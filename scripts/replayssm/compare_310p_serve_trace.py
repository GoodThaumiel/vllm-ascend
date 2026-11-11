# SPDX-License-Identifier: Apache-2.0
"""Compare native and teacher-forced 310P ReplaySSM traces from eager serve."""

import argparse
import json
from pathlib import Path

import torch


def error(native, replay, atol, rtol):
    if native.shape != replay.shape:
        return {"shapes": [list(native.shape), list(replay.shape)], "outside_tolerance": True}
    native, replay = native.float(), replay.float()
    finite = bool(torch.isfinite(native).all() and torch.isfinite(replay).all())
    return {
        "max_abs": float((native - replay).abs().max()) if finite else None,
        "outside_tolerance": not finite or not torch.allclose(native, replay, atol=atol, rtol=rtol),
    }


def advance(state, layer, suffix, ring_index):
    gate = layer[f"g_{suffix}"][:, ring_index].float().exp()
    delta = layer[f"d_{suffix}"][:, ring_index].float()
    key = layer[f"k_{suffix}"][:, ring_index].float()
    key = key.repeat_interleave(state.shape[0] // key.shape[0], dim=0)
    return gate[:, None, None] * state + delta[:, :, None] * key[:, None, :]


def dense_verify(history, layer, length):
    state = history.double()
    repeats = state.shape[0] // layer["k"].shape[1]
    key = layer["k"].double().repeat_interleave(repeats, dim=1)
    query = layer["q"].double().repeat_interleave(repeats, dim=1)
    scale = query.shape[-1] ** -0.5
    output = []
    deltas = []
    for token in range(length):
        state = state * layer["g"][token].double().exp()[:, None, None]
        projected = (state @ key[token, :, :, None]).squeeze(-1)
        delta = layer["beta"][token].double()[:, None] * (layer["v"][token].double() - projected)
        state = state + delta[:, :, None] * key[token, :, None, :]
        output.append((state @ query[token, :, :, None]).squeeze(-1) * scale)
        deltas.append(delta)
    return torch.stack(output).half(), torch.stack(deltas).half()


def compare_layer(native, replay, atol, rtol):
    result = {
        name: error(native[name], replay[name], atol, rtol)
        for name in ("q", "k", "v", "g", "beta", "gdn_out")
        if name in native and name in replay
    }
    if "positions" not in replay:
        if "state_after" in native and "state_after" in replay:
            result["state_after"] = error(native["state_after"], replay["state_after"], atol, rtol)
        return result
    metadata = ("slots", "ring_slots", "positions", "bases", "lengths", "flush", "reset")
    result["ring"] = {key: replay[key].tolist() for key in metadata}
    result["status"] = {
        key: replay[key].unique().tolist() for key in ("cursor_status", "verify_status", "compact_status")
    }
    pos, base, length = (int(replay[key][0]) for key in ("positions", "bases", "lengths"))
    history = replay["state_before"].float()
    ring_len = replay["d_before"].shape[1]
    for offset in range(pos):
        history = advance(history, replay, "before", (base + offset) % ring_len)
    if "state_before" in native:
        result["state_before"] = error(native["state_before"], history, atol, rtol)
    if length:
        reference_output, reference_delta = dense_verify(history, replay, length)
        offsets = [(base + pos + offset) % ring_len for offset in range(length)]
        actual_delta = replay["d_after"][:, offsets].transpose(0, 1)
        result["verify_out_vs_reference"] = error(reference_output, replay["gdn_out"], atol, rtol)
        result["verify_delta_vs_reference"] = error(reference_delta, actual_delta, atol, rtol)
    candidate = history
    if "state_after" in native and len(native["state_after"]) == length and length:
        reconstructed = []
        for offset in range(length):
            candidate = advance(candidate, replay, "after", (base + pos + offset) % ring_len)
            reconstructed.append(candidate)
        result["candidate_states"] = error(native["state_after"], torch.stack(reconstructed), atol, rtol)
    if int(replay["flush"][0]):
        result["compact_checkpoint"] = error(history, replay["state_after"], atol, rtol)
    else:
        result["compact_checkpoint"] = error(replay["state_before"], replay["state_after"], 0, 0)
    return result


def compare_logits(native, replay, atol, rtol):
    result = error(native, replay, atol, rtol)
    if native.shape != replay.shape:
        return result
    n_values, n_ids = native.float().topk(2, dim=-1)
    r_values, r_ids = replay.float().topk(2, dim=-1)
    result["rows"] = [
        {
            "native_argmax": int(n_id[0]),
            "replay_argmax": int(r_id[0]),
            "native_margin": float(n_value[0] - n_value[1]),
            "replay_margin": float(r_value[0] - r_value[1]),
            "replay_margin_on_native_pair": float(replay[i, n_id[0]] - replay[i, n_id[1]]),
        }
        for i, (n_value, n_id, r_value, r_id) in enumerate(zip(n_values, n_ids, r_values, r_ids))
    ]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("native", type=Path)
    parser.add_argument("replay", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=0.002)
    parser.add_argument("--rtol", type=float, default=0.002)
    args = parser.parse_args()
    report = {}
    for rank in ("rank0", "rank1"):
        native_paths = sorted((args.native / rank).glob("*.pt"))
        replay_paths = sorted((args.replay / rank).glob("*.pt"))
        if not native_paths or [p.name for p in native_paths] != [p.name for p in replay_paths]:
            raise ValueError(f"Incomplete or unequal trace at {rank}")
        steps = []
        first = {}
        offset = 0
        for step, (native_path, replay_path) in enumerate(zip(native_paths, replay_paths)):
            native = torch.load(native_path, map_location="cpu", mmap=True, weights_only=True)
            replay = torch.load(replay_path, map_location="cpu", mmap=True, weights_only=True)
            for key in ("input_ids", "sampled"):
                if not torch.equal(native[key], replay[key]):
                    raise ValueError(f"{rank} step {step} has different {key}")
            for key in ("drafts", "computed", "length"):
                if native[key] != replay[key]:
                    raise ValueError(f"{rank} step {step} has different {key}")
            layer_stats = {
                name: compare_layer(native["layers"][name], replay["layers"][name], args.atol, args.rtol)
                for name in native["layers"].keys() & replay["layers"].keys()
            }
            for name, fields in sorted(layer_stats.items(), key=lambda item: list(native["layers"]).index(item[0])):
                for field, stats in fields.items():
                    if stats.get("max_abs") or stats.get("outside_tolerance"):
                        first.setdefault("nonzero", {"step": step, "layer": name, "tensor": field})
                    if stats.get("outside_tolerance"):
                        first.setdefault("outside_tolerance", {"step": step, "layer": name, "tensor": field})
                if any(code for codes in fields.get("status", {}).values() for code in codes):
                    first.setdefault("nonzero_status", {"step": step, "layer": name, "status": fields["status"]})
            logits = compare_logits(native["logits"], replay["logits"], args.atol, args.rtol)
            accepted = int((native["sampled"] >= 0).sum())
            for row, entry in enumerate(logits.get("rows", [])):
                if entry["native_argmax"] != entry["replay_argmax"]:
                    mismatch = {"step": step, "logit_row": row, "output_offset": offset, **entry}
                    first.setdefault("argmax", mismatch)
                    if row < accepted:
                        first.setdefault("accepted_window_argmax", {**mismatch, "token_index": offset + row})
            steps.append(
                {
                    "step": step,
                    "output_offset": offset,
                    "computed": native["computed"],
                    "drafts": native["drafts"],
                    "window": native["length"],
                    "accepted": accepted,
                    "native_previous_accepted": native["previous_accepted"],
                    "replay_previous_accepted": replay["previous_accepted"],
                    "logits": logits,
                    "layers": layer_stats,
                    "native_only_layers": sorted(native["layers"].keys() - replay["layers"].keys()),
                    "replay_only_layers": sorted(replay["layers"].keys() - native["layers"].keys()),
                }
            )
            offset += accepted
        report[rank] = {"first": first, "steps": steps}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({rank: result["first"] for rank, result in report.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
