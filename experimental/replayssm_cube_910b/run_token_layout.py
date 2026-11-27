# SPDX-License-Identifier: Apache-2.0
"""Compare fixed-slot and compact V24 on the same data and NPU graph."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experimental.replayssm_cube_910b.run import batches, measure, select_flush_rows  # noqa: E402
from experimental.replayssm_sd_ascendc.lab import CACHE_NAMES, MAX_TOKENS, make_case  # noqa: E402
from scripts.replayssm.profile_sd_operator_compare import package_versions  # noqa: E402

INPUTS = ("q", "k", "v", "g", "beta")
PATHS = ("fixed", "pack_unpack", "compact")
RING_LEN = 32
TILE_V = 128
CORES = 40
FLUSH_POSITION = 15
NONFLUSH_POSITION = 7


def layout_tensors(lengths, device):
    offsets = torch.cat((lengths.new_zeros(1), lengths.cumsum(0, dtype=torch.int32)))
    pick = (torch.arange(MAX_TOKENS) < lengths[:, None]).flatten().nonzero().flatten()
    return offsets.to(device), pick.to(device)


def trial(args, batch, device):
    tp = 4 if args.geometry == "27b" else 1
    cpu = make_case(
        geometry=args.geometry,
        tp=tp,
        batch=batch,
        position=FLUSH_POSITION,
        length=MAX_TOKENS,
        base=31,
        tile_v=TILE_V,
        seed=42,
    )
    lengths = cpu["lengths"]
    if args.layout != "uniform":
        lengths.copy_((torch.arange(batch) % MAX_TOKENS + 1).to(torch.int32))
    if args.layout == "padded":
        # Include an interior pad and a trailing pad when the batch allows it.
        lengths[1::5].zero_()
        if batch > 1:
            lengths[-1] = 0
    padding = lengths == 0
    flush_count = min(int(batch * args.flush_fraction), int((~padding).sum()))
    flush_rows = (~padding).nonzero().flatten()[select_flush_rows(int((~padding).sum()), flush_count, "random", 42)]
    cpu["positions"].fill_(NONFLUSH_POSITION)
    cpu["positions"][flush_rows] = FLUSH_POSITION
    cpu["positions"][padding] = 0
    cpu["bases"][padding] = 0
    cpu["slots"][padding] = 0

    case = {name: cpu[name].to(device) for name in ("slots", "positions", "bases", "lengths", *CACHE_NAMES)}
    # Exercise independent checkpoint and ring IDs.
    case["checkpoint"] = torch.cat((case["checkpoint"][:1], case["checkpoint"]))
    slots = torch.where(case["slots"] > 0, case["slots"] + 1, 0)
    value_heads = cpu["v"].size(2)
    fixed_inputs = [cpu[name].to(device) for name in INPUTS]
    offsets, pick = layout_tensors(lengths, device)
    tokens = pick.numel()
    compact_inputs = [tensor.flatten(0, 1).index_select(0, pick) for tensor in fixed_inputs]
    if args.layout == "padded":
        # A captured graph may reserve physical token capacity beyond offsets[-1].
        compact_inputs = [
            torch.cat((src, src.new_full((batch * MAX_TOKENS - tokens, *src.shape[1:]), float("nan"))))
            for src in compact_inputs
        ]
    flush = torch.zeros(batch, dtype=torch.int32, device=device)
    flush[flush_rows.to(device)] = 1
    restore_slots = slots[flush_rows.to(device)].long()
    restore_state = case["checkpoint"].index_select(0, restore_slots)
    verify_status = torch.empty(batch * value_heads, dtype=torch.int32, device=device)
    compact_status = torch.empty(batch * value_heads * 4, dtype=torch.int32, device=device)
    fixed_out = torch.empty_like(fixed_inputs[2])
    compact_out = torch.empty_like(compact_inputs[2])
    op = torch.ops.replayssm_sd_lab.fused_

    def invoke(cache, inputs, out, token_offsets=None):
        op(
            *inputs,
            *(cache[name] for name in CACHE_NAMES),
            slots,
            case["positions"],
            case["bases"],
            case["lengths"],
            flush,
            out,
            verify_status,
            compact_status,
            TILE_V**-0.5,
            TILE_V,
            CORES,
            case["slots"],
            24,
            token_offsets,
        )

    def reset_cache(cache):
        if flush_count:
            cache["checkpoint"].index_copy_(0, restore_slots, restore_state)

    def reset_status():
        verify_status.fill_(-1)
        compact_status.fill_(-1)

    def check_status():
        torch.testing.assert_close(verify_status.cpu(), torch.zeros_like(verify_status, device="cpu"))
        torch.testing.assert_close(compact_status.cpu(), torch.zeros_like(compact_status, device="cpu"))

    def capture_expected(valid_indices=pick):
        reset_cache(case)
        reset_status()
        invoke(case, fixed_inputs, fixed_out)
        check_status()
        expected_out = fixed_out.flatten(0, 1).index_select(0, valid_indices).cpu()
        expected_cache = {name: case[name].cpu() for name in CACHE_NAMES}
        reset_cache(case)
        return expected_out, expected_cache

    expected_out, expected_cache = capture_expected()
    layers = [case] + [{name: case[name].clone() for name in CACHE_NAMES} for _ in range(args.layers - 1)]

    def reset_stack():
        for cache in layers:
            reset_cache(cache)

    def check_cache(cache, reference):
        for name in CACHE_NAMES:
            torch.testing.assert_close(cache[name].cpu(), reference[name], atol=0, rtol=0, equal_nan=True, msg=name)

    def check_compact(out, reference):
        check_status()
        torch.testing.assert_close(out[:tokens].cpu(), reference, atol=0, rtol=0)
        if out.size(0) > tokens:
            assert torch.isnan(out[tokens:]).all().item(), "compact output wrote into padding"

    rows = []
    outputs = []
    for path in PATHS:
        if path == "compact":

            def call_stack():
                for cache in layers:
                    invoke(cache, compact_inputs, compact_out, offsets)

        elif path == "fixed":

            def call_stack():
                for cache in layers:
                    invoke(cache, fixed_inputs, fixed_out)

        else:

            def call_stack():
                for index, cache in enumerate(layers):
                    packed = [tensor.new_zeros(tensor.shape) for tensor in fixed_inputs]
                    for dest, src in zip(packed, compact_inputs):
                        dest.flatten(0, 1).index_copy_(0, pick, src[:tokens])
                    invoke(cache, packed, fixed_out)
                    unpacked = fixed_out.flatten(0, 1).index_select(0, pick)
                    if len(outputs) <= index:
                        outputs.append(unpacked)
                    else:
                        outputs[index] = unpacked

        def check_stack(graph, path=path):
            reset_stack()
            reset_status()
            compact_out.fill_(float("nan"))
            fixed_out.fill_(float("nan"))
            graph.replay()
            if path == "compact":
                check_compact(compact_out, expected_out)
            else:
                check_status()
                result = outputs[-1] if path == "pack_unpack" else fixed_out.flatten(0, 1).index_select(0, pick)
                torch.testing.assert_close(result.cpu(), expected_out, atol=0, rtol=0)
            check_cache(layers[0], expected_cache)
            if len(layers) > 1:
                check_cache(layers[-1], expected_cache)

            if path == "compact" and args.layout == "ragged" and batch > 1:
                # Keep the same number of tokens and buffer addresses, but change
                # their request boundaries after capture. Rebuild the fixed oracle
                # using those same flat Q/K/V/g/beta values.
                changed_lengths = lengths.roll(1)
                changed_offsets, changed_pick = layout_tensors(changed_lengths, device)
                offsets.copy_(changed_offsets)
                case["lengths"].copy_(changed_lengths)
                for dest, src in zip(fixed_inputs, compact_inputs):
                    dest.flatten(0, 1).index_copy_(0, changed_pick, src)
                changed_expected, changed_cache = capture_expected(changed_pick)
                # The positive check also compares bytes outside the new valid
                # history. Restore the fixed oracle's input ring before replay.
                for name in CACHE_NAMES[1:]:
                    case[name].copy_(expected_cache[name])
                reset_stack()
                reset_status()
                compact_out.fill_(float("nan"))
                graph.replay()
                check_compact(compact_out, changed_expected)
                check_cache(layers[0], changed_cache)
                if len(layers) > 1:
                    check_cache(layers[-1], changed_cache)

                if batch == 4:
                    # Make only the last row invalid. It must write a status
                    # without touching its output or checkpoint/ring slots.
                    offsets[-1] = tokens + 1
                    reset_stack()
                    reset_status()
                    compact_out.fill_(float("nan"))
                    graph.replay()
                    statuses = verify_status.cpu().view(batch, value_heads)
                    assert (statuses[:-1] == 0).all() and (statuses[-1] == 6).all()
                    statuses = compact_status.cpu().view(batch, value_heads * 4)
                    assert (statuses[:-1] == 0).all() and (statuses[-1] == 6).all()
                    last_start = int(changed_lengths[:-1].sum())
                    assert torch.isnan(compact_out[last_start:]).all().item()
                    bad_checkpoint_slot = int(cpu["slots"][-1]) + 1
                    last_initial = cpu["checkpoint"][bad_checkpoint_slot - 1]
                    torch.testing.assert_close(layers[-1]["checkpoint"][bad_checkpoint_slot].cpu(), last_initial)
                    bad_ring_slot = int(cpu["slots"][-1])
                    for name in CACHE_NAMES[1:]:
                        torch.testing.assert_close(
                            layers[-1][name][bad_ring_slot].cpu(),
                            changed_cache[name][bad_ring_slot],
                            atol=0,
                            rtol=0,
                            equal_nan=True,
                        )

        label = f"{args.geometry}/b{batch}/{args.layout}/f{flush_count}/{path}/layers{args.layers}"
        timing = measure(call_stack, reset_stack, args.iterations, label, check_graph=check_stack)
        rows.append(dict(path=path, batch=batch, tokens=tokens, flush_rows=flush_count, **timing))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", choices=("27b", "9b"), required=True)
    parser.add_argument("--batches", type=batches, required=True)
    parser.add_argument("--layout", choices=("uniform", "ragged", "padded"), default="ragged")
    parser.add_argument("--flush-fraction", type=float, default=0.125)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--verify-library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.flush_fraction <= 1 or not 1 <= args.layers <= 48 or args.iterations < 1:
        parser.error("Expected flush-fraction in [0,1], 1..48 layers, and at least one iteration")

    import torch_npu  # noqa: F401

    torch.npu.set_device(args.device)
    device = torch.device(f"npu:{args.device}")
    library = args.verify_library.resolve(strict=True)
    torch.ops.load_library(str(library))
    report = dict(
        status="RUNNING",
        config=vars(args),
        packages=package_versions(),
        device_name=torch.npu.get_device_name(args.device),
        library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
        rows=[],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for batch in args.batches:
            report["rows"].extend(trial(args, batch, device))
        report["status"] = "PASSED"
    except Exception as exc:
        report.update(status="FAILED", failed_batch=batch, error=repr(exc))
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
