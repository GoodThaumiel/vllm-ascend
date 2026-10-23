# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Small NPU profile: native speculative RGDR versus Replay SD lifecycle ops.

The same recurrent inputs are used for both paths. Each measured call starts
from a restored state; restoration runs outside the labelled/profiled region.
This is a one-layer operator comparison, not an HTTP serving benchmark.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import statistics
import subprocess
import sys
import traceback
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experimental/replayssm_sd_ascendc"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lab import CACHE_NAMES, clone_page_strided_case, make_case  # noqa: E402
from profile_sd_operator_compare import native_state_page, reconstruct_native_start  # noqa: E402

HEAD_DIM = 128
VALUE_HEADS = 48
KEY_HEADS = 16
GDN_LAYERS = 48


def p50_p90(samples_us: list[float]) -> dict[str, float]:
    ordered = sorted(samples_us)
    return {
        "p50_us": statistics.median(ordered),
        "p90_us": ordered[math.ceil(0.9 * len(ordered)) - 1],
        "min_us": ordered[0],
    }


def make_cursor_case(batch: int, position: int, base: int, flush: bool, accepted: int, device: torch.device):
    """One block per request, with slot zero reserved for padding."""
    write_pos = torch.zeros(batch + 1, dtype=torch.int32, device=device)
    cache_base = torch.zeros_like(write_pos)
    is_flush = torch.zeros(batch + 1, dtype=torch.int8, device=device)
    write_pos[1:] = position
    cache_base[1:] = base
    is_flush[1:] = int(flush)
    return {
        "write_pos": write_pos,
        "cache_base": cache_base,
        "is_flush": is_flush,
        "accepted": torch.full((batch,), accepted, dtype=torch.int32, device=device),
        "slots": torch.arange(1, batch + 1, dtype=torch.int32, device=device),
        "do_reset": torch.zeros(batch, dtype=torch.int8, device=device),
        "status": torch.empty(batch, dtype=torch.int32, device=device),
    }


def run_case(args: argparse.Namespace, batch: int, position: int, length: int, flush: bool) -> dict:
    from vllm_ascend.ops.gdn import recurrent_gated_delta_rule
    from vllm_ascend.ops.replayssm.ascendc import (
        replayssm_sd_compact,
        replayssm_sd_cursor,
        replayssm_sd_verify,
    )

    device = torch.device(f"npu:{args.device}")
    value_heads = VALUE_HEADS // args.tp
    key_heads = KEY_HEADS // args.tp
    cpu = make_case(
        tp=args.tp,
        batch=batch,
        position=position,
        length=length,
        base=args.base,
        ring_len=args.ring,
        tile_v=args.tile_v,
        seed=args.seed + batch * 100 + position * 10 + length,
    )
    native_beta = cpu["beta"].bfloat16().to(device)
    cpu["beta"] = cpu["beta"].bfloat16().float()
    replay = clone_page_strided_case(cpu, device)
    replay["flush"] = torch.full((batch,), int(flush), dtype=torch.int32, device=device)
    replay["compact_status"] = torch.empty_like(replay["status"])
    cursor = make_cursor_case(batch, position, args.base, flush, args.accepted, device)
    native_state, native_page_bytes = native_state_page(device, args.tp, batch, length)
    native_start = reconstruct_native_start(cpu, position).to(device)
    native_slots = torch.arange(1, batch * length + 1, dtype=torch.int32, device=device)
    native_lengths = torch.tensor([0] + [length] * batch, dtype=torch.int32, device=device)
    native_accepted = torch.full((batch,), args.accepted, dtype=torch.int32, device=device)
    q = replay["q"][:, :length].contiguous().view(batch * length, key_heads, HEAD_DIM)
    k = replay["k"][:, :length].contiguous().view_as(q)
    v = replay["v"][:, :length].contiguous().view(batch * length, value_heads, HEAD_DIM)
    g = replay["g"][:, :length].contiguous().view(batch * length, value_heads)
    beta = native_beta[:, :length].contiguous().view_as(g)
    scale = HEAD_DIM**-0.5
    pristine_replay = {name: replay[name].clone() for name in (*CACHE_NAMES, "out", "status")}
    pristine_cursor = {name: cursor[name].clone() for name in ("write_pos", "cache_base", "is_flush")}

    def reset_native() -> None:
        for row in range(batch):
            native_state[1 + row * length + args.accepted - 1].copy_(native_start[row])

    def reset_replay() -> None:
        for name, source in pristine_replay.items():
            replay[name].copy_(source)
        for name, source in pristine_cursor.items():
            cursor[name].copy_(source)
        replay["compact_status"].fill_(-1)
        cursor["status"].fill_(-1)

    def native_call() -> torch.Tensor:
        return recurrent_gated_delta_rule(
            q,
            k,
            v,
            native_state,
            g=g,
            beta=beta,
            scale=scale,
            actual_seq_lengths=native_lengths,
            ssm_state_indices=native_slots,
            num_accepted_tokens=native_accepted,
        )

    def verify_call() -> None:
        replayssm_sd_verify(
            *(
                replay[name]
                for name in (
                    "q",
                    "k",
                    "v",
                    "g",
                    "beta",
                    "checkpoint",
                    "d_cache",
                    "k_cache",
                    "g_cache",
                    "slots",
                    "positions",
                    "bases",
                    "lengths",
                    "out",
                    "status",
                )
            ),
            scale,
            args.tile_v,
            args.cores,
        )

    def compact_call() -> None:
        replayssm_sd_compact(
            *(
                replay[name]
                for name in (
                    "checkpoint",
                    "d_cache",
                    "k_cache",
                    "g_cache",
                    "slots",
                    "positions",
                    "bases",
                    "flush",
                    "compact_status",
                )
            ),
            args.tile_v,
            args.cores,
        )

    def cursor_call() -> None:
        replayssm_sd_cursor(
            *(
                cursor[name]
                for name in (
                    "write_pos",
                    "cache_base",
                    "is_flush",
                    "accepted",
                    "slots",
                    "do_reset",
                    "status",
                )
            ),
            20,
            4,
            args.ring,
        )

    reset_native()
    reset_replay()
    torch.npu.synchronize()
    native_first = native_call().view(batch, length, value_heads, HEAD_DIM)
    verify_call()
    compact_call()
    cursor_call()
    torch.npu.synchronize()
    for name, status in (
        ("verify", replay["status"]),
        ("compact", replay["compact_status"]),
        ("cursor", cursor["status"]),
    ):
        if bool(torch.any(status.cpu() != 0)):
            raise RuntimeError(f"{name} returned nonzero status")
    difference = (native_first.float() - replay["out"][:, :length].float()).abs()
    output_max_abs = difference.max().item()
    output_relative_l2 = (difference.norm() / native_first.float().norm().clamp_min(1e-12)).item()
    if not math.isfinite(output_max_abs) or output_max_abs > args.max_abs_error:
        raise RuntimeError(f"native/Replay output max_abs={output_max_abs:.6g}")
    if flush:
        checkpoint_max_abs = (replay["checkpoint"][1:].float() - native_start.float()).abs().max().item()
        if not math.isfinite(checkpoint_max_abs) or checkpoint_max_abs > args.max_checkpoint_error:
            raise RuntimeError(f"compact checkpoint max_abs={checkpoint_max_abs:.6g}")
    else:
        checkpoint_max_abs = 0.0
        torch.testing.assert_close(replay["checkpoint"].cpu(), cpu["checkpoint"], rtol=0, atol=0)
    expected_pos = args.accepted if flush else position + args.accepted
    expected_base = ((args.base + position) & (args.ring - 1)) if flush else args.base
    if not bool(torch.all(cursor["write_pos"][1:].cpu() == expected_pos)):
        raise RuntimeError("cursor write_pos mismatch")
    if not bool(torch.all(cursor["cache_base"][1:].cpu() == expected_base)):
        raise RuntimeError("cursor cache_base mismatch")
    expected_flush = int(expected_pos + 8 > 20)
    if not bool(torch.all(cursor["is_flush"][1:].cpu() == expected_flush)):
        raise RuntimeError("cursor is_flush mismatch")

    label = f"tp{args.tp}_b{batch}_t{length}_p{position}_f{int(flush)}_a{args.accepted}_c{args.cores}"
    timings: dict[str, list[float]] = {name: [] for name in ("native", "verify", "compact", "cursor", "replay_total")}
    for iteration in range(args.warmup + args.repetitions):
        reset_native()
        reset_replay()
        torch.npu.synchronize()
        events = [torch.npu.Event(enable_timing=True) for _ in range(5)]
        events[0].record()
        with torch.profiler.record_function(f"REPLAYSSM_SD_LIFECYCLE/{label}/native"):
            native_call()
        events[1].record()
        with torch.profiler.record_function(f"REPLAYSSM_SD_LIFECYCLE/{label}/verify"):
            verify_call()
        events[2].record()
        with torch.profiler.record_function(f"REPLAYSSM_SD_LIFECYCLE/{label}/compact"):
            compact_call()
        events[3].record()
        with torch.profiler.record_function(f"REPLAYSSM_SD_LIFECYCLE/{label}/cursor"):
            cursor_call()
        events[4].record()
        torch.npu.synchronize()
        if iteration >= args.warmup:
            timings["native"].append(events[0].elapsed_time(events[1]) * 1000)
            timings["verify"].append(events[1].elapsed_time(events[2]) * 1000)
            timings["compact"].append(events[2].elapsed_time(events[3]) * 1000)
            timings["cursor"].append(events[3].elapsed_time(events[4]) * 1000)
            timings["replay_total"].append(events[1].elapsed_time(events[4]) * 1000)

    summary = {name: p50_p90(samples) for name, samples in timings.items()}
    return {
        "label": label,
        "batch": batch,
        "tp_geometry": args.tp,
        "verify_tokens": length,
        "history_position": position,
        "flush": flush,
        "accepted": args.accepted,
        "cores": args.cores,
        "output_max_abs": output_max_abs,
        "output_relative_l2": output_relative_l2,
        "checkpoint_max_abs": checkpoint_max_abs,
        "device_timings": summary,
        "analytical_48_layer_recurrent_only_us": {
            "native": GDN_LAYERS * summary["native"]["p50_us"],
            "replay": GDN_LAYERS * (summary["verify"]["p50_us"] + summary["compact"]["p50_us"])
            + summary["cursor"]["p50_us"],
        },
        "native_state_page_bytes": native_page_bytes,
        "replay_state_page_bytes": replay["checkpoint"].stride(0) * 2,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tp", type=int, choices=(4, 8), default=4)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 16, 32])
    parser.add_argument("--positions", type=int, nargs="+", default=[0, 12, 16])
    parser.add_argument("--length", type=int, choices=(1, 4), default=4)
    parser.add_argument("--accepted", type=int, default=1)
    parser.add_argument("--base", type=int, default=29)
    parser.add_argument("--ring", type=int, choices=(32, 64), default=32)
    parser.add_argument("--tile-v", type=int, choices=(16, 32), default=32)
    parser.add_argument("--cores", type=int, choices=(8, 20, 40), default=40)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--max-abs-error", type=float, default=0.05)
    parser.add_argument("--max-checkpoint-error", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(batch < 1 for batch in args.batches) or any(pos < 0 or pos > 16 for pos in args.positions):
        parser.error("batches must be positive; positions must be 0..16")
    if args.accepted < 1 or args.accepted > args.length:
        parser.error("accepted must be in 1..length")
    if args.base < 0 or args.base >= args.ring or args.repetitions < 1 or args.warmup < 0:
        parser.error("invalid base, repetitions or warmup")
    if args.max_abs_error <= 0 or args.max_checkpoint_error <= 0:
        parser.error("error tolerances must be positive")
    if any(pos + args.length > min(20, args.ring) for pos in args.positions):
        parser.error("position + length exceeds the logical or physical window")

    import torch_npu  # noqa: F401

    from vllm_ascend.utils import enable_custom_op

    torch.npu.set_device(args.device)
    names = ("replayssm_sd_verify_", "replayssm_sd_compact_", "replayssm_sd_cursor_")
    if not enable_custom_op() or any(not hasattr(torch.ops._C_ascend, name) for name in names):
        raise RuntimeError("Rebuild the Ascend910B vllm_ascend_C extension with all Replay SD operators")
    report = {
        "status": "RUNNING",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "packages": {name: importlib.metadata.version(name) for name in ("torch", "torch-npu", "vllm", "vllm-ascend")},
        "note": (
            "One-layer recurrent boundary, common q/k/v/g/beta, no conv/gating/norm/merge/serve. "
            "Eager NPU Event timings may include host dispatch gaps; do not compare with graph replay p50. "
            "Restore copies are outside labelled timing regions. Replay cursor is one call per batch step, "
            "not 48 calls. Analytical 48-layer estimate is a sum of isolated p50s, not measured serving latency."
        ),
        "cases": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for batch in args.batches:
        for pos in args.positions:
            flush = pos + 8 > 20
            try:
                case = run_case(args, batch, pos, args.length, flush)
            except Exception:
                report["status"] = "FAILED"
                report["failed_case"] = {"batch": batch, "position": pos, "traceback": traceback.format_exc()}
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                raise
            report["cases"].append(case)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
    report["status"] = "PASSED"
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"PASSED {len(report['cases'])} cases; report={args.output}", flush=True)


if __name__ == "__main__":
    main()
