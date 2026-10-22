# SPDX-License-Identifier: Apache-2.0
"""Profile the native vLLM Ascend GDN recurrent op against ReplaySSM AR.

Run under the server's CANN/torch_npu environment on an otherwise idle NPU.
The two measured ranges contain only the recurrent operator launches; setup,
correctness checks, and warmup remain outside those ranges.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experimental/replayssm_ar_ascendc"))

HEAD_DIM = 128
PAGE_ALIGNMENT_BYTES = 256 * 1024
CONV_WIDTH_MULTIPLIER = 3
DEFAULT_NUM_SLOTS = 1510


def parse_ints(value: str) -> tuple[int, ...]:
    result = tuple(int(part) for part in value.split(","))
    if not result or any(number < 0 for number in result) or len(result) != len(set(result)):
        raise argparse.ArgumentTypeError("expected unique nonnegative comma-separated integers")
    return result


def layout_geometry(tp: int, capacity: int, page_bytes: int = 0) -> tuple[int, int, dict[str, int]]:
    """Match the 0.30 hybrid GDN state order and 256-KiB page alignment."""
    key_heads = 16 // tp
    value_heads = 48 // tp
    conv_bytes = CONV_WIDTH_MULTIPLIER * (value_heads + 2 * key_heads) * HEAD_DIM * 2
    regions = {
        "checkpoint": value_heads * HEAD_DIM * HEAD_DIM * 2,
        "d_cache": value_heads * capacity * HEAD_DIM * 2,
        "k_cache": key_heads * capacity * HEAD_DIM * 2,
        "g_cache": value_heads * capacity * 4,
    }
    payload_bytes = conv_bytes + sum(regions.values())
    if page_bytes == 0:
        page_bytes = math.ceil(payload_bytes / PAGE_ALIGNMENT_BYTES) * PAGE_ALIGNMENT_BYTES
    if page_bytes < payload_bytes or page_bytes % 32:
        raise ValueError(f"page_bytes={page_bytes} must fit {payload_bytes} payload bytes and be 32-byte aligned")
    return conv_bytes, page_bytes, regions


def make_page_views(
    *, device: torch.device, num_slots: int, tp: int, capacity: int, page_bytes: int
) -> tuple[dict[str, torch.Tensor], int]:
    """Allocate one server-like page-strided backing for one GDN layer."""
    key_heads = 16 // tp
    value_heads = 48 // tp
    conv_bytes, page_bytes, regions = layout_geometry(tp, capacity, page_bytes)
    raw = torch.empty(num_slots * page_bytes, dtype=torch.uint8, device=device)
    shapes = {
        "checkpoint": (num_slots, value_heads, HEAD_DIM, HEAD_DIM),
        "d_cache": (num_slots, value_heads, capacity, HEAD_DIM),
        "k_cache": (num_slots, key_heads, capacity, HEAD_DIM),
        "g_cache": (num_slots, value_heads, capacity),
    }
    dtypes = {
        "checkpoint": torch.bfloat16,
        "d_cache": torch.float16,
        "k_cache": torch.float16,
        "g_cache": torch.float32,
    }
    views = {}
    offset_bytes = conv_bytes
    for name, shape in shapes.items():
        dtype = dtypes[name]
        item_bytes = torch.empty(0, dtype=dtype, device="cpu").element_size()
        typed = raw.view(dtype)
        row_strides = []
        running = 1
        for size in reversed(shape[1:]):
            row_strides.append(running)
            running *= size
        strides = (page_bytes // item_bytes, *reversed(row_strides))
        views[name] = torch.as_strided(
            typed,
            size=shape,
            stride=strides,
            storage_offset=offset_bytes // item_bytes,
        )
        offset_bytes += regions[name]
    return views, page_bytes


def reconstruct_state(case: dict[str, torch.Tensor], position: int) -> torch.Tensor:
    """Build the native pre-token state corresponding to Replay's checkpoint + ring."""
    batch = case["q"].shape[0]
    key_heads = case["k"].shape[1]
    value_heads = case["v"].shape[1]
    state = case["checkpoint"][1 : batch + 1].float().clone()
    for step in range(position):
        decay = case["g_cache"][1 : batch + 1, :, step].exp().float()
        delta = case["d_cache"][1 : batch + 1, :, step].float()
        key = case["k_cache"][1 : batch + 1, :, step].float()
        key = key.repeat_interleave(value_heads // key_heads, dim=1)
        state = decay[..., None, None] * state + delta[..., None] * key[..., None, :]
    return state.to(torch.bfloat16)


def package_versions() -> dict[str, str]:
    names = ("torch", "torch-npu", "vllm", "vllm-ascend", "flash-linear-attention-npu-a2")
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed under this distribution name"
    return versions


def source_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def run_case(args: argparse.Namespace, batch: int, context_length: int, position: int) -> dict:
    from lab import make_case

    from vllm_ascend.ops.gdn import recurrent_gated_delta_rule
    from vllm_ascend.ops.replayssm.ascendc import replayssm_ar_step

    device = torch.device(f"npu:{args.device}")
    flush = position == args.capacity - 1
    seed = args.seed + batch * 100_000 + context_length + position
    cpu = make_case(args.tp, batch, args.capacity, position, flush, args.tile_v, seed)
    beta_native = cpu["beta"].to(torch.bfloat16).to(device)
    inputs = {name: cpu[name].to(device) for name in ("q", "k", "v", "g", "slots", "positions", "flush")}
    inputs["beta_replay"] = beta_native.float()
    actual_seq_lengths = torch.tensor([0] + [1] * batch, dtype=torch.int32, device=device)
    native_views, page_bytes = make_page_views(
        device=device, num_slots=args.num_slots, tp=args.tp, capacity=args.capacity, page_bytes=args.page_bytes
    )
    replay_views, _ = make_page_views(
        device=device, num_slots=args.num_slots, tp=args.tp, capacity=args.capacity, page_bytes=page_bytes
    )
    active = slice(1, batch + 1)
    reconstructed = reconstruct_state(cpu, position)
    native_views["checkpoint"][active].copy_(reconstructed.to(device))
    for name in replay_views:
        replay_views[name][active].copy_(cpu[name][active].to(device))
    out = torch.empty((batch, 48 // args.tp, HEAD_DIM), dtype=torch.bfloat16, device=device)
    first_out = torch.empty_like(out)
    tasks = batch * (48 // args.tp) * (HEAD_DIM // args.tile_v)
    status = torch.empty(tasks, dtype=torch.int32, device=device)
    scale = HEAD_DIM**-0.5

    def native_call() -> torch.Tensor:
        # This is the exact symbol used by vllm_ascend.ops.gdn when Replay is off.
        return recurrent_gated_delta_rule(
            inputs["q"],
            inputs["k"],
            inputs["v"],
            native_views["checkpoint"],
            g=inputs["g"],
            beta=beta_native,
            scale=scale,
            actual_seq_lengths=actual_seq_lengths,
            ssm_state_indices=inputs["slots"],
        )

    def replay_call(output: torch.Tensor = out) -> None:
        replayssm_ar_step(
            inputs["q"],
            inputs["k"],
            inputs["v"],
            inputs["g"],
            inputs["beta_replay"],
            replay_views["checkpoint"],
            replay_views["d_cache"],
            replay_views["k_cache"],
            replay_views["g_cache"],
            inputs["slots"],
            inputs["positions"],
            inputs["flush"],
            output,
            status,
            scale,
            args.tile_v,
            args.cores,
        )

    # Optional warmup runs outside the measured ranges. With --warmup 0 (the
    # default), the complete script launches exactly 100 calls of each operator
    # per case, which simplifies whole-process profiler filtering.
    for _ in range(args.warmup):
        native_call()
        replay_call()
    torch.npu.synchronize()
    native_views["checkpoint"][active].copy_(reconstructed.to(device))
    replay_views["checkpoint"][active].copy_(cpu["checkpoint"][active].to(device))
    torch.npu.synchronize()

    label = f"tp{args.tp}_b{batch}_ctx{context_length}_pos{position}_flush{int(flush)}_cores{args.cores}"
    durations = {}
    print(f"PROFILE_BEGIN {label} native repetitions={args.repetitions}", flush=True)
    torch.npu.synchronize()
    start = time.perf_counter()
    first_native_result = None
    with torch.profiler.record_function(f"REPLAYSSM_PROFILE/{label}/native"):
        for repeat in range(args.repetitions):
            result = native_call()
            if repeat == 0:
                first_native_result = result
        torch.npu.synchronize()
    durations["native"] = time.perf_counter() - start
    print(f"PROFILE_END {label} native", flush=True)

    print(f"PROFILE_BEGIN {label} replay repetitions={args.repetitions}", flush=True)
    torch.npu.synchronize()
    start = time.perf_counter()
    with torch.profiler.record_function(f"REPLAYSSM_PROFILE/{label}/replay"):
        for repeat in range(args.repetitions):
            replay_call(first_out if repeat == 0 else out)
        torch.npu.synchronize()
    durations["replay"] = time.perf_counter() - start
    print(f"PROFILE_END {label} replay", flush=True)

    assert first_native_result is not None
    status_values = status.cpu()
    if torch.any(status_values != 0):
        raise RuntimeError(f"Replay status contains {status_values.unique().tolist()} for position {position}")
    difference = (first_native_result.float() - first_out.float()).abs()
    max_abs = difference.max().item()
    relative_l2 = (difference.norm() / first_native_result.float().norm().clamp_min(1e-12)).item()
    if not math.isfinite(max_abs) or max_abs > args.max_abs_error:
        raise RuntimeError(
            f"Native/Replay first-output mismatch at B={batch}, context={context_length}, pos={position}: "
            f"max_abs={max_abs:.6g}, relative_l2={relative_l2:.6g}"
        )

    return {
        "label": label,
        "tp_geometry": args.tp,
        "batch": batch,
        "context_length_label": context_length,
        "ring_position": position,
        "flush": flush,
        "repetitions_per_operator": args.repetitions,
        "warmup_per_operator": args.warmup,
        "num_slots": args.num_slots,
        "page_bytes": page_bytes,
        "tile_v": args.tile_v,
        "requested_cores": args.cores,
        "launched_blocks": min(args.cores, tasks),
        "tasks": tasks,
        "output_max_abs_error": max_abs,
        "output_relative_l2_error": relative_l2,
        "host_range_seconds_including_dispatch": durations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tp", type=int, choices=(4, 8), default=4)
    parser.add_argument("--batches", type=parse_ints, default=(1, 16, 32))
    parser.add_argument("--context-lengths", type=parse_ints, default=(4096, 8192, 32768))
    parser.add_argument("--positions", type=parse_ints, default=(0, 8, 15))
    parser.add_argument("--capacity", type=int, default=16)
    parser.add_argument("--num-slots", type=int, default=DEFAULT_NUM_SLOTS)
    parser.add_argument("--page-bytes", type=int, default=0, help="0 derives the 0.30 hybrid page size")
    parser.add_argument("--tile-v", type=int, choices=(16, 32, 64), default=32)
    parser.add_argument("--cores", type=int, default=8)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--max-abs-error", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.capacity <= 64 or any(pos >= args.capacity for pos in args.positions):
        parser.error("positions must be in [0, capacity), capacity in [1, 64]")
    if any(batch < 1 for batch in args.batches) or args.num_slots <= max(args.batches):
        parser.error("batches must be positive and num-slots must exceed the largest batch")
    if any(length < 1 for length in args.context_lengths) or args.cores < 1 or args.repetitions < 1 or args.warmup < 0:
        parser.error("context lengths, cores and repetitions must be positive; warmup must be nonnegative")
    if args.max_abs_error <= 0:
        parser.error("max-abs-error must be positive")

    import torch_npu  # noqa: F401
    from lab import load_framework

    torch.npu.set_device(args.device)
    _, extension_path = load_framework()
    # Fail early if the installed plugin does not expose the same native path
    # that vLLM Ascend uses when --use-replayssm is absent.
    from vllm_ascend.ops.gdn import recurrent_gated_delta_rule

    report = {
        "status": "RUNNING",
        "native_source": "vllm_ascend.ops.gdn.recurrent_gated_delta_rule",
        "native_implementation_module": recurrent_gated_delta_rule.__module__,
        "replay_source": "vllm_ascend.ops.replayssm.ascendc.replayssm_ar_step",
        "source_commit": source_commit(),
        "extension_path": str(extension_path),
        "packages": package_versions(),
        "note": "context_length labels the state seed; single-token GDN work does not scale with it",
        "cases": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for context_length in args.context_lengths:
        for batch in args.batches:
            for position in args.positions:
                try:
                    case_result = run_case(args, batch, context_length, position)
                except Exception:
                    report["status"] = "FAILED"
                    report["failed_case"] = {
                        "batch": batch,
                        "context_length": context_length,
                        "position": position,
                        "traceback": traceback.format_exc(),
                    }
                    args.output.write_text(json.dumps(report, indent=2) + "\n")
                    raise
                report["cases"].append(case_result)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
    report["status"] = "PASSED"
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"PASSED {len(report['cases'])} cases; report={args.output}", flush=True)


if __name__ == "__main__":
    main()
