# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Profile native speculative RGDR against framework-packaged Replay SD verify.

Run under an external NPU profiler. Each labelled range contains only 100
recurrent calls by default; preparation, restoration and correctness checks
are outside the ranges. This is a non-flush operator comparison, not serving.
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
sys.path.insert(0, str(ROOT / "experimental/replayssm_sd_ascendc"))
from lab import CACHE_NAMES, clone_page_strided_case, make_case  # noqa: E402

HEAD_DIM = 128
PAGE_ALIGNMENT_BYTES = 256 * 1024


def positive_ints(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated positive integers") from exc
    if not result or len(result) != len(set(result)) or any(item < 1 for item in result):
        raise argparse.ArgumentTypeError("expected unique positive integers")
    return result


def nonnegative_ints(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated nonnegative integers") from exc
    if not result or len(result) != len(set(result)) or any(item < 0 for item in result):
        raise argparse.ArgumentTypeError("expected unique nonnegative integers")
    return result


def aligned_bytes(size: int) -> int:
    return math.ceil(size / PAGE_ALIGNMENT_BYTES) * PAGE_ALIGNMENT_BYTES


def native_state_page(device: torch.device, tp: int, batch: int, length: int):
    """Use unique token slots and a server-like per-slot state stride."""
    key_heads, value_heads = 16 // tp, 48 // tp
    conv_bytes = 3 * (value_heads + 2 * key_heads) * HEAD_DIM * 2
    state_bytes = value_heads * HEAD_DIM * HEAD_DIM * 2
    page_bytes = aligned_bytes(conv_bytes + state_bytes)
    num_slots = batch * length + 1
    raw = torch.empty(num_slots * page_bytes, dtype=torch.uint8, device=device)
    view = torch.as_strided(
        raw.view(torch.bfloat16),
        size=(num_slots, value_heads, HEAD_DIM, HEAD_DIM),
        stride=(page_bytes // 2, HEAD_DIM * HEAD_DIM, HEAD_DIM, 1),
        storage_offset=conv_bytes // 2,
    )
    return view, page_bytes


def reconstruct_native_start(cpu: dict[str, torch.Tensor], position: int) -> torch.Tensor:
    """Convert the Replay checkpoint and committed ring into one BF16 state."""
    state = cpu["checkpoint"][1:].float().clone()
    value_heads = cpu["v"].shape[2]
    key_heads = cpu["k"].shape[2]
    group = value_heads // key_heads
    base = int(cpu["bases"][0])
    ring = cpu["d_cache"].shape[2]
    for i in range(position):
        phys = (base + i) & (ring - 1)
        alpha = cpu["g_cache"][1:, :, phys].exp()
        delta = cpu["d_cache"][1:, :, phys].float()
        key = cpu["k_cache"][1:, :, phys].float().repeat_interleave(group, dim=1)
        state = alpha[..., None, None] * state + delta[..., None] * key[..., None, :]
    return state.bfloat16()


def package_versions() -> dict[str, str]:
    versions = {}
    for name in ("torch", "torch-npu", "vllm", "vllm-ascend", "triton-ascend"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def run_case(args: argparse.Namespace, batch: int, context: int, length: int, position: int) -> dict:
    from vllm_ascend.ops.gdn import recurrent_gated_delta_rule
    from vllm_ascend.ops.replayssm.ascendc import replayssm_sd_verify

    device = torch.device(f"npu:{args.device}")
    seed = args.seed + batch * 1000 + position * 10 + length
    cpu = make_case(
        tp=args.tp,
        batch=batch,
        position=position,
        length=length,
        base=args.base,
        ring_len=args.ring,
        tile_v=args.tile_v,
        seed=seed,
    )
    # Match native BF16 beta quantization at the recurrent boundary.
    beta_native_cpu = cpu["beta"].bfloat16()
    beta_native = beta_native_cpu.to(device)
    cpu["beta"] = beta_native_cpu.float()
    replay = clone_page_strided_case(cpu, device)
    native_state, native_page_bytes = native_state_page(device, args.tp, batch, length)
    native_start = reconstruct_native_start(cpu, position).to(device)
    state_indices = torch.arange(1, batch * length + 1, dtype=torch.int32, device=device)
    actual_lengths = torch.tensor([0] + [length] * batch, dtype=torch.int32, device=device)
    accepted = torch.full((batch,), args.accepted, dtype=torch.int32, device=device)
    q = replay["q"][:, :length].contiguous().view(batch * length, 16 // args.tp, HEAD_DIM)
    k = replay["k"][:, :length].contiguous().view_as(q)
    v = replay["v"][:, :length].contiguous().view(batch * length, 48 // args.tp, HEAD_DIM)
    g = replay["g"][:, :length].contiguous().view(batch * length, 48 // args.tp)
    beta = beta_native[:, :length].contiguous().view_as(g)
    scale = HEAD_DIM**-0.5

    def reset_native() -> None:
        for row in range(batch):
            native_state[1 + row * length + args.accepted - 1].copy_(native_start[row])

    def reset_replay() -> None:
        for name in (*CACHE_NAMES, "out", "status"):
            replay[name].copy_(cpu[name].to(device))

    def native_call() -> torch.Tensor:
        # Exact vllm_ascend.ops.gdn speculative recurrent symbol and arguments.
        return recurrent_gated_delta_rule(
            q,
            k,
            v,
            native_state,
            g=g,
            beta=beta,
            scale=scale,
            actual_seq_lengths=actual_lengths,
            ssm_state_indices=state_indices,
            num_accepted_tokens=accepted,
        )

    def replay_call() -> None:
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
            args.variant,
        )

    reset_native()
    reset_replay()
    torch.npu.synchronize()
    native_first = native_call()
    replay_call()
    torch.npu.synchronize()
    if torch.any(replay["status"].cpu() != 0):
        raise RuntimeError("Replay SD status is nonzero")
    native_first = native_first.view(batch, length, 48 // args.tp, HEAD_DIM)
    replay_first = replay["out"][:, :length]
    diff = (native_first.float() - replay_first.float()).abs()
    max_abs = diff.max().item()
    relative_l2 = (diff.norm() / native_first.float().norm().clamp_min(1e-12)).item()
    if not math.isfinite(max_abs) or max_abs > args.max_abs_error:
        raise RuntimeError(f"Native/Replay output mismatch: max_abs={max_abs:.6g}, relative_l2={relative_l2:.6g}")

    for _ in range(args.warmup):
        native_call()
        replay_call()
    torch.npu.synchronize()
    reset_native()
    reset_replay()
    torch.npu.synchronize()

    label = f"tp{args.tp}_b{batch}_ctx{context}_t{length}_a{args.accepted}_pos{position}_c{args.cores}_v{args.variant}"
    durations = {}
    for name, call in (("native", native_call), ("replay", replay_call)):
        print(f"PROFILE_BEGIN {label} {name} repetitions={args.repetitions}", flush=True)
        torch.npu.synchronize()
        start = time.perf_counter()
        with torch.profiler.record_function(f"REPLAYSSM_SD/{label}/{name}"):
            for _ in range(args.repetitions):
                call()
            torch.npu.synchronize()
        durations[name] = time.perf_counter() - start
        print(f"PROFILE_END {label} {name}", flush=True)

    value_heads = 48 // args.tp
    state_bytes = batch * value_heads * HEAD_DIM * HEAD_DIM * 2
    replay_page_bytes = replay["checkpoint"].stride(0) * 2
    return {
        "label": label,
        "tp_geometry": args.tp,
        "batch": batch,
        "context_length_label": context,
        "verify_tokens_per_request": length,
        "previous_accepted_tokens": args.accepted,
        "history_position": position,
        "ring_base": args.base,
        "ring_length": args.ring,
        "repetitions_per_path": args.repetitions,
        "warmup_per_path": args.warmup,
        "tile_v": args.tile_v,
        "requested_cores": args.cores,
        "variant": args.variant,
        "effective_replay_cores": min(
            args.cores, batch * value_heads * (1 if args.variant >= 3 else HEAD_DIM // args.tile_v)
        ),
        "output_max_abs_error": max_abs,
        "output_relative_l2_error": relative_l2,
        "host_range_seconds_including_dispatch": durations,
        "native_state_slots": batch * length + 1,
        "native_state_page_bytes": native_page_bytes,
        "native_state_allocated_bytes": (batch * length + 1) * native_page_bytes,
        "replay_state_slots": batch + 1,
        "replay_state_page_bytes": replay_page_bytes,
        "replay_state_allocated_bytes": (batch + 1) * replay_page_bytes,
        "native_state_read_write_lower_bound_bytes": 2 * state_bytes,
        "replay_checkpoint_read_lower_bound_bytes": state_bytes,
        "note": "Profiler counters, not these byte bounds, determine measured roofline traffic.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tp", type=int, choices=(4, 8), default=4)
    parser.add_argument("--batches", type=positive_ints, default=(1, 16, 32, 64))
    parser.add_argument("--context-lengths", type=positive_ints, default=(32768,))
    parser.add_argument("--lengths", type=positive_ints, default=(1, 4))
    parser.add_argument("--accepted", type=int, default=1)
    parser.add_argument("--positions", type=nonnegative_ints, default=(0, 7, 12))
    parser.add_argument("--base", type=int, default=29)
    parser.add_argument("--ring", type=int, choices=(16, 32, 64), default=32)
    parser.add_argument("--tile-v", type=int, choices=(16, 32, 64), default=32)
    parser.add_argument("--cores", type=int, choices=(8, 20, 40), default=40)
    parser.add_argument("--variant", type=int, choices=range(8), default=0)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--max-abs-error", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(length > 4 for length in args.lengths) or any(pos > 16 for pos in args.positions):
        parser.error("lengths must be 1..4 and positions 0..16")
    if args.accepted < 1 or args.accepted > min(args.lengths):
        parser.error("accepted must be between 1 and every requested verify length")
    if not 0 <= args.base < args.ring or args.repetitions < 1 or args.warmup < 0 or args.max_abs_error <= 0:
        parser.error("invalid base, repetitions, warmup, or error threshold")
    import torch_npu  # noqa: F401

    from vllm_ascend.utils import enable_custom_op

    torch.npu.set_device(args.device)
    if not enable_custom_op() or not hasattr(torch.ops._C_ascend, "replayssm_sd_verify_"):
        raise RuntimeError("rebuild the 910B vllm_ascend_C extension: replayssm_sd_verify_ is absent")
    report = {
        "status": "RUNNING",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "packages": package_versions(),
        "native_source": "vllm_ascend.ops.gdn.recurrent_gated_delta_rule",
        "replay_source": "vllm_ascend.ops.replayssm.ascendc.replayssm_sd_verify",
        "replay_variant": args.variant,
        "note": (
            "Context length is a label only: recurrent tensor shapes and seed are identical across labels. "
            "Non-flush recurrent core only; no causal conv, gating, L2 norm, cursor commit, or HTTP serving."
        ),
        "cases": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for context in args.context_lengths:
        for batch in args.batches:
            for length in args.lengths:
                for position in args.positions:
                    if position + length > min(20, args.ring):
                        continue
                    try:
                        case = run_case(args, batch, context, length, position)
                    except Exception:
                        report["status"] = "FAILED"
                        report["failed_case"] = {
                            "context": context,
                            "batch": batch,
                            "length": length,
                            "position": position,
                            "traceback": traceback.format_exc(),
                        }
                        args.output.write_text(json.dumps(report, indent=2) + "\n")
                        raise
                    report["cases"].append(case)
                    args.output.write_text(json.dumps(report, indent=2) + "\n")
    report["status"] = "PASSED"
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"PASSED {len(report['cases'])} cases; report={args.output}", flush=True)


if __name__ == "__main__":
    main()
