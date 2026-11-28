# SPDX-License-Identifier: Apache-2.0
"""Trace ReplaySSM SD numerical error across accepted windows and flushes.

The local reference starts each step from the *actual* BF16 checkpoint and
FP16 ring. The dense reference carries every accepted state in FP64. Comparing
both separates an operator/state-machine error from expected cache rounding.
The CPU reference mode compares checkpoint schedules without an NPU.
This script is diagnostic and does not measure serving latency.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experimental.replayssm_sd_ascendc.lab import (  # noqa: E402
    clone_page_strided_case,
    make_case,
    recurrent_golden,
)
from experimental.replayssm_sd_compact_ascendc.lab import compact_golden  # noqa: E402
from experimental.replayssm_sd_cursor_ascendc import lab as cursor_lab  # noqa: E402

HEAD_DIM = 128
MAX_TOKENS = 4
RING_LEN = 32
LOGICAL_LIMIT = 20
MIXED_CORES = 20
MIXED_PANEL_WIDTH = 16
MIXED_FUSED_FOLD = 4
GEOMETRIES = {"27b": (4, 8), "9b": (1, 2)}


def compare(actual: torch.Tensor, expected: torch.Tensor, *, atol: float, rtol: float) -> dict:
    a, e = actual.float().cpu(), expected.float().cpu()
    finite = torch.isfinite(a) & torch.isfinite(e)
    mismatch = ~torch.isclose(a, e, atol=atol, rtol=rtol, equal_nan=True)
    error = (a[finite] - e[finite]).abs()
    first = torch.nonzero(mismatch, as_tuple=False)
    return {
        "max_abs": float(error.max()) if error.numel() else None,
        "mismatch_count": int(mismatch.sum()),
        "first_mismatch_index": first[0].tolist() if first.numel() else None,
        "all_close": not bool(mismatch.any()),
    }


def dense_window(
    state: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """FP64 accepted-state recurrence; returns four candidate outputs/states."""
    group = v.shape[1] // q.shape[1]
    states: list[torch.Tensor] = []
    outputs: list[torch.Tensor] = []
    for token in range(MAX_TOKENS):
        key = k[token].double().repeat_interleave(group, dim=0)
        query = q[token].double().repeat_interleave(group, dim=0) * (HEAD_DIM**-0.5)
        decayed = torch.exp(g[token].double())[:, None, None] * state
        delta = beta[token].double()[:, None] * (v[token].double() - torch.einsum("hvk,hk->hv", decayed, key))
        state = decayed + delta[:, :, None] * key[:, None, :]
        outputs.append(torch.einsum("hvk,hk->hv", state, query).bfloat16())
        states.append(state)
    return torch.stack(outputs), states


def candidate_inputs(args, initial, generator):
    def sample(name, maximum, uniform=False):
        template = initial[name][0]
        draw = torch.rand if uniform else torch.randn
        return (draw(template.shape, generator=generator) * maximum).to(template.dtype)

    values = {
        "q": sample("q", 0.09),
        "k": sample("k", 0.09),
        "v": sample("v", 0.09),
        "g": -sample("g", args.gate_max, uniform=True),
        "beta": sample("beta", 0.8, uniform=True),
    }
    if args.beta_native_precision:
        values["beta"] = values["beta"].bfloat16().float()
    return values


def initial_case(args):
    return make_case(
        geometry=args.geometry,
        tp=args.tp,
        batch=1,
        position=0,
        length=MAX_TOKENS,
        ring_len=RING_LEN,
        tile_v=HEAD_DIM,
        seed=args.seed,
    )


def run_cpu_reference(args: argparse.Namespace) -> dict:
    # Use the production cursor contract and the independent golden ring and
    # folding equations. Neither this path nor the dense references import torch_npu.
    torch.set_num_threads(1)
    case = initial_case(args)
    cursor = cursor_lab.make_case(batch=1, slots=2, logical_limit=args.logical_limit)
    cursor["do_reset"].fill_(1)
    cursor["accepted"].zero_()
    generator = torch.Generator().manual_seed(args.seed + 1)
    exact_state = case["checkpoint"][1].double()
    stored_state = exact_state.clone()
    rows = []

    for step in range(args.steps):
        cursor = cursor_lab.reference_step(cursor, logical_limit=args.logical_limit)
        cursor["do_reset"].zero_()
        position = int(cursor["write_pos"][1])
        base = int(cursor["cache_base"][1])
        flush = int(cursor["is_flush"][1]) & cursor_lab.FLUSH_BIT
        if int(cursor["status"][0]) or position + MAX_TOKENS > args.logical_limit:
            raise RuntimeError(f"Invalid cursor at step {step}: position={position}")
        case["positions"].fill_(position)
        case["bases"].fill_(base)
        case["flush"] = torch.full((1,), flush, dtype=torch.int32)
        candidates = candidate_inputs(args, case, generator)
        for name, value in candidates.items():
            case[name][0].copy_(value)
        verified = recurrent_golden(case)
        case = compact_golden(verified) if flush else verified
        actual = case["out"][0]
        values = tuple(candidates[name] for name in ("q", "k", "v", "g", "beta"))
        exact_out, exact_states = dense_window(exact_state, *values)
        stored_out, stored_states = dense_window(stored_state, *values)
        if not torch.isfinite(actual).all():
            raise RuntimeError(f"Nonfinite replay output at step {step}")
        accepted = args.accepted[step % len(args.accepted)]
        exact_state = exact_states[accepted - 1]
        stored_state = stored_states[accepted - 1].bfloat16().double()
        cursor["accepted"].fill_(accepted)
        rows.append(
            {
                "step": step,
                "accepted": accepted,
                "position": position,
                "base": base,
                "flush": flush,
                "output_vs_dense_fp64": compare(actual, exact_out, atol=args.atol, rtol=args.rtol),
                "output_vs_dense_bf16_boundary": compare(actual, stored_out, atol=args.atol, rtol=args.rtol),
                "dense_bf16_boundary_vs_fp64": compare(stored_out, exact_out, atol=args.atol, rtol=args.rtol),
            }
        )

    summary = {}
    for metric in ("output_vs_dense_fp64", "output_vs_dense_bf16_boundary", "dense_bf16_boundary_vs_fp64"):
        summary[metric] = {
            "max_abs": max(row[metric]["max_abs"] for row in rows),
            "steps_outside_tolerance": sum(not row[metric]["all_close"] for row in rows),
        }
    return {
        "status": "REFERENCE_COMPLETED",
        "backend": "cpu_reference",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "torch": torch.__version__,
        "geometry": args.geometry,
        "tp": args.tp,
        "seed": args.seed,
        "gate_max": args.gate_max,
        "accepted_pattern": args.accepted,
        "beta_native_precision": args.beta_native_precision,
        "logical_limit": args.logical_limit,
        "atol": args.atol,
        "rtol": args.rtol,
        "flush_count": sum(row["flush"] for row in rows),
        "max_pending_history": max(row["position"] for row in rows),
        "summary": summary,
        "steps": rows,
    }


def run(args: argparse.Namespace) -> dict:
    import torch_npu  # noqa: F401

    from vllm_ascend.ops.replayssm.ascendc import replayssm_sd_compact, replayssm_sd_cursor, replayssm_sd_verify
    from vllm_ascend.utils import enable_custom_op

    torch.set_num_threads(1)
    torch.npu.set_device(args.device)
    if not enable_custom_op():
        raise RuntimeError("ReplaySSM AscendC operators are unavailable; rebuild this checkout")
    device = torch.device(f"npu:{args.device}")
    initial = initial_case(args)
    value_heads = initial["v"].shape[2]
    replay = clone_page_strided_case(initial, device)
    replay["flush"] = torch.zeros(1, dtype=torch.int32, device=device)
    mixed_library = args.mixed_library
    if mixed_library is None:
        replay["compact_status"] = torch.empty(value_heads * 4, dtype=torch.int32, device=device)
    else:
        mixed_library = mixed_library.resolve(strict=True)
        torch.ops.load_library(str(mixed_library))
        for name in ("d_cache", "k_cache", "g_cache"):
            replay[name] = initial[name].to(device)
        zeros = torch.zeros((MIXED_PANEL_WIDTH, HEAD_DIM), dtype=torch.bfloat16, device=device)
        blocks = min(MIXED_CORES, initial["q"].shape[2])
        workspace = torch.empty((blocks, HEAD_DIM, MIXED_PANEL_WIDTH), dtype=torch.float32, device=device)
        offsets = torch.tensor([0, MAX_TOKENS], dtype=torch.int32, device=device)
    cursor = {
        "write_pos": torch.zeros(2, dtype=torch.int32, device=device),
        "cache_base": torch.zeros(2, dtype=torch.int32, device=device),
        "is_flush": torch.zeros(2, dtype=torch.int8, device=device),
        "accepted": torch.zeros(1, dtype=torch.int32, device=device),
        "slots": replay["slots"],
        "do_reset": torch.ones(1, dtype=torch.int8, device=device),
        "status": torch.empty(1, dtype=torch.int32, device=device),
    }
    generator = torch.Generator().manual_seed(args.seed + 1)
    dense_state = initial["checkpoint"][1].double().clone()
    cumulative_accepted = 0
    report = {
        "status": "RUNNING",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "packages": {name: importlib.metadata.version(name) for name in ("torch", "torch-npu", "vllm", "vllm-ascend")},
        "geometry": args.geometry,
        "tp": args.tp,
        "seed": args.seed,
        "logical_limit": args.logical_limit,
        "gate_max": args.gate_max,
        "accepted_pattern": args.accepted,
        "beta_native_precision": args.beta_native_precision,
        "atol": args.atol,
        "rtol": args.rtol,
        "steps": [],
    }
    if mixed_library is not None:
        report["mixed_library_sha256"] = hashlib.sha256(mixed_library.read_bytes()).hexdigest()
    report["backend"] = "mixed_fused_fold" if mixed_library is not None else "framework_v13_compact_v1"
    for step in range(args.steps):
        replayssm_sd_cursor(
            cursor["write_pos"],
            cursor["cache_base"],
            cursor["is_flush"],
            cursor["accepted"],
            cursor["slots"],
            cursor["do_reset"],
            cursor["status"],
            args.logical_limit,
            MAX_TOKENS,
            RING_LEN,
        )
        cursor["do_reset"].zero_()
        replay["positions"].copy_(cursor["write_pos"][1:2])
        replay["bases"].copy_(cursor["cache_base"][1:2])
        replay["flush"].copy_(cursor["is_flush"][1:2].to(torch.int32))
        torch.npu.synchronize()
        position = int(replay["positions"].cpu()[0])
        base = int(replay["bases"].cpu()[0])
        flush = int(replay["flush"].cpu()[0])
        cursor_status = int(cursor["status"].cpu()[0])
        if position > 16 or position + MAX_TOKENS > args.logical_limit or cursor_status:
            raise RuntimeError(f"cursor invalid at step {step}: pos={position}, status={cursor_status}")

        candidates = candidate_inputs(args, initial, generator)
        for name, tensor in candidates.items():
            replay[name][0].copy_(tensor.to(device))
        replay["out"].fill_(float("nan"))
        replay["status"].fill_(-1)
        if mixed_library is None:
            replay["compact_status"].fill_(-1)
        before = {name: replay[name].cpu() for name in initial}
        before["flush"] = replay["flush"].cpu()
        local = recurrent_golden(before)
        expected_compact = compact_golden({**local, "flush": before["flush"]})
        dense_output, dense_states = dense_window(dense_state, **candidates)

        if mixed_library is None:
            replayssm_sd_verify(
                replay["q"],
                replay["k"],
                replay["v"],
                replay["g"],
                replay["beta"],
                replay["checkpoint"],
                replay["d_cache"],
                replay["k_cache"],
                replay["g_cache"],
                replay["slots"],
                replay["positions"],
                replay["bases"],
                replay["lengths"],
                replay["out"],
                replay["status"],
                HEAD_DIM**-0.5,
                128,
                40,
                13,
            )
            replayssm_sd_compact(
                replay["checkpoint"],
                replay["d_cache"],
                replay["k_cache"],
                replay["g_cache"],
                replay["slots"],
                replay["positions"],
                replay["bases"],
                replay["flush"],
                replay["compact_status"],
                32,
                40,
                1,
            )
        else:
            torch.ops.replayssm_mix_910b.verify_(
                *(replay[name][0] for name in ("q", "k", "v", "g", "beta")),
                replay["checkpoint"],
                replay["d_cache"],
                replay["k_cache"],
                replay["g_cache"],
                replay["slots"],
                replay["slots"],
                replay["positions"],
                replay["bases"],
                replay["lengths"],
                replay["out"][0],
                replay["status"],
                zeros,
                workspace,
                MIXED_CORES,
                True,
                False,
                True,
                False,
                MIXED_FUSED_FOLD,
                replay["flush"],
                offsets,
            )
        torch.npu.synchronize()
        actual = replay["out"][0].cpu()
        accepted = args.accepted[step % len(args.accepted)]
        candidate_physical = [(base + position + token) & (RING_LEN - 1) for token in range(MAX_TOKENS)]
        local_cache = {
            name: compare(
                replay[name][1, :, candidate_physical].cpu(),
                local[name][1, :, candidate_physical],
                atol=args.atol,
                rtol=args.rtol,
            )
            for name in ("d_cache", "k_cache", "g_cache")
        }
        row = {
            "step": step,
            "cumulative_accepted_before": cumulative_accepted,
            "accepted": accepted,
            "position": position,
            "base": base,
            "flush": flush,
            "cursor_status": cursor_status,
            "verify_status_max": int(replay["status"].cpu().abs().max()),
            "compact_status_max": int(replay["compact_status"].cpu().abs().max()) if mixed_library is None else None,
            "output_vs_local": compare(actual, local["out"][0], atol=args.atol, rtol=args.rtol),
            "output_vs_dense": compare(actual, dense_output, atol=args.atol, rtol=args.rtol),
            "checkpoint_vs_local": compare(
                replay["checkpoint"][1].cpu(), expected_compact["checkpoint"][1], atol=args.atol, rtol=args.rtol
            ),
            "candidate_cache_vs_local": local_cache,
        }
        report["steps"].append(row)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        dense_state = dense_states[accepted - 1]
        cumulative_accepted += accepted
        cursor["accepted"].fill_(accepted)
    report["first_local_mismatch_step"] = next(
        (
            row["step"]
            for row in report["steps"]
            if row["verify_status_max"]
            or row["compact_status_max"]
            or not row["output_vs_local"]["all_close"]
            or not row["checkpoint_vs_local"]["all_close"]
            or any(not value["all_close"] for value in row["candidate_cache_vs_local"].values())
        ),
        None,
    )
    report["status"] = "PASSED" if report["first_local_mismatch_step"] is None else "LOCAL_REFERENCE_MISMATCH"
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", choices=GEOMETRIES, default="27b")
    parser.add_argument("--tp", type=int, choices=(1, 2, 4, 8), default=4)
    backend = parser.add_mutually_exclusive_group()
    backend.add_argument("--cpu-reference", action="store_true")
    backend.add_argument("--mixed-library", type=Path, help="Use the packed grouped-fused verify library")
    parser.add_argument(
        "--logical-limit", type=int, choices=range(2 * MAX_TOKENS, LOGICAL_LIMIT + 1), default=LOGICAL_LIMIT
    )
    parser.add_argument("--gate-max", type=float, default=0.04)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--accepted", type=str, default="3,1,4,2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--beta-native-precision", action="store_true")
    parser.add_argument("--atol", type=float, required=True)
    parser.add_argument("--rtol", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.accepted = [int(item) for item in args.accepted.split(",")]
    if args.steps < 1 or not args.accepted or any(not 1 <= item <= 4 for item in args.accepted):
        parser.error("steps must be positive and accepted counts must be 1..4")
    if args.tp not in GEOMETRIES[args.geometry] or (args.mixed_library and args.tp != GEOMETRIES[args.geometry][0]):
        parser.error("unsupported geometry/TP combination for this backend")
    if args.atol < 0 or args.rtol < 0 or not 0 <= args.gate_max <= 1:
        parser.error("atol and rtol must be nonnegative and gate-max must be 0..1")
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.cpu_reference:
        report = run_cpu_reference(args)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    else:
        report = run(args)
    print(json.dumps({key: value for key, value in report.items() if key != "steps"}, indent=2))
    return 0 if report["status"] in ("PASSED", "REFERENCE_COMPLETED") else 1


if __name__ == "__main__":
    raise SystemExit(main())
