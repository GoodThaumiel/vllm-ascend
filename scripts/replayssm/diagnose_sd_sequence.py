# SPDX-License-Identifier: Apache-2.0
"""Trace ReplaySSM SD numerical error across accepted windows and flushes.

The local reference starts each step from the *actual* BF16 checkpoint and
FP16 ring. The dense reference carries every accepted state in FP64. Comparing
both separates an operator/state-machine error from expected cache rounding.
This script is diagnostic and does not measure serving latency.
"""

from __future__ import annotations

import argparse
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

HEAD_DIM = 128
MAX_TOKENS = 4
RING_LEN = 32
LOGICAL_LIMIT = 20


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


def run(args: argparse.Namespace) -> dict:
    import torch_npu  # noqa: F401

    from vllm_ascend.ops.replayssm.ascendc import replayssm_sd_compact, replayssm_sd_cursor, replayssm_sd_verify
    from vllm_ascend.utils import enable_custom_op

    torch.set_num_threads(1)
    torch.npu.set_device(args.device)
    if not enable_custom_op():
        raise RuntimeError("ReplaySSM AscendC operators are unavailable; rebuild this checkout")
    device = torch.device(f"npu:{args.device}")
    key_heads, value_heads = 16 // args.tp, 48 // args.tp
    initial = make_case(tp=args.tp, batch=1, position=0, length=4, ring_len=RING_LEN, tile_v=128, seed=args.seed)
    replay = clone_page_strided_case(initial, device)
    replay["compact_status"] = torch.empty(value_heads * 4, dtype=torch.int32, device=device)
    replay["flush"] = torch.zeros(1, dtype=torch.int32, device=device)
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
        "tp": args.tp,
        "seed": args.seed,
        "accepted_pattern": args.accepted,
        "beta_native_precision": args.beta_native_precision,
        "atol": args.atol,
        "rtol": args.rtol,
        "steps": [],
    }
    for step in range(args.steps):
        replayssm_sd_cursor(
            cursor["write_pos"],
            cursor["cache_base"],
            cursor["is_flush"],
            cursor["accepted"],
            cursor["slots"],
            cursor["do_reset"],
            cursor["status"],
            LOGICAL_LIMIT,
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
        if position > 16 or position + MAX_TOKENS > LOGICAL_LIMIT or cursor_status:
            raise RuntimeError(f"cursor invalid at step {step}: pos={position}, status={cursor_status}")

        q = (torch.randn(MAX_TOKENS, key_heads, HEAD_DIM, generator=generator) * 0.09).bfloat16()
        k = (torch.randn(MAX_TOKENS, key_heads, HEAD_DIM, generator=generator) * 0.09).bfloat16()
        v = (torch.randn(MAX_TOKENS, value_heads, HEAD_DIM, generator=generator) * 0.09).bfloat16()
        g = -(torch.rand(MAX_TOKENS, value_heads, generator=generator) * 0.04)
        beta = torch.rand(MAX_TOKENS, value_heads, generator=generator) * 0.8
        if args.beta_native_precision:
            beta = beta.bfloat16().float()
        for name, tensor in (("q", q), ("k", k), ("v", v), ("g", g), ("beta", beta)):
            replay[name][0].copy_(tensor.to(device))
        replay["out"].fill_(float("nan"))
        replay["status"].fill_(-1)
        replay["compact_status"].fill_(-1)
        before = {name: replay[name].cpu() for name in initial}
        before["flush"] = replay["flush"].cpu()
        local = recurrent_golden(before)
        expected_compact = compact_golden({**local, "flush": before["flush"]})
        dense_output, dense_states = dense_window(dense_state, q, k, v, g, beta)

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
            "compact_status_max": int(replay["compact_status"].cpu().abs().max()),
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
    parser.add_argument("--tp", type=int, choices=(4, 8), default=4)
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
    if args.atol < 0 or args.rtol < 0:
        parser.error("atol and rtol must be nonnegative")
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = run(args)
    print(json.dumps({key: value for key, value in report.items() if key != "steps"}, indent=2))
    return 0 if report["status"] == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
