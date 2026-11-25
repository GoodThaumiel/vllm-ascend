# SPDX-License-Identifier: Apache-2.0
"""One same-input graph benchmark. Use --path to isolate msprof captures."""

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experimental.replayssm_cube_910b.projections import (  # noqa: E402
    checkpoint_projection,
    compact,
    compact_inputs,
    panel,
    verify,
)
from experimental.replayssm_sd_ascendc.lab import make_case  # noqa: E402
from scripts.replayssm.profile_sd_operator_compare import (  # noqa: E402
    native_state_page,
    package_versions,
    reconstruct_native_start,
)

PATHS = (
    "native",
    "v20",
    "compact_v1",
    "cube_verify",
    "cube_compact",
    "projection",
    "grouped_projection",
    "fold_gemm",
    "vector_pair",
    "cube_pair",
)
DEFAULT_BATCHES = (1, 2, 4, 8, 16, 32, 64, 128)


def batches(text):
    if text == "all":
        return range(1, 129)
    values = tuple(int(x) for x in text.split(","))
    if not values or min(values) < 1 or max(values) > 128 or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("expected unique batches in 1..128, or all")
    return values


def measure(call, reset, iterations):
    for _ in range(2):
        reset()
        call()
    reset()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        call()
    us = []
    for _ in range(iterations):
        reset()
        start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        us.append(start.elapsed_time(end) * 1000)
    return {"graph_us": us, "p50_us": statistics.median(us), "p90_us": sorted(us)[(9 * len(us) - 1) // 10]}


def trial(args, b, dev):
    from vllm_ascend.ops.gdn import recurrent_gated_delta_rule
    from vllm_ascend.ops.replayssm.ascendc import replayssm_sd_compact, replayssm_sd_verify

    p, t, base = args.pos, args.length, args.base
    active = {"none": 0, "mixed": max(1, b // 8), "full": b}[args.flush]
    cpu = make_case(tp=4, batch=b, position=p, length=t, base=base, tile_v=128, seed=42)
    cpu["beta"] = cpu["beta"].bfloat16().float()
    c = {n: v.to(dev) for n, v in cpu.items()}
    c["checkpoint"] = torch.cat((torch.zeros_like(c["checkpoint"][:1]).expand(4, -1, -1, -1), c["checkpoint"]))
    initial = c["checkpoint"].clone()
    ring = c["slots"]
    slots = ring + 4
    flush = (torch.arange(b, device=dev) < active).int()
    status = torch.empty(b * 12 * 4, dtype=torch.int32, device=dev)
    state, _ = native_state_page(dev, 4, b, t)
    start_state = reconstruct_native_start(cpu, p).to(dev)
    first = state[1::t]
    indices = torch.arange(1, b * t + 1, device=dev, dtype=torch.int32)
    lengths = torch.tensor([0] + [t] * b, device=dev, dtype=torch.int32)
    accepted = torch.ones(b, device=dev, dtype=torch.int32)
    inputs = [c[n][:, :t].contiguous().flatten(0, 1) for n in ("q", "k", "v", "g", "beta")]

    def native():
        return recurrent_gated_delta_rule(
            *inputs[:3],
            state,
            g=inputs[3],
            beta=inputs[4],
            scale=128**-0.5,
            actual_seq_lengths=lengths,
            ssm_state_indices=indices,
            num_accepted_tokens=accepted,
        )

    inputs[4] = inputs[4].bfloat16()

    def vector():
        replayssm_sd_verify(
            *(c[n] for n in ("q", "k", "v", "g", "beta", "checkpoint", "d_cache", "k_cache", "g_cache")),
            slots,
            c["positions"],
            c["bases"],
            c["lengths"],
            c["out"],
            c["status"],
            128**-0.5,
            128,
            40,
            20,
            ring,
        )

    def fold():
        replayssm_sd_compact(
            c["checkpoint"],
            c["d_cache"],
            c["k_cache"],
            c["g_cache"],
            slots,
            c["positions"],
            c["bases"],
            flush,
            status,
            32,
            40,
            1,
            ring,
        )

    def reset_state():
        c["checkpoint"].copy_(initial)

    # Candidate positions and committed entries are disjoint. Keep goldens on CPU outside timing.
    vector()
    assert not c["status"].cpu().any()
    golden = {n: c[n].cpu() for n in ("out", "d_cache", "k_cache", "g_cache")}
    first.copy_(start_state)
    y_native = native().view(b, t, 12, 128)
    fold()
    assert not status.cpu().any()
    checkpoint = c["checkpoint"].cpu()
    reset_state()
    # Poison only destinations, so a missing candidate write cannot inherit the vector result.
    c["out"].fill_(float("nan"))
    for s in range(t):
        for n in ("d_cache", "k_cache", "g_cache"):
            c[n][1:, :, (base + p + s) % 32].fill_(float("nan"))
    verify(c, t, p, base)
    errors = {}
    for n, ref in golden.items():
        actual = c[n].cpu()
        torch.testing.assert_close(actual.float(), ref.float(), atol=0.01, rtol=0.03, equal_nan=True)
        errors[n] = (actual.float() - ref.float()).nan_to_num().abs().max().item()
    torch.testing.assert_close(y_native.float().cpu(), golden["out"][:, :t].float(), atol=0.01, rtol=0.03)
    compact(c, active, p, base, status)
    torch.testing.assert_close(c["checkpoint"].cpu().float(), checkpoint.float(), atol=0.01, rtol=0.03)
    errors["checkpoint"] = (c["checkpoint"].cpu().float() - checkpoint.float()).abs().max().item()
    reset_state()
    rhs = panel(c, t)
    shared_rhs = panel(c, t, grouped=True)
    calls = dict(
        native=native,
        v20=vector,
        compact_v1=fold,
        cube_verify=lambda: verify(c, t, p, base),
        cube_compact=lambda: compact(c, active, p, base, status),
        grouped_projection=lambda: checkpoint_projection(c["checkpoint"][5:], shared_rhs),
        projection=lambda: checkpoint_projection(c["checkpoint"][5:], rhs),
        vector_pair=lambda: (vector(), fold()),
        cube_pair=lambda: (verify(c, t, p, base), compact(c, active, p, base, status)),
    )
    if active and p:
        _, dl, kh = compact_inputs(c, active, p, base)
        calls["fold_gemm"] = lambda: torch.bmm(dl, kh)
    rows = []
    for path in args.path:
        if path not in calls:
            continue
        reset = (
            (lambda: first.copy_(start_state))
            if path == "native"
            else reset_state
            if path in ("vector_pair", "compact_v1", "cube_pair", "cube_compact")
            else lambda: None
        )
        reset_state()
        print(f"MEASURE batch={b} path={path}", flush=True)
        rows.append(
            dict(
                batch=b,
                pos=p,
                length=t,
                base=base,
                active_flush=active,
                path=path,
                errors=errors,
                **measure(calls[path], reset, args.iterations),
            )
        )
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batches", type=batches, default=DEFAULT_BATCHES)
    ap.add_argument("--length", type=int, choices=(1, 4), default=4)
    ap.add_argument("--pos", type=int, choices=range(17), default=15)
    ap.add_argument("--base", type=int, choices=range(32), default=31)
    ap.add_argument("--flush", choices=("none", "mixed", "full"), default="full")
    ap.add_argument("--path", nargs="+", choices=PATHS, default=PATHS)
    ap.add_argument("--iterations", type=int, default=100)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.iterations < 1:
        ap.error("iterations must be positive")
    import torch_npu  # noqa: F401

    from vllm_ascend.utils import enable_custom_op

    torch.npu.set_device(args.device)
    if not enable_custom_op():
        raise RuntimeError("framework v20 + compact v1 extension required")
    report = dict(
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        packages=package_versions(),
        config=vars(args),
        soc=torch.npu.get_device_name(args.device),
        rows=[],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for b in args.batches:
        try:
            report["rows"].extend(trial(args, b, torch.device(f"npu:{args.device}")))
        except Exception as exc:
            report.update(failed_batch=b, error=repr(exc))
            raise
        finally:
            args.output.write_text(json.dumps(report, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
