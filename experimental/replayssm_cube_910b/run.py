# SPDX-License-Identifier: Apache-2.0
"""One same-input graph benchmark. Use --path to isolate msprof captures."""

import argparse
import hashlib
import importlib
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

VECTOR_PATHS = ("native", "v20", "compact_v1", "vector_pair")
MIXED_PATHS = ("mixed_verify", "mixed_pair")
CUBE_PATHS = (
    "cube_verify",
    "cube_compact",
    "projection",
    "grouped_projection",
    "fold_gemm",
    "cube_pair",
)
PATHS = (*VECTOR_PATHS, *CUBE_PATHS)
DEFAULT_BATCHES = (1, 2, 4, 8, 16, 32, 64, 128)


def batches(text):
    if text == "all":
        return tuple(range(1, 129))
    try:
        values = tuple(int(x) for x in text.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated batches in 1..128, or all") from exc
    if not values or min(values) < 1 or max(values) > 128 or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("expected unique batches in 1..128, or all")
    return values


def measure(call, reset, iterations, label, check_graph=None):
    for _ in range(2):
        reset()
        call()
    reset()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        call()
    torch.npu.synchronize()
    us = []
    print(f"MEASURE_BEGIN {label} replays={iterations}", flush=True)
    with torch.profiler.record_function(f"REPLAYSSM_SD/{label}"):
        for _ in range(iterations):
            reset()
            start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            us.append(start.elapsed_time(end) * 1000)
    print(f"MEASURE_END {label}", flush=True)
    if check_graph is not None:
        check_graph(graph)
    return {"graph_us": us, "p50_us": statistics.median(us), "p90_us": sorted(us)[(9 * len(us) - 1) // 10]}


def last_projections(case, blocks):
    """Reference the last head produced in each AIC's reusable workspace slot."""
    batch, _, hk, _ = case["k"].shape
    group = case["v"].shape[2] // hk
    first = torch.arange(blocks)
    tasks = first + ((batch * hk - 1 - first) // blocks) * blocks
    rows, keys = tasks // hk, tasks % hk
    checkpoint = case["checkpoint"][rows + 1, keys * group + group - 1].float()
    rhs = torch.zeros((blocks, 128, 16), dtype=torch.float32)
    rhs[:, :, :4] = case["k"][rows, :, keys].transpose(1, 2)
    rhs[:, :, 4:8] = case["q"][rows, :, keys].transpose(1, 2)
    return torch.bmm(checkpoint, rhs)


def trial(args, b, dev):
    from vllm_ascend.ops.gdn import recurrent_gated_delta_rule
    from vllm_ascend.ops.replayssm.ascendc import replayssm_sd_compact, replayssm_sd_verify

    p, t, base = args.pos, args.length, args.base
    active = {"none": 0, "mixed": max(1, b // 8), "full": b}[args.flush]
    cpu = make_case(geometry=args.geometry, tp=args.tp, batch=b, position=p, length=t, base=base, tile_v=128, seed=42)
    key_heads, value_heads = cpu["k"].shape[2], cpu["v"].shape[2]
    cpu["beta"] = cpu["beta"].bfloat16().float()
    c = {n: v.to(dev) for n, v in cpu.items()}
    c["checkpoint"] = torch.cat((torch.zeros_like(c["checkpoint"][:1]).expand(4, -1, -1, -1), c["checkpoint"]))
    initial = c["checkpoint"].clone()
    ring = c["slots"]
    slots = ring + 4
    flush = (torch.arange(b, device=dev) < active).int()
    status = torch.full((b * value_heads * 4,), -1, dtype=torch.int32, device=dev)
    state, state_page_bytes = native_state_page(dev, args.tp, b, t, geometry=args.geometry)
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
        # Only the active prefix is mutable. Keep the same inputs for every replay.
        if p and active:
            c["checkpoint"][5 : 5 + active].copy_(initial[5 : 5 + active])

    # Native materializes the committed ring in its initial state. Replay reads
    # the same history directly and writes its candidates into disjoint slots.
    vector()
    if c["status"].cpu().any():
        raise RuntimeError("V20 returned nonzero status")
    expected_out = c["out"][:, :t].float().cpu()
    first.copy_(start_state)
    y_native = native().view_as(expected_out).float().cpu()
    torch.testing.assert_close(y_native, expected_out, atol=0.01, rtol=0.03)
    errors = {"native_vs_v20_out": (y_native - expected_out).abs().max().item()}
    fold()
    if status.cpu().any():
        raise RuntimeError("Compact V1 returned nonzero status")
    checkpoint = c["checkpoint"].cpu()
    if p and active:
        torch.testing.assert_close(checkpoint[5 : 5 + active], start_state[:active].cpu(), atol=0.001, rtol=0.015)
    unchanged = 5 + (active if p else 0)
    torch.testing.assert_close(checkpoint[unchanged:], cpu["checkpoint"][unchanged - 4 :], atol=0, rtol=0)
    reset_state()

    calls = dict(native=native, v20=vector, compact_v1=fold, vector_pair=lambda: (vector(), fold()))
    compare_demo = any(path in CUBE_PATHS for path in args.path)
    compare_mixed = any(path in MIXED_PATHS for path in args.path)
    if compare_demo or compare_mixed:
        # Poison destinations so a missing Cube candidate write cannot inherit V20 results.
        golden = {n: c[n].cpu() for n in ("out", "d_cache", "k_cache", "g_cache")}

        def poison():
            c["out"].fill_(float("nan"))
            c["status"].fill_(-1)
            for s in range(t):
                for n in ("d_cache", "k_cache", "g_cache"):
                    c[n][1:, :, (base + p + s) % 32].fill_(float("nan"))

        def check_outputs(label, error_prefix=""):
            if c["status"].cpu().any():
                raise RuntimeError(f"{label} returned nonzero status")
            for n, ref in golden.items():
                actual = c[n].cpu().float()
                expected = ref.float()
                torch.testing.assert_close(actual, expected, atol=0.01, rtol=0.03, equal_nan=True)
                errors[f"{error_prefix}{n}"] = (actual - expected).nan_to_num().abs().max().item()

    if compare_demo:
        poison()
        verify(c, t, p, base)
        check_outputs("cube_demo")
        compact(c, active, p, base, status)
        actual = c["checkpoint"].cpu().float()
        torch.testing.assert_close(actual, checkpoint.float(), atol=0.01, rtol=0.03)
        errors["checkpoint"] = (actual - checkpoint.float()).abs().max().item()
        reset_state()
        rhs = panel(c, t)
        shared_rhs = panel(c, t, grouped=True)
        calls.update(
            cube_verify=lambda: verify(c, t, p, base),
            cube_compact=lambda: compact(c, active, p, base, status),
            grouped_projection=lambda: checkpoint_projection(c["checkpoint"][5:], shared_rhs),
            projection=lambda: checkpoint_projection(c["checkpoint"][5:], rhs),
            cube_pair=lambda: (verify(c, t, p, base), compact(c, active, p, base, status)),
        )
        if active and p:
            _, dl, kh = compact_inputs(c, active, p, base)
            calls["fold_gemm"] = lambda: torch.bmm(dl, kh)

    if compare_mixed:
        blocks = min(b * key_heads, args.mixed_cores)
        zeros = torch.zeros((16, 128), dtype=torch.bfloat16, device=dev)
        scratch = torch.full((blocks, 128, 16), float("nan"), dtype=torch.float32, device=dev)
        expected_projection = last_projections(cpu, blocks)

        def verify_mixed():
            torch.ops.replayssm_mix_910b.verify_(
                *(c[n] for n in ("q", "k", "v", "g", "beta", "checkpoint", "d_cache", "k_cache", "g_cache")),
                slots,
                ring,
                c["positions"],
                c["bases"],
                c["lengths"],
                c["out"],
                c["status"],
                zeros,
                scratch,
                args.mixed_cores,
            )

        poison()
        verify_mixed()
        torch.testing.assert_close(scratch.cpu(), expected_projection, atol=2e-5, rtol=1e-3)
        check_outputs("mixed eager", "mixed_eager_")
        torch.testing.assert_close(c["checkpoint"], initial, atol=0, rtol=0)
        calls.update(mixed_verify=verify_mixed, mixed_pair=lambda: (verify_mixed(), fold()))
    rows = []
    for path in args.path:
        if path not in calls:
            continue
        reset = (
            (lambda: first.copy_(start_state))
            if path == "native"
            else reset_state
            if path in ("vector_pair", "compact_v1", "cube_pair", "cube_compact", "mixed_pair")
            else lambda: None
        )
        reset_state()
        label = f"{args.geometry}_tp{args.tp}_b{b}_t{t}_pos{p}_base{base}_{args.flush}/{path}"
        mix_metadata = {}
        if path in MIXED_PATHS:

            def check_graph(graph, reset=reset, path=path):
                reset()
                poison()
                scratch.fill_(float("nan"))
                if path == "mixed_pair":
                    status.fill_(-1)
                graph.replay()
                torch.testing.assert_close(scratch.cpu(), expected_projection, atol=2e-5, rtol=1e-3)
                check_outputs("mixed graph", "mixed_graph_")
                expected_state = checkpoint if path == "mixed_pair" else initial.cpu()
                torch.testing.assert_close(c["checkpoint"].cpu(), expected_state, atol=0, rtol=0)
                if path == "mixed_pair" and status.cpu().any():
                    raise RuntimeError("Captured compact returned nonzero status")

            timings = measure(calls[path], reset, args.iterations, label, check_graph=check_graph)
            mix_metadata = dict(aic_blocks=blocks, aiv_blocks=2 * blocks, workspace_bytes=scratch.nbytes)
        else:
            timings = measure(calls[path], reset, args.iterations, label)
        rows.append(
            dict(
                batch=b,
                pos=p,
                length=t,
                base=base,
                active_flush=active,
                key_heads=key_heads,
                value_heads=value_heads,
                native_state_page_bytes=state_page_bytes,
                path=path,
                errors=dict(errors),
                **timings,
                **mix_metadata,
            )
        )
    return rows


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", choices=("all", "vector", "mixed"), default="all")
    ap.add_argument("--geometry", choices=("27b", "9b"), default="27b")
    ap.add_argument("--batches", type=batches)
    ap.add_argument("--length", type=int, choices=(1, 4), default=4)
    ap.add_argument("--pos", type=int, choices=range(17), default=15)
    ap.add_argument("--base", type=int, choices=range(32), default=31)
    ap.add_argument("--flush", choices=("none", "mixed", "full"), default="full")
    ap.add_argument("--path", nargs="+", choices=(*PATHS, *MIXED_PATHS))
    ap.add_argument("--mixed-library", type=Path, help="Path to libreplayssm_sd_mix_torch.so")
    ap.add_argument("--mixed-cores", type=int, choices=range(1, 21), default=20)
    ap.add_argument("--iterations", type=int, default=100)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.iterations < 1:
        ap.error("iterations must be positive")
    vector_only = args.suite == "vector" or (args.geometry == "9b" and args.suite == "all")
    if args.suite == "mixed":
        defaults = (*VECTOR_PATHS, *MIXED_PATHS)
    else:
        defaults = VECTOR_PATHS if vector_only else PATHS
    args.batches = args.batches or (DEFAULT_BATCHES if defaults == PATHS else batches("all"))
    args.path = args.path or defaults
    if (args.suite != "all" or args.geometry == "9b") and any(path in CUBE_PATHS for path in args.path):
        ap.error("the PyTorch Cube demo uses the 27B TP4 all suite")
    if any(path in MIXED_PATHS for path in args.path) and not args.mixed_library:
        ap.error("mixed paths require --mixed-library")
    if args.suite == "vector" and any(path in MIXED_PATHS for path in args.path):
        ap.error("use --suite mixed for AIC/AIV paths")
    if len(set(args.path)) != len(args.path):
        ap.error("paths must be unique")
    args.tp = 4 if args.geometry == "27b" else 1
    return args


def main():
    args = parse_args()
    import torch_npu  # noqa: F401

    from vllm_ascend.utils import enable_custom_op

    torch.npu.set_device(args.device)
    if not enable_custom_op():
        raise RuntimeError("framework v20 + compact v1 extension required")
    library = Path(importlib.import_module("vllm_ascend.vllm_ascend_C").__file__).resolve()
    report = dict(
        status="RUNNING",
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        packages=package_versions(),
        config=vars(args),
        soc=torch.npu.get_device_name(args.device),
        library=str(library),
        library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
        rows=[],
    )
    if any(path in MIXED_PATHS for path in args.path):
        mixed_library = args.mixed_library.resolve(strict=True)
        torch.ops.load_library(str(mixed_library))
        report.update(
            mixed_library=str(mixed_library),
            mixed_library_sha256=hashlib.sha256(mixed_library.read_bytes()).hexdigest(),
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for b in args.batches:
        try:
            report["rows"].extend(trial(args, b, torch.device(f"npu:{args.device}")))
            if b == args.batches[-1]:
                report["status"] = "PASSED"
        except Exception as exc:
            report.update(status="FAILED", failed_batch=b, error=repr(exc))
            raise
        finally:
            args.output.write_text(json.dumps(report, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
