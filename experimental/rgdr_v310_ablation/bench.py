"""Isolated 310P native GDR ablations: MTP3/7, batch 32, one graph launch per iteration."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import statistics
import subprocess
import time
import traceback
from pathlib import Path

import torch
import torch_npu

ROOT = Path(__file__).resolve().parents[2]
HEADS_K = 8
HEADS_V = 16
DIM = 128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("native", "0", "1", "2", "3"), required=True)
    parser.add_argument("--mtp", type=int, choices=(3, 7), required=True)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--cores", type=int, choices=range(1, 8), default=7)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--library", type=Path, default=Path(__file__).parent / "build/librgdr_v310_vec_torch.so")
    parser.add_argument("--golden", type=Path)
    parser.add_argument("--native-golden", type=Path, help="native output/state for tolerance comparison")
    parser.add_argument("--write-golden", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def make_inputs(args: argparse.Namespace, device: torch.device) -> dict[str, torch.Tensor]:
    tokens = args.mtp + 1
    total = args.batch * tokens
    gen = torch.Generator().manual_seed(3102026)

    def random(shape: tuple[int, ...], scale: float) -> torch.Tensor:
        return (torch.randn(shape, generator=gen) * scale).half()

    q = random((total, HEADS_K, DIM), 0.09)
    k = random((total, HEADS_K, DIM), 0.09)
    v = random((total, HEADS_V, DIM), 0.09)
    beta = torch.rand((total, HEADS_V), generator=gen).mul(0.8).half()
    g = torch.rand((total, HEADS_V), generator=gen).mul(-0.04)
    state = torch.zeros((total + 1, HEADS_V, DIM, DIM), dtype=torch.float16)
    accepted = torch.arange(args.batch, dtype=torch.int32).remainder(tokens).add(1)
    for row in range(args.batch):
        slot = 1 + row * tokens + int(accepted[row]) - 1
        state[slot] = random((HEADS_V, DIM, DIM), 0.02)
    cpu = {
        "q": q,
        "k": k,
        "v": v,
        "beta": beta,
        "g": g,
        "state": state,
        "lengths": torch.full((args.batch,), tokens, dtype=torch.int32),
        "indices": torch.arange(1, total + 1, dtype=torch.int32),
        "accepted": accepted,
    }
    return {name: value.to(device) for name, value in cpu.items()}


def elapsed_us(samples: list[float], source: str) -> dict[str, float | int | str]:
    ordered = sorted(samples)
    return {
        "p50_us": statistics.median(ordered),
        "p90_us": ordered[int(0.9 * (len(ordered) - 1))],
        "min_us": ordered[0],
        "iterations": len(samples),
        "timing_source": source,
    }


def compare(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float | int]:
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise AssertionError("shape or dtype differs from vector variant 0")
    actual = actual.cpu()
    expected = expected.cpu()
    mismatches = actual.view(torch.int16) != expected.view(torch.int16)
    difference = (actual.float() - expected.float()).abs()
    return {
        "exact_mismatch_count": int(mismatches.sum()),
        "max_abs": float(difference.max()),
    }


def compare_native(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float | int]:
    summary = compare(actual, expected)
    torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=2e-3, rtol=2e-3)
    return summary


def run(args: argparse.Namespace) -> dict:
    torch.npu.set_device(args.device)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    device = torch.device(f"npu:{args.device}")
    if args.variant == "native":
        from vllm_ascend.utils import enable_custom_op

        if not enable_custom_op():
            raise RuntimeError("native V310 custom op is unavailable")
    else:
        torch.ops.load_library(str(args.library.resolve()))
    inp = make_inputs(args, device)
    original_state = inp["state"].clone()
    output = torch.empty_like(inp["v"])

    def call() -> torch.Tensor:
        if args.variant == "native":
            return torch.ops._C_ascend.npu_recurrent_gated_delta_rule_310(
                query=inp["q"],
                key=inp["k"],
                value=inp["v"],
                beta=inp["beta"],
                state=inp["state"],
                actual_seq_lengths=inp["lengths"],
                ssm_state_indices=inp["indices"],
                g=inp["g"],
                gk=None,
                num_accepted_tokens=inp["accepted"],
                scale_value=DIM**-0.5,
            )
        torch.ops.rgdr_v310_ablation.run_(
            inp["q"],
            inp["k"],
            inp["v"],
            inp["beta"],
            inp["state"],
            inp["lengths"],
            inp["indices"],
            inp["g"],
            inp["accepted"],
            output,
            DIM**-0.5,
            int(args.variant),
            args.cores,
        )
        return output

    inp["state"].copy_(original_state)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        graph_output = call()

    for _ in range(args.warmup):
        inp["state"].copy_(original_state)
        graph.replay()
    torch.npu.synchronize()

    source = "npu_event"
    try:
        start_probe = torch.npu.Event(enable_timing=True)
        end_probe = torch.npu.Event(enable_timing=True)
        start_probe.record()
        end_probe.record()
        end_probe.synchronize()
        start_probe.elapsed_time(end_probe)
    except RuntimeError as error:
        if "107006" not in str(error) and "timestamp invalid" not in str(error):
            raise
        source = "wall_clock"

    samples: list[float] = []
    for _ in range(args.iterations):
        inp["state"].copy_(original_state)
        if source == "npu_event":
            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) * 1000)
        else:
            torch.npu.synchronize()
            start_ns = time.perf_counter_ns()
            graph.replay()
            torch.npu.synchronize()
            samples.append((time.perf_counter_ns() - start_ns) / 1000)

    inp["state"].copy_(original_state)
    graph.replay()
    torch.npu.synchronize()
    actual_output = graph_output.cpu()
    actual_state = inp["state"].cpu()
    result = {"graph_perf": elapsed_us(samples, source)}
    if args.native_golden:
        native = torch.load(args.native_golden, map_location="cpu", weights_only=True)
        result["native_correctness"] = {
            "output": compare_native(actual_output, native["output"]),
            "state": compare_native(actual_state, native["state"]),
        }
    if args.golden:
        golden = torch.load(args.golden, map_location="cpu", weights_only=True)
        result["correctness"] = {
            "output": compare(actual_output, golden["output"]),
            "state": compare(actual_state, golden["state"]),
        }
        if any(value["exact_mismatch_count"] for value in result["correctness"].values()):
            raise AssertionError(
                f"variant {args.variant} differs bitwise from vector variant 0: {result['correctness']}"
            )
    if args.write_golden:
        args.write_golden.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"output": actual_output, "state": actual_state}, args.write_golden)
    return result


def main() -> None:
    args = parse_args()
    report = {
        "status": "FAILED",
        "variant": args.variant,
        "mtp": args.mtp,
        "batch": args.batch,
        "cores": args.cores if args.variant != "native" else "host tiling",
        "device": args.device,
        "scope": "isolated native V310 op; Q/K normalization and cache reset excluded",
    }
    try:
        report["source_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        for package in ("torch", "torch-npu", "vllm", "vllm-ascend"):
            try:
                report[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                report[package] = "not installed"
        report["device_name"] = torch.npu.get_device_name(args.device)
        report.update(run(args))
        report["status"] = "PASSED"
    except Exception:
        report["error"] = traceback.format_exc()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False))
    if report["status"] != "PASSED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
