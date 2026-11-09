"""Compare 310P LoadData/Mmad and native-style vector matvec microprobes."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
import traceback
from pathlib import Path

import torch
import torch_npu


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mtp", type=int, choices=(3, 7), required=True)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--stage", choices=("smoke", "perf"), default="smoke")
    parser.add_argument("--tiles", type=int, default=0, help="override tile count; default perf = B*16*(MTP+1)*8")
    parser.add_argument("--cores", type=int, choices=range(1, 9), default=8)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--library", type=Path, default=Path(__file__).parent / "build/librgdr_v310_cube_torch.so")
    parser.add_argument(
        "--vector-library", type=Path, default=Path(__file__).parent / "build/librgdr_v310_vec_torch.so"
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def graph_perf(call, args: argparse.Namespace) -> dict:
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        call()
    for _ in range(args.warmup):
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
    samples = []
    for _ in range(args.iterations):
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
    ordered = sorted(samples)
    return {
        "p50_us": statistics.median(ordered),
        "p90_us": ordered[int(0.9 * (len(ordered) - 1))],
        "min_us": ordered[0],
        "iterations": len(ordered),
        "timing_source": source,
    }


def run(args: argparse.Namespace) -> dict:
    torch.npu.set_device(args.device)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    torch.ops.load_library(str(args.library.resolve()))
    torch.ops.load_library(str(args.vector_library.resolve()))
    device = torch.device(f"npu:{args.device}")
    tiles = args.tiles or (32 if args.stage == "smoke" else args.batch * 16 * (args.mtp + 1) * 8)
    gen = torch.Generator().manual_seed(310128)
    a_cpu = (torch.randn((tiles, 16, 128), generator=gen) * 0.02).half()
    vec_cpu = (torch.randn((tiles, 128), generator=gen) * 0.09).half()
    b_cpu = torch.zeros((tiles, 128, 16), dtype=torch.float16)
    b_cpu[:, :, 0] = vec_cpu
    a, b, vec = a_cpu.to(device), b_cpu.to(device), vec_cpu.to(device)
    cube_out = torch.empty((tiles, 16, 16), device=device, dtype=torch.float32)
    vector_out = torch.empty((tiles, 16), device=device, dtype=torch.float32)

    def call_cube() -> None:
        torch.ops.rgdr_v310_cube_probe.matvec_(a, b, cube_out, args.cores)

    def call_vector() -> None:
        torch.ops.rgdr_v310_ablation.matvec_(a, vec, vector_out, min(args.cores, 7))

    call_vector()
    call_cube()
    torch.npu.synchronize()
    cube_actual = cube_out.cpu()
    vector_actual = vector_out.cpu()
    # The probe keeps N=16 to satisfy Mmad's cube tile. Only column 0 is live.
    expected_first = (a_cpu.float() * vec_cpu.float()[:, None, :]).sum(dim=-1)
    expected = torch.zeros_like(cube_actual)
    expected[:, :, 0] = expected_first
    torch.testing.assert_close(cube_actual, expected, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(vector_actual, expected_first, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(cube_actual[:, :, 0], vector_actual, atol=2e-3, rtol=2e-3)
    result: dict = {
        "tiles": tiles,
        "input_bytes": {
            "matrix": a.numel() * a.element_size(),
            "vector": vec.numel() * vec.element_size(),
            "cube_padded_rhs": b.numel() * b.element_size(),
        },
        "correctness": {
            "cube_max_abs": float((cube_actual - expected).abs().max()),
            "vector_max_abs": float((vector_actual - expected_first).abs().max()),
            "cube_vector_max_abs": float((cube_actual[:, :, 0] - vector_actual).abs().max()),
            "unused_columns_max_abs": float(cube_actual[:, :, 1:].abs().max()),
        },
    }
    if args.stage == "smoke":
        return result

    result["graph_perf"] = {
        "vector": graph_perf(call_vector, args),
        "cube": graph_perf(call_cube, args),
    }
    result["graph_perf"]["cube_over_vector_p50"] = (
        result["graph_perf"]["cube"]["p50_us"] / result["graph_perf"]["vector"]["p50_us"]
    )
    return result


def main() -> None:
    args = parse_args()
    report = {
        "status": "FAILED",
        "stage": args.stage,
        "mtp": args.mtp,
        "batch": args.batch,
        "cores": args.cores,
        "device": args.device,
    }
    try:
        report["source_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], text=True
        ).strip()
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
