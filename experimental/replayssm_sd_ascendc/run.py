# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""On-device smoke, graph, and timing for the isolated SD verify operator."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import statistics
import subprocess
import traceback
from pathlib import Path

from lab import (
    CACHE_NAMES,
    check_result,
    clone_case,
    clone_page_strided_case,
    invoke,
    make_case,
    recurrent_golden,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--library", type=Path, default=Path(__file__).resolve().parent / "build/libreplayssm_sd_torch.so"
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--backend", choices=("lab", "framework"), default="lab")
    parser.add_argument("--tp", type=int, choices=(4, 8), default=4)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--tile-v", type=int, choices=(16, 32, 64), default=32)
    parser.add_argument("--cores", type=int, choices=(8, 20, 40), default=40)
    parser.add_argument("--variant", type=int, choices=(0, 1, 2, 3, 4, 5, 6), default=0)
    parser.add_argument("--layout", choices=("compact", "page_strided"), default="page_strided")
    parser.add_argument("--stage", choices=("smoke", "graph", "perf", "all"), default="smoke")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _cpu_case(args: argparse.Namespace, pos: int, length: int, base: int, seed: int, padding: bool = False):
    case = make_case(
        tp=args.tp,
        batch=args.batch,
        position=pos,
        length=length,
        base=base,
        tile_v=args.tile_v,
        seed=seed,
        padding=padding,
    )
    if padding and args.batch >= 2:
        case["lengths"][0] = 1
    return case


def _device_case(case, device, layout):
    return (clone_page_strided_case if layout == "page_strided" else clone_case)(case, device)


def _run_once(args, device, pos, length, base, seed, padding=False):
    import torch

    cpu = _cpu_case(args, pos, length, base, seed, padding=padding)
    expected = recurrent_golden(cpu)
    gpu = _device_case(cpu, device, args.layout)
    invoke(gpu, tile_v=args.tile_v, cores=args.cores, variant=args.variant, backend=args.backend)
    torch.npu.synchronize()
    actual = clone_case(gpu)
    return check_result(actual, expected)


def _graph_case(args, device, perf=False):
    import torch

    cpu = _cpu_case(args, 7, 4, 29, 157)
    gpu = _device_case(cpu, device, args.layout)
    invoke(gpu, tile_v=args.tile_v, cores=args.cores, variant=args.variant, backend=args.backend)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        invoke(gpu, tile_v=args.tile_v, cores=args.cores, variant=args.variant, backend=args.backend)
    records = []
    for pos, length, base, padding in (
        (0, 1, 0, False),
        (7, 2, 29, False),
        (7, 3, 30, False),
        (7, 4, 29, False),
        (12, 4, 31, False),
        (16, 4, 31, False),
        (7, 4, 30, True),
    ):
        cpu = _cpu_case(args, pos, length, base, seed=160 + pos, padding=padding)
        for name in cpu:
            gpu[name].copy_(cpu[name].to(device))
        graph.replay()
        torch.npu.synchronize()
        records.append(
            {
                "position": pos,
                "length": length,
                "base": base,
                "padding": padding,
                "errors": check_result(clone_case(gpu), recurrent_golden(cpu)),
            }
        )
    if not perf:
        return {"graph_cases": records}

    timings = {}
    for pos, length, base in ((0, 1, 0), (7, 4, 29), (12, 4, 31), (16, 4, 31)):
        pristine = _cpu_case(args, pos, length, base, seed=177 + pos)
        pristine_device = {name: tensor.to(device) for name, tensor in pristine.items()}
        for name in pristine:
            gpu[name].copy_(pristine_device[name])
        graph.replay()
        torch.npu.synchronize()
        samples = []
        for _ in range(args.iterations):
            for name in (*CACHE_NAMES, "out", "status"):
                gpu[name].copy_(pristine_device[name])
            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) * 1000)
        check_result(clone_case(gpu), recurrent_golden(pristine))
        ordered = sorted(samples)
        timings[f"pos{pos}_length{length}"] = {
            "p50": statistics.median(samples),
            "p90": ordered[int(0.9 * (len(ordered) - 1))],
            "min": ordered[0],
            "iterations": len(samples),
            "reset_in_timed_region": False,
        }
    return {
        "graph_cases": records,
        "graph_replay_us": timings,
    }


def main() -> None:
    args = parse_args()
    report = {
        "stage": args.stage,
        "device": args.device,
        "tp": args.tp,
        "batch": args.batch,
        "tile_v": args.tile_v,
        "cores": args.cores,
        "variant": args.variant,
        "effective_cores": min(
            args.cores,
            args.batch * (48 // args.tp) * (1 if args.variant >= 3 else (128 // args.tile_v)),
        ),
        "layout": args.layout,
        "backend": args.backend,
        "status": "FAILED",
        "cases": [],
    }
    repo_root = Path(__file__).resolve().parents[2]
    try:
        report["source_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        report["source_commit"] = "unavailable"
    report["packages"] = {}
    for name in ("torch", "torch-npu", "vllm", "vllm-ascend", "triton-ascend"):
        try:
            report["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report["packages"][name] = "not installed"
    report["ascend_home_path"] = os.environ.get("ASCEND_HOME_PATH", "not set")
    try:
        import torch
        import torch_npu  # noqa: F401

        torch.npu.set_device(args.device)
        if args.backend == "lab":
            torch.ops.load_library(str(args.library.resolve()))
            library = args.library.resolve()
        else:
            from vllm_ascend.utils import enable_custom_op

            if not enable_custom_op() or not hasattr(torch.ops._C_ascend, "replayssm_sd_verify_"):
                raise RuntimeError("Rebuild vllm_ascend_C with Replay SD for Ascend910B")
            library = Path(importlib.import_module("vllm_ascend.vllm_ascend_C").__file__).resolve()
        report["library"] = str(library)
        report["library_sha256"] = hashlib.sha256(library.read_bytes()).hexdigest()
        device = f"npu:{args.device}"
        if args.stage in ("smoke", "all"):
            for pos, length, base, padding in (
                (0, 1, 0, False),
                (7, 2, 29, False),
                (7, 3, 30, False),
                (7, 4, 29, False),
                (12, 4, 31, False),
                (16, 4, 31, False),
                (7, 4, 30, True),
            ):
                errors = _run_once(args, device, pos, length, base, 101 + pos, padding=padding)
                report["cases"].append(
                    {
                        "position": pos,
                        "length": length,
                        "base": base,
                        "padding": padding,
                        "status": "PASSED",
                        "errors": errors,
                    }
                )
        if args.stage in ("graph", "perf", "all"):
            report.update(_graph_case(args, device, perf=args.stage in ("perf", "all")))
        report["status"] = "PASSED"
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(
        json.dumps({k: report[k] for k in ("stage", "device", "tp", "batch", "cores", "status")}, ensure_ascii=False),
        flush=True,
    )
    if report["status"] != "PASSED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
