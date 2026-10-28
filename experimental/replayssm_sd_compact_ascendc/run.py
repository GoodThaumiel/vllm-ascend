# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Independent Ascend910B4 smoke, graph and timing for Replay SD compaction."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import statistics
import subprocess
import traceback
from pathlib import Path

import torch
from lab import (
    CACHE_NAMES,
    clone_case,
    clone_page_strided_case,
    compact_golden,
    compare,
    invoke,
    make_case,
    pack_metadata_views,
)

ROOT = Path(__file__).resolve().parents[2]


def cases(tp: int, batch: int, tile_v: int):
    for pos, base, flush in ((0, 0, 1), (13, 31, 1), (16, 31, 1), (13, 30, 0)):
        case = make_case(tp=tp, batch=batch, position=pos, base=base, tile_v=tile_v, seed=991 + pos + base)
        case["flush"].fill_(flush)
        yield f"pos{pos}_base{base}_flush{flush}", case
    if batch >= 2:
        case = make_case(tp=tp, batch=batch, position=13, base=31, tile_v=tile_v, seed=1151)
        case["flush"][0] = 0
        yield "mixed_flush", case
        case = make_case(tp=tp, batch=batch, position=13, base=31, tile_v=tile_v, seed=1201)
        last = batch - 1
        for name in ("slots", "positions", "bases", "flush"):
            case[name][last] = 0
        yield "mixed_padding", case


def compare_case(cpu, gpu):
    torch.npu.synchronize()
    return compare(clone_case(gpu), compact_golden(cpu))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--library", type=Path, default=Path(__file__).resolve().parent / "build/libreplayssm_sd_compact_torch.so"
    )
    parser.add_argument("--stage", choices=("smoke", "graph", "perf", "all"), default="smoke")
    parser.add_argument("--backend", choices=("lab", "framework"), default="lab")
    parser.add_argument("--tp", type=int, choices=(4, 8), default=4)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--tile-v", type=int, choices=(16, 32), default=32)
    parser.add_argument("--cores", type=int, choices=(8, 20, 40), default=40)
    parser.add_argument("--variant", type=int, choices=(0, 1, 2, 3, 4), default=1)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.batch < 1 or args.batch > 1024 or args.iterations < 1:
        parser.error("batch must be 1..1024 and iterations positive")

    import torch_npu  # noqa: F401

    torch.npu.set_device(args.device)
    device = torch.device(f"npu:{args.device}")
    if args.backend == "lab":
        library = args.library.resolve()
        torch.ops.load_library(str(library))
    else:
        from vllm_ascend.utils import enable_custom_op

        if not enable_custom_op() or not hasattr(torch.ops._C_ascend, "replayssm_sd_compact_"):
            raise RuntimeError("Rebuild vllm_ascend_C with Replay SD compact for Ascend910B")
        library = Path(importlib.import_module("vllm_ascend.vllm_ascend_C").__file__).resolve()
    try:
        source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        source_commit = "unavailable"
    report = {
        "stage": args.stage,
        "status": "RUNNING",
        "source_commit": source_commit,
        "library": str(library),
        "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        "backend": args.backend,
        "device": args.device,
        "tp": args.tp,
        "batch": args.batch,
        "tile_v": args.tile_v,
        "cores": args.cores,
        "variant": args.variant,
        "metadata_pack_in_timed_region": False,
        "packages": {name: importlib.metadata.version(name) for name in ("torch", "torch-npu", "vllm", "vllm-ascend")},
        "cases": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for label, cpu in cases(args.tp, args.batch, args.tile_v):
            gpu = clone_page_strided_case(cpu, device)
            if args.variant == 4:
                gpu = pack_metadata_views(gpu)
            if args.stage in ("smoke", "all"):
                invoke(gpu, tile_v=args.tile_v, cores=args.cores, variant=args.variant, backend=args.backend)
                errors = compare_case(cpu, gpu)
                report["cases"].append({"case": label, "stage": "smoke", "errors": errors})
            if args.stage in ("graph", "perf", "all"):
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    invoke(gpu, tile_v=args.tile_v, cores=args.cores, variant=args.variant, backend=args.backend)
                for name in cpu:
                    gpu[name].copy_(cpu[name].to(device))
                graph.replay()
                errors = compare_case(cpu, gpu)
                record = {"case": label, "stage": "graph", "errors": errors}
                if args.stage in ("perf", "all"):
                    pristine = {name: cpu[name].to(device) for name in (*CACHE_NAMES, "status")}
                    samples = []
                    for _ in range(args.iterations):
                        for name, tensor in pristine.items():
                            gpu[name].copy_(tensor)
                        start = torch.npu.Event(enable_timing=True)
                        end = torch.npu.Event(enable_timing=True)
                        start.record()
                        graph.replay()
                        end.record()
                        end.synchronize()
                        samples.append(start.elapsed_time(end) * 1000)
                    ordered = sorted(samples)
                    record["graph_replay_us"] = {
                        "p50": statistics.median(samples),
                        "p90": ordered[int(0.9 * (len(ordered) - 1))],
                        "min": ordered[0],
                        "iterations": len(samples),
                        "reset_in_timed_region": False,
                    }
                    compare_case(cpu, gpu)
                report["cases"].append(record)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
        report["status"] = "PASSED"
    except Exception:
        report["status"] = "FAILED"
        report["error"] = traceback.format_exc()
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        raise
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"PASSED {len(report['cases'])} cases; report={args.output}", flush=True)


if __name__ == "__main__":
    main()
