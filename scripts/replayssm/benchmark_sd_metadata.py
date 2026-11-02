# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Measure five separate metadata reads against one packed read on NPU."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import statistics
import subprocess
import sys
import traceback
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experimental.replayssm_sd_ascendc.lab import pack_metadata_  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--cores", type=int, default=40)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--mode", choices=("separate", "packed", "both"), default="both")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not (1 <= args.batch <= 1024 and 1 <= args.heads <= 32 and 1 <= args.cores <= 40 and args.iterations > 0):
        parser.error("invalid batch, heads, cores, or iterations")
    report = {
        "status": "FAILED",
        "batch": args.batch,
        "heads": args.heads,
        "cores": args.cores,
        "mode": args.mode,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "measurements": {},
    }
    try:
        import torch_npu  # noqa: F401

        from vllm_ascend.utils import enable_custom_op

        torch.npu.set_device(args.device)
        if not enable_custom_op() or not hasattr(torch.ops._C_ascend, "replayssm_sd_metadata_probe_"):
            raise RuntimeError("rebuild the 910B extension with the metadata probe")
        library = Path(importlib.import_module("vllm_ascend.vllm_ascend_C").__file__).resolve()
        report["library"] = str(library)
        report["library_sha256"] = hashlib.sha256(library.read_bytes()).hexdigest()
        report["ascend_home_path"] = os.environ.get("ASCEND_HOME_PATH", "not set")
        report["packages"] = {}
        for package in ("torch", "torch-npu", "vllm", "vllm-ascend"):
            try:
                report["packages"][package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                report["packages"][package] = "not installed"
        device = f"npu:{args.device}"
        rows = torch.arange(args.batch, dtype=torch.int32, device=device)
        case = {"slots": rows + 101, "positions": rows % 17, "bases": rows % 32, "lengths": rows % 4 + 1}
        ring_slots = rows + 1
        packed = torch.zeros((args.batch, 8), dtype=torch.int32, device=device)
        out = torch.empty((args.batch * args.heads, 8), dtype=torch.int32, device=device)

        def pack():
            pack_metadata_(case, packed, ring_slots)

        def invoke(mode):
            torch.ops._C_ascend.replayssm_sd_metadata_probe_(
                case["slots"],
                ring_slots,
                case["positions"],
                case["bases"],
                case["lengths"],
                packed,
                out,
                args.heads,
                args.cores,
                mode,
            )

        def check():
            expected = torch.stack(
                (case["slots"], ring_slots, case["positions"], case["bases"], case["lengths"]), dim=1
            ).repeat_interleave(args.heads, dim=0)
            torch.testing.assert_close(out[:, :5], expected, rtol=0, atol=0)
            torch.testing.assert_close(out[:, 5:], torch.zeros_like(out[:, 5:]), rtol=0, atol=0)

        modes = ("separate", "packed") if args.mode == "both" else (args.mode,)
        for name in modes:
            mode = 0 if name == "separate" else 1
            case["positions"].copy_(rows % 17)
            pack()
            invoke(mode)
            torch.npu.synchronize()
            check()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                invoke(mode)
            # Exercise changing metadata after graph capture.
            case["positions"].add_(1)
            pack()
            graph.replay()
            torch.npu.synchronize()
            check()
            for _ in range(5):
                graph.replay()
            torch.npu.synchronize()
            samples = []
            print(f"PROFILE_BEGIN metadata_{name} batch={args.batch}", flush=True)
            for _ in range(args.iterations):
                start = torch.npu.Event(enable_timing=True)
                end = torch.npu.Event(enable_timing=True)
                start.record()
                graph.replay()
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end) * 1000)
            print(f"PROFILE_END metadata_{name} batch={args.batch}", flush=True)
            check()
            samples.sort()
            report["measurements"][name] = {
                "p50_us": statistics.median(samples),
                "p90_us": samples[int(0.9 * (len(samples) - 1))],
                "min_us": samples[0],
                "iterations": len(samples),
                "includes": "read metadata and write aligned per-task output; excludes packing",
            }
        report["status"] = "PASSED"
    except Exception as exc:
        report.update({"error": repr(exc), "traceback": traceback.format_exc()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if report["status"] != "PASSED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
