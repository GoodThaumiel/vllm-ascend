# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""On-device smoke, graph, and graph-replay timing for the SD cursor op."""

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
from lab import MUTABLE, clone_case, invoke, make_case, reference_step


def cases(batch: int, mtp_size: int) -> list[tuple[str, dict[str, torch.Tensor]]]:
    max_tokens = mtp_size + 1
    logical_limit = 16 + max_tokens

    def initial() -> dict[str, torch.Tensor]:
        return make_case(batch=batch, slots=max(batch + 1, 129), logical_limit=logical_limit, max_spec_len=max_tokens)

    normal = initial()
    normal["write_pos"][1 : batch + 1] = 7
    normal["accepted"][:] = 2

    boundary = initial()
    boundary["write_pos"][1 : batch + 1] = 16 - max_tokens
    boundary["accepted"][:] = max_tokens

    flush = initial()
    flush["write_pos"][1 : batch + 1] = 16
    flush["cache_base"][1 : batch + 1] = 29
    flush["is_flush"][1 : batch + 1] = 1
    flush["accepted"][:] = 2

    reset = initial()
    reset["write_pos"][1 : batch + 1] = 16
    reset["cache_base"][1 : batch + 1] = 31
    reset["is_flush"][1 : batch + 1] = 1
    reset["accepted"][:] = -1
    reset["do_reset"][:] = 1

    reject = initial()
    reject["write_pos"][1 : batch + 1] = 7
    reject["accepted"][:] = 0

    padded = initial()
    padded["slots"][-1] = 0
    padded["do_reset"][-1] = 1
    padded["accepted"][-1] = -1
    result = [
        ("normal_accept2", normal),
        ("boundary_accept4", boundary),
        ("flush_wrap", flush),
        ("first_decode_reset", reset),
        ("zero_accept_noop", reject),
        ("padding", padded),
    ]
    if batch >= 4:
        invalid = initial()
        invalid["slots"][0] = -1
        invalid["accepted"][1] = max_tokens + 1
        invalid["cache_base"][3] = -1
        invalid["write_pos"][4] = logical_limit - 1
        invalid["accepted"][3] = max_tokens
        result.append(("invalid_rows", invalid))
    return result


def compare(actual: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]) -> None:
    for name in MUTABLE:
        torch.testing.assert_close(actual[name].cpu(), expected[name], rtol=0, atol=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--library", type=Path, default=Path(__file__).resolve().parent / "build/libreplayssm_sd_cursor_torch.so"
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--stage", choices=("smoke", "graph", "perf"), default="smoke")
    parser.add_argument("--backend", choices=("lab", "framework"), default="lab")
    parser.add_argument("--mtp-size", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    max_tokens = args.mtp_size + 1
    logical_limit = 16 + max_tokens
    if args.batch < 1 or args.batch > 128 or args.iterations < 1:
        parser.error("batch must be 1..128 and iterations must be positive")

    report = {
        "stage": args.stage,
        "device": args.device,
        "batch": args.batch,
        "backend": args.backend,
        "mtp_size": args.mtp_size,
        "status": "FAILED",
        "cases": [],
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], text=True
        ).strip(),
        "packages": {},
    }
    for name in ("torch", "torch-npu", "vllm", "vllm-ascend"):
        try:
            report["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report["packages"][name] = "not installed"
    try:
        import torch_npu  # noqa: F401

        torch.npu.set_device(args.device)
        if args.backend == "lab":
            library = args.library.resolve()
            torch.ops.load_library(str(library))
        else:
            from vllm_ascend.utils import enable_custom_op

            if not enable_custom_op() or not hasattr(torch.ops._C_ascend, "replayssm_sd_cursor_"):
                raise RuntimeError("Rebuild vllm_ascend_C with Replay SD cursor for Ascend910B")
            library = Path(importlib.import_module("vllm_ascend.vllm_ascend_C").__file__).resolve()
        report["library"] = str(library)
        report["library_sha256"] = hashlib.sha256(library.read_bytes()).hexdigest()
        device = torch.device(f"npu:{args.device}")
        scenarios = cases(args.batch, args.mtp_size)
        allocated = clone_case(scenarios[0][1], device)
        if args.stage in ("graph", "perf"):
            invoke(allocated, logical_limit, max_tokens, backend=args.backend)
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                invoke(allocated, logical_limit, max_tokens, backend=args.backend)

        for name, source in scenarios:
            for key, tensor in source.items():
                allocated[key].copy_(tensor.to(device))
            if args.stage == "smoke":
                invoke(allocated, logical_limit, max_tokens, backend=args.backend)
            else:
                graph.replay()
            torch.npu.synchronize()
            compare(allocated, reference_step(source, logical_limit=logical_limit, max_spec_len=max_tokens))
            report["cases"].append({"name": name, "status": "PASSED"})

        if args.stage == "perf":
            timings = {}
            for name, source in scenarios[:4]:
                source_device = clone_case(source, device)
                samples = []
                for _ in range(args.iterations):
                    for key in MUTABLE:
                        allocated[key].copy_(source_device[key])
                    start = torch.npu.Event(enable_timing=True)
                    end = torch.npu.Event(enable_timing=True)
                    start.record()
                    graph.replay()
                    end.record()
                    end.synchronize()
                    samples.append(start.elapsed_time(end) * 1000)
                compare(allocated, reference_step(source, logical_limit=logical_limit, max_spec_len=max_tokens))
                ordered = sorted(samples)
                timings[name] = {
                    "device_us_p50": statistics.median(samples),
                    "device_us_p90": ordered[int(0.9 * (len(ordered) - 1))],
                    "device_us_min": ordered[0],
                    "iterations": len(samples),
                    "reset_in_timed_region": False,
                }
            report["graph_replay_us"] = timings
        report["status"] = "PASSED"
    except Exception:
        report["error"] = traceback.format_exc()
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"{report['status']} report={args.output}", flush=True)


if __name__ == "__main__":
    main()
