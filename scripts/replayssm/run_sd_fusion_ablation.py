# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Compare serial v13 verify + v1 compact against one fused MTP3 27B kernel."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import statistics
import subprocess
import sys
import traceback
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experimental.replayssm_sd_ascendc.lab import (  # noqa: E402
    CACHE_NAMES,
    clone_case,
    clone_page_strided_case,
    make_case,
    recurrent_golden,
)
from experimental.replayssm_sd_compact_ascendc.lab import compact_golden  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tp", type=int, choices=(4, 8), default=4)
    parser.add_argument("--batch", type=int, choices=(1, 2, 32, 64), default=2)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--stage", choices=("smoke", "graph", "perf"), default="smoke")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--cores", type=int, choices=(8, 20, 40), default=40)
    parser.add_argument("--ring-layout", choices=("tied", "detached"), default="detached")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def make_fusion_case(
    tp: int, batch: int, pos: int, base: int, flush_mode: str, *, detached: bool = False, padding: bool = False
):
    case = make_case(
        tp=tp, geometry="27b", batch=batch, position=pos, length=4, base=base, tile_v=128, seed=123 + pos + base
    )
    case["flush"] = torch.zeros(batch, dtype=torch.int32)
    if flush_mode == "all":
        case["flush"].fill_(1)
    elif flush_mode == "mixed":
        case["flush"][::2] = 1
    case["compact_status"] = torch.full((batch * (48 // tp) * 4,), -1, dtype=torch.int32)
    if detached:
        case["ring_slots"] = case["slots"].clone()
        case["checkpoint"] = torch.cat((case["checkpoint"], case["checkpoint"][1:].clone()))
        case["slots"] = case["slots"] + batch
    if padding:
        case["slots"][0] = 0
        if detached:
            case["ring_slots"][0] = 0
        for name in ("positions", "bases", "lengths", "flush"):
            case[name][0] = 0
    return case


def invoke(case: dict[str, torch.Tensor], fused: bool, cores: int) -> None:
    from vllm_ascend.ops.replayssm.ascendc import (
        replayssm_sd_compact,
        replayssm_sd_verify,
        replayssm_sd_verify_compact_fused,
    )

    common = (
        case["q"],
        case["k"],
        case["v"],
        case["g"],
        case["beta"],
        case["checkpoint"],
        case["d_cache"],
        case["k_cache"],
        case["g_cache"],
        case["slots"],
        case["positions"],
        case["bases"],
        case["lengths"],
    )
    ring_slots = case.get("ring_slots")
    if fused:
        replayssm_sd_verify_compact_fused(
            *common,
            case["flush"],
            case["out"],
            case["status"],
            case["compact_status"],
            128**-0.5,
            128,
            cores,
            ring_slots,
        )
    else:
        replayssm_sd_verify(
            *common,
            case["out"],
            case["status"],
            128**-0.5,
            128,
            cores,
            13,
            ring_slots,
        )
        replayssm_sd_compact(
            case["checkpoint"],
            case["d_cache"],
            case["k_cache"],
            case["g_cache"],
            case["slots"],
            case["positions"],
            case["bases"],
            case["flush"],
            case["compact_status"],
            32,
            cores,
            1,
            ring_slots,
        )


def expected_case(case: dict[str, torch.Tensor]):
    reference = clone_case(case)
    if "ring_slots" in case:
        for row in range(case["slots"].numel()):
            reference["checkpoint"][int(case["ring_slots"][row])] = case["checkpoint"][int(case["slots"][row])]
        reference["slots"] = case["ring_slots"].clone()
    verified = recurrent_golden(reference)
    compact_input = clone_case(verified)
    compact_input["status"] = case["compact_status"].clone()
    folded = compact_golden(compact_input)
    expected = clone_case(case)
    for name in ("out", "d_cache", "k_cache", "g_cache"):
        expected[name] = folded[name]
    if "ring_slots" in case:
        for row in range(case["slots"].numel()):
            expected["checkpoint"][int(case["slots"][row])] = folded["checkpoint"][int(case["ring_slots"][row])]
    else:
        expected["checkpoint"] = folded["checkpoint"]
    expected["status"] = verified["status"]
    expected["compact_status"] = torch.zeros_like(case["compact_status"])
    return expected


def check(actual: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]):
    errors = {}
    for name in ("out", *CACHE_NAMES):
        left, right = actual[name].cpu(), expected[name].cpu()
        torch.testing.assert_close(left, right, atol=0.01, rtol=0.03, equal_nan=True)
        finite = torch.isfinite(right)
        errors[name] = float((left[finite].float() - right[finite].float()).abs().max()) if finite.any() else 0.0
    for name in ("status", "compact_status"):
        torch.testing.assert_close(actual[name].cpu(), expected[name], atol=0, rtol=0)
    return errors


def run_case(args, cpu, *, graph: bool, perf: bool):
    device = f"npu:{args.device}"
    expected = expected_case(cpu)
    results = {}
    for mode, fused in (("serial", False), ("fused", True)):
        gpu = clone_page_strided_case(cpu, device)
        invoke(gpu, fused, args.cores)
        torch.npu.synchronize()
        if graph:
            graph_obj = torch.npu.NPUGraph()
            with torch.npu.graph(graph_obj):
                invoke(gpu, fused, args.cores)
            pristine = {name: tensor.to(device) for name, tensor in cpu.items()}
            for name in (*CACHE_NAMES, "out", "status", "compact_status"):
                gpu[name].copy_(pristine[name])
            graph_obj.replay()
            torch.npu.synchronize()
        errors = check(clone_case(gpu), expected)
        result = {"errors": errors}
        if graph and not perf:
            alternate = (
                make_fusion_case(
                    args.tp,
                    args.batch,
                    12,
                    31,
                    "all",
                    detached="ring_slots" in cpu,
                )
                if int(cpu["flush"].sum()) == 0
                else make_fusion_case(
                    args.tp,
                    args.batch,
                    0,
                    0,
                    "none",
                    detached="ring_slots" in cpu,
                )
            )
            for name, tensor in alternate.items():
                gpu[name].copy_(tensor.to(device))
            graph_obj.replay()
            torch.npu.synchronize()
            result["dynamic_graph_errors"] = check(clone_case(gpu), expected_case(alternate))
        if perf:
            samples = []
            for _ in range(args.iterations):
                for name in (*CACHE_NAMES, "out", "status", "compact_status"):
                    gpu[name].copy_(pristine[name])
                start = torch.npu.Event(enable_timing=True)
                end = torch.npu.Event(enable_timing=True)
                start.record()
                graph_obj.replay()
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end) * 1000)
            check(clone_case(gpu), expected)
            ordered = sorted(samples)
            result["graph_replay_us"] = {
                "p50": statistics.median(samples),
                "p90": ordered[int(0.9 * (len(ordered) - 1))],
                "min": ordered[0],
                "iterations": len(samples),
                "reset_in_timed_region": False,
            }
        results[mode] = result
    return results


def main() -> None:
    args = parse_args()
    report = {
        "stage": args.stage,
        "tp": args.tp,
        "batch": args.batch,
        "cores": args.cores,
        "device": args.device,
        "ring_layout": args.ring_layout,
        "status": "FAILED",
        "cases": [],
    }
    report["source_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    report["packages"] = {}
    for name in ("torch", "torch-npu", "vllm", "vllm-ascend"):
        try:
            report["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report["packages"][name] = "not installed"
    try:
        import torch_npu  # noqa: F401

        from vllm_ascend.utils import enable_custom_op

        if not enable_custom_op() or not hasattr(torch.ops._C_ascend, "replayssm_sd_verify_compact_fused_"):
            raise RuntimeError("Rebuild vllm_ascend_C with the fused SD operator")
        torch.npu.set_device(args.device)
        cases = (
            (0, 0, "none", False),
            (7, 29, "none", False),
            (7, 29, "all", False),
            (12, 31, "all", False),
            (16, 31, "all", False),
        )
        if args.batch >= 2:
            cases += ((12, 31, "mixed", False), (12, 31, "all", True))
        for pos, base, flush_mode, padding in cases:
            cpu = make_fusion_case(
                args.tp, args.batch, pos, base, flush_mode, detached=args.ring_layout == "detached", padding=padding
            )
            result = run_case(args, cpu, graph=args.stage != "smoke", perf=args.stage == "perf")
            report["cases"].append({"position": pos, "base": base, "flush": flush_mode, "padding": padding, **result})
        report["status"] = "PASSED"
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({key: report[key] for key in ("stage", "tp", "batch", "status")}, ensure_ascii=False))
    if report["status"] != "PASSED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
