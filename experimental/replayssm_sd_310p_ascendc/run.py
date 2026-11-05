"""Ascend310P3 isolated correctness, graph, and native-op comparison."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import statistics
import subprocess
import traceback
from pathlib import Path

import torch
import torch_npu  # noqa: F401
from reference import (
    CACHE,
    DIM,
    RING,
    clone,
    compact_reference,
    cursor_reference,
    fp16_error,
    make_case,
    verify_reference,
)

ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=Path(__file__).parent / "build/libreplayssm_sd_310p_torch.so")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--stage", choices=("cursor_probe", "smoke", "sequence", "graph", "perf", "all"), default="smoke"
    )
    parser.add_argument("--mtp", type=int, choices=(3, 5), required=True)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--tile-v", type=int, choices=(32, 64), default=32)
    parser.add_argument("--cores", type=int, choices=range(1, 8), default=7)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def verify(case: dict[str, torch.Tensor], tile_v: int, cores: int) -> None:
    torch.ops.replayssm_sd_310p.verify_(
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
        case["out"],
        case["verify_status"],
        DIM**-0.5,
        tile_v,
        cores,
        case["ring_slots"],
    )


def compact(case: dict[str, torch.Tensor], cores: int) -> None:
    torch.ops.replayssm_sd_310p.compact_(
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
        case["ring_slots"],
    )


def compare_verify(actual: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]) -> dict:
    result = {"out": fp16_error(actual["out"].cpu(), expected["out"])}
    for name in CACHE:
        a, e = actual[name].cpu(), expected[name]
        if name in ("d_cache", "k_cache", "checkpoint"):
            result[name] = fp16_error(a, e)
        else:
            torch.testing.assert_close(a, e, atol=0, rtol=0, equal_nan=True)
            result[name] = {"max_abs": 0.0}
    torch.testing.assert_close(actual["verify_status"].cpu(), expected["verify_status"], atol=0, rtol=0)
    return result


def compare_compact(actual: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]) -> dict:
    torch.testing.assert_close(actual["compact_status"].cpu(), expected["compact_status"], atol=0, rtol=0)
    result = {"checkpoint": fp16_error(actual["checkpoint"].cpu(), expected["checkpoint"])}
    for name in ("d_cache", "k_cache", "g_cache"):
        torch.testing.assert_close(actual[name].cpu(), expected[name], atol=0, rtol=0, equal_nan=True)
    return result


def cursor_case(case: dict[str, torch.Tensor], mtp: int) -> dict[str, torch.Tensor]:
    slots = case["checkpoint"].shape[0]
    batch = case["slots"].numel()
    result = {
        "write_pos": torch.zeros(slots, dtype=torch.int32),
        "cache_base": torch.zeros(slots, dtype=torch.int32),
        "is_flush": torch.zeros(slots, dtype=torch.int8),
        "accepted": torch.full((batch,), mtp, dtype=torch.int32),
        "slots": case["slots"].clone(),
        "do_reset": torch.zeros(batch, dtype=torch.int8),
        "status": torch.full((batch,), -1, dtype=torch.int32),
    }
    for row in range(batch):
        slot = int(case["slots"][row])
        if slot > 0:
            result["write_pos"][slot] = case["positions"][row]
            result["cache_base"][slot] = case["bases"][row]
            result["is_flush"][slot] = case["flush"][row]
    return result


def cursor(case: dict[str, torch.Tensor], mtp: int) -> None:
    torch.ops.replayssm_sd_310p.cursor_(
        case["write_pos"],
        case["cache_base"],
        case["is_flush"],
        case["accepted"],
        case["slots"],
        case["do_reset"],
        case["status"],
        16,
        mtp,
        RING,
    )


def cursor_probe(args: argparse.Namespace, device: torch.device) -> dict:
    """Minimal device writeback gate, independent of verify and compact."""
    case = make_case(mtp=args.mtp, batch=1, pos=0)
    cpu = cursor_case(case, args.mtp)
    cpu["write_pos"][1] = 12
    cpu["cache_base"][1] = 31
    cpu["is_flush"][1] = 1
    gpu = clone(cpu, device)
    expected = cursor_reference(
        cpu["write_pos"],
        cpu["cache_base"],
        cpu["is_flush"],
        cpu["accepted"],
        cpu["slots"],
        cpu["do_reset"],
        args.mtp,
    )
    cursor(gpu, args.mtp)
    torch.npu.synchronize()
    fields = ("write_pos", "cache_base", "is_flush", "status")
    actual = {name: gpu[name].cpu().tolist() for name in fields}
    wanted = {name: value.tolist() for name, value in zip(fields, expected)}
    if actual != wanted:
        raise AssertionError(f"cursor device writeback failed: actual={actual}, expected={wanted}")
    return {"actual": actual, "expected": wanted}


def smoke(args: argparse.Namespace, device: torch.device) -> list[dict]:
    records = []
    for pos, base, flush, padding in (
        (0, 0, 0, False),
        (7, 29, 0, False),
        (12, 31, 1, False),
        (16, 31, 1, False),
        (7, 30, 1, args.batch > 1),
    ):
        cpu = make_case(
            mtp=args.mtp, batch=args.batch, pos=pos, base=base, tile_v=args.tile_v, seed=42 + pos, padding=padding
        )
        cpu["flush"].fill_(flush)
        if args.batch > 1:
            cpu["flush"][0] = 1 - flush  # mixed flush rows
        detached_ring = args.batch > 1 and pos == 12
        if detached_ring:
            # Keep checkpoint ids unchanged while exchanging request-owned ring
            # allocations. This catches accidental use of the checkpoint slot
            # to address d/k/g history.
            for name in ("d_cache", "k_cache", "g_cache"):
                cpu[name][[1, 2]] = cpu[name][[2, 1]].clone()
            cpu["ring_slots"][:2] = torch.tensor([2, 1], dtype=torch.int32)
        gpu = clone(cpu, device)
        expected = verify_reference(cpu)
        verify(gpu, args.tile_v, args.cores)
        torch.npu.synchronize()
        verify_errors = compare_verify(gpu, expected)
        compact_expected = compact_reference(expected)
        compact(gpu, args.cores)
        torch.npu.synchronize()
        compact_errors = compare_compact(gpu, compact_expected)
        cursor_cpu = cursor_case(cpu, args.mtp)
        cursor_gpu = clone(cursor_cpu, device)
        wp, base_expected, fl, status = cursor_reference(
            cursor_cpu["write_pos"],
            cursor_cpu["cache_base"],
            cursor_cpu["is_flush"],
            cursor_cpu["accepted"],
            cursor_cpu["slots"],
            cursor_cpu["do_reset"],
            args.mtp,
        )
        cursor(cursor_gpu, args.mtp)
        torch.npu.synchronize()
        for name, expected_tensor in (
            ("write_pos", wp),
            ("cache_base", base_expected),
            ("is_flush", fl),
            ("status", status),
        ):
            torch.testing.assert_close(cursor_gpu[name].cpu(), expected_tensor, atol=0, rtol=0)
        records.append(
            {
                "pos": pos,
                "base": base,
                "flush": flush,
                "padding": padding,
                "detached_ring": detached_ring,
                "verify": verify_errors,
                "compact": compact_errors,
                "cursor": "PASSED",
            }
        )
    records.append({"invalid_bindings": invalid_bindings(args, device)})
    return records


def invalid_bindings(args: argparse.Namespace, device: torch.device) -> list[str]:
    """Ensure malformed inputs fail on the host, before a device launch."""
    case = clone(make_case(mtp=args.mtp, batch=1, pos=0), device)
    checks = []
    for label, change in (
        ("verify_width", {"q": case["q"][:, :3]}),
        ("verify_dtype", {"beta": case["beta"].float()}),
    ):
        broken = dict(case)
        broken.update(change)
        try:
            verify(broken, args.tile_v, args.cores)
        except RuntimeError:
            checks.append(label)
        else:
            raise AssertionError(f"{label} unexpectedly accepted")
    broken = dict(case)
    broken["checkpoint"] = case["checkpoint"].float()
    try:
        compact(broken, args.cores)
    except RuntimeError:
        checks.append("compact_dtype")
    else:
        raise AssertionError("compact_dtype unexpectedly accepted")
    cur = clone(cursor_case(case, args.mtp), device)
    try:
        cursor(cur, 4)
    except RuntimeError:
        checks.append("cursor_spec_len")
    else:
        raise AssertionError("cursor_spec_len unexpectedly accepted")
    return checks


def sequence(args: argparse.Namespace, device: torch.device) -> list[dict]:
    """Exercise 24 accepted-prefix commits through wrap and flush transitions."""
    cpu = make_case(mtp=args.mtp, batch=1, pos=0, base=0, tile_v=args.tile_v)
    gpu = clone(cpu, device)
    cur_cpu = cursor_case(cpu, args.mtp)
    cur_gpu = clone(cur_cpu, device)
    records = []
    accepted_pattern = (1, args.mtp, 2, args.mtp - 1)
    for step in range(24):
        fresh = make_case(mtp=args.mtp, batch=1, pos=0, base=0, tile_v=args.tile_v, seed=1000 + step)
        for name in ("q", "k", "v", "g", "beta"):
            cpu[name].copy_(fresh[name])
            gpu[name].copy_(fresh[name].to(device))
        for name in ("positions", "bases", "flush"):
            gpu[name].copy_(cpu[name].to(device))
        expected_verify = verify_reference(cpu)
        verify(gpu, args.tile_v, args.cores)
        torch.npu.synchronize()
        verify_errors = compare_verify(gpu, expected_verify)
        expected_compact = compact_reference(expected_verify)
        compact(gpu, args.cores)
        torch.npu.synchronize()
        compact_errors = compare_compact(gpu, expected_compact)
        cpu = expected_compact

        accepted = accepted_pattern[step % len(accepted_pattern)]
        cur_cpu["accepted"].fill_(accepted)
        cur_gpu["accepted"].fill_(accepted)
        wp, base, flush, status = cursor_reference(
            cur_cpu["write_pos"],
            cur_cpu["cache_base"],
            cur_cpu["is_flush"],
            cur_cpu["accepted"],
            cur_cpu["slots"],
            cur_cpu["do_reset"],
            args.mtp,
        )
        cursor(cur_gpu, args.mtp)
        torch.npu.synchronize()
        for name, expected_tensor in (("write_pos", wp), ("cache_base", base), ("is_flush", flush), ("status", status)):
            torch.testing.assert_close(cur_gpu[name].cpu(), expected_tensor, atol=0, rtol=0)
            cur_cpu[name] = expected_tensor
        cpu["positions"][0] = wp[1]
        cpu["bases"][0] = base[1]
        cpu["flush"][0] = flush[1]
        records.append(
            {
                "step": step,
                "accepted": accepted,
                "next_pos": int(wp[1]),
                "next_base": int(base[1]),
                "next_flush": int(flush[1]),
                "verify": verify_errors,
                "compact": compact_errors,
            }
        )
    return records


def native_inputs(cpu: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    batch, tokens = cpu["q"].shape[:2]
    indices = torch.arange(1, batch * tokens + 1, dtype=torch.int32).reshape(batch, tokens)
    state = torch.empty((batch * tokens + 1, 16, DIM, DIM), dtype=torch.float16)
    for row in range(batch):
        initial = cpu["checkpoint"][row + 1].double()
        pos, base = int(cpu["positions"][row]), int(cpu["bases"][row])
        for i in range(pos):
            phys = (base + i) & (RING - 1)
            alpha = cpu["g_cache"][row + 1, :, phys].double().exp()
            delta = cpu["d_cache"][row + 1, :, phys].double()
            key = cpu["k_cache"][row + 1, :, phys].double().repeat_interleave(2, dim=0)
            initial = alpha[:, None, None] * initial + delta[:, :, None] * key[:, None, :]
        state[indices[row].long()] = initial.half()
    state[0].zero_()
    return {
        "query": cpu["q"].reshape(-1, 8, DIM).to(device),
        "key": cpu["k"].reshape(-1, 8, DIM).to(device),
        "value": cpu["v"].reshape(-1, 16, DIM).to(device),
        "g": cpu["g"].reshape(-1, 16).to(device),
        "beta": cpu["beta"].reshape(-1, 16).to(device),
        "state": state.to(device),
        "actual_seq_lengths": torch.full((batch,), tokens, dtype=torch.int32, device=device),
        "ssm_state_indices": indices.reshape(-1).to(device),
        "num_accepted_tokens": torch.ones(batch, dtype=torch.int32, device=device),
    }


def native(inp: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.ops._C_ascend.npu_recurrent_gated_delta_rule_310(
        **inp,
        gk=None,
        scale_value=DIM**-0.5,
    )


def graph_of(callable_):
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        callable_()
    return graph


def time_graph(graph, reset, *, warmup: int, iterations: int) -> dict[str, float]:
    for _ in range(warmup):
        reset()
        graph.replay()
    torch.npu.synchronize()
    samples = []
    for _ in range(iterations):
        reset()
        start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000)
    ordered = sorted(samples)
    return {
        "p50_us": statistics.median(samples),
        "p90_us": ordered[int(0.9 * (len(ordered) - 1))],
        "min_us": ordered[0],
        "iterations": len(samples),
    }


def _graph_case(args: argparse.Namespace, device: torch.device, pos: int, base: int, flush: int, perf: bool) -> dict:
    cpu = make_case(mtp=args.mtp, batch=args.batch, pos=pos, base=base, tile_v=args.tile_v)
    cpu["flush"].fill_(flush)
    gpu = clone(cpu, device)
    cur = clone(cursor_case(cpu, args.mtp), device)
    cur_before = {name: cur[name].clone() for name in ("write_pos", "cache_base", "is_flush", "status")}
    replay_before = {name: gpu[name].clone() for name in CACHE}

    verify_graph = graph_of(lambda: verify(gpu, args.tile_v, args.cores))
    compact_graph = graph_of(lambda: compact(gpu, args.cores))
    cursor_graph = graph_of(lambda: cursor(cur, args.mtp))
    combined_graph = graph_of(
        lambda: (verify(gpu, args.tile_v, args.cores), compact(gpu, args.cores), cursor(cur, args.mtp))
    )

    # The same captured graph must consume new device metadata and cache
    # contents. Rebuilding a graph per position would not test this ABI.
    alternate = make_case(
        mtp=args.mtp,
        batch=args.batch,
        pos=7 if pos == 0 else 0,
        base=30 if pos == 0 else 0,
        tile_v=args.tile_v,
        seed=900 + pos,
    )
    alternate["flush"].fill_(1 - flush)
    for name in alternate:
        gpu[name].copy_(alternate[name].to(device))
    verify_graph.replay()
    torch.npu.synchronize()
    dynamic_verify = compare_verify(gpu, verify_reference(alternate))
    for name in alternate:
        gpu[name].copy_(alternate[name].to(device))
    compact_graph.replay()
    torch.npu.synchronize()
    dynamic_compact = compare_compact(gpu, compact_reference(alternate))
    for name in cpu:
        gpu[name].copy_(cpu[name].to(device))
    for name, value in cur_before.items():
        cur[name].copy_(value)
    cur["accepted"].fill_(2)
    cur["do_reset"].fill_(1)
    cursor_graph.replay()
    torch.npu.synchronize()
    dynamic_cursor = all(
        bool(torch.equal(cur[name].cpu(), expected_tensor))
        for name, expected_tensor in zip(
            ("write_pos", "cache_base", "is_flush", "status"),
            cursor_reference(
                cur_before["write_pos"].cpu(),
                cur_before["cache_base"].cpu(),
                cur_before["is_flush"].cpu(),
                torch.full_like(cur["accepted"].cpu(), 2),
                cur["slots"].cpu(),
                torch.ones_like(cur["do_reset"].cpu()),
                args.mtp,
            ),
        )
    )
    if not dynamic_cursor:
        raise AssertionError("Captured cursor ignored dynamic reset/acceptance metadata")
    cur["accepted"].fill_(args.mtp)
    cur["do_reset"].zero_()

    def reset_replay() -> None:
        for name in CACHE:
            gpu[name].copy_(replay_before[name])
        gpu["out"].zero_()
        gpu["verify_status"].fill_(-1)
        gpu["compact_status"].fill_(-1)

    def reset_cursor() -> None:
        for name, value in cur_before.items():
            cur[name].copy_(value)

    def reset_combined() -> None:
        reset_replay()
        reset_cursor()

    reset_replay()
    verify_graph.replay()
    torch.npu.synchronize()
    expected = verify_reference(cpu)
    result = compare_verify(gpu, expected)

    record = {
        "pos": pos,
        "base": base,
        "flush": flush,
        "errors": result,
        "dynamic_verify": dynamic_verify,
        "dynamic_compact": dynamic_compact,
        "dynamic_cursor": dynamic_cursor,
    }
    if perf:
        native_in = native_inputs(cpu, device)
        native_before = native_in["state"].clone()
        native_graph = graph_of(lambda: native(native_in))

        def reset_native() -> None:
            native_in["state"].copy_(native_before)

        reset_native()
        native_out = native(native_in).reshape_as(gpu["out"])
        torch.npu.synchronize()
        # The native path starts from a materialized FP16 history state.
        # Report its additional rounding separately from graph timing.
        try:
            result["native_vs_dense_out"] = fp16_error(native_out.cpu(), expected["out"])
        except AssertionError as error:
            if pos == 0:
                raise
            result["native_vs_dense_out"] = {"status": "DIFFERENT_ROUNDING", "detail": str(error)}
        record["graph_replay_us"] = {
            "verify": time_graph(verify_graph, reset_replay, warmup=args.warmup, iterations=args.iterations),
            "compact": time_graph(compact_graph, reset_replay, warmup=args.warmup, iterations=args.iterations),
            "cursor": time_graph(cursor_graph, reset_cursor, warmup=args.warmup, iterations=args.iterations),
            "combined": time_graph(combined_graph, reset_combined, warmup=args.warmup, iterations=args.iterations),
            "native": time_graph(native_graph, reset_native, warmup=args.warmup, iterations=args.iterations),
        }
    return record


def graph_and_perf(args: argparse.Namespace, device: torch.device, perf: bool) -> list[dict]:
    if perf:
        from vllm_ascend.utils import enable_custom_op

        if not enable_custom_op():
            raise RuntimeError("vllm_ascend native custom operators are unavailable")

    return [
        _graph_case(args, device, pos, base, flush, perf)
        for pos, base, flush in ((0, 0, 0), (12, 31, 0), (12, 31, 1), (16, 31, 1))
    ]


def main() -> None:
    args = parse_args()
    report = {
        "status": "FAILED",
        "mtp": args.mtp,
        "batch": args.batch,
        "tile_v": args.tile_v,
        "cores": args.cores,
        "device": args.device,
        "stage": args.stage,
    }
    try:
        torch.ops.load_library(str(args.library.resolve()))
        torch.npu.set_device(args.device)
        torch_npu.npu.set_compile_mode(jit_compile=False)
        device = torch.device(f"npu:{args.device}")
        report["device_name"] = torch.npu.get_device_name(args.device)
        report["source_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip()
        for package in ("torch", "torch-npu", "vllm", "vllm-ascend"):
            try:
                report[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                report[package] = "not installed"
        if args.stage == "cursor_probe":
            report["cursor_probe"] = cursor_probe(args, device)
        if args.stage in ("smoke", "all"):
            report["smoke"] = smoke(args, device)
        if args.stage in ("sequence", "all"):
            report["sequence"] = sequence(args, device)
        if args.stage in ("graph", "perf", "all"):
            report["graph"] = graph_and_perf(args, device, args.stage in ("perf", "all"))
        report["status"] = "PASSED"
    except Exception:
        report["error"] = traceback.format_exc()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    summary = {key: value for key, value in report.items() if key not in ("smoke", "sequence", "graph")}
    print(json.dumps(summary, ensure_ascii=False))
    if report["status"] != "PASSED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
