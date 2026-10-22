# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Isolated, staged AscendC validation. Parent process never initializes an NPU."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import statistics
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

STAGES = ("smoke", "replay", "flush", "lifecycle", "graph_perf")


def write_json(path, record):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def child_command(args, device, report):
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--stage",
        args.stage,
        "--tp",
        str(args.tp),
        "--capacity",
        str(args.capacity),
        "--batch",
        str(args.batch),
        "--tile-v",
        str(args.tile_v),
        "--cores",
        str(args.cores),
        "--library",
        str(args.library.resolve()),
        "--output",
        str(report.parent),
        "--child-device",
        str(device),
        "--iterations",
        str(args.iterations),
        "--backend",
        getattr(args, "backend", "standalone"),
        "--stack-layers",
        str(getattr(args, "stack_layers", 0)),
    ]
    if args.native_baseline:
        command.append("--native-baseline")
    return command


def run_single_case(op, cpu_case, args, device, label, page_strided=False):
    import torch
    from lab import check_result, clone_case, clone_page_strided_case, golden, invoke

    expected = golden(cpu_case)
    device_case = clone_page_strided_case(cpu_case, device) if page_strided else clone_case(cpu_case, device)
    print(f"LAUNCH {label}", flush=True)
    begin = time.perf_counter()
    invoke(op, device_case, args.tile_v, args.cores)
    torch.npu.synchronize()
    first_ms = (time.perf_counter() - begin) * 1000
    actual = clone_case(device_case)
    metrics = check_result(actual, expected, cpu_case)
    print(f"PASS {label}", flush=True)
    return {"case": label, "status": "PASSED", "launch_and_sync_ms": first_ms, "errors": metrics}


def run_lifecycle(op, args, device, save):
    import torch
    from lab import BUFFER_NAMES, check_result, clone_case, golden, invoke, make_case

    cpu = make_case(args.tp, args.batch, args.capacity, flush=args.capacity == 1, tile_v=args.tile_v, seed=800)
    gpu = clone_case(cpu, device)
    for step in range(2 * args.capacity + 3):
        inputs = make_case(
            args.tp, args.batch, args.capacity, flush=args.capacity == 1, tile_v=args.tile_v, seed=801 + step
        )
        for name in ("q", "k", "v", "g", "beta"):
            cpu[name].copy_(inputs[name])
            gpu[name].copy_(inputs[name])
        # One row flushes early. Rows are reordered to test physical-slot lookup.
        if step == 2 and args.batch > 1:
            cpu["slots"] = cpu["slots"].roll(1)
            cpu["positions"] = cpu["positions"].roll(1)
        if step == args.capacity + 1:
            # Simulate a new request reusing a physical slot: reset only its
            # checkpoint/cursor, leaving the old history in storage.
            slot = int(cpu["slots"][0])
            cpu["checkpoint"][slot].zero_()
            gpu["checkpoint"][slot].zero_()
            cpu["positions"][0] = 0
        cpu["flush"] = (cpu["positions"] == args.capacity - 1).int()
        if step == 1:
            cpu["flush"][0] = 1
        for name in ("slots", "positions", "flush"):
            gpu[name].copy_(cpu[name])
        expected = golden(cpu)
        preserved = clone_case(gpu)
        print(f"LAUNCH lifecycle step={step}", flush=True)
        invoke(op, gpu, args.tile_v, args.cores)
        torch.npu.synchronize()
        # Carry each implementation's own state forward to expose accumulated
        # numerical drift. Do not overwrite device caches with the golden.
        metrics = check_result(clone_case(gpu), expected, cpu, preserved)
        save({"case": f"lifecycle_step_{step}", "status": "PASSED", "errors": metrics})
        for name in BUFFER_NAMES:
            cpu[name] = expected[name]
        cpu["positions"] = torch.where(cpu["flush"].bool(), 0, cpu["positions"] + 1)


def run_invalid_metadata(op, args, device, save):
    from lab import check_result, clone_case, golden, invoke, make_case

    variants = (("slots", args.batch + 2, 1), ("positions", args.capacity, 2), ("flush", 2, 3))
    for field, value, code in variants:
        cpu = make_case(args.tp, args.batch, args.capacity, flush=args.capacity == 1, tile_v=args.tile_v)
        # Golden for this row is the padding/no-write behavior; only status
        # differs when the device receives invalid metadata.
        cpu["slots"][0] = 0
        cpu["flush"][0] = 0
        expected = golden(cpu)
        expected["status"][: (48 // args.tp) * (128 // args.tile_v)] = code
        gpu = clone_case(cpu, device)
        gpu[field][0] = value
        invoke(op, gpu, args.tile_v, args.cores)
        import torch

        torch.npu.synchronize()
        metrics = check_result(clone_case(gpu), expected, cpu)
        save({"case": f"invalid_{field}", "status": "PASSED", "expected_status": code, "errors": metrics})


def event_timing(call, reset, iterations):
    import torch

    for _ in range(3):
        reset()
        call()
    torch.npu.synchronize()
    samples = []
    for _ in range(iterations):
        reset()
        start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        start.record()
        call()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000)
    ordered = sorted(samples)
    return {
        "device_us_p50": statistics.median(samples),
        "device_us_p90": ordered[min(len(ordered) - 1, int(len(ordered) * 0.9))],
        "device_us_min": min(samples),
        "iterations": iterations,
        "reset_in_timed_region": False,
    }


def run_graph_perf(op, args, device, save):
    import torch
    from lab import BUFFER_NAMES, check_result, clone_case, golden, invoke, make_case

    cpu = make_case(args.tp, args.batch, args.capacity, flush=args.capacity == 1, tile_v=args.tile_v, seed=900)
    gpu = clone_case(cpu, device)
    # Warm up before capture; test current-stream launch on a nondefault stream.
    stream = torch.npu.Stream(device=device)
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        invoke(op, gpu, args.tile_v, args.cores)
    torch.npu.current_stream().wait_stream(stream)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    print("CAPTURE ar_step_", flush=True)
    with torch.npu.graph(graph):
        invoke(op, gpu, args.tile_v, args.cores)
    # Same addresses, different device-side position/flush and cache contents.
    for pos, flush in ((0, False), (args.capacity // 2, False), (args.capacity - 1, True), (0, True)):
        flush = flush or pos == args.capacity - 1
        cpu = make_case(args.tp, args.batch, args.capacity, pos, flush, args.tile_v, seed=901 + pos)
        for name in cpu:
            gpu[name].copy_(cpu[name])
        print(f"REPLAY pos={pos} flush={flush}", flush=True)
        graph.replay()
        torch.npu.synchronize()
        metrics = check_result(clone_case(gpu), golden(cpu), cpu)
        save({"case": f"graph_pos{pos}_flush{int(flush)}", "status": "PASSED", "errors": metrics})

    # Event timing around eager Python calls may include host dispatch gaps.
    # Report captured replay separately; reset mutations outside timing.
    pristine_graph = {name: cpu[name].to(device) for name in BUFFER_NAMES}

    def reset_graph_device():
        for name in BUFFER_NAMES:
            gpu[name].copy_(pristine_graph[name])

    timing = event_timing(graph.replay, reset_graph_device, args.iterations)
    check_result(clone_case(gpu), golden(cpu), cpu)
    save({"case": "perf_graph_single_empty_history_flush", "status": "PASSED", **timing})

    for pos, flush in ((0, False), (args.capacity // 2, False), (args.capacity - 1, True)):
        flush = flush or pos == args.capacity - 1
        cpu = make_case(args.tp, args.batch, args.capacity, pos, flush, args.tile_v, seed=910 + pos)
        gpu = clone_case(cpu, device)
        pristine = {name: gpu[name].clone() for name in BUFFER_NAMES}

        def reset(gpu=gpu, pristine=pristine):
            for name in BUFFER_NAMES:
                gpu[name].copy_(pristine[name])

        timing = event_timing(lambda gpu=gpu: invoke(op, gpu, args.tile_v, args.cores), reset, args.iterations)
        check_result(clone_case(gpu), golden(cpu), cpu)
        save({"case": f"perf_pos{pos}_flush{int(flush)}", "status": "PASSED", **timing})

    if args.native_baseline:
        from fla_npu.ops.ascendc import recurrent_gated_delta_rule

        # Only compare the same state/output semantics: empty history + flush.
        cpu = make_case(args.tp, args.batch, args.capacity, 0, True, args.tile_v, seed=950)
        gpu = clone_case(cpu, device)
        state = gpu["checkpoint"]
        pristine = state.clone()
        lengths = torch.tensor([0] + [1] * args.batch, dtype=torch.int32, device=device)
        beta = gpu["beta"].to(torch.bfloat16)
        native_output = None

        def native_call():
            nonlocal native_output
            native_output = recurrent_gated_delta_rule(
                gpu["q"],
                gpu["k"],
                gpu["v"],
                state,
                g=gpu["g"],
                beta=beta,
                scale=128**-0.5,
                actual_seq_lengths=lengths,
                ssm_state_indices=gpu["slots"],
            )

        timing = event_timing(native_call, lambda: state.copy_(pristine), args.iterations)
        gpu["out"].copy_(native_output)
        gpu["status"].zero_()
        check_result(clone_case(gpu), golden(cpu), cpu)
        save({"case": "native_ar_empty_history_flush", "status": "PASSED", **timing})
        replay_timing = event_timing(
            lambda: invoke(op, gpu, args.tile_v, args.cores), lambda: state.copy_(pristine), args.iterations
        )
        check_result(clone_case(gpu), golden(cpu), cpu)
        save({"case": "replay_ar_empty_history_flush", "status": "PASSED", **replay_timing})

    if args.stack_layers:
        run_layer_stack(op, args, device, save)


def run_layer_stack(op, args, device, save):
    import torch
    from lab import check_result, clone_case, golden, invoke, make_case

    if args.capacity < 3:
        raise ValueError("layer-stack benchmark requires capacity >= 3")
    # Independent allocations preserve a realistic multi-layer working set.
    # The non-flush append position is outside the read prefix, so repeated
    # replay is valid without resetting state inside the measured graph.
    cases = [
        make_case(args.tp, args.batch, args.capacity, args.capacity // 2, False, args.tile_v, seed=1000 + i)
        for i in range(args.stack_layers)
    ]
    buffers = [clone_case(case, device) for case in cases]

    def call_stack():
        for case in buffers:
            invoke(op, case, args.tile_v, args.cores)

    call_stack()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        call_stack()
    timing = event_timing(graph.replay, lambda: None, args.iterations)
    for before, after in zip(cases, buffers):
        check_result(clone_case(after), golden(before), before)
    save(
        {
            "case": "perf_graph_layer_stack",
            "status": "PASSED",
            "layers": args.stack_layers,
            "calls_per_rank_per_replay": args.stack_layers,
            "graph_ms_p50": timing["device_us_p50"] / 1000,
            "amortized_layer_us_p50": timing["device_us_p50"] / args.stack_layers,
            **timing,
        }
    )


def run_child(args, report):
    record = {
        "status": "RUNNING",
        "stage": args.stage,
        "backend": args.backend,
        "stack_layers": args.stack_layers,
        "device": args.child_device,
        "tp_geometry": args.tp,
        "batch": args.batch,
        "capacity": args.capacity,
        "tile_v": args.tile_v,
        "cores": args.cores,
        "cases": [],
        "next_level_allowed": False,
    }
    write_json(report, record)

    def save(case):
        record["cases"].append(case)
        write_json(report, record)

    try:
        import torch
        import torch_npu  # noqa: F401
        from lab import load_framework, load_library, make_case, ub_bytes

        record["packages"] = {}
        for name in (
            "torch",
            "torch-npu",
            "vllm",
            "vllm-ascend",
            "triton-ascend",
            "fla-npu",
            "flash-linear-attention-npu",
            "flash-linear-attention-npu-a2",
        ):
            try:
                record["packages"][name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                record["packages"][name] = "not installed under this distribution name"
        record["source_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent, text=True
        ).strip()
        record["source_status"] = subprocess.check_output(
            ["git", "status", "--short"], cwd=Path(__file__).resolve().parent, text=True
        ).splitlines()
        if args.backend == "standalone":
            record["library"] = str(args.library.resolve())
            record["library_sha256"] = hashlib.sha256(args.library.read_bytes()).hexdigest()
            kernel_library = args.library.parent / "lib/libreplayssm_ar_kernel.so"
            if kernel_library.exists():
                record["kernel_sha256"] = hashlib.sha256(kernel_library.read_bytes()).hexdigest()
        cache = args.library.parent / "CMakeCache.txt"
        if args.backend == "standalone" and cache.exists():
            record["build_config"] = [
                line
                for line in cache.read_text().splitlines()
                if line.startswith(
                    ("SOC_VERSION:", "ASCEND_CANN_PACKAGE_PATH:", "Python3_EXECUTABLE:", "CMAKE_CXX_COMPILER:")
                )
            ]
        record["ub_bytes_per_core"] = ub_bytes(args.capacity, args.tile_v)
        write_json(report, record)
        torch.set_num_threads(1)
        torch.npu.set_device(args.child_device)
        device = torch.device(f"npu:{args.child_device}")
        record["device_name"] = torch.npu.get_device_name(args.child_device)
        if args.backend == "framework":
            op, artifact = load_framework()
            record["library"] = str(artifact)
            record["library_sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
            record["framework_kernels"] = [
                {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                for path in artifact.parent.glob("**/libvllm_ascend_kernels.so")
            ]
        else:
            op = load_library(args.library)
        write_json(report, record)
        if args.stage in ("smoke", "replay", "flush"):
            if args.stage == "smoke":
                variants = [(0, args.capacity == 1, False)]
            elif args.stage == "replay":
                variants = [
                    (p, p == args.capacity - 1, False)
                    for p in sorted({0, 1, 7, args.capacity // 2, args.capacity - 1})
                    if p < args.capacity
                ]
            else:
                variants = [
                    (p, True, pad) for p in sorted({0, args.capacity // 2, args.capacity - 1}) for pad in (False, True)
                ]
            for pos, flush, pad in variants:
                cpu = make_case(
                    args.tp, args.batch, args.capacity, pos, flush, args.tile_v, seed=700 + pos, padding=pad
                )
                label = f"{args.stage}_pos{pos}_flush{int(flush)}_pad{int(pad)}"
                record["active_case"] = label
                write_json(report, record)
                save(run_single_case(op, cpu, args, device, label))
                if args.stage == "smoke" and not pad:
                    page_label = f"{label}_page_strided"
                    record["active_case"] = page_label
                    write_json(report, record)
                    save(
                        run_single_case(
                            op,
                            cpu,
                            args,
                            device,
                            page_label,
                            page_strided=True,
                        )
                    )
        elif args.stage == "lifecycle":
            run_lifecycle(op, args, device, save)
            run_invalid_metadata(op, args, device, save)
        else:
            run_graph_perf(op, args, device, save)
        record["status"] = "PASSED"
    except Exception as exc:
        record.update(status="FAILED", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        print(record["traceback"], flush=True)
    write_json(report, record)
    return 0 if record["status"] == "PASSED" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, default="smoke")
    parser.add_argument("--backend", choices=("standalone", "framework"), default="standalone")
    parser.add_argument(
        "--stack-layers", type=int, default=0, help="Optional independent-layer captured graph benchmark"
    )
    parser.add_argument("--tp", type=int, choices=(4, 8), default=4, help="Local head geometry; no distributed launch")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--capacity", type=int, choices=range(1, 65), default=16)
    parser.add_argument("--tile-v", type=int, choices=(16, 32, 64), default=32)
    parser.add_argument("--cores", type=int, choices=range(1, 65), default=1)
    parser.add_argument("--devices", default="0", help="Comma-separated device IDs; children run in parallel")
    parser.add_argument("--library", type=Path, default=Path(__file__).parent / "build/libreplayssm_ar_torch.so")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=180, help="Host child timeout, not a device reset")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--native-baseline", action="store_true")
    parser.add_argument("--child-device", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (
        args.batch < 1
        or args.batch > 1024
        or args.timeout <= 0
        or args.iterations < 1
        or not 0 <= args.stack_layers <= 128
    ):
        parser.error("invalid batch/timeout/iterations")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.child_device is not None:
        return run_child(args, args.output / f"device{args.child_device}.json")
    try:
        devices = [int(device) for device in args.devices.split(",")]
        if not devices or min(devices) < 0 or len(set(devices)) != len(devices):
            raise ValueError
    except ValueError:
        parser.error("--devices must contain unique nonnegative integers")
    if (args.output / "summary.json").exists() or any(args.output.glob("device*.json")):
        parser.error("use a fresh output directory to preserve previous reports")

    def worker(device):
        report = args.output / f"device{device}.json"
        with (args.output / f"device{device}.log").open("w") as log:
            try:
                result = subprocess.run(
                    child_command(args, device, report), stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout
                )
                data = (
                    json.loads(report.read_text())
                    if report.exists()
                    else {"status": "FAILED", "error": "child exited without report"}
                )
                if result.returncode and data.get("status") != "FAILED":
                    data.update(status="FAILED", error=f"child returncode={result.returncode}")
                return data
            except subprocess.TimeoutExpired:
                data = json.loads(report.read_text()) if report.exists() else {}
                data.update(status="TIMEOUT", device=device, error="Host timeout; check NPU health before rerunning")
                write_json(report, data)
                return data

    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        results = list(pool.map(worker, devices))
    summary = {
        "stage": args.stage,
        "status": "PASSED" if all(r["status"] == "PASSED" for r in results) else "FAILED",
        "devices": results,
        "next_level_allowed": False,
    }
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["status"] == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
