# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Run isolated ReplaySSM Triton feature probes concurrently across NPUs."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

PROBES = (
    "cumsum",
    "serial_prefix",
    "dot",
    "masked_zero_dot",
    "runtime_branch",
    "masked_2d_io",
)


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def child_command(probe: str, device: int, report: Path) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child-probe",
        probe,
        "--child-device",
        str(device),
        "--child-report",
        str(report),
    ]


def run_probe(probe: str, device: int) -> dict:
    import torch
    import torch_npu  # noqa: F401

    from vllm_ascend.ops.triton.fla import replayssm_probes
    from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

    torch.npu.set_device(device)
    init_device_properties_triton()
    npu = torch.device(f"npu:{device}")
    started = time.perf_counter()
    repetitions = 1

    if probe in ("cumsum", "serial_prefix"):
        x_cpu = torch.linspace(-0.8, 0.7, 16, dtype=torch.float32)
        x = x_cpu.to(npu)
        out = torch.empty_like(x)
        kernel = (
            replayssm_probes.cumsum_probe_kernel if probe == "cumsum" else replayssm_probes.serial_prefix_probe_kernel
        )
        repetitions = 8
        for _ in range(repetitions):
            kernel[(1,)](x, out, BLOCK=16, num_warps=1, num_stages=1)
        torch.npu.synchronize()
        actual = out.cpu()
        expected = torch.cumsum(x_cpu, dim=0)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    elif probe in ("dot", "masked_zero_dot"):
        generator = torch.Generator().manual_seed(20261020 + device)
        a_cpu = (torch.randn(64, 16, generator=generator) * 0.1).to(torch.float16)
        b_cpu = (torch.randn(16, 64, generator=generator) * 0.1).to(torch.float16)
        a = a_cpu.to(npu)
        b = b_cpu.to(npu)
        out = torch.empty(64, 64, dtype=torch.float32, device=npu)
        repetitions = 4
        if probe == "dot":
            for _ in range(repetitions):
                replayssm_probes.dot_probe_kernel[(1,)](a, b, out, BM=64, BK=16, BN=64, num_warps=1, num_stages=1)
            expected = a_cpu.float() @ b_cpu.float()
        else:
            valid_length = torch.zeros(1, dtype=torch.int32, device=npu)
            for _ in range(repetitions):
                replayssm_probes.masked_zero_dot_probe_kernel[(1,)](
                    a,
                    b,
                    out,
                    valid_length,
                    BM=64,
                    BK=16,
                    BN=64,
                    num_warps=1,
                    num_stages=1,
                )
            expected = torch.zeros(64, 64, dtype=torch.float32)
        torch.npu.synchronize()
        actual = out.cpu()
        torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
    elif probe == "runtime_branch":
        x_cpu = torch.linspace(-1.0, 1.0, 64, dtype=torch.float32)
        x = x_cpu.to(npu)
        flag = torch.zeros(1, dtype=torch.int32, device=npu)
        out = torch.empty_like(x)
        branch_cases = ((0, -1.0), (1, 1.0), (0, -1.0), (1, 1.0))
        repetitions = len(branch_cases)
        for flag_value, delta in branch_cases:
            flag.fill_(flag_value)
            replayssm_probes.runtime_branch_probe_kernel[(1,)](x, flag, out, BLOCK=64, num_warps=1, num_stages=1)
            torch.npu.synchronize()
            torch.testing.assert_close(out.cpu(), x_cpu + delta, atol=0, rtol=0)
    elif probe == "masked_2d_io":
        x_cpu = torch.arange(64 * 64, dtype=torch.float32).reshape(64, 64).to(torch.bfloat16)
        x = x_cpu.to(npu)
        out = torch.full((64, 64), -7.0, dtype=torch.bfloat16, device=npu)
        repetitions = 4
        for _ in range(repetitions):
            replayssm_probes.masked_2d_io_probe_kernel[(1,)](x, out, 63, 61, BM=64, BN=64, num_warps=1, num_stages=1)
        torch.npu.synchronize()
        actual = out.cpu()
        expected = torch.full_like(x_cpu, -7.0)
        expected[:63, :61] = x_cpu[:63, :61]
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    else:
        raise ValueError(f"unknown probe: {probe}")

    return {
        "status": "PASSED",
        "probe": probe,
        "device": device,
        "repetitions": repetitions,
        "elapsed_ms_including_compile": (time.perf_counter() - started) * 1000,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "torch-npu", "triton-ascend", "vllm", "vllm-ascend")
        },
        "module_path": replayssm_probes.__file__,
        "ascend_home_path": os.environ.get("ASCEND_HOME_PATH"),
    }


def git_evidence() -> dict:
    root = Path(__file__).resolve().parents[2]
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False)
    status = subprocess.run(["git", "status", "--short"], cwd=root, capture_output=True, text=True, check=False)
    return {"commit": head.stdout.strip(), "status_short": status.stdout.splitlines()}


def stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=10)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probes", default=",".join(PROBES))
    parser.add_argument("--devices", default="0,1,2,3,4,5")
    parser.add_argument("--probe-timeout", type=int, default=900)
    parser.add_argument("--child-probe", choices=PROBES)
    parser.add_argument("--child-device", type=int, choices=range(64))
    parser.add_argument("--child-report", type=Path)
    args = parser.parse_args()

    if args.child_probe is not None:
        if args.child_device is None or args.child_report is None:
            parser.error("--child-device and --child-report are required for a child probe")
        record = {"status": "FAILED", "probe": args.child_probe, "device": args.child_device}
        try:
            record.update(run_probe(args.child_probe, args.child_device))
        except Exception as exc:
            record.update(error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        write_json(args.child_report, record)
        return 0 if record["status"] == "PASSED" else 1

    probes = parse_csv(args.probes)
    devices_raw = parse_csv(args.devices)
    if not probes or any(probe not in PROBES for probe in probes):
        parser.error(f"--probes must be a comma-separated subset of {','.join(PROBES)}")
    try:
        devices = [int(device) for device in devices_raw]
    except ValueError:
        parser.error("--devices must contain comma-separated integer logical device IDs")
    if len(devices) != len(probes):
        parser.error("--devices must contain exactly one distinct logical device for each probe")
    if len(set(devices)) != len(devices):
        parser.error("parallel probes require distinct logical devices")
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    args.output.mkdir(parents=True)

    children = []
    for probe, device in zip(probes, devices, strict=True):
        report = args.output / f"{probe}-device{device}.json"
        log_path = args.output / f"{probe}-device{device}.log"
        cache_path = args.output / f"triton-cache-{probe}"
        cache_path.mkdir()
        stream = log_path.open("w")
        env = os.environ.copy()
        env["TRITON_CACHE_DIR"] = str(cache_path)
        process = subprocess.Popen(
            child_command(probe, device, report),
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            start_new_session=True,
        )
        children.append((probe, device, report, stream, process, time.monotonic()))

    while any(process.poll() is None for _, _, _, _, process, _ in children):
        now = time.monotonic()
        for _, _, _, _, process, started in children:
            if process.poll() is None and now - started > args.probe_timeout:
                stop_process(process)
        time.sleep(0.2)

    records = []
    for probe, device, report, stream, process, _ in children:
        stream.close()
        if report.is_file():
            record = json.loads(report.read_text())
        elif process.returncode is not None and process.returncode < 0:
            record = {"status": "FAILED", "probe": probe, "device": device, "error": "probe timed out"}
        else:
            record = {
                "status": "FAILED",
                "probe": probe,
                "device": device,
                "error": f"missing report; returncode={process.returncode}",
            }
        records.append(record)

    summary = {
        "level": 2,
        "scope": "isolated Triton-Ascend feature probes for ReplaySSM AR",
        "status": "AUTOMATED_CHECKS_PASSED"
        if all(record["status"] == "PASSED" for record in records)
        else "DIAGNOSTIC_FAILURES_RECORDED",
        "source": git_evidence(),
        "probes": records,
        "next_level_allowed": False,
    }
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "AUTOMATED_CHECKS_PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
