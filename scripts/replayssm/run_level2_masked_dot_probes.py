# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Compare masked-dot workarounds concurrently on separate Ascend NPUs."""

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
    "where_after_load",
    "multiply_after_load",
    "remask_after_masked_load",
    "masked_a_only",
    "masked_b_only",
    "materialize_then_dot",
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


def launch_probe(probe: str, kernels, a, b, out, valid_length, scratch_a, scratch_b) -> None:
    launch = dict(BM=64, BK=16, BN=64, num_warps=1, num_stages=1)
    if probe == "where_after_load":
        kernels.dot_where_after_load_probe_kernel[(1,)](a, b, out, valid_length, **launch)
    elif probe == "multiply_after_load":
        kernels.dot_multiply_after_load_probe_kernel[(1,)](a, b, out, valid_length, **launch)
    elif probe == "remask_after_masked_load":
        kernels.dot_remask_after_masked_load_probe_kernel[(1,)](a, b, out, valid_length, **launch)
    elif probe == "masked_a_only":
        kernels.dot_masked_a_only_probe_kernel[(1,)](a, b, out, valid_length, **launch)
    elif probe == "masked_b_only":
        kernels.dot_masked_b_only_probe_kernel[(1,)](a, b, out, valid_length, **launch)
    elif probe == "materialize_then_dot":
        kernels.materialize_masked_dot_operands_probe_kernel[(1,)](a, b, scratch_a, scratch_b, valid_length, **launch)
        kernels.dot_probe_kernel[(1,)](scratch_a, scratch_b, out, **launch)
    else:
        raise ValueError(f"unknown probe: {probe}")


def run_probe(probe: str, device: int) -> dict:
    import torch
    import torch_npu  # noqa: F401

    from vllm_ascend.ops.triton.fla import replayssm_probes
    from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

    torch.npu.set_device(device)
    init_device_properties_triton()
    npu = torch.device(f"npu:{device}")
    generator = torch.Generator().manual_seed(20261020 + device)
    a_cpu = (torch.randn(64, 16, generator=generator) * 0.1).to(torch.float16)
    b_cpu = (torch.randn(16, 64, generator=generator) * 0.1).to(torch.float16)
    a = a_cpu.to(npu)
    b = b_cpu.to(npu)
    out = torch.empty(64, 64, dtype=torch.float32, device=npu)
    valid_length = torch.zeros(1, dtype=torch.int32, device=npu)
    scratch_a = torch.empty_like(a)
    scratch_b = torch.empty_like(b)
    cases = []
    started = time.perf_counter()

    for valid in (0, 7, 16):
        valid_length.fill_(valid)
        for _ in range(4):
            launch_probe(probe, replayssm_probes, a, b, out, valid_length, scratch_a, scratch_b)
        torch.npu.synchronize()
        actual = out.cpu()
        expected = a_cpu[:, :valid].float() @ b_cpu[:valid, :].float()
        absolute_error = (actual - expected).abs()
        finite = actual.isfinite()
        failed = (~finite) | (absolute_error > (2e-3 + 2e-3 * expected.abs()))
        reported_error = absolute_error.masked_fill(~finite, float("inf"))
        max_error = float(reported_error.max())
        mismatches = int(failed.sum())
        cases.append({"valid_length": valid, "max_abs_error": max_error, "mismatched_elements": mismatches})
        if mismatches:
            raise AssertionError(
                f"{probe} failed for valid_length={valid}: mismatches={mismatches}/4096, max_abs_error={max_error}"
            )

    return {
        "status": "PASSED",
        "probe": probe,
        "device": device,
        "repetitions_per_case": 4,
        "cases": cases,
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
        "scope": "masked tl.dot workaround probes for ReplaySSM AR",
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
