# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Run isolated native baseline checks and retain a report even after failure."""

import argparse
import json
import os
import subprocess
import sys
import tarfile
import time
from datetime import UTC, datetime
from pathlib import Path

from level1_common import CASES, summarize, write_json


def run_case(command: list[str], output: Path, case: str, timeout: int, env: dict) -> dict:
    report_path = output / f"{case}.json"
    started = time.monotonic()
    try:
        with (output / f"{case}.log").open("w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            try:
                returncode = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                # Stop the entire TP process group, not just the Python parent.
                import signal

                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                return {"case": case, "status": "FAILED", "error": f"Timed out after {timeout}s"}
        if report_path.is_file():
            try:
                record = json.loads(report_path.read_text())
                if not isinstance(record, dict):
                    raise ValueError("worker report is not a JSON object")
            except (ValueError, OSError) as exc:
                record = {"case": case, "status": "FAILED", "error": f"Unreadable worker report: {exc}"}
        else:
            record = {"case": case, "status": "FAILED", "error": "Worker exited without a report; inspect case log"}
        if record.get("case") != case or returncode != 0 or record.get("status") != "PASSED":
            record.update(case=case, status="FAILED")
        record.update(returncode=returncode, elapsed_seconds=round(time.monotonic() - started, 3))
        return record
    except OSError as exc:
        return {"case": case, "status": "FAILED", "error": str(exc)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", required=True, type=Path, help="Local Qwen3.6-27B directory; never downloads weights"
    )
    parser.add_argument("--tp", required=True, type=int, choices=(4, 8))
    parser.add_argument("--runner", choices=("v1", "v2"), default="v1")
    parser.add_argument("--output", type=Path, required=True, help="New directory; existing directories are refused")
    parser.add_argument("--memory-utilization", type=float, default=0.85)
    parser.add_argument("--spec-tokens", type=int, default=3, choices=(1, 3))
    parser.add_argument("--timeout", type=int, default=1800, help="Timeout per isolated case, seconds")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only; does not validate hardware")
    args = parser.parse_args()
    if not 0 < args.memory_utilization < 1 or args.timeout <= 0:
        parser.error("memory-utilization must be in (0, 1) and timeout must be positive")
    model = args.model.expanduser().resolve()
    if not (model / "config.json").is_file():
        parser.error("model must be a local directory containing config.json")
    config = json.loads((model / "config.json").read_text())
    if config.get("model_type") not in ("qwen3_5", "qwen3_5_text"):
        parser.error("this level-1 matrix targets the Qwen3.5/Qwen3.6 dense GDN architecture")
    output = args.output.expanduser().resolve()
    archive = output.with_name(output.name + ".tar.gz")
    if archive.exists():
        parser.error(f"archive already exists: {archive}")
    worker = Path(__file__).with_name("level1_worker.py")
    commands = {
        case: [
            sys.executable,
            str(worker),
            "--case",
            case,
            "--model",
            str(model),
            "--tp",
            str(args.tp),
            "--runner",
            args.runner,
            "--report",
            str(output / f"{case}.json"),
            "--memory-utilization",
            str(args.memory_utilization),
            "--spec-tokens",
            str(args.spec_tokens),
        ]
        for case in CASES
    }
    if args.dry_run:
        print(json.dumps({"status": "DRY_RUN_NOT_VALIDATED", "commands": commands}, indent=2))
        return 0
    output.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[2]
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=False)
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo, capture_output=True, text=True, check=False
    )
    manifest = {
        "schema_version": 1,
        "level": 1,
        "created_utc": datetime.now(UTC).isoformat(),
        "source_commit": revision.stdout.strip(),
        "tracked_source_dirty": bool(dirty.stdout.strip()),
        "model": str(model),
        "tp": args.tp,
        "runner": args.runner,
        "cann_required": "9.2.0",
        "commands": commands,
    }
    write_json(output / "manifest.json", manifest)
    env = os.environ.copy()
    env["VLLM_USE_V2_MODEL_RUNNER"] = "1" if args.runner == "v2" else "0"
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    # Prepend this checkout while retaining the caller's existing module path.
    env["PYTHONPATH"] = str(repo) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    records = []
    summary = summarize(records)
    write_json(output / "summary.json", summary)
    for case, command in commands.items():
        print(f"[{case}] running; log: {output / (case + '.log')}", flush=True)
        record = run_case(command, output, case, args.timeout, env)
        records.append(record)
        summary = summarize(records)
        write_json(output / "summary.json", summary)
        print(f"[{case}] {record['status']}", flush=True)
        if record["status"] != "PASSED":
            break
    print(f"{summary['status']}; user confirmation PENDING; report: {output / 'summary.json'}", flush=True)
    with tarfile.open(archive, "x:gz") as bundle:
        for path in sorted(output.iterdir()):
            if path.suffix in (".json", ".log"):
                bundle.add(path, arcname=f"{output.name}/{path.name}")
    print(f"Feedback bundle: {archive}", flush=True)
    return 0 if summary["status"] == "AUTOMATED_CHECKS_PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
