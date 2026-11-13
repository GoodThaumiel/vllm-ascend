"""Compare two RGDR device task times from msprof CSVs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path


def task_times(path: Path, kernel: str, warmup: int, iterations: int) -> list[float]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        fields = {re.sub(r"[^a-z0-9]", "", name.lower()): name for name in reader.fieldnames or ()}
        names = fields.get("opname")
        start = fields.get("taskstarttimeus") or fields.get("taskstarttime")
        duration = fields.get("taskdurationus") or fields.get("taskduration")
        if not all((names, start, duration)):
            raise ValueError(f"{path}: expected Op Name, Task Start Time and Task Duration columns")
        rows = [(float(row[start]), float(row[duration])) for row in reader if kernel in row[names]]
    if len(rows) != warmup + iterations:
        raise ValueError(f"{path}: expected {warmup + iterations} {kernel} launches, found {len(rows)}")
    if any(not math.isfinite(time) or not math.isfinite(cost) or cost <= 0 for time, cost in rows):
        raise ValueError(f"{path}: invalid task timestamp or duration")
    return [cost for _, cost in sorted(rows)[warmup:]]


def summarize(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    return {"p50_us": statistics.median(ordered), "p90_us": ordered[int(0.9 * (len(ordered) - 1))]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path, help="baseline op_summary CSV")
    parser.add_argument("candidate", type=Path, help="cube-grouped op_summary CSV")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--baseline-kernel", default="rgdr_v310_cube_fp16")
    parser.add_argument("--candidate-kernel", default="rgdr_v310_cube_grouped")
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations < 1:
        parser.error("--warmup must be non-negative and --iterations must be positive")
    try:
        baseline = summarize(task_times(args.baseline, args.baseline_kernel, args.warmup, args.iterations))
        candidate = summarize(task_times(args.candidate, args.candidate_kernel, args.warmup, args.iterations))
    except (OSError, ValueError) as error:
        parser.error(str(error))
    passed = candidate["p50_us"] < baseline["p50_us"] and candidate["p90_us"] <= baseline["p90_us"]
    print(
        json.dumps(
            {
                "status": "PASSED" if passed else "FAILED",
                "baseline": baseline,
                "candidate": candidate,
                "p50_speedup": baseline["p50_us"] / candidate["p50_us"],
            },
            indent=2,
        )
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
