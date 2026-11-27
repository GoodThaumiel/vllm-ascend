# SPDX-License-Identifier: Apache-2.0
"""Summarize Ascend Torch Profiler kernels inside annotated target forwards."""

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

TARGET_PREFIX = "REPLAY_STEADY_TARGET/"


def kernel_group(name, op_type):
    key = (name + op_type).lower().replace("_", "")
    if "recurrentgateddeltarule" in key:
        return "native_gdr"
    if "replayssm" in key:
        if "verifycompactfused" in key:
            return "replay_fused"
        for part in ("verify", "compact", "cursor"):
            if part in key:
                return "replay_" + part
    if "indexcopy" in key or "indexselect" in key:
        return "index_copy_or_select"
    if "scatter" in key or "gather" in key:
        return "scatter_or_gather"
    if "memcpy" in key or "viewcopy" in key or "stridedslice" in key:
        return "copy_or_slice"
    if "fill" in key:
        return "fill"
    if "matmul" in key or "gemm" in key:
        return "matmul"
    if "attention" in key:
        return "attention"
    return "other"


def union_duration(events):
    elapsed = 0.0
    end = float("-inf")
    for start, stop in sorted(events):
        elapsed += max(0.0, stop - max(end, start))
        end = max(end, stop)
    return elapsed


def summarize(trace, kernels):
    if isinstance(trace, dict):
        trace = trace["traceEvents"]
    windows = []
    for event in trace:
        if event.get("ph") == "X" and event.get("name", "").startswith(TARGET_PREFIX):
            start = float(event["ts"])
            windows.append((start, start + float(event["dur"]), event["name"]))
    if not windows:
        raise ValueError("No annotated target decode forwards in trace_view.json")

    rows = [
        (float(row["Start Time(us)"]), float(row["Duration(us)"]), row)
        for row in kernels
        if row["Start Time(us)"].strip() and row["Duration(us)"].strip()
    ]
    forwards = []
    named_samples = defaultdict(list)
    for start, end, label in sorted(windows):
        selected = [(ts, dur, row) for ts, dur, row in rows if start <= ts and ts + dur <= end + 1]
        if not selected:
            raise ValueError(f"No device kernels in {label}; check timestamp alignment")
        groups = defaultdict(list)
        names = defaultdict(list)
        for ts, dur, row in selected:
            group = kernel_group(row["Name"], row["Type"])
            groups[group].append((ts, ts + dur))
            names[row["Name"]].append(dur)
            named_samples[row["Name"]].append(dur)
        intervals = [(ts, ts + dur) for ts, dur, _ in selected]
        forwards.append(
            {
                "label": label,
                "cpu_scope_us": end - start,
                "device_span_us": max(stop for _, stop in intervals) - min(begin for begin, _ in intervals),
                "device_busy_union_us": union_duration(intervals),
                "model_ids": dict(Counter(row["Model ID"] for _, _, row in selected)),
                "groups": {
                    group: {
                        "calls": len(values),
                        "sum_us": sum(stop - begin for begin, stop in values),
                        "union_us": union_duration(values),
                    }
                    for group, values in sorted(groups.items())
                },
                "kernel_sums_us": {name: sum(durations) for name, durations in names.items()},
            }
        )
    return {
        "forwards": forwards,
        "kernel_medians": {
            name: {
                "calls": len(values),
                "call_p50_us": statistics.median(values),
                "per_forward_sum_p50_us": statistics.median(f["kernel_sums_us"].get(name, 0) for f in forwards),
            }
            for name, values in sorted(named_samples.items())
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profiler_output", type=Path, help="ASCEND_PROFILER_OUTPUT directory")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.profiler_output
    with (source / "kernel_details.csv").open(newline="") as file:
        report = summarize(json.loads((source / "trace_view.json").read_text()), csv.DictReader(file))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
