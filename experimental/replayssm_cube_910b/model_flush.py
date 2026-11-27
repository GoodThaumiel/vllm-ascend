# SPDX-License-Identifier: Apache-2.0
"""Weight mixed-flush device graph timings by a measured per-forward histogram."""

import argparse
import bisect
import json
import statistics
from collections import defaultdict
from pathlib import Path


def latency_curves(reports, batch, layout, paths, layers):
    samples = defaultdict(list)
    stack_samples = defaultdict(list)
    schedules = {}
    workload = None
    for report in reports:
        if report["status"] != "PASSED":
            raise ValueError("Use only completed graph reports")
        for row in report["rows"]:
            if row["batch"] != batch or row["path"] not in paths or row["flush_layout"] != layout:
                continue
            key = (
                report["config"]["geometry"],
                row["length"],
                row["pos"],
                row["nonflush_pos"],
                row["base"],
            )
            if workload is None:
                workload = key
            elif workload != key:
                raise ValueError("Reports must use the same geometry, lengths and history positions")
            schedule = tuple(
                row.get(name)
                for name in ("aiv_panel_cols", "aiv_history_suffix", "aiv_bulk_qkv", "mixed_balanced", "compact_tile")
            )
            if schedules.setdefault(row["path"], schedule) != schedule:
                raise ValueError(f"Reports mix operator configurations for {row['path']}")
            bucket = row["path"], row["active_flush"]
            samples[bucket].extend(row["graph_us"])
            if row.get("gdn_stack", {}).get("layers") == layers:
                stack_samples[bucket].extend(row["gdn_stack"]["graph_us"])

    curves = {path: {} for path in paths}
    for bucket, durations in samples.items():
        path, count = bucket
        ordered = sorted(durations)
        stack = stack_samples[bucket]
        curves[path][count] = {
            "samples": len(ordered),
            "layer_p50_us": statistics.median(ordered),
            "layer_p90_us": ordered[(9 * len(ordered) - 1) // 10],
            "gdn_mean_us": statistics.fmean(stack) if stack else layers * statistics.fmean(ordered),
            "gdn_source": "stack" if stack else "per_layer_estimate",
        }
    if not workload or any(not curve for curve in curves.values()):
        raise ValueError("Every path needs at least one timing at the requested batch and layout")
    return workload, curves


def estimate_mean(curve, histogram):
    """Interpolate means between sampled counts. Do not extrapolate."""
    counts = sorted(curve)
    total = sum(histogram.values())
    if total <= 0 or any(weight < 0 for weight in histogram.values()):
        raise ValueError("Histogram counts must be nonnegative, with at least one forward")
    estimate = 0.0
    for count, weight in histogram.items():
        if not weight:
            continue
        index = bisect.bisect_left(counts, count)
        if count in curve:
            mean = curve[count]["gdn_mean_us"]
        elif index == 0 or index == len(counts):
            raise ValueError(f"Flush count {count} is outside measured range {counts[0]}..{counts[-1]}")
        else:
            low, high = counts[index - 1], counts[index]
            low_value, high_value = curve[low]["gdn_mean_us"], curve[high]["gdn_mean_us"]
            mean = low_value + (count - low) / (high - low) * (high_value - low_value)
        estimate += weight * mean
    return estimate / total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path, nargs="+")
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--layout", choices=("prefix", "spread", "random"), default="random")
    parser.add_argument("--paths", nargs="+", default=("native", "mixed_packed_compact_v6", "fused_v24"))
    parser.add_argument("--histogram", type=Path, help="JSON mapping flush row count to observed decode forward count")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.batch < 1 or args.layers < 1 or "native" not in args.paths:
        parser.error("Positive batch and layer counts and a native path are required")
    reports = (json.loads(path.read_text()) for path in args.reports)
    try:
        workload, curves = latency_curves(reports, args.batch, args.layout, args.paths, args.layers)
        result = {
            "batch": args.batch,
            "layers": args.layers,
            "layout": args.layout,
            "workload": workload,
            "curves": curves,
        }
        if args.histogram:
            histogram = {int(count): int(weight) for count, weight in json.loads(args.histogram.read_text()).items()}
            mean_by_path = {path: estimate_mean(curve, histogram) for path, curve in curves.items()}
            result["observed_forwards_by_flush_count"] = histogram
            result["expected_gdn_mean_us"] = mean_by_path
            result["expected_gdn_delta_us"] = {
                path: value - mean_by_path["native"] for path, value in mean_by_path.items()
            }
    except ValueError as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
