# SPDX-License-Identifier: Apache-2.0
"""Draw batch scaling and calibrated rooflines for ONE workload JSON."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def contract(row):
    """Leading multiply/add FLOPs and logical payload bytes, one invocation.

    FMA=2. Excludes gates, casts, metadata, UB/GM temporaries and reuse effects.
    Full-window demo explicitly computes 16 columns (including padding).
    """
    b, t, p, f, path = (row[k] for k in ("batch", "length", "pos", "active_flush", "path"))
    # Older reports predate the 9B option and contain only 27B TP4 rows.
    h, k = row.get("value_heads", 12), row.get("key_heads", 4)
    d, n = 128, 16
    # Native: loaded initial state and t candidate state writes. Core operations:
    # two projections, decay and one outer-product add per candidate.
    native = (b * h * t * 7 * d * d, 2 * b * h * d * d * (1 + t) + b * t * (4 * k * d + 4 * h * d + 6 * h))
    # Target V20: 2T checkpoint matvecs; history & speculative dot/corrections.
    dots = 2 * d * (2 * t * p + t * t)
    history_tree = 2 * t * (16 + 15) * d if p else 0
    spec_updates = 2 * d * t * (t - 1)
    qkv = b * t * (4 * k * d + 4 * h * d + 8 * h)
    ring = b * ((p + t) * (2 * (h + k) * d) + 4 * h * (32 + t))
    verify_io = 2 * b * h * d * d + qkv + ring
    v20 = (b * h * (4 * t * d * d + dots + history_tree + spec_updates), verify_io)
    # Per row rank-one updates all panel columns; dot plus weighted update.
    cube = (b * h * (2 * d * d * n + (p + t) * 4 * d * n), verify_io)
    moved = f * h * (4 * d * d + 2 * p * d) + f * k * 2 * p * d if p else 0
    compact = (f * h * 3 * p * d * d, moved + 4 * b * h * 4)
    short = (f * h * (2 * d * d * n + 2 * d * d) if p else 0, compact[1])
    # Both AIVs prepare the K-side table once per key-head group. Count the
    # FP32 panel stores and tile reads as logical GM traffic.
    mixed = (
        2 * b * h * d * d * n + 4 * b * k * (p + t) * d * n + 3 * b * h * (p + t) * d * n,
        verify_io + 2 * b * h * d * n * 4,
    )
    if path in ("mixed_fused_fold", "mixed_column_recurrence"):
        cols = row.get("aiv_panel_cols", 8)
        pos_counts = ((p, f), (row.get("nonflush_pos", p), b - f))
        histories = sum(pos * count for pos, count in pos_counts)
        with_history = sum(count for pos, count in pos_counts if pos)
        tree_adds = sum(count * ((4 if pos <= 4 else 8 if pos <= 8 else 16) - 1) for pos, count in pos_counts if pos)
        projected_history = (histories + tree_adds + 2 * with_history) * h * d * cols + histories * h * d
        candidate_panel = 3 * t * cols
        if path == "mixed_column_recurrence":
            candidate_panel = t * (t + 1) + t * cols + t * t
        candidate_work = b * h * d * (candidate_panel + 3 * t)
        dot_work = 2 * k * (histories + b * t) * cols * (2 * d - 1)
        folded = f if p else 0
        fold_work = folded * h * d * d * (1 + 2 * p)
        ring_payload = (histories + b * t) * (2 * d * (h + k) + 4 * h)
        checkpoint_io = (2 * b + 4 * folded) * h * d * d
        handoff_io = 4 * b * h * d * (n + cols)
        return (
            2 * b * h * d * d * n + projected_history + candidate_work + dot_work + fold_work,
            checkpoint_io + handoff_io + qkv + ring_payload,
        )
    if path in ("vector_pair", "cube_pair", "mixed_pair"):
        a, c = {"vector_pair": (v20, compact), "cube_pair": (cube, short), "mixed_pair": (mixed, compact)}[path]
        return a[0] + c[0], a[1] + c[1]
    return {
        "native": native,
        "v20": v20,
        "cube_verify": cube,
        "mixed_verify": mixed,
        "compact_v1": compact,
        "cube_compact": short,
        "grouped_projection": (b * h * 2 * d * d * n, b * (h * 2 * d * d + k * 2 * d * n + h * 2 * d * n)),
        "projection": (b * h * 2 * d * d * n, b * h * (2 * d * d + 2 * d * n + 2 * d * n)),
        "fold_gemm": (f * h * 2 * d * d * n, f * h * 2 * (2 * d * n + d * d)),
    }[path]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("report", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bandwidth-gbs", type=float, required=True)
    ap.add_argument("--cube-tflops", type=float, required=True)
    ap.add_argument("--vector-tflops", type=float, required=True)
    ap.add_argument("--roof-source", required=True, help="board calibration run ID or source of ceiling values")
    ap.add_argument("--counters", type=Path, help="CSV: batch,path,flops,hbm_read_bytes,hbm_write_bytes (per replay)")
    args = ap.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not all(math.isfinite(x) and x > 0 for x in (args.bandwidth_gbs, args.cube_tflops, args.vector_tflops)):
        ap.error("all ceilings must be positive and finite")
    report = json.loads(args.report.read_text())
    if report.get("status", "PASSED") != "PASSED" or "error" in report or not report["rows"]:
        ap.error("incomplete report")
    counters = {}
    if args.counters:
        with args.counters.open() as stream:
            for c in csv.DictReader(stream):
                key = int(c["batch"]), c["path"]
                if key in counters:
                    ap.error(f"duplicate {key}")
                counters[key] = tuple(float(c[k]) for k in ("flops", "hbm_read_bytes", "hbm_write_bytes"))
        if set(counters) != {(r["batch"], r["path"]) for r in report["rows"]}:
            ap.error("counter keys differ from report")
    by = defaultdict(list)
    points = []
    for row in report["rows"]:
        b, path, us = row["batch"], row["path"], row["p50_us"]
        if args.counters:
            flops, rd, wr = counters[b, path]
            if not all(math.isfinite(x) and x >= 0 for x in (flops, rd, wr)):
                ap.error("invalid counter")
            size = rd + wr
        else:
            flops, size = contract(row)
        ai = flops / size if size else 0
        rate = flops / (us * 1e6)
        by[path].append((b, us, ai, rate))
        points.append(dict(batch=b, path=path, p50_us=us, flops=flops, bytes=size, flop_per_byte=ai, tflops=rate))
    fig, (lat, roof) = plt.subplots(1, 2, figsize=(14, 6), layout="constrained")
    path_names = {
        "native": "Native RGDR",
        "v20": "Verify V20",
        "compact_v1": "Compact v1",
        "vector_pair": "V20 + compact v1",
        "mixed_verify": "AIC/AIV verify",
        "mixed_pair": "AIC/AIV verify + compact v1",
    }
    for name, rows in by.items():
        rows.sort()
        b, us, ai, rate = zip(*rows)
        label = path_names.get(name, name)
        (line,) = lat.plot(b, us, ".-", label=label)
        positive = [r for r in rows if r[2] > 0 and r[3] > 0]
        if positive:
            roof.plot([r[2] for r in positive], [r[3] for r in positive], ".", color=line.get_color(), label=label)
    xs = [pt["flop_per_byte"] for pt in points if pt["flop_per_byte"] > 0]
    if xs:
        lo = math.log10(min(xs) / 2)
        hi = math.log10(max(max(xs) * 2, 1500 * args.cube_tflops / args.bandwidth_gbs))
        x = [10 ** (lo + (hi - lo) * i / 199) for i in range(200)]
        for label, cap, style in (("Cube", args.cube_tflops, "--"), ("Vector", args.vector_tflops, ":")):
            roof.plot(x, [min(v * args.bandwidth_gbs / 1000, cap) for v in x], style, label=label + " ceiling")
    lat.set(xlabel="Requests per batch", ylabel="Graph replay p50 (µs)", xscale="log", yscale="log")
    roof.set(
        xlabel="FLOP / byte",
        ylabel="TFLOP/s",
        xscale="log",
        yscale="log",
        title="Measured work & HBM traffic" if args.counters else "Model: leading work / logical payload",
    )
    for ax in (lat, roof):
        ax.grid(True, which="both", alpha=0.2)
        ax.legend(fontsize=8)
    config = report["config"]
    fig.suptitle(
        f"{report.get('soc', 'Ascend')} Qwen3.5-{config.get('geometry', '27b').upper()} TP{config.get('tp', 4)} | "
        f"T={config['length']} pos={config['pos']} flush={config['flush']} | ceilings: {args.roof_source}"
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    with args.out.with_suffix(".csv").open("w", newline="") as stream:
        w = csv.DictWriter(stream, fieldnames=list(points[0]))
        w.writeheader()
        w.writerows(points)


if __name__ == "__main__":
    main()
