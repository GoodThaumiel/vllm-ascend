# SPDX-License-Identifier: Apache-2.0
"""Compare opt-in vllm serve ReplaySSM memory snapshots from worker logs."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import median

MARKER = "REPLAYSSM_MEMORY_JSON "
STAGES = ("model_loaded", "kv_cache_initialized", "graph_capture_before", "graph_capture_after")
FIELDS = (
    "torch_allocated_bytes",
    "torch_reserved_bytes",
    "torch_peak_allocated_bytes",
)


def parse_log(path: Path) -> list[dict]:
    records = []
    for line in path.read_text(errors="replace").splitlines():
        if MARKER not in line:
            continue
        records.append(json.loads(line.split(MARKER, 1)[1]))
    if not records:
        raise ValueError(f"No {MARKER.strip()} records found in {path}")
    return records


def summarize(records: list[dict]) -> dict[str, dict]:
    by_stage: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_stage[record["stage"]].append(record)
    summary = {}
    for stage, stage_records in by_stage.items():
        row = {field: int(median(record[field] for record in stage_records)) for field in FIELDS}
        row["ranks"] = len(stage_records)
        plans = [record["cache_plan"] for record in stage_records if "cache_plan" in record]
        if plans:
            row["planned_device_backing_bytes"] = int(median(plan["planned_device_backing_bytes"] for plan in plans))
            row["logical_state_component_bytes"] = plans[0]["logical_state_component_bytes"]
            row["num_blocks"] = plans[0]["num_blocks"]
        views = [record["cache_views"] for record in stage_records if "cache_views" in record]
        if views:
            row["unique_view_backing_bytes"] = int(median(view["unique_backing_bytes"] for view in views))
            row["sample_views"] = views[0]["sample_views"]
        summary[stage] = row
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True, help="Native GDN vllm serve worker log")
    parser.add_argument("--replay", type=Path, required=True, help="Replay AR vllm serve worker log")
    args = parser.parse_args()
    native = summarize(parse_log(args.native))
    replay = summarize(parse_log(args.replay))
    print("stage | metric | native GiB | replay GiB | replay-native GiB")
    print("--- | --- | ---: | ---: | ---:")
    for stage in STAGES:
        if stage not in native or stage not in replay:
            continue
        for field in (*FIELDS, "planned_device_backing_bytes", "unique_view_backing_bytes"):
            if field not in native[stage] or field not in replay[stage]:
                continue
            a, b = native[stage][field], replay[stage][field]
            gib = 1024**3
            print(f"{stage} | {field} | {a / gib:.3f} | {b / gib:.3f} | {(b - a) / gib:+.3f}")
    print("\nNative cache plan:")
    print(json.dumps(native.get("kv_cache_initialized", {}), indent=2, sort_keys=True))
    print("\nReplay cache plan:")
    print(json.dumps(replay.get("kv_cache_initialized", {}), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
