#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Estimate and optionally allocate the independent 310P ReplaySSM SD ring."""

import argparse
import json
from pathlib import Path

from vllm_ascend._310p.ops.replayssm_sd_demo import (
    ReplaySSM310PGeometry,
    ReplaySSM310PRingDemo,
    estimate_memory,
    geometry_from_hf_config,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--linear-layers", type=int, default=30)
    parser.add_argument("--local-key-heads", type=int, default=8)
    parser.add_argument("--local-value-heads", type=int, default=16)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--mtp", type=int, choices=(3, 5), required=True)
    parser.add_argument("--model-config", type=Path, help="Read Qwen3.5 GDN layer/head counts from config.json")
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--active-requests", type=int)
    parser.add_argument("--allocate-device", help="Allocate D/K/G rings on cpu or npu:0 and report actual payload")
    args = parser.parse_args()
    if args.model_config:
        geometry = geometry_from_hf_config(
            json.loads(args.model_config.read_text()),
            tp=args.tp,
            max_num_seqs=args.max_num_seqs,
            mtp_tokens=args.mtp,
        )
    else:
        geometry = ReplaySSM310PGeometry(
            linear_layers=args.linear_layers,
            local_key_heads=args.local_key_heads,
            local_value_heads=args.local_value_heads,
            max_num_seqs=args.max_num_seqs,
            mtp_tokens=args.mtp,
        )
    demo = ReplaySSM310PRingDemo(geometry)
    result = estimate_memory(geometry, active_requests=args.active_requests)
    result["geometry_source"] = str(args.model_config) if args.model_config else "explicit_or_example_defaults"
    initial_ids = ["request-a", "request-b"][: geometry.max_num_seqs]
    first_assignment = demo.update(initial_ids)
    after_reuse = demo.update(initial_ids[1:] + ["request-c"], ["request-a"])
    result["lifecycle_probe"] = {
        "first_assignment": first_assignment,
        "after_release_and_reuse": after_reuse,
    }
    if args.allocate_device:
        memory_before = None
        if args.allocate_device.startswith("npu"):
            import torch
            import torch_npu  # noqa: F401

            npu_device = torch.device(args.allocate_device)
            torch.npu.synchronize(npu_device)
            memory_before = torch.npu.memory_allocated(npu_device)
        rings = demo.allocate_rings(args.allocate_device)
        result["allocated_ring_bytes_measured"] = demo.allocated_bytes(rings)
        result["allocation_matches_projection"] = (
            result["allocated_ring_bytes_measured"] == result["ring_bytes_allocated"]
        )
        if memory_before is not None:
            torch.npu.synchronize(npu_device)
            result["npu_allocator_delta_bytes"] = torch.npu.memory_allocated(npu_device) - memory_before
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
