# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Read-only cache and allocator accounting for ReplaySSM serve comparisons."""

from __future__ import annotations

import os
from collections import defaultdict
from math import prod
from typing import Any

import torch


def cache_plan_snapshot(kv_cache_config: Any) -> dict[str, Any]:
    """Account for planned storage and Mamba components without NPU allocations.

    Component bytes describe logical payload. The configured allocation sum
    can double-count overlaid hybrid groups; deduplicated live view storage
    and allocator deltas are needed to estimate physical cache residency.
    """
    num_blocks = int(kv_cache_config.num_blocks)
    allocations = kv_cache_config.kv_cache_tensors
    component_bytes: dict[str, int] = defaultdict(int)
    group_pages: list[dict[str, Any]] = []
    for group in kv_cache_config.kv_cache_groups:
        spec = group.kv_cache_spec
        layer_count = len(group.layer_names)
        layer_specs = getattr(spec, "kv_cache_specs", None)
        if layer_specs is None:
            layer_specs = dict.fromkeys(group.layer_names, spec)
        entry: dict[str, Any] = {
            "kind": type(spec).__name__,
            "layers": layer_count,
            "page_bytes": int(spec.page_size_bytes),
            "logical_page_bytes": int(getattr(spec, "real_page_size_bytes", spec.page_size_bytes)),
        }
        sample_components: dict[str, list[dict[str, Any]]] = {}
        sample_mamba_layout: dict[str, dict[str, int]] = {}
        for layer_spec in layer_specs.values():
            if not (hasattr(layer_spec, "shapes") and hasattr(layer_spec, "dtypes")):
                continue
            parts: list[dict[str, Any]] = []
            for index, (shape, dtype) in enumerate(zip(layer_spec.shapes, layer_spec.dtypes)):
                bytes_per_state = prod(shape) * torch.empty(0, dtype=dtype, device="cpu").element_size()
                part = {
                    "index": index,
                    "shape": list(shape),
                    "dtype": str(dtype),
                    "bytes_per_state": bytes_per_state,
                }
                parts.append(part)
                component_bytes[f"{type(layer_spec).__name__}.state_{index}"] += bytes_per_state * num_blocks
            sample_components.setdefault(type(layer_spec).__name__, parts)
            if hasattr(layer_spec, "num_speculative_blocks"):
                sample_mamba_layout.setdefault(
                    type(layer_spec).__name__,
                    {
                        "num_speculative_blocks": int(layer_spec.num_speculative_blocks),
                        "page_size_bytes": int(layer_spec.page_size_bytes),
                        "blocks_per_request_none": 1 + int(layer_spec.num_speculative_blocks),
                    },
                )
        if sample_components:
            entry["sample_components_by_kind"] = sample_components
        if sample_mamba_layout:
            entry["sample_mamba_layout_by_kind"] = sample_mamba_layout
        group_pages.append(entry)
    return {
        "num_blocks": num_blocks,
        "planned_device_backing_bytes": sum(int(tensor.size) for tensor in allocations if not tensor.host_resident),
        "planned_host_backing_bytes": sum(int(tensor.size) for tensor in allocations if tensor.host_resident),
        "planned_allocation_count": len(allocations),
        "logical_state_component_bytes": dict(component_bytes),
        "groups": group_pages,
    }


def cache_view_snapshot(kv_caches: dict[str, Any], sample_layers: set[str] | None = None) -> dict[str, Any]:
    """Describe live views and deduplicate their backing storages by pointer."""
    backing: dict[int, int] = {}
    samples: dict[str, list[dict[str, Any]]] = {}
    view_count = 0
    for layer_name, cache in kv_caches.items():
        views = cache if isinstance(cache, (tuple, list)) else (cache,)
        layer_views = []
        for view in views:
            if not isinstance(view, torch.Tensor):
                continue
            storage = view.untyped_storage()
            backing[storage.data_ptr()] = storage.nbytes()
            view_count += 1
            layer_views.append(
                {
                    "shape": list(view.shape),
                    "stride": list(view.stride()),
                    "storage_offset": view.storage_offset(),
                    "contiguous": view.is_contiguous(),
                    "dtype": str(view.dtype),
                }
            )
        if layer_views and (layer_name in sample_layers if sample_layers is not None else len(samples) < 2):
            samples[layer_name] = layer_views
    return {
        "view_count": view_count,
        "unique_backing_count": len(backing),
        "unique_backing_bytes": sum(backing.values()),
        "sample_views": samples,
    }


def memory_snapshot(
    *, stage: str, use_replayssm: bool, kv_cache_config: Any = None, kv_caches: Any = None
) -> dict[str, Any]:
    """Collect a phase snapshot; NPU memory APIs are called only at phase boundaries."""
    report: dict[str, Any] = {
        "stage": stage,
        "pid": os.getpid(),
        "device": torch.npu.current_device(),
        "use_replayssm": use_replayssm,
        "torch_allocated_bytes": torch.npu.memory_allocated(),
        "torch_reserved_bytes": torch.npu.memory_reserved(),
        "torch_peak_allocated_bytes": torch.npu.max_memory_allocated(),
    }
    if kv_cache_config is not None:
        report["cache_plan"] = cache_plan_snapshot(kv_cache_config)
    if kv_caches is not None:
        sample_layers = None
        if kv_cache_config is not None:
            sample_layers = set()
            for group in kv_cache_config.kv_cache_groups:
                layer_specs = getattr(group.kv_cache_spec, "kv_cache_specs", None)
                if layer_specs is None:
                    if group.layer_names:
                        sample_layers.add(group.layer_names[0])
                else:
                    seen_kinds = set()
                    for layer_name, layer_spec in layer_specs.items():
                        kind = type(layer_spec).__name__
                        if kind not in seen_kinds:
                            sample_layers.add(layer_name)
                            seen_kinds.add(kind)
        report["cache_views"] = cache_view_snapshot(kv_caches, sample_layers)
    return report
