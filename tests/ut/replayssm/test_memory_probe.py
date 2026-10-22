# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks for ReplaySSM cache accounting."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "vllm_ascend/ops/replayssm/memory_probe.py"
SPEC = importlib.util.spec_from_file_location("replayssm_memory_probe", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


class MambaSpec:
    shapes = ((2, 3), (4, 5), (2, 2))
    dtypes = (torch.bfloat16, torch.bfloat16, torch.float32)
    page_size_bytes = 256
    real_page_size_bytes = 68


def test_cache_plan_counts_components_and_physical_allocation_once():
    config = SimpleNamespace(
        num_blocks=10,
        kv_cache_tensors=[SimpleNamespace(size=1024, host_resident=False)],
        kv_cache_groups=[SimpleNamespace(layer_names=["gdn.0", "gdn.1"], kv_cache_spec=MambaSpec())],
    )
    result = probe.cache_plan_snapshot(config)
    assert result["planned_device_backing_bytes"] == 1024
    assert result["logical_state_component_bytes"] == {
        "MambaSpec.state_0": 240,
        "MambaSpec.state_1": 800,
        "MambaSpec.state_2": 320,
    }
    assert result["groups"][0]["page_bytes"] == 256


def test_cache_views_deduplicate_page_strided_backing():
    backing = torch.empty(128, dtype=torch.bfloat16)
    first = torch.as_strided(backing, (2, 4), (32, 1))
    second = torch.as_strided(backing, (2, 4), (32, 1), storage_offset=8)
    result = probe.cache_view_snapshot({"gdn.0": (first, second)}, {"gdn.0"})
    assert result["view_count"] == 2
    assert result["unique_backing_count"] == 1
    assert result["unique_backing_bytes"] == backing.untyped_storage().nbytes()
    assert result["sample_views"]["gdn.0"][0]["stride"] == [32, 1]
    assert not result["sample_views"]["gdn.0"][0]["contiguous"]


def test_uniform_group_counts_embedded_mamba_specs():
    uniform = SimpleNamespace(
        kv_cache_specs={"gdn.0": MambaSpec(), "gdn.1": MambaSpec()},
        page_size_bytes=512,
    )
    config = SimpleNamespace(
        num_blocks=10,
        kv_cache_tensors=[SimpleNamespace(size=1024, host_resident=False)],
        kv_cache_groups=[SimpleNamespace(layer_names=["gdn.0", "gdn.1"], kv_cache_spec=uniform)],
    )
    result = probe.cache_plan_snapshot(config)
    assert result["logical_state_component_bytes"]["MambaSpec.state_1"] == 800
    assert result["groups"][0]["page_bytes"] == 512
