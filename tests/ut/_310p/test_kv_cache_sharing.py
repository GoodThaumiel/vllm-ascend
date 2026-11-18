# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheGroupSpec,
    MambaSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.kv_cache_layout import KVCacheLayout

from vllm_ascend._310p.dflash_kv_layout import (
    dflash_blocks_per_request,
    dflash_shared_kv_layout,
    make_dflash_shared_kv_groups,
)
from vllm_ascend._310p.kv_cache_sharing import get_310p_dflash_private_page_bytes, get_310p_shared_cache_slots


def _make_groups() -> list[KVCacheGroupSpec]:
    attention = FullAttentionSpec(block_size=4, num_kv_heads=1, head_size=4, dtype=torch.float16)
    mamba = MambaSpec(block_size=4, shapes=((16,),), dtypes=(torch.float32,))
    assert attention.page_size_bytes == mamba.page_size_bytes
    return [
        KVCacheGroupSpec(["L3", "L7"], attention),
        KVCacheGroupSpec(["L0", "L4"], mamba),
        KVCacheGroupSpec(["L1", "L5"], mamba),
        KVCacheGroupSpec(["L2", "L6"], mamba),
    ]


def test_shares_matching_mamba_slots_across_groups_only() -> None:
    assert get_310p_shared_cache_slots(_make_groups(), KVCacheLayout.LBHNC) == {
        "L0": 0,
        "L4": 1,
        "L1": 0,
        "L5": 1,
        "L2": 0,
        "L6": 1,
    }


def test_mtp_attention_layer_does_not_disable_mamba_slot_sharing() -> None:
    groups = _make_groups()
    expected_slots = get_310p_shared_cache_slots(groups, KVCacheLayout.LBNHC)
    assert expected_slots
    groups[0] = KVCacheGroupSpec(
        [*groups[0].layer_names, "mtp.layers.0.self_attn.attn"],
        groups[0].kv_cache_spec,
    )

    assert get_310p_shared_cache_slots(groups, KVCacheLayout.LBNHC) == expected_slots


@pytest.mark.parametrize("layout", [KVCacheLayout.LHBNC, KVCacheLayout.BLHNC])
def test_rejects_layout_without_contiguous_per_layer_pages(layout: KVCacheLayout) -> None:
    assert get_310p_shared_cache_slots(_make_groups(), layout) == {}


def test_rejects_nonuniform_group_sizes_and_mamba_specs() -> None:
    groups = _make_groups()
    assert get_310p_shared_cache_slots(groups[:1], KVCacheLayout.LBHNC) == {}

    groups[2] = KVCacheGroupSpec(["L1"], groups[2].kv_cache_spec)
    assert get_310p_shared_cache_slots(groups, KVCacheLayout.LBHNC) == {}

    groups = _make_groups()
    different_mamba = MambaSpec(block_size=4, shapes=((32,),), dtypes=(torch.float16,))
    assert different_mamba.page_size_bytes == groups[1].kv_cache_spec.page_size_bytes
    groups[2] = KVCacheGroupSpec(["L1", "L5"], different_mamba)
    assert get_310p_shared_cache_slots(groups, KVCacheLayout.LBHNC) == {}


def test_rejects_mismatched_page_sizes_or_multiple_attention_groups() -> None:
    groups = _make_groups()
    smaller_mamba = MambaSpec(block_size=4, shapes=((8,),), dtypes=(torch.float32,))
    for index in range(1, len(groups)):
        groups[index] = KVCacheGroupSpec(groups[index].layer_names, smaller_mamba)
    assert get_310p_shared_cache_slots(groups, KVCacheLayout.LBHNC) == {}

    groups = _make_groups()
    groups[2] = KVCacheGroupSpec(groups[2].layer_names, groups[0].kv_cache_spec)
    assert get_310p_shared_cache_slots(groups, KVCacheLayout.LBHNC) == {}


def test_uses_per_layer_specs_from_uniform_type_group() -> None:
    groups = _make_groups()
    mamba = groups[1].kv_cache_spec
    groups[1] = KVCacheGroupSpec(
        ["L0", "L4"],
        UniformTypeKVCacheSpecs(
            block_size=mamba.block_size,
            kv_cache_specs={"L0": mamba, "L4": mamba},
        ),
    )
    assert get_310p_shared_cache_slots(groups, KVCacheLayout.LBNHC)["L4"] == 1


def test_dflash_private_pages_charge_backing_bytes_without_changing_cache_geometry() -> None:
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(method="dflash", num_speculative_tokens=7),
        use_v2_model_runner=False,
        parallel_config=SimpleNamespace(tensor_parallel_size=2),
    )
    common_page = 640 * 1024
    target = FullAttentionSpec(
        block_size=640, num_kv_heads=1, head_size=256, dtype=torch.float16, page_size_padded=common_page
    )
    draft_swa = SlidingWindowSpec(
        block_size=128,
        num_kv_heads=2,
        head_size=128,
        dtype=torch.float16,
        sliding_window=4096,
        page_size_padded=common_page,
    )
    state = MambaSpec(block_size=4096, shapes=((16, 128, 128),), dtypes=(torch.float16,), page_size_padded=common_page)
    groups = [
        KVCacheGroupSpec(["target.attn"], target),
        KVCacheGroupSpec(["draft.attn"], draft_swa),
        KVCacheGroupSpec(["target.linear_attn"], state),
    ]

    assert get_310p_dflash_private_page_bytes(config, groups) == [640 * 1024, 128 * 1024, 512 * 1024]
    assert [group.kv_cache_spec.block_size for group in groups] == [640, 128, 4096]
    assert [group.kv_cache_spec.page_size_bytes for group in groups] == [common_page] * 3
    config.speculative_config.method = "mtp"
    assert get_310p_dflash_private_page_bytes(config, groups) is None


def test_310p_dflash_grouping_reuses_slots_and_preserves_draft_attention(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_ASCEND_310P_DFLASH_SHARED_KV", "1")
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(method="dflash", num_speculative_tokens=7),
        use_v2_model_runner=False,
        parallel_config=SimpleNamespace(tensor_parallel_size=2, pipeline_parallel_size=1),
        model_config=SimpleNamespace(
            get_total_num_hidden_layers=lambda: 40,
            max_model_len=4096,
            original_max_model_len=4096,
        ),
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        kv_transfer_config=None,
    )
    target = FullAttentionSpec(block_size=640, num_kv_heads=1, head_size=256, dtype=torch.float16)
    draft = FullAttentionSpec(block_size=640, num_kv_heads=4, head_size=128, dtype=torch.float16, non_causal=True)
    sliding = SlidingWindowSpec(
        block_size=128,
        num_kv_heads=4,
        head_size=128,
        dtype=torch.float16,
        sliding_window=4096,
    )
    state = MambaSpec(
        block_size=4096,
        shapes=((10, 2560), (16, 128, 128)),
        dtypes=(torch.float16, torch.float16),
        num_speculative_blocks=7,
    )
    specs = {f"model.layers.{idx}.linear_attn": state for idx in range(30)}
    specs.update({f"model.layers.{idx}.self_attn.attn": target for idx in range(30, 40)})
    specs.update({f"model.layers.{idx}.self_attn.attn": sliding for idx in range(40, 45)})
    specs["model.layers.45.self_attn.attn"] = draft

    groups = make_dflash_shared_kv_groups(config, specs)
    assert groups is not None
    assert sorted(name for group in groups for name in group.layer_names) == sorted(specs)
    assert [len(group.layer_names) for group in groups] == [6] * 5 + [5, 5, 5, 1]
    assert [group.kv_cache_spec.block_size for group in groups] == [4096] * 5 + [768, 768, 384, 384]
    assert groups[-2].kv_cache_spec.sliding_window == 4096
    assert groups[-1].kv_cache_spec.non_causal
    assert sliding.block_size == 128 and draft.block_size == 640
    layout = dflash_shared_kv_layout(config, groups)
    assert layout is not None
    assert layout.bytes_per_pool_block == 6 * state.real_page_size_bytes + 5 * 768 * 1024
    assert dflash_blocks_per_request(groups, 2455) == 62
    for group in groups:
        slots = layout.state_slots if isinstance(group.kv_cache_spec, MambaSpec) else layout.attention_slots
        assert len({slots[name] for name in group.layer_names}) == len(group.layer_names)
    config.speculative_config.num_speculative_tokens = 5
    specs = {
        name: replace(spec, num_speculative_blocks=5) if isinstance(spec, MambaSpec) else spec
        for name, spec in specs.items()
    }
    groups = make_dflash_shared_kv_groups(config, specs)
    assert groups is not None and dflash_shared_kv_layout(config, groups) is not None
    assert dflash_blocks_per_request(groups, 2453) == 52
    monkeypatch.setenv("VLLM_ASCEND_310P_DFLASH_SHARED_KV", "0")
    assert make_dflash_shared_kv_groups(config, specs) is None
    assert dflash_shared_kv_layout(config, groups) is None


@pytest.mark.parametrize("mode, expected_blocks", [("physical", 6), ("v24", 10), ("explicit", 6)])
def test_310p_dflash_worker_advertises_the_correct_number_of_blocks(monkeypatch, mode, expected_blocks) -> None:
    from vllm_ascend._310p import worker_310p

    page = SimpleNamespace(page_size_bytes=1024)
    groups = [SimpleNamespace(layer_names=["state.0", "state.1", "state.2"], kv_cache_spec=page)]
    monkeypatch.setattr(worker_310p, "get_kv_cache_groups", lambda *_: groups)
    monkeypatch.setattr(worker_310p, "dflash_shared_kv_layout", lambda *_: SimpleNamespace(bytes_per_pool_block=5120))
    worker = SimpleNamespace(
        get_kv_cache_spec=lambda: {},
        vllm_config=SimpleNamespace(cache_config=SimpleNamespace(get_resolved_kv_cache_layout=lambda: None)),
    )
    # The upstream planner always divides by the largest logical group.
    planner_bytes = worker_310p.NPUWorker310._scale_kv_cache_memory_for_multi_group(worker, 30720, mode=mode)
    assert planner_bytes == expected_blocks * 3072
