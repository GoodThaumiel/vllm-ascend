# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    FullAttentionSpec,
    MambaSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend._310p.dflash_config import DFLASH_MAX_DRAFT_TOKENS


def get_310p_dflash_private_page_bytes(vllm_config, groups) -> list[int] | None:
    """Return per-layer backing bytes for each group in the 310P DFlash MRv1 pilot."""
    spec_config = vllm_config.speculative_config
    if (
        spec_config is None
        or spec_config.method != "dflash"
        or not 1 <= spec_config.num_speculative_tokens <= DFLASH_MAX_DRAFT_TOKENS
        or vllm_config.use_v2_model_runner
        or vllm_config.parallel_config.tensor_parallel_size != 2
        # The pilot uses one state group and separate attention groups. The
        # existing allocator can share state storage in other group layouts.
        or len(groups) < 3
        or sum(isinstance(group.kv_cache_spec, MambaSpec) for group in groups) != 1
    ):
        return None

    pages = []
    for group in groups:
        spec = group.kv_cache_spec
        if isinstance(spec, MambaSpec):
            pages.append(spec.real_page_size_bytes)
        elif type(spec) in (FullAttentionSpec, SlidingWindowSpec) and (
            spec.head_size == spec.head_size_v
            and spec.num_heads == spec.num_kv_heads
            and spec.tokens_per_state == 1
            and spec.state_content_bytes is None
            and spec.block_size % 64 == 0
            and spec.num_kv_heads * spec.head_size % 16 == 0
        ):
            # 310P allocates independent NZ K/V tensors of this size. Splitting
            # a manager block into 64/128-token kernel pages preserves its bytes.
            pages.append(spec.real_page_size_bytes)
        else:
            return None
    return pages


def get_310p_shared_cache_slots(kv_cache_groups, layout) -> dict[str, int]:
    """Map layers to slots when uniform Mamba groups can share storage."""
    if not (layout.is_layer_compact and layout.is_block_compact) or not kv_cache_groups:
        return {}

    page_sizes = set()
    attention_group_count = 0
    mamba_spec = None
    mamba_group_size = None
    shared_slots: dict[str, int] = {}
    for group in kv_cache_groups:
        if not group.layer_names:
            return {}
        has_attention = False
        has_mamba = False
        for slot, layer_name in enumerate(group.layer_names):
            group_spec = group.kv_cache_spec
            spec = (
                group_spec.kv_cache_specs[layer_name] if isinstance(group_spec, UniformTypeKVCacheSpecs) else group_spec
            )
            page_sizes.add(spec.page_size_bytes)
            if isinstance(spec, AttentionSpec):
                has_attention = True
            elif isinstance(spec, MambaSpec):
                has_mamba = True
                if mamba_spec is not None and spec != mamba_spec:
                    return {}
                mamba_spec = spec
                shared_slots[layer_name] = slot
            else:
                return {}
        if has_attention and has_mamba:
            return {}
        attention_group_count += has_attention
        if has_mamba:
            # MTP may add attention layers, but Mamba groups still share slots.
            if mamba_group_size is not None and len(group.layer_names) != mamba_group_size:
                return {}
            mamba_group_size = len(group.layer_names)

    if len(page_sizes) != 1 or attention_group_count != 1 or mamba_spec is None:
        return {}
    return shared_slots
