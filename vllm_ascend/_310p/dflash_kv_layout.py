# SPDX-License-Identifier: Apache-2.0
"""Shared backing for the 310P DFlash MRv1 KV groups."""

import math
import re
from collections import defaultdict
from dataclasses import dataclass, replace

import torch
from vllm.utils.math_utils import cdiv, round_up
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec, MambaSpec, SlidingWindowSpec

from vllm_ascend import envs
from vllm_ascend._310p.dflash_config import DFLASH_MAX_DRAFT_TOKENS


def dflash_shared_kv_enabled(config) -> bool:
    spec = config.speculative_config
    return (
        envs.VLLM_ASCEND_310P_DFLASH_SHARED_KV
        and spec is not None
        and spec.method == "dflash"
        and 1 <= spec.num_speculative_tokens <= DFLASH_MAX_DRAFT_TOKENS
        and not config.use_v2_model_runner
        and config.parallel_config.tensor_parallel_size == 2
        and config.parallel_config.pipeline_parallel_size == 1
        and config.model_config.original_max_model_len != -1
        and not config.cache_config.enable_prefix_caching
        and config.kv_transfer_config is None
    )


def _is_dense_attention(spec) -> bool:
    return (
        type(spec) in (FullAttentionSpec, SlidingWindowSpec)
        and spec.dtype == torch.float16
        and spec.head_size == spec.head_size_v
        and spec.head_size in (64, 128, 256)
        and spec.num_heads == spec.num_kv_heads
        and spec.tokens_per_state == 1
        and spec.state_content_bytes is None
    )


@dataclass(frozen=True)
class DFlashSharedKV:
    state_slots: dict[str, int]
    attention_slots: dict[str, int]
    bytes_per_pool_block: int


def dflash_shared_kv_layout(config, groups) -> DFlashSharedKV | None:
    if not dflash_shared_kv_enabled(config):
        return None
    state_groups = [group for group in groups if isinstance(group.kv_cache_spec, MambaSpec)]
    attention_groups = [group for group in groups if _is_dense_attention(group.kv_cache_spec)]
    state_specs = {group.kv_cache_spec for group in state_groups}
    attention_pages = {group.kv_cache_spec.real_page_size_bytes for group in attention_groups}
    if (
        not all(group.layer_names and not group.host_resident for group in groups)
        or len(state_groups) + len(attention_groups) != len(groups)
        or len(state_specs) != 1
        or len(attention_pages) != 1
        or any(group.kv_cache_spec.block_size % 128 for group in attention_groups)
    ):
        return None
    state_spec = next(iter(state_specs))
    if state_spec.mamba_cache_mode != "none":
        return None
    state_slots = {name: slot for group in state_groups for slot, name in enumerate(group.layer_names)}
    attention_slots = {name: slot for group in attention_groups for slot, name in enumerate(group.layer_names)}
    physical_bytes = (max(state_slots.values()) + 1) * state_spec.real_page_size_bytes + (
        max(attention_slots.values()) + 1
    ) * next(iter(attention_pages))
    return DFlashSharedKV(state_slots, attention_slots, physical_bytes)


def dflash_blocks_per_request(groups, length: int) -> int:
    return sum(
        cdiv(min(length, spec.sliding_window + spec.block_size - 1), spec.block_size)
        if isinstance(spec, SlidingWindowSpec)
        else cdiv(length, spec.block_size) + getattr(spec, "num_speculative_blocks", 0)
        for spec in (group.kv_cache_spec for group in groups)
    )


def make_dflash_shared_kv_groups(config, specs) -> list[KVCacheGroupSpec] | None:
    """Split large target buckets into shareable slots, preserving both draft tables."""
    if not dflash_shared_kv_enabled(config):
        return None
    target_layers = config.model_config.get_total_num_hidden_layers()
    state = defaultdict(list)
    target = defaultdict(list)
    draft = defaultdict(list)
    for name, spec in specs.items():
        layer = re.search(r"(?:^|\.)layers\.(\d+)\.", name)
        if layer is None:
            return None
        if isinstance(spec, MambaSpec) and int(layer[1]) < target_layers:
            state[spec].append(name)
        elif _is_dense_attention(spec):
            bucket = target if int(layer[1]) < target_layers else draft
            bucket[spec].append(name)
        else:
            return None
    if len(state) != 1 or len(target) != 1 or not 1 <= len(draft) <= 2:
        return None
    state_spec, state_names = next(iter(state.items()))
    target_spec, target_names = next(iter(target.items()))
    if state_spec.mamba_cache_mode != "none" or config.model_config.max_model_len > state_spec.block_size:
        return None

    # Pages from different groups can alias only if an ID has the same byte
    # span in every NZ view. Choose the smallest aligned common page that can
    # accommodate the existing target page, then express it in each group's
    # tokens. 310P still runs attention on the underlying 64/128-token pages.
    bytes_per_token = {spec: spec.real_page_size_bytes // spec.block_size for spec in (target_spec, *draft)}
    page_alignment = math.lcm(*(128 * size for size in bytes_per_token.values()))
    attention_page = round_up(target_spec.real_page_size_bytes, page_alignment)
    if attention_page > 2 * target_spec.real_page_size_bytes:
        return None
    planner_page = max(attention_page, state_spec.real_page_size_bytes)
    normalized = {
        spec: replace(spec, block_size=attention_page // size, page_size_padded=planner_page)
        for spec, size in bytes_per_token.items()
    }
    state_spec = replace(state_spec, page_size_padded=planner_page)
    draft_groups = [KVCacheGroupSpec(names, normalized[spec]) for spec, names in draft.items()]
    length = config.model_config.max_model_len
    draft_blocks = dflash_blocks_per_request(draft_groups, length)
    draft_width = max(len(names) for names in draft.values())
    state_blocks = 1 + state_spec.num_speculative_blocks
    target_blocks = cdiv(length, normalized[target_spec].block_size)

    # v24 reuses equal-sized slots across groups. More groups reduce backing
    # per ID but make each request claim more IDs, so score both together.
    _, state_count, target_count = min(
        (
            (
                cdiv(len(state_names), states) * state_spec.real_page_size_bytes
                + max(cdiv(len(target_names), targets), draft_width) * attention_page
            )
            * (states * state_blocks + targets * target_blocks + draft_blocks),
            states,
            targets,
        )
        for states in range(1, len(state_names) + 1)
        for targets in range(1, len(target_names) + 1)
    )
    return [
        *[KVCacheGroupSpec(state_names[i::state_count], state_spec) for i in range(state_count)],
        *[KVCacheGroupSpec(target_names[i::target_count], normalized[target_spec]) for i in range(target_count)],
        *draft_groups,
    ]
