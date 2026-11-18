# SPDX-License-Identifier: Apache-2.0
"""Fixed-K MRv1 DFlash for 310P. Decode graphs own draft forward only."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch_npu
from vllm.config import CUDAGraphMode
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.logger import logger
from vllm.v1.attention.backends.utils import CommonAttentionMetadata

from vllm_ascend._310p.attention.dflash_splitfuse import DFlashSplitFuse310, draft_kv_group_order
from vllm_ascend._310p.dflash_config import (
    DFLASH_MAX_DRAFT_TOKENS,
    DFLASH_MRV1_LOADING,
    DFLASH_QUERY_DMA_LANES,
    pack_query_lanes,
    retained_page_range,
    usable_decode_sizes,
    validate_packed_context,
)
from vllm_ascend.ascend_forward_context import set_ascend_forward_context
from vllm_ascend.attention.attention_v1 import AscendMetadata
from vllm_ascend.compilation.acl_graph import ACLGraphWrapper
from vllm_ascend.ops.replayssm.cache import replayssm_sd_enabled
from vllm_ascend.spec_decode.dflash_proposer import AscendDflashProposer

QUERY_FIELDS = 6
REQUEST_INFO = 5


@dataclass(frozen=True)
class DraftKVGroup:
    gid: int
    page_size: int
    page_count: int
    table_width: int
    null_pages: int
    window: int | None
    causal: bool


class DFlashRotary310(nn.Module):
    """Use drafter frequencies with descriptor-owned query cosine/sine."""

    dflash_310p = True

    def __init__(self, rope) -> None:
        super().__init__()
        if rope.head_size != 128 or rope.rotary_dim != rope.head_size:
            raise ValueError("310P DFlash currently requires full 128-wide RoPE")
        self.rope = rope
        self.query_tables = None

    def tables(self, positions, dtype):
        cos, sin = self.rope.cos_sin_cache.index_select(0, positions.to(torch.int32)).to(dtype).chunk(2, dim=-1)
        if self.rope.is_neox_style:
            cos, sin = [torch.cat((part, part), dim=-1) for part in (cos, sin)]
        else:
            cos, sin = [part.repeat_interleave(2, dim=-1) for part in (cos, sin)]
        return tuple(part.contiguous().view(1, positions.numel(), 1, 128) for part in (cos, sin))

    def forward(self, positions, query, key):
        cos, sin = self.query_tables if self.query_tables is not None else self.tables(positions, query.dtype)
        query_rot, key_rot = torch_npu.npu_apply_rotary_pos_emb(
            query.contiguous().view(1, positions.numel(), -1, 128),
            key.contiguous().view(1, positions.numel(), -1, 128),
            cos,
            sin,
            rotary_mode="half" if self.rope.is_neox_style else "interleave",
        )
        return query_rot.view_as(query), key_rot.view_as(key)


class AscendDFlashProposer310(AscendDflashProposer):
    """The 35B TP2 pilot with a per-service DFlash proposal width."""

    def __init__(self, vllm_config, device, runner):
        super().__init__(vllm_config, device, runner)
        spec = self.speculative_config
        parallel = vllm_config.parallel_config
        cache = vllm_config.cache_config
        mode = vllm_config.compilation_config.cudagraph_mode
        if (
            not 1 <= spec.num_speculative_tokens <= DFLASH_MAX_DRAFT_TOKENS
            or spec.draft_sample_method != "greedy"
            or spec.disable_padded_drafter_batch
            or spec.draft_tensor_parallel_size != 2
            or parallel.tensor_parallel_size != 2
            or parallel.pipeline_parallel_size != 1
            or parallel.data_parallel_size != 1
            or parallel.enable_expert_parallel
            or self.dcp_size != 1
            or parallel.prefill_context_parallel_size != 1
            or self.uses_mrope
            or self.use_async_scheduling
            or self.dynamic_spec is not None
            or cache.enable_prefix_caching
            or self.dtype != torch.float16
            or replayssm_sd_enabled(vllm_config)
            or cache.use_replayssm
            or mode not in (CUDAGraphMode.NONE, CUDAGraphMode.FULL_DECODE_ONLY)
            or (mode == CUDAGraphMode.FULL_DECODE_ONLY and not self.use_cuda_graph)
        ):
            raise ValueError("310P DFlash L2 requires text FP16, K=1–7, greedy, TP2 and MRv1; see LEVEL2.md")
        self.query_width = self.num_speculative_tokens + 1
        self._groups: tuple[DraftKVGroup, ...] = ()
        self._batch_buffers = {}
        self._rope = None
        self._draft_route_logged = set()

    def _get_model(self):
        token = DFLASH_MRV1_LOADING.set(True)
        try:
            return super()._get_model()
        finally:
            DFLASH_MRV1_LOADING.reset(token)

    def load_model(self, model):
        super().load_model(model)
        core = self.model.model
        if not hasattr(core, "_num_attn_layers"):
            core._build_fused_kv_buffers()
        # Context KV and query forward must both use the draft's own frequencies.
        # _build_fused_kv_buffers has already saved the original RoPE metadata.
        self._rope = DFlashRotary310(core.layers[0].self_attn.rotary_emb)
        self._rope.rope.cos_sin_cache = self._rope.rope.cos_sin_cache.to(device=self.device, dtype=self.dtype)
        for layer in core.layers:
            if layer.self_attn.rotary_emb is not self._rope.rope:
                raise ValueError("310P DFlash requires one RoPE configuration across draft layers")
            layer.self_attn.rotary_emb = self._rope

    def initialize_attn_backend(self, kv_cache_config, kernel_block_sizes=None):
        core = self.model.model
        layers = list(core.layers)
        names = self.model.get_draft_kv_cache_layer_names()
        gids, self._layer_group_indices = draft_kv_group_order(
            names, [group.layer_names for group in kv_cache_config.kv_cache_groups]
        )
        if len(gids) > 2:
            raise ValueError("310P DFlash supports up to two physical draft KV groups")
        groups: list[DraftKVGroup | None] = [None] * len(gids)
        static_context = self.vllm_config.compilation_config.static_forward_context
        for layer, name, index in zip(layers, names, self._layer_group_indices):
            gid = gids[index]
            table = self.runner.input_batch.block_table[gid]
            key_cache = static_context[name].kv_cache[0]
            if key_cache.dtype != torch.float16 or key_cache.shape[-2] != table.block_size:
                raise ValueError(f"310P DFlash cache and page-table layout disagree for {name}")
            group = DraftKVGroup(
                gid,
                table.block_size,
                key_cache.shape[0],
                table.block_table.np.shape[1],
                table.blocks_per_phys_block,
                layer.self_attn.sliding_window,
                layer.self_attn.causal,
            )
            if groups[index] is not None and groups[index] != group:
                raise ValueError(f"310P DFlash layers in KV group {gid} have incompatible attention layouts")
            groups[index] = group
        self._groups = tuple(group for group in groups if group is not None)
        self.kv_cache_gid = gids[0]
        self.block_size = self._groups[0].page_size
        if self.use_cuda_graph:
            sizes = usable_decode_sizes(
                self.vllm_config.compilation_config.cudagraph_capture_sizes,
                self.max_batch_size,
                self.max_num_tokens,
                self.query_width,
            )
            if not sizes:
                raise ValueError(
                    f"310P DFlash FULL_DECODE_ONLY needs a usable capture size divisible by {self.query_width}"
                )
            logger.info("310P DFlash exact capture token sizes: %s", sorted(sizes))
        if not hasattr(torch.ops.dflash_310p, "prepare_"):
            root = Path(__file__).resolve().parents[3]
            library = root / "experimental/dflash_310p_ascendc/build/libdflash_310p_torch.so"
            if not library.is_file():
                raise FileNotFoundError(f"Build the 310P DFlash Level 1 operator first: {library}")
            torch.ops.load_library(str(library))
        self._context_capacity = (self.max_num_tokens + 7) // 8 * 8
        self._positions_in = torch.empty(self._context_capacity, dtype=torch.int32, device=self.device)
        self._context_out = torch.empty(
            (len(gids), 2, self._context_capacity),
            dtype=torch.int32,
            device=self.device,
        )
        self._group_info = torch.tensor(
            [[group.page_size, group.page_count, 0, 0, 0, 0, 0, 0] for group in self._groups],
            dtype=torch.int32,
            device=self.device,
        )
        logger.info(
            "310P DFlash MRv1 draft KV groups rank=%d groups=%s", get_tensor_model_parallel_rank(), self._groups
        )

    def _buffers(self, rows):
        if rows not in self._batch_buffers:
            caches = []
            for group in self._groups:
                attention = DFlashSplitFuse310(
                    rows=rows,
                    query_width=self.query_width,
                    table_width=group.table_width,
                    page_size=group.page_size,
                    device=self.device,
                    causal=group.causal,
                    window=group.window,
                )
                slots = torch.empty(rows * self.query_width, dtype=torch.int32, device=self.device)
                metadata = AscendMetadata(num_actual_tokens=rows * self.query_width, slot_mapping=slots)
                metadata.dflash_splitfuse = attention
                caches.append(metadata)
            metadata_by_layer = {
                name: caches[index]
                for name, index in zip(self.model.get_draft_kv_cache_layer_names(), self._layer_group_indices)
            }
            rope_tables = tuple(
                torch.empty((1, rows * self.query_width, 1, 128), dtype=self.dtype, device=self.device)
                for _ in range(2)
            )
            self._batch_buffers[rows] = caches, metadata_by_layer, rope_tables
        return self._batch_buffers[rows]

    def _host_tables(self, rows):
        tables = []
        for group in self._groups:
            block_table = self.runner.input_batch.block_table[group.gid]
            table = block_table.block_table.np[:rows].copy()
            allocated = np.arange(group.table_width)[None, :] < block_table.num_blocks_per_row[:rows, None]
            table[~allocated] = -1
            # A physical null block may have been split into multiple kernel pages.
            table[(table >= 0) & (table < group.null_pages)] = 0
            tables.append(torch.from_numpy(table))
        return tables

    def _prepare(self, cad: CommonAttentionMetadata, positions, next_token_ids, rejected, tables):
        rows = cad.num_reqs
        count = positions.shape[-1]
        if positions.ndim == 2:
            # The target VL-shaped wrapper replicates text positions across axes.
            positions = positions[0]
        if positions.ndim != 1 or count > self._context_capacity:
            raise ValueError("310P DFlash expects packed text positions within the allocated context capacity")
        validate_packed_context(cad.query_start_loc_cpu[: rows + 1].tolist(), rows, count)
        reject = torch.zeros_like(next_token_ids) if rejected is None else rejected
        starts = cad.query_start_loc[:rows].to(torch.int32)
        ends = cad.query_start_loc[1 : rows + 1].to(torch.int32)
        request = torch.stack(
            (
                starts,
                ends,
                reject[:rows].to(torch.int32),
                next_token_ids[:rows].to(torch.int32),
                starts,
                starts,
                starts,
                starts,
            ),
            dim=1,
        )
        self._positions_in[:count].copy_(positions.to(torch.int32).contiguous())
        width = max(table.shape[1] for table in tables)
        padded = [torch.nn.functional.pad(table, (0, width - table.shape[1]), value=-1) for table in tables]
        block_tables = torch.stack(padded).to(self.device)
        query = torch.empty(
            (len(tables), rows, QUERY_FIELDS, DFLASH_QUERY_DMA_LANES), dtype=torch.int32, device=self.device
        )
        torch.ops.dflash_310p.prepare_(
            request,
            self._positions_in,
            block_tables,
            self._group_info,
            query,
            self._context_out,
            self.num_speculative_tokens,
            self.max_model_len,
            self.parallel_drafting_token_id,
            False,
        )
        self.runner._trace_dflash("draft_prepare_readback_begin")
        query_host = query.cpu()
        self.runner._trace_dflash("draft_prepare_readback_end")
        if torch.any(query_host[:, :, REQUEST_INFO, 3] != 0):
            raise RuntimeError(f"310P DFlash invalid prepared metadata: {query_host[:, :, REQUEST_INFO, 3].tolist()}")
        return query_host

    def _stage(self, prepared, tables):
        rows = prepared.shape[1]
        total = rows * self.query_width
        caches, _, rope_tables = self._buffers(rows)
        self.input_ids[:total].copy_(pack_query_lanes(prepared[0, :, 0], self.query_width))
        self.positions[:total].copy_(pack_query_lanes(prepared[0, :, 1], self.query_width))
        for index, metadata in enumerate(caches):
            group = self._groups[index]
            lengths = prepared[index, :, REQUEST_INFO, 2].contiguous()
            table = tables[index].clone()
            for row, length in enumerate(lengths.tolist()):
                earliest, latest = retained_page_range(length, self.query_width, group.page_size, group.window)
                retained = table[row, earliest:latest].clone()
                if (
                    retained.numel() != latest - earliest
                    or torch.any(retained < group.null_pages)
                    or torch.any(retained >= group.page_count)
                ):
                    raise RuntimeError(f"310P DFlash invalid visible KV page in group {group.gid}, row {row}")
                table[row].zero_()
                table[row, : retained.numel()].copy_(retained)
                lengths[row] -= earliest * group.page_size
            metadata.slot_mapping.copy_(pack_query_lanes(prepared[index, :, 2], self.query_width))
            metadata.dflash_splitfuse.refresh(lengths, table)
        for target, source in zip(rope_tables, self._rope.tables(self.positions[:total], self.dtype)):
            target.copy_(source)

    def _run_merged_draft(self, num_input_tokens):
        return self.model(
            input_ids=self.input_ids[:num_input_tokens],
            positions=self.positions[:num_input_tokens],
            inputs_embeds=None,
        )

    def _forward(self, rows, mode, descriptor=None, *, is_profile=False):
        metadata, rope_tables = (None, None)
        if self._groups:
            _, metadata, rope_tables = self._buffers(rows)
        with set_ascend_forward_context(
            metadata,
            self.vllm_config,
            num_tokens=rows * self.query_width,
            num_actual_tokens=rows * self.query_width,
            batch_descriptor=descriptor,
            aclgraph_runtime_mode=mode,
            is_draft_model=True,
            in_profile_run=is_profile,
            skip_compiled=True,
        ):
            self._rope.query_tables = rope_tables
            try:
                return self._runnable(rows * self.query_width)
            finally:
                self._rope.query_tables = None

    def _propose(
        self,
        num_speculative_tokens,
        target_token_ids,
        target_positions,
        target_hidden_states,
        next_token_ids,
        token_indices_to_sample,
        common_attn_metadata,
        target_model_batch_desc,
        sampling_metadata,
        num_rejected_tokens_gpu=None,
        **kwargs,
    ):
        self._last_draft_probs = None
        if num_speculative_tokens == 0:
            return torch.empty((common_attn_metadata.num_reqs, 0), dtype=torch.long, device=self.device)
        if num_speculative_tokens != self.num_speculative_tokens:
            raise ValueError(f"310P DFlash expects the configured K={self.num_speculative_tokens}")
        scheduler_output = kwargs.get("scheduler_output")
        if scheduler_output is not None and scheduler_output.scheduled_encoder_inputs:
            raise ValueError("310P DFlash L2 accepts text requests only")
        rows = common_attn_metadata.num_reqs
        count = target_token_ids.numel()
        self.runner._trace_dflash("draft_begin", rows=rows, context_tokens=count)
        # Do not update graph-retained storage until the previous replay has completed.
        if self.use_cuda_graph:
            self.runner._trace_dflash("draft_prior_stream_wait_begin")
            torch.npu.current_stream().synchronize()
            self.runner._trace_dflash("draft_prior_stream_wait_end")
        tables = self._host_tables(rows)
        prepared = self._prepare(
            common_attn_metadata, target_positions, next_token_ids, num_rejected_tokens_gpu, tables
        )
        context_positions = self._context_out[0, 0, :count]
        context_slots = [self._context_out[index, 1, :count] for index in self._layer_group_indices]
        self.runner._trace_dflash("draft_context_kv_begin")
        context = self.model.combine_hidden_states(target_hidden_states)
        with set_ascend_forward_context(
            None, self.vllm_config, num_tokens=count, is_draft_model=True, skip_compiled=True
        ):
            self.model.precompute_and_store_context_kv(context, context_positions, context_slots)
        self.runner._trace_dflash("draft_context_kv_enqueued")
        self.runner._trace_dflash("draft_stage_begin")
        self._stage(prepared, tables)
        self.runner._trace_dflash("draft_stage_enqueued")
        descriptor = getattr(self.runner, "_dflash_target_graph", None)
        mode = CUDAGraphMode.FULL if descriptor is not None else CUDAGraphMode.NONE
        if mode == CUDAGraphMode.FULL:
            if (
                descriptor != target_model_batch_desc
                or descriptor.num_tokens != rows * self.query_width
                or not isinstance(self._runnable, ACLGraphWrapper)
                or descriptor not in self._runnable.concrete_aclgraph_entries
            ):
                raise RuntimeError(f"Missing matching 310P DFlash draft graph for {descriptor}")
        self.runner._trace_dflash("draft_forward_begin", mode=mode.name, tokens=rows * self.query_width)
        hidden = self._forward(rows, mode, descriptor)
        self.runner._trace_dflash("draft_forward_enqueued")
        if mode == CUDAGraphMode.FULL and rows not in self._draft_route_logged:
            self._draft_route_logged.add(rows)
            logger.info(
                "310P DFlash graph replay component=draft rank=%d tokens=%d",
                get_tensor_model_parallel_rank(),
                rows * self.query_width,
            )
        indices = pack_query_lanes(prepared[0, :, 3], self.num_speculative_tokens).to(self.device)
        self.runner._trace_dflash("draft_sample_begin")
        draft_ids, _ = self.compute_draft_token_ids(hidden.index_select(0, indices), None)
        self.runner._trace_dflash("draft_sample_end")
        return draft_ids.view(rows, self.num_speculative_tokens)

    def dummy_run(
        self,
        num_tokens,
        num_reqs=0,
        num_tokens_across_dp=None,
        aclgraph_runtime_mode=CUDAGraphMode.NONE,
        batch_descriptor=None,
        dummy_compute_logits=lambda hidden_states: None,
        is_profile=False,
        **kwargs,
    ):
        rows = min(max(num_reqs, 1), self.max_batch_size)
        if self._groups:
            prepared = torch.zeros((len(self._groups), rows, QUERY_FIELDS, DFLASH_QUERY_DMA_LANES), dtype=torch.int32)
            query_lanes = torch.arange(self.query_width, dtype=torch.int32)
            prepared[:, :, 1, : self.query_width] = query_lanes
            prepared[:, :, REQUEST_INFO, 2] = self.query_width
            tables = []
            for index, group in enumerate(self._groups):
                first_pages = torch.arange(group.null_pages, group.null_pages + rows, dtype=torch.int32)
                if group.page_count < group.null_pages + rows or group.page_size < self.query_width:
                    raise ValueError("310P DFlash graph dummy rows do not fit in allocated draft cache")
                table = torch.zeros((rows, group.table_width), dtype=torch.int32)
                table[:, 0] = first_pages
                tables.append(table)
                prepared[index, :, 2, : self.query_width] = first_pages[:, None] * group.page_size + query_lanes
            self._stage(prepared, tables)
        elif is_profile:
            count = min(num_tokens, self.max_num_tokens)
            with set_ascend_forward_context(
                None, self.vllm_config, num_tokens=count, is_draft_model=True, skip_compiled=True
            ):
                self.model.precompute_and_store_context_kv(
                    self._dflash_hidden_states[:count],
                    self._context_positions_buffer[:count],
                )
        hidden_states = self._forward(rows, aclgraph_runtime_mode, batch_descriptor, is_profile=is_profile)
        dummy_compute_logits(hidden_states)
        if aclgraph_runtime_mode == CUDAGraphMode.FULL:
            logger.info(
                "310P DFlash graph ready component=draft rank=%d tokens=%d",
                get_tensor_model_parallel_rank(),
                rows * self.query_width,
            )
