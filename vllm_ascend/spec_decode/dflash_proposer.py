from copy import copy
from typing import Any

import torch
from vllm.config import CUDAGraphMode, VllmConfig, get_layers_from_vllm_config
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.triton_utils import triton
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import SlidingWindowSpec, UniformTypeKVCacheSpecs
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import _EXTRA_CTX, set_ascend_forward_context
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.ops.triton.spec_decode.utils import copy_and_expand_dflash_and_dspark_inputs_kernel
from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num, init_device_properties_triton
from vllm_ascend.spec_decode.eagle_proposer import AscendEagleProposer
from vllm_ascend.spec_decode.utils import DynamicSpecScheduler, SlidingWindowAdapter

_COPY_EXPAND_TILE_SIZE = 256


def _compute_num_programs(num_context_total: int, num_query_total: int) -> int:
    """Number of programs to launch for copy_and_expand_dflash_and_dspark_inputs_kernel:
    one program per TILE_SIZE chunk of the larger work range, capped at the
    vector-core count (the kernel grid-strides, so any count fully covers the work).

    ``init_device_properties_triton`` is idempotent (guarded by an unset
    sentinel), so calling it here keeps the helper self-contained for unit
    tests that bypass worker startup.
    """
    init_device_properties_triton()
    # The kernel runs two independent grid-stride loops, over
    # [0, num_context_total) and [0, num_query_total); each is fully covered for
    # any program count (a too-small count just iterates more). Only the larger
    # bound decides how many programs do real work, so size on max(...). Using
    # the sum would launch extra programs that are idle in *both* loops.
    num_blocks_needed = triton.cdiv(max(num_context_total, num_query_total), _COPY_EXPAND_TILE_SIZE)
    return min(num_blocks_needed, get_vectorcore_num())


class AscendDflashProposer(AscendEagleProposer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        super().__init__(
            vllm_config,
            device,
            runner=runner,
        )

        self.mixed_dflash_attention = self.method == "dflash" and vllm_config._dflash_needs_multi_kv_group()
        self.max_query_tokens = self.max_batch_size * (1 + self.num_speculative_tokens)
        self.max_positions = self.max_num_tokens + self.max_query_tokens

        self._context_slot_mapping_buffers = torch.zeros(
            self.max_num_tokens,
            dtype=torch.int32,
            device=device,
        )

        self._slot_mapping_buffer = torch.zeros(
            self.max_query_tokens,
            dtype=torch.int32,
            device=device,
        )

        self._context_positions_buffer = torch.zeros(
            self.max_num_tokens,
            dtype=torch.int32,
            device=device,
        )

        self.positions = torch.zeros(
            self.max_query_tokens,
            dtype=torch.int32,
            device=device,
        )

        self.arange_dflash = torch.arange(self.max_positions + 1, device=device, dtype=torch.int32)

        self._dflash_hidden_states = torch.zeros(
            (self.max_num_tokens, self.hidden_size), dtype=self.dtype, device=self.device
        )

        self.parallel_drafting_hidden_state_tensor = None

        dynamic_spec_config = get_ascend_config().dynamic_spec_config
        self.dynamic_spec = None

        if dynamic_spec_config.method == "dflash":
            self.dynamic_spec = DynamicSpecScheduler(
                method="dflash",
                method_params=dynamic_spec_config.method_params,
                max_batch_size=self.max_batch_size,
                num_speculative_tokens=self.num_speculative_tokens,
                device=device,
            )

    def initialize_attn_backend(self, kv_cache_config, kernel_block_sizes=None) -> None:
        if not self.mixed_dflash_attention:
            return super().initialize_attn_backend(kv_cache_config, kernel_block_sizes)
        if self.dcp_size != 1 or self.vllm_config.parallel_config.prefill_context_parallel_size != 1:
            raise ValueError("Mixed sliding/full DFlash on Ascend MRv1 requires DCP=PCP=1")
        if self.draft_window_size is not None:
            raise ValueError("Mixed sliding/full DFlash uses the checkpoint's per-layer attention windows")

        names = self.model.get_draft_kv_cache_layer_names()
        self.attn_layer_names = names
        self._dflash_layer_causal = dict(zip(names, self.model.get_draft_attn_causal(), strict=True))
        all_layers = get_layers_from_vllm_config(self.vllm_config, AttentionLayerBase, names)
        by_key = {}
        layer_gid = {}
        self._dflash_block_sizes = {}
        self._dflash_sliding_windows = {}
        for gid, group in enumerate(kv_cache_config.kv_cache_groups):
            for name in names:
                if name not in group.layer_names:
                    continue
                spec = group.kv_cache_spec
                if isinstance(spec, UniformTypeKVCacheSpecs):
                    spec = spec.kv_cache_specs[name]
                backend = all_layers[name].get_attn_backend()
                key = (gid, backend, spec, self._dflash_layer_causal[name])
                kernel_size = kernel_block_sizes[gid] if kernel_block_sizes is not None else spec.block_size
                layer_gid[name] = gid
                self._dflash_block_sizes[gid] = kernel_size
                if key not in by_key:
                    attn_group = AttentionGroup(backend, [], spec, gid)
                    attn_group.create_metadata_builders(self.vllm_config, self.device, kernel_block_size=kernel_size)
                    by_key[key] = attn_group
                by_key[key].layer_names.append(name)
                if isinstance(spec, SlidingWindowSpec) and gid not in self._dflash_sliding_windows:
                    self._dflash_sliding_windows[gid] = SlidingWindowAdapter(
                        spec.sliding_window + self.num_speculative_tokens,
                        kernel_size,
                        self.max_batch_size,
                        0,
                        self.device,
                    )
        if layer_gid.keys() != set(names):
            raise RuntimeError("Missing KV cache groups for DFlash layers")

        self.draft_attn_groups = list(by_key.values())
        self.kv_cache_gid = layer_gid[names[0]]
        self.kernel_block_size = self.block_size = self._dflash_block_sizes[self.kv_cache_gid]
        self._dflash_layer_gids = [layer_gid[name] for name in names]
        self._per_group_block_table_buffers = {}
        self._dflash_input_slot_mappings = {}
        self._per_group_query_slot_mapping_buffers = {
            gid: torch.full_like(self._slot_mapping_buffer, -1) for gid in self._dflash_block_sizes
        }
        self._per_group_context_slot_mapping_buffers = {
            gid: torch.full_like(self._context_slot_mapping_buffers, -1) for gid in self._dflash_block_sizes
        }

    def set_per_group_attn_metadata(self, gid, block_table, slot_mapping) -> None:
        if gid in self._dflash_block_sizes:
            self._per_group_block_table_buffers[gid] = block_table
            self._dflash_input_slot_mappings[gid] = slot_mapping

    def _dflash_group_metadata(self, common, group, num_input_tokens):
        gid = group.kv_cache_group_id
        view = copy(common)
        view.causal = self._dflash_layer_causal[group.layer_names[0]]
        view.block_table_tensor = self._adjust_tensor(self._per_group_block_table_buffers[gid], common.num_reqs)
        view.slot_mapping = self._per_group_query_slot_mapping_buffers[gid][:num_input_tokens]
        if adapter := self._dflash_sliding_windows.get(gid):
            adapter.apply(view)
        return view

    def build_draft_attn_metadata(self, common_attn_metadata, num_input_tokens, num_actual_tokens):
        if not self.mixed_dflash_attention:
            return super().build_draft_attn_metadata(common_attn_metadata, num_input_tokens, num_actual_tokens)
        per_layer = {}
        for group in self.draft_attn_groups:
            view = self._dflash_group_metadata(common_attn_metadata, group, num_input_tokens)
            metadata = group.get_metadata_builder().build(0, view)
            per_layer.update(dict.fromkeys(group.layer_names, metadata))
        return [per_layer], per_layer[self.attn_layer_names[0]]

    def set_inputs_first_pass(
        self,
        target_token_ids: torch.Tensor,
        next_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        cad: CommonAttentionMetadata,
        num_rejected_tokens_gpu: torch.Tensor | None,
        req_scheduled_tokens=None,
        long_seq_metadata=None,
        num_prefill_reqs=0,
        num_decode_reqs=0,
    ) -> tuple[int, torch.Tensor, CommonAttentionMetadata, tuple[Any, Any] | None]:
        # DFlash cross-attention: context K/V from target hidden states,
        # Q from query embeddings (bonus + mask tokens).
        batch_size = cad.num_reqs
        num_context = target_token_ids.shape[0]
        num_query_per_req = 1 + self.num_speculative_tokens
        num_query_total = batch_size * num_query_per_req

        self._dflash_num_context = num_context
        self._dflash_hidden_states[:num_context] = target_hidden_states

        token_indices_to_sample = torch.empty(
            batch_size * self.num_speculative_tokens,
            dtype=torch.int32,
            device=self.device,
        )

        has_num_rejected = num_rejected_tokens_gpu is not None

        if self.mixed_dflash_attention:
            group_inputs = [
                (
                    self._per_group_block_table_buffers[gid],
                    self._dflash_input_slot_mappings[gid],
                    self._per_group_context_slot_mapping_buffers[gid],
                    self._per_group_query_slot_mapping_buffers[gid],
                    block_size,
                )
                for gid, block_size in self._dflash_block_sizes.items()
            ]
            self._context_slot_mapping_buffers = [
                self._per_group_context_slot_mapping_buffers[gid] for gid in self._dflash_layer_gids
            ]
            query_slots = self._per_group_query_slot_mapping_buffers[self.kv_cache_gid]
        else:
            query_slots = self._slot_mapping_buffer
            group_inputs = [
                (
                    cad.block_table_tensor,
                    cad.slot_mapping,
                    self._context_slot_mapping_buffers,
                    query_slots,
                    self.kernel_block_size,
                )
            ]
        num_programs = _compute_num_programs(num_context, num_query_total)
        for block_table, context_slots, out_context_slots, out_query_slots, block_size in group_inputs:
            copy_and_expand_dflash_and_dspark_inputs_kernel[(num_programs,)](
                next_token_ids_ptr=next_token_ids,
                target_positions_ptr=target_positions,
                context_slot_mapping_ptr=context_slots,
                out_input_ids_ptr=self.input_ids,
                out_context_positions_ptr=self._context_positions_buffer,
                out_query_positions_ptr=self.positions,
                out_context_slot_mapping_ptr=out_context_slots,
                out_query_slot_mapping_ptr=out_query_slots,
                out_token_indices_ptr=token_indices_to_sample,
                block_table_ptr=block_table,
                block_table_stride=block_table.stride(0),
                query_start_loc_ptr=cad.query_start_loc,
                seq_lens_ptr=cad.seq_lens,
                num_rejected_tokens_ptr=(num_rejected_tokens_gpu if has_num_rejected else 0),
                parallel_drafting_token_id=self.parallel_drafting_token_id,
                block_size=block_size,
                num_query_per_req=num_query_per_req,
                num_speculative_tokens=self.num_speculative_tokens,
                total_input_tokens=num_context,
                batch_size=batch_size,
                HAS_NUM_REJECTED=has_num_rejected,
            )

        query_slot_mapping = query_slots[:num_query_total]
        new_query_start_loc = self.arange_dflash[: batch_size + 1] * num_query_per_req

        effective_seq_lens = cad.seq_lens
        if has_num_rejected:
            effective_seq_lens = effective_seq_lens - num_rejected_tokens_gpu

        cad.query_start_loc = new_query_start_loc
        cad.seq_lens = effective_seq_lens + num_query_per_req
        cad.query_start_loc_cpu = (
            torch.from_numpy(self.token_arange_np[: batch_size + 1]).clone() * num_query_per_req
        ).to(torch.int32)

        if hasattr(cad, "actual_seq_lengths_q"):
            cad.actual_seq_lengths_q = [num_query_per_req] * batch_size
        if hasattr(cad, "decode_token_per_req"):
            cad.decode_token_per_req = num_query_per_req

        cad.num_actual_tokens = num_query_total
        cad.max_query_len = num_query_per_req
        cad.max_seq_len = cad.max_seq_len + num_query_per_req
        cad.slot_mapping = query_slot_mapping
        cad.causal = False
        cad.attn_mask = None
        cad.attn_state = AscendAttentionState.ChunkedPrefill

        return num_query_total, token_indices_to_sample, cad, None

    def _clear_dummy_slot_mappings(self) -> None:
        self._slot_mapping_buffer.fill_(-1)
        if self.mixed_dflash_attention:
            for buffers in (
                self._per_group_query_slot_mapping_buffers,
                self._per_group_context_slot_mapping_buffers,
            ):
                for mapping in buffers.values():
                    mapping.fill_(-1)
            return
        self._context_slot_mapping_buffers.fill_(-1)

    @torch.inference_mode()
    def dummy_run(
        self,
        num_tokens: int,
        num_reqs: int = 0,
        num_tokens_across_dp: torch.Tensor | None = None,
        aclgraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        batch_descriptor=None,
        dummy_compute_logits=lambda hidden_states: None,
        is_profile=False,
        **kwargs,
    ) -> None:
        num_query_tokens = min(num_tokens, self.max_query_tokens)

        (
            num_input_tokens,
            num_tokens_across_dp,
            _,
        ) = self.runner._sync_metadata_across_dp(num_query_tokens, is_draft_model=True)

        if not self.use_cuda_graph:
            aclgraph_runtime_mode = CUDAGraphMode.NONE
        num_query_per_req = 1 + self.num_speculative_tokens
        num_query_total = num_reqs * num_query_per_req

        context_positions = self._context_positions_buffer[:num_input_tokens]
        context_states = self.hidden_states[:num_input_tokens]

        multi_steps_attn_metadata = []
        if aclgraph_runtime_mode == CUDAGraphMode.FULL and len(self.runner.attn_groups) > 0:
            common_attn_metadata = AscendCommonAttentionMetadata(
                query_start_loc=self.arange_dflash[: num_reqs + 1] * num_query_per_req,
                query_start_loc_cpu=torch.from_numpy(self.token_arange_np[: num_reqs + 1]).clone() * num_query_per_req,
                seq_lens_cpu=self.runner.optimistic_seq_lens_cpu,
                seq_lens_cpu_upper_bound=self.runner.optimistic_seq_lens_cpu,
                seq_lens=self.runner.seq_lens[:num_reqs],
                num_reqs=num_reqs,
                num_actual_tokens=num_query_tokens,
                max_query_len=num_query_per_req,
                max_seq_len=0,
                slot_mapping=self._slot_mapping_buffer[:num_query_total],
                attn_state=AscendAttentionState.ChunkedPrefill,
                causal=False,
                is_prefilling=torch.zeros(num_reqs, dtype=torch.bool),
                block_table_tensor=self.runner.input_batch.block_table[self.kv_cache_gid].get_device_tensor()[
                    :num_reqs
                ],
            )

            per_layer_attn_metadata = {}
            groups = self.draft_attn_groups if self.mixed_dflash_attention else self.draft_attn_groups[:1]
            for group in groups:
                view = common_attn_metadata
                names = self.attn_layer_names
                if self.mixed_dflash_attention:
                    gid = group.kv_cache_group_id
                    self._per_group_block_table_buffers[gid] = self.runner.input_batch.block_table[
                        gid
                    ].get_device_tensor()[:num_reqs]
                    view = self._dflash_group_metadata(common_attn_metadata, group, num_query_total)
                    names = group.layer_names
                metadata = group.get_metadata_builder().build_for_graph_capture(
                    view,
                    AscendAttentionState.ChunkedPrefill,
                )
                if not view.causal:
                    metadata.attn_mask = None
                metadata.attn_state = AscendAttentionState.ChunkedPrefill
                per_layer_attn_metadata.update(dict.fromkeys(names, metadata))
            multi_steps_attn_metadata.append(per_layer_attn_metadata)

        self.token_indices_to_sample.fill_(0)
        self._clear_dummy_slot_mappings()

        if aclgraph_runtime_mode == CUDAGraphMode.FULL:
            self._maybe_update_metadata(
                self.draft_attn_groups[0].backend,
                multi_steps_attn_metadata,
            )

        with set_ascend_forward_context(
            multi_steps_attn_metadata[0] if multi_steps_attn_metadata else None,
            self.vllm_config,
            num_tokens=num_input_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            num_actual_tokens=num_input_tokens,
            in_profile_run=is_profile,
            batch_descriptor=batch_descriptor,
            aclgraph_runtime_mode=aclgraph_runtime_mode,
            is_draft_model=True,
            draft_attn_metadatas=multi_steps_attn_metadata,
        ):
            if is_profile:
                self.model.precompute_and_store_context_kv(context_states, context_positions)
                self.model(
                    input_ids=self.input_ids[:num_input_tokens],
                    positions=self._get_positions(num_input_tokens),
                    inputs_embeds=None,
                )

            else:
                self._dflash_num_context = num_input_tokens
                self._runnable(
                    num_input_tokens=num_input_tokens,
                    batch_size=num_reqs,
                    token_indices_to_sample=self.token_indices_to_sample[: num_reqs * self.num_speculative_tokens],
                    target_positions=self._get_positions(num_input_tokens),
                    inputs_embeds=None,
                    multi_steps_attn_metadata=multi_steps_attn_metadata,
                    num_tokens=num_input_tokens,
                )

            forward_context = get_forward_context()
            if forward_context.cudagraph_runtime_mode == CUDAGraphMode.FULL and not _EXTRA_CTX.capturing:
                self._update_full_graph_params(forward_context, num_tokens, multi_steps_attn_metadata)

    def build_model_inputs_first_pass(
        self,
        num_input_tokens: int,
        _context_slots: torch.Tensor | list[torch.Tensor],
    ) -> None:
        num_context = self._dflash_num_context

        if _context_slots is None:
            _context_slots = None
        elif isinstance(_context_slots, list):
            _context_slots = [_one_context_slots[:num_context] for _one_context_slots in _context_slots]
        else:
            _context_slots = _context_slots[:num_context]

        self.model.precompute_and_store_context_kv(
            self._dflash_hidden_states[:num_context],
            self._context_positions_buffer[:num_context],
            _context_slots,
        )

    def _raise_if_multimodal(self):
        pass
