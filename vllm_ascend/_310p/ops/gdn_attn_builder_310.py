# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""310P RC GDN metadata builder.

Share request grouping and graph materialization, without building outputs
that are only consumed by the common GDN/KDA kernels.
"""

from __future__ import annotations

import torch
from vllm.v1.attention.backends.utils import PAD_SLOT_ID

from vllm_ascend._310p.ops.replayssm_sd import aligned_metadata_size, replay_token_gather
from vllm_ascend.ops.gdn_attn_builder import (
    AscendGDNAttentionBackend,
    AscendGDNAttentionMetadataBuilder,
)
from vllm_ascend.ops.replayssm.cache import REPLAYSSM_SD_MAX_HISTORY


class GDNAttentionMetadataBuilder310(AscendGDNAttentionMetadataBuilder):
    """310P overrides on top of :class:`AscendGDNAttentionMetadataBuilder`.

    310P consumes top-level non-spec fields and derives recurrent lengths in
    its kernel wrapper; it still uses the shared speculative conv metadata.
    """

    # The 310P conv kernel expects PAD_SLOT_ID, not a valid state block.
    _SPEC_GRAPH_PAD_SLOT_ID = PAD_SLOT_ID
    _USE_COMMON_KERNEL_METADATA = False

    def _allocate_replayssm_sd_rows(self, capacity, device):
        # Pad physical storage, while passing logical row counts to the ABI.
        rows = aligned_metadata_size(capacity, 1)
        self.sd_accepted = torch.zeros(rows, dtype=torch.int32, device=device)
        self.sd_reset = torch.zeros(rows, dtype=torch.int8, device=device)
        self.sd_token_gather = torch.empty(rows * (self.num_spec + 1), dtype=torch.int64, device=device)
        return torch.zeros(rows, dtype=torch.int32, device=device)

    def _allocate_replayssm_sd_cursor_state(self, device):
        slots = self.sd_ring_capacity + 1
        self.sd_write_pos = torch.zeros(aligned_metadata_size(slots, 4), dtype=torch.int32, device=device)[:slots]
        self.sd_cache_base = torch.zeros(aligned_metadata_size(slots, 4), dtype=torch.int32, device=device)[:slots]
        self.sd_is_flush = torch.zeros(aligned_metadata_size(slots, 1), dtype=torch.int8, device=device)[:slots]

    def _advance_replayssm_sd_cursor(self, rows, accepted, reset):
        ring_slots = self.sd_ring_slots[:rows]
        padding = ring_slots == 0
        # Native conv retains PAD_SLOT_ID. Only the Replay ABI uses slot 0.
        self.sd_slots[:rows].masked_fill_(padding, 0)
        self.sd_lengths[:rows].masked_fill_(padding, 0)
        self.sd_accepted[:rows].copy_(accepted[:rows])
        self.sd_reset[:rows].copy_(reset)
        torch.ops.replayssm_sd_310p.cursor_(
            self.sd_write_pos,
            self.sd_cache_base,
            self.sd_is_flush,
            self.sd_accepted[:rows],
            ring_slots,
            self.sd_reset[:rows],
            self.sd_cursor_status[:rows],
            REPLAYSSM_SD_MAX_HISTORY,
            self.num_spec + 1,
            self.sd_ring_len,
        )

    def _stage_replayssm_sd_pack(self, pack, rows, attn_metadata):
        super()._stage_replayssm_sd_pack(pack, rows, attn_metadata)
        attn_metadata.replayssm_sd_reset = self.sd_reset[:rows]
        if pack.numel() == rows * (self.num_spec + 1):
            return
        # Eager mixed and short decode windows use gather, avoiding 310P
        # index_copy_ / IndexPutV2 on the packed GDN tensors.
        gather = replay_token_gather(pack, rows)
        self.sd_token_gather[: gather.numel()].copy_(gather, non_blocking=True)
        attn_metadata.replayssm_sd_token_gather = self.sd_token_gather[: gather.numel()]

    def build(self, common_prefix_len, common_attn_metadata, *args, **kwargs):
        m = common_attn_metadata
        if self.use_replayssm_sd and not m.offload_dummy:
            # After preemption vLLM may recompute generated context as well
            # as the prompt. That context has to rebuild the checkpoint.
            decode_base = m.replayssm_decode_base_cpu
            seq_lens = m.seq_lens_cpu_upper_bound
            if decode_base is not None and seq_lens is not None and m.is_prefilling is not None:
                computed = seq_lens - torch.diff(m.query_start_loc_cpu)
                m = m.replace(is_prefilling=m.is_prefilling | (computed < decode_base))
        return super().build(common_prefix_len, m, *args, **kwargs)

    def _can_pad_spec_decode(self, graph_request_count: int, num_spec_decode_tokens: int) -> bool:
        # MTP token buffers can hold (1 + K) tokens per request. Do not apply
        # the common single-token limit to 310P concurrent spec replay.
        return (
            graph_request_count <= self.decode_cudagraph_max_bs
            and num_spec_decode_tokens <= self.spec_token_indx.numel()
        )


# Keep the name introduced by the 310P ACL graph padding patch so existing
# imports and tests from that patch continue to work after rebasing onto
# upstream/main, whose class name is GDNAttentionMetadataBuilder310.
AscendGDNAttentionMetadataBuilder310 = GDNAttentionMetadataBuilder310


class AscendGDNAttentionBackend310(AscendGDNAttentionBackend):
    @staticmethod
    def get_builder_cls() -> type[AscendGDNAttentionMetadataBuilder310]:
        return AscendGDNAttentionMetadataBuilder310
