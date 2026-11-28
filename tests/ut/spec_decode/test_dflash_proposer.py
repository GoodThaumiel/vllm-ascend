#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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
# This file is a part of the vllm-ascend project.
#
"""Unit tests for the DFlash speculative-decoding proposer."""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
from vllm.config import CUDAGraphMode

from vllm_ascend.spec_decode.dflash_proposer import AscendDflashProposer


class TestDummySlotMappingCleanup:
    def test_dummy_run_clears_single_group_buffers_before_forward(self, monkeypatch):
        proposer = AscendDflashProposer.__new__(AscendDflashProposer)
        proposer.mixed_dflash_attention = False
        proposer._slot_mapping_buffer = torch.full((8,), 123, dtype=torch.int32)
        proposer._context_slot_mapping_buffers = torch.full((8,), 456, dtype=torch.int32)

        @contextmanager
        def forward_context(*args, **kwargs):
            yield

        def runnable(**kwargs):
            assert torch.all(proposer._slot_mapping_buffer == -1)
            assert torch.all(proposer._context_slot_mapping_buffers == -1)

        monkeypatch.setattr("vllm_ascend.spec_decode.dflash_proposer.set_ascend_forward_context", forward_context)
        monkeypatch.setattr(
            "vllm_ascend.spec_decode.dflash_proposer.get_forward_context",
            lambda: SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode.NONE),
        )
        proposer.max_query_tokens = 8
        proposer.runner = SimpleNamespace(
            _sync_metadata_across_dp=lambda num_tokens, **kwargs: (num_tokens, None, None),
        )
        proposer.use_cuda_graph = False
        proposer.num_speculative_tokens = 4
        proposer._context_positions_buffer = torch.zeros(8, dtype=torch.int32)
        proposer.hidden_states = torch.zeros((8, 8), dtype=torch.float32)
        proposer.token_indices_to_sample = torch.zeros(4, dtype=torch.int32)
        proposer.vllm_config = SimpleNamespace()
        proposer._get_positions = lambda num_tokens: torch.zeros(num_tokens, dtype=torch.int32)
        proposer._runnable = MagicMock(side_effect=runnable)

        proposer.dummy_run(num_tokens=5, num_reqs=1, is_profile=False)

        assert torch.all(proposer._slot_mapping_buffer == -1)
        assert torch.all(proposer._context_slot_mapping_buffers == -1)
        proposer._runnable.assert_called_once()


def test_mixed_dflash_metadata_keeps_cache_groups_and_causality_separate():
    proposer = AscendDflashProposer.__new__(AscendDflashProposer)
    proposer.mixed_dflash_attention = True
    proposer.attn_layer_names = ["sliding", "full"]
    proposer._dflash_layer_causal = {"sliding": True, "full": False}
    proposer._dflash_sliding_windows = {}
    proposer._per_group_block_table_buffers = {1: torch.tensor([[10]]), 2: torch.tensor([[20]])}
    proposer._per_group_query_slot_mapping_buffers = {1: torch.tensor([101, 102]), 2: torch.tensor([201, 202])}
    proposer.draft_attn_groups = [
        SimpleNamespace(
            kv_cache_group_id=gid,
            layer_names=[name],
            get_metadata_builder=lambda: SimpleNamespace(build=lambda _, view: view),
        )
        for gid, name in enumerate(proposer.attn_layer_names, start=1)
    ]
    common = SimpleNamespace(num_reqs=1, causal=False, block_table_tensor=None, slot_mapping=None)
    (metadata,), primary = proposer.build_draft_attn_metadata(common, 2, 2)
    assert primary is metadata["sliding"]
    assert metadata["sliding"].causal
    assert not metadata["full"].causal
    for gid, name in enumerate(proposer.attn_layer_names, start=1):
        torch.testing.assert_close(metadata[name].block_table_tensor, proposer._per_group_block_table_buffers[gid])
        torch.testing.assert_close(metadata[name].slot_mapping, proposer._per_group_query_slot_mapping_buffers[gid])
    assert common.block_table_tensor is None
    assert common.slot_mapping is None
