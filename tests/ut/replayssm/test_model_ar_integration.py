# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "replayssm_cache_contract",
    ROOT / "vllm_ascend/ops/replayssm/cache.py",
)
assert SPEC is not None and SPEC.loader is not None
CACHE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CACHE)


class ReplaySSMModelARIntegrationTests(unittest.TestCase):
    def test_qwen_tp4_tp8_cache_contract_uses_bf16_checkpoint_and_fp16_ring(self) -> None:
        base_dtypes = (torch.bfloat16, torch.bfloat16)
        self.assertEqual(
            CACHE.append_gdn_replayssm_ar_dtypes(base_dtypes, torch.bfloat16),
            (
                torch.bfloat16,
                torch.bfloat16,
                torch.float16,
                torch.float16,
                torch.float32,
            ),
        )
        for tp_size, local_k_heads, local_v_heads in ((4, 4, 12), (8, 2, 6)):
            base_shapes = ((6144, 3), (local_v_heads, 128, 128))
            self.assertEqual(
                CACHE.append_gdn_replayssm_ar_shapes(
                    base_shapes,
                    num_k_heads=16,
                    tp_size=tp_size,
                    buffer_len=16,
                ),
                (
                    *base_shapes,
                    (local_v_heads, 16, 128),
                    (local_k_heads, 16, 128),
                    (local_v_heads, 16),
                ),
            )

    def test_cache_contract_rejects_non_bf16_baseline_and_unsupported_shapes(self) -> None:
        with self.assertRaisesRegex(ValueError, "BF16 model"):
            CACHE.append_gdn_replayssm_ar_dtypes((torch.float16, torch.bfloat16), torch.float16)
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            CACHE.append_gdn_replayssm_ar_dtypes((torch.bfloat16, torch.float32), torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "head dimension 128"):
            CACHE.append_gdn_replayssm_ar_shapes(
                ((256, 3), (2, 64, 128)),
                num_k_heads=8,
                tp_size=4,
                buffer_len=16,
            )

    def test_ar_cursor_flushes_window_and_restarts_from_checkpoint(self) -> None:
        positions, flush = CACHE.derive_replayssm_ar_cursors(
            num_computed_tokens=torch.tensor([100, 115, 116]),
            decode_base=torch.tensor([100, 100, 100]),
            query_lens=torch.ones(3, dtype=torch.int64),
            buffer_len=16,
            block_size=128,
            align_mode=False,
        )
        torch.testing.assert_close(positions, torch.tensor([0, 15, 0], dtype=torch.int32))
        torch.testing.assert_close(flush, torch.tensor([0, 1, 0], dtype=torch.int32))

    def test_ar_cursor_reanchors_on_resume_and_flushes_align_boundary(self) -> None:
        positions, flush = CACHE.derive_replayssm_ar_cursors(
            num_computed_tokens=torch.tensor([300, 383]),
            decode_base=torch.tensor([300, 300]),
            query_lens=torch.ones(2, dtype=torch.int64),
            buffer_len=16,
            block_size=128,
            align_mode=True,
        )
        # Resumed row starts with an empty ring; the second row materializes the
        # state at the end of its aligned cache block even though pos != L-1.
        torch.testing.assert_close(positions, torch.tensor([0, 3], dtype=torch.int32))
        torch.testing.assert_close(flush, torch.tensor([0, 1], dtype=torch.int32))

    def test_ar_cursor_materializes_single_token_extend_and_ignores_padding(self) -> None:
        positions, flush = CACHE.derive_replayssm_ar_cursors(
            num_computed_tokens=torch.tensor([127, 0]),
            decode_base=torch.tensor([128, 0]),
            query_lens=torch.tensor([1, 0]),
            buffer_len=16,
            block_size=128,
            align_mode=False,
        )
        # A captured one-token extend can precede its decode-base update. It
        # rebuilds from the checkpoint at position zero and flushes immediately.
        # A padded graph row is a no-op and must never request a flush.
        torch.testing.assert_close(positions, torch.tensor([0, 0], dtype=torch.int32))
        torch.testing.assert_close(flush, torch.tensor([1, 0], dtype=torch.int32))

    def test_ar_mode_rejects_speculation_until_sd_lifecycle_is_connected(self) -> None:
        config = SimpleNamespace(
            cache_config=SimpleNamespace(use_replayssm=True, mamba_cache_mode="none"),
            speculative_config=SimpleNamespace(num_speculative_tokens=3),
        )
        with self.assertRaisesRegex(ValueError, "ReplaySSM SD"):
            CACHE.reject_replayssm_ar_speculative_decode(config)

    def test_ar_mode_rejects_all_cache_lifecycle(self) -> None:
        config = SimpleNamespace(
            cache_config=SimpleNamespace(use_replayssm=True, mamba_cache_mode="all"),
            speculative_config=None,
        )
        with self.assertRaisesRegex(ValueError, "moves the checkpoint every token"):
            CACHE.reject_replayssm_ar_speculative_decode(config)

    def test_gdn_dispatch_passes_graph_metadata_and_all_five_cache_tensors(self) -> None:
        try:
            from vllm_ascend.ops.gdn import AscendGatedDeltaNetAttention
        except ModuleNotFoundError as exc:
            self.skipTest(f"optional local vLLM dependencies are unavailable: {exc}")

        batch, key_heads, value_heads, dim, capacity, slots = 2, 1, 3, 128, 16, 5
        layer = SimpleNamespace(
            num_v_heads=value_heads,
            tp_size=1,
            head_v_dim=dim,
            head_k_dim=dim,
            _replayssm_ar_status=torch.empty(batch * value_heads * 4, dtype=torch.int32),
            kv_cache=(
                torch.empty(slots, 1, 1),
                torch.empty(slots, value_heads, dim, dim, dtype=torch.bfloat16),
                torch.empty(slots, value_heads, capacity, dim, dtype=torch.float16),
                torch.empty(slots, key_heads, capacity, dim, dtype=torch.float16),
                torch.empty(slots, value_heads, capacity, dtype=torch.float32),
            ),
        )
        query = torch.empty(1, batch, key_heads, dim, dtype=torch.bfloat16)
        key = torch.empty_like(query)
        value = torch.empty(1, batch, value_heads, dim, dtype=torch.bfloat16)
        g = torch.empty(1, batch, value_heads, dtype=torch.float32)
        beta = torch.empty_like(g)
        state_indices = torch.tensor([1, 2], dtype=torch.int32)
        write_pos = torch.tensor([0, 15], dtype=torch.int32)
        is_flush = torch.tensor([0, 1], dtype=torch.int32)
        output = torch.empty(batch, value_heads, dim, dtype=torch.bfloat16)

        with patch("vllm_ascend.ops.gdn.replayssm_ar_step") as replay:
            result = AscendGatedDeltaNetAttention._replayssm_ar_decode(
                layer,
                query=query,
                key=key,
                value=value,
                g=g,
                beta=beta,
                state_indices=state_indices,
                write_pos=write_pos,
                is_flush=is_flush,
                output=output,
            )

        self.assertEqual(result.shape, (1, batch, value_heads, dim))
        args = replay.call_args.args
        self.assertTrue(all(actual is expected for actual, expected in zip(args[5:9], layer.kv_cache[1:5])))
        # Slicing/contiguous may return a distinct Tensor object depending on
        # the backend. The operator contract is its metadata and values, not
        # Python object identity.
        for actual, expected in zip(args[9:12], (state_indices, write_pos, is_flush)):
            torch.testing.assert_close(actual.cpu(), expected.cpu())
            self.assertEqual(actual.dtype, torch.int32)
            self.assertEqual(actual.device, expected.device)
            self.assertTrue(actual.is_contiguous())
        self.assertIs(args[12], output)
        self.assertEqual(args[13].shape, (batch * value_heads * 4,))
        self.assertEqual(args[13].dtype, torch.int32)
        self.assertEqual(args[13].device, output.device)


if __name__ == "__main__":
    unittest.main()
