# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import ast
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
    def test_sd_mtp3_ring_and_packed_token_mapping(self) -> None:
        self.assertEqual(CACHE.replayssm_sd_ring_len(16, 3), 32)
        lengths = torch.tensor([1, 4, 2], dtype=torch.int32)
        indices = CACHE.replayssm_sd_pack_indices(lengths)
        torch.testing.assert_close(indices, torch.tensor([0, 4, 5, 6, 7, 8, 9]))
        source = torch.arange(indices.numel(), dtype=torch.float32)
        padded = torch.zeros(12, dtype=torch.float32)
        padded.index_copy_(0, indices, source)
        torch.testing.assert_close(padded.index_select(0, indices), source)
        torch.testing.assert_close(padded[[1, 2, 3, 10, 11]], torch.zeros(5))

    def test_sd_mtp1_mtp2_use_four_token_verify_rows(self) -> None:
        for num_spec, lengths, expected in (
            (1, [1, 2, 1], [0, 4, 5, 8]),
            (2, [1, 3, 2], [0, 4, 5, 6, 8, 9]),
        ):
            with self.subTest(num_spec=num_spec):
                self.assertEqual(CACHE.replayssm_sd_ring_len(16, num_spec), 32)
                indices = CACHE.replayssm_sd_pack_indices(
                    torch.tensor(lengths, dtype=torch.int32), num_spec + 1, CACHE.REPLAYSSM_SD_MAX_TOKENS
                )
                torch.testing.assert_close(indices, torch.tensor(expected))
                padded = torch.zeros(3 * CACHE.REPLAYSSM_SD_MAX_TOKENS)
                padded.index_copy_(0, indices, torch.arange(len(expected), dtype=padded.dtype))
                torch.testing.assert_close(
                    padded.index_select(0, indices), torch.arange(len(expected), dtype=padded.dtype)
                )

    def test_sd_rejects_unsupported_window_before_cache_allocation(self) -> None:
        with self.assertRaisesRegex(ValueError, "buffer_len=16"):
            CACHE.replayssm_sd_ring_len(32, 3)
        for num_spec in (0, 4):
            with self.assertRaisesRegex(ValueError, "MTP1/2/3"):
                CACHE.replayssm_sd_ring_len(16, num_spec)
        with self.assertRaisesRegex(ValueError, "query length"):
            CACHE.replayssm_sd_pack_indices(torch.tensor([1, 5]))
        with self.assertRaisesRegex(ValueError, "query length"):
            CACHE.replayssm_sd_pack_indices(torch.tensor([3]), 2, 4)

    def test_sd_opt_in_reuses_decode_base_without_ar_spec_guard(self) -> None:
        config = SimpleNamespace(
            additional_config={"replayssm_sd": True},
            cache_config=SimpleNamespace(use_replayssm=False, mamba_cache_mode="none"),
            speculative_config=SimpleNamespace(num_speculative_tokens=3),
        )
        self.assertTrue(CACHE.replayssm_sd_enabled(config))
        self.assertTrue(CACHE.replayssm_gdn_enabled(config))
        self.assertFalse(config.cache_config.use_replayssm)
        CACHE.reject_replayssm_ar_speculative_decode(config)
        config.additional_config["replayssm_sd"] = "false"
        self.assertFalse(CACHE.replayssm_sd_enabled(config))
        self.assertFalse(CACHE.replayssm_gdn_enabled(config))
        with self.assertRaisesRegex(ValueError, "boolean"):
            config.additional_config["replayssm_sd"] = "enabled"
            CACHE.replayssm_sd_enabled(config)

    def test_request_ring_slots_survive_reorder_and_reuse_after_release(self) -> None:
        slots = CACHE.ReplaySSMSDRingSlots(2)
        slots.assign(["first", "second"])
        self.assertEqual(slots.slots_for(["second", "first"]), [2, 1])
        slots.release(["first"])
        slots.assign(["third"])
        self.assertEqual(slots.slots_for(["second", "third"]), [2, 1])
        with self.assertRaisesRegex(RuntimeError, "ring slots exhausted"):
            slots.assign(["fourth"])

    def test_sd_ring_capacity_defaults_to_max_num_seqs(self) -> None:
        config = SimpleNamespace(scheduler_config=SimpleNamespace(max_num_seqs=64), additional_config={})
        self.assertEqual(CACHE.replayssm_sd_ring_capacity(config), 64)
        config.additional_config["replayssm_sd_ring_capacity"] = None
        self.assertEqual(CACHE.replayssm_sd_ring_capacity(config), 64)
        config.additional_config["replayssm_sd_ring_capacity"] = 96
        self.assertEqual(CACHE.replayssm_sd_ring_capacity(config), 96)
        config.additional_config["replayssm_sd_ring_capacity"] = 32
        with self.assertRaisesRegex(ValueError, ">= max_num_seqs"):
            CACHE.replayssm_sd_ring_capacity(config)

    def test_sd_opt_in_does_not_set_upstream_kda_recover_flag(self) -> None:
        # The vLLM mode="after" validator runs after the Ascend platform hook.
        # Writing this flag there turns Qwen MTP into KDA RecoverSSM.
        tree = ast.parse((ROOT / "vllm_ascend/platform.py").read_text())
        assignments = (node for node in ast.walk(tree) if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)))
        for node in assignments:
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                self.assertFalse(
                    isinstance(target, ast.Attribute) and target.attr == "use_replayssm",
                    "the Ascend platform hook must leave the upstream RecoverSSM flag unset for SD",
                )

    def test_sd_serve_verify_selects_v13_tile128(self) -> None:
        self.assertEqual((CACHE.REPLAYSSM_SD_VERIFY_VARIANT, CACHE.REPLAYSSM_SD_VERIFY_TILE_V), (13, 128))
        try:
            from vllm_ascend.ops.gdn import AscendGatedDeltaNetAttention
        except ModuleNotFoundError as exc:
            self.skipTest(f"optional local vLLM dependencies are unavailable: {exc}")

        batch, key_heads, value_heads, dim = 2, 1, 2, 128
        layer = SimpleNamespace(
            num_spec=3,
            num_k_heads=key_heads,
            num_v_heads=value_heads,
            tp_size=1,
            head_k_dim=dim,
            head_v_dim=dim,
            kv_cache=(None, *(torch.empty(1) for _ in range(4))),
            _replayssm_sd_d_ring=torch.empty(1),
            _replayssm_sd_k_ring=torch.empty(1),
            _replayssm_sd_g_ring=torch.empty(1),
            _replayssm_sd_verify_status=torch.empty(batch * value_heads, dtype=torch.int32),
            _replayssm_sd_compact_status=torch.empty(batch * value_heads * 4, dtype=torch.int32),
        )
        metadata = SimpleNamespace(
            replayssm_sd_slots=torch.tensor([1, 2], dtype=torch.int32),
            replayssm_sd_block_slots=torch.tensor([1, 2], dtype=torch.int32),
            replayssm_sd_ring_slots=torch.tensor([2, 1], dtype=torch.int32),
            replayssm_sd_positions=torch.zeros(batch, dtype=torch.int32),
            replayssm_sd_bases=torch.zeros(batch, dtype=torch.int32),
            replayssm_sd_flush=torch.zeros(batch, dtype=torch.int32),
            replayssm_sd_lengths=torch.tensor([1, 4], dtype=torch.int32),
            replayssm_sd_pack_indices=torch.tensor([0, 4, 5, 6, 7]),
        )
        query = torch.zeros(1, 5, key_heads, dim)
        value = torch.arange(5 * value_heads * dim).view(1, 5, value_heads, dim).float()
        gate = torch.zeros(1, 5, value_heads)

        def fake_verify(*args):
            args[13].copy_(args[2])

        with (
            patch("vllm_ascend.ops.gdn.replayssm_sd_verify", side_effect=fake_verify) as verify,
            patch("vllm_ascend.ops.gdn.replayssm_sd_compact") as compact,
        ):
            result = AscendGatedDeltaNetAttention._replayssm_sd_decode(
                layer, query=query, key=query, value=value, g=gate, beta=gate, attn_metadata=metadata
            )
        torch.testing.assert_close(result, value)
        self.assertEqual(verify.call_args.args[-4:-1], (128, 40, 13))
        self.assertEqual(compact.call_args.args[-4:-1], (32, 40, 1))
        torch.testing.assert_close(verify.call_args.args[-1], metadata.replayssm_sd_ring_slots)
        torch.testing.assert_close(compact.call_args.args[-1], metadata.replayssm_sd_ring_slots)

    def test_sd_decode_pads_mtp1_mtp2_without_changing_output_order(self) -> None:
        try:
            from vllm_ascend.ops.gdn import AscendGatedDeltaNetAttention
        except ModuleNotFoundError as exc:
            self.skipTest(f"optional local vLLM dependencies are unavailable: {exc}")

        for num_spec in (1, 2):
            with self.subTest(num_spec=num_spec):
                lengths = torch.tensor([1, num_spec + 1], dtype=torch.int32)
                pack = CACHE.replayssm_sd_pack_indices(lengths, num_spec + 1, CACHE.REPLAYSSM_SD_MAX_TOKENS)
                tokens = int(lengths.sum())
                layer = SimpleNamespace(
                    num_spec=num_spec,
                    num_k_heads=1,
                    num_v_heads=2,
                    tp_size=1,
                    head_k_dim=128,
                    head_v_dim=128,
                    kv_cache=(None, torch.empty(1)),
                    _replayssm_sd_d_ring=torch.empty(1),
                    _replayssm_sd_k_ring=torch.empty(1),
                    _replayssm_sd_g_ring=torch.empty(1),
                    _replayssm_sd_verify_status=torch.empty(4, dtype=torch.int32),
                    _replayssm_sd_compact_status=torch.empty(16, dtype=torch.int32),
                )
                metadata = SimpleNamespace(
                    replayssm_sd_block_slots=torch.tensor([1, 2], dtype=torch.int32),
                    replayssm_sd_ring_slots=torch.tensor([2, 1], dtype=torch.int32),
                    replayssm_sd_positions=torch.zeros(2, dtype=torch.int32),
                    replayssm_sd_bases=torch.zeros(2, dtype=torch.int32),
                    replayssm_sd_flush=torch.zeros(2, dtype=torch.int32),
                    replayssm_sd_lengths=lengths,
                    replayssm_sd_pack_indices=pack,
                )
                query = torch.zeros(1, tokens, 1, 128)
                value = torch.arange(tokens * 2 * 128).reshape(1, tokens, 2, 128).float()
                gate = torch.zeros(1, tokens, 2)

                def fake_verify(*args):
                    args[13].copy_(args[2])

                with (
                    patch("vllm_ascend.ops.gdn.replayssm_sd_verify", side_effect=fake_verify) as verify,
                    patch("vllm_ascend.ops.gdn.replayssm_sd_compact"),
                ):
                    result = AscendGatedDeltaNetAttention._replayssm_sd_decode(
                        layer, query=query, key=query, value=value, g=gate, beta=gate, attn_metadata=metadata
                    )
                self.assertEqual(tuple(verify.call_args.args[0].shape), (2, 4, 1, 128))
                torch.testing.assert_close(result, value)

    def test_all_npu_input_batch_owners_keep_replayssm_lifecycle(self) -> None:
        input_batch_source = (ROOT / "vllm_ascend/worker/npu_input_batch.py").read_text()
        self.assertIn("self.replayssm_decode_base_cpu_tensor = torch.zeros(", input_batch_source)
        self.assertIn(
            "self.replayssm_decode_base = self.replayssm_decode_base_cpu_tensor.numpy()",
            input_batch_source,
        )

        # Ascend replaces upstream InputBatch construction both before and
        # after final KV-cache geometry is known. The 310P subclass is audited
        # as well so every NPU-owned constructor preserves the upstream flag.
        callsites = (
            ROOT / "vllm_ascend/worker/model_runner_v1.py",
            ROOT / "vllm_ascend/_310p/model_runner_310p.py",
        )
        seen = 0
        for path in callsites:
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                if not isinstance(func, ast.Name) or func.id != "NPUInputBatch":
                    continue
                seen += 1
                keywords = {keyword.arg for keyword in node.keywords}
                self.assertIn("use_replayssm", keywords, f"missing at {path}:{node.lineno}")
        self.assertEqual(seen, 4)

    def test_ascend_overrides_preserve_sd_cache_geometry_and_decode_base(self) -> None:
        worker = (ROOT / "vllm_ascend/worker/worker.py").read_text()
        metadata = (ROOT / "vllm_ascend/attention/utils.py").read_text()
        self.assertIn("self.cache_config.num_gpu_blocks = kv_cache_config.num_blocks", worker)
        self.assertIn("replayssm_decode_base_cpu=_slice_reqs(self.replayssm_decode_base_cpu)", metadata)

    def test_decode_base_uses_live_buffer_and_zeroes_capture_dummy(self) -> None:
        live = torch.tensor([17, 33, 0], dtype=torch.int32)
        actual = CACHE.prepare_replayssm_decode_base_cpu(live, is_dummy_run=False)
        self.assertIs(actual, live)

        dummy = CACHE.prepare_replayssm_decode_base_cpu(live, is_dummy_run=True)
        self.assertIsNot(dummy, live)
        torch.testing.assert_close(dummy, torch.zeros_like(live))
        torch.testing.assert_close(live, torch.tensor([17, 33, 0], dtype=torch.int32))

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
