# SPDX-License-Identifier: Apache-2.0
import unittest
from pathlib import Path
from runpy import run_path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

ROOT = Path(__file__).resolve().parents[3] / "vllm_ascend/_310p"


class Prefill310DispatchTest(unittest.TestCase):
    def setUp(self):
        self.envs = SimpleNamespace(
            VLLM_ASCEND_310P_PREFILL_SELECTED_SOFTMAX=True,
            VLLM_ASCEND_310P_PREFILL_LEGACY_GDN_HO=True,
            VLLM_ASCEND_310P_PREFILL_RMSNORM_PAD16=True,
        )
        self.npu = SimpleNamespace(
            npu_rms_norm=Mock(
                side_effect=lambda x, w, eps: (
                    (torch.nn.functional.rms_norm(x.float(), (x.shape[-1],), w.float(), eps)).to(x.dtype),
                    None,
                )
            ),
            npu_moe_gating_top_k_softmax=Mock(
                side_effect=lambda logits, k: (*logits.float().softmax(-1).topk(k), None)
            ),
        )
        self.context = SimpleNamespace(attn_metadata={"layer": SimpleNamespace(num_prefills=1)}, is_draft_model=False)

        class RouterBase:
            def _renormalize_topk_weights(self, weights):
                return weights / weights.sum(dim=-1, keepdim=True) if self.renormalize else weights

        with patch.dict(
            "sys.modules",
            {
                "torch_npu": self.npu,
                "vllm_ascend": SimpleNamespace(envs=self.envs),
                "vllm.forward_context": SimpleNamespace(
                    get_forward_context=lambda: self.context, is_forward_context_available=lambda: True
                ),
                "vllm.third_party.flash_linear_attention.ops.utils": SimpleNamespace(tensor_cache=lambda fn: fn),
                "vllm_ascend.ops.fused_moe.router.grouped_topk_router": SimpleNamespace(
                    AscendGroupedTopKRouter=RouterBase
                ),
                "vllm_ascend._310p.ops.compute_wy": SimpleNamespace(compute_wy_for_prefill=Mock()),
            },
        ):
            self.l2norm = run_path(str(ROOT / "ops/fla/l2norm.py"))["l2norm_310p"]
            with patch.dict(
                "sys.modules", {"vllm_ascend._310p.ops.fla.l2norm": SimpleNamespace(l2norm_310p=self.l2norm)}
            ):
                self.chunk = run_path(str(ROOT / "ops/fla/chunk_gated_delta_rule.py"))
            self.router = run_path(str(ROOT / "fused_moe/grouped_topk_router.py"))["AscendGroupedTopKRouter310"]()
        self.router.scoring_func = "softmax"
        self.router.use_grouped_topk = False
        self.router.e_score_correction_bias = None
        self.router.renormalize = True
        self.router.routed_scaling_factor = 2.0
        self.router.top_k = 8

    def test_selected_softmax_matches_full_softmax_then_renormalize(self):
        logits = torch.arange(256, dtype=torch.float16).div_(32).expand(1025, 256).contiguous()
        weights, ids = self.router._compute_routing(logits, logits, None)
        expected, expected_ids = logits.float().softmax(-1).topk(8)
        expected = (expected / expected.sum(-1, keepdim=True)).half() * 2
        torch.testing.assert_close(ids, expected_ids.int(), atol=0, rtol=0)
        torch.testing.assert_close(weights, expected)
        self.npu.npu_moe_gating_top_k_softmax.assert_not_called()

    def test_selected_softmax_keeps_native_for_decode_draft_and_unrenormalized(self):
        logits = torch.ones((32, 256), dtype=torch.float16)
        for prefills, draft, renormalize in ((0, False, True), (1, True, True), (1, False, False)):
            with self.subTest(prefills=prefills, draft=draft, renormalize=renormalize):
                self.context.attn_metadata["layer"].num_prefills = prefills
                self.context.is_draft_model = draft
                self.router.renormalize = renormalize
                self.npu.npu_moe_gating_top_k_softmax.reset_mock()
                self.router._compute_routing(logits, logits, None)
                self.npu.npu_moe_gating_top_k_softmax.assert_called_once()

    def test_chunk_qk_rmsnorm_pads_and_crops_only_when_needed(self):
        x = torch.randn((1, 3, 8, 128), dtype=torch.float16)
        actual = self.chunk["_maybe_l2norm"](x, True)
        padded_input = self.npu.npu_rms_norm.call_args.args[0]
        self.assertEqual(padded_input.shape, (32, 128))
        self.assertEqual(torch.count_nonzero(padded_input[24:]).item(), 0)
        torch.testing.assert_close(actual, self.l2norm(x), atol=0, rtol=0)
        self.assertEqual(self.npu.npu_rms_norm.call_args.args[0].shape, (24, 128))

        self.chunk["_maybe_l2norm"](x[:, :2], True)
        self.assertEqual(self.npu.npu_rms_norm.call_args.args[0].shape, (16, 128))

    def test_legacy_ho_selects_pair_only_for_supported_checkpoint_contract(self):
        legacy_h, legacy_o, v30_h, v30_o = (object() for _ in range(4))
        ops = SimpleNamespace(
            chunk_gated_delta_rule_fwd_h=legacy_h,
            chunk_fwd_o=legacy_o,
            chunk_gated_delta_rule_fwd_h_vllm=v30_h,
            chunk_fwd_o_vllm=v30_o,
        )
        select = self.chunk["_select_chunk_ho_ops"]
        fp16 = SimpleNamespace(dtype=torch.float16)
        self.assertEqual(select(ops, 128, 128, fp16, [0, 64, 128]), (legacy_h, legacy_o))
        self.assertEqual(select(ops, 128, 128, fp16, [0, 0, 64]), (v30_h, v30_o))
        self.assertEqual(select(ops, 128, 256, fp16, None), (v30_h, v30_o))
        self.assertEqual(select(ops, 128, 128, SimpleNamespace(dtype=torch.float32), None), (v30_h, v30_o))
        with self.assertRaisesRegex(RuntimeError, "Rebuild"):
            select(SimpleNamespace(chunk_gated_delta_rule_fwd_h=legacy_h), 128, 128, fp16, None)
        self.envs.VLLM_ASCEND_310P_PREFILL_LEGACY_GDN_HO = False
        self.assertEqual(select(ops, 128, 128, fp16, None), (v30_h, v30_o))
