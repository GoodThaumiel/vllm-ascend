# SPDX-License-Identifier: Apache-2.0
import unittest
from pathlib import Path
from runpy import run_path

import torch

MODULE = Path(__file__).resolve().parents[3] / "vllm_ascend/_310p/attention/dflash_splitfuse.py"
_components = run_path(str(MODULE))
draft_additive_mask = _components["draft_additive_mask"]
draft_kv_group_order = _components["draft_kv_group_order"]


class DFlash310PKVGroupTest(unittest.TestCase):
    def test_layer_order_and_physical_group_identity(self):
        # Layers 0 and 2 share the SWA group, with the full layer between them.
        self.assertEqual(
            draft_kv_group_order(
                ["draft.0", "draft.1", "draft.2"],
                [["target.0", "draft.1"], ["target.gdn"], ["draft.0", "draft.2"]],
            ),
            ((0, 2), (1, 0, 1)),
        )

    def test_missing_or_ambiguous_layer_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "do not cover"):
            draft_kv_group_order(["draft.0"], [["target.0"]])
        with self.assertRaisesRegex(ValueError, "multiple KV groups"):
            draft_kv_group_order(["draft.0"], [["draft.0"], ["draft.0"]])


class DFlash310PMaskTest(unittest.TestCase):
    def mask(self, lengths, causal, window=None):
        return draft_additive_mask(
            torch.tensor(lengths, dtype=torch.int32),
            torch.arange(8).view(1, 1, -1),
            torch.arange(3).view(1, -1, 1),
            causal=causal,
            window=window,
        )

    def check_visibility(self, mask, expected):
        self.assertEqual(mask.dtype, torch.float16)
        self.assertEqual(
            [row.nonzero().flatten().tolist() for row in (mask == 0)],
            expected,
        )
        self.assertTrue(torch.all((mask == 0) | (mask == -10000)))

    def test_noncausal_group_sees_entire_valid_context_and_query(self):
        self.check_visibility(self.mask([5, 7], False), [list(range(5))] * 3 + [list(range(7))] * 3)

    def test_sliding_group_has_causal_window_per_query(self):
        self.check_visibility(
            self.mask([4, 7], True, 3), [[0, 1], [0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]]
        )

    def test_explicit_noncausal_override_retains_both_window_bounds(self):
        self.check_visibility(self.mask([7], False, 2), [[3, 4, 5], [4, 5, 6], [5, 6]])
