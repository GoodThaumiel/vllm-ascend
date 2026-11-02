# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""CPU contract for the fused verify-then-compact experiment."""

import unittest

import torch

from scripts.replayssm.run_sd_fusion_ablation import expected_case, make_fusion_case


class SDFusionReferenceTests(unittest.TestCase):
    def test_flush_only_folds_committed_history(self):
        for tp in (4, 8):
            for mode in ("none", "all", "mixed"):
                for detached in (False, True):
                    with self.subTest(tp=tp, mode=mode, detached=detached):
                        case = make_fusion_case(tp, 2, 12, 31, mode, detached=detached)
                        self._check_case(case, 48 // tp)

    def test_padding_preserves_status_and_checkpoint(self):
        case = make_fusion_case(4, 2, 12, 31, "all", detached=True, padding=True)
        self._check_case(case, 12)

    def _check_case(self, case, value_heads):
        expected = expected_case(case)
        self.assertEqual(tuple(expected["out"].shape), (2, 4, value_heads, 128))
        self.assertEqual(int(torch.count_nonzero(expected["status"])), 0)
        self.assertEqual(int(torch.count_nonzero(expected["compact_status"])), 0)
        for row in range(2):
            slot = int(case["slots"][row])
            if int(case["flush"][row]) == 0:
                torch.testing.assert_close(expected["checkpoint"][slot], case["checkpoint"][slot])
            else:
                self.assertFalse(torch.equal(expected["checkpoint"][slot], case["checkpoint"][slot]))


if __name__ == "__main__":
    unittest.main()
