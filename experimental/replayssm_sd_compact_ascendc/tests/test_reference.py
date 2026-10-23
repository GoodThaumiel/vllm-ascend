# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import unittest

import torch

from experimental.replayssm_sd_compact_ascendc.lab import (
    clone_page_strided_case,
    compact_golden,
    make_case,
)


class TestSDCompactReference(unittest.TestCase):
    def test_wraparound_and_uncommitted_draft_tail(self):
        case = make_case(tp=8, batch=1, position=16, base=31, ring_len=32)
        slot = 1
        case["checkpoint"][slot].zero_()
        for name in ("d_cache", "k_cache", "g_cache"):
            for i in range(16):
                phys = (31 + i) & 31
                case[name][slot, :, phys].zero_()
        for i, d, key in ((0, 2.0, 3.0), (12, 5.0, 7.0)):
            phys = (31 + i) & 31
            case["d_cache"][slot, 0, phys, 0] = d
            case["k_cache"][slot, 0, phys, 0] = key
        result = compact_golden(case)
        self.assertEqual(float(result["checkpoint"][slot, 0, 0, 0]), 41.0)
        self.assertTrue(torch.isfinite(result["checkpoint"][slot]).all())
        for name in ("d_cache", "k_cache", "g_cache"):
            torch.testing.assert_close(result[name], case[name], atol=0, rtol=0, equal_nan=True)

    def test_nonflush_row_and_page_strides(self):
        case = make_case(tp=4, batch=2, position=13, base=30)
        case["flush"][1] = 0
        packed = clone_page_strided_case(case)
        for name in ("checkpoint", "d_cache", "k_cache", "g_cache"):
            self.assertFalse(packed[name].is_contiguous())
        result = compact_golden(packed)
        torch.testing.assert_close(result["checkpoint"][2], packed["checkpoint"][2], atol=0, rtol=0)
        self.assertTrue(torch.isfinite(result["checkpoint"][1]).all())
        self.assertTrue(torch.all(result["status"] == 0))


if __name__ == "__main__":
    unittest.main()
