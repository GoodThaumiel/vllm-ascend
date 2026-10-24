# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab import clone_case, clone_page_strided_case, make_case, output_only_reference, recurrent_golden  # noqa: E402


class SDReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_output_only_matches_dense_recurrence(self):
        for tp, pos, length, base in ((4, 0, 1, 0), (4, 7, 4, 29), (8, 12, 4, 31), (8, 4, 2, 13)):
            with self.subTest(tp=tp, pos=pos, length=length, base=base):
                case = make_case(tp=tp, position=pos, length=length, base=base, seed=pos + length)
                dense = recurrent_golden(case)["out"]
                replay = output_only_reference(case)
                torch.testing.assert_close(replay, dense, rtol=0, atol=0)

    def test_verify_keeps_checkpoint_and_poisoned_tail(self):
        case = make_case(position=7, length=2, base=31)
        before = clone_case(case)
        result = recurrent_golden(case)
        torch.testing.assert_close(result["checkpoint"], before["checkpoint"], rtol=0, atol=0)
        self.assertTrue(result["d_cache"][1, :, 12].isnan().all())
        self.assertTrue(result["k_cache"][1, :, 12].isnan().all())
        self.assertTrue(result["g_cache"][1, :, 12].isnan().all())
        self.assertEqual(int(torch.count_nonzero(result["out"][:, 2:])), 0)

    def test_page_strided_shared_backing_and_wrap(self):
        case = make_case(position=7, length=4, base=31)
        packed = clone_page_strided_case(case)
        self.assertEqual(
            len(
                {packed[name].untyped_storage().data_ptr() for name in ("checkpoint", "d_cache", "k_cache", "g_cache")}
            ),
            1,
        )
        self.assertFalse(packed["checkpoint"].is_contiguous())
        expected = recurrent_golden(case)
        actual = recurrent_golden(packed)
        for name in ("out", "checkpoint", "d_cache", "k_cache", "g_cache"):
            torch.testing.assert_close(actual[name], expected[name], equal_nan=True, rtol=0, atol=0)

    def test_window_boundary_and_invalid_geometry(self):
        self.assertEqual(recurrent_golden(make_case(position=12, length=4))["out"].shape[1], 4)
        self.assertEqual(recurrent_golden(make_case(position=16, length=4, ring_len=32))["out"].shape[1], 4)
        with self.assertRaisesRegex(ValueError, "physical ring"):
            make_case(position=13, length=4, ring_len=16)
        with self.assertRaisesRegex(ValueError, "Invalid TP"):
            make_case(tp=3)

    def test_tile64_uses_two_value_tiles_per_head(self):
        for tp in (4, 8):
            with self.subTest(tp=tp):
                case = make_case(tp=tp, batch=2, position=12, length=4, tile_v=64)
                self.assertEqual(case["status"].numel(), 2 * (48 // tp) * 2)
                torch.testing.assert_close(output_only_reference(case), recurrent_golden(case)["out"], rtol=0, atol=0)

    def test_padding_and_mixed_window_lengths(self):
        case = make_case(batch=3, position=7, length=4, base=30, padding=True)
        case["lengths"][0] = 1
        actual = output_only_reference(case)
        expected = recurrent_golden(case)["out"]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(int(torch.count_nonzero(actual[-1])), 0)


if __name__ == "__main__":
    unittest.main()
