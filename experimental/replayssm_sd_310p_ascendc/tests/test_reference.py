"""CPU contracts for the two candidate widths and request-owned ring."""

from __future__ import annotations

import unittest

import torch

from experimental.replayssm_sd_310p_ascendc.reference import (
    HISTORY,
    RING,
    compact_reference,
    cursor_reference,
    fp16_error,
    make_case,
    verify_reference,
)


class ReferenceTests(unittest.TestCase):
    def test_candidate_width_and_fp16_dtypes(self):
        for mtp in (3, 5):
            with self.subTest(mtp=mtp):
                case = make_case(mtp=mtp, pos=0)
                self.assertEqual(case["q"].shape, (1, mtp + 1, 8, 128))
                self.assertEqual(case["v"].shape, (1, mtp + 1, 16, 128))
                self.assertEqual(case["checkpoint"].dtype, torch.float16)
                self.assertEqual(case["beta"].dtype, torch.float16)
                self.assertEqual(case["g_cache"].dtype, torch.float32)
                result = verify_reference(case)
                self.assertTrue(torch.isfinite(result["out"]).all())
                self.assertTrue(torch.equal(result["checkpoint"], case["checkpoint"]))
                fp16_error(result["out"], result["out"])

    def test_wrap_and_flush_keep_candidate_ring(self):
        case = make_case(mtp=5, pos=16, base=31)
        case["flush"].fill_(1)
        verified = verify_reference(case)
        compacted = compact_reference(verified)
        self.assertFalse(torch.equal(compacted["checkpoint"], verified["checkpoint"]))
        for name in ("d_cache", "k_cache", "g_cache"):
            torch.testing.assert_close(compacted[name], verified[name], atol=0, rtol=0, equal_nan=True)
        self.assertTrue(torch.isfinite(verified["d_cache"][1, :, (31 + 16) & 31]).all())

    def test_detached_ring_matches_tied_geometry(self):
        tied = make_case(mtp=5, batch=2, pos=12, base=31)
        tied["flush"].fill_(1)
        detached = {name: value.clone() for name, value in tied.items()}
        for name in ("d_cache", "k_cache", "g_cache"):
            detached[name][[1, 2]] = detached[name][[2, 1]].clone()
        detached["ring_slots"][:] = torch.tensor([2, 1], dtype=torch.int32)
        tied_result = compact_reference(verify_reference(tied))
        detached_result = compact_reference(verify_reference(detached))
        torch.testing.assert_close(detached_result["out"], tied_result["out"], atol=0, rtol=0)
        torch.testing.assert_close(detached_result["checkpoint"], tied_result["checkpoint"], atol=0, rtol=0)
        for name in ("d_cache", "k_cache", "g_cache"):
            torch.testing.assert_close(
                detached_result[name][[2, 1]], tied_result[name][1:], atol=0, rtol=0, equal_nan=True
            )

    def test_cursor_mtp5_flush_and_reset(self):
        wp = torch.tensor([0, 12], dtype=torch.int32)
        base = torch.tensor([0, 31], dtype=torch.int32)
        flush = torch.tensor([0, 1], dtype=torch.int8)
        accepted = torch.tensor([5], dtype=torch.int32)
        slots = torch.tensor([1], dtype=torch.int32)
        reset = torch.zeros(1, dtype=torch.int8)
        next_wp, next_base, next_flush, status = cursor_reference(
            wp,
            base,
            flush,
            accepted,
            slots,
            reset,
            5,
        )
        self.assertEqual(int(next_wp[1]), 5)
        self.assertEqual(int(next_base[1]), (31 + 12) & (RING - 1))
        self.assertEqual(int(next_flush[1]), int(HISTORY < 5 + 2 * 5))
        self.assertEqual(int(status[0]), 0)
        reset[0] = 1
        next_wp, next_base, next_flush, status = cursor_reference(
            wp,
            base,
            flush,
            accepted,
            slots,
            reset,
            5,
        )
        self.assertEqual((int(next_wp[1]), int(next_base[1]), int(next_flush[1]), int(status[0])), (0, 0, 0, 0))

    def test_padding_has_zero_status(self):
        case = make_case(mtp=3, batch=2, padding=True)
        result = verify_reference(case)
        self.assertTrue(torch.equal(result["out"][1], torch.zeros_like(result["out"][1])))
        self.assertTrue(torch.equal(result["verify_status"], torch.zeros_like(result["verify_status"])))

    def test_invalid_geometry(self):
        for kwargs in ({"mtp": 4}, {"pos": 17}, {"base": 32}, {"tile_v": 128}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                make_case(**kwargs)


if __name__ == "__main__":
    unittest.main()
