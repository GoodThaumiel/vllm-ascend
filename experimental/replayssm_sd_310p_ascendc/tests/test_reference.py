"""CPU contracts for candidate windows and the request-owned ring."""

from __future__ import annotations

import unittest

import torch

from experimental.replayssm_sd_310p_ascendc.reference import (
    HISTORY,
    RING,
    compact_reference,
    cursor_reference,
    fp16_error,
    make_capacity_case,
    make_case,
    verify_reference,
)


class ReferenceTests(unittest.TestCase):
    def test_capacity_mixed_workload_preserves_row_distribution(self):
        for mtp in (3, 7):
            with self.subTest(mtp=mtp):
                width = mtp + 1
                case = make_capacity_case(mtp, 32, "mixed")
                rows = torch.stack([case[name] for name in ("positions", "lengths", "flush")], dim=1)
                self.assertEqual(len(rows.unique(dim=0)), 16)
                torch.testing.assert_close(rows[:16], rows[16:])
                self.assertGreater(int(case["flush"].sum()), 0)
                self.assertLess(int(case["flush"].sum()), 32)
                self.assertEqual(int(case["lengths"].max()), width)
                self.assertTrue(torch.all(case["positions"] + case["lengths"] <= HISTORY))

    def test_projected_verify_matches_dense_recurrence(self):
        for mtp, length, pos, base in ((3, 4, 0, 0), (7, 1, 7, 29), (7, 8, 16, 31)):
            with self.subTest(mtp=mtp, length=length, pos=pos, base=base):
                case = make_case(mtp=mtp, pos=pos, base=base)
                case["lengths"].fill_(length)
                expected = verify_reference(case)
                history = (torch.arange(pos) + base) % RING
                for key_head in range(8):
                    keys = case["k"][0, :length, key_head].double()
                    queries = case["q"][0, :length, key_head].double()
                    panel = torch.cat((keys, queries))
                    history_dots = case["k_cache"][1, key_head, history].double() @ panel.T
                    candidate_dots = keys @ panel.T
                    for head in (2 * key_head, 2 * key_head + 1):
                        projected = case["checkpoint"][1, head].double() @ panel.T
                        for i, physical in enumerate(history):
                            alpha = case["g_cache"][1, head, physical].double().exp()
                            delta = case["d_cache"][1, head, physical].double()
                            projected = alpha * projected + delta[:, None] * history_dots[i]
                        for token in range(length):
                            projected *= case["g"][0, token, head].double().exp()
                            value = case["v"][0, token, head].double()
                            beta = case["beta"][0, token, head].double()
                            delta = beta * (value - projected[:, token])
                            projected += delta[:, None] * candidate_dots[token]
                            out = (projected[:, length + token] * 128**-0.5).half()
                            physical = (base + pos + token) % RING
                            torch.testing.assert_close(out, expected["out"][0, token, head], atol=0, rtol=0)
                            torch.testing.assert_close(
                                delta.half(), expected["d_cache"][1, head, physical], atol=0, rtol=0
                            )

    def test_candidate_width_and_fp16_dtypes(self):
        for mtp in (3, 5, 7):
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

    def test_fp32_suffix_fold_matches_sequential_compact(self):
        for pos, base in ((1, 0), (7, 29), (12, 31), (16, 0), (16, 29)):
            with self.subTest(pos=pos, base=base):
                case = make_case(pos=pos, base=base)
                case["flush"].fill_(1)
                expected = compact_reference(case)["checkpoint"][1]
                physical = [(base + i) % RING for i in range(pos)]
                gate = case["g_cache"][1, :, physical].exp()
                key = case["k_cache"][1, :, physical].float().repeat_interleave(2, dim=0)
                delta = case["d_cache"][1, :, physical].float()
                weights = torch.empty_like(gate)
                scale = torch.ones_like(gate[:, 0])
                for i in reversed(range(pos)):
                    weights[:, i] = scale
                    scale = scale * gate[:, i]
                state = case["checkpoint"][1].float() * scale[:, None, None]
                for i in range(pos):
                    weighted_delta = delta[:, i] * weights[:, i, None]
                    state += weighted_delta[:, :, None] * key[:, i, None, :]
                torch.testing.assert_close(state.half(), expected, atol=2**-14, rtol=0)

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

    def test_mtp7_full_acceptance_wraps_and_keeps_pending_candidates(self):
        width = 8
        case = make_case(mtp=7, pos=8, base=28)
        case["flush"].fill_(1)
        verified = verify_reference(case)
        compacted = compact_reference(verified)
        wp, base, flush, status = cursor_reference(
            torch.tensor([0, 8], dtype=torch.int32),
            torch.tensor([0, 28], dtype=torch.int32),
            torch.tensor([0, 1], dtype=torch.int8),
            torch.tensor([width], dtype=torch.int32),
            case["ring_slots"],
            torch.zeros(1, dtype=torch.int8),
            width,
        )
        self.assertEqual((int(wp[1]), int(base[1]), int(flush[1]), int(status[0])), (8, 4, 1, 0))
        self.assertFalse(torch.equal(compacted["checkpoint"], case["checkpoint"]))
        for name in ("d_cache", "k_cache", "g_cache"):
            torch.testing.assert_close(compacted[name], verified[name], atol=0, rtol=0, equal_nan=True)
            self.assertTrue(torch.isfinite(compacted[name][1, :, 4:12]).all())

    def test_compact_metadata_errors_preserve_state(self):
        changes = (
            ("slots", -1, 1),
            ("slots", 0, 4),
            ("ring_slots", -1, 6),
            ("positions", HISTORY + 1, 2),
            ("bases", RING, 3),
            ("flush", 2, 5),
        )
        for field, value, error in changes:
            with self.subTest(field=field, value=value):
                case = make_case(mtp=3, pos=0)
                if field == "slots":
                    case["flush"][0] = 1 if value == 0 else 0
                elif field == "ring_slots":
                    case["flush"][0] = 1
                case[field][0] = value
                result = compact_reference(case)
                self.assertTrue(torch.all(result["compact_status"] == error))
                for name in ("checkpoint", "d_cache", "k_cache", "g_cache"):
                    torch.testing.assert_close(result[name], case[name], atol=0, rtol=0, equal_nan=True)

    def test_verify_metadata_errors_preserve_cache(self):
        changes = (
            ("slots", -1, 1),
            ("slots", 0, 4),
            ("ring_slots", -1, 5),
            ("positions", HISTORY + 1, 2),
            ("bases", RING, 3),
            ("lengths", 0, 2),
        )
        for field, value, error in changes:
            with self.subTest(field=field, value=value):
                case = make_case(mtp=3, pos=0)
                case[field][0] = value
                result = verify_reference(case)
                self.assertTrue(torch.all(result["verify_status"] == error))
                self.assertTrue(torch.all(result["out"] == 0))
                for name in ("checkpoint", "d_cache", "k_cache", "g_cache"):
                    torch.testing.assert_close(result[name], case[name], atol=0, rtol=0, equal_nan=True)

    def test_invalid_geometry(self):
        for kwargs in ({"mtp": 4}, {"pos": 17}, {"base": 32}, {"tile_v": 128}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                make_case(**kwargs)


if __name__ == "__main__":
    unittest.main()
