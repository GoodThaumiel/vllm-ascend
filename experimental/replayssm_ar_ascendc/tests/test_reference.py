# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import sys
import unittest
from argparse import Namespace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab import (  # noqa: E402
    BUFFER_NAMES,
    check_result,
    clone_case,
    golden,
    make_case,
    ub_bytes,
)
from run import child_command  # noqa: E402


class ReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_analytic_rank_one_and_layout(self):
        case = make_case(tp=8, batch=1, flush=True)
        for name in ("q", "k", "checkpoint", "g"):
            case[name].zero_()
        case["beta"].fill_(1)
        case["q"][:, :, 11] = 1
        case["k"][:, :, 11] = 1
        expected = golden(case)
        torch.testing.assert_close(expected["out"], (case["v"].float() * 128**-0.5).bfloat16())
        torch.testing.assert_close(expected["checkpoint"][1, :, :, 11], case["v"][0])
        self.assertEqual(torch.count_nonzero(expected["checkpoint"][1, :, :, :11]), 0)
        for name in ("d_cache", "k_cache", "g_cache"):
            torch.testing.assert_close(expected[name], case[name], equal_nan=True)

    def test_closed_form_matches_independent_sequential_recurrence(self):
        for tp, pos in ((4, 0), (4, 7), (8, 15), (8, 63)):
            with self.subTest(tp=tp, pos=pos):
                case = make_case(tp=tp, batch=1, capacity=64, position=pos, flush=True, seed=pos)
                actual = golden(case)
                group = case["v"].shape[1] // case["q"].shape[1]
                for h in range(case["v"].shape[1]):
                    state = case["checkpoint"][1, h].double()
                    for i in range(pos):
                        state = case["g_cache"][1, h, i].double().exp() * state
                        state += (
                            case["d_cache"][1, h, i].double()[:, None]
                            * case["k_cache"][1, h // group, i].double()[None, :]
                        )
                    k, q = case["k"][0, h // group].double(), case["q"][0, h // group].double()
                    state *= case["g"][0, h].double().exp()
                    delta = case["beta"][0, h].double() * (case["v"][0, h].double() - state @ k)
                    state += delta[:, None] * k[None, :]
                    torch.testing.assert_close(actual["checkpoint"][1, h], state.bfloat16(), rtol=0, atol=0)
                    torch.testing.assert_close(actual["out"][0, h], (state @ q * 128**-0.5).bfloat16(), rtol=0, atol=0)

    def test_nonflush_changes_only_current_cache_position(self):
        case = make_case(tp=8, batch=2, position=7, padding=True)
        expected = golden(case)
        check_result(expected, expected, case)
        torch.testing.assert_close(expected["checkpoint"], case["checkpoint"], rtol=0, atol=0)
        self.assertEqual(torch.count_nonzero(expected["out"][-1]), 0)
        for name in ("d_cache", "k_cache", "g_cache"):
            self.assertTrue(expected[name][1, :, 8:].isnan().all())
            self.assertTrue(expected[name][1, :, 7].isfinite().all())

    def test_capacity_one_flush_and_padding(self):
        case = make_case(tp=8, batch=2, capacity=1, flush=True, padding=True)
        expected = golden(case)
        check_result(expected, expected, case)
        self.assertEqual(torch.count_nonzero(expected["out"][-1]), 0)

    def test_metadata_rejects_duplicate_and_out_of_bounds_slots(self):
        case = make_case(tp=8, batch=2)
        case["slots"][1] = case["slots"][0]
        with self.assertRaisesRegex(ValueError, "unique"):
            golden(case)
        case["slots"][1] = 99
        with self.assertRaisesRegex(ValueError, "outside"):
            golden(case)

    def test_last_position_requires_flush(self):
        with self.assertRaisesRegex(ValueError, "must flush"):
            make_case(tp=8, position=15)

    def test_ub_budget_covers_all_tilings(self):
        for capacity in range(1, 65):
            for tile_v in (16, 32, 64):
                self.assertLess(ub_bytes(capacity, tile_v), 192 * 1024)
        self.assertEqual(ub_bytes(64, 32), 104096)
        self.assertEqual(ub_bytes(64, 64), 153632)

    def test_checker_rejects_out_of_range_write_and_nonfinite_output(self):
        before = make_case(tp=8, batch=1, position=7)
        expected = golden(before)
        actual = clone_case(expected)
        actual["d_cache"][0, 0, 0, 0] = 1
        with self.assertRaises(AssertionError):
            check_result(actual, expected, before)
        actual = clone_case(expected)
        actual["out"][0, 0, 0] = float("nan")
        with self.assertRaisesRegex(AssertionError, "nonfinite"):
            check_result(actual, expected, before)
        actual = clone_case(expected)
        actual["out"].zero_()
        with self.assertRaises(AssertionError):
            check_result(actual, expected, before)

    def test_checker_preserves_device_history_with_prior_rounding(self):
        before = make_case(tp=8, batch=1, position=7)
        expected = golden(before)
        preserved = clone_case(before)
        preserved["checkpoint"][1, 0, 0, 0] += 0.001
        actual = clone_case(expected)
        actual["checkpoint"].copy_(preserved["checkpoint"])
        check_result(actual, expected, before, preserved)

    def test_many_steps_do_not_read_poisoned_tail(self):
        case = make_case(tp=8, batch=1, capacity=4)
        for step in range(13):
            inputs = make_case(tp=8, batch=1, capacity=4, seed=step + 10)
            for name in ("q", "k", "v", "g", "beta"):
                case[name] = inputs[name]
            case["flush"] = (case["positions"] == 3).int()
            result = golden(case)
            check_result(result, result, case)
            for name in BUFFER_NAMES:
                case[name] = result[name]
            case["positions"] = torch.where(case["flush"].bool(), 0, case["positions"] + 1)

    def test_child_command_passes_required_output(self):
        args = Namespace(
            stage="smoke",
            tp=4,
            capacity=16,
            batch=2,
            tile_v=32,
            cores=1,
            library=Path("/tmp/lib.so"),
            iterations=20,
            native_baseline=False,
        )
        command = child_command(args, 3, Path("/tmp/results/device3.json"))
        self.assertEqual(command[command.index("--output") + 1], "/tmp/results")
        self.assertEqual(command[command.index("--child-device") + 1], "3")
        self.assertEqual(command[command.index("--stage") + 1], "smoke")


if __name__ == "__main__":
    unittest.main()
