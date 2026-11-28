# SPDX-License-Identifier: Apache-2.0
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from experimental.replayssm_cube_910b.plot import contract
from experimental.replayssm_cube_910b.run import (
    assert_stage,
    last_projections,
    parse_args,
    projection_error_summary,
    reference_projections,
)
from experimental.replayssm_sd_ascendc.lab import make_case, recurrent_golden


class TestMixedVerifyProbe(unittest.TestCase):
    def test_per_head_diagnostic_keeps_early_projections(self):
        case = make_case(geometry="9b", tp=1, batch=2)
        heads = torch.arange(32).repeat(2)
        rows = torch.arange(2).repeat_interleave(32)
        all_projections = reference_projections(case, rows, heads)
        torch.testing.assert_close(last_projections(case, 16), all_projections[33::2])
        for index in (0, 18, 63):
            row, head = divmod(index, 32)
            rhs = torch.cat((case["k"][row, :, head // 2], case["q"][row, :, head // 2])).float()
            torch.testing.assert_close(all_projections[index, :, :8], case["checkpoint"][row + 1, head].float() @ rhs.T)

    def test_wide_probe_places_q_after_eight_k_columns(self):
        for window in (6, 8):
            with self.subTest(window=window):
                case = make_case(batch=1, length=window, window=window)
                state = case["checkpoint"][1, 0].float()
                projected = reference_projections(case, torch.tensor([0]), torch.tensor([0]), panel_tokens=8)[0]
                torch.testing.assert_close(projected[:, :window], state @ case["k"][0, :, 0].float().T)
                torch.testing.assert_close(projected[:, 8 : 8 + window], state @ case["q"][0, :, 0].float().T)
                self.assertEqual(projected[:, window:8].count_nonzero().item(), 0)
                self.assertEqual(projected[:, 8 + window :].count_nonzero().item(), 0)

    def test_wide_column_recurrence_reduces_modeled_work(self):
        row = dict(batch=32, length=8, pos=8, active_flush=16, nonflush_pos=4, aiv_panel_cols=16)
        original = contract({**row, "path": "mixed_fused_fold"})
        columns = contract({**row, "path": "mixed_column_recurrence"})
        self.assertLess(columns[0], original[0])
        self.assertEqual(columns[1], original[1])

    def test_failure_names_tensor_and_preserves_values(self):
        actual, expected = torch.tensor([0.0, 0.0227]), torch.zeros(2)
        with TemporaryDirectory() as directory:
            dump = Path(directory) / "failed.pt"
            with self.assertRaisesRegex(AssertionError, "mixed eager/d_cache"):
                assert_stage(actual, expected, "mixed eager/d_cache", atol=0.01, rtol=0.03, dump=dump)
            result = torch.load(dump, weights_only=True)
            self.assertEqual(result["stage"], "mixed eager/d_cache")
            torch.testing.assert_close(result["actual"], actual)
            torch.testing.assert_close(result["expected"], expected)

    def test_projection_failure_summary_groups_k_q_and_padding_by_tile_half(self):
        expected = torch.zeros(2, 128, 16)
        expected[..., :8] = 0.01
        actual = expected.clone()
        actual[0, :, :4] = 0
        actual[1, 64:, 4:8] = 0.02
        actual[1, 0, 8] = float("nan")
        summary = projection_error_summary(actual, expected)
        self.assertEqual(summary["k"]["mismatches"], 512)
        self.assertEqual(summary["k"]["zero_values"], 512)
        self.assertEqual(summary["k"]["tile_halves"], [(0, 0, 256), (0, 1, 256)])
        self.assertEqual(summary["q"]["tile_halves"], [(1, 1, 256)])
        self.assertEqual(summary["q"]["zero_values"], 0)
        self.assertEqual(summary["padding"]["nonfinite_values"], 1)
        self.assertEqual(summary["padding"]["tile_halves"], [(1, 0, 1)])

    def test_workspace_reference_uses_last_task_and_head(self):
        for geometry, tp in (("27b", 4), ("9b", 1)):
            case = make_case(geometry=geometry, tp=tp, batch=3)
            _, _, hk, dim = case["k"].shape
            group = case["v"].shape[2] // hk
            for blocks in (1, 5, min(20, 3 * hk)):
                expected = last_projections(case, blocks)
                for block in range(blocks):
                    row, key = divmod(list(range(block, 3 * hk, blocks))[-1], hk)
                    state = case["checkpoint"][row + 1, (key + 1) * group - 1].float()
                    rhs = torch.cat((case["k"][row, :, key], case["q"][row, :, key])).float()
                    torch.testing.assert_close(expected[block, :, :8], state @ rhs.T)
                self.assertEqual(expected.shape, (blocks, dim, 16))
                self.assertEqual(expected[:, :, 8:].count_nonzero().item(), 0)

    def test_balanced_workspace_records_last_value_head_even_if_group_splits(self):
        case = make_case(geometry="27b", tp=4, batch=16)
        expected = reference_projections(case, torch.tensor([0, 1, 15]), torch.tensor([8, 6, 11]))
        actual = last_projections(case, 20, balanced=True)
        torch.testing.assert_close(actual[[0, 1, 19]], expected)

    def test_two_tiles_projected_recurrence_matches_dense(self):
        # Simulate the 128x16 FP32 handoff to two V64 consumers. Exercise both
        # head-sharing ratios, a wrapped committed ring and padded K/Q columns.
        for geometry, tp, pos, length in (("27b", 4, 0, 1), ("9b", 1, 16, 4), ("27b", 4, 15, 1)):
            case = make_case(geometry=geometry, tp=tp, batch=1, position=pos, length=length, base=31)
            target = recurrent_golden(case)
            _, _, hk, _ = case["k"].shape
            hv = case["v"].shape[2]
            group = hv // hk
            actual = torch.zeros_like(case["out"])
            actual_d = case["d_cache"].clone()
            for key_head in range(hk):
                rhs = torch.zeros((128, 16), dtype=torch.float64)
                keys = case["k"][0, :length, key_head].double()
                rhs[:, :length] = keys.T
                rhs[:, 4 : 4 + length] = case["q"][0, :length, key_head].double().T
                history_indices = [(31 + i) % 32 for i in range(pos)]
                history_keys = case["k_cache"][1, key_head, history_indices].double()
                dots = torch.cat((history_keys, keys)) @ rhs
                for head in range(key_head * group, (key_head + 1) * group):
                    projection = case["checkpoint"][1, head].double() @ rhs
                    for begin in (0, 64):
                        tile = slice(begin, begin + 64)
                        y = projection[tile].clone()
                        for i, phys in enumerate(history_indices):
                            y.mul_(case["g_cache"][1, head, phys].double().exp())
                            y.add_(case["d_cache"][1, head, phys, tile].double()[:, None] * dots[i])
                        for t in range(length):
                            y.mul_(case["g"][0, t, head].double().exp())
                            delta = (case["v"][0, t, head, tile].double() - y[:, t]) * case["beta"][0, t, head]
                            y.add_(delta[:, None] * dots[pos + t])
                            actual[0, t, head, tile] = y[:, 4 + t] * 128**-0.5
                            actual_d[1, head, (31 + pos + t) % 32, tile] = delta
            torch.testing.assert_close(actual, target["out"], atol=0, rtol=0)
            torch.testing.assert_close(actual_d, target["d_cache"], atol=0, rtol=0, equal_nan=True)

    def test_sweep_mapping_and_roofline(self):
        for geometry in ("27b", "9b"):
            args = parse_args(
                [
                    "--geometry",
                    geometry,
                    "--suite",
                    "mixed",
                    "--mixed-library",
                    "fixture.so",
                    "--output",
                    "unused.json",
                ]
            )
            self.assertEqual(args.batches, tuple(range(1, 129)))
            self.assertEqual(args.path[-2:], ("mixed_verify", "mixed_pair"))
            common = dict(batch=128, length=4, pos=15, active_flush=16, value_heads=32, key_heads=16)
            verify = contract(dict(common, path="mixed_verify"))
            compact = contract(dict(common, path="compact_v1"))
            pair = contract(dict(common, path="mixed_pair"))
            self.assertEqual(pair, tuple(a + b for a, b in zip(verify, compact)))
