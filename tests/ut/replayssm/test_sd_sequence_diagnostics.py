# SPDX-License-Identifier: Apache-2.0
"""CPU checks for the sequential SD numerical diagnostic."""

import importlib.util
import unittest
from pathlib import Path

import torch

from experimental.replayssm_sd_ascendc.lab import make_case, recurrent_golden

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "diagnose_sd_sequence", ROOT / "scripts/replayssm/diagnose_sd_sequence.py"
)
assert SPEC is not None and SPEC.loader is not None
DIAG = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DIAG)


class SDSequenceDiagnosticsTests(unittest.TestCase):
    def test_dense_window_matches_reference_at_empty_history(self) -> None:
        case = make_case(tp=4, batch=1, position=0, length=4, ring_len=32, tile_v=128, seed=7)
        state = case["checkpoint"][1].double()
        actual, states = DIAG.dense_window(
            state,
            case["q"][0],
            case["k"][0],
            case["v"][0],
            case["g"][0],
            case["beta"][0],
        )
        expected = recurrent_golden(case)["out"][0]
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        self.assertEqual(len(states), 4)

    def test_compare_reports_first_mismatch(self) -> None:
        result = DIAG.compare(torch.tensor([1.0, 2.1]), torch.tensor([1.0, 2.0]), atol=0.01, rtol=0)
        self.assertEqual(result["mismatch_count"], 1)
        self.assertEqual(result["first_mismatch_index"], [1])


if __name__ == "__main__":
    unittest.main()
