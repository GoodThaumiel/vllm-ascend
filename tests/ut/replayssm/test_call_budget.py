# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts/replayssm"))
from call_budget import estimate  # noqa: E402


class CallBudgetTests(unittest.TestCase):
    def test_tp_is_parallel_and_batch_does_not_multiply_launches(self):
        config = {
            "text_config": {"num_hidden_layers": 64, "layer_types": ["linear_attention"] * 48 + ["full_attention"] * 16}
        }
        for tp in (4, 8):
            budget = estimate(config, tp, 150, 220)
            self.assertEqual(budget["calls_all_ranks"], 48 * tp)
            self.assertEqual(budget["calls_per_rank_per_pure_decode_forward"], 48)
            self.assertAlmostEqual(budget["illustrative_replay_ms"], 7.2)
            self.assertAlmostEqual(budget["illustrative_saved_ms"], 3.36)
            self.assertEqual(budget["flush_extra_calls"], 0)

    def test_rejects_config_without_explicit_layer_types(self):
        with self.assertRaises(ValueError):
            estimate({"num_hidden_layers": 64}, 4, 150, 220)

    def test_all_full_attention_model_has_zero_replay_calls(self):
        budget = estimate({"num_hidden_layers": 2, "layer_types": ["full_attention"] * 2}, 8, 150, 220)
        self.assertEqual(budget["calls_all_ranks"], 0)


if __name__ == "__main__":
    unittest.main()
