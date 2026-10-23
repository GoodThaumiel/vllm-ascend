# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""CPU checks for the SD lifecycle profiler's per-block cursor setup."""

import importlib.util
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "profile_sd_lifecycle_compare", ROOT / "scripts/replayssm/profile_sd_lifecycle_compare.py"
)
PROFILE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROFILE)


class SDLifecycleSetupTests(unittest.TestCase):
    def test_cursor_is_block_keyed_and_padding_slot_is_reserved(self):
        case = PROFILE.make_cursor_case(32, position=16, base=31, flush=True, accepted=4, device=torch.device("cpu"))
        self.assertEqual(case["write_pos"].shape, (33,))
        self.assertEqual(case["slots"].tolist(), list(range(1, 33)))
        self.assertEqual(int(case["write_pos"][0]), 0)
        self.assertEqual(int(case["cache_base"][0]), 0)
        self.assertEqual(int(case["is_flush"][0]), 0)
        self.assertTrue(torch.all(case["write_pos"][1:] == 16))
        self.assertTrue(torch.all(case["cache_base"][1:] == 31))
        self.assertTrue(torch.all(case["is_flush"][1:] == 1))
        self.assertTrue(torch.all(case["accepted"] == 4))

    def test_p50_p90_use_device_samples(self):
        self.assertEqual(
            PROFILE.p50_p90([10.0, 11.0, 12.0, 100.0]),
            {
                "p50_us": 11.5,
                "p90_us": 100.0,
                "min_us": 10.0,
            },
        )


if __name__ == "__main__":
    unittest.main()
