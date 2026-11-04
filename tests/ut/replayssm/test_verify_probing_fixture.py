# SPDX-License-Identifier: Apache-2.0
import unittest

import torch

from experimental.replayssm_sd_ascendc.lab import make_case
from experimental.replayssm_sd_ascendc.probing.run import fixture


class TestProbeFixture(unittest.TestCase):
    def test_detached_mapping(self):
        for pos in (0, 1, 7, 16):
            for base in (0, 31):
                case = fixture(2, pos, 4, base)
                tied = make_case(tp=4, batch=2, position=16, length=4, base=base, ring_len=32, tile_v=128, seed=42)
                for name in ("d_cache", "k_cache", "g_cache"):
                    tied[name].nan_to_num_(nan=0.0)
                self.assertEqual(case["slots"].tolist(), [5, 6])
                self.assertEqual(case["ring_slots"].tolist(), [1, 2])
                torch.testing.assert_close(case["checkpoint"][5:], tied["checkpoint"][1:])
                for name in ("d_cache", "k_cache", "g_cache", "q", "k", "v"):
                    torch.testing.assert_close(case[name], tied[name])
                history = {(base + i) % 32 for i in range(pos)}
                candidates = {(base + pos + i) % 32 for i in range(4)}
                self.assertFalse(history & candidates)

    def test_padding(self):
        case = fixture(2, 7, 4, 31, True)
        self.assertEqual(case["slots"][-1].item(), 0)
        self.assertEqual(case["ring_slots"][-1].item(), 0)


if __name__ == "__main__":
    unittest.main()
