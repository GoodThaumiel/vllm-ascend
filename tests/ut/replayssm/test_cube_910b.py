# SPDX-License-Identifier: Apache-2.0
import unittest

import torch

from experimental.replayssm_cube_910b.projections import compact, verify
from experimental.replayssm_sd_ascendc.lab import make_case, recurrent_golden
from experimental.replayssm_sd_compact_ascendc.lab import compact_golden


class TestCubeDemo(unittest.TestCase):
    def test_window_and_fold(self):
        for t, p, base, active in [(1, 0, 0, 0), (4, 1, 31, 2), (4, 15, 31, 1), (1, 16, 20, 2)]:
            with self.subTest(t=t, p=p, base=base, active=active):
                c = make_case(batch=2, length=t, position=p, base=base, tile_v=128)
                expected = recurrent_golden(c)
                c["flush"] = torch.tensor([int(i < active) for i in range(2)])
                expected_ckpt = compact_golden(c)["checkpoint"]
                c["checkpoint"] = torch.cat(
                    (torch.zeros_like(c["checkpoint"][:1]).expand(4, -1, -1, -1), c["checkpoint"])
                )
                verify(c, t, p, base)
                for key in ("out", "d_cache", "k_cache", "g_cache"):
                    torch.testing.assert_close(c[key], expected[key], atol=0.001, rtol=0.015, equal_nan=True)
                compact(c, active, p, base, torch.empty(96, dtype=torch.int32))
                torch.testing.assert_close(c["checkpoint"][4:], expected_ckpt, atol=0.001, rtol=0.015)
                self.assertTrue(torch.equal(c["checkpoint"][5 + active :], expected_ckpt[1 + active :]))
