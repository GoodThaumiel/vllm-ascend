# SPDX-License-Identifier: Apache-2.0
import unittest

from experimental.replayssm_cube_910b.model_flush import estimate_mean


class TestFlushLatencyModel(unittest.TestCase):
    def test_weights_the_locally_measured_curve(self):
        curve = {f: {"gdn_mean_us": latency} for f, latency in ((0, 100), (4, 180), (32, 340))}
        self.assertEqual(estimate_mean(curve, {0: 1, 2: 2, 4: 1}), 140)
        with self.assertRaisesRegex(ValueError, "outside measured range"):
            estimate_mean(curve, {33: 1})
