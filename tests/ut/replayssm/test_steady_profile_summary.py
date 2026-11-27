# SPDX-License-Identifier: Apache-2.0
import unittest

from scripts.replayssm.summarize_steady_decode_profile import summarize


class TestSteadyDecodeProfileSummary(unittest.TestCase):
    def test_filters_to_target_and_counts_overlapping_device_time_once(self):
        trace = [{"name": "REPLAY_STEADY_TARGET/FULL/B128/step6", "ph": "X", "ts": "100", "dur": 40}]

        def row(start, duration, name):
            return {"Start Time(us)": start, "Duration(us)": duration, "Name": name, "Type": "", "Model ID": "41"}

        report = summarize(
            trace,
            [
                row("99", "2", "MTP"),
                row("108", "20", "RecurrentGatedDeltaRule"),
                row("120", "14", "IndexCopy"),
                row("141", "3", "MTP"),
            ],
        )
        forward = report["forwards"][0]
        self.assertEqual(forward["device_busy_union_us"], 26)
        self.assertEqual(forward["groups"]["native_gdr"]["sum_us"], 20)
        self.assertEqual(forward["groups"]["index_copy_or_select"]["calls"], 1)
        self.assertNotIn("MTP", report["kernel_medians"])
