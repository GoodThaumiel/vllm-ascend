# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import sys
import unittest
from argparse import Namespace
from pathlib import Path

CONTRACT = Path(__file__).resolve().parents[3] / "vllm_ascend" / "ops" / "replayssm"
sys.path.insert(0, str(CONTRACT))

from contract import ReplaySSMARLayout  # noqa: E402

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts" / "replayssm"
sys.path.insert(0, str(SCRIPTS))

from run_level2_ar import child_command as ar_child_command  # noqa: E402
from run_level2_masked_dot_probes import PROBES as MASKED_DOT_PROBES  # noqa: E402
from run_level2_masked_dot_probes import child_command as masked_dot_child_command  # noqa: E402
from run_level2_triton_probes import PROBES  # noqa: E402
from run_level2_triton_probes import child_command as probe_child_command  # noqa: E402


class ReplaySSMARContractTests(unittest.TestCase):
    def test_tp4_and_tp8_qwen_shapes(self):
        for tp, key_heads, value_heads in ((4, 4, 12), (8, 2, 6)):
            with self.subTest(tp=tp):
                layout = ReplaySSMARLayout(2, 5, key_heads, value_heads, 128, 128, 16)
                self.assertEqual(layout.shapes["mixed_qkv"], (2, 2 * key_heads * 128 + value_heads * 128))
                self.assertEqual(layout.shapes["checkpoint"], (5, value_heads, 128, 128))
                self.assertEqual(layout.shapes["d_cache"], (5, value_heads, 16, 128))
                self.assertEqual(layout.shapes["k_cache"], (5, key_heads, 16, 128))

    def test_cursor_requires_flush_at_last_position(self):
        layout = ReplaySSMARLayout(1, 2, 1, 3, 16, 16, 4)
        layout.validate_metadata([1], [3], [1])
        with self.assertRaisesRegex(ValueError, "must flush"):
            layout.validate_metadata([1], [3], [0])
        self.assertEqual(layout.advance(3, True), 0)
        with self.assertRaisesRegex(ValueError, "must flush"):
            layout.advance(3, False)

    def test_slot_zero_is_padding_and_never_carries_history(self):
        layout = ReplaySSMARLayout(2, 3, 1, 2, 16, 16, 4)
        layout.validate_metadata([0, 2], [0, 1], [0, 0])
        with self.assertRaisesRegex(ValueError, "padded row"):
            layout.validate_metadata([0, 2], [1, 1], [0, 0])

    def test_invalid_head_mapping_and_metadata_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "divisible"):
            ReplaySSMARLayout(1, 2, 2, 3, 16, 16, 4)
        layout = ReplaySSMARLayout(1, 2, 1, 2, 16, 16, 4)
        with self.assertRaisesRegex(ValueError, "outside allocated"):
            layout.validate_metadata([2], [0], [0])
        with self.assertRaisesRegex(ValueError, "lengths"):
            layout.validate_metadata([], [0], [0])

    def test_hardware_probe_isolates_each_npu_in_a_child(self):
        args = Namespace(tp=8, capacity=16)
        command = ar_child_command(args, 7, Path("/tmp/replayssm-level2/device7.json"))
        self.assertEqual(command[command.index("--child-device") + 1], "7")
        self.assertEqual(command[command.index("--tp") + 1], "8")
        self.assertEqual(command[command.index("--capacity") + 1], "16")

    def test_diagnostic_probes_are_independently_assignable(self):
        self.assertEqual(len(PROBES), len(set(PROBES)))
        for device, probe in enumerate(PROBES):
            command = probe_child_command(probe, device, Path(f"/tmp/{probe}.json"))
            self.assertEqual(command[command.index("--child-probe") + 1], probe)
            self.assertEqual(command[command.index("--child-device") + 1], str(device))

        self.assertEqual(len(MASKED_DOT_PROBES), len(set(MASKED_DOT_PROBES)))
        for device, probe in enumerate(MASKED_DOT_PROBES):
            command = masked_dot_child_command(probe, device, Path(f"/tmp/{probe}.json"))
            self.assertEqual(command[command.index("--child-probe") + 1], probe)
            self.assertEqual(command[command.index("--child-device") + 1], str(device))

    def test_production_kernel_avoids_known_cumsum_trap(self):
        source = (
            Path(__file__).resolve().parents[3] / "vllm_ascend" / "ops" / "triton" / "fla" / "replayssm_ar.py"
        ).read_text()
        self.assertNotIn("tl." + "cumsum(", source)
        self.assertIn("get_element(b_g_all", source)


if __name__ == "__main__":
    unittest.main()
