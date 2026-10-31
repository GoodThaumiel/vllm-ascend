# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Keep the ReplaySSM SD host guard aligned with the AscendC UB capacity."""

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


class SDHeadCapacityContractTests(unittest.TestCase):
    def test_verify_binding_accepts_kernel_head_capacity_for_every_variant(self):
        binding = (ROOT / "csrc/replayssm_sd.cpp").read_text()
        kernel = (ROOT / "csrc/kernels/replayssm_sd_verify.cpp").read_text()
        host_limit = int(re.search(r"constexpr int64_t MAX_VALUE_HEADS = (\d+);", binding).group(1))
        kernel_limit = int(re.search(r"constexpr uint32_t MAX_VALUE_HEADS = (\d+);", kernel).group(1))
        self.assertEqual(host_limit, kernel_limit)
        self.assertIn("value_heads <= MAX_VALUE_HEADS", binding)
        self.assertNotRegex(binding, r"variant\s*[><=]+\s*\d+\s*\|\|\s*value_heads\s*<=\s*\d+")


if __name__ == "__main__":
    unittest.main()
