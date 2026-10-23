# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
import unittest

import torch

from experimental.replayssm_sd_cursor_ascendc.lab import make_case, reference_step


class CursorReferenceTests(unittest.TestCase):
    def test_partial_accept_reject_and_padding(self):
        case = make_case()
        case["write_pos"][1:5] = torch.tensor([0, 7, 12, 9], dtype=torch.int32)
        case["accepted"][:] = torch.tensor([4, 2, 1, 0], dtype=torch.int32)
        case["slots"][3] = 0
        result = reference_step(case)
        self.assertEqual(result["write_pos"][:5].tolist(), [0, 4, 9, 13, 9])
        self.assertEqual(result["is_flush"][3].item(), 1)
        self.assertEqual(result["status"].tolist(), [0, 0, 0, 0])

    def test_flush_commit_wrap_and_reset_priority(self):
        case = make_case()
        case["write_pos"][1:5] = torch.tensor([16, 12, 8, 9], dtype=torch.int32)
        case["cache_base"][1:5] = torch.tensor([29, 30, 31, 27], dtype=torch.int32)
        case["is_flush"][1:5] = torch.tensor([1, 1, 0, 1], dtype=torch.int8)
        case["accepted"][:] = torch.tensor([2, 4, 3, -1], dtype=torch.int32)
        case["do_reset"][3] = 1
        result = reference_step(case)
        self.assertEqual(result["cache_base"][1:5].tolist(), [13, 10, 31, 0])
        self.assertEqual(result["write_pos"][1:5].tolist(), [2, 4, 11, 0])
        self.assertEqual(result["is_flush"][1:5].tolist(), [0, 0, 0, 0])

    def test_error_codes_do_not_mutate_cursor(self):
        case = make_case()
        case["slots"][:] = torch.tensor([-1, 2, 3, 4], dtype=torch.int32)
        case["accepted"][:] = torch.tensor([1, 5, 1, 4], dtype=torch.int32)
        case["cache_base"][3] = -1
        case["write_pos"][4] = 19
        original = {name: case[name].clone() for name in ("write_pos", "cache_base", "is_flush")}
        result = reference_step(case)
        self.assertEqual(result["status"].tolist(), [1, 2, 3, 4])
        for name, expected in original.items():
            torch.testing.assert_close(result[name], expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
