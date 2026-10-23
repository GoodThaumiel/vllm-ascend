# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""CPU checks for the native/Replay SD profiler's state layout."""

import importlib.util
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "profile_sd_operator_compare", ROOT / "scripts/replayssm/profile_sd_operator_compare.py"
)
PROFILE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROFILE)
from lab import make_case  # noqa: E402


class SDProfileSetupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_native_token_slots_are_page_strided(self):
        for tp, batch, length in ((4, 2, 4), (8, 32, 1)):
            with self.subTest(tp=tp, batch=batch, length=length):
                state, page_bytes = PROFILE.native_state_page(torch.device("cpu"), tp, batch, length)
                self.assertEqual(state.shape, (batch * length + 1, 48 // tp, 128, 128))
                self.assertEqual(state.stride(0) * 2, page_bytes)
                self.assertEqual(page_bytes % (256 * 1024), 0)
                self.assertFalse(state.is_contiguous())

    def test_native_start_matches_committed_history_recurrence(self):
        case = make_case(tp=8, batch=2, position=7, length=4, base=29, seed=31)
        actual = PROFILE.reconstruct_native_start(case, 7)
        state = case["checkpoint"][1:].double()
        for i in range(7):
            physical = (29 + i) & 31
            decay = case["g_cache"][1:, :, physical].double().exp()
            delta = case["d_cache"][1:, :, physical].double()
            key = case["k_cache"][1:, :, physical].double().repeat_interleave(3, dim=1)
            state = decay[..., None, None] * state + delta[..., None] * key[..., None, :]
        torch.testing.assert_close(actual.float(), state.bfloat16().float(), atol=0.002, rtol=0.01)


if __name__ == "__main__":
    unittest.main()
