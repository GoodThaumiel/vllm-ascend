# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""CPU algebra and ring boundary checks for v13-derived SD ablations.

These do not execute AscendC instructions; device tests remain mandatory.
"""

import unittest

import torch


class SDVerifyAblationReferenceTests(unittest.TestCase):
    def test_history_row_accumulation_matches_dense(self):
        torch.manual_seed(48)
        history = torch.randn(16, 128) * 0.1
        dots = torch.randn(16) * 0.1
        suffix = torch.rand(16)
        base = torch.randn(128) * 0.1
        for pos in (0, 1, 7, 15, 16):
            actual = base.clone()
            for i in range(pos):
                actual += history[i] * (dots[i] * suffix[i])
            expected = base + (history[:pos] * (dots * suffix)[:pos, None]).sum(dim=0)
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)

    def test_speculative_suffix_and_current_term(self):
        torch.manual_seed(49)
        alpha = torch.rand(4)
        dots = torch.randn(4, 4)
        previous_d = torch.randn(4, 128)
        for length in (1, 2, 3, 4):
            for s in range(length):
                suffix = torch.ones(4)
                product = 1.0
                for j in range(s - 1, -1, -1):
                    suffix[j] = product
                    product *= float(alpha[j])
                result = torch.zeros(128)
                for j in range(s - 1, -1, -1):
                    result += previous_d[j] * dots[s, j] * suffix[j]
                expected = sum(
                    (previous_d[j] * dots[s, j] * alpha[j + 1 : s].prod() for j in range(s)),
                    torch.zeros(128),
                )
                torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-5)
                torch.testing.assert_close(previous_d[s] * dots[s, s], previous_d[s] * dots[s][s])

    def test_paired_matvec_tiles_match_full_matvec(self):
        torch.manual_seed(50)
        state = torch.randn(128, 128)
        q, k = torch.randn(128), torch.randn(128)
        out_q, out_k = [], []
        for start in range(0, 128, 16):
            block = state[start : start + 16]
            out_q.append((block[:, :64] * q[:64] + block[:, 64:] * q[64:]).sum(dim=1))
            out_k.append((block[:, :64] * k[:64] + block[:, 64:] * k[64:]).sum(dim=1))
        torch.testing.assert_close(torch.cat(out_q), state @ q, atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(torch.cat(out_k), state @ k, atol=1e-5, rtol=1e-4)

    def test_batched_gate_write_preserves_other_ring_entries(self):
        for base in (0, 28, 29, 30, 31):
            for pos in (0, 7, 15, 16):
                for length in (1, 2, 3, 4):
                    expected = torch.arange(32, dtype=torch.float32)
                    actual = expected.clone()
                    values = torch.arange(length, dtype=torch.float32) + 100
                    for i in range(length):
                        expected[(base + pos + i) % 32] = values[i]
                    start = (base + pos) % 32
                    first = min(length, 32 - start)
                    actual[start : start + first] = values[:first]
                    actual[: length - first] = values[first:]
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
