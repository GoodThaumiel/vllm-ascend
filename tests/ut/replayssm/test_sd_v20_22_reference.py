# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""CPU checks for request metadata and key-head reuse algebra."""

import unittest

import torch

from experimental.replayssm_sd_ascendc.lab import clone_case, make_case, pack_metadata_, recurrent_golden


def grouped_key_reference(case):
    """Compute shared raw key dots once; keep each value head's gates/state separate."""
    result = clone_case(case)
    result["out"].zero_()
    result["status"].zero_()
    heads = case["q"].shape[2]
    value_heads = case["v"].shape[2]
    group_size = value_heads // heads
    ring = case["d_cache"].shape[2]
    for row in range(case["q"].shape[0]):
        slot = int(case["slots"][row])
        if slot == 0:
            continue
        pos, base, length = (int(case[name][row]) for name in ("positions", "bases", "lengths"))
        for key_head in range(heads):
            hist_indices = [(base + i) % ring for i in range(pos)]
            hist_k = case["k_cache"][slot, key_head, hist_indices].double()
            q = case["q"][row, :length, key_head].double() * 128**-0.5
            k = case["k"][row, :length, key_head].double()
            hist_q = hist_k @ q.T
            hist_kdot = hist_k @ k.T
            spec_q = k @ q.T
            spec_k = k @ k.T
            for member in range(group_size):
                head = key_head * group_size + member
                state = case["checkpoint"][slot, head].double()
                history_d = case["d_cache"][slot, head, hist_indices].double()
                history_a = case["g_cache"][slot, head, hist_indices].double().exp()
                alpha = case["g"][row, :length, head].double().exp()
                beta = case["beta"][row, :length, head].double()
                spec_d = []
                for s in range(length):
                    decay = history_a.prod() * alpha[:s].prod()
                    prev_q = decay * (state @ q[s])
                    prev_k = decay * (state @ k[s])
                    for i in range(pos):
                        suffix = history_a[i + 1 :].prod() * alpha[:s].prod()
                        prev_q += suffix * history_d[i] * hist_q[i, s]
                        prev_k += suffix * history_d[i] * hist_kdot[i, s]
                    for j in range(s):
                        suffix = alpha[j + 1 : s].prod()
                        prev_q += suffix * spec_d[j] * spec_q[j, s]
                        prev_k += suffix * spec_d[j] * spec_k[j, s]
                    d = beta[s] * (case["v"][row, s, head].double() - alpha[s] * prev_k)
                    out = alpha[s] * prev_q + d * spec_q[s, s]
                    spec_d.append(d)
                    phys = (base + pos + s) % ring
                    result["out"][row, s, head] = out.bfloat16()
                    result["d_cache"][slot, head, phys] = d.half()
                    result["k_cache"][slot, key_head, phys] = k[s].half()
                    result["g_cache"][slot, head, phys] = case["g"][row, s, head]
    return result


class SD20To22Tests(unittest.TestCase):
    def test_metadata_pack_is_dynamic_and_uses_independent_ring(self):
        case = {
            "slots": torch.tensor([11, 12, 0], dtype=torch.int32),
            "positions": torch.tensor([15, 7, 0], dtype=torch.int32),
            "bases": torch.tensor([16, 31, 0], dtype=torch.int32),
            "lengths": torch.tensor([4, 1, 0], dtype=torch.int32),
        }
        ring = torch.tensor([1, 2, 0], dtype=torch.int32)
        packed = torch.zeros((3, 8), dtype=torch.int32)
        pack_metadata_(case, packed, ring)
        self.assertEqual(packed.tolist(), [[11, 1, 15, 16, 4, 0, 0, 0], [12, 2, 7, 31, 1, 0, 0, 0], [0] * 8])
        case["positions"][0] = 0
        case["lengths"][0] = 1
        ring[0] = 3
        pack_metadata_(case, packed, ring)
        self.assertEqual(packed[0].tolist(), [11, 3, 0, 16, 1, 0, 0, 0])

    def test_grouped_task_mapping_covers_every_head_once(self):
        for batch in (1, 2, 8, 16, 32, 64):
            blocks = min(40, batch * 4)
            seen = []
            for block in range(blocks):
                for task in range(block, batch * 4, blocks):
                    row, key_head = divmod(task, 4)
                    seen.extend((row, key_head * 3 + member) for member in range(3))
            self.assertEqual(len(seen), batch * 12)
            self.assertEqual(set(seen), {(row, head) for row in range(batch) for head in range(12)})

    def test_grouped_raw_dots_match_dense_for_distinct_head_gates(self):
        for position, length, base in ((0, 1, 0), (7, 2, 29), (15, 4, 16), (16, 4, 31)):
            case = make_case(tp=4, batch=3, position=position, length=length, base=base, tile_v=128, seed=71)
            # Mixed length and a padding row while the other heads have independent gates.
            case["lengths"][0] = 1
            case["slots"][1] = 0
            case["positions"][1] = 0
            case["bases"][1] = 0
            case["lengths"][1] = 0
            expected = recurrent_golden(case)
            actual = grouped_key_reference(case)
            for name in ("out", "d_cache", "k_cache", "g_cache", "checkpoint"):
                torch.testing.assert_close(actual[name], expected[name], rtol=0.003, atol=1e-5, equal_nan=True)
            torch.testing.assert_close(actual["status"], expected["status"], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
