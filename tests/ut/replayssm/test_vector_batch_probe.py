# SPDX-License-Identifier: Apache-2.0
import argparse
import sys
import unittest
from types import ModuleType
from unittest.mock import patch

import torch
import torch.distributed.tensor  # noqa: F401 - load before patch.dict snapshots sys.modules

from experimental.replayssm_cube_910b import run
from experimental.replayssm_cube_910b.plot import contract
from scripts.replayssm.profile_sd_operator_compare import native_state_page


class TestVectorBatchProbe(unittest.TestCase):
    def test_sweeps_every_batch_for_both_geometries(self):
        for geometry, tp in (("27b", 4), ("9b", 1)):
            args = run.parse_args(["--suite", "vector", "--geometry", geometry, "--output", "unused.json"])
            self.assertEqual(args.batches, tuple(range(1, 129)))
            self.assertEqual(args.path, ("native", "v20", "compact_v1", "vector_pair"))
            self.assertEqual(args.tp, tp)
        for invalid in ("0,1", "129", "5,5", "abc"):
            with self.assertRaises(argparse.ArgumentTypeError):
                run.batches(invalid)

    def test_native_9b_state_at_bs128(self):
        state, page_bytes = native_state_page(torch.device("meta"), 1, 128, 4, geometry="9b")
        self.assertEqual(tuple(state.shape), (513, 32, 128, 128))
        self.assertEqual(state.stride(0) * state.element_size(), page_bytes)
        self.assertGreater(state.storage_offset(), 0)
        self.assertLessEqual(state.storage_offset() + state[0].numel(), state.stride(0))

    def test_vector_trial_bypasses_cube_and_uses_both_head_geometries(self):
        for geometry, key_heads, value_heads in (("27b", 4, 12), ("9b", 16, 32)):
            with self.subTest(geometry=geometry):
                self.run_stubbed_trial(geometry, key_heads, value_heads)

    def run_stubbed_trial(self, geometry, key_heads, value_heads):
        # Only host wiring is stubbed. The actual NPU numerics and graph timing run on-board.
        args = run.parse_args(
            ["--suite", "vector", "--geometry", geometry, "--pos", "0", "--flush", "mixed", "--output", "unused.json"]
        )
        native_module = ModuleType("vllm_ascend.ops.gdn")
        replay_module = ModuleType("vllm_ascend.ops.replayssm.ascendc")

        def native(q, k, v, state, **kwargs):
            self.assertEqual(q.shape, (8, key_heads, 128))
            self.assertEqual(state.shape, (9, value_heads, 128, 128))
            self.assertEqual(kwargs["beta"].dtype, torch.bfloat16)
            return torch.zeros_like(v)

        def verify_call(q, k, v, g, beta, checkpoint, d, kc, gc, slots, pos, base, lengths, out, status, *tail):
            self.assertEqual(beta.dtype, torch.float32)
            self.assertEqual(status.numel(), 2 * value_heads)
            self.assertEqual(tail[-2], 20)
            torch.testing.assert_close(slots, tail[-1] + 4)
            out.zero_()
            status.zero_()

        def compact_call(checkpoint, d, k, g, slots, pos, base, flush, status, tile, cores, variant, ring_slots):
            self.assertEqual((tile, cores, variant), (32, 40, 1))
            self.assertEqual(status.numel(), 2 * value_heads * 4)
            self.assertEqual(flush.tolist(), [1, 0])
            torch.testing.assert_close(slots, ring_slots + 4)
            status.zero_()

        def measure_stub(call, reset, iterations, label):
            reset()
            call()
            self.assertIn(geometry, label)
            return {"p50_us": 1.0}

        native_module.recurrent_gated_delta_rule = native
        replay_module.replayssm_sd_verify = verify_call
        replay_module.replayssm_sd_compact = compact_call
        with (
            patch.dict(sys.modules, {native_module.__name__: native_module, replay_module.__name__: replay_module}),
            patch.object(run, "measure", measure_stub),
            patch.object(run, "verify", side_effect=AssertionError("Cube demo entered")),
            patch.object(run, "panel", side_effect=AssertionError("Cube panel prepared")),
        ):
            rows = run.trial(args, 2, torch.device("cpu"))
        self.assertEqual([r["path"] for r in rows], list(run.VECTOR_PATHS))
        for row in rows:
            self.assertEqual((row["key_heads"], row["value_heads"], row["active_flush"]), (key_heads, value_heads, 1))
        v20 = rows[1]
        self.assertEqual(contract(v20)[0] // value_heads, contract({**v20, "value_heads": 12})[0] // 12)


if __name__ == "__main__":
    unittest.main()
