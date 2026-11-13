import unittest
from argparse import Namespace

import torch

from experimental.rgdr_v310_ablation.bench import make_inputs
from experimental.rgdr_v310_ablation.task_mapping import assigned_tasks


class TestTaskMapping(unittest.TestCase):
    def test_uniform_mtp3_and_mtp7_batch32(self):
        for tokens in (4, 8):
            for cores in (1, 7, 8):
                with self.subTest(tokens=tokens, cores=cores):
                    assigned = assigned_tasks([tokens] * 32, 16, cores)
                    flattened = [task for core in assigned for task in core]
                    expected = [(batch, head, batch * tokens) for batch in range(32) for head in range(16)]
                    self.assertCountEqual(flattened, expected)
                    self.assertEqual(len(flattened), len(set(flattened)))

    def test_mixed_and_zero_lengths(self):
        lengths = [4, 0, 1, 8, 2, 0, 7]
        for cores in (1, 2, 7, 9):
            with self.subTest(cores=cores):
                flattened = [task for core in assigned_tasks(lengths, 16, cores) for task in core]
                expected = []
                seq0 = 0
                for batch, length in enumerate(lengths):
                    if length:
                        expected.extend((batch, head, seq0) for head in range(16))
                    seq0 += length
                self.assertCountEqual(flattened, expected)

    def test_reject_bad_geometry(self):
        for args in (([4], 0, 7), ([4], 16, 0), ([-1], 16, 7)):
            with self.assertRaises(ValueError):
                assigned_tasks(*args)


class TestPackedInputs(unittest.TestCase):
    def test_ragged_mtp1_to_7_uses_same_native_slots(self):
        for mtp in range(1, 8):
            with self.subTest(mtp=mtp):
                args = Namespace(mtp=mtp, batch=3, ragged=True, fp16_gate_input=True, variant="native")
                native = make_inputs(args, torch.device("cpu"))
                args.variant = "cube-grouped"
                grouped = make_inputs(args, torch.device("cpu"))
                self.assertEqual(grouped["slot_table"].shape, (args.batch, mtp + 1))
                for row, length in enumerate(native["lengths"].tolist()):
                    start, end = native["offsets"][row : row + 2].tolist()
                    torch.testing.assert_close(grouped["slot_table"][row, :length], native["indices"][start:end])
                    self.assertTrue(1 <= int(grouped["accepted"][row]) <= length)
                torch.testing.assert_close(native["state"], grouped["state"], atol=0, rtol=0)
                torch.testing.assert_close(native["g"], grouped["g"].float(), atol=0, rtol=0)
