import unittest

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
