# SPDX-License-Identifier: Apache-2.0
import unittest

from .reference import cases, prepare_reference


class PrepareReferenceTest(unittest.TestCase):
    def check(self, case, k=7, anchor=False):
        return prepare_reference(*cases()[case], k, 32, 248077, anchor)

    def test_group_specific_pages_and_padding(self):
        query, context = self.check("mixed")
        self.assertEqual(query[0][0][2], list(range(180, 188)))
        self.assertEqual(query[1][0][2], [52, 53, 54, 55, 72, 73, 74, 75])
        self.assertEqual(context[0][1][11:16], [179, -1, -1, -1, -1])
        self.assertEqual(context[1][1][7:9], [47, 48])
        self.assertEqual(query[1][2][2], [-1] * 8)
        self.assertEqual(query[1][2][5][:4], [16, 16, 0, 0])

    def test_fully_rejected_restarts_at_own_position_and_caps_sequence(self):
        query, _ = self.check("mixed")
        ids, positions, slots, _, _, info = query[1][1]
        self.assertEqual(ids[0], 104)
        self.assertEqual(positions, [28, 29, 30, 31, 31, 31, 31, 31])
        self.assertEqual(slots, [116, 117, 118, 119] + [-1] * 4)
        self.assertEqual(info[:4], [8, 16, 32, 0])

    def test_context_extent_is_independent_of_query_extent(self):
        query, context = self.check("long_context")
        self.assertEqual(query[0][0][1][0], 20)
        self.assertEqual(context[0][0][:19], list(range(1, 20)))
        self.assertEqual(context[0][1][19:], [-1] * 5)
        self.assertEqual(context[1][0], context[0][0])
        self.assertEqual(context[1][1][:7], [-1] * 7)
        self.assertEqual(query[1][0][5][3], 0)

    def test_all_widths_and_both_sampling_conventions(self):
        for k in range(1, 8):
            for anchor in (False, True):
                with self.subTest(k=k, anchor=anchor):
                    query, _ = self.check("mixed", k, anchor)
                    self.assertEqual(query[0][0][3][:k], list(range(0 if anchor else 1, k + int(not anchor))))
                    self.assertEqual(query[0][0][4][:k], list(range(21, 21 + k)))
                    self.assertEqual(query[0][0][2][k + 1 :], [-1] * (7 - k))

    def test_invalid_page_and_span_do_not_produce_slots(self):
        query, context = self.check("invalid")
        self.assertEqual(query[1][0][5][3], 2)
        self.assertEqual(query[1][1][5][3], 1)
        self.assertEqual(query[1][0][2][:3], [-1] * 3)
        self.assertEqual(context[1][1], [-1] * 24)


if __name__ == "__main__":
    unittest.main()
