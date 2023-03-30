import unittest

from trees.utils import has_overlap, cut_overlaps, breaks_box


class HasOverlapTestCase(unittest.TestCase):

    def test_1d_true(self):
        ps1 = ([0],[5])
        ps2 = ([1],[6])

        self.assertTrue(has_overlap(ps1, ps2))

    def test_1d_false(self):
        ps1 = ([0],[5])
        ps2 = ([5],[6])

        self.assertFalse(has_overlap(ps1, ps2))

    def test_1d_true_domination(self):
        ps1 = ([0],[5])
        ps2 = ([2],[5])

        self.assertTrue(has_overlap(ps1, ps2))

    def test_2d_true(self):
        ps1 = ([0,0],[5,5])
        ps2 = ([2,2],[7,7])

        self.assertTrue(has_overlap(ps1, ps2))

    def test_2d_false(self):
        ps1 = ([0,5],[5,7])
        ps2 = ([2,0],[5,5])

        self.assertFalse(has_overlap(ps1, ps2))

    def test_4d_false(self):
        ps1 = ([0, 0, 5, 120], [109, 5, 14, 128])
        ps2 = ([0, 0, 0, 128], [54, 5, 14, 131])

        self.assertFalse(has_overlap(ps1, ps2))


class CutOverlapsTestCase(unittest.TestCase):
    def test_1d_max_overlap(self):
        ps_main = ([0], [5])
        ps_to_cut = ([3], [7])

        res = cut_overlaps(ps_main, ps_to_cut)
        self.assertEqual(res, ([5], [7]))
        self.assertNotEqual(res, ps_to_cut)

    def test_1d_min_overlap(self):
        ps_main = ([5], [10])
        ps_to_cut = ([3], [7])

        res = cut_overlaps(ps_main, ps_to_cut)
        self.assertEqual(res, ([3], [5]))
        self.assertNotEqual(res, ps_to_cut)

    def test_1d_no_overlap(self):
        ps_main = ([0], [5])
        ps_to_cut = ([5], [7])

        res = cut_overlaps(ps_main, ps_to_cut)
        self.assertEqual(res, ps_to_cut)

    def test_1d_total_domination(self):
        ps_main = ([0], [5])
        ps_to_cut = ([0], [5])

        res = cut_overlaps(ps_main, ps_to_cut)
        self.assertEqual(res, ps_to_cut)

    def test_2d_max_overlap_in_1_dim(self):
        ps_main = ([0, 0], [5, 5])
        ps_to_cut = ([3, 0], [7, 5])

        res = cut_overlaps(ps_main, ps_to_cut)
        self.assertEqual(res, ([5, 0], [7, 5]))
        self.assertNotEqual(res, ps_to_cut)

    def test_2d_assertion_error_when_multiple_cuts(self):
        ps_main = ([0, 0], [5, 3])
        ps_to_cut = ([3, 2], [7, 5])
        self.assertRaises(AssertionError, cut_overlaps, ps_main, ps_to_cut)

    # def test_4d(self):
    #     ps_main = ([0, 0, 5, 120], [109, 5, 14, 128])
    #     ps_to_cut = ([0, 0, 0, 128], [54, 5, 14, 131])

    #     res = cut_overlaps(ps_main, ps_to_cut)
    #     # self.assertEqual(res, ([5, 4], [7, 5]))
    #     self.assertEqual(res, ps_to_cut)


class BreaksBoxTestCase(unittest.TestCase):
    def test_1d_true(self):
        box = ([0], [5])
        breaker = ([1], [4])
        self.assertTrue(breaks_box(box, breaker))

    def test_1d_false_min_aligned(self):
        box = ([0], [5])
        breaker = ([0], [4])
        self.assertFalse(breaks_box(box, breaker))

    def test_2d_true_upper_right(self):
        box = ([3,0], [7,5])
        breaker = ([0,0], [5,3])
        self.assertTrue(breaks_box(box, breaker))

    def test_2d_true_upper_left(self):
        box = ([3,0], [7,5])
        breaker = ([5,0], [9,3])
        self.assertTrue(breaks_box(box, breaker))

    def test_2d_true_lower_left(self):
        box = ([3,0], [7,5])
        breaker = ([5,3], [9,7])
        self.assertTrue(breaks_box(box, breaker))

    def test_2d_true_lower_right(self):
        box = ([3,0], [7,5])
        breaker = ([0,3], [5,7])
        self.assertTrue(breaks_box(box, breaker))

    def test_2d_false_no_overlap(self):
        box = ([3,0], [7,5])
        breaker = ([0,0], [3,3])
        self.assertFalse(breaks_box(box, breaker))

    def test_2d_false_adjacent_max(self):
        box = ([3,0], [7,5])
        breaker = ([7,0], [9,3])
        self.assertFalse(breaks_box(box, breaker))

    def test_2d_false_adjacent_min(self):
        box = ([3,0], [7,5])
        breaker = ([0,0], [3,3])
        self.assertFalse(breaks_box(box, breaker))

    def test_3d_true(self):
        box = ([1,0,1], [2,2,3])
        breaker = ([1,0,0], [3,1,2])
        self.assertTrue(breaks_box(box, breaker))
