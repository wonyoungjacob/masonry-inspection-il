import unittest
import numpy as np
from wall3.volume import box, union_iou


class VolumeTests(unittest.TestCase):
    def test_translation_has_analytic_iou(self):
        a = box(np.zeros(3), np.eye(3), np.array([.045, .021, .0135]))
        b = box(np.array([.03, 0, 0]), np.eye(3), np.array([.045, .021, .0135]))
        self.assertAlmostEqual(union_iou([a], [b]), .5)

    def test_pitch_is_not_ignored(self):
        half = np.array([.045, .021, .0135])
        a = box(np.zeros(3), np.eye(3), half)
        rotation = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
        b = box(np.zeros(3), rotation, half)
        intersection = .027 * .027 * .042
        volume = .09 * .042 * .027
        self.assertAlmostEqual(union_iou([a], [b]), intersection / (2 * volume - intersection))

    def test_union_does_not_double_count(self):
        a = box(np.zeros(3), np.eye(3), np.ones(3))
        self.assertAlmostEqual(union_iou([a, a], [a]), 1.)

    def test_disjoint_boxes(self):
        a = box(np.zeros(3), np.eye(3), np.ones(3))
        b = box(np.ones(3) * 10, np.eye(3), np.ones(3))
        self.assertEqual(union_iou([a], [b]), 0.)
