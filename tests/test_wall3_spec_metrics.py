"""평가지표 md 6절 시험: 설계 자세 그대로면 ①=10 mm(1단 아랫면 0), ②③④=0, ⑤=0, IoU=1, 시방 합격.
벽돌 하나를 3°·4° 돌리거나 5·7 mm 앞뒤로 밀면 2.8절(②③④ 6 mm 문언 적용)의 걸리는 크기와 맞는지 확인한다.
2026-11-03: 기대값을 2026-10-27 기준 변경(c29b74c, 6 mm 문언 적용)에 맞춤(사용자 결정). 구 C안 하한 시험은 mode="scaled"로.
실행: PYTHONPATH=. venv/bin/python -m unittest discover -s tests -p "test_wall3_*.py" -v
"""
import unittest

import numpy as np

from wall3.config import design_slots, W, H, WALL_Y
from wall3.spec_metrics import evaluate

DESIGN = design_slots()
LEN = [s["L"] for s in DESIGN]
YF = WALL_Y + W / 2


def design_poses():
    return np.array([[s["x"], s["y"], s["z"], 1.0, 0.0, 0.0, 0.0] for s in DESIGN])


def yaw_q(a):
    return [np.cos(a / 2), 0, 0, np.sin(a / 2)]


def run(poses):
    return evaluate(poses, LEN, DESIGN, W, H, front_y=YF)


def rot_z(poses, k, deg):
    p = poses.copy(); p[k, 3:7] = yaw_q(np.deg2rad(deg)); return p


def shift_y(poses, k, mm):
    p = poses.copy(); p[k, 1] -= mm / 1000.0        # −y world = towards the robot = backwards from the front face
    return p


class DesignPose(unittest.TestCase):
    def test_design_is_perfect(self):
        r = run(design_poses())
        self.assertTrue(r["exists"])
        self.assertAlmostEqual(r["1_bed_joint"]["upper_min_mm"], 10.0, places=6)
        self.assertAlmostEqual(r["1_bed_joint"]["upper_max_mm"], 10.0, places=6)
        self.assertAlmostEqual(r["1_bed_joint"]["base_min_mm"], 0.0, places=6)
        self.assertAlmostEqual(r["1_bed_joint"]["base_max_mm"], 0.0, places=6)
        for key in ("2_line_plane", "3_plumb", "4_position"):
            self.assertAlmostEqual(r[key]["value_mm"], 0.0, places=6, msg=key)
            self.assertAlmostEqual(r[key]["tol_mm"], 6.0, places=6, msg=key)
        self.assertAlmostEqual(r["5_thickness"]["min_mm"], 0.0, places=6)
        self.assertAlmostEqual(r["5_thickness"]["max_mm"], 0.0, places=6)
        self.assertAlmostEqual(r["iou_mean"], 1.0, places=9)
        self.assertAlmostEqual(r["iou_min"], 1.0, places=9)
        self.assertTrue(r["spec_pass"])

    def test_brick_order_does_not_matter(self):
        p = design_poses()
        perm = list(range(16)); perm[1], perm[3] = perm[3], perm[1]; perm[5], perm[10] = perm[10], perm[5]
        r = evaluate(p[perm], [LEN[i] for i in perm], DESIGN, W, H, front_y=YF)
        self.assertTrue(r["spec_pass"])
        self.assertAlmostEqual(r["iou_min"], 1.0, places=9)


class Rotation(unittest.TestCase):
    """2.8절: 수평 회전 θ는 ②면 · ③(2·3단) · ④(1단)에서 95·sin θ ≥ 6 mm, 약 3.6°부터 걸린다."""

    @staticmethod
    def corner_shift(deg):
        """Exact front-corner displacement 95·sin θ + 45·(1 − cos θ) mm (2.8절 95·sin θ is its approximation)."""
        a = np.deg2rad(deg)
        return 95 * np.sin(a) + 45 * (1 - np.cos(a))

    def check(self, k, deg, key):
        r = run(rot_z(design_poses(), k, deg))
        return r, r[key]["value_mm"]

    def test_course2_rotation(self):
        for deg, expect_pass in ((3.0, True), (4.0, False)):
            r, v = self.check(7, deg, "3_plumb")
            self.assertAlmostEqual(v, self.corner_shift(deg), delta=1e-6)
            self.assertAlmostEqual(v, 95 * np.sin(np.deg2rad(deg)), delta=0.15)      # 45·(1 − cos 4°) = 0.11 mm
            self.assertEqual(r["spec_pass"], expect_pass, f"{deg} deg")
        r = run(rot_z(design_poses(), 7, 4.0))
        self.assertFalse(r["3_plumb"]["pass"])

    def test_course1_rotation(self):
        for deg, expect_pass in ((3.0, True), (4.0, False)):
            r, v = self.check(2, deg, "4_position")
            self.assertAlmostEqual(v, self.corner_shift(deg), delta=1e-6)
            self.assertAlmostEqual(v, 95 * np.sin(np.deg2rad(deg)), delta=0.15)
            self.assertEqual(r["4_position"]["pass"], expect_pass)
            self.assertEqual(r["spec_pass"], expect_pass)

    def test_threshold_near_3_6_deg(self):
        """Exact corner shift reaches 6 mm at about 3.57° (5.88 mm at 3.5°, 6.14 mm at 3.65°)."""
        self.assertTrue(run(rot_z(design_poses(), 7, 3.5))["spec_pass"])
        self.assertFalse(run(rot_z(design_poses(), 7, 3.65))["spec_pass"])

    def test_rotation_iou_matches_table(self):
        for deg, iou in ((1.0, 0.978), (2.0, 0.957)):
            r = run(rot_z(design_poses(), 7, deg))
            self.assertAlmostEqual(r["iou"][7], iou, delta=0.002)


class Shift(unittest.TestCase):
    """2.8절: 앞뒤 밀림 s는 ③(2·3단) · ④(1단)에서 s ≈ 6 mm, ②면에서 s ≈ 12 mm(띠 반폭)부터 걸린다."""

    def test_course2_shift(self):
        r = run(shift_y(design_poses(), 7, 5.0))
        self.assertAlmostEqual(r["3_plumb"]["value_mm"], 5.0, delta=1e-6)
        self.assertTrue(r["3_plumb"]["pass"])
        self.assertTrue(r["spec_pass"])
        self.assertAlmostEqual(r["2_line_plane"]["plane_mm"], 2.5, delta=0.3)   # ~s/2
        r = run(shift_y(design_poses(), 7, 7.0))
        self.assertAlmostEqual(r["3_plumb"]["value_mm"], 7.0, delta=1e-6)
        self.assertFalse(r["3_plumb"]["pass"])
        self.assertFalse(r["spec_pass"])
        self.assertAlmostEqual(r["2_line_plane"]["plane_mm"], 3.5, delta=0.3)   # ~s/2, under 6 mm
        self.assertTrue(r["2_line_plane"]["pass"])
        self.assertAlmostEqual(r["5_thickness"]["max_mm"], 7.0, delta=1e-6)

    def test_course1_shift(self):
        r = run(shift_y(design_poses(), 2, 5.0))
        self.assertAlmostEqual(r["4_position"]["value_mm"], 5.0, delta=1e-6)
        self.assertTrue(r["4_position"]["pass"])
        r = run(shift_y(design_poses(), 2, 7.0))
        self.assertAlmostEqual(r["4_position"]["value_mm"], 7.0, delta=1e-6)
        self.assertFalse(r["4_position"]["pass"])
        self.assertFalse(r["spec_pass"])

    def test_iou_shift_width(self):
        r = run(shift_y(design_poses(), 7, 1.0))
        self.assertAlmostEqual(r["iou"][7], 89 / 91, places=6)


class OtherItems(unittest.TestCase):
    def test_tilt_along_length_hits_bed_joint(self):
        """2.8절: 길이 방향 기울기 φ는 ①(≥ 3 mm → 1.8°), ②수평선(95·sin φ ≥ 6 → 3.6°)."""
        def tilted(deg):
            p = design_poses()
            a = np.deg2rad(deg)                                   # about the y axis
            p[7, 3:7] = [np.cos(a / 2), 0, np.sin(a / 2), 0]
            return run(p)
        r = tilted(2.0)
        self.assertFalse(r["1_bed_joint"]["pass"])
        self.assertTrue(r["2_line_plane"]["pass"])
        r = tilted(4.0)
        self.assertFalse(r["1_bed_joint"]["pass"])
        self.assertFalse(r["2_line_plane"]["pass"])

    def test_raised_brick_level(self):
        p = design_poses(); p[7, 2] += 0.004                      # z +4 mm: ② band half-width 2 mm, ① 14 mm under it fails
        r = run(p)
        self.assertAlmostEqual(r["2_line_plane"]["level_mm"]["2"], 2.0, delta=1e-6)
        self.assertFalse(r["1_bed_joint"]["pass"])

    def test_missing_brick(self):
        p = design_poses(); p[13, :3] = [2.0, -2.0, 0.0285]
        r = run(p)
        self.assertFalse(r["exists"])
        self.assertFalse(r["slots_filled"][13])
        self.assertEqual(r["iou"][13], 0.0)
        self.assertFalse(r["spec_pass"])

    def test_base_penetration_counts_as_zero(self):
        p = design_poses(); p[:5, 2] -= 0.0002
        r = run(p)
        self.assertAlmostEqual(r["1_bed_joint"]["base_min_raw_mm"], -0.2, places=6)
        self.assertTrue(r["1_bed_joint"]["pass"])

    def test_floor_sensitivity(self):
        """floor_tol is the lower bound of the old C안 (mode="scaled"); the literal mode (default) does not use it."""
        p = shift_y(design_poses(), 7, 2.5)
        self.assertFalse(evaluate(p, LEN, DESIGN, W, H, front_y=YF, floor_tol=0.002, mode="scaled")["spec_pass"])
        self.assertTrue(evaluate(p, LEN, DESIGN, W, H, front_y=YF, floor_tol=0.003, mode="scaled")["spec_pass"])
        self.assertTrue(evaluate(p, LEN, DESIGN, W, H, front_y=YF, floor_tol=0.002)["spec_pass"])


if __name__ == "__main__":
    unittest.main()
