"""나4 G1 시험: 발동 조건(닫고 0.3 s 뒤 1.5 cm 상승, 닫을 때마다 한 번), 판정(문턱 없으면 오류), 정상 집기와 빈손의 하중·각도.
실행: MUJOCO_GL=egl PYTHONPATH=. venv/bin/python -m unittest discover -s tests -p "test_wall3_*.py" -v
"""
import unittest

from wall3.g1 import G1Config, G1Trigger
from wall3.test_g1 import trial


class TestG1(unittest.TestCase):
    def test_trigger(self):
        t = G1Trigger(G1Config(), 0.05)
        fired = [t.update(0.3, 0.50 - 0.001 * i) for i in range(10)]           # closed, still going down
        self.assertFalse(any(fired))
        fired = [t.update(0.3, 0.491 + 0.004 * i) for i in range(10)]          # lifting
        self.assertEqual(sum(fired), 1)
        self.assertTrue(fired[4])                                               # 0.491 + 0.016 >= z_min 0.491 + 0.015
        self.assertFalse(t.update(0.3, 0.6))                                    # once per closing
        t.update(0.0, 0.6)                                                      # opened: re-armed
        self.assertFalse(t.update(0.3, 0.6))

    def test_decide_needs_thresholds(self):
        from wall3.g1 import G1
        g = G1.__new__(G1)
        g.cfg = G1Config()
        with self.assertRaises(ValueError):
            g.decide({"load_N": 19.0, "angle_rad": 0.24})
        g.cfg = G1Config(thr_load=4.5, thr_angle=0.47)
        self.assertTrue(g.decide({"load_N": 19.0, "angle_rad": 0.24}))
        self.assertFalse(g.decide({"load_N": 0.0, "angle_rad": 0.70}))

    def test_grasp_vs_empty(self):
        n, e = trial(("main_v1", "normal", 0)), trial(("main_v1", "empty", 0))
        self.assertAlmostEqual(n["load_N"], 19.12, delta=0.3)
        self.assertLess(abs(e["load_N"]), 1.0)
        self.assertLess(n["angle_rad"], 0.3)
        self.assertGreater(e["angle_rad"], 0.6)
        self.assertLess(n["slip_during_angle_check_mm"], 1.0)


if __name__ == "__main__":
    unittest.main()
