"""held_grasp 기준 자리 판별 (2026-11-03, 우분투지시_검측정보진단_20261103.md 2.5-3).

G1 순간 손(TCP) xy에서 가장 가까운 공급칸 = 손에 든 벽돌의 자리. 공급칸 중심 ± 10 mm · ± 10° 흔들림(P6), 손이 벽돌 길이 방향으로
집을 수 있는 범위(검측이 받아들이는 |along| <= L/2 - 15 mm)에서 항상 맞는 칸을 고르고, 10 cm 밖이면 판별 불가인지 본다.
실행: PYTHONPATH=. venv/bin/python -m unittest discover -s tests -p "test_wall3_*.py" -v
"""
import itertools
import unittest

import numpy as np

from wall3.config import SUPPLY_SLOTS, design_slots
from wall3.gates import GateConfig, nearest_supply_slot

MAX_D = GateConfig().held_grasp_max
D = design_slots()


def hand_xy(i, dx, dy, yaw_deg, along):
    x, y = SUPPLY_SLOTS[i]
    yaw = np.deg2rad(yaw_deg)
    return np.array([x + dx + along * np.cos(yaw), y + dy + along * np.sin(yaw)])


class GateTargetTest(unittest.TestCase):
    def test_defaults_are_the_old_gates(self):
        c = GateConfig()
        self.assertEqual((c.pose_source, c.target, c.held_grasp_max), ("camera", "turn", 0.10))

    def test_unknown_values_rejected(self):
        for bad in (dict(pose_source="cam"), dict(target="held")):
            with self.assertRaises(ValueError):
                GateConfig(**bad)

    def test_grid_scatter_and_grasp_positions(self):
        n = 0
        for i in range(len(SUPPLY_SLOTS)):
            a_max = D[i]["L"] / 2 - 0.015
            for dx, dy, yd, a in itertools.product((-0.01, 0.0, 0.01), (-0.01, 0.0, 0.01), (-10.0, 0.0, 10.0), (-a_max, 0.0, a_max)):
                slot, d = nearest_supply_slot(hand_xy(i, dx, dy, yd, a), SUPPLY_SLOTS, MAX_D)
                self.assertEqual(slot, i, (i, dx, dy, yd, a, d))
                self.assertLessEqual(d, MAX_D)
                n += 1
        self.assertEqual(n, 16 * 81)

    def test_random_scatter(self):
        rng = np.random.default_rng(0)
        for _ in range(20000):
            i = int(rng.integers(16))
            a_max = D[i]["L"] / 2 - 0.015
            xy = hand_xy(i, rng.uniform(-0.01, 0.01), rng.uniform(-0.01, 0.01), rng.uniform(-10, 10), rng.uniform(-a_max, a_max))
            self.assertEqual(nearest_supply_slot(xy, SUPPLY_SLOTS, MAX_D)[0], i)

    def test_far_is_undetermined(self):
        for x, y in SUPPLY_SLOTS:
            for ang in np.linspace(0, 2 * np.pi, 24, endpoint=False):
                slot, d = nearest_supply_slot(np.array([x, y]) + 0.105 * np.array([np.cos(ang), np.sin(ang)]), SUPPLY_SLOTS, MAX_D)
                if d > MAX_D:
                    self.assertIsNone(slot)
        for s in D:                                                # wall slots: far from every supply slot
            self.assertIsNone(nearest_supply_slot((s["x"], s["y"]), SUPPLY_SLOTS, MAX_D)[0])
        self.assertIsNone(nearest_supply_slot((0.0, 0.40), SUPPLY_SLOTS, MAX_D)[0])     # retract pose
        slot, d = nearest_supply_slot((-0.91 - 0.105, -0.25), SUPPLY_SLOTS, MAX_D)       # 10.5 cm off the corner slot
        self.assertIsNone(slot)
        self.assertAlmostEqual(d, 0.105)


if __name__ == "__main__":
    unittest.main()
