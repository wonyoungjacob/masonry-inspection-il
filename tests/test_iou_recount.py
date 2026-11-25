"""벽돌별 IoU 재집계 도구 점검 (2026-11-25). 결과 파일을 읽지 않는 계산 부분만 본다."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from wall3.iou_recount import CONDS, PAIRS, SEEDS, boot, load, per_wall, summarize, threshold_check, thresholds


def wall(seed, filled, iou, spec=False):
    return {"seed": seed, "eval": {"slots_filled": filled, "iou": iou, "spec_pass": spec}}


HALF = [False] * 5 + [True] + [False] * 4 + [True] + [False] * 5


class TestThresholds(unittest.TestCase):
    def test_half_slots_get_half_threshold(self):
        t = thresholds(HALF, "A")
        self.assertEqual(t[5], 0.702)
        self.assertEqual(t[0], 0.749)
        self.assertEqual(thresholds(HALF, "B")[10], 0.953)

    def test_threshold_check_matches_hand_values(self):
        r = threshold_check()
        self.assertAlmostEqual(r["온장 이동 (6, 6, 3) mm"], 184 * 84 * 54 / (2 * 190 * 90 * 57 - 184 * 84 * 54), places=6)
        self.assertAlmostEqual(r["반장 이동 (0, 6, 0) mm"], 84 / 96, places=6)


class TestSummary(unittest.TestCase):
    def test_pass_counts_and_empty_slots(self):
        iou = [0.80] * 16
        iou[5] = 0.71          # half: passes A (0.702), fails 0.749
        iou[0] = 0.0           # empty
        filled = [True] * 16
        filled[0] = False
        walls = {s: wall(s, filled, iou) for s in SEEDS}
        r, v = summarize(walls, HALF)
        self.assertEqual(r["bricks_filled"], 1500)
        self.assertEqual(r["pass_A"], 1500)
        self.assertEqual(r["pass_B"], 0)
        self.assertEqual(r["pass_A_half"], 200)
        self.assertAlmostEqual(r["iou_mean_1600"], (0.80 * 14 + 0.71) / 16)
        self.assertEqual(r["walls_complete"], 0)

    def test_nonzero_iou_on_empty_slot_is_rejected(self):
        with self.assertRaises(ValueError):
            per_wall(wall(1, [False] * 16, [0.5] + [0.0] * 15), HALF)

    def test_load_rejects_duplicate_and_missing_seeds(self):
        with tempfile.TemporaryDirectory() as d:
            w = [wall(s, [True] * 16, [0.9] * 16) for s in SEEDS[:60]]
            Path(d, "p0").mkdir(); Path(d, "p1").mkdir()
            Path(d, "p0", "rollout.json").write_text(json.dumps({"walls": w}))
            Path(d, "p1", "rollout.json").write_text(json.dumps({"walls": w[:5]}))
            with self.assertRaises(ValueError):
                load(f"{d}/p*/rollout.json")            # duplicate seeds
            Path(d, "p1", "rollout.json").write_text(json.dumps({"walls": []}))
            with self.assertRaises(ValueError):
                load(f"{d}/p*/rollout.json")            # only 60 seeds


class TestBootPairs(unittest.TestCase):
    def test_boot_constant_and_determinism(self):
        self.assertEqual(boot(np.full(100, 2.0)), (2.0, 2.0))
        x = np.linspace(0, 16, 100)
        self.assertEqual(boot(x), boot(x))

    def test_pairs_name_existing_conditions(self):
        names = [c[0] for c in CONDS]
        self.assertEqual(len(set(names)), len(names))
        for h, c, _ in PAIRS:
            self.assertIn(h, names)
            self.assertIn(c, names)
        self.assertEqual(sum(k.startswith("설계") for _, _, k in PAIRS), 6)
        self.assertEqual(sum(k.startswith("역방향") for _, _, k in PAIRS), 6)


class TestDiag(unittest.TestCase):
    def test_spearman_monotone(self):
        from experiments.diag_iou_A_20261125 import spearman
        x = np.arange(10.0)
        self.assertAlmostEqual(spearman(x, x ** 3), 1.0)
        self.assertAlmostEqual(spearman(x, -x), -1.0)


if __name__ == "__main__":
    unittest.main()
