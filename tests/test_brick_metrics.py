"""벽돌 기준 집계 · 공급 따라가기 진단 도구 점검 (2026-11-17). 결과 파일을 읽지 않는 계산 부분만 본다."""
import unittest

import numpy as np

from wall3.brick_metrics import boot, CONDS, PAIRS
from experiments.diag_follow_supply_20261117 import expected_if_following, first_takeover


class TestBoot(unittest.TestCase):
    def test_same_seed_same_interval(self):
        x = np.r_[np.full(60, 100.0), np.linspace(0, 93.75, 40)]
        self.assertEqual(boot(x), boot(x))

    def test_interval_brackets_mean(self):
        x = np.r_[np.full(60, 100.0), np.linspace(0, 93.75, 40)]
        lo, hi = boot(x)
        self.assertLess(lo, x.mean())
        self.assertGreater(hi, x.mean())

    def test_constant_has_zero_width(self):
        self.assertEqual(boot(np.full(100, 100.0)), (100.0, 100.0))

    def test_pairs_name_existing_conditions(self):
        names = {n for n, _ in CONDS}
        self.assertEqual(len(names), len(CONDS))
        for h, c in PAIRS:
            self.assertIn(h, names)
            self.assertIn(c, names)


class TestFollow(unittest.TestCase):
    def test_expected_change_grows_with_shift(self):
        dn = np.full(1000, 20.0)
        self.assertEqual(expected_if_following(dn, 0.0), 0.0)
        self.assertGreater(expected_if_following(dn, 30.0), 10.0)      # a hand that follows a 30 mm move changes the distance a lot

    def test_first_takeover_keeps_first_try_only(self):
        w = {"bricks": [{"gates": {"g0": [{"try": 1, "supply_slot": 3, "tcp_off_mm": 12.0}, {"try": 2, "supply_slot": 3, "tcp_off_mm": 40.0}]}},
                        {"gates": {"g0": [{"try": 1, "supply_slot": 5, "tcp_off_mm": 25.0}]}}, {"gates": {}}]}
        self.assertEqual(first_takeover(w), {3: 12.0, 5: 25.0})


if __name__ == "__main__":
    unittest.main()
