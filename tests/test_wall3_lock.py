"""나2 결정 시험 (쥔 동안 고정): N_min 상수, 버전 분리, 정상 잡기는 고정되고 빈손·한쪽 패드 잡기는 고정되지 않음, 눌러 박으면 'force' 해제.
실행: MUJOCO_GL=egl PYTHONPATH=. venv/bin/python -m unittest discover -s tests -p "test_wall3_*.py" -v
"""
import unittest

import numpy as np

from wall3.config import scene_config
from wall3.expert import expert_config
from wall3.scene import WallEnv
from wall3.test_lock import trial


class TestGripLock(unittest.TestCase):
    def test_nmin(self):
        env = WallEnv(scene_config("main_v1"))
        env.reset(700000)
        g, c = 9.81, env.cfg
        for i, b in enumerate(env.brick_bid):
            m = env.model.body_mass[b]
            self.assertAlmostEqual(env.lock_nmin[i], m * np.sqrt(g * g + c.lock_amax ** 2) / (2 * c.lock_mu))
        full = [n for n, s in zip(env.lock_nmin, env.design) if not s["half"]]
        half = [n for n, s in zip(env.lock_nmin, env.design) if s["half"]]
        self.assertTrue(19 < min(full) < 21 and 9 < min(half) < 11)
        env.close()

    def test_versions(self):
        self.assertFalse(scene_config("pilot_20261016").grip_lock)
        self.assertTrue(scene_config("main_v1").grip_lock)
        self.assertEqual(expert_config("pilot_20261016").t_close, 0.45)
        self.assertFalse(expert_config("pilot_20261016").tcp_same_pass)

    def test_grasps(self):
        r = {m: trial((m, 700000, 0)) for m in ("normal", "empty", "one_pad", "overload")}
        self.assertTrue(r["normal"]["locked"] and not r["normal"]["releases"])
        self.assertTrue(all(e.get("jump_mm", 0) <= 0.1 for e in r["normal"]["events"]))
        self.assertFalse(r["empty"]["locked"])
        self.assertFalse(r["one_pad"]["locked"])
        self.assertIn("force", r["overload"]["releases"])


if __name__ == "__main__":
    unittest.main()
