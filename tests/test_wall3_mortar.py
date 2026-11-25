"""나3 모르타르 강성 공식 시험: 임피던스 함수, 벽돌 밑 모르타르 판 수, 버전별 강성(파일럿은 옛 공식).
실행: MUJOCO_GL=egl PYTHONPATH=. venv/bin/python -m unittest discover -s tests -p "test_wall3_*.py" -v
"""
import unittest

from wall3.config import scene_config
from wall3.scene import WallEnv, impedance, mortar_solref


class TestMortar(unittest.TestCase):
    def test_impedance(self):
        self.assertAlmostEqual(impedance(0.0), 0.95)
        self.assertAlmostEqual(impedance(0.0005), 0.97)
        self.assertAlmostEqual(impedance(0.002), 0.99)

    def test_slabs_and_pairs(self):
        env = WallEnv(scene_config("main_v1", mortar_stiffness="impedance"))      # 나3 formula (reverted in main_v1 by 나5)
        n = [env._slabs_under(i) for i in range(env.n)]
        for i, s in enumerate(env.design):
            self.assertEqual(n[i], 0 if s["course"] == 1 else (1 if s["half"] else 3))
        m = env.model
        m_full = max(m.body_mass[b] for b in env.brick_bid)
        for p in range(m.npair):
            i = int(m.geom(m.pair_geom1[p]).name.split("_")[1])
            k, _ = mortar_solref(env.cfg, m.body_mass[env.brick_bid[i]] / m_full, 4 * n[i])
            self.assertAlmostEqual(-m.pair_solref[p, 0] / k, 1.0, places=6)
        env.close()

    def test_pilot_unchanged(self):
        c = scene_config("pilot_20261016")
        self.assertEqual(c.mortar_stiffness, "brick_sim")
        env = WallEnv(c)
        self.assertAlmostEqual(-env.model.pair_solref[0, 0], 9.81 / c.mortar_sag)
        env.close()


if __name__ == "__main__":
    unittest.main()
