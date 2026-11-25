"""현실 제약 ablation 도구 점검 (2026-11-16): 오차 뽑기 · 믿는 도면 · 손목 카메라 측정(G0v).

MuJoCo EGL 렌더가 필요하다(측정 시험). 시작값 700300(개발 배치, 동작 확인만).
"""
import os
import unittest

import numpy as np

from wall3.realism import make_perturb, belief_design, B_PALLET_MM


class TestPerturb(unittest.TestCase):
    def test_same_wall_same_values(self):
        a, b = make_perturb(700300, True, True), make_perturb(700300, True, True)
        self.assertEqual(a.to_dict(), b.to_dict())
        self.assertNotEqual(a.to_dict(), make_perturb(700301, True, True).to_dict())

    def test_flags(self):
        ab, a, b, n = (make_perturb(700310, x, y) for x, y in ((1, 1), (1, 0), (0, 1), (0, 0)))
        self.assertEqual(n.pallet, (0.0, 0.0))
        self.assertEqual(n.wall_dx, 0.0)
        self.assertEqual(a.pallet, (0.0, 0.0))
        self.assertEqual(b.wall_dx, 0.0)
        self.assertEqual(ab.wall_dx, a.wall_dx)                 # the A values do not depend on B
        self.assertEqual(ab.pallet, b.pallet)
        self.assertAlmostEqual(float(np.hypot(*b.pallet)) * 1000, B_PALLET_MM, places=6)
        self.assertLessEqual(abs(a.wall_dx), 0.003)
        self.assertLessEqual(abs(np.degrees(a.wall_dth)), 0.3)

    def test_belief_design_rigid(self):
        from wall3.config import design_slots
        d = design_slots()
        self.assertEqual(belief_design(d, make_perturb(700300, False, False)), d)
        bd = belief_design(d, make_perturb(700300, True, False))
        for i, j in ((0, 15), (3, 8), (5, 10)):
            a = np.hypot(d[i]["x"] - d[j]["x"], d[i]["y"] - d[j]["y"])
            b = np.hypot(bd[i]["x"] - bd[j]["x"], bd[i]["y"] - bd[j]["y"])
            self.assertAlmostEqual(a, b, places=9)


class TestWristCamera(unittest.TestCase):
    def test_g0v_measures_supply_brick(self):
        os.environ.setdefault("MUJOCO_GL", "egl")
        from wall3.config import scene_config, RETRACT_Q, H
        from wall3.scene import WallEnv
        from wall3.expert import WallExpert
        from wall3.ik import down_R
        from wall3.realism import NoisyEstimator, shift_pallet
        env = WallEnv(scene_config("main_v1_eval"))
        env.reset(700300, arm_q=np.array(RETRACT_Q))
        p = make_perturb(700300, True, True)
        shift_pallet(env, p)
        ex = WallExpert(env)
        ex.q_cmd, ex.grip_cmd, ex.fail, ex.steps, ex.touch, ex.brick_k, ex.in_hand, ex.on_step = env.arm_q().copy(), 0.0, None, 0, [], 0, None, None
        ex._watch = lambda h, ph: None
        cam = NoisyEstimator(env, "wrist", 480, 640, z_min=env.cfg.table_h + 0.02)
        cam.set_noise(p.wrist_t, p.wrist_r, p.depth_wrist, p.depth_pix, np.random.default_rng(0))
        pos, _ = env.supply_pose(0)                              # drawing (the pallet moved 30 mm)
        q = ex._solve(np.r_[pos[:2], pos[2] + ex.dzp + ex.cfg.approach], down_R(0.0), ex.q_ret, ex.q_ret[0])
        ex._joint_move(q, 0.0, None, "t"); ex._hold(5, 0.0, None, "h")
        dets = [d for d in cam.detect(force=True) if abs(d["z_top"] - (env.cfg.table_h + H)) < 0.015]
        self.assertTrue(dets)
        tcp = env.tcp()[0]
        best = min((cam.pose_of(d, L=env.design[0]["L"], yaw_prior=0.0) for d in dets), key=lambda pp: np.hypot(*(pp["pos"][:2] - tcp[:2])))
        err = float(np.hypot(*(best["pos"][:2] - env.brick_pose(0)[0][:2])))
        self.assertLess(err, 0.005, err)                          # within 5 mm with the A errors (audit)
        cam.close(); env.close()


if __name__ == "__main__":
    unittest.main()
