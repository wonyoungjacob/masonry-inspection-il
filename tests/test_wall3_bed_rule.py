"""모르타르 판 까는 규칙 시험 (2026-10-30, main_v1_eval_bed): 1단이 삐뚤어도 판이 깔리는지, 옛 버전은 그대로인지.
실행: MUJOCO_GL=egl PYTHONPATH=. venv/bin/python -m unittest discover -s tests -p "test_wall3_bed_rule.py" -v
"""
import hashlib, json, unittest

import mujoco
import numpy as np

from wall3.config import scene_config, RETRACT_Q, H, JOINT
from wall3.scene import WallEnv, yaw_quat

HASH = {"main_v1": "10909a503935beef2b26ed12088d2a0e6d7383d55923407590cc4cebbe6eb864",       # before 2026-10-30
        "main_v1_eval": "e656246dbd0ef361730a3e5156973d319ff95ebb09ca8c4fa1104844b49605cd",  # = eval_gates_v2_20261026
        "pilot_20261016": "ef1ad719851f471eff97f13ad49930c07dd65b820fdc021a992b7a55ed90a6e1"}
SEED = 700300
COURSE1 = range(5)


def put(env, i, x, y, z, yaw=0.0):
    a, v = env.brick_qadr[i], env.brick_dofadr[i]
    env.data.qpos[a:a + 3] = [x, y, z]
    env.data.qpos[a + 3:a + 7] = yaw_quat(yaw)
    env.data.qvel[v:v + 6] = 0.0


def run(env, n):
    for _ in range(n):
        env.step(np.array(RETRACT_Q), env.cfg.gripper.open_ctrl)


def course1(env, shift=None):
    """Course 1 on its design slots; shift = {brick: (dx, dy, dz)} in m."""
    for i in COURSE1:
        s = env.design[i]
        dx, dy, dz = (shift or {}).get(i, (0.0, 0.0, 0.0))
        put(env, i, s["x"] + dx, s["y"] + dy, s["z"] + dz)
    mujoco.mj_forward(env.model, env.data)


def fresh(version):
    env = WallEnv(scene_config(version))
    env.reset(SEED, arm_q=np.array(RETRACT_Q))
    return env


class TestBedRule(unittest.TestCase):
    def test_old_versions_unchanged(self):
        for v, h in HASH.items():
            got = hashlib.sha256(json.dumps(scene_config(v).to_dict(), sort_keys=True, default=float).encode()).hexdigest()
            self.assertEqual(got, h, v)
        self.assertEqual(scene_config("main_v1_eval").bed_rule, "in_slot")
        self.assertEqual(scene_config("main_v1_eval_bed").bed_rule, "placed")
        self.assertEqual(scene_config("main_v1_eval_bed").cure_time_s, 7200.0)
        with self.assertRaises(ValueError):
            scene_config("main_v1", bed_rule="x")

    def test_in_slot_rule(self):
        env = fresh("main_v1_eval")
        course1(env)
        run(env, 45)
        self.assertGreaterEqual(env.bed_laid[1], 0, "straight course 1: bed laid (old rule)")
        env.reset(SEED, arm_q=np.array(RETRACT_Q))
        course1(env, {3: (0.0, 0.015, 0.0)})
        run(env, 80)
        self.assertEqual(env.bed_laid[1], -1, "brick 4 off by 15 mm (across the wall): no bed under the old rule")
        env.close()

    def test_placed_rule_crooked(self):
        env = fresh("main_v1_eval_bed")
        course1(env, {3: (0.0, 0.015, 0.0), 1: (-0.004, -0.012, 0.0)})
        run(env, 45)
        self.assertGreaterEqual(env.bed_laid[1], 0, "crooked course 1: bed laid")
        before = env.brick_poses()
        p, q = env.brick_pose(3)
        slab = env.data.mocap_pos[env.bed_mocap[env.beds[1]["slabs"][3]]]
        self.assertLess(np.abs(slab - (p + [0, 0, H / 2 + JOINT / 2])).max(), 1e-6, "slab on the actual top of brick 4")
        run(env, 20)
        self.assertLess(np.abs(env.brick_poses()[:, :3] - before[:, :3]).max(), 1e-3, "nothing moved when the bed appeared")
        env.close()

    def test_placed_rule_perched(self):
        env = fresh("main_v1_eval_bed")
        course1(env)
        s, s2 = env.design[3], env.design[2]
        put(env, 3, s2["x"] + s2["L"] / 2 + s["L"] / 2 - 0.02, s["y"], s["z"] + H + 0.002)   # brick 4 dropped onto brick 3's end
        mujoco.mj_forward(env.model, env.data)
        run(env, 140)
        p, q = env.brick_pose(3)
        tilt = np.degrees(np.arccos(np.clip(1 - 2 * (q[1] ** 2 + q[2] ** 2), -1, 1)))
        self.assertGreater(tilt, 10.0, "brick 4 perched (tilted)")
        self.assertFalse(env.in_slot(3))
        self.assertGreaterEqual(env.bed_laid[1], 0, "perched course 1: bed laid")
        before = env.brick_poses()
        run(env, 20)
        self.assertLess(np.abs(env.brick_poses()[:, :3] - before[:, :3]).max(), 1e-3)
        env.close()

    def test_placed_rule_waits_for_course(self):
        env = fresh("main_v1_eval_bed")
        course1(env)
        x, y = env.supply[4][0][:2]
        put(env, 4, x, y, env.cfg.table_h + H / 2)                  # brick 5 still on the supply table
        mujoco.mj_forward(env.model, env.data)
        self.assertTrue(env.on_table(4))
        run(env, 80)
        self.assertEqual(env.bed_laid[1], -1, "course 1 not finished: no bed")
        env.close()

    def test_placed_rule_upper_brick_already_there(self):
        env = fresh("main_v1_eval_bed")
        course1(env, {3: (0.0, 0.015, 0.0)})
        s = env.design[6]
        put(env, 6, s["x"], s["y"], H + H / 2 + 0.0005)              # course-2 brick put on the bare course 1
        mujoco.mj_forward(env.model, env.data)
        run(env, 80)
        self.assertEqual(env.bed_laid[1], -1, "the bed would sink into brick 7: not laid")
        self.assertGreater(env.bed_blocked[1], 0)
        self.assertLess(abs(env.brick_pose(6)[0][2] - 1.5 * H), 0.002, "brick 7 not thrown up")
        env.close()


if __name__ == "__main__":
    unittest.main()
