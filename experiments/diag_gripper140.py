"""단계 1 진단: robosuite Robotiq 2F-140 MuJoCo 모델을 받은 그대로 쓸 수 있는가.

집게를 직교 3축 이송대(위치 구동 x·y·z)에 아래를 보게 달고, 온장(190 x 90 x 57 mm, 2,000 kg/m3, 약 1.95 kg)을
위에서 집어 0.3 m 들고 수평으로 0.5 m 빠르게 옮긴 뒤 내려놓는다. 팔 IK와 섞이지 않게 집게만 시험한다.

조건
  as_received : 집게 xml을 그대로(단위만 robosuite처럼 radian으로 컴파일, robosuite base.xml의 impratio 20·elliptic 원뿔)
  exclude_self: 위 + 집게 몸체끼리 접촉 제외(진단용. 이것이 필요하면 '받은 그대로 동작하지 않음'에 해당)

실행: MUJOCO_GL=egl venv/bin/python -m experiments.diag_gripper140 [--out outputs/wall3/diag_gripper140]
"""
import argparse, json, subprocess
from pathlib import Path

import mujoco
import numpy as np

SIM = Path(__file__).resolve().parents[1]
GRIPPER_XML = SIM / "third_party/robosuite_gripper140/robotiq_gripper_140.xml"
L, W, H, RHO = 0.190, 0.090, 0.057, 2000.0
TS = 0.001


def build(exclude_self: bool, with_brick: bool = True):
    s = mujoco.MjSpec()
    s.option.timestep = TS
    s.option.impratio = 20
    s.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    wb = s.worldbody
    wb.add_light(pos=[0, -1, 2], dir=[0, 0.5, -1])
    wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[2, 2, 0.05], friction=[0.6, 0.005, 0.0001])
    car = wb.add_body(name="carrier", pos=[0, 0, 0.6])
    car.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.02, 0.02, 0.02], mass=2.0, contype=0, conaffinity=0)
    for ax, name in ((0, "cx"), (1, "cy"), (2, "cz")):
        a = [0, 0, 0]; a[ax] = 1
        car.add_joint(name=name, type=mujoco.mjtJoint.mjJNT_SLIDE, axis=a, damping=0)
        act = s.add_actuator(name=name, target=name, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        act.set_to_position(kp=50000, kv=3000)
    mount = car.add_site(name="mount", pos=[0, 0, 0], quat=[0, 1, 0, 0])    # gripper +z (approach) = world -z
    gr = mujoco.MjSpec.from_file(str(GRIPPER_XML))
    gr.compiler.degree = False                       # robosuite base.xml: <compiler angle="radian">
    gr.compiler.inertiagrouprange = [0, 0]           # robosuite base.xml
    mount.attach_body(gr.body("right_gripper"), "g_", "")
    if exclude_self:
        names = [b.name for b in s.bodies if b.name.startswith("g_")]
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                e = s.add_exclude(); e.bodyname1 = a; e.bodyname2 = b
    if with_brick:
        br = wb.add_body(name="brick", pos=[0, 0, H / 2])
        br.add_freejoint(name="brick_free")
        br.add_geom(name="brick", type=mujoco.mjtGeom.mjGEOM_BOX, size=[L / 2, W / 2, H / 2],
                    mass=L * W * H * RHO, friction=[0.6, 0.005, 0.0001], rgba=[0.6, 0.6, 0.58, 1])
    return s.compile()


def minjerk(a, b, n):
    t = np.linspace(0, 1, n)[:, None]
    return a + (b - a) * (10 * t**3 - 15 * t**4 + 6 * t**5)


def run(exclude_self: bool, with_brick: bool = True, move_time: float = 0.7):
    m = build(exclude_self, with_brick)
    d = mujoco.MjData(m)
    lp = lambda: d.geom("g_left_fingerpad_collision").xpos.copy()
    rp = lambda: d.geom("g_right_fingerpad_collision").xpos.copy()
    fj = lambda: float(d.joint("g_finger_joint").qpos[0])
    rk = lambda: float(d.joint("g_right_outer_knuckle_joint").qpos[0])
    carrier = lambda: d.body("carrier").xpos.copy()
    brick = lambda: d.body("brick").xpos.copy() if with_brick else np.zeros(3)
    log = {}

    def hold(target, grip, sec):
        n = int(sec / TS)
        for _ in range(n):
            d.ctrl[:3] = target; d.ctrl[3:5] = [grip, -grip]
            mujoco.mj_step(m, d)

    def move(a, b, grip, sec, rec=None):
        n = int(sec / TS)
        for p in minjerk(np.asarray(a, float), np.asarray(b, float), n):
            d.ctrl[:3] = p; d.ctrl[3:5] = [grip, -grip]
            mujoco.mj_step(m, d)
            if rec is not None:
                rec.append(brick() - carrier())

    # carrier joint position = displacement from body pos (0,0,0.6); open, then measure pad height vs carrier
    start = np.array([0.0, 0.0, 0.0])
    hold(start, 0.0, 1.0)
    pads_open = np.linalg.norm(lp() - rp())
    pad_drop = carrier()[2] - (lp()[2] + rp()[2]) / 2          # carrier-to-pad-centre distance (m)
    log["open_pad_distance_mm"] = pads_open * 1000
    log["carrier_to_pad_m"] = pad_drop
    # descend so that pad centres sit 8 mm above the brick mid-height
    z_grasp = (H / 2 + 0.008 + pad_drop) - 0.6
    move(start, [0, 0, z_grasp], 0.0, 1.0)
    hold([0, 0, z_grasp], 0.0, 0.3)
    hold([0, 0, z_grasp], 0.7, 1.0)                           # close
    log["closed_finger_joint"] = fj(); log["closed_right_knuckle"] = rk()
    log["closed_pad_distance_mm"] = np.linalg.norm(lp() - rp()) * 1000
    rel0 = brick() - carrier()
    up = [0, 0, z_grasp + 0.3]
    rec = []
    move([0, 0, z_grasp], up, 0.7, 1.0, rec)
    hold(up, 0.7, 0.3)
    log["lift_brick_z_m"] = float(brick()[2]) if with_brick else None
    if with_brick:
        log["slip_after_lift_mm"] = ((brick() - carrier() - rel0) * 1000).round(2).tolist()   # x, y, z
    rel1 = brick() - carrier()
    over = [0.5, 0, z_grasp + 0.3]
    move(up, over, 0.7, move_time, rec)
    hold(over, 0.7, 0.5)
    rec = np.array(rec)
    if with_brick:
        drift = np.linalg.norm(rec - rel0, axis=1)
        log["carry_slip_max_mm"] = float(drift.max() * 1000)
        log["carry_slip_end_mm"] = float(drift[-1] * 1000)
        log["slip_during_move_mm"] = ((brick() - carrier() - rel1) * 1000).round(2).tolist()
        q = d.body("brick").xquat
        log["carry_tilt_deg"] = float(np.degrees(np.arccos(np.clip(1 - 2 * (q[1] ** 2 + q[2] ** 2), -1, 1))))
        log["pad_contacts"] = sorted({(m.geom(c.geom1).name, m.geom(c.geom2).name) for c in d.contact[:d.ncon]
                                      if "brick" in (m.geom(c.geom1).name, m.geom(c.geom2).name)})
    down = [0.5, 0, z_grasp + 0.002]
    move(over, down, 0.7, 1.0)
    hold(down, 0.0, 0.8)
    move(down, [0.5, 0, z_grasp + 0.15], 0.0, 0.5)
    if with_brick:
        b = brick(); q = d.body("brick").xquat
        tilt = np.degrees(np.arccos(np.clip(1 - 2 * (q[1] ** 2 + q[2] ** 2), -1, 1)))
        log["placed_xy_err_mm"] = float(np.hypot(b[0] - 0.5, b[1]) * 1000)
        log["placed_z_m"] = float(b[2]); log["placed_tilt_deg"] = float(tilt)
    log["max_tendon_violation_rad"] = float(np.abs(d.ten_length).max())
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(SIM / "outputs/wall3/diag_gripper140_v2"))
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} 이미 있음: 덮어쓰지 않음")
    out.mkdir(parents=True)
    res = {"gripper_xml_sha256": subprocess.check_output(["sha256sum", str(GRIPPER_XML)], text=True).split()[0],
           "git_commit": subprocess.check_output(["git", "-C", str(SIM), "rev-parse", "HEAD"], text=True).strip(),
           "brick_mass_kg": L * W * H * RHO}
    for cond, ex in (("as_received", False), ("exclude_self", True)):
        res[cond] = {"brick": run(ex, True), "empty_close": run(ex, False)}
    (out / "result.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
