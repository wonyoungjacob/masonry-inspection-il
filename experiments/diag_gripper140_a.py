"""단계 1 재개(선택지 A): 조립 조정(①②③)을 적용한 2F-140 집게 시험.

집게를 직교 3축 이송대(위치 구동)에 아래로 달고, 벽돌을 위에서 집어 0.3 m 들기(1 s) → 수평 0.5 m 옮기기(0.7 s) →
내려놓기. 온장(190×90×57 mm)과 반장(90×90×57 mm), 밀도 2,000 kg/m³, 벽돌 마찰 0.6(패드 기본 1.0).
③ 집는 높이: 닫힌 패드는 손가락 끝 쪽 모서리가 안쪽으로 약 9° 들어가 있어 그 모서리가 벽돌에 닿는다. 그 모서리를
벽돌 옆면 가운데 높이(무게중심 높이)에 맞춘다.

통과 기준(지시서 단계 1 재개): 쥠/빈손 닫힘 각도 구분, 들기·옮기기 중 미끄러짐 ≤ 1 mm, 운반 중 기울기 ≤ 1°, 반장도 통과.

실행: MUJOCO_GL=egl venv/bin/python -m experiments.diag_gripper140_a --torque 2 4 6 8 --out outputs/wall3/<이름>
"""
import argparse, json, subprocess
from pathlib import Path

import mujoco
import numpy as np

from wall3 import gripper as G

SIM = Path(__file__).resolve().parents[1]
W, H, RHO, MU_BRICK = 0.090, 0.057, 2000.0, 0.6
BRICKS = {"full": 0.190, "half": 0.090}
TS = 0.001
TIP_Z = 0.2080        # pad tip-edge (contact) depth below the mount along the approach axis, measured at a 90 mm closure
CLEAR = 0.002         # A1+: gap between the deepest finger point and the brick bottom (floor / lower course)


def build(cfg: G.GripperConfig, length=None, impratio=20.0, noslip=0):
    """impratio/noslip: solver options. brick_sim/scene.py (지난 실험) uses impratio 10, noslip_iterations 3."""
    s = mujoco.MjSpec()
    s.option.timestep = TS
    s.option.impratio = impratio
    s.option.noslip_iterations = noslip
    s.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    wb = s.worldbody
    wb.add_light(pos=[0, -1, 2], dir=[0, 0.5, -1])
    wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[2, 2, 0.05], friction=[MU_BRICK, 0.005, 0.0001])
    car = wb.add_body(name="carrier", pos=[0, 0, 0.6])
    car.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.02, 0.02, 0.02], mass=2.0, contype=0, conaffinity=0)
    for ax, name in ((0, "cx"), (1, "cy"), (2, "cz")):
        a = [0, 0, 0]; a[ax] = 1
        car.add_joint(name=name, type=mujoco.mjtJoint.mjJNT_SLIDE, axis=a)
        s.add_actuator(name=name, target=name, trntype=mujoco.mjtTrn.mjTRN_JOINT).set_to_position(kp=50000, kv=3000)
    mount = car.add_site(name="mount", quat=[0, 1, 0, 0])          # approach (+z gripper) = world −z; fingers close along world y
    G.attach(mount, s, cfg)
    if length is not None:
        br = wb.add_body(name="brick", pos=[0, 0, H / 2])
        br.add_freejoint(name="brick_free")
        br.add_geom(name="brick", type=mujoco.mjtGeom.mjGEOM_BOX, size=[length / 2, W / 2, H / 2],
                    mass=length * W * H * RHO, friction=[MU_BRICK, 0.005, 0.0001], rgba=[0.6, 0.6, 0.58, 1])
    return s.compile()


def minjerk(a, b, n):
    t = np.linspace(0, 1, n)[:, None]
    return a + (b - a) * (10 * t**3 - 15 * t**4 + 6 * t**5)


def quat_angle(q1, q2):
    return float(np.degrees(2 * np.arccos(np.clip(abs(np.dot(q1, q2)), -1, 1))))


def pad_forces(m, d):
    """Normal force (N) on the brick from each finger (pad + fingertip geoms) and from anything else of the gripper."""
    bid = m.geom("brick").id
    f6 = np.zeros(6); out = {"left": 0.0, "right": 0.0, "other_gripper": 0.0, "floor": 0.0}
    for i in range(d.ncon):
        c = d.contact[i]
        if bid not in (c.geom1, c.geom2):
            continue
        other = m.geom(c.geom2 if c.geom1 == bid else c.geom1).name
        mujoco.mj_contactForce(m, d, i, f6)
        if other.startswith("g_left_finger"):
            out["left"] += f6[0]
        elif other.startswith("g_right_finger"):
            out["right"] += f6[0]
        elif other.startswith("g_"):
            out["other_gripper"] += f6[0]
        else:
            out["floor"] += f6[0]
    return out


def run(cfg: G.GripperConfig, kind=None, move_time=0.7, dz=0.0, impratio=20.0, noslip=0, rest_after_move=0.5):
    """dz: pad tip edge height above the brick mid-height (m), ③."""
    length = BRICKS[kind] if kind else None
    m = build(cfg, length, impratio, noslip)
    d = mujoco.MjData(m)
    carrier = lambda: d.body("carrier").xpos.copy()
    has = length is not None
    log = {"brick": kind or "none"}
    seq = {"lift": [], "move": []}

    state = {"close": False, "t0": 0.0, "c0": cfg.open_ctrl, "cmd": cfg.open_ctrl}

    def step(target, close):
        if close != state["close"]:                       # new order: ramp from the current command
            state.update(close=close, t0=d.time, c0=state["cmd"])
        c = G.ramp(cfg, close, d.time - state["t0"], state["c0"])
        state["cmd"] = c
        d.ctrl[:3] = target; d.ctrl[3:5] = [c, -c]
        mujoco.mj_step(m, d)

    def hold(target, close, sec):
        for _ in range(int(sec / TS)):
            step(target, close)

    def move(a, b, close, sec, tag=None):
        for p in minjerk(np.asarray(a, float), np.asarray(b, float), int(sec / TS)):
            step(p, close)
            if tag and has:
                seq[tag].append((d.body("brick").xpos - carrier()).copy())
                seq[tag + "_q"] = seq.get(tag + "_q", []) + [d.body("brick").xquat.copy()]

    hold([0, 0, 0], False, 0.8)
    lp, rp = d.geom("g_left_fingerpad_collision"), d.geom("g_right_fingerpad_collision")
    log["open_inner_gap_mm"] = float((rp.xpos[1] - lp.xpos[1] - 2 * m.geom_size[lp.id][2]) * 1000)
    # ③ pad tip edge (contact) at the brick mid-height
    if cfg.parallel_pads:
        # A1+ ③: parallel pads touch with their face; put the deepest finger point CLEAR above the brick bottom
        cal = G.calibrate(cfg, W)
        pad_c = (cal["deepest_finger_depth"] - cal["pad_center_depth"]) + CLEAR + dz   # pad centre above brick bottom
        z_grasp = pad_c + cal["pad_center_depth"] - 0.6
        log["pad_center_above_mid_mm"] = (pad_c - H / 2) * 1000
    else:
        z_grasp = (H / 2 + dz + TIP_Z) - 0.6
    move([0, 0, 0], [0, 0, z_grasp], False, 1.0)
    hold([0, 0, z_grasp], False, 0.2)
    zb0 = d.body("brick").xpos[2] if has else 0.0
    pen, zmax = 0.0, zb0
    for _ in range(int(1.0 / TS)):                     # close, watching the brick and the finger penetration
        step([0, 0, z_grasp], True)
        if has:
            zmax = max(zmax, d.body("brick").xpos[2])
            bid = m.geom("brick").id
            for i in range(d.ncon):
                c = d.contact[i]
                if bid in (c.geom1, c.geom2) and m.geom(c.geom2 if c.geom1 == bid else c.geom1).name.startswith("g_"):
                    pen = min(pen, c.dist)
    log["closure_rad"] = G.closure(d)
    if has:
        log["close_brick_rise_mm"] = float((zmax - zb0) * 1000)
        log["close_max_penetration_mm"] = float(-pen * 1000)
    if not has:
        return log
    f = pad_forces(m, d)
    log["grasp_normal_N"] = {k: round(v, 2) for k, v in f.items()}
    rel0 = d.body("brick").xpos - carrier(); q0 = d.body("brick").xquat.copy()
    up = [0, 0, z_grasp + 0.3]
    move([0, 0, z_grasp], up, True, 1.0, "lift")
    hold(up, True, 0.3)
    f = pad_forces(m, d)
    log["hold_normal_N"] = {k: round(v, 2) for k, v in f.items()}
    log["hold_closure_rad"] = G.closure(d)
    over = [0.5, 0, z_grasp + 0.3]
    move(up, over, True, move_time, "move")
    for _ in range(int(rest_after_move / TS)):
        step(over, True)
        seq["move"].append((d.body("brick").xpos - carrier()).copy()); seq["move_q"].append(d.body("brick").xquat.copy())
    seq["move"].append((d.body("brick").xpos - carrier()).copy()); seq["move_q"].append(d.body("brick").xquat.copy())
    for tag in ("lift", "move"):
        disp = np.linalg.norm(np.array(seq[tag]) - rel0, axis=1)
        log[f"{tag}_slip_max_mm"] = float(disp.max() * 1000)
        log[f"{tag}_tilt_max_deg"] = max(quat_angle(q0, q) for q in seq[tag + "_q"])
    down = [0.5, 0, z_grasp + 0.001]
    move(over, down, True, 1.0)
    hold(down, False, 0.6)
    move(down, [0.5, 0, z_grasp + 0.15], False, 0.5)
    b = d.body("brick").xpos; q = d.body("brick").xquat
    log["placed_xy_err_mm"] = float(np.hypot(b[0] - 0.5, b[1]) * 1000)
    log["placed_z_mm"] = float(b[2] * 1000)
    log["placed_tilt_deg"] = float(np.degrees(np.arccos(np.clip(1 - 2 * (q[1] ** 2 + q[2] ** 2), -1, 1))))
    log["max_tendon_violation_rad"] = float(np.abs(d.ten_length).max())
    log["pass"] = bool(max(log["lift_slip_max_mm"], log["move_slip_max_mm"]) <= 1.0
                       and max(log["lift_tilt_max_deg"], log["move_tilt_max_deg"]) <= 1.0)
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--torque", type=float, nargs="+", default=[2, 4, 6, 8])
    ap.add_argument("--kp", type=float, default=100.0)
    ap.add_argument("--kv", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} 이미 있음: 덮어쓰지 않음")
    out.mkdir(parents=True)
    res = {"git_commit": subprocess.check_output(["git", "-C", str(SIM), "rev-parse", "HEAD"], text=True).strip(),
           "gripper_xml_sha256": subprocess.check_output(["sha256sum", str(G.GRIPPER_XML)], text=True).split()[0],
           "tip_z_m": TIP_Z, "runs": []}
    for tq in a.torque:
        cfg = G.GripperConfig(kp=a.kp, kv=a.kv, torque_limit=tq)
        r = {"cfg": cfg.to_dict(), "empty": run(cfg, None)}
        for kind in ("full", "half"):
            r[kind] = run(cfg, kind)
        res["runs"].append(r)
        e = r["empty"]["closure_rad"]
        for kind in ("full", "half"):
            x = r[kind]
            print(f"tau {tq:4.1f} {kind:4s} pass {x['pass']!s:5s} closure {x['closure_rad']:.3f}/empty {e:.3f} "
                  f"hold N L/R {x['hold_normal_N']['left']:.1f}/{x['hold_normal_N']['right']:.1f} other {x['hold_normal_N']['other_gripper']:.1f} | "
                  f"slip lift/move {x['lift_slip_max_mm']:.2f}/{x['move_slip_max_mm']:.2f} mm tilt {x['lift_tilt_max_deg']:.2f}/{x['move_tilt_max_deg']:.2f} deg | "
                  f"placed xy {x['placed_xy_err_mm']:.2f} tilt {x['placed_tilt_deg']:.2f} | tendon {x['max_tendon_violation_rad']:.3f}", flush=True)
    (out / "result.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
