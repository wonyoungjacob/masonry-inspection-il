"""단계 2 도달 지도: 벽 자리 16개 × (놓는 자세 + 0.10 m 위 접근 자세), 공급 칸 × (집는 자세 + 접근 자세).

각 자세: 위에서 수직으로 집는 TCP 자세의 IK 해, 관절 여유(팔꿈치 15° 이상, wrist_2가 0·π에서 15° 이상, 손목 중심이 베이스 축에서
0.10 m 이상), 받침대·탁자·바닥·벽(이미 쌓인 벽돌)·다른 공급 벽돌과의 최소 간격(mj_geomDistance)을 잰다.
공급 칸은 후보 격자(로봇 뒤쪽, 베이스 거리 0.55 ~ 0.95 m)에서 가능한 것만 남기고, 철수 자세 → 공급 → 벽 → 철수의 어깨 회전량
합이 최소가 되게 16칸을 벽 자리에 배정한다(헝가리안). 공급 흩기(±10 mm, ±10°)의 양 끝도 검사한다.

실행: MUJOCO_GL=egl venv/bin/python -m wall3.reach_map --out outputs/wall3/reach_map_<날짜>
"""
import argparse, json, subprocess
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import linear_sum_assignment

from wall3 import gripper as G
from wall3.config import SceneConfig, W, H, JOINT, WALL_Y
from wall3.ik import UR10eIK, wrap, PAN_OFFSET
from wall3.poses import tcp_for_brick, grip_joint_qpos
from wall3.scene import WallEnv, yaw_quat, _quat_rotmat

SIM = Path(__file__).resolve().parents[1]
APPROACH = 0.10
RETRACT_TCP = np.array([0.0, 0.40, 0.75])       # 철수 자세: TCP above the robot, towards the wall, gripper down
CLEAR_MIN = 0.0005                               # m: required clearance for "no collision" at a pose
ROWS = [-0.25, -0.43, -0.61, -0.79]
COLS = [-0.91, -0.65, -0.39, -0.13, 0.13, 0.39, 0.65, 0.91]


def candidates():
    out = []
    for y in ROWS:
        for x in COLS:
            r = np.hypot(x, y)
            if 0.55 <= r <= 0.95:
                out.append((x, y))
    return out


class Checker:
    def __init__(self, env: WallEnv):
        self.env = env
        m = env.model
        self.m, self.d = m, mujoco.MjData(m)
        self.ik = UR10eIK(m)
        self.robot_g = [g for g in range(m.ngeom) if m.geom_bodyid[g] in env.robot_bodies
                        and (m.geom_contype[g] or m.geom_conaffinity[g])]
        self.grip_g = [g for g in self.robot_g if m.geom_bodyid[g] in env.gripper_bodies]
        names = ["floor", "pedestal", "table"]
        self.env_g = {n: m.geom(n).id for n in names}
        self.theta_c = env.cal["closure"]
        self.jq = {n: m.joint(G.PREFIX + n).qposadr[0] for n in grip_joint_qpos(0)}
        self.ft = np.zeros(6)

    def set_state(self, q, theta, bricks, beds=()):
        """bricks: {i: (pos, yaw)} for every brick. beds: boundaries laid at design poses."""
        m, d, env = self.m, self.d, self.env
        d.qpos[:] = 0
        d.qpos[env.arm_qadr] = q
        for n, v in grip_joint_qpos(theta).items():
            d.qpos[self.jq[n]] = v
        for i, (p, yaw) in bricks.items():
            a = env.brick_qadr[i]
            d.qpos[a:a + 3] = p; d.qpos[a + 3:a + 7] = yaw_quat(yaw)
        for nm, mid in env.bed_mocap.items():
            d.mocap_pos[mid] = [0, 0, -1]
        for bnd in beds:
            info = env.beds[bnd]
            for nm, i in zip(info["slabs"], info["lower"]):
                s = env.design[i]
                d.mocap_pos[env.bed_mocap[nm]] = [s["x"], s["y"], s["z"] + H / 2 + JOINT / 2]
            for j, nm in enumerate(info["joints"]):
                a, b = env.design[info["lower"][j]], env.design[info["lower"][j + 1]]
                d.mocap_pos[env.bed_mocap[nm]] = [((a["x"] + a["L"] / 2) + (b["x"] - b["L"] / 2)) / 2, WALL_Y,
                                                  a["z"] + H / 2 + JOINT / 2]
        mujoco.mj_forward(m, d)

    def clearance(self, held=None, support=None):
        """Min distance robot↔environment and held brick↔environment (m), with the closest pair names."""
        m, d, env = self.m, self.d, self.env
        others = list(self.env_g.values()) + [env.brick_gid[i] for i in range(env.n) if i != held]
        best = (1.0, "")
        for g1 in self.robot_g:
            for g2 in others:
                dist = mujoco.mj_geomDistance(m, d, g1, g2, 0.05, self.ft)
                if dist < best[0]:
                    best = (dist, f"{m.geom(g1).name or m.body(m.geom_bodyid[g1]).name}~{m.geom(g2).name}")
            if held is not None and g1 not in self.grip_g:
                dist = mujoco.mj_geomDistance(m, d, g1, env.brick_gid[held], 0.05, self.ft)
                if dist < best[0]:
                    best = (dist, f"{m.body(m.geom_bodyid[g1]).name}~held")
        if held is not None:
            for g2 in others:
                if g2 == support:
                    continue
                dist = mujoco.mj_geomDistance(m, d, env.brick_gid[held], g2, 0.05, self.ft)
                if dist < best[0]:
                    best = (dist, f"held~{m.geom(g2).name}")
        return float(best[0]), best[1]


def ik_seeds(az_target, q_ref):
    seeds = []
    for dpan in (0.0, 0.25, -0.25):
        for lift, elbow in ((-1.2, 1.6), (-1.6, 2.0), (-0.9, 1.2), (-2.0, 2.3)):
            pan = wrap(az_target + PAN_OFFSET + dpan)
            seeds.append(np.array([pan, lift, elbow, -np.pi / 2 - (lift + elbow) + 0.0, -np.pi / 2, q_ref[5]]))
    return seeds


def solve_pose(ck: Checker, p, R, q_ref):
    az = np.arctan2(p[1], p[0])
    q, info = ck.ik.solve(p, R, q_ref, seeds=ik_seeds(az, q_ref) + [q_ref])
    if q is not None:
        q = q.copy()
        q[0] = q_ref[0] + wrap(q[0] - q_ref[0])          # keep the pan continuous with the reference
        q[5] = q_ref[5] + wrap(q[5] - q_ref[5])
    return q, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--slots", default="", help="json list of 16 (x,y) to verify instead of searching")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} 이미 있음: 덮어쓰지 않음")
    out.mkdir(parents=True)
    cand = json.loads(a.slots) if a.slots else candidates()
    pad = tuple(cand[:16]) + tuple((0.0, -5.0 - 0.3 * k) for k in range(max(0, 16 - len(cand))))
    cfg = SceneConfig(supply_slots=pad[:16])
    env = WallEnv(cfg)
    ck = Checker(env)
    cal, clear = env.cal, cfg.grasp_clear
    far = {i: (np.array([3.0 + 0.3 * i, 3.0, H / 2]), 0.0) for i in range(env.n)}   # bricks out of the way

    # retract pose
    R_ret = tcp_for_brick([0, 0, 0], 0.0, cal, clear)[1]
    q_home = np.array([-1.92, -1.6, 1.9, -1.871, -1.571, 0.0])
    q_ret, info_ret = solve_pose(ck, RETRACT_TCP, R_ret, q_home)
    if q_ret is None:
        raise SystemExit("retract pose has no IK solution")
    ck.set_state(q_ret, ck.theta_c, far)
    res = {"retract": {"tcp": RETRACT_TCP.tolist(), "q": q_ret.tolist(), "margins": info_ret["margins"],
                       "clearance_m": ck.clearance()[0]}}

    # wall slots: place (closed, brick held) and approach above
    wall = []
    for k, s in enumerate(env.design):
        bricks = dict(far)
        for j in range(k):
            bricks[j] = (np.array([env.design[j]["x"], env.design[j]["y"], env.design[j]["z"]]), 0.0)
        pb = np.array([s["x"], s["y"], s["z"]])
        bricks[k] = (pb, 0.0)
        beds = tuple(b for b in (1, 2) if s["course"] > b)
        support = ck.env_g["floor"] if s["course"] == 1 else None
        row = {"slot": k, "course": s["course"], "index": s["index"], "half": s["half"],
               "base_dist_m": float(np.hypot(s["x"], s["y"])), "poses": {}}
        for yaw in (0.0, np.pi):
            ok_all, poses = True, {}
            q_prev = q_ret
            for name, lift in (("approach", APPROACH), ("place", 0.0)):
                p, R = tcp_for_brick(pb, yaw, cal, clear, lift)
                q, info = solve_pose(ck, p, R, q_prev)
                if q is None:
                    poses[name] = {"ok": False, "reason": "no IK within joint conditions"}; ok_all = False; break
                held = dict(bricks); held[k] = (pb + [0, 0, lift], 0.0)
                ck.set_state(q, ck.theta_c, held, beds)
                cl, who = ck.clearance(held=k, support=support if lift == 0 else None)
                ok = cl >= CLEAR_MIN
                poses[name] = {"ok": ok, "q": q.tolist(), "margins": info["margins"], "clearance_m": cl, "closest": who}
                ok_all &= ok
                q_prev = q
            row["poses"][f"yaw{int(round(np.degrees(yaw)))}"] = {"ok": ok_all, **poses}
        row["ok"] = any(v["ok"] for v in row["poses"].values())
        wall.append(row)
    res["wall"] = wall

    # supply slots: grasp (open, then closed) and approach (open), nominal and scatter extremes
    sup = []
    corners = [(0, 0, 0)] + [(dx, dy, dyaw) for dx in (-1, 1) for dy in (-1, 1) for dyaw in (-1, 1)]
    for c_i, (x, y) in enumerate(cand):
        row = {"cand": c_i, "x": x, "y": y, "base_dist_m": float(np.hypot(x, y)), "cases": [], "pan": None}
        ok_slot = True
        for dx, dy, dyaw in corners:
            p0 = np.array([x + dx * cfg.scatter_pos, y + dy * cfg.scatter_pos, cfg.table_h + H / 2])
            yaw0 = np.deg2rad(dyaw * cfg.scatter_yaw_deg)
            # brick 0 = the brick under test; bricks 1..15 lie on the 15 nearest other candidate slots (worst case)
            bricks = dict(far)
            near = sorted([j for j in range(len(cand)) if j != c_i], key=lambda j: np.hypot(cand[j][0] - x, cand[j][1] - y))[:env.n - 1]
            for idx, j in enumerate(near, start=1):
                bricks[idx] = (np.array([cand[j][0], cand[j][1], cfg.table_h + H / 2]), 0.0)
            hi = 0
            bricks[hi] = (p0, yaw0)
            case = {"d": [dx, dy, dyaw], "ok": False}
            best = None
            for yaw in (yaw0, yaw0 + np.pi):
                okc, info_c = True, {}
                q_prev = q_ret
                for name, lift, theta in (("approach", APPROACH, 0.0), ("grasp_open", 0.0, 0.0), ("grasp_closed", 0.0, None)):
                    p, R = tcp_for_brick(p0, yaw, cal, clear, lift)
                    q, info = solve_pose(ck, p, R, q_prev)
                    if q is None:
                        okc = False; info_c[name] = "no IK"; break
                    ck.set_state(q, ck.theta_c if theta is None else theta, bricks)
                    cl, who = ck.clearance(held=hi if theta is None else None, support=ck.env_g["table"])
                    info_c[name] = {"clearance_m": cl, "closest": who, "margins": info["margins"], "q": q.tolist()}
                    okc &= cl >= CLEAR_MIN
                    q_prev = q
                if okc:
                    cost = abs(wrap(q_prev[5] - q_ret[5]))
                    if best is None or cost < best[0]:
                        best = (cost, yaw, info_c)
            if best is not None:
                case.update(ok=True, yaw=float(best[1]), detail=best[2])
                if (dx, dy, dyaw) == (0, 0, 0):
                    row["pan"] = best[2]["grasp_closed"]["q"][0]
            ok_slot &= case["ok"]
            row["cases"].append(case)
        row["ok"] = bool(ok_slot)
        sup.append(row)
    res["supply_candidates"] = sup

    # assignment: wall slot k <- supply slot minimising pan travel retract → supply → wall → retract (net pan turn 0)
    feas = [r for r in sup if r["ok"]]
    res["n_wall_ok"] = sum(r["ok"] for r in wall)
    res["n_supply_feasible"] = len(feas)
    if res["n_wall_ok"] == 16 and len(feas) >= 16 and not a.slots:
        pr = q_ret[0]
        cost = np.zeros((16, len(feas)))
        for k, w in enumerate(wall):
            v = next(v for v in w["poses"].values() if v["ok"])
            pw = v["place"]["q"][0]
            for j, r in enumerate(feas):
                ps = r["pan"]
                d1, d3 = wrap(ps - pr), wrap(pr - pw)          # out to the table and back from the wall the short way;
                cost[k, j] = abs(d1) + abs(d1 + d3) + abs(d3)    # table → wall unwinds them (no net turn of the pan)
        ri, ci = linear_sum_assignment(cost)
        res["assignment"] = [{"wall_slot": int(k), "supply": [feas[j]["x"], feas[j]["y"]], "pan_travel_deg": float(np.degrees(cost[k, j]))}
                             for k, j in zip(ri, ci)]
        res["supply_slots"] = [[feas[j]["x"], feas[j]["y"]] for k, j in sorted(zip(ri, ci))]
    res["meta"] = {"git_commit": subprocess.check_output(["git", "-C", str(SIM), "rev-parse", "HEAD"], text=True).strip(),
                   "config": cfg.to_dict(), "approach_m": APPROACH, "clear_min_m": CLEAR_MIN, "calibration": cal}
    (out / "reach_map.json").write_text(json.dumps(res, indent=1, default=float))
    print("retract q", np.round(q_ret, 3), "margins", info_ret["margins"])
    for w in wall:
        v = {kk: (vv["ok"], round(min([vv[n]["clearance_m"] for n in ("approach", "place") if n in vv and isinstance(vv[n], dict) and "clearance_m" in vv[n]] or [-1]) * 1000, 1)) for kk, vv in w["poses"].items()}
        print(f"wall {w['slot']:2d} c{w['course']} {'half' if w['half'] else 'full'} r {w['base_dist_m']:.3f} ok {w['ok']} {v}")
    for r in sup:
        print(f"cand ({r['x']:+.2f},{r['y']:+.2f}) r {r['base_dist_m']:.2f} ok {r['ok']} cases ok {sum(c['ok'] for c in r['cases'])}/{len(r['cases'])}")
    print("wall ok", res["n_wall_ok"], "/ 16, supply feasible", len(feas), "/", len(sup))
    if "assignment" in res:
        for x in res["assignment"]:
            print(x)
    env.close()


if __name__ == "__main__":
    main()
