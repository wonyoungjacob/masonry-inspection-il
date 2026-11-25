"""UR10e 역기구학 대본: 위에서 수직으로 집고 놓는 벽 쌓기 시연 (지시서 단계 4).

벽돌 한 장 = 철수 자세에서 출발 → 공급 칸 접근(0.10 m 위) → 내려가 집기 → 들기 → 벽 자리 접근 → 내려놓기(놓기 직전 위치 보정)
→ 놓기 → 올라가기 → 철수 자세 복귀. 재시도·다시 집기 없음: 실패하면 벽을 멈추고 사유를 기록한다.
정답 좌표(벽돌 참값 자세)는 시연 생성이라 써도 된다(연구 원칙 1). 정책 관측에는 넣지 않는다.

관절 목표(=기록하는 행동) = q_d + (τ_ff + kd·q̇_d) / kp. τ_ff는 원하는 궤적의 역동역학(로봇+집게) + 쥔 벽돌의 무게(자코비안).
UR10e 위치 서보(kp 5000, kd 500)의 처짐·추종 지연을 없애 놓기 정확도를 확보한다. 관절 속도는 UR10e 사양 안
(어깨 회전·들기 120°/s, 팔꿈치·손목 180°/s), 큰 이동은 최소 저크 곡선.
"""
from dataclasses import dataclass, asdict

import mujoco
import numpy as np

from wall3 import gripper as G
from wall3.config import RETRACT_Q, H, JOINT
from wall3.ik import UR10eIK, wrap, down_R
from wall3.poses import dz_pad, yaw_of_quat
from wall3.scene import WallEnv, _quat_rotmat


def rot_angle(R, Rt):
    """True rotation angle between two rotation matrices (rad)."""
    return float(np.arccos(np.clip((np.trace(R.T @ Rt) - 1.0) / 2.0, -1.0, 1.0)))


KP, KD = 5000.0, 500.0                      # UR10e Menagerie actuator gain / damping (gainprm, biasprm)


@dataclass
class ExpertConfig:
    vmax: tuple = (2.094, 2.094, 3.142, 3.142, 3.142, 3.142)   # rad/s, UR10e joint speed limits
    vfrac: float = 0.95                     # fraction of vmax used at the profile peak
    amax: tuple = (6.0, 6.0, 8.0, 10.0, 10.0, 10.0)   # rad/s^2, peak joint acceleration of the big moves
    approach: float = 0.10                  # m above grasp / place
    t_down: float = 0.50                    # s, approach → grasp / hover
    t_up: float = 0.40                      # s, grasp → approach, release → approach
    t_close: float = 0.75                   # s, gripper closing at the supply (command ramp + settle; the grip lock needs the 0.5 s ramp + 0.1 s still)
    tcp_same_pass: bool = True              # in-hand pose from one kinematics pass (False = pilot: TCP from qpos, brick from xpos)
    t_open: float = 0.25                    # s, gripper opening at the wall (ramp from just above the stalled closure)
    hover: float = 0.003                    # m, brick bottom above its support during the pre-release correction
    place_gap: float = 0.0005               # m, brick bottom above its support when released
    t_align: float = 0.25                   # s, re-aimed sideways correction at the approach height
    t_hover: float = 0.20                   # s, hover → place
    servo_iters: int = 3                    # pre-release position corrections (ground truth)
    servo_steps: int = 3                    # control steps per correction
    min_move: float = 0.40                  # s, shortest joint move
    grip_empty: float = 0.40                # rad, closure above this after closing = nothing in the hand (G1 candidate)
    slip_abort: float = 0.030               # m, brick drift in the hand → abort (it drifts 13 ~ 19 mm during the big pan move: MuJoCo friction creep on the robosuite pads; recorded per brick, corrected before release)
    wait_bed: float = 6.0                   # s, max wait at the wall approach for the mortar bed
    grip_squeeze: float = 0.06              # rad beyond the 90 mm contact closure commanded to hold a brick (see grip_close_cmd)
    vert_lag: float = 0.45                  # fraction of a joint move during which the vertical motion waits (down) / is done (up)
    lag_supply: float = 0.30                # retract → supply: descend over the table late
    lag_wall: float = 0.35                  # supply → wall: rotate away from the table before descending
    t_unload: float = 0.10                  # s, before opening: bring the finger command to the stalled closure (grip force → 0
                                            # on both fingers together) so the opening finger does not shove the brick
    # main_v2 (2026-10-22): one wrist branch per brick. main_v1 demos split between two (or more) wrist_3 angles for the same
    # brick: (a) a 180° grasp / place flip chosen by a near-tie rule, (b) IK accepting a pose turned 180° about the tool axis
    # (ik.rot_err is 0 at 180°), fixed above the wall by spinning the wrist half a turn in one step. Off = main_v1 behaviour.
    ik_angle_check: bool = False            # accept an IK solution only if the true rotation error (angle) is small
    grasp_wrist_ref: tuple | None = None    # rad per slot: grasp with the flip whose wrist_3 is closest to this
    place_wrist_ref: tuple | None = None    # rad per slot: place with the flip whose wrist_3 is closest to this
    max_cart_wrist_step: float | None = None  # rad, a straight-line move whose wrist_3 jumps more than this in one step fails

    def to_dict(self):
        d = asdict(self)
        for k in ("ik_angle_check", "grasp_wrist_ref", "place_wrist_ref", "max_cart_wrist_step"):
            if not d[k]:
                d.pop(k)
        return d


# expert settings per environment version (pilot_20261016 reproduces the pilot demos exactly)
EXPERT_VERSIONS = {
    "pilot_20261016": dict(t_close=0.45, tcp_same_pass=False),
    "main_v1": dict(),
    "main_v1_eval": dict(),          # same expert as main_v1 (only the mortar set time differs)
    # main_v2 (2026-10-22, 사용자 결정): main_v1 + one wrist branch per brick for the grasp and one for the place (not necessarily
    # the same). References = wrist_3 of the largest branch in the main_v1 dataset (data/wall3_main_20261019, 1000 walls) at the
    # moment the fingers start closing / opening (experiments/diag_demo_modes.py); slots 0, 2, 3, 4, 5, 15 had other branches.
    "main_v2": dict(ik_angle_check=True, max_cart_wrist_step=0.3,
                    grasp_wrist_ref=(-1.487, -1.455, -1.935, 0.946, 0.761, -1.211, -1.036, -0.657, 0.62, 0.433, 0.327, -1.012, -0.815, 0.264, -0.06, -0.074),
                    place_wrist_ref=(-2.796, 0.043, -0.294, -0.6, -0.831, 0.407, 0.201, -0.125, -0.455, -0.726, -0.878, 0.344, 0.043, -0.294, -0.6, -0.831)),
}


def expert_config(version, **kw):
    return ExpertConfig(**{**EXPERT_VERSIONS[version], **kw})


def minjerk(n):
    s = np.linspace(0.0, 1.0, n + 1)[1:]
    return 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5


RHO = 0.3                                    # smooth trapezoid: raised-cosine accel/decel over 30 % of the move each
VP_NORM = 1.0 / (1.0 - RHO)                  # peak of the normalised velocity
AP_NORM = VP_NORM * np.pi / (2 * RHO)        # peak of the normalised acceleration


def smooth_trap(u):
    """Position profile s(u), u in [0, 1]: velocity rises along a half cosine over [0, RHO], cruises, and falls over
    [1 − RHO, 1] (finite jerk at the ends, zero velocity/acceleration at both ends). s(0) = 0, s(1) = 1."""
    u = np.clip(np.asarray(u, float), 0, 1)
    r = RHO
    up = VP_NORM * (u / 2 - r / (2 * np.pi) * np.sin(np.pi * u / r))
    s_r = VP_NORM * r / 2
    cruise = s_r + VP_NORM * (u - r)
    w = 1 - u
    down = 1 - VP_NORM * (w / 2 - r / (2 * np.pi) * np.sin(np.pi * w / r))
    return np.where(u < r, up, np.where(u > 1 - r, down, cruise))


class WallExpert:
    def __init__(self, env: WallEnv, cfg: ExpertConfig | None = None):
        self.env = env
        self.cfg = cfg or expert_config(env.cfg.env_version)
        m = env.model
        self.ik = UR10eIK(m)
        self.dt = env.cfg.ctrl_dt
        self.scratch = mujoco.MjData(m)
        self.jacp = np.zeros((3, m.nv)); self.jacr = np.zeros((3, m.nv))
        self.w3 = m.body("wrist_3_link").id
        self.grip_q = [m.joint(G.PREFIX + n).qposadr[0] for n in
                       ("finger_joint", "left_inner_finger_joint", "left_inner_knuckle_joint",
                        "right_outer_knuckle_joint", "right_inner_finger_joint", "right_inner_knuckle_joint")]
        self.dzp = dz_pad(env.cal, env.cfg.grasp_clear)
        self.q_ret = np.array(RETRACT_Q)

    # ------------------------------------------------------------------ low level
    def _ff(self, q, qd, qdd, held):
        """Feed-forward joint force for the arm at the desired state (inverse dynamics + held brick weight)."""
        env, s = self.env, self.scratch
        s.qpos[:] = env.data.qpos
        s.qvel[:] = 0.0
        s.qacc[:] = 0.0
        s.qpos[env.arm_qadr] = q
        s.qvel[env.arm_dofadr] = qd
        s.qacc[env.arm_dofadr] = qdd
        for a in env.brick_qadr:                  # bricks far away: no contacts in the scratch copy
            s.qpos[a:a + 3] = [5.0, 5.0, -5.0]
        mujoco.mj_inverse(env.model, s)
        tau = s.qfrc_inverse[env.arm_dofadr].copy()
        if held is not None:
            pb = env.brick_pose(held)[0]
            mujoco.mj_jac(env.model, s, self.jacp, self.jacr, pb, self.w3)
            mass = env.model.body_mass[env.brick_bid[held]]
            tau += self.jacp[:, env.arm_dofadr].T @ np.array([0.0, 0.0, mass * 9.81])
        return tau

    def _send(self, q, qd, qdd, grip, held, phase):
        tau = self._ff(q, qd, qdd, held)          # includes joint (passive) damping when the servo damping lives in the joint
        ctrl = q + (tau + self.env.arm_kd * qd) / self.env.arm_kp
        action = np.r_[ctrl, grip]
        if self.on_step is not None:
            self.on_step(self.env, action, phase, self.brick_k)
        self.env.step(ctrl, grip)
        self.q_cmd = q.copy()
        self.steps += 1
        return action

    def _play(self, path, grip, held, phase):
        """path: (N, 6) joint positions at 20 Hz; grip: finger_1 command (scalar or per step).
        Velocities/accelerations by finite differences."""
        if len(path) == 0:
            return
        grip = np.broadcast_to(np.asarray(grip, float), (len(path),))
        prev = self.q_cmd
        P = np.vstack([prev, path])
        V = np.gradient(P, self.dt, axis=0)
        A = np.gradient(V, self.dt, axis=0)
        V[-1] = 0.0; A[-1] = 0.0
        for i in range(1, len(P)):
            self.grip_cmd = float(grip[i - 1])
            self._send(P[i], V[i], A[i], self.grip_cmd, held, phase)
            self._watch(held, phase)
            if self.fail:
                return

    def _joint_move(self, q_to, grip, held, phase, t_min=None, lag=None):
        """Min-jerk joint move. The three parallel pitch joints (lift, elbow, wrist_1) share one profile, so the tool stays
        vertical; it is delayed when going down (rotate first, then descend) and advanced when going up (rise first),
        so the hand and the held brick never sweep low over the supply table or the wall."""
        dq = q_to - self.q_cmd
        vm = np.array(self.cfg.vmax) * self.cfg.vfrac
        z0, z1 = self.ik.fk(self.q_cmd)[0][2], self.ik.fk(q_to)[0][2]
        f = self.cfg.vert_lag if lag is None else lag
        am = np.array(self.cfg.amax)
        Tj = np.maximum(VP_NORM * np.abs(dq) / vm, np.sqrt(AP_NORM * np.abs(dq) / am))   # per joint, smooth trapezoid
        T_rot = float(np.max(Tj[[0, 4, 5]]))
        T_pitch = float(np.max(Tj[[1, 2, 3]])) / (1.0 - f)
        T = max(t_min or self.cfg.min_move, T_rot, T_pitch)
        n = max(2, int(np.ceil(T / self.dt)))
        u = np.linspace(0.0, 1.0, n + 1)[1:]
        s = smooth_trap(u)
        if f <= 0:
            w = u
        elif z1 < z0 - 0.02:        # going down: pitch joints start late
            w = np.clip((u - f) / (1 - f), 0, 1)
        elif z1 > z0 + 0.02:        # going up: pitch joints finish early
            w = np.clip(u / (1 - f), 0, 1)
        else:
            w = u
        sp = smooth_trap(w)
        prof = np.stack([s, sp, sp, sp, s, s], axis=1)
        self._play(self.q_cmd + prof * dq, grip, held, phase)

    def _cart_move(self, p_to, R, T, grip, held, phase):
        """Straight-line TCP move (min-jerk) with IK at every control step."""
        p0, _ = self.ik.fk(self.q_cmd)
        n = max(1, int(np.ceil(T / self.dt)))
        path, q = [], self.q_cmd
        for s in minjerk(n):
            q_prev = q
            q, ep, er = self.ik.solve_from(q, p0 + s * (np.asarray(p_to) - p0), R)
            if self.cfg.ik_angle_check:
                er = rot_angle(self.ik.fk(q)[1], R)
            if ep > 1e-3 or er > 1e-2:
                self.fail = f"ik_cartesian:{phase}"
                return
            if self.cfg.max_cart_wrist_step and abs(q[5] - q_prev[5]) > self.cfg.max_cart_wrist_step:
                self.fail = f"wrist_jump:{phase}:{q[5] - q_prev[5]:+.2f}"
                return
            path.append(q)
        self._play(np.array(path), grip, held, phase)

    def _hold(self, n, grip, held, phase):
        self._play(np.repeat(self.q_cmd[None], n, axis=0), grip, held, phase)

    def _solve(self, p, R, q_ref, pan_ref):
        if self.cfg.ik_angle_check:
            q = self._solve_checked(p, R, q_ref)
        else:
            q, info = self.ik.solve(p, R, q_ref, seeds=self._seeds(p, q_ref))
        if q is None:
            return None
        q = q.copy()
        q[0] = pan_ref + wrap(q[0] - pan_ref)          # keep the pan on the side it came from (no net winding)
        q[5] = q_ref[5] + wrap(q[5] - q_ref[5])
        return q

    def _solve_checked(self, p, R, q_ref, tol_p=2e-4, tol_ang=2e-3):
        """ik.solve with the true rotation angle as the orientation test (ik.rot_err vanishes at 180°) and the seeds repeated
        with wrist_3 turned by ±90° / 180°, so a seed half a turn off can still converge to the right pose."""
        best = None
        for s0 in self._seeds(p, q_ref):
            for dw in (0.0, np.pi / 2, -np.pi / 2, np.pi):
                sd = np.array(s0, float); sd[5] += dw
                q, ep, _ = self.ik.solve_from(sd, p, R)
                if ep > tol_p or rot_angle(self.ik.fk(q)[1], R) > tol_ang:
                    continue
                if min(self.ik.margins(q).values()) < 0 or not self.ik.branch_ok(q, p):
                    continue
                cost = float(np.sum(wrap(q - q_ref) ** 2))
                if best is None or cost < best[1]:
                    best = (q, cost)
        return None if best is None else best[0]

    def _seeds(self, p, q_ref):
        from wall3.ik import PAN_OFFSET
        az = np.arctan2(p[1], p[0])
        out = [q_ref]
        for lift, elbow in ((-1.2, 1.6), (-0.7, 2.0), (-1.6, 2.0), (-0.9, 1.2)):
            out.append(np.array([wrap(az + PAN_OFFSET), lift, elbow, -np.pi / 2 - (lift + elbow), -np.pi / 2, q_ref[5]]))
        return out

    def grip_close_cmd(self):
        """Finger command used to hold a brick: the 90 mm contact closure + grip_squeeze (position mode: each finger is a
        spring around this target, which centres the brick between the two independently driven fingers), or the full
        close command when grip_squeeze <= 0 (both fingers torque-saturated: no centring)."""
        gc = self.env.cfg.gripper
        if self.cfg.grip_squeeze <= 0:
            return gc.close_ctrl
        return float(min(gc.close_ctrl, self.env.cal["closure"] + self.cfg.grip_squeeze))

    def _support_top(self, k):
        """Height of what slot k rests on: floor (course 1) or the top of the mortar bed actually laid under it."""
        env = self.env
        s = env.design[k]
        if s["course"] == 1:
            return 0.0
        info = env.beds[s["course"] - 1]
        if env.bed_laid[s["course"] - 1] < 0:            # not laid yet: design value (recomputed before going down)
            return s["z"] - H / 2
        tops = []
        for nm, i in zip(info["slabs"], info["lower"]):
            lo = env.design[i]
            if abs(lo["x"] - s["x"]) < (lo["L"] + s["L"]) / 2:    # slab under the footprint
                tops.append(env.data.mocap_pos[env.bed_mocap[nm]][2] + JOINT / 2)
        return float(np.mean(tops)) if tops else s["z"] - H / 2

    # ------------------------------------------------------------------ monitoring
    def _watch(self, held, phase):
        env = self.env
        for name_r, name_o, dist, g_r, g_o in env.contacts():
            if held is not None and g_o == env.brick_gid[held] and g_r in env.pad_gids:
                continue
            if phase in ("grasp_close", "grasp_lift", "place_unload", "place_open") and g_o == env.brick_gid[self.brick_k] and g_r in env.pad_gids:
                continue
            if dist < -0.0005:
                self.fail = f"collision:{phase}:{name_r}~{name_o}"
                return
            self.touch.append((self.env.t, phase, name_r, name_o, round(dist * 1000, 2)))
        if held is not None and self.in_hand is not None:
            rel = self._brick_in_tcp(held)
            drift = np.linalg.norm(rel[0] - self.in_hand[0])
            self.max_drift = max(self.max_drift, drift)
            if drift > self.cfg.slip_abort:
                self.fail = f"slip:{phase}:{drift * 1000:.1f}mm"

    def _brick_in_tcp(self, i):
        """Brick pose in the TCP frame. Both from the same kinematics pass (xpos/site_xpos are the pre-integration state of
        the last physics step; mixing them with the post-integration qpos biased this by speed × 1 ms, up to ~2 mm)."""
        if not self.cfg.tcp_same_pass:
            pt, Rt = self.env.tcp()
            pb, qb = self.env.brick_pose(i)
            return Rt.T @ (pb - pt), Rt.T @ _quat_rotmat(qb)
        d = self.env.data
        pt, Rt = d.site_xpos[self.env.tcp_sid], d.site_xmat[self.env.tcp_sid].reshape(3, 3)
        b = self.env.brick_bid[i]
        return Rt.T @ (d.xpos[b] - pt), Rt.T @ d.xmat[b].reshape(3, 3)

    # ------------------------------------------------------------------ one brick
    def _brick(self, k):
        env, c = self.env, self.cfg
        s = env.design[k]
        held = None
        self.in_hand = None
        self.max_drift = 0.0
        self.drift_before_place = 0.0
        start = env.t
        # --- supply: grasp yaw closest to the current wrist_3
        pb, qb = env.brick_pose(k)
        yaw_b = yaw_of_quat(qb)
        best = None
        ref = self.q_ret[5] if c.grasp_wrist_ref is None else c.grasp_wrist_ref[k]
        for yaw in (yaw_b, yaw_b + np.pi):
            R = down_R(yaw)
            q_app = self._solve(pb + [0, 0, self.dzp + c.approach], R, self.q_ret, self.q_ret[0])
            if q_app is not None and (best is None or abs(q_app[5] - ref) < abs(best[1][5] - ref)):
                best = (R, q_app)
        if best is None:
            return {"fail": "ik_supply"}
        if c.grasp_wrist_ref is not None and abs(best[1][5] - ref) > np.pi / 2:      # only the other branch was reachable
            self.fail = f"wrist_branch:grasp:{best[1][5]:+.2f}"
            return {}
        R_s, q_app = best
        self._joint_move(q_app, 0.0, None, "to_supply", lag=c.lag_supply)
        if self.fail:
            return {}
        self._cart_move(pb + [0, 0, self.dzp], R_s, c.t_down, 0.0, None, "grasp_descend")
        if self.fail:
            return {}
        gc = env.cfg.gripper
        n = int(round(c.t_close / self.dt))
        cg = self.grip_close_cmd()
        self._hold(n, [min(cg, G.ramp(gc, True, (i + 1) * self.dt, gc.open_ctrl)) for i in range(n)], None, "grasp_close")
        if self.fail:
            return {}
        closure = env.closure()
        if closure > c.grip_empty:
            self.fail = f"grasp_empty:closure={closure:.3f}"
            return {"closure": closure}
        grip = cg
        held = k
        self._cart_move(pb + [0, 0, self.dzp + c.approach], R_s, c.t_up, grip, held, "grasp_lift")
        if self.fail:
            return {}
        self.in_hand = self._brick_in_tcp(k)
        if np.linalg.norm(self.in_hand[0] - [0, 0, self.dzp]) > 0.02:   # brick centre sits dzp below the TCP (TCP z = down)
            self.fail = "grasp_lost"
            return {}
        # --- wall: TCP pose that puts the brick (as it sits in the hand) at its design x, y, yaw, resting on the real support
        p_d = np.array([s["x"], s["y"], self._support_top(k) + H / 2])
        R_d = np.array([[np.cos(s["yaw"]), -np.sin(s["yaw"]), 0], [np.sin(s["yaw"]), np.cos(s["yaw"]), 0], [0, 0, 1.0]])
        rel_p, rel_R = self.in_hand
        cands = []
        for flip in (0.0, np.pi):                       # brick is symmetric under a 180° turn about z
            Rf = R_d @ np.array([[np.cos(flip), -np.sin(flip), 0], [np.sin(flip), np.cos(flip), 0], [0, 0, 1.0]])
            R_tcp = Rf @ rel_R.T
            p_tcp = p_d - R_tcp @ rel_p
            q_w = self._solve(p_tcp + [0, 0, c.approach], R_tcp, self.q_ret, self.q_ret[0])
            if q_w is not None:
                cost = (abs(q_w[5] - self.q_cmd[5]) + 0.3 * abs(q_w[5] - self.q_ret[5]) if c.place_wrist_ref is None
                        else abs(q_w[5] - c.place_wrist_ref[k]))
                cands.append((cost, Rf, R_tcp, p_tcp, q_w))
        if not cands:
            self.fail = "ik_wall"
            return {}
        if c.place_wrist_ref is not None and min(x[0] for x in cands) > np.pi / 2:  # only the other branch was reachable
            self.fail = f"wrist_branch:place:{min(cands, key=lambda x: x[0])[4][5]:+.2f}"
            return {}
        _, Rf, R_tcp, p_tcp, q_w = min(cands, key=lambda x: x[0])
        self._joint_move(q_w, grip, held, "to_wall", lag=c.lag_wall)
        if self.fail:
            return {}
        # mortar bed must be there before a course-2/3 brick goes down
        if s["course"] > 1:
            waited = 0
            while env.bed_laid[s["course"] - 1] < 0 and waited * self.dt < c.wait_bed:
                self._hold(1, grip, held, "wait_bed"); waited += 1
                if self.fail:
                    return {}
            if env.bed_laid[s["course"] - 1] < 0:
                self.fail = f"bed_not_laid:{s['course'] - 1}"
                return {}
        # the brick creeps in the hand during the big move: re-measure it above the wall and re-aim (ground truth, demo only)
        p_d[2] = self._support_top(k) + H / 2
        self.in_hand = self._brick_in_tcp(k)
        self.drift_before_place = self.max_drift
        rel_p, rel_R = self.in_hand
        R_tcp = Rf @ rel_R.T
        p_tcp = p_d - R_tcp @ rel_p
        hover = c.hover
        # line up above the slot at the approach height first, then go straight down (no sideways motion near the wall)
        self._cart_move(p_tcp + [0, 0, c.approach], R_tcp, c.t_align, grip, held, "place_align")
        if self.fail:
            return {}
        self._cart_move(p_tcp + [0, 0, hover], R_tcp, c.t_down, grip, held, "place_descend")
        if self.fail:
            return {}
        for _ in range(c.servo_iters):
            pb_now, qb_now = env.brick_pose(k)
            e = pb_now - (p_d + [0, 0, hover])
            e_yaw = wrap(yaw_of_quat(qb_now) - s["yaw"])
            e_yaw = (e_yaw + np.pi / 2) % np.pi - np.pi / 2
            Rz = np.array([[np.cos(-e_yaw), -np.sin(-e_yaw), 0], [np.sin(-e_yaw), np.cos(-e_yaw), 0], [0, 0, 1.0]])
            R_tcp = Rz @ R_tcp
            p_tcp = p_tcp - e
            self._cart_move(p_tcp + [0, 0, hover], R_tcp, c.servo_steps * self.dt, grip, held, "place_servo")
            if self.fail:
                return {}
        self._cart_move(p_tcp + [0, 0, c.place_gap], R_tcp, c.t_hover, grip, held, "place_lower")
        if self.fail:
            return {}
        pre = env.brick_pose(k)
        cl = env.closure()
        nu = max(1, int(round(c.t_unload / self.dt)))
        self._hold(nu, list(np.linspace(self.grip_cmd, cl, nu + 1)[1:]), held, "place_unload")
        if self.fail:
            return {}
        n = int(round(c.t_open / self.dt))
        self._hold(n, [G.ramp(gc, False, (i + 1) * self.dt, cl) for i in range(n)], None, "place_open")
        if self.fail:
            return {}
        held = None
        self.in_hand = None
        self._cart_move(p_tcp + [0, 0, c.approach], R_tcp, c.t_up, 0.0, None, "retreat")
        if self.fail:
            return {}
        self._joint_move(self.q_ret, 0.0, None, "to_retract", lag=0.0)
        if self.fail:
            return {}
        end = env.t
        pb_end, qb_end = env.brick_pose(k)
        static = bool(np.linalg.norm(env.brick_vel(k)[:3]) < 0.01)
        ev = [e for e in env.lock_events if e["brick"] == k and start <= e["t"] <= end]
        lock_info = {"locks": sum(e["event"] == "lock" for e in ev),
                     "releases": [e["reason"] for e in ev if e["event"] == "release"],
                     "lock_max_dev_mm": max([e.get("max_dev_while_locked_mm", 0.0) for e in ev if e["event"] == "release"] or [0.0]),
                     "lock_events": ev}
        return {"start": int(start), "end": int(end), "support_top_m": float(p_d[2] - H / 2), **lock_info, "duration_s": (end - start) * self.dt, "closure": float(closure),
                "max_drift_mm": float(self.max_drift * 1000), "drift_transport_mm": float(self.drift_before_place * 1000),
                "static_at_end": static,
                "release_err_mm": ((np.asarray(pre[0]) - p_d - [0, 0, c.place_gap]) * 1000).round(3).tolist(),
                "end_err_mm": ((pb_end - [s["x"], s["y"], s["z"]]) * 1000).round(3).tolist(),
                "end_yaw_err_deg": float(np.degrees((wrap(yaw_of_quat(qb_end) - s["yaw"]) + np.pi / 2) % np.pi - np.pi / 2))}

    # ------------------------------------------------------------------ one wall
    def run(self, seed, on_step=None, settle_s=2.0, max_bricks=None):
        env = self.env
        env.reset(seed, arm_q=self.q_ret)
        self.on_step = on_step
        self.q_cmd = self.q_ret.copy()
        self.grip_cmd = self.env.cfg.gripper.open_ctrl
        self.steps = 0
        self.fail = None
        self.touch = []
        self.brick_k = -1
        out = {"seed": int(seed), "bricks": []}
        for k in range(env.n if max_bricks is None else max_bricks):
            self.brick_k = k
            r = self._brick(k)
            r["slot"] = k
            out["bricks"].append(r)
            if self.fail:
                out["fail"] = {"brick": k, "reason": self.fail, "t": int(env.t)}
                break
        if not self.fail:
            self.brick_k = -1
            self._hold(int(round(settle_s / self.dt)), 0.0, None, "settle")
        out["steps"] = int(env.t)
        out["touch_events"] = self.touch[:200]
        out["n_touch_events"] = len(self.touch)
        out["bed_laid_t"] = {str(k): int(v) for k, v in env.bed_laid.items()}
        out["cured"] = env.cured.tolist()
        out["sag_at_cure_mm"] = [None if np.isnan(v) else float(v * 1000) for v in env.sag_at_cure]
        out["sag_end_mm"] = [None if np.isnan(v) else float(v * 1000) for v in (env.bed_sag(i) for i in range(env.n))]
        return out


def _r2q(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).ravel())
    return q
