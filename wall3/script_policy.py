"""정책 없음: 학습 정책 자리에 대본 제어기를 넣는다 (2026-11-16, 사용자 지시 '정책 없음: 대본 순서 + 같은 검측').

목적: Full(G0 + G1 + G2 + G2w + G3 + H)에서 정책이 맡던 일을 코드로 바꿔, 정책이 그 일을 대본보다 잘하는지 공평하게 비교한다.
Full 기록(eval_lf_full_20261110)에서 정책이 하던 일 = 다음 벽돌 고르기 · 공급칸 위로 대략 접근 · 설계 자리 근처까지 운반 · 놓기 시작.
집기(G0) · 쥠 확인(G1) · 놓기 보정(G2, G2w) · 놓은 뒤 재배치(G3) · 손에 든 벽돌 기준(H)은 Full과 같은 코드(wall3/gates.py)다.

이 제어기가 쓰는 정보 (사용자 조건: 참값 좌표 금지, 도면 좌표 허용)
  - 정책과 같은 입력 중 로봇 상태: 관절값 6 (observation.state), 차례의 목표(observation.environment_state). 카메라 영상은 쓰지 않는다.
  - 집게 두 손가락의 엔코더(닫힘 각도 = 두 구동 관절의 평균): 벽돌을 쥐었는지. 정책 입력은 왼쪽 손가락 하나였지만, 벽돌이 손 안에서
    치우치면 한쪽 각도만으로는 틀려(시험 벽 700300에서 확인, 11-16) 두 손가락 평균을 쓴다. 로봇 자신의 센서다.
  - 도면: 작업 순서(차례 = 자리 0 → 15), 설계 자리 x · y · z · 방향, 공급칸 중심(env.supply_pose(i), rng 없음 = 흩어짐 모름).
  - 로봇 · 공구 사양: 기구학(역기구학), 역동역학 모델(피드포워드), 서보 이득, 집게 보정값(쥔 벽돌 높이 dzp, 벽돌을 쥔 집게 각도),
    철수 자세, 벽돌 자재 무게(도면의 온장 · 반장).
  - 쓰지 않는 것: 벽돌의 실제 위치 · 자세(참값), 접촉 정보, 모르타르 상태, 검측 내부 상태. 시험(test_script_policy)이
    소스와 실행 중 호출을 모두 확인한다.

동작 (차례 k마다, 대본 wall3/expert.py의 이동 곡선과 시간을 그대로 씀)
  1. 손이 비었으면: 공급칸 k의 도면 중심 위 10 cm로 관절 이동(늦게 내려감) → 곧장 집기 높이로 내려감 → 집게 닫기 → 10 cm 들기.
     G0가 켜져 있으면 내려가기 시작할 때 G0가 넘겨받아 집고 G1을 한 뒤 돌려준다. G0 횟수(공급칸당 2번)가 끝나면 이 제어기가
     직접 집는다(정책이 직접 집던 경우와 같음). 시도는 차례당 max_attempts번, 넘으면 철수(그 벽돌은 포기).
  2. 벽돌을 쥐었으면(집게 각도가 벽돌 폭에서 멈춤): 집기 높이 위 10 cm로 곧장 올림 → 설계 자리 k 위 10 cm로 관절 이동
     (벽돌이 손 가운데에 반듯이 있다고 가정, 도면 방향) → 집게를 풀기 시작한다. G2가 여기서 넘겨받아 재고 · 맞추고 · 놓는다.
  3. 놓기를 시작한 뒤에는 철수 자세로 돌아간다(롤아웃이 철수를 보고 다음 차례로 넘어감).
검측이 손을 돌려줄 때(policy.reset)마다 지금 로봇 상태에서 계획을 다시 세운다. 차례의 기억(놓기 시작함 · 시도 횟수)은 유지한다.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import mujoco
import numpy as np

from wall3 import gripper as G
from wall3.expert import WallExpert, smooth_trap, minjerk, VP_NORM, AP_NORM
from wall3.ik import down_R, wrap
from wall3.to_lerobot import target_of

NAME = "script_20261116"


@dataclass
class ScriptConfig:
    approach: float = 0.10        # m: above the grasp height at the supply / above the place height at the wall (expert)
    t_down: float = 0.50          # s: straight descent onto the supply slot (expert)
    t_up: float = 0.40            # s: straight lift after the grasp (expert)
    t_close: float = 0.75         # s: jaw closing at the supply (expert)
    lag_supply: float = 0.30      # retract → supply: descend over the table late (expert)
    lag_wall: float = 0.35        # supply → wall: rotate away from the table before descending (expert)
    min_move: float = 0.40        # s: shortest joint move (expert)
    release_drop: float = 0.10    # rad: grip command this far below the holding command = start of the release
                                  #      (G2 takes over below holding − 0.02, gates.GateConfig.release_drop)
    t_open: float = 0.30          # s: jaw opening if no gate takes the release over
    max_attempts: int = 4         # grasp attempts per turn (G0 two per supply slot, then this controller itself)
    hold_min: float = 0.10        # rad: jaw closure (mean of both fingers) above this ...
    hold_max_frac: float = 0.5    # ... and below closure_brick + this × (holding command − closure_brick) = a brick between
                                  #     the fingers (empty: the fingers reach the holding command 0.281; brick: 0.221)

    def to_dict(self):
        return asdict(self)


class ScriptPolicy:
    """Drop-in for the learned policy in wall3.rollout.run_wall: select_action(obs) -> (1, 7) joint targets + grip, reset()."""

    def __init__(self, env, cfg: ScriptConfig | None = None):
        import torch
        self.torch = torch
        self.cfg = cfg or ScriptConfig()
        self.env = env
        self.ex = WallExpert(env)                               # IK, motion profiles, tool geometry only (never its plan)
        self.dt = env.cfg.ctrl_dt
        self.gc = env.cfg.gripper
        self.q_ret = np.array(self.ex.q_ret)
        self.dzp = float(self.ex.dzp)
        self.cg = float(self.ex.grip_close_cmd())               # holding command (= what G0 uses)
        closure_brick = float(env.cal["closure"])
        self.hold_max = closure_brick + self.cfg.hold_max_frac * (self.cg - closure_brick)
        self.targets = np.array([target_of(s) for s in env.design])          # drawing: the turn's target, as the policy sees it
        self.mass = [float(env.model.body_mass[b]) for b in env.brick_bid]   # material spec (full / half brick)
        m = env.model
        self.scratch = mujoco.MjData(m)
        self.jacp = np.zeros((3, m.nv)); self.jacr = np.zeros((3, m.nv))
        self.w3 = m.body("wrist_3_link").id
        self.grip_qadr = list(self.ex.grip_q)                   # gripper joint encoders (robot state)
        self.knuckle_qadr = [m.joint(G.PREFIX + "finger_joint").qposadr[0], m.joint(G.PREFIX + "right_outer_knuckle_joint").qposadr[0]]
        self.k = None
        self.log = []                                           # per turn: what the controller did (for the report)
        self.reset()

    # ------------------------------------------------------------------ interface
    def reset(self):
        """Called at every turn start (rollout) and whenever a gate hands the arm back: drop the current plan only."""
        self.queue = []

    def config(self):
        return {"name": NAME, **self.cfg.to_dict(), "hold_max": round(self.hold_max, 4), "holding_cmd": round(self.cg, 4)}

    def select_action(self, obs):
        st = obs["observation.state"].detach().cpu().numpy().reshape(-1).astype(float)
        q = st[:6]
        qp = self.env.data.qpos
        finger = 0.5 * (float(qp[self.knuckle_qadr[0]]) - float(qp[self.knuckle_qadr[1]]))   # jaw closure from both encoders
        k = int(np.argmin(np.abs(self.targets - obs["observation.environment_state"].detach().cpu().numpy().reshape(1, -1)).sum(1)))
        if k != self.k:                                          # a new turn (or a new wall)
            self.k, self.queue = k, []
            self.turn = {"turn": k, "attempts": 0, "release_started": False, "gave_up": False, "plans": []}
            self.log.append(self.turn)
        if not self.queue:
            self._plan(q, finger)
        P, V, A, grip, held, label = self.queue[0]
        i = self._i
        act = self._ctrl(P[i], V[i], A[i], float(grip[i]), held)
        if label == "release" and i == 0:
            self.turn["release_started"] = True
        self._i += 1
        if self._i >= len(P):
            self.queue.pop(0)
            self._i = 0
        return self.torch.from_numpy(act[None])

    # ------------------------------------------------------------------ planning (robot state + drawing only)
    def holding(self, finger):
        return self.cfg.hold_min < finger < self.hold_max

    def _plan(self, q, finger):
        c, t = self.cfg, self.turn
        p, _ = self.ex.ik.fk(q)
        segs = []
        if t["release_started"] or t["gave_up"]:
            segs += self._retract(q)
            t["plans"].append("retract")
        elif self.holding(finger):
            segs += self._transport(q)
            t["plans"].append("transport")
        elif t["attempts"] >= c.max_attempts:
            t["gave_up"] = True
            segs += self._retract(q)
            t["plans"].append("give_up")
        else:
            t["attempts"] += 1
            segs += self._grasp(q)
            t["plans"].append("grasp")
        self.queue = segs
        self._i = 0
        self._prev = q.copy()
        self._finish(q)

    def _grasp(self, q):
        """Supply slot k from the drawing: above it, straight down, close, straight up."""
        c, ex, k = self.cfg, self.ex, self.k
        pos, yaw = self.env.supply_pose(k)                       # drawing: nominal slot centre (the scatter is unknown)
        z_grasp = pos[2] + self.dzp
        z_app = z_grasp + c.approach
        segs = []
        p, R = ex.ik.fk(q)
        q_now = q
        if p[2] < z_app - 0.005 and np.hypot(*(p[:2] - pos[:2])) < 0.08:   # still low over the table (after a failed grasp)
            path = self._cart(q_now, np.r_[p[:2], z_app], R, c.t_up)
            if path is not None:
                segs.append((path, c.t_up, self.gc.open_ctrl, None, "up"))
                q_now = path[-1]
        best = None
        for yy in (yaw, yaw + np.pi):                            # 180° symmetric brick: the wrist closest to the retract pose
            qa = ex._solve(np.r_[pos[:2], z_app], down_R(yy), self.q_ret, self.q_ret[0])
            if qa is not None and (best is None or abs(qa[5] - self.q_ret[5]) < abs(best[1][5] - self.q_ret[5])):
                best = (down_R(yy), qa)
        if best is None:
            self.turn["gave_up"] = True
            return self._retract(q_now)
        R_s, q_app = best
        segs.append((self._joint(q_now, q_app, c.lag_supply), None, self.gc.open_ctrl, None, "to_supply"))
        down = self._cart(q_app, np.r_[pos[:2], z_grasp], R_s, c.t_down)
        if down is None:
            self.turn["gave_up"] = True
            return segs + self._retract(q_app)
        segs.append((down, None, self.gc.open_ctrl, None, "descend"))
        n = int(round(c.t_close / self.dt))
        close = [min(self.cg, G.ramp(self.gc, True, (j + 1) * self.dt, self.gc.open_ctrl)) for j in range(n)]
        segs.append((np.repeat(down[-1][None], n, axis=0), None, close, None, "close"))
        up = self._cart(down[-1], np.r_[pos[:2], z_app], R_s, c.t_up)
        if up is not None:
            segs.append((up, None, self.cg, k, "lift"))       # G1 fires 1.5 cm up (gates.after)
        return segs

    def _transport(self, q):
        """Brick in the hand (assumed centred and aligned): up, then over design slot k at the approach height, start the release."""
        c, ex, k = self.cfg, self.ex, self.k
        s = self.env.design[k]
        segs = []
        p, R = ex.ik.fk(q)
        q_now = q
        pos, _ = self.env.supply_pose(k)
        z_app_supply = pos[2] + self.dzp + c.approach
        if p[2] < z_app_supply - 0.005:
            path = self._cart(q_now, np.r_[p[:2], z_app_supply], R, c.t_up)
            if path is not None:
                segs.append((path, None, self.cg, k, "up"))
                q_now = path[-1]
        tgt = np.array([s["x"], s["y"], s["z"] + self.dzp + c.approach])
        cands = []
        for flip in (0.0, np.pi):
            qw = ex._solve(tgt, down_R(s["yaw"] + flip), self.q_ret, self.q_ret[0])
            if qw is not None:
                cands.append((abs(qw[5] - q_now[5]) + 0.3 * abs(qw[5] - self.q_ret[5]), qw))   # expert's wrist cost
        if not cands:
            self.turn["gave_up"] = True
            return segs + self._retract(q_now)
        q_w = min(cands, key=lambda x: x[0])[1]
        segs.append((self._joint(q_now, q_w, c.lag_wall), None, self.cg, k, "to_wall"))
        n = max(1, int(round(c.t_open / self.dt)))
        start = self.cg - c.release_drop
        opening = [max(self.gc.open_ctrl, start - (start - self.gc.open_ctrl) * (j + 1) / n) for j in range(n)]
        segs.append((np.repeat(q_w[None], n + 1, axis=0), None, [start] + opening, None, "release"))
        return segs

    def _retract(self, q):
        segs = [(self._joint(q, self.q_ret, 0.0), None, self.gc.open_ctrl, None, "to_retract")]
        segs.append((np.repeat(self.q_ret[None], 40, axis=0), None, self.gc.open_ctrl, None, "wait"))
        return segs

    # ------------------------------------------------------------------ motion (expert profiles, no stepping)
    def _joint(self, q0, q1, lag):
        """wall3.expert.WallExpert._joint_move without executing: (N, 6) joint positions."""
        ec = self.ex.cfg
        dq = q1 - q0
        vm = np.array(ec.vmax) * ec.vfrac
        am = np.array(ec.amax)
        z0, z1 = self.ex.ik.fk(q0)[0][2], self.ex.ik.fk(q1)[0][2]
        Tj = np.maximum(VP_NORM * np.abs(dq) / vm, np.sqrt(AP_NORM * np.abs(dq) / am))
        T_rot = float(np.max(Tj[[0, 4, 5]]))
        T_pitch = float(np.max(Tj[[1, 2, 3]])) / (1.0 - lag)
        T = max(self.cfg.min_move, T_rot, T_pitch)
        n = max(2, int(np.ceil(T / self.dt)))
        u = np.linspace(0.0, 1.0, n + 1)[1:]
        s = smooth_trap(u)
        if lag <= 0:
            w = u
        elif z1 < z0 - 0.02:
            w = np.clip((u - lag) / (1 - lag), 0, 1)
        elif z1 > z0 + 0.02:
            w = np.clip(u / (1 - lag), 0, 1)
        else:
            w = u
        sp = smooth_trap(w)
        prof = np.stack([s, sp, sp, sp, s, s], axis=1)
        return q0 + prof * dq

    def _cart(self, q0, p_to, R, T):
        """Straight-line TCP move (min-jerk) with IK at every control step; None if the IK fails."""
        p0, _ = self.ex.ik.fk(q0)
        n = max(1, int(np.ceil(T / self.dt)))
        path, q = [], np.asarray(q0, float)
        for s in minjerk(n):
            q, ep, er = self.ex.ik.solve_from(q, p0 + s * (np.asarray(p_to) - p0), R)
            if ep > 1e-3 or er > 1e-2:
                return None
            path.append(q)
        return np.array(path)

    def _finish(self, q):
        """Turn the planned segments into (positions, velocities, accelerations, grip, held, label), continuous across segments."""
        out, prev = [], np.asarray(q, float)
        for path, _, grip, held, label in self.queue:
            path = np.asarray(path, float)
            P = np.vstack([prev, path])
            V = np.gradient(P, self.dt, axis=0)
            A = np.gradient(V, self.dt, axis=0)
            V[-1] = 0.0; A[-1] = 0.0
            g = np.broadcast_to(np.asarray(grip, float), (len(path),))
            out.append((P[1:], V[1:], A[1:], g, held, label))
            prev = path[-1]
        self.queue = out

    def _ctrl(self, q, qd, qdd, grip, held):
        """Joint target as recorded in the demos: q + (τ_ff + kd·q̇) / kp. τ_ff = inverse dynamics of the robot (encoders only;
        bricks parked far away) + the weight of a brick assumed centred dzp below the TCP (drawing mass)."""
        env, s = self.env, self.scratch
        s.qpos[:] = env.model.qpos0
        for a in env.brick_qadr:
            s.qpos[a:a + 3] = [5.0, 5.0, -5.0]
        s.qpos[self.grip_qadr] = env.data.qpos[self.grip_qadr]  # gripper encoders
        s.qvel[:] = 0.0
        s.qacc[:] = 0.0
        s.qpos[env.arm_qadr] = q
        s.qvel[env.arm_dofadr] = qd
        s.qacc[env.arm_dofadr] = qdd
        mujoco.mj_inverse(env.model, s)
        tau = s.qfrc_inverse[env.arm_dofadr].copy()
        if held is not None:
            p, R = self.ex.ik.fk(q)
            pb = p + self.dzp * R[:, 2]                          # nominal: brick centre dzp below the TCP (TCP z = down)
            mujoco.mj_jac(env.model, s, self.jacp, self.jacr, pb, self.w3)
            tau += self.jacp[:, env.arm_dofadr].T @ np.array([0.0, 0.0, self.mass[held] * 9.81])
        ctrl = q + (tau + env.arm_kd * qd) / env.arm_kp
        return np.r_[ctrl, grip].astype(np.float64)
