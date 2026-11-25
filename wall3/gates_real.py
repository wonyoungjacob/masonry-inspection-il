"""현실 제약 실험의 검측 (2026-11-16, 01_사기설/우분투지시_현실제약_ablation_20261116.md 3 · 4절).

Full + G2t(wall3/gates_carry.py)에서 바꾼 것. 하이브리드 · 제어만 8조건에 똑같이 쓴다(공정 규칙).
  G0v   G0(도면 공급칸 중심에서 집기) 대신 손목 카메라로 집기. 넘겨받는 때는 G0와 같다(gates._g0_slot: 집게를 연 채
        공급칸 6 cm 안, 접근 높이 + 3 cm 아래로 내려옴). 넘겨받으면 곧장 접근 높이로 올린 뒤 손목 카메라(640 × 480 RGB-D)로
        손 아래 8 cm 안의 공급 벽돌(윗면 높이 = 공급대 + H ± 15 mm)을 재고, 잰 중심 · 방향으로 집는다. 못 보면 도면의 칸 중심 위로
        가서 한 번 더 본다. 그래도 못 보면 집지 않고 돌려준다(시도 1회). 측정 모듈은 작업대 카메라와 같은 perception.py.
  피드포워드  검측 동작의 쥔 벽돌 무게 보상에 벽돌 참값 위치 대신 도면상 위치(TCP 아래 dzp)와 도면 무게를 쓴다.
  A     설계 자리 = 믿는 도면(A1), 카메라 = 보정 오차 · 깊이 잡음이 든 측정 모듈(A2 · A3 · A4), G1 힘 = 잡음(A5, rollout에서 설치).
참값은 기록(audit_*)에만 쓴다.
"""
from __future__ import annotations

import types

import mujoco
import numpy as np

from wall3 import gripper as G
from wall3.config import H
from wall3.gates_carry import WallGatesCarry
from wall3.realism import NoisyEstimator, belief_design
from wall3.poses import yaw_of_quat
from wall3.perception import _fold

G0V_R = 0.08            # m: a supply brick within this of the hand (xy) is the one to grasp
G0V_DZ = 0.015          # m: its top within this of the nominal supply-brick top height


def _ff_nominal(ex, q, qd, qdd, held):
    """WallExpert._ff with the held brick assumed centred dzp below the TCP (drawing mass), not at its true position."""
    env, s = ex.env, ex.scratch
    s.qpos[:] = env.data.qpos
    s.qvel[:] = 0.0
    s.qacc[:] = 0.0
    s.qpos[env.arm_qadr] = q
    s.qvel[env.arm_dofadr] = qd
    s.qacc[env.arm_dofadr] = qdd
    for a in env.brick_qadr:
        s.qpos[a:a + 3] = [5.0, 5.0, -5.0]
    mujoco.mj_inverse(env.model, s)
    tau = s.qfrc_inverse[env.arm_dofadr].copy()
    if held is not None:
        p, R = ex.ik.fk(q)
        pb = p + ex.dzp * R[:, 2]
        mujoco.mj_jac(env.model, s, ex.jacp, ex.jacr, pb, ex.w3)
        tau += ex.jacp[:, env.arm_dofadr].T @ np.array([0.0, 0.0, env.model.body_mass[env.brick_bid[held]] * 9.81])
    return tau


class WallGatesReal(WallGatesCarry):
    def __init__(self, env, policy=None, **kw):
        super().__init__(env, policy, **kw)
        if self.cam is not None:
            self.cam.close()
            self.cam = NoisyEstimator(env, "top", 960, 1280)
        self.wcam = NoisyEstimator(env, "wrist", 480, 640, z_min=env.cfg.table_h + 0.02)   # above the wooden table top
        self.ex._ff = types.MethodType(_ff_nominal, self.ex)
        self.design_belief = [dict(d) for d in env.design]
        self.pert = None

    def set_perturb(self, p, rng_top, rng_wrist):
        self.pert = p
        self.design_belief = belief_design(self.env.design, p)
        self.cam.set_noise(p.top_t, p.top_r, p.depth_top, p.depth_pix, rng_top)
        self.wcam.set_noise(p.wrist_t, p.wrist_r, p.depth_wrist, p.depth_pix, rng_wrist)

    def _design(self, k):
        s = self.design_belief[k]
        return np.array([s["x"], s["y"]]), float(s["yaw"]), s["L"], s

    # ------------------------------------------------------------------ G0v
    def _g0v_look(self, i):
        env = self.env
        p, _ = env.tcp()
        L = self.design_belief[i]["L"]                     # drawing: supply slot i holds the brick of design slot i
        zt = env.cfg.table_h + H
        best, bd = None, G0V_R
        for det in self.wcam.detect(force=True):
            if not (zt - G0V_DZ <= det["z_top"] <= zt + G0V_DZ):
                continue
            pp = self.wcam.pose_of(det, L=L, yaw_prior=0.0)
            dd = float(np.hypot(*(pp["pos"][:2] - p[:2])))
            if dd < bd:
                best, bd = pp, dd
        return best

    def _g0_grasp(self, i):
        env, ex = self.env, self.ex
        gc = env.cfg.gripper
        go = gc.open_ctrl
        self.g0_tries[i] += 1
        pos_d, z_grasp_d, z_app = self._g0_heights(i)       # drawing: slot centre, table height
        p0, _ = env.tcp()
        ev = {"t": int(env.t), "supply_slot": int(i), "turn": int(self.k), "try": self.g0_tries[i], "vision": True,
              "tcp_off_mm": round(float(np.hypot(*(p0[:2] - pos_d[:2]))) * 1000, 1),
              "tcp_z_above_grasp_mm": round((p0[2] - z_grasp_d) * 1000, 1)}
        self._sync_expert(go)
        p, R = env.tcp()
        if p[2] < z_app:
            ex._cart_move(np.r_[p[:2], z_app], R, 0.3, go, None, "g0v_up")
            ex.fail = None
        det, where = self._g0v_look(i), "here"
        if det is None and self._move_tcp(np.r_[pos_d[:2], z_app], 0.0, go, None, "g0v_to_slot", T=0.5):
            det, where = self._g0v_look(i), "slot"
        ev["seen"] = where if det is not None else None
        if det is None:
            ev["result"] = "not_seen"
            self.ev["g0"].append(ev)
            self._handback()
            return
        pos_m, yaw_m = np.asarray(det["pos"], float), float(det["yaw"])
        bp, bq = env.brick_pose(i)                            # audit only
        ev.update(meas_full=bool(det["full"]), audit_meas_err_mm=round(float(np.hypot(*(pos_m[:2] - bp[:2]))) * 1000, 2),
                  audit_meas_yaw_err_deg=round(float(np.degrees(_fold(yaw_m - yaw_of_quat(bq)))), 2),
                  audit_meas_z_err_mm=round(float(pos_m[2] - bp[2]) * 1000, 2))
        z_grasp = float(pos_m[2]) + ex.dzp
        if not self._move_tcp(np.r_[pos_m[:2], z_grasp + ex.cfg.approach], yaw_m, go, None, "g0v_align", T=0.5):
            ev["result"] = "ik_failed"
            self.ev["g0"].append(ev)
            self._handback()
            return
        p, R = env.tcp()
        ex._cart_move(np.r_[pos_m[:2], z_grasp], R, ex.cfg.t_down, go, None, "g0v_descend")
        ex.fail = None
        n = int(round(ex.cfg.t_close / self.dt))
        cg = ex.grip_close_cmd()
        ex._hold(n, [min(cg, G.ramp(gc, True, (j + 1) * self.dt, go)) for j in range(n)], None, "g0v_close")
        p, R = env.tcp()
        ex._cart_move(p + [0, 0, self.g1cfg.lift], R, 0.3, cg, None, "g0v_lift")
        ex.fail = None
        a = np.asarray(env.last_action, float).copy()
        ng1 = len(self.ev["g1"])
        self.trig.reset()
        self._g1_moment(a)                                    # G1 check, then G2t carries if grasped
        if self.g1 is None:
            self._handback()
        ev["g1_grasped"] = self.ev["g1"][ng1]["grasped"] if len(self.ev["g1"]) > ng1 else None
        ev["result"] = "done"
        self.ev["g0"].append(ev)
