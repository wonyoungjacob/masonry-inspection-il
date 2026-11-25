"""UR10e 6자유도 감쇠 최소자승 IK (TCP 위치 + 방향). brick_sim.ik.DLSIK의 6관절 판.

자세 조건(지시서 단계 2): 팔꿈치 15° 이상 굽힘, 손목 특이점 회피(wrist_2가 0·π에서 15° 이상), 어깨 특이점 회피
(손목 중심이 베이스 축에서 0.10 m 이상). 여러 시작값에서 풀고 조건을 만족하는 해 중 기준 자세에 가장 가까운 해를 고른다.
"""
import mujoco
import numpy as np

ELBOW_MIN = np.deg2rad(15.0)
PAN_OFFSET = np.deg2rad(160.0)     # shoulder_pan ≈ target azimuth + 160° when the arm faces the target (this base mounting)
PAN_WINDOW = np.deg2rad(60.0)
WRIST2_MIN = np.deg2rad(15.0)
SHOULDER_MIN_R = 0.10


def down_R(yaw):
    """TCP rotation: approach (z) = world −z, closing axis (y) horizontal; x = (cos yaw, sin yaw, 0) = brick long axis."""
    x = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    z = np.array([0.0, 0.0, -1.0])
    return np.stack([x, np.cross(z, x), z], axis=1)


def rot_err(R, Rt):
    return 0.5 * (np.cross(R[:, 0], Rt[:, 0]) + np.cross(R[:, 1], Rt[:, 1]) + np.cross(R[:, 2], Rt[:, 2]))


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class UR10eIK:
    def __init__(self, model, site="tcp", qadr=None, dofadr=None, damping=1e-3):
        self.m = model
        self.d = mujoco.MjData(model)
        self.site = model.site(site).id
        names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        self.qadr = np.array([model.joint(n).qposadr[0] for n in names]) if qadr is None else qadr
        self.dofadr = np.array([model.joint(n).dofadr[0] for n in names]) if dofadr is None else dofadr
        self.lo = model.jnt_range[[model.joint(n).id for n in names], 0]
        self.hi = model.jnt_range[[model.joint(n).id for n in names], 1]
        self.wrist1_body = model.body("wrist_1_link").id
        self.base_body = model.body("base").id
        self.lam2 = damping
        self.jp = np.zeros((3, model.nv)); self.jr = np.zeros((3, model.nv))

    def fk(self, q):
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        return self.d.site_xpos[self.site].copy(), self.d.site_xmat[self.site].reshape(3, 3).copy()

    def solve_from(self, q0, p_t, R_t, iters=200, tol_p=1e-5, tol_r=1e-4):
        m, d = self.m, self.d
        q = np.array(q0, float)
        for _ in range(iters):
            p, R = self.fk(q)
            ep, er = p_t - p, rot_err(R, R_t)
            if np.linalg.norm(ep) < tol_p and np.linalg.norm(er) < tol_r:
                break
            mujoco.mj_comPos(m, d)
            mujoco.mj_jacSite(m, d, self.jp, self.jr, self.site)
            J = np.vstack([self.jp[:, self.dofadr], self.jr[:, self.dofadr]])
            e = np.r_[ep, er]
            dq = J.T @ np.linalg.solve(J @ J.T + self.lam2 * np.eye(6), e)
            n = np.linalg.norm(dq)
            if n > 0.4:
                dq *= 0.4 / n
            q = np.clip(q + dq, self.lo, self.hi)
        p, R = self.fk(q)
        return q, float(np.linalg.norm(p_t - p)), float(np.linalg.norm(rot_err(R, R_t)))

    def margins(self, q):
        """Joint-condition margins (positive = satisfied): elbow bend, wrist_2 away from 0/π, wrist centre off the base axis."""
        self.fk(q)
        wc = self.d.xpos[self.wrist1_body]; bc = self.d.xpos[self.base_body]
        return dict(elbow=float(abs(q[2]) - ELBOW_MIN),
                    wrist2=float(abs(np.sin(q[4])) - np.sin(WRIST2_MIN)),
                    shoulder=float(np.hypot(wc[0] - bc[0], wc[1] - bc[1]) - SHOULDER_MIN_R))

    def branch_ok(self, q, p_t):
        """One posture family for every pose: the arm faces the target (pan within ±60° of azimuth + 160°),
        shoulder lift in (−π, 0) and elbow bent positive (elbow up). Excludes shoulder-flip solutions over the base."""
        az = np.arctan2(p_t[1], p_t[0])
        return abs(wrap(q[0] - az - PAN_OFFSET)) < PAN_WINDOW and -np.pi < q[1] < 0 and q[2] > 0

    def solve(self, p_t, R_t, q_ref, seeds=None, tol_p=2e-4, tol_r=2e-3):
        """Best solution near q_ref that meets the joint conditions and the posture family; (q, info) or (None, info)."""
        if seeds is None:
            seeds = [q_ref]
        best, info = None, {"tried": 0}
        for s in seeds:
            q, ep, er = self.solve_from(s, p_t, R_t)
            info["tried"] += 1
            if ep > tol_p or er > tol_r:
                continue
            mg = self.margins(q)
            if min(mg.values()) < 0 or not self.branch_ok(q, p_t):
                continue
            cost = float(np.sum(wrap(q - q_ref) ** 2))
            if best is None or cost < best[1]:
                best = (q, cost, ep, er, mg)
        if best is None:
            return None, info
        q, cost, ep, er, mg = best
        info.update(pos_err=ep, rot_err=er, margins=mg, cost=cost)
        return q, info
