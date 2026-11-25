"""G1 집기 확인 (2차 지시 나4, 실험설계 md 12절 2): 하중이 주 신호, 집게 각도가 보조 신호.

로봇 자신의 신호만 쓴다(현실성 조건 6: 고정 상태·패드 법선력·참값 자세는 쓰지 않음).
  - 발동: 집게 명령이 닫힘(명령 ≥ close_cmd_min)이 된 뒤 0.3 s 넘게 유지되고, 그 뒤 TCP가 가장 낮았던 높이보다 1.5 cm 올라감
          (관절 상태의 순기구학). 한 번 닫을 때마다 한 번.
  - 확인: 팔을 마지막 관절 목표에 멈춤 → t_settle 뒤 t_load 동안 손목 힘 센서의 연직 성분 평균 − 집게 자체 무게(tare) = 하중
          → 집게만 끝까지 닫는 명령 t_squeeze → 그 끝의 집게 각도(왼쪽 구동 관절, 관측 상태의 집게 값과 같은 것)
          → 원래 쥠 명령으로 t_restore.
  - 판정: 하중 ≥ thr_load 그리고 각도 ≤ thr_angle 이면 쥠. 문턱은 검증 배치로 정해 동결(이번에는 정하지 않음).
tare: 집게 자체 무게. 실제 로봇의 공구 하중 보정처럼, 별도 계산에서 철수 자세로 1.5 s 멈춘 집게의 연직 힘을 한 번 잰다.
"""
from dataclasses import dataclass, asdict

import mujoco
import numpy as np

from wall3.config import RETRACT_Q


@dataclass
class G1Config:
    lift: float = 0.015                 # m TCP rise after closing that triggers the check
    close_cmd_min: float = 0.2          # gripper command counted as "closed" (rad; brick contact closure is ~0.22)
    t_closed_min: float = 0.3           # s the close command must have lasted
    t_settle: float = 0.3               # s arm held still before the load window
    t_load: float = 0.2                 # s load averaging window
    t_squeeze: float = 0.2              # s full-close command for the angle
    t_restore: float = 0.1              # s back to the holding command
    t_open: float = 0.4                 # s open command after a "not grasped" result (before the policy retries)
    thr_load: float | None = None       # N, decided later on a validation batch
    thr_angle: float | None = None      # rad

    def to_dict(self):
        return asdict(self)


def tare(env, t=1.5):
    """Vertical wrist force (N, world z) of the empty gripper held still at the retract pose, in a separate MjData."""
    m = env.model
    d = mujoco.MjData(m)
    mujoco.mj_copyData(d, m, env.data)
    d.qpos[env.arm_qadr] = RETRACT_Q
    d.qvel[:] = 0.0
    d.ctrl[env.arm_act] = RETRACT_Q
    d.ctrl[env.grip_act] = [env.cfg.gripper.open_ctrl, -env.cfg.gripper.open_ctrl]
    d.eq_active[[e for e in env.grip_eq]] = 0
    n = int(round(t / m.opt.timestep))
    fz = []
    for i in range(n):
        mujoco.mj_step(m, d)
        if i >= n - 100:
            R = d.site_xmat[env.ft_site].reshape(3, 3)
            fz.append((R @ d.sensordata[env.ft_force:env.ft_force + 3])[2])
    return float(np.mean(fz))


class G1Trigger:
    """Fires once per closing when the TCP has risen `lift` above its lowest point since the close command."""

    def __init__(self, cfg: G1Config, dt: float):
        self.cfg, self.dt = cfg, dt
        self.reset()

    def reset(self):
        self.z_min, self.n_closed, self.armed = None, 0, True

    def update(self, grip_cmd, tcp_z):
        c = self.cfg
        if grip_cmd < c.close_cmd_min:
            self.reset()
            return False
        self.n_closed += 1
        self.z_min = tcp_z if self.z_min is None else min(self.z_min, tcp_z)
        if self.armed and self.n_closed * self.dt >= c.t_closed_min and tcp_z >= self.z_min + c.lift:
            self.armed = False
            return True
        return False


class G1:
    def __init__(self, env, cfg: G1Config | None = None):
        self.env = env
        self.cfg = cfg or G1Config()
        self.f0 = tare(env)

    def measure(self, q_hold, grip_hold, on_step=None):
        """Run the check holding the arm at q_hold. on_step(label) is called after every control step (recording)."""
        env, c = self.env, self.cfg
        dt = env.cfg.ctrl_dt
        n = lambda t: max(1, int(round(t / dt)))

        def step(grip, label):
            env.step(q_hold, grip)
            if on_step is not None:
                on_step(label)

        for _ in range(n(c.t_settle)):
            step(grip_hold, "g1_settle")
        fz = []
        for _ in range(n(c.t_load)):
            step(grip_hold, "g1_load")
            fz.append(env.wrist_ft()[0][2])
        load = float(np.mean(fz) - self.f0)
        for _ in range(n(c.t_squeeze)):
            step(env.cfg.gripper.close_ctrl, "g1_squeeze")
        angle = float(env.data.joint("g_finger_joint").qpos[0])
        for _ in range(n(c.t_restore)):
            step(grip_hold, "g1_restore")
        return {"load_N": load, "angle_rad": angle, "fz_samples": [float(x) for x in fz], "tare_N": self.f0,
                "steps": n(c.t_settle) + n(c.t_load) + n(c.t_squeeze) + n(c.t_restore)}

    def decide(self, r):
        c = self.cfg
        if c.thr_load is None or c.thr_angle is None:
            raise ValueError("G1 thresholds are not set")
        return bool(r["load_N"] >= c.thr_load and r["angle_rad"] <= c.thr_angle)
