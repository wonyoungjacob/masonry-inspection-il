"""robosuite Robotiq 2F-140 집게를 MjSpec에 붙이는 조립 코드 (실험설계 md v9 P1 + 사용자 결정 A1+).

모델 파일(sim/third_party/robosuite_gripper140/robotiq_gripper_140.xml)은 고치지 않는다. 조립할 때 아래만 적용한다.
  ① 집게 몸체끼리 접촉 끄기(contact exclude). 벽돌·벽·탁자와의 접촉은 그대로.
     MuJoCo Menagerie 2F-85(2f85.xml)도 base–driver, base–spring_link, coupler–follower 사이를 <exclude>로 끈다.
  ② 쥐는 힘: 손가락 구동기(position) 이득과 토크 한계. 실제 2F-140 쥐는 힘 사양(약 10 ~ 125 N) 안.
  ③ 집는 높이: 이 파일이 아니라 대본에서 정한다(calibrate()가 닫힌 형상을 잰다).
  A1+ (2026-10-15 사용자 결정, 단계 1 재개 2차)
  ④ 손가락 링크 구속(tendon 등식)의 강성: Menagerie 2F-85와 같은 solref 0.005 1, solimp 0.95 0.99 0.001.
  ⑤ 패드 평행 결합: robosuite 결합 비율(1.5, 3.5)은 닫을수록 패드가 기울어(90 mm에서 약 9°) 모서리로 닿는다.
     실제 2F-140처럼 패드가 평행하도록 inner_finger = ∓driver, inner_knuckle = 평행사변형 비율로 바꾼다.
그 밖에 robosuite가 이 파일을 합치는 base.xml의 설정(radian 단위, inertiagrouprange 0 0)을 맞춘다.
모양·치수·질량·관성·패드 마찰(기본 1.0)은 바꾸지 않는다.
"""
from dataclasses import dataclass, asdict
from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np

SIM = Path(__file__).resolve().parents[1]
GRIPPER_XML = SIM / "third_party/robosuite_gripper140/robotiq_gripper_140.xml"
PREFIX = "g_"
PADS = ("left_fingerpad_collision", "right_fingerpad_collision")

# tendon name -> [(joint, coef), ...]  (fixed tendons; the equality holds each length at 0)
COUPLING_ROBOSUITE = {
    "finger2_12_cpl": [("finger_joint", 1.0), ("left_inner_finger_joint", 1.5)],
    "finger2_23_cpl": [("left_inner_finger_joint", 1.0), ("left_inner_knuckle_joint", 3.5)],
    "finger1_12_cpl": [("right_outer_knuckle_joint", 1.0), ("right_inner_finger_joint", -1.5)],
    "finger1_23_cpl": [("right_inner_finger_joint", 1.0), ("right_inner_knuckle_joint", 3.5)],
}
COUPLING_PARALLEL = {                                   # ⑤ pads stay parallel over the whole stroke (checked: tilt 0.0°)
    "finger2_12_cpl": [("finger_joint", 1.0), ("left_inner_finger_joint", -1.0)],
    "finger2_23_cpl": [("left_inner_finger_joint", 1.0), ("left_inner_knuckle_joint", 1.0)],
    "finger1_12_cpl": [("right_outer_knuckle_joint", 1.0), ("right_inner_finger_joint", 1.0)],
    "finger1_23_cpl": [("right_inner_finger_joint", 1.0), ("right_inner_knuckle_joint", 1.0)],
}


@dataclass
class GripperConfig:
    exclude_self: bool = True        # ①
    kp: float = 20.0                 # ② position gain (robosuite 20)
    kv: float = 0.0                  #    velocity damping of the position servo (robosuite 0)
    torque_limit: float = 0.0        # ② |actuator torque| limit, N·m (0 = robosuite: no limit)
    close_ctrl: float = 0.7          # closing command (robosuite ctrlrange upper end)
    open_ctrl: float = 0.0
    eq_solref: tuple = ()            # ④ finger coupling equality solref. () = as received (MuJoCo default 0.02 1)
    eq_solimp: tuple = ()            # ④ same, solimp. () = as received (0.9 0.95 0.001)
    parallel_pads: bool = False      # ⑤
    close_time: float = 0.0          # s, command ramp open → close (0 = step command). Real 2F-140 closes at 30 ~ 250 mm/s;
                                     # a step command slams the light fingers shut in ~5 ms (≈ 7 m/s) and pops the brick up.
    open_time: float = 0.0           # s, command ramp close → open

    def to_dict(self):
        return asdict(self)


def a1plus(torque_limit: float, kp: float = 100.0, kv: float = 1.0, close_time: float = 0.5, open_time: float = 0.3) -> GripperConfig:
    """A1+ configuration (①②④⑤ applied; ③ is done by the expert). Command ramps: closing 0.5 s ≈ 105 mm/s per pad."""
    return GripperConfig(kp=kp, kv=kv, torque_limit=torque_limit, eq_solref=(0.005, 1.0),
                         eq_solimp=(0.95, 0.99, 0.001), parallel_pads=True, close_time=close_time, open_time=open_time)


def ramp(cfg: GripperConfig, close: bool, t: float, start: float | None = None) -> float:
    """finger_1 command t seconds after a close/open order (start = command when the order was given)."""
    if close:
        s0 = cfg.open_ctrl if start is None else start
        T = cfg.close_time
        return cfg.close_ctrl if T <= 0 else float(min(cfg.close_ctrl, s0 + (cfg.close_ctrl - cfg.open_ctrl) * t / T))
    s0 = cfg.close_ctrl if start is None else start
    T = cfg.open_time
    return cfg.open_ctrl if T <= 0 else float(max(cfg.open_ctrl, s0 - (cfg.close_ctrl - cfg.open_ctrl) * t / T))


def _gripper_spec(cfg: GripperConfig) -> "mujoco.MjSpec":
    gr = mujoco.MjSpec.from_file(str(GRIPPER_XML))
    gr.compiler.degree = False                     # robosuite base.xml: <compiler angle="radian">
    gr.compiler.inertiagrouprange = [0, 0]         # robosuite base.xml
    for a in gr.actuators:                         # ②
        a.gainprm[0] = cfg.kp
        a.biasprm[1] = -cfg.kp
        a.biasprm[2] = -cfg.kv
        if cfg.torque_limit > 0:
            a.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
            a.forcerange = [-cfg.torque_limit, cfg.torque_limit]
    if cfg.parallel_pads:                          # ⑤ re-create the four coupling tendons with new coefficients
        old = {t.name: t for t in gr.tendons}
        for name, wraps in COUPLING_PARALLEL.items():
            gr.delete(old[name])
            t = gr.add_tendon(name=name)
            for j, c in wraps:
                t.wrap_joint(j, c)
    for e in gr.equalities:                        # ④
        if cfg.eq_solref:
            e.solref = list(cfg.eq_solref)
        if cfg.eq_solimp:
            e.solimp = (list(cfg.eq_solimp) + [0.5, 2.0])[:5]
    return gr


def attach(parent_site: "mujoco.MjsSite", spec: "mujoco.MjSpec", cfg: GripperConfig):
    """Attach the gripper to `parent_site` of `spec` (gripper +z = approach direction = site +z). Returns the frame."""
    gr = _gripper_spec(cfg)
    frame = parent_site.attach_body(gr.body("right_gripper"), PREFIX, "")
    if cfg.exclude_self:                           # ①
        names = [b.name for b in spec.bodies if b.name.startswith(PREFIX)]
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                e = spec.add_exclude()
                e.bodyname1, e.bodyname2 = a, b
    return frame


def grip_ctrl(close: bool, cfg: GripperConfig):
    """Control values for (finger_1, finger_2)."""
    v = cfg.close_ctrl if close else cfg.open_ctrl
    return [v, -v]


def closure(d) -> float:
    """Gripper closure angle (rad) = mean of the two driven knuckle angles (left +, right −). Used by G1."""
    return 0.5 * (float(d.joint(PREFIX + "finger_joint").qpos[0]) - float(d.joint(PREFIX + "right_outer_knuckle_joint").qpos[0]))


def pad_gap(m, d, site: str) -> float:
    """Inner gap between the two pad faces along the closing axis (y) of `site` (m)."""
    R = d.site(site).xmat.reshape(3, 3); o = d.site(site).xpos
    L = (_geom_points(m, d, m.geom(PREFIX + PADS[0]).id) - o) @ R
    Rr = (_geom_points(m, d, m.geom(PREFIX + PADS[1]).id) - o) @ R
    return float(Rr[:, 1].min() - L[:, 1].max())


def _geom_points(m, d, gid):
    """World points spanning a geom: box corners or mesh vertices."""
    R = d.geom_xmat[gid].reshape(3, 3); p = d.geom_xpos[gid]
    if m.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH:
        mid = m.geom_dataid[gid]
        v = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
    else:
        s = m.geom_size[gid]
        v = np.array([[x * s[0], y * s[1], z * s[2]] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    return p + v @ R.T


@lru_cache(maxsize=32)
def _calibrate(cfg_items, width):
    cfg = GripperConfig(**dict(cfg_items))
    s = mujoco.MjSpec(); s.option.timestep = 0.001; s.option.gravity = [0, 0, 0]
    site = s.worldbody.add_body(name="base", pos=[0, 0, 1]).add_site(name="mount")   # approach = site +z = world +z
    attach(site, s, cfg)
    m = s.compile(); d = mujoco.MjData(m)
    pads = [m.geom(PREFIX + p).id for p in PADS]
    fingers = [g for g in range(m.ngeom) if m.geom(g).name.startswith(PREFIX) and "finger" in m.geom(g).name
               and (m.geom_contype[g] or m.geom_conaffinity[g])]
    for _ in range(1000):
        mujoco.mj_step(m, d)
    out = {"open_inner_gap": pad_gap(m, d, "mount")}
    for c in np.linspace(0.0, cfg.close_ctrl, 701):      # quasi-static closing sweep, no object
        d.ctrl[:] = [c, -c]
        for _ in range(30):
            mujoco.mj_step(m, d)
        if pad_gap(m, d, "mount") <= width:
            pc = 0.5 * (d.geom_xpos[pads[0]] + d.geom_xpos[pads[1]])
            out.update(closure=closure(d), gap=pad_gap(m, d, "mount"), pad_center_depth=float(pc[2] - 1.0),
                       deepest_finger_depth=float(max(_geom_points(m, d, g)[:, 2].max() for g in fingers) - 1.0),
                       pad_half_length=float(m.geom_size[pads[0]][1]))
            return out
    raise RuntimeError("gripper does not close to %.3f m" % width)


def calibrate(cfg: GripperConfig, width: float = 0.090) -> dict:
    """Closed geometry when the pads just touch an object `width` wide (m): closure angle, distance from the mount to the pad
    centre and to the deepest finger point along the approach axis (m), and the open inner gap (m)."""
    return dict(_calibrate(tuple(sorted(cfg.to_dict().items())), float(width)))
