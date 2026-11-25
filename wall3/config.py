"""3층 벽체 파일럿 설정 (실험설계 md 10절, 지시서 3절). 모든 값은 실행 결과와 함께 저장한다(to_dict).

좌표: 세계 원점 = 로봇 받침대 아래 바닥 중심. x = 벽 길이 방향, y = 벽면에 수직(+y가 로봇에서 벽 쪽), z = 연직 위.
벽 중심선 y = WALL_Y(0.6 m), 벽 중앙 x = 0. 벽의 '앞면'(평가지표 md의 y_f)은 로봇 반대쪽 면(y = WALL_Y + W/2)이며
앞 카메라가 이쪽에서 벽을 본다. 공급대는 로봇 뒤쪽(y < 0).
"""
from dataclasses import dataclass, field, asdict

import numpy as np

from wall3 import gripper as G

# ---- brick (P2) and wall (평가지표 md 1절) -------------------------------------------------------------------------
L_FULL, L_HALF, W, H = 0.190, 0.090, 0.090, 0.057
RHO = 2000.0                    # kg/m^3 (v8)
MU_BRICK = 0.6                  # brick friction (floor and table use the same value)
JOINT = 0.010                   # bed and head joint (KCS 41 34 02 3.4.2(1))
WALL_Y = 0.60                   # wall centre line, 0.6 m in front of the base (P4)


def design_slots():
    """16 design slots in laying order (P7: course 1 L→R, course 2 L→R with half bricks at the ends, course 3 L→R).
    Each: dict(course, index, x, y, z, yaw, L, half)."""
    out = []
    pitch = L_FULL + JOINT
    xs_full = [(-2 + i) * pitch for i in range(5)]                      # -0.4 .. 0.4
    z = [H / 2, H / 2 + H + JOINT, H / 2 + 2 * (H + JOINT)]
    x_half = 0.5 * (5 * L_FULL + 4 * JOINT) - L_HALF / 2               # 0.45
    xs_c2 = [-x_half] + [(-1.5 + i) * pitch for i in range(4)] + [x_half]
    for k, xs in enumerate([xs_full, xs_c2, xs_full]):
        for i, x in enumerate(xs):
            half = (k == 1 and i in (0, len(xs) - 1))
            out.append(dict(course=k + 1, index=i, x=float(x), y=WALL_Y, z=float(z[k]), yaw=0.0,
                            L=L_HALF if half else L_FULL, half=half))
    return out


# ---- supply table (P5): set from the reach map (단계 2, wall3/reach_map.py) ---------------------------------------
# slot centre (x, y) on the table; slot i holds the brick for design slot i. Brick long axis along x.
# Chosen by reach_map.py (2026-10-15, outputs/wall3/reach_map_try3): 16 of 18 grid candidates (rows y = -0.25 … -0.79,
# columns 0.26 m apart, base distance 0.55 ~ 0.95 m) assigned to the wall slots to minimise the pan travel.
SUPPLY_SLOTS = ((-0.91, -0.25), (-0.65, -0.25), (+0.91, -0.25), (+0.65, -0.25), (+0.65, -0.43), (-0.65, -0.43), (-0.39, -0.43), (-0.39, -0.79), (+0.65, -0.61), (+0.39, -0.43), (+0.39, -0.61), (-0.65, -0.61), (-0.39, -0.61), (+0.39, -0.79), (+0.13, -0.79), (+0.13, -0.61))


# 철수 자세 (reach_map.py): TCP (0, 0.40, 0.75) m, gripper down. Every brick starts and ends here.
RETRACT_TCP = (0.0, 0.40, 0.75)
RETRACT_Q = (-2.020805, -2.131279, 2.263128, -1.702647, -1.570801, -0.450007)


@dataclass
class SceneConfig:
    # environment version (2차 지시 나1). Build with scene_config(version); the dataclass defaults are main_v1.
    env_version: str = "main_v1"
    brick_rgba: tuple = ((0.64, 0.29, 0.21, 1.0), (0.58, 0.26, 0.19, 1.0))   # 적벽돌 색 (2026-10-16), two shades
    ctrl_interp: str = "linear"                 # how a 20 Hz target is spread over the 50 physics steps (나2):
                                                # linear | hermite (C1: tangents from the previous action) | lowpass
    integrator: str = "implicitfast"            # UR10e Menagerie file's integrator | euler | implicit | rk4 (나2)
    arm_damping: str = "actuator"               # UR10e servo damping kd 500 in the actuator (file) | joint: same kd as joint
                                                # damping (Euler integrates joint damping implicitly) (나2)
    arm_kd: tuple = ()                          # per-joint actuator damping override (N·m·s/rad), () = file values (500 each)
    arm_joint_damping: tuple = ()               # per-joint passive damping override, () = file values (10 10 5 2 2 2)
    mortar_stiffness: str = "brick_sim"         # "brick_sim": k = g / sag (pilot, ignores the constraint impedance);
                                                # main_v1 too: 나3 correction reverted by its rule (나5 recheck 5/10 < 80 %, 2026-10-17);
                                                # "impedance": k = (1 - d)/d · g / sag so the bed really sags `mortar_sag` (나3)
    # grip lock (나2 결정 2026-10-17): weld the held brick to the gripper only while real friction could hold it
    grip_lock: bool = True                      # pilot_20261016: False
    lock_mu: float = 0.6                        # ② ③ friction used for the capacity (brick μ, smaller than the pad's 1.0)
    lock_amax: float = 7.4                      # ② m/s², largest TCP acceleration during transport (나2 diagnosis)
    lock_vlin: float = 0.005                    # ① m/s, brick ↔ gripper relative speed below which the grasp has settled
    lock_vang: float = 0.05                     # ① rad/s
    lock_settle: float = 0.1                    # ① s the relative speed must stay below the limits
    lock_cap_window: int = 10                   # ③ twist limit = max over the last 10 physics steps (10 ms; 추가 지시 2)
    lock_open_eps: float = 0.002                # ⑤ rad: finger command below the finger angle by this = opening order
    lock_drop: float = 0.01                     # ⑤ rad: finger command dropping this far below its max since closing = opening
    lock_solref: tuple = (0.004, 1.0)           # grip weld stiffness (as the cure weld)
    # robot (P3)
    pedestal_h: float = 0.60
    pedestal_half: float = 0.15                 # square pedestal half-size (m)
    # supply table (P5)
    table_h: float = 0.35                       # table top height (m)
    table_x: tuple = (-1.00, 1.00)
    table_y: tuple = (-1.00, -0.17)
    supply_slots: tuple = SUPPLY_SLOTS          # ((x, y), ...) x16, design-slot order
    scatter_pos: float = 0.010                  # P6: +-10 mm
    scatter_yaw_deg: float = 10.0               # P6: +-10 deg
    # mortar (mortarB as in brick_sim, scaled to a 10 mm joint)
    mortar_sag: float = 0.0005 * 0.010 / 0.004737   # 1.06 mm
    mortar_zeta: float = 1.0
    mortar_friction: float = 0.75
    bed_overhang: float = 0.002
    cure_settle_steps: int = 40                 # 2 s at 20 Hz: bed laying (next course) waits for this, and the brick cure below
    # brick cure (mortar set) time. None = cure_settle_steps (2 s, main_v1 as frozen). The evaluation environment
    # (2026-10-22, 사용자 결정) uses a real mortar's initial setting time: placed bricks stay movable on the fresh bed for
    # the whole wall (they rest on the compliant bed with friction), and the next course is laid as before (2 s still).
    cure_time_s: float | None = None
    hand_clear: float = 0.05                    # TCP farther than this from the brick centre = hand off the brick
    # bed laying rule. "in_slot" (main_v1, frozen): every brick of the course sits in its design slot within the slot_tol_*
    # tolerances below. "placed" (2026-10-30, 사용자 결정): every brick of the course has left the supply table, however
    # crooked; the bed is not laid while a slab would sink into a brick of the course above (one already put on the bare course).
    bed_rule: str = "in_slot"
    # bed laying check (brick_sim brick_in_slot tolerances)
    slot_tol_xy: float = 0.008
    slot_tol_z: float = 0.003
    slot_tol_yaw_deg: float = 10.0
    slot_tol_tilt_deg: float = 10.0
    # physics (P8, 10.2) — solver options as brick_sim/scene.py (지난 실험)
    timestep: float = 0.001
    ctrl_dt: float = 0.05                       # 20 Hz
    impratio: float = 10.0
    noslip_iterations: int = 3
    # cameras (P9, 10.3)
    img: int = 128
    hi_w: int = 1280
    hi_h: int = 960
    # gripper (P1, A1+)
    gripper: G.GripperConfig = field(default_factory=lambda: G.a1plus(4.0))
    grasp_clear: float = 0.005                  # ③ deepest finger point above the brick bottom (2 mm let the fingertips touch the table: pads sink ~2 mm into the brick under 50 N and keep descending as they close)

    def __post_init__(self):
        if self.bed_rule not in ("in_slot", "placed"):
            raise ValueError(f"unknown bed_rule {self.bed_rule!r}")

    def to_dict(self):
        d = asdict(self)
        if d.get("cure_time_s") is None:        # keeps the frozen main_v1 config dict (and its hash) unchanged
            d.pop("cure_time_s", None)
        if d.get("bed_rule") == "in_slot":      # same: the versions before 2026-10-30 keep their dict and hash
            d.pop("bed_rule", None)
        return d


# Environment versions. pilot_20261016 reproduces the pilot collection exactly (gray bricks, linear target interpolation,
# brick_sim mortar stiffness). main_v1 = the dataclass defaults, filled in step by step by 나2 ~ 나4 and frozen in 나5.
ENV_VERSIONS = {
    "pilot_20261016": dict(brick_rgba=((0.62, 0.62, 0.60, 1.0), (0.58, 0.58, 0.56, 1.0)), ctrl_interp="linear",
                           mortar_stiffness="brick_sim", integrator="implicitfast", arm_damping="actuator", grip_lock=False),
    "main_v1": dict(),
    # 평가용 (2026-10-22, 사용자 결정): main_v1 + 실제 모르타르처럼 벽 하나 쌓는 동안 굳지 않음. 시멘트 모르타르 초결은 보통 수 시간
    # (KS L 5103 / ASTM C191 계열 시험, 조적 모르타르 가사 시간 약 2 ~ 2.5 h) → 2 h. 정책 학습 데이터(main_v1, 2 s 경화)와 다른 점은 이것뿐
    "main_v1_eval": dict(cure_time_s=7200.0),
    # 2026-10-30 (사용자 결정): main_v1_eval + 모르타르 판을 아래 단이 삐뚤어도 깐다. 아래 단 벽돌이 모두 공급대를 떠나 2 s 멈춰 있고
    # 손이 떠나 있으면 깐다(정확도 조건 없음) = 다음 단 첫 벽돌을 집으러 가기 전. 1단 → 2단, 2단 → 3단 모두. 다음 단 벽돌이 이미 맨
    # 아래 단 위에 놓여 판이 그 벽돌에 파고들 자리면 깔지 않는다(벽돌이 튀어 오르는 시뮬레이션 오류 방지, env.bed_blocked에 기록).
    # 학습 데이터(main_v1)와 다른 점: 경화 시간(main_v1_eval과 같음), 판 까는 규칙
    "main_v1_eval_bed": dict(cure_time_s=7200.0, bed_rule="placed"),
}


def scene_config(version: str, **kw) -> "SceneConfig":
    if version not in ENV_VERSIONS:
        raise ValueError(f"unknown env version {version!r}; known: {sorted(ENV_VERSIONS)}")
    return SceneConfig(env_version=version, **{**ENV_VERSIONS[version], **kw})


def slot_distance(x, y):
    return float(np.hypot(x, y))
