"""정책 밖 검측 G1 · G2 · G3 (3단 벽, 2026-10-20; 실험설계 md 1 · 3절, 소형 실험 brick_sim/grasp_gate.py를 옮김).

정책을 밖에서 지켜보다가 세 순간에만 개입한다. 판단에는 로봇이 실제로 가진 정보만 쓴다:
관절값(순기구학), 손목 힘 센서, 집게 각도, 작업대 위 카메라(top 고해상도 RGB-D, wall3/perception.py), 도면(설계 자리 · 벽돌 크기 ·
작업 순서). **시뮬레이터 참값은 판단에 쓰지 않고**, 기록(측정 오차 · 모르타르 경화 여부)에만 쓴다.

G1 들기 확인 (wall3/g1.py): 집게를 닫고 1.5 cm 들면 멈춰 손목 하중 · 집게 각도로 쥠 판정. 못 쥐었으면 열고 정책을 reset해 다시 집게 함.
G2 놓기 전 보정: '쥐고 있음' 상태에서 정책이 집게를 풀기 시작하면(명령이 쥠 명령보다 내려감) 가로챈다. 자리 위 10 cm로 올려
   쥔 벽돌을 카메라로 잰 뒤(이웃 벽돌과 높이로 구분), 벽돌이 설계 자리 · 방향에 오도록 손을 옮기고, 받침 위 0.5 mm까지 내려 놓는다.
G3 놓은 뒤 재검사: 놓은 직후 팔을 벽 위에서 비켜(로봇 쪽으로) 놓인 벽돌을 잰다. 허용치를 넘으면 모르타르가 굳기 전에(손이 5 cm 밖에서
   2 s 정지하면 굳음) 한 번 다시 집어 G2 방식으로 다시 놓는다(벽돌당 1회).
G2를 끄고 G3만 켜면 놓는 순간 가로채 그 자리에서 그대로 놓은 뒤(보정 없음) G3를 한다.

동작은 대본(wall3/expert.py)의 이동 함수(역동역학 피드포워드 포함)를 그대로 쓴다. 검측 동작 시간도 벽돌당 시간 제한(17 s)에 들어간다.
문턱 · 허용치는 임시값(검증 배치로 정해 동결 전). 결과는 '정책 + 검측'으로 따로 보고한다.

검측이 쓰는 정보 진단 (2026-11-03, 우분투지시_검측정보진단_20261103.md 2절). 기본값(camera, turn)은 그 전과 같은 동작이다.
  pose_source "truth" (진단 전용): 동작은 카메라 모드와 같고, 판단에 쓰는 값(손 안 벽돌 · 놓인 벽돌 자세, 받침 높이)만 참값.
  target: G2 · G3의 기준 자리. "turn" 차례 번호 k / "held_truth" G1 순간 참값으로 들린 벽돌의 자리 (진단) /
          "held_grasp" G1 순간 손(관절값 순기구학) xy에서 가장 가까운 공급칸의 자리 (도면의 공급대 배치만 씀, 참값 안 씀).
  turn이 아니면 감사 기록(audit_*)은 실제로 손에 든 벽돌(참값)로 남긴다.

실패 원인을 하나씩 없애는 추가 제어 (2026-11-10, 01_사기설/실험파이프라인_실패원인제거_20261110.md). 기본값은 모두 꺼짐(그 전과 같은 동작).
  K1 g1_fail_retract: G1이 '안 쥠'으로 판정해 집게를 연 뒤, 5 cm 들고 철수 자세(시연의 시작 자세)로 보낸 다음 정책에 돌려준다
     (그 전: 그 자리에서 집게만 열고 돌려줌 → 정책이 시연에 없던 공중 자세에서 다시 시작). 롤아웃은 이 철수를 '정책의 철수 완료'로
     세지 않는다(pop_gate_retract).
  K6 g3_find_r: G3가 놓인 벽돌을 찾는 반경(설계 자리 중심에서). 그 전 4 cm. 이웃 벽돌 중심은 15 cm 이상 떨어져 있다.
  K2 g0_align (G0 집기 정렬): 정책이 집게를 연 채 어떤 공급칸 중심에서 g0_zone_xy 안, 접근 높이 + g0_zone_z 아래로 내려오면
     가로챈다. 도면의 공급칸 중심 · 방향(env.supply_pose, 흩어짐 ±10 mm · ±10°는 모름)으로 대본의 집기 동작(위로 → 정렬 →
     내려감 → 닫기 → 1.5 cm 들기)을 한 뒤 G1 판정(켜져 있으면)을 하고 정책에 돌려준다. 공급칸마다 g0_max_tries번까지,
     G1이 '쥠'으로 판정한 공급칸(정책이 집었든 G0가 집었든)은 다시 가로채지 않는다. 정책은 어느 벽돌로 갈지와 운반을 맡는다.
     사전 확인(experiments/diag_nominal_grasp.py, 10벽 × 16장): 160 / 160 G1 '쥠', 손 안 어긋남 길이 ≤ 10.7 mm · 폭 ≤ 5.1 mm.
  K7 near_slot 0.40 (G2 넓게 받기, R_11): 손에 든 벽돌을 정책이 놓기 시작할 때, 손이 그 벽돌의 설계 자리에서 이 거리(xy) 안이면
     G2 · G3가 넘겨받는다. 그 전 15 cm. 근거(L1 추적, K2): 벽돌 11을 쥔 90번 중 놓기 시작한 곳이 15 cm 안인 64번은 모두 G2가 놓아
     채웠고, 19 ~ 33 cm인 26번은 모두 G2가 넘겨받지 않아 정책이 놓았고 벽 뒤로 떨어졌다. 공급대와 벽은 80 cm 넘게 떨어져 있다.
  K8 sdcg (G2c 확약 게이트, 방법론 제안 SDCG (b)(c), 2026-11-11): 놓기(확약)는 측정과 힘으로 확인된 뒤에만 한다.
     (b) G2가 손 안 벽돌을 어느 시점에서도 재지 못하면, 손을 10 cm 올리고 로봇 쪽으로 12 cm 옮긴 다른 팔 자세에서 다시 잰다.
     (c) 마지막 8 cm는 천천히 내리며 손목 힘을 본다(2 s, 0.5 s 멈춰 영점, 8 N 넘게 2스텝 = 닿음). 놓을 높이보다 30 mm 넘게
         위에서 닿으면(벽돌 끝이 같은 단 이웃 위에 걸림, 약 57 mm) 놓지 않고, 먼저 놓인 이웃(k − 1)의 반대쪽으로 6 mm 옮겨 다시
         내린다. 또 닿으면 다른 쪽으로 6 mm. 끝내 못 맞추면 그 자리에 둔다.
         손 안 높이를 모를 때(재지 못함)는 닿을 때까지 3 cm 더 내려간다.
     근거(Full 추적, 진단): 자리 6 처음 실패 9벽 중 8벽은 첫 G2가 손 안을 재지 못한 채 '어긋남 0'으로 놓아 7 ~ 11 mm 어긋났고,
     벽돌 끝이 자리 5 위에 걸쳐 떨어졌다. 한 벽(700315)은 정책이 먼저 벽돌을 자리 5에 대고 눌러 손 안에서 22 mm 밀려 올라가 있었다.
     시방 합격 기준(채점) · G3 문턱은 바꾸지 않는다(SDCG (a)의 시방 유도 합격 영역은 구현하지 않음).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from wall3 import gripper as G
from wall3.config import H, W, JOINT
from wall3.expert import WallExpert
from wall3.g1 import G1, G1Config, G1Trigger
from wall3.ik import down_R, wrap
from wall3.poses import yaw_of_quat
from wall3.perception import BrickPoseEstimator, _fold

POSE_SOURCES = ("camera", "truth")
TARGETS = ("turn", "held_truth", "held_grasp")


def nearest_supply_slot(xy, supply_slots, max_d):
    """held_grasp: the supply slot (nominal centre from the layout drawing; slot i holds the brick of design slot i) nearest to
    the hand's xy. Returns (slot or None if farther than max_d, distance m)."""
    d = np.hypot(*(np.asarray(supply_slots, float) - np.asarray(xy, float)[:2]).T)
    i = int(np.argmin(d))
    return (i if d[i] <= max_d else None), float(d[i])


@dataclass
class GateConfig:
    near_slot: float = 0.15          # m: TCP within this (xy) of the brick's design slot = a release meant for that slot
    release_drop: float = 0.02       # rad: grip command this far below the holding command = the policy starts to release
    view_up: float = 0.25            # m: G2 measures the held brick this high above its place height (clear of neighbours;
                                     #    2026-10-20: at 0.10 the elbow hid 3 / 16 slots, at 0.25 all 16 were seen, <= 1.9 mm)
    view_shift: float = 0.035        # m: second / third G2 view shifted along the jaw axis when no brick end was visible
    place_gap: float = 0.0005        # m: brick bottom above its support when released (expert)
    g2_tol_xy: float = 0.0015        # m: after moving, re-measure; correct again above this (once)
    g3_view_back: float = 0.38       # m: G3 views the laid brick with the TCP this far toward the robot (arm off the wall strip)
    g3_view_up: float = 0.25         # m: ... and this high above the brick top (2026-10-20: 0.22 / 0.12 left 2.5 mm errors)
    g3_tol_xy: float = 0.003         # m: re-place above this (임시; 시방 ④ 등에서 벽돌 단위 기준을 유도해 검증 배치로 동결 예정)
    g3_tol_yaw_deg: float = 1.5
    g3_tol_z: float = 0.002          # m: brick centre above its place height by more than this (perched)
    g3_tol_tilt_deg: float = 2.0
    g3_partial_margin: float | None = None  # m: decide 'bad' on a partial view beyond tol + this (None = wait for a full view;
                                     #    only needed while the mortar set in 2 s, main_v1)
    min_time_left_s: float = 0.0     # G3 re-placement only if this much of the brick time limit is left (0 = always; 예비 확인 2026-10-20)
    pose_source: str = "camera"      # 2026-11-03: brick poses / support height for G2 · G3 from the camera, or "truth" (진단 전용)
    target: str = "turn"             # 2026-11-03: G2 · G3 reference slot: "turn" (k) | "held_truth" (진단) | "held_grasp"
    held_grasp_max: float = 0.10     # m: held_grasp nearest supply slot farther than this = undetermined (use k)
    held_truth_min_rise: float = 0.01  # m: held_truth needs a brick lifted more than this (else k)
    g1_fail_retract: bool = False    # 2026-11-10 K1: after a G1 'not grasped', lift and go to the retract pose before handing back
    g1_fail_lift: float = 0.05       # m: K1 straight lift before the joint move to the retract pose
    g3_find_r: float = 0.04          # m: 2026-11-10 K6: G3 looks for the laid brick within this of its design centre
    g0_align: bool = False           # 2026-11-10 K2: grasp at the drawing's supply-slot centre when the policy descends there open
    g0_zone_xy: float = 0.06         # m: TCP within this (xy) of a supply-slot centre
    g0_zone_z: float = 0.03          # m: ... and below the approach height (grasp height + expert approach) plus this
    g0_open_max: float = 0.15        # policy grip command below this = the jaw is (still) open
    g0_max_tries: int = 2            # G0 grasps per supply slot and wall
    sdcg: bool = False               # 2026-11-11 K8 G2c (SDCG): re-measure an unmeasured held brick, guarded last descent
    sdcg_guard_from: float = 0.08    # m: the last part of the G2 descent (above the place height) is slow and force-guarded
    sdcg_guard_T: float = 2.0        # s: duration of that guarded part (slow, 2 mm per control step: stop before the brick
                                     #    turns between the pads; at 4 mm / step it did, unit test 11-11)
    sdcg_f_thr: float = 8.0          # N: wrist force change from the tare above this, sdcg_n_over steps in a row = contact
                                     #    (tare after 0.5 s still; a contact changes it by 10 ~ 50 N; unit test 11-11)
    sdcg_n_over: int = 2             # control steps in a row over sdcg_f_thr = contact
    sdcg_early: float = 0.030        # m: contact with the TCP higher than place height + this = early (a brick end on a same-
                                     #    course neighbour: about H = 57 mm; the bed itself was felt up to 13 mm high, test 11-11)
    sdcg_shift: float = 0.006        # m: after an early contact, shift the brick this far (x try) away from the touching end
    sdcg_tries: int = 2              # re-tries after an early contact
    sdcg_view_up2: float = 0.10      # m: extra in-hand views this much higher ...
    sdcg_view_back: float = 0.12     # m: ... and this far toward the robot (another arm pose) when no view saw a brick end
    sdcg_extra_down: float = 0.03    # m: unmeasured brick (height in the hand unknown): go on down to contact, at most this

    def __post_init__(self):
        if self.pose_source not in POSE_SOURCES:
            raise ValueError(f"unknown pose_source {self.pose_source!r}")
        if self.target not in TARGETS:
            raise ValueError(f"unknown target {self.target!r}")

    def to_dict(self):
        return asdict(self)


# 2026-11-10: added controls, one per failure cause (실험파이프라인_실패원인제거_20261110.md). name -> GateConfig fields
FIXES = {
    "k1": {"g1_fail_retract": True},      # R_10S: G1 '안 쥠' 뒤 철수 자세로 돌려줌
    "k6": {"g3_find_r": 0.07},            # R_6G: G3 찾는 반경 4 -> 7 cm
    "k2": {"g0_align": True},             # R_10S · R_7*: 도면 공급칸 중심에서 집기 (G0 집기 정렬)
    "k7": {"near_slot": 0.40},            # R_11: G2가 놓기를 넘겨받는 거리 15 -> 40 cm (G2 넓게 받기)
    "k8": {"sdcg": True},                 # 2026-11-11 R_6G: G2c 확약 게이트 (SDCG): 못 재면 다시 재기 + 마지막 하강 힘 감시
}


def fix_fields(fixes):
    """GateConfig fields for a list of fix names (order kept, unknown names rejected)."""
    out = {}
    for f in fixes or ():
        if f not in FIXES:
            raise ValueError(f"unknown gate fix {f!r} (known: {sorted(FIXES)})")
        out.update(FIXES[f])
    return out


class WallGates:
    def __init__(self, env, policy=None, g1: bool = False, g2: bool = False, g3: bool = False,
                 g1_cfg: G1Config | None = None, cfg: GateConfig | None = None, on_frame=None):
        self.env, self.policy = env, policy
        self.on_g1, self.on_g2, self.on_g3 = bool(g1), bool(g2), bool(g3)
        self.cfg = cfg or GateConfig()
        self.g1cfg = g1_cfg or G1Config(thr_load=4.5, thr_angle=0.47)   # 임시 문턱 (나4 분포의 가운데)
        self.g1 = G1(env, self.g1cfg) if g1 else None
        self.cam = BrickPoseEstimator(env) if (g2 or g3) else None
        self.ex = WallExpert(env)                       # motion primitives only (IK + feed-forward), never its plan
        self.ex._watch = lambda held, phase: None        # gate motions are not aborted on contact
        self.dt = env.cfg.ctrl_dt
        self.on_frame = on_frame
        self.ex.on_step = (lambda e, a, ph, k: on_frame(ph, k)) if on_frame else None
        self.trig = G1Trigger(self.g1cfg, self.dt)
        self.replaced = set()
        self._t_release = None
        self.debug_dir = None
        self.truth = self.cfg.pose_source == "truth"
        self.default = self.cfg.pose_source == "camera" and self.cfg.target == "turn"     # = the gates before 2026-11-03
        self.carrying, self.tgt, self.held_b = False, 0, None
        self._gate_retracted = False                    # K1: the arm is at the retract pose because of a gate, not the policy
        self.g0_tries, self.g0_taken = [0] * env.n, set()  # K2 bookkeeping per wall

    # ------------------------------------------------------------------ per brick
    def start_brick(self, k, deadline_t):
        # target != turn: a brick still in the hand keeps its reference slot into the next turn (the policy sometimes places
        # in the next turn the brick it picked in this one); a new wall (k = 0) always starts empty
        keep = self.cfg.target != "turn" and k > 0 and self.carrying
        if k == 0:
            self.g0_tries, self.g0_taken = [0] * self.env.n, set()
        self.k, self.deadline = k, deadline_t
        if not keep:
            self.carrying, self.hold_cmd = False, None
            self.tgt, self.held_b = k, None
        self.trig.reset()
        self.ev = {"g1": [], "g2": [], "g3": []}
        if self.cfg.g0_align:
            self.ev["g0"] = []
        self.z_start = [self.env.brick_pose(j)[0][2] for j in range(self.env.n)]      # audit only

    def end_brick(self):
        return self.ev

    def _time_left(self):
        return (self.deadline - self.env.t) * self.dt

    def _audit(self, k):
        """Ground truth for the log only. Default gates: brick k (as before); otherwise the brick really in the hand."""
        if not self.default and self.held_b is not None:
            k = self.held_b
        p, q = self.env.brick_pose(k)
        return p, yaw_of_quat(q)

    def _ab(self, k):
        return k if self.default or self.held_b is None else self.held_b

    # ------------------------------------------------------------------ reference slot (2026-11-03)
    def _lifted_truth(self):
        """Ground truth: the brick in the hand at the G1 moment = the grip-locked one, else the one lifted most since the
        turn began (more than held_truth_min_rise); None if none."""
        env = self.env
        if env.lock_b >= 0:
            return int(env.lock_b)
        rise = [env.brick_pose(j)[0][2] - self.z_start[j] for j in range(env.n)]
        j = int(np.argmax(rise))
        return j if rise[j] > self.cfg.held_truth_min_rise else None

    def _decide_target(self):
        """At the G1 moment (closed and lifted 1.5 cm; the same trigger without G1): the reference slot for G2 · G3 and the
        brick really in the hand (ground truth, audit only unless held_truth / truth)."""
        c = self.cfg
        held = self._lifted_truth()
        info = {"t": int(self.env.t), "turn": int(self.k), "method": c.target, "audit_brick": held}
        if c.target == "turn":
            slot = self.k
        elif c.target == "held_truth":
            slot = held if held is not None else self.k
            info["undetermined"] = held is None
        else:
            p, _ = self.env.tcp()
            i, d = nearest_supply_slot(p[:2], self.env.cfg.supply_slots, c.held_grasp_max)
            slot = i if i is not None else self.k
            info.update(supply_slot=i, supply_dist_mm=round(d * 1000, 1), undetermined=i is None)
        info["slot"] = int(slot)
        return int(slot), held, info

    def _take(self, act, dec, g1_ok):
        slot, held, info = dec
        self.carrying, self.hold_cmd = True, float(act[6])
        self.tgt, self.held_b = slot, held
        if not self.default:
            self.ev.setdefault("target", []).append({**info, "g1_grasped": g1_ok, "used": True})

    def _design(self, k):
        s = self.env.design[k]
        return np.array([s["x"], s["y"]]), float(s["yaw"]), s["L"], s

    # ------------------------------------------------------------------ hooks around the policy's step
    def before(self, act):
        """Called with the policy's action before it is executed. True = the gate handled the step(s) itself."""
        if self.cfg.g0_align and not self.carrying:
            i = self._g0_slot(act)
            if i is not None:
                self._g0_grasp(i)
                return True
        if not self.carrying or self.hold_cmd is None:
            return False
        if float(act[6]) >= self.hold_cmd - self.cfg.release_drop:
            return False
        p, _ = self.env.tcp()
        t = self.tgt                                      # reference slot (= k for target "turn")
        xy, _, _, _ = self._design(t)
        dist = float(np.hypot(*(p[:2] - xy)))
        if dist > self.cfg.near_slot or not (self.on_g2 or self.on_g3):
            self.carrying = False                         # a release elsewhere (or no G2/G3): the policy does it
            if self.on_g2 or self.on_g3:                  # 2026-11-03: logged only (not a G2 release in the summaries)
                self.ev["g2"].append({"t": int(self.env.t), "slot": int(t), "skipped": "far", "dist_mm": round(dist * 1000, 1)})
            return False
        self.carrying = False
        if self.on_g2:
            self._g2_place(t, self.hold_cmd, measure=True)
        else:
            self._release_here(t, self.hold_cmd)          # G3 only: release where the policy is, no correction
        if self.on_g3:
            self._g3(t)
        self._handback()
        return True

    def after(self, act):
        """Called after the policy's action was executed (G1 trigger: closed and lifted 1.5 cm)."""
        if self.carrying:
            return
        p, _ = self.env.tcp()
        if not self.trig.update(float(act[6]), p[2]):
            return
        self._g1_moment(act)

    def _g1_moment(self, act):
        """The G1 moment (closed and lifted 1.5 cm, by the policy or by G0): reference slot, G1 check, then hand back."""
        dec = self._decide_target()                         # the G1 moment: reference slot (kept until the release)
        if self.g1 is None:                                 # no check: treat as carrying
            self._take(act, dec, None)
            self._g0_mark_taken()
            return
        m = self.g1.measure(act[:6], float(act[6]), (lambda lb: self.on_frame(lb, self.k)) if self.on_frame else None)
        ok = self.g1.decide(m)
        rise = [self.env.brick_pose(j)[0][2] - self.z_start[j] for j in range(self.env.n)]     # audit only
        self.ev["g1"].append({"t": int(self.env.t), "load_N": m["load_N"], "angle_rad": m["angle_rad"], "grasped": ok,
                              "audit_target_rise_mm": round(rise[self.k] * 1000, 1), "audit_max_rise_brick": int(np.argmax(rise)),
                              "audit_max_rise_mm": round(max(rise) * 1000, 1)})
        if ok:
            self._take(act, dec, True)
            self._g0_mark_taken()
        else:
            if not self.default:
                self.ev.setdefault("target", []).append({**dec[2], "g1_grasped": False, "used": False})
            for _ in range(max(1, int(round(self.g1cfg.t_open / self.dt)))):
                self.env.step(act[:6], self.env.cfg.gripper.open_ctrl)
                if self.on_frame:
                    self.on_frame("G1 open", self.k)
            if self.cfg.g1_fail_retract:
                self.ev["g1"][-1]["retract"] = self._g1_fail_retract()
            self.trig.reset()
        if self.policy is not None:
            self.policy.reset()

    def _g1_fail_retract(self):
        """K1: open hand straight up by g1_fail_lift, then the expert's joint move to the retract pose (pitch joints rise first)."""
        env, ex = self.env, self.ex
        go = env.cfg.gripper.open_ctrl
        self._sync_expert(go)
        p, R = env.tcp()
        ex._cart_move(p + [0, 0, self.cfg.g1_fail_lift], R, 0.3, go, None, "g1_fail_up")
        ex._joint_move(np.array(ex.q_ret), go, None, "g1_fail_retract")
        self._gate_retracted = True
        return {"t_done": int(env.t), "q_err_rad": round(float(np.abs(env.arm_q() - ex.q_ret).max()), 4)}

    # ------------------------------------------------------------------ K2 G0 grasp alignment (2026-11-10)
    def _g0_mark_taken(self):
        p, _ = self.env.tcp()
        i, _ = nearest_supply_slot(p[:2], self.env.cfg.supply_slots, self.cfg.held_grasp_max)
        if i is not None:
            self.g0_taken.add(i)

    def _g0_heights(self, i):
        pos, _ = self.env.supply_pose(i)                     # drawing: slot centre, table height (no scatter)
        z_grasp = pos[2] + self.ex.dzp
        return pos, z_grasp, z_grasp + self.ex.cfg.approach

    def _g0_slot(self, act):
        """The supply slot the policy is descending onto with an open jaw (None if not)."""
        c, env = self.cfg, self.env
        if float(act[6]) >= c.g0_open_max:
            return None
        p, _ = env.tcp()
        i, d = nearest_supply_slot(p[:2], env.cfg.supply_slots, c.g0_zone_xy)
        if i is None or i in self.g0_taken or self.g0_tries[i] >= c.g0_max_tries:
            return None
        _, _, z_app = self._g0_heights(i)
        if p[2] > z_app + c.g0_zone_z:
            return None
        z_next = float(self.ex.ik.fk(np.asarray(act[:6], float))[0][2])
        if z_next >= p[2] - 1e-4:                            # not going down
            return None
        return i

    def _g0_grasp(self, i):
        """Expert grasp at the drawing's centre of supply slot i, then the G1 moment."""
        env, ex = self.env, self.ex
        gc = env.cfg.gripper
        go = gc.open_ctrl
        self.g0_tries[i] += 1
        pos, z_grasp, z_app = self._g0_heights(i)
        p0, _ = env.tcp()
        ev = {"t": int(env.t), "supply_slot": int(i), "turn": int(self.k), "try": self.g0_tries[i],
              "tcp_off_mm": round(float(np.hypot(*(p0[:2] - pos[:2]))) * 1000, 1), "tcp_z_above_grasp_mm": round((p0[2] - z_grasp) * 1000, 1)}
        self._sync_expert(go)
        p, R = env.tcp()
        if p[2] < z_app:                                     # straight up first: the fingers may be near the brick already
            ex._cart_move(np.r_[p[:2], z_app], R, 0.3, go, None, "g0_up")
        if not self._move_tcp(np.r_[pos[:2], z_app], 0.0, go, None, "g0_align", T=0.5):
            ev["result"] = "ik_failed"
            self.ev["g0"].append(ev)
            self._handback()
            return
        p, R = env.tcp()
        ex._cart_move(np.r_[pos[:2], z_grasp], R, ex.cfg.t_down, go, None, "g0_descend")
        n = int(round(ex.cfg.t_close / self.dt))
        cg = ex.grip_close_cmd()
        ex._hold(n, [min(cg, G.ramp(gc, True, (j + 1) * self.dt, go)) for j in range(n)], None, "g0_close")
        p, R = env.tcp()
        ex._cart_move(p + [0, 0, self.g1cfg.lift], R, 0.3, cg, None, "g0_lift")
        a = np.asarray(env.last_action, float).copy()
        ng1 = len(self.ev["g1"])
        self.trig.reset()
        self._g1_moment(a)                                   # G1 check (if on) and hand back
        if self.g1 is None:                                  # 2026-11-11: G1 off → _g1_moment returns before its policy reset;
            self._handback()                                 # G0 moved the arm, so the policy's old action chunk must go
        ev["g1_grasped"] = self.ev["g1"][ng1]["grasped"] if len(self.ev["g1"]) > ng1 else None
        ev["result"] = "done"
        self.ev["g0"].append(ev)

    def pop_gate_retract(self):
        """True once after a gate moved the arm to the retract pose (the rollout must not count it as the policy's retract)."""
        r, self._gate_retracted = self._gate_retracted, False
        return r

    def _handback(self):
        self.trig.reset()
        if self.policy is not None:
            self.policy.reset()

    # ------------------------------------------------------------------ helpers
    def _sync_expert(self, grip):
        ex = self.ex
        ex.q_cmd = self.env.arm_q().copy()
        ex.grip_cmd, ex.fail, ex.brick_k, ex.in_hand = grip, None, self.k, None
        ex.steps, ex.touch, ex.max_drift, ex.drift_before_place = getattr(ex, "steps", 0), [], 0.0, 0.0

    def _support_top(self, k):
        """Top of what slot k rests on: the floor for course 1, else the laid mortar bed measured by the camera next to
        the slot (head-joint gaps / bed edge), falling back to the drawing."""
        xy, yaw, L, s = self._design(k)
        if s["course"] == 1:
            return 0.0, "floor"
        if self.truth:
            return self._support_top_truth(k)
        if self.cam is not None:
            z = self.cam.surface_z(xy[0], xy[1], yaw, offsets=(-(L / 2 + 0.005), L / 2 + 0.005), across=(0.0,),
                                   z_lo=s["z"] - H / 2 - 0.01, z_hi=s["z"] - H / 2 + 0.01)
            if z is not None:
                return z, "camera"
        return s["z"] - H / 2, "drawing"

    def _support_top_truth(self, k):
        """pose_source truth (진단): the real top of the mortar bed under slot k (slab mocap pose + slab thickness) at the
        points the camera samples (the head-joint gaps at both ends); no bed yet: the real top of the lower-course bricks."""
        from wall3.scene import _quat_rotmat
        env, m, d = self.env, self.env.model, self.env.data
        xy, yaw, L, s = self._design(k)
        bnd = s["course"] - 1
        ex_ = np.array([np.cos(yaw), np.sin(yaw)])
        pts = [xy + o * ex_ for o in (-(L / 2 + 0.005), L / 2 + 0.005)]
        if env.bed_laid[bnd] >= 0:
            src = "truth_bed"
            info = env.beds[bnd]
            boxes = [(d.geom_xpos[g].copy(), d.geom_xmat[g].reshape(3, 3).copy(), m.geom_size[g].copy())
                     for g in (env.bed_gid[nm] for nm in info["slabs"] + info["joints"])]
        else:
            src = "truth_brick"
            boxes = []
            for i in range(env.n):
                if self.env.design[i]["course"] == bnd:
                    p, q = env.brick_pose(i)
                    boxes.append((p, _quat_rotmat(q), np.array([self.env.design[i]["L"] / 2, W / 2, H / 2])))
        zs = []
        for pt in pts:
            tops = []
            for c, R, half in boxes:
                n = R[:, 2] if R[2, 2] >= 0 else -R[:, 2]
                if n[2] < 0.5:
                    continue
                t0 = c + n * half[2]
                z = t0[2] - (n[0] * (pt[0] - t0[0]) + n[1] * (pt[1] - t0[1])) / n[2]
                loc = R.T @ (np.r_[pt, z] - c)
                if abs(loc[0]) <= half[0] and abs(loc[1]) <= half[1]:
                    tops.append(float(z))
            if tops:
                zs.append(max(tops))
        if zs:
            return float(np.median(zs)), src
        return s["z"] - H / 2, "drawing"

    def _in_hand_truth(self):
        """pose_source truth (진단): the brick hanging in the jaw now (grip-locked, else the one found at the G1 moment if it
        still hangs about dzp below the TCP); None = empty hand."""
        env = self.env
        b = env.lock_b if env.lock_b >= 0 else self.held_b
        if b is None or b < 0:
            return None
        p, R = env.tcp()
        bp, _ = env.brick_pose(b)
        ax = R[:2, 0] / np.linalg.norm(R[:2, 0])
        ay = np.array([-ax[1], ax[0]])
        rel = bp[:2] - p[:2]
        if abs(bp[2] - (p[2] - self.ex.dzp)) > 0.02 or abs(rel @ ax) > self.env.design[b]["L"] / 2 or abs(rel @ ay) > W / 2 + 0.01:
            return None
        return int(b)

    def _held_meas_truth(self, b, p, R):
        """pose_source truth (진단): brick b in the TCP frame at TCP pose (p, R), in the camera's held_brick format."""
        bp, bq = self.env.brick_pose(b)
        byaw = yaw_of_quat(bq)
        ax = R[:2, 0] / np.linalg.norm(R[:2, 0])
        ay = np.array([-ax[1], ax[0]])
        rel = bp[:2] - p[:2]
        return {"pos": bp, "yaw": byaw, "yaw_in_hand_deg": float(np.degrees(_fold(byaw - float(np.arctan2(R[1, 0], R[0, 0]))))),
                "across_m": float(rel @ ay), "along_m": float(rel @ ax), "src": "truth", "brick": int(b)}

    def _find_truth(self, xy, r, z_range):
        """pose_source truth (진단), for cam.find: the brick just released (the one in the hand at the G1 moment) if its real
        centre is within r of xy with its top in z_range (the camera's search window), else None; with no known brick in the
        hand, the brick whose real centre is nearest xy in that window."""
        from wall3.scene import _quat_rotmat
        env = self.env
        best, bd = None, r
        for j in (range(env.n) if self.held_b is None else (self.held_b,)):
            p, q = env.brick_pose(j)
            R = _quat_rotmat(q)
            z_top = p[2] + abs(R[2, 2]) * H / 2
            if z_range is not None and not (z_range[0] <= z_top <= z_range[1]):
                continue
            dd = float(np.hypot(*(p[:2] - np.asarray(xy)[:2])))
            if dd < bd:
                best, bd = {"pos": p, "yaw": yaw_of_quat(q), "tilt_deg": float(np.degrees(np.arccos(np.clip(abs(R[2, 2]), -1, 1)))),
                            "full": True, "src": "truth", "brick": j}, dd
        return best

    def _move_tcp(self, p, yaw, grip, held, phase, T=0.6, exact_yaw=False):
        """Joint-space move to a TCP pose (gripper down, x axis = yaw); unless exact_yaw, the yaw is chosen modulo pi
        closest to the wrist (an empty jaw or a 180-degree symmetric brick)."""
        ex, env = self.ex, self.env
        _, R0 = env.tcp()
        cur = float(np.arctan2(R0[1, 0], R0[0, 0]))
        if not exact_yaw:
            yaw = yaw if abs(wrap(yaw - cur)) <= abs(wrap(yaw + np.pi - cur)) else yaw + np.pi
        q = ex._solve(np.asarray(p, float), down_R(yaw), env.arm_q(), env.arm_q()[0])
        if q is None:
            return False
        ex._joint_move(q, grip, held, phase, t_min=T, lag=0.0)
        return True

    def _measure_held(self, k):
        """In-hand brick from the camera at the current pose; up to two extra views shifted along the jaw axis."""
        env, ex = self.env, self.ex
        _, _, L, _ = self._design(k)
        tries = []
        p_home, R_home = env.tcp()
        yaw_home = float(np.arctan2(R_home[1, 0], R_home[0, 0]))
        out = np.r_[self._design(k)[0] / np.linalg.norm(self._design(k)[0]), 0.0]          # away from the robot
        views = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.08), (self.cfg.view_shift, 0.0, 0.0), (-self.cfg.view_shift, 0.0, 0.0),
                 (0.0, np.deg2rad(35), 0.0), (0.0, -np.deg2rad(35), 0.0)]
        got = []                                                   # (hb, p, R) of the views that saw a brick end
        for shift, turn, o in views:
            if len(got) >= 3:
                break
            if shift or turn or o:
                tgt = p_home + shift * R_home[:, 0] + o * out
                if turn:
                    self._move_tcp(tgt, yaw_home + turn, ex.grip_cmd, k, "g2_view_turn", T=0.4, exact_yaw=True)
                else:
                    self._move_tcp(tgt, yaw_home, ex.grip_cmd, k, "g2_view_shift", T=0.3, exact_yaw=True)
            p, R = env.tcp()
            if self.truth:
                b = self._in_hand_truth()
                hb = self._held_meas_truth(b, p, R) if b is not None else None
            else:
                hb = self.cam.held_brick(p, R, L, ex.dzp)
            tries.append(hb is not None)
            if hb is None and self.debug_dir is not None and not self.truth:          # what the camera saw when no brick end was visible
                from PIL import Image
                self.debug_dir.mkdir(parents=True, exist_ok=True)
                Image.fromarray(self.cam.last_rgb).save(self.debug_dir / f"g2_t{env.t}_k{k}_shift{shift:+.3f}_turn{np.degrees(turn):+.0f}_out{o:.2f}.png")
            if hb is not None:
                got.append((hb, p, R))
        if (p - p_home).any():
            self._move_tcp(p_home, yaw_home, ex.grip_cmd, k, "g2_view_home", T=0.3, exact_yaw=True)
        p, R = env.tcp()
        if not got:
            return None, p, R, tries
        if self.truth:                                             # the true in-hand pose at the pose G2 continues from
            b = self._in_hand_truth()
            if b is None:
                return None, p, R, tries
            hb = self._held_meas_truth(b, p, R)
            hb["n_views"] = len(got)
            return hb, p, R, tries
        # the grasp position along the brick is a property of the grasp: the median over the views that saw it
        hb = dict(got[0][0])
        hb["along_m"] = float(np.median([h["along_m"] for h, _, _ in got]))
        hb["across_m"] = 0.0
        hb["n_views"] = len(got)
        return hb, p, R, tries

    # ------------------------------------------------------------------ G2
    def _g2_place(self, k, grip, measure=True):
        env, ex, c = self.env, self.ex, self.cfg
        self._sync_expert(grip)
        xy, yaw_d, L, s = self._design(k)
        sup, sup_src = self._support_top(k)
        tcp_place_z = sup + c.place_gap + H / 2 + ex.dzp       # brick hangs dzp below the TCP
        ev = {"t": int(env.t), "slot": int(k), "support_src": sup_src}
        # 1) up to the view height above where the hand is, measure the brick in the hand
        p, R = env.tcp()
        ex._cart_move(np.r_[p[:2], max(p[2], tcp_place_z + c.view_up)], R, 0.4, grip, k, "g2_up")
        hb, p, R, tries = self._measure_held(k) if measure else (None, *env.tcp(), [])
        ev["views"] = tries
        if hb is None and measure and c.sdcg:                    # K8 (SDCG b): not measured -> do not commit yet, look again
            hb, p, R, tries2 = self._sdcg_remeasure(k, grip)     # from another arm pose (higher, toward the robot)
            ev["sdcg_remeasure"] = {"views": tries2, "ok": hb is not None}
        tcp_yaw = float(np.arctan2(R[1, 0], R[0, 0]))
        if hb is None:
            u0, c0, dpsi, src = 0.0, 0.0, 0.0, "nominal"         # grasp position unknown: assume a centred, aligned grasp
        else:
            u0, c0, dpsi, src = hb["along_m"], hb["across_m"], np.deg2rad(hb["yaw_in_hand_deg"]), ("truth" if self.truth else "camera")
        ev.update(in_hand_along_mm=round(u0 * 1000, 2), in_hand_across_mm=round(c0 * 1000, 2),
                  in_hand_yaw_deg=round(float(np.degrees(dpsi)), 2), in_hand_src=src)
        bp, byaw = self._audit(k)                                # log only: true in-hand offset
        ax = R[:2, 0] / np.linalg.norm(R[:2, 0])
        ay = np.array([-ax[1], ax[0]])
        ev["audit_in_hand_err_mm"] = round(float(np.hypot(*((p[:2] + u0 * ax + c0 * ay) - bp[:2]))) * 1000, 2) if hb is not None else None
        ev["audit_in_hand_yaw_err_deg"] = round(float(np.degrees(_fold(tcp_yaw + dpsi - byaw))), 2) if hb is not None else None
        if not self.default:
            ev["audit_brick"] = self._ab(k)
        if self.debug_dir is not None and hb is not None and ev["audit_in_hand_err_mm"] > 3 and not self.truth:
            from PIL import Image
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            self.cam._key = None; self.cam.detect()
            Image.fromarray(self.cam.last_rgb).save(self.debug_dir / f"g2bad_t{env.t}_k{k}_err{ev['audit_in_hand_err_mm']:.1f}.png")
            (self.debug_dir / f"g2bad_t{env.t}_k{k}.json").write_text(__import__("json").dumps(
                {k2: (v.tolist() if hasattr(v, "tolist") else v) for k2, v in hb.items()}, default=float))
        # 2) over the slot: TCP so the brick (offset u0 along / c0 across the jaw, turned dpsi in the hand) lands on the
        # design x, y, yaw. Brick yaw = TCP yaw + dpsi -> TCP yaw = design yaw - dpsi (modulo pi, closest to the wrist)
        yaw_t = yaw_d - dpsi
        yaw_t = yaw_t if abs(wrap(yaw_t - tcp_yaw)) <= abs(wrap(yaw_t + np.pi - tcp_yaw)) else yaw_t + np.pi
        ax_t = np.array([np.cos(yaw_t), np.sin(yaw_t)]); ay_t = np.array([-ax_t[1], ax_t[0]])
        tgt = np.r_[xy - u0 * ax_t - c0 * ay_t, tcp_place_z + c.view_up]
        ok = self._move_tcp(tgt, yaw_t, grip, k, "g2_align", T=0.8, exact_yaw=True)
        ev["moved_mm"] = round(float(np.hypot(*(tgt[:2] - p[:2]))) * 1000, 1)
        # 3) re-measure once at the target and correct the residual
        if ok and measure:
            p2, R2 = env.tcp()
            if self.truth:
                b2 = self._in_hand_truth()
                hb2 = self._held_meas_truth(b2, p2, R2) if b2 is not None else None
            else:
                hb2 = self.cam.held_brick(p2, R2, L, ex.dzp)
            if hb2 is not None:
                err = xy - hb2["pos"][:2]
                ev["residual_mm"] = round(float(np.hypot(*err)) * 1000, 2)      # logged only
        # 4) straight down to the place height and release (K8: the last part slow and force-guarded)
        if c.sdcg:
            ev["sdcg"] = self._sdcg_descend(k, tcp_place_z, grip, ax_t, unknown_height=hb is None)
        else:
            p, R = env.tcp()
            ex._cart_move(np.r_[p[:2], tcp_place_z], R, 0.6, grip, k, "g2_descend")
        self._open_and_lift(k, grip)
        bp, byaw = self._audit(k)
        ev["audit_after_mm"] = round(float(np.hypot(*(bp[:2] - xy))) * 1000, 2)
        ev["audit_after_yaw_deg"] = round(float(np.degrees(_fold(byaw - yaw_d))), 2)
        self.ev["g2"].append(ev)

    # ------------------------------------------------------------------ K8 G2c commitment gate (SDCG, 2026-11-11)
    def _sdcg_remeasure(self, k, grip):
        """SDCG (b): no view saw a brick end -> measure again from another arm pose (higher, toward the robot), then
        come back. In the Full trace the failed views were at the wall's left end (wrist over the held brick, 2026-11-11)."""
        env, ex, c = self.env, self.ex, self.cfg
        p0, R0 = env.tcp()
        yaw0 = float(np.arctan2(R0[1, 0], R0[0, 0]))
        toward_robot = -p0[:2] / max(float(np.linalg.norm(p0[:2])), 1e-6)
        q = np.r_[p0[:2] + c.sdcg_view_back * toward_robot, p0[2] + c.sdcg_view_up2]
        if not self._move_tcp(q, yaw0, grip, k, "g2c_view_move", T=0.5, exact_yaw=True):
            return None, *env.tcp(), []
        hb, p, R, tries = self._measure_held(k)
        self._move_tcp(p0, yaw0, grip, k, "g2c_view_back", T=0.5, exact_yaw=True)
        p, R = env.tcp()
        return hb, p, R, tries                                   # the grasp position along the brick is the same here

    def _sdcg_descend(self, k, z_place, grip, ax_t, unknown_height=False):
        """SDCG (c): fast down to z_place + guard_from, then slow and force-guarded down to the place height.
        A contact well above the place height (> sdcg_early) means a brick end is on something (a neighbour): lift, shift
        the brick and try again: first away from the same-course neighbour laid before it (slot k - 1, the building order),
        then to the other side. The wrist torque is not used for the side: at the contact the brick turns between the pads
        (unit test 2026-11-11: the force / torque contact point came out on the wrong end); it is only logged.
        unknown_height (the camera never measured the held brick): go on down to contact, at most sdcg_extra_down."""
        env, ex, c = self.env, self.ex, self.cfg
        out = {"tries": []}
        p, R = env.tcp()
        xy0, shift = p[:2].copy(), 0.0
        z_g = z_place + c.sdcg_guard_from
        noop = lambda held, phase: None
        for attempt in range(c.sdcg_tries + 1):
            xy = xy0 + shift * ax_t
            p, R = env.tcp()
            if p[2] < z_g - 1e-3:
                ex._cart_move(np.r_[p[:2], z_g], R, 0.3, grip, k, "g2c_up")
            p, R = env.tcp()
            ex._cart_move(np.r_[xy, z_g], R, 0.8 if attempt == 0 else 0.4, grip, k, "g2c_to_guard")
            if ex.fail:                                          # IK: fall back to the plain descent
                out["result"] = f"fallback:{ex.fail}"
                ex.fail = None
                p, R = env.tcp()
                ex._cart_move(np.r_[p[:2], z_place], R, 0.6, grip, k, "g2_descend")
                return out
            fts = []
            ex._watch = lambda held, phase: fts.append(env.wrist_ft())
            try:
                ex._hold(10, grip, k, "g2c_tare")               # 0.5 s still: the arm stops ringing; tare = last 5 steps
            finally:
                ex._watch = noop
            F0 = np.mean([f for f, _ in fts[-5:]], axis=0)
            T0 = np.mean([t for _, t in fts[-5:]], axis=0)
            hit, over = {}, [0]

            def watch(held, phase, F0=F0, T0=T0, hit=hit, over=over):
                F, Tq = env.wrist_ft()
                dF = F - F0
                over[0] = over[0] + 1 if abs(float(dF[2])) > c.sdcg_f_thr else 0
                if over[0] >= c.sdcg_n_over:                     # control steps in a row
                    hit.update(dF=dF, dT=Tq - T0, z=float(env.tcp()[0][2]), t=int(env.t))
                    ex.fail = "sdcg_contact"
            z_end = z_place - (c.sdcg_extra_down if unknown_height else 0.0)
            T = c.sdcg_guard_T * (z_g - z_end) / c.sdcg_guard_from
            ex._watch = watch
            try:
                ex._cart_move(np.r_[xy, z_end], R, T, grip, k, "g2c_descend")
            finally:
                ex._watch = noop
            if ex.fail and ex.fail != "sdcg_contact":
                out["result"] = f"ik:{ex.fail}"
                ex.fail = None
                return out
            ex.fail = None
            rec = {"shift_mm": round(shift * 1000, 1)}
            if not hit:
                rec["contact"] = None                            # reached the end height without contact: release here
                out["tries"].append(rec)
                out["result"] = "placed"
                return out
            dz = hit["z"] - z_place
            rec.update(contact_dz_mm=round(dz * 1000, 1), dFz_N=round(float(hit["dF"][2]), 1), t=hit["t"])
            if dz <= c.sdcg_early:                               # touched at about the expected height: release here
                out["tries"].append(rec)
                out["result"] = "placed"
                return out
            # early contact: which end? the contact point (perpendicular to the force) r = (F x T) / |F|^2,
            # the same whatever sign convention the sensor uses
            dF, dT = hit["dF"], hit["dT"]
            r = np.cross(dF, dT) / max(float(dF @ dF), 1e-9)
            rec["contact_along_mm_log"] = round(float(r[:2] @ ax_t) * 1000, 1)       # logged only (see docstring)
            out["tries"].append(rec)
            if attempt == c.sdcg_tries:
                break
            away = self._sdcg_away(k, ax_t)
            shift = away * c.sdcg_shift if attempt == 0 else -away * c.sdcg_shift
        out["result"] = "gave_up"                                # still early contact: lift and put it down unguarded
        p, R = env.tcp()
        ex._cart_move(np.r_[p[:2], z_g], R, 0.3, grip, k, "g2c_up")
        ex._cart_move(np.r_[xy0 + shift * ax_t, z_place], R, 0.6, grip, k, "g2_descend")
        return out

    def _sdcg_away(self, k, ax_t):
        """+1 / -1 along ax_t: away from the same-course neighbour laid before slot k (k - 1, else k + 1)."""
        d = self.env.design
        nb = k - 1 if k > 0 and d[k - 1]["course"] == d[k]["course"] else k + 1
        if nb >= len(d) or d[nb]["course"] != d[k]["course"]:
            return 1.0
        v = np.array([d[k]["x"] - d[nb]["x"], d[k]["y"] - d[nb]["y"]])     # from the neighbour to slot k
        return 1.0 if float(v @ ax_t) >= 0 else -1.0

    def _release_here(self, k, grip):
        """G3 without G2: the policy's release, done where the hand is (no measurement, no correction)."""
        self._sync_expert(grip)
        self._open_and_lift(k, grip)
        self.ev["g2"].append({"t": int(self.env.t), "slot": int(k), "release_here": True})

    def _open_and_lift(self, k, grip):
        env, ex = self.env, self.ex
        gc = env.cfg.gripper
        cl = env.closure()
        nu = max(1, int(round(ex.cfg.t_unload / self.dt)))
        ex._hold(nu, list(np.linspace(grip, cl, nu + 1)[1:]), k, "release_unload")
        n = int(round(ex.cfg.t_open / self.dt))
        ex._hold(n, [G.ramp(gc, False, (i + 1) * self.dt, cl) for i in range(n)], None, "release_open")
        self._t_release = int(env.t)
        p, R = env.tcp()
        ex._cart_move(p + [0, 0, 0.05], R, 0.25, 0.0, None, "release_up")

    # ------------------------------------------------------------------ G3
    def _view_laid(self, k, fast=True):
        """Move the hand off the wall strip toward the robot and up, checking the camera at every step: decide as soon as
        the laid brick is fully visible (the mortar sets 2 s after the hand leaves a still brick, so G3 has to be quick).
        If it never shows in full, look once more from the retract pose (too late for a re-placement, logged)."""
        env, ex, c = self.env, self.ex, self.cfg
        self._sync_expert(0.0)
        xy, yaw_d, L, s = self._design(k)
        zr = (s["z"] + H / 2 - 0.02, s["z"] + H / 2 + H + 0.01)       # also a brick perched on a neighbour
        p0, R = env.tcp()
        yaw0 = float(np.arctan2(R[1, 0], R[0, 0]))
        toward_robot = -xy / np.linalg.norm(xy)
        view = np.r_[xy + c.g3_view_back * toward_robot, s["z"] + H / 2 + ex.dzp + c.g3_view_up]
        q_v = ex._solve(view, down_R(yaw0), env.arm_q(), env.arm_q()[0])
        d = None
        self.view_steps = 0
        if q_v is not None:
            q0 = env.arm_q().copy()
            n = max(2, int(round(0.7 / self.dt)))
            from wall3.expert import smooth_trap
            prof = smooth_trap(np.linspace(0, 1, n + 1)[1:])
            path = q0[None] + prof[:, None] * (q_v - q0)[None]
            for i in range(n):
                ex._play(path[i:i + 1], 0.0, None, "g3_view")
                self.view_steps += 1
                if i >= 3:
                    dd = self._find_truth(xy, c.g3_find_r, zr) if self.truth else self.cam.find(xy, r=c.g3_find_r, z_range=zr, L=L, yaw_prior=yaw_d)
                    if dd is not None and dd["full"]:
                        return dd
                    if c.g3_partial_margin is not None and dd is not None and \
                            float(np.hypot(*(dd["pos"][:2] - xy))) > c.g3_tol_xy + c.g3_partial_margin:
                        dd["decided_on_partial"] = True                  # clearly off even with a partial view: no time to lose
                        return dd
                    d = dd if dd is not None else d
        ex._joint_move(np.array(ex.q_ret), 0.0, None, "g3_view_retract", lag=0.0)
        if self.truth:
            d2 = self._find_truth(xy, c.g3_find_r, zr)
        else:
            self.cam._key = None
            d2 = self.cam.find(xy, r=c.g3_find_r, z_range=zr, L=L, yaw_prior=yaw_d)
        return d2 if d2 is not None else d

    def _g3(self, k):
        env, c = self.env, self.cfg
        self._sync_expert(0.0)
        xy, yaw_d, L, s = self._design(k)
        d = self._view_laid(k)
        ev = {"t": int(env.t), "slot": int(k), "seen": d is not None, "t_release_done": self._t_release}
        if d is not None:
            sup, _ = self._support_top(k)
            exy = float(np.hypot(*(d["pos"][:2] - xy)))
            eyaw = float(np.degrees(_fold(d["yaw"] - yaw_d)))
            ez = float(d["pos"][2] - (sup + H / 2))
            bp, byaw = self._audit(k)
            ev.update(exy_mm=round(exy * 1000, 2), eyaw_deg=round(eyaw, 2), ez_mm=round(ez * 1000, 2), tilt_deg=round(d["tilt_deg"], 2),
                      audit_meas_err_mm=round(float(np.hypot(*(d["pos"][:2] - bp[:2]))) * 1000, 2))
            if not self.default:
                ev["audit_brick"] = self._ab(k)
                if self.truth:
                    ev["truth_found_brick"] = d.get("brick")
            bad = exy > c.g3_tol_xy or abs(eyaw) > c.g3_tol_yaw_deg or ez > c.g3_tol_z or d["tilt_deg"] > c.g3_tol_tilt_deg
            ev["bad"] = bool(bad)
            if bad and k not in self.replaced and d["tilt_deg"] < 45:
                if c.min_time_left_s > 0 and self._time_left() < c.min_time_left_s:
                    ev["replace"] = "skipped_time"
                else:
                    self.replaced.add(k)
                    ev["replace"] = self._replace(k, d)
        self.ev["g3"].append(ev)

    def _replace(self, k, d):
        """Re-grasp the laid brick at its measured pose and place it again with G2 (once)."""
        env, ex = self.env, self.ex
        xy, yaw_d, L, s = self._design(k)
        gc = env.cfg.gripper
        pos, yaw_b = d["pos"], d["yaw"]
        above = np.r_[pos[:2], pos[2] + ex.dzp + 0.045]
        out = {"audit_cured_before": bool(env.cured[self._ab(k)])}
        if not self._move_tcp(above, yaw_b, 0.0, None, "g3_to_brick", T=0.4):
            out["result"] = "ik_failed"
            return out
        p, R = env.tcp()
        ex._cart_move(np.r_[pos[:2], pos[2] + ex.dzp], R, 0.35, 0.0, None, "g3_descend")
        n = int(round(ex.cfg.t_close / self.dt))
        cg = ex.grip_close_cmd()
        ex._hold(n, [min(cg, G.ramp(gc, True, (i + 1) * self.dt, gc.open_ctrl)) for i in range(n)], None, "g3_close")
        closure = env.closure()
        if not self.default:                                             # the brick really re-grasped (audit; truth mode: judged by it)
            if self.truth and d.get("brick") is not None:
                self.held_b = int(d["brick"])
            elif env.lock_b >= 0:
                self.held_b = int(env.lock_b)
            out["audit_brick"] = self._ab(k)
        out["audit_cured_at_close"] = bool(env.cured[self._ab(k)])       # log only: mortar set before the gate got there?
        out["audit_cure_t"] = int(env.cure_t[self._ab(k)]); out["t_close"] = int(env.t)
        if closure > ex.cfg.grip_empty:
            out["result"] = "grasp_empty"
            self._open_and_lift(k, cg)
            return out
        self._g2_place(k, cg, measure=True)                              # (self.k stays the turn; k = the reference slot)
        d2 = self._view_laid(k)
        if d2 is not None:
            out["after_exy_mm"] = round(float(np.hypot(*(d2["pos"][:2] - xy))) * 1000, 2)
            out["after_eyaw_deg"] = round(float(np.degrees(_fold(d2["yaw"] - yaw_d))), 2)
        bp, byaw = self._audit(k)
        out["audit_after_mm"] = round(float(np.hypot(*(bp[:2] - xy))) * 1000, 2)
        out["result"] = "replaced"
        return out

    def summary_config(self):
        return {"g1": self.on_g1, "g2": self.on_g2, "g3": self.on_g3, "gate_cfg": self.cfg.to_dict(),
                "g1_cfg": {**self.g1cfg.to_dict(), "tare_N": self.g1.f0 if self.g1 else None}}
