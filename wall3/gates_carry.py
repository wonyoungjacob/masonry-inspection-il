"""G2t: G2가 운반 전체를 넘겨받는다 (2026-11-16, 사용자 지시. 스토리라인 '방법 후보 3').

Full(G0 + G1 + G2 + G2w + G3 + H)과 같되, G1이 '쥠'으로 판정한 바로 그 순간부터 G2가 운반까지 맡는다:
  쥔 자리에서 곧장 10 cm 들기(대본의 grasp_lift 높이) → 기준 자리(H: 쥔 순간 가장 가까운 공급칸의 설계 자리) 위 10 cm로 관절 이동
  (대본의 to_wall과 같은 곡선 · 손목 선택, 벽돌이 손 가운데에 반듯이 있다고 가정, 도면 방향) → 그 뒤는 Full의 G2 그대로
  (손 안 벽돌을 카메라로 재고 맞춰 놓음) → G3 → 정책에 돌려줌.
정책이 맡는 일: 다음 벽돌 고르기 · 공급칸 위로 접근(G0가 넘겨받을 때까지) · 놓은 뒤 철수.
쓰는 정보: 도면(설계 자리 · 방향), 관절값, 공구 사양. 운반 목표에 카메라 · 참값을 쓰지 않는다.
이동 함수는 다른 검측과 같다(wall3/expert.py, gates.py의 G2와 같은 held 인자).

wall3/gates.py는 고치지 않는다(Full과 같은 코드). 이 파일의 하위 클래스만 G1 '쥠' 직후에 운반을 끼워 넣는다.
"""
from __future__ import annotations

import numpy as np

from wall3.gates import WallGates
from wall3.ik import down_R


class WallGatesCarry(WallGates):
    def _take(self, act, dec, g1_ok):
        super()._take(act, dec, g1_ok)
        if g1_ok is True and self.on_g2:
            self._g0_mark_taken()                 # before the hand leaves the supply table (the base marks it after _take)
            self._carry_and_place()

    def _carry_and_place(self):
        env, ex = self.env, self.ex
        k, grip = self.tgt, self.hold_cmd
        xy, yaw_d, L, s = self._design(k)
        ev = {"t": int(env.t), "slot": int(k)}
        self._sync_expert(grip)
        p, R = env.tcp()
        ex._cart_move(p + [0.0, 0.0, ex.cfg.approach - self.g1cfg.lift], R, ex.cfg.t_up, grip, k, "g2t_up")
        ev["up_fail"] = ex.fail
        ex.fail = None
        tgt = np.array([xy[0], xy[1], s["z"] + ex.dzp + ex.cfg.approach])
        cands = []
        for flip in (0.0, np.pi):
            qw = ex._solve(tgt, down_R(yaw_d + flip), ex.q_ret, ex.q_ret[0])
            if qw is not None:
                cands.append((abs(qw[5] - ex.q_cmd[5]) + 0.3 * abs(qw[5] - ex.q_ret[5]), qw))   # expert's wrist cost
        if not cands:
            ev["result"] = "ik_failed"             # leave the brick to the policy (as in Full)
            self.ev.setdefault("g2t", []).append(ev)
            return
        ex._joint_move(min(cands, key=lambda x: x[0])[1], grip, k, "g2t_carry", lag=ex.cfg.lag_wall)
        ev["carry_fail"] = ex.fail
        ex.fail = None
        p2, _ = env.tcp()
        ev["tcp_to_target_mm"] = round(float(np.linalg.norm(p2 - tgt)) * 1000, 1)
        ev["t_done"] = int(env.t)
        ev["result"] = "carried"
        self.ev.setdefault("g2t", []).append(ev)
        self.carrying = False
        self._g2_place(k, grip, measure=True)
        if self.on_g3:
            self._g3(k)
        self._handback()
