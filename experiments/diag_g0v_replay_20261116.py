"""진단용 재실행(결과 아님, 2026-11-17): 하이브리드 · A · 벽 하나에서 G0v가 볼 때 손목 카메라 검출 목록과 공급 벽돌 참값 (진단)을 남긴다.

검측 동작은 바꾸지 않는다(_g0v_look을 감싸 기록만 더함). 참값은 기록에만 쓴다.
실행: MUJOCO_GL=egl PYTHONPATH=. venv/bin/python experiments/diag_g0v_replay_20261116.py <출력 폴더(없어야 함)> 700342
"""
import sys, json, os
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))
import numpy as np
import wall3.gates_real as GR
from wall3.config import H
LOG = []
orig = GR.WallGatesReal._g0v_look
def look(self, i):
    env = self.env
    p, R = env.tcp()
    dets = self.wcam.detect(force=True)
    rows = []
    for d in dets:
        pp = self.wcam.pose_of(d, L=self.design_belief[i]["L"], yaw_prior=0.0)
        rows.append({"z_top": round(float(d["z_top"]), 4), "pos": [round(float(x), 4) for x in pp["pos"]], "dist_mm": round(float(np.hypot(*(pp["pos"][:2] - p[:2]))) * 1000, 1),
                     "px": int(d.get("n", d.get("px", -1))) if isinstance(d.get("n", d.get("px", -1)), (int, np.integer)) else str(type(d.get("n")))})
    bp, bq = env.brick_pose(i)
    zax = None
    try:
        import mujoco
        m = np.zeros(9); mujoco.mju_quat2Mat(m, np.asarray(bq, float)); zax = m.reshape(3, 3)[:, 2]
    except Exception as e:
        zax = str(e)
    LOG.append({"t": int(env.t), "slot": int(i), "k": int(self.k), "tcp": [round(float(x), 4) for x in p], "tcp_zaxis": [round(float(x), 3) for x in R[:, 2]],
                "tcp_xaxis": [round(float(x), 3) for x in R[:, 0]],
                "zt_nominal": round(env.cfg.table_h + H, 4), "brick_true": [round(float(x), 4) for x in bp], "brick_tilt_deg": round(float(np.degrees(np.arccos(min(1, abs(zax[2]))))), 2) if not isinstance(zax, str) else zax,
                "dets": rows})
    return orig(self, i)
GR.WallGatesReal._g0v_look = look
import wall3.rollout_real as RR
out = sys.argv[1]
sys.argv = ["x", "--controller", "act", "--env-version", "main_v1_eval", "--seeds", sys.argv[2], "--out", out, "--A", "--record-first"]
try:
    RR.main()
finally:
    json.dump(LOG, open(out + "_g0v_log.json", "w"), indent=1)
