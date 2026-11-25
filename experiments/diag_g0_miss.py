"""K2(G0 집기 정렬)가 작동하지 않은 채 멈춘 차례에서, 정책이 어디서 집게를 닫았나 (2026-11-10, 진단).

diag_trace 출력(trace.json + trace_<seed>.npz, --gate-fix k2)만 읽는다. 새 시뮬레이션 없음. 평가 결과가 아니다.
- 대상: 처음 못 채운 자리가 있는 벽에서, 마지막으로 벽돌을 쥔(G1 '쥠') 차례 뒤의 차례들(= 아무것도 못 쥐고 끝난 차례).
  화면 · 분류 수는 벽마다 첫 멈춘 차례만 센다(뒤 차례는 대개 빈 칸을 되풀이해 집는 결과). json에는 모든 차례를 남긴다.
  처음 실패 원인 자체는 diag_trace_causes가 가린다. 이 스크립트는 'G0가 왜 안 작동했나'만 본다.
- 정책이 집게를 닫기 시작한 순간(집게 명령이 g0_open_max를 넘는 순간, 정책 단계만)마다:
  손끝 xy에서 가장 가까운 공급칸(도면 중심)과 거리, 그 칸 벽돌이 아직 칸에 있나(참값 (진단): 시작 위치에서 3 cm 안),
  접근 높이 대비 손끝 높이, 처음 못 채운 자리의 공급칸까지 거리.
- G0 작동 조건(gates._g0_slot)과 같은 기준으로, 집게가 열린 정책 단계 중 처음 못 채운 자리의 공급칸에 가장 가까이 간 거리와 높이도 낸다.
  '내려가는 중' 조건은 기록된 다음 단계 손끝 높이로 근사한다.

실행: MUJOCO_GL=egl PYTHONPATH=. venv/bin/python -m experiments.diag_g0_miss outputs/wall3/diag_trace_ladder1_k2_20261110 --out <json>
"""
import argparse, glob, json
from collections import Counter
from pathlib import Path

import numpy as np

SRC_POLICY = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace_dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--present-mm", type=float, default=30.0, help="brick counts as still in its supply cell within this (truth, diagnosis)")
    a = ap.parse_args()
    if a.out and Path(a.out).exists():
        raise SystemExit(f"{a.out} 이미 있음: 덮어쓰지 않음")
    from wall3.config import scene_config
    from wall3.gates import GateConfig
    from wall3.scene import WallEnv
    from wall3.expert import WallExpert
    env = WallEnv(scene_config("main_v1_eval"))
    ex = WallExpert(env)
    gc = GateConfig()
    cells = np.asarray(env.cfg.supply_slots, float)
    z_grasp = env.supply_pose(0)[0][2] + ex.dzp
    z_app = z_grasp + ex.cfg.approach
    env.close()

    rows = []
    for tj in sorted(glob.glob(str(Path(a.trace_dir) / "part*" / "trace.json"))):
        d = json.loads(Path(tj).read_text())
        assert "k2" in d["meta"]["gate_fix"], f"{tj}: K2 아님"
        for w in d["walls"]:
            s = w["eval"]["slots_filled"]
            if all(s):
                continue
            fu = s.index(False)
            z = np.load(Path(tj).parent / f"trace_{w['seed']}.npz")
            t, src, tcp, grip, pose = z["t"], z["src"], z["tcp"], z["grip"], z["pose"]
            home = pose[0, :, :2]                                   # supply positions at reset (truth, diagnosis)
            got = [i for i, b in enumerate(w["bricks"]) if any(e["grasped"] for e in b["gates"]["g1"])]
            last = got[-1] if got else -1
            stuck = w["bricks"][last + 1:]
            for b in stuck:
                m = (t >= b["start"]) & (t < b["end"])
                idx = np.flatnonzero(m)
                if len(idx) == 0:
                    continue
                pol = idx[src[idx] == SRC_POLICY]
                closes = [i for i in pol if i > 0 and grip[i - 1] < gc.g0_open_max <= grip[i]]
                ev = []
                for i in closes:
                    xy = tcp[i, :2]
                    dd = np.hypot(*(cells - xy).T)
                    c = int(np.argmin(dd))
                    pres = float(np.hypot(*(pose[i, c, :2] - home[c]))) * 1000 <= a.present_mm
                    ev.append({"t": int(t[i]), "cell": c, "cell_dist_mm": round(float(dd[c]) * 1000, 1),
                               "cell_brick_present_truth": pres, "z_above_app_mm": round(float(tcp[i, 2] - z_app) * 1000, 1),
                               "z_above_grasp_mm": round(float(tcp[i, 2] - z_grasp) * 1000, 1),
                               "fu_cell_dist_mm": round(float(dd[fu]) * 1000, 1)})
                op = [i for i in pol if grip[i] < gc.g0_open_max]
                near = None
                if op:
                    dfu = np.hypot(*(tcp[op, :2] - cells[fu]).T)
                    j = int(np.argmin(dfu))
                    i = op[j]
                    near = {"t": int(t[i]), "dist_mm": round(float(dfu[j]) * 1000, 1),
                            "z_above_app_mm": round(float(tcp[i, 2] - z_app) * 1000, 1),
                            "in_zone_xy": bool(dfu[j] <= gc.g0_zone_xy)}
                    inz = [i for i, dv in zip(op, dfu) if dv <= gc.g0_zone_xy]
                    near["steps_in_zone_xy"] = len(inz)
                    near["steps_in_zone_xy_low"] = sum(1 for i in inz if tcp[i, 2] <= z_app + gc.g0_zone_z)
                    near["steps_in_zone_xy_low_desc"] = sum(1 for i in inz if tcp[i, 2] <= z_app + gc.g0_zone_z
                                                            and i + 1 < len(tcp) and tcp[i + 1, 2] < tcp[i, 2] - 1e-4)
                fu_present = float(np.hypot(*(pose[idx[0], fu, :2] - home[fu]))) * 1000
                rows.append({"seed": w["seed"], "first_unfilled": fu, "turn": b["slot"], "end_reason": b["end_reason"],
                             "g0": len(b["gates"].get("g0", [])), "g1": len(b["gates"]["g1"]),
                             "fu_brick_moved_mm_truth": round(fu_present, 1), "closes": ev, "nearest_open_to_fu_cell": near})
    walls = sorted({r["seed"] for r in rows})
    print(f"멈춘 차례 {len(rows)}개 (벽 {len(walls)}개). 공급칸 중심은 도면값, '칸에 있음'은 참값 (진단). 진단이며 평가 아님.")
    print(f"접근 높이 = 집기 높이 + {ex.cfg.approach * 1000:.0f} mm, G0 구역 = 칸 중심 {gc.g0_zone_xy * 1000:.0f} mm 안 · 접근 높이 + {gc.g0_zone_z * 1000:.0f} mm 아래 · 집게 < {gc.g0_open_max}")
    cnt, seen = Counter(), set()
    for r in rows:
        n = r["nearest_open_to_fu_cell"]
        cl = r["closes"]
        cells_c = Counter(e["cell"] for e in cl)
        emp = sum(1 for e in cl if not e["cell_brick_present_truth"])
        if r["fu_brick_moved_mm_truth"] > a.present_mm:
            kind = "처음 못 채운 자리의 벽돌이 이미 칸에 없음 (참값 (진단))"
        elif n is None:
            kind = "집게를 연 정책 단계 없음"
        elif not n["in_zone_xy"]:
            kind = "그 칸 6 cm 안에 집게를 연 채 안 감"
        elif n["steps_in_zone_xy_low"] == 0:
            kind = "그 칸 위에는 갔으나 너무 높음"
        elif n["steps_in_zone_xy_low_desc"] == 0:
            kind = "그 칸 위 낮게 갔으나 내려가는 중이 아님"
        else:
            kind = "조건은 맞는데 안 작동 (확인 필요)"
        r["kind"] = kind
        r["first_stuck_turn"] = r["seed"] not in seen
        if not r["first_stuck_turn"]:
            continue
        seen.add(r["seed"])
        cnt[kind] += 1
        print(f"{r['seed']} 처음 못 채움 {r['first_unfilled']:2d} 차례 {r['turn']:2d} {r['end_reason']:8s} G0 {r['g0']} G1 {r['g1']} "
              f"| 그 칸까지 최단 {n['dist_mm'] if n else '-'} mm (높이 접근+{n['z_above_app_mm'] if n else '-'}) "
              f"| 닫은 곳 {dict(cells_c)} 빈칸 {emp}/{len(cl)} | {kind}")
    print("분류 (벽마다 첫 멈춘 차례):", dict(cnt))
    if a.out:
        Path(a.out).write_text(json.dumps({"trace_dir": a.trace_dir, "z_app_m": z_app, "z_grasp_m": z_grasp,
                                           "zone": [gc.g0_zone_xy, gc.g0_zone_z, gc.g0_open_max], "kinds": dict(cnt),
                                           "turns": rows}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
