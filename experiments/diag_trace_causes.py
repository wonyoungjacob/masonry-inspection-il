"""diag_trace 결과로 못 채운 자리마다 원인을 가린다 (2026-10-28, 진단용 · 정답 좌표는 분석에만).

벽돌마다 움직인 구간(스텝 사이 1.5 mm 넘게 움직임, 6스텝 안 끊김은 한 구간)을 뽑아, 손(TCP)과 5 cm 안에서 함께 움직인 구간 = 운반, 아니면 = 밀림 · 넘어짐으로 본다.
구간을 시작한 주체(정책 / G1 / G2·G3)와 그때의 차례 번호 k(rollout이 정책에 준 목표 자리)를 붙인다.
못 채운 자리 j(평가의 slots_filled 기준)마다 벽돌 j의 최종 상태(어디에 어떤 모양으로)와 마지막으로 옮긴 주체 · 차례를 붙인다:
  최종 상태: 공급대 제자리 / 공급대 다른 곳 / 제자리에서 이웃 벽돌 끝에 걸쳐 기울어짐 / 제자리에서 45 mm 넘게 벗어남 /
            다른 자리에 들어감 / 벽 뒤 바닥 / 벽 앞 · 옆 바닥 / 벽 위 엉뚱한 곳
  주체: 마지막으로 손에 쥐고 옮기거나(벽돌이 손과 5 cm 안에서 함께 움직임) 30 mm 넘게 민 동작이 정책인지 G2 · G3인지, 그때 차례 k
        (1.5 s 안에 이어진 움직임은 한 덩어리: 놓은 뒤 넘어지거나 미끄러진 것은 놓은 주체에 붙임)
  못 집음(한 번도 안 옮겨짐): 정책이 집게를 닫기 시작한 손 위치에 가장 가까운 처음 공급 위치의 벽돌 m으로 세분
     (m = j: 그 벽돌 앞에서 헛집음, m ≠ j이고 m이 이미 공급대에 없음: 옮겨진 벽돌 m의 빈자리를 집으려 함)

실행: venv/bin/python -m experiments.diag_trace_causes outputs/wall3/diag_trace_20261028

2026-11-03 선택 옵션 (10-29 검증 반영, 우분투지시_검측정보진단_20261103.md 3절 C). 옵션 없이 돌리면 10-28과 같은 분류다.
  --aim-max-mm 50 : '못 집음'의 노린 공급칸은 손이 그 처음 공급 위치 50 mm 안에서 오므린 경우만. 더 멀면 '공급칸 밖에서 오므림'
  --perch-mm 20   : 평가가 '채움'으로 셌지만 그 자리를 차지한 벽돌(평가의 짝짓기)이 설계 높이보다 20 mm 넘게 뜬 자리(이웃 벽돌에
                    걸침)를 따로 표시하고, '처음 문제 자리' = min(처음 못 채운 자리, 처음 걸친 자리) 표를 함께 낸다
  추적에 검측 기준 자리(tgt, 2026-11-03 diag_trace)가 있으면 G2·G3 동작에 '기준 자리 = 그 벽돌의 자리인지'를 붙인다
  --out FILE      : 결과 json (기본 <root>/causes.json, 이미 있으면 멈춤)
"""
import argparse, collections, json, sys
from pathlib import Path

import numpy as np

SRCN = {0: "정책", 1: "G1", 2: "G2·G3", 3: "마무리", 4: "G0"}   # 4: 2026-11-10 K2 G0 집기 정렬
EXIST_XY, EXIST_Z, EXIST_TILT = 0.045, 0.0285, 45.0          # = spec_metrics (채움 기준)


def tilt_deg(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.degrees(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1)))


def segments(P, thr=0.0015, gap=6):
    """Per brick: (start, end) step index of motion stretches."""
    d = np.r_[np.zeros((1, P.shape[1])), np.linalg.norm(np.diff(P, axis=0), axis=2)]
    out = {}
    for b in range(P.shape[1]):
        m = np.flatnonzero(d[:, b] > thr)
        segs = []
        if len(m):
            s = e = m[0]
            for i in m[1:]:
                if i - e <= gap:
                    e = i
                else:
                    segs.append((int(s), int(e)))
                    s = e = i
            segs.append((int(s), int(e)))
        out[b] = segs
    return out


def analyse(z, wall, design, aim_max=None, perch=None):
    k, src, pose, tcp, grip = z["k"], z["src"], z["pose"].astype(float), z["tcp"].astype(float), z["grip"]
    tgt = z["tgt"] if "tgt" in z.files else None
    P, Q = pose[:, :, :3], pose[:, :, 3:7]
    n = P.shape[1]
    D = np.array([[s["x"], s["y"], s["z"]] for s in design])
    inslot = (np.hypot(P[:, :, 0] - D[None, :, 0], P[:, :, 1] - D[None, :, 1]) <= EXIST_XY) & \
             (np.abs(P[:, :, 2] - D[None, :, 2]) <= EXIST_Z) & (tilt_deg(Q) < EXIST_TILT)
    segs = segments(P)
    supply0 = P[0]
    carried = {b: [] for b in range(n)}
    moves = {b: [] for b in range(n)}
    for b in range(n):
        for s, e in segs[b]:
            a = max(s - 1, 0)
            lift = float((P[s:e + 1, b, 2].max() - P[a, b, 2]) * 1000)
            dxy = float(np.hypot(*(P[e, b] - P[a, b])[:2]) * 1000)
            if lift < 5 and dxy < 3:
                continue
            near = int(np.argmin(np.hypot(D[:, 0] - P[e, b, 0], D[:, 1] - P[e, b, 1]) + 10 * (np.abs(D[:, 2] - P[e, b, 2]) > EXIST_Z)))
            held = float(np.median(np.linalg.norm(tcp[s:e + 1, :3] - P[s:e + 1, b], axis=1))) < 0.05     # moves with the hand
            mv = {"s": s, "e": e, "k": int(k[s]), "src": int(src[s]), "carry": bool(held and (lift > 15 or dxy > 30)), "held": held,
                  "lift_mm": round(lift), "dxy_mm": round(dxy),
                  "end": P[e, b].round(3).tolist(), "end_near_slot": near,
                  "end_near_mm": round(float(np.hypot(*(P[e, b, :2] - D[near, :2])) * 1000)), "end_in_own": bool(inslot[e, b])}
            if tgt is not None:
                mv["tgt"] = int(tgt[s])
            moves[b].append(mv)
            if mv["carry"]:
                carried[b].append(mv)
    # policy grasp attempts: grip command starts closing in a policy step -> nearest initial supply spot
    on = np.flatnonzero((grip[1:] > 0.15) & (grip[:-1] <= 0.15)) + 1
    attempts = []
    for i in on:
        if src[i] != 0:
            continue
        dd = np.hypot(supply0[:, 0] - tcp[i, 0], supply0[:, 1] - tcp[i, 1])
        m = int(np.argmin(dd))
        if aim_max is not None and dd[m] * 1000 > aim_max:
            attempts.append({"i": int(i), "k": int(k[i]), "aim": None, "aim_mm": round(float(dd[m]) * 1000), "aim_gone": None})
            continue
        attempts.append({"i": int(i), "k": int(k[i]), "aim": m, "aim_mm": round(float(dd[m]) * 1000),
                         "aim_gone": bool(np.hypot(*(P[i, m, :2] - supply0[m, :2])) > 0.02)})
    filled = wall["eval"]["slots_filled"]
    match = wall["eval"].get("match")
    out = []
    todo = [(j, j, False) for j in range(n) if not filled[j]]
    if perch is not None and match is not None:                 # 'filled' slot j whose occupant brick sits > perch mm high
        for j in range(n):
            o = match[j]
            if filled[j] and o >= 0 and (P[-1, o, 2] - D[j, 2]) * 1000 > perch:
                todo.append((j, o, True))
        todo.sort()
    for j_slot, j, perched in todo:
        fp, tl = P[-1, j], float(tilt_deg(Q[-1, j]))
        dj = float(np.hypot(*(fp[:2] - D[j_slot, :2])) * 1000)
        dz = float((fp[2] - D[j_slot, 2]) * 1000)
        on_sup = bool(fp[2] > 0.3)                                  # supply table top ~0.35 m
        untouched = bool(np.hypot(*(fp[:2] - supply0[j, :2])) < 0.05) and on_sup
        near = int(np.argmin(np.hypot(D[:, 0] - fp[0], D[:, 1] - fp[1]) + 10 * (np.abs(D[:, 2] - fp[2]) > EXIST_Z)))
        dn = float(np.hypot(*(fp[:2] - D[near, :2])) * 1000)
        # final state
        if perched:
            st = "채움으로 셌지만 이웃 벽돌에 걸침(+%d mm 넘게 뜸)" % perch
        elif untouched:
            st = "공급대 제자리(안 옮겨짐)"
        elif on_sup:
            st = "공급대 위 다른 곳"
        elif dj <= 45 and dz > 10 and tl >= 5:
            st = "제자리에서 이웃 벽돌 끝에 걸쳐 기울어짐"
        elif dj <= 80 and abs(dz) <= EXIST_Z:
            st = "제자리에서 45 mm 넘게 벗어남"
        elif near != j and dn <= 45 and abs(fp[2] - D[near, 2]) <= EXIST_Z:
            st = f"다른 자리 {near}에 들어감"
        elif fp[2] < 0.08 and fp[1] > 0.62:
            st = "벽 뒤 바닥으로 떨어짐"
        elif fp[2] < 0.08:
            st = "벽 앞 · 옆 바닥으로 떨어짐"
        else:
            st = "벽 위 엉뚱한 곳"
        # who moved it last: moves with <= 30 steps (1.5 s) between them form one chain (a carry and the slide / tip-over after the
        # release belong to the carry); the chain's actor = its first carry, else its first move (a push)
        chains = []
        for m in moves[j]:
            c = chains[-1] if chains else None
            same = c is not None and m["s"] - c[-1]["e"] <= 30 and (not m["carry"] or (c[-1]["carry"] and c[-1]["src"] == m["src"]))
            if same:                                 # a fall / slide right after, or the same hand still carrying
                c.append(m)
            else:
                chains.append([m])
        big = []
        for c in chains:
            disp = float(np.hypot(*(np.array(c[-1]["end"][:2]) - P[max(c[0]["s"] - 1, 0), j, :2])) * 1000)
            car = [m for m in c if m["carry"]]
            if car or disp > 30:
                big.append(dict(car[0] if car else c[0], chain_end=c[-1]["end"], chain_disp_mm=round(disp)))
        wall_carry = [m for m in moves[j] if m["carry"] and m["dxy_mm"] > 300 and m["src"] == 0]
        r = {"slot": j_slot, "brick": j, "perched": perched, "final": fp.round(3).tolist(), "final_tilt_deg": round(tl, 1), "final_dxy_mm": round(dj), "final_dz_mm": round(dz),
             "state": st, "moves": moves[j], "policy_transport_k": [m["k"] for m in wall_carry],
             "gate_moved": [m["k"] for m in moves[j] if m["src"] in (2, 4) and (m["carry"] or m["dxy_mm"] > 30)]}
        if untouched or not big:
            att = [x for x in attempts if x["k"] == j]
            aims = collections.Counter((x["aim"], x["aim_gone"]) for x in att)
            r.update(attempts=len(att), aims=[[m, g, c] for (m, g), c in aims.most_common()])
            if not att:
                why = "정책이 집으려 하지 않음"
            else:
                m, g, _ = r["aims"][0]
                why = ("공급칸 밖에서 집게를 오므림" if m is None else "자기 벽돌 앞에서 헛집음" if m == j else
                       f"이미 옮겨진 벽돌 {m}의 빈 공급 위치를 집으려 함" if g else f"아직 공급대에 있는 다른 벽돌 {m}을 노림")
            r.update(actor="못 집음", why=why, text=f"못 집음: {why} (차례 {j} 시도 {len(att)}회) → {st}")
        else:
            m = big[-1]
            who = SRCN[m["src"]]
            ref = ""
            if m["src"] == 2 and "tgt" in m:                         # the gate's reference slot when it moved this brick
                ref = "(기준=제 자리)" if m["tgt"] == j else "(기준≠제 자리)"
            if m["carry"]:
                act = f"{who}가 차례 {m['k']}에 들어 옮김" + (f"(검측 기준 자리 {m['tgt']})" if ref else "")
            else:
                busy = [b for b in range(n) if b != j and any(c["s"] - 2 <= m["s"] <= c["e"] + 2 for c in carried[b])]
                act = f"{who} 동작 중 밀림(차례 {m['k']}" + (f", 그때 벽돌 {busy} 운반 중)" if busy else ")")
            r.update(actor=({2: "G2·G3", 4: "G0"}.get(m["src"], "정책")) + (" 운반" if m["carry"] else " 밀림") + ref, last_k=m["k"],
                     text=f"{act} → {st} (xy {dj:.0f} mm, 높이 {dz:+.0f} mm, 기울기 {tl:.0f}°)")
        out.append(r)
    return out, moves, attempts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--aim-max-mm", type=float, default=None)
    ap.add_argument("--perch-mm", type=float, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    root = Path(a.root)
    out_json = Path(a.out) if a.out else root / "causes.json"
    if out_json.exists():
        raise SystemExit(f"{out_json} 이미 있음: 덮어쓰지 않음")
    from wall3.config import scene_config
    from wall3.scene import WallEnv
    res = {}
    walls = []
    for part in sorted(root.glob("part*/trace.json")):
        walls += [(part.parent, w) for w in json.loads(part.read_text())["walls"]]
    env = WallEnv(scene_config(json.loads(next(root.glob("part*/trace.json")).read_text())["meta"]["env_version"]))
    for d, w in walls:
        env.reset(w["seed"])
        z = np.load(d / f"trace_{w['seed']}.npz")
        fails, moves, attempts = analyse(z, w, env.design, a.aim_max_mm, a.perch_mm)
        first = next((j for j in range(16) if not w["eval"]["slots_filled"][j]), 16)
        res[w["seed"]] = {"same_as_eval": w.get("same_as_eval"), "first_unfilled": first, "fails": [f for f in fails if not f.get("perched")],
                          "moves": {b: m for b, m in moves.items() if m}, "attempts": attempts}
        if a.perch_mm is not None:
            pf = [f for f in fails if f.get("perched")]
            res[w["seed"]]["perched"] = pf
            res[w["seed"]]["first_problem"] = min([first] + [f["slot"] for f in pf])
    env.close()
    out_json.write_text(json.dumps({"_options": {"aim_max_mm": a.aim_max_mm, "perch_mm": a.perch_mm}, **res} if (a.aim_max_mm or a.perch_mm) else res,
                                   indent=1, default=float))
    for s, r in res.items():
        print(f"{s} 처음 못 채운 자리 {r['first_unfilled']} 재현 {r['same_as_eval'] and r['same_as_eval']['slots_filled']}")
        for f in r["fails"]:
            if f["slot"] <= 8:
                print(f"   자리 {f['slot']}: {f['text']}")
    print("\n== 요약 ==")
    print("처음 못 채운 자리(0부터):", sorted(collections.Counter(r["first_unfilled"] for r in res.values()).items()))
    print("처음 못 채운 자리 뒤에 채운 자리가 있는 벽:", [(s, r["first_unfilled"], [f for f in range(r["first_unfilled"] + 1, 16)
          if f not in {x["slot"] for x in r["fails"]}]) for s, r in res.items()
          if any(f not in {x["slot"] for x in r["fails"]} for f in range(r["first_unfilled"] + 1, 16))])

    def label(f, first):
        if f["actor"] == "못 집음":
            w = f["why"]
            w = "이미 옮겨진 벽돌의 빈 공급 위치를 집으려 함" if w.startswith("이미 옮겨진") else "아직 있는 다른 벽돌을 노림" if w.startswith("아직") else w
            return f"못 집음: {w}"
        return f"{f['actor']} → {f['state']}"
    tables = [("처음 못 채운 자리", "first_unfilled", "fails", lambda f, r: f["slot"] == r["first_unfilled"]),
              ("그 뒤 못 채운 자리", "first_unfilled", "fails", lambda f, r: f["slot"] > r["first_unfilled"])]
    if a.perch_mm is not None:
        tables.append(("처음 문제 자리(걸침 포함)", "first_problem", "both",
                       lambda f, r: f["slot"] == r["first_problem"]))
        print("\n걸친 채움 자리(0부터):", sorted(collections.Counter(f["slot"] for r in res.values() for f in r["perched"]).items()),
              "| 걸침이 처음 못 채운 자리보다 앞인 벽:", sum(r["first_problem"] < r["first_unfilled"] for r in res.values()))
    for name, key, which, pick in tables:
        tab = collections.defaultdict(collections.Counter)
        for r in res.values():
            fs = r["fails"] + (r.get("perched", []) if which == "both" else [])
            for f in fs:
                if pick(f, r):
                    tab[label(f, r[key])][f["slot"]] += 1
        print(f"\n{name}: 원인 × 자리(0부터)")
        for lab, c in sorted(tab.items()):
            print(f"  {lab}: 계 {sum(c.values())} | " + " ".join(f"{k}:{v}" for k, v in sorted(c.items())))


if __name__ == "__main__":
    main()
