"""역방향 제거 보고서의 표 · 그림 (2026-11-11 밤샘 지시 5절). 결과 파일(compare json · summary)만 읽는다. 시뮬레이션 없음.

표 1 역방향 제거(Full과 짝 비교) · 표 2 사다리 짝 비교(L1 · L2 · L3) · 표 3 Full 자리별 처음 실패 비율(0.05 이상)
· 한눈에 사다리 · 그림: 처음 못 채운 자리 분포 막대(Full / Full − G0 / 검측 없음, 위아래 3칸, 같은 축).
숫자는 모두 개발 배치 100벽(700300 ~ 700399) · 최종 평가 전. 참값 줄은 '참값 (진단)'.

실행: PYTHONPATH=. venv/bin/python -m experiments.report_lf_20261111 --out-md <md> --out-png <png>
"""
import argparse, json
from pathlib import Path

W = Path("outputs/wall3")
NOTE = "개발 배치 100벽(700300 ~ 700399) · 최종 평가 전"

# 사다리: (회차, compare json, 기준 키, [(키, 줄 이름, 판정)])
LADDER = [
    ("L1", "eval_ladder1_20261110_compare.json", "eval_ladder1_20261110/act_all_held", [
        ("eval_ladder1_20261110/act_all_held+k1", "+ G1r (k1)", "하한 > 0 → L2에서 G0 위로 다시 재어 뺌"),
        ("eval_ladder1_20261110/act_all_held+k2", "+ G0 (k2)", "남김 (하한 > 0)"),
        ("eval_ladder1_20261110/act_all_held+k6", "+ G3w (k6)", "보류 (0 걸침, 겨냥 원인 절반 아래로 줄음) → L2에서 뺌"),
    ]),
    ("L2", "eval_ladder2_20261110_compare.json", "eval_ladder2_20261110/act_all_held+k2", [
        ("eval_ladder2_20261110/act_all_held+k2+k1", "+ G1r (k1)", "뺌 (0 걸침)"),
        ("eval_ladder2_20261110/act_all_held+k2+k6", "+ G3w (k6)", "뺌 (0 걸침)"),
        ("eval_ladder2_20261110/act_all_held+k2+k1+k6", "+ G1r + G3w", "뺌 (0 걸침)"),
    ]),
    ("L3", "eval_ladder3a_20261110_compare.json", "eval_ladder3a_20261110/act_all_held+k2", [
        ("eval_ladder3b_20261110/act_all_held+k2+k7", "+ G2w (k7)", None),       # None: 규칙으로 판정
    ]),
]
LADDER_BASE = {"L1": "E3 = G1 + G2 + G3 + H (다시)", "L2": "G0 + G1 + G2 + G3 + H (다시)", "L3": "G0 + G1 + G2 + G3 + H (다시)"}


# 줄 주석 (11-10 밤 확인, 진행상황.md): 이 줄은 참고 줄로 강등
ROW_NOTE = {"Lf07_noG1": "참고 줄: G1을 끄면 G0가 손을 돌려줄 때 정책 리셋이 빠지는 구현 누락이 있어(gates.py `_g1_moment`의 G1 꺼짐 분기), "
                         "G1의 몫이 아니라 그 누락의 영향을 잰 값이다"}


def rule(ci):
    lo, hi = ci
    return "남김 (하한 > 0)" if lo > 0 else ("뺌 (상한 < 0)" if hi < 0 else "뺌 (0 걸침, 마감 규칙 (a))")


def fmt_vs(v):
    return (f"{v['diff_mean']:+.2f} [{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}]".replace("-", "−")) if v else "–"


def row_stats(r):
    return f"{r['filled_mean']:.2f}장 | {r['fill_rate_pct']:.1f} % | {r['complete']} | {r['spec_pass']}"


def first_rates(hist, floor=0.05):
    h = {int(k): v for k, v in hist.items()}
    out = []
    for s in range(16):
        reach = sum(v for k, v in h.items() if k >= s)
        f = h.get(s, 0)
        if reach and f / reach >= floor:
            out.append((s, f, reach, f / reach))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--out-png", default=None)
    a = ap.parse_args()
    for p in (a.out_md, a.out_png):
        if p and Path(p).exists():
            raise SystemExit(f"{p} 이미 있음: 덮어쓰지 않음")
    import experiments.plan_lf_20261110 as P
    lf = json.loads((W / "eval_lf_20261110_compare.json").read_text())
    C = lf["conditions"]
    key = {name: f"eval_lf_{tail}_20261110/act_{cond}" for name, tail, cond, _ in P.ROWS}
    full = C[key["Lf01_full"]]
    md = [f"모든 숫자: {NOTE}. ACT(main_train_v2_20261022/act_s1000 last), env main_v1_eval. 짝 차이 = 같은 시작값끼리 뺀 값의 평균, "
          f"[95 % 구간] = 벽 단위 부트스트랩 10000번.", ""]
    md += ["### 표 1. 역방향 제거 (Full에서 하나씩 끔)", "",
           "| 줄 | 조건 (코드 조건 이름) | 벽당 채운 자리 | 채움률 | 16장 완성 /100 | 시방 합격 /100 | Full 대비 짝 차이 [95 % 구간] | 늘어난 · 줄어든 벽 |",
           "|---|---|---|---|---|---|---|---|"]
    for i, (name, tail, cond, label) in enumerate(P.ROWS, 1):
        r = C.get(key[name])
        if r is None or "error" in r:
            md.append(f"| {i} | {label} (`{cond}`) | 없음{(' (' + r['error'] + ')') if r else ' (돌지 않음 · 포기)'} | | | | | |")
            continue
        v = r.get("vs_base")
        lab2 = f"{label} (`{cond}`)" + (" ※" if name in ROW_NOTE else "")
        md.append(f"| {i} | {lab2} | {row_stats(r)} | {fmt_vs(v) if v else '기준'} | "
                  f"{(str(v['walls_up']) + ' · ' + str(v['walls_down'])) if v else '–'} |")
    md += [""] + [f"- ※ {ROW_NOTE[n]}." for n, _, _, _ in P.ROWS if n in ROW_NOTE]
    md += ["- 한 제어의 몫 = Full − (Full에서 그것만 끈 줄) = 위 짝 차이의 부호를 뒤집은 값.",
           "- (진단) 줄은 G2 · G3의 위치와 손에 든 벽돌 판별을 참값으로 바꾼 조건(`all_truth_held`)이다. 실제 로봇에서는 불가능하다.", ""]
    md += ["### 표 2. 사다리 짝 비교 (같은 회차에서 다시 돌린 기준과)", "",
           "| 회차 | 기준 (값) | 더한 제어 | 벽당 채운 자리 | 16장 완성 | 시방 합격 | 짝 차이 [95 % 구간] | 판정 |", "|---|---|---|---|---|---|---|---|"]
    for lab, f, base, rows in LADDER:
        p = W / f
        if not p.exists():
            md.append(f"| {lab} | (비교 파일 없음 `{f}`) | | | | | | |")
            continue
        D = json.loads(p.read_text())["conditions"]
        b = D[base]
        for k, nm, judge in rows:
            r = D.get(k)
            if r is None or "error" in r:
                md.append(f"| {lab} | {LADDER_BASE[lab]} {b['filled_mean']:.2f}장 | {nm} | 없음 | | | | |")
                continue
            v = r["vs_base"]
            md.append(f"| {lab} | {LADDER_BASE[lab]} {b['filled_mean']:.2f}장 · 완성 {b['complete']} · 시방 {b['spec_pass']} | {nm} | "
                      f"{r['filled_mean']:.2f}장 | {r['complete']} | {r['spec_pass']} | {fmt_vs(v)} | {judge or rule(v['ci95'])} |")
    md += ["", "### 표 3. Full 자리별 처음 실패 비율 (그 자리까지 간 벽 중 거기서 처음 못 채운 벽, 0.05 이상)", "",
           "| 자리 (0부터) | 처음 실패 / 그 자리까지 간 벽 | 비율 |", "|---|---|---|"]
    fr = first_rates(full["first_unfilled_hist"])
    md += [f"| {s} | {f} / {n} | {x:.2f} |" for s, f, n, x in fr] or ["| – | 0.05 이상 없음 | – |"]
    md += ["", f"- Full 처음 못 채운 자리 분포(벽 수, 16 = 완성): {full['first_unfilled_hist']}", ""]
    md += ["### 한눈에 사다리", "", "| 시점 | 설정 | 벽당 채운 자리 | 16장 완성 | 시방 합격 |", "|---|---|---|---|---|",
           "| 10-22 | 처음 학습한 정책 + G1 · G2 · G3 | 2.08 | 0 | 0 |",
           "| 10-26 | 손목 방향 통일 데이터로 재학습 + G1 · G2 · G3 | 5.38 | 0 | 0 |",
           "| 11-04 | + H (손에 든 벽돌 기준) | 8.55 | 12 | 11 |"]
    for lab, f, base, rows in LADDER:
        p = W / f
        if p.exists():
            D = json.loads(p.read_text())["conditions"]
            for k, nm, judge in rows:
                r = D.get(k)
                if r and "error" not in r and (judge or rule(r["vs_base"]["ci95"])) == "남김 (하한 > 0)":   # 끝까지 남긴 것만
                    md.append(f"| 11-10 {lab} | 그 회차 기준 {nm} | {r['filled_mean']:.2f} | {r['complete']} | {r['spec_pass']} |")
    md.append(f"| 11-10 Lf | **Full** (`{P.FULL_COND}`) | **{full['filled_mean']:.2f}** | **{full['complete']}** | **{full['spec_pass']}** |")
    Path(a.out_md).write_text("\n".join(md) + "\n")
    print("\n".join(md))
    if a.out_png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager as fm
        fp = [f for f in fm.findSystemFonts() if "NotoSerifCJK-Regular" in f or "NotoSansCJK-Regular" in f]
        if fp:
            fm.fontManager.addfont(fp[0])
            plt.rcParams["font.family"] = fm.FontProperties(fname=fp[0]).get_name()
        ink, ink2, grid, surf = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
        series = [("Lf01_full", "Full", "#2a78d6"), ("Lf03_noG0", "Full − G0", "#eb6834"), ("Lf02_none", "검측 없음 (정책만)", "#1baf7a")]
        fig, axs = plt.subplots(3, 1, figsize=(9, 7.2), sharex=True, sharey=True, facecolor=surf)
        xs = list(range(17))
        for ax, (name, lab, col) in zip(axs, series):
            r = C.get(key[name])
            ax.set_facecolor(surf)
            if r is None or "error" in r:
                ax.text(0.5, 0.5, f"{lab}: 결과 없음", transform=ax.transAxes, ha="center", color=ink2)
                continue
            h = {int(k): v for k, v in r["first_unfilled_hist"].items()}
            ys = [h.get(x, 0) for x in xs]
            ax.bar(xs, ys, width=0.6, color=col, linewidth=0, zorder=3)
            for x, y in zip(xs, ys):
                if y >= 5:
                    ax.text(x, y + 1.5, str(y), ha="center", va="bottom", fontsize=8, color=ink2)
            ax.set_title(f"{lab}: 벽당 {r['filled_mean']:.2f}장 · 16장 완성 {r['complete']} · 시방 합격 {r['spec_pass']} (/100)",
                         loc="left", fontsize=10, color=ink)
            ax.grid(axis="y", color=grid, linewidth=0.8, zorder=0)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                ax.spines[s].set_color(grid)
            ax.tick_params(colors=ink2, labelsize=8)
            ax.set_ylabel("벽 수", color=ink2, fontsize=9)
        axs[-1].set_xticks(xs, [str(x) for x in range(16)] + ["완성"])
        axs[-1].set_xlabel("처음 못 채운 자리 (0부터, '완성' = 16장 다 채움)", color=ink2, fontsize=9)
        axs[0].set_ylim(0, 115)
        fig.suptitle(f"처음 못 채운 자리 분포 ({NOTE})", fontsize=11, color=ink, x=0.01, ha="left")
        fig.tight_layout()
        fig.savefig(a.out_png, dpi=200, facecolor=surf)
        print("그림:", a.out_png)


if __name__ == "__main__":
    main()
