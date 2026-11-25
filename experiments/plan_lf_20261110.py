"""역방향 제거 (leave-one-out) 계획 (2026-11-10 밤샘, 01_사기설/실험파이프라인_실패원인제거_20261110.md 6절 · 11-10 마감 규칙).

Full = G0 + G1 + G2 + G3 + H (+ G2w: L3 짝 비교 구간 하한 > 0이면, 마감 규칙 (a)). G1r · G3w는 L2에서 뺐다.
지시(01_사기설/11.10일 우분투 프롬프트.txt) 1절 우선순위대로 한 조건씩 단계 하나(조건 안 프로세스 8개). 시간이 모자라면 뒤부터 포기.
  Lf01 Full · Lf02 검측 없음 · Lf03 Full − G0 · Lf04 Full − H · Lf05 Full − G2 (G2w도 꺼짐) · Lf06 Full − G3 · Lf07 Full − G1
  · Lf08 Full − G2w (Full에 G2w가 있을 때만) · Lf09 (진단) Full, 위치 = 참값 (진단) (기존 조건 이름 all_truth_held: 손에 든 벽돌도 참값으로 판별)
  → Lf_compare (Full과 짝 비교) → Lf_trace (Full 추적 5묶음) → Lf_causes → Lf_rank → (선택) Lf_diffusion (Diffusion Full 1줄, 참고)
조건은 결과를 보기 전에 정한다: K7_IN_FULL은 L3 판정 직후, Lf 시작 전에 채우고 커밋한다. 시작값 700300 ~ 700399만.

실행: venv/bin/python -m wall3.supervisor --plan experiments.plan_lf_20261110
"""
K7_IN_FULL = True            # L3 판정 (11-10 19:00): E3+G0+G2w 대 E3+G0 다시 +0.41 [+0.01, +0.81], 하한 > 0 → 넣음 (마감 규칙 (a))

if K7_IN_FULL not in (True, False):
    raise SystemExit("plan_lf_20261110: K7_IN_FULL을 L3 판정으로 채운 뒤 돌린다")

W = "outputs/wall3"
PY = "venv/bin/python"
SUP_NAME = "supervisor_lf_20261110"
TRAIN = f"{W}/main_train_v2_20261022"
CKPT = f"{TRAIN}/act_s1000/checkpoints/last/pretrained_model"
FIX = "+k2+k7" if K7_IN_FULL else "+k2"          # Full의 추가 제어 (G0, G2w)
FIX_NO_G0 = "+k7" if K7_IN_FULL else ""
FIX_NO_G2 = "+k2"                                  # G2를 끄면 G2w도 꺼짐

# (단계 이름, 출력 폴더 꼬리, 조건 이름, 줄 이름)
ROWS = [
    ("Lf01_full", "full", f"all_held{FIX}", "Full"),
    ("Lf02_none", "none", "none", "검측 없음"),
    ("Lf03_noG0", "noG0", f"all_held{FIX_NO_G0}", "Full − G0"),
    ("Lf04_noH", "noH", f"all{FIX}", "Full − H"),
    ("Lf05_noG2", "noG2", f"g1g3_held{FIX_NO_G2}", "Full − G2"),
    ("Lf06_noG3", "noG3", f"g1g2_held{FIX}", "Full − G3"),
    ("Lf07_noG1", "noG1", f"g2g3_held{FIX}", "Full − G1"),
] + ([("Lf08_noG2w", "noG2w", "all_held+k2", "Full − G2w")] if K7_IN_FULL else []) + [
    ("Lf09_truth", "truth", f"all_truth_held{FIX}", "(진단) Full, 위치 = 참값 (진단)"),
]


def out_of(tail):
    return f"{W}/eval_lf_{tail}_20261110"


def ev(name, out, cond, policy="act"):
    return (name, [dict(name=name, kind="summary", module="wall3.eval_gates", out=out,
                        cmd=[PY, "-m", "wall3.eval_gates", "--out", out, "--train-dir", TRAIN, "--policies", policy,
                             "--conditions", cond, "--chunks", "8", "--parallel", "8", "--env-version", "main_v1_eval"])])


def seeds(i, n=100, chunks=5, s0=700300):
    k = n // chunks
    return list(range(s0 + i * k, s0 + (i + 1) * k))


FULL_COND = ROWS[0][2]
T = f"{W}/diag_trace_lf_full_20261110"
STAGES = [ev(name, out_of(tail), cond) for name, tail, cond, _ in ROWS] + [
    ("Lf_compare", [dict(name="Lf_compare", kind="rc", module="wall3.compare_ladder", out=f"{W}/eval_lf_20261110_compare.json",
                         cmd=[PY, "-m", "wall3.compare_ladder", "--evals", *[out_of(t) for _, t, _, _ in ROWS],
                              "--base", f"eval_lf_full_20261110/act_{FULL_COND}", "--out", f"{W}/eval_lf_20261110_compare.json"],
                         log=f"{W}/eval_lf_20261110_compare.txt")]),
    ("Lf_trace", [dict(name=f"Lf_trace_part{i}", kind="rc", module="experiments.diag_trace", out=f"{T}/part{i}",
                       cmd=[PY, "-m", "experiments.diag_trace", "--ckpt", CKPT, "--env-version", "main_v1_eval",
                            "--seeds", *map(str, seeds(i)), "--out", f"{T}/part{i}",
                            "--gate-pose-source", "camera", "--gate-target", "held_grasp",
                            "--gate-fix", *FIX.strip("+").split("+"),
                            "--compare", *[f"{out_of('full')}/act_{FULL_COND}_part{c}/rollout.json" for c in range(8)]])
                  for i in range(5)]),
    ("Lf_causes", [dict(name="Lf_causes", kind="rc", module="experiments.diag_trace_causes", out=f"{T}/causes.json",
                        cmd=[PY, "-m", "experiments.diag_trace_causes", T, "--aim-max-mm", "50", "--perch-mm", "20",
                             "--out", f"{T}/causes.json"], log=f"{T}/causes.txt")]),
    ("Lf_rank", [dict(name="Lf_rank", kind="rc", module="experiments.rank_fail_causes", out=f"{T}/rank.json",
                      cmd=[PY, "-m", "experiments.rank_fail_causes", "--trace", T, "--out", f"{T}/rank.json"],
                      log=f"{T}/rank.txt")]),
    ev("Lf_diffusion", f"{W}/eval_lf_full_diffusion_20261110", FULL_COND, policy="diffusion"),
]
