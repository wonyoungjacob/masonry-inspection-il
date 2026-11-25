"""실패 원인 제거 사다리 3단계 계획 (2026-11-10, 01_사기설/실험파이프라인_실패원인제거_20261110.md 5절). wall3.supervisor가 돌린다.

L2 판정(4절 규칙): G1r(k1) · G3w(k6)은 G0(k2) 위에서 구간이 0을 걸치고 겨냥한 원인이 줄지 않아 뺀다. 기준 = E3 + G0.
밤샘 지시(01_사기설/11.10일 우분투 프롬프트.txt) "한 번에 한 조건": 조건마다 단계 하나, 조건 안에서 프로세스 8개(시작값 8묶음).
L3_base   : 같은 100벽(700300 ~ 700399)에서 E3+G0 다시 (같은 묶음 기준)          → eval_ladder3a_20261110
L3_k7     : E3+G0+G2w (k7, 다시 매긴 1위 자리 11 겨냥)                          → eval_ladder3b_20261110
L3_compare: E3+G0 다시와 짝 비교 (L1 · L2 조건도 표에)
조건은 L3 결과를 보기 전에 정했다(04b9ef4, 18:15 추적 빼고 조건마다 단계로 나눔 · L3 시작 전). 계산만 한다.

실행: venv/bin/python -m wall3.supervisor --plan experiments.plan_ladder3_20261110
"""
W = "outputs/wall3"
PY = "venv/bin/python"
SUP_NAME = "supervisor_ladder3_20261110"
TRAIN = f"{W}/main_train_v2_20261022"
L1 = f"{W}/eval_ladder1_20261110"
L2 = f"{W}/eval_ladder2_20261110"
L3A = f"{W}/eval_ladder3a_20261110"
L3B = f"{W}/eval_ladder3b_20261110"


def ev(name, out, cond):
    return (name, [dict(name=name, kind="summary", module="wall3.eval_gates", out=out,
                        cmd=[PY, "-m", "wall3.eval_gates", "--out", out, "--train-dir", TRAIN, "--policies", "act",
                             "--conditions", cond, "--chunks", "8", "--parallel", "8", "--env-version", "main_v1_eval"])])


STAGES = [
    ev("L3_base", L3A, "all_held+k2"),
    ev("L3_k7", L3B, "all_held+k2+k7"),
    ("L3_compare", [dict(name="L3_compare", kind="rc", module="wall3.compare_ladder", out=f"{L3A}_compare.json",
                         cmd=[PY, "-m", "wall3.compare_ladder", "--evals", L3A, L3B, L2, L1,
                              "--base", "eval_ladder3a_20261110/act_all_held+k2", "--out", f"{L3A}_compare.json"],
                         log=f"{L3A}_compare.txt")]),
]
