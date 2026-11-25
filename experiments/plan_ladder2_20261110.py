"""실패 원인 제거 사다리 2단계 계획 (2026-11-10, 01_사기설/실험파이프라인_실패원인제거_20261110.md 5절). wall3.supervisor가 돌린다.

L1 판정: K2 남김 · K1 남김 · K6 보류 (5.1절). 새 기준 = E3 + K2.
L2_eval   : 같은 100벽(700300 ~ 700399)에서 E3+K2 다시(같은 묶음 기준) · +K1 · +K6 · +K1+K6 (ACT)
L2_compare: 같은 묶음 E3+K2와 짝 비교, L1 조건들과도 함께 표로 (평가 바로 뒤)
조건은 L2 결과를 보기 전에 정했다. 계산만 한다. 판정 · 보고 · 커밋은 결과를 읽고 한다.
추적(원인 다시 나누기)은 L2 판정으로 남길 설정이 정해진 뒤 따로 돌린다.

실행: venv/bin/python -m wall3.supervisor --plan experiments.plan_ladder2_20261110
"""
W = "outputs/wall3"
PY = "venv/bin/python"
SUP_NAME = "supervisor_ladder2_20261110"
L1 = f"{W}/eval_ladder1_20261110"
L2 = f"{W}/eval_ladder2_20261110"
CONDS = ["all_held+k2", "all_held+k2+k1", "all_held+k2+k6", "all_held+k2+k1+k6"]

STAGES = [
    ("L2_eval", [dict(name="L2_eval", kind="summary", module="wall3.eval_gates", out=L2,
                      cmd=[PY, "-m", "wall3.eval_gates", "--out", L2, "--train-dir", f"{W}/main_train_v2_20261022",
                           "--policies", "act", "--conditions", *CONDS, "--chunks", "5", "--parallel", "8",
                           "--env-version", "main_v1_eval"])]),
    ("L2_compare", [dict(name="L2_compare", kind="rc", module="wall3.compare_ladder", out=f"{L2}_compare.json",
                         cmd=[PY, "-m", "wall3.compare_ladder", "--evals", L2, L1,
                              "--base", "eval_ladder2_20261110/act_all_held+k2", "--out", f"{L2}_compare.json"],
                         log=f"{L2}_compare.txt")]),
]
