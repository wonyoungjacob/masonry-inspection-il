"""Full − G1 다시 (2026-11-11, 사용자 승인). G1 꺼짐 + G0의 정책 리셋 누락을 고친 뒤 그 줄만 다시 잰다.

11-10 Lf07(`eval_lf_noG1_20261110`)은 G1이 꺼지면 G0가 손을 돌려줄 때 정책 리셋이 빠진 채 쟀다(gates.py `_g0_grasp`, 11-11 고침).
고친 코드가 바뀌는 경우는 'G0 켜짐 + G1 꺼짐'뿐이다. Full과 다른 Lf 줄(G1 켜짐, 또는 G0 없음)의 실행 경로는 그대로라, Full은 11-10 값을 기준으로 짝 비교한다.
같은 100벽(700300 ~ 700399), 같은 조건 이름, 같은 프로세스 8개. 새 출력 폴더(덮어쓰지 않음).

실행: venv/bin/python -m wall3.supervisor --plan experiments.plan_lf_g1fix_20261111
"""
W = "outputs/wall3"
PY = "venv/bin/python"
SUP_NAME = "supervisor_lf_g1fix_20261111"
TRAIN = f"{W}/main_train_v2_20261022"
FULL = f"{W}/eval_lf_full_20261110"
OUT = f"{W}/eval_lf_noG1fix_20261111"
COND = "g2g3_held+k2+k7"                     # Full − G1 (11-10 Lf07과 같은 조건 이름)

STAGES = [
    ("Lf07b_noG1fix", [dict(name="Lf07b_noG1fix", kind="summary", module="wall3.eval_gates", out=OUT,
                            cmd=[PY, "-m", "wall3.eval_gates", "--out", OUT, "--train-dir", TRAIN, "--policies", "act",
                                 "--conditions", COND, "--chunks", "8", "--parallel", "8", "--env-version", "main_v1_eval"])]),
    ("Lf07b_compare", [dict(name="Lf07b_compare", kind="rc", module="wall3.compare_ladder", out=f"{W}/eval_lf_g1fix_20261111_compare.json",
                            cmd=[PY, "-m", "wall3.compare_ladder", "--evals", FULL, f"{W}/eval_lf_noG1_20261110", OUT,
                                 "--base", "eval_lf_full_20261110/act_all_held+k2+k7", "--out", f"{W}/eval_lf_g1fix_20261111_compare.json"],
                            log=f"{W}/eval_lf_g1fix_20261111_compare.txt")]),
]
