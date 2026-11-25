"""정책 없음 · G2t 계획 (2026-11-16, 사용자 지시). 조건은 결과 전에 정했다(experiments/eval_20261116.py CONDITIONS).

  N01_nopolicy  정책 없음: 대본 제어기 + Full 검측        → outputs/wall3/eval_nopolicy_20261116
  N02_g2t       Full + G2t: ACT + Full 검측, G2가 운반   → outputs/wall3/eval_g2t_20261116
  (둘은 한 단계에서 동시에, 각 프로세스 8개)
  N_compare     Full(eval_lf_full_20261110, ⑤ 구성 기준 실행)과 같은 100벽 짝 비교 (wall3.compare_ladder)
시작값 700300 ~ 700399(개발 배치), 최종 평가 아님. 참값 금지(--gate-pose-source truth · --gate-target held_truth 없음).

실행: venv/bin/python -m wall3.supervisor --plan experiments.plan_nopolicy_g2t_20261116
"""
W = "outputs/wall3"
PY = "venv/bin/python"
SUP_NAME = "supervisor_nopolicy_g2t_20261116"
NOPOL = f"{W}/eval_nopolicy_20261116"
G2T = f"{W}/eval_g2t_20261116"
FULL = f"{W}/eval_lf_full_20261110"
CMP = f"{W}/eval_nopolicy_g2t_20261116_compare.json"


def ev(name, out, cond):
    return dict(name=name, kind="summary", module="experiments.eval_20261116", out=out,
                cmd=[PY, "-m", "experiments.eval_20261116", "--out", out, "--condition", cond, "--chunks", "8", "--parallel", "8",
                     "--env-version", "main_v1_eval"])


STAGES = [
    ("N_eval", [ev("N01_nopolicy", NOPOL, "script_all_held+k2+k7"), ev("N02_g2t", G2T, "act_all_held+k2+k7+g2t")]),
    ("N_compare", [dict(name="N_compare", kind="rc", module="wall3.compare_ladder", out=CMP,
                        cmd=[PY, "-m", "wall3.compare_ladder", "--evals", FULL, NOPOL, G2T,
                             "--base", "eval_lf_full_20261110/act_all_held+k2+k7", "--out", CMP],
                        log=f"{W}/eval_nopolicy_g2t_20261116_compare.txt")]),
]
