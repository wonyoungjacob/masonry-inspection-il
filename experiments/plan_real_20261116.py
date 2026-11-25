"""현실 제약 ablation 계획 (2026-11-16, 01_사기설/우분투지시_현실제약_ablation_20261116.md). 조건은 결과 전에 정했다.

칸마다 한 단계(하이브리드 · 제어만 동시, 조건마다 프로세스 6개 = 최대 12개), 칸은 차례로: 제약 없음 → A → B → A + B → 보고.
시작값 700300 ~ 700399(개발 배치), 최종 평가 아님. 참값은 검측 · 대본의 판단과 동작에 쓰지 않는다(기록에만).

실행: venv/bin/python -m wall3.supervisor --plan experiments.plan_real_20261116
"""
W = "outputs/wall3"
PY = "venv/bin/python"
SUP_NAME = "supervisor_real_20261116"
CELLS = ("none", "A", "B", "AB")


def out_of(ctrl, cell):
    return f"{W}/eval_real_{ctrl}_{cell}_20261116"


def ev(ctrl, cell):
    out = out_of(ctrl, cell)
    return dict(name=f"R_{ctrl}_{cell}", kind="summary", module="wall3.eval_real", out=out,
                cmd=[PY, "-m", "wall3.eval_real", "--out", out, "--condition", f"{ctrl}_real_{cell}",
                     "--chunks", "6", "--parallel", "6", "--env-version", "main_v1_eval"])


REPORT = f"{W}/report_real_20261116.json"
STAGES = [(f"R_{cell}", [ev("act", cell), ev("script", cell)]) for cell in CELLS] + [
    ("R_report", [dict(name="R_report", kind="rc", module="experiments.report_real_20261116", out=REPORT,
                       cmd=[PY, "-m", "experiments.report_real_20261116", "--out", REPORT], log=f"{W}/report_real_20261116.txt")]),
]
