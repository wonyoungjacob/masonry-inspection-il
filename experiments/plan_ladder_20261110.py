"""실패 원인 제거 사다리 1단계 계획 (2026-11-10, 01_사기설/실험파이프라인_실패원인제거_20261110.md 5절). wall3.supervisor가 돌린다.

L1_eval   : 같은 100벽(700300 ~ 700399)에서 E3(all_held) 다시 · +K1 · +K2 · +K6 (각 제어 하나씩, ACT)
L1_trace  : +K2 조건을 추적 기록과 함께 다시 돌림 (평가와 같은지 비교) → L1_causes 원인 분류 → L1_rank 다음 순위
L1_compare: 기준(같은 묶음의 E3 다시 돌림)과 짝 비교, 11-04 E3와도 비교 (평가 바로 뒤)
계산만 한다. 판정 · 보고 · 커밋은 사람이(또는 Claude가) 결과를 읽고 한다.

실행: venv/bin/python -m wall3.supervisor --plan experiments.plan_ladder_20261110
"""
W = "outputs/wall3"
PY = "venv/bin/python"
SUP_NAME = "supervisor_ladder_20261110"
CKPT = f"{W}/main_train_v2_20261022/act_s1000/checkpoints/last/pretrained_model"
L1 = f"{W}/eval_ladder1_20261110"
T1 = f"{W}/diag_trace_ladder1_k2_20261110"
CONDS = ["all_held", "all_held+k1", "all_held+k2", "all_held+k6"]


def seeds(i, n=100, chunks=5, s0=700300):
    k = n // chunks
    return list(range(s0 + i * k, s0 + (i + 1) * k))


STAGES = [
    ("L1_eval", [dict(name="L1_eval", kind="summary", module="wall3.eval_gates", out=L1,
                      cmd=[PY, "-m", "wall3.eval_gates", "--out", L1, "--train-dir", f"{W}/main_train_v2_20261022",
                           "--policies", "act", "--conditions", *CONDS, "--chunks", "5", "--parallel", "8",
                           "--env-version", "main_v1_eval"])]),
    ("L1_compare", [dict(name="L1_compare", kind="rc", module="wall3.compare_ladder", out=f"{L1}_compare.json",
                         cmd=[PY, "-m", "wall3.compare_ladder", "--evals", L1, f"{W}/eval_gates_info_20261104",
                              "--base", "eval_ladder1_20261110/act_all_held", "--out", f"{L1}_compare.json"],
                         log=f"{L1}_compare.txt")]),
    ("L1_trace", [dict(name=f"L1_trace_part{i}", kind="rc", module="experiments.diag_trace", out=f"{T1}/part{i}",
                       cmd=[PY, "-m", "experiments.diag_trace", "--ckpt", CKPT, "--env-version", "main_v1_eval",
                            "--seeds", *map(str, seeds(i)), "--out", f"{T1}/part{i}",
                            "--gate-pose-source", "camera", "--gate-target", "held_grasp", "--gate-fix", "k2",
                            "--compare", f"{L1}/act_all_held+k2_part{i}/rollout.json"]) for i in range(5)]),
    ("L1_causes", [dict(name="L1_causes", kind="rc", module="experiments.diag_trace_causes", out=f"{T1}/causes.json",
                        cmd=[PY, "-m", "experiments.diag_trace_causes", T1, "--aim-max-mm", "50", "--perch-mm", "20",
                             "--out", f"{T1}/causes.json"], log=f"{T1}/causes.txt")]),
    ("L1_rank", [dict(name="L1_rank", kind="rc", module="experiments.rank_fail_causes", out=f"{T1}/rank.json",
                      cmd=[PY, "-m", "experiments.rank_fail_causes", "--trace", T1, "--out", f"{T1}/rank.json"],
                      log=f"{T1}/rank.txt")]),
]
