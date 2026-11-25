"""역방향 제거 2차: SDCG(G2c)를 넣은 Full에서 하나씩 끈다 (2026-11-11, 사용자 지시).

사용자 지시: "sdcg와 vpra도 구현해서 full을 만들고 거기서 하나씩 빼는 실험. 그 2개가 영향력이 적은 거 같으면 무시하고 스토리라인에만".
- VPRA: 구현하지 않음. Full 완성 60벽 모두 시방 합격, 최대 오차가 허용 6 mm 안(스토리라인 'SDCG · VPRA 검토').
- SDCG: G2c(k8, gates.py)로 구현. 근거: Full 추적에서 자리 6 처음 실패 9벽 중 8벽이 손 안 측정 실패 뒤 놓기.
Full2 = G0 + G1 + G2 + G2w + G2c + G3 + H (`all_held+k2+k7+k8`). 조건은 결과를 보기 전에 정했다.
순서: Full2 → Full2 − G2c(= 11-10 Full 설정) 먼저(SDCG 몫), 그다음 나머지. 같은 100벽(700300 ~ 700399), 조건마다 단계 하나 · 프로세스 8개.
G1 꺼짐 + G0의 정책 리셋 누락은 e2050e9에서 고쳤다(Full2 − G1 줄은 고친 코드).

실행: venv/bin/python -m wall3.supervisor --plan experiments.plan_lf2_sdcg_20261111
"""
W = "outputs/wall3"
PY = "venv/bin/python"
SUP_NAME = "supervisor_lf2_sdcg_20261111"
TRAIN = f"{W}/main_train_v2_20261022"

# (단계 이름, 출력 폴더 꼬리, 조건 이름, 줄 이름)
ROWS = [
    ("Lg01_full2", "full2", "all_held+k2+k7+k8", "Full2"),
    ("Lg02_noG2c", "noG2c", "all_held+k2+k7", "Full2 − G2c"),
    ("Lg03_noG0", "noG0", "all_held+k7+k8", "Full2 − G0"),
    ("Lg04_noH", "noH", "all+k2+k7+k8", "Full2 − H"),
    ("Lg05_noG2", "noG2", "g1g3_held+k2", "Full2 − G2 (G2w · G2c도 꺼짐)"),
    ("Lg06_noG3", "noG3", "g1g2_held+k2+k7+k8", "Full2 − G3"),
    ("Lg07_noG1", "noG1", "g2g3_held+k2+k7+k8", "Full2 − G1"),
    ("Lg08_noG2w", "noG2w", "all_held+k2+k8", "Full2 − G2w"),
    ("Lg09_none", "none", "none", "검측 없음"),
    ("Lg10_truth", "truth", "all_truth_held+k2+k7+k8", "(진단) Full2, 위치 = 참값 (진단)"),
]


def out_of(tail):
    return f"{W}/eval_lg_{tail}_20261111"


def ev(name, out, cond):
    return (name, [dict(name=name, kind="summary", module="wall3.eval_gates", out=out,
                        cmd=[PY, "-m", "wall3.eval_gates", "--out", out, "--train-dir", TRAIN, "--policies", "act",
                             "--conditions", cond, "--chunks", "8", "--parallel", "8", "--env-version", "main_v1_eval"])])


STAGES = [ev(name, out_of(tail), cond) for name, tail, cond, _ in ROWS] + [
    ("Lg_compare", [dict(name="Lg_compare", kind="rc", module="wall3.compare_ladder", out=f"{W}/eval_lg_20261111_compare.json",
                         cmd=[PY, "-m", "wall3.compare_ladder", "--evals", *[out_of(t) for _, t, _, _ in ROWS],
                              f"{W}/eval_lf_full_20261110",
                              "--base", f"eval_lg_full2_20261111/act_{ROWS[0][2]}", "--out", f"{W}/eval_lg_20261111_compare.json"],
                         log=f"{W}/eval_lg_20261111_compare.txt")]),
]
