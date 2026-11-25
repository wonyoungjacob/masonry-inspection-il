"""예비 확인(결과 아님, 2026-11-17): 공급 팔레트를 도면에서 멀리 옮기면 지금 ACT가 벽돌을 따라가는가.

사용자 질문('블록이 아예 다른 곳에서 공급되면 당연히 하이브리드가 낫겠지?')에 답하려는 2벽짜리 확인이다. 보고 수치로 쓰지 않는다.
현실 제약 롤아웃(wall3/rollout_real.py)을 그대로 쓰고, B의 팔레트 이동만 (0, -shift) m로 바꾼다(방향 고정: -y는
작업대 안쪽이라 벽돌이 작업대 밖으로 나가지 않는다). 검측 · 정책 · 대본 코드는 바꾸지 않는다.

읽는 법 (기록의 g0 사건)
  - 정책이 도면 공급칸 중심 60 mm 안으로 내려오면 G0v가 넘겨받는다. tcp_off_mm = 그 순간 손과 도면 중심의 거리.
  - G0v는 손에서 80 mm 안의 벽돌만 집는다(G0V_R). 벽돌이 그보다 멀면 result = not_seen.
  - ACT가 벽돌을 영상으로 따라간다면 도면 중심 60 mm 밖(벽돌 쪽)으로 내려가므로 g0 사건이 거의 없어야 한다.

실행: MUJOCO_GL=egl PYTHONPATH=. venv/bin/python experiments/pilot_supplyfar_20261117.py <출력 폴더(없어야 함)> act|script <shift_mm> 700300 700301
"""
import sys

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))
import wall3.realism as RZ

out, controller, shift_mm, seeds = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4:]
orig = RZ.make_perturb


def make_perturb(seed, A, B):
    p = orig(seed, A, B)
    if B:
        p.pallet = (0.0, -shift_mm / 1000)
    return p


RZ.make_perturb = make_perturb
import wall3.rollout_real as RR

sys.argv = ["x", "--controller", controller, "--env-version", "main_v1_eval", "--seeds", *seeds, "--out", out, "--B"]
RR.main()
