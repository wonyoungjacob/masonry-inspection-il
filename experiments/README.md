# experiments/ — 실험 계획 · 진단 스크립트

`wall3/`가 환경 · 시연기 · 검측 · 채점 같은 **재사용하는 코드**라면, 이 폴더는 연구 중에 **한 번씩 돌린 실험의 정의와 실패 분석 스크립트**다.
날짜가 붙은 파일은 그날의 실험 조건을 결과를 보기 전에 고정해 둔 기록이라 이름과 내용을 바꾸지 않았다.

| 접두어 | 내용 | 예 |
|---|---|---|
| `plan_*` | `wall3.supervisor`가 차례로 돌리는 실험 계획(조건 목록 · 출력 폴더 · 판정 순서) | `plan_lf_20261110.py` = ⑤ 구성의 역방향 제거(leave-one-out) 실험 |
| `diag_*` | 실패 원인 진단. 정책이 어디서, 왜 실패했는지 추적 · 영상 · 그림으로 확인 | `diag_trace.py`, `diag_fail_mechanism.py`, `diag_slip.py` |
| `rank_*`, `est_*` | 진단 결과를 원인별로 세고, 원인을 없앴을 때의 상한을 추정 | `rank_fail_causes.py`, `est_fix_ceiling.py` |
| `sweep_*`, `pilot_*` | 집게 파라미터 탐색, 파일럿 확인 | `sweep_gripper140_a.py` |
| `report_*`, `*_train_report.py`, `summarize_*` | 결과 JSON을 표 · 보고서로 요약 | `report_lf_20261111.py` |

진단 스크립트는 시뮬레이터 참값(정답 좌표)을 **기록과 분석에만** 쓴다. 검측 판단에 참값을 쓴 조건은 결과에서 항상 "참값 (진단)"으로 표시했다.

실행은 저장소 루트에서 `python -m experiments.<이름>` 또는 `python -m wall3.supervisor --plan experiments.<plan 이름>`.
대부분 학습된 체크포인트와 수집 데이터(수백 GB, 저장소에 없음)가 필요하다.
