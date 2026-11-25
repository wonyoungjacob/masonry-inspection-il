#!/usr/bin/env bash
# 3층 벽체 수집 화면 보기 (MuJoCo 뷰어 창, 실시간). 데스크톱 화면에 연결된 터미널에서 실행한다.
#   bash wall3/view_collect.sh --seed 700000 --walls 3 [--speed 1.0]
# 녹화(화면 없이 mp4)는 view_collect.py 머리말 참고.
set -e
cd "$(dirname "$0")/.."
export MUJOCO_GL=glfw
exec venv/bin/python -m wall3.view_collect "$@"
