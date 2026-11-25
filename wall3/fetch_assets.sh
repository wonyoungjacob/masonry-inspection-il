#!/usr/bin/env bash
# 3층 벽체 환경 자산 받기 (pip 설치 없음). 커밋을 고정해 같은 파일을 받는다.
#   UR10e  : MuJoCo Menagerie universal_robots_ur10e  -> sim/menagerie/universal_robots_ur10e/
#   2F-140 : robosuite robotiq_gripper_140.xml + 메시 -> sim/third_party/robosuite_gripper140/
# 기존 sim/menagerie/trs_so_arm100 은 건드리지 않는다. 대상 폴더가 이미 있으면 멈춘다.
set -euo pipefail
MENAGERIE_COMMIT=62ab00f2bc44f6f20c1c9b7a13f3fb1dd41a4e35
ROBOSUITE_COMMIT=5ce6643f3092639d08f7b0f90ed1c6a84f50552c
SIM="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT

for d in "$SIM/menagerie/universal_robots_ur10e" "$SIM/third_party/robosuite_gripper140"; do
  [ -e "$d" ] && { echo "이미 있음: $d (덮어쓰지 않음)"; exit 1; }
done

git clone -q --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git "$TMP/menagerie"
git -C "$TMP/menagerie" checkout -q "$MENAGERIE_COMMIT"
git -C "$TMP/menagerie" sparse-checkout set universal_robots_ur10e
mkdir -p "$SIM/menagerie"
cp -r "$TMP/menagerie/universal_robots_ur10e" "$SIM/menagerie/"
printf 'repo:   https://github.com/google-deepmind/mujoco_menagerie\ncommit: %s\nfolder: universal_robots_ur10e (copied unchanged)\n' \
  "$MENAGERIE_COMMIT" > "$SIM/menagerie/universal_robots_ur10e/SOURCE.txt"

git clone -q --filter=blob:none --sparse https://github.com/ARISE-Initiative/robosuite.git "$TMP/robosuite"
git -C "$TMP/robosuite" checkout -q "$ROBOSUITE_COMMIT"
git -C "$TMP/robosuite" sparse-checkout set --no-cone /LICENSE /robosuite/models/assets/grippers/robotiq_gripper_140.xml \
  '/robosuite/models/assets/grippers/meshes/robotiq_140_gripper/*'
G="$SIM/third_party/robosuite_gripper140"; mkdir -p "$G/meshes"
cp "$TMP/robosuite/robosuite/models/assets/grippers/robotiq_gripper_140.xml" "$G/"
cp -r "$TMP/robosuite/robosuite/models/assets/grippers/meshes/robotiq_140_gripper" "$G/meshes/"
cp "$TMP/robosuite/LICENSE" "$G/LICENSE"
printf 'Robotiq 2F-140 gripper MuJoCo model, copied unchanged from robosuite\nrepo:   https://github.com/ARISE-Initiative/robosuite\ncommit: %s\nlicense: MIT (LICENSE)\n' \
  "$ROBOSUITE_COMMIT" > "$G/SOURCE.txt"

sha256sum "$G/robotiq_gripper_140.xml" "$SIM/menagerie/universal_robots_ur10e/ur10e.xml"
