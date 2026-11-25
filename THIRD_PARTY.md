# 외부 자산과 라이선스

이 저장소는 아래 자산을 **포함하지 않고**, `wall3/fetch_assets.sh`가 커밋을 고정해 내려받는다. 파일은 고치지 않고 그대로 쓴다.

| 자산 | 출처 | 고정 커밋 | 라이선스 |
|---|---|---|---|
| UR10e MJCF + 메시 | [google-deepmind/mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie) `universal_robots_ur10e` | `62ab00f2` | 폴더 안 `LICENSE` 참고 |
| Robotiq 2F-140 집게 | [ARISE-Initiative/robosuite](https://github.com/ARISE-Initiative/robosuite) `robotiq_gripper_140.xml` + 메시 | `5ce6643f` | robosuite `LICENSE` (MIT) |

학습·평가에 [LeRobot](https://github.com/huggingface/lerobot)의 ACT · Diffusion Policy 구현을 쓴다. 시뮬레이터는 [MuJoCo](https://github.com/google-deepmind/mujoco)다.
