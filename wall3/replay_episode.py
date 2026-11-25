"""데이터셋 에피소드 재생 (사용자 요청 2026-10-19): LeRobot 데이터셋에 기록된 행동을 MuJoCo 장면에 그대로 다시 넣어(행동 재생)
그 에피소드를 3D로 보여 주고, 화면 오른쪽 위에 데이터셋에 저장된 카메라 3대 영상(정책이 보는 128² 영상)을 겹쳐 띄운다.

에피소드 = 벽돌 한 장. 앞 벽돌들이 놓인 상태가 필요하므로 같은 벽의 앞 에피소드 행동부터 빠르게 다시 넣은 뒤, 고른 에피소드를 실시간으로
보여 준다. 재생이 데이터셋과 같은지 매 스텝 관절 상태를 데이터셋의 observation.state와 비교해 최대 차이를 출력한다.

보기:  DISPLAY=:1 venv/bin/python -m wall3.replay_episode --root data/wall3_main_20261018 --episode 12
점검:  MUJOCO_GL=egl venv/bin/python -m wall3.replay_episode --root ... --episode 12 --check   (창 없이 비교만)
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np

SIM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
CAMS = ("front", "top", "wrist")


def episode_rows(root: Path, episodes):
    """action, observation.state of the given episodes (in episode order)."""
    import pyarrow.dataset as pds
    d = pds.dataset(str(root / "data"), format="parquet")
    t = d.to_table(columns=["index", "episode_index", "action", "observation.state"],
                   filter=pds.field("episode_index").isin(list(episodes))).to_pandas().sort_values("index")
    return np.stack(t["action"].to_numpy()).astype(np.float64), np.stack(t["observation.state"].to_numpy()).astype(np.float64), t["episode_index"].to_numpy()


def episode_images(root: Path, ep):
    """Decoded camera frames of one episode from the dataset videos: dict cam -> (T, 128, 128, 3) uint8."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(f"local/{root.name}", root=str(root), episodes=[ep], video_backend="pyav")
    out = {c: [] for c in CAMS}
    for i in range(len(ds)):
        item = ds[i]
        for c in CAMS:
            out[c].append((item[f"observation.images.{c}"].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))
    return {c: np.stack(v) for c, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--check", action="store_true", help="no window: replay and compare with the dataset only")
    a = ap.parse_args()
    from wall3.config import scene_config, RETRACT_Q
    from wall3.scene import WallEnv
    root = Path(a.root)
    eps = json.loads((root / "meta" / "wall3_episodes.json").read_text())
    e = eps[a.episode]
    wall = [x for x in eps if x["wall_index"] == e["wall_index"] and x["episode_index"] <= a.episode]
    act, state, epi = episode_rows(root, [x["episode_index"] for x in wall])
    print(f"episode {a.episode}: wall seed {e['seed']}, slot {e['slot']} (course {e['course']}, {'half' if e['half'] else 'full'}), "
          f"{e['frames']} frames; replaying {len(act)} steps from the wall start", flush=True)
    env = WallEnv(scene_config("main_v1"))
    env.reset(e["seed"], arm_q=np.array(RETRACT_Q))
    viewer = imgs = None
    if not a.check:
        import mujoco, mujoco.viewer
        imgs = episode_images(root, a.episode)
        viewer = mujoco.viewer.launch_passive(env.model, env.data, show_left_ui=False, show_right_ui=False)
        viewer.cam.lookat[:] = [0.0, 0.2, 0.3]; viewer.cam.distance = 2.6; viewer.cam.azimuth = 120; viewer.cam.elevation = -25
    dev, k0 = 0.0, None
    dt = env.cfg.ctrl_dt / a.speed
    for i in range(len(act)):
        if viewer is not None and not viewer.is_running():
            return
        obs = np.r_[env.arm_q(), float(env.data.joint("g_finger_joint").qpos[0])]
        dev = max(dev, float(np.abs(obs - state[i]).max()))
        shown = epi[i] == a.episode
        t0 = time.time()
        env.step(act[i, :6], float(act[i, 6]))
        if viewer is not None and (shown or i % 10 == 0):
            if shown:
                k0 = i if k0 is None else k0
                f = i - k0
                import mujoco
                vp = viewer.viewport
                s = max(128, min(256, vp.height // 3)) if vp is not None else 256
                x = (vp.width - s) if vp is not None else 0
                top = vp.height if vp is not None else 3 * s
                viewer.set_images([(mujoco.MjrRect(x, top - (j + 1) * s, s, s),
                                    np.ascontiguousarray(np.kron(imgs[c][min(f, len(imgs[c]) - 1)], np.ones((s // 128, s // 128, 1), np.uint8))))
                                   for j, c in enumerate(CAMS)])
                viewer.set_texts((None, None, f"episode {a.episode} (seed {e['seed']} slot {e['slot']})  frame {f + 1}/{e['frames']}",
                                  "right: dataset cameras front / top / wrist"))
            else:
                viewer.set_texts((None, None, f"fast-forward: bricks before slot {e['slot']}  step {i}/{len(act)}", ""))
            viewer.sync()
            if shown:
                time.sleep(max(0.0, dt - (time.time() - t0)))
    print(f"replay matches the dataset: max |joint/gripper state - observation.state| = {dev:.2e} rad", flush=True)
    if viewer is not None:
        viewer.set_texts((None, None, f"episode {a.episode} done - close the window (Esc) to exit", ""))
        while viewer.is_running():
            viewer.sync(); time.sleep(0.05)


if __name__ == "__main__":
    main()
