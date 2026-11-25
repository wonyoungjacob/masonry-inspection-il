"""벽 단위 원자료 → LeRobot 데이터셋(v3) 변환 (지시서 단계 5). 벽돌 한 장 = 에피소드 1개.

- 시방 합격한 벽만 넣는다(--all 이면 전부, 스모크 확인용).
- 에피소드 = 벽돌 k의 [시작, 끝) 스텝 (시작 = 철수 자세에서 출발, 끝 = 벽돌이 받침에 정지하고 집게가 철수 자세로 돌아옴).
- 정책 관측: observation.images.{front,top,wrist}(128²), observation.state(관절 6 + 집게 왼쪽 구동 관절 각도),
  observation.environment_state = 목표 정보(도면에서 계산한 그 벽돌의 목표 x·y·z·yaw + 반장 표시 1/0).
- 정책 관측이 아닌 것(info.*, LeRobot은 observation.*·action만 정책 입력으로 씀): 집게 닫힘 각도, 손목 힘·토크, 현재 벽돌 참값 자세,
  벽돌 16장 참값 자세, 대본 단계. 에피소드·벽 메타데이터(벽 번호, 자리 단·순서, 시작값, IoU, 판정)는 meta/wall3_*.json.
- 다 만든 뒤 LeRobot으로 다시 읽어 프레임 수·카메라·목표 정보를 검사하고, 파일 해시를 남긴다.

실행: source venv/bin/activate && python -m wall3.to_lerobot --raw outputs/wall3/<수집> [<수집2> ...] --root data/wall3_pilot_<YYYYMMDD>
      [--max-walls 50] [--all]
      (activate가 torchcodec용 LD_LIBRARY_PATH를 설정한다)
"""
import argparse, hashlib, json, time
from pathlib import Path

import numpy as np

SIM = Path(__file__).resolve().parents[1]
CAMS = ("front", "top", "wrist")
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]
TASK = "Lay the next brick of the wall at its target position."


def features(img=128, lock=False):
    f = {f"observation.images.{c}": {"dtype": "video", "shape": (img, img, 3), "names": ["height", "width", "channels"]} for c in CAMS}
    f["observation.state"] = {"dtype": "float32", "shape": (7,), "names": JOINTS + ["gripper"]}
    f["observation.environment_state"] = {"dtype": "float32", "shape": (5,),
                                          "names": ["target_x", "target_y", "target_z", "target_yaw", "target_is_half"]}
    f["action"] = {"dtype": "float32", "shape": (7,), "names": JOINTS + ["gripper"]}
    f["info.gripper_closure"] = {"dtype": "float32", "shape": (1,), "names": ["closure"]}
    f["info.wrist_ft"] = {"dtype": "float32", "shape": (6,), "names": ["fx", "fy", "fz", "tx", "ty", "tz"]}
    f["info.brick_pose"] = {"dtype": "float32", "shape": (7,), "names": ["x", "y", "z", "qw", "qx", "qy", "qz"]}
    f["info.all_brick_poses"] = {"dtype": "float32", "shape": (112,), "names": [f"b{i}_{c}" for i in range(16) for c in "xyzwijk"]}
    f["info.phase"] = {"dtype": "float32", "shape": (1,), "names": ["phase_id"]}
    if lock:                            # main_v1 grip lock state (not a policy input: 현실성 조건 6)
        f["info.grip_lock"] = {"dtype": "float32", "shape": (4,), "names": ["locked", "lock_brick", "pad_n_left", "pad_n_right"]}
    return f


def target_of(slot):
    return np.array([slot["x"], slot["y"], slot["z"], slot["yaw"], 1.0 if slot["half"] else 0.0], np.float32)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def build(raw_dirs, root: Path, repo_id: str, include_all: bool, max_walls: int = 0, video_mb: int = 20, part=None, exclude=()):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    infos = []
    for rd in raw_dirs:
        for p in rd.glob("raw/wall_*.json"):
            w = json.loads(p.read_text())
            w["_raw_dir"] = rd
            infos.append(w)
    infos.sort(key=lambda w: w["seed"])
    if len({w["seed"] for w in infos}) != len(infos):
        raise SystemExit("같은 시작값이 두 번 들어 있음")
    # clean pass (추가 지시 2): spec pass, finished, no expert overload release (pilot raw has no such key → 0)
    use = [w for w in infos if include_all or (w["eval"]["spec_pass"] and not w.get("fail") and not w.get("overload_releases", 0))]
    missing = set(exclude) - {w["seed"] for w in infos}
    if missing:
        raise SystemExit(f"제외할 시작값이 원자료에 없음: {sorted(missing)}")
    use = [w for w in use if w["seed"] not in set(exclude)]          # walls dropped by a recorded decision (--exclude-seeds)
    if max_walls:
        if len(use) < max_walls:
            raise SystemExit(f"합격 벽 {len(use)}개 < 요청 {max_walls}개")
        use = use[:max_walls]                                    # first passing walls in seed order
    selected = use
    if part is not None:                                         # 나7: contiguous slice i of n (seed order), merged later
        i, n = part
        bounds = np.linspace(0, len(use), n + 1).round().astype(int)
        use = use[bounds[i]:bounds[i + 1]]
    if not use:
        raise SystemExit("변환할 벽이 없음")
    # video_files_size_in_mb: LeRobot appends every episode to one growing mp4 per camera until this size; small files keep
    # the per-episode append cheap (the 48-wall pilot with the default slowed from 17 to 70 s per wall).
    with np.load(use[0]["_raw_dir"] / "raw" / use[0]["raw_file"]) as zf:
        lock = "lock" in zf.files                                # main_v1 raw has the grip lock state, pilot raw does not
    ds = LeRobotDataset.create(repo_id, fps=20, features=features(lock=lock), root=str(root), robot_type="ur10e_robotiq2f140", use_videos=True,
                               video_files_size_in_mb=video_mb)
    ep_meta, wall_meta = [], []
    ep = 0
    t0 = time.time()
    for wi, w in enumerate(use):
        with np.load(w["_raw_dir"] / "raw" / w["raw_file"]) as zf:      # decompress each array once (indexing an NpzFile
            z = {k: zf[k] for k in zf.files}                     # re-reads the whole array every time)
        assert np.array_equal(z["t"], np.arange(len(z["t"]))), "steps must be contiguous"
        assert ("lock" in z) == lock, "raw files mix environments with and without the grip lock"
        for b in w["bricks"]:
            k = b["slot"]
            s = w["design"][k]
            tgt = target_of(s)
            for i in range(b["start"], b["end"]):
                ds.add_frame({"task": TASK,
                              **{f"observation.images.{c}": z[f"img_{c}"][i] for c in CAMS},
                              "observation.state": z["state"][i],
                              "observation.environment_state": tgt,
                              "action": z["action"][i],
                              "info.gripper_closure": np.array([z["closure"][i]], np.float32),
                              "info.wrist_ft": z["ft"][i],
                              "info.brick_pose": z["bricks"][i][k],
                              "info.all_brick_poses": z["bricks"][i].reshape(-1),
                              "info.phase": np.array([z["phase"][i]], np.float32),
                              **({"info.grip_lock": z["lock"][i]} if lock else {})})
            ds.save_episode()
            ep_meta.append({"episode_index": ep, "wall_index": wi, "seed": w["seed"], "slot": k, "course": s["course"],
                            "position": s["index"], "half": s["half"], "start_step": b["start"], "end_step": b["end"],
                            "frames": b["end"] - b["start"], "duration_s": b["duration_s"], "target": tgt.tolist(),
                            "base_dist_m": w["base_dist_m"][k], "iou": w["eval"]["iou"][k], "end_err_mm": b["end_err_mm"],
                            "wall_spec_pass": w["eval"]["spec_pass"]})
            ep += 1
        wall_meta.append({"wall_index": wi, "seed": w["seed"], "spec_pass": w["eval"]["spec_pass"], "eval": w["eval"],
                          "raw_dir": str(w["_raw_dir"]), "raw_file": w["raw_file"], "raw_sha256": w["raw_sha256"], "marks": [[b["slot"], b["start"], b["end"]] for b in w["bricks"]]})
        print(f"wall {w['seed']}: 16 episodes ({ep} total, {time.time() - t0:.0f} s)", flush=True)
    ds.finalize()
    meta = root / "meta"
    (meta / "wall3_episodes.json").write_text(json.dumps(ep_meta, indent=1, default=float))
    (meta / "wall3_walls.json").write_text(json.dumps(wall_meta, indent=1, default=float))
    (meta / "wall3_source.json").write_text(json.dumps({"raw_dirs": [str(r) for r in raw_dirs],
                                                        "collect_configs": [json.loads((r / "collect_config.json").read_text()) for r in raw_dirs],
                                                        "include_all": include_all, "max_walls": max_walls, "video_files_size_in_mb": video_mb,
                                                        "n_walls_raw": len(infos), "n_walls_used": len(use), "part": part,
                                                        "excluded_seeds": sorted(exclude),
                                                        "seeds_used": [w["seed"] for w in use],
                                                        "seeds_not_used": [w["seed"] for w in infos if w not in selected],
                                                        "task": TASK, "features": features(lock=lock)}, indent=1, default=float))
    return ep_meta


def _table(root: Path):
    """All non-video columns of a dataset as numpy arrays, rows in global index order."""
    import pyarrow.parquet as pq
    import pyarrow as pa
    files = sorted((root / "data").rglob("*.parquet"))
    t = pa.concat_tables([pq.read_table(f) for f in files]).combine_chunks()
    order = np.argsort(t.column("index").to_numpy())
    out = {}
    for name in t.column_names:
        col = t.column(name).combine_chunks()
        if pa.types.is_list(col.type) or pa.types.is_fixed_size_list(col.type):
            v = col.flatten().to_numpy(zero_copy_only=False)
            out[name] = v.reshape(len(col), -1)[order]
        else:
            out[name] = col.to_numpy(zero_copy_only=False)[order]
    return out


def content_sha256(root: Path):
    """Hash of every non-video column in global index order (independent of how rows are split into files)."""
    tab = _table(root)
    h = hashlib.sha256()
    for k in sorted(tab):
        a = np.ascontiguousarray(tab[k])
        h.update(k.encode()); h.update(str(a.dtype).encode()); h.update(a.tobytes())
    return h.hexdigest()


def check_raw(root: Path, ep_meta, wall_meta):
    """Every episode's rows equal the raw recording slice (what a one-shot conversion writes), in wall/seed order."""
    tab = _table(root)
    problems = []
    if not np.array_equal(tab["index"], np.arange(len(tab["index"]))):
        problems.append("index not contiguous")
    by_wall = {}
    for e in ep_meta:
        by_wall.setdefault(e["wall_index"], []).append(e)
    pos = 0
    keys = {"action": "action", "observation.state": "state", "info.wrist_ft": "ft", "info.grip_lock": "lock"}
    for w in wall_meta:
        with np.load(Path(w["raw_dir"]) / "raw" / w["raw_file"]) as zf:
            z = {k: zf[k] for k in ("action", "state", "ft", "closure", "phase", "bricks", "lock") if k in zf.files}
        for e in by_wall[w["wall_index"]]:
            a, b = e["start_step"], e["end_step"]
            sl = slice(pos, pos + (b - a))
            if not np.array_equal(tab["episode_index"][sl], np.full(b - a, e["episode_index"])):
                problems.append(f"episode {e['episode_index']} rows")
            for col, rk in keys.items():
                if col in tab and not np.array_equal(tab[col][sl], z[rk][a:b].reshape(b - a, -1)):
                    problems.append(f"episode {e['episode_index']} {col}")
            if not np.array_equal(tab["info.gripper_closure"][sl].ravel(), z["closure"][a:b]):
                problems.append(f"episode {e['episode_index']} closure")
            if not np.array_equal(tab["info.brick_pose"][sl], z["bricks"][a:b, e["slot"]]):
                problems.append(f"episode {e['episode_index']} brick pose")
            if not np.allclose(tab["observation.environment_state"][sl], np.asarray(e["target"], np.float32)[None], atol=0):
                problems.append(f"episode {e['episode_index']} target")
            pos += b - a
    if pos != len(tab["index"]):
        problems.append(f"rows {len(tab['index'])} != episodes {pos}")
    return {"rows": int(len(tab["index"])), "episodes_checked": len(ep_meta), "walls_checked": len(wall_meta), "problems": problems[:50],
            "ok": not problems}


def merge(part_roots, root: Path, repo_id: str, video_mb: int = 20):
    """나7: aggregate part datasets (in the given order) into one and rebuild the wall3 meta with global indices."""
    from lerobot.datasets.aggregate import aggregate_datasets
    part_roots = [Path(r) for r in part_roots]
    aggregate_datasets([f"local/{r.name}" for r in part_roots], repo_id, roots=part_roots, aggr_root=root,
                       video_files_size_in_mb=video_mb)
    ep_meta, wall_meta, src = [], [], []
    for r in part_roots:
        e = json.loads((r / "meta" / "wall3_episodes.json").read_text())
        w = json.loads((r / "meta" / "wall3_walls.json").read_text())
        de, dw = len(ep_meta), len(wall_meta)
        ep_meta += [dict(x, episode_index=x["episode_index"] + de, wall_index=x["wall_index"] + dw) for x in e]
        wall_meta += [dict(x, wall_index=x["wall_index"] + dw) for x in w]
        src.append(json.loads((r / "meta" / "wall3_source.json").read_text()))
    seeds = [w["seed"] for w in wall_meta]
    if seeds != sorted(seeds) or len(set(seeds)) != len(seeds):
        raise SystemExit("parts are not in seed order")
    meta = root / "meta"
    (meta / "wall3_episodes.json").write_text(json.dumps(ep_meta, indent=1, default=float))
    (meta / "wall3_walls.json").write_text(json.dumps(wall_meta, indent=1, default=float))
    (meta / "wall3_source.json").write_text(json.dumps({"merged_from": [str(r) for r in part_roots], "parts": src, "seeds_used": seeds,
                                                        "n_walls_used": len(seeds), "video_files_size_in_mb": video_mb,
                                                        "task": TASK, "features": src[0]["features"]}, indent=1, default=float))
    return ep_meta, wall_meta


def verify(root: Path, repo_id: str, ep_meta):
    """Read back with LeRobot: episode count, frames per episode, cameras, target info per episode."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(repo_id, root=str(root), video_backend="pyav")
    rep = {"num_episodes": ds.num_episodes, "num_frames": ds.num_frames, "expected_episodes": len(ep_meta),
           "expected_frames": int(sum(e["frames"] for e in ep_meta)), "camera_keys": list(ds.meta.camera_keys), "problems": []}
    if ds.num_episodes != len(ep_meta):
        rep["problems"].append("episode count")
    if ds.num_frames != rep["expected_frames"]:
        rep["problems"].append("frame count")
    if sorted(ds.meta.camera_keys) != sorted(f"observation.images.{c}" for c in CAMS):
        rep["problems"].append("cameras")
    eps = ds.meta.episodes
    for e in ep_meta:
        i = e["episode_index"]
        a, b = int(eps["dataset_from_index"][i]), int(eps["dataset_to_index"][i])
        if b - a != e["frames"]:
            rep["problems"].append(f"episode {i} length {b - a} != {e['frames']}")
        for j in (a, (a + b) // 2, b - 1):
            item = ds[j]
            if int(item["episode_index"]) != i:
                rep["problems"].append(f"frame {j} episode index")
            if not np.allclose(item["observation.environment_state"].numpy(), e["target"], atol=1e-6):
                rep["problems"].append(f"episode {i} frame {j} target")
            for c in CAMS:
                if tuple(item[f"observation.images.{c}"].shape) != (3, 128, 128):
                    rep["problems"].append(f"episode {i} frame {j} {c} shape {tuple(item[f'observation.images.{c}'].shape)}")
    rep["checked_frames"] = 3 * len(ep_meta)
    rep["ok"] = not rep["problems"]
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", nargs="+", default=None, help="collect.py output folder(s)")
    ap.add_argument("--root", required=True, help="new dataset folder (sim/data/wall3_pilot_<YYYYMMDD>)")
    ap.add_argument("--repo-id", default="")
    ap.add_argument("--all", action="store_true", help="include walls that failed the spec (smoke check only)")
    ap.add_argument("--max-walls", type=int, default=0, help="use the first N passing walls in seed order (0 = all)")
    ap.add_argument("--video-mb", type=int, default=20, help="LeRobot video_files_size_in_mb")
    ap.add_argument("--part", type=int, nargs=2, default=None, metavar=("I", "N"), help="convert only slice I of N (seed order)")
    ap.add_argument("--merge", nargs="+", default=None, help="part dataset folders to merge into --root (in seed order)")
    ap.add_argument("--exclude-seeds", type=int, nargs="*", default=[], help="walls to leave out (record the reason in the report)")
    a = ap.parse_args()
    root = Path(a.root)
    if root.exists():
        raise SystemExit(f"{root} 이미 있음: 덮어쓰지 않음")
    repo_id = a.repo_id or f"local/{root.name}"
    t0 = time.time()
    if a.merge:
        ep_meta, wall_meta = merge(a.merge, root, repo_id, a.video_mb)
    else:
        ep_meta = build([Path(r) for r in a.raw or []], root, repo_id, a.all, a.max_walls, a.video_mb, tuple(a.part) if a.part else None,
                        tuple(a.exclude_seeds))
        wall_meta = json.loads((root / "meta" / "wall3_walls.json").read_text())
    t_build = time.time() - t0
    rep = verify(root, repo_id, ep_meta)
    rep["raw_check"] = check_raw(root, ep_meta, wall_meta)
    rep["content_sha256"] = content_sha256(root)
    rep["ok"] = rep["ok"] and rep["raw_check"]["ok"]
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.name != "wall3_hashes.json")
    hashes = {str(p.relative_to(root)): sha256(p) for p in files}
    total = hashlib.sha256("".join(f"{k}:{v}\n" for k, v in hashes.items()).encode()).hexdigest()
    size = sum(p.stat().st_size for p in files)
    (root / "meta" / "wall3_hashes.json").write_text(json.dumps({"dataset_sha256": total, "bytes": size, "files": hashes}, indent=1))
    rep.update(dataset_sha256=total, bytes=size, build_s=t_build)
    (root / "meta" / "wall3_verify.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps({k: v for k, v in rep.items() if k != "problems"}, indent=1, default=float), "\nproblems:", rep["problems"][:10])
    if not rep["ok"]:
        raise SystemExit("verification failed")


if __name__ == "__main__":
    main()
