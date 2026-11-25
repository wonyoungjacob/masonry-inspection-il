"""Hugging Face 업로드 (사용자 요청 2026-10-19): 본 데이터셋과 본 학습 체크포인트를 사용자 계정에 비공개로 올린다.

1) 데이터셋: 합치기 로그에 "done:"이 있고(영상 전수 검사까지 통과) 최종 검사 JSON이 ok일 때만 올린다. 올린 뒤 LeRobot이 허브에서
   읽을 수 있게 info.json의 codebase_version 태그를 단다. 로컬 데이터셋 폴더는 바꾸지 않는다(카드는 허브에만 올림).
2) 모델: 학습마다 상태 파일의 result가 "done"이 되면 checkpoints/last/pretrained_model을 올린다. 실패·멈춤이면 올리지 않는다.

실행: venv/bin/python -m wall3.hf_upload --dataset data/wall3_main_20261019 --dataset-repo <id> --merge-log <log>
        --train outputs/wall3/main_train_<날짜> --model act_s1000=<id> --model diffusion_s1000=<id> [--public]
"""
import argparse, json, time
from pathlib import Path

POLL = 60


def say(*a):
    print(time.strftime("[%m-%d %H:%M:%S]"), *a, flush=True)


def dataset_card(root: Path, repo: str):
    v = json.loads((root / "meta" / "wall3_verify.json").read_text())
    src = json.loads((root / "meta" / "wall3_source.json").read_text())
    info = json.loads((root / "meta" / "info.json").read_text())
    seeds = src["seeds_used"]
    return f"""---
tags: [LeRobot, robotics, ur10e, masonry, mujoco]
task_categories: [robotics]
---
# {repo.split('/')[1]}

Simulated (MuJoCo) UR10e + Robotiq 2F-140 laying a 3-course brick wall (16 bricks, running bond, mortar beds).
Scripted inverse-kinematics expert demonstrations, LeRobot format (codebase {info.get('codebase_version')}).

| | |
|---|---|
| Walls / episodes / frames | {len(seeds):,} / {v['num_episodes']:,} / {v['num_frames']:,} (one episode = one brick) |
| FPS | {info.get('fps')} |
| Cameras (policy input, 128x128) | front, top, wrist |
| observation.state | 6 joint angles + gripper |
| observation.environment_state | target of this brick from the drawing: x, y, z, yaw, is_half |
| action | 6 joint targets + gripper command |
| info.* (not policy inputs) | gripper closure, wrist force/torque, true brick pose(s), expert phase, grip-lock state |
| Seeds | {seeds[0]} ~ {seeds[-1]}, only clean walls (spec pass, no grip-lock overload release); excluded by decision: {', '.join(map(str, sorted({s for p in src.get('parts', [src]) for s in p.get('excluded_seeds', [])})))} |
| Content hash | `{v.get('content_sha256')}` |

Every video file was decoded end to end and its frame count checked against the episode lengths before upload.
"""


def model_card(train: Path, name: str, repo: str, dataset_repo: str):
    ck = train / name / "checkpoints" / "last" / "pretrained_model"
    cfg = json.loads((ck / "config.json").read_text())
    tc = json.loads((ck / "train_config.json").read_text())
    return f"""---
tags: [LeRobot, robotics, ur10e, masonry, {cfg.get('type')}]
datasets: [{dataset_repo}]
---
# {repo.split('/')[1]}

LeRobot 0.6.1 `{cfg.get('type')}` policy (default settings), trained on [{dataset_repo}](https://huggingface.co/datasets/{dataset_repo}):
UR10e + Robotiq 2F-140 brick laying in MuJoCo, one episode per brick.

| | |
|---|---|
| Steps / batch | {tc.get('steps'):,} / {tc.get('batch_size')} (last checkpoint) |
| Seed | {tc.get('seed')} |
| Inputs | {', '.join(cfg.get('input_features', {}).keys())} |
| Action chunk | {', '.join(f'{k} {cfg[k]}' for k in ('chunk_size', 'n_action_steps', 'horizon', 'n_obs_steps') if cfg.get(k) is not None)} |
"""


def verify_remote(api, repo: str, repo_type: str, local: Path, skip=(".cache",)):
    """Every local file is on the Hub with the same size (README.md is the card, written on the Hub only)."""
    info = (api.dataset_info if repo_type == "dataset" else api.model_info)(repo, files_metadata=True)
    remote = {s.rfilename: s.size for s in info.siblings}
    loc = {str(f.relative_to(local)): f.stat().st_size for f in local.rglob("*") if f.is_file() and not f.relative_to(local).parts[0] in skip}
    missing = sorted(f for f in loc if f not in remote)
    differ = sorted(f for f in loc if f in remote and remote[f] != loc[f])
    res = {"repo": repo, "type": repo_type, "private": info.private, "local_files": len(loc), "remote_files": len(remote),
           "missing": missing[:20], "size_differs": differ[:20], "card": "README.md" in remote,
           "bytes": sum(loc.values()), "ok": not missing and not differ and "README.md" in remote}
    say("verify", repo, "OK" if res["ok"] else "PROBLEM", {k: res[k] for k in ("local_files", "remote_files", "card", "private")},
        res["missing"][:3], res["size_differs"][:3])
    return res


def upload_dataset(api, root: Path, repo: str, private: bool):
    info = json.loads((root / "meta" / "info.json").read_text())
    api.create_repo(repo, repo_type="dataset", private=private, exist_ok=True)
    say(f"dataset upload start: {root} -> {repo}")
    api.upload_large_folder(repo_id=repo, folder_path=str(root), repo_type="dataset")
    api.upload_file(path_or_fileobj=dataset_card(root, repo).encode(), path_in_repo="README.md", repo_id=repo, repo_type="dataset")
    ver = info.get("codebase_version")
    if ver:
        try:
            api.create_tag(repo, tag=ver, repo_type="dataset")
        except Exception as e:                                          # tag already there on a re-run
            say("tag:", repr(e)[:200])
    say(f"dataset upload done: https://huggingface.co/datasets/{repo}")


def upload_model(api, train: Path, name: str, repo: str, dataset_repo: str, private: bool):
    ck = train / name / "checkpoints" / "last" / "pretrained_model"
    api.create_repo(repo, repo_type="model", private=private, exist_ok=True)
    say(f"model upload start: {ck} -> {repo}")
    api.upload_folder(repo_id=repo, folder_path=str(ck), repo_type="model")
    api.upload_file(path_or_fileobj=model_card(train, name, repo, dataset_repo).encode(), path_in_repo="README.md", repo_id=repo, repo_type="model")
    say(f"model upload done: https://huggingface.co/{repo}")
    return verify_remote(api, repo, "model", ck)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--dataset-repo", required=True)
    ap.add_argument("--merge-log", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--model", action="append", default=[], help="run_name=repo_id")
    ap.add_argument("--public", action="store_true")
    ap.add_argument("--skip-dataset", action="store_true", help="dataset already uploaded: only wait for the models")
    ap.add_argument("--record", default=str(Path(__file__).resolve().parents[2] / "01_사기설" / "근거_우분투_wall3_본학습" / "HF업로드.json"),
                    help="upload + verification results (committed with the report)")
    a = ap.parse_args()
    from huggingface_hub import HfApi
    api = HfApi()
    say("logged in as", api.whoami()["name"])
    private = not a.public
    root, log, train = Path(a.dataset), Path(a.merge_log), Path(a.train)
    # 1) dataset
    while not a.skip_dataset:
        txt = log.read_text() if log.exists() else ""
        if any(l.startswith("done:") for l in txt.splitlines()):
            break
        if "깨짐" in txt or "실패" in txt:
            say("merge failed: dataset not uploaded"); return
        time.sleep(POLL)
    if not a.skip_dataset:
        chk = json.loads((root / "meta" / "wall3_check_videos.json").read_text())
        ver = json.loads((root / "meta" / "wall3_verify.json").read_text())
        if not (chk.get("ok") and ver.get("ok")):
            say("dataset checks not ok: not uploaded"); return
        upload_dataset(api, root, a.dataset_repo, private)
    # 2) models, as each training finishes
    rec = {"dataset": verify_remote(api, a.dataset_repo, "dataset", root), "models": {}}
    todo = dict(m.split("=", 1) for m in a.model)
    while todo:
        for name in list(todo):
            st = train / f"{name}.status.json"
            res = json.loads(st.read_text()).get("result") if st.exists() else None
            if res == "done":
                rec["models"][name] = upload_model(api, train, name, todo.pop(name), a.dataset_repo, private)
                Path(a.record).write_text(json.dumps(rec, indent=1, ensure_ascii=False))
            elif res in ("failed", "stopped", "oom"):
                say(f"{name} {res}: not uploaded"); todo.pop(name)
                rec["models"][name] = {"uploaded": False, "result": res}
        if todo:
            time.sleep(POLL)
    rec["time"] = time.strftime("%Y-%m-%d %H:%M")
    Path(a.record).write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    say("all uploads finished")


if __name__ == "__main__":
    main()
