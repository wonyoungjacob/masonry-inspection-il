"""본 학습 속도 설정 고르기 (본학습 지시 4절: 데이터 로딩 작업자 수·영상 디코더는 처음 10분 안에 정해도 됨).

lerobot-train과 같은 방식으로 데이터 로더를 만들어(정책 기본 설정의 delta_timestamps, 배치 32, spawn, 섞기) 디코더 × 작업자 수마다
초당 샘플 수를 잰다. 학습 설정(정책·스텝·배치)은 바꾸지 않는다.

실행: source venv/bin/activate && python -m experiments.bench_loader --root data/wall3_main_<날짜> --out <json>
"""
import argparse, json, time
from pathlib import Path


def bench(root, policy, backend, workers, batches, warmup, max_s=50.0):
    import torch
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
    import dataclasses
    tdef = {f.name: f.default for f in dataclasses.fields(TrainPipelineConfig)}
    repo = f"local/{Path(root).name}"
    meta = LeRobotDatasetMetadata(repo, root=root)
    cfg = ACTConfig() if policy == "act" else DiffusionConfig()
    ds = LeRobotDataset(repo, root=root, delta_timestamps=resolve_delta_timestamps(cfg, meta), video_backend=backend)
    dl = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True, num_workers=workers, pin_memory=True, drop_last=False,
                                     prefetch_factor=tdef.get("prefetch_factor", 2), persistent_workers=True,
                                     multiprocessing_context="spawn")
    it = iter(dl)
    for _ in range(warmup):
        next(it)
    t0 = time.time()
    n = 0
    while n < batches and time.time() - t0 < max_s:          # time cap so the whole sweep stays inside the 10 minutes
        next(it)
        n += 1
    dt = time.time() - t0
    del it, dl
    return {"policy": policy, "backend": backend, "workers": workers, "batches": n, "seconds": dt, "samples_per_s": n * 32 / dt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, nargs="+", default=[4, 8])
    ap.add_argument("--backends", nargs="+", default=["pyav", "torchcodec"])
    ap.add_argument("--batches", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--max-s", type=float, default=50.0, help="time cap per setting (s)")
    a = ap.parse_args()
    rows = []
    for policy in ("act", "diffusion"):
        for b in a.backends:
            for w in a.workers:
                try:
                    r = bench(a.root, policy, b, w, a.batches, a.warmup, a.max_s)
                except Exception as e:                    # a backend that cannot load is simply not chosen
                    r = {"policy": policy, "backend": b, "workers": w, "error": repr(e)[:300]}
                print(json.dumps(r), flush=True)
                rows.append(r)
    ok = [r for r in rows if "samples_per_s" in r]
    best_by = {}
    for r in ok:
        best_by.setdefault(r["policy"], max(x["samples_per_s"] for x in ok if x["policy"] == r["policy"]))
    # one setting for all six runs: the best mean of (throughput / best throughput of that policy) over both policies
    combos = sorted({(r["backend"], r["workers"]) for r in ok})
    score = {c: sum(r["samples_per_s"] / best_by[r["policy"]] for r in ok if (r["backend"], r["workers"]) == c) / 2 for c in combos
             if sum((r["backend"], r["workers"]) == c for r in ok) == 2}
    choice = max(score, key=score.get) if score else ("pyav", 4)
    res = {"rows": rows, "score": {f"{b}/{w}": s for (b, w), s in score.items()}, "choice": {"backend": choice[0], "workers": choice[1]}}
    Path(a.out).write_text(json.dumps(res, indent=1))
    print("CHOICE", choice[0], choice[1], flush=True)


if __name__ == "__main__":
    main()
