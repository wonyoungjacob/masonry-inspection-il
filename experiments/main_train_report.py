"""본 학습 보고 (본학습 지시 6절): 학습 6번의 기록·학습 곡선·정상 동작 확인 2벽을 모아 보고 폴더의 요약 JSON·보고서·작은 PNG를 다시 쓴다.

train_main.sh가 학습 하나가 끝날 때마다 부른다(학습 결과는 다시 계산하지 않고 로그·상태 파일·롤아웃 JSON에서 읽음).
실행: venv/bin/python -m experiments.main_train_report --train <sim/outputs/wall3/main_train_날짜> --dataset <sim/data/...> --out-dir <보고 폴더>
      [--final]  (6번 다 끝났을 때: 실험일지 항목도 씀)
"""
import argparse, hashlib, json, time
from pathlib import Path

import numpy as np

from experiments.mini_train_report import parse_log, summarize_rollout

SIM = Path(__file__).resolve().parents[1]
REPO = SIM.parent
ORDER = [("act", 1000), ("diffusion", 1000), ("act", 2000), ("diffusion", 2000), ("act", 3000), ("diffusion", 3000)]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fmt_t(t):
    return time.strftime("%m-%d %H:%M", time.localtime(t)) if t else "—"


def parse_runs(s):
    return [(r.split()[0], int(r.split()[1])) for r in s.split(",") if r.strip()]


def run_info(train: Path, pol, seed):
    name = f"{pol}_s{seed}"
    st = train / f"{name}.status.json"
    if not st.exists():
        return {"name": name, "policy": pol, "seed": seed, "result": "not started"}
    s = json.loads(st.read_text())
    out = {"name": name, "policy": pol, "seed": seed, **s}
    log = train / f"{name}.log"
    if log.exists():
        rows, speed = parse_log(log)
        if rows:
            a = np.array(rows)
            out.update(steps_logged=int(a[-1, 0]), loss_first=float(a[0, 1]), loss_last=float(a[-1, 1]),
                       loss_last5_mean=float(a[-5:, 1].mean()), loss_min=float(a[:, 1].min()),
                       steps_per_s=float(np.median(speed[len(speed) // 10:])) if speed else None,
                       update_s_median=float(np.median(a[:, 2])), data_s_median=float(np.median(a[:, 3])))
            out["_curve"] = a[:, :2].tolist()
    ck = train / name / "checkpoints" / "last" / "pretrained_model"
    if (ck / "model.safetensors").exists():
        out["checkpoint"] = str(ck.resolve().relative_to(SIM)) if ck.resolve().is_relative_to(SIM) else str(ck.resolve())
        out["checkpoint_step_dirs"] = sorted(p.name for p in (train / name / "checkpoints").iterdir() if p.name != "last")
        out["model_sha256"] = sha256(ck / "model.safetensors")
        cfg = json.loads((ck / "config.json").read_text())
        out["input_features"] = list(cfg.get("input_features", {}).keys())
        out["chunk"] = {k: cfg.get(k) for k in ("chunk_size", "n_action_steps", "horizon", "n_obs_steps") if k in cfg}
    ro = train / f"rollout_{name}"
    if (ro / "rollout.json").exists():
        r = summarize_rollout(ro)
        d = json.loads((ro / "rollout.json").read_text())
        r["iou_mean_per_wall"] = [w["eval"]["iou_mean"] for w in d["walls"]]
        r["filled_per_wall"] = [int(sum(w["eval"]["slots_filled"])) for w in d["walls"]]
        r["lifted_per_wall"] = [int(sum(b["grasped_lifted"] for b in w["bricks"])) for w in d["walls"]]
        r["spec_per_wall"] = [bool(w["eval"]["spec_pass"]) for w in d["walls"]]
        r["wall_clock_s_per_wall"] = [w["wall_clock_s"] for w in d["walls"]]
        r["scene_sha256"] = d["meta"].get("scene_sha256")
        out["rollout"] = r
    return out


def curve_png(info, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    a = np.array(info["_curve"])
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    ax.semilogy(a[:, 0], a[:, 1], lw=0.8)
    ax.set_title(f"{info['name']} (main, 1,000 walls)", fontsize=9)
    ax.set_xlabel("step", fontsize=8); ax.set_ylabel("train loss", fontsize=8); ax.tick_params(labelsize=7); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=80); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--runs", default=",".join(f"{p} {s}" for p, s in ORDER), help='"act 1000,diffusion 1000,..."')
    ap.add_argument("--journal", default=str(REPO / "실험일지.md"), help="lab log the --final entry goes to")
    ap.add_argument("--title", default="3층 벽체 본 학습", help="report title / lab-log heading")
    ap.add_argument("--note", default=None, help="markdown paragraph under the title (e.g. what differs from the main training)")
    a = ap.parse_args()
    train, ds, out = Path(a.train), Path(a.dataset), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    runs = [run_info(train, p, s) for p, s in parse_runs(a.runs)]
    for r in runs:
        if "_curve" in r and r.get("result") in ("done", "failed", "skipped", "stopped"):
            curve_png(r, out / f"학습곡선_{r['name']}.png")
    ver = json.loads((ds / "meta" / "wall3_verify.json").read_text())
    src = json.loads((ds / "meta" / "wall3_source.json").read_text())
    bench = json.loads((train / "bench_loader.json").read_text()) if (train / "bench_loader.json").exists() else None
    setting = json.loads((train / "setting.json").read_text()) if (train / "setting.json").exists() else {}
    stop = (train / "STOP").read_text().strip() if (train / "STOP").exists() else None
    summ = {"dataset": {"root": str(ds), "episodes": ver["num_episodes"], "frames": ver["num_frames"], "dataset_sha256": ver.get("dataset_sha256"),
                        "content_sha256": ver.get("content_sha256"), "bytes": ver.get("bytes"), "walls": len(src["seeds_used"]),
                        "seeds": [src["seeds_used"][0], src["seeds_used"][-1]], "excluded_seeds": sorted({s for p in src.get("parts", [src]) for s in p.get("excluded_seeds", [])}),
                        "freeze_commit": "482da46"},
            "setting": setting, "bench": bench, "stop": stop, "runs": [{k: v for k, v in r.items() if k != "_curve"} for r in runs]}
    (out / "요약.json").write_text(json.dumps(summ, indent=1, ensure_ascii=False, default=float))

    L = [f"# {a.title} 보고 (Ubuntu)", "", *([a.note, ""] if a.note else []),
         "지시서: [`우분투지시_3층벽체_본학습.md`](../우분투지시_3층벽체_본학습.md). 이 보고서는 `sim/experiments/main_train_report.py`가 학습이 하나 끝날 때마다 다시 쓴다. "
         "요약 [`요약.json`](요약.json).", ""]
    if (out / "경위.md").exists():
        L += ["재시작·결정 경위: [`경위.md`](경위.md)", ""]
    if stop:
        L += [f"> **멈춤:** {stop}", ""]
    d = summ["dataset"]
    L += ["## 데이터셋", "", "| 항목 | 값 |", "|---|---|",
          f"| 경로 | `sim/{Path(d['root']).relative_to(SIM) if Path(d['root']).is_absolute() else d['root']}` |",
          f"| 벽 · 에피소드 · 프레임 | {d['walls']:,} · {d['episodes']:,} · {d['frames']:,} |",
          f"| 시작값 | {d['seeds'][0]} ~ {d['seeds'][1]} (과부하 벽 3개와 결정으로 뺀 {', '.join(map(str, d['excluded_seeds']))} 제외) |",
          f"| 데이터 해시 (파일) | `{d['dataset_sha256']}` |", f"| 내용 해시 | `{d['content_sha256']}` |",
          f"| 크기 | {d['bytes'] / 1e9:.1f} GB |" if d.get("bytes") else "| 크기 | — |", f"| 동결 커밋 | `{d['freeze_commit']}` |", ""]
    L += ["## 학습 설정 (6번 모두 같음)", "", "| 항목 | 값 |", "|---|---|",
          "| 정책 | ACT, Diffusion Policy (LeRobot 0.6.1 기본 구현·기본 설정 = 미니 학습과 같음) |",
          "| 행동 묶음 | ACT chunk 100 · 실행 100, Diffusion horizon 64 · 실행 32 · 관측 2 (미니 체크포인트 config.json과 같음) |",
          f"| 스텝 · 배치 · 저장 | 100,000 · 32 · {setting.get('save_freq', 20000):,} 스텝마다, 쓸 체크포인트 = 마지막 |",
          "| 학습 | " + ", ".join(f"{r['policy']} s{r['seed']}" for r in runs) + " |",
          f"| 데이터 로딩 (속도 설정) | 디코더 **{setting.get('backend', '—')}**, 작업자 **{setting.get('workers', '—')}**, 동시 실행 **{setting.get('concurrent', '—')}** |", ""]
    if bench:
        L += ["### 속도 설정 측정 (학습 시작 전, 초당 샘플 수)", "", "| 정책 | 디코더 | 작업자 | 샘플/s |", "|---|---|---|---|"]
        for r in bench["rows"]:
            L.append(f"| {r['policy']} | {r['backend']} | {r['workers']} | {r.get('samples_per_s', float('nan')):.0f} |" if "samples_per_s" in r
                     else f"| {r['policy']} | {r['backend']} | {r['workers']} | 실패 |")
        L += ["", f"고른 값: {bench['choice']['backend']} · 작업자 {bench['choice']['workers']} (두 정책 평균 상대 속도가 가장 높은 조합, 6번 모두 같게)", ""]
    L += [f"## 학습 {len(runs)}번", "", "| 정책 | seed | 결과 | 시작 | 끝 | 걸린 시간 | 스텝/s | 처음 → 마지막 손실 | 이어 하기 | 체크포인트 해시 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in runs:
        dur = f"{(r['end'] - r['start']) / 3600:.1f} h" if r.get("end") and r.get("start") else "—"
        loss = f"{r['loss_first']:.3f} → {r['loss_last']:.4f}" if "loss_first" in r else "—"
        sps = f"{r['steps_per_s']:.1f}" if r.get("steps_per_s") else "—"
        L.append(f"| {r['policy']} | {r['seed']} | {r.get('result')} | {fmt_t(r.get('start'))} | {fmt_t(r.get('end'))} | {dur} | {sps} | {loss} | "
                 f"{max(r.get('attempts', 1) - 1, 0) if r.get('attempts') else '—'} | {('`' + r['model_sha256'][:12] + '…`') if r.get('model_sha256') else '—'} |")
    L += ["", "체크포인트: `sim/outputs/wall3/main_train_<날짜>/<정책>_s<seed>/checkpoints/last/pretrained_model`(저장소 밖). 학습 곡선 PNG는 이 폴더의 `학습곡선_*.png`.", ""]
    L += ["## 정상 동작 확인 (학습마다 정책 단독 2벽, 결과로 쓰지 않음)", "",
          "동결 `main_v1`(설정 해시 `10909a50…`), 시작값 700300 · 700301, G1·G2·G3 끔, 벽돌당 17 s, 도면 순서표로 넘어감. 녹화는 첫 벽(저장소 밖).", "",
          "| 정책 | seed | 들어 올린 벽돌 (벽별) | 채워진 자리 (벽별) | IoU 평균 (벽별) | 시방 | 벽 하나 평가 시간 |", "|---|---|---|---|---|---|---|"]
    for r in runs:
        ro = r.get("rollout")
        if not ro:
            continue
        L.append(f"| {r['policy']} | {r['seed']} | {' / '.join(map(str, ro['lifted_per_wall']))} (16장 중) | {' / '.join(map(str, ro['filled_per_wall']))} | "
                 f"{' / '.join(f'{x:.3f}' for x in ro['iou_mean_per_wall'])} | {sum(ro['spec_per_wall'])} / {len(ro['spec_per_wall'])} | "
                 f"{' / '.join(f'{x:.0f}' for x in ro['wall_clock_s_per_wall'])} s |")
    L += ["", "## 실험설계 md와 다르게 한 것", "", "| # | 항목 | 실험설계 md / 지시 | 실제 |", "|---|---|---|---|",
          f"| 1 | 데이터 로딩 | (미니 학습: pyav, 작업자 4) | {setting.get('backend', '—')}, 작업자 {setting.get('workers', '—')} (지시 4절이 허용한 속도 설정만) |",
          "| 3 | 학습 seed | 1000 · 2000 · 3000 (6번) | **사용자 결정(10-19): 우선 seed 1000만, ACT·Diffusion 2번** |",
          "| 4 | 체크포인트 저장 | 2만 스텝마다 | **사용자 결정(10-19): 1만 스텝마다** |",
          "| 5 | 시작 조건 | 다시 읽기 검사(에피소드마다 3프레임) | 영상 파일 전부 끝까지 디코딩해 프레임 수 확인 추가(깨진 영상 3개 때문, 경위.md) |",
          "| 2 | 정상 동작 확인 시작값 700300 · 700301 | 파일럿 블록 미사용 값 | 이 두 값은 2차 작업 때 수집 일정 시험(대본, `outputs/wall3/collect_sched_test_20261018`, 어느 데이터셋에도 안 들어감)에 한 번 쓰였음. 정책 롤아웃에는 처음 |", ""]
    (out / "보고서.md").write_text("\n".join(L))
    if a.final:
        j = Path(a.journal)
        u = j.read_text()
        done = [r for r in runs if r.get("result") == "done"]
        lines = [f"- {r['name']}: {(r['end'] - r['start']) / 3600:.1f} h, 손실 {r['loss_first']:.3f} → {r['loss_last']:.4f}"
                 + (f", 확인 2벽 들어 올림 {'/'.join(map(str, r['rollout']['lifted_per_wall']))}, 채움 {'/'.join(map(str, r['rollout']['filled_per_wall']))}" if r.get("rollout") else "")
                 for r in done]
        head = ("3층 벽체 본 학습 6번 (ACT·Diffusion × seed 1000·2000·3000, 10만 스텝)" if a.title == "3층 벽체 본 학습"
                else f"{a.title} ({', '.join(r['name'] for r in runs)}, 10만 스텝)")
        entry = (f"# 실험일지\n\n## {time.strftime('%Y-%m-%d')} — {head} (Ubuntu, Claude Code)\n\n"
                 f"상세 `01_사기설/{Path(a.out_dir).name}/보고서.md`. 데이터 `{d['root']}` (에피소드 {d['episodes']:,}, 내용 해시 `{(d['content_sha256'] or '')[:12]}…`).\n"
                 f"- 끝난 학습 {len(done)} / {len(runs) if a.title != '3층 벽체 본 학습' else 6}" + (f", 멈춤: {stop}" if stop else "") + f", 데이터 로딩 {setting.get('backend')} · 작업자 {setting.get('workers')} · 동시 {setting.get('concurrent')}\n"
                 + "\n".join(lines) + "\n\n---\n\n")
        if not u.startswith("# 실험일지\n"):
            u = "# 실험일지\n" + u
        j.write_text(entry + u[len("# 실험일지\n"):].lstrip("\n"))
    print(json.dumps({r["name"]: r.get("result") for r in runs}))


if __name__ == "__main__":
    main()
