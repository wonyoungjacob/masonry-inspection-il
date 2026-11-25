"""데이터셋 영상 전수 검사 (본 학습 재시작 전, 2026-10-19): 모든 mp4를 pyav로 처음부터 끝까지 디코딩하고,
파일마다 디코딩된 프레임 수가 에피소드 메타(그 파일에 든 에피소드 길이의 합)와 같은지 확인한다.

이유: 10묶음 병렬 변환에서 영상 600개 중 3개가 깨졌는데(디코더 "Invalid data found"), 에피소드마다 3프레임만 읽는
기존 다시 읽기 검사로는 잡히지 않았다. 학습이 14,000스텝에서 그 파일을 만나 죽었다.

실행: venv/bin/python -m experiments.check_videos --root <데이터셋 폴더> [--out <json>] [--workers 12]
끝 줄 "VIDEOS OK" / "VIDEOS BAD"; 문제가 있으면 종료 코드 1.
"""
import argparse, json, sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path


def decode(path):
    import av
    try:
        n = 0
        with av.open(str(path)) as c:
            for _ in c.decode(video=0):
                n += 1
        return str(path), n, None
    except Exception as e:                                   # noqa: BLE001 - every decoder failure is a finding
        return str(path), -1, repr(e)[:300]


def expected_frames(root: Path):
    import pandas as pd
    ep = pd.concat([pd.read_parquet(f) for f in sorted((root / "meta" / "episodes").rglob("*.parquet"))])
    info = json.loads((root / "meta" / "info.json").read_text())
    vpath = info["video_path"]
    exp = defaultdict(int)
    cams = [k for k, v in info["features"].items() if v.get("dtype") == "video"]
    for cam in cams:
        ci, fi = f"videos/{cam}/chunk_index", f"videos/{cam}/file_index"
        for c, f, n in zip(ep[ci], ep[fi], ep["length"]):
            exp[str(root / vpath.format(video_key=cam, chunk_index=int(c), file_index=int(f)))] += int(n)
    return exp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    root = Path(a.root)
    exp = expected_frames(root)
    files = sorted(str(p) for p in (root / "videos").rglob("*.mp4"))
    with Pool(a.workers) as p:
        res = p.map(decode, files)
    got = {f: (n, err) for f, n, err in res}
    problems = []
    for f in sorted(set(exp) | set(got)):
        if f not in got:
            problems.append({"file": f, "problem": "missing file"})
        elif got[f][1]:
            problems.append({"file": f, "problem": got[f][1]})
        elif f not in exp:
            problems.append({"file": f, "problem": "file not referenced by any episode"})
        elif got[f][0] != exp[f]:
            problems.append({"file": f, "problem": f"frames {got[f][0]} != episodes {exp[f]}"})
    rep = {"root": str(root), "files": len(files), "frames_decoded": sum(n for n, e in got.values() if n > 0),
           "frames_expected": sum(exp.values()), "problems": problems, "ok": not problems}
    if a.out:
        Path(a.out).write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k != "problems"}), "problems:", problems[:5])
    print("VIDEOS OK" if rep["ok"] else "VIDEOS BAD")
    sys.exit(0 if rep["ok"] else 1)


if __name__ == "__main__":
    main()
