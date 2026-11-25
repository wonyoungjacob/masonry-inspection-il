"""나5 환경 동결 기록: 커밋, wall3 코드 변경 여부, 장면·대본·G1 설정과 그 해시, 자산 해시.

실행: venv/bin/python -m wall3.freeze --env-version main_v1 --out <json>
"""
import argparse, hashlib, json, subprocess
from pathlib import Path

SIM = Path(__file__).resolve().parents[1]


def h(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=float).encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-version", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from wall3.config import scene_config
    from wall3.expert import expert_config
    from wall3.g1 import G1Config
    from wall3.collect import asset_info
    scene, expert, g1 = scene_config(a.env_version).to_dict(), expert_config(a.env_version).to_dict(), G1Config().to_dict()
    code = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((SIM / "wall3").glob("*.py"))}
    rec = {"env_version": a.env_version,
           "git_commit": subprocess.check_output(["git", "-C", str(SIM), "rev-parse", "HEAD"], text=True).strip(),
           "git_dirty_wall3": bool(subprocess.check_output(["git", "-C", str(SIM), "status", "--porcelain", "--", "wall3"], text=True).strip()),
           "scene_sha256": h(scene), "expert_sha256": h(expert), "g1_sha256": h(g1), "code_sha256": h(code),
           "scene": scene, "expert": expert, "g1": g1, "code_files": code, "assets": asset_info()}
    Path(a.out).write_text(json.dumps(rec, indent=1, default=float, ensure_ascii=False))
    print(json.dumps({k: rec[k] for k in ("env_version", "git_commit", "git_dirty_wall3", "scene_sha256", "expert_sha256", "g1_sha256", "code_sha256")}, indent=1))


if __name__ == "__main__":
    main()
