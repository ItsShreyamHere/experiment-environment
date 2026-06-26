"""Kaggle (P100) runner for CDE MP0b experiments.

One-cell entry point for running the current `cde study` experiment on a Kaggle P100:
  1. prints GPU / torch info;
  2. patches ONLY the mp0b.workers key to --workers (default 6) on this ephemeral clone
     (the committed config stays at 3, safe for the 4GB laptop);
  3. runs a P100-internal DETERMINISM SELF-CHECK (same cell trained twice must be identical) —
     this replaces the laptop-byte-match gate, which cannot hold across GPUs (different cuBLAS);
  4. runs the experiment that config.yaml dispatches to (currently MP0b-16 data_switch);
  5. prints a result summary (controls + rm/add crossovers);
  6. zips the run outputs to /kaggle/working for download.

Usage on Kaggle (P100, internet on):
    !git clone <your-repo-url> cde
    %cd cde
    !pip -q install click pyyaml rich
    !python kaggle/run_kaggle.py --workers 6
"""
from __future__ import annotations

import os
# Pin deterministic cuBLAS workspace BEFORE torch/CUDA initialises (matches train.py), so the
# P100 determinism self-check is valid. Must run before any `import torch`.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))   # so `import src.*` works when run as `python kaggle/run_kaggle.py`


def patch_mp0b_workers(cfg_path: Path, n: int) -> None:
    """Set workers ONLY inside the top-level `mp0b:` block (config has another under icp0)."""
    lines = cfg_path.read_text(encoding="utf-8").splitlines()
    in_mp0b = False
    out = []
    for ln in lines:
        top = re.match(r"^([A-Za-z_][\w-]*):", ln)      # a top-level key resets the section
        if top:
            in_mp0b = top.group(1) == "mp0b"
        if in_mp0b and re.match(r"^\s+workers:\s*\d+", ln):
            ln = re.sub(r"(workers:\s*)\d+", rf"\g<1>{n}", ln)
        out.append(ln)
    cfg_path.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"[kaggle] patched mp0b.workers -> {n}")


def determinism_selfcheck() -> None:
    import yaml
    from src.lab.train import train_cell
    mp = yaml.safe_load((ROOT / "config" / "config.yaml").read_text())["mp0b"]
    drop = ("data_switch", "scale", "grid2d", "sweep", "interp",
            "perturb_mag", "perturb_control", "hybrid", "init_predictor")
    base = {k: v for k, v in mp.items() if k not in drop}
    base.update(steps=500, log_every=100, warmup_steps=50, init_seed=3, data_seed=2)
    N, K = int(mp["N"]), int(mp["K"])
    a = train_cell("diag_ssm", N, K, 0, dict(base))
    b = train_cell("diag_ssm", N, K, 0, dict(base))
    same = (a["k_correct"] == b["k_correct"] and a["final_loss"] == b["final_loss"])
    print(f"[kaggle] P100 DETERMINISM SELF-CHECK (same cell x2, 500 steps): "
          f"{'PASS — identical' if same else '*** FAIL — NON-DETERMINISTIC ***'}  "
          f"(kA={a['k_correct']:.6f} kB={b['k_correct']:.6f}, lossA={a['final_loss']} lossB={b['final_loss']})")
    if not same:
        print("[kaggle] WARNING: P100 is not bit-deterministic with current settings. Results remain "
              "QUALITATIVELY usable, but A3-style exact reproducibility does not hold on this GPU. "
              "Flag this in the write-up.")


def summarize() -> None:
    base = ROOT / "data" / "runs" / "mp0b"
    scale = base / "scale"
    if scale.exists() and any(scale.glob("p*_s*.json")):
        print("\n=== MP0b-22 SCALE (N>=32) RESULT SUMMARY ===")
        pts = {}
        for p in scale.glob("p*_s*.json"):
            d = json.loads(p.read_text())
            pts.setdefault(int(d.get("point", 0)), []).append(d)
        for i in sorted(pts):
            recs = pts[i]
            N, K = recs[0].get("N"), recs[0].get("K")
            fk = sorted(r["final_k"] for r in recs)
            jumps = sorted(r.get("jump") for r in recs if r.get("jump"))
            hi = max(fk) if fk else 1
            stuck = sum(1 for x in fk if x < 0.5 * hi)
            gaps = [fk[j + 1] - fk[j] for j in range(len(fk) - 1)]
            print(f"  N={N} K={K}: finals={[round(x, 1) for x in fk]}")
            print(f"          n={len(fk)}  stuck(<0.5*max)={stuck}/{len(fk)}  "
                  f"max_gap={round(max(gaps), 1) if gaps else 0}  "
                  f"jumped={len(jumps)}/{len(fk)}  jump_range={(jumps[0], jumps[-1]) if jumps else None}")
        print("  (a split at a given N => the bimodal critical-period trap PERSISTS at that N;")
        print("   all-stuck @budget => transition likely exceeds budget, consistent with steep scaling.)")
        return
    ds = base / "data_switch"
    if not ds.exists():
        print("[kaggle] no scale or data_switch outputs found to summarize.")
        return
    recs = {json.loads(p.read_text())["tag"]: json.loads(p.read_text()) for p in ds.glob("*.json")}
    print(f"\n=== data_switch summary ({len(recs)} cells) ===")
    for tag in sorted(recs):
        print(f"  {tag}: final_k={round(recs[tag]['final_k'], 1)}")


def zip_outputs() -> None:
    base = ROOT / "data" / "runs" / "mp0b"
    if not base.exists():
        return
    dest_dir = Path("/kaggle/working") if Path("/kaggle/working").exists() else ROOT
    archive = shutil.make_archive(str(dest_dir / "mp0b_results"), "zip", str(base))
    print(f"[kaggle] zipped outputs -> {archive}  (download this for the laptop record)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6,
                    help="mp0b parallel cells (P100 16GB fits ~6; Kaggle has 4 vCPU)")
    ap.add_argument("--skip-selfcheck", action="store_true")
    args = ap.parse_args()

    import torch
    ngpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"[kaggle] torch {torch.__version__}  CUDA={torch.cuda.is_available()}  GPUs={ngpu}  "
          f"device0={torch.cuda.get_device_name(0) if ngpu else 'CPU'}")
    if ngpu > 1:
        print(f"[kaggle] multi-GPU: scale cells round-robin across all {ngpu} GPUs (use --workers ~{2*ngpu})")

    patch_mp0b_workers(ROOT / "config" / "config.yaml", args.workers)
    if not args.skip_selfcheck:
        determinism_selfcheck()

    print("[kaggle] launching `cde study` (dispatches to the active mp0b block) ...")
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    subprocess.run([sys.executable, "-m", "src.cli", "study"], cwd=str(ROOT), check=True, env=env)

    summarize()
    zip_outputs()


if __name__ == "__main__":
    main()
