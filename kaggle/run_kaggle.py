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
    ds = ROOT / "data" / "runs" / "mp0b" / "data_switch"
    if not ds.exists():
        print("[kaggle] no data_switch outputs found.")
        return
    recs = {}
    for p in ds.glob("*.json"):
        d = json.loads(p.read_text())
        recs[d["tag"]] = d
    def fk(t):
        return round(recs[t]["final_k"], 1) if t in recs else None
    Ts = [10000, 20000, 30000, 50000, 75000]
    print("\n=== MP0b-16 (P100) RESULT SUMMARY ===")
    print(f"  P100 controls:  ctrl_rescue(i3,d2) = {fk('ctrl_rescue')}   ctrl_stuck(i3,d0) = {fk('ctrl_stuck')}")
    print(f"  rm  (start d2 rescue, ->d0 at T): " +
          "  ".join(f"T{T//1000}k={fk(f'rm_T{T}')}" for T in Ts))
    print(f"  add (start d0 stuck,  ->d2 at T): " +
          "  ".join(f"T{T//1000}k={fk(f'add_T{T}')}" for T in Ts))
    print("  (rm crossover = rescue LOCK-IN step; add crossover = window-CLOSE step.)")
    print(f"  cells present: {len(recs)}/12")


def zip_outputs() -> None:
    ds = ROOT / "data" / "runs" / "mp0b" / "data_switch"
    if not ds.exists():
        return
    dest_dir = Path("/kaggle/working") if Path("/kaggle/working").exists() else ROOT
    archive = shutil.make_archive(str(dest_dir / "mp0b16_p100_results"), "zip", str(ds))
    print(f"[kaggle] zipped outputs -> {archive}  (download this for the laptop record)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6,
                    help="mp0b parallel cells (P100 16GB fits ~6; Kaggle has 4 vCPU)")
    ap.add_argument("--skip-selfcheck", action="store_true")
    args = ap.parse_args()

    import torch
    print(f"[kaggle] torch {torch.__version__}  CUDA={torch.cuda.is_available()}  "
          f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

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
