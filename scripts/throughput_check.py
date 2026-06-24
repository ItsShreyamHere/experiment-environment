"""Quick GPU throughput probe: time the same cells via the process pool at
different worker counts. Not a test; run directly:  python -m scripts.throughput_check
"""
from __future__ import annotations

import multiprocessing as mp_lib
import time
from concurrent.futures import ProcessPoolExecutor

import torch

from src.lab.icp0 import _train_worker

MP = dict(vocab_size=64, num_queries=32, d_model=128, n_layers=2, n_heads=4,
          batch_size=32, steps=800, lr=5e-4, amp=True, early_stop_acc=2.0,
          device="cuda", seq_len=1024, grad_clip=1.0, warmup_steps=200)
CELLS = [("diag_ssm", 64, 32, s) for s in range(6)]   # 6 same-shape cells, different seeds


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}  cells={len(CELLS)}  steps={MP['steps']}")
    base = None
    for W in (1, 4, 5, 6):
        ctx = mp_lib.get_context("spawn")
        mp_cfg = dict(MP, mem_fraction=round(0.9 / W, 3))
        t = time.time()
        try:
            with ProcessPoolExecutor(max_workers=W, mp_context=ctx) as ex:
                list(ex.map(_train_worker, [(*c, mp_cfg) for c in CELLS]))
            dt = time.time() - t
            if base is None:
                base = dt
            print(f"workers={W}: {dt:.1f}s  ({dt/len(CELLS):.1f}s/cell)  speedup x{base/dt:.2f}")
        except Exception as e:
            print(f"workers={W}: ERROR {str(e)[:100]}")


if __name__ == "__main__":
    main()
