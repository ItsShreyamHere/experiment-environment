"""Instrument Calibration Program 0 (ICP0).

We are calibrating a telescope, not yet observing galaxies. ICP0 sweeps a pure
fixed-state recurrence (and an attention control) over state size N and recall
load K, then asks: does the instrument satisfy its Measurement Axioms? Only if it
does may the fitted slope of K*(N) be read as the constant b. Otherwise the run
is INCONCLUSIVE and contributes exactly zero evidence to any law.

It still records the Phenomena it observes (recall cliff, K* conservation), which
persist regardless of any law's fate.

Resumable: each cell commits its measurement + 'done' status atomically, so a
Ctrl-C leaves finished cells persisted; re-running `cde measure` skips them and
continues. Aggregation reads the measurements table (not in-memory state), so a
resumed run produces the same verdict as an uninterrupted one.
"""

from __future__ import annotations

import json
import math
from typing import Any

from ..db.connection import Database, utcnow
from ..phenomena import phenomenon as ph
from ..quantities import quantity as q
from ..utils import logging as log
from ..utils.hashing import short_hash
from . import estimator

ICP_ID = "ICP0"
STAGE = "icp0"


def _meas_row(quantity: str, arch: str, N: int, K, seed, log_V: float, value: float,
              run_id: str, **extra: Any) -> dict[str, Any]:
    meas_id = "meas|" + short_hash([ICP_ID, quantity, arch, N, K, seed])
    return {
        "meas_id": meas_id, "quantity": quantity, "arch": arch, "dataset": "mqar",
        "N": N, "d_state": N, "K": K, "seed": seed, "log_V": log_V, "value": value,
        "ci_low": extra.get("ci_low"), "ci_high": extra.get("ci_high"),
        "repeatability": extra.get("repeatability"),
        "grad_norm": extra.get("grad_norm"), "grad_nonfinite": extra.get("grad_nonfinite"),
        "steps_run": extra.get("steps_run"), "method": ICP_ID,
        "mp_id": ICP_ID, "run_id": run_id, "created_at": utcnow(),
    }


def _k_values(N: int, mp: dict[str, Any]) -> list[int]:
    """K values for a given N. Per-N (relative to capacity) if k_multipliers is set,
    so every cell sits at a uniform overload ratio and can plateau cleanly; else the
    rectangular num_pairs list."""
    mults = mp.get("k_multipliers")
    if mults:
        ks = sorted({max(2, int(round(m * N))) for m in mults})
        cap = mp.get("k_max")
        if cap:
            ks = [k for k in ks if k <= int(cap)] or [min(ks)]
        return ks
    return list(mp.get("num_pairs", []))


def _cells(mp: dict[str, Any]):
    for arch in mp.get("arches", ["diag_ssm", "attention"]):
        for N in mp.get("state_dims", []):
            for K in _k_values(N, mp):
                for seed in mp.get("seeds", [0]):
                    yield arch, N, K, seed


def _train_worker(args):
    """Top-level (picklable) worker: train one cell. Runs in its own process so
    several tiny cells share the GPU concurrently (one model barely occupies it)."""
    arch, N, K, seed, mp = args
    from .train import train_cell
    return (arch, N, K, seed), train_cell(arch, N, K, seed, mp)


def run(db: Database, cfg, run_id: str, quick: bool = False) -> dict[str, Any]:
    mp = cfg.icp0(quick=quick)
    V = int(mp.get("vocab_size", 64))
    log_V = math.log2(V)
    workers = int(mp.get("workers", 1))
    cells = list(_cells(mp))
    done = db.done_units(STAGE)
    pending = [c for c in cells if f"{c[0]}:N{c[1]}:K{c[2]}:s{c[3]}" not in done]
    log.info(f"[ICP0] Instrument Calibration Program 0 — {len(cells)} cells; "
             f"{len(done)} done, {len(pending)} to run; workers={workers}.")

    runs_dir = cfg.data_path("runs") / "icp0"
    runs_dir.mkdir(parents=True, exist_ok=True)

    def record(cell, out) -> None:
        arch, N, K, seed = cell
        unit = f"{arch}:N{N}:K{K}:s{seed}"
        (runs_dir / f"{unit.replace(':', '_')}.json").write_text(
            json.dumps({"unit": unit, "arch": arch, "N": N, "K": K, "seed": seed,
                        "accuracy": out["accuracy"], "max_grad_norm": out.get("max_grad_norm"),
                        "nonfinite_steps": out.get("nonfinite_steps"), "steps_run": out.get("steps"),
                        "final_loss": out.get("final_loss"), "history": out.get("history", [])},
                       indent=1), encoding="utf-8")
        db.insert("measurements", _meas_row(
            "recall_accuracy", arch, N, K, seed, log_V, out["accuracy"], run_id,
            grad_norm=out.get("max_grad_norm"), grad_nonfinite=out.get("nonfinite_steps"),
            steps_run=out.get("steps")))
        db.mark(STAGE, unit, "done")
        log.dim(f"[ICP0] {unit} acc={out['accuracy']:.3f} k_correct={out['k_correct']:.1f} "
                f"max_gnorm={out.get('max_grad_norm')} nonfinite={out.get('nonfinite_steps')}")

    interrupted = False
    if workers > 1 and len(pending) > 1:
        import multiprocessing as _mp
        from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait
        # NOTE: per-process memory caps backfired on this 8GB GPU (per-process OOM /
        # cuBLAS workspace failures). Leave the allocator uncapped and keep workers low.
        ctx = _mp.get_context("spawn")
        ex = ProcessPoolExecutor(max_workers=workers, mp_context=ctx)
        futs = {}
        try:
            for cell in pending:
                db.mark(STAGE, f"{cell[0]}:N{cell[1]}:K{cell[2]}:s{cell[3]}", "running")
                futs[ex.submit(_train_worker, (*cell, mp))] = cell
            remaining = set(futs)
            while remaining:
                fdone, remaining = wait(remaining, return_when=FIRST_COMPLETED)
                for fut in fdone:
                    cell = futs[fut]
                    unit = f"{cell[0]}:N{cell[1]}:K{cell[2]}:s{cell[3]}"
                    try:
                        _, out = fut.result()
                        record(cell, out)
                    except Exception as exc:
                        db.mark(STAGE, unit, "failed", str(exc))
                        log.warn(f"[ICP0] cell {unit} failed: {exc}")
            ex.shutdown(wait=True)
        except KeyboardInterrupt:
            interrupted = True
            ex.shutdown(wait=False, cancel_futures=True)
            log.warn("[ICP0] interrupted (Ctrl-C). Finished cells saved; re-run `cde measure` to resume.")
    else:
        from .train import train_cell  # lazy (imports torch)
        try:
            for cell in pending:
                unit = f"{cell[0]}:N{cell[1]}:K{cell[2]}:s{cell[3]}"
                db.mark(STAGE, unit, "running")
                try:
                    out = train_cell(*cell, mp)
                except Exception as exc:
                    db.mark(STAGE, unit, "failed", str(exc))
                    log.warn(f"[ICP0] cell {unit} failed: {exc}")
                    continue
                record(cell, out)
        except KeyboardInterrupt:
            interrupted = True
            log.warn("[ICP0] interrupted (Ctrl-C). Finished cells saved; re-run `cde measure` to resume.")

    if interrupted:
        return {"verdict": "interrupted", "b": None, "done": db.count(
            "processing_status", "stage=? AND status='done'", (STAGE,))}

    return _aggregate(db, cfg, run_id, log_V)


def _aggregate(db: Database, cfg, run_id: str, log_V: float) -> dict[str, Any]:
    """Compute K* per (arch,N,seed) and the quantity b — entirely from the DB."""
    rows = db.query(
        "SELECT arch, N, K, seed, value, grad_nonfinite, steps_run, MAX(created_at) AS t "
        "FROM measurements WHERE quantity='recall_accuracy' "
        "GROUP BY arch, N, K, seed"
    )
    kcorr: dict[tuple, float] = {}
    attention_acc: list[float] = []
    grad_cells: list[dict[str, float]] = []
    for r in rows:
        acc = float(r["value"])
        kcorr[(r["arch"], int(r["N"]), int(r["K"]), int(r["seed"]))] = acc * int(r["K"])
        if r["arch"] == "attention":
            attention_acc.append(acc)
        elif r["arch"] == "diag_ssm":
            grad_cells.append({"N": int(r["N"]), "K": int(r["K"]),
                               "nonfinite": r["grad_nonfinite"] or 0, "steps": r["steps_run"] or 0})

    # K* per (arch, N, seed) = max over K of k_correct
    per_n_ssm: dict[int, list[float]] = {}
    seen: set[tuple] = set()
    for (arch, N, K, seed) in kcorr:
        if (arch, N, seed) in seen:
            continue
        seen.add((arch, N, seed))
        ks = [kcorr[(arch, N, kk, seed)] for (a2, n2, kk, s2) in kcorr
              if a2 == arch and n2 == N and s2 == seed]
        k_star = max(ks)
        db.insert("measurements", _meas_row("k_star", arch, N, None, seed, log_V, k_star, run_id))
        if arch == "diag_ssm":
            per_n_ssm.setdefault(N, []).append(k_star)
    db.commit()

    ssm_grid = _ssm_grid(kcorr)
    est = estimator.estimate(per_n_ssm, log_V, cfg.get("estimator", default={}),
                             attention_acc, ssm_grid=ssm_grid, grad_cells=grad_cells)
    if est.get("b") is not None:
        q.set_measured(db, "b", est["b"], est["b_ci"][0], est["b_ci"][1], run_id,
                       method="ICP0: slope of K*(N) * log V")
    _record_phenomena(db, kcorr, run_id, est)
    est["mp_id"] = ICP_ID
    log.success(f"[ICP0] verdict={est['verdict']} b={est.get('b')} "
                f"exponent={est.get('exponent')} reasons={est.get('reasons')}")
    return est


def _record_phenomena(db, kcorr, run_id, est) -> None:
    """Record observed regularities. These survive regardless of L1's fate."""
    ssm_N = sorted({n for (a, n, k, s) in kcorr if a == "diag_ssm"})
    ssm_K = sorted({k for (a, n, k, s) in kcorr if a == "diag_ssm"})
    if not ssm_N or len(ssm_K) < 2:
        return
    N_max = max(ssm_N)
    k_small = _mean_k(kcorr, "diag_ssm", N_max, min(ssm_K))
    k_large = _mean_k(kcorr, "diag_ssm", N_max, max(ssm_K))
    acc_small = (k_small / min(ssm_K)) if k_small is not None else None
    acc_large = (k_large / max(ssm_K)) if k_large is not None else None
    if acc_small and acc_large and acc_small > 0.7 and acc_large < 0.5:
        ph.record(db, ph.Phenomenon(
            name="recall cliff",
            statement=(f"At fixed state N={N_max}, per-item recall collapses as the number of "
                       f"stored pairs K grows (acc {acc_small:.2f}@K={min(ssm_K)} -> "
                       f"{acc_large:.2f}@K={max(ssm_K)})."),
            support=[f"ICP0 diag_ssm N={N_max}"],
        ), run_id)
    if est.get("verdict") == estimator.LINEAR:
        ph.record(db, ph.Phenomenon(
            name="K* conservation",
            statement=("K* scales linearly with state size N "
                       f"(exponent≈{est.get('exponent'):.2f}); K* conserved across input length."),
            support=[f"ICP0 b={est.get('b'):.3f}"],
        ), run_id)


def _mean_k(kcorr, arch, N, K):
    vals = [v for (a, n, k, s), v in kcorr.items() if a == arch and n == N and k == K]
    return sum(vals) / len(vals) if vals else None


def _ssm_grid(kcorr) -> dict[tuple[int, int], float]:
    """(N,K) -> mean k_correct over seeds, for the SSM only."""
    acc: dict[tuple[int, int], list[float]] = {}
    for (arch, N, K, seed), v in kcorr.items():
        if arch == "diag_ssm":
            acc.setdefault((N, K), []).append(v)
    return {nk: sum(vs) / len(vs) for nk, vs in acc.items()}
