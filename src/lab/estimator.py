"""Constant estimator — measurements -> the quantity b, and L1's verdict.

Pure, deterministic functions over stored measurement records. The crown-jewel
relation is K* = N·b/log V, so for the pure recurrence:

    slope of K*(N)  =  b / log V      =>      b = slope * log V

The *shape* of K*(N) decides L1's fate (roadmap §XI / decision tree):
    exponent p ≈ 1  (linear)      -> conservation law holds      -> L1 SURVIVES
    p > superlinear_threshold     -> capacity amplification       -> L1 COLLAPSES
    p < sublinear_threshold       -> interference-dominated        -> L1 WOUNDED
A non-flat attention control or too few points -> INVALID (no update).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from . import axioms

# verdicts
LINEAR = "linear"
SUPERLINEAR = "superlinear"
SUBLINEAR = "sublinear"
INVALID = "invalid"

# attack outcomes (imported names kept local to avoid a hard dep cycle)
_OUTCOME = {LINEAR: "survived", SUPERLINEAR: "killed", SUBLINEAR: "wounded", INVALID: "invalid"}


def verdict_to_outcome(verdict: str) -> str:
    return _OUTCOME.get(verdict, "invalid")


def _ols_slope_through_origin(N: np.ndarray, K: np.ndarray) -> float:
    denom = float(np.sum(N * N))
    return float(np.sum(N * K) / denom) if denom > 0 else 0.0


def _exponent(N: np.ndarray, K: np.ndarray) -> float | None:
    """Fit log K* = p log N + c; return p (the scaling exponent)."""
    mask = (N > 0) & (K > 0)
    if mask.sum() < 2:
        return None
    p, _ = np.polyfit(np.log(N[mask]), np.log(K[mask]), 1)
    return float(p)


def estimate(
    per_n: dict[int, list[float]],
    log_V: float,
    cfg_est: dict[str, Any] | None = None,
    attention_acc: list[float] | None = None,
    ssm_grid: dict[tuple[int, int], float] | None = None,
    grad_cells: list[dict[str, float]] | None = None,
) -> dict[str, Any]:
    """Estimate b and classify K*(N).

    per_n: N -> list of K* samples (one per seed) for the pure recurrence.
    attention_acc: per-cell recall accuracies of the attention control (sanity).
    ssm_grid: (N,K) -> mean k_correct, fed to the Measurement Axioms.
    grad_cells: per-cell {N,K,nonfinite,steps} for axiom A5.

    The verdict is INVALID (zero evidence) whenever a Measurement Axiom is
    violated — the instrument refuses to read an uncalibrated number as nature.
    """
    cfg_est = cfg_est or {}
    sup_thr = float(cfg_est.get("superlinear_threshold", 1.15))
    sub_thr = float(cfg_est.get("sublinear_threshold", 0.85))
    n_boot = int(cfg_est.get("bootstrap_samples", 1000))

    Ns = sorted(per_n.keys())
    N = np.array(Ns, dtype=float)
    means = np.array([float(np.mean(per_n[n])) for n in Ns], dtype=float)

    result: dict[str, Any] = {
        "per_N": [{"N": int(n), "k_star": float(np.mean(per_n[n])),
                   "samples": [float(x) for x in per_n[n]]} for n in Ns],
        "log_V": log_V,
    }

    # attention sanity: the control must recall well (flat, high) to validate the cell
    att_ok = True
    if attention_acc is not None and len(attention_acc) > 0:
        att_ok = float(np.mean(attention_acc)) >= 0.85
    result["attention_ok"] = att_ok
    result["attention_mean_acc"] = float(np.mean(attention_acc)) if attention_acc else None

    # Measurement Axioms: refuse to read an uncalibrated number as evidence.
    guard_reasons = axioms.check(ssm_grid, per_n, attention_acc, grad_cells, cfg_est)
    result["reasons"] = guard_reasons

    if len(Ns) < 2 or guard_reasons or np.all(means <= 0):
        result.update({"verdict": INVALID, "b": None, "b_ci": [None, None],
                       "slope": None, "exponent": None, "repeatability": None})
        return result

    slope = _ols_slope_through_origin(N, means)
    b = slope * log_V
    p = _exponent(N, means)

    # bootstrap CI for b over seed resamples
    boot_bs: list[float] = []
    rng = np.random.default_rng(0)  # fixed seed -> deterministic CI
    max_seeds = max(len(v) for v in per_n.values())
    if max_seeds >= 2:
        for _ in range(n_boot):
            resampled = []
            for n in Ns:
                s = per_n[n]
                idx = rng.integers(0, len(s), size=len(s))
                resampled.append(float(np.mean([s[i] for i in idx])))
            bs = _ols_slope_through_origin(N, np.array(resampled)) * log_V
            boot_bs.append(bs)
        ci = [float(np.percentile(boot_bs, 2.5)), float(np.percentile(boot_bs, 97.5))]
    else:
        ci = [b, b]

    # repeatability: across-seed agreement averaged over N
    reps = []
    for n in Ns:
        s = np.array(per_n[n], dtype=float)
        if s.mean() > 0 and len(s) > 1:
            reps.append(max(0.0, 1.0 - float(s.std()) / float(s.mean())))
    repeatability = float(np.mean(reps)) if reps else 1.0

    if p is None:
        verdict = INVALID
    elif p > sup_thr:
        verdict = SUPERLINEAR
    elif p < sub_thr:
        verdict = SUBLINEAR
    else:
        verdict = LINEAR

    result.update({
        "verdict": verdict,
        "b": b,
        "b_ci": ci,
        "slope": slope,
        "exponent": p,
        "repeatability": repeatability,
    })
    return result
