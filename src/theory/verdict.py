"""Turn stored MP0 measurements into an attack on a law.

Deterministic and torch-free: it re-reads the `measurements` table, re-runs the
(numpy-only) estimator, and applies the resulting verdict to the theory's attack
ledger. `cde measure` and `cde attack` therefore agree by construction.
"""

from __future__ import annotations

from typing import Any

from ..db.connection import Database
from ..lab import estimator
from . import objects as T
from .attack import Attack, apply


def _measurements(db: Database):
    """Read the latest value per cell across all runs (resume-safe), like _aggregate.

    Returns (per_n, attention_acc, ssm_grid).
    """
    per_n: dict[int, list[float]] = {}
    for r in db.query(
        "SELECT N, seed, value, MAX(created_at) AS t FROM measurements "
        "WHERE quantity='k_star' AND arch='diag_ssm' GROUP BY N, seed"
    ):
        per_n.setdefault(int(r["N"]), []).append(float(r["value"]))
    att = [float(r["value"]) for r in db.query(
        "SELECT value, MAX(created_at) AS t FROM measurements "
        "WHERE quantity='recall_accuracy' AND arch='attention' GROUP BY N, K, seed"
    )]
    grid_acc: dict[tuple[int, int], list[float]] = {}
    grad_cells: list[dict[str, float]] = []
    for r in db.query(
        "SELECT N, K, seed, value, grad_nonfinite, steps_run, MAX(created_at) AS t FROM measurements "
        "WHERE quantity='recall_accuracy' AND arch='diag_ssm' GROUP BY N, K, seed"
    ):
        grid_acc.setdefault((int(r["N"]), int(r["K"])), []).append(float(r["value"]) * int(r["K"]))
        grad_cells.append({"N": int(r["N"]), "K": int(r["K"]),
                           "nonfinite": r["grad_nonfinite"] or 0, "steps": r["steps_run"] or 0})
    ssm_grid = {nk: sum(vs) / len(vs) for nk, vs in grid_acc.items()}
    return per_n, att, ssm_grid, grad_cells


def attack_from_measurements(db: Database, cfg, theory_id: str, run_id: str) -> dict[str, Any]:
    """Estimate b/K*(N) from stored measurements and attack `theory_id` with the verdict."""
    per_n, att, ssm_grid, grad_cells = _measurements(db)
    est = estimator.estimate(per_n, _log_v(db), cfg.get("estimator", default={}), att,
                             ssm_grid=ssm_grid, grad_cells=grad_cells)
    verdict = est["verdict"]
    outcome = estimator.verdict_to_outcome(verdict)

    reasons = est.get("reasons") or []
    cause = {
        estimator.LINEAR: "K*(N) is linear; the recall conservation law held under attack.",
        estimator.SUPERLINEAR: "K*(N) grows super-linearly; the conserved item-count K*=N·b/log V is falsified.",
        estimator.SUBLINEAR: "K*(N) grows sub-linearly; interference degrades b below the linear law.",
        estimator.INVALID: "Measurement INCONCLUSIVE — instrument refuses to interpret: "
                           + ("; ".join(reasons) if reasons else "too few state sizes / control not flat"),
    }[verdict]
    detail = (
        f"verdict={verdict} exponent={est.get('exponent')} b={est.get('b')} "
        f"attention_ok={est.get('attention_ok')} reasons={reasons}"
    )
    summary = apply(
        db,
        Attack(theory_id=theory_id, kind="measurement", source="MP0", outcome=outcome, detail=detail),
        run_id,
        killer_detail={"cause": cause, "evidence": [detail]},
    )
    summary.update({"verdict": verdict, "estimate": est})
    return summary


def _log_v(db: Database) -> float:
    row = db.one("SELECT log_V FROM measurements WHERE log_V IS NOT NULL LIMIT 1")
    return float(row["log_V"]) if row else 6.0
