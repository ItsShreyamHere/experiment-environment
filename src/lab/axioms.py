"""Measurement Axioms — properties the *instrument* must obey before any number
it produces may be read as evidence about nature.

These are not laws or theories about sequence models. They are calibration
requirements: structural facts a trustworthy recall measurement cannot violate.
A measurement that breaks an axiom tells us about the instrument (overflow,
undertraining, seed noise, a broken control), not about L1 — so it must
contribute *exactly zero* evidence for or against any theory.

  A1  Monotone recall: for fixed (arch, N), recalled-count K_correct(K) must be
      non-decreasing in K up to a plateau. (More stored pairs cannot reduce the
      number correctly recalled.)
  A2  Plateaus allowed, drops forbidden: a fall in K_correct(K) is an instrument
      failure (undertraining / interference), never a property of nature.
  A3  Seed agreement: across seeds, per-N K* must be consistent (low dispersion).
  A4  Control saturation: the attention control must stay saturated (near-perfect
      recall) — otherwise the cell is mis-specified.
  A5  Finite optimization: nonfinite-gradient steps must stay below a small frac.

(There is also a censoring check: if K* is achieved only at the smallest K for
every N, K was never pushed past saturation and K* is a floor, not a plateau.)
"""

from __future__ import annotations

from typing import Any

# A1/A2 are one criterion (monotone-then-plateau); A2 is the "no drop" half.
AXIOMS = {
    "A1": "K_correct(K) is non-decreasing then plateaus for fixed (arch, N)",
    "A2": "K_correct(K) never drops (a drop = undertraining/interference, not nature)",
    "A3": "per-N K* agrees across seeds (low dispersion)",
    "A4": "attention control stays saturated",
    "A5": "no nonfinite gradients beyond a small fraction of steps",
}


def check(
    ssm_grid: dict[tuple[int, int], float] | None,
    per_n_seeds: dict[int, list[float]] | None = None,
    attention_acc: list[float] | None = None,
    grad_cells: list[dict[str, float]] | None = None,
    cfg: dict[str, Any] | None = None,
) -> list[str]:
    """Return a list of axiom-violation reasons (empty list == calibrated)."""
    cfg = cfg or {}
    drop_tol = float(cfg.get("monotonic_drop_tol", 0.8))   # must stay >= tol * running max
    seed_cv_max = float(cfg.get("seed_cv_max", 0.25))
    att_min = float(cfg.get("attention_min", 0.85))
    nonfinite_frac_max = float(cfg.get("nonfinite_frac_max", 0.02))

    reasons: list[str] = []

    # A1 / A2 — monotone-then-plateau (the instrument's central theorem). No exceptions.
    if ssm_grid:
        Ns = sorted({n for (n, _) in ssm_grid})
        for n in Ns:
            ks = sorted(k for (nn, k) in ssm_grid if nn == n)
            run_max = 0.0
            for k in ks:
                v = ssm_grid[(n, k)]
                if run_max > 0 and v < drop_tol * run_max:
                    reasons.append(
                        f"A2 violated: K_correct drops at N={n} "
                        f"(K={k} -> {v:.1f} < {drop_tol:.0%} of {run_max:.1f}); undertraining")
                    break
                run_max = max(run_max, v)
        # censoring: K* achieved only at the smallest K for every N
        if len({k for (_, k) in ssm_grid}) >= 2:
            censored = True
            for n in Ns:
                kc = {k: ssm_grid[(n, k)] for k in sorted(k for (nn, k) in ssm_grid if nn == n)}
                if kc and max(kc, key=kc.get) != min(kc):
                    censored = False
                    break
            if censored:
                reasons.append("censored: larger K never increased recalled items; "
                               "K below saturation, K* is a floor not a plateau")

    # A3 — seed agreement (only meaningful with >=2 seeds)
    if per_n_seeds:
        for n, samples in per_n_seeds.items():
            if len(samples) >= 2:
                mean = sum(samples) / len(samples)
                if mean > 0:
                    var = sum((x - mean) ** 2 for x in samples) / len(samples)
                    cv = (var ** 0.5) / mean
                    if cv > seed_cv_max:
                        reasons.append(f"A3 violated: K* dispersion across seeds at N={n} "
                                       f"(CV={cv:.2f} > {seed_cv_max})")

    # A4 — attention control saturated
    if attention_acc:
        amean = sum(attention_acc) / len(attention_acc)
        if amean < att_min:
            reasons.append(f"A4 violated: attention control not saturated (mean {amean:.2f} < {att_min})")

    # A5 — finite optimization
    if grad_cells:
        for c in grad_cells:
            steps = c.get("steps") or 0
            nf = c.get("nonfinite") or 0
            if steps and nf > nonfinite_frac_max * steps:
                reasons.append(f"A5 violated: {nf}/{steps} nonfinite-grad steps "
                               f"(> {nonfinite_frac_max:.0%}) at N={c.get('N')} K={c.get('K')}")
                break

    return reasons
