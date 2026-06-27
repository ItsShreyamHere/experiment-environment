# Capstone Measurement — measure the constant `b` on the trap-escapee subpopulation

> Status: SPEC, not yet run. Picks up after MP0b-23. Goal: turn the suggestive "K* ∝ N" from MP0b-22
> into a CI-bounded measurement of `b` with an L1 curvature verdict. This is CDE's founding deliverable.

## 1. Goal

Measure `b` (effective bits per state coordinate) and the curvature of `K*(N)`, **conditioned on the
cells that escape the early-training optimization trap**, and read the verdict on L1:
- `K* ∝ N` (exponent p in [0.85, 1.15]) ⇒ **L1 holds**, `b ≈ K*·logV / N` is a real scale-stable constant.
- p > 1.15 ⇒ **super-linear ⇒ L1 collapses** (pure compression has no hard limit).
- p < 0.85 ⇒ sub-linear (also damages L1).

## 2. Why escapees (rationale — established MP0b-15..23)

The bimodal "stuck vs rescued" outcome is an **early-training critical-period trap** that is **specific to
bounded-state recurrence** (absent in attention, MP0b-23). Most seeds fall into a stuck basin → that is
*why* `b` was irreproducible and why A3 refused `b=6.54`. But the cells that **escape** the trap obey L1
cleanly: MP0b-22 found escapee K*≈24 @N=16 and ≈48.5 @N=32 → ratio 2.0 for 2× N → p≈1.01. So the clean
measurement is recoverable by **filtering to escapees** before the fit. This doc operationalizes that.

## 3. The measurement method (escapee-aware estimator)

Per capacity point N (with K=2N, V=16 ⇒ logV: keep the SAME log base the estimator already uses):
1. **Escapee selection.** Run many seeds; classify each cell's final outcome. An escapee = a cell in the
   HIGH basin (above the bimodal gap). Use the existing gap logic (`run_scale` already computes the
   largest-gap split / `stuck_frac`), OR a fixed acc threshold (e.g. final acc ≥ 0.6). Record both
   stuck-fraction(N) (the trap prevalence) and the escapee set.
2. **Convergence gate.** An escapee counts toward K* ONLY if it has plateaued (tail slope ≈ 0 over the
   last ~10% of steps). Under-converged escapees give K* LOWER BOUNDS — flag them, don't average them in.
3. **K*(N).** K* = max (or high-quantile) `k_correct` over converged escapees at N. (Max is closest to the
   true capacity; mean under-counts if some escapees are mid-ramp.)
4. **Fit + CI + curvature.** Feed the `(N, K*)` points to `src/lab/estimator.py` (it already does the
   `K*∝N^p` fit, bootstrap CI over seeds, and the p∈[0.85,1.15] linear / >1.15 superlinear thresholds —
   see `estimator.superlinear_threshold=1.15`, `sublinear_threshold=0.85`, `bootstrap_samples=1000`).
   **TODO before running:** add an escapee-filter + convergence-gate step in front of the estimator (it
   currently fits ALL cells; we need it to fit the converged-escapee K* per N). Small change in the
   estimator or a wrapper script.
5. **Report:** `b = value ± CI`, exponent `p ± CI`, the L1 verdict, AND stuck-fraction(N) alongside (the
   measurement is explicitly conditional on escape; the trap prevalence is part of the honest result).

## 4. Design (parameters)

- **arch:** `diag_ssm` (the architecture with the trap; attention is the control, not measured here).
- **N grid:** {16, 24, 32, 48} primary; **64 = stretch** (escape time may exceed feasible budget). More N
  points ⇒ better-conditioned `K*(N)` fit. K = 2N, d_model = 64, V = 16, num_queries = 32.
- **Seeds:** ≥16 per N (escape fraction was ~1/3 at warmup600 and DROPS at larger N within fixed budget,
  so larger N needs more seeds and/or more steps to yield ≥3–4 converged escapees).
- **Budget (steps):** the hard part — escape/jump time scales steeply with N (~13k@N=8 → ~100k+@N=16,
  MP0b-13; N=32 escapees were still under-converged at 400k, MP0b-22). Use **per-N budgets that grow with
  N** (e.g. N16:200k, N24:300k, N32:500k, N48:800k) rather than one fixed budget, so each N yields
  CONVERGED escapees. (`run_scale` uses one `steps` for all points — **TODO:** allow per-point `steps`,
  or run each N as a separate scale point/run.)
- **Hyperparameters:** canonical (lr 5e-4, warmup 600, AMP off, deterministic). RATIONALE: measure the
  natural capacity. (Optional variant: a gentler warmup raises the escapee yield — MP0b-20 — which is a
  legitimate way to get more converged cells for a *capacity* measurement; if used, report it explicitly.)
- **Platform:** Kaggle **T4 × 2** (the scale harness round-robins across both GPUs). **Use Commit mode**
  (Save Version → Save & Run All) so outputs persist to the version even if the live session ends —
  this is what bit us in MP0b-22 (zip lost when the interactive session closed).

## 5. Feasibility & staging (BE HONEST: this is a multi-session job)

Rough single-T4 cost ≈ minutes that scale ~linearly with N×steps. A full {16,24,32,48}×16-seed grid at
growing budgets is **tens of GPU-hours → several 12h Commit sessions**, even across 2 T4s. Plan it staged
and resumable; combine the `(N, K*)` points across stages for the final fit:
- **Stage 1 (1 session):** N∈{16,32}, 16 seeds, budgets 200k/500k → refine MP0b-22 to a 2-point `b` with
  real escapee counts + convergence gating. Already enough for a first CI on `b`.
- **Stage 2 (1 session):** add N∈{24,48} → 4 points → proper `K*(N)` fit + curvature verdict.
- **Stage 3 (stretch):** N=64 with a large budget → test super-linearity at the top end.
Resumability across sessions: download each session's `mp0b_results.zip` (Commit mode persists it) and
re-upload as a Kaggle Dataset for the next session, OR just treat each stage's `(N,K*)` as independent
inputs to the combined fit (cleanest — no state to carry).

## 6. How to run (when ready)

1. Set the `mp0b.scale` block to the stage's points/seeds/steps (I'll wire exact config + the per-point
   `steps` + estimator escapee-filter when we start — items flagged **TODO** above).
2. Push to GitHub `ItsShreyamHere/experiment-environment`.
3. Kaggle: **T4 ×2**, Internet on, **Commit mode**:
   ```python
   !git clone https://github.com/ItsShreyamHere/experiment-environment.git cde
   %cd cde && pip -q install click pyyaml rich
   !python kaggle/run_kaggle.py --workers 6
   ```
4. After: download `mp0b_results.zip` from the version output; I fold it into `data/runs/` and run the
   escapee-aware estimator → `b ± CI`, `p ± CI`, L1 verdict.

## 7. "Done" looks like

- A reported **`b = X ± CI`** (escapee subpopulation), an **exponent `p ± CI`**, and an explicit **L1
  verdict** (holds / super-linear-collapse / sub-linear).
- **stuck_fraction(N)** reported alongside (the measurement is conditional on escape; trap prevalence is
  part of the honest result).
- A short writeup appended to `MP0B_FINDINGS.md` (MP0b-24) + a phenomenon in `cde.db`.

## 8. Open decisions / risks

- **Budget vs N** is the dominant risk: too little ⇒ under-converged escapees ⇒ K* is a lower bound ⇒ p
  biased. Mitigate with per-N growing budgets + the convergence gate (only count plateaued escapees).
- **Escapee definition** (gap-split vs fixed acc threshold): report the fit under both; it should be robust.
- **logV convention:** match whatever `estimator.py`/L1 already use so `b`'s absolute value is comparable
  to the historical `b≈6.54` (MP0b-22 gave ~6.0 under log2(16)=4 — confirm the base before quoting `b`).
- **N=64 may be infeasible** on T4 within a session ⇒ treat as stretch; the fit can stand on {16,24,32,48}.

## 9. Inputs already in hand (don't redo)

- MP0b-22 (T4): escapee K*≈24@N16, ≈48.5@N32 (N32 under-converged ⇒ lower bound). Raw zip was lost; the
  per-cell finals are in `MP0B_FINDINGS.md` §MP0b-22. (Re-running N16/N32 with convergence gating is Stage 1.)
- Laptop archives for N=16 baselines: `data/runs/mp0b/scale_mp0b13_warmup600/` (warmup600),
  `scale_mp0b21_warmup3000/` (warmup3000). diag_ssm trap is laptop-deterministic; T4 is its own platform.
- Estimator: `src/lab/estimator.py` (fit + bootstrap + curvature thresholds) — reuse, add escapee filter.
