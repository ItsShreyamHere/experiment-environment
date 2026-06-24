# ICP0 Calibration Report — autonomous run

**Outcome:** the Constant Discovery Engine instrument is sound and behaves correctly;
it **refused to produce `b`** because the underlying recall measurement could not be
*calibrated* on this hardware. L1 received **zero evidence**. This is a successful
instrument outcome (it revealed apparatus limits, not a fact about nature).

## What was attempted
Calibrate the recall instrument: sweep a pure fixed-state recurrence (gated linear
attention) and an attention control over state size `N` and recall load `K`, and read
`b` from the slope of `K*(N)` — but **only if the Measurement Axioms hold** (A1/A2
monotone-then-plateau, A3 seed agreement, A4 control saturated, A5 finite gradients).

## Pathology ladder (each correctly caught and fixed)
1. **fp16 overflow** in the chunk-parallel scan → NaN gradients → flat/garbage `K*`.
   Fix: bound the decay (`A_MIN`) and run the scan in **fp32**. Result: `nonfinite=0`.
2. **Undertraining / instability** at high `K` → A2 "drops". Fixes: gradient clipping,
   LR warmup, `num_queries=32` (gradient signal), per-arch LR, longer training.
3. **Censoring** (K* floored at the smallest K) → fixed with **per-N K** multipliers
   `{0.25, 0.5, 1, 2}×N` so the curve rises before plateauing.
4. **GPU nondeterminism** (same cell → 32.0 or 5.2) → added deterministic algorithms +
   cuBLAS workspace config + 3-seed averaging.
5. **The compute wall (root cause).** A 30k-step convergence diagnostic showed the
   "drops" were **undertraining**: e.g. `N=8, K=16` climbs to `k_correct ≈ 13` (its true
   capacity) at 30 000 steps — but `N=16, K=32` still hadn't converged at 30 000 steps.

## The finding
- **Single-seed calibration succeeds at tiny N.** N∈{6,7,8}, K∈{0.5,1,2}×N, 30 000 steps,
  fp32: every per-cell `K_correct(K)` curve rises-then-plateaus (no drops), the attention
  control saturates, nothing is censored → **all axioms pass** → the instrument reads
  `b = 6.54 bits/dim`, exponent 0.529 (sublinear). The pipeline *can* produce a verdict.
- **But the measurement is not reproducible across seeds — A3 is the real wall.** Per-seed
  `K*`: N6 → {11.1, 6.0, 10.8}; N8 → {13.0, 7.9, 14.5}. The overloaded cells converge
  **bimodally**: some seeds reach the true plateau (~11–14), others get stuck at the
  recall-all-but-not-beyond value (~6–8). So the single-seed `b` was one lucky draw; the
  3-seed run is correctly flagged **A3 (seed disagreement)** → INVALID → L1 zero evidence.
- **The signal underlying L1 is real** (`K*` grows with `N` on converged seeds) — recorded
  as a Phenomenon — but a *reproducible* `b` needs an optimization that reliably reaches the
  plateau (more steps / LR schedule / a delta-rule overwrite model), beyond this GPU's reach.
- The instrument **correctly refused at every stage** (A2 under-convergence, then A3
  irreproducibility), so no spurious verdict ever reached L1. The A3 axiom caught what
  single-seed could not see — the deepest validation of the "refuse to over-interpret" design.

## Reproduce / continue on capable hardware
Working recipe is in `config/config.yaml` (`icp0`): `vocab_size 16`, `num_queries 32`,
`d_model 64`, `amp: false` (fp32), `A_MIN 0.5`, per-N `k_multipliers [0.25,0.5,1,2]`,
deterministic. Set **`steps: 30000`** (or higher; it must scale with N) and run
`cde measure` on a GPU that can afford it; then `cde attack L1` reads the verdict.
A **delta-rule / overwrite** recurrence (graceful degradation) is the principled model
change to make overloaded cells plateau at far fewer steps.

## Status of the engine itself
Complete and tested (21 tests). Resumable (Ctrl-C safe), parallel (`workers: 3`, ~1.43×),
offline/deterministic, no LLM. The Measurement Axioms (`src/lab/axioms.py`) are the
durable contribution: the instrument knows when it is not calibrated and says so.
