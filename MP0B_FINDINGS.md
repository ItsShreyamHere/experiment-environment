# MP0b — Bimodal Convergence Investigation

A *study* of the phenomenon ICP0 uncovered (A3 / irreproducibility), not a fix.
Cell under study: `diag_ssm`, N=8, K=16 (2× overload), V=16, d_model=64, fp32,
lr=5e-4, identical hyperparameters across seeds; only the random seed varies.

## MP0b-1 — distribution & metastability (16 seeds, 30 000 steps)

**Final K_correct, sorted:**
```
STUCK:  5.1 5.2 5.7 5.8 6.0 6.6 6.7 7.4   |  gap 3.8  |   PLATEAU: 11.2 11.5 12.4 13.0 14.0 14.3 14.5 14.6
```
- **Cleanly bimodal:** two well-separated clusters, **nothing in the 7.4→11.2 gap**, an
  exact **8/16 (50%) split**. CV = 0.39. Same architecture + same hyperparameters → two worlds.
- **Metastability signature (from trajectories):** the populations are caught
  *mid-transition* at 30k steps:
  - plateau seeds split into **early-jumpers** (acc flat-high by ~15k) and
    **still-rising-at-the-end** (seeds 4/10/13: slope **+0.07…+0.09** over the last 3k steps —
    not converged; their reported K* underestimates their true plateau);
  - stuck seeds split into **flat** (3/7/8/11/12: slope ≈ 0 — possibly trapped) and
    **creeping** (9: +0.047, 1/14: +0.02 — slowly transitioning).
- **Interpretation:** the 30k bimodality is partly a *snapshot of an ongoing stochastic
  transition* (grokking-like delayed jumps), not necessarily two permanent basins.

## MP0b-2 — extended training (16 seeds, 100 000 steps): METASTABILITY CONFIRMED
- **Stuck fraction fell 8/16 (50%) → 3/16 (19%)** as 5 seeds underwent late transitions
  (e.g. seed 1: 5.7→13.2, seed 3: 6.6→13.0, seed 9: 6.0→10.0).
- **None of the remaining "stuck" seeds are flat-trapped** — all 3 are still rising at 100k
  (seed 12 +0.053, seed 8 +0.025, seed 11 +0.013).
- **Jump-time distribution (acc crosses 0.6):** `4600 … 93400`, a **20× spread**, heavy-tailed,
  median ~23 200; one seed jumped at 93 400 (near the end). The empty 30k gap (7.4→11.2) fills
  in with intermediate finals (8.0, 9.2, 9.7, 10.0).
- **Conclusion:** bimodal convergence is **metastable, grokking-like delayed generalization**,
  NOT a lottery-ticket permanent basin. A3 irreproducibility at a fixed step budget is the
  *tail* of the jump-time distribution. A reproducible `b` is recoverable in principle, but the
  required budget is set by the heavy tail (very large).

## MP0b-3 — weight-decay sweep: NULL. `weight_decay ∈ {0, 0.01, 0.1}` (incl. zero), 6 seeds each.
Stuck fraction 2/6 and median jump ~13–15k steps at **every** value; bimodal structure intact.
→ **Not** canonical (Power et al.) grokking; weight decay does not control this transition.

## MP0b-4 — LR sweep: (largely) NULL. `lr ∈ {2e-4, 5e-4, 1.5e-3}`, 6 seeds each.
Stuck fraction **2/6 at every LR** (7.5× range); only a weak, non-monotone timing hint
(lowest LR jumps latest, median 24k vs ~13–15k). → **Not cleanly thermal** either; neither
weight decay nor LR controls the basin split.

## Crystallized picture (after 30k/100k + WD + LR)
A robust metastable transition with a **broad, heavy-tailed jump-time distribution** that
**resolves with training time** (MP0b-2) but is **insensitive to weight decay and LR**. The
fraction not-yet-jumped at a fixed budget is remarkably stable (~2/6 at 40k). This is *not*
classic grokking and *not* simple thermal escape.

## MP0b-5 — init × data decoupling: INIT-DETERMINED (lottery-ticket structure)
4×4 init×data grid, 40k steps. Variance of final recall: **66% by INIT, 13% by DATA, 21%
interaction**. `init0` → plateau for *every* data order; `init1` → stuck for *every* data order
(winning vs dead tickets); `init2/3` borderline (data tips them). Combined with MP0b-2 (the dead
`init1` *does* escape at ~71k with more training), the synthesis is an **initialisation-determined
ordering of metastable jump times**: everyone eventually crosses, but *when* is set largely at init.

## Free analysis — the die is cast at INIT but HIDDEN in early dynamics
From the 100k trajectories: eventual-plateau vs eventual-stuck accuracy ranges **overlap at 1k,
2k, 5k, 10k** (plateau [0.31,0.87] vs stuck [0.31,0.34] at 10k). So the winning-ticket property is
latent in the init weights but **not visible in the early training curve** — it only manifests at
the (variable, often late) jump.

## MP0b-6 — init-weight predictor: a seductive correlation
16 inits: only **`a_bias_mean` (mean decay-gate bias) correlated, r = −0.62**; param norm, q/out
spectral norm, stable rank all ~null. Sign read mechanistically: winning inits start with negative
decay bias (more forgetting). Looked like "the winning ticket = initialized to forget."

## MP0b-7 — CAUSAL intervention: the correlation is REFUTED
Force `init_decay_bias ∈ {−1, −0.3, 0, 0.3}` (6 seeds each). **No effect:** stuck fraction ~3/6 at
every value, mean final_k flat, correlation of *forced* bias with final +0.11 (near-null, opposite
sign to the observed −0.62). → **The decay bias is a confounded correlate, not the cause.** The real
winning-ticket coordinate remains unidentified (coarse init metrics do not capture it).

## Epistemic capstone
The project refuses to over-interpret at every level: **A3 killed a seductive result** (`b = 6.54`);
**MP0b-7 killed a seductive correlate** (decay bias). Born from an axiom refusing a measurement, the
investigation reaches its capstone with an intervention refusing the explanation of that measurement.

## MP0b-8 — causal block ablation (W=init0, L=init1): DISTRIBUTED, not localized
Swap one parameter block at a time; pure-W=13.6, pure-L=5.7. Mean final_k (3 seeds):
```
block   W_block->L (rescue?)   L_block->W (break?)
embed     5.6                   14.3     (irrelevant both ways)
to_q     14.8  RESCUES          12.7
to_k     13.0  RESCUES           6.6  BREAKS
to_v     12.4  RESCUES           9.5  partial break
to_a     12.7  RESCUES          13.9
out      12.2  RESCUES          10.3
head     13.8  RESCUES          14.0
```
**Case C/D.** Sufficiency is *redundant* — almost every W block rescues L (to_q is NOT special;
the earlier to_q-localization read is dead). Necessity is sparse/asymmetric — only `to_k` clearly
breaks W; `embed` is irrelevant both ways. Likely metastable barrier-kicking: any nudge toward W's
configuration crosses the barrier within 40k steps.

## MP0b-9 — perturb control (running): is the rescue even about W's weights?
If almost any W block rescues L, maybe a RANDOM block does too. Inject donor ∈ {0=W, 50,51,52=random}
block ∈ {to_q, to_a, embed} into L. If random ≈ W → the rescue is **perturbation / barrier-kicking**,
and "winning tickets transfer" dies. If only real-W rescues → W's specific configuration matters.

## MP0b-9 — perturb control: ticket transfer is DEAD
Inject a block from donor ∈ {W=0, 50,51,52=random} into L: rescue ~half the time, ~independent of
donor identity AND block. W's `embed` fails to rescue, a *random* `embed` succeeds. So the rescue is
not transferring W's config and not about which block — it's whether the specific perturbation kicks
L over the barrier. **"Winning tickets transfer" killed.**

## MP0b-10 — re-draw rate: "fresh 50% lottery" is DEAD (null)
16 random `to_q` donors into L → cleanly **bimodal** (stuck ~5.8 vs rescued ~13.9, clean gap) but
**rescued 11/16 = 69%, not 50%**. A random block scramble is *biased toward rescue*, not a neutral
re-draw. Surviving: two-basin structure robust under perturbation; rate is perturbation-dependent.

## Killed so far / Surviving so far
KILLED: weight decay · learning rate · decay-bias correlate · to_q localization · winning-block
transfer · "fresh 50% lottery". SURVIVING: bimodality · metastability · heavy-tailed jump times ·
initialization dominance (66%) · delayed transitions · two-basin structure robust under perturbation.

## MP0b-11 — GEOMETRY hypothesis (attack, not confirm)
Suspect: there is no hidden coordinate/ticket; the relevant quantity is geometric (distance to a
basin boundary, barrier height, landscape structure). Treat exactly like every prior dead story.
- **11a interpolation W→L (running):** train from θ_α=(1−α)W+αL; sharp phase boundary in α? jump-time
  divergence near it (critical slowing → barrier)? smooth or discontinuous?
- **11b perturbation magnitude:** θ=L+ε·noise across ε and many noise dirs; does rescue rate track ε
  (geometry) or the specific direction (kills the clean geometric reading)?
If geometry produces a clean, beautiful picture, distrust it *because* it is clean and attack it
(mode connectivity; a basin-distance metric must out-predict α and ε). Geometry earns survival by
intervention + prediction or it dies like the rest.

## MP0b-11 — GEOMETRY hypothesis: the predictive version is DEAD
(Harness optimized first: vectorized `make_batch` [byte-exact, 5.5×] + `scan_chunk=64` [≈ to
sequential, 5e-6] → ~2.7× faster, GPU util 27%→79%; W/L identities preserved.)
- **11a interpolation θ=(1−α)W+αL:** the winner is *isolated* — 20% toward L already falls into the
  stuck basin (5.8); middle uniformly stuck; a non-monotone winning *pocket* at α=0.8. **No linear
  mode connectivity; winning set is non-convex** (block-swaps rescue, averaging does not). Smooth-
  landscape geometry: dead.
- **11b perturbation magnitude θ=L+ε·noise:** ε=0 → 0% rescue; ε∈[0.25,2] → ~80% (FLAT, direction-
  scattered); ε=4 → collapses. **Not monotone** → "rescue ∝ magnitude/distance" is dead. Only a
  perturbation *window* is real.
- **Surviving (qualitative only):** L sits at the edge of a large, easily-reached winning basin;
  the winner is fragile along the W→L line; the winning set is non-convex. **No scalar geometric
  quantity predicts the outcome.** Geometry-as-prediction joins the graveyard; geometry-as-
  description is restatement.

## THE GRAVEYARD (explanations killed by intervention) vs THE SURVIVORS
KILLED: b=6.54 (A3) · weight decay · learning rate · decay-bias correlate · to_q localization ·
winning-block transfer · "fresh 50% lottery" · smooth-landscape geometry · magnitude/distance law.
SURVIVING (phenomena, repeatedly confirmed): **bimodality · metastability · heavy-tailed jump
times · initialization dominance (66%) · delayed transitions · two-basin structure robust under
perturbation · a non-convex winning set with a large basin and a shallow metastable stuck basin.**
Eleven experiments; every *explanation* died; the *phenomenon* (and its qualitative structure) did
not. Explanations are cheap; phenomena are expensive. Protect the mystery.

## Open (the one geometry test not yet run)
**Mode connectivity via curved paths:** linear interpolation found no low-loss connection W→L, but a
*nonlinear* low-loss path might exist (Garipov-style curve finding). And: can ANY learned weight-
space metric out-predict the coordinate attempts (all of which failed)? These are the last ways
geometry could earn predictive survival; absent that, the standing result is the characterized
mystery itself.

## MP0b-12/13 — THE DEFLATIONARY ATTACK: the phenomenon SURVIVED
**12 (scale, K=2N, 30k steps):** N=8 clean bimodal (6/12 jumped); N=16/32 → 0/12 jumped → a single
pre-jump cluster (the split *appears* to vanish). Confounded: jump time may simply exceed the budget.
**13 (disentangle, N=16, 150k steps):** late jumps **RETURN** — 4/12 reach the high basin (up to 22.3),
with recorded jumps at **steps 106k and 145k** after >100k stuck; bimodality re-emerges (cv 0.22→0.44).

**CONCLUSION — survived the hardest attack:** the metastable two-basin transition is **real, not a
toy artifact.** The metastable **jump time scales steeply with N** (median ~13k @N=8 → ~100k+ @N=16,
≈8–10× for 2× N), so the bimodal split *vanishes at fixed budget and returns with enough steps.*
**This closes the loop:** the steep jump-time-vs-N scaling **explains the original A3 irreproducibility**
— ICP0 cells never converged because their jump times exceeded any feasible budget. The phenomenon
that killed `b=6.54` and the phenomenon MP0b chased are the **same thing**.

## STANDING RECORD — what survived 13 experiments
**Dead (killed by intervention):** b=6.54 · weight decay · learning rate · decay-bias correlate ·
to_q localization · winning-block transfer · "fresh 50% lottery" · smooth/predictive geometry ·
"toy-scale artifact" (the deflationary null).
**Survived (and earned seriousness via the scale attack):** a **metastable, initialization-determined,
bimodal delayed transition** in associative-recall training, with heavy-tailed jump times whose scale
grows steeply with model size; the winning set is non-convex; the cause is not any hyperparameter,
coordinate, block, weight statistic, distance, or magnitude. **Real, characterized, unexplained.**

## Hardware limit + open directions (for stronger compute)
This 4 GB laptop GPU caps clean runs at ~N≤16 (N=64 OOMs; N=32 needs budgets we can't afford).
Open: (a) the jump-time scaling LAW (median jump vs N — linear? super-linear? exponential?);
(b) sudden-vs-gradual confirmation of the transition (trajectory-logged jumpers); (c) N≥32 survival;
(d) whether the same phenomenon appears in other recurrences/attention. All require more compute.

## Where the investigation stands (open)
Bimodal recall convergence is a real, **initialisation-determined, metastable delayed transition**,
insensitive to weight decay and LR, invisible in early dynamics, resolved only by training time —
but the **specific init coordinate that sets the basin is not yet identified** (it is not the decay
bias, nor any coarse weight statistic tried). Next probes (open): a classifier on the full init
weights; perturbation/ablation of init sub-blocks; the joint q–decay geometry; jump-time-vs-init
mapping. No architectural change. (Trajectories preserved: mp0b_30k/100k/wd_sweep/lr_sweep/grid2d/
init_predictor.)

## Synthesis (the MP0b story)
Bimodal recall convergence is an **initialisation-determined, metastable delayed transition**:
two outcomes (full-recall plateau vs partial-recall stuck), set primarily by the initialisation
(which fixes a broad, heavy-tailed distribution of jump times), crossed stochastically during
training, **insensitive to weight decay and learning rate**, **invisible in early accuracy**, and
only resolved by sheer training time. Not classic (weight-decay) grokking; a distinct phenomenon.
(Trajectories preserved: `mp0b_30k/100k/wd_sweep/lr_sweep/grid2d`.)

## Open questions (later MP0b experiments — vary one knob, multi-seed)
critical LR (phase-transition threshold?) · init-only vs data-only seed (lottery-ticket?) ·
weight decay · optimizer (Adam vs SGD/momentum) · cosine schedule · the bifurcation step
distribution. **No architectural change (no delta-rule) until the phenomenon is understood.**

## MP0b-14 — THE LAST EXPERIMENT: attack the word "jump." It does NOT survive.
Same N=16/d_model=64/K=32/150k setting as MP0b-13, but **dense trajectory logging**
(`log_every=50` → 3000 acc samples/seed) to resolve the *morphology* of the late transition.
World A = true discontinuous jump · B = continuous ramp · C = staircase · D = other.

**The two seeds that crossed 0.6 did so by SLOW MONOTONE RAMPS, not jumps:**
```
step(k):     1    5   10   20   30   45   60   75   90  105  120  135  150
s1 fk=22.3: .13  .25  .27  .36  .44  .53  .55  .54  .56  .59  .67  .68  .70   (0.40->0.60 took 84,050 steps)
s2 fk=19.0: .13  .19  .25  .27  .31  .33  .40  .40  .41  .47  .50  .54  .59   (0.40->0.60 took 90,200 steps)
stuck s7  : .10  .19  .23  .24  .24  .24  .23  .24  .23  .24  .23  .23  .23   (flat after ~5k forever)
```
- **No discontinuity anywhere.** Max acc gain in any 1000-step window = **0.10**, and that maximum
  is the *initial* easy-structure rise at ~step 1k — **not** the late "jump." Max gain in any single
  50-step eval interval = **0.03**. One downward jolt >0.03 per jumper (eval-batch noise; n=2 batches).
  Ramps are monotone within noise. **Verdict: World B (continuous ramp).** Not A, not C.
- **"Jump" was a measurement artifact.** The MP0b-13 "jumps at 106k/145k" were just where a slow
  monotone ramp happened to cross the 0.6 threshold (`_jump` records the first step acc≥0.6). s1 was
  already at acc 0.55 by step **60k** — 45k steps *before* its recorded "jump." The heavy-tailed
  "jump-time distribution" (MP0b-2/13) is really the distribution of *when a slow ramp crosses
  threshold*, governed by the ramp's slope — not by any sudden event.
- **Collateral damage to "metastability."** Strict metastability implies a barrier and a *sudden*
  hop between basins. There is no hop. The honest picture is an **init-gated continuous capacity
  ramp**: everyone learns the easy structure fast (acc ~0.24 by 5k); only some inits then continue
  a slow monotone climb, and the climb *rate* (including rate≈0 = flat-stuck) appears set at init.
  "Barrier-kicking" (MP0b-8/9) survives only as metaphor; nothing in the trajectory is discontinuous.
- **Honest complication to "bimodal."** At N=16/150k the finals are
  `7.3 7.6 7.7 7.7 7.8 7.8 8.0 | 10.7 11.2 11.6 | 19.0 22.3` — closer to a *spread of ramp rates*
  (flat-stuck ×7, partial ×3, high ×2) than a clean two-basin split. Clean bimodality was a small-N
  (N=8) feature; at N=16 it relaxes toward a continuum. Do not over-claim "two basins" at scale.
- **What still survives:** outcome variation is init-determined; the easy structure is learned fast
  by all seeds; only some inits sustain a slow capacity ramp; the mechanism that sets the ramp rate
  remains unidentified. (Caveat: "ramp" is a *description of shape*, not a mechanism — it explains
  nothing; it only corrects the false "jump"/"sudden" reading.)

## FINAL GRAVEYARD vs FINAL SURVIVORS (14 experiments)
**KILLED by intervention/measurement:** `b=6.54` (A3) · weight decay · learning rate · decay-bias
correlate · to_q localization · winning-block transfer · "fresh 50% lottery" · smooth/predictive
geometry · magnitude/distance law · "toy-scale artifact" (deflationary null) · **the word "jump" /
discontinuous transition (MP0b-14)** · and, downgraded: strict "metastable barrier-hop" and clean
"two-basin" at N≥16.
**SURVIVED (phenomena, repeatedly confirmed):** outcome variation in associative-recall training is
**initialisation-determined**; all seeds learn the easy structure fast (~0.24 acc by 5k); a subset
then undergo a **slow, monotone, continuous capacity ramp** whose completion time is heavy-tailed and
**grows steeply with N** (this is what produced the original A3 irreproducibility); the ramp rate
(including ≈0) is fixed largely at init by a coordinate that is **not** any hyperparameter, weight
statistic, block, distance, or magnitude tried. **Real, characterized, unexplained — and now correctly
described as a ramp, not a jump.**

---

# CONSOLIDATED SYNTHESIS — the MP0b investigation (ICP0 → 14 experiments)

**Origin.** ICP0 calibrated the recall instrument and produced a clean single-seed constant `b≈6.54`.
Measurement Axiom **A3** (3-seed agreement) refused it as irreproducible. That refusal — *the
instrument declining to publish a seductive number* — opened the investigation. The thing that broke
A3 became the object of study.

**Method (held to the end).** Count wounds, not beliefs. Every explanation was treated as a murder
target; only intervention (not correlation) counted as evidence; nulls were recorded as honestly as
hits; no fix was attempted before the phenomenon was understood. Fourteen experiments, eleven dead
explanations, one surviving (and now sharpened) phenomenon.

**The arc.**
1. *Distribution & metastability (1–2):* bimodal finals at 30k; extending to 100k, late risers appear →
   first read as "grokking-like delayed jumps."
2. *Hyperparameter attacks (3–4):* weight decay and LR do **not** control it → not classic grokking,
   not simple thermal escape.
3. *Init vs data (5–7):* outcome is **66% init-determined**; a seductive init correlate (decay bias)
   is **causally refuted** by forcing it.
4. *Localization attacks (8–10):* the "winning ticket" is **not** localized, **does not transfer**,
   and a random block scramble rescues as well as the real winner → "ticket transfer" dead.
5. *Geometry attack (11):* no linear mode connectivity; the winning set is non-convex; **no scalar
   geometric quantity** (interpolation α, perturbation magnitude ε) predicts the outcome → geometry-
   as-prediction dead.
6. *Deflationary scale attack (12–13):* the split appears to vanish at N=16/32 (fixed budget) but
   **returns** at N=16 with 150k steps → the phenomenon is **real, not a toy artifact**; its
   timescale **scales steeply with N**, which **explains the original A3 failure** (ICP0 cells never
   reached their transition within budget). The A3-killer and the MP0b phenomenon are the *same thing*.
7. *Morphology attack (14, this experiment):* the transition is a **continuous monotone ramp, not a
   jump**; "jump"/"sudden"/"barrier-hop"/"clean two-basin" are corrected; the underlying init-gated,
   N-scaling, unexplained ramp survives.

**Standing statement (what CDE measured).** In tiny gated-linear-attention recall training, final
recall capacity is set largely at initialisation, which fixes the *rate* of a slow monotone capacity
ramp (rate≈0 ⇒ stuck). The ramp's completion time is heavy-tailed and grows steeply with state size
N, so at any fixed compute budget some inits look "stuck," producing the irreproducibility A3 caught.
The controlling init coordinate is unidentified and is **not** any hyperparameter, weight statistic,
parameter block, weight-space distance, or perturbation magnitude tested. There is **no constant `b`**
to report from this cell — the cell does not converge reproducibly, and the instrument correctly
declines to manufacture one.

**Why this is a success (not a failure).** CDE was built to *refuse to over-interpret*. It refused at
every level: overflow → undertraining → censoring → nondeterminism → single-seed success (A3) →
decay-bias correlate (MP0b-7) → ticket transfer (8–10) → geometry (11) → toy-artifact null (12–13) →
the word "jump" (14). A measurement instrument that declines a seductive `b`, then spends fourteen
experiments killing its own favorite explanations and reporting the survivors as *characterized but
unexplained*, is behaving exactly as a scientific instrument should. **Explanations are cheap;
phenomena are expensive. The mystery was protected, not the stories about it.**

**Hardware boundary.** 4 GB laptop GPU caps clean runs at N≤16 (N=64 OOMs). The two open quantitative
questions — the **ramp-rate-vs-N scaling law** and whether the same ramp appears in other recurrences/
attention — require more compute and are deferred. No architectural change (no delta-rule) was made;
the investigation rests here, as planned.

---

# THE INVESTIGATION REOPENED — autonomous "next attack" directive

The post-MP0b-14 rest was conditional. A new directive ("you are the next attack; truth over
continuity; kill any surviving explanation by intervention") reopened the case to test the **one
load-bearing survivor** — *"outcome is initialization-determined"* — at the scale where it had never
actually been tested by intervention.

## MP0b-15 — init×data decoupling at N=16/150k: "init-determined" SURVIVES as leading-order, but the STRONG form is KILLED

**Why this was the highest-information experiment.** "Initialization-determined" rested entirely on
MP0b-5 (a 4×4 init×data grid at **N=8/40k**) — a scale MP0b-14 itself flagged as unrepresentative. At
N=16 (the scale MP0b-12/13 validated as *real*), `run_scale` left **init==data coupled**, so the 5
ramped seeds were confounded: not one could be attributed to init vs. data. MP0b-15 breaks the
coupling — a 4×4 grid, `init_seed ∈ {0,1,2,3}` × `data_seed ∈ {0,1,2,3}`, same N=16/K=32/d64/150k
cell, reusing `run_grid2d` + `_analyze_grid2d`. **Determinism gate passed exactly** (the 4 diagonal
cells i==d reproduced the scale run: 7.69 / 22.34 / 19.05 / 11.58) — so the grid is bit-comparable to
all prior results and, crucially, **has a zero noise floor** (same init×data ⇒ identical result).

**Final `k_correct` table (rows=init, cols=data):**
```
        d0    d1    d2    d3    row mean
 i0     7.7   7.8   9.4   9.0    8.5   DEAD ticket  — stuck under ALL data
 i1    27.1  22.3  21.8  25.6   24.2  WINNING ticket — high under ALL data
 i2    29.4  23.9  19.0  11.5   21.0  high, but data d3 BREAKS it (->11.5)
 i3    12.1  11.5  29.4  11.6   16.2  stuck, but data d2 RESCUES it (->29.4, acc 0.92)
                                  (col means: 19.1 16.4 19.9 14.4)
```
**Variance decomposition:** init **0.567**, data **0.077**, interaction/residual **0.356**. Because the
run is deterministic, within-cell noise = 0, so that **0.356 is PURE, reproducible init×data
interaction** (the second-largest term, 4.6× the data main effect) — not measurement error.

**Verdict — a partial kill, stated honestly:**
- **SURVIVES (leading order):** init is the single largest determinant of the final basin (57% of
  variance). Two inits are *data-robust*: `i0` is dead under every data stream, `i1` is alive under
  every data stream. For these, fate is effectively fixed at init → the coordinate hunt remains
  justified *for the data-robust inits*.
- **KILLED (strong form):** the claim "outcome is fixed at init; data is a minor ~13% nuisance"
  (the MP0b-5/N=8 reading) is **dead**. A 36% noise-free interaction with **two-way basin crossings**:
  data stream d2 *rescues* the stuck init i3 (7→29, acc 0.92); data stream d3 *breaks* the strong init
  i2 (29→11.5). Init sets a **propensity, not a fate**.

**The decisive timing (free trajectory analysis, no extra compute).** For the borderline inits the
basin is **selected mid-training, not at init.** `i3_d2` tracks its stuck row-mates until ~10–20k,
then diverges sharply over **steps 20k→50k** (acc 0.33→0.62→0.81) while i3_d0/d1/d3 stay flat at ~0.36
forever; `i2_d0` lifts off at 20k–30k while `i2_d3` never does. So:
- **WOUNDS the MP0b-free-analysis claim "the die is cast at INIT, hidden in early dynamics":** true
  only for the *data-robust* inits (i0, i1). For *borderline* inits the die is cast **mid-training
  (~20–50k), by the data stream**, and is visible as a clean divergence — not latent at step 0.
- **WOUNDS the MP0b-14 sub-claim "the climb *rate* is set at init":** within the winning init i1, the
  cross-0.6 time spans **42.9k / 105.4k / 120.7k / 48.9k** purely from the data seed (≈3× spread). Init
  gates basin *reachability*; **data sets the timing/rate** and, for borderline inits, the basin itself.

**Refined three-tier picture (what now stands):**
| Init class | Behaviour across data seeds | Decider |
|---|---|---|
| `i0` (dead), `i1` (alive) | data-robust (same basin every time) | **init** |
| `i2`, `i3` | data-decidable (basin flips by data) | **data, mid-training ~20–50k** |

**Bearing on L1 / `b`.** Effective `K*` at this cell spans **~7.7 → ~29.4** (a ~3.8× range, acc up to
0.92) and is **flippable by the data stream** for borderline inits. So effective `b` is neither
seed-stable *nor* purely init-stable here — fixing the init does **not** fix `b` for borderline inits.
This reinforces (and deepens) the standing result: **no reproducible `b` from this cell**, now wounded
from a second direction (the init×data interaction), not just from init-variability.

**Casualties (MP0b-15):** strong-form "init fixes the outcome, data is a 13% nuisance" · "ramp rate is
set at init" (MP0b-14) · "die cast at init, hidden early" (now: true only for data-robust inits).
**Survivors:** init is the largest single factor (57%) for the *final basin*; all seeds learn the easy
structure fast (~0.24 by 10k); the transition is a slow continuous ramp once a basin is selected
(MP0b-14). **New object:** a large, reproducible, **noise-free init×data interaction** — borderline
inits have their basin **selected mid-training by the data stream**, in a ~20–50k window.

## MP0b-16 — critical-window data-switch: the basin is committed EARLY (<10k) and IRREVERSIBLY

**The attack.** MP0b-15 left the timing ambiguous ("selected ~20-50k"). MP0b-16 intervenes: take the
borderline init i3 (stuck under d0, rescued by d2) and **switch its training data stream** between the
two worlds at step T (opt-in `data_switch` hook in `train.py`; eval set fixed). Two arms:
- **rm** (start rescuing d2 -> switch to stuck d0 at T): the crossover marks the rescue **lock-in** step.
- **add** (start stuck d0 -> switch to rescuing d2 at T): the crossover marks the **window-close** step.
T in {10,20,30,50,75}k, 150k total. **Determinism gate PASSED** - the two no-switch controls reproduce
the grid2d i3 cells *exactly* (ctrl_rescue 29.42 = i3xd2; ctrl_stuck 12.12 = i3xd0).

**Result - both arms are flat; they agree:**
```
rm  (d2 -> d0 at T):   T=10k 29.7 | 20k 29.8 | 30k 29.3 | 50k 29.2 | 75k 29.2   ALL RESCUED (~ctrl 29.4)
add (d0 -> d2 at T):   T=10k  9.6 | 20k 11.8 | 30k 11.9 | 50k 12.0 | 75k 11.8   ALL STUCK   (~ctrl 12.1)
```
- **rm:** removing the rescuing stream changes nothing - even at **T=10k, where acc@switch was still
  0.28** (flat, indistinguishable from stuck). The first <10k steps of d2 **lock in** the rescue.
- **add:** adding the rescuing stream never rescues - even add_T10k, which then trains on d2 for the
  remaining **140k steps**, stays stuck (9.6). This **refutes the remaining-steps caveat**: it is genuine
  **irreversibility**, not insufficient time. The first <10k steps of d0 lock in the stuck basin, and
  140k steps of the rescuing data cannot escape it within budget.

**Verdict - an early, data-driven, irreversible critical period.** Whichever data world is present in
the **first <10k steps** determines the basin, and the commitment is **irreversible within the 150k
budget**. The basin is set long *before* it is visible: i3xd2 commits by <10k but its accuracy does not
begin rising until ~20-30k (MP0b-15). The visible ramp is a **lagging readout of a commitment already made**.

**Casualties (MP0b-16):**
- MP0b-15's "data selects the basin **mid-training (~20-50k)**" - **CORRECTED**: ~20-50k is only when the
  already-made choice becomes *visible*; the actual selection is **<10k and latent**.
- the "fewer-remaining-steps" caveat on the add arm - **refuted** (140k of d2 still fails).
- weakens any "metastable late hop" residue further: there is no late hop and no late decision.

**Survivors / sharpened object:** a **critical-period** mechanism - for a borderline init, the basin
(hence effective K* and effective `b`) is **committed in the first <10k training steps by the data
stream, then frozen**. (Caveat: "irreversible" = not reversed within 150k; at N=8 stuck seeds eventually
moved, so the escape time, if finite, exceeds this budget.) **For L1/`b`:** the basin is a critical-period
quantity fixed by the first <10k steps of the init x data trajectory - not an asymptotic constant.

## MP0b-17 — finer early sweep: the critical period is PINNED to ~steps 1000-2000

MP0b-16 was flat for T>=10k, so the lock-in is earlier; this sweeps T in {0.5,1,2,4,8}k in both arms
(init i3, d2<->d0, 150k; determinism gate PASS - controls reproduce grid2d i3 exactly).
```
rm  (d2 -> d0 at T):   0.5k 12.9 stuck | 1k  9.5 stuck | 2k 29.2 RESC | 4k 27.5 RESC | 8k 29.2 RESC
add (d0 -> d2 at T):   0.5k 28.5 RESC  | 1k 26.4 RESC  | 2k 12.0 stuck| 4k  9.0 stuck| 8k  8.9 stuck
```
**Both arms cross-validate the same sharp boundary between step 1k and 2k:**
- **rm:** <=1k of d2 is insufficient (reverts to stuck); >=2k of d2 locks in the rescue.
- **add:** switching to d2 by <=1k still rescues; >=2k of d0 first locks in stuck.
- Approached from below the rescue weakens monotonically toward the boundary (add 0.5k->28.5,
  1k->26.4, 2k->stuck): a sharp but continuous transition zone, not a single-step cliff.

**The commitment is invisible when it happens.** At step 2k, acc@switch is **0.17-0.18 for BOTH** the
cell that will rescue (->0.91) and the one that stays stuck (->0.37); indistinguishable at the moment
their fates diverge, and so until the visible ramp at ~20-30k.

**Verdict - a hidden critical period.** For borderline init i3 the final capacity basin is selected by
the data stream within a sharp window at **~steps 1000-2000**, latent ~20k steps before it shows, and
irreversible thereafter (MP0b-16). A *pinned, two-sided cross-validated* mechanism. **Next attacks:**
(1) the window sits just after LR warmup ends (600 steps) - vary warmup; (2) replicate on another
borderline init (i2xd3) to test i3-specificity.

## MP0b-18 — generality: the critical period is GENERAL, its timing is INIT-SPECIFIC

Replicate the fine sweep on a DIFFERENT borderline init: i2 (a winner under d0=29.4, broken by d3=11.5).
init=2, rescue=d0, stuck=d3, T in {0.5,1,2,4,8}k, 150k. Determinism gate PASS (controls reproduce
grid2d i2 exactly: 29.36 / 11.53).
```
i2 rm  (d0 -> d3 at T):  0.5k 29.2 RESC | 1k 24.1 RESC | 2k 28.4 RESC | 4k 29.4 RESC | 8k 28.9 RESC
i2 add (d3 -> d0 at T):  0.5k 23.4 RESC | 1k 11.6 stuck| 2k 11.8 stuck| 4k  9.3 stuck| 8k 11.5 stuck
```
- **Same phenomenon as i3:** a sharp, two-sided, **latent** (at the boundary acc@switch ~0.12 for BOTH
  the future-rescued and future-stuck cell), irreversible early critical period.
- **Different timing:** i2's boundary is **~0.5-1k** (rm locks in <0.5k; add closes between 0.5k and 1k),
  vs i3's **~1-2k**. i2 (robust winner) commits ~2x faster than i3 (default-loser). Commitment speed
  tracks how "reachable" the basin is for that init.

**Verdict - generality CONFIRMED (2/2 borderline inits).** The early, latent, irreversible, data-driven
critical period is **not i3-specific**; it is a general mechanism whose **window location is init-dependent**.
**Casualty:** the "i3-specific fluke" worry - killed. **Bonus:** i2 and i3 commit at *different* steps
under *identical* warmup (600), so the window is **not set by the LR schedule alone**.

## MP0b-19 — WARMUP attack: the clean story CRACKS. The basin is schedule-dependent (an early-LR trap)

Rerun i3's fine sweep at **warmup_steps=3000** (5x the usual 600). (Determinism-vs-grid2d mismatch is
EXPECTED - warmup changes training; the controls instead define the warmup=3000 basins.)
```
                       ctrl i3xd0 (stuck stream)   ctrl i3xd2 (rescue stream)
warmup=600  (MP0b-17):        12.1  STUCK                 29.4  rescued
warmup=3000 (MP0b-19):        27.0  HIGH (!)              28.1  high
```
**The longer warmup RESCUED the default-stuck cell** (i3xd0: 12.1 -> 27.0). At warmup=3000 i3 reaches
the high basin **regardless of data stream** - the clean data-dependence that defined the critical
period **vanishes**, and the data-switch cells scatter non-monotonically (no clean window).

**Verdict - the critical period is SCHEDULE-DEPENDENT, not intrinsic.** The LR warmup length is a
**causal lever on the basin**: a gentler early-LR ramp flips "stuck" -> "rescued."

**Reframe (richer + more honest).** Combining MP0b-15..19: the **"stuck basin" is an early-training
OPTIMIZATION TRAP**, entered or avoided in the first ~1-2k steps, escapable by **either** (a) the
rescuing data stream during the critical window (MP0b-16/17/18) **or** (b) a gentler LR warmup
(MP0b-19). It is **not** a capacity/data limit. So the bimodal stuck/rescued outcome - and the original
**A3 irreproducibility** - is largely an **optimization artifact**, not a property of the data or state
size. **Casualty:** "intrinsic data-driven critical period" (downgraded: schedule-modulable). **`b`:**
the irreproducibility was the optimizer stochastically falling into an early trap; gentler schedules
may recover convergence.

## MP0b-20 — warmup sweep on i3xd0: a THRESHOLD escape (confirms the early-LR trap)

Sweep warmup on the fixed flipped cell i3xd0. Built-in determinism checks PASS (warmup=600 -> 12.12 =
grid2d i3xd0; warmup=3000 -> 26.97 = MP0b-19 ctrl) - bit-consistent across the whole investigation.
```
warmup:     300    600   1500   3000   6000
i3xd0 fk:  13.8   12.1   22.5   27.0   23.0      stuck,stuck,RESC,RESC,RESC
acc@150k:  0.43   0.38   0.70   0.84   0.72
```
**Threshold-shaped, not strictly monotone:** trapped at warmup<=600, escapes sharply by 1500, optimum
~3000, slight decline at 6000 (too-long warmup wastes early low-LR steps). Trajectories: warmup<=600
never ramps; >=1500 becomes a late-riser. **Verdict: warmup is a clean causal lever on the basin** -
a too-fast early-LR ramp traps i3xd0; a gentler ramp escapes it. Confirms the **early-high-LR
optimization trap** (threshold-shaped escape). **Capstone (MP0b-21, RUNNING):** does a gentler warmup
dissolve the WHOLE 12-seed bimodality (baseline warmup=600: 7/12 stuck) or only i3xd0?

## MP0b-21 — CAPSTONE: warmup is NOT a master switch; the bimodality is robust

Does a gentler warmup dissolve the WHOLE 12-seed bimodality, or only i3xd0 (MP0b-19/20)? Re-run the
original 12-seed coupled cell at warmup=3000 vs the warmup=600 baseline (both 150k, determinism intact).
```
seed:    s0   s1   s2   s3   s4   s5   s6   s7   s8   s9  s10  s11   | stuck(<10)
w=600:  7.7 22.3 19.0 11.6 11.2  7.7  7.6  7.3  7.8  7.8  8.0 10.7   |  7/12
w=3000: 9.9 25.2 26.6 26.4  9.9  7.9  7.7 12.4  7.7  7.8  7.8 16.1   |  7/12
                          TRAP            RESC
```
**Net effect of a 5x gentler warmup: ZERO change in stuck fraction (7/12 -> 7/12).** Per-seed: 1 rescued
(s7 7.3->12.4), 1 trapped (s4 11.2->9.9), 10 basins unchanged. Warmup *amplified* the existing basins
(winners win bigger: s3 11.6->26.4, s2 19->27) and added spread (cv 0.54), but did not move the
population across the split.

**Verdict - warmup is a per-cell nudge, not a universal lever.** The MP0b-19/20 rescue of i3xd0 was
REAL but **cell-specific**; warmup does not generally escape the stuck basin. **This KILLS the over-claim**
(my own, MP0b-19) that "the bimodality is just an LR-schedule artifact": it survives a 5x warmup change.
Symmetrically, the *data* lever is also not universal - the rescuing stream d2 flips *borderline* inits
(i2/i3) but not the robust loser i0 (i0xd2 = 9.4, still stuck; MP0b-15 grid2d). **Neither data nor
schedule is a master switch.** The bimodality is robust.

---

# THE DISCOVERY (consolidated, MP0b-15..21) — the reopened investigation

**What it is.** The bimodal "stuck vs rescued" convergence in tiny gated-linear-attention associative
recall - the phenomenon that produced the A3 irreproducibility that launched CDE - is an **early-training
basin-selection process**: the final recall-capacity basin is decided in a **sharp, latent critical
period in the first ~1-2k training steps**, then held essentially irreversibly within feasible budgets.

**The evidence chain (each step an intervention, all bit-deterministic):**
1. **Init x data, not init alone (MP0b-15).** Decoupling init and data seeds at N=16: 57% of outcome
   variance is init, but a **36% noise-free init x data interaction** flips borderline inits across
   basins by data alone. "Initialization-determined" (the prior survivor) is only the leading term.
2. **Early, latent, irreversible (MP0b-16).** Switching the data stream shows the basin is committed in
   the first <10k steps and cannot be undone by 140k later steps of the opposite stream - *before* any
   visible accuracy change (the ramp at ~20-30k is a lagging readout).
3. **Pinned to ~1-2k, cross-validated (MP0b-17).** A fine two-sided sweep pins the commitment to a sharp
   window at ~steps 1000-2000, latent (acc ~0.17 for both fates at the moment they diverge).
4. **General across inits (MP0b-18).** A second borderline init (i2) shows the same critical period at a
   different, init-specific location (~0.5-1k); commitment speed tracks basin reachability.
5. **A co-determinant, not a single cause (MP0b-19/20/21).** The LR warmup schedule is *also* a lever -
   it flips i3xd0 (12->27) with a clear threshold (MP0b-20) - but at the population level a 5x warmup
   change leaves the stuck fraction unchanged (MP0b-21). Neither data nor schedule is a master switch.

**Why it matters / connection to `b`.** The effective recall capacity K* (hence the constant `b = K* log
V / N`) at an overloaded cell is **not an asymptotic property** - it is fixed stochastically in an early
critical period, co-determined by init x data x schedule, with no single controlling coordinate (every
one tried - decay bias, weight blocks, geometry, LR, weight decay, warmup - is at most a partial lever;
MP0b-3..11, 19-21). **This is *why* CDE could not measure a reproducible `b` at this cell, and A3 was
right to refuse `b=6.54`:** the irreproducibility is a genuine early-training critical-period
stochasticity, not measurement error and not a removable artifact. The instrument's refusal was correct,
and the thing it refused is now *characterized*: a real, robust, mechanistically-located phenomenon.

**The graveyard (killed by intervention).** b=6.54 (A3) - weight decay - learning rate - decay-bias
correlate - to_q localization - winning-block transfer - "fresh 50% lottery" - smooth/predictive
geometry - magnitude/distance law - "toy-scale artifact" - the word "jump" (it's a continuous ramp) -
strict "init-determined" (it's init x data) - "intrinsic data-driven critical period" (schedule co-
determines) - **and "bimodality is just an LR-schedule artifact" (MP0b-21: survives 5x warmup).**

**The survivor (a characterized mechanism, not a mystery).** An **early-training (~1-2k step), latent,
near-irreversible, init-specific critical period that selects the recall-capacity basin, co-determined
by initialization, data stream, and optimization schedule, with no single master coordinate.** It is
the mechanism behind the original A3 irreproducibility. Explanations were cheap and all died; the
phenomenon survived 21 experiments and is now located in training time, shown general across inits, and
shown robust to (while modulable by) both data and schedule.

**Open (compute-gated; needs cloud/4090 or N>=32).** Does the critical period appear in other
recurrences/attention? Does it persist / its window scale at N>=32? Is there a schedule that *does*
collapse the bimodality (cosine, lower peak LR, longer total budget)? These extend the discovery; they
do not threaten its core, which is established on this 4 GB laptop.

## MP0b-22 — N>=32 (Kaggle T4 x2): the trap PERSISTS at N=32, and the escapees reveal K* proportional to N

First experiment past the 4 GB-laptop ceiling: scale sweep N in {16 (on-platform anchor), 32}, K=2N,
d_model=64, 6 seeds, 400k steps, canonical warmup=600, on Kaggle dual-T4 (cells round-robin'd across
both GPUs; P/T4-internal determinism self-check PASSED). Data: mp0b_results.zip (T4 platform).
```
N=16 K=32 (400k):  finals [7.5, 10.5, 12.0, 12.9, 23.1, 24.1]   BIMODAL gap_frac 0.61 stuck 3/6  jumps 88.6k,144k
N=32 K=64 (400k):  finals [8.2, 16.4, 18.3, 18.4, 35.1, 48.5]   BIMODAL gap_frac 0.41 stuck 4/6  jump 108k (1 crossed 0.6)
```
**(1) The bimodal critical-period trap PERSISTS at N=32** - a clear stuck cluster (acc ~0.13-0.29) vs
escapees reaching the high basin (35.1=acc0.55, 48.5=acc0.76). **NOT a small-N (N<=16) artifact.**
Consistent with steep jump-time-vs-N scaling: at N=32 only 1-2 cells fully escaped within 400k (vs the
transition already taking >100k at N=16), so most look pre-/partially-transitioned at this budget.

**(2) Bonus - the escapees reveal the hidden L1 measurement.** The trap-escaping WINNERS scale cleanly:
K* ~ 24 at N=16, ~48.5 at N=32 -> ratio 2.0 for 2x N -> **K* proportional to N (exponent ~1.01, LINEAR)**,
exactly L1's prediction `K* = N b / log V` with ~constant `b`. So **conditioned on trap-escapees, K*(N) is
linear and `b` is roughly scale-stable** - the clean measurement the trap normally hides. This closes the
arc back to CDE's origin: `b` was irreproducible *because most cells fall into the early-training trap*;
the escapees show L1 holding (linear, NOT super-linear => L1 survives this small test).

**Caveats (honest):** 6 seeds, only 1-2 clear escapees per N; the N=32 winner s3 (35.1, acc 0.55) is
UNDER-CONVERGED at 400k, so K*(32) >= 48.5 is a lower bound; T4 platform (internally deterministic, not
byte-comparable to laptop). Exponent ~1.0 is suggestive, not a CI-bounded measurement. **Next to harden
it:** more seeds + longer budget at N=32 (and N=48/64) to (a) get more converged escapees, (b) fit K*(N)
with a CI and read `b` + its curvature - i.e. run the actual MP0 b-measurement *on the escapee subpopulation*.
