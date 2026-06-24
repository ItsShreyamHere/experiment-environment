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
