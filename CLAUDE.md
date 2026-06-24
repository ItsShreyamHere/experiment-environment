# CDE — hard rules

1. **CDE is an instrument, not a synthesis engine.** The spine is
   `Measurements → Phenomena → Quantities → Constants → Laws`. Never invert it. Laws depend on
   quantities; quantities depend on measurements.
2. **Count wounds, not beliefs.** No Bayesian confidence anywhere. A theory's state is an attack
   ledger (status + attack/failed/successful counts + survival history).
3. **Strictly downstream.** All sibling DBs are opened `mode=ro`. Never mutate an upstream artifact.
4. **No LLMs in v1.** No network, no API key. Evidence graph = deterministic retrieval; estimator =
   statistics; obituaries = templates. (LLMs may arrive in a later milestone, never before.)
5. **Append-only.** Re-running appends by `run_id`; never mutate prior output. Resume via
   `processing_status`.
6. **Determinism.** Fixed seeds, fixed data order. Report exponents + CI, not single accuracy
   numbers. Same inputs ⇒ same measurements.
7. **Tiny models only.** Single 8GB laptop GPU. Plain-PyTorch recurrence; no `mamba-ssm` /
   `causal-conv1d` / Triton. seq ≤ 1024, batch ≤ 64, AMP on.
8. **Dead laws are discoveries.** Obituaries and the cemetery are first-class outputs, as valuable
   as survivors.
9. **Phenomena outlive theories.** Record observed regularities (e.g. the recall cliff) separately;
   they persist even if the law that named them dies.
