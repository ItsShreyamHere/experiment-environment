# Constant Discovery Engine (CDE)

A **scientific instrument**, not a synthesis engine. CDE's sole purpose is to determine
whether the *Laws of State Compression* (crystallized by the sibling `research-synthesis-engine`)
are genuine invariants of nature or artifacts of current architectures.

It works the way physics actually developed — bottom-up from measurement:

```
Measurements → Phenomena → Quantities → Constants → Laws
```

The one missing constant is **`b`** (effective bits per state coordinate). From the crown-jewel
law **L1** (`ā(m) ≤ min(1, N·b/(m·log V))`), the conserved item count is `K* = N·b/log V`, and the
slope of `K*(N)` *measures* `b`. CDE trains tiny sequence models, measures `K*(N)`, fits `b`, and
uses that measurement to **attack** L1.

## Method: count wounds, not beliefs

A theory has no confidence number. It has an **attack ledger**:
`status ∈ {unknown, surviving, damaged, collapsed}` plus `attack_count / failed_attacks /
successful_attacks`. Science here is *attempted murder → survival*. Dead laws are discoveries:
they get an **obituary** and a place in the **Theory Cemetery**.

## v1 scope

- Validate **one law (L1)** end-to-end. `K6`, `K9` are dormant (the structural dependency graph
  `L1 → K6 → K9` exists, so L1's death propagates).
- **Real PyTorch lab**, tiny models only (single 8GB RTX 3050 Laptop GPU). Plain-PyTorch
  recurrence — no `mamba-ssm`/`causal-conv1d`/Triton.
- **No LLMs, no network, no API key.** Deterministic → 0 API cost, 0 hallucination, max reproducibility.

## Quickstart

```bash
pip install -e .            # plus a CUDA torch build (see requirements.txt)
cde init                    # create data dirs + cde.db
cde verify                  # check sibling DBs reachable RO + torch/CUDA + configs load
cde ingest                  # build the evidence graph; seed quantities/theories/dependencies
cde measure --quick         # fast end-to-end Measurement Program 0 (minutes)
cde measure                 # full MP0 grid -> Quantity b ± CI
cde attack L1               # apply the measurement verdict to L1's ledger
cde export                  # survival table, constant atlas, quantity sheet, cemetery
cde status                  # snapshot of quantities, theories, measurements
```

## Pipeline

`corpus-builder → SPE → RSE → research-director → **CDE**` (strictly read-only on all upstreams).

See `BUILDING.md` for conventions and `CLAUDE.md` for the hard rules.
