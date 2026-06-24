# Building CDE

## Conventions (shared with the research family)

- Python ≥3.10, **Click** CLI, **PyYAML** config (+ env override), **SQLite WAL append-only** tagged
  by `run_id`, **Rich** console, per-unit resumability via `processing_status`, deterministic IDs via
  `utils.hashing.stable_hash`, `kind|key` node ids, graceful Ctrl+C.
- Own store: `data/db/cde.db`. Upstreams opened read-only (`src/db/readonly.py`).
- Entry point: `cde = "src.cli:cli"`.

## Layout

```
config/   config.yaml, theory_objects.yaml, quantities.yaml
src/db/   schema.py, connection.py, readonly.py, {corpus,pressure,director,rse}_reader.py
src/quantities/   quantity.py        # Quantity object — the heart
src/phenomena/    phenomenon.py      # observed regularities; survive theories
src/theory/       objects.py, dependency.py, attack.py, cemetery.py, obituaries/obituary.py
src/ingest/       evidence_graph.py  # deterministic, no LLM
src/lab/          models/, tasks/mqar.py, train.py, mp0.py, estimator.py
src/export/       exporters.py
src/utils/        logging.py, hashing.py
data/db | data/runs | data/exports
```

## Milestones

- **M1 (this build):** instrument core + L1 measured end-to-end (ingest → MP0 → Quantity `b` →
  attack L1 → survival table / constant atlas / cemetery).
- **M2:** more architectures (Mamba-2/RWKV/S4/Hyena/RetNet, plain-Torch, Tier-1) → fill the atlas;
  activate K6, K9.
- **M3:** more Measurement Programs (copy length, state-tracking, MI probe).
- **M4:** automatic adversary + richer obituaries (LLMs may enter here).
- **M5:** Memory Geometry Atlas.

## Testing

`pytest` — fully offline. `tests/conftest.py` builds tiny synthetic sibling DBs and a tiny MQAR set.
No network, no API key, no GPU required for the unit tests (lab tests run on CPU at toy scale).
