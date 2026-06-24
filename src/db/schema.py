"""DDL for CDE's own append-only store (cde.db).

The instrument's spine, in tables, bottom-up:
    measurements -> quantities -> phenomena -> theory_objects (+ attack ledger).

Every produced row is tagged with a `run_id`. There are NO confidence/belief
columns anywhere (by design): a theory's state is its attack ledger. Cross-DB
references to sibling objects are plain TEXT (no cross-database foreign keys).
"""

from __future__ import annotations

SCHEMA: list[str] = [
    # -- provenance / infra -------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS runs (
        run_id      TEXT PRIMARY KEY,
        started_at  TEXT,
        finished_at TEXT,
        note        TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS processing_status (
        stage      TEXT,
        unit_id    TEXT,
        status     TEXT,              -- pending|running|done|failed|skipped
        attempts   INTEGER DEFAULT 0,
        last_error TEXT,
        updated_at TEXT,
        PRIMARY KEY (stage, unit_id)
    )
    """,
    # -- Measurements: the atomic empirical unit ---------------------------
    """
    CREATE TABLE IF NOT EXISTS measurements (
        meas_id       TEXT PRIMARY KEY,
        quantity      TEXT,           -- e.g. k_star, recall_accuracy
        arch          TEXT,           -- diag_ssm | attention
        dataset       TEXT,           -- mqar
        N             INTEGER,        -- state size (state dim per channel)
        d_state       INTEGER,        -- alias retained for clarity (== N)
        K             INTEGER,        -- number of key->value pairs (= m, recall load)
        seed          INTEGER,        -- training seed (for per-seed aggregation / CI)
        log_V         REAL,           -- bits per value (= log2 of value alphabet)
        value         REAL,           -- the measured number
        ci_low        REAL,
        ci_high       REAL,
        repeatability REAL,           -- across-seed agreement in [0,1]
        grad_norm     REAL,           -- max finite grad norm during training (axiom A5)
        grad_nonfinite INTEGER,       -- count of nonfinite-gradient steps (axiom A5)
        steps_run     INTEGER,        -- optimizer steps actually taken
        method        TEXT,
        mp_id         TEXT,           -- which Measurement Program produced it
        run_id        TEXT,
        created_at    TEXT
    )
    """,
    # -- Quantities: built from measurements; laws depend on these ----------
    """
    CREATE TABLE IF NOT EXISTS quantities (
        id                 TEXT PRIMARY KEY,   -- quantity name
        name               TEXT,
        symbol             TEXT,
        units              TEXT,
        measurement_method TEXT,
        description        TEXT,
        value              REAL,
        ci_low             REAL,
        ci_high            REAL,
        status             TEXT,               -- unmeasured | measured
        run_id             TEXT,
        created_at         TEXT
    )
    """,
    # -- Phenomena: observed regularities; outlive theories ----------------
    """
    CREATE TABLE IF NOT EXISTS phenomena (
        id           TEXT PRIMARY KEY,
        name         TEXT,
        statement    TEXT,
        status       TEXT,            -- observed | absent
        support_json TEXT,            -- supporting measurement ids / notes
        run_id       TEXT,
        created_at   TEXT
    )
    """,
    # -- Theory Objects: NO confidence; just an attack ledger --------------
    """
    CREATE TABLE IF NOT EXISTS theory_objects (
        id                 TEXT PRIMARY KEY,
        type               TEXT,        -- law | conjecture | invariant
        mode               TEXT,        -- active | dormant
        priority           INTEGER,
        statement          TEXT,
        formula            TEXT,
        measures_json      TEXT,        -- quantities this theory is stated in
        falsification      TEXT,
        status             TEXT,        -- unknown | surviving | damaged | collapsed | undermined
        attack_count       INTEGER DEFAULT 0,
        failed_attacks     INTEGER DEFAULT 0,   -- attacks the theory survived
        successful_attacks INTEGER DEFAULT 0,   -- attacks that wounded/killed it
        run_id             TEXT,
        created_at         TEXT
    )
    """,
    # -- Structural dependency graph (NOT Bayesian) ------------------------
    """
    CREATE TABLE IF NOT EXISTS theory_dependencies (
        parent_id TEXT,               -- e.g. L1
        child_id  TEXT,               -- e.g. K6 (depends on L1)
        relation  TEXT,               -- supports | specializes | unifies
        run_id    TEXT,
        PRIMARY KEY (parent_id, child_id)
    )
    """,
    # -- Attack ledger / survival history ----------------------------------
    """
    CREATE TABLE IF NOT EXISTS attacks (
        id         TEXT PRIMARY KEY,
        theory_id  TEXT,
        kind       TEXT,              -- measurement | structural | ...
        source     TEXT,              -- e.g. MP0
        outcome    TEXT,              -- survived | wounded | killed | invalid
        detail     TEXT,
        run_id     TEXT,
        created_at TEXT
    )
    """,
    # -- Obituaries: dead theories are discoveries -------------------------
    """
    CREATE TABLE IF NOT EXISTS obituaries (
        theory_id              TEXT PRIMARY KEY,
        cause_of_death         TEXT,
        killer                 TEXT,
        died_at                TEXT,
        historical_significance TEXT,
        descendants_json       TEXT,
        text                   TEXT,
        run_id                 TEXT
    )
    """,
    # -- Evidence graph (deterministic, from siblings) ---------------------
    """
    CREATE TABLE IF NOT EXISTS nodes (
        node_id    TEXT PRIMARY KEY,  -- kind|key
        kind       TEXT,
        key        TEXT,
        label      TEXT,
        attrs_json TEXT,
        run_id     TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS edges (
        src      TEXT,
        dst      TEXT,
        relation TEXT,                -- supports | contradicts | tests | derived_from | attacks
        weight   REAL DEFAULT 1.0,
        run_id   TEXT,
        PRIMARY KEY (src, dst, relation)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS evidence (
        id           TEXT PRIMARY KEY,
        target_kind  TEXT,            -- theory | phenomenon
        target_id    TEXT,
        source_db    TEXT,            -- pressure | research | rse | corpus
        source_table TEXT,
        source_id    TEXT,
        polarity     TEXT,            -- support | contradict
        note         TEXT,
        run_id       TEXT
    )
    """,
]

INDEXES: list[str] = [
    "CREATE INDEX IF NOT EXISTS idx_meas_quantity ON measurements(quantity, arch)",
    "CREATE INDEX IF NOT EXISTS idx_meas_mp ON measurements(mp_id)",
    "CREATE INDEX IF NOT EXISTS idx_attacks_theory ON attacks(theory_id)",
    "CREATE INDEX IF NOT EXISTS idx_evidence_target ON evidence(target_kind, target_id)",
    "CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src)",
]
