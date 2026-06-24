"""Knowledge layer — build the Evidence Graph from sibling outputs.

Strictly deterministic: keyword/topic retrieval against each theory's term set,
no LLM. Sibling rows that mention a law's subject become signed `evidence`
attached to that theory, plus `nodes`/`edges` in the graph. Evidence here is
*context*; the actual attack on a law comes from a Measurement Program, not from
this retrieval.
"""

from __future__ import annotations

from typing import Any

from ..config import Config
from ..db.connection import Database
from ..db.corpus_reader import CorpusReader
from ..db.director_reader import DirectorReader
from ..db.pressure_reader import PressureReader
from ..db.rse_reader import RSEReader
from ..quantities.quantity import seed_quantities
from ..theory import dependency
from ..theory.objects import seed_theories
from ..utils import logging as log
from ..utils.hashing import short_hash

# Per-theory subject terms (lowercase). L1 is the active law in v1.
KEYWORDS: dict[str, list[str]] = {
    "L1": [
        "recall", "associative recall", "associative memory", "copying", "copy",
        "rate-distortion", "rate distortion", "index", "retrieval", "mqar",
        "needle", "key-value", "key value", "memorization", "kv cache",
    ],
    "K6": ["sparse", "sparsity", "access pattern", "in-context", "selective"],
    "K9": ["hybrid", "attention fraction", "ssm", "state space", "interpolat"],
}

# Tables whose presence of a law-term we treat as SUPPORTING the ceiling
# (failures/limits the law predicts), vs. CONTRADICTING (tensions/surprises).
_CONTRADICT_TABLES = {"contradictions", "anomalies"}


def _row_text(row: dict[str, Any]) -> str:
    return " ".join(str(v) for v in row.values() if isinstance(v, str)).lower()


def _matches(text: str, terms: list[str]) -> bool:
    return any(t in text for t in terms)


def _source_id(table: str, row: dict[str, Any]) -> str:
    key = row.get("id") or row.get("paper_id") or row.get("topic") or short_hash(row)
    return f"{table}:{key}"


def _attach(
    db: Database,
    theory_id: str,
    source_db: str,
    table: str,
    row: dict[str, Any],
    run_id: str,
) -> None:
    polarity = "contradict" if table in _CONTRADICT_TABLES else "support"
    sid = _source_id(table, row)
    node_id = f"{source_db}|{sid}"
    db.insert("nodes", {
        "node_id": node_id, "kind": source_db, "key": sid,
        "label": (_row_text(row)[:140] or table), "attrs_json": None, "run_id": run_id,
    })
    db.insert("edges", {
        "src": node_id, "dst": f"theory|{theory_id}", "relation": polarity,
        "weight": 1.0, "run_id": run_id,
    })
    db.insert("evidence", {
        "id": "ev|" + short_hash([theory_id, node_id, polarity]),
        "target_kind": "theory", "target_id": theory_id,
        "source_db": source_db, "source_table": table, "source_id": sid,
        "polarity": polarity, "note": _row_text(row)[:200], "run_id": run_id,
    })


def _scan(db: Database, source_db: str, table_rows: dict[str, list[dict[str, Any]]], run_id: str) -> int:
    """Attach evidence from a sibling's tables to every theory it mentions."""
    n = 0
    for table, rows in table_rows.items():
        for row in rows:
            text = _row_text(row)
            for theory_id, terms in KEYWORDS.items():
                if _matches(text, terms):
                    _attach(db, theory_id, source_db, table, row, run_id)
                    n += 1
    db.commit()
    return n


def build(db: Database, cfg: Config, run_id: str) -> dict[str, Any]:
    """Seed quantities/theories/dependencies, then build the evidence graph."""
    stats: dict[str, Any] = {}

    # 1. Seed the instrument's objects (idempotent).
    stats["quantities"] = seed_quantities(db, cfg.quantities, run_id)
    theories = seed_theories(db, cfg.theories, run_id)
    stats["theories"] = len(theories)
    stats["dependencies"] = dependency.seed_dependencies(db, cfg.theories, run_id)
    for t in theories:
        db.insert("nodes", {
            "node_id": f"theory|{t.id}", "kind": "theory", "key": t.id,
            "label": t.id, "attrs_json": None, "run_id": run_id,
        })
    db.commit()

    accepted = cfg.accepted_only
    evidence_total = 0

    # 2. SPE pressure.db
    pr = PressureReader(cfg.source_path("pressure_db"), accepted_only=accepted)
    if pr.available:
        rows = {t: pr.fetch(t, limit=5000) for t in (
            "failure_modes", "contradictions", "anomalies", "bottlenecks",
            "tradeoffs", "assumptions", "forgotten_ideas", "premature_rejections",
        )}
        evidence_total += _scan(db, "pressure", rows, run_id)
    pr.close()

    # 3. RD research.db
    rd = DirectorReader(cfg.source_path("research_db"), accepted_only=accepted)
    if rd.available:
        rows = {t: rd.fetch(t, limit=5000) for t in (
            "skeptic_reports", "questions", "programs", "experiment_plans",
        )}
        evidence_total += _scan(db, "research", rows, run_id)
    rd.close()

    # 4. RSE rse.db (survivors + attacks)
    rse = RSEReader(cfg.source_path("rse_db"), laws_path=cfg.source_path("rse_laws"))
    if rse.available:
        rows = {"survivors": rse.survivors(limit=2000), "attacks": rse.attacks(limit=5000)}
        evidence_total += _scan(db, "rse", rows, run_id)
    stats["laws_doc"] = rse.laws_available
    rse.close()

    # 5. corpus.db prior art (titles only, keyword search per theory)
    cr = CorpusReader(cfg.source_path("corpus_db"))
    if cr.available:
        papers: list[dict[str, Any]] = []
        for terms in KEYWORDS.values():
            papers.extend(cr.search_by_keywords(terms, k=20))
        evidence_total += _scan(db, "corpus", {"papers": papers}, run_id)
    cr.close()

    stats["evidence"] = evidence_total
    stats["nodes"] = db.count("nodes")
    stats["edges"] = db.count("edges")
    log.info(
        f"[ingest] theories={stats['theories']} quantities={stats['quantities']} "
        f"deps={stats['dependencies']} evidence={evidence_total} nodes={stats['nodes']}"
    )
    return stats
