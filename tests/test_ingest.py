"""Deterministic ingestion attaches sibling rows as evidence on the right law."""

from __future__ import annotations

from src.ingest import evidence_graph


def test_evidence_attached_to_L1(db, cfg):
    run_id = db.start_run(note="test")
    stats = evidence_graph.build(db, cfg, run_id)
    # the fake failure_modes/contradictions/skeptic rows all mention recall/copying -> L1
    assert stats["evidence"] >= 3
    l1_ev = db.query("SELECT * FROM evidence WHERE target_id='L1'")
    assert len(l1_ev) >= 3
    tables = {e["source_table"] for e in l1_ev}
    assert "failure_modes" in tables
    # contradictions are recorded with contradict polarity
    assert any(e["polarity"] == "contradict" for e in l1_ev)


def test_missing_siblings_are_tolerated(db, cfg):
    run_id = db.start_run(note="test")
    # corpus_db / rse_db point at missing files; build must not raise
    stats = evidence_graph.build(db, cfg, run_id)
    assert stats["theories"] >= 1
