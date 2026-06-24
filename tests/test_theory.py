"""Theory seeding, the attack ledger, and structural death propagation."""

from __future__ import annotations

from src.ingest import evidence_graph
from src.theory import dependency, objects as T
from src.theory.attack import Attack, KILLED, SURVIVED, apply


def _seed(db, cfg):
    run_id = db.start_run(note="test")
    evidence_graph.build(db, cfg, run_id)
    return run_id


def test_seeds_present(db, cfg):
    _seed(db, cfg)
    l1 = T.get_theory(db, "L1")
    assert l1 is not None and l1["mode"] == "active" and l1["status"] == T.UNKNOWN
    assert db.count("quantities") >= 3
    assert db.count("theory_dependencies") >= 2  # L1->K6, K6->K9


def test_survived_attack_promotes_to_surviving(db, cfg):
    run_id = _seed(db, cfg)
    apply(db, Attack("L1", "measurement", "MP0", SURVIVED, "linear"), run_id)
    l1 = T.get_theory(db, "L1")
    assert l1["status"] == T.SURVIVING
    assert l1["failed_attacks"] == 1 and l1["attack_count"] == 1


def test_collapse_propagates_structurally(db, cfg):
    run_id = _seed(db, cfg)
    summary = apply(db, Attack("L1", "measurement", "MP0", KILLED, "superlinear"), run_id,
                    killer_detail={"cause": "super-linear K*(N)"})
    assert T.get_theory(db, "L1")["status"] == T.COLLAPSED
    # K6 depends on L1, K9 on K6 -> both undermined transitively
    assert set(summary["undermined"]) == {"K6", "K9"}
    assert T.get_theory(db, "K6")["status"] == T.UNDERMINED
    assert T.get_theory(db, "K9")["status"] == T.UNDERMINED
    # an obituary was written
    assert db.one("SELECT * FROM obituaries WHERE theory_id='L1'") is not None
