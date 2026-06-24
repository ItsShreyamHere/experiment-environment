"""Attacks — the unit of survival history.

An Attack is an attempted murder of a theory. Its `outcome` updates the theory's
ledger and status, and (if it kills) triggers an obituary + structural
propagation. The measurement IS the attack: MP0's verdict on K*(N) arrives here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..db.connection import Database, utcnow
from ..utils.hashing import short_hash
from . import dependency, objects as T
from .obituaries import obituary as obit

# Outcomes
SURVIVED = "survived"   # the theory withstood the attack
WOUNDED = "wounded"     # the theory is damaged but not dead
KILLED = "killed"       # the theory collapses
INVALID = "invalid"     # the attack was inconclusive (e.g. degenerate run)


@dataclass
class Attack:
    theory_id: str
    kind: str            # measurement | structural
    source: str          # e.g. MP0
    outcome: str
    detail: str = ""


def apply(db: Database, attack: Attack, run_id: str, *, killer_detail: dict[str, Any] | None = None) -> dict[str, Any]:
    """Record an attack, update the ledger/status, and handle death.

    Returns a summary dict {recorded, new_status, undermined}.
    """
    aid = "attack|" + short_hash([attack.theory_id, attack.source, attack.outcome, attack.detail, run_id])
    db.insert(
        "attacks",
        {
            "id": aid,
            "theory_id": attack.theory_id,
            "kind": attack.kind,
            "source": attack.source,
            "outcome": attack.outcome,
            "detail": attack.detail,
            "run_id": run_id,
            "created_at": utcnow(),
        },
    )

    undermined: list[str] = []
    row = T.get_theory(db, attack.theory_id) or {}
    new_status = row.get("status", T.UNKNOWN)

    if attack.outcome == SURVIVED:
        T.bump_ledger(db, attack.theory_id, failed=1)
        # surviving an attack promotes unknown -> surviving (never downgrades a wound)
        if new_status in (T.UNKNOWN, T.SURVIVING):
            new_status = T.SURVIVING
            T.set_status(db, attack.theory_id, new_status)
    elif attack.outcome == WOUNDED:
        T.bump_ledger(db, attack.theory_id, successful=1)
        new_status = T.DAMAGED
        T.set_status(db, attack.theory_id, new_status)
    elif attack.outcome == KILLED:
        T.bump_ledger(db, attack.theory_id, successful=1)
        new_status = T.COLLAPSED
        T.set_status(db, attack.theory_id, new_status)
        obit.write_obituary(db, attack.theory_id, attack.source, run_id, detail=killer_detail or {})
        undermined = dependency.propagate_collapse(db, attack.theory_id, run_id)
    # INVALID: no ledger/status change

    db.commit()
    return {"recorded": aid, "new_status": new_status, "undermined": undermined}
