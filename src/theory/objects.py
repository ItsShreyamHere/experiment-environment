"""Theory Objects and their attack ledger.

A theory carries NO confidence number. Its entire state is:
    status ∈ {unknown, surviving, damaged, collapsed, undermined}
    + attack_count / failed_attacks / successful_attacks
Science here is attempted murder → survival. `unknown` means not yet attacked.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ..db.connection import Database, utcnow

# Statuses
UNKNOWN = "unknown"
SURVIVING = "surviving"
DAMAGED = "damaged"
COLLAPSED = "collapsed"
UNDERMINED = "undermined"   # a structural consequence: a dependency died

ALIVE_STATUSES = {UNKNOWN, SURVIVING, DAMAGED}
DEAD_STATUSES = {COLLAPSED, UNDERMINED}


@dataclass
class TheoryObject:
    id: str
    type: str
    statement: str
    formula: str = ""
    mode: str = "active"          # active | dormant
    priority: int = 99
    measures: list[str] = field(default_factory=list)
    falsification: str = ""
    depends_on: list[str] = field(default_factory=list)
    status: str = UNKNOWN
    attack_count: int = 0
    failed_attacks: int = 0
    successful_attacks: int = 0

    def to_row(self, run_id: str) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "mode": self.mode,
            "priority": self.priority,
            "statement": self.statement.strip(),
            "formula": self.formula,
            "measures_json": json.dumps(self.measures, ensure_ascii=False),
            "falsification": self.falsification.strip(),
            "status": self.status,
            "attack_count": self.attack_count,
            "failed_attacks": self.failed_attacks,
            "successful_attacks": self.successful_attacks,
            "run_id": run_id,
            "created_at": utcnow(),
        }


def seed_theories(db: Database, cfg_theories: dict[str, Any], run_id: str) -> list[TheoryObject]:
    """Insert theory seeds from config (idempotent). Dependencies seeded separately."""
    out: list[TheoryObject] = []
    for t in cfg_theories.get("theories", []):
        theory = TheoryObject(
            id=t["id"],
            type=t.get("type", "conjecture"),
            statement=t.get("statement", ""),
            formula=t.get("formula", ""),
            mode=t.get("mode", "active"),
            priority=int(t.get("priority", 99)),
            measures=list(t.get("measures", [])),
            falsification=t.get("falsification", ""),
            depends_on=list(t.get("depends_on", [])),
        )
        db.insert("theory_objects", theory.to_row(run_id))
        out.append(theory)
    db.commit()
    return out


def get_theory(db: Database, theory_id: str) -> dict[str, Any] | None:
    return db.one("SELECT * FROM theory_objects WHERE id=?", (theory_id,))


def set_status(db: Database, theory_id: str, status: str) -> None:
    db.conn.execute("UPDATE theory_objects SET status=? WHERE id=?", (status, theory_id))
    db.commit()


def bump_ledger(db: Database, theory_id: str, *, failed: int = 0, successful: int = 0) -> None:
    db.conn.execute(
        """
        UPDATE theory_objects
        SET attack_count = attack_count + ?,
            failed_attacks = failed_attacks + ?,
            successful_attacks = successful_attacks + ?
        WHERE id = ?
        """,
        (failed + successful, failed, successful, theory_id),
    )
    db.commit()
