"""Phenomena — observed regularities that outlive theories.

A Phenomenon (e.g. "the recall cliff") is recorded directly from measurements. It
is deliberately decoupled from any Law: if L1 is later falsified, the recall cliff
it tried to explain may still be a real, recorded observation. Phenomena are the
durable empirical bedrock beneath the (mortal) theories.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ..db.connection import Database, utcnow
from ..utils.hashing import short_hash

OBSERVED = "observed"
ABSENT = "absent"


@dataclass
class Phenomenon:
    name: str
    statement: str
    status: str = OBSERVED
    support: list[str] = field(default_factory=list)  # measurement ids / notes

    @property
    def id(self) -> str:
        return "phenomenon|" + self.name.lower().replace(" ", "_")

    def to_row(self, run_id: str) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "statement": self.statement,
            "status": self.status,
            "support_json": json.dumps(self.support, ensure_ascii=False),
            "run_id": run_id,
            "created_at": utcnow(),
        }


def record(db: Database, phenomenon: Phenomenon, run_id: str) -> str:
    db.insert("phenomena", phenomenon.to_row(run_id))
    db.commit()
    return phenomenon.id
