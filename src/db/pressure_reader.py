"""Read-only access to SPE's pressure.db (scientific-knowledge-extractor).

Exposes the idea-level object tables. Object rows usually carry a `status`
column; by default we only return 'accepted' rows. Tables without a status
column (deterministic signals) are returned whole.
Adapted from research-synthesis-engine/src/db/pressure_reader.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .readonly import ReadOnlyDB

OBJECT_TABLES = [
    "pressure_zones",
    "stress_tensor",
    "contradictions",
    "assumptions",
    "failure_modes",
    "anomalies",
    "tradeoffs",
    "bottlenecks",
    "forgotten_ideas",
    "rare_ideas",
    "lineages",
    "premature_rejections",
    "shadow_importance",
    "long_tail_candidates",
    "rediscovery_signals",
    "atomic_claims",
]

_NO_STATUS = {"pressure_zones", "stress_tensor", "rediscovery_signals", "shadow_importance"}


class PressureReader(ReadOnlyDB):
    def __init__(self, db_path: str | Path, accepted_only: bool = True):
        super().__init__(db_path)
        self.accepted_only = accepted_only

    def fetch(self, table: str, limit: int | None = None) -> list[dict[str, Any]]:
        if not self.has_table(table):
            return []
        sql = f"SELECT * FROM {table}"
        if self.accepted_only and table not in _NO_STATUS and "status" in self.columns(table):
            sql += " WHERE status='accepted'"
        if limit:
            sql += f" LIMIT {int(limit)}"
        return self.query(sql)

    def counts(self) -> dict[str, int]:
        return {t: self.count(t) for t in OBJECT_TABLES if self.has_table(t)}
