"""Read-only access to research-director's research.db (RD).

Append-only and tagged by run_id, so we read the most recent run by default.
`skeptic_reports` are RD's attacks; `experiment_plans` its proposed experiments.
Adapted from research-synthesis-engine/src/db/director_reader.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .readonly import ReadOnlyDB

OBJECT_TABLES = [
    "programs",
    "questions",
    "skeptic_reports",
    "experiment_plans",
    "counterfactuals",
    "contradiction_priorities",
    "assumption_reports",
    "clusters",
    "priority_scores",
]


class DirectorReader(ReadOnlyDB):
    def __init__(self, db_path: str | Path, accepted_only: bool = True):
        super().__init__(db_path)
        self.accepted_only = accepted_only

    def _latest_run(self, table: str) -> str | None:
        if "run_id" not in self.columns(table):
            return None
        row = self.require().execute(
            f"SELECT run_id FROM {table} ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        return row["run_id"] if row else None

    def fetch(self, table: str, limit: int | None = None, latest_only: bool = True) -> list[dict[str, Any]]:
        if not self.has_table(table):
            return []
        cols = self.columns(table)
        clauses: list[str] = []
        params: list[Any] = []
        if latest_only and "run_id" in cols:
            rid = self._latest_run(table)
            if rid:
                clauses.append("run_id=?")
                params.append(rid)
        if self.accepted_only and "status" in cols:
            clauses.append("status IN ('accepted','approved','proposed')")
        sql = f"SELECT * FROM {table}"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        if limit:
            sql += f" LIMIT {int(limit)}"
        return self.query(sql, tuple(params))

    def counts(self) -> dict[str, int]:
        return {t: self.count(t) for t in OBJECT_TABLES if self.has_table(t)}
