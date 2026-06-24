"""Read-only access to research-synthesis-engine outputs.

Two sources:
  - rse.db: the `survivors` (candidate architectures) and `attacks` (destroyer
    verdicts) tables — surviving architectures carry `remaining_risks`/attacks
    that are first-class evidence for/against the laws.
  - 16_laws_of_state_compression.md: the theory source document (read as text so
    the laws' provenance is traceable; the structured seeds live in config/).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .readonly import ReadOnlyDB


class RSEReader(ReadOnlyDB):
    def __init__(self, db_path: str | Path, laws_path: str | Path | None = None):
        super().__init__(db_path)
        self.laws_path = Path(laws_path) if laws_path else None

    def _latest_run(self, table: str) -> str | None:
        if "run_id" not in self.columns(table):
            return None
        row = self.require().execute(
            f"SELECT run_id FROM {table} ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
        return row["run_id"] if row else None

    def fetch(self, table: str, limit: int | None = None, latest_only: bool = True) -> list[dict[str, Any]]:
        if not self.has_table(table):
            return []
        clauses: list[str] = []
        params: list[Any] = []
        if latest_only and "run_id" in self.columns(table):
            rid = self._latest_run(table)
            if rid:
                clauses.append("run_id=?")
                params.append(rid)
        sql = f"SELECT * FROM {table}"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        if limit:
            sql += f" LIMIT {int(limit)}"
        return self.query(sql, tuple(params))

    def survivors(self, limit: int | None = None) -> list[dict[str, Any]]:
        return self.fetch("survivors", limit)

    def attacks(self, limit: int | None = None) -> list[dict[str, Any]]:
        return self.fetch("attacks", limit)

    def counts(self) -> dict[str, int]:
        return {t: self.count(t) for t in ("survivors", "attacks", "candidate_genomes") if self.has_table(t)}

    def laws_text(self) -> str:
        if self.laws_path and self.laws_path.exists():
            return self.laws_path.read_text(encoding="utf-8")
        return ""

    @property
    def laws_available(self) -> bool:
        return bool(self.laws_path and self.laws_path.exists())
