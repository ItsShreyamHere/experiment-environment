"""Read-only SQLite base for upstream sibling stores.

Opens databases with `mode=ro` via a URI so CDE can never mutate a sibling's
artifact. Missing files/tables are surfaced as clean states rather than crashes.
Mirrors research-synthesis-engine/src/db/readonly.py.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


class ReadOnlyDB:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.available = self.path.exists()
        self.conn: sqlite3.Connection | None = None
        if self.available:
            uri = f"file:{self.path.as_posix()}?mode=ro"
            self.conn = sqlite3.connect(uri, uri=True)
            self.conn.row_factory = sqlite3.Row

    def require(self) -> sqlite3.Connection:
        if self.conn is None:
            raise FileNotFoundError(f"upstream database not found: {self.path}")
        return self.conn

    def has_table(self, name: str) -> bool:
        if self.conn is None:
            return False
        row = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone()
        return row is not None

    def columns(self, table: str) -> set[str]:
        if not self.has_table(table):
            return set()
        return {c["name"] for c in self.query(f"PRAGMA table_info({table})")}

    def query(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        return [dict(r) for r in self.require().execute(sql, params).fetchall()]

    def count(self, table: str, where: str = "", params: tuple = ()) -> int:
        if not self.has_table(table):
            return 0
        sql = f"SELECT COUNT(*) AS n FROM {table}"
        if where:
            sql += f" WHERE {where}"
        return int(self.require().execute(sql, params).fetchone()["n"])

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()
