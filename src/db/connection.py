"""CDE's own append-only SQLite store (cde.db).

WAL mode, dict rows, run tracking, generic append/query helpers, and per-unit
processing status for resumability. No LLM cache (v1 is LLM-free). Cross-DB
foreign keys are not used, so foreign_keys stays off.
Mirrors research-synthesis-engine/src/db/connection.py.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .schema import INDEXES, SCHEMA


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class Database:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self._migrate()

    def _migrate(self) -> None:
        cur = self.conn.cursor()
        for ddl in SCHEMA:
            cur.execute(ddl)
        for ddl in INDEXES:
            cur.execute(ddl)
        # Additive migrations for stores created before a column existed.
        for table, col, decl in [
            ("measurements", "seed", "INTEGER"),
            ("measurements", "grad_norm", "REAL"),
            ("measurements", "grad_nonfinite", "INTEGER"),
            ("measurements", "steps_run", "INTEGER"),
        ]:
            cols = {r["name"] for r in cur.execute(f"PRAGMA table_info({table})")}
            if col not in cols:
                cur.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")
        self.conn.commit()

    # -- runs ---------------------------------------------------------------
    def start_run(self, note: str = "") -> str:
        run_id = "run_" + utcnow().replace(":", "").replace("-", "").replace(".", "")[:18]
        self.conn.execute(
            "INSERT OR REPLACE INTO runs(run_id, started_at, finished_at, note) VALUES (?,?,?,?)",
            (run_id, utcnow(), None, note),
        )
        self.conn.commit()
        return run_id

    def finish_run(self, run_id: str) -> None:
        self.conn.execute(
            "UPDATE runs SET finished_at=? WHERE run_id=?", (utcnow(), run_id)
        )
        self.conn.commit()

    def latest_run(self) -> str | None:
        row = self.conn.execute(
            "SELECT run_id FROM runs ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
        return row["run_id"] if row else None

    # -- generic writes -----------------------------------------------------
    def insert(self, table: str, row: dict[str, Any], replace: bool = True) -> None:
        cols = list(row.keys())
        placeholders = ",".join("?" for _ in cols)
        verb = "INSERT OR REPLACE" if replace else "INSERT OR IGNORE"
        col_sql = ",".join(f'"{c}"' for c in cols)
        self.conn.execute(
            f"{verb} INTO {table} ({col_sql}) VALUES ({placeholders})",
            tuple(row[c] for c in cols),
        )

    def insert_many(self, table: str, rows: Iterable[dict[str, Any]], replace: bool = True) -> int:
        n = 0
        for r in rows:
            self.insert(table, r, replace=replace)
            n += 1
        self.conn.commit()
        return n

    def commit(self) -> None:
        self.conn.commit()

    # -- generic reads ------------------------------------------------------
    def query(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        return [dict(r) for r in self.conn.execute(sql, params).fetchall()]

    def one(self, sql: str, params: tuple = ()) -> dict[str, Any] | None:
        row = self.conn.execute(sql, params).fetchone()
        return dict(row) if row else None

    def count(self, table: str, where: str = "", params: tuple = ()) -> int:
        sql = f"SELECT COUNT(*) AS n FROM {table}"
        if where:
            sql += f" WHERE {where}"
        return int(self.conn.execute(sql, params).fetchone()["n"])

    # -- processing status (resumability) -----------------------------------
    def mark(self, stage: str, unit_id: str, status: str, error: str | None = None) -> None:
        self.conn.execute(
            """
            INSERT INTO processing_status(stage, unit_id, status, attempts, last_error, updated_at)
            VALUES (?,?,?,1,?,?)
            ON CONFLICT(stage, unit_id) DO UPDATE SET
                status=excluded.status,
                attempts=processing_status.attempts+1,
                last_error=excluded.last_error,
                updated_at=excluded.updated_at
            """,
            (stage, unit_id, status, error, utcnow()),
        )
        self.conn.commit()

    def done_units(self, stage: str) -> set[str]:
        rows = self.conn.execute(
            "SELECT unit_id FROM processing_status WHERE stage=? AND status='done'", (stage,)
        ).fetchall()
        return {r["unit_id"] for r in rows}

    def reset_stage(self, stage: str) -> None:
        self.conn.execute("DELETE FROM processing_status WHERE stage=?", (stage,))
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()
