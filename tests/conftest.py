"""Offline test fixtures: tiny synthetic sibling DBs + a temp CDE config.

No network, no API key, no GPU. The real theory_objects.yaml / quantities.yaml
sidecars are reused (read from config/), only config.yaml is overridden so the
fakes and temp paths are used.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from src.config import Config, load_config
from src.db.connection import Database


def _make_pressure(path: Path) -> None:
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE TABLE failure_modes (id INTEGER PRIMARY KEY, paper_id TEXT, statement TEXT, "
        "category TEXT, status TEXT)"
    )
    conn.executemany(
        "INSERT INTO failure_modes (paper_id, statement, category, status) VALUES (?,?,?,?)",
        [
            ("arxiv:1", "SSMs fail at verbatim copying beyond a length set by state size", "memory", "accepted"),
            ("arxiv:2", "associative recall degrades as the number of stored pairs grows", "memory", "accepted"),
            ("arxiv:3", "training instability unrelated to memory", "optimization", "accepted"),
        ],
    )
    conn.execute(
        "CREATE TABLE contradictions (id INTEGER PRIMARY KEY, topic TEXT, position_a TEXT, "
        "position_b TEXT, status TEXT)"
    )
    conn.execute(
        "INSERT INTO contradictions (topic, position_a, position_b, status) VALUES (?,?,?,?)",
        ("recall vs throughput", "SSMs match attention on recall", "attention strictly beats SSMs at recall", "accepted"),
    )
    conn.commit()
    conn.close()


def _make_research(path: Path) -> None:
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE TABLE skeptic_reports (id INTEGER PRIMARY KEY, program_id TEXT, attack_type TEXT, "
        "evidence TEXT, status TEXT, run_id TEXT, created_at TEXT)"
    )
    conn.execute(
        "INSERT INTO skeptic_reports (program_id, attack_type, evidence, status, run_id, created_at) "
        "VALUES (?,?,?,?,?,?)",
        ("p1", "capacity", "the recall ceiling is a hard capacity bound", "accepted", "run_x", "2026-01-01"),
    )
    conn.commit()
    conn.close()


@pytest.fixture
def fakes(tmp_path: Path) -> dict[str, Path]:
    pressure = tmp_path / "pressure.db"
    research = tmp_path / "research.db"
    _make_pressure(pressure)
    _make_research(research)
    return {"pressure": pressure, "research": research}


@pytest.fixture
def cfg(tmp_path: Path, fakes: dict[str, Path]) -> Config:
    base = load_config()  # reuse real theory/quantity sidecars
    raw = {
        "sources": {
            "corpus_db": str(tmp_path / "missing_corpus.db"),
            "pressure_db": str(fakes["pressure"]),
            "research_db": str(fakes["research"]),
            "rse_db": str(tmp_path / "missing_rse.db"),
            "rse_laws": str(tmp_path / "missing_laws.md"),
        },
        "database": {"path": str(tmp_path / "cde.db")},
        "paths": {"exports": str(tmp_path / "exports")},
        "upstream": {"accepted_only": True},
        "estimator": {"superlinear_threshold": 1.15, "sublinear_threshold": 0.85, "bootstrap_samples": 200},
    }
    return Config(raw=raw, theories=base.theories, quantities=base.quantities)


@pytest.fixture
def db(cfg: Config) -> Database:
    return Database(cfg.db_path)
