"""The CLI runs fully offline: init, ingest, status, export, and a no-data attack."""

from __future__ import annotations

from pathlib import Path

import yaml
from click.testing import CliRunner

from src.cli import cli


def _write_config(tmp_path: Path) -> Path:
    cfg = {
        "sources": {
            "corpus_db": str(tmp_path / "missing_corpus.db"),
            "pressure_db": str(tmp_path / "missing_pressure.db"),
            "research_db": str(tmp_path / "missing_research.db"),
            "rse_db": str(tmp_path / "missing_rse.db"),
            "rse_laws": str(tmp_path / "missing_laws.md"),
        },
        "database": {"path": str(tmp_path / "db" / "cde.db")},
        "paths": {"exports": str(tmp_path / "exports"), "runs": str(tmp_path / "runs"),
                  "logs": str(tmp_path / "logs")},
        "upstream": {"accepted_only": True},
    }
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return p


def test_offline_pipeline(tmp_path: Path):
    cfg_path = _write_config(tmp_path)
    runner = CliRunner()

    def run(*args):
        return runner.invoke(cli, ["--config", str(cfg_path), *args])

    assert run("init").exit_code == 0
    assert run("ingest").exit_code == 0

    r = run("status")
    assert r.exit_code == 0 and "L1" in r.output

    # attack with no measurements -> invalid verdict, no crash, L1 stays unknown
    r = run("attack", "L1")
    assert r.exit_code == 0

    r = run("export")
    assert r.exit_code == 0
    assert (tmp_path / "exports" / "survival_table.md").exists()
    assert (tmp_path / "exports" / "quantities.json").exists()
