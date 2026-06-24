"""CDE command line — the instrument's control panel.

    cde init      create data dirs + cde.db
    cde verify    check sibling DBs (read-only), torch/CUDA, configs
    cde ingest    build the evidence graph; seed quantities/theories/dependencies
    cde measure   run Instrument Calibration Program 0 (ICP0) -> Quantity b (+ phenomena)
    cde attack    apply the measured verdict to a law's attack ledger
    cde status    snapshot of quantities, theories, measurements
    cde export    survival table, constant atlas, quantity sheet, cemetery

v1 is offline and deterministic: no network, no API key, no LLM.
"""

from __future__ import annotations

from pathlib import Path

import click

from .config import load_config
from .db.connection import Database
from .utils import logging as log


class App:
    def __init__(self, config_path: str | None):
        self.cfg = load_config(config_path)
        self._db: Database | None = None

    @property
    def db(self) -> Database:
        if self._db is None:
            self._db = Database(self.cfg.db_path)
        return self._db


pass_app = click.make_pass_decorator(App)


@click.group()
@click.option("--config", "config_path", default=None, help="Path to config.yaml")
@click.pass_context
def cli(ctx: click.Context, config_path: str | None) -> None:
    ctx.obj = App(config_path)


@cli.command()
@pass_app
def init(app: App) -> None:
    """Create data directories and initialise cde.db."""
    for name in ("exports", "runs", "logs"):
        try:
            app.cfg.data_path(name).mkdir(parents=True, exist_ok=True)
        except KeyError:
            pass
    _ = app.db  # triggers schema migration
    log.success(f"[init] cde.db ready at {app.cfg.db_path}")


@cli.command()
@pass_app
def verify(app: App) -> None:
    """Preflight: sibling DBs reachable read-only, torch/CUDA, configs load."""
    ok = True

    # 1. sibling sources
    from .db.readonly import ReadOnlyDB
    for name in ("corpus_db", "pressure_db", "research_db", "rse_db"):
        try:
            p = app.cfg.source_path(name)
        except KeyError:
            log.warn(f"  sources.{name}: not configured"); continue
        present = ReadOnlyDB(p).available
        (log.success if present else log.warn)(f"  {name}: {'found' if present else 'MISSING'} ({p})")

    laws = app.cfg.source_path("rse_laws") if app.cfg.get("sources", "rse_laws") else None
    if laws:
        (log.success if Path(laws).exists() else log.warn)(
            f"  rse_laws: {'found' if Path(laws).exists() else 'MISSING'} ({laws})")

    # 2. torch / CUDA
    try:
        import torch
        cuda = torch.cuda.is_available()
        dev = torch.cuda.get_device_name(0) if cuda else "cpu"
        (log.success if cuda else log.warn)(f"  torch {torch.__version__}: CUDA={cuda} ({dev})")
        if not cuda:
            log.warn("  (CPU only — MP0 --quick works but is slow; full grid wants the GPU)")
    except Exception as exc:
        ok = False
        log.error(f"  torch: NOT importable ({exc})")

    # 3. configs
    n_theories = len(app.cfg.theories.get("theories", []))
    n_quant = len(app.cfg.quantities.get("quantities", []))
    (log.success if n_theories and n_quant else log.error)(
        f"  configs: {n_theories} theories, {n_quant} quantities")

    log.info("[verify] done." if ok else "[verify] issues found (see above).")


@cli.command()
@pass_app
def ingest(app: App) -> None:
    """Build the evidence graph and seed quantities/theories/dependencies."""
    from .ingest import evidence_graph
    run_id = app.db.start_run(note="ingest")
    stats = evidence_graph.build(app.db, app.cfg, run_id)
    app.db.finish_run(run_id)
    log.success(f"[ingest] {stats}")


@cli.command()
@click.option("--quick", is_flag=True, help="Tiny grid for fast end-to-end validation.")
@pass_app
def measure(app: App, quick: bool) -> None:
    """Run Instrument Calibration Program 0 (ICP0) -> Quantity b (+ phenomena).

    Resumable: Ctrl-C is safe; re-run to continue from the last finished cell.
    """
    from .lab import icp0  # lazy (imports torch)
    run_id = app.db.start_run(note="ICP0" + (" quick" if quick else ""))
    est = icp0.run(app.db, app.cfg, run_id, quick=quick)
    app.db.finish_run(run_id)
    b = est.get("b")
    if b is not None:
        ci = est.get("b_ci") or [None, None]
        log.success(f"[measure] b = {b:.4f} bits/dim  CI=[{ci[0]:.4f}, {ci[1]:.4f}]  verdict={est['verdict']}")
    else:
        reasons = est.get("reasons") or []
        log.warn(f"[measure] INCONCLUSIVE (verdict={est.get('verdict')}): the instrument is not calibrated, "
                 f"so b is not read. Measurement Axiom violations:")
        for r in (reasons or ["(too few state sizes / interrupted)"]):
            log.warn(f"    - {r}")


@cli.command()
@pass_app
def study(app: App) -> None:
    """Run MP0b — the Bimodal Convergence Investigation (study, not a fix).

    Trains one overloaded cell across many seeds (identical hyperparameters) and
    characterises the two-basin outcome: distribution, stuck fraction, metastability.
    Resumable (Ctrl-C safe).
    """
    from .lab import mp0b  # lazy (imports torch)
    run_id = app.db.start_run(note="MP0b")
    res = mp0b.run(app.db, app.cfg, run_id)
    app.db.finish_run(run_id)
    log.success(f"[study] MP0b-1 summary: {res}")


@cli.command()
@click.argument("theory_id")
@pass_app
def attack(app: App, theory_id: str) -> None:
    """Apply the measured verdict to a law's attack ledger (e.g. `cde attack L1`)."""
    from .theory import verdict
    run_id = app.db.latest_run() or app.db.start_run(note="attack")
    summary = verdict.attack_from_measurements(app.db, app.cfg, theory_id, run_id)
    log.success(
        f"[attack] {theory_id}: verdict={summary['verdict']} -> status={summary['new_status']}")
    if summary.get("undermined"):
        log.warn(f"  structurally undermined dependents: {', '.join(summary['undermined'])}")


@cli.command()
@pass_app
def status(app: App) -> None:
    """Snapshot of quantities, theories, measurements, evidence."""
    db = app.db
    log.info("[bold]Quantities[/bold]")
    for q in db.query("SELECT * FROM quantities ORDER BY id"):
        val = f"{q['value']:.4f}" if q["value"] is not None else "-"
        log.info(f"  {q['symbol']:>4} = {val:>10}  [{q['status']}]  ({q['units']})")
    log.info("[bold]Theories[/bold]")
    for t in db.query("SELECT * FROM theory_objects ORDER BY priority"):
        log.info(f"  {t['id']:>3} [{t['mode']}] {t['status']:>10}  "
                 f"attacks={t['attack_count']} survived={t['failed_attacks']} wounds={t['successful_attacks']}")
    log.info(f"[bold]Counts[/bold]  measurements={db.count('measurements')} "
             f"evidence={db.count('evidence')} nodes={db.count('nodes')} phenomena={db.count('phenomena')}")


@cli.command()
@pass_app
def export(app: App) -> None:
    """Write survival table, constant atlas, quantity sheet, and cemetery."""
    from .export import exporters
    out = app.cfg.data_path("exports")
    stats = exporters.export_all(app.db, out)
    log.success(f"[export] -> {out}  {stats}")


if __name__ == "__main__":
    cli()
