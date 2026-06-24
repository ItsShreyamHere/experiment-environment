"""Template-based exporters (no LLM).

Writes the instrument's outputs to data/exports/:
  - survival_table.{json,md} : every theory's status + attack ledger
  - constant_atlas.json      : measured quantities (b ...) with CI + method
  - quantities.json          : the full quantity sheet
  - cemetery/{collapsed,damaged,surviving}/*.md : each theory's life + obituary
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..db.connection import Database
from ..theory import cemetery


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def export_all(db: Database, exports_dir: Path) -> dict[str, Any]:
    exports_dir.mkdir(parents=True, exist_ok=True)
    theories = db.query("SELECT * FROM theory_objects ORDER BY priority")
    quantities = db.query("SELECT * FROM quantities ORDER BY id")
    phenomena = db.query("SELECT * FROM phenomena ORDER BY id")

    # -- survival table ----------------------------------------------------
    _dump(exports_dir / "survival_table.json", theories)
    rows = ["| Theory | Type | Mode | Status | attacks | survived | wounded |",
            "|---|---|---|---|---|---|---|"]
    for t in theories:
        rows.append(
            f"| {t['id']} | {t['type']} | {t['mode']} | **{t['status']}** | "
            f"{t['attack_count']} | {t['failed_attacks']} | {t['successful_attacks']} |"
        )
    _write(exports_dir / "survival_table.md", "# Law Survival Table\n\n" + "\n".join(rows) + "\n")

    # -- constant atlas + quantity sheet ----------------------------------
    atlas = [{
        "name": q["name"], "symbol": q["symbol"], "units": q["units"],
        "value": q["value"], "ci": [q["ci_low"], q["ci_high"]],
        "status": q["status"], "method": q["measurement_method"],
    } for q in quantities]
    _dump(exports_dir / "constant_atlas.json", [a for a in atlas if a["status"] == "measured"])
    _dump(exports_dir / "quantities.json", atlas)
    _dump(exports_dir / "phenomena.json", phenomena)

    # -- cemetery ----------------------------------------------------------
    parts = cemetery.partitions(db)
    cem_dir = exports_dir / "cemetery"
    counts = {}
    for part_name, members in parts.items():
        counts[part_name] = len(members)
        for t in members:
            md = _theory_card(t)
            _write(cem_dir / part_name / f"{t['id']}.md", md)

    return {
        "theories": len(theories),
        "quantities_measured": sum(1 for a in atlas if a["status"] == "measured"),
        "phenomena": len(phenomena),
        "cemetery": counts,
    }


def _theory_card(t: dict[str, Any]) -> str:
    lines = [
        f"# {t['id']} — {t['type']} ({t['status']})",
        "",
        f"**Mode:** {t['mode']}  **Priority:** {t['priority']}",
        "",
        f"**Statement:** {t['statement']}",
        "",
        f"**Formula:** `{t['formula']}`",
        "",
        f"**Falsification:** {t['falsification']}",
        "",
        "## Attack ledger",
        f"- attacks: {t['attack_count']}",
        f"- survived: {t['failed_attacks']}",
        f"- wounding/killing: {t['successful_attacks']}",
    ]
    if t.get("obituary"):
        lines += ["", "---", "", t["obituary"]["text"]]
    return "\n".join(lines) + "\n"
