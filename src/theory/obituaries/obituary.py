"""Theory obituaries — dead laws are discoveries.

Template-based (no LLM). When a theory collapses, we record precisely why: the
cause of death, the killer (which Measurement Program), the date, its historical
significance, and the descendants it leaves behind (its structural dependents).
"""

from __future__ import annotations

import json
from typing import Any

from ...db.connection import Database, utcnow
from .. import objects as T

_TEMPLATE = """# Obituary — {tid}: {ttype}

**Died:** {died_at}
**Killer:** {killer}
**Cause of death:** {cause}

## Statement (what it claimed)
{statement}

## How it was falsified
{falsification}

{evidence_block}

## Historical significance
{significance}

## Descendants (theories that depended on it)
{descendants}

> A dead law is not a failure of the instrument. It is a measurement of nature.
"""


def _significance(theory: dict[str, Any]) -> str:
    deps = theory.get("_dependents", [])
    if deps:
        return (
            f"High. {theory['id']} was load-bearing: {len(deps)} theory(ies) "
            f"({', '.join(deps)}) were built on it and are now structurally undermined."
        )
    return f"Notable. {theory['id']} collapsed without surviving dependents."


def write_obituary(
    db: Database,
    theory_id: str,
    killer: str,
    run_id: str,
    detail: dict[str, Any] | None = None,
) -> None:
    theory = T.get_theory(db, theory_id) or {"id": theory_id}
    detail = detail or {}
    deps = [r["child_id"] for r in db.query(
        "SELECT child_id FROM theory_dependencies WHERE parent_id=?", (theory_id,)
    )]
    theory["_dependents"] = deps

    cause = detail.get("cause", "falsified by measurement")
    evidence_block = ""
    if detail.get("evidence"):
        evidence_block = "## Evidence at death\n" + "\n".join(f"- {e}" for e in detail["evidence"])

    text = _TEMPLATE.format(
        tid=theory_id,
        ttype=theory.get("type", "theory"),
        died_at=utcnow(),
        killer=killer,
        cause=cause,
        statement=(theory.get("statement") or "").strip(),
        falsification=(theory.get("falsification") or "").strip(),
        evidence_block=evidence_block,
        significance=_significance(theory),
        descendants=", ".join(deps) if deps else "(none)",
    )

    db.insert(
        "obituaries",
        {
            "theory_id": theory_id,
            "cause_of_death": cause,
            "killer": killer,
            "died_at": utcnow(),
            "historical_significance": _significance(theory),
            "descendants_json": json.dumps(deps, ensure_ascii=False),
            "text": text,
            "run_id": run_id,
        },
    )
    db.commit()
