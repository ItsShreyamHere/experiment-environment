"""Structural theory dependency graph (NOT Bayesian).

Edges record that one theory is built on another (L1 -> K6 -> K9). When a parent
collapses, dependents are marked `undermined` — a structural consequence, not a
confidence adjustment. Propagation is transitive.
"""

from __future__ import annotations

from typing import Any

from ..db.connection import Database
from . import objects as T


def seed_dependencies(db: Database, cfg_theories: dict[str, Any], run_id: str) -> int:
    """Create parent->child edges from each theory's `depends_on` list."""
    n = 0
    for t in cfg_theories.get("theories", []):
        child = t["id"]
        for parent in t.get("depends_on", []):
            db.insert(
                "theory_dependencies",
                {"parent_id": parent, "child_id": child, "relation": "supports", "run_id": run_id},
            )
            n += 1
    db.commit()
    return n


def dependents(db: Database, theory_id: str) -> list[str]:
    rows = db.query(
        "SELECT child_id FROM theory_dependencies WHERE parent_id=?", (theory_id,)
    )
    return [r["child_id"] for r in rows]


def propagate_collapse(db: Database, theory_id: str, run_id: str) -> list[str]:
    """Mark all (transitive) dependents of a collapsed theory as `undermined`.

    Returns the list of theory ids that were undermined.
    """
    undermined: list[str] = []
    frontier = dependents(db, theory_id)
    seen: set[str] = set()
    while frontier:
        child = frontier.pop()
        if child in seen:
            continue
        seen.add(child)
        row = T.get_theory(db, child)
        if row and row["status"] not in T.DEAD_STATUSES:
            T.set_status(db, child, T.UNDERMINED)
            undermined.append(child)
        frontier.extend(dependents(db, child))
    return undermined
