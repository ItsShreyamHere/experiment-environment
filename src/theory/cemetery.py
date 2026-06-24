"""The Theory Cemetery — every theory's life, partitioned by fate.

A read model over theory_objects + obituaries. Partitions:
    surviving : status in {unknown, surviving}
    damaged   : status == damaged
    collapsed : status in {collapsed, undermined}  (+ obituary if any)
"""

from __future__ import annotations

from typing import Any

from ..db.connection import Database
from . import objects as T


def partitions(db: Database) -> dict[str, list[dict[str, Any]]]:
    rows = db.query("SELECT * FROM theory_objects ORDER BY priority")
    out: dict[str, list[dict[str, Any]]] = {"surviving": [], "damaged": [], "collapsed": []}
    for r in rows:
        status = r["status"]
        if status in (T.UNKNOWN, T.SURVIVING):
            out["surviving"].append(r)
        elif status == T.DAMAGED:
            out["damaged"].append(r)
        else:  # COLLAPSED or UNDERMINED
            obit = db.one("SELECT * FROM obituaries WHERE theory_id=?", (r["id"],))
            r = dict(r)
            r["obituary"] = obit
            out["collapsed"].append(r)
    return out
