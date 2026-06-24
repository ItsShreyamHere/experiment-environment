"""Scientific Quantity — the heart of the instrument.

Laws depend on quantities; quantities depend only on measurements. A Quantity is
seeded `unmeasured` and becomes `measured` once a Measurement Program assigns it
a value with a confidence interval. There are no beliefs here — only numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..db.connection import Database, utcnow

UNMEASURED = "unmeasured"
MEASURED = "measured"


@dataclass
class Quantity:
    name: str
    symbol: str
    units: str
    measurement_method: str
    description: str = ""
    value: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    status: str = UNMEASURED

    def to_row(self, run_id: str) -> dict[str, Any]:
        return {
            "id": self.name,
            "name": self.name,
            "symbol": self.symbol,
            "units": self.units,
            "measurement_method": self.measurement_method,
            "description": self.description,
            "value": self.value,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "status": self.status,
            "run_id": run_id,
            "created_at": utcnow(),
        }


def seed_quantities(db: Database, cfg_quantities: dict[str, Any], run_id: str) -> int:
    """Insert the seed Quantity definitions (idempotent, unmeasured)."""
    n = 0
    for q in cfg_quantities.get("quantities", []):
        quantity = Quantity(
            name=q["name"],
            symbol=q.get("symbol", q["name"]),
            units=q.get("units", ""),
            measurement_method=q.get("measurement_method", ""),
            description=q.get("description", ""),
            status=q.get("status", UNMEASURED),
        )
        db.insert("quantities", quantity.to_row(run_id))
        n += 1
    db.commit()
    return n


def set_measured(
    db: Database,
    name: str,
    value: float,
    ci_low: float,
    ci_high: float,
    run_id: str,
    method: str | None = None,
) -> None:
    """Record a measured value for a quantity (promotes it to `measured`)."""
    row = db.one("SELECT * FROM quantities WHERE id=?", (name,)) or {}
    db.insert(
        "quantities",
        {
            "id": name,
            "name": name,
            "symbol": row.get("symbol", name),
            "units": row.get("units", ""),
            "measurement_method": method or row.get("measurement_method", ""),
            "description": row.get("description", ""),
            "value": value,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "status": MEASURED,
            "run_id": run_id,
            "created_at": utcnow(),
        },
    )
    db.commit()


def get_quantity(db: Database, name: str) -> dict[str, Any] | None:
    return db.one("SELECT * FROM quantities WHERE id=?", (name,))
