"""Configuration loading and path resolution for CDE.

Loads config/config.yaml plus the theory_objects.yaml and quantities.yaml
sidecars, and resolves sibling source paths relative to the project root so the
checkouts under ../ are found regardless of the current working directory.
Mirrors research-synthesis-engine/src/config.py.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Project root = two levels up from this file (src/config.py -> src -> root).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@dataclass
class Config:
    raw: dict[str, Any]
    theories: dict[str, Any] = field(default_factory=dict)
    quantities: dict[str, Any] = field(default_factory=dict)
    root: Path = PROJECT_ROOT

    # -- generic access -----------------------------------------------------
    def get(self, *keys: str, default: Any = None) -> Any:
        node: Any = self.raw
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    # -- path resolution ----------------------------------------------------
    def resolve(self, relative: str | os.PathLike[str]) -> Path:
        p = Path(relative)
        return p if p.is_absolute() else (self.root / p).resolve()

    def source_path(self, name: str) -> Path:
        rel = self.get("sources", name)
        if rel is None:
            raise KeyError(f"sources.{name} not configured")
        return self.resolve(rel)

    def data_path(self, name: str) -> Path:
        rel = self.get("paths", name)
        if rel is None:
            raise KeyError(f"paths.{name} not configured")
        return self.resolve(rel)

    @property
    def db_path(self) -> Path:
        return self.resolve(self.get("database", "path", default="data/db/cde.db"))

    @property
    def accepted_only(self) -> bool:
        return bool(self.get("upstream", "accepted_only", default=True))

    def icp0(self, quick: bool = False) -> dict[str, Any]:
        """Return the ICP0 grid config, with the `quick` overrides merged in."""
        base = dict(self.get("icp0", default={}) or {})
        if quick:
            base.update(base.get("quick", {}) or {})
        base.pop("quick", None)
        return base


def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    cfg_path = Path(path) if path else (CONFIG_DIR / "config.yaml")
    raw = _load_yaml(cfg_path)
    return Config(
        raw=raw,
        theories=_load_yaml(CONFIG_DIR / "theory_objects.yaml"),
        quantities=_load_yaml(CONFIG_DIR / "quantities.yaml"),
    )
