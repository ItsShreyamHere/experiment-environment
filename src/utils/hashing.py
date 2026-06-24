"""Deterministic hashing helpers (stable ids, cache keys).

All hashes are stable across runs and processes: keys are sorted and a fixed
encoding is used so the same logical input always yields the same digest.
Mirrors the family convention (research-synthesis-engine/src/utils/hashing.py).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_hash(obj: Any) -> str:
    """Stable SHA-256 over an arbitrary JSON-serialisable object."""
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    return sha256_text(payload)


def short_hash(obj: Any, n: int = 16) -> str:
    return stable_hash(obj)[:n]
