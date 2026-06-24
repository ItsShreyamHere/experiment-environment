"""Read-only access to corpus-builder's corpus.db.

v1 only needs deterministic keyword lookup over paper titles/abstracts (to attach
prior-art evidence to a law). No embeddings/ChromaDB are used, keeping CDE offline.
Adapted from research-synthesis-engine/src/db/corpus_reader.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .readonly import ReadOnlyDB


class CorpusReader(ReadOnlyDB):
    def __init__(self, db_path: str | Path):
        super().__init__(db_path)

    def num_papers(self) -> int:
        return self.count("papers")

    def search_by_keywords(self, terms: list[str], k: int = 12) -> list[dict[str, Any]]:
        """Deterministic OR keyword match over titles/abstracts, ranked by corpus score."""
        terms = [t for t in terms if len(t) > 2]
        if not terms or not self.has_table("papers"):
            return []
        cols = self.columns("papers")
        order = "ranking_score DESC" if "ranking_score" in cols else "citation_count DESC"
        clauses = " OR ".join("title LIKE ? OR abstract LIKE ?" for _ in terms)
        params: list[Any] = []
        for t in terms:
            like = f"%{t}%"
            params.extend([like, like])
        return self.query(
            f"SELECT id AS paper_id, title FROM papers WHERE {clauses} ORDER BY {order} LIMIT ?",
            (*params, k),
        )
