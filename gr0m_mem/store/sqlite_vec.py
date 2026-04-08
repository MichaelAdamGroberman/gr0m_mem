"""Pure-SQLite vector backend — the ``chromadb``-free path.

Stores each chunk as one row with the embedding as a BLOB (little-endian
float32). Retrieval loads all rows for the target corpus, computes cosine
similarity in Python (numpy if available, stdlib otherwise), and returns
the top-N.

This is O(N) per query, which is fine at the scales Gr0m_Mem targets
(tens of thousands of chunks per corpus). For millions of chunks upgrade
to the ``chromadb`` backend or wait for the planned sqlite-vec ANN
integration.
"""

from __future__ import annotations

import json
import math
import sqlite3
import struct
from pathlib import Path
from typing import Any

from gr0m_mem.store.base import QueryResult, VectorBackend
from gr0m_mem.types import Corpus

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    corpus     TEXT NOT NULL,
    id         TEXT NOT NULL,
    document   TEXT NOT NULL,
    embedding  BLOB NOT NULL,
    dim        INTEGER NOT NULL,
    metadata   TEXT NOT NULL,
    PRIMARY KEY (corpus, id)
);

CREATE INDEX IF NOT EXISTS idx_chunks_corpus ON chunks(corpus);
"""


def _pack_embedding(emb: list[float]) -> bytes:
    return struct.pack(f"<{len(emb)}f", *emb)


def _unpack_embedding(blob: bytes, dim: int) -> list[float]:
    return list(struct.unpack(f"<{dim}f", blob))


def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b, strict=True):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def _cosine_batch(query: list[float], rows: list[list[float]]) -> list[float]:
    """Optionally-vectorized batch cosine. Falls back to the stdlib loop."""
    try:
        import numpy as np
    except ImportError:
        return [_cosine(query, r) for r in rows]

    if not rows:
        return []
    q = np.asarray(query, dtype=np.float32)
    m = np.asarray(rows, dtype=np.float32)
    q_norm = np.linalg.norm(q)
    row_norms = np.linalg.norm(m, axis=1)
    denom = q_norm * row_norms
    denom[denom == 0] = 1.0
    scores = (m @ q) / denom
    return [float(s) for s in scores]


class SqliteVectorBackend(VectorBackend):
    """SQLite-backed vector store with Python-side cosine similarity."""

    name = "sqlite_vec"
    requires_embeddings = True

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._path = db_path
        self._conn = sqlite3.connect(str(db_path), isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)

    @property
    def path(self) -> Path:
        return self._path

    # ── VectorBackend API ─────────────────────────────────

    def add(
        self,
        corpus: Corpus,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]] | None,
        metadatas: list[dict[str, Any]],
    ) -> None:
        if embeddings is None:
            raise ValueError("SqliteVectorBackend requires embeddings")
        if not (len(ids) == len(documents) == len(embeddings) == len(metadatas)):
            raise ValueError("ids, documents, embeddings, metadatas must match in length")
        if not ids:
            return
        rows = []
        for cid, doc, emb, meta in zip(ids, documents, embeddings, metadatas, strict=True):
            rows.append(
                (
                    corpus.name,
                    cid,
                    doc,
                    _pack_embedding(emb),
                    len(emb),
                    json.dumps(meta, sort_keys=True),
                )
            )
        self._conn.executemany(
            """
            INSERT INTO chunks (corpus, id, document, embedding, dim, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(corpus, id) DO UPDATE SET
                document = excluded.document,
                embedding = excluded.embedding,
                dim = excluded.dim,
                metadata = excluded.metadata
            """,
            rows,
        )

    def query(
        self,
        corpus: Corpus,
        query_text: str,
        query_embedding: list[float] | None,
        n_results: int,
        where: dict[str, Any] | None = None,
    ) -> QueryResult:
        if query_embedding is None:
            raise ValueError("SqliteVectorBackend requires a query_embedding")
        rows = self._conn.execute(
            "SELECT id, document, embedding, dim, metadata FROM chunks WHERE corpus = ?",
            (corpus.name,),
        ).fetchall()
        if not rows:
            return QueryResult()

        # Optional metadata filter — we apply it Python-side because the
        # stored metadata is JSON text. Keeps the SQL simple and avoids
        # shipping a metadata indexer.
        if where:
            filtered = []
            for r in rows:
                meta = json.loads(r["metadata"])
                if all(meta.get(k) == v for k, v in where.items()):
                    filtered.append(r)
            rows = filtered
            if not rows:
                return QueryResult()

        embeddings = [_unpack_embedding(r["embedding"], r["dim"]) for r in rows]
        scores = _cosine_batch(query_embedding, embeddings)

        ranked = sorted(
            range(len(rows)), key=lambda i: scores[i], reverse=True
        )[:n_results]

        return QueryResult(
            ids=[rows[i]["id"] for i in ranked],
            documents=[rows[i]["document"] for i in ranked],
            metadatas=[json.loads(rows[i]["metadata"]) for i in ranked],
            scores=[max(0.0, scores[i]) for i in ranked],
        )

    def count(self, corpus: Corpus) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM chunks WHERE corpus = ?", (corpus.name,)
        ).fetchone()
        return int(row["n"])

    def list_corpora(self) -> list[str]:
        rows = self._conn.execute(
            "SELECT DISTINCT corpus FROM chunks ORDER BY corpus"
        ).fetchall()
        return [r["corpus"] for r in rows]

    def sample_metadatas(self, corpus: Corpus, limit: int) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT metadata FROM chunks WHERE corpus = ? LIMIT ?",
            (corpus.name, limit),
        ).fetchall()
        return [json.loads(r["metadata"]) for r in rows]

    def delete(self, corpus: Corpus, ids: list[str]) -> None:
        if not ids:
            return
        self._conn.executemany(
            "DELETE FROM chunks WHERE corpus = ? AND id = ?",
            [(corpus.name, i) for i in ids],
        )

    def close(self) -> None:
        self._conn.close()
