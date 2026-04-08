"""SQLite FTS5 backend — the zero-dependency fallback.

FTS5 ships with CPython's ``sqlite3`` on every mainstream platform.
This backend ignores embeddings entirely and uses BM25 full-text search.
It's what runs on a laptop where the user doesn't have Ollama, doesn't
have chromadb, and just wants Claude to stop forgetting things.

Gr0m_Mem never refuses to start because of a missing dependency. If
nothing else works, this does.

Known limitation: FTS5 BM25 is a lexical match — it won't find
"vehicle" when you search for "car". For semantic retrieval install
either ``gr0m-mem[chromadb]`` or Ollama + mxbai-embed-large.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from gr0m_mem.store.base import QueryResult, VectorBackend
from gr0m_mem.types import Corpus

# Contentless FTS5 table — we manage the document text ourselves so we
# can tokenize whatever the caller hands us without imposing a schema.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    corpus    TEXT NOT NULL,
    id        TEXT NOT NULL,
    document  TEXT NOT NULL,
    metadata  TEXT NOT NULL,
    PRIMARY KEY (corpus, id)
);

CREATE INDEX IF NOT EXISTS idx_chunks_corpus ON chunks(corpus);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    corpus UNINDEXED,
    chunk_id UNINDEXED,
    document,
    tokenize='porter unicode61'
);
"""


def _check_fts5_available(conn: sqlite3.Connection) -> None:
    """Best-effort check that FTS5 is compiled in. Raises on failure."""
    try:
        conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS _fts5_probe USING fts5(x)")
        conn.execute("DROP TABLE IF EXISTS _fts5_probe")
    except sqlite3.OperationalError as e:  # pragma: no cover - platform-specific
        raise RuntimeError(
            "SQLite FTS5 is not available in this Python build. "
            "Upgrade to a CPython from python.org, or install "
            "`gr0m-mem[chromadb]` for the chromadb backend."
        ) from e


def _escape_fts_query(q: str) -> str:
    """Escape a user query for FTS5 MATCH.

    FTS5's MATCH syntax is its own small language (NEAR, OR, phrase
    groups, column filters...). For a user's free-text question we want
    simple AND-over-bag-of-words semantics, so we strip punctuation,
    wrap each token in double-quotes to disable operators, and join with
    spaces (implicit AND).
    """
    out: list[str] = []
    current: list[str] = []
    for ch in q:
        if ch.isalnum() or ch == "_":
            current.append(ch.lower())
        else:
            if current:
                out.append("".join(current))
                current = []
    if current:
        out.append("".join(current))
    # Drop very short fragments (FTS5 tokenizer would drop them anyway)
    tokens = [t for t in out if len(t) > 1]
    if not tokens:
        return "\"\""  # never-matches fallback
    # OR semantics so BM25 can rank partial matches instead of requiring
    # every query token to hit. This is what users expect from natural-
    # language questions, and BM25 handles the ranking for us.
    return " OR ".join(f'"{t}"' for t in tokens)


class SqliteFtsBackend(VectorBackend):
    """Lexical full-text search via SQLite FTS5. No embeddings needed."""

    name = "sqlite_fts"
    requires_embeddings = False

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._path = db_path
        self._conn = sqlite3.connect(str(db_path), isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        _check_fts5_available(self._conn)
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
        if not (len(ids) == len(documents) == len(metadatas)):
            raise ValueError("ids, documents, metadatas must match in length")
        if not ids:
            return
        # embeddings are ignored on purpose.
        for cid, doc, meta in zip(ids, documents, metadatas, strict=True):
            # Delete any existing row (upsert) before re-indexing the FTS
            # row so we never leave orphan FTS entries pointing at a stale
            # document.
            self._conn.execute(
                "DELETE FROM chunks_fts WHERE corpus = ? AND chunk_id = ?",
                (corpus.name, cid),
            )
            self._conn.execute(
                """
                INSERT INTO chunks (corpus, id, document, metadata)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(corpus, id) DO UPDATE SET
                    document = excluded.document,
                    metadata = excluded.metadata
                """,
                (corpus.name, cid, doc, json.dumps(meta, sort_keys=True)),
            )
            self._conn.execute(
                "INSERT INTO chunks_fts (corpus, chunk_id, document) VALUES (?, ?, ?)",
                (corpus.name, cid, doc),
            )

    def query(
        self,
        corpus: Corpus,
        query_text: str,
        query_embedding: list[float] | None,
        n_results: int,
        where: dict[str, Any] | None = None,
    ) -> QueryResult:
        # query_embedding ignored on purpose.
        match_expr = _escape_fts_query(query_text)
        rows = self._conn.execute(
            """
            SELECT c.id, c.document, c.metadata, bm25(chunks_fts) AS rank
              FROM chunks_fts
              JOIN chunks c
                ON c.corpus = chunks_fts.corpus
               AND c.id = chunks_fts.chunk_id
             WHERE chunks_fts.corpus = ?
               AND chunks_fts MATCH ?
             ORDER BY rank
             LIMIT ?
            """,
            (corpus.name, match_expr, n_results * 3),
        ).fetchall()

        if where:
            rows = [
                r
                for r in rows
                if all(json.loads(r["metadata"]).get(k) == v for k, v in where.items())
            ][:n_results]
        else:
            rows = rows[:n_results]

        if not rows:
            return QueryResult()

        # BM25 scores are negative (lower = better). Normalize to [0, 1]
        # by shifting with the max magnitude seen in this response.
        raw = [-float(r["rank"]) for r in rows]
        max_raw = max(raw) if raw else 1.0
        denom = max_raw if max_raw > 0 else 1.0
        scores = [min(1.0, max(0.0, s / denom)) for s in raw]

        return QueryResult(
            ids=[r["id"] for r in rows],
            documents=[r["document"] for r in rows],
            metadatas=[json.loads(r["metadata"]) for r in rows],
            scores=scores,
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
        for cid in ids:
            self._conn.execute(
                "DELETE FROM chunks_fts WHERE corpus = ? AND chunk_id = ?",
                (corpus.name, cid),
            )
            self._conn.execute(
                "DELETE FROM chunks WHERE corpus = ? AND id = ?",
                (corpus.name, cid),
            )

    def close(self) -> None:
        self._conn.close()
