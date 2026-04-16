"""Backend protocol for Gr0m_Mem's storage layer.

Gr0m_Mem supports three backends, in descending order of retrieval
quality:

1. ``chromadb``   — full HNSW vector search over Ollama embeddings.
                    Best retrieval, requires ``pip install gr0m-mem[chromadb]``.
2. ``sqlite_vec`` — SQLite rows with BLOB embeddings, cosine similarity
                    computed in Python. Requires Ollama for embeddings but
                    no extra Python deps.
3. ``sqlite_fts`` — SQLite FTS5 BM25 full-text search. No embeddings, no
                    Ollama, no external deps. The fallback that works
                    anywhere Python does.

All backends implement :class:`VectorBackend`. The :class:`Brain` facade
picks one at init time — callers never see which.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from gr0m_mem.types import Corpus


@dataclass(frozen=True)
class QueryResult:
    """Normalized result shape returned by every backend.

    ``scores`` are in the range ``0.0..1.0`` where higher is better —
    backends are responsible for converting their native ranking (cosine
    distance, BM25, etc.) into this normalized space so ``Brain`` never
    has to special-case them.
    """

    ids: list[str] = field(default_factory=list)
    documents: list[str] = field(default_factory=list)
    metadatas: list[dict[str, Any]] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.ids)


@runtime_checkable
class VectorBackend(Protocol):
    """The storage abstraction every backend must provide."""

    #: short identifier, e.g. ``"chromadb"`` or ``"sqlite_fts"``
    name: str

    #: whether :meth:`add` and :meth:`query` need pre-computed embeddings
    requires_embeddings: bool

    def add(
        self,
        corpus: Corpus,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]] | None,
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Insert or update chunks.

        ``embeddings`` must be non-None when
        :attr:`requires_embeddings` is True, and is ignored otherwise.
        """
        ...

    def query(
        self,
        corpus: Corpus,
        query_text: str,
        query_embedding: list[float] | None,
        n_results: int,
        where: dict[str, Any] | None = None,
    ) -> QueryResult:
        """Retrieve the top ``n_results`` chunks matching the query."""
        ...

    def count(self, corpus: Corpus) -> int: ...

    def list_corpora(self) -> list[str]: ...

    def sample_metadatas(self, corpus: Corpus, limit: int) -> list[dict[str, Any]]:
        """Return up to ``limit`` metadatas for the ``mem_analyze`` tool."""
        ...

    def delete(self, corpus: Corpus, ids: list[str]) -> None: ...

    def close(self) -> None: ...
