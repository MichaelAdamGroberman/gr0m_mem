"""ChromaDB backend.

Lazy-imports ``chromadb`` so the module loads even when the optional
dependency is not installed. :class:`ChromaBackend` raises a clear error
at construction time if chromadb is missing, and the auto-selection
cascade in :class:`gr0m_mem.brain.Brain` falls back to
:class:`gr0m_mem.store.sqlite_vec.SqliteVectorBackend` or
:class:`gr0m_mem.store.sqlite_fts.SqliteFtsBackend` instead.
"""

from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING, Any

from gr0m_mem.store.base import QueryResult, VectorBackend
from gr0m_mem.types import Corpus

if TYPE_CHECKING:
    from chromadb.api.models.Collection import Collection


def chromadb_available() -> bool:
    return find_spec("chromadb") is not None


class ChromaBackend(VectorBackend):
    """Best-quality backend — HNSW via chromadb, cosine space."""

    name = "chromadb"
    requires_embeddings = True

    def __init__(self, path: Path) -> None:
        if not chromadb_available():
            raise RuntimeError(
                "ChromaBackend requested but `chromadb` is not installed. "
                "Install with `pip install gr0m-mem[chromadb]` or set "
                "GR0M_MEM_BACKEND=sqlite_vec to use the pure-SQLite backend."
            )
        # Import here so the module still loads without the optional dep.
        import chromadb
        from chromadb.config import Settings

        path.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._client = chromadb.PersistentClient(
            path=str(path),
            settings=Settings(anonymized_telemetry=False, allow_reset=False),
        )
        self._collections: dict[str, Collection] = {}

    # ── Internal ──────────────────────────────────────────

    def _collection(self, corpus: Corpus) -> Collection:
        key = corpus.collection_name
        cached = self._collections.get(key)
        if cached is not None:
            return cached
        col = self._client.get_or_create_collection(
            name=key,
            metadata={"hnsw:space": "cosine", "gr0m_mem:corpus": corpus.name},
        )
        self._collections[key] = col
        return col

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
            raise ValueError("ChromaBackend requires embeddings")
        if not (len(ids) == len(documents) == len(embeddings) == len(metadatas)):
            raise ValueError("ids, documents, embeddings, metadatas must match in length")
        if not ids:
            return
        self._collection(corpus).upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,  # type: ignore[arg-type]
            metadatas=metadatas,  # type: ignore[arg-type]
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
            raise ValueError("ChromaBackend requires a query_embedding")
        raw = self._collection(corpus).query(
            query_embeddings=[query_embedding],  # type: ignore[arg-type]
            n_results=n_results,
            where=where,
        )
        ids = (raw.get("ids") or [[]])[0]
        docs = (raw.get("documents") or [[]])[0]
        metas = (raw.get("metadatas") or [[]])[0]
        dists = (raw.get("distances") or [[]])[0]
        # Chroma returns cosine distance; we return similarity in [0, 1].
        scores = [max(0.0, 1.0 - float(d)) for d in dists]
        return QueryResult(
            ids=list(ids),
            documents=list(docs),
            metadatas=[dict(m) if m else {} for m in metas],
            scores=scores,
        )

    def count(self, corpus: Corpus) -> int:
        return int(self._collection(corpus).count())

    def list_corpora(self) -> list[str]:
        return sorted(c.name for c in self._client.list_collections())

    def sample_metadatas(self, corpus: Corpus, limit: int) -> list[dict[str, Any]]:
        col = self._collection(corpus)
        total = col.count()
        if not total:
            return []
        raw = col.get(limit=min(limit, total), include=["metadatas"])
        return [dict(m) if m else {} for m in (raw.get("metadatas") or [])]

    def delete(self, corpus: Corpus, ids: list[str]) -> None:
        if not ids:
            return
        self._collection(corpus).delete(ids=ids)

    def close(self) -> None:
        # Chroma's PersistentClient has no explicit close; GC handles it.
        self._collections.clear()
