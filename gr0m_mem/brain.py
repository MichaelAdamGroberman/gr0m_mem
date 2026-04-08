"""High-level Gr0m_Mem facade.

This is what both the CLI and the MCP server talk to. It owns a
:class:`VectorBackend` (chosen at init time via the selection cascade),
an :class:`EmbeddingClient` (only used if the backend needs embeddings),
a temporal :class:`KnowledgeGraph`, and a :class:`Wakeup` store.

The five core operations:

* ``learn``   — ingest a document (header/body/context chunks, embed if
                the backend supports it, upsert)
* ``search``  — retrieval within a corpus
* ``query``   — alias with a simpler return shape
* ``rag``     — return a context block ready to inline into a prompt
* ``analyze`` — aggregate stats about a corpus

The loop-prevention layer (:class:`Wakeup`) and the temporal KG
(:class:`KnowledgeGraph`) live alongside as separate, composable
subsystems. Callers pick and mix them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from gr0m_mem.config import Config
from gr0m_mem.graph.kg import KnowledgeGraph
from gr0m_mem.store.base import VectorBackend
from gr0m_mem.store.chroma import ChromaBackend, chromadb_available
from gr0m_mem.store.embedding import Document, EmbeddingClient, chunk_document
from gr0m_mem.store.sqlite_fts import SqliteFtsBackend
from gr0m_mem.store.sqlite_vec import SqliteVectorBackend
from gr0m_mem.types import Corpus, DocumentId
from gr0m_mem.wakeup import Wakeup

log = logging.getLogger("gr0m_mem.brain")


@dataclass(frozen=True, slots=True)
class SearchHit:
    document_id: str
    chunk_type: str
    score: float
    text: str
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class BackendChoice:
    name: str
    reason: str


def _select_backend(config: Config) -> tuple[VectorBackend, BackendChoice]:
    """Pick a vector backend following the auto-selection cascade."""
    requested = config.backend

    if requested == "chromadb":
        return ChromaBackend(config.chroma_path), BackendChoice(
            "chromadb", "explicitly requested via GR0M_MEM_BACKEND"
        )
    if requested == "sqlite_vec":
        return SqliteVectorBackend(config.sqlite_vec_path), BackendChoice(
            "sqlite_vec", "explicitly requested via GR0M_MEM_BACKEND"
        )
    if requested == "sqlite_fts":
        return SqliteFtsBackend(config.sqlite_fts_path), BackendChoice(
            "sqlite_fts", "explicitly requested via GR0M_MEM_BACKEND"
        )

    # auto: chromadb → sqlite_vec (if ollama) → sqlite_fts
    if chromadb_available():
        return ChromaBackend(config.chroma_path), BackendChoice(
            "chromadb", "auto: chromadb package is installed"
        )

    embed = EmbeddingClient(url=config.ollama_url, model=config.embed_model)
    try:
        if embed.health():
            return SqliteVectorBackend(config.sqlite_vec_path), BackendChoice(
                "sqlite_vec",
                f"auto: chromadb not installed; ollama reachable with {config.embed_model}",
            )
    finally:
        embed.close()

    return SqliteFtsBackend(config.sqlite_fts_path), BackendChoice(
        "sqlite_fts",
        "auto: chromadb not installed and ollama unreachable — "
        "falling back to SQLite FTS5 lexical search",
    )


class Brain:
    """Gr0m_Mem core facade. Thread-unsafe — one instance per process."""

    def __init__(self, config: Config) -> None:
        config.ensure_dirs()
        self._config = config
        self._backend, self._choice = _select_backend(config)
        log.info(
            "gr0m_mem backend=%s reason=%r",
            self._choice.name,
            self._choice.reason,
        )
        self._embed: EmbeddingClient | None = None
        if self._backend.requires_embeddings:
            self._embed = EmbeddingClient(
                url=config.ollama_url,
                model=config.embed_model,
            )
        self._kg = KnowledgeGraph(
            db_path=config.graph_db_path,
            fact_check_mode=config.fact_check_mode,
        )
        self._wakeup = Wakeup(config.wakeup_db_path)

    # ── Properties ───────────────────────────────────────

    @property
    def config(self) -> Config:
        return self._config

    @property
    def backend(self) -> VectorBackend:
        return self._backend

    @property
    def backend_choice(self) -> BackendChoice:
        return self._choice

    @property
    def embedding(self) -> EmbeddingClient | None:
        return self._embed

    @property
    def kg(self) -> KnowledgeGraph:
        return self._kg

    @property
    def wakeup(self) -> Wakeup:
        return self._wakeup

    # ── Core operations ──────────────────────────────────

    def learn(
        self,
        corpus: Corpus,
        document_id: str,
        title: str,
        body: str,
        context: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Ingest a document into a corpus."""
        doc = Document(
            id=DocumentId(document_id),
            title=title,
            body=body,
            context=context,
            metadata=metadata or {},
        )
        chunks = chunk_document(doc)
        embeddings: list[list[float]] | None = None
        if self._backend.requires_embeddings:
            assert self._embed is not None
            embeddings = self._embed.embed_many([c.text for c in chunks])
        self._backend.add(
            corpus=corpus,
            ids=[c.id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=embeddings,
            metadatas=[c.metadata for c in chunks],
        )
        return {
            "corpus": corpus.name,
            "document_id": document_id,
            "backend": self._backend.name,
            "chunks_indexed": len(chunks),
            "chunk_ids": [c.id for c in chunks],
        }

    def search(
        self,
        corpus: Corpus,
        query: str,
        n_results: int = 10,
        where: dict[str, Any] | None = None,
    ) -> list[SearchHit]:
        """Retrieve within a corpus, fusing scores across the 3 chunks of each doc."""
        query_embedding: list[float] | None = None
        if self._backend.requires_embeddings:
            assert self._embed is not None
            query_embedding = self._embed.embed(query)

        raw = self._backend.query(
            corpus=corpus,
            query_text=query,
            query_embedding=query_embedding,
            n_results=max(n_results * 3, 10),
            where=where,
        )

        # Collapse chunks -> documents. Keep the best-scoring chunk and
        # give a modest bonus per additional matching chunk of the same
        # document.
        agg: dict[str, SearchHit] = {}
        for chunk_id, text, meta, score in zip(
            raw.ids, raw.documents, raw.metadatas, raw.scores, strict=False
        ):
            doc_id = str(meta.get("document_id", chunk_id.split("::", 1)[0]))
            existing = agg.get(doc_id)
            if existing is None or score > existing.score:
                agg[doc_id] = SearchHit(
                    document_id=doc_id,
                    chunk_type=str(meta.get("chunk_type", "")),
                    score=score,
                    text=text,
                    metadata=dict(meta),
                )
            else:
                # Additional matching chunks on the same document get a
                # modest bonus, capped so the public score stays in [0, 1].
                bumped = min(1.0, existing.score + score * 0.3)
                agg[doc_id] = SearchHit(
                    document_id=existing.document_id,
                    chunk_type=existing.chunk_type,
                    score=bumped,
                    text=existing.text,
                    metadata=existing.metadata,
                )

        return sorted(agg.values(), key=lambda h: h.score, reverse=True)[:n_results]

    def query(
        self,
        corpus: Corpus,
        query: str,
        n_results: int = 5,
    ) -> dict[str, Any]:
        hits = self.search(corpus=corpus, query=query, n_results=n_results)
        return {
            "corpus": corpus.name,
            "query": query,
            "backend": self._backend.name,
            "count": len(hits),
            "results": [
                {
                    "document_id": h.document_id,
                    "score": round(h.score, 4),
                    "chunk_type": h.chunk_type,
                    "text": h.text,
                    "metadata": h.metadata,
                }
                for h in hits
            ],
        }

    def rag(
        self,
        corpus: Corpus,
        query: str,
        n_results: int = 5,
        max_context_chars: int = 4000,
    ) -> dict[str, Any]:
        """Return a concatenated context block suitable for inlining into a prompt.

        Gr0m_Mem does **not** call an LLM here. The calling agent is the LLM.
        """
        hits = self.search(corpus=corpus, query=query, n_results=n_results)
        parts: list[str] = []
        used = 0
        for h in hits:
            chunk = f"[{h.document_id}] ({h.chunk_type}, score={h.score:.3f})\n{h.text}"
            if used + len(chunk) > max_context_chars:
                break
            parts.append(chunk)
            used += len(chunk)
        return {
            "corpus": corpus.name,
            "query": query,
            "backend": self._backend.name,
            "context": "\n\n---\n\n".join(parts),
            "context_chars": used,
            "hit_count": len(parts),
        }

    def analyze(self, corpus: Corpus) -> dict[str, Any]:
        total_chunks = self._backend.count(corpus)
        metadatas = self._backend.sample_metadatas(corpus, limit=200)
        unique_docs = {str(m.get("document_id", "")) for m in metadatas if m}
        unique_docs.discard("")
        return {
            "corpus": corpus.name,
            "collection": corpus.collection_name,
            "backend": self._backend.name,
            "total_chunks": total_chunks,
            "sampled_unique_documents": len(unique_docs),
            "embed_model": self._embed.model if self._embed else None,
        }

    def list_corpora(self) -> list[str]:
        return self._backend.list_corpora()

    def close(self) -> None:
        if self._embed is not None:
            self._embed.close()
        self._backend.close()
        self._kg.close()
        self._wakeup.close()
