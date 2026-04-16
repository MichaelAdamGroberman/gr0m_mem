"""High-level Gr0m_Mem facade (main branch — zero-install core).

This is what both the CLI and the MCP server talk to. It owns the
SQLite FTS5 vector backend, a temporal :class:`KnowledgeGraph`, and a
:class:`Wakeup` store. On this branch there are no semantic retrieval
backends and no Ollama client — every dependency is either in the
Python stdlib or a pure-Python wheel.

The five core retrieval operations:

* ``learn``   — ingest a document (header/body/context chunks, upsert)
* ``search``  — retrieval within a corpus (BM25 via FTS5)
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
from gr0m_mem.store.chunking import Document, chunk_document
from gr0m_mem.store.sqlite_fts import SqliteFtsBackend
from gr0m_mem.types import Corpus, DocumentId
from gr0m_mem.wakeup import Wakeup

log = logging.getLogger("gr0m_mem.brain")


@dataclass(frozen=True)
class SearchHit:
    document_id: str
    chunk_type: str
    score: float
    text: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class BackendChoice:
    name: str
    reason: str


def _select_backend(config: Config) -> tuple[VectorBackend, BackendChoice]:
    """Pick a backend. The main branch only has one.

    The ``semantic`` branch has a real cascade (chromadb → sqlite_vec →
    sqlite_fts). On this branch we always land on ``sqlite_fts`` — but
    we keep the ``BackendChoice`` return shape so callers (and the
    ``mem_status`` diagnostic tool) stay compatible with the semantic
    build.
    """
    if config.backend not in ("auto", "sqlite_fts"):
        log.warning(
            "backend=%r requested but this is the zero-install main branch; "
            "only sqlite_fts is available — switching to sqlite_fts. Install "
            "the semantic branch for chromadb or sqlite_vec.",
            config.backend,
        )
    return SqliteFtsBackend(config.sqlite_fts_path), BackendChoice(
        "sqlite_fts",
        "main branch ships only the SQLite FTS5 backend — no Ollama, "
        "no chromadb, no downloads. Install gr0m-mem from the "
        "`semantic` branch for semantic retrieval backends.",
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
        self._backend.add(
            corpus=corpus,
            ids=[c.id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=None,
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
        raw = self._backend.query(
            corpus=corpus,
            query_text=query,
            query_embedding=None,
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
            "embed_model": None,
        }

    def list_corpora(self) -> list[str]:
        return self._backend.list_corpora()

    def close(self) -> None:
        self._backend.close()
        self._kg.close()
        self._wakeup.close()
