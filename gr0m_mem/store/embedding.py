"""Embedding client + 3-chunk document chunking.

The 3-chunk strategy (header / body / context) is generalized from the
production RedAI brain pattern (``brain_mcp.py:264-327``) into a
domain-agnostic form:

* ``header`` — short one-liner: id + title + optional metadata summary
* ``body``   — main content, truncated to ``body_chars`` characters
* ``context`` — relationship/graph context summary (e.g., "referenced by N
  other documents"), optional

Retrieval fuses scores across the three chunks per document. The embedding
model is Ollama-hosted (default: ``mxbai-embed-large``). We call the Ollama
HTTP API directly via ``httpx`` to avoid pulling in the ``ollama`` Python
package and its transitive deps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx

from gr0m_mem.types import DocumentId


@dataclass(frozen=True, slots=True)
class Chunk:
    id: str
    text: str
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Document:
    id: DocumentId
    title: str
    body: str
    context: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def chunk_document(doc: Document, body_chars: int = 1200) -> list[Chunk]:
    """Split a document into (up to) 3 chunks for multi-aspect retrieval.

    The caller is responsible for ensuring chunk ids are unique across the
    corpus — we derive them from ``doc.id`` plus a fixed suffix.
    """
    base_meta: dict[str, Any] = {
        "document_id": str(doc.id),
        **doc.metadata,
    }

    chunks: list[Chunk] = []

    header = f"[{doc.id}] {doc.title}".strip()
    chunks.append(
        Chunk(
            id=f"{doc.id}::header",
            text=header,
            metadata={**base_meta, "chunk_type": "header"},
        )
    )

    body = doc.body[:body_chars].strip()
    if body:
        chunks.append(
            Chunk(
                id=f"{doc.id}::body",
                text=body,
                metadata={**base_meta, "chunk_type": "body"},
            )
        )

    context = doc.context.strip()
    if context:
        chunks.append(
            Chunk(
                id=f"{doc.id}::context",
                text=context[:body_chars],
                metadata={**base_meta, "chunk_type": "context"},
            )
        )

    return chunks


class EmbeddingClient:
    """Ollama embeddings client (mxbai-embed-large by default).

    Kept sync for simplicity in v0.1.0. The MCP server runs each tool call
    in its own task, so one blocking embed call per tool does not stall the
    server. Async + batching land in v0.2.0 with a measured benchmark of the
    delta — not before.
    """

    def __init__(self, url: str, model: str, timeout_s: float = 30.0) -> None:
        self._url = url.rstrip("/")
        self._model = model
        self._client = httpx.Client(timeout=timeout_s)

    @property
    def model(self) -> str:
        return self._model

    def embed(self, text: str) -> list[float]:
        """Embed a single text. Raises ``httpx.HTTPError`` on failure."""
        r = self._client.post(
            f"{self._url}/api/embeddings",
            json={"model": self._model, "prompt": text},
        )
        r.raise_for_status()
        data = r.json()
        embedding = data.get("embedding")
        if not isinstance(embedding, list) or not embedding:
            raise RuntimeError(f"ollama returned empty embedding for model {self._model!r}")
        return embedding

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts sequentially. See class docstring for rationale."""
        return [self.embed(t) for t in texts]

    def health(self) -> bool:
        """Check if Ollama is reachable and the configured model is installed."""
        try:
            r = self._client.get(f"{self._url}/api/tags")
            r.raise_for_status()
            models = [m.get("name", "") for m in r.json().get("models", [])]
            # Ollama model names often have ":latest" suffix — match prefix.
            return any(m.split(":")[0] == self._model.split(":")[0] for m in models)
        except httpx.HTTPError:
            return False

    def close(self) -> None:
        self._client.close()
