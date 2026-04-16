"""Document chunking — backend-agnostic, zero-dependency.

The 3-chunk strategy (header / body / context) is the only part of the
embedding module that is actually backend-independent. Split out here so
the ``main`` branch can ship without any Ollama or embedding-client
code at all.

* ``header`` — short one-liner: id + title + optional metadata summary
* ``body``   — main content, truncated to ``body_chars`` characters
* ``context`` — optional relationship/graph context summary

Empty chunks are dropped — we never pad the collection with meaningless
rows just to hit a "3-chunk strategy" marketing story.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from gr0m_mem.types import DocumentId


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class Document:
    id: DocumentId
    title: str
    body: str
    context: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def chunk_document(doc: Document, body_chars: int = 1200) -> list[Chunk]:
    """Split a document into (up to) 3 chunks for multi-aspect retrieval.

    The caller is responsible for ensuring chunk ids are unique across
    the corpus — we derive them from ``doc.id`` plus a fixed suffix.
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
