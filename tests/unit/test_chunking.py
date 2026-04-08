"""Unit tests for the 3-chunk strategy."""

from __future__ import annotations

from gr0m_mem.store.embedding import Document, chunk_document
from gr0m_mem.types import DocumentId


def _doc(body: str = "body text", context: str = "") -> Document:
    return Document(
        id=DocumentId("doc1"),
        title="Hello world",
        body=body,
        context=context,
        metadata={"custom": "meta"},
    )


def test_all_three_chunks_when_all_fields_present() -> None:
    chunks = chunk_document(_doc(body="a body", context="a context"))
    kinds = [c.metadata["chunk_type"] for c in chunks]
    assert kinds == ["header", "body", "context"]
    ids = [c.id for c in chunks]
    assert ids == ["doc1::header", "doc1::body", "doc1::context"]


def test_empty_context_omits_context_chunk() -> None:
    chunks = chunk_document(_doc(context=""))
    kinds = [c.metadata["chunk_type"] for c in chunks]
    assert "context" not in kinds
    assert kinds == ["header", "body"]


def test_empty_body_omits_body_chunk() -> None:
    chunks = chunk_document(_doc(body="   "))
    kinds = [c.metadata["chunk_type"] for c in chunks]
    assert kinds == ["header"]


def test_body_truncation() -> None:
    long = "x" * 5000
    chunks = chunk_document(_doc(body=long), body_chars=500)
    body_chunk = next(c for c in chunks if c.metadata["chunk_type"] == "body")
    assert len(body_chunk.text) == 500


def test_metadata_carried_through() -> None:
    chunks = chunk_document(_doc())
    for c in chunks:
        assert c.metadata["document_id"] == "doc1"
        assert c.metadata["custom"] == "meta"
