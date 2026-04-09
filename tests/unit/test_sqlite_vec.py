"""SqliteVectorBackend — pure-SQLite cosine similarity."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from gr0m_mem.store.sqlite_vec import (
    SqliteVectorBackend,
    _cosine,
    _pack_embedding,
    _unpack_embedding,
)
from gr0m_mem.types import Corpus


@pytest.fixture
def backend(tmp_path: Path) -> SqliteVectorBackend:
    return SqliteVectorBackend(tmp_path / "vec.db")


def test_cosine_identical() -> None:
    assert _cosine([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]) == pytest.approx(1.0)


def test_cosine_orthogonal() -> None:
    assert _cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_opposite() -> None:
    assert _cosine([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_cosine_length_mismatch_returns_zero() -> None:
    assert _cosine([1.0, 0.0], [1.0, 0.0, 0.0]) == 0.0


def test_pack_roundtrip() -> None:
    original = [0.1, -0.2, 3.14159, 0.0, 100.0]
    blob = _pack_embedding(original)
    restored = _unpack_embedding(blob, len(original))
    for a, b in zip(original, restored, strict=True):
        assert math.isclose(a, b, rel_tol=1e-6)


class TestVectorBackend:
    def test_add_and_query(self, backend: SqliteVectorBackend) -> None:
        corpus = Corpus("notes")
        backend.add(
            corpus=corpus,
            ids=["doc1::body", "doc2::body"],
            documents=["auth decision", "db decision"],
            embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            metadatas=[
                {"document_id": "doc1", "chunk_type": "body"},
                {"document_id": "doc2", "chunk_type": "body"},
            ],
        )

        result = backend.query(
            corpus=corpus,
            query_text="unused",
            query_embedding=[1.0, 0.0, 0.0],
            n_results=5,
        )
        assert result.ids[0] == "doc1::body"
        assert result.scores[0] > result.scores[1]

    def test_requires_embeddings(self, backend: SqliteVectorBackend) -> None:
        with pytest.raises(ValueError, match="requires embeddings"):
            backend.add(
                corpus=Corpus("notes"),
                ids=["x"],
                documents=["hi"],
                embeddings=None,
                metadatas=[{}],
            )
        with pytest.raises(ValueError, match="query_embedding"):
            backend.query(Corpus("notes"), "x", None, 5)

    def test_upsert_updates_existing(self, backend: SqliteVectorBackend) -> None:
        corpus = Corpus("notes")
        backend.add(
            corpus=corpus,
            ids=["doc::body"],
            documents=["v1"],
            embeddings=[[1.0, 0.0]],
            metadatas=[{"document_id": "doc", "chunk_type": "body"}],
        )
        backend.add(
            corpus=corpus,
            ids=["doc::body"],
            documents=["v2"],
            embeddings=[[0.5, 0.5]],
            metadatas=[{"document_id": "doc", "chunk_type": "body"}],
        )
        assert backend.count(corpus) == 1
        result = backend.query(corpus, "x", [0.5, 0.5], 5)
        assert result.documents == ["v2"]

    def test_corpus_isolation(self, backend: SqliteVectorBackend) -> None:
        a = Corpus("project-a")
        b = Corpus("project-b")
        backend.add(a, ["id1"], ["a doc"], [[1.0, 0.0]], [{"document_id": "id1"}])
        backend.add(b, ["id1"], ["b doc"], [[1.0, 0.0]], [{"document_id": "id1"}])

        ra = backend.query(a, "x", [1.0, 0.0], 5)
        rb = backend.query(b, "x", [1.0, 0.0], 5)
        assert ra.documents == ["a doc"]
        assert rb.documents == ["b doc"]

    def test_delete(self, backend: SqliteVectorBackend) -> None:
        corpus = Corpus("notes")
        backend.add(
            corpus,
            ["a", "b"],
            ["x", "y"],
            [[1.0, 0.0], [0.0, 1.0]],
            [{}, {}],
        )
        backend.delete(corpus, ["a"])
        assert backend.count(corpus) == 1
