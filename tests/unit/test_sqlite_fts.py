"""SqliteFtsBackend — the zero-dep fallback."""

from __future__ import annotations

from pathlib import Path

import pytest

from gr0m_mem.store.sqlite_fts import SqliteFtsBackend, _escape_fts_query
from gr0m_mem.types import Corpus


@pytest.fixture
def backend(tmp_path: Path) -> SqliteFtsBackend:
    return SqliteFtsBackend(tmp_path / "fts.db")


def _add(backend: SqliteFtsBackend, corpus: Corpus, cid: str, text: str) -> None:
    backend.add(
        corpus=corpus,
        ids=[cid],
        documents=[text],
        embeddings=None,
        metadatas=[{"document_id": cid.split("::")[0], "chunk_type": "body"}],
    )


class TestEscape:
    def test_basic_tokens(self) -> None:
        assert _escape_fts_query("hello world") == '"hello" OR "world"'

    def test_strips_punctuation(self) -> None:
        assert _escape_fts_query("what about SSRF!") == '"what" OR "about" OR "ssrf"'

    def test_short_tokens_dropped(self) -> None:
        assert _escape_fts_query("a b car") == '"car"'

    def test_empty_returns_never_match(self) -> None:
        assert _escape_fts_query("") == '""'
        assert _escape_fts_query("!!!") == '""'


class TestFtsBackend:
    def test_add_and_search(self, backend: SqliteFtsBackend) -> None:
        corpus = Corpus("notes")
        _add(backend, corpus, "auth::body", "we picked Clerk for authentication")
        _add(backend, corpus, "db::body", "we picked Postgres for the database")

        result = backend.query(
            corpus=corpus,
            query_text="authentication decision",
            query_embedding=None,
            n_results=5,
        )
        assert result.ids
        assert result.ids[0] == "auth::body"
        assert 0.0 <= result.scores[0] <= 1.0

    def test_no_match_returns_empty(self, backend: SqliteFtsBackend) -> None:
        corpus = Corpus("notes")
        _add(backend, corpus, "doc::body", "Postgres over SQLite")
        result = backend.query(
            corpus=corpus,
            query_text="kubernetes",
            query_embedding=None,
            n_results=5,
        )
        assert len(result) == 0

    def test_corpus_isolation(self, backend: SqliteFtsBackend) -> None:
        a = Corpus("project-a")
        b = Corpus("project-b")
        _add(backend, a, "doc1::body", "authentication via Clerk")
        _add(backend, b, "doc2::body", "authentication via Auth0")

        ra = backend.query(a, "authentication", None, 5)
        rb = backend.query(b, "authentication", None, 5)
        assert ra.ids == ["doc1::body"]
        assert rb.ids == ["doc2::body"]
        assert sorted(backend.list_corpora()) == ["project-a", "project-b"]

    def test_delete(self, backend: SqliteFtsBackend) -> None:
        corpus = Corpus("notes")
        _add(backend, corpus, "a::body", "keep this")
        _add(backend, corpus, "b::body", "delete this")
        backend.delete(corpus, ["b::body"])
        assert backend.count(corpus) == 1

    def test_requires_embeddings_is_false(self, backend: SqliteFtsBackend) -> None:
        assert backend.requires_embeddings is False

    def test_sample_metadatas(self, backend: SqliteFtsBackend) -> None:
        corpus = Corpus("notes")
        _add(backend, corpus, "a::body", "hello")
        _add(backend, corpus, "b::body", "world")
        metas = backend.sample_metadatas(corpus, limit=10)
        assert len(metas) == 2
        assert {m["document_id"] for m in metas} == {"a", "b"}
