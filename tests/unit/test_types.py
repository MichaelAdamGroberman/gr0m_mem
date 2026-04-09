"""Unit tests for gr0m_mem.types."""

from __future__ import annotations

import pytest

from gr0m_mem.types import Corpus, DocumentId


class TestCorpus:
    def test_valid_names(self) -> None:
        for name in ["myproject", "research-notes", "a_b_c", "abc123", "long-name-42"]:
            c = Corpus(name)
            assert str(c) == name

    def test_invalid_names(self) -> None:
        for name in ["", "ab", "A", "has space", "-leading", "trailing-", "has.dot"]:
            with pytest.raises(ValueError):
                Corpus(name)

    def test_collection_name_replaces_hyphens(self) -> None:
        assert Corpus("research-notes").collection_name == "research_notes"
        assert Corpus("plain").collection_name == "plain"


class TestDocumentId:
    def test_valid(self) -> None:
        for v in ["doc1", "uuid-abc-123", "anything printable works!"]:
            assert str(DocumentId(v)) == v

    def test_invalid(self) -> None:
        for v in ["", "   ", "has\nnewline", "has\x00null", "has\ttab"]:
            with pytest.raises(ValueError):
                DocumentId(v)
