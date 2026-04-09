"""Brain.search score fusion — scores must stay in [0, 1]."""

from __future__ import annotations

from pathlib import Path

import pytest

from gr0m_mem.brain import Brain
from gr0m_mem.config import Config


def _cfg(tmp_path: Path) -> Config:
    return Config(
        home=tmp_path,
        chroma_path=tmp_path / "chroma_db",
        sqlite_vec_path=tmp_path / "vectors.db",
        sqlite_fts_path=tmp_path / "fts.db",
        graph_db_path=tmp_path / "graph.db",
        state_db_path=tmp_path / "state.db",
        wakeup_db_path=tmp_path / "wakeup.db",
        ollama_url="http://127.0.0.1:1",
        embed_model="mxbai-embed-large",
        fact_check_mode="strict",
        backend="sqlite_fts",
    )


@pytest.fixture
def brain(tmp_path: Path) -> Brain:
    b = Brain(_cfg(tmp_path))
    yield b
    b.close()


def test_score_stays_in_unit_range(brain: Brain) -> None:
    """A document whose header and body both match must not exceed score 1.0."""
    from gr0m_mem.types import Corpus

    brain.learn(
        corpus=Corpus("notes"),
        document_id="d1",
        title="Backend choice",
        body="We picked backend zero dep default so backend always works",
    )
    hits = brain.search(Corpus("notes"), "backend zero dep", n_results=5)
    assert hits, "expected at least one hit"
    for h in hits:
        assert 0.0 <= h.score <= 1.0, f"score out of range: {h.score}"


def test_fts_backend_used_without_ollama(brain: Brain) -> None:
    """Confirm auto-selection actually lands on sqlite_fts when forced."""
    assert brain.backend.name == "sqlite_fts"
    assert brain.backend.requires_embeddings is False
    assert brain.embedding is None
