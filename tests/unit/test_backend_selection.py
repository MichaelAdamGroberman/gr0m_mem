"""Backend auto-selection cascade."""

from __future__ import annotations

from pathlib import Path

import pytest

from gr0m_mem.brain import _select_backend
from gr0m_mem.config import Config


def _config(tmp_path: Path, backend: str) -> Config:
    return Config(
        home=tmp_path,
        chroma_path=tmp_path / "chroma_db",
        sqlite_vec_path=tmp_path / "vectors.db",
        sqlite_fts_path=tmp_path / "fts.db",
        graph_db_path=tmp_path / "graph.db",
        state_db_path=tmp_path / "state.db",
        wakeup_db_path=tmp_path / "wakeup.db",
        ollama_url="http://127.0.0.1:1",  # guaranteed unreachable
        embed_model="mxbai-embed-large",
        fact_check_mode="strict",
        backend=backend,
    )


def test_forced_sqlite_fts(tmp_path: Path) -> None:
    cfg = _config(tmp_path, "sqlite_fts")
    backend, choice = _select_backend(cfg)
    assert choice.name == "sqlite_fts"
    assert "explicitly requested" in choice.reason
    assert backend.requires_embeddings is False
    backend.close()


def test_forced_sqlite_vec(tmp_path: Path) -> None:
    cfg = _config(tmp_path, "sqlite_vec")
    backend, choice = _select_backend(cfg)
    assert choice.name == "sqlite_vec"
    assert backend.requires_embeddings is True
    backend.close()


def test_auto_falls_back_to_fts_when_ollama_unreachable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Pretend chromadb is missing so auto skips it.
    monkeypatch.setattr("gr0m_mem.brain.chromadb_available", lambda: False)
    cfg = _config(tmp_path, "auto")
    backend, choice = _select_backend(cfg)
    assert choice.name == "sqlite_fts"
    assert "unreachable" in choice.reason
    backend.close()


def test_auto_prefers_chromadb_when_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Only run if chromadb is actually installed — otherwise skip, because
    # we refuse to lie about availability in tests.
    from importlib.util import find_spec

    if find_spec("chromadb") is None:
        pytest.skip("chromadb not installed; auto cascade tested in other tests")
    cfg = _config(tmp_path, "auto")
    backend, choice = _select_backend(cfg)
    assert choice.name == "chromadb"
    backend.close()


def test_forced_chromadb_errors_when_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("gr0m_mem.store.chroma.chromadb_available", lambda: False)
    cfg = _config(tmp_path, "chromadb")
    with pytest.raises(RuntimeError, match="chromadb"):
        _select_backend(cfg)
