"""Backend selection on the main (zero-install) branch.

The main branch ships with only the SQLite FTS5 backend. The selection
function preserves the same return shape as the semantic branch so
diagnostic tools (``mem_status``) and cross-branch configs stay
compatible, but it always lands on ``sqlite_fts`` regardless of what
the caller requested.
"""

from __future__ import annotations

from pathlib import Path

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
        ollama_url="http://127.0.0.1:1",
        embed_model="mxbai-embed-large",
        fact_check_mode="strict",
        backend=backend,
    )


def test_auto_always_returns_sqlite_fts(tmp_path: Path) -> None:
    backend, choice = _select_backend(_config(tmp_path, "auto"))
    assert choice.name == "sqlite_fts"
    assert backend.requires_embeddings is False
    assert "zero-install" in choice.reason or "main branch" in choice.reason
    backend.close()


def test_forced_sqlite_fts(tmp_path: Path) -> None:
    backend, choice = _select_backend(_config(tmp_path, "sqlite_fts"))
    assert choice.name == "sqlite_fts"
    backend.close()


def test_semantic_backend_request_falls_back(tmp_path: Path) -> None:
    """Asking for chromadb on the main branch must not crash.

    Cross-branch configs (e.g. a shared ``.env`` that sets
    ``GR0M_MEM_BACKEND=chromadb`` for the semantic branch) should still
    start successfully on the main branch, just with a warning and the
    FTS fallback.
    """
    backend, choice = _select_backend(_config(tmp_path, "chromadb"))
    assert choice.name == "sqlite_fts"
    backend.close()

    backend2, choice2 = _select_backend(_config(tmp_path, "sqlite_vec"))
    assert choice2.name == "sqlite_fts"
    backend2.close()
