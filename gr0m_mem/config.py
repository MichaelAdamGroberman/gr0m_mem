"""Runtime configuration for Gr0m_Mem.

All state lives under ``GR0M_MEM_HOME`` (default ``~/.gr0m_mem``). Each path
is overridable via an env var so tests and power users can redirect.

The vector backend is chosen via ``GR0M_MEM_BACKEND``:

* ``auto`` (default) — chromadb if installed, else sqlite_vec if Ollama is
  reachable, else sqlite_fts (lexical). This means ``pip install gr0m-mem``
  always produces a working brain, even with no extras and no Ollama.
* ``chromadb``   — force chromadb; errors if not installed.
* ``sqlite_vec`` — SQLite rows + Python cosine over Ollama embeddings.
* ``sqlite_fts`` — SQLite FTS5 BM25. No embeddings required.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

VALID_BACKENDS = ("auto", "chromadb", "sqlite_vec", "sqlite_fts")


def _env_path(var: str, default: Path) -> Path:
    val = os.environ.get(var)
    return Path(val).expanduser() if val else default


@dataclass(frozen=True, slots=True)
class Config:
    home: Path
    chroma_path: Path
    sqlite_vec_path: Path
    sqlite_fts_path: Path
    graph_db_path: Path
    state_db_path: Path
    wakeup_db_path: Path
    ollama_url: str
    embed_model: str
    fact_check_mode: str  # "strict" | "warn" | "off"
    backend: str  # "auto" | "chromadb" | "sqlite_vec" | "sqlite_fts"

    @classmethod
    def from_env(cls) -> Config:
        home = _env_path("GR0M_MEM_HOME", Path.home() / ".gr0m_mem")
        backend = os.environ.get("GR0M_MEM_BACKEND", "auto").lower()
        if backend not in VALID_BACKENDS:
            raise ValueError(
                f"GR0M_MEM_BACKEND={backend!r} must be one of {VALID_BACKENDS}"
            )
        return cls(
            home=home,
            chroma_path=_env_path("GR0M_MEM_CHROMA_PATH", home / "chroma_db"),
            sqlite_vec_path=_env_path("GR0M_MEM_SQLITE_VEC", home / "vectors.db"),
            sqlite_fts_path=_env_path("GR0M_MEM_SQLITE_FTS", home / "fts.db"),
            graph_db_path=_env_path("GR0M_MEM_GRAPH_DB", home / "graph.db"),
            state_db_path=_env_path("GR0M_MEM_STATE_DB", home / "state.db"),
            wakeup_db_path=_env_path("GR0M_MEM_WAKEUP_DB", home / "wakeup.db"),
            ollama_url=os.environ.get("GR0M_MEM_OLLAMA_URL", "http://localhost:11434"),
            embed_model=os.environ.get("GR0M_MEM_EMBED_MODEL", "mxbai-embed-large"),
            fact_check_mode=os.environ.get("GR0M_MEM_FACT_CHECK_MODE", "strict"),
            backend=backend,
        )

    def ensure_dirs(self) -> None:
        """Create the directories needed for persistence. Idempotent."""
        self.home.mkdir(parents=True, exist_ok=True)
        self.chroma_path.mkdir(parents=True, exist_ok=True)
        for p in (
            self.sqlite_vec_path,
            self.sqlite_fts_path,
            self.graph_db_path,
            self.state_db_path,
            self.wakeup_db_path,
        ):
            p.parent.mkdir(parents=True, exist_ok=True)
