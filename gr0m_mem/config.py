"""Runtime configuration for Gr0m_Mem (main branch — zero-install core).

All state lives under ``GR0M_MEM_HOME`` (default ``~/.gr0m_mem``). Each
path is overridable via an env var so tests and power users can
redirect.

The ``main`` branch ships only the SQLite FTS5 backend. The
``GR0M_MEM_BACKEND`` knob still exists and still accepts every backend
name the ``semantic`` branch accepts, but unknown / semantic-only
backends fall back to ``sqlite_fts`` with a warning — the only hard
guarantee this branch makes is "always works, zero extras".

For semantic retrieval backends (chromadb / sqlite_vec) install Gr0m_Mem
from the ``semantic`` branch of the repository.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

VALID_BACKENDS = ("auto", "sqlite_fts")


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
        # Accept any value — the Brain will fall back to sqlite_fts with
        # a warning if the caller asks for a backend this branch does
        # not ship. This keeps the env var interface compatible with the
        # semantic branch for cross-branch configuration.
        backend = os.environ.get("GR0M_MEM_BACKEND", "auto").lower()
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
