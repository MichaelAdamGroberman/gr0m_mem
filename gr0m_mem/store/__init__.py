"""Storage layer — zero-dependency SQLite FTS5 backend.

The ``main`` branch of Gr0m_Mem ships with only the SQLite FTS5 backend.
This gives ``pip install gr0m-mem`` a working memory brain with zero
extras: FTS5 is part of CPython's built-in ``sqlite3`` on every
mainstream platform.

Semantic retrieval backends (ChromaDB, Ollama + SQLite vector) live on
the ``semantic`` branch. They require either the ``chromadb`` extra or
a running Ollama with the ``mxbai-embed-large`` (~1 GB) model — both of
which are deliberately kept off the default install path so nobody has
to download a gigabyte of weights before the library becomes useful.
"""

from gr0m_mem.store.base import QueryResult, VectorBackend
from gr0m_mem.store.chunking import Chunk, Document, chunk_document
from gr0m_mem.store.sqlite_fts import SqliteFtsBackend

__all__ = [
    "Chunk",
    "Document",
    "QueryResult",
    "SqliteFtsBackend",
    "VectorBackend",
    "chunk_document",
]
