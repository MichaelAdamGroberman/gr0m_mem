"""Storage layer — three backends with auto-selection."""

from gr0m_mem.store.base import QueryResult, VectorBackend
from gr0m_mem.store.chroma import ChromaBackend, chromadb_available
from gr0m_mem.store.embedding import EmbeddingClient, chunk_document
from gr0m_mem.store.sqlite_fts import SqliteFtsBackend
from gr0m_mem.store.sqlite_vec import SqliteVectorBackend

__all__ = [
    "ChromaBackend",
    "EmbeddingClient",
    "QueryResult",
    "SqliteFtsBackend",
    "SqliteVectorBackend",
    "VectorBackend",
    "chromadb_available",
    "chunk_document",
]
