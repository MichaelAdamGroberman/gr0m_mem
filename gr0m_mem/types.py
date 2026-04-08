"""Core validated types for Gr0m_Mem.

Kept deliberately general-purpose. No domain-specific taxonomies (security,
medical, legal, ...) live in the core. Callers can attach arbitrary metadata
to documents and edges via the `metadata: dict` field.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Corpus names are opaque tenant identifiers — one ChromaDB collection per corpus.
# We enforce a conservative charset so collection names are stable across
# filesystems and ChromaDB's own naming rules.
_CORPUS_RE = re.compile(r"[a-z0-9][a-z0-9_-]{1,62}[a-z0-9]")


@dataclass(frozen=True, slots=True)
class Corpus:
    """An isolated memory tenant (project, topic, workspace).

    Examples: ``Corpus("myproject")``, ``Corpus("research-notes")``.
    """

    name: str

    def __post_init__(self) -> None:
        if not _CORPUS_RE.fullmatch(self.name):
            raise ValueError(
                f"invalid corpus name {self.name!r}: "
                "must be 3-64 chars, lowercase alnum + - _, start/end alnum"
            )

    def __str__(self) -> str:
        return self.name

    @property
    def collection_name(self) -> str:
        """ChromaDB collection name for this corpus.

        One collection per corpus — never mixed. ``-`` is replaced with ``_``
        because ChromaDB historically disliked hyphens in some versions.
        """
        return self.name.replace("-", "_")


@dataclass(frozen=True, slots=True)
class DocumentId:
    """Opaque document identifier, scoped within a corpus.

    Gr0m_Mem does not interpret the format — callers choose their own scheme
    (UUIDs, hashes, semantic IDs, etc). We only enforce that it's a non-empty
    printable string.
    """

    value: str

    def __post_init__(self) -> None:
        if not self.value or not self.value.strip():
            raise ValueError("DocumentId must be non-empty")
        if any(c in self.value for c in "\x00\n\r\t"):
            raise ValueError(f"DocumentId contains control chars: {self.value!r}")

    def __str__(self) -> str:
        return self.value
