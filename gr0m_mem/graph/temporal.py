"""Temporal edge type + validity helpers.

The graph is an open, domain-agnostic triple store: any caller can invent
its own predicate vocabulary. We keep ``EdgePredicate`` as a plain string
newtype (not an enum) so the core has zero baked-in taxonomy — no security
categories, no people-relationship verbs, nothing.

Time is always UTC, always timezone-aware. Naive datetimes are rejected at
construction time to avoid the silent timezone drift that bit MemPalace.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import uuid4

_PREDICATE_RE = re.compile(r"[a-z][a-z0-9_]{0,62}")


@dataclass(frozen=True)
class EdgePredicate:
    """An opaque relationship name. Lowercase snake_case, 1-63 chars."""

    name: str

    def __post_init__(self) -> None:
        if not _PREDICATE_RE.fullmatch(self.name):
            raise ValueError(
                f"invalid predicate {self.name!r}: "
                "must be lowercase snake_case, start with a letter, 1-63 chars"
            )

    def __str__(self) -> str:
        return self.name


def _require_aware(dt: datetime, field_name: str) -> datetime:
    if dt.tzinfo is None:
        raise ValueError(f"{field_name} must be timezone-aware (got naive {dt!r})")
    return dt


@dataclass(frozen=True)
class TemporalEdge:
    """A triple with a validity window.

    * ``valid_from`` — when the fact became true (UTC-aware)
    * ``valid_to``   — when the fact stopped being true, or ``None`` if still current
    * ``confidence`` — caller-supplied 0.0..=1.0
    * ``weight``     — caller-supplied score (e.g., for shortest-path traversal)
    * ``source_doc`` — optional pointer back to a document id in ChromaDB
    * ``data``       — arbitrary JSON metadata (no schema imposed by Gr0m_Mem)
    """

    source: str
    target: str
    predicate: EdgePredicate
    valid_from: datetime
    valid_to: datetime | None = None
    weight: float = 1.0
    confidence: float = 1.0
    source_doc: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid4()))

    def __post_init__(self) -> None:
        if not self.source or not self.target:
            raise ValueError("source and target must be non-empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence {self.confidence} out of [0,1]")
        _require_aware(self.valid_from, "valid_from")
        if self.valid_to is not None:
            _require_aware(self.valid_to, "valid_to")
            if self.valid_to < self.valid_from:
                raise ValueError(
                    f"valid_to {self.valid_to} precedes valid_from {self.valid_from}"
                )

    def is_active_at(self, as_of: datetime | None) -> bool:
        """Return True if this edge is valid at ``as_of``.

        ``as_of=None`` means "currently valid" — i.e., the edge has no
        ``valid_to`` (it is open-ended from ``valid_from`` to now).
        """
        if as_of is None:
            return self.valid_to is None
        _require_aware(as_of, "as_of")
        if as_of < self.valid_from:
            return False
        return not (self.valid_to is not None and as_of > self.valid_to)
