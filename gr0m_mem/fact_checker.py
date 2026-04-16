"""Contradiction detector, wired into :meth:`KnowledgeGraph.add_triple`.

Policy: a new assertion ``(s, p, o, valid_from)`` **contradicts** an
existing currently-active edge ``(s, p, o')`` if:

1. The predicate ``p`` matches, and
2. The existing edge is still open (``valid_to IS NULL``), and
3. The existing ``o'`` differs from the incoming ``o``, and
4. The incoming ``valid_from`` is not strictly earlier than the existing
   ``valid_from`` (otherwise the new assertion is simply historical).

This captures the common case "Kai works_on Orion" followed by "Kai
works_on Nova" without first invalidating the old edge. Callers with
legitimately multi-valued predicates (e.g. ``knows``, ``tagged_with``)
should either use ``fact_check_mode="off"`` on that :class:`KnowledgeGraph`
instance or invalidate the old edge explicitly before adding the new one.

MemPalace shipped its fact-checker as a separate utility that was not
called during writes. Gr0m_Mem calls this during **every** ``add_triple``,
because an unwired safety check is worse than no safety check at all — it
advertises a guarantee it does not provide.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gr0m_mem.graph.store import GraphStore
    from gr0m_mem.graph.temporal import EdgePredicate


@dataclass(frozen=True)
class Contradiction:
    """A single conflict between a proposed and an existing edge."""

    existing_edge_id: str
    subject: str
    predicate: str
    existing_target: str
    new_target: str
    existing_valid_from: datetime

    def message(self) -> str:
        return (
            f"contradiction on ({self.subject}, {self.predicate}): "
            f"existing active edge -> {self.existing_target!r} "
            f"(valid_from={self.existing_valid_from.isoformat()}, "
            f"id={self.existing_edge_id}) conflicts with new target "
            f"{self.new_target!r}. Invalidate the existing edge first, "
            "or switch this KnowledgeGraph to fact_check_mode='off'."
        )


class ContradictionError(ValueError):
    """Raised by :class:`FactChecker` in strict mode when conflicts are found."""

    def __init__(self, conflicts: list[Contradiction]) -> None:
        self.conflicts = conflicts
        super().__init__("; ".join(c.message() for c in conflicts))


class FactChecker:
    """Stateless checker that consults a :class:`GraphStore` on demand."""

    def __init__(self, store: GraphStore) -> None:
        self._store = store

    def check(
        self,
        subject: str,
        predicate: EdgePredicate,
        target: str,
        valid_from: datetime,
    ) -> list[Contradiction]:
        """Return the list of contradictions the new triple would cause.

        Empty list means the triple can be safely inserted.
        """
        active = self._store.active_edges_between(
            source=subject, target=None, predicate=predicate
        )
        conflicts: list[Contradiction] = []
        for edge in active:
            if edge.target == target:
                # Same triple — not a contradiction, just a no-op or update.
                continue
            if valid_from < edge.valid_from:
                # New assertion is historically earlier — it's describing
                # an older state, not conflicting with the current one.
                continue
            conflicts.append(
                Contradiction(
                    existing_edge_id=edge.id,
                    subject=subject,
                    predicate=str(predicate),
                    existing_target=edge.target,
                    new_target=target,
                    existing_valid_from=edge.valid_from,
                )
            )
        return conflicts
