"""Temporal-aware graph traversal.

The only way to get a NetworkX view of the graph is through
:func:`active_view`, which **requires** the caller to explicitly decide
what temporal slice they want. ``as_of=None`` is a valid, explicit choice
(it means "currently valid"). There is no unfiltered path.

This is the design point MemPalace fell short on: in MemPalace the
contradiction utility existed but was not wired into queries. Here we
refuse to compile a graph that ignores time.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import networkx as nx

if TYPE_CHECKING:
    from gr0m_mem.graph.temporal import TemporalEdge


_MISSING = object()


def active_view(
    edges: list[TemporalEdge],
    as_of: datetime | None | object = _MISSING,
) -> nx.MultiDiGraph:
    """Build a NetworkX MultiDiGraph of the edges valid at ``as_of``.

    Parameters
    ----------
    edges:
        The full list of candidate edges (typically from
        :meth:`gr0m_mem.graph.store.GraphStore.all_edges`).
    as_of:
        * ``None`` — currently valid (open-ended) edges only
        * a UTC-aware ``datetime`` — edges active at that point in time
        * (sentinel) if omitted entirely, raises ``TypeError`` — callers
          must make the temporal choice explicit.

    Returns
    -------
    ``nx.MultiDiGraph`` — a fresh graph on every call. We use MultiDiGraph
    because two nodes can be connected by many predicates, and historical
    queries may return multiple edges on the same (source, target) pair.
    """
    if as_of is _MISSING:
        raise TypeError(
            "active_view() requires an explicit `as_of` argument "
            "(pass `as_of=None` for currently-valid edges, or a "
            "timezone-aware datetime for a historical slice)"
        )
    if as_of is not None and not isinstance(as_of, datetime):
        raise TypeError(f"as_of must be datetime or None, got {type(as_of).__name__}")

    g: nx.MultiDiGraph = nx.MultiDiGraph()
    for edge in edges:
        if not edge.is_active_at(as_of):
            continue
        g.add_edge(
            edge.source,
            edge.target,
            key=edge.id,
            predicate=str(edge.predicate),
            weight=edge.weight,
            confidence=edge.confidence,
            valid_from=edge.valid_from.isoformat(),
            valid_to=edge.valid_to.isoformat() if edge.valid_to else None,
            source_doc=edge.source_doc,
            data=edge.data,
        )
    return g
