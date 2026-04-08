"""Temporal knowledge graph with validity-window semantics.

Every edge has a ``valid_from`` and an optional ``valid_to``. Traversal
cannot happen without a caller explicitly deciding on the temporal filter —
see :func:`gr0m_mem.graph.traverse.active_view`. This is the feature
MemPalace's README promised and never wired; we make it unavoidable here.
"""

from gr0m_mem.graph.kg import KnowledgeGraph
from gr0m_mem.graph.store import GraphStore
from gr0m_mem.graph.temporal import EdgePredicate, TemporalEdge
from gr0m_mem.graph.traverse import active_view

__all__ = [
    "EdgePredicate",
    "GraphStore",
    "KnowledgeGraph",
    "TemporalEdge",
    "active_view",
]
