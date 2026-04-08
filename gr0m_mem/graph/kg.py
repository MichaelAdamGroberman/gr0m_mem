"""High-level :class:`KnowledgeGraph` facade.

This is the only class callers (CLI, MCP server, Python API) should need
from the ``graph`` package. It owns a :class:`GraphStore` and a
:class:`FactChecker`, wires them together, and exposes the temporal-first
API (``add_triple``, ``invalidate``, ``query_entity``, ``timeline``,
``stats``, ``traverse``).

Every traversal path here goes through
:func:`gr0m_mem.graph.traverse.active_view`, which forces an explicit
``as_of`` decision. The facade exposes that decision as a required
argument too — we do not paper over it with a ``None`` default that reads
like "doesn't matter".
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import networkx as nx

from gr0m_mem.fact_checker import Contradiction, ContradictionError, FactChecker
from gr0m_mem.graph.store import GraphStore
from gr0m_mem.graph.temporal import EdgePredicate, TemporalEdge
from gr0m_mem.graph.traverse import active_view

log = logging.getLogger("gr0m_mem.graph")


_VALID_MODES = ("strict", "warn", "off")


class KnowledgeGraph:
    """Temporal KG with fact-checking wired into every write."""

    def __init__(
        self,
        db_path: Path,
        fact_check_mode: str = "strict",
    ) -> None:
        if fact_check_mode not in _VALID_MODES:
            raise ValueError(
                f"fact_check_mode must be one of {_VALID_MODES}, got {fact_check_mode!r}"
            )
        self._store = GraphStore(db_path)
        self._checker = FactChecker(self._store)
        self._mode = fact_check_mode

    @property
    def store(self) -> GraphStore:
        return self._store

    @property
    def fact_check_mode(self) -> str:
        return self._mode

    # ── Writes ────────────────────────────────────────────

    def add_triple(
        self,
        subject: str,
        predicate: str,
        target: str,
        *,
        valid_from: datetime | None = None,
        weight: float = 1.0,
        confidence: float = 1.0,
        source_doc: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> TemporalEdge:
        """Add a triple, running the fact-checker first.

        In ``strict`` mode (default), a contradiction raises
        :class:`ContradictionError` and no edge is inserted. In ``warn``
        mode a log line is emitted and the edge is inserted anyway. In
        ``off`` mode the checker is skipped entirely.
        """
        pred = EdgePredicate(predicate)
        vf = valid_from or datetime.now(timezone.utc)

        conflicts: list[Contradiction] = []
        if self._mode != "off":
            conflicts = self._checker.check(
                subject=subject,
                predicate=pred,
                target=target,
                valid_from=vf,
            )
            if conflicts and self._mode == "strict":
                raise ContradictionError(conflicts)
            if conflicts and self._mode == "warn":
                for c in conflicts:
                    log.warning("fact_checker: %s", c.message())

        edge = TemporalEdge(
            source=subject,
            target=target,
            predicate=pred,
            valid_from=vf,
            weight=weight,
            confidence=confidence,
            source_doc=source_doc,
            data=data or {},
        )
        self._store.insert(edge)
        return edge

    def invalidate(
        self,
        subject: str,
        predicate: str,
        target: str,
        ended_at: datetime | None = None,
    ) -> int:
        """Mark every currently-active matching edge as ended at ``ended_at``.

        Does not delete. Returns the number of edges updated.
        """
        return self._store.invalidate(
            source=subject,
            target=target,
            predicate=EdgePredicate(predicate),
            ended_at=ended_at or datetime.now(timezone.utc),
        )

    # ── Reads ─────────────────────────────────────────────

    def query_entity(
        self,
        entity: str,
        *,
        as_of: datetime | None,
        direction: str = "both",
    ) -> list[TemporalEdge]:
        """Return edges touching ``entity`` active at ``as_of``.

        ``as_of`` is required — callers must decide whether they want the
        current state (``None``) or a historical slice.
        """
        return self._store.edges_for_entity(
            entity=entity, as_of=as_of, direction=direction
        )

    def timeline(self, entity: str | None = None) -> list[TemporalEdge]:
        """Return every edge touching ``entity`` (or all edges, if ``None``),
        ordered by ``valid_from``.

        Unlike :meth:`query_entity`, this does not filter by time — it's
        the literal full history. Intended for audit, visualisation, and
        the ``mem_kg_timeline`` MCP tool.
        """
        edges = self._store.all_edges()
        if entity is not None:
            edges = [e for e in edges if e.source == entity or e.target == entity]
        edges.sort(key=lambda e: e.valid_from)
        return edges

    def traverse(self, *, as_of: datetime | None) -> nx.MultiDiGraph:
        """Return a NetworkX MultiDiGraph of edges active at ``as_of``.

        Callers can then use NetworkX algorithms (shortest path, BFS, …).
        Every call returns a fresh view — we do not cache because the
        store is mutable and stale views are worse than rebuild cost for
        the scales Gr0m_Mem targets.
        """
        return active_view(self._store.all_edges(), as_of=as_of)

    def stats(self) -> dict[str, int]:
        return self._store.count()

    def close(self) -> None:
        self._store.close()
