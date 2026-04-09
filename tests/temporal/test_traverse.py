"""active_view — the mandatory temporal-decision traversal API."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from gr0m_mem.graph.temporal import EdgePredicate, TemporalEdge
from gr0m_mem.graph.traverse import active_view

UTC = timezone.utc


def _edge(s: str, t: str, vf: datetime, vt: datetime | None = None) -> TemporalEdge:
    return TemporalEdge(
        source=s,
        target=t,
        predicate=EdgePredicate("rel"),
        valid_from=vf,
        valid_to=vt,
    )


def test_active_view_requires_explicit_as_of() -> None:
    with pytest.raises(TypeError, match="explicit `as_of`"):
        active_view([])  # type: ignore[call-arg]


def test_active_view_rejects_non_datetime() -> None:
    with pytest.raises(TypeError, match="datetime"):
        active_view([], as_of="2025-01-01")  # type: ignore[arg-type]


def test_active_view_none_returns_current_edges() -> None:
    vf = datetime(2025, 1, 1, tzinfo=UTC)
    edges = [
        _edge("a", "b", vf),
        _edge("c", "d", vf, vt=datetime(2025, 6, 1, tzinfo=UTC)),  # closed
    ]
    g = active_view(edges, as_of=None)
    assert g.number_of_edges() == 1
    assert g.has_edge("a", "b")
    assert not g.has_edge("c", "d")


def test_active_view_historical_slice() -> None:
    e1 = _edge(
        "a",
        "b",
        vf=datetime(2025, 1, 1, tzinfo=UTC),
        vt=datetime(2025, 12, 31, tzinfo=UTC),
    )
    e2 = _edge("c", "d", vf=datetime(2026, 1, 1, tzinfo=UTC))
    edges = [e1, e2]

    mid_2025 = datetime(2025, 6, 1, tzinfo=UTC)
    g = active_view(edges, as_of=mid_2025)
    assert g.has_edge("a", "b")
    assert not g.has_edge("c", "d")


def test_active_view_is_multidigraph() -> None:
    """Two predicates on the same (s, t) pair must both survive."""
    vf = datetime(2025, 1, 1, tzinfo=UTC)
    edges = [
        TemporalEdge(source="a", target="b", predicate=EdgePredicate("knows"), valid_from=vf),
        TemporalEdge(source="a", target="b", predicate=EdgePredicate("trusts"), valid_from=vf),
    ]
    g = active_view(edges, as_of=None)
    assert g.number_of_edges() == 2
    preds = {d["predicate"] for _, _, d in g.edges(data=True)}
    assert preds == {"knows", "trusts"}
