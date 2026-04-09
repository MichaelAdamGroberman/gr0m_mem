"""GraphStore persistence + temporal queries."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from gr0m_mem.graph.store import GraphStore
from gr0m_mem.graph.temporal import EdgePredicate, TemporalEdge

UTC = timezone.utc


@pytest.fixture
def store(tmp_path: Path) -> GraphStore:
    return GraphStore(tmp_path / "graph.db")


def _edge(s: str, p: str, t: str, vf: datetime, vt: datetime | None = None) -> TemporalEdge:
    return TemporalEdge(
        source=s,
        target=t,
        predicate=EdgePredicate(p),
        valid_from=vf,
        valid_to=vt,
    )


def test_insert_and_retrieve(store: GraphStore) -> None:
    e = _edge("Kai", "works_on", "Orion", datetime(2025, 6, 1, tzinfo=UTC))
    store.insert(e)
    all_edges = store.all_edges()
    assert len(all_edges) == 1
    assert all_edges[0].source == "Kai"
    assert all_edges[0].predicate.name == "works_on"


def test_invalidate_sets_valid_to(store: GraphStore) -> None:
    start = datetime(2025, 6, 1, tzinfo=UTC)
    end = datetime(2026, 3, 1, tzinfo=UTC)
    store.insert(_edge("Kai", "works_on", "Orion", start))
    n = store.invalidate("Kai", "Orion", EdgePredicate("works_on"), ended_at=end)
    assert n == 1
    edges = store.all_edges()
    assert len(edges) == 1  # NOT deleted
    assert edges[0].valid_to == end


def test_historical_slice(store: GraphStore) -> None:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    end = datetime(2025, 12, 31, tzinfo=UTC)
    store.insert(_edge("Kai", "works_on", "Orion", start, vt=end))
    store.insert(_edge("Kai", "works_on", "Nova", datetime(2026, 1, 1, tzinfo=UTC)))

    # Current — only Nova (Orion is closed).
    current = store.edges_at(None)
    assert len(current) == 1
    assert current[0].target == "Nova"

    # Mid-2025 — only Orion.
    past = store.edges_at(datetime(2025, 6, 1, tzinfo=UTC))
    assert len(past) == 1
    assert past[0].target == "Orion"


def test_edges_for_entity_direction(store: GraphStore) -> None:
    vf = datetime(2025, 1, 1, tzinfo=UTC)
    store.insert(_edge("Kai", "works_on", "Orion", vf))
    store.insert(_edge("Maya", "knows", "Kai", vf))

    out = store.edges_for_entity("Kai", None, direction="out")
    assert [e.target for e in out] == ["Orion"]

    inn = store.edges_for_entity("Kai", None, direction="in")
    assert [e.source for e in inn] == ["Maya"]

    both = store.edges_for_entity("Kai", None, direction="both")
    assert len(both) == 2


def test_count(store: GraphStore) -> None:
    vf = datetime(2025, 1, 1, tzinfo=UTC)
    store.insert(_edge("a", "p", "b", vf))
    store.insert(_edge("a", "p", "c", vf, vt=datetime(2025, 6, 1, tzinfo=UTC)))
    counts = store.count()
    assert counts == {"total": 2, "active": 1, "closed": 1}
