"""KnowledgeGraph facade + FactChecker wiring."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import pytest

from gr0m_mem.fact_checker import ContradictionError
from gr0m_mem.graph.kg import KnowledgeGraph

UTC = timezone.utc


@pytest.fixture
def kg(tmp_path: Path) -> KnowledgeGraph:
    return KnowledgeGraph(tmp_path / "graph.db", fact_check_mode="strict")


def _now() -> datetime:
    return datetime.now(tz=UTC)


def test_add_and_query_current(kg: KnowledgeGraph) -> None:
    kg.add_triple("Kai", "works_on", "Orion", valid_from=_now())
    edges = kg.query_entity("Kai", as_of=None)
    assert len(edges) == 1
    assert edges[0].target == "Orion"


def test_query_entity_requires_keyword_as_of(kg: KnowledgeGraph) -> None:
    # Positional call should fail — as_of is keyword-only.
    with pytest.raises(TypeError):
        kg.query_entity("Kai")  # type: ignore[call-arg]


def test_contradiction_strict_mode_raises(kg: KnowledgeGraph) -> None:
    start = datetime(2025, 6, 1, tzinfo=UTC)
    kg.add_triple("Kai", "works_on", "Orion", valid_from=start)
    with pytest.raises(ContradictionError) as exc:
        kg.add_triple(
            "Kai", "works_on", "Nova", valid_from=datetime(2026, 3, 1, tzinfo=UTC)
        )
    assert exc.value.conflicts
    assert exc.value.conflicts[0].existing_target == "Orion"
    assert exc.value.conflicts[0].new_target == "Nova"
    # Nothing inserted on failure.
    assert kg.stats()["total"] == 1


def test_contradiction_warn_mode_inserts_anyway(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    kg = KnowledgeGraph(tmp_path / "g.db", fact_check_mode="warn")
    start = datetime(2025, 6, 1, tzinfo=UTC)
    kg.add_triple("Kai", "works_on", "Orion", valid_from=start)
    with caplog.at_level(logging.WARNING, logger="gr0m_mem.graph"):
        kg.add_triple(
            "Kai", "works_on", "Nova", valid_from=datetime(2026, 3, 1, tzinfo=UTC)
        )
    assert kg.stats()["total"] == 2
    assert any("contradiction" in r.message for r in caplog.records)


def test_contradiction_off_mode_silent(tmp_path: Path) -> None:
    kg = KnowledgeGraph(tmp_path / "g.db", fact_check_mode="off")
    start = datetime(2025, 6, 1, tzinfo=UTC)
    kg.add_triple("Kai", "works_on", "Orion", valid_from=start)
    kg.add_triple(
        "Kai", "works_on", "Nova", valid_from=datetime(2026, 3, 1, tzinfo=UTC)
    )
    assert kg.stats()["total"] == 2


def test_invalidate_then_add_clean(kg: KnowledgeGraph) -> None:
    start = datetime(2025, 6, 1, tzinfo=UTC)
    kg.add_triple("Kai", "works_on", "Orion", valid_from=start)
    n = kg.invalidate(
        "Kai", "works_on", "Orion", ended_at=datetime(2026, 2, 28, tzinfo=UTC)
    )
    assert n == 1
    # After invalidation the new triple is not a contradiction.
    kg.add_triple("Kai", "works_on", "Nova", valid_from=datetime(2026, 3, 1, tzinfo=UTC))
    assert kg.stats() == {"total": 2, "active": 1, "closed": 1}


def test_historical_query_still_returns_closed_edges(kg: KnowledgeGraph) -> None:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    end = datetime(2025, 12, 31, tzinfo=UTC)
    edge = kg.add_triple("Kai", "works_on", "Orion", valid_from=start)
    kg.store.invalidate("Kai", "Orion", edge.predicate, ended_at=end)

    current = kg.query_entity("Kai", as_of=None)
    assert current == []

    mid_2025 = datetime(2025, 6, 1, tzinfo=UTC)
    historical = kg.query_entity("Kai", as_of=mid_2025)
    assert len(historical) == 1
    assert historical[0].target == "Orion"


def test_traverse_requires_as_of_keyword(kg: KnowledgeGraph) -> None:
    with pytest.raises(TypeError):
        kg.traverse()  # type: ignore[call-arg]


def test_traverse_builds_multidigraph(kg: KnowledgeGraph) -> None:
    vf = datetime(2025, 1, 1, tzinfo=UTC)
    kg.add_triple("Kai", "knows", "Maya", valid_from=vf)
    kg.add_triple("Maya", "works_on", "Orion", valid_from=vf)
    g = kg.traverse(as_of=None)
    assert g.number_of_nodes() == 3
    assert g.number_of_edges() == 2


def test_timeline_ordering(kg: KnowledgeGraph) -> None:
    kg.add_triple("a", "rel", "b", valid_from=datetime(2026, 1, 1, tzinfo=UTC))
    kg.add_triple("c", "rel", "d", valid_from=datetime(2025, 1, 1, tzinfo=UTC))
    tl = kg.timeline()
    assert [e.valid_from.year for e in tl] == [2025, 2026]


def test_invalid_fact_check_mode(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="fact_check_mode"):
        KnowledgeGraph(tmp_path / "g.db", fact_check_mode="loose")  # type: ignore[arg-type]
