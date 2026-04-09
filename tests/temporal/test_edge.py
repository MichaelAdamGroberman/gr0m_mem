"""TemporalEdge construction + is_active_at."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from gr0m_mem.graph.temporal import EdgePredicate, TemporalEdge

UTC = timezone.utc


def _edge(
    *,
    valid_from: datetime,
    valid_to: datetime | None = None,
) -> TemporalEdge:
    return TemporalEdge(
        source="Kai",
        target="Orion",
        predicate=EdgePredicate("works_on"),
        valid_from=valid_from,
        valid_to=valid_to,
    )


def test_predicate_validation() -> None:
    EdgePredicate("works_on")
    EdgePredicate("a")
    for bad in ["", "1leading_digit", "Has-Hyphen", "HAS_UPPER", "has space"]:
        with pytest.raises(ValueError):
            EdgePredicate(bad)


def test_naive_datetime_rejected() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        _edge(valid_from=datetime(2025, 1, 1))  # naive


def test_valid_to_before_valid_from_rejected() -> None:
    start = datetime(2025, 6, 1, tzinfo=UTC)
    end = datetime(2024, 1, 1, tzinfo=UTC)
    with pytest.raises(ValueError, match="precedes"):
        _edge(valid_from=start, valid_to=end)


def test_confidence_bounds() -> None:
    with pytest.raises(ValueError):
        TemporalEdge(
            source="a",
            target="b",
            predicate=EdgePredicate("p"),
            valid_from=datetime(2025, 1, 1, tzinfo=UTC),
            confidence=1.5,
        )


class TestIsActiveAt:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    mid = datetime(2025, 6, 1, tzinfo=UTC)
    end = datetime(2025, 12, 31, tzinfo=UTC)
    before = datetime(2024, 1, 1, tzinfo=UTC)
    after = datetime(2026, 6, 1, tzinfo=UTC)

    def test_open_edge_current(self) -> None:
        e = _edge(valid_from=self.start)
        assert e.is_active_at(None) is True
        assert e.is_active_at(self.mid) is True
        assert e.is_active_at(self.before) is False

    def test_closed_edge_not_current(self) -> None:
        e = _edge(valid_from=self.start, valid_to=self.end)
        # `None` = currently valid = no valid_to — closed edges are NOT current.
        assert e.is_active_at(None) is False

    def test_closed_edge_historical(self) -> None:
        e = _edge(valid_from=self.start, valid_to=self.end)
        assert e.is_active_at(self.mid) is True
        assert e.is_active_at(self.before) is False
        assert e.is_active_at(self.after) is False
