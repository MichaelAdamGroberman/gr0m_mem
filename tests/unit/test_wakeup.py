"""Wakeup — loop-prevention core."""

from __future__ import annotations

from pathlib import Path

import pytest

from gr0m_mem.wakeup import VALID_KINDS, Wakeup


@pytest.fixture
def wake(tmp_path: Path) -> Wakeup:
    return Wakeup(tmp_path / "wakeup.db")


def test_remember_rejects_invalid_kind(wake: Wakeup) -> None:
    with pytest.raises(ValueError, match="kind must be one of"):
        wake.remember(kind="nonsense", text="x")


def test_remember_rejects_empty_text(wake: Wakeup) -> None:
    with pytest.raises(ValueError, match="text must be non-empty"):
        wake.remember(kind="identity", text="   ")


def test_remember_and_retrieve(wake: Wakeup) -> None:
    fact = wake.remember(kind="identity", text="Michael, SWE")
    assert fact.kind == "identity"
    assert fact.text == "Michael, SWE"
    assert fact.hit_count == 0
    all_facts = wake.all_facts()
    assert len(all_facts) == 1
    assert all_facts[0].id == fact.id


def test_record_and_recall_decision(wake: Wakeup) -> None:
    wake.record_decision(
        subject="database",
        decision="Postgres over SQLite",
        rationale="concurrent writes + >10GB dataset",
    )
    wake.record_decision(
        subject="database",
        decision="connection pool size 20",
    )
    wake.record_decision(
        subject="auth",
        decision="Clerk over Auth0",
        rationale="better DX",
    )

    db_decisions = wake.recall_decisions("database")
    assert len(db_decisions) == 2
    # Newest first.
    assert db_decisions[0].text == "connection pool size 20"
    assert db_decisions[1].rationale == "concurrent writes + >10GB dataset"

    auth_decisions = wake.recall_decisions("auth")
    assert len(auth_decisions) == 1


def test_scope_isolation(wake: Wakeup) -> None:
    wake.remember(kind="project", text="active: alpha", scope="global")
    wake.remember(kind="project", text="beta milestone", scope="beta")

    beta_scope = wake.all_facts(scope="beta")
    # beta sees itself + global
    assert len(beta_scope) == 2

    alpha_scope = wake.all_facts(scope="alpha")
    # alpha sees only global (no alpha-scoped facts)
    assert len(alpha_scope) == 1


class TestSnapshot:
    def _seed(self, wake: Wakeup) -> None:
        wake.remember(kind="identity", text="Michael, software engineer on macOS")
        wake.remember(kind="preference", text="terse responses, no trailing summaries")
        wake.remember(kind="project", text="active: gr0m_mem public launch")
        wake.record_decision(
            subject="backend",
            decision="sqlite_fts is the zero-dep default",
            rationale="pip install gr0m-mem must never fail",
        )
        wake.remember(kind="question", text="publish benchmarks before v0.1?", subject="launch")

    def test_snapshot_empty_store(self, wake: Wakeup) -> None:
        snap = wake.snapshot()
        assert snap["facts_included"] == 0
        assert snap["text"] == ""

    def test_snapshot_full(self, wake: Wakeup) -> None:
        self._seed(wake)
        snap = wake.snapshot(token_budget=500)
        assert snap["facts_included"] == 5
        text = snap["text"]
        assert "## IDENTITY" in text
        assert "## PREFERENCE" in text
        assert "## PROJECT" in text
        assert "## DECISION" in text
        assert "## QUESTION" in text
        # Decision should include the rationale.
        assert "zero-dep default" in text
        assert "pip install gr0m-mem must never fail" in text

    def test_snapshot_respects_tight_budget(self, wake: Wakeup) -> None:
        self._seed(wake)
        tight = wake.snapshot(token_budget=20)
        assert tight["tokens_used"] <= 20
        # Identity is always first and should fit by itself.
        assert "IDENTITY" in tight["text"]
        # With 20 tokens we should NOT get all five sections.
        assert tight["facts_included"] < 5

    def test_snapshot_tokens_used_monotonic(self, wake: Wakeup) -> None:
        self._seed(wake)
        small = wake.snapshot(token_budget=30)
        big = wake.snapshot(token_budget=500)
        assert small["tokens_used"] <= big["tokens_used"]
        assert small["facts_included"] <= big["facts_included"]


def test_stats(wake: Wakeup) -> None:
    wake.remember(kind="identity", text="a")
    wake.remember(kind="preference", text="b")
    wake.record_decision(subject="x", decision="y")
    stats = wake.stats()
    assert stats["total"] == 3
    assert stats["identity"] == 1
    assert stats["preference"] == 1
    assert stats["decision"] == 1


def test_touch_bumps_counter(wake: Wakeup) -> None:
    fact = wake.remember(kind="identity", text="Michael")
    wake.touch(fact.id)
    wake.touch(fact.id)
    refreshed = wake.all_facts()[0]
    assert refreshed.hit_count == 2


def test_forget(wake: Wakeup) -> None:
    fact = wake.remember(kind="context", text="delete me")
    assert wake.forget(fact.id) is True
    assert wake.forget(fact.id) is False
    assert wake.all_facts() == []


def test_all_valid_kinds_accepted(wake: Wakeup) -> None:
    for kind in VALID_KINDS:
        wake.remember(kind=kind, text=f"example {kind}")
    assert wake.stats()["total"] == len(VALID_KINDS)
