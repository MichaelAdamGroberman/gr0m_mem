"""End-to-end: CLI `hook` subcommand records facts into the wakeup store.

Uses a temporary ``GR0M_MEM_HOME`` and forces ``sqlite_fts`` backend so
the test needs neither Ollama nor chromadb.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gr0m_mem.brain import Brain
from gr0m_mem.cli import main as cli_main
from gr0m_mem.config import Config


@pytest.fixture
def gr0m_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("GR0M_MEM_HOME", str(tmp_path))
    monkeypatch.setenv("GR0M_MEM_BACKEND", "sqlite_fts")
    # Guarantee Ollama is unreachable for the test so we never accidentally
    # hit a real Ollama process on the dev machine.
    monkeypatch.setenv("GR0M_MEM_OLLAMA_URL", "http://127.0.0.1:1")
    return tmp_path


def _count_hook_facts(event: str, session_id: str) -> int:
    brain = Brain(Config.from_env())
    try:
        return sum(
            1
            for f in brain.wakeup.all_facts()
            if f.metadata.get("source") == "hook"
            and f.metadata.get("event") == event
            and f.metadata.get("session_id") == session_id
        )
    finally:
        brain.close()


def test_stop_hook_records_milestone(gr0m_env: Path, capsys: pytest.CaptureFixture) -> None:
    rc = cli_main(["hook", "stop", "--session-id", "abc_123"])
    assert rc == 0
    assert _count_hook_facts("stop", "abc_123") == 1


def test_precompact_hook_records_milestone(gr0m_env: Path) -> None:
    rc = cli_main(["hook", "precompact", "--session-id", "xyz-789"])
    assert rc == 0
    assert _count_hook_facts("precompact", "xyz-789") == 1


def test_hook_elapsed_metadata_populated_on_second_call(gr0m_env: Path) -> None:
    # First fire has no prior; second fire records elapsed seconds.
    cli_main(["hook", "stop", "--session-id", "s1"])
    cli_main(["hook", "stop", "--session-id", "s1"])
    brain = Brain(Config.from_env())
    try:
        hook_facts = sorted(
            (
                f
                for f in brain.wakeup.all_facts()
                if f.metadata.get("source") == "hook"
            ),
            key=lambda f: f.added_at,
        )
    finally:
        brain.close()
    assert len(hook_facts) == 2
    assert hook_facts[0].metadata.get("prior_hook_count") == 0
    assert hook_facts[0].metadata.get("elapsed_since_last_s") is None
    assert hook_facts[1].metadata.get("prior_hook_count") == 1
    elapsed = hook_facts[1].metadata.get("elapsed_since_last_s")
    assert elapsed is not None and elapsed >= 0.0


def test_sessions_are_isolated(gr0m_env: Path) -> None:
    cli_main(["hook", "stop", "--session-id", "session_a"])
    cli_main(["hook", "stop", "--session-id", "session_b"])
    assert _count_hook_facts("stop", "session_a") == 1
    assert _count_hook_facts("stop", "session_b") == 1


def test_remember_cli_persists(gr0m_env: Path, capsys: pytest.CaptureFixture) -> None:
    cli_main(
        [
            "remember",
            "--kind",
            "identity",
            "--text",
            "Michael, software engineer",
        ]
    )
    out = capsys.readouterr().out
    assert "remembered" in out
    assert "identity" in out

    brain = Brain(Config.from_env())
    try:
        facts = brain.wakeup.all_facts()
    finally:
        brain.close()
    assert len(facts) == 1
    assert facts[0].text == "Michael, software engineer"


def test_wakeup_cli_renders_snapshot(gr0m_env: Path, capsys: pytest.CaptureFixture) -> None:
    cli_main(["remember", "--kind", "identity", "--text", "Michael"])
    cli_main(["remember", "--kind", "preference", "--text", "terse replies"])
    cli_main(["wakeup", "--tokens", "500"])
    out = capsys.readouterr().out
    assert "IDENTITY" in out
    assert "PREFERENCE" in out
    assert "Michael" in out
