"""End-to-end: CLI `hook` subcommand records facts into the wakeup store.

Uses a temporary ``GR0M_MEM_HOME`` and forces ``sqlite_fts`` backend so
the test needs neither Ollama nor chromadb.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from gr0m_mem.brain import Brain
from gr0m_mem.cli import main as cli_main
from gr0m_mem.config import Config

_SAVE_HOOK = Path(__file__).resolve().parents[2] / "gr0m_mem" / "hooks" / "save_hook.sh"
_PRECOMPACT_HOOK = (
    Path(__file__).resolve().parents[2] / "gr0m_mem" / "hooks" / "precompact_hook.sh"
)
# Resolve once, as an absolute path -- tests below sometimes run the hook
# with a deliberately-restricted PATH, and `bash` itself must still be
# findable without relying on it.
_BASH = shutil.which("bash") or "/bin/bash"


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


def test_hook_elapsed_metadata_populated_on_second_call(
    gr0m_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Disable throttling so both fires land -- this test is about the
    # elapsed-seconds bookkeeping, not the throttle window (see
    # test_stop_hook_throttled_within_window below for that).
    monkeypatch.setenv("GR0M_MEM_HOOK_THROTTLE_SECONDS", "0")
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


def test_stop_hook_throttled_within_window(gr0m_env: Path) -> None:
    # Default throttle window applies: a second "stop" fire for the same
    # session, immediately after the first, is dropped -- no new fact.
    cli_main(["hook", "stop", "--session-id", "s1"])
    cli_main(["hook", "stop", "--session-id", "s1"])
    assert _count_hook_facts("stop", "s1") == 1


def test_stop_hook_not_throttled_across_sessions(gr0m_env: Path) -> None:
    # Throttling is keyed on session id -- a different session is
    # unaffected by another session's recent fire.
    cli_main(["hook", "stop", "--session-id", "s1"])
    cli_main(["hook", "stop", "--session-id", "s2"])
    assert _count_hook_facts("stop", "s1") == 1
    assert _count_hook_facts("stop", "s2") == 1


def test_precompact_hook_never_throttled(gr0m_env: Path) -> None:
    # PreCompact is a last-chance flush and must never be throttled,
    # even with back-to-back fires for the same session.
    cli_main(["hook", "precompact", "--session-id", "s1"])
    cli_main(["hook", "precompact", "--session-id", "s1"])
    assert _count_hook_facts("precompact", "s1") == 2


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


def test_wakeup_snapshot_excludes_hook_milestones(gr0m_env: Path) -> None:
    # This is the actual bug that motivated hook throttling + this
    # exclusion: 49 identical hook milestones were drowning out real
    # facts in a 400-token snapshot. Real facts must always win.
    cli_main(["remember", "--kind", "identity", "--text", "Michael, software engineer"])
    cli_main(["hook", "stop", "--session-id", "s1"])
    cli_main(["hook", "stop", "--session-id", "s2"])  # different session, not throttled

    brain = Brain(Config.from_env())
    try:
        all_facts = brain.wakeup.all_facts()
        snap = brain.wakeup.snapshot(token_budget=500)
    finally:
        brain.close()

    # The hook facts are durable and still queryable directly...
    assert sum(1 for f in all_facts if f.metadata.get("source") == "hook") == 2
    # ...but never appear in the snapshot.
    assert "claude-code stop" not in snap["text"]
    assert "Michael, software engineer" in snap["text"]
    assert snap["facts_total"] == 1
    assert snap["facts_included"] == 1


# ── Shell hook scripts (save_hook.sh / precompact_hook.sh) ──────────────
#
# Claude Code invokes these as raw shell scripts and feeds the hook
# payload as JSON on stdin -- it does not set $SESSION_ID. These tests
# run the real scripts as subprocesses against a temp GR0M_MEM_HOME.


def _hook_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ)
    env["GR0M_MEM_PYTHON"] = sys.executable
    if extra:
        env.update(extra)
    return env


def _run_hook(
    script: Path,
    *,
    stdin: str | None,
    env: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [_BASH, str(script)],
        input=stdin,
        text=True,
        env=env,
        capture_output=True,
        timeout=30,
    )


def _path_without_jq(tmp_path: Path) -> str:
    """Build a PATH containing only cat/tr/python3, no jq.

    Used to exercise the pure-python JSON-parsing fallback the hook
    scripts fall back to when jq is not installed.
    """
    cat, tr, py = shutil.which("cat"), shutil.which("tr"), shutil.which("python3")
    if not (cat and tr and py):
        pytest.skip("cat/tr/python3 not resolvable on this machine")
    bin_dir = tmp_path / "no-jq-bin"
    bin_dir.mkdir(exist_ok=True)
    for name, src in (("cat", cat), ("tr", tr), ("python3", py)):
        (bin_dir / name).symlink_to(src)
    return str(bin_dir)


class TestSaveHookScript:
    def test_extracts_and_sanitises_session_id_from_stdin_json(self, gr0m_env: Path) -> None:
        payload = json.dumps({"session_id": "abc-123$(rm)", "hook_event_name": "Stop"})
        result = _run_hook(_SAVE_HOOK, stdin=payload, env=_hook_env())
        assert result.returncode == 0
        # `$`, `(`, `)` are stripped by the whitelist; `-` is kept.
        assert _count_hook_facts("stop", "abc-123rm") == 1

    def test_empty_stdin_falls_back_to_unknown(self, gr0m_env: Path) -> None:
        result = subprocess.run(
            [_BASH, str(_SAVE_HOOK)],
            stdin=subprocess.DEVNULL,
            text=True,
            env=_hook_env(),
            capture_output=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert _count_hook_facts("stop", "unknown") == 1

    def test_falls_back_to_session_id_env_var_when_stdin_has_none(self, gr0m_env: Path) -> None:
        payload = json.dumps({"hook_event_name": "Stop"})
        result = _run_hook(
            _SAVE_HOOK, stdin=payload, env=_hook_env({"SESSION_ID": "env-fallback"})
        )
        assert result.returncode == 0
        assert _count_hook_facts("stop", "env-fallback") == 1

    def test_falls_back_to_claude_session_id_env_var(self, gr0m_env: Path) -> None:
        result = subprocess.run(
            [_BASH, str(_SAVE_HOOK)],
            stdin=subprocess.DEVNULL,
            text=True,
            env=_hook_env({"CLAUDE_SESSION_ID": "claude-env-fallback"}),
            capture_output=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert _count_hook_facts("stop", "claude-env-fallback") == 1

    def test_json_session_id_wins_over_env_var(self, gr0m_env: Path) -> None:
        payload = json.dumps({"session_id": "from-json"})
        result = _run_hook(
            _SAVE_HOOK, stdin=payload, env=_hook_env({"SESSION_ID": "from-env"})
        )
        assert result.returncode == 0
        assert _count_hook_facts("stop", "from-json") == 1
        assert _count_hook_facts("stop", "from-env") == 0

    def test_falls_back_to_python_json_parsing_without_jq(self, gr0m_env: Path) -> None:
        path_without_jq = _path_without_jq(gr0m_env)
        payload = json.dumps({"session_id": "nojq-session"})
        result = _run_hook(
            _SAVE_HOOK, stdin=payload, env=_hook_env({"PATH": path_without_jq})
        )
        assert result.returncode == 0
        assert _count_hook_facts("stop", "nojq-session") == 1

    def test_malformed_stdin_exits_zero_and_falls_back_to_unknown(self, gr0m_env: Path) -> None:
        result = _run_hook(_SAVE_HOOK, stdin="not json at all {{{", env=_hook_env())
        assert result.returncode == 0
        assert _count_hook_facts("stop", "unknown") == 1

    def test_exits_zero_even_when_python_interpreter_is_missing(self, gr0m_env: Path) -> None:
        payload = json.dumps({"session_id": "s1"})
        result = _run_hook(
            _SAVE_HOOK,
            stdin=payload,
            env=_hook_env({"GR0M_MEM_PYTHON": "/nonexistent/python3"}),
        )
        assert result.returncode == 0


class TestPrecompactHookScript:
    def test_extracts_and_sanitises_session_id_from_stdin_json(self, gr0m_env: Path) -> None:
        payload = json.dumps({"session_id": "xyz-789", "hook_event_name": "PreCompact"})
        result = _run_hook(_PRECOMPACT_HOOK, stdin=payload, env=_hook_env())
        assert result.returncode == 0
        assert _count_hook_facts("precompact", "xyz-789") == 1

    def test_empty_stdin_falls_back_to_unknown(self, gr0m_env: Path) -> None:
        result = subprocess.run(
            [_BASH, str(_PRECOMPACT_HOOK)],
            stdin=subprocess.DEVNULL,
            text=True,
            env=_hook_env(),
            capture_output=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert _count_hook_facts("precompact", "unknown") == 1
