"""The loop-prevention benchmark must stay at 100% on every commit."""

from __future__ import annotations

from benchmarks.loop_prevention.run import run


def test_all_scenarios_pass() -> None:
    result = run()
    assert result["scenarios_failed"] == 0, (
        f"scenarios failed: {[s['id'] for s in result['scenarios'] if not s['passed']]}"
    )
    assert result["probe_pass_rate"] == 1.0
    assert result["scenarios_total"] >= 1
