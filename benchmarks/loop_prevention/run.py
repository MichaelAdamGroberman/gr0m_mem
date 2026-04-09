"""Loop-prevention benchmark runner.

Simulates a two-session workflow per scenario:

1. **Session 1** — records the scenario's ``setup`` facts and decisions
   into a fresh wakeup store.
2. **Session 2** — loads the wakeup snapshot and runs every ``probe``
   against the persistent state. Each probe is a specific loop-prevention
   check (snapshot contains a phrase, a decision exists, the newest
   decision on a subject is the expected one, a scoped decision is
   isolated to its scope).

A scenario passes iff every probe passes. The aggregate headline number
is the fraction of probes that passed.

Writes a JSON results file with per-scenario traces plus environment
metadata so every committed result is reproducible on the hardware it
was run on.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gr0m_mem import __version__
from gr0m_mem.wakeup import Wakeup

SCENARIOS_FILE = Path(__file__).resolve().parent / "scenarios.json"


@dataclass(frozen=True, slots=True)
class ProbeResult:
    kind: str
    subject: str | None
    needle: str
    passed: bool
    detail: str


@dataclass(frozen=True, slots=True)
class ScenarioResult:
    id: str
    description: str
    probes: list[ProbeResult]
    passed: bool

    @property
    def probe_pass_rate(self) -> float:
        if not self.probes:
            return 1.0
        return sum(1 for p in self.probes if p.passed) / len(self.probes)


# ── Probe primitives ─────────────────────────────────────


def _probe_snapshot_contains(wake: Wakeup, needle: str) -> ProbeResult:
    snap = wake.snapshot(token_budget=500)
    ok = needle.lower() in snap["text"].lower()
    return ProbeResult(
        kind="snapshot_must_contain",
        subject=None,
        needle=needle,
        passed=ok,
        detail=f"tokens_used={snap['tokens_used']}, text_len={len(snap['text'])}",
    )


def _probe_decision_exists(wake: Wakeup, subject: str, needle: str) -> ProbeResult:
    decisions = wake.recall_decisions(subject)
    ok = any(needle.lower() in d.text.lower() for d in decisions)
    return ProbeResult(
        kind="decision_must_exist",
        subject=subject,
        needle=needle,
        passed=ok,
        detail=f"decision_count={len(decisions)}",
    )


def _probe_newest_decision(wake: Wakeup, subject: str, needle: str) -> ProbeResult:
    decisions = wake.recall_decisions(subject, limit=1)
    ok = bool(decisions) and needle.lower() in decisions[0].text.lower()
    detail = decisions[0].text if decisions else "<no decisions>"
    return ProbeResult(
        kind="newest_decision_is",
        subject=subject,
        needle=needle,
        passed=ok,
        detail=detail,
    )


def _probe_scoped_decision(
    wake: Wakeup, scope: str, subject: str, needle: str
) -> ProbeResult:
    decisions = wake.recall_decisions(subject, scope=scope, limit=1)
    ok = bool(decisions) and needle.lower() in decisions[0].text.lower()
    detail = decisions[0].text if decisions else "<no decisions>"
    return ProbeResult(
        kind="scoped_decision_must_be",
        subject=f"{scope}/{subject}",
        needle=needle,
        passed=ok,
        detail=detail,
    )


_PROBES = {
    "snapshot_must_contain": _probe_snapshot_contains,
    "decision_must_exist": _probe_decision_exists,
    "newest_decision_is": _probe_newest_decision,
    "scoped_decision_must_be": _probe_scoped_decision,
}


# ── Runner ───────────────────────────────────────────────


def _run_scenario(scenario: dict[str, Any], db_path: Path) -> ScenarioResult:
    wake = Wakeup(db_path)
    try:
        # Session 1: seed the store.
        for fact in scenario["setup"].get("facts", []):
            wake.remember(
                kind=fact["kind"],
                text=fact["text"],
                subject=fact.get("subject"),
                rationale=fact.get("rationale"),
                scope=fact.get("scope", "global"),
            )
        for dec in scenario["setup"].get("decisions", []):
            wake.record_decision(
                subject=dec["subject"],
                decision=dec["decision"],
                rationale=dec.get("rationale"),
                scope=dec.get("scope", "global"),
            )

        # Session 2: probe the persistent state.
        probe_results: list[ProbeResult] = []
        for p in scenario["probes"]:
            kind = p["kind"]
            if kind == "snapshot_must_contain":
                probe_results.append(_probe_snapshot_contains(wake, p["needle"]))
            elif kind == "decision_must_exist":
                probe_results.append(
                    _probe_decision_exists(wake, p["subject"], p["needle"])
                )
            elif kind == "newest_decision_is":
                probe_results.append(
                    _probe_newest_decision(wake, p["subject"], p["needle"])
                )
            elif kind == "scoped_decision_must_be":
                probe_results.append(
                    _probe_scoped_decision(wake, p["scope"], p["subject"], p["needle"])
                )
            else:
                probe_results.append(
                    ProbeResult(
                        kind=kind,
                        subject=None,
                        needle=p.get("needle", ""),
                        passed=False,
                        detail=f"unknown probe kind {kind!r}",
                    )
                )
    finally:
        wake.close()

    return ScenarioResult(
        id=scenario["id"],
        description=scenario["description"],
        probes=probe_results,
        passed=all(p.passed for p in probe_results),
    )


def run(scenarios_path: Path = SCENARIOS_FILE) -> dict[str, Any]:
    raw = json.loads(scenarios_path.read_text(encoding="utf-8"))
    scenarios = raw["scenarios"]

    results: list[ScenarioResult] = []
    for scenario in scenarios:
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "wakeup.db"
            results.append(_run_scenario(scenario, db))

    total_probes = sum(len(r.probes) for r in results)
    passed_probes = sum(sum(1 for p in r.probes if p.passed) for r in results)
    scenarios_passed = sum(1 for r in results if r.passed)

    return {
        "benchmark": "loop_prevention",
        "run_at": datetime.now(tz=timezone.utc).isoformat(),
        "environment": {
            "gr0m_mem_version": __version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "scenarios_total": len(results),
        "scenarios_passed": scenarios_passed,
        "scenarios_failed": len(results) - scenarios_passed,
        "scenario_pass_rate": scenarios_passed / len(results) if results else 1.0,
        "probes_total": total_probes,
        "probes_passed": passed_probes,
        "probe_pass_rate": passed_probes / total_probes if total_probes else 1.0,
        "scenarios": [
            {
                "id": r.id,
                "description": r.description,
                "passed": r.passed,
                "probes": [asdict(p) for p in r.probes],
            }
            for r in results
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gr0m_mem-bench-loop-prevention")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write results JSON to this path (default: stdout)",
    )
    parser.add_argument(
        "--scenarios",
        type=Path,
        default=SCENARIOS_FILE,
        help="Override the scenarios file",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any scenario fails",
    )
    args = parser.parse_args(argv)

    result = run(args.scenarios)
    payload = json.dumps(result, indent=2)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n", encoding="utf-8")
        print(
            f"loop_prevention: {result['scenarios_passed']}/{result['scenarios_total']} "
            f"scenarios passed ({result['probe_pass_rate'] * 100:.1f}% probe pass rate)",
            file=sys.stderr,
        )
        print(f"results written to {args.out}", file=sys.stderr)
    else:
        print(payload)

    if args.strict and result["scenarios_failed"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
