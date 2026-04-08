"""Loop-prevention subsystem.

Claude's "memory loops" — re-asking questions it already answered,
re-deriving conclusions it already reached, re-introducing itself every
session — happen because the model has no persistent state between
sessions. :class:`Wakeup` is the minimal fix: a tiny SQLite table of
high-value facts with a compact snapshot function that the agent loads
at session start.

This is intentionally **separate** from the vector store. The vector
store is for open-ended recall ("what did we discuss about auth?"); the
wakeup store is for facts the model needs **every session** without
searching: who the user is, what projects are active, what decisions
are locked in, what questions are still open.

Kinds of facts (tags, not a strict taxonomy):

* ``identity``    — "I am Michael, a software engineer working from macOS"
* ``preference``  — "prefers terse responses"
* ``project``     — "active: gr0m_mem public launch"
* ``decision``    — "chose sqlite_fts over sqlite_vec for zero-dep default"
* ``question``    — "still deciding whether to publish benchmarks before v0.1"
* ``milestone``   — "shipped Phase B 2026-04-08"
* ``context``     — anything else worth keeping tiny and always-loaded

The snapshot respects a token budget (via ``tiktoken`` when installed,
or a 4-chars-per-token heuristic otherwise) so the same call works
whether you give it 200 tokens or 2000.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

VALID_KINDS = (
    "identity",
    "preference",
    "project",
    "decision",
    "question",
    "milestone",
    "context",
)

# Ordering used by the snapshot when budget is tight. Earlier kinds are
# rendered first and get priority under the token budget.
_SNAPSHOT_ORDER = (
    "identity",
    "preference",
    "project",
    "decision",
    "question",
    "milestone",
    "context",
)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    id           TEXT PRIMARY KEY,
    kind         TEXT NOT NULL,
    scope        TEXT NOT NULL,      -- corpus name or 'global'
    subject      TEXT,               -- for decisions/questions: what this is about
    text         TEXT NOT NULL,
    rationale    TEXT,
    added_at     TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    hit_count    INTEGER NOT NULL DEFAULT 0,
    metadata     TEXT NOT NULL       -- JSON
);

CREATE INDEX IF NOT EXISTS idx_facts_kind    ON facts(kind);
CREATE INDEX IF NOT EXISTS idx_facts_scope   ON facts(scope);
CREATE INDEX IF NOT EXISTS idx_facts_subject ON facts(subject);
"""


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _count_tokens(text: str) -> int:
    """Real token count via tiktoken, or a 4-chars-per-token fallback."""
    try:
        import tiktoken
    except ImportError:
        return max(1, len(text) // 4)
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


@dataclass(frozen=True, slots=True)
class Fact:
    id: str
    kind: str
    scope: str
    subject: str | None
    text: str
    rationale: str | None
    added_at: datetime
    last_seen_at: datetime
    hit_count: int
    metadata: dict[str, Any]


def _row_to_fact(row: sqlite3.Row) -> Fact:
    return Fact(
        id=row["id"],
        kind=row["kind"],
        scope=row["scope"],
        subject=row["subject"],
        text=row["text"],
        rationale=row["rationale"],
        added_at=datetime.fromisoformat(row["added_at"]),
        last_seen_at=datetime.fromisoformat(row["last_seen_at"]),
        hit_count=int(row["hit_count"]),
        metadata=json.loads(row["metadata"]),
    )


class Wakeup:
    """SQLite-backed persistent facts + snapshot builder."""

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._path = db_path
        self._conn = sqlite3.connect(str(db_path), isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)

    @property
    def path(self) -> Path:
        return self._path

    # ── Writes ───────────────────────────────────────────

    def remember(
        self,
        kind: str,
        text: str,
        *,
        scope: str = "global",
        subject: str | None = None,
        rationale: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Fact:
        """Record a new fact.

        ``kind`` must be one of :data:`VALID_KINDS`. ``scope`` is either
        a corpus name (facts local to that project) or ``"global"``
        (facts that apply everywhere).
        """
        if kind not in VALID_KINDS:
            raise ValueError(f"kind must be one of {VALID_KINDS}, got {kind!r}")
        if not text.strip():
            raise ValueError("text must be non-empty")
        now = _now()
        fact_id = str(uuid4())
        self._conn.execute(
            """
            INSERT INTO facts
                (id, kind, scope, subject, text, rationale,
                 added_at, last_seen_at, hit_count, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
            """,
            (
                fact_id,
                kind,
                scope,
                subject,
                text.strip(),
                rationale,
                now.isoformat(),
                now.isoformat(),
                json.dumps(metadata or {}, sort_keys=True),
            ),
        )
        return Fact(
            id=fact_id,
            kind=kind,
            scope=scope,
            subject=subject,
            text=text.strip(),
            rationale=rationale,
            added_at=now,
            last_seen_at=now,
            hit_count=0,
            metadata=metadata or {},
        )

    def record_decision(
        self,
        subject: str,
        decision: str,
        rationale: str | None = None,
        *,
        scope: str = "global",
        metadata: dict[str, Any] | None = None,
    ) -> Fact:
        """Log a concrete decision about ``subject``.

        This is the key loop-prevention primitive: when the agent makes
        a choice, it should call this so future sessions find it via
        :meth:`recall_decisions` instead of re-deriving it.
        """
        return self.remember(
            kind="decision",
            text=decision,
            scope=scope,
            subject=subject,
            rationale=rationale,
            metadata=metadata,
        )

    def touch(self, fact_id: str) -> None:
        """Bump the hit count and last-seen time for a fact."""
        self._conn.execute(
            """
            UPDATE facts
               SET hit_count = hit_count + 1,
                   last_seen_at = ?
             WHERE id = ?
            """,
            (_now().isoformat(), fact_id),
        )

    def forget(self, fact_id: str) -> bool:
        cur = self._conn.execute("DELETE FROM facts WHERE id = ?", (fact_id,))
        return cur.rowcount > 0

    # ── Reads ────────────────────────────────────────────

    def recall_decisions(
        self,
        subject: str,
        *,
        scope: str = "global",
        limit: int = 10,
    ) -> list[Fact]:
        """Return the most recent decisions about ``subject``.

        Results are ordered newest-first. The agent should call this
        **before** asking the user a question that sounds familiar.
        """
        rows = self._conn.execute(
            """
            SELECT * FROM facts
             WHERE kind = 'decision'
               AND subject = ?
               AND (scope = ? OR scope = 'global')
             ORDER BY added_at DESC
             LIMIT ?
            """,
            (subject, scope, limit),
        ).fetchall()
        return [_row_to_fact(r) for r in rows]

    def all_facts(
        self,
        *,
        scope: str = "global",
        kinds: tuple[str, ...] | None = None,
    ) -> list[Fact]:
        """Return every fact in ``scope`` (and global), newest-first."""
        clauses = ["(scope = ? OR scope = 'global')"]
        params: list[Any] = [scope]
        if kinds:
            placeholders = ",".join("?" for _ in kinds)
            clauses.append(f"kind IN ({placeholders})")
            params.extend(kinds)
        rows = self._conn.execute(
            f"SELECT * FROM facts WHERE {' AND '.join(clauses)} "
            "ORDER BY added_at DESC",
            params,
        ).fetchall()
        return [_row_to_fact(r) for r in rows]

    def snapshot(
        self,
        *,
        scope: str = "global",
        token_budget: int = 200,
    ) -> dict[str, Any]:
        """Return a compact context block sized to ``token_budget``.

        The snapshot is structured by kind in :data:`_SNAPSHOT_ORDER`:
        identity first, then preferences, active projects, recent
        decisions, open questions, milestones, context. Each section is
        added whole-or-not — we never truncate a fact in the middle.
        """
        facts = self.all_facts(scope=scope)
        if not facts:
            return {
                "scope": scope,
                "token_budget": token_budget,
                "tokens_used": 0,
                "facts_included": 0,
                "facts_total": 0,
                "text": "",
            }

        by_kind: dict[str, list[Fact]] = {k: [] for k in _SNAPSHOT_ORDER}
        for f in facts:
            by_kind.setdefault(f.kind, []).append(f)

        lines: list[str] = []
        tokens_used = 0
        included = 0

        def _fact_line(f: Fact) -> str:
            if f.kind == "decision" and f.subject:
                base = f"- {f.subject}: {f.text}"
                if f.rationale:
                    base += f" ({f.rationale})"
                return base
            if f.kind == "question" and f.subject:
                return f"- {f.subject}? {f.text}"
            return f"- {f.text}"

        for kind in _SNAPSHOT_ORDER:
            section = by_kind.get(kind, [])
            if not section:
                continue
            header = f"\n## {kind.upper()}"
            header_tokens = _count_tokens(header)
            if tokens_used + header_tokens > token_budget and lines:
                break
            tentative = [header]
            t_used = header_tokens
            any_added = False
            for f in section:
                line = _fact_line(f)
                line_tokens = _count_tokens(line)
                if tokens_used + t_used + line_tokens > token_budget:
                    if any_added:
                        break
                    # Not even one line fits — skip this section entirely.
                    tentative = []
                    t_used = 0
                    break
                tentative.append(line)
                t_used += line_tokens
                any_added = True
                included += 1
            if tentative:
                lines.extend(tentative)
                tokens_used += t_used

        text = "\n".join(lines).strip()
        return {
            "scope": scope,
            "token_budget": token_budget,
            "tokens_used": tokens_used,
            "facts_included": included,
            "facts_total": len(facts),
            "text": text,
        }

    def stats(self) -> dict[str, int]:
        row = self._conn.execute("SELECT COUNT(*) AS n FROM facts").fetchone()
        total = int(row["n"])
        by_kind_rows = self._conn.execute(
            "SELECT kind, COUNT(*) AS n FROM facts GROUP BY kind"
        ).fetchall()
        out: dict[str, int] = {"total": total}
        for r in by_kind_rows:
            out[r["kind"]] = int(r["n"])
        return out

    def close(self) -> None:
        self._conn.close()
