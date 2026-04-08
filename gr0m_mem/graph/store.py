"""SQLite persistence for the temporal knowledge graph.

Schema and indexes:

* ``graph_edges`` stores every edge ever inserted. Edges are never deleted
  by application code — they are "invalidated" by setting ``valid_to``.
* Active-edge lookups (``valid_to IS NULL``) are the hot path, so we
  maintain partial indexes on ``source`` and ``target`` for them.
* Time-window lookups (``WHERE valid_from <= ? AND (valid_to IS NULL OR valid_to >= ?)``)
  are served by a compound index on ``(source, valid_from, valid_to)``.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gr0m_mem.graph.temporal import EdgePredicate, TemporalEdge

_SCHEMA = """
CREATE TABLE IF NOT EXISTS graph_edges (
    id           TEXT PRIMARY KEY,
    source       TEXT NOT NULL,
    target       TEXT NOT NULL,
    predicate    TEXT NOT NULL,
    weight       REAL NOT NULL,
    valid_from   TEXT NOT NULL,
    valid_to     TEXT,
    confidence   REAL NOT NULL,
    source_doc   TEXT,
    data         TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_edges_src_active
    ON graph_edges(source) WHERE valid_to IS NULL;

CREATE INDEX IF NOT EXISTS idx_edges_tgt_active
    ON graph_edges(target) WHERE valid_to IS NULL;

CREATE INDEX IF NOT EXISTS idx_edges_src_time
    ON graph_edges(source, valid_from, valid_to);

CREATE INDEX IF NOT EXISTS idx_edges_tgt_time
    ON graph_edges(target, valid_from, valid_to);

CREATE INDEX IF NOT EXISTS idx_edges_pred
    ON graph_edges(predicate);
"""


def _encode_dt(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        raise ValueError(f"naive datetime cannot be stored: {dt!r}")
    return dt.astimezone(timezone.utc).isoformat()


def _decode_dt(s: str | None) -> datetime | None:
    if s is None:
        return None
    return datetime.fromisoformat(s)


def _row_to_edge(row: sqlite3.Row) -> TemporalEdge:
    valid_from = _decode_dt(row["valid_from"])
    if valid_from is None:
        raise ValueError(f"edge {row['id']} has NULL valid_from")
    return TemporalEdge(
        id=row["id"],
        source=row["source"],
        target=row["target"],
        predicate=EdgePredicate(row["predicate"]),
        weight=row["weight"],
        valid_from=valid_from,
        valid_to=_decode_dt(row["valid_to"]),
        confidence=row["confidence"],
        source_doc=row["source_doc"],
        data=json.loads(row["data"]),
    )


class GraphStore:
    """SQLite-backed persistence. One instance per process/path."""

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._path = db_path
        self._conn = sqlite3.connect(str(db_path), isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)

    @property
    def path(self) -> Path:
        return self._path

    # ── Writes ────────────────────────────────────────────

    def insert(self, edge: TemporalEdge) -> None:
        self._conn.execute(
            """
            INSERT INTO graph_edges
                (id, source, target, predicate, weight, valid_from, valid_to,
                 confidence, source_doc, data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                edge.id,
                edge.source,
                edge.target,
                str(edge.predicate),
                edge.weight,
                _encode_dt(edge.valid_from),
                _encode_dt(edge.valid_to),
                edge.confidence,
                edge.source_doc,
                json.dumps(edge.data, sort_keys=True),
            ),
        )

    def invalidate(
        self,
        source: str,
        target: str,
        predicate: EdgePredicate,
        ended_at: datetime,
    ) -> int:
        """Set ``valid_to = ended_at`` on every currently-active matching edge.

        Returns the number of edges updated. Does not delete rows —
        historical queries still see the edge.
        """
        cur = self._conn.execute(
            """
            UPDATE graph_edges
               SET valid_to = ?
             WHERE source = ? AND target = ? AND predicate = ?
               AND valid_to IS NULL
            """,
            (_encode_dt(ended_at), source, target, str(predicate)),
        )
        return cur.rowcount

    def invalidate_by_id(self, edge_id: str, ended_at: datetime) -> bool:
        cur = self._conn.execute(
            """
            UPDATE graph_edges
               SET valid_to = ?
             WHERE id = ? AND valid_to IS NULL
            """,
            (_encode_dt(ended_at), edge_id),
        )
        return cur.rowcount > 0

    # ── Reads ─────────────────────────────────────────────

    def all_edges(self) -> list[TemporalEdge]:
        rows = self._conn.execute("SELECT * FROM graph_edges").fetchall()
        return [_row_to_edge(r) for r in rows]

    def edges_at(self, as_of: datetime | None) -> list[TemporalEdge]:
        """Return every edge active at ``as_of``.

        ``as_of=None`` means "currently valid" — i.e., rows with
        ``valid_to IS NULL``. A past or future datetime matches rows whose
        validity window contains it.
        """
        if as_of is None:
            rows = self._conn.execute(
                "SELECT * FROM graph_edges WHERE valid_to IS NULL"
            ).fetchall()
        else:
            stamp = _encode_dt(as_of)
            rows = self._conn.execute(
                """
                SELECT * FROM graph_edges
                 WHERE valid_from <= ?
                   AND (valid_to IS NULL OR valid_to >= ?)
                """,
                (stamp, stamp),
            ).fetchall()
        return [_row_to_edge(r) for r in rows]

    def edges_for_entity(
        self,
        entity: str,
        as_of: datetime | None,
        direction: str = "both",
    ) -> list[TemporalEdge]:
        """Return edges touching ``entity`` and active at ``as_of``.

        ``direction`` is one of ``"out"``, ``"in"``, ``"both"``.
        """
        if direction not in ("out", "in", "both"):
            raise ValueError(f"direction must be out|in|both, got {direction!r}")

        clauses: list[str]
        if direction == "out":
            clauses = ["source = ?"]
            params: tuple[Any, ...] = (entity,)
        elif direction == "in":
            clauses = ["target = ?"]
            params = (entity,)
        else:
            clauses = ["(source = ? OR target = ?)"]
            params = (entity, entity)

        if as_of is None:
            clauses.append("valid_to IS NULL")
        else:
            stamp = _encode_dt(as_of)
            clauses.append("valid_from <= ? AND (valid_to IS NULL OR valid_to >= ?)")
            params = (*params, stamp, stamp)

        where = " AND ".join(clauses)
        rows = self._conn.execute(
            f"SELECT * FROM graph_edges WHERE {where}", params
        ).fetchall()
        return [_row_to_edge(r) for r in rows]

    def active_edges_between(
        self,
        source: str,
        target: str | None,
        predicate: EdgePredicate | None,
    ) -> list[TemporalEdge]:
        """Return currently-active edges matching an (s, [p], [o]) pattern.

        Used by the fact checker to look up what might conflict with an
        incoming triple. Only considers open-ended (``valid_to IS NULL``)
        edges because closed edges cannot contradict a new assertion.
        """
        clauses = ["source = ?", "valid_to IS NULL"]
        params: list[Any] = [source]
        if target is not None:
            clauses.append("target = ?")
            params.append(target)
        if predicate is not None:
            clauses.append("predicate = ?")
            params.append(str(predicate))
        rows = self._conn.execute(
            "SELECT * FROM graph_edges WHERE " + " AND ".join(clauses),
            tuple(params),
        ).fetchall()
        return [_row_to_edge(r) for r in rows]

    def count(self) -> dict[str, int]:
        row = self._conn.execute(
            """
            SELECT
                COUNT(*)                                             AS total,
                SUM(CASE WHEN valid_to IS NULL THEN 1 ELSE 0 END)    AS active,
                SUM(CASE WHEN valid_to IS NOT NULL THEN 1 ELSE 0 END) AS closed
            FROM graph_edges
            """
        ).fetchone()
        return {
            "total": int(row["total"] or 0),
            "active": int(row["active"] or 0),
            "closed": int(row["closed"] or 0),
        }

    def close(self) -> None:
        self._conn.close()
