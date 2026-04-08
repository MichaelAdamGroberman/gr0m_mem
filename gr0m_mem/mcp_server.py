"""Gr0m_Mem MCP server.

Exposes the five core tools to any MCP-compatible client (Claude Code,
Cursor, Gemini CLI, Copilot, etc.) over stdio.

SDK: ``mcp.server.fastmcp.FastMCP`` — same SDK the production RedAI brain
uses. Tool signatures use plain Python types so FastMCP can infer the
JSON schema from annotations.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from mcp.server.fastmcp import FastMCP

from gr0m_mem import __version__
from gr0m_mem.brain import Brain
from gr0m_mem.config import Config
from gr0m_mem.fact_checker import ContradictionError
from gr0m_mem.graph.temporal import TemporalEdge
from gr0m_mem.types import Corpus
from gr0m_mem.wakeup import VALID_KINDS, Fact

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [gr0m_mem] %(message)s",
)
log = logging.getLogger("gr0m_mem.mcp")


# ── Server init ──────────────────────────────────────────

_config = Config.from_env()
_brain = Brain(_config)

mcp = FastMCP(
    "Gr0m_Mem",
    instructions=(
        "Persistent memory brain that prevents conversation loops.\n\n"
        "MANDATORY LOOP-PREVENTION PROTOCOL — follow this every session:\n\n"
        "1. AT SESSION START, call `mem_wakeup` once. It returns a compact "
        "context block of identity / preferences / active projects / "
        "recent decisions / open questions. Inline the returned `text` "
        "into your working context immediately. This is how you avoid "
        "re-introducing yourself and re-asking questions you already "
        "answered in previous sessions.\n\n"
        "2. BEFORE ASKING THE USER A QUESTION that sounds like something "
        "you might have decided already, call `mem_recall_decisions` with "
        "the subject first. If a prior decision exists, use it — do not "
        "re-derive, do not re-ask.\n\n"
        "3. AFTER MAKING A DECISION with the user, call `mem_record_decision` "
        "with the subject, the decision, and the rationale. Future sessions "
        "will find it and skip the debate.\n\n"
        "4. AFTER LEARNING A DURABLE FACT about the user (preference, "
        "project, identity, milestone), call `mem_remember` with the "
        "appropriate kind.\n\n"
        "RETRIEVAL TOOLS (open-ended search within a named corpus):\n"
        " - `mem_learn`       — ingest a document (3-chunk header/body/context)\n"
        " - `mem_search`      — semantic search, returns ranked hits\n"
        " - `mem_query`       — smaller result shape for agent loops\n"
        " - `mem_rag`         — concatenated context block ready to inline\n"
        " - `mem_analyze`     — aggregate stats about a corpus\n"
        " - `mem_list_corpora`— every corpus currently in the store\n\n"
        "TEMPORAL KNOWLEDGE GRAPH (facts with validity windows):\n"
        " - `mem_kg_add`         — insert a triple; strict-mode fact checker "
        "rejects contradictions by default\n"
        " - `mem_kg_query`       — edges touching an entity, filterable by `as_of`\n"
        " - `mem_kg_invalidate`  — close an edge (never delete)\n"
        " - `mem_kg_timeline`    — full history of an entity\n"
        " - `mem_kg_stats`       — counts of active/closed edges\n\n"
        "Every corpus is an isolated collection — documents from different "
        "corpora are never mixed. All tools require an explicit `corpus` "
        "argument; there is no default tenant.\n\n"
        "Backend: Gr0m_Mem auto-selects chromadb if installed, falls back "
        "to SQLite vector search (with Ollama embeddings) or finally SQLite "
        "FTS5 lexical search. Call `mem_status` any time to see which "
        "backend is active and why."
    ),
)


# ── Helpers ──────────────────────────────────────────────


def _corpus(name: str) -> Corpus:
    """Validate and wrap a corpus name from a tool call argument."""
    return Corpus(name.strip())


def _parse_iso(ts: str | None) -> datetime | None:
    """Parse an ISO-8601 timestamp argument. ``None`` passes through."""
    if ts is None or ts == "":
        return None
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        raise ValueError(f"timestamp must be timezone-aware (UTC), got {ts!r}")
    return dt


def _fact_to_json(fact: Fact) -> dict[str, Any]:
    return {
        "id": fact.id,
        "kind": fact.kind,
        "scope": fact.scope,
        "subject": fact.subject,
        "text": fact.text,
        "rationale": fact.rationale,
        "added_at": fact.added_at.isoformat(),
        "last_seen_at": fact.last_seen_at.isoformat(),
        "hit_count": fact.hit_count,
        "metadata": fact.metadata,
    }


def _edge_to_json(edge: TemporalEdge) -> dict[str, Any]:
    return {
        "id": edge.id,
        "source": edge.source,
        "target": edge.target,
        "predicate": str(edge.predicate),
        "weight": edge.weight,
        "confidence": edge.confidence,
        "valid_from": edge.valid_from.isoformat(),
        "valid_to": edge.valid_to.isoformat() if edge.valid_to else None,
        "source_doc": edge.source_doc,
        "data": edge.data,
    }


# ── Tools ────────────────────────────────────────────────


@mcp.tool()
def mem_learn(
    corpus: str,
    document_id: str,
    title: str,
    body: str,
    context: str = "",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Ingest a document into a corpus.

    The document is split into up to three chunks (header / body / context)
    and each chunk is embedded separately for multi-aspect retrieval.
    Upserts by ``document_id`` — re-ingesting with the same id replaces the
    existing chunks.
    """
    return _brain.learn(
        corpus=_corpus(corpus),
        document_id=document_id,
        title=title,
        body=body,
        context=context,
        metadata=metadata or {},
    )


@mcp.tool()
def mem_search(
    corpus: str,
    query: str,
    n_results: int = 10,
) -> dict[str, Any]:
    """Semantic search within a corpus.

    Returns documents ranked by fused similarity across the
    header/body/context chunks. Never mixes corpora.
    """
    hits = _brain.search(
        corpus=_corpus(corpus),
        query=query,
        n_results=n_results,
    )
    return {
        "corpus": corpus,
        "query": query,
        "count": len(hits),
        "results": [
            {
                "document_id": h.document_id,
                "score": round(h.score, 4),
                "chunk_type": h.chunk_type,
                "text": h.text,
                "metadata": h.metadata,
            }
            for h in hits
        ],
    }


@mcp.tool()
def mem_query(
    corpus: str,
    query: str,
    n_results: int = 5,
) -> dict[str, Any]:
    """Simpler search variant with fewer default results, intended for agent loops."""
    return _brain.query(
        corpus=_corpus(corpus),
        query=query,
        n_results=n_results,
    )


@mcp.tool()
def mem_rag(
    corpus: str,
    query: str,
    n_results: int = 5,
    max_context_chars: int = 4000,
) -> dict[str, Any]:
    """Return a concatenated context block to inline into your next prompt.

    Gr0m_Mem intentionally does NOT call an LLM here — you (the calling
    agent) are the LLM. This just assembles the most relevant chunks into
    one block up to the character budget.
    """
    return _brain.rag(
        corpus=_corpus(corpus),
        query=query,
        n_results=n_results,
        max_context_chars=max_context_chars,
    )


@mcp.tool()
def mem_analyze(corpus: str) -> dict[str, Any]:
    """Return aggregate stats about a corpus (chunk counts, unique documents, model)."""
    return _brain.analyze(_corpus(corpus))


@mcp.tool()
def mem_list_corpora() -> dict[str, Any]:
    """List every corpus collection currently in the store."""
    return {
        "corpora": _brain.list_corpora(),
        "chroma_path": str(_config.chroma_path),
        "version": __version__,
    }


# ── Temporal knowledge graph tools ───────────────────────


@mcp.tool()
def mem_kg_add(
    subject: str,
    predicate: str,
    target: str,
    valid_from: str | None = None,
    confidence: float = 1.0,
    weight: float = 1.0,
    source_doc: str | None = None,
    data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Add a temporal triple to the knowledge graph.

    ``valid_from`` is an optional ISO-8601 UTC timestamp — when omitted we
    use the current wall clock. The fact-checker runs before the insert;
    in strict mode (default) a contradicting triple is rejected and this
    tool returns ``{"ok": false, "conflicts": [...]}``. Invalidate the
    existing edge with ``mem_kg_invalidate`` first, or flip this instance
    to ``fact_check_mode="off"`` via the ``GR0M_MEM_FACT_CHECK_MODE``
    environment variable.
    """
    try:
        edge = _brain.kg.add_triple(
            subject=subject,
            predicate=predicate,
            target=target,
            valid_from=_parse_iso(valid_from),
            weight=weight,
            confidence=confidence,
            source_doc=source_doc,
            data=data or {},
        )
    except ContradictionError as e:
        return {
            "ok": False,
            "conflicts": [
                {
                    "existing_edge_id": c.existing_edge_id,
                    "existing_target": c.existing_target,
                    "new_target": c.new_target,
                    "existing_valid_from": c.existing_valid_from.isoformat(),
                    "message": c.message(),
                }
                for c in e.conflicts
            ],
        }
    return {"ok": True, "edge": _edge_to_json(edge)}


@mcp.tool()
def mem_kg_query(
    entity: str,
    as_of: str | None = None,
    direction: str = "both",
) -> dict[str, Any]:
    """Return edges touching ``entity`` active at ``as_of``.

    ``as_of=None`` means "currently valid". Pass an ISO-8601 UTC timestamp
    for a historical slice (e.g. ``"2025-01-15T00:00:00+00:00"``).
    ``direction`` is ``"out"``, ``"in"``, or ``"both"``.
    """
    edges = _brain.kg.query_entity(
        entity=entity,
        as_of=_parse_iso(as_of),
        direction=direction,
    )
    return {
        "entity": entity,
        "as_of": as_of,
        "direction": direction,
        "count": len(edges),
        "edges": [_edge_to_json(e) for e in edges],
    }


@mcp.tool()
def mem_kg_invalidate(
    subject: str,
    predicate: str,
    target: str,
    ended_at: str | None = None,
) -> dict[str, Any]:
    """Mark every currently-active ``(subject, predicate, target)`` edge as ended.

    Does not delete rows — historical queries still see the edge. Returns
    the count of edges that were updated.
    """
    count = _brain.kg.invalidate(
        subject=subject,
        predicate=predicate,
        target=target,
        ended_at=_parse_iso(ended_at),
    )
    return {
        "subject": subject,
        "predicate": predicate,
        "target": target,
        "ended_at": ended_at,
        "invalidated": count,
    }


@mcp.tool()
def mem_kg_timeline(entity: str | None = None) -> dict[str, Any]:
    """Return every edge touching ``entity`` (or all edges), ordered by ``valid_from``.

    Unlike ``mem_kg_query``, this does not filter by time — it returns the
    literal full history, including closed edges.
    """
    edges = _brain.kg.timeline(entity=entity)
    return {
        "entity": entity,
        "count": len(edges),
        "edges": [_edge_to_json(e) for e in edges],
    }


@mcp.tool()
def mem_kg_stats() -> dict[str, Any]:
    """Aggregate counts for the knowledge graph (total / active / closed edges)."""
    return {
        "graph_db": str(_config.graph_db_path),
        "fact_check_mode": _brain.kg.fact_check_mode,
        **_brain.kg.stats(),
    }


# ── Loop-prevention: wakeup + decisions ──────────────────


@mcp.tool()
def mem_wakeup(scope: str = "global", token_budget: int = 200) -> dict[str, Any]:
    """Return a compact persistent-memory snapshot for session start.

    Call this ONCE at the beginning of every conversation. The returned
    ``text`` field is a short Markdown block of who the user is, what
    projects are active, what decisions are locked in, and what
    questions are still open — inline it into your working context
    immediately so you don't re-ask things you already know.

    ``token_budget`` caps the snapshot size; defaults to 200 tokens,
    which is typically enough for identity + a few preferences + the
    most important active decisions.
    """
    return _brain.wakeup.snapshot(scope=scope, token_budget=token_budget)


@mcp.tool()
def mem_remember(
    kind: str,
    text: str,
    scope: str = "global",
    subject: str | None = None,
    rationale: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Record a durable fact that should survive across sessions.

    ``kind`` must be one of: identity, preference, project, decision,
    question, milestone, context. Use this after learning anything the
    user would not want to tell you again next time.
    """
    if kind not in VALID_KINDS:
        return {
            "ok": False,
            "error": f"kind must be one of {VALID_KINDS}, got {kind!r}",
        }
    fact = _brain.wakeup.remember(
        kind=kind,
        text=text,
        scope=scope,
        subject=subject,
        rationale=rationale,
        metadata=metadata or {},
    )
    return {"ok": True, "fact": _fact_to_json(fact)}


@mcp.tool()
def mem_record_decision(
    subject: str,
    decision: str,
    rationale: str | None = None,
    scope: str = "global",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Log a concrete decision about ``subject``.

    Call this after the user locks in a choice ("we're using Postgres",
    "skip the refactor for now", "call the project Gr0m_Mem"). Future
    sessions will find it via ``mem_recall_decisions`` and won't re-ask.
    """
    fact = _brain.wakeup.record_decision(
        subject=subject,
        decision=decision,
        rationale=rationale,
        scope=scope,
        metadata=metadata or {},
    )
    return {"ok": True, "fact": _fact_to_json(fact)}


@mcp.tool()
def mem_recall_decisions(
    subject: str,
    scope: str = "global",
    limit: int = 10,
) -> dict[str, Any]:
    """Look up prior decisions about ``subject`` before asking the user.

    Call this BEFORE asking a question that feels familiar. If the
    return is non-empty, trust the prior decision instead of
    re-deriving or re-asking.
    """
    facts = _brain.wakeup.recall_decisions(subject, scope=scope, limit=limit)
    return {
        "subject": subject,
        "scope": scope,
        "count": len(facts),
        "decisions": [_fact_to_json(f) for f in facts],
    }


@mcp.tool()
def mem_forget(fact_id: str) -> dict[str, Any]:
    """Remove a fact from the wakeup store by id.

    Use this when a ``mem_remember`` record is stale or no longer
    applies (project retired, preference changed, question answered).
    Deletion is immediate and not reversible — the wakeup store does
    not keep tombstones. For facts that should be retained but no
    longer apply, prefer recording a newer fact of the same ``subject``
    that supersedes the old one.
    """
    removed = _brain.wakeup.forget(fact_id)
    return {"ok": removed, "fact_id": fact_id}


@mcp.tool()
def mem_status() -> dict[str, Any]:
    """Summary of backend choice, databases, and loop-prevention state.

    Useful for debugging why retrieval or wakeup isn't behaving as
    expected. Returns the active vector backend, why it was chosen, and
    key counts across all subsystems.
    """
    return {
        "version": __version__,
        "backend": _brain.backend_choice.name,
        "backend_reason": _brain.backend_choice.reason,
        "embed_model": _brain.embedding.model if _brain.embedding else None,
        "chroma_path": str(_config.chroma_path),
        "sqlite_vec_path": str(_config.sqlite_vec_path),
        "sqlite_fts_path": str(_config.sqlite_fts_path),
        "graph_db": str(_config.graph_db_path),
        "wakeup_db": str(_config.wakeup_db_path),
        "fact_check_mode": _brain.kg.fact_check_mode,
        "kg": _brain.kg.stats(),
        "wakeup": _brain.wakeup.stats(),
    }


# ── Entrypoint ───────────────────────────────────────────


def main() -> None:
    """Run the MCP server over stdio."""
    log.info("gr0m_mem %s starting (chroma=%s)", __version__, _config.chroma_path)
    mcp.run()


if __name__ == "__main__":
    main()
