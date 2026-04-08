"""Gr0m_Mem command-line interface.

Subcommands:

* ``init``    — create the on-disk home and print configuration
* ``learn``   — ingest a document from stdin or a file
* ``search``  — semantic search within a corpus
* ``status``  — summary of all corpora in the store
* ``doctor``  — verify python / chromadb / ollama / embed model
* ``serve``   — run the MCP server (equivalent to ``python -m gr0m_mem.mcp_server``)

The CLI is intentionally thin — it just wraps :class:`gr0m_mem.brain.Brain`.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from gr0m_mem import __version__
from gr0m_mem.brain import Brain
from gr0m_mem.config import Config
from gr0m_mem.types import Corpus


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gr0m_mem",
        description="Honest memory brain for Claude Code and other MCP agents.",
    )
    parser.add_argument("--version", action="version", version=f"gr0m_mem {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="Create the on-disk home and print config")

    p_learn = sub.add_parser("learn", help="Ingest a document into a corpus")
    p_learn.add_argument("--corpus", required=True)
    p_learn.add_argument("--id", dest="document_id", required=True, help="Document id")
    p_learn.add_argument("--title", required=True)
    p_learn.add_argument(
        "--body",
        help="Body text. If omitted, reads from --file or stdin.",
    )
    p_learn.add_argument("--file", type=Path, help="Read body from this file")
    p_learn.add_argument("--context", default="", help="Optional context/relationship text")

    p_search = sub.add_parser("search", help="Semantic search within a corpus")
    p_search.add_argument("query")
    p_search.add_argument("--corpus", required=True)
    p_search.add_argument("-n", "--n-results", type=int, default=10)

    sub.add_parser("status", help="Show all corpora in the store")

    sub.add_parser("doctor", help="Verify python, backend, ollama, embed model")

    sub.add_parser("serve", help="Run the MCP server over stdio")

    p_hook = sub.add_parser("hook", help="Claude Code hook entry points")
    p_hook.add_argument("event", choices=["stop", "precompact"])
    p_hook.add_argument("--session-id", default="unknown")

    p_wakeup = sub.add_parser("wakeup", help="Print the persistent-memory snapshot")
    p_wakeup.add_argument("--scope", default="global")
    p_wakeup.add_argument("--tokens", type=int, default=200)

    p_remember = sub.add_parser("remember", help="Record a durable fact")
    p_remember.add_argument("--kind", required=True)
    p_remember.add_argument("--text", required=True)
    p_remember.add_argument("--scope", default="global")
    p_remember.add_argument("--subject", default=None)
    p_remember.add_argument("--rationale", default=None)

    return parser


# ── Command handlers ─────────────────────────────────────


def _cmd_init(_: argparse.Namespace) -> int:
    config = Config.from_env()
    config.ensure_dirs()
    print(f"gr0m_mem {__version__}")
    print(f"  home:         {config.home}")
    print(f"  chroma_path:  {config.chroma_path}")
    print(f"  graph_db:     {config.graph_db_path}")
    print(f"  state_db:     {config.state_db_path}")
    print(f"  ollama_url:   {config.ollama_url}")
    print(f"  embed_model:  {config.embed_model}")
    print("Initialized.")
    return 0


def _cmd_learn(args: argparse.Namespace) -> int:
    if args.body is not None:
        body = args.body
    elif args.file is not None:
        body = args.file.read_text(encoding="utf-8")
    elif not sys.stdin.isatty():
        body = sys.stdin.read()
    else:
        print("error: provide --body, --file, or pipe content on stdin", file=sys.stderr)
        return 2

    brain = Brain(Config.from_env())
    try:
        result = brain.learn(
            corpus=Corpus(args.corpus),
            document_id=args.document_id,
            title=args.title,
            body=body,
            context=args.context,
        )
    finally:
        brain.close()
    print(json.dumps(result, indent=2))
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    brain = Brain(Config.from_env())
    try:
        hits = brain.search(
            corpus=Corpus(args.corpus),
            query=args.query,
            n_results=args.n_results,
        )
    finally:
        brain.close()
    payload = {
        "corpus": args.corpus,
        "query": args.query,
        "count": len(hits),
        "results": [
            {
                "document_id": h.document_id,
                "score": round(h.score, 4),
                "chunk_type": h.chunk_type,
                "text": h.text,
            }
            for h in hits
        ],
    }
    print(json.dumps(payload, indent=2))
    return 0


def _cmd_status(_: argparse.Namespace) -> int:
    brain = Brain(Config.from_env())
    try:
        corpora = brain.list_corpora()
        print(f"gr0m_mem {__version__}")
        print(f"  chroma_path: {brain.config.chroma_path}")
        print(f"  corpora:     {len(corpora)}")
        for name in corpora:
            try:
                c = Corpus(name)
                stats = brain.analyze(c)
                print(
                    f"    - {name}: {stats['total_chunks']} chunks, "
                    f"~{stats['sampled_unique_documents']} documents sampled"
                )
            except ValueError:
                # Collections not created by Gr0m_Mem may not satisfy the name regex.
                print(f"    - {name}: (not a gr0m_mem corpus, skipped)")
    finally:
        brain.close()
    return 0


def _cmd_doctor(_: argparse.Namespace) -> int:
    import platform

    print(f"gr0m_mem {__version__}")
    print(f"  python:       {platform.python_version()} ({sys.executable})")

    try:
        import chromadb

        print(f"  chromadb:     {chromadb.__version__} (optional extra installed)")
    except Exception:  # noqa: BLE001
        print("  chromadb:     not installed (optional — fallback backend will be used)")

    try:
        import mcp  # noqa: F401

        print("  mcp sdk:      installed")
    except Exception as e:  # noqa: BLE001
        print(f"  mcp sdk:      MISSING ({e})")
        return 1

    config = Config.from_env()
    brain = Brain(config)
    try:
        print(f"  backend:      {brain.backend_choice.name}")
        print(f"    reason:     {brain.backend_choice.reason}")
        if brain.embedding is not None:
            if brain.embedding.health():
                print(f"  ollama:       reachable, {config.embed_model} present")
            else:
                print(f"  ollama:       UNREACHABLE ({config.ollama_url})")
                print(f"    hint: ollama pull {config.embed_model}")
        else:
            print("  ollama:       not needed (backend is lexical-only)")
        print(f"  kg stats:     {brain.kg.stats()}")
        print(f"  wakeup stats: {brain.wakeup.stats()}")
        print(f"  chroma_path:  {config.chroma_path}")
        print(f"  vectors.db:   {config.sqlite_vec_path}")
        print(f"  fts.db:       {config.sqlite_fts_path}")
        print(f"  graph.db:     {config.graph_db_path}")
        print(f"  wakeup.db:    {config.wakeup_db_path}")
    finally:
        brain.close()

    return 0


def _cmd_wakeup(args: argparse.Namespace) -> int:
    brain = Brain(Config.from_env())
    try:
        snap = brain.wakeup.snapshot(scope=args.scope, token_budget=args.tokens)
    finally:
        brain.close()
    print(json.dumps(snap, indent=2))
    return 0


def _cmd_remember(args: argparse.Namespace) -> int:
    from gr0m_mem.wakeup import VALID_KINDS

    if args.kind not in VALID_KINDS:
        print(f"error: --kind must be one of {VALID_KINDS}", file=sys.stderr)
        return 2
    brain = Brain(Config.from_env())
    try:
        fact = brain.wakeup.remember(
            kind=args.kind,
            text=args.text,
            scope=args.scope,
            subject=args.subject,
            rationale=args.rationale,
        )
    finally:
        brain.close()
    print(f"remembered: {fact.id} ({fact.kind}) {fact.text}")
    return 0


def _cmd_hook(args: argparse.Namespace) -> int:
    """Entry point for the Claude Code Stop / PreCompact hooks.

    Records a ``milestone`` fact so every hook fire is durable. Metadata
    includes the event type, sanitized session id, wall-clock timestamp,
    and (when we can compute it) the elapsed seconds since the last hook
    of the same event for the same session.

    Richer summarization (parsing the JSONL transcript on stdin, pulling
    out decisions and quoted user statements) is intentionally deferred:
    the transcript format is stream-shaped and harness-specific, and
    getting it wrong would silently drop state. For now the durable
    record is "a hook fired at T with session S", which is already
    enough to reconstruct the session graph from the wakeup store.
    """
    from gr0m_mem.wakeup import VALID_KINDS

    _ = VALID_KINDS  # imported for the side effect of validating the enum
    brain = Brain(Config.from_env())
    try:
        prior = [
            f
            for f in brain.wakeup.all_facts()
            if f.metadata.get("source") == "hook"
            and f.metadata.get("event") == args.event
            and f.metadata.get("session_id") == args.session_id
        ]
        elapsed_since_last: float | None = None
        if prior:
            newest = max(prior, key=lambda f: f.added_at)
            elapsed_since_last = (
                datetime.now(timezone.utc) - newest.added_at
            ).total_seconds()

        brain.wakeup.remember(
            kind="milestone",
            text=f"claude-code {args.event} session={args.session_id}",
            metadata={
                "source": "hook",
                "event": args.event,
                "session_id": args.session_id,
                "prior_hook_count": len(prior),
                "elapsed_since_last_s": elapsed_since_last,
            },
        )
    finally:
        brain.close()
    return 0


def _cmd_serve(_: argparse.Namespace) -> int:
    from gr0m_mem.mcp_server import main as mcp_main

    mcp_main()
    return 0


_HANDLERS = {
    "init": _cmd_init,
    "learn": _cmd_learn,
    "search": _cmd_search,
    "status": _cmd_status,
    "doctor": _cmd_doctor,
    "serve": _cmd_serve,
    "wakeup": _cmd_wakeup,
    "remember": _cmd_remember,
    "hook": _cmd_hook,
}


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return _HANDLERS[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
