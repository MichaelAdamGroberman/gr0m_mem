"""LoCoMo runner for Gr0m_Mem (session-level R@10).

Dataset format expected (matches the published LoCoMo release): a JSON
list where each item is a conversation with segmented sessions::

    {
        "conversation_id": "...",
        "sessions": [
            {"id": "...", "turns": [...]},
            ...
        ],
        "questions": [
            {
                "id": "...",
                "question": "...",
                "gold_session_ids": ["...", ...]
            },
            ...
        ]
    }

Each session is ingested as one Gr0m_Mem document (we flatten the
turns into one body). Each question searches the conversation's corpus
and we check whether any of the top-10 hits' document_id matches a
gold session id.

Like :mod:`benchmarks.longmemeval.run`, this runner refuses to guess if
the dataset is missing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gr0m_mem import __version__
from gr0m_mem.brain import Brain
from gr0m_mem.config import Config
from gr0m_mem.types import Corpus


def _config_for_scratch(tmp_path: Path, backend: str) -> Config:
    return Config(
        home=tmp_path,
        chroma_path=tmp_path / "chroma_db",
        sqlite_vec_path=tmp_path / "vectors.db",
        sqlite_fts_path=tmp_path / "fts.db",
        graph_db_path=tmp_path / "graph.db",
        state_db_path=tmp_path / "state.db",
        wakeup_db_path=tmp_path / "wakeup.db",
        ollama_url="http://localhost:11434",
        embed_model="mxbai-embed-large",
        fact_check_mode="off",
        backend=backend,
    )


def _flatten_turns(turns: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for t in turns:
        speaker = t.get("speaker", "")
        text = t.get("text") or t.get("content") or ""
        parts.append(f"{speaker}: {text}".strip())
    return "\n".join(parts)


def _safe_corpus(raw: str) -> Corpus:
    try:
        return Corpus(raw.lower().replace(" ", "_"))
    except ValueError:
        h = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
        return Corpus(f"locomo_{h}")


def _run_conversation(
    brain: Brain,
    conversation: dict[str, Any],
    k: int,
) -> list[dict[str, Any]]:
    corpus = _safe_corpus(str(conversation.get("conversation_id", "conversation")))

    for session in conversation.get("sessions", []):
        body = _flatten_turns(list(session.get("turns", [])))
        brain.learn(
            corpus=corpus,
            document_id=str(session["id"]),
            title=f"session {session['id']}",
            body=body,
        )

    out: list[dict[str, Any]] = []
    for question in conversation.get("questions", []):
        hits = brain.search(
            corpus=corpus,
            query=str(question["question"]),
            n_results=k,
        )
        retrieved = [h.document_id for h in hits]
        gold_set = {str(g) for g in question.get("gold_session_ids", [])}
        out.append(
            {
                "conversation_id": conversation.get("conversation_id"),
                "question_id": question.get("id"),
                "question": question.get("question"),
                "gold_ids": sorted(gold_set),
                "retrieved": retrieved,
                "hit_at_k": any(r in gold_set for r in retrieved),
            }
        )
    return out


def run(dataset_path: Path, backend: str, k: int = 10) -> dict[str, Any]:
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(
            f"expected a JSON list at top level of {dataset_path}, got {type(data).__name__}"
        )

    per_question: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as td:
        cfg = _config_for_scratch(Path(td), backend)
        brain = Brain(cfg)
        try:
            for conversation in data:
                per_question.extend(_run_conversation(brain, conversation, k=k))
        finally:
            brain.close()

    hits = sum(1 for q in per_question if q["hit_at_k"])
    total = len(per_question)

    return {
        "benchmark": "locomo",
        "run_at": datetime.now(tz=timezone.utc).isoformat(),
        "environment": {
            "gr0m_mem_version": __version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "backend": backend,
            "k": k,
        },
        "dataset": str(dataset_path),
        "total_questions": total,
        "hits_at_k": hits,
        "recall_at_k": hits / total if total else 0.0,
        "per_question": per_question,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gr0m_mem-bench-locomo")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--backend", default="auto")
    parser.add_argument("-k", type=int, default=10)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    if not args.dataset.exists():
        print(
            f"error: dataset not found: {args.dataset}\n"
            "Download LoCoMo from the published release and point --dataset "
            "at the JSON file. Gr0m_Mem does not fabricate benchmark numbers.",
            file=sys.stderr,
        )
        return 2

    result = run(args.dataset, backend=args.backend, k=args.k)
    payload = json.dumps(result, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n", encoding="utf-8")
        print(
            f"locomo: R@{args.k}={result['recall_at_k']:.4f} "
            f"({result['hits_at_k']}/{result['total_questions']})",
            file=sys.stderr,
        )
        print(f"results written to {args.out}", file=sys.stderr)
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
