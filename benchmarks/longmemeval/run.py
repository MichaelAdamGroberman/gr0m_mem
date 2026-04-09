"""LongMemEval runner for Gr0m_Mem.

Measures R@5 on the LongMemEval benchmark (Hu et al., 2024). The dataset
is not committed — it's too large and lives in academic repos — so this
runner expects a path to a local copy downloaded by the user or by CI.

Dataset format expected (matches the published LongMemEval-s release):
a JSON list where each item is::

    {
        "question_id": "...",
        "question": "...",
        "answer": "...",
        "haystack": [
            {"id": "...", "content": "..."},
            ...
        ],
        "gold_evidence_ids": ["...", ...]
    }

The runner ingests every haystack document into a fresh ``Corpus``
named after the question_id (so questions never contaminate each other),
calls :meth:`gr0m_mem.brain.Brain.search` with the question text, and
reports whether any of the top-5 hits' ``document_id`` appears in
``gold_evidence_ids``.

If the dataset path does not exist the runner exits with a clear
message instead of guessing — we never fabricate benchmark results.
"""

from __future__ import annotations

import argparse
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


def _run_one_question(
    brain: Brain,
    question_id: str,
    question: str,
    haystack: list[dict[str, Any]],
    gold_ids: list[str],
    k: int = 5,
) -> dict[str, Any]:
    # Use the question_id as a corpus name so questions never contaminate
    # each other. LongMemEval question ids can contain characters the
    # Corpus validator rejects — fall back to a stable hash when needed.
    try:
        corpus = Corpus(question_id.lower().replace(" ", "_"))
    except ValueError:
        import hashlib

        h = hashlib.sha1(question_id.encode("utf-8")).hexdigest()[:16]
        corpus = Corpus(f"lme_{h}")

    for doc in haystack:
        brain.learn(
            corpus=corpus,
            document_id=str(doc["id"]),
            title=str(doc.get("title", doc["id"])),
            body=str(doc.get("content", "")),
        )

    hits = brain.search(corpus=corpus, query=question, n_results=k)
    recovered = [h.document_id for h in hits]
    gold_set = set(gold_ids)
    hit_at_k = any(r in gold_set for r in recovered)

    return {
        "question_id": question_id,
        "question": question,
        "gold_ids": gold_ids,
        "retrieved": recovered,
        "hit_at_k": hit_at_k,
    }


def run(dataset_path: Path, backend: str, k: int = 5) -> dict[str, Any]:
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
            for item in data:
                per_question.append(
                    _run_one_question(
                        brain=brain,
                        question_id=str(item.get("question_id", "")),
                        question=str(item.get("question", "")),
                        haystack=list(item.get("haystack", [])),
                        gold_ids=[str(g) for g in item.get("gold_evidence_ids", [])],
                        k=k,
                    )
                )
        finally:
            brain.close()

    hits = sum(1 for q in per_question if q["hit_at_k"])
    total = len(per_question)

    return {
        "benchmark": "longmemeval",
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
    parser = argparse.ArgumentParser(prog="gr0m_mem-bench-longmemeval")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--backend", default="auto")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    if not args.dataset.exists():
        print(
            f"error: dataset not found: {args.dataset}\n"
            "Download LongMemEval from the published release and point "
            "--dataset at the JSON file. Gr0m_Mem does not fabricate "
            "benchmark numbers.",
            file=sys.stderr,
        )
        return 2

    result = run(args.dataset, backend=args.backend, k=args.k)
    payload = json.dumps(result, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n", encoding="utf-8")
        print(
            f"longmemeval: R@{args.k}={result['recall_at_k']:.4f} "
            f"({result['hits_at_k']}/{result['total_questions']})",
            file=sys.stderr,
        )
        print(f"results written to {args.out}", file=sys.stderr)
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
