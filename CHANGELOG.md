# Changelog

All notable changes to Gr0m_Mem are documented here. The format is based
on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] — 2026-04-08

First public alpha. Ships the minimum viable loop-prevention core and
both branches (zero-dep FTS5 on `main`, ChromaDB + SQLite-vector + Ollama
on `semantic`).

### Added

- **Core retrieval** (Phase A) — `ChunkedDocument` 3-chunk strategy
  (header / body / context), in-process ChromaDB backend on `semantic`,
  and a thin score-fusion facade (`Brain.search`) that keeps
  backend-specific retrieval shapes hidden from callers.
- **Temporal knowledge graph** (Phase B) — `TemporalEdge`, SQLite
  persistence with partial indexes on `valid_to IS NULL`,
  `active_view(as_of=…)` traversal that refuses to run without an
  explicit temporal decision, `KnowledgeGraph` facade, and a
  `FactChecker` wired into every `add_triple` with `strict` / `warn` /
  `off` modes. `ContradictionError` carries actionable resolution text.
- **Backend cascade** (Phase C) — three backends with auto-selection:
  - `chromadb`   — best retrieval, HNSW cosine, `pip install gr0m-mem[chromadb]`
  - `sqlite_vec` — pure-SQLite with Python cosine over Ollama embeddings
  - `sqlite_fts` — SQLite FTS5 BM25, zero Python extras, works anywhere
- **Wakeup loop-prevention** (Phase C) — `Wakeup` SQLite store with
  `remember`, `record_decision`, `recall_decisions`, `touch`, `forget`,
  and a `snapshot(token_budget=N)` that orders facts by priority and
  respects the budget using `tiktoken` (or a 4-chars-per-token fallback).
- **MCP tools** — 17 total, grouped into core retrieval (6), temporal KG
  (5), loop prevention (5), and diagnostics (1). Strong protocol in the
  server `instructions` string: wakeup at session start, recall before
  asking, record after deciding.
- **Claude Code hooks** — Stop and PreCompact shell scripts with
  sanitized `SESSION_ID` (`tr -cd 'a-zA-Z0-9_-'`) fixing MemPalace
  issue #110 from day one. Registered in `.claude-plugin/plugin.json`
  using `${CLAUDE_PLUGIN_ROOT}`.
- **CLI** — `init`, `learn`, `search`, `status`, `doctor`, `wakeup`,
  `remember`, `hook`, `serve`. `doctor` reports the active backend and
  the reason it was chosen.
- **Benchmarks** (Phase D) — loop prevention runner with 8 committed
  scenarios, LongMemEval runner, LoCoMo runner. First results file:
  [`benchmarks/results/2026-04-08-loop-prevention.json`](benchmarks/results/2026-04-08-loop-prevention.json)
  — 8/8 scenarios passed, 15/15 probes passed.
- **CI** — GitHub Actions for ruff / mypy / pytest on macOS and Ubuntu
  across Python 3.10 / 3.11 / 3.12, weekly benchmark cron, tag-triggered
  wheel build + PyPI publish via OIDC trusted publisher.

### Decisions locked in

- **`sqlite_fts` is the zero-dep default.** `pip install gr0m-mem` must
  never fail because of a missing embedding model or Rust bindings. The
  FTS5 tokenizer is Porter + unicode61.
- **The chunk-fusion score is clamped to `[0, 1]`.** Additional matching
  chunks get a 30% bonus, capped so the public score stays in the
  documented range.
- **No LLM is called by `mem_rag`.** Gr0m_Mem produces the context; the
  calling agent is the LLM. This is deliberate — it's what MemPalace's
  96.6% raw-mode result comes from and what makes this brain cheap.
- **Temporal filtering is mandatory.** `active_view()` raises `TypeError`
  if the caller forgets to pass `as_of`. `None` is a valid, explicit
  choice meaning "currently valid"; omitting the argument is not.
- **Ollama + 1GB embedding model live on a separate `semantic` branch.**
  Main branch ships a memory brain that works before you download
  anything else.

### Honest limitations

- `sqlite_fts` is lexical only — it won't find "vehicle" when you search
  for "car". Install `gr0m-mem[chromadb]` or switch to the `semantic`
  branch for semantic retrieval.
- The Claude Code hooks currently record a milestone per fire and
  include elapsed-since-last-hook metadata, but do not yet parse the
  JSONL transcript. Richer transcript summarization is tracked for a
  follow-up release — the current behavior is durable but not smart.
- Benchmark runs for LongMemEval and LoCoMo require you to download the
  upstream dataset yourself; runners refuse to fabricate results if the
  dataset is missing.
