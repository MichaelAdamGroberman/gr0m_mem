# Gr0m_Mem

[![CI](https://github.com/MichaelAdamGroberman/gr0m_mem/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/MichaelAdamGroberman/gr0m_mem/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/gr0m-mem.svg?logo=pypi&logoColor=white&label=PyPI)](https://pypi.org/project/gr0m-mem/)
[![Python](https://img.shields.io/pypi/pyversions/gr0m-mem.svg?logo=python&logoColor=white)](https://pypi.org/project/gr0m-mem/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Downloads](https://static.pepy.tech/badge/gr0m-mem)](https://pepy.tech/project/gr0m-mem)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)

Zero-install persistent memory brain for **any** LLM runtime (Claude Code, Claude Desktop, Cursor, Gemini CLI, Continue, Cline, Zed, OpenAI Codex CLI, Aider, raw OpenAI / Anthropic / Gemini APIs, or a local Llama) that **stops the model from re-asking and re-deriving** across sessions.

> **This is the `main` branch — the zero-install core.**
> No ChromaDB, no Ollama, no 1 GB embedding model. Pure CPython stdlib + a couple of pure-Python wheels. `pip install gr0m-mem` and it just works.
>
> For semantic retrieval (ChromaDB HNSW + Ollama embeddings) switch to the [`semantic`](https://github.com/MichaelAdamGroberman/gr0m_mem/tree/semantic) branch.

## Works with any LLM

Gr0m_Mem is universally compatible through three integration paths:

- **MCP server** — Claude Code, Claude Desktop, Cursor, Gemini CLI, Continue, Cline, Zed, OpenAI Codex CLI, and any other [Model Context Protocol](https://modelcontextprotocol.io) client. Setup snippets for every major client in [`docs/integrations.md`](docs/integrations.md).
- **CLI shell-out** — any agent framework that can run shell commands (OpenAI Agents SDK, LangChain, LlamaIndex, Aider, raw API callers): wrap `gr0m_mem wakeup`, `gr0m_mem remember`, and `gr0m_mem search` as tools.
- **Paste-into-system-prompt** — models with no MCP and no tool calling at all: copy [`UNIVERSAL_PROMPT.md`](UNIVERSAL_PROMPT.md) into your system prompt and the model will drive the CLI via shell.

The loop-prevention protocol is the same across all three paths.

## The problem

Claude forgets everything when a session ends. Next time you talk to it:

- It re-introduces itself.
- It asks what you're working on — again.
- It re-derives the same architectural decision you already locked in yesterday.
- It loses track of which features shipped and re-suggests them.

Other memory systems try to fix this with "let an LLM decide what to remember." That path is expensive, loses context, and still leaks reasoning. Gr0m_Mem takes the other path: **record everything important explicitly, surface it at session start, and refuse to contradict it without you saying so.**

## How it fixes the loop

Four tools (and two Claude Code hooks) are the entire product:

| When | Tool | Effect |
|---|---|---|
| Session start | `mem_wakeup` | Returns a token-budgeted snapshot of identity / preferences / projects / decisions / open questions. Claude inlines it and stops re-introducing. |
| After a decision | `mem_record_decision` | Persists the decision + rationale against a subject. |
| Before asking a familiar question | `mem_recall_decisions` | Retrieves prior decisions on that subject. If any exist, Claude uses them instead of re-asking. |
| Learning anything durable | `mem_remember` | Stores a preference, project, milestone, context fact, or open question. |

The plugin's Stop and PreCompact hooks flush a milestone after every session and before every context compaction, so nothing high-value is lost to `/clear` or window compression. Session ids are whitelisted (`tr -cd 'a-zA-Z0-9_-'`) before any path touch — the shell-injection bug MemPalace had to patch (Issue #110) is fixed by design here.

## Zero-install promise

`pip install gr0m-mem` always produces a working brain. The main branch has exactly one backend:

- **`sqlite_fts`** — SQLite FTS5 BM25 full-text search. Ships with CPython's stdlib `sqlite3` on every mainstream platform. No compiled extras, no embedding model, no Ollama, no network. Lexical-only, but `mem_wakeup` + `mem_record_decision` don't care about the backend — they use their own SQLite table.

Run `gr0m_mem doctor` to verify.

### Want semantic retrieval too?

Switch to the [`semantic`](https://github.com/MichaelAdamGroberman/gr0m_mem/tree/semantic) branch. It adds two more backends with auto-selection:

- **`chromadb`** — HNSW cosine over ChromaDB, best retrieval quality
- **`sqlite_vec`** — pure-Python cosine over SQLite rows, using Ollama for embeddings

Both require either the `chromadb` optional extra or a running Ollama with `mxbai-embed-large` (~1 GB). The semantic branch is a drop-in replacement — same tools, same API, richer retrieval.

## Also in the box

- **Temporal knowledge graph** with mandatory `as_of` filtering. `TemporalEdge` carries `valid_from` / `valid_to`, SQLite persistence with partial indexes on `valid_to IS NULL`, and `active_view()` that refuses to run without an explicit temporal decision. The `FactChecker` runs on every `add_triple` and rejects contradictions in strict mode (the thing MemPalace's `fact_checker.py` advertised but never wired).
- **Per-corpus isolation.** Every corpus is its own FTS5 table — documents from different projects are never mixed. All tools require an explicit `corpus` argument; there is no default tenant.
- **Reproducible benchmarks.** The loop-prevention benchmark (8 scenarios, 15 probes) runs in CI on every push and must stay at 100%. First committed result: [`benchmarks/results/2026-04-08-loop-prevention.json`](benchmarks/results/2026-04-08-loop-prevention.json).

## Install

### As a Claude Code plugin (preferred)

```bash
claude plugin marketplace add MichaelAdamGroberman/gr0m_mem
claude plugin install --scope user gr0m_mem
```

The plugin registers the MCP server plus the Stop / PreCompact hooks in one step.

### From PyPI

```bash
pip install gr0m-mem              # zero-install core (this branch)
pip install "gr0m-mem[tokens]"    # + real tiktoken token counts
```

Then point your MCP client at `python -m gr0m_mem.mcp_server`.

## Quick start

```bash
gr0m_mem doctor
gr0m_mem remember --kind identity --text "Michael, software engineer, macOS"
gr0m_mem remember --kind preference --text "terse responses, no trailing summaries"
gr0m_mem wakeup --tokens 200
```

Now open Claude Code. It calls `mem_wakeup` at session start and sees you before the first message.

## What it actually looks like

Real, unedited terminal output from a fresh install on macOS:

### `gr0m_mem doctor`

```console
$ gr0m_mem doctor
gr0m_mem 0.1.0 (main branch — zero-install core)
  python:       3.12.12 (/Users/michaelgroberman/Gr0m_Mem/.venv/bin/python)
  mcp sdk:      installed
  backend:      sqlite_fts
    reason:     main branch ships only the SQLite FTS5 backend — no Ollama,
                no chromadb, no downloads.
  ollama:       not needed on this branch
  kg stats:     {'total': 0, 'active': 0, 'closed': 0}
  wakeup stats: {'total': 0}
  fts.db:       ~/.gr0m_mem/fts.db
  graph.db:     ~/.gr0m_mem/graph.db
  wakeup.db:    ~/.gr0m_mem/wakeup.db
```

### Recording a few facts and a decision

```console
$ gr0m_mem remember --kind identity --text "Michael, software engineer on macOS"
remembered: d5b976ce-7832-4e61-8323-d08cd56d9177 (identity) Michael, software engineer on macOS

$ gr0m_mem remember --kind preference --text "terse responses, no trailing summaries"
remembered: 3f28a0be-78f4-4fae-ac71-4e6db6818f03 (preference) terse responses, no trailing summaries

$ python -c "
> from gr0m_mem.brain import Brain
> from gr0m_mem.config import Config
> b = Brain(Config.from_env())
> b.wakeup.record_decision(
>     subject='backend',
>     decision='sqlite_fts is the zero-dep default',
>     rationale='pip install gr0m-mem must never fail',
> )
> b.close()
> "
```

### `gr0m_mem wakeup --tokens 200` — what Claude sees at session start

```console
$ gr0m_mem wakeup --tokens 200
{
  "scope": "global",
  "token_budget": 200,
  "tokens_used": 50,
  "facts_included": 3,
  "facts_total": 3,
  "text": "## IDENTITY\n- Michael, software engineer on macOS\n\n## PREFERENCE\n- terse responses, no trailing summaries\n\n## DECISION\n- backend: sqlite_fts is the zero-dep default (pip install gr0m-mem must never fail)"
}
```

The agent inlines the `text` field at the top of every conversation and stops re-asking who you are.

## Requirements

- **CPython 3.10, 3.11, or 3.12.** 3.13+ blocked until chromadb and the MCP SDK publish compatible wheels (affects the `semantic` branch; not an issue here).

## License

MIT — see [LICENSE](LICENSE).

## Contact

Maintained by **Michael Adam Groberman**.

- **GitHub**: [@MichaelAdamGroberman](https://github.com/MichaelAdamGroberman)
- **LinkedIn**: [michael-adam-groberman](https://www.linkedin.com/in/michael-adam-groberman/)

For security reports, use GitHub private vulnerability advisories (see [SECURITY.md](SECURITY.md)) — **do not** use LinkedIn DMs for sensitive disclosures.
