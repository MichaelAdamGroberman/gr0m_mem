# Gr0m_Mem

Zero-install persistent memory brain for Claude Code and Claude Desktop that **stops Claude from re-asking and re-deriving** across sessions.

> **This is the `main` branch — the zero-install core.**
> No ChromaDB, no Ollama, no 1 GB embedding model. Pure CPython stdlib + a couple of pure-Python wheels. `pip install gr0m-mem` and it just works.
>
> For semantic retrieval (ChromaDB HNSW + Ollama embeddings) switch to the [`semantic`](https://github.com/MichaelAdamGroberman/gr0m_mem/tree/semantic) branch.

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

## Requirements

- **CPython 3.10, 3.11, or 3.12.** 3.13+ blocked until chromadb and the MCP SDK publish compatible wheels (affects the `semantic` branch; not an issue here).

## License

MIT — see [LICENSE](LICENSE).
