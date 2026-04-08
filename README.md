# Gr0m_Mem

Persistent memory brain for Claude Code and Claude Desktop that **stops Claude from re-asking and re-deriving** across sessions.

> **Status: v0.1.0 alpha.** Benchmarks are being wired in. Every claim in this README that cites a number will link to a committed `benchmarks/results/*.json` file. If you see a claim without a link, open an issue.

## The problem

Claude forgets everything when a session ends. Next time you talk to it:

- It re-introduces itself.
- It asks what you're working on — again.
- It re-derives the same architectural decision you already locked in yesterday.
- It loses track of which features shipped and re-suggests them.

Other memory systems try to fix this with "let an LLM decide what to remember." That path is expensive, loses context, and still leaks the reasoning. Gr0m_Mem takes a different path: **record everything important explicitly, surface it at session start, and refuse to contradict it without you saying so.**

## How it fixes the loop

Four tools (and a Claude Code hook) are all you need:

| When | Tool | Effect |
|---|---|---|
| Session start | `mem_wakeup` | Returns a 200-token snapshot of identity / preferences / projects / decisions / open questions. Claude inlines it and stops re-introducing. |
| After a decision | `mem_record_decision` | Persists the decision + rationale against a subject. |
| Before asking a familiar question | `mem_recall_decisions` | Retrieves prior decisions on that subject. If any exist, Claude uses them instead of re-asking. |
| Learning anything durable | `mem_remember` | Stores a preference, project, milestone, context fact, or open question. |

The plugin's Stop and PreCompact hooks flush a milestone after every session and before every context compaction, so nothing high-value is lost to `/clear` or window compression.

## Zero-install promise

`pip install gr0m-mem` always produces a working brain. Gr0m_Mem auto-picks a backend based on what's available:

1. **`chromadb`** — best retrieval. Install with `pip install "gr0m-mem[chromadb]"` if you want semantic search over Ollama embeddings backed by ChromaDB HNSW.
2. **`sqlite_vec`** — pure Python cosine similarity over embeddings stored in SQLite. Only needs Ollama (`ollama pull mxbai-embed-large`). No compiled extras.
3. **`sqlite_fts`** — SQLite FTS5 BM25. Ships with CPython. No embeddings, no Ollama, no extras. Lexical only, but `mem_wakeup` + `mem_record_decision` still prevent loops — retrieval quality is the only thing affected.

Run `gr0m_mem doctor` to see which backend was picked and why. Force a specific one with `GR0M_MEM_BACKEND=chromadb|sqlite_vec|sqlite_fts`.

## Also in the box

- **Temporal knowledge graph** wired into every retrieval path. Edges carry `valid_from` / `valid_to` and `mem_kg_query` accepts an explicit `as_of`. The fact checker runs on every `add_triple` and rejects contradictions in strict mode (the thing MemPalace's `fact_checker.py` advertised but never wired).
- **Per-corpus isolation**. Every corpus is its own collection — documents from different projects are never mixed.
- **Reproducible benchmarks**. Runners for LongMemEval, LoCoMo, and a loop-prevention scenario land in `benchmarks/` in Phase D, with committed results.

## Install

### As a Claude Code plugin (preferred)

```bash
claude plugin marketplace add <owner>/gr0m_mem
claude plugin install --scope user gr0m_mem
```

The plugin registers the MCP server plus the Stop / PreCompact hooks in one step.

### From PyPI

```bash
pip install gr0m-mem                      # minimal, sqlite_fts fallback
pip install "gr0m-mem[chromadb]"          # with ChromaDB backend
pip install "gr0m-mem[chromadb,tokens]"   # + real tiktoken counts
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

- **CPython 3.10, 3.11, or 3.12.** 3.13+ blocked until chromadb and the MCP SDK publish compatible wheels.
- **(Optional) Ollama** with `mxbai-embed-large` for the vector backends.

## License

MIT — see [LICENSE](LICENSE).
