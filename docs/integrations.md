# Gr0m_Mem — integrations

Gr0m_Mem is universally compatible with any LLM runtime through one of
three paths, in descending order of preference:

1. **MCP server** — for any Model Context Protocol client
2. **CLI shell-out** — for any agent that can run shell commands
3. **Paste-into-system-prompt** — see [`UNIVERSAL_PROMPT.md`](../UNIVERSAL_PROMPT.md)

This document shows the exact setup for every mainstream client. If
your client is missing, open an issue — we will add it.

---

## Claude Code (recommended)

Native Claude Code plugin with Stop + PreCompact hooks:

```bash
claude plugin marketplace add MichaelAdamGroberman/gr0m_mem
claude plugin install --scope user gr0m_mem
```

Restart Claude Code. Type `/mcp` to verify `gr0m_mem` is connected.

The plugin registers the MCP server and both hooks (Stop +
PreCompact) in one step. Session state is flushed automatically.

---

## Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "gr0m_mem": {
      "command": "python",
      "args": ["-m", "gr0m_mem.mcp_server"]
    }
  }
}
```

Restart Claude Desktop. Claude Desktop does not have hooks, but it
reads the MCP server `instructions` field and follows the
loop-prevention protocol at session start.

---

## Cursor

Add to `~/.cursor/mcp.json` (or the project-local `.cursor/mcp.json`):

```json
{
  "mcpServers": {
    "gr0m_mem": {
      "command": "python",
      "args": ["-m", "gr0m_mem.mcp_server"]
    }
  }
}
```

For session-start protocol, also add to your Cursor Rules (Settings
→ Rules for AI):

```
You have Gr0m_Mem memory. On every new conversation, call
mem_wakeup(token_budget=200) first and inline the returned text
into your context. Before asking a question that sounds familiar,
call mem_recall_decisions(subject). After locking in a decision,
call mem_record_decision(subject, decision, rationale).
```

---

## Gemini CLI

Edit `~/.config/gemini/config.json`:

```json
{
  "mcpServers": {
    "gr0m_mem": {
      "command": "python",
      "args": ["-m", "gr0m_mem.mcp_server"]
    }
  }
}
```

Gemini CLI reads MCP server instructions automatically.

---

## Continue (VS Code / JetBrains)

Add to your Continue config (`~/.continue/config.yaml`):

```yaml
mcpServers:
  - name: gr0m_mem
    command: python
    args: ["-m", "gr0m_mem.mcp_server"]
```

---

## Cline (VS Code)

Open the Cline MCP panel and add a server:

- **Name:** `gr0m_mem`
- **Command:** `python -m gr0m_mem.mcp_server`

---

## Zed

Add to your Zed settings (`~/.config/zed/settings.json`):

```json
{
  "context_servers": {
    "gr0m_mem": {
      "command": {
        "path": "python",
        "args": ["-m", "gr0m_mem.mcp_server"]
      }
    }
  }
}
```

---

## Aider

Aider does not speak MCP directly, so use the CLI shell-out path. Add
to your `.aider.conf.yml`:

```yaml
read:
  - UNIVERSAL_PROMPT.md
```

Then `aider` will load [`UNIVERSAL_PROMPT.md`](../UNIVERSAL_PROMPT.md)
as part of every session, and Claude (or whichever model you route
Aider to) will call `gr0m_mem wakeup` via shell.

---

## OpenAI Agents SDK

Use the MCP support if your runtime exposes it; otherwise add a tool
wrapper around the CLI:

```python
from agents import Agent, function_tool
import subprocess
import json

@function_tool
def mem_wakeup(token_budget: int = 200) -> dict:
    """Return the Gr0m_Mem persistent-memory snapshot."""
    out = subprocess.check_output(
        ["gr0m_mem", "wakeup", "--tokens", str(token_budget)],
        text=True,
    )
    return json.loads(out)

@function_tool
def mem_record_decision(subject: str, decision: str, rationale: str = "") -> None:
    subprocess.run(
        ["gr0m_mem", "remember",
         "--kind", "decision",
         "--subject", subject,
         "--text", decision,
         "--rationale", rationale],
        check=True,
    )

agent = Agent(
    name="...",
    instructions="...",
    tools=[mem_wakeup, mem_record_decision],
)
```

---

## LangChain / LlamaIndex

Wrap the CLI with a standard `Tool`:

```python
from langchain.tools import ShellTool, Tool
import subprocess, json

def wakeup(_: str = "") -> str:
    out = subprocess.check_output(
        ["gr0m_mem", "wakeup", "--tokens", "200"], text=True
    )
    return json.loads(out)["text"]

mem_wakeup = Tool(
    name="mem_wakeup",
    description="Return the persistent memory snapshot for this user.",
    func=wakeup,
)
```

---

## Codex CLI / OpenAI Codex

The OpenAI Codex CLI speaks MCP. Add to `~/.codex/config.toml`:

```toml
[mcp_servers.gr0m_mem]
command = "python"
args    = ["-m", "gr0m_mem.mcp_server"]
```

---

## Any other MCP client

The MCP server is launched as:

```
python -m gr0m_mem.mcp_server
```

Point your client's MCP launcher at that command. The server uses
stdio transport by default; it exposes 17 tools and sends a
loop-prevention protocol in its `instructions` string that any
MCP-compliant client will surface to the model.

---

## No MCP at all?

Use [`UNIVERSAL_PROMPT.md`](../UNIVERSAL_PROMPT.md). It defines the
same four-rule protocol in prose and tells the model to use the
`gr0m_mem` CLI via shell. Works with raw OpenAI API calls, locally
hosted Llama, Mistral, Gemini via REST, etc. — anything that can
execute a command.
