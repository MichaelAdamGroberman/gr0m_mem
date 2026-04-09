# Security Policy

## Supported versions

Gr0m_Mem is in `0.1.x` alpha. Only the latest tagged release on `main`
and the latest tagged release on `semantic` receive security fixes.

| Branch     | Status  | Fixes |
|------------|---------|-------|
| `main`     | current | ✅    |
| `semantic` | current | ✅    |
| earlier    | n/a     | ❌    |

## Reporting a vulnerability

**Please do not open a public issue for security reports.** Instead:

1. Open a GitHub **private vulnerability report** via the "Report a
   vulnerability" button on the repository's Security tab:
   <https://github.com/MichaelAdamGroberman/gr0m_mem/security/advisories/new>
2. Or email the maintainer directly (see GitHub profile).

Include:

- The affected version and branch
- A minimal reproduction (scenario, config, and the exact command that
  demonstrates the issue)
- Impact — what data, state, or capability an attacker could gain

You will receive an initial response within 72 hours. Fixes for
confirmed vulnerabilities are prioritized; credit is given by default
unless you request anonymity.

## Scope

In scope:

- Shell injection, path traversal, and input-handling bugs in the CLI
  (`gr0m_mem ...`) and the Claude Code hooks (`hooks/*.sh`)
- Data leakage across corpora (Gr0m_Mem guarantees corpus isolation)
- Temporal KG bypasses of the fact checker that allow silent
  contradictions in strict mode
- Memory disclosure from the wakeup store or the FTS database to
  unauthorized processes on multi-user systems
- Package supply chain issues in the build and release workflows

Out of scope:

- Denial of service from a local caller flooding the SQLite database
  (Gr0m_Mem is designed for a single local agent — quotas are not a
  security feature here)
- Anything requiring an attacker who already has filesystem read
  access to the Gr0m_Mem data directory

## Hardening applied by default

- Hook scripts sanitize `SESSION_ID` with a strict whitelist before
  any path construction (`tr -cd 'a-zA-Z0-9_-'`) — fixes the class of
  bug tracked as MemPalace issue #110 in the related prior-art
  project.
- Every corpus is an isolated SQLite collection; there is no "default"
  tenant and tools require an explicit corpus argument.
- The fact checker runs on every `add_triple` and rejects
  contradictions in strict mode by default.
- Branch protection on `main` and `semantic` blocks force pushes and
  deletions and requires pull requests for outside contributions.
- CI runs ruff, mypy `--strict`, the full test suite, and the
  loop-prevention benchmark in strict mode on every push.
