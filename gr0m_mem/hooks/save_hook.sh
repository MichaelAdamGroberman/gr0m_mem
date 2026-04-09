#!/usr/bin/env bash
# Gr0m_Mem save hook (Claude Code Stop event).
#
# Flushes the most recent exchange into the wakeup store so a crash or
# unexpected /clear does not lose the last few minutes of work. Runs on
# every Claude Code Stop event; the Python side handles throttling so we
# don't thrash the DB.
#
# Security: the Claude Code session id is passed in as $SESSION_ID from
# the hook payload. We whitelist it hard before using it in any path
# construction — this is the fix for the shell-injection bug that hit
# MemPalace (Issue #110). Only alphanumerics, underscore, and hyphen are
# allowed; anything else is stripped. Empty session ids become "unknown".

set -euo pipefail

SESSION_ID="${SESSION_ID:-}"
SESSION_ID="$(printf %s "$SESSION_ID" | tr -cd 'a-zA-Z0-9_-')"
if [ -z "$SESSION_ID" ]; then
    SESSION_ID="unknown"
fi
export GR0M_MEM_SESSION_ID="$SESSION_ID"

PYTHON_BIN="${GR0M_MEM_PYTHON:-python3}"

exec "$PYTHON_BIN" -m gr0m_mem hook stop --session-id "$SESSION_ID" "$@"
