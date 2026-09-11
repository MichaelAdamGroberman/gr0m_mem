#!/usr/bin/env bash
# Gr0m_Mem save hook (Claude Code Stop event).
#
# Flushes the most recent exchange into the wakeup store so a crash or
# unexpected /clear does not lose the last few minutes of work. Runs on
# every Claude Code Stop event; the Python side throttles repeats for
# the same session so we don't thrash the DB (see _cmd_hook in cli.py).
#
# Claude Code delivers the hook payload as JSON on stdin, shaped like
# {"session_id":"...","transcript_path":"...","cwd":"...",
#  "hook_event_name":"Stop","stop_hook_active":false} -- it does NOT set
# a $SESSION_ID environment variable. We read stdin first and fall back
# to $SESSION_ID / $CLAUDE_SESSION_ID env vars for manual or test
# invocation where nothing is piped in.
#
# Security: whatever the source, the raw session id is whitelisted hard
# before use in any path construction -- this is the fix for the
# shell-injection bug that hit MemPalace (Issue #110). Only
# alphanumerics, underscore, and hyphen are allowed; anything else is
# stripped. Empty session ids become "unknown".
#
# This hook must never block Claude Code: it always exits 0, even if
# stdin is empty/unparseable, jq and python3 are both missing, or the
# gr0m_mem CLI call itself fails.

set -o pipefail

# Read the full hook payload from stdin (JSON), if any is piped in.
HOOK_INPUT=""
if [ ! -t 0 ]; then
    HOOK_INPUT="$(cat 2>/dev/null)"
fi

RAW_SESSION_ID=""
HOOK_EVENT_NAME=""

if [ -n "$HOOK_INPUT" ]; then
    if command -v jq >/dev/null 2>&1; then
        RAW_SESSION_ID="$(printf '%s' "$HOOK_INPUT" | jq -r '.session_id // empty' 2>/dev/null)"
        HOOK_EVENT_NAME="$(printf '%s' "$HOOK_INPUT" | jq -r '.hook_event_name // empty' 2>/dev/null)"
    elif command -v python3 >/dev/null 2>&1; then
        RAW_SESSION_ID="$(printf '%s' "$HOOK_INPUT" | python3 -c '
import json, sys
try:
    data = json.load(sys.stdin)
    print(data.get("session_id") or "")
except Exception:
    pass
' 2>/dev/null)"
        HOOK_EVENT_NAME="$(printf '%s' "$HOOK_INPUT" | python3 -c '
import json, sys
try:
    data = json.load(sys.stdin)
    print(data.get("hook_event_name") or "")
except Exception:
    pass
' 2>/dev/null)"
    fi
fi

# Fall back to env vars (manual invocation / tests) if stdin gave nothing.
if [ -z "$RAW_SESSION_ID" ]; then
    RAW_SESSION_ID="${SESSION_ID:-${CLAUDE_SESSION_ID:-}}"
fi

SESSION_ID="$(printf %s "$RAW_SESSION_ID" | tr -cd 'a-zA-Z0-9_-')"
if [ -z "$SESSION_ID" ]; then
    SESSION_ID="unknown"
fi
export GR0M_MEM_SESSION_ID="$SESSION_ID"

PYTHON_BIN="${GR0M_MEM_PYTHON:-python3}"

if [ -n "$HOOK_EVENT_NAME" ]; then
    "$PYTHON_BIN" -m gr0m_mem hook stop --session-id "$SESSION_ID" \
        --hook-event-name "$HOOK_EVENT_NAME" "$@"
else
    "$PYTHON_BIN" -m gr0m_mem hook stop --session-id "$SESSION_ID" "$@"
fi

exit 0
