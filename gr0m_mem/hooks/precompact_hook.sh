#!/usr/bin/env bash
# Gr0m_Mem pre-compaction hook (Claude Code PreCompact event).
#
# Fires immediately before Claude Code compresses the context window. We
# force a synchronous flush so any high-value facts from the soon-to-be-
# discarded turns land in the wakeup store before they are lost. Unlike
# the save hook, this one does NOT throttle — every compaction event is
# a last chance.
#
# Security: same SESSION_ID whitelist as save_hook.sh. See MemPalace
# Issue #110 for the bug we are fixing by default here.

set -euo pipefail

SESSION_ID="${SESSION_ID:-}"
SESSION_ID="$(printf %s "$SESSION_ID" | tr -cd 'a-zA-Z0-9_-')"
if [ -z "$SESSION_ID" ]; then
    SESSION_ID="unknown"
fi
export GR0M_MEM_SESSION_ID="$SESSION_ID"

PYTHON_BIN="${GR0M_MEM_PYTHON:-python3}"

exec "$PYTHON_BIN" -m gr0m_mem hook precompact --session-id "$SESSION_ID" "$@"
