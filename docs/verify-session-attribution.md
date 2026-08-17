# verify-session-attribution

End-to-end test for the four-link session attribution chain.

## What it proves

Every `agent-run --harness` run records a session id and asserts that the same id
travels all the way from the harness to the llmproxy billing database. The chain has
four links, and a silent break in any one means cost is silently unattributed:

```
session.json  →  harness disk record  →  wire header  →  llmproxy client_sessions row
   [link2]            [link3]               (proxy)           [link4]
```

The script also exercises link 1 (run actually completed) and, for sub-agent cells,
link 5 (child row's `parent_client_session_id` resolves to the root session):

```
root client_sessions row  ←  parent_client_session_id  ←  child client_sessions row
         [link4]                                                   [link5]
```

It covers all three harnesses (claude, opencode, codex) in both one-shot and
interactive modes, plus sub-agent cells for opencode and claude.

## How to run

```sh
# Full suite (~5 min, ~$0.10 at current haiku/gpt-5.4 rates)
python3 scripts/verify-session-attribution

# Single harness
python3 scripts/verify-session-attribution --harness claude

# Single cell (cheapest: ~$0.002)
python3 scripts/verify-session-attribution --only claude-oneshot

# Skip sub-agent cells (faster, no sub-agent cost)
python3 scripts/verify-session-attribution --skip-subagents

# Preserve run dirs for debugging
python3 scripts/verify-session-attribution --only claude-oneshot --keep

# Machine-readable JSON output (stdout is pure JSON; human log goes to stderr)
python3 scripts/verify-session-attribution --json
```

Exit code is **0** when all cells are PASS or SKIP, **1** on any FAIL.

## Prerequisites

### SSH access (two hops)

DB access requires: `ssh vibes → ssh llmproxy → sudo docker exec postgres psql`

```sh
# Verify the path works:
ssh vibes 'ssh llmproxy "sudo docker exec $(sudo docker ps -q -f name=llmproxy_postgres) psql -U llmproxy -d llmproxy -c \"SELECT 1\""'
```

Override hosts with environment variables:
```sh
VIBES_HOST=myhost LLMPROXY_HOST=proxy python3 scripts/verify-session-attribution
```

### Harness credentials

- **claude**: `~/.claude/settings.json` with `ANTHROPIC_BASE_URL` pointing to llmproxy
- **opencode**: `~/.config/opencode/opencode.json` with `llmproxy-anthropic` provider defined
- **codex**: `~/.codex/auth.json` with `OPENAI_API_KEY`; `~/.codex/config.toml` with
  `model_provider = "llmproxy"` and `[model_providers.llmproxy]`. The script reads
  the key from auth.json automatically if `OPENAI_API_KEY` is not in the environment.

### opencode.db access

The script opens `~/.local/share/opencode/opencode.db` read-only via
`file:...?mode=ro` (WAL mode, live owner safe). Never writes to it.

## Cells

| Cell | Harness | Mode | Links asserted |
|------|---------|------|----------------|
| `claude-oneshot` | claude | `--print` one-shot | 1, 2, 3, 4 |
| `claude-interactive` | claude | TUI + two steers | 1, 2, 3, 4 |
| `opencode-oneshot` | opencode | `run` one-shot | 1, 2, 3, 4 |
| `opencode-interactive` | opencode | TUI + two steers | 1, 2, 3, 4 |
| `codex-oneshot` | codex | app-server one-shot | 1, 2, 3, 4 |
| `codex-interactive` | codex | app-server interactive | 1, 2, 3, 4 |
| `opencode-subagent` | opencode | build agent with explore sub-agent | 1–5 |
| `claude-subagent` | claude | Task sub-agent | 1–5 |
| `codex-subagent` | codex | — | **SKIP** (see below) |

## What each failure means

| Link | Failure message | What broke |
|------|----------------|------------|
| `link1-launch` | agent-run exited N / timed out | harness binary not found, or bad arg |
| `link1-run` | status='launch_failed'/'failed' | harness crashed, auth/model error |
| `link1-interactive-first-reply` | '42' not in assistant region | TUI not starting, model not responding |
| `link1-steer` | agent-run steer failed | interactive FIFO not set up |
| `link1-interactive-second-reply` | '27' not in assistant region | steer not reaching agent |
| `link2-session-json` | session.json missing/malformed | agent-run acquisition failed |
| `link2-session-json` | confidence != 'certain' | mint/push was uncertain or absent |
| `link3-disk-record` | no transcript / no session row / no rollout | session not written to harness state |
| `link4-db-row` | no client_sessions row after Ns | proxy never received the session id |
| `link4-db-row` | client mismatch | wrong harness identified by proxy |
| `link5-parent-linkage` | no child row with parent=root | sub-agent requests not carrying parent header |

### Harness-specific link-3 checks

- **claude**: `~/.claude/projects/<mangled-cwd>/<session-id>.jsonl` must exist
- **opencode**: `~/.local/share/opencode/opencode.db` must have a `session` row for
  the id **and at least one `message` row** (an empty decoy session is a FAIL)
- **codex**: `~/.codex/sessions/YYYY/MM/DD/rollout-*-<session-id>.jsonl` must exist

## codex sub-agent: SKIP rationale

codex sends one id under three header names (`x-client-request-id`, `session-id`,
`thread-id`) and no parent header. The app-server JSON-RPC schema has
`Thread.parentThreadId` and `MultiAgentMode`, but:

- `ThreadStartParams` has no `parentThreadId` input field
- `TurnStartParams` has no `multiAgentMode` field
- The `AgentControl` orchestration that sets `agentNickname`/`agentRole`/`parentThreadId`
  is internal to the codex runtime, not triggerable via a user prompt

Prompting codex to "spawn a sub-agent" produces a text response but never creates a
second thread with a `parentThreadId`. The schema confirms the plumbing exists; the
triggering surface does not. Cell is SKIP with a stated reason, not deleted.

Evidence: `ACCEPTANCE-EVIDENCE.md §4`; schema inspection of `ThreadStartParams.json`
and `TurnStartParams.json` in `codex app-server generate-json-schema --out DIR`.

## DB schema reference

```sql
-- client_sessions columns
api_key_id               -- FK to api_keys
client_session_id        -- the csid the harness sent
session_id               -- internal proxy session
org_id                   -- organization
client                   -- 'claude-code' | 'opencode' | 'codex'
parent_client_session_id -- set for sub-agents; null for roots
created_at               -- timestamp of first request for this csid
```

## Environment variables

| Variable | Default | Meaning |
|----------|---------|---------|
| `VIBES_HOST` | `vibes` | First SSH hop hostname |
| `LLMPROXY_HOST` | `llmproxy` | Second SSH hop hostname |
| `POSTGRES_USER` | `llmproxy` | psql username |
| `POSTGRES_DB` | `llmproxy` | psql database name |
| `DB_POLL_TIMEOUT` | `15` | Seconds to poll DB for a new row |
| `RUN_TIMEOUT` | `120` | Seconds to wait for a run to reach done |
| `INTERACTIVE_FIRST_REPLY_TIMEOUT` | `90` | Seconds to wait for first interactive reply |
| `INTERACTIVE_SECOND_REPLY_TIMEOUT` | `90` | Seconds to wait for second interactive reply |
| `CLAUDE_MODEL` | `claude-haiku-4-5` | Model for claude cells |
| `OPENCODE_MODEL` | `llmproxy-anthropic/claude-haiku-4.5` | Model for opencode cells |
| `CODEX_MODEL` | `gpt-5.4` | Model for codex cells |
| `CODEX_PROVIDER` | `llmproxy` | Provider for codex cells |
| `OPENAI_API_KEY` | (from `~/.codex/auth.json`) | Key for codex |
