# VERIFY-RESULTS

Full run of `scripts/verify-session-attribution` against the production llmproxy,
macmini, 2026-08-17, deployed `b8d3172` (managed mode).

## Run command

```sh
cd /tmp && python3 /Users/marcio/git/toolbox-verify/scripts/verify-session-attribution
```

## Full output

```
verify-session-attribution
cells: claude-oneshot, claude-interactive, opencode-oneshot, opencode-interactive, codex-oneshot, codex-interactive, opencode-subagent, claude-subagent, codex-subagent
DB: ssh vibes → ssh llmproxy → psql -U llmproxy -d llmproxy
keep=False skip-subagents=False


============================================================
CELL: claude-oneshot  (run=vsa-claude_oneshot-a951cb44)
============================================================
  Launching claude one-shot → run='vsa-claude_oneshot-a951cb44'
  Waiting for vsa-claude_oneshot-a951cb44 to reach done (timeout=120.0s)…
  [link1] PASS status=done
  [link2] PASS session_id='e8340923-6604-42cd-8c9b-432dbefc54f4' confidence=certain acquisition=pushed
  [link3] PASS disk: /Users/marcio/.claude/projects/-private-tmp/e8340923-6604-42cd-8c9b-432dbefc54f4.jsonl
  Polling DB for client_sessions row (timeout=15.0s)…
  [link4] PASS DB: client='claude-code' created_at='2026-08-17 11:13:50.072762+00'

  → PASS: claude-oneshot
     session_id='e8340923-6604-42cd-8c9b-432dbefc54f4'
     link1: done
     link2: confidence=certain acquisition=pushed
     link3: /Users/marcio/.claude/projects/-private-tmp/e8340923-6604-42cd-8c9b-432dbefc54f4.jsonl
     link4: client=claude-code

============================================================
CELL: claude-interactive  (run=vsa-claude_interactive-f839b0c8)
============================================================
  Launching claude interactive → run='vsa-claude_interactive-f839b0c8'
  Waiting for first PONG reply (timeout=90.0s)…
  [link1a] PASS: first PONG reply seen
  Steering second prompt…
  Waiting for second PONG2 reply (timeout=90.0s)…
  [link1b] PASS: second PONG2 reply seen
  [link2] PASS session_id='3b66c662-9b7e-4293-9f9e-28d2fe4df22f' confidence=certain acquisition=pushed
  [link3] PASS disk: /Users/marcio/.claude/projects/-private-tmp/3b66c662-9b7e-4293-9f9e-28d2fe4df22f.jsonl
  Polling DB for client_sessions row…
  [link4] PASS DB: client='claude-code'

  → PASS: claude-interactive
     session_id='3b66c662-9b7e-4293-9f9e-28d2fe4df22f'
     link1: two turns answered
     link2: confidence=certain acquisition=pushed
     link3: /Users/marcio/.claude/projects/-private-tmp/3b66c662-9b7e-4293-9f9e-28d2fe4df22f.jsonl
     link4: client=claude-code

============================================================
CELL: opencode-oneshot  (run=vsa-opencode_oneshot-7dbe3aa2)
============================================================
  Launching opencode one-shot → run='vsa-opencode_oneshot-7dbe3aa2'
  Waiting for vsa-opencode_oneshot-7dbe3aa2 to reach done (timeout=120.0s)…
  [link1] PASS status=done
  [link2] PASS session_id='ses_ff09234dbffeX9MRVbbYEpFNXe' confidence=certain acquisition=minted
  [link3] PASS disk: opencode.db: session has 2 message(s)
  Polling DB for client_sessions row (timeout=15.0s)…
  [link4] PASS DB: client='opencode' created_at='2026-08-17 11:14:11.643831+00'

  → PASS: opencode-oneshot
     session_id='ses_ff09234dbffeX9MRVbbYEpFNXe'
     link1: done
     link2: confidence=certain acquisition=minted
     link3: opencode.db: session has 2 message(s)
     link4: client=opencode

============================================================
CELL: opencode-interactive  (run=vsa-opencode_interactive-af0f9987)
============================================================
  Launching opencode interactive → run='vsa-opencode_interactive-af0f9987'
  Waiting for first PONG reply (timeout=90.0s)…
  [link1a] PASS: first PONG reply seen
  Steering second prompt…
  Waiting for second PONG2 reply (timeout=90.0s)…
  [link1b] PASS: second PONG2 reply seen
  [link2] PASS session_id='ses_ff0921ffcffeQyyjtKf5ZdfoGN' confidence=certain acquisition=minted
  [link3] PASS disk: opencode.db: session has 5 message(s)
  Polling DB for client_sessions row…
  [link4] PASS DB: client='opencode'

  → PASS: opencode-interactive
     session_id='ses_ff0921ffcffeQyyjtKf5ZdfoGN'
     link1: two turns answered
     link2: confidence=certain acquisition=minted
     link3: opencode.db: session has 5 message(s)
     link4: client=opencode

============================================================
CELL: codex-oneshot  (run=vsa-codex_oneshot-84c2a67e)
============================================================
  Launching codex one-shot → run='vsa-codex_oneshot-84c2a67e'
  Waiting for vsa-codex_oneshot-84c2a67e to reach done (timeout=120.0s)…
  [link1] PASS status=done
  [link2] PASS session_id='01a00f6e-0955-77b0-9d35-b87335ffa680' confidence=certain acquisition=minted
  [link3] PASS disk: /Users/marcio/.codex/sessions/2026/08/17/rollout-2026-08-17T12-14-24-01a00f6e-0955-77b0-9d35-b87335ffa680.jsonl
  Polling DB for client_sessions row (timeout=15.0s)…
  [link4] PASS DB: client='codex' created_at='2026-08-17 11:14:27.686612+00'

  → PASS: codex-oneshot
     session_id='01a00f6e-0955-77b0-9d35-b87335ffa680'
     link1: done
     link2: confidence=certain acquisition=minted
     link3: /Users/marcio/.codex/sessions/2026/08/17/rollout-2026-08-17T12-14-24-01a00f6e-0955-77b0-9d35-b87335ffa680.jsonl
     link4: client=codex

============================================================
CELL: codex-interactive  (run=vsa-codex_interactive-77593c08)
============================================================
  Launching codex interactive → run='vsa-codex_interactive-77593c08'
  Waiting for first PONG reply (timeout=90.0s)…
  [link1a] PASS: first PONG reply seen
  Steering second prompt…
  Waiting for second PONG2 reply (timeout=90.0s)…
  [link1b] PASS: second PONG2 reply seen
  [link2] PASS session_id='01a00f6e-194b-7512-8e33-f7637a3bebe5' confidence=certain acquisition=minted
  [link3] PASS disk: /Users/marcio/.codex/sessions/2026/08/17/rollout-2026-08-17T12-14-28-01a00f6e-194b-7512-8e33-f7637a3bebe5.jsonl
  Polling DB for client_sessions row…
  [link4] PASS DB: client='codex'

  → PASS: codex-interactive
     session_id='01a00f6e-194b-7512-8e33-f7637a3bebe5'
     link1: two turns answered
     link2: confidence=certain acquisition=minted
     link3: /Users/marcio/.codex/sessions/2026/08/17/rollout-2026-08-17T12-14-28-01a00f6e-194b-7512-8e33-f7637a3bebe5.jsonl
     link4: client=codex

============================================================
CELL: opencode-subagent  (run=vsa-opencode_subagent-94ada268)
============================================================
  Launching opencode sub-agent → run='vsa-opencode_subagent-94ada268'
  [link1] PASS status=done
  [link2] PASS root_id='ses_ff091c027ffe4F8RGO23JoqSwC'
  [link3] PASS disk: opencode.db: session has 3 message(s)
  [link4] PASS root row: client='opencode'
  Polling DB for child row (parent='ses_ff091c027ffe4F8RGO23JoqSwC')…
  [link5] PASS child='ses_ff091b1e7ffeGRzbFDM5fdShL2' parent='ses_ff091c027ffe4F8RGO23JoqSwC' → resolves to root

  → PASS: opencode-subagent
     session_id='ses_ff091c027ffe4F8RGO23JoqSwC'
     link1: done
     link2: confidence=certain
     link3: opencode.db: session has 3 message(s)
     link4: root row present
     link5: child='ses_ff091b1e7ffeGRzbFDM5fdShL2' parent='ses_ff091c027ffe4F8RGO23JoqSwC'

============================================================
CELL: claude-subagent  (run=vsa-claude_subagent-aafece56)
============================================================
  Launching claude sub-agent → run='vsa-claude_subagent-aafece56'
  [link1] PASS status=done
  [link2] PASS root_id='6e15ca3c-6366-4eb4-880d-989acc8a5a89'
  [link3] PASS disk: /Users/marcio/.claude/projects/-private-tmp/6e15ca3c-6366-4eb4-880d-989acc8a5a89.jsonl
  [link4] PASS root row: client='claude-code'
  Polling DB for child row (parent='6e15ca3c-6366-4eb4-880d-989acc8a5a89', timeout=15.0s)…
  [link5] PASS child='a69f6d0fe57861ca4' parent='6e15ca3c-6366-4eb4-880d-989acc8a5a89' (via ??-fallback)

  → PASS: claude-subagent
     session_id='6e15ca3c-6366-4eb4-880d-989acc8a5a89'
     link1: done
     link2: confidence=certain
     link3: /Users/marcio/.claude/projects/-private-tmp/6e15ca3c-6366-4eb4-880d-989acc8a5a89.jsonl
     link4: root row present
     link5: child='a69f6d0fe57861ca4' parent='6e15ca3c-6366-4eb4-880d-989acc8a5a89' (depth-1 ??-fallback)

============================================================
CELL: codex-subagent  (run=vsa-codex_subagent-2967ca9a)
============================================================

  → SKIP: codex-subagent
     SKIP: codex sub-agent SKIP: no reliably triggerable sub-agent mechanism.
     The app-server schema has Thread.parentThreadId and MultiAgentMode but no
     ThreadStartParams.parentThreadId input and no TurnStartParams.multiAgentMode —
     internal AgentControl orchestration only.

============================================================
SUMMARY
============================================================
  ✓ PASS  claude-oneshot  sid='e8340923-6604-42cd-8c9b-432dbefc54f4'
  ✓ PASS  claude-interactive  sid='3b66c662-9b7e-4293-9f9e-28d2fe4df22f'
  ✓ PASS  opencode-oneshot  sid='ses_ff09234dbffeX9MRVbbYEpFNXe'
  ✓ PASS  opencode-interactive  sid='ses_ff0921ffcffeQyyjtKf5ZdfoGN'
  ✓ PASS  codex-oneshot  sid='01a00f6e-0955-77b0-9d35-b87335ffa680'
  ✓ PASS  codex-interactive  sid='01a00f6e-194b-7512-8e33-f7637a3bebe5'
  ✓ PASS  opencode-subagent  sid='ses_ff091c027ffe4F8RGO23JoqSwC'
  ✓ PASS  claude-subagent  sid='6e15ca3c-6366-4eb4-880d-989acc8a5a89'
  ~ SKIP  codex-subagent

  PASS=8  FAIL=0  SKIP=1

EXIT: 0
```

## Session IDs observed

| Cell | session_id | DB client | acquisition |
|------|-----------|-----------|-------------|
| claude-oneshot | `e8340923-6604-42cd-8c9b-432dbefc54f4` | `claude-code` | pushed |
| claude-interactive | `3b66c662-9b7e-4293-9f9e-28d2fe4df22f` | `claude-code` | pushed |
| opencode-oneshot | `ses_ff09234dbffeX9MRVbbYEpFNXe` | `opencode` | minted |
| opencode-interactive | `ses_ff0921ffcffeQyyjtKf5ZdfoGN` | `opencode` | minted |
| codex-oneshot | `01a00f6e-0955-77b0-9d35-b87335ffa680` | `codex` | minted |
| codex-interactive | `01a00f6e-194b-7512-8e33-f7637a3bebe5` | `codex` | minted |
| opencode-subagent (root) | `ses_ff091c027ffe4F8RGO23JoqSwC` | `opencode` | minted |
| opencode-subagent (child) | `ses_ff091b1e7ffeGRzbFDM5fdShL2` | `opencode` | — |
| claude-subagent (root) | `6e15ca3c-6366-4eb4-880d-989acc8a5a89` | `claude-code` | pushed |
| claude-subagent (child) | `a69f6d0fe57861ca4` | `claude-code` | — |

All `confidence: certain`, all found in `client_sessions` within seconds of the model
call. claude sub-agent child row verified via the `??`-fallback
(`x-claude-code-parent-agent-id` absent at depth 1; proxy resolves parent via
`x-claude-code-session-id`).

## Break-test evidence

To prove each assertion is independently detectable as a failure, we verified that
bad inputs at each link produce the expected outcome:

| Break condition | Expected outcome | Confirmed |
|----------------|-----------------|-----------|
| link2: `session.json` has `confidence='guessed'` | link2 fails: `confidence='guessed' != 'certain'` | ✓ |
| link2: `session.json` absent | link2 fails: `session.json missing or malformed` | ✓ |
| link3 claude: fake session_id → no `.jsonl` transcript | link3 fails: no transcript file | ✓ |
| link3 opencode: fake session_id → no session row in opencode.db | link3 fails: no session row | ✓ |
| link3 codex: fake session_id → no rollout file | link3 fails: no rollout file | ✓ |
| link4: bogus session id `00000000-…` → no DB row | link4 fails: no row after timeout | ✓ |
| link4: real session id but wrong expected client (`opencode` vs `claude-code`) | link4 fails: client mismatch | ✓ |
| link5: fake root `ses_fffakeroot…` → no child row | link5 fails: no child row | ✓ |

Each test was verified live against the production DB. Corrupt session.json was also
demonstrated on a real run by overwriting `session_id` to `00000000-…` and confirming
the DB returned no row for the bogus value — which the script would report as:

```
FAIL [link4-db-row]: no client_sessions row for '00000000-...' after 15.0s
```

Exit code from a FAIL cell is `1`. Exit code from all-SKIP is `0`. Exit from
all-PASS is `0`.

## Issues found during build

1. **opencode model string** — `agent-run --model claude-haiku-4-5` is not enough; opencode
   requires the full provider-qualified name `llmproxy-anthropic/claude-haiku-4.5`.
   The script defaults to the correct full form (`OPENCODE_MODEL` env var).

2. **opencode.db column name** — the `message` table uses `session_id` (snake_case),
   not `sessionID` (camelCase). Fixed in script.

3. **codex gpt-4o-mini unavailable** — the llmproxy does not route `gpt-4o-mini` for
   codex. The script uses `gpt-5.4` (from `~/.codex/config.toml`) by default.

4. **codex sub-agent** — no reliably triggerable sub-agent surface found (see cell
   description and `ACCEPTANCE-EVIDENCE.md §4`). Cell is SKIP with full rationale.
   The schema plumbing (`Thread.parentThreadId`, `MultiAgentMode`) exists but
   `ThreadStartParams` has no `parentThreadId` input and `TurnStartParams` has no
   `multiAgentMode` field — the AgentControl mechanism is internal only.

## Existing tests

```
uv run pytest -q   # 894 passed, 1 skipped — unchanged
```
