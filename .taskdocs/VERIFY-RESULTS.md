# VERIFY-RESULTS

Full run of `scripts/verify-session-attribution` against the production llmproxy,
macmini, 2026-08-17, deployed `477b993` (managed mode, post-review fixes).

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

Checking DB path (preflight SELECT 1)…
DB preflight OK


============================================================
CELL: claude-oneshot  (run=vsa-claude_oneshot-14bed537)
============================================================
  Launching claude one-shot → run='vsa-claude_oneshot-14bed537'
  Waiting for vsa-claude_oneshot-14bed537 to reach done (timeout=120.0s)…
  [link1] PASS status=done
  [link2] PASS session_id='a5f71608-3556-489d-a4f5-bb651174ad28' confidence=certain acquisition=pushed
  [link3] PASS disk: /Users/marcio/.claude/projects/-private-tmp/a5f71608-3556-489d-a4f5-bb651174ad28.jsonl
  Polling DB for client_sessions row (timeout=15.0s)…
  [link4] PASS DB: client='claude-code' created_at='2026-08-17 13:02:27.881152+00'

  → PASS: claude-oneshot
     session_id='a5f71608-3556-489d-a4f5-bb651174ad28'
     link1: done
     link2: confidence=certain acquisition=pushed
     link3: /Users/marcio/.claude/projects/-private-tmp/a5f71608-3556-489d-a4f5-bb651174ad28.jsonl
     link4: client=claude-code

============================================================
CELL: claude-interactive  (run=vsa-claude_interactive-6ccac09f)
============================================================
  Launching claude interactive → run='vsa-claude_interactive-6ccac09f'
  Waiting for turn-1 reply '42' in assistant region (timeout=90.0s)…
  [link1a] PASS: turn-1 reply '42' seen in assistant region (log count=4)
  Steering second prompt…
  Waiting for turn-2 reply '27' in assistant region (timeout=90.0s)…
  [link1b] PASS: turn-2 reply '27' seen in assistant region
  [link2] PASS session_id='fca925cb-865c-4e41-b0c4-dfd852b4a50d' confidence=certain acquisition=pushed
  [link3] PASS disk: /Users/marcio/.claude/projects/-private-tmp/fca925cb-865c-4e41-b0c4-dfd852b4a50d.jsonl
  Polling DB for client_sessions row…
  [link4] PASS DB: client='claude-code'

  → PASS: claude-interactive
     session_id='fca925cb-865c-4e41-b0c4-dfd852b4a50d'
     link1: two turns answered
     link2: confidence=certain acquisition=pushed
     link3: /Users/marcio/.claude/projects/-private-tmp/fca925cb-865c-4e41-b0c4-dfd852b4a50d.jsonl
     link4: client=claude-code

============================================================
CELL: opencode-oneshot  (run=vsa-opencode_oneshot-0390e5a9)
============================================================
  Launching opencode one-shot → run='vsa-opencode_oneshot-0390e5a9'
  Waiting for vsa-opencode_oneshot-0390e5a9 to reach done (timeout=120.0s)…
  [link1] PASS status=done
  [link2] PASS session_id='ses_ff02e8f5affeMjqpuOO625Oo3W' confidence=certain acquisition=minted
  [link3] PASS disk: opencode.db: session has 2 message(s)
  Polling DB for client_sessions row (timeout=15.0s)…
  [link4] PASS DB: client='opencode' created_at='2026-08-17 13:03:07.627344+00'

  → PASS: opencode-oneshot
     session_id='ses_ff02e8f5affeMjqpuOO625Oo3W'
     link1: done
     link2: confidence=certain acquisition=minted
     link3: opencode.db: session has 2 message(s)
     link4: client=opencode

============================================================
CELL: opencode-interactive  (run=vsa-opencode_interactive-94f7f7d9)
============================================================
  Launching opencode interactive → run='vsa-opencode_interactive-94f7f7d9'
  Waiting for turn-1 reply '42' in assistant region (timeout=90.0s)…
  [link1a] PASS: turn-1 reply '42' seen in assistant region (log count=4)
  Steering second prompt…
  Waiting for turn-2 reply '27' in assistant region (timeout=90.0s)…
  [link1b] PASS: turn-2 reply '27' seen in assistant region
  [link2] PASS session_id='ses_ff02e62b8ffe15tTbTGBqDvAhJ' confidence=certain acquisition=minted
  [link3] PASS disk: opencode.db: session has 6 message(s)
  Polling DB for client_sessions row…
  [link4] PASS DB: client='opencode'

  → PASS: opencode-interactive
     session_id='ses_ff02e62b8ffe15tTbTGBqDvAhJ'
     link1: two turns answered
     link2: confidence=certain acquisition=minted
     link3: opencode.db: session has 6 message(s)
     link4: client=opencode

============================================================
CELL: codex-oneshot  (run=vsa-codex_oneshot-3b06b795)
============================================================
  Launching codex one-shot → run='vsa-codex_oneshot-3b06b795'
  Waiting for vsa-codex_oneshot-3b06b795 to reach done (timeout=120.0s)…
  [link1] PASS status=done
  [link2] PASS session_id='01a00fd1-f79e-7210-97a1-2c7e7d6e1a73' confidence=certain acquisition=minted
  [link3] PASS disk: /Users/marcio/.codex/sessions/2026/08/17/rollout-2026-08-17T14-03-33-01a00fd1-f79e-7210-97a1-2c7e7d6e1a73.jsonl
  Polling DB for client_sessions row (timeout=15.0s)…
  [link4] PASS DB: client='codex' created_at='2026-08-17 13:03:35.983444+00'

  → PASS: codex-oneshot
     session_id='01a00fd1-f79e-7210-97a1-2c7e7d6e1a73'
     link1: done
     link2: confidence=certain acquisition=minted
     link3: /Users/marcio/.codex/sessions/2026/08/17/rollout-2026-08-17T14-03-33-01a00fd1-f79e-7210-97a1-2c7e7d6e1a73.jsonl
     link4: client=codex

============================================================
CELL: codex-interactive  (run=vsa-codex_interactive-af218cb2)
============================================================
  Launching codex interactive → run='vsa-codex_interactive-af218cb2'
  Waiting for turn-1 reply '42' in assistant region (timeout=90.0s)…
  [link1a] PASS: turn-1 reply '42' seen in assistant region (log count=1)
  Steering second prompt…
  Waiting for turn-2 reply '27' in assistant region (timeout=90.0s)…
  [link1b] PASS: turn-2 reply '27' seen in assistant region
  [link2] PASS session_id='01a00fd2-03d0-78c2-8293-511afb0eaf61' confidence=certain acquisition=minted
  [link3] PASS disk: /Users/marcio/.codex/sessions/2026/08/17/rollout-2026-08-17T14-03-36-01a00fd2-03d0-78c2-8293-511afb0eaf61.jsonl
  Polling DB for client_sessions row…
  [link4] PASS DB: client='codex'

  → PASS: codex-interactive
     session_id='01a00fd2-03d0-78c2-8293-511afb0eaf61'
     link1: two turns answered
     link2: confidence=certain acquisition=minted
     link3: /Users/marcio/.codex/sessions/2026/08/17/rollout-2026-08-17T14-03-36-01a00fd2-03d0-78c2-8293-511afb0eaf61.jsonl
     link4: client=codex

============================================================
CELL: opencode-subagent  (run=vsa-opencode_subagent-08fc4ca5)
============================================================
  Launching opencode sub-agent → run='vsa-opencode_subagent-08fc4ca5'
  [link1] PASS status=done
  [link2] PASS root_id='ses_ff02dd211ffeJml5713O3mT9GV'
  [link3] PASS disk: opencode.db: session has 3 message(s)
  [link4] PASS root row: client='opencode'
  Polling DB for child row (parent='ses_ff02dd211ffeJml5713O3mT9GV')…
  [link5] PASS child='ses_ff02d9bb5ffeRyV0DsjetbbeAh' parent='ses_ff02dd211ffeJml5713O3mT9GV' → resolves to root
  [link5+] PASS sub-agent reply 'SUBPONG' confirmed in log

  → PASS: opencode-subagent
     session_id='ses_ff02dd211ffeJml5713O3mT9GV'
     link1: done
     link2: confidence=certain
     link3: opencode.db: session has 3 message(s)
     link4: client=opencode
     link5: child='ses_ff02d9bb5ffeRyV0DsjetbbeAh' parent='ses_ff02dd211ffeJml5713O3mT9GV'

============================================================
CELL: claude-subagent  (run=vsa-claude_subagent-ac1160eb)
============================================================
  Launching claude sub-agent → run='vsa-claude_subagent-ac1160eb'
  [link1] PASS status=done
  [link2] PASS root_id='58d9c8d4-ab94-4071-96f1-0e1cda6d7e86'
  [link3] PASS disk: /Users/marcio/.claude/projects/-private-tmp/58d9c8d4-ab94-4071-96f1-0e1cda6d7e86.jsonl
  [link4] PASS root row: client='claude-code'
  Polling DB for child row (parent='58d9c8d4-ab94-4071-96f1-0e1cda6d7e86', timeout=15.0s)…
  [link5] PASS child='abf7e8ab902b397e9' parent='58d9c8d4-ab94-4071-96f1-0e1cda6d7e86' (via ??-fallback)
  [link5+] PASS sub-agent reply 'SUBPONG' confirmed in log

  → PASS: claude-subagent
     session_id='58d9c8d4-ab94-4071-96f1-0e1cda6d7e86'
     link1: done
     link2: confidence=certain
     link3: /Users/marcio/.claude/projects/-private-tmp/58d9c8d4-ab94-4071-96f1-0e1cda6d7e86.jsonl
     link4: client=claude-code
     link5: child='abf7e8ab902b397e9' parent='58d9c8d4-ab94-4071-96f1-0e1cda6d7e86' (depth-1 ??-fallback)

============================================================
CELL: codex-subagent  (run=vsa-codex_subagent-2459d33e)
============================================================

  → SKIP: codex-subagent
     SKIP: codex sub-agent SKIP: no reliably triggerable sub-agent mechanism.
     The app-server schema has Thread.parentThreadId and MultiAgentMode but no
     ThreadStartParams.parentThreadId input and no TurnStartParams.multiAgentMode —
     internal AgentControl orchestration only.

============================================================
SUMMARY
============================================================
  ✓ PASS  claude-oneshot  sid='a5f71608-3556-489d-a4f5-bb651174ad28'
  ✓ PASS  claude-interactive  sid='fca925cb-865c-4e41-b0c4-dfd852b4a50d'
  ✓ PASS  opencode-oneshot  sid='ses_ff02e8f5affeMjqpuOO625Oo3W'
  ✓ PASS  opencode-interactive  sid='ses_ff02e62b8ffe15tTbTGBqDvAhJ'
  ✓ PASS  codex-oneshot  sid='01a00fd1-f79e-7210-97a1-2c7e7d6e1a73'
  ✓ PASS  codex-interactive  sid='01a00fd2-03d0-78c2-8293-511afb0eaf61'
  ✓ PASS  opencode-subagent  sid='ses_ff02dd211ffeJml5713O3mT9GV'
  ✓ PASS  claude-subagent  sid='58d9c8d4-ab94-4071-96f1-0e1cda6d7e86'
  ~ SKIP  codex-subagent

  PASS=8  FAIL=0  SKIP=1

EXIT: 0
```

## Session IDs observed

| Cell | session_id | DB client | acquisition |
|------|-----------|-----------|-------------|
| claude-oneshot | `a5f71608-3556-489d-a4f5-bb651174ad28` | `claude-code` | pushed |
| claude-interactive | `fca925cb-865c-4e41-b0c4-dfd852b4a50d` | `claude-code` | pushed |
| opencode-oneshot | `ses_ff02e8f5affeMjqpuOO625Oo3W` | `opencode` | minted |
| opencode-interactive | `ses_ff02e62b8ffe15tTbTGBqDvAhJ` | `opencode` | minted |
| codex-oneshot | `01a00fd1-f79e-7210-97a1-2c7e7d6e1a73` | `codex` | minted |
| codex-interactive | `01a00fd2-03d0-78c2-8293-511afb0eaf61` | `codex` | minted |
| opencode-subagent (root) | `ses_ff02dd211ffeJml5713O3mT9GV` | `opencode` | minted |
| opencode-subagent (child) | `ses_ff02d9bb5ffeRyV0DsjetbbeAh` | `opencode` | — |
| claude-subagent (root) | `58d9c8d4-ab94-4071-96f1-0e1cda6d7e86` | `claude-code` | pushed |
| claude-subagent (child) | `abf7e8ab902b397e9` | `claude-code` | — |

All `confidence: certain`, all found in `client_sessions` within seconds of the model
call. claude sub-agent child row verified via the `??`-fallback
(`x-claude-code-parent-agent-id` absent at depth 1; proxy resolves parent via
`x-claude-code-session-id`).

## C1 fix proof (vacuous-pass prevention)

Launched `claude-interactive` with the model instructed NOT to produce `42`
(prompt: `"Do NOT say the number 42 under any circumstances. Reply with the word: BANANA"`).
The new `_token_in_assistant_region` check correctly returned False throughout:

```
t+4.1s  status=running  in_assistant=False  log_count=7
...
t+50.0s status=running  in_assistant=False  log_count=7
result: FAIL [link1-interactive-first-reply]

clean transcript showed:
  ❯ Do NOT say the number 42 under any circumstances. Reply with the word: BANANA
  ⏺ BANANA
```

`log_count=7` because `42` appears 7 times in the raw PTY echo of the prompt — the old
code would have reported PASS after 4s. The new code correctly FAILs because `42` never
appears in a non-`❯` line of the clean transcript.

## H2/H3 fix proof (cleanup on interrupt)

Launched `--only claude-oneshot`, sent SIGINT 3s in (mid-cell, after run directory
was created):

```
At t+3s: new runs={'vsa-claude_oneshot-dc7fef9e'}
Sending SIGINT to pid=78424…
Process exited with code 1
After interrupt+cleanup: remaining new runs=set()
After interrupt+cleanup: remaining new dirs=set()
>>> CORRECT: H2/H3 fix works — SIGINT cleanup left no vsa-* remnants
```

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

## Issues fixed since b8d3172 (review findings)

- **C1**: Interactive `link1` asserted the raw PTY echo instead of the model's reply.
  Fixed by requiring the token to appear in a non-`❯`/`┃` line of `agent-run clean`
  output (`_token_in_assistant_region`). Turn tokens are now synthesised values absent
  from prompts (`42` = 7×6, `27` = 3×9), disjoint from each other, and drawn from
  short prompts (≤19 chars) to stay within the claude TUI steer buffer limit.
- **H1**: Teardown now escalates to SIGKILL if SIGTERM doesn't land within budget
  (`_kill_and_reap`), and uses `agent-run reap` instead of bare `shutil.rmtree`.
- **H2**: `run_cell` `finally` now calls `_reap_run` for any run still in `_active_runs`
  (covers launch-failure and unexpected-exception paths).
- **H3**: SIGINT/SIGTERM handlers and `atexit` installed at startup; all launched run
  names tracked in `_active_runs`.
- **M1**: `_psql_query_csv` returns `_DbError` on transport failure (not `None`);
  callers report `link4-db-unreachable` separately from `link4-db-row`. DB preflight
  (`SELECT 1`) runs before any cell.
- **M2**: `_log` sends output to stderr when `--json` is set; stdout is pure JSON.
- **M3**: Sub-agent cells now compare `db_client != expected_client` at link4.
- **M4**: Sub-agent cells assert `SUBPONG` appears in the run log after link5.
- **L1**: Dead `_psql_query` function removed; `_parse_psql_csv` docstring rewritten.
- **L2**: Unused `re` and `tempfile` imports removed; `re` re-added for `_SESSION_ID_RE`.
- **L3**: `session_id` validated against `^[A-Za-z0-9_-]+$` before SQL interpolation.
- **L4**: `_read_log` reads the full log; `_read_log_tail` kept for diagnostic snippets only.
- **L6**: `--keep` help text now says "leaves the TUI process running".
- **L7**: Cost figure unified at `~$0.02–0.05` in both `--help` and docs.
- **ACCEPTANCE-EVIDENCE.md §4**: Updated "codex — no parent header exists" to accurately
  describe the schema plumbing that exists vs the triggering surface that does not.

## Existing tests

```
uv run pytest -q   # 894 passed, 1 skipped — unchanged
```
