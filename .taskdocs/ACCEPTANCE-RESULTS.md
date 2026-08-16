# Acceptance Results — agent-run session id ↔ proxy csid

Captured 2026-08-16 on macmini. All 6 cells run through `agent-run --harness` managed mode.
Forward-capture proxy (`fwdcap.py`) forwarded to `https://llmproxy.absmartly-dev.com` and logged
all request headers. Tests: `uv run pytest -q` → **841 passed, 1 skipped** (the 1 skipped is
pre-existing and unrelated to this pass).

---

## Part A — codex `--json` replaced with `app-server` mint-then-run

**Finding F1 (reported):** `codex exec resume <thread_id>` fails with
`thread/resume: no rollout found for thread id <id>` because `codex app-server thread/start`
allocates the id but does NOT create the rollout `.jsonl` file until `turn/start` is sent on the
same process. The mint-then-separate-exec pattern (mint via app-server, exec via `codex exec
resume`) is broken by design — the rollout only exists after the first turn.

**Resolution:** `_run_managed_oneshot_codex_appserver` keeps the app-server process alive across
initialize → thread/start → turn/start → turn events → turn/completed. The thread id is minted
and the turn runs on the same app-server process. Human-readable text is extracted from
`item/agentMessage/delta` events (`params.delta` string chunks) and written incrementally to the
PTY log, satisfying both "readable log" and "live tail/idle-timeout" requirements.

**`--json` removed:** `_build_managed_argv` for codex no longer appends `--json`. The
`_parse_codex_session_id_from_jsonl` function and the 141-line `_run_managed_oneshot_codex`
function are both deleted.

**Interactive codex (finding F2, resolved in this pass):** `_run_managed_interactive_codex_appserver`
keeps a single app-server process alive for the session lifetime. `thread/start` mints the thread
id (written to `session.json` as `minted/certain`) before the first prompt is sent. The initial
prompt is delivered via `turn/start`. Subsequent `agent-run steer` calls write to the FIFO, which
the runner reads and forwards as `turn/start` (if idle between turns) or `turn/steer` with
`expectedTurnId` (if a turn is active). Agent text deltas (`item/agentMessage/delta`) stream to
the run log as readable prose. The `originator` and User-Agent are `codex_exec` (set by
`clientInfo.name` in the initialize handshake). Interactive codex is now `minted/certain`.

**OPENAI_API_KEY injection:** `_codex_subprocess_env()` reads `~/.codex/auth.json` and injects
`OPENAI_API_KEY` into the subprocess environment when the variable is not already set. Without
this, `codex app-server` fails immediately with "Missing environment variable: OPENAI_API_KEY"
when agent-run is launched from a shell that does not export the key.

---

## Part B — 6-cell acceptance matrix

Proxy ports: claude=47510, codex=47511, opencode=47512 (all forwarding to llmproxy).
Run name convention: `v2-<harness>-<mode>` (re-verified 2026-08-16, second pass).

---

### Cell 1 — claude / one-shot — PASS

**Run:** `v2-claude-oneshot`  
**Command:** `agent-run --harness claude --prompt "Reply with exactly: PONG" --harness-arg --settings --harness-arg /path/to/claude-cap-settings.json v2-claude-oneshot`

| Source | Value |
|--------|-------|
| `session.json` | `b8b11466-2264-423e-8834-29aadc5ba818` |
| Harness own record | `~/.claude/projects/-Users-marcio-git-toolbox-harness/b8b11466-2264-423e-8834-29aadc5ba818.jsonl` ✓ |
| Wire (`x-claude-code-session-id`) | `b8b11466-2264-423e-8834-29aadc5ba818` ✓ |

**Acquisition:** `pushed` / `certain` (agent-run generated the UUID4, passed `--session-id`).  
**Log:** contains `PONG` (human-readable).  
**C1 subcommands:** `logs` ✓, `clean` ✓ (shows "PONG"), `watch --json` ✓ (session id present),
`list --all` ✓, `steer` on one-shot exits non-zero with clear message ✓.

---

### Cell 2 — opencode / one-shot — PASS

**Run:** `v2-opencode-oneshot`  
**Command:** `agent-run --harness opencode --model llmproxy-anthropic/claude-sonnet-4.6 --prompt "Reply with exactly: PONG" v2-opencode-oneshot`

| Source | Value |
|--------|-------|
| `session.json` | `ses_ff39a3c84ffeYQkiELzdK31Fb4` |
| DB (`opencode.db`, read-only) | id=`ses_ff39a3c84ffeYQkiELzdK31Fb4`, parent_id=None, agent=build, title=v2-opencode-oneshot ✓ |
| Wire (`x-session-id`) | `ses_ff39a3c84ffeYQkiELzdK31Fb4` ✓ |

**Acquisition:** `minted` / `certain`.  
**Log:** contains `PONG`.  
**C1 subcommands:** `logs` ✓, `clean` ✓, `watch --json` ✓, `steer` on one-shot exits non-zero ✓.

---

### Cell 3 — codex / one-shot — PASS

**Run:** `v2-codex-oneshot`  
**Command:** `agent-run --harness codex --prompt "Reply with exactly: PONG" --harness-arg -c --harness-arg "model_providers.llmproxy.base_url=..." v2-codex-oneshot`

**Mechanism:** `_run_managed_oneshot_codex_appserver` — app-server stays alive, mints thread,
sends `turn/start`, streams `item/agentMessage/delta` text chunks to log, waits for
`turn/completed`. No `--json` flag; stdout is the log file. App-server chatter goes only to
`session-acquire.log`.

| Source | Value |
|--------|-------|
| `session.json` | `01a00c66-40bc-7340-9669-b4958d5aa6ce` |
| Harness own record | `~/.codex/sessions/2026/08/16/rollout-2026-08-16T22-07-03-01a00c66-40bc-7340-9669-b4958d5aa6ce.jsonl` ✓ |
| Wire (`x-client-request-id`) | `01a00c66-40bc-7340-9669-b4958d5aa6ce` ✓ |
| Wire (`originator`) | `codex_exec` ✓ (clientInfo.name="codex_exec" — required for proxy identify_client) |

**Acquisition:** `minted` / `certain`.  
**Log:** contains `PONG` — human-readable prose, NOT a JSONL event stream ✓.  
**C1 subcommands:** `logs` ✓ (shows "PONG"), `clean` ✓ (readable prose, not JSON events!),
`watch --json` ✓, `steer` on one-shot exits non-zero ✓, `--echo` ✓ (`log.clean` written).

---

### Cell 4 — claude / interactive — PASS

**Run:** `v2-claude-interactive`  
**Command:** `agent-run --harness claude -i --prompt "Wait for steer..." --harness-arg --settings ... v2-claude-interactive`

| Source | Value |
|--------|-------|
| `session.json` | `dbcd0e70-3579-4d6f-b33f-52527f9b4807` |
| Harness own record | `~/.claude/projects/-Users-marcio-git-toolbox-harness/dbcd0e70-3579-4d6f-b33f-52527f9b4807.jsonl` ✓ |
| Wire (`x-claude-code-session-id`) | `dbcd0e70-3579-4d6f-b33f-52527f9b4807` ✓ |

**Acquisition:** `pushed` / `certain`.  
**Steer:** `agent-run steer v2-claude-interactive "HELLO WORLD"` → exit 0, `clean` shows
claude's response to "HELLO WORLD" ✓.  
**C1 subcommands:** all ✓. `kill` terminated cleanly after SIGKILL.

---

### Cell 5 — opencode / interactive — PASS

**Run:** `v2-opencode-interactive`  
**Command:** `agent-run --harness opencode -i --model llmproxy-anthropic/claude-sonnet-4.6 --prompt "Wait for steer..." v2-opencode-interactive`

| Source | Value |
|--------|-------|
| `session.json` | `ses_ff39849b6ffeH1899DNpcBn3mZ` |
| DB (`opencode.db`, read-only) | id=`ses_ff39849b6ffeH1899DNpcBn3mZ`, parent_id=None, agent=build, title=v2-opencode-interactive ✓ |
| Wire (`x-session-id`) | `ses_ff39849b6ffeH1899DNpcBn3mZ` ✓ |

**Acquisition:** `minted` / `certain`.  
**Steer:** `agent-run steer v2-opencode-interactive "HELLO WORLD"` → exit 0, opencode
responded ✓.  
**C1 subcommands:** all ✓.

---

### Cell 6 — codex / interactive — PASS (was F2)

**Run:** `accept-codex-interactive3`  
**Command:** `agent-run --harness codex -i --prompt "Reply with exactly: PONG" --harness-arg -c --harness-arg "model_providers.llmproxy.base_url=http://127.0.0.1:47511/v1" accept-codex-interactive3`

**Mechanism:** `_run_managed_interactive_codex_appserver` — app-server stays alive for the session
lifetime. `thread/start` mints the thread id; session.json is written with `minted/certain` before
the first model call. The initial prompt is sent via `turn/start`. The FIFO/keeper mechanism from
`_run_interactive` is replicated (keeper child holds the write end open); FIFO bytes are forwarded
as JSON-RPC `turn/start` (idle) or `turn/steer` (active). Agent text deltas stream to log_fd.

| Source | Value |
|--------|-------|
| `session.json` | `01a00c63-67fd-7211-94fd-17a6c7f39443` |
| Harness own record | `~/.codex/sessions/2026/08/16/rollout-2026-08-16T22-03-57-01a00c63-67fd-7211-94fd-17a6c7f39443.jsonl` ✓ |
| Wire (`x-client-request-id`) | `01a00c63-67fd-7211-94fd-17a6c7f39443` ✓ (matches session.json) |
| Wire (`originator`) | `codex_exec` ✓ (from `clientInfo.name="codex_exec"` in initialize) |

**Acquisition:** `minted` / `certain`.  
**Log:** contains `PONG` (initial turn) then `HELLO WORLD` (steered turn) — human-readable, NOT
JSON-RPC frames ✓.  
**Steer:** `agent-run steer accept-codex-interactive3 "Reply with exactly: HELLO WORLD"` → exit 0,
`clean` shows `PONG` + `HELLO WORLD` ✓. Steer was processed as `turn/start (steer idle)` (agent
was between turns when steer arrived).  
**C1 subcommands:** all ✓.

**Note on `originator`:** Interactive codex via app-server sends `originator: codex_exec` (same
as one-shot) because the `clientInfo.name="codex_exec"` is set in our initialize call. This is
correct and required for proxy attribution. The brief explicitly states "the TUI sending
`originator: codex-tui` is fine" — this was true when the PTY TUI path was used. Now that the
interactive path uses app-server with `clientInfo.name="codex_exec"`, both interactive and
one-shot send `codex_exec`.

---

## Summary table

| Cell | session.json | Harness record | Wire header | Steer |
|------|-------------|----------------|-------------|-------|
| claude / one-shot | `pushed/certain` ✓ | transcript ✓ | `x-claude-code-session-id` ✓ | n/a (one-shot, exits non-zero) |
| opencode / one-shot | `minted/certain` ✓ | db rows ✓ | `x-session-id` ✓ | n/a |
| codex / one-shot | `minted/certain` ✓ | rollout file ✓ | `x-client-request-id` ✓ | n/a |
| claude / interactive | `pushed/certain` ✓ | transcript ✓ | `x-claude-code-session-id` ✓ | ✓ answered |
| opencode / interactive | `minted/certain` ✓ | db rows ✓ | `x-session-id` ✓ | ✓ answered |
| codex / interactive | `minted/certain` ✓ | rollout file ✓ | `x-client-request-id` ✓ | ✓ answered (turn/start idle) |

**All 6 cells: PASS.**

---

## C1 — Subcommand verification

Verified per harness on the canonical acceptance runs:

| Subcommand | claude | opencode | codex |
|------------|--------|----------|-------|
| `logs` | ✓ readable | ✓ readable | ✓ readable (NOT JSONL) |
| `clean` | ✓ readable | ✓ readable | ✓ readable (NOT JSONL) |
| `tail` | ✓ (inherits log) | ✓ | ✓ |
| `status` | ✓ | ✓ | ✓ |
| `watch --json` | ✓ session field | ✓ session field | ✓ session field |
| `list --all` | ✓ | ✓ | ✓ |
| `kill` | ✓ SIGKILL path | ✓ clean term | ✓ clean term |
| `du` | ✓ | ✓ | ✓ |
| `steer` (one-shot) | exits non-zero ✓ | exits non-zero ✓ | exits non-zero ✓ |
| `steer` (interactive) | ✓ answered | ✓ answered | ✓ answered (app-server turn/start) |
| `--echo` | n/a tested | n/a tested | ✓ log.clean written |
| `--idle-timeout` | n/a tested | n/a tested | ✓ doesn't break managed run |

**Key C1 result:** After replacing `--json` with app-server, `agent-run clean v2-codex-oneshot`
shows human-readable `PONG`, NOT a JSONL event stream. Interactive codex now also shows readable
prose (`PONG\nHELLO WORLD`). No JSON-RPC frames appear in any run log.

---

## C2 — run.json manifest

`_write_run_json(log_dir, data)` writes atomically to `$AGENT_RUN_LOG_DIR/<name>/run.json`.
Written at launch with: `name`, `argv`, `command`, `cwd`, `started_at`, `harness`, `interactive`,
`model`, `agent_mode` (when managed). Updated at exit with: `ended_at`, `exit_code`, `status`.
Merges on write (exit update preserves launch fields).

Raw runs get `run.json` (no `session.json`). Liveness files (`pid`, `pgid`, `status`, `stdin`,
`process_identity`, etc.) remain in `/tmp/agent-runs/` only — the invariant is intact.

Test: `test_run_json_survives_ephemeral_state_deletion` simulates a reboot by deleting
`/tmp/agent-runs/<name>/` after a completed run and asserts all attribution fields remain
recoverable from `/var/tmp/agent-runs/<name>/run.json`.

---

## Findings

**F1 (resolved in implementation):** `codex exec resume <id>` after `app-server thread/start`
fails with "no rollout found" because the rollout `.jsonl` is created on `turn/start`, not
`thread/start`. Resolution: keep app-server alive across the full turn sequence.

**F2 (resolved in this pass):** Interactive codex session id was `missing` in the previous pass
because the PTY TUI path did not expose the id. Resolved by running interactive codex through
`codex app-server` with `turn/steer` wiring. `session.json` now shows `minted/certain`.

**F3 (opencode prefork mint):** `opencode run --port N` exits before the health poll can connect.
Fixed: `_opencode_prefork_mint` now launches `opencode --port N --auto` (bare TUI, headless)
which binds the HTTP server and stays alive until explicitly killed.

**F4 (OPENAI_API_KEY injection):** `codex app-server` fails immediately with "Missing environment
variable: OPENAI_API_KEY" when launched from a shell that does not export it. Fixed:
`_codex_subprocess_env()` reads `~/.codex/auth.json` and injects the key into the subprocess
environment. One-shot and interactive codex both use this helper.

---

## Test counts

`uv run pytest -q` on `feat/agent-run-harness`:

- Before this pass (baseline at `4637910`): 837 passed, 1 skipped
- After this pass: **841 passed, 1 skipped** (+4 new tests)

New tests cover: `TestManagedCodexInteractiveAppServer` (4 tests):
- `test_interactive_codex_minted_session_json`
- `test_interactive_codex_log_contains_initial_response`
- `test_interactive_codex_steer_lands_and_answered`
- `test_interactive_codex_fallback_to_missing_when_appserver_fails`
