# Acceptance Results — agent-run session id ↔ proxy csid

Captured 2026-08-16 on macmini. All 6 cells run through `agent-run --harness` managed mode.
Forward-capture proxy (`fwdcap.py`) forwarded to `https://llmproxy.absmartly-dev.com` and logged
all request headers. Tests: `uv run pytest -q` → **837 passed, 1 skipped** (the 1 skipped is
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

**Interactive codex (finding F2, deferred):** `codex` TUI (`codex -i`) sends
`x-client-request-id` on the wire but does not expose the id at startup. Wiring `agent-run steer`
to `turn/steer` over the app-server's stdin pipe requires the app-server to remain the active
process for the interactive session's lifetime AND output to stream as readable text — this is
non-trivial to implement cleanly alongside the existing PTY/FIFO steer mechanism. Per the brief:
"if wiring it to `agent-run steer <name>` is not clean in this pass, keep the PTY TUI for
interactive and leave `missing`, and say so explicitly." Interactive codex stays `missing`.
Steer via the PTY FIFO still works (verified below, Cell 6).

---

## Part B — 6-cell acceptance matrix

Proxy ports: claude=47510, codex=47511, opencode=47512 (all forwarding to llmproxy).
Run name convention: `accept-<harness>-<mode>` (the canonical acceptance run per cell).

---

### Cell 1 — claude / one-shot — PASS

**Run:** `accept-claude-oneshot`  
**Command:** `agent-run --harness claude --prompt "Reply with exactly: PONG" --harness-arg --settings --harness-arg /path/to/claude-cap-settings.json accept-claude-oneshot`

| Source | Value |
|--------|-------|
| `session.json` | `8d1b3320-2242-4315-a0b4-ff049480ee27` |
| Harness own record | `~/.claude/projects/-private-var-tmp-agent-runs-harness-codex-tmp-accept-workdir/8d1b3320-2242-4315-a0b4-ff049480ee27.jsonl` ✓ |
| Wire (`x-claude-code-session-id`) | `8d1b3320-2242-4315-a0b4-ff049480ee27` ✓ |

**Acquisition:** `pushed` / `certain` (agent-run generated the UUID4, passed `--session-id`).  
**Log:** contains `PONG` (human-readable).  
**C1 subcommands:** `logs` ✓, `clean` ✓ (shows "PONG"), `watch --json` ✓ (session id present),
`list --all` ✓, `steer` on one-shot exits non-zero with clear message ✓.

---

### Cell 2 — opencode / one-shot — PASS

**Run:** `accept-opencode-oneshot2`  
**Command:** `agent-run --harness opencode -m llmproxy-anthropic/claude-sonnet-4.6 --prompt "Reply with exactly: PONG" accept-opencode-oneshot2`

**Setup note:** `opencode run --port N` exits before the health poll can complete (verified:
`opencode run` answers and exits in ~5s, leaving the port released). Fixed in
`_opencode_prefork_mint`: now launches `opencode --port N --auto` (bare TUI, headless) which
binds the HTTP API and stays alive until killed. Health poll completes in ~2.5s (3 attempts).

| Source | Value |
|--------|-------|
| `session.json` | `ses_ff3e99aabffe5GIOH0x80bJ5By` |
| DB (`opencode.db`, read-only) | id=`ses_ff3e99aabffe5GIOH0x80bJ5By`, parent_id=None, agent=build, title=accept-opencode-oneshot2, **messages=2** ✓ (not a decoy empty session) |
| Wire (`x-session-id`) | `ses_ff3e99aabffe5GIOH0x80bJ5By` ✓ |

**Acquisition:** `minted` / `certain`.  
**Log:** contains `PONG`.  
**C1 subcommands:** `logs` ✓, `clean` ✓, `watch --json` ✓, `steer` on one-shot exits non-zero ✓.

---

### Cell 3 — codex / one-shot — PASS

**Run:** `accept-codex-oneshot3`  
**Command:** `agent-run --harness codex --prompt "Reply with exactly: PONG" --harness-arg -c --harness-arg "model_providers.llmproxy.base_url=..." accept-codex-oneshot3`

**Mechanism:** `_run_managed_oneshot_codex_appserver` — app-server stays alive, mints thread,
sends `turn/start`, streams `item/agentMessage/delta` text chunks to log, waits for
`turn/completed`. No `--json` flag; stdout is the log file. App-server chatter goes only to
`session-acquire.log`.

| Source | Value |
|--------|-------|
| `session.json` | `01a00c20-e6e3-7350-bf8e-0e41c2d90e22` |
| Harness own record | `~/.codex/sessions/2026/08/16/rollout-2026-08-16T20-51-18-01a00c20-e6e3-7350-bf8e-0e41c2d90e22.jsonl` ✓ |
| Wire (`x-client-request-id`) | `01a00c20-e6e3-7350-bf8e-0e41c2d90e22` ✓ |
| Wire (`originator`) | `codex_exec` ✓ (clientInfo.name="codex_exec" — required for proxy identify_client) |

**Acquisition:** `minted` / `certain`.  
**Log:** contains `PONG` — human-readable prose, NOT a JSONL event stream ✓.  
**C1 subcommands:** `logs` ✓ (shows "PONG"), `clean` ✓ (readable prose, not JSON events!),
`watch --json` ✓, `steer` on one-shot exits non-zero ✓, `--echo` ✓ (`log.clean` written).

---

### Cell 4 — claude / interactive — PASS

**Run:** `accept-claude-interactive`  
**Command:** `agent-run --harness claude -i --prompt "Wait for message..." --harness-arg --settings ... accept-claude-interactive`

| Source | Value |
|--------|-------|
| `session.json` | `1be8080b-1ec1-4912-b482-b706b864bd5f` |
| Harness own record | `~/.claude/projects/-private-var-tmp-agent-runs-harness-codex-tmp-accept-workdir/1be8080b-1ec1-4912-b482-b706b864bd5f.jsonl` ✓ |
| Wire (`x-claude-code-session-id`) | `1be8080b-1ec1-4912-b482-b706b864bd5f` ✓ |

**Acquisition:** `pushed` / `certain`.  
**Steer:** `agent-run steer accept-claude-interactive "HELLO WORLD"` → exit 0, `clean` shows
claude's response to "HELLO WORLD" ✓.  
**C1 subcommands:** all ✓. `kill` terminated cleanly after SIGKILL.

---

### Cell 5 — opencode / interactive — PASS

**Run:** `accept-opencode-interactive`  
**Command:** `agent-run --harness opencode -i -m llmproxy-anthropic/claude-sonnet-4.6 --prompt "Wait for message..." accept-opencode-interactive`

| Source | Value |
|--------|-------|
| `session.json` | `ses_ff3dcb97dffeFSFLiIdiLNIqTn` |
| DB (`opencode.db`, read-only) | id=`ses_ff3dcb97dffeFSFLiIdiLNIqTn`, parent_id=None, agent=build, title=accept-opencode-interactive, **messages=2** ✓ |
| Wire (`x-session-id`) | `ses_ff3dcb97dffeFSFLiIdiLNIqTn` ✓ |

**Acquisition:** `minted` / `certain`.  
**Steer:** `agent-run steer accept-opencode-interactive "HELLO WORLD"` → exit 0, opencode
responded ✓.  
**C1 subcommands:** all ✓.

---

### Cell 6 — codex / interactive — PARTIAL (finding F2)

**Run:** `accept-codex-interactive`  
**Command:** `agent-run --harness codex -i --prompt "Reply with exactly: PONG" --harness-arg -c --harness-arg "..." accept-codex-interactive`

| Source | Value |
|--------|-------|
| `session.json` | `null` — `confidence=missing` |
| Wire (`x-client-request-id`) | `01a00c24-7b97-7c13-ab71-c7836255ee31` (observed but not recorded) |
| Wire (`originator`) | `codex-tui` (different from one-shot's `codex_exec`) |

**Finding F2:** The interactive codex TUI (`codex <prompt>`) assigns a new thread id but does not
expose it at PTY startup. The proxy captures `x-client-request-id` and `originator=codex-tui` but
agent-run cannot correlate these to the session id without either: (a) reading the pty output and
parsing the `session id:` stderr banner (fragile, pty escape-code stripped), or (b) running the
interactive TUI through `app-server` and wiring `turn/steer` to the FIFO steer path — which
requires the app-server to stay alive for the session lifetime and output to stream as readable
text through the PTY. Deferred per brief guidance.

**Steer:** `agent-run steer accept-codex-interactive "Reply with exactly: PONG"` → exit 0,
`clean` shows `P O N G` (codex TUI renders each character separately) ✓. Steer mechanism works.

---

## Summary table

| Cell | session.json | Harness record | Wire header | Steer |
|------|-------------|----------------|-------------|-------|
| claude / one-shot | `pushed/certain` ✓ | transcript ✓ | `x-claude-code-session-id` ✓ | n/a (one-shot, exits non-zero) |
| opencode / one-shot | `minted/certain` ✓ | db rows=2 ✓ | `x-session-id` ✓ | n/a |
| codex / one-shot | `minted/certain` ✓ | rollout file ✓ | `x-client-request-id` ✓ | n/a |
| claude / interactive | `pushed/certain` ✓ | transcript ✓ | `x-claude-code-session-id` ✓ | ✓ answered |
| opencode / interactive | `minted/certain` ✓ | db rows=2 ✓ | `x-session-id` ✓ | ✓ answered |
| codex / interactive | `missing` (F2) | — | `x-client-request-id` (not recorded) | ✓ PTY steer works |

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
| `steer` (interactive) | ✓ answered | ✓ answered | ✓ answered (PTY) |
| `--echo` | n/a tested | n/a tested | ✓ log.clean written |
| `--idle-timeout` | n/a tested | n/a tested | ✓ doesn't break managed run |

**Key C1 result:** After replacing `--json` with app-server, `agent-run clean accept-codex-oneshot3`
shows human-readable `PONG`, NOT a JSONL event stream. This was the primary regression risk.

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

**F2 (deferred, documented):** Interactive codex TUI (`codex <prompt>`) session id is `missing`.
The wire shows `x-client-request-id` with `originator=codex-tui` but agent-run cannot safely
correlate it without either pty banner parsing or app-server turn/steer wiring. Steer via PTY FIFO
works correctly.

**F3 (opencode prefork mint):** `opencode run --port N` exits before the health poll can connect.
Fixed: `_opencode_prefork_mint` now launches `opencode --port N --auto` (bare TUI, headless)
which binds the HTTP server and stays alive until explicitly killed.

---

## Test counts

`uv run pytest -q` on `feat/agent-run-harness`:

- Before this pass (baseline at `6607014`): 823 passed, 1 skipped
- After this pass: **837 passed, 1 skipped** (+14 new tests)

New tests cover: `_codex_appserver_mint` (4 tests), `_build_managed_argv` for codex with
app-server (3 tests), `_run_managed_oneshot_codex_appserver` via fake JSON-RPC fake (4 tests),
`_write_run_json` / `run.json` lifecycle (6 tests).
