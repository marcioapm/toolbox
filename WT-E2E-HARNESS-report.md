# WT-E2E-HARNESS report

`tests/test_agent_run_worktrees_e2e.py`, section `E2E7`, extends the OpenCode-only
E2E1-E2E6 suite to `claude` and `codex`. Four commits, one per numbered item in the
task:

- `01882aa` — item 1: `TestE2E7ClaudeWorktreeLaunch`
- `66c5d72` — item 1: `TestE2E7CodexWorktreeLaunch`
- `c310307` — items 2+3: `TestE2E7RollbackAppliesToEveryHarness`,
  `TestE2E7NoPreforkProcessForClaudeOrCodex`
- `84cc317` — item 4: `TestE2E7InteractiveWorktreeLaunch`

No changes to `src/toolbox/agent_run.py` in the final state (two mutations were
applied and reverted during verification; see the mutation table below). No
existing E2E1-E2E6 test was modified.

## What was added

### 1. Happy-path launch per harness

**`TestE2E7ClaudeWorktreeLaunch::test_worktree_created_and_session_id_pushed_via_argv`**
Real `agent-run --harness claude --worktree ...` through `agent_run.main()`. The
`claude` binary is substituted with a real shell script on `PATH` that writes its
own argv to a file and exits 0 — the same class of substitution `_FAKE_OPENCODE_SCRIPT`
uses for E2E3. Asserts:
- the worktree is registered (`git worktree list --porcelain`), branch created at
  the base commit, `<state_dir>/cwd` records the worktree path, terminal status
  `done` — the same contract E2E1 checks for raw launches.
- the fake's recorded argv contains `--session-id` followed by a value that parses
  as `uuid.UUID(...)` with `version == 4`.
- that value equals `session.json`'s `session_id`, with `acquisition == "pushed"`
  and `confidence == "certain"`.

**`TestE2E7CodexWorktreeLaunch::test_worktree_created_and_thread_id_lands_in_session_json`**
Real `agent-run --harness codex --worktree ...`. Codex never execs an argv — both
modes drive `codex app-server` over JSON-RPC (`agent_run.py:9415`) — so the fake
substitutes the app-server process itself. It is a real Python script invoked as
`codex app-server` that implements the exact frame sequence
`_CodexAppServer.mint_thread` (agent_run.py:11606) and
`_run_managed_oneshot_codex_appserver` (agent_run.py:11715) require:
`initialize` → response → `initialized` notification (no response) → `thread/start`
→ response with `result.thread.id` → `turn/start` → response → one
`item/agentMessage/delta` → `turn/completed`. This is not a guess: each frame shape
was read directly from `mint_thread`'s parsing (`result.get("thread") or result`,
`thread.get("id") or thread.get("sessionId")`) and from the one-shot runner's
frame-dispatch loop. The fake also records its own `os.getcwd()` to a file before
the handshake, proving the app-server subprocess itself — not just
`_apply_launch_cwd`'s effect on the launcher — was started with the worktree as its
cwd (`_CodexAppServer.start()` passes `cwd=cwd` to `Popen`, agent_run.py:11550).
Asserts the same worktree/branch/cwd contract as the claude test, plus the
recorded cwd equals the worktree, plus `session.json`'s `session_id` equals the
thread id the fake returned, with `acquisition == "minted"` and
`confidence == "certain"`.

This test passed on the first run with no protocol guessing required beyond
reading the two functions named above — the fake reached a real `thread/start`
round-trip and a real completed turn.

### 2. Rollback contract per harness

**`TestE2E7RollbackAppliesToEveryHarness`**, parametrized over
`["claude", "codex", "opencode"]`. Reuses E2E2's exact trigger (a missing
`-f/--prompt-file`, rejected inside `_cmd_launch_locked` at agent_run.py:9200-9202,
before any harness-specific session acquisition and before `os.fork()`). This
single failure point is harness-agnostic — none of the three harnesses' external
binaries need to be faked, since acquisition never runs. Asserts the worktree and
branch are removed, no run state dir exists, and the leaked-worktree warning
(`"leaving worktree in place"`) is never printed to stderr.

### 3. The pre-fork process claim, as a test

**`TestE2E7NoPreforkProcessForClaudeOrCodex::test_prefork_mark_reflects_only_the_harnesss_own_acquisition_path`**,
parametrized over `["claude", "codex", "opencode"]`. This is the test the task
exists for. A failure (`_persist_submit_mode` raising) is injected strictly after
every harness's own session-acquisition block completes (claude's push, codex's
argv stub, opencode's real prefork mint against a fake `opencode` binary) and
strictly before the unconditional `args._worktree_process_started = True` every
harness hits right before `os.fork()` (agent_run.py:9478). Because the injection
point sits after acquisition and before the unconditional mark, only a harness's
*own* acquisition path can have set the mark by the time the induced failure fires
— isolating exactly the claim under test rather than an artifact of where the
failure was placed.

Through a real `cmd_launch(args)` call (not a mock), the test reads
`args._worktree_process_started` after the induced failure:
- `claude`, `codex`: asserts `mark is False`, and that the worktree/branch were
  rolled back — the positive form of the audit claim.
- `opencode`: asserts `mark is True` (its real `_opencode_prefork_mint` genuinely
  ran `Popen(cwd=worktree)` against the fake `opencode` binary and set the mark
  unconditionally just before that `Popen`, agent_run.py:11215-11224), and that the
  worktree survived. This is the explicit non-vacuity inversion the task asked
  for: the test does not silently `if harness == "opencode": pass`; it asserts the
  *opposite* mark value and additionally asserts, via
  `with pytest.raises(AssertionError): assert mark is False`, that the
  claude/codex-shaped assertion is genuinely wrong for opencode — proving the
  parametrization is not vacuously true for all three harnesses at once.

### 4. Interactive mode for both

**`TestE2E7InteractiveWorktreeLaunch`**: one `-i --harness claude --worktree ...`
and one `-i --harness codex --worktree ...` real launch each. Both wait for
`status == "running"` (not just launch return), assert the worktree/branch/cwd
contract, and for codex additionally wait for the fake app-server's recorded cwd
file and assert it equals the worktree. Both then `SIGTERM` the runner pid and
assert a terminal status is reached. The codex fake
(`_make_fake_codex_appserver_interactive`) differs from the one-shot fake only in
that it blocks on a further `recv()` after the initial turn completes instead of
exiting — mirroring `_run_managed_interactive_codex_appserver`'s long-lived
session loop, which keeps the app-server alive across steers until the runner's
own teardown kills it.

## Mutation table (non-vacuity proof)

Two production mutations were applied, shown to cause the named test to fail
with the exact message below, then reverted. SHA-256 of
`src/toolbox/agent_run.py` before and after each mutation+revert cycle:

| # | Target test | Mutation | Location | Result before mutation (SHA-256) | Failure while mutated | Result after revert (SHA-256) |
|---|---|---|---|---|---|---|
| 1 | `TestE2E7RollbackAppliesToEveryHarness` (all 3 params) | `if getattr(args, "_worktree_process_started", False):` → `if True:` in `cmd_launch`'s except-clause, forcing the "leave worktree in place" branch unconditionally | `agent_run.py:9132` | `93f993b532cc092ad891f4e879ff9b143fcd3c412e149459c218a155b4c17823` | `AssertionError: assert not True` (`wt.exists()` for all of `claude`, `codex`, `opencode`) — 3 failed | `93f993b532cc092ad891f4e879ff9b143fcd3c412e149459c218a155b4c17823` (identical) |
| 2 | `TestE2E7NoPreforkProcessForClaudeOrCodex[claude]` | Added `args._worktree_process_started = True` immediately after claude's push-acquisition block, simulating a hypothetical pre-fork process creator for claude | `agent_run.py:9382` (inserted line, immediately after `_record_session(...)` in the `if harness == "claude":` block) | `93f993b532cc092ad891f4e879ff9b143fcd3c412e149459c218a155b4c17823` | `AssertionError: claude must never set the pre-fork mark` / `assert True is False` — 1 failed, `codex` and `opencode` unaffected (2 passed) | `93f993b532cc092ad891f4e879ff9b143fcd3c412e149459c218a155b4c17823` (identical) |

Both mutations left a real worktree/branch on disk in `tmp_path`-scoped
directories during the failing run (the point of the test); each was removed by
hand (`git worktree remove --force` + `git branch -D`) before the next
verification pass, and `tmp_path` itself is pytest-managed and rotates out.

## Verification results, verbatim

### 1. `AGENT_RUN_LIVE_TESTS=1 uv run pytest tests/test_agent_run_worktrees_e2e.py -q`

```
.........................                                                [100%]
=============================== warnings summary ===============================
.venv/lib/python3.13/site-packages/_pytest/config/__init__.py:1464
  /private/tmp/tb-comb/.venv/lib/python3.13/site-packages/_pytest/config/__init__.py:1464: PytestConfigWarning: Unknown config option: timeout
  
    self._warn_or_fail_if_strict(f"Unknown config option: {key}\n")

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
25 passed, 1 warning in 14.78s
```

25 = 15 original (E2E1-E2E6) + 10 new (2 happy-path + 3 rollback-parametrized +
3 pre-fork-mark-parametrized + 2 interactive).

### 2. Ungated (`AGENT_RUN_LIVE_TESTS` unset)

```
sssssssssssssssssssssssss                                                [100%]
=============================== warnings summary ===============================
...
25 skipped, 1 warning in 0.04s
```

All 25 skip; nothing fails or errors.

### 3. `uv run pytest tests/test_agent_run_worktrees.py -q`

```
........................................................................ [ 33%]
........................................................................ [ 67%]
....................................................................     [100%]
=============================== warnings summary ===============================
...
212 passed, 1 warning in 111.16s (0:01:51)
```

Still 212 passed, as required. This file was not modified.

### 4. Full suite (`uv run pytest -q`)

```
ssssssssss.............................................................. [ 92%]
........................................................................ [ 97%]
.................................................                        [100%]
=============================== warnings summary ===============================
...
1670 passed, 35 skipped, 134 warnings in 510.09s (0:08:30)
```

**1670 passed, 35 skipped, 0 failed.** Re-ran the three tests named as
pre-existing-and-out-of-scope in isolation to double check:

```
$ uv run pytest tests/test_agent_run_attach.py::test_tail_exits_promptly_on_sigint_during_a_continuous_output_burst tests/test_agent_run_hooks.py::TestHookResolution::test_ancestry_fallback_via_pgid tests/test_agent_run_hooks.py::TestHookResolution::test_ancestry_direct_ppid_match_wins -q
...
3 passed, 1 warning in 2.26s
```

All three passed, both embedded in the full run and in isolation. Nothing beyond
those three (which turned out not to fail at all) failed anywhere in the suite.
The 134 warnings are all pre-existing `DeprecationWarning: ... multi-threaded,
use of fork()` noise from `os.fork()` under pytest's threaded test runner,
unrelated to this change.

### 5. `ps` check for stray processes

After every live-gated run (including the two mutation runs), checked for:
- any process whose argv references this suite's `bin_dir` paths, or any of the
  E2E7 run names (`e2e7-claude-happy`, `e2e7-codex-happy`, `e2e7-rollback-*`,
  `e2e7-premark-*`, `e2e7-claude-interactive`, `e2e7-codex-interactive`):
  none found.
- any `sleep 60` process (the fake interactive `claude`/mint scripts' payload):
  none found.
- the two `sleep 600` processes visible in `ps aux` at verification time are
  unrelated background sessions already running on this shared development
  machine before this work started (substring collision with the `sleep 60`
  grep, not started by any test here).

Every one-shot test either lets the fake exit on its own (`exit 0`) or is
cleaned up by an explicit `finally` block; every interactive test SIGTERMs the
runner pid and calls `_kill_and_reap` in `finally`; the mutation-run leaked
worktrees (expected, since that is what the mutation defeats) were removed by
hand via `git worktree remove --force` / `git branch -D` before the next
verification pass.

## What could not be tested, and why

Nothing in the required scope was left uncovered by a weaker substitute. In
particular, the codex JSON-RPC fake was not needed as a fallback path: reading
`_CodexAppServer.mint_thread` and `_run_managed_oneshot_codex_appserver` directly
gave the exact frame sequence, and the resulting fake reached a real
`thread/start` round-trip and a real completed turn on the first attempt for both
the one-shot and interactive happy-path tests — this is asserted directly against
`session.json`'s `session_id`, not inferred from a weaker signal like exit code
alone.

Two things are outside what this suite can prove, by design, and are noted here
rather than silently asserted around:

- **Real `claude` and `codex` binaries are never invoked.** Every test in E2E7
  substitutes the external agent binary, exactly as E2E3 already does for
  `opencode`. This suite proves agent-run's own worktree/session-acquisition/
  rollback machinery is correct against a binary that behaves as documented; it
  does not and cannot prove the real `claude` CLI accepts `--session-id` the way
  agent-run assumes, or that the real `codex app-server` JSON-RPC wire format
  matches what `_CodexAppServer` parses. That gap exists in the pre-existing
  `test_agent_run_harness.py` fakes too and is unchanged by this work.
- **The `--worktree-reuse` and worktree-collision paths are not re-parametrized
  across harnesses.** E2E5 already covers those paths exhaustively for a raw
  launch; E2E7 does not duplicate that matrix per-harness, since none of E2E5's
  refusal paths depend on which harness is selected (they are all rejected
  inside `_create_launch_worktree`, before `harness` is read at all).
