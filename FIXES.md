# Consolidated review fixes — `verify/adversarial-x4`

---

## H1 — Subcommand-dispatch hijack via `NAME -- COMMAND`

**What changed:** Removed the `explicit_launch` gate in `_parse_launch_argv`
(`src/toolbox/agent_run.py`). The gate keyed on `tokens[1] == "--"` and caused
any `SUBCOMMAND -- ARG` invocation to launch a new background run named after
the subcommand instead of dispatching it. A run may not be named after a known
subcommand (the pre-existing safe rule). Removed two tests that asserted the
hijacking behaviour; added parametrised regression tests asserting subcommand
dispatch is preserved when `--` follows a subcommand name.

**Mutation result:** Restoring the `explicit_launch` gate fails
`test_subcommand_name_with_dashdash_dispatches_as_subcommand` for every
subcommand (11 failures).

---

## H2 — Walk-error treated as empty dir deletes fresh preserved logs

**What changed:** `_newest_mtime_recursive` now returns the module-level
`_WALK_INCOMPLETE` sentinel (distinct from `None`) when an `OSError` prevented
the walk from completing. `None` means walk completed with no entries (genuinely
empty). All four call sites in `cmd_reap` check for the sentinel and skip the
candidate; only a completed walk with `None` applies the `log_before.st_mtime`
fallback.

**Mutation result:** Removing the `if newest is _WALK_INCOMPLETE: continue`
guard in pass 2.5 fails `test_scandir_error_retains_fresh_content_log`
(1 failure).

---

## M1 — `_force_kill_legacy` ESRCH from `getpgid` refuses instead of falling through

**What changed:** Added a separate `except ProcessLookupError` clause before the
`except OSError` clause in the `getpgid` call inside `_force_kill_legacy`. On
`ESRCH` the process is already gone; the fix falls through to the existing
"not running" handler rather than calling `sys.exit`.

**Mutation result:** Reverting to `except OSError as exc: sys.exit(...)` fails
`test_force_kill_legacy_esrch_from_getpgid_is_not_a_refusal` (1 failure).

---

## M2 — `_du_collect_rows` comprehension-wide `lstat` abort

**What changed:** Replaced the set comprehension in `_du_collect_rows` (both
`STATE_ROOT` and `LOG_ROOT` blocks) with a per-entry `try/except OSError` loop,
so a single vanishing entry is skipped rather than aborting the entire scan.

**Mutation result:** Reverting to the comprehension fails
`test_du_per_entry_lstat_error_skips_that_entry_only` (1 failure per root).

---

## M3 — `cmd_list` and `_dir_size_bytes` still follow symlinks

**What changed:**
- Introduced `_lstat_st_mode(p)` helper (returns `p.lstat().st_mode` or 0 on
  error) and applied it in `cmd_list` for both `STATE_ROOT` and `LOG_ROOT`
  enumeration, replacing the old `p.is_dir()` calls.
- `_dir_size_bytes` now `lstat`s its own top argument and returns 0 for any
  non-real-directory (symlink or missing).  Updated the docstring accordingly.

**Mutation result:** Reverting `cmd_list` to `p.is_dir()` fails
`test_list_ignores_top_level_symlink` (2 failures, one per root kind).
Reverting `_dir_size_bytes`'s top-argument lstat fails
`test_symlink_top_returns_zero` (1 failure).

---

## M4 — Idle watchdog abandons run on transient `_process_identity` failure

**What changed:** `_idle_watchdog_loop` now only returns when `current_identity`
is readable *and* differs from `runner_identity`.  When `current_identity is
None` (probe failed), it falls back to `_pid_alive`; the loop exits only if
both agree the runner is gone.  The escalation grace poll after SIGTERM also
switches to `_pid_alive` (avoiding ~40 `ps` subprocesses per firing).  Fixed
`test_watchdog_stops_when_runner_identity_changes` and
`test_watchdog_does_not_fire_*` tests to monkeypatch `_pid_alive` so the
identity check is the only termination path.

**Mutation result:** Reverting to `if _process_identity(runner_pid) !=
runner_identity: return` fails
`test_watchdog_transient_identity_probe_failure_does_not_abandon_run`
(1 failure).

---

## M5 — `GIT_GRAFT_FILE` not stripped from git subprocess env

**What changed:** Added `"GIT_GRAFT_FILE"` to `WATCH_GIT_ENV_VARS_TO_STRIP`.
An inherited graft file silently rewrites parent pointers, making
`commits_since_start` return a wrong value with no `git_error`.  Also split
the combined `test_replacement_refs_are_disabled` into independent tests for
`GIT_REPLACE_REF_BASE` (strip set membership) and `GIT_NO_REPLACE_OBJECTS`
(env override in `_watch_run_git_checked`), so each mechanism has its own
coverage.

**Mutation result:** Removing `"GIT_GRAFT_FILE"` from the strip set fails
`test_git_graft_file_is_stripped` (1 failure).  Asserting
`GIT_REPLACE_REF_BASE in WATCH_GIT_ENV_VARS_TO_STRIP` independently catches
removal of that entry (1 failure).

---

## M6 — Dead `_watch_log_facts`, `_watch_signals`, `_watch_tail_lines`

**What changed:** Deleted all three functions (zero production callers since
`_watch_log_observation` replaced them).  Repointed
`TestWatchTailLinesByteBound` and
`test_large_log_line_count_is_unknown_instead_of_scanned` at
`_watch_tail_lines_from_file` / `_watch_log_facts_from_file` so the
`WATCH_TAIL_MAX_BYTES` cap and the `WATCH_LINE_COUNT_MAX_BYTES` cap are
actually covered on the live code path.

**Mutation result:** Removing the `bytes_read < WATCH_TAIL_MAX_BYTES` bound
from `_watch_tail_lines_from_file` fails
`test_byte_cap_enforced_on_newline_free_file` (1 failure). Setting
`WATCH_LINE_COUNT_MAX_BYTES` to `sys.maxsize` fails
`test_large_log_line_count_is_unknown_instead_of_scanned` (1 failure).

---

## M7 — Vacuous tests

Each item below was proven vacuous by the mutation listed; the fix makes the
test load-bearing.

**`_watch_tail_lines_from_file` byte cap** — covered by M6 above.

**`test_watchdog_stops_when_runner_identity_changes`** — the test used pid
4242 (not alive), so the loop exited on the liveness check before reaching the
identity comparison.  Fixed by monkeypatching `_pid_alive` to always return
`True`; the identity check is now the only exit path.  
*Mutation:* reverting loop guard to `_pid_alive` alone → 1 failure.

**`test_watchdog_escalation_publishes_killed_with_a_reason`** — stubs both new
guards to always-agree values.  Added three negative tests:
`test_watchdog_escalation_does_not_publish_when_identity_changed` (mismatched
identity → no signal, no terminal state),
`test_watchdog_escalation_does_not_signal_aux_pid_with_wrong_parent` (parent
check), and the transient-probe test from M4.  
*Mutation:* removing the identity guard from `_watchdog_escalate` → 1 failure.

**Monotonic reap budget** — the budget comparison `time.monotonic() -
reap_start > reap_budget` was only covered by timing, not a discriminating
mock.  New test mocks `time.monotonic()` to return `start + 1000.0` after the
first call (guaranteeing budget exhaustion) while `time.time()` returns a
past value (so a `time.time()`-based bug would see elapsed ≈ 0 and never
defer).  
*Mutation:* reverting six `time.monotonic()` sites to `time.time()` → 1 failure.

**Both reap inode/age rechecks** — existing tests only exercised the
pre-existing `expected=` inode gate.  Added
`test_preserved_log_inode_recheck_retains_same_inode_with_fresh_content` (same
inode, fresh child file written under lock — only the age recheck can save it)
and `test_preserved_log_age_recheck_passes_when_log_was_written_inside_lock`
(confirms the locked age walk fires even when the inode recheck passes).
Same for pass 3 scratch: `test_orphan_scratch_age_recheck_retains_freshly_written_file`.  
*Mutation:* removing the age recheck from pass 2.5 fails each new test (1 failure each).

**`GIT_REPLACE_REF_BASE` / `GIT_NO_REPLACE_OBJECTS`** — split into independent
assertions; see M5 above.

---

## L1 — `untracked_files` over-counts rename/copy origins

**What changed:** The `-z` porcelain stream emits two NUL-delimited records for
each rename/copy: a status record (`R  <new>`) and a bare origin path with no
XY prefix.  If the origin path began with `?? ` it was counted as an untracked
file.  Replaced the filter with a state machine: when XY field is `R`/`C`,
consume the next record as the origin path and do not classify it.

**Mutation result:** Reverting to `sum(1 for r in records if r.startswith("?? "))`
fails `test_rename_origin_path_starting_with_question_marks_not_counted`
(1 failure).

---

## L2 — `log.lines: null` undocumented in `cmd_watch` contract

**What changed:** Added a sentence to the `cmd_watch` docstring documenting
that `log.lines` is `null` above `WATCH_LINE_COUNT_MAX_BYTES` and that
`log.bytes` is always populated.  Added the 16 MiB rationale comment next to
the constant.

*No behaviour change; no mutation applicable.*

---

## L3 — `--untracked-files=all` cost undocumented

**What changed:** Added a comment next to the `--untracked-files=all` flag
noting that it disables git's directory-collapse optimisation (~10x on a large
untracked tree) and that the failure mode is `git_error: "timeout"`.

*Comment only; no behaviour change.*

---

## L4 — `_runner`'s `setsid()` causal comment in the wrong function

**What changed:** Rewrote the `# After setsid(), pid == pgid` comment to
reference `_cmd_launch_locked` as the actual call site, since `setsid()` moved
there in the prior commit.

*Comment only; no behaviour change.*

---

## L5 — Unreachable `or pid <= 0` in `_watch_effective_status`

**What changed:** Dropped `or pid <= 0` from the `if pid is None or pid <= 0`
guard.  The `pid <= 0` arm is unreachable because `_effective_status` already
resolves a non-positive pid to `died` via `_pid_alive` before `_watch_effective_status`
sees it.

*Mutation:* removing the `or pid <= 0` (the now-deleted clause) has no effect
on the suite — the guard remains via `_pid_alive`'s own protection.

---

## Full suite result

```
744 passed, 1 skipped in 34.04s
```

(725 baseline + 19 new tests)

## `git log --oneline main..HEAD`

```
589fbaf agent-run: fix kill/watchdog safety gaps, git env hardening, and dead helpers
febeec0 agent-run: fix subcommand-dispatch hijack via NAME--COMMAND and walk-error data loss
9d781f6 docs(agent-run): describe --max-seconds as a soft candidate-admission budget
855c919 agent-run: fix process-group, watch-contract, and reap-lifecycle defects
```
