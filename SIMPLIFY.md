# Simplifier pass — `verify/adversarial-x4`

Net effect: `src/toolbox/agent_run.py` −33 lines, tests net −2 cases (744 → 742),
zero behaviour change. Every guard that survived was proven load-bearing by
mutation; two guards that no test pinned are now pinned.

```
 src/toolbox/agent_run.py                | 111 +++++++++++---------------------
 tests/test_agent_run_launch_watchdog.py |  12 ++++
 tests/test_agent_run_reap.py            |  60 ++++++++---------
 tests/test_agent_run_watch.py           |  35 ++++++----
```

## Observed test run

```
uv run pytest -q
742 passed, 1 skipped, 1 warning in 34.17s
```

Baseline was 744 passed, 1 skipped. The delta is exactly the four removals and
two additions itemised below; no test was dropped without either a proven
duplicate or a strictly stronger replacement.

## `git log --oneline main..HEAD`

```
7f330b8 FIXES.md: document consolidated review findings and mutation results
589fbaf agent-run: fix kill/watchdog safety gaps, git env hardening, and dead helpers
febeec0 agent-run: fix subcommand-dispatch hijack via NAME--COMMAND and walk-error data loss
9d781f6 docs(agent-run): describe --max-seconds as a soft candidate-admission budget
855c919 agent-run: fix process-group, watch-contract, and reap-lifecycle defects
```

---

## What was removed, and why it was safe

### 1. Duplicated `iterdir` + `lstat` + `S_ISDIR` scan → `_real_subdir_names`

The same "list this root's real subdirectories, skipping dot-names and
symlinks" logic had accreted in four places in two shapes: `cmd_list` used a
comprehension over a `_lstat_st_mode(p)` helper (round 1), while
`_du_collect_rows` used an open-coded per-entry `try/except OSError` loop
(round 2). Both shapes implement the same contract; the `_lstat_st_mode`
indirection existed only so the comprehension form could swallow `OSError`.

Replaced all four with one `_real_subdir_names(root)` and deleted
`_lstat_st_mode` (now zero callers). Behaviour is identical in both directions:

- `cmd_list` previously let an `OSError` from `iterdir()` itself propagate; the
  helper still propagates it and `cmd_list` still does not wrap. `du` wrapped it
  before and still wraps it.
- Per-entry `lstat` failure is skipped in both, as before.

Proven by mutation — all four symlink-exclusion tests and both per-entry
tolerance tests fail against a reverted helper (see mutation table).

### 2. Assertion-free `inspect.getsource` test → a boundary observation

`test_git_no_replace_objects_is_set` grepped the *source text* of
`_watch_run_git_checked` for the string `GIT_NO_REPLACE_OBJECTS`, and
`test_git_replace_ref_base_is_stripped` asserted membership in a module
constant. Neither observed the git subprocess. Both are replaced by one test
that captures the actual `env=` dict handed to `subprocess.run` and asserts
`GIT_REPLACE_REF_BASE` is absent and `GIT_NO_REPLACE_OBJECTS == "1"`.

This is strictly stronger, not merely equal: the source-grep version passes if
the line is present but dead (e.g. assigned to a dict that is then discarded),
and the constant-membership version passes if the constant is never consulted.
The replacement still discriminates **both** mutations independently — verified
separately, not just together.

### 3. Genuinely duplicate reap tests (2 removed)

- `test_old_empty_preserved_log_is_collected` vs.
  `test_genuine_empty_dir_with_old_mtime_is_collected` — same fixture shape
  (empty dir, backdated container mtime), same call, same assertion, only the
  age constant differed (48 h vs. 72 h, both far past the 24 h threshold). Kept
  the one with the explanatory docstring.
- `test_preserved_log_age_recheck_passes_when_log_was_written_inside_lock` vs.
  `test_preserved_log_inode_recheck_retains_same_inode_with_fresh_content` —
  identical mechanism (write a child under the mocked lock, assert the dir
  survives). Despite its name the former exercised the *age* recheck, not the
  inode recheck; both mapped to the same guard. Merged into one test, renamed to
  say what it actually pins (`..._age_recheck_retains_same_inode_with_fresh_content`).

### 4. Comment bloat

Trimmed roughly 30 lines of comment that restated adjacent code or hedged, at
these sites: the `_WALK_INCOMPLETE` sentinel definition, two inline comments
inside `_newest_mtime_recursive`, the `WATCH_LINE_COUNT_MAX_BYTES` constant, the
porcelain-parsing block, two `cmd_reap` skip comments, the `cmd_watch` docstring,
the `_force_kill_legacy` ESRCH branch, two `_idle_watchdog_loop` branches, and
the `_parse_launch_argv` subcommand gate.

Every comment naming a real failure mode, bound, or unit was kept — the ~10x
output expansion from `--untracked-files=all` and its `timeout` failure mode,
the 16 MiB / ~180 bytes-per-second derivation, the container-mtime semantics in
the empty-dir fallback, and the "may not be named after a subcommand" rule.
Comments that were pure narration ("Process slot has been reused; a different
process occupies this pid. Stop guarding immediately — the original runner is
gone") were compressed to the fact ("PID reuse: a different process holds this
pid, so stop guarding").

---

## What was added (two guards that no test pinned)

Simplification surfaced two under-tested guards. Neither is a new feature; both
pin existing behaviour that was silently unprotected.

### `test_preserved_log_inode_recheck_survives_old_replacement`

Deleting the pass-2.5 inode recheck failed **zero** tests, which made it look
redundant with `_crash_safe_rmtree(..., expected=log_before)`. It is not. I
probed it directly (`/var/tmp/.../probe_inode.py`, two runs, one per code
variant):

- **With** the recheck: the swapped-in directory is left alone.
- **Without** it: `_crash_safe_rmtree` renames the replacement to a
  `.reaping-victim.<pid>.<ns>` sentinel *before* the `expected=` inode
  comparison rejects the delete. The `expected=` check then correctly refuses to
  empty it — but the directory is now stranded under a dot-prefixed sentinel
  name, invisible to every scan in the module, and the **next** reap's
  `_reap_stale_sentinels` sweep deletes it unconditionally. Probe output
  confirmed `PRECIOUS SURVIVED: False` under the mutation and the stranded
  `.reaping-victim.*` entry in `LOG_ROOT`.

The guard is load-bearing on a destructive path across two invocations, so it
stays and now has a test. The test deliberately uses an *equally old*
replacement so the age recheck cannot mask the inode recheck, and additionally
asserts no sentinel-named entry is left behind.

### Bounding `test_watchdog_recognizes_output_before_first_poll`

This test was load-bearing but failed by **hanging**: without the
`last_signature = initial_log_stat` seed the watchdog never records first
output, so the 600 s startup grace suppresses the kill forever and the test
never returns. I confirmed this (no termination after 45 s of wall clock).

A test that hangs under regression is a CI hazard, not a signal. Added a
`time.sleep` counter that asserts at 10 polls (the fixed path needs ~2). The
test now fails in ~10 s instead of hanging, with the same discriminating power.

---

## Mutation results — every touched test still fails without its fix

Each mutation was applied in isolation to a pristine copy and reverted
immediately afterwards.

| Mutation applied to `agent_run.py` | Test(s) that failed |
| --- | --- |
| `_real_subdir_names` uses `is_dir()` (follows symlinks) | `test_list_ignores_top_level_symlink[state,log]`, `test_du_ignores_top_level_symlink[state,log]` (4 failed) |
| `_real_subdir_names` per-entry `OSError` tolerance removed | `test_du_per_entry_lstat_error_skips_that_entry_only[state,log]` (2 failed) |
| `_dir_size_bytes` top `lstat`/`S_ISDIR` guard removed | `test_symlink_top_returns_zero` |
| `GIT_REPLACE_REF_BASE` removed from strip set | `test_git_replace_env_is_neutralised_at_the_subprocess_boundary` |
| `env["GIT_NO_REPLACE_OBJECTS"]` line deleted | `test_git_replace_env_is_neutralised_at_the_subprocess_boundary` |
| `GIT_GRAFT_FILE` removed from strip set | `test_git_graft_file_is_stripped` |
| Final HEAD recheck removed | `test_head_change_during_observation_degrades`, `test_per_call_cap_holds` |
| Porcelain rename-origin skip removed | `test_rename_origin_path_starting_with_question_marks_not_counted` |
| `--untracked-files=all` removed | `test_untracked_files_below_one_directory_are_counted_individually` |
| Pass-2.5 outer `_WALK_INCOMPLETE` guard removed | `test_scandir_error_retains_fresh_content_log` |
| Empty-dir container-mtime fallback removed | `test_genuine_empty_dir_with_old_mtime_is_collected` |
| Pass-2.5 inode recheck removed | `test_preserved_log_inode_recheck_survives_old_replacement` (new) |
| Pass-2.5 age recheck removed | `test_preserved_log_age_recheck_retains_same_inode_with_fresh_content` (merged) |
| Pass-3 age recheck removed | `test_orphan_scratch_age_recheck_retains_freshly_written_file` |
| `_pid_alive` `pid <= 0` guard removed | `test_nonpositive_pid_is_not_probed_or_published[-1,-42]` (3 failed) |
| Watch non-positive pid nulling removed | `test_nonpositive_pid_is_not_probed_or_published[-1,-42]` (3 failed) |
| `"starting"` dropped from verified status set | `test_starting_pid_is_verified[...]` (3 failed) |
| `WATCH_LINE_COUNT_MAX_BYTES` cap bypassed | `test_large_log_line_count_is_unknown_instead_of_scanned` |
| Watch argparse exit 2 no longer remapped to 1 | `test_watch_parse_errors_exit_1[argv1,argv2]` (3 failed) |
| `_runner_state_root` returns `STATE_ROOT` on non-Linux | `test_darwin_unknown_state_root_skips_candidate` |
| Reap budget reverted to `time.time()` | `test_monotonic_budget_is_not_fooled_by_wall_clock_step_back` |
| `--max-seconds` help reverted to "wall-clock budget" | `test_help_describes_budget_as_soft` |
| `_watchdog_escalate` identity precheck removed | `test_watchdog_escalation_does_not_publish_when_identity_changed` |
| `_watchdog_escalate` aux-pid parent check removed | `test_watchdog_escalation_does_not_signal_aux_pid_with_wrong_parent` |
| Watchdog identity check made strict (`None` exits) | `test_watchdog_transient_identity_probe_failure_does_not_abandon_run` |
| `last_signature` no longer seeded from `initial_log_stat` | `test_watchdog_recognizes_output_before_first_poll` |
| `_force_kill_legacy` pgid containment check removed | `test_force_kill_legacy_refuses_mismatched_recorded_group` |
| `_force_kill_legacy` ESRCH clause removed | `test_force_kill_legacy_esrch_from_getpgid_is_not_a_refusal` |
| Log facts + signals reopen the log separately | `test_one_observation_opens_log_once` |

---

## Deliberately left alone

**The pass-3 orphan-scratch inode recheck.** This is the one guard I could prove
is individually redundant: removing *either* the explicit
`(st_dev, st_ino) != (scratch_before...)` comparison *or* the
`expected=scratch_before` argument to `_safe_rmtree` leaves the full suite green
(743 passed with `expected=` dropped), because each alone stops the deletion.
Unlike pass 2.5, `_safe_rmtree` does **not** rename before deleting, so there is
no sentinel-stranding failure mode to make the outer check independently
necessary.

I left it. It is four lines on a destructive path, and the two checks fail in
different ways: the explicit comparison skips cleanly with no output, while
`expected=` returns from inside `_safe_rmtree` after the directory has already
been opened. Deleting the outer check would make correctness depend entirely on
an argument passed to a function three call levels away — exactly the
defence-in-depth the brief says not to strip from a destructive path. I also
wrote and then **discarded** a test for it, because no fixture I could construct
distinguished the two guards; a test that passes with either guard present has
no discriminating power over the one it claims to pin.

**`os.setsid()` placement.** Moving it back into the intermediate child leaves
the suite fully green (742 passed) — no test pins it. I did not touch it. The
guarantee is real (the runner must be the session and process-group leader so
`pgid == pid`, which `_force_kill_legacy`'s group-targeted SIGKILL depends on),
but it is only observable across a real double-fork with a live PTY, which no
unit test can reach without spawning actual processes. Reverting it would be a
behaviour change; writing a test for it is out of scope for a simplifier pass.
Flagging it as a genuine coverage gap rather than silently leaving it.

**`_watch_tail_lines_from_file`'s `WATCH_TAIL_MAX_BYTES - bytes_read` clamp.**
Dropping the third term from the `min()` leaves all five
`TestWatchTailLinesByteBound` cases green, because they assert against a bound
of `WATCH_TAIL_MAX_BYTES + WATCH_TAIL_READ_BLOCK_BYTES` — which is exactly the
one-block overshoot the clamp prevents. The clamp is a real tightening (exact
cap vs. cap-plus-one-block) and the tests were written to the looser bound
before it existed. Tightening the assertions would be a behaviour-adjacent
change to a test I did not otherwise need to touch, so I left both alone and
note the slack here.

**`_WALK_INCOMPLETE`'s `# type: ignore[return-value]` casts.** The sentinel
widens the return type beyond `Optional[float]`. A cleaner encoding (a small
result dataclass, or `Union[float, None, _WalkIncomplete]`) would touch all four
call sites and the signature. Correctness is unaffected and the current form is
locally obvious; not worth the churn under a behaviour-preserving mandate.

**`_watch_log_facts_from_file` / `_watch_tail_lines_from_file` docstrings.**
Both are undocumented after the round-2 split extracted them from
`_watch_log_facts` / `_watch_tail_lines`. The prose that used to describe them
now lives on `_watch_open_validated_log` (descriptor-vs-pathname validation) and
`cmd_watch` (the `log.lines` null contract), which is where it belongs — both
functions now take an already-validated descriptor and neither owns the
contract any more. Adding docstrings back would re-duplicate that prose.
