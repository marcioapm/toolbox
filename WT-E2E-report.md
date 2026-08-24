# WT-E2E: end-to-end coverage for `--worktree` launch

New file: `tests/test_agent_run_worktrees_e2e.py` (15 tests), added on top of the
existing hermetic suite `tests/test_agent_run_worktrees.py` (unchanged). Follows the
structure and skip convention of `tests/test_agent_run_live_acceptance.py`.

## Which cases are genuinely end-to-end

All six case groups drive the real CLI entrypoint (`agent_run.main()` in-process, or
`python -m toolbox.agent_run` as a real subprocess when a real interrupt or a real
long-lived process is needed), against a real git repository under `tmp_path`, with
real forked/double-forked processes and real filesystem state.

| Group | Cases | Real seams exercised | Substituted seam |
|---|---|---|---|
| E2E1 | happy path, `--worktree-branch`, annotated-tag base | real `git worktree add`, real fork/exec, real `git worktree list`, real HEAD | none |
| E2E2 | pre-process rollback removes; post-process rollback preserves | real preflight `sys.exit` after worktree creation; real `os.fork()`, real SIGINT delivered to a real launcher process with a real live runner | none |
| E2E3 | prefork-mint boundary (round-4 High) | real `subprocess.Popen` mint child with the worktree as its cwd, real descendant `sleep`, real SIGINT escaping the mint cleanup path | the external `opencode` binary is replaced with a real shell script that forks a real sleeping descendant and never reports healthy, so the mint step blocks in its own real health-poll loop. This mirrors the A2 precedent in `test_agent_run_live_acceptance.py`: a genuine managed OpenCode TUI/app-server launch is impractical to drive deterministically and quickly in a test, so only the far end of the process (an actual OpenCode server) is faked. The fork, the cwd, the ownership mark, and the rollback refusal all execute unmodified. |
| E2E4 | leaked-worktree cleanup commands actually work | commands are captured from real stderr of a real subprocess and executed through a real shell (`subprocess.run(..., shell=True)`), with a space in both the repo and worktree paths and a shell-significant branch name | none |
| E2E5 | collision/reuse refusals (nonempty dir, registered-worktree-no-reuse, reuse-attaches, branch-checked-out-elsewhere, `--worktree`+`--cwd`) | real git state inspected before/after each refusal | none |
| E2E6 | `du` and `reap --include-worktrees` | real linked worktree sized by `du`, really removed by `reap --include-worktrees` (dry-run then real), a real live run's worktree confirmed un-reaped | none |

No case fakes the thing it claims to exercise; the only substitution (E2E3) is
plainly stated in the module docstring, in the exact style of the existing A2 note.

## Gate

Every test is marked `@live_only`, gated on `AGENT_RUN_LIVE_TESTS == "1"`, identical
in form to `tests/test_agent_run_live_acceptance.py`'s `live_only` marker. Every git
repo, worktree, and state/log root is created under pytest's `tmp_path` via the
existing `isolated_runs_root`/`isolated_log_root` fixtures (which monkeypatch
`AGENT_RUN_STATE_DIR`/`AGENT_RUN_LOG_DIR` and `agent_run.STATE_ROOT`/`LOG_ROOT`), or,
for the tests that must launch a real subprocess, by exporting those same two env
vars into that subprocess's environment. No test invokes an internal helper
(`cmd_launch`, `_create_launch_worktree`, etc.) directly — every launch goes through
`agent_run.main()` or a real `python -m toolbox.agent_run` subprocess.

## Verification run and results

Environment: macOS (darwin/arm64), repo venv is Python 3.13.13; a separate `uv`-managed
venv was created for the Python 3.12 regression run. All commands below were run from
the repo root, `/private/tmp/tb-e2e`, on branch `feat/worktree-create-e2e`.

### 1. Whole new file, `AGENT_RUN_LIVE_TESTS=1`, five times

```
for i in 1 2 3 4 5; do
  AGENT_RUN_LIVE_TESTS=1 .venv/bin/python -m pytest tests/test_agent_run_worktrees_e2e.py -v
done
```

Result: **15 passed / 0 failed / 0 skipped, all five runs**, each in 9.5-9.8s wall
time (E2E3's fake-opencode health-poll timeout — bounded, ~1s in practice because the
interrupt lands well before the 30s ceiling — dominates runtime). No flake observed;
no seed to report because none of these tests use randomised ordering or seeded
inputs — every wait is a bounded poll on a real file/process condition
(`_wait_for`), with no wall-clock assertions and no fixed sleep used as a
synchronisation primitive.

Per-run summary (identical across all five):
```
tests/test_agent_run_worktrees_e2e.py::TestE2E1RealLaunch:: (3 tests) PASSED
tests/test_agent_run_worktrees_e2e.py::TestE2E2RollbackContract:: (2 tests) PASSED
tests/test_agent_run_worktrees_e2e.py::TestE2E3PreforkMintBoundary:: (1 test) PASSED
tests/test_agent_run_worktrees_e2e.py::TestE2E4LeakedWorktreeCleanupCommandsWork:: (1 test) PASSED
tests/test_agent_run_worktrees_e2e.py::TestE2E5CollisionAndReuseShapes:: (5 tests) PASSED
tests/test_agent_run_worktrees_e2e.py::TestE2E6DuAndReapAccountForRealWorktree:: (3 tests) PASSED
15 passed, 1 warning in ~9.5-9.8s
```

### 2. Whole new file, without the env var

```
unset AGENT_RUN_LIVE_TESTS
.venv/bin/python -m pytest tests/test_agent_run_worktrees_e2e.py -v
```

Result: **15 skipped, 0 failed, 0 errors**, in 0.03s.

### 3. Proof the rollback tests are not vacuous

Target: the guard in `cmd_launch` (`src/toolbox/agent_run.py:9126`):
```python
if getattr(args, "_worktree_process_started", False):
```

Baseline hash before any mutation:
```
775c36e5e63307a2f5c78107427b7f13e6510f1813cae50b9abf7d0bafa2ffe4  src/toolbox/agent_run.py
```

**Mutation A — force the guard always-true** (`if True or getattr(...)`, i.e. *never
roll back, always warn-and-leave*):
```
AGENT_RUN_LIVE_TESTS=1 .venv/bin/python -m pytest tests/test_agent_run_worktrees_e2e.py -v -k "E2E2 or E2E3"
```
Result: `test_preflight_failure_after_worktree_creation_removes_worktree_and_branch`
**FAILS** (`assert not wt.exists()` — the worktree that should have been rolled back
back on a genuine preflight failure survives). The other two tests in that selection
(`test_post_process_failure_preserves_worktree_and_branch`,
`test_interrupt_during_mint_cleanup_refuses_rollback_with_live_child`) still pass,
as expected — an always-true guard never rolls back post-process failures either, so
those two preservation assertions are unaffected by this mutation; the one case this
mutation breaks is exactly the one it should break.

**Mutation B — force the guard always-false** (`if False and getattr(...)`, i.e.
*always attempt rollback, ignore the ownership mark*):
```
AGENT_RUN_LIVE_TESTS=1 .venv/bin/python -m pytest tests/test_agent_run_worktrees_e2e.py -v -k "E2E2 or E2E3"
```
Result: **both**
`test_post_process_failure_preserves_worktree_and_branch` (E2E2) and
`test_interrupt_during_mint_cleanup_refuses_rollback_with_live_child` (E2E3) **FAIL**
(`assert wt.is_dir()` — the worktree that a live process is using gets deleted out
from under it). `test_preflight_failure_after_worktree_creation_removes_worktree_and_branch`
still passes, as expected: that case never depended on the mark being read correctly.

Both mutations were applied and reverted one at a time (never combined), each
followed immediately by the targeted pytest run above, and then the file was
restored with `Edit` to its original text.

**Restoration verified by hash:**
```
$ shasum -a 256 src/toolbox/agent_run.py
775c36e5e63307a2f5c78107427b7f13e6510f1813cae50b9abf7d0bafa2ffe4  src/toolbox/agent_run.py
$ git diff --stat src/toolbox/agent_run.py
(no output — clean)
```
Matches the pre-mutation hash exactly. A final full run of the new file after
restoration reconfirmed **15 passed** (`AGENT_RUN_LIVE_TESTS=1 .venv/bin/python -m
pytest tests/test_agent_run_worktrees_e2e.py -q` → `15 passed`).

This demonstrates the rollback tests are load-bearing on the exact production guard
they exist to pin, not vacuously true regardless of its logic.

### 4. No test touched anything outside `tmp_path`

Snapshot of the real GC roots and the real repo's worktree list, taken immediately
before and after the full five-run live loop in step 1 (and again around the
mutation-testing runs in step 3):

```
ls /tmp/agent-runs | sort            > before   (117 entries)
ls /var/tmp/agent-runs | sort        > before   (159 entries)
git -C /private/tmp/tb-e2e worktree list > before  (49 entries)
... run the full live suite ...
ls /tmp/agent-runs | sort            > after
ls /var/tmp/agent-runs | sort        > after
git -C /private/tmp/tb-e2e worktree list > after
diff before after   # empty for /tmp/agent-runs and /var/tmp/agent-runs
diff before after   # worktree list: only this repo's own HEAD commit line changed
                     # (from committing test files), no new/removed worktree rows
```

Result: **zero new entries** in either real GC root across all runs (five live runs
in step 1, plus the two mutation runs in step 3). The only difference in
`git worktree list` between snapshots was this repo's own recorded HEAD commit
advancing as new test-file commits were made — no worktree row was added, removed,
or changed elsewhere. Every git repository and worktree the tests themselves created
lived under a pytest `tmp_path` (confirmed by construction: every `_make_repo`,
`_launch_argv`, and subprocess `env` in the new file roots exclusively at `tmp_path`
and the `isolated_runs_root`/`isolated_log_root` fixtures).

### 5. Full suite on Python 3.12, before/after comparison

A second venv was created with `uv venv --python 3.12` and the package installed
editable (`uv pip install -e ".[dev]"`) into it, run against the repo **as-is**
(current tree, "after") and against a temporary `git worktree` checked out at
`d0274f2` — the commit immediately preceding this task's first commit — ("before").

`def test_` counts:
```
before (d0274f2):  1235
after  (HEAD):      1250   (+15, exactly the new file's test count)
```

Full suite, Python 3.12.13, `AGENT_RUN_LIVE_TESTS` unset (matches CI default — the
new file's 15 tests all skip):

```
before (d0274f2):  1657 passed, 10 skipped, 133 warnings in 503.05s
after  (HEAD):      1657 passed, 25 skipped, 133 warnings in 504.13s
```

Passed count is identical (**1657 → 1657**, zero regression); skipped count rose by
exactly 15 (**10 → 25**), matching the 15 new gated tests skipping without the live
env var. Zero failures, zero errors in either run.

The temporary baseline worktree was removed after the comparison
(`git worktree remove ... --force`); `git worktree list` in the real repo returned to
its pre-check state (49 entries, confirmed by diff against the pre-verification
snapshot).

## Summary of what was skipped

Nothing in the required verification list was skipped. All five steps were executed
and are reported above with the actual commands and actual output.

## Commits

One commit per case group, none squashed:

1. `tests: e2e coverage for --worktree happy-path launch (group 1)`
2. `tests: e2e coverage for the worktree rollback contract (group 2)`
3. `tests: e2e coverage for the prefork-mint ownership boundary (group 3)`
4. `tests: e2e coverage for the leaked-worktree cleanup commands (group 4)`
5. `tests: e2e coverage for worktree collision/reuse and du/reap accounting (groups 5-6)`
