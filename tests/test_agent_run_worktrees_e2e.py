"""End-to-end coverage for `--worktree` launch: real CLI, real git, real
processes, real filesystem.

Gate: AGENT_RUN_LIVE_TESTS=1, exactly as tests/test_agent_run_live_acceptance.py.
Every test skips unless that variable is set.

E2E1 -- real launch: worktree registration, branch name, HEAD, the recorded
        launch cwd, --worktree-branch, and an annotated-tag base.
E2E2 -- rollback contract: a preflight failure reached after worktree
        creation removes the worktree and branch; a failure once a real
        forked runner exists preserves both.
E2E3 -- prefork-mint boundary (round-4 High): a real child process holds
        the worktree as its cwd, and an interrupt escapes mint cleanup.
        The `opencode` binary is substituted with a real sleeping shell
        script -- the ownership-mark/rollback machinery under test runs
        unmodified against a real child process with a real pid and a real
        cwd. This mirrors the A2 substitution already documented in
        test_agent_run_live_acceptance.py: driving a genuine managed
        opencode launch deterministically and quickly in a test is
        impractical, so only the external agent binary is faked.
E2E4 -- the leaked-worktree cleanup commands, captured from real stderr and
        executed through a real shell, must actually remove the worktree
        and branch they name.
E2E5 -- collision/reuse refusals through the real CLI, checked against real
        exit codes, real stderr, and real untouched filesystem state.
E2E6 -- `du` and `reap --include-worktrees` against a real linked worktree.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Optional

import pytest

from toolbox import agent_run

# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------

LIVE = os.environ.get("AGENT_RUN_LIVE_TESTS") == "1"
live_only = pytest.mark.skipif(
    not LIVE, reason="set AGENT_RUN_LIVE_TESTS=1 to run live worktree e2e tests"
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _git(cwd: Path, *args: str) -> str:
    """Run git in ``cwd`` with a deterministic identity and no user config."""
    env = {
        **os.environ,
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@example.invalid",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@example.invalid",
    }
    result = subprocess.run(
        ["git", "-C", str(cwd), *args],
        capture_output=True, text=True, check=True, env=env,
    )
    return result.stdout


def _make_repo(root: Path, name: str = "repo") -> Path:
    """A real repo with two commits, an annotated tag at HEAD, and a bare
    'origin' remote holding the same commit, so a freshly branched worktree
    has nothing unpushed."""
    repo = root / name
    repo.mkdir(parents=True)
    _git(repo, "init", "-q", "-b", "main", ".")
    (repo / "a.txt").write_text("one\n")
    _git(repo, "add", "a.txt")
    _git(repo, "commit", "-qm", "first")
    (repo / "a.txt").write_text("one\ntwo\n")
    _git(repo, "add", "a.txt")
    _git(repo, "commit", "-qm", "second")
    _git(repo, "tag", "-a", "v1", "-m", "annotated", "HEAD")
    origin = root / f"{name}-origin.git"
    subprocess.run(["git", "init", "-q", "--bare", str(origin)], check=True)
    _git(repo, "remote", "add", "origin", str(origin))
    _git(repo, "push", "-q", "origin", "main")
    return repo


def _wait_for(predicate: Callable[[], bool], timeout: float, message: str) -> None:
    """Poll ``predicate`` until true, or fail with a diagnostic message."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    pytest.fail(f"timed out after {timeout}s waiting for: {message}")


def _wait_terminal(state_dir: Path, timeout: float = 20.0) -> str:
    """Poll <state_dir>/status until it reaches a terminal value."""
    status = "starting"

    def _check() -> bool:
        nonlocal status
        try:
            status = (state_dir / "status").read_text().strip()
        except FileNotFoundError:
            status = "starting"
        return status in agent_run.TERMINAL_STATUSES

    _wait_for(_check, timeout, f"terminal status in {state_dir} (last={status!r})")
    return status


def _wait_pid_file(state_dir: Path, field: str = "pid", timeout: float = 10.0) -> int:
    _wait_for(lambda: (state_dir / field).exists(), timeout, f"{field} file in {state_dir}")
    return int((state_dir / field).read_text().strip())


def _pid_gone(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


def _kill_and_reap(pid: Optional[int], timeout: float = 10.0) -> None:
    """Terminate a real process this test started, and wait for it to exit."""
    if not pid or pid <= 0:
        return
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return
    try:
        os.kill(pid, __import__("signal").SIGKILL)
    except ProcessLookupError:
        return
    _wait_for(lambda: _pid_gone(pid), timeout, f"pid {pid} to exit after SIGKILL")


def _exit_status(code: object) -> int:
    """The OS-visible exit status ``sys.exit(code)`` would produce."""
    if code is None:
        return 0
    if isinstance(code, int):
        return code
    return 1


def _launch_argv(
    *,
    name: str,
    worktree: Path,
    worktree_base: str,
    worktree_repo: Path,
    worktree_branch: Optional[str] = None,
    worktree_reuse: bool = False,
    command: Optional[list[str]] = None,
    prompt_file: Optional[str] = None,
    cwd: Optional[str] = None,
) -> list[str]:
    """Build a real agent-run CLI argv for a raw `--worktree` launch."""
    argv: list[str] = []
    if cwd is not None:
        argv += ["--cwd", cwd]
    argv += [
        "--worktree", str(worktree),
        "--worktree-base", worktree_base,
        "--worktree-repo", str(worktree_repo),
    ]
    if worktree_branch is not None:
        argv += ["--worktree-branch", worktree_branch]
    if worktree_reuse:
        argv.append("--worktree-reuse")
    if prompt_file is not None:
        argv += ["-f", prompt_file]
    argv.append(name)
    if command is not None:
        argv += ["--", *command]
    return argv


@pytest.fixture(autouse=True)
def _restore_cwd():
    """`_create_launch_worktree`'s invocation-cwd discovery and
    `_apply_launch_cwd` both chdir the real test process; put it back so
    later tests in the same session are unaffected."""
    origin = os.getcwd()
    yield
    os.chdir(origin)


# ---------------------------------------------------------------------------
# E2E1: real launch
# ---------------------------------------------------------------------------

@live_only
class TestE2E1RealLaunch:
    """`agent-run --worktree DIR --worktree-base REF NAME` through the real
    CLI entrypoint, a real git repository, and a real double-forked process."""

    def test_happy_path_worktree_registered_branch_head_and_cwd(
        self, isolated_runs_root, isolated_log_root, tmp_path
    ):
        repo = _make_repo(tmp_path, "repo1")
        base_oid = _git(repo, "rev-parse", "main").strip()
        wt = tmp_path / "wt1"
        name = "e2e-happy"
        argv = _launch_argv(
            name=name, worktree=wt, worktree_base=base_oid, worktree_repo=repo,
            command=["true"],
        )

        rc = agent_run.main(argv)
        assert rc == 0

        state_dir = isolated_runs_root / name
        status = _wait_terminal(state_dir)
        assert status == "done"

        # Real `git worktree list` sees it as a registered linked worktree
        # of the fixture repo.
        porcelain = _git(repo, "worktree", "list", "--porcelain")
        assert str(wt.resolve()) in porcelain

        assert _git(wt, "rev-parse", "HEAD").strip() == base_oid
        assert name in _git(repo, "branch", "--list", name)
        assert (state_dir / "cwd").read_text().strip() == str(wt.resolve())

    def test_explicit_worktree_branch_overrides_run_name(
        self, isolated_runs_root, isolated_log_root, tmp_path
    ):
        repo = _make_repo(tmp_path, "repo2")
        wt = tmp_path / "wt2"
        name = "e2e-branch"
        branch = "e2e-explicit-branch"
        argv = _launch_argv(
            name=name, worktree=wt, worktree_base="main", worktree_repo=repo,
            worktree_branch=branch, command=["true"],
        )

        assert agent_run.main(argv) == 0
        state_dir = isolated_runs_root / name
        assert _wait_terminal(state_dir) == "done"

        branches = _git(repo, "branch", "--list")
        assert branch in branches
        assert name not in branches
        assert (
            _git(wt, "symbolic-ref", "--quiet", "HEAD").strip() == f"refs/heads/{branch}"
        )

    def test_annotated_tag_base_resolves_to_the_peeled_commit(
        self, isolated_runs_root, isolated_log_root, tmp_path
    ):
        repo = _make_repo(tmp_path, "repo3")
        tag_commit = _git(repo, "rev-parse", "v1^{commit}").strip()
        wt = tmp_path / "wt3"
        name = "e2e-tag-base"
        argv = _launch_argv(
            name=name, worktree=wt, worktree_base="v1", worktree_repo=repo,
            command=["true"],
        )

        assert agent_run.main(argv) == 0
        state_dir = isolated_runs_root / name
        assert _wait_terminal(state_dir) == "done"

        assert _git(wt, "rev-parse", "HEAD").strip() == tag_commit
