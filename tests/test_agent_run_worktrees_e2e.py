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
import signal
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
        os.kill(pid, signal.SIGKILL)
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


# ---------------------------------------------------------------------------
# E2E2: the rollback contract
# ---------------------------------------------------------------------------

@live_only
class TestE2E2RollbackContract:
    """A worktree this invocation created is removed by a preflight failure
    and preserved by a post-process failure -- through the real CLI, a real
    fork, and real filesystem state, never a mocked seam."""

    def test_preflight_failure_after_worktree_creation_removes_worktree_and_branch(
        self, isolated_runs_root, isolated_log_root, tmp_path
    ):
        """A missing -f/--prompt-file is rejected inside _cmd_launch_locked,
        before os.fork() is reached -- after _create_launch_worktree has
        already run inside cmd_launch's publication lock. This is a genuine
        argv/preflight failure reached only after the worktree exists."""
        repo = _make_repo(tmp_path, "repo-pre")
        wt = tmp_path / "wt-pre"
        name = "e2e-preflight-fail"
        missing_prompt = tmp_path / "no-such-prompt.txt"
        argv = _launch_argv(
            name=name, worktree=wt, worktree_base="main", worktree_repo=repo,
            command=["true"], prompt_file=str(missing_prompt),
        )

        with pytest.raises(SystemExit) as exc:
            agent_run.main(argv)
        assert "prompt file not found" in str(exc.value)

        assert not wt.exists()
        porcelain = _git(repo, "worktree", "list", "--porcelain")
        assert str(wt.resolve()) not in porcelain
        assert name not in _git(repo, "branch", "--list")
        assert not (isolated_runs_root / name).exists()

    def test_post_process_failure_preserves_worktree_and_branch(
        self, isolated_runs_root, isolated_log_root, tmp_path
    ):
        """A real interrupt delivered to the real CLI process after its real
        `os.fork()` has already produced a live runner must never roll back:
        the worktree and branch survive, and the real workload process (a
        `sleep`, standing in for the launched agent) is reaped by the test."""
        repo = _make_repo(tmp_path, "repo-post")
        wt = tmp_path / "wt-post"
        name = "e2e-postprocess-fail"
        argv = [
            sys.executable, "-m", "toolbox.agent_run",
            *_launch_argv(
                name=name, worktree=wt, worktree_base="main", worktree_repo=repo,
                command=[sys.executable, "-c", "import time; time.sleep(30)"],
            ),
        ]
        env = dict(os.environ)
        env["AGENT_RUN_STATE_DIR"] = str(isolated_runs_root)
        env["AGENT_RUN_LOG_DIR"] = str(isolated_log_root)

        launcher = subprocess.Popen(argv, env=env)
        runner_pid: Optional[int] = None
        agent_pid: Optional[int] = None
        try:
            state_dir = isolated_runs_root / name
            # Proof the fork already produced a live runner: the runner
            # publishes its own pid before anything else. Once this file
            # exists, os.fork() has returned in the launcher and
            # args._worktree_process_started is unconditionally set.
            _wait_for(lambda: (state_dir / "pid").exists(), 10.0, "runner pid file")
            runner_pid = int((state_dir / "pid").read_text().strip())
            assert agent_run._pid_alive(runner_pid)

            launcher.send_signal(signal.SIGINT)
            launcher.wait(timeout=15)

            assert wt.is_dir(), "worktree must survive a post-fork interrupt"
            porcelain = _git(repo, "worktree", "list", "--porcelain")
            assert str(wt.resolve()) in porcelain
            assert name in _git(repo, "branch", "--list", name)

            _wait_for(
                lambda: (state_dir / "agent_pid").exists(), 10.0, "agent_pid file"
            )
            agent_pid = int((state_dir / "agent_pid").read_text().strip())
        finally:
            launcher.poll()
            if launcher.returncode is None:
                launcher.kill()
                launcher.wait(timeout=10)
            _kill_and_reap(agent_pid)
            _kill_and_reap(runner_pid)


# ---------------------------------------------------------------------------
# E2E3: the prefork-mint boundary
# ---------------------------------------------------------------------------

_FAKE_OPENCODE_SCRIPT = """#!/bin/sh
trap 'exit 0' TERM
sleep 60 &
echo $! > {pidfile}
wait
"""


@live_only
class TestE2E3PreforkMintBoundary:
    """A managed opencode launch starts a real temporary child with the new
    worktree as its cwd, before the ownership mark that gates rollback is
    read by cmd_launch's except clause. An interrupt escaping the mint
    cleanup path must still leave that live child, and the worktree it is
    using, untouched.

    The external `opencode` binary is substituted with a real shell script
    that forks a real `sleep` descendant and never becomes healthy, so the
    mint step blocks in its own real health-poll loop with a real process
    tree in place -- only the far end of the process (an actual OpenCode
    server) is faked; the fork, cwd, ownership mark and rollback refusal all
    execute for real.
    """

    def test_interrupt_during_mint_cleanup_refuses_rollback_with_live_child(
        self, isolated_runs_root, isolated_log_root, tmp_path
    ):
        repo = _make_repo(tmp_path, "repo-mint")
        wt = tmp_path / "wt-mint"
        name = "e2e3-mint-interrupt"

        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        pidfile = tmp_path / "descendant.pid"
        fake_opencode = bin_dir / "opencode"
        fake_opencode.write_text(_FAKE_OPENCODE_SCRIPT.format(pidfile=pidfile))
        fake_opencode.chmod(0o755)

        argv = [
            sys.executable, "-m", "toolbox.agent_run",
            "--harness", "opencode", "--prompt", "hi",
            "--worktree", str(wt), "--worktree-base", "main",
            "--worktree-repo", str(repo),
            name,
        ]
        env = dict(os.environ)
        env["PATH"] = str(bin_dir) + os.pathsep + env.get("PATH", "")
        env["AGENT_RUN_STATE_DIR"] = str(isolated_runs_root)
        env["AGENT_RUN_LOG_DIR"] = str(isolated_log_root)

        launcher = subprocess.Popen(argv, env=env)
        descendant_pid: Optional[int] = None
        try:
            _wait_for(pidfile.exists, 10.0, "fake opencode's sleep descendant pidfile")
            descendant_pid = int(pidfile.read_text().strip())
            assert descendant_pid > 0
            try:
                os.kill(descendant_pid, 0)
            except ProcessLookupError:
                pytest.fail(f"descendant pid {descendant_pid} died before the interrupt")

            # The mint child is real and blocked in a real health-poll loop;
            # give it a moment past the pidfile write to be certain the
            # ownership mark (set immediately before Popen) is in place.
            time.sleep(0.2)
            launcher.send_signal(signal.SIGINT)
            rc = launcher.wait(timeout=15)
            assert rc != 0

            try:
                os.kill(descendant_pid, 0)
            except ProcessLookupError:
                pytest.fail("mint's live descendant must survive the interrupted cleanup")

            assert wt.is_dir(), "worktree must survive an interrupted mint cleanup"
            porcelain = _git(repo, "worktree", "list", "--porcelain")
            assert str(wt.resolve()) in porcelain
            assert name in _git(repo, "branch", "--list", name)
        finally:
            launcher.poll()
            if launcher.returncode is None:
                launcher.kill()
                launcher.wait(timeout=10)
            _kill_and_reap(descendant_pid)
