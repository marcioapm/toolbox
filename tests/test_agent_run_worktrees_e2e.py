"""End-to-end coverage for `--worktree` launch: real CLI, real git, real
processes, real filesystem.

Gate: AGENT_RUN_LIVE_TESTS=1, exactly as tests/test_agent_run_live_acceptance.py.
Every test skips unless that variable is set.

E2E1 -- real launch: worktree registration, branch name, HEAD, the recorded
        launch cwd, --worktree-branch, and an annotated-tag base.
E2E2 -- rollback contract: a preflight failure reached after worktree
        creation removes the worktree and branch; a failure once a real
        forked runner exists preserves both.
E2E3 -- prefork-mint boundary: a real child process holds
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
E2E7 -- claude and codex acquire a managed session id with no process created
        in the worktree before the runner fork, unlike opencode's prefork
        mint; this section proves the resulting rollback contract holds for
        every harness, not just opencode.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import uuid
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
        # Widens the real window between the runner publishing its pid and
        # the launcher's parent process returning, so the SIGINT below
        # reliably lands while the launcher is still alive to receive it,
        # rather than racing the launcher's own fast normal exit.
        env["AGENT_RUN_TEST_SLOW_IDENTITY"] = "1"

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

            # The pidfile can only exist after the mint Popen, which itself
            # runs after the ownership mark is set, so the mark is already
            # in place here and no further wait is needed.
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


# ---------------------------------------------------------------------------
# E2E4: the leaked-worktree cleanup commands must actually work
# ---------------------------------------------------------------------------

@live_only
class TestE2E4LeakedWorktreeCleanupCommandsWork:
    """When a post-process failure leaves a worktree behind, the cleanup
    commands agent-run prints to stderr must be independently runnable: this
    executes them through a real shell and proves the worktree and branch
    are actually gone afterward. A space in both the repository path and the
    worktree path, plus a shell-significant branch name, forces every
    shlex.quote() call in the printed line to do real work."""

    def test_printed_cleanup_commands_executed_through_a_real_shell_remove_everything(
        self, isolated_runs_root, isolated_log_root, tmp_path
    ):
        git_root = tmp_path / "git roots"
        git_root.mkdir()
        repo = _make_repo(git_root, "main repo")
        wt = git_root / "wt dir; rm -rf nope"
        name = "e2e4-cleanup"
        branch = "feature;touch-pwned"
        argv = [
            sys.executable, "-m", "toolbox.agent_run",
            *_launch_argv(
                name=name, worktree=wt, worktree_base="main", worktree_repo=repo,
                worktree_branch=branch,
                command=[sys.executable, "-c", "import time; time.sleep(30)"],
            ),
        ]
        env = dict(os.environ)
        env["AGENT_RUN_STATE_DIR"] = str(isolated_runs_root)
        env["AGENT_RUN_LOG_DIR"] = str(isolated_log_root)
        # See E2E2: widens the runner-publication window so the SIGINT below
        # reliably lands on a still-alive launcher.
        env["AGENT_RUN_TEST_SLOW_IDENTITY"] = "1"

        launcher = subprocess.Popen(argv, env=env, stderr=subprocess.PIPE, text=True)
        runner_pid: Optional[int] = None
        agent_pid: Optional[int] = None
        try:
            state_dir = isolated_runs_root / name
            _wait_for(lambda: (state_dir / "pid").exists(), 10.0, "runner pid file")
            runner_pid = int((state_dir / "pid").read_text().strip())

            launcher.send_signal(signal.SIGINT)
            _, stderr = launcher.communicate(timeout=15)

            assert wt.is_dir(), "precondition: the worktree must have leaked"
            marker = "clean up by hand: "
            assert marker in stderr, f"no cleanup line printed; stderr={stderr!r}"
            cleanup_line = stderr[stderr.index(marker) + len(marker):].splitlines()[0]

            _wait_for(
                lambda: (state_dir / "agent_pid").exists(), 10.0, "agent_pid file"
            )
            agent_pid = int((state_dir / "agent_pid").read_text().strip())
            _kill_and_reap(agent_pid)
            _kill_and_reap(runner_pid)
            runner_pid = agent_pid = None

            # The exact recovery path advertised to an operator: run the
            # printed line through a real shell, not a re-parsed argv list.
            shell_result = subprocess.run(
                cleanup_line, shell=True, capture_output=True, text=True,
            )
            assert shell_result.returncode == 0, shell_result.stderr

            assert not wt.exists()
            porcelain = _git(repo, "worktree", "list", "--porcelain")
            assert str(wt.resolve()) not in porcelain
            assert branch not in _git(repo, "branch", "--list", branch)
        finally:
            launcher.poll()
            if launcher.returncode is None:
                launcher.kill()
                launcher.wait(timeout=10)
            _kill_and_reap(agent_pid)
            _kill_and_reap(runner_pid)


# ---------------------------------------------------------------------------
# E2E5: collision and reuse shapes
# ---------------------------------------------------------------------------

@live_only
class TestE2E5CollisionAndReuseShapes:
    """Every refusal path through the real CLI leaves pre-existing state on
    disk exactly as it was: no partial worktree, no partial branch, no
    state dir for the rejected run name."""

    def test_existing_nonempty_directory_without_reuse_is_refused_and_untouched(
        self, isolated_runs_root, isolated_log_root, tmp_path
    ):
        repo = _make_repo(tmp_path, "repo-nonempty")
        target = tmp_path / "already-here"
        target.mkdir()
        marker = target / "f.txt"
        marker.write_text("pre-existing content\n")
        name = "e2e5-nonempty"
        argv = _launch_argv(
            name=name, worktree=target, worktree_base="main", worktree_repo=repo,
            command=["true"],
        )

        with pytest.raises(SystemExit) as exc:
            agent_run.main(argv)
        assert _exit_status(exc.value.code) == 1
        assert "already exists" in str(exc.value)

        assert marker.read_text() == "pre-existing content\n"
        assert not (isolated_runs_root / name).exists()

    def test_existing_registered_worktree_without_reuse_is_refused(
        self, isolated_runs_root, isolated_log_root, tmp_path
    ):
        repo = _make_repo(tmp_path, "repo-registered")
        existing = tmp_path / "existing-wt"
        _git(repo, "worktree", "add", "-q", str(existing), "-b", "existing-branch", "main")
        head_before = _git(existing, "rev-parse", "HEAD").strip()
        name = "e2e5-registered-no-reuse"
        argv = _launch_argv(
            name=name, worktree=existing, worktree_base="main", worktree_repo=repo,
            worktree_branch="a-different-branch", command=["true"],
        )

        with pytest.raises(SystemExit) as exc:
            agent_run.main(argv)
        assert _exit_status(exc.value.code) == 1
        assert "already exists" in str(exc.value)

        assert _git(existing, "rev-parse", "HEAD").strip() == head_before
        porcelain = _git(repo, "worktree", "list", "--porcelain")
        assert str(existing.resolve()) in porcelain
        assert not (isolated_runs_root / name).exists()

    def test_existing_registered_worktree_with_reuse_attaches_and_launches(
        self, isolated_runs_root, isolated_log_root, tmp_path
    ):
        repo = _make_repo(tmp_path, "repo-reuse")
        existing = tmp_path / "reuse-wt"
        _git(repo, "worktree", "add", "-q", str(existing), "-b", "reuse-branch", "main")
        name = "e2e5-reuse-attach"
        argv = _launch_argv(
            name=name, worktree=existing, worktree_base="main", worktree_repo=repo,
            worktree_branch="reuse-branch", worktree_reuse=True, command=["true"],
        )

        assert agent_run.main(argv) == 0
        state_dir = isolated_runs_root / name
        assert _wait_terminal(state_dir) == "done"
        assert (state_dir / "cwd").read_text().strip() == str(existing.resolve())
        # Attaching to a pre-existing worktree creates nothing new.
        assert not (state_dir / "worktree_created").exists()

    def test_branch_checked_out_elsewhere_is_refused_and_leaves_no_new_worktree(
        self, isolated_runs_root, isolated_log_root, tmp_path
    ):
        repo = _make_repo(tmp_path, "repo-branch-collision")
        other = tmp_path / "other-wt"
        _git(repo, "worktree", "add", "-q", str(other), "-b", "shared-branch", "main")
        new_dir = tmp_path / "new-wt"
        name = "e2e5-branch-collision"
        argv = _launch_argv(
            name=name, worktree=new_dir, worktree_base="main", worktree_repo=repo,
            worktree_branch="shared-branch", worktree_reuse=True, command=["true"],
        )

        with pytest.raises(SystemExit) as exc:
            agent_run.main(argv)
        assert _exit_status(exc.value.code) == 1

        assert not new_dir.exists()
        porcelain = _git(repo, "worktree", "list", "--porcelain")
        assert str(other.resolve()) in porcelain
        assert str(new_dir.resolve()) not in porcelain
        assert not (isolated_runs_root / name).exists()

    def test_worktree_and_cwd_together_is_a_usage_error_before_any_mutation(
        self, isolated_runs_root, isolated_log_root, tmp_path
    ):
        repo = _make_repo(tmp_path, "repo-cwd-conflict")
        wt = tmp_path / "wt-cwd-conflict"
        name = "e2e5-cwd-conflict"
        argv = [
            "--cwd", str(tmp_path),
            "--worktree", str(wt), "--worktree-base", "main",
            "--worktree-repo", str(repo),
            name, "--", "true",
        ]

        with pytest.raises(SystemExit) as exc:
            agent_run.main(argv)
        assert exc.value.code == 2

        assert not wt.exists()
        assert not (isolated_runs_root / name).exists()


# ---------------------------------------------------------------------------
# E2E6: du and reap account for a real worktree
# ---------------------------------------------------------------------------

@live_only
class TestE2E6DuAndReapAccountForRealWorktree:
    """`du` charges a real linked worktree's bytes; `reap --include-worktrees`
    predicts and then really removes it once the run is terminal, and never
    while the run is still live."""

    def test_du_charges_the_linked_worktree_bytes(
        self, isolated_runs_root, isolated_log_root, tmp_path, capsys
    ):
        repo = _make_repo(tmp_path, "repo-du")
        wt = tmp_path / "wt-du"
        name = "e2e6-du"
        argv = _launch_argv(
            name=name, worktree=wt, worktree_base="main", worktree_repo=repo,
            command=["true"],
        )
        assert agent_run.main(argv) == 0
        state_dir = isolated_runs_root / name
        assert _wait_terminal(state_dir) == "done"

        # A real file of known size inside the worktree, beyond what git
        # itself puts there, so the byte count is unambiguous.
        payload = wt / "payload.bin"
        payload.write_bytes(b"x" * 65536)

        capsys.readouterr()
        rc = agent_run.cmd_du(argparse.Namespace(by_run=True, top=None, bytes=False, json=True))
        assert rc == 0
        out = json.loads(capsys.readouterr().out)
        run_row = next(r for r in out["runs"] if r["name"] == name)
        assert run_row["worktree_bytes"] >= 65536

    def test_reap_include_worktrees_dry_run_predicts_then_real_run_removes(
        self, isolated_runs_root, isolated_log_root, tmp_path, capsys
    ):
        repo = _make_repo(tmp_path, "repo-reap")
        wt = tmp_path / "wt-reap"
        name = "e2e6-reap"
        argv = _launch_argv(
            name=name, worktree=wt, worktree_base="main", worktree_repo=repo,
            command=["true"],
        )
        assert agent_run.main(argv) == 0
        state_dir = isolated_runs_root / name
        assert _wait_terminal(state_dir) == "done"

        reap_kwargs = dict(
            dry_run=True, idle_hours=None, min_age_hours=None, log_min_age_hours=None,
            name=None, force_unknown=False, include_logs=False, orphan_processes=False,
            orphan_min_age_hours=None, max_seconds=None, include_worktrees=True,
            worktree_min_age_hours=0.0, force_dirty=False, all=False,
        )
        capsys.readouterr()
        rc = agent_run.cmd_reap(argparse.Namespace(**reap_kwargs))
        assert rc == 0
        dry_out = capsys.readouterr().out
        assert "worktrees_removed=1 worktrees_skipped=0" in dry_out
        assert wt.is_dir(), "dry-run must not remove anything"

        real_kwargs = dict(reap_kwargs, dry_run=False)
        rc = agent_run.cmd_reap(argparse.Namespace(**real_kwargs))
        assert rc == 0
        real_out = capsys.readouterr().out
        assert "worktrees_removed=1 worktrees_skipped=0" in real_out

        assert not wt.exists()
        porcelain = _git(repo, "worktree", "list", "--porcelain")
        assert str(wt.resolve()) not in porcelain

    def test_reap_include_worktrees_never_removes_a_still_live_run(
        self, isolated_runs_root, isolated_log_root, tmp_path, capsys
    ):
        repo = _make_repo(tmp_path, "repo-reap-live")
        wt = tmp_path / "wt-reap-live"
        name = "e2e6-reap-live"
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
        agent_pid: Optional[int] = None
        try:
            launcher.wait(timeout=15)
            state_dir = isolated_runs_root / name
            _wait_for(
                lambda: (state_dir / "status").exists()
                and (state_dir / "status").read_text().strip() == "running",
                10.0, "status=running",
            )
            agent_pid = int((state_dir / "agent_pid").read_text().strip())
            assert agent_run._pid_alive(agent_pid)

            reap_kwargs = dict(
                dry_run=False, idle_hours=None, min_age_hours=None, log_min_age_hours=None,
                name=None, force_unknown=False, include_logs=False, orphan_processes=False,
                orphan_min_age_hours=None, max_seconds=None, include_worktrees=True,
                worktree_min_age_hours=0.0, force_dirty=False, all=False,
            )
            capsys.readouterr()
            rc = agent_run.cmd_reap(argparse.Namespace(**reap_kwargs))
            assert rc == 0
            out = capsys.readouterr().out
            assert "worktrees_removed=0" in out

            assert wt.is_dir(), "a still-live run's worktree must never be reaped"
            porcelain = _git(repo, "worktree", "list", "--porcelain")
            assert str(wt.resolve()) in porcelain
        finally:
            _kill_and_reap(agent_pid)
            launcher.poll()
            if launcher.returncode is None:
                launcher.kill()
                launcher.wait(timeout=10)


# ---------------------------------------------------------------------------
# E2E7: managed-harness worktree coverage (claude, codex) plus the rollback
# contract common to every harness that acquires its session id without a
# pre-fork process.
# ---------------------------------------------------------------------------

_FAKE_CLAUDE_SCRIPT = """#!/bin/sh
: > {argv_file}
for arg in "$@"; do
    printf '%s\\n' "$arg" >> {argv_file}
done
exit 0
"""


@live_only
class TestE2E7ClaudeWorktreeLaunch:
    """A real `agent-run --harness claude --worktree ...` launch through the
    real CLI, a real git worktree, and a real forked runner -- only the
    `claude` binary itself is substituted, with a real shell script that
    records its own argv. This proves two things through one real launch:
    the worktree/branch/cwd machinery a raw launch already exercises (E2E1)
    behaves identically in managed mode, and agent-run's push-acquired
    UUID4 session id actually reaches the harness's argv as --session-id
    and is the same id recorded in session.json.
    """

    def test_worktree_created_and_session_id_pushed_via_argv(
        self, isolated_runs_root, isolated_log_root, tmp_path, monkeypatch
    ):
        repo = _make_repo(tmp_path, "repo-claude-wt")
        base_oid = _git(repo, "rev-parse", "main").strip()
        wt = tmp_path / "wt-claude"
        name = "e2e7-claude-happy"

        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        argv_capture = tmp_path / "claude-argv.txt"
        fake_claude = bin_dir / "claude"
        fake_claude.write_text(_FAKE_CLAUDE_SCRIPT.format(argv_file=argv_capture))
        fake_claude.chmod(0o755)
        monkeypatch.setenv("PATH", str(bin_dir) + os.pathsep + os.environ.get("PATH", ""))

        argv = [
            "--harness", "claude", "--prompt", "hi",
            "--worktree", str(wt), "--worktree-base", base_oid,
            "--worktree-repo", str(repo),
            name,
        ]
        rc = agent_run.main(argv)
        assert rc == 0

        state_dir = isolated_runs_root / name
        status = _wait_terminal(state_dir)
        assert status == "done"

        # The same worktree/branch/cwd contract E2E1 checks for a raw launch.
        porcelain = _git(repo, "worktree", "list", "--porcelain")
        assert str(wt.resolve()) in porcelain
        assert _git(wt, "rev-parse", "HEAD").strip() == base_oid
        assert name in _git(repo, "branch", "--list", name)
        assert (state_dir / "cwd").read_text().strip() == str(wt.resolve())

        # Proof agent-run passed --session-id: the fake's own recorded argv.
        recorded_argv = argv_capture.read_text().splitlines()
        assert "--session-id" in recorded_argv, f"argv missing --session-id: {recorded_argv!r}"
        session_id_flag_idx = recorded_argv.index("--session-id")
        pushed_session_id = recorded_argv[session_id_flag_idx + 1]
        parsed = uuid.UUID(pushed_session_id)
        assert parsed.version == 4, f"session id is not a UUID4: {pushed_session_id!r}"

        session = agent_run._read_session_json(isolated_log_root / name)
        assert session is not None
        assert session["harness"] == "claude"
        assert session["acquisition"] == "pushed"
        assert session["confidence"] == "certain"
        assert session["session_id"] == pushed_session_id


def _make_fake_codex_appserver(bin_dir: Path, *, thread_id: str, cwd_file: Path) -> None:
    """Write a fake `codex` speaking the minimum JSON-RPC subset
    `_run_managed_oneshot_codex_appserver` and `_CodexAppServer.mint_thread`
    drive: initialize -> initialized (notification, no response) ->
    thread/start -> turn/start, then one agentMessage delta and a completed
    turn. Frame shapes match the real app-server response shapes read by
    `_CodexAppServer.mint_thread` (``result.thread.id``) and the one-shot
    runner's frame loop (``item/agentMessage/delta``, ``turn/completed``).
    cwd is recorded before the handshake so the test can prove app-server
    was started with the worktree as its cwd, the same fact
    `_opencode_prefork_mint`'s Popen(cwd=...) establishes for opencode.
    """
    fake = bin_dir / "codex"
    fake.write_text(
        "#!/usr/bin/env python3\n"
        "import sys, os, json\n"
        "def send(obj):\n"
        "    sys.stdout.write(json.dumps(obj) + '\\n')\n"
        "    sys.stdout.flush()\n"
        "def recv():\n"
        "    line = sys.stdin.readline()\n"
        "    if not line:\n"
        "        sys.exit(0)\n"
        "    return json.loads(line)\n"
        f"with open({str(cwd_file)!r}, 'w') as f:\n"
        "    f.write(os.getcwd())\n"
        "if len(sys.argv) < 2 or sys.argv[1] != 'app-server':\n"
        "    sys.exit(1)\n"
        f"THREAD_ID = {thread_id!r}\n"
        "msg = recv()  # initialize\n"
        "send({'jsonrpc': '2.0', 'id': msg['id'], 'result': {'userAgent': 'codex_exec/e2e7'}})\n"
        "recv()  # initialized notification -- no response\n"
        "msg = recv()  # thread/start\n"
        "send({'jsonrpc': '2.0', 'id': msg['id'],\n"
        "      'result': {'thread': {'id': THREAD_ID, 'path': '~/.codex/sessions/e2e7.jsonl'}}})\n"
        "msg = recv()  # turn/start\n"
        "send({'jsonrpc': '2.0', 'id': msg['id'],\n"
        "      'result': {'turn': {'id': 'turn-1', 'status': 'inProgress'}}})\n"
        "send({'jsonrpc': '2.0', 'method': 'item/agentMessage/delta',\n"
        "      'params': {'delta': 'worktree answer'}})\n"
        "send({'jsonrpc': '2.0', 'method': 'turn/completed',\n"
        "      'params': {'turn': {'id': 'turn-1', 'status': 'completed'}}})\n"
        "sys.exit(0)\n"
    )
    fake.chmod(0o755)


@live_only
class TestE2E7CodexWorktreeLaunch:
    """A real `agent-run --harness codex --worktree ...` launch. Codex never
    execs an argv -- both modes drive `codex app-server` over JSON-RPC
    (agent_run.py:9415), so the fake substitutes the app-server process
    itself rather than a harness argv, and speaks just enough of the
    protocol for `_CodexAppServer.mint_thread` and the one-shot turn loop
    to complete.
    """

    def test_worktree_created_and_thread_id_lands_in_session_json(
        self, isolated_runs_root, isolated_log_root, tmp_path, monkeypatch
    ):
        repo = _make_repo(tmp_path, "repo-codex-wt")
        base_oid = _git(repo, "rev-parse", "main").strip()
        wt = tmp_path / "wt-codex"
        name = "e2e7-codex-happy"

        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        cwd_file = tmp_path / "codex-cwd.txt"
        thread_id = "e2e7-thread-happy"
        _make_fake_codex_appserver(bin_dir, thread_id=thread_id, cwd_file=cwd_file)
        monkeypatch.setenv("PATH", str(bin_dir) + os.pathsep + os.environ.get("PATH", ""))

        argv = [
            "--harness", "codex", "--prompt", "hi",
            "--worktree", str(wt), "--worktree-base", base_oid,
            "--worktree-repo", str(repo),
            name,
        ]
        rc = agent_run.main(argv)
        assert rc == 0

        state_dir = isolated_runs_root / name
        status = _wait_terminal(state_dir)
        assert status == "done"

        # The same worktree/branch/cwd contract E2E1 checks for a raw launch.
        porcelain = _git(repo, "worktree", "list", "--porcelain")
        assert str(wt.resolve()) in porcelain
        assert _git(wt, "rev-parse", "HEAD").strip() == base_oid
        assert name in _git(repo, "branch", "--list", name)
        assert (state_dir / "cwd").read_text().strip() == str(wt.resolve())

        # The app-server process itself ran with the worktree as its cwd --
        # mint_thread happens post-fork inside the runner, so this is the
        # process-tree analogue of opencode's Popen(cwd=...) prefork mint.
        assert cwd_file.read_text().strip() == str(wt.resolve())

        session = agent_run._read_session_json(isolated_log_root / name)
        assert session is not None
        assert session["harness"] == "codex"
        assert session["acquisition"] == "minted"
        assert session["confidence"] == "certain"
        assert session["session_id"] == thread_id


@live_only
class TestE2E7RollbackAppliesToEveryHarness:
    """The rollback contract E2E2 proves for a raw run applies identically
    to every managed harness: a preflight failure reached after worktree
    creation but before ``os.fork()`` removes the worktree and branch this
    invocation created, and never prints the leaked-worktree warning that
    only applies once a runner process exists. The trigger -- a missing
    ``-f/--prompt-file`` -- is rejected inside ``_cmd_launch_locked`` before
    any harness-specific session acquisition runs (agent_run.py:9200-9202,
    ahead of the managed-mode block starting at 9351), so the same failure
    point exercises all three harnesses without needing to fake any of
    their external binaries.
    """

    @pytest.mark.parametrize("harness", ["claude", "codex", "opencode"])
    def test_preflight_failure_after_worktree_creation_removes_worktree_and_branch(
        self, isolated_runs_root, isolated_log_root, tmp_path, capsys, harness
    ):
        repo = _make_repo(tmp_path, f"repo-rollback-{harness}")
        wt = tmp_path / f"wt-rollback-{harness}"
        name = f"e2e7-rollback-{harness}"
        missing_prompt = tmp_path / "no-such-prompt.txt"
        argv = [
            "--harness", harness, "-f", str(missing_prompt),
            "--worktree", str(wt), "--worktree-base", "main",
            "--worktree-repo", str(repo),
            name,
        ]

        with pytest.raises(SystemExit) as exc:
            agent_run.main(argv)
        assert "prompt file not found" in str(exc.value)

        assert not wt.exists()
        porcelain = _git(repo, "worktree", "list", "--porcelain")
        assert str(wt.resolve()) not in porcelain
        assert name not in _git(repo, "branch", "--list")
        assert not (isolated_runs_root / name).exists()

        stderr = capsys.readouterr().err
        assert "leaving worktree in place" not in stderr, (
            f"a preflight failure must never print the post-fork leak warning; got: {stderr!r}"
        )


@live_only
class TestE2E7NoPreforkProcessForClaudeOrCodex:
    """The audit claim under test: for claude and codex, nothing creates a
    process in the worktree before the runner fork, so
    ``args._worktree_process_started`` stays False through a failure
    reached after session acquisition -- unlike opencode, whose prefork
    mint sets the mark as the literal last statement before its ``Popen``
    call (agent_run.py:11215-11224), regardless of whether that Popen ever
    produces a usable session.

    One failure point -- ``_persist_submit_mode`` raising -- is injected
    strictly after every harness's session-acquisition block (claude's
    push, codex's argv stub, opencode's real prefork mint against a fake
    `opencode` binary) and strictly before the unconditional mark every
    harness sets right before ``os.fork()`` (agent_run.py:9478). Only a
    harness's own acquisition path can set the mark before that point, so
    a difference between harnesses here isolates exactly the claim in
    question rather than an artifact of where the failure was injected.
    """

    @pytest.mark.parametrize("harness", ["claude", "codex", "opencode"])
    def test_prefork_mark_reflects_only_the_harnesss_own_acquisition_path(
        self, isolated_runs_root, isolated_log_root, tmp_path, monkeypatch, harness
    ):
        repo = _make_repo(tmp_path, f"repo-premark-{harness}")
        wt = tmp_path / f"wt-premark-{harness}"
        name = f"e2e7-premark-{harness}"

        # opencode's real prefork mint runs against this fake binary; claude
        # and codex never invoke an external process before the fork, so
        # its presence on PATH is inert for them.
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        fake_opencode = bin_dir / "opencode"
        fake_opencode.write_text("#!/bin/sh\ntrap 'exit 0' TERM\nsleep 60\n")
        fake_opencode.chmod(0o755)
        monkeypatch.setenv("PATH", str(bin_dir) + os.pathsep + os.environ.get("PATH", ""))
        # Short-circuits the up-to-30s health poll: the mark is already set
        # (it precedes Popen unconditionally), so a fast, deterministic
        # "never healthy" outcome is all this test needs from the mint.
        monkeypatch.setattr(agent_run, "_opencode_health_poll", lambda *a, **k: False)

        def _boom(*_a, **_k):
            raise RuntimeError("forced failure between acquisition and fork")
        monkeypatch.setattr(agent_run, "_persist_submit_mode", _boom)

        args = argparse.Namespace(
            name=name, command=[], interactive=False,
            prompt_file=None, submit_mode=None, idle_timeout=None,
            harness=harness, prompt="hi", model=None, agent_mode=None,
            harness_args=[], permissions="bypass", cwd=None,
            worktree=str(wt), worktree_base="main", worktree_branch=None,
            worktree_repo=str(repo), worktree_reuse=False,
            enable_planning=False, enable_questions=False,
        )

        try:
            with pytest.raises(RuntimeError, match="forced failure"):
                agent_run.cmd_launch(args)

            mark = getattr(args, "_worktree_process_started", None)
            if harness == "opencode":
                # Non-vacuity: the claude/codex branch's own assertion
                # (mark is False) is wrong here -- opencode's prefork mint
                # genuinely ran and set the mark, so asserting False must
                # itself fail. The inversion is explicit, not a silent if.
                with pytest.raises(AssertionError):
                    assert mark is False
                assert mark is True, "opencode's prefork mint must have set the mark"
                assert wt.is_dir(), "a set mark must refuse rollback"
                assert name in _git(repo, "branch", "--list", name)
            else:
                assert mark is False, f"{harness} must never set the pre-fork mark"
                assert not wt.exists(), f"{harness} preflight failure must roll back"
                porcelain = _git(repo, "worktree", "list", "--porcelain")
                assert str(wt.resolve()) not in porcelain
                assert name not in _git(repo, "branch", "--list")
        finally:
            if harness == "opencode" and wt.is_dir():
                _git(repo, "worktree", "remove", "--force", str(wt))
                _git(repo, "branch", "-D", name)
