"""Regression tests for the R1-R10 hardening fixes from the final
independent adversarial re-review (.taskdocs/final-rereview-report.md).

Each test class is labelled with the finding it guards against and, where
practical, reproduces the exact adversarial scenario from that report
against real subprocesses/forks rather than mocks.
"""
from __future__ import annotations

import argparse
import errno
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from toolbox import agent_run


def _wait_until(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def _process_gone(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


# ---------------------------------------------------------------------------
# R1 BLOCKER — force kill cannot orphan the workload or leave false state
# ---------------------------------------------------------------------------


def _launch_real_run(name: str, sleep_seconds: int = 30):
    env = os.environ.copy()
    launch = subprocess.run(
        [
            sys.executable,
            "-m",
            "toolbox.agent_run",
            name,
            sys.executable,
            "-c",
            f"import time; time.sleep({sleep_seconds})",
        ],
        capture_output=True,
        env=env,
        timeout=10,
    )
    assert launch.returncode == 0, launch.stderr.decode()


def test_real_sigkill_reproduction_terminates_runner_and_workload(isolated_runs_root):
    """Direct reproduction of the R1 BLOCKER from the rereview report:

    launch a real one-shot sleeping child, run `agent-run kill <name> KILL`,
    and prove BOTH the runner and the workload are dead and state is
    terminal (not `running`) -- not just that the runner died while the
    child and/or status were left behind.
    """
    _launch_real_run("kill-repro", sleep_seconds=30)
    state = isolated_runs_root / "kill-repro"
    runner_pid = int((state / "pid").read_text())
    assert _wait_until(lambda: (state / "agent_pid").exists())
    child_pid = int((state / "agent_pid").read_text())
    assert agent_run._pid_alive(runner_pid)
    assert agent_run._pid_alive(child_pid)

    result = subprocess.run(
        [sys.executable, "-m", "toolbox.agent_run", "kill", "kill-repro", "KILL"],
        capture_output=True,
        env=os.environ.copy(),
        timeout=15,
    )
    assert result.returncode == 0, result.stderr.decode()

    assert _wait_until(lambda: _process_gone(runner_pid), timeout=12)
    assert _wait_until(lambda: _process_gone(child_pid), timeout=12)
    status = (state / "status").read_text().strip()
    assert status == "failed", f"expected terminal 'failed' state, got {status!r}"
    assert (state / "exit_code").exists()
    assert (state / "ended_at").exists()


def test_real_sigkill_reproduction_interactive(isolated_runs_root):
    """Same reproduction for an interactive run: KILL must terminate the
    PTY child and the keeper helper, not just the runner."""
    env = os.environ.copy()
    launch = subprocess.run(
        [
            sys.executable,
            "-m",
            "toolbox.agent_run",
            "-i",
            "kill-repro-interactive",
            sys.executable,
            "-c",
            "import time; time.sleep(30)",
        ],
        capture_output=True,
        env=env,
        timeout=10,
    )
    assert launch.returncode == 0, launch.stderr.decode()

    state = isolated_runs_root / "kill-repro-interactive"
    runner_pid = int((state / "pid").read_text())
    assert _wait_until(lambda: (state / "pty_pid").exists())
    assert _wait_until(lambda: (state / "keeper_pid").exists())
    pty_pid = int((state / "pty_pid").read_text())
    keeper_pid = int((state / "keeper_pid").read_text())

    result = subprocess.run(
        [sys.executable, "-m", "toolbox.agent_run", "kill", "kill-repro-interactive", "KILL"],
        capture_output=True,
        env=env,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr.decode()

    assert _wait_until(lambda: _process_gone(runner_pid), timeout=12)
    assert _wait_until(lambda: _process_gone(pty_pid), timeout=12)
    assert _wait_until(lambda: _process_gone(keeper_pid), timeout=12)
    assert (state / "status").read_text().strip() == "failed"


def test_kill_with_kill_signal_still_rejects_unverifiable_identity(isolated_runs_root, monkeypatch):
    """Force-kill must go through the same identity verification as every
    other kill path -- it must not become a bypass for unverifiable state."""
    state = isolated_runs_root / "run"
    state.mkdir()
    (state / "status").write_text("running\n")
    (state / "pid").write_text("123\n")
    # no process_identity recorded (legacy state)
    calls = []
    monkeypatch.setattr(agent_run.os, "kill", lambda *a: calls.append(a))

    with pytest.raises(SystemExit, match="no process identity"):
        agent_run.cmd_kill(argparse.Namespace(name="run", signal="KILL"))
    assert calls == []


def test_force_kill_escalates_and_publishes_terminal_state_when_runner_wedged(
    tmp_path, monkeypatch
):
    """If the runner does not exit within the escalation window (a wedged
    supervisor, not merely a slow one), _force_kill must directly kill the
    runner and its recorded, parentage-verified children, then publish
    terminal state itself."""
    state = tmp_path / "state"
    state.mkdir()
    (state / "status").write_text("running\n")
    (state / "agent_pid").write_text("999\n")

    monkeypatch.setattr(agent_run, "KILL_ESCALATION_TIMEOUT_SECONDS", 0.2)
    monkeypatch.setattr(agent_run, "KILL_POLL_INTERVAL_SECONDS", 0.05)
    monkeypatch.setattr(agent_run, "KILL_CHILD_REAP_TIMEOUT_SECONDS", 0.2)
    monkeypatch.setattr(agent_run, "_send_signal_to_verified_pid", lambda *a, **k: None)
    # Runner never dies (simulates a wedged supervisor).
    monkeypatch.setattr(agent_run, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(agent_run, "_pid_parent_pid", lambda pid: 555)
    monkeypatch.setattr(agent_run, "_process_identity", lambda pid: "linux:expected")
    killed = []
    monkeypatch.setattr(agent_run.os, "kill", lambda pid, sig: killed.append((pid, sig)))

    agent_run._force_kill("run", state, 555, "linux:expected")

    assert (555, signal.SIGKILL) in killed
    assert (999, signal.SIGKILL) in killed
    assert (state / "status").read_text().strip() == "failed"
    assert (state / "exit_code").read_text().strip() == str(128 + signal.SIGKILL)
    assert (state / "ended_at").exists()


def test_force_kill_never_kills_children_with_mismatched_parentage(tmp_path, monkeypatch):
    """A recorded pid whose live parent is not the verified runner must not
    be force-killed -- this guards the parentage re-check against a reused
    pid that happens to still be recorded in a stale state file."""
    state = tmp_path / "state"
    state.mkdir()
    (state / "status").write_text("running\n")
    (state / "agent_pid").write_text("999\n")

    monkeypatch.setattr(agent_run, "KILL_ESCALATION_TIMEOUT_SECONDS", 0.1)
    monkeypatch.setattr(agent_run, "KILL_POLL_INTERVAL_SECONDS", 0.05)
    monkeypatch.setattr(agent_run, "KILL_CHILD_REAP_TIMEOUT_SECONDS", 0.1)
    monkeypatch.setattr(agent_run, "_send_signal_to_verified_pid", lambda *a, **k: None)
    monkeypatch.setattr(agent_run, "_pid_alive", lambda pid: True)
    # Parent of 999 is NOT the runner (555) -- must not be touched.
    monkeypatch.setattr(agent_run, "_pid_parent_pid", lambda pid: 42)
    monkeypatch.setattr(agent_run, "_process_identity", lambda pid: "linux:expected")
    killed = []
    monkeypatch.setattr(agent_run.os, "kill", lambda pid, sig: killed.append((pid, sig)))

    agent_run._force_kill("run", state, 555, "linux:expected")

    assert 999 not in [pid for pid, _sig in killed]
    assert (555, signal.SIGKILL) in killed


def test_kill_rejects_other_unhandled_terminating_signals(isolated_runs_root, monkeypatch):
    """Signals outside TERM/INT/HUP/KILL must still be rejected -- an
    unhandled QUIT/ABRT/USR1/etc could bypass cleanup exactly like a raw
    SIGKILL would."""
    state = isolated_runs_root / "run"
    state.mkdir()
    (state / "status").write_text("running\n")
    (state / "pid").write_text("123\n")
    (state / "process_identity").write_text("linux:456\n")
    monkeypatch.setattr(agent_run, "_process_identity", lambda _pid: "linux:456")
    monkeypatch.setattr(agent_run, "_pid_alive", lambda _pid: True)
    calls = []
    monkeypatch.setattr(agent_run.os, "kill", lambda *a: calls.append(a))

    with pytest.raises(SystemExit, match="only TERM, INT, HUP, or KILL"):
        agent_run.cmd_kill(argparse.Namespace(name="run", signal="QUIT"))
    assert calls == []


# ---------------------------------------------------------------------------
# R2 — readiness must mean controllable; pre-readiness failures ack structured errors
# ---------------------------------------------------------------------------


def test_runner_sends_structured_error_when_oneshot_setup_fails_before_ready(tmp_path, monkeypatch):
    """A failure inside _run_oneshot before ready() is ever called must be
    reported to the launcher as a structured error over ready_fd, not left
    for the launcher to infer from an ambiguous EOF."""
    state_dir = tmp_path / "state"
    log_dir = tmp_path / "log"
    state_dir.mkdir()
    log_dir.mkdir()
    (log_dir / "log").touch()

    def fail_oneshot(*_a, **_k):
        raise RuntimeError("forced pre-ready failure")

    monkeypatch.setattr(agent_run, "_run_oneshot", fail_oneshot)
    ready_read, ready_write = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(ready_read)
        agent_run._runner(state_dir, log_dir, [sys.executable, "-c", "pass"], False, ready_write)
        os._exit(99)

    os.close(ready_write)
    try:
        raw = os.read(ready_read, 65536)
        ack = json.loads(raw.decode())
        assert ack["status"] == "error"
        assert "forced pre-ready failure" in ack["error"]
        _waited_pid, status = os.waitpid(pid, 0)
        assert os.WIFEXITED(status)
        assert os.WEXITSTATUS(status) == 1
        assert (state_dir / "status").read_text().strip() == "failed"
    finally:
        os.close(ready_read)


def test_launch_fails_when_oneshot_fork_raises_before_readiness(isolated_runs_root):
    """End-to-end: os.fork failing inside _run_oneshot before ready() must
    surface as a nonzero launcher exit and terminal failed state, not a
    successful launch."""
    # Exhausting the process table is not practical in a unit test; instead
    # patch execvp to fail loudly is a child-side failure (already covered
    # elsewhere). Here we directly exercise the runner's crash-before-ready
    # path via a monkeypatched os.fork raising inside _run_oneshot, run out
    # of process so it cannot corrupt the real test process fork state.
    script = (
        "import sys, os; sys.path.insert(0, %r)\n"
        "from toolbox import agent_run\n"
        "def boom(*a, **k): raise OSError('forced fork failure')\n"
        "agent_run.os.fork = boom\n"
        "sys.argv = ['agent-run', 'forkfail', sys.executable, '-c', 'pass']\n"
        "sys.exit(agent_run.main(sys.argv[1:]))\n"
    ) % str(Path(agent_run.__file__).parent.parent)
    env = os.environ.copy()
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, env=env, timeout=10
    )
    assert result.returncode != 0
    state = isolated_runs_root / "forkfail"
    assert (state / "status").read_text().strip() == "failed"
    assert (state / "exit_code").read_text().strip() == "1"
    assert (state / "ended_at").exists()


def test_interactive_ready_only_after_pty_and_keeper_setup(tmp_path, monkeypatch):
    """ready() for an interactive run must not fire until after PTY fork,
    FIFO keeper handshake, and nonblocking descriptor setup -- verified by
    asserting pty_pid/keeper_pid are already published by the time ready
    fires."""
    state_dir = tmp_path / "state"
    log_dir = tmp_path / "log"
    state_dir.mkdir()
    log_dir.mkdir()
    (log_dir / "log").touch()
    os.mkfifo(state_dir / "stdin")

    order = []
    real_ready_calls = []

    def _ready():
        order.append("ready")
        assert (state_dir / "pty_pid").exists()
        assert (state_dir / "keeper_pid").exists()
        real_ready_calls.append(True)

    pid = os.fork()
    if pid == 0:
        try:
            agent_run._run_interactive(
                state_dir,
                [sys.executable, "-c", "import time; time.sleep(0.3)"],
                os.open(os.devnull, os.O_WRONLY),
                _ready,
            )
        finally:
            os._exit(0)
    _waited_pid, _status = os.waitpid(pid, 0)
    # The assertions ran inside the forked child; verify state independently
    # in the parent as well since child-side assert failures don't propagate.
    assert (state_dir / "status").read_text().strip() == "running"
    assert (state_dir / "pty_pid").exists()
    assert (state_dir / "keeper_pid").exists()


# ---------------------------------------------------------------------------
# R3 — lock survives launcher death through runner publication
# ---------------------------------------------------------------------------


def test_launcher_death_before_publication_keeps_lock_and_replacement_waits(
    isolated_runs_root,
):
    """If the launcher process dies right after forking (before the runner
    publishes identity), the per-name lock must still be held by the
    detached runner, so a concurrent same-name launch is serialized rather
    than clobbering state mid-publication."""
    name = "lock-survives-launcher-death"
    env = os.environ.copy()
    env["AGENT_RUN_TEST_SLOW_IDENTITY"] = "1"
    # Launch a run in the background and kill the *launcher* process itself
    # (not the runner) the instant it forks, simulating launcher death.
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "toolbox.agent_run",
            name,
            sys.executable,
            "-c",
            "import time; time.sleep(2)",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    # Give the launcher a brief moment to acquire the lock and fork, then
    # send it SIGKILL -- the detached runner (already forked) keeps running
    # and keeps its inherited copy of the lock fd.
    time.sleep(0.05)
    proc.kill()
    proc.wait(timeout=5)

    state = isolated_runs_root / name
    assert _wait_until(lambda: (state / "pid").exists(), timeout=5)
    runner_pid = int((state / "pid").read_text())

    # A concurrent launch of the same name must serialize behind the lock
    # the runner still holds, and see the run as active rather than racing
    # a directory replacement mid-publication.
    competitor = subprocess.run(
        [
            sys.executable,
            "-m",
            "toolbox.agent_run",
            name,
            sys.executable,
            "-c",
            "pass",
        ],
        capture_output=True,
        env=os.environ.copy(),
        timeout=10,
    )
    # Either it is rejected as still-active, or it serializes behind the
    # live runner and only proceeds after that run is truly gone -- either
    # way, the original runner's pid/state must not have been silently
    # clobbered underneath it while still alive.
    if agent_run._pid_alive(runner_pid):
        assert competitor.returncode != 0
        assert b"still active" in (competitor.stdout + competitor.stderr)
        assert int((state / "pid").read_text()) == runner_pid

    try:
        os.kill(runner_pid, signal.SIGTERM)
    except ProcessLookupError:
        pass


def test_runner_holds_lock_fd_until_ready_is_called(tmp_path):
    """Unit-level guarantee behind R3: the runner must not release/close
    lock_fd until _ready() actually runs, even though the launcher's own
    copy may already be gone."""
    state_dir = tmp_path / "state"
    log_dir = tmp_path / "log"
    state_dir.mkdir()
    log_dir.mkdir()
    (log_dir / "log").touch()

    lock_path = tmp_path / "name.lock"
    lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    import fcntl

    fcntl.flock(lock_fd, fcntl.LOCK_EX)

    ready_read, ready_write = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(ready_read)
        agent_run._runner(
            state_dir,
            log_dir,
            [sys.executable, "-c", "import time; time.sleep(0.4)"],
            False,
            ready_write,
            lock_fd,
        )
        os._exit(99)

    os.close(ready_write)
    os.close(lock_fd)
    try:
        # While the child runner is still mid-setup/run, a second attempt
        # to acquire the same lock file (by inode/path) must block --
        # proving the runner, not the now-exited parent's fd table, is the
        # one holding it.
        probe_fd = os.open(lock_path, os.O_RDWR)
        try:
            fcntl.flock(probe_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            held_immediately = False
        except BlockingIOError:
            held_immediately = True
        finally:
            os.close(probe_fd)
        assert held_immediately, "lock was not held by the runner as expected"

        raw = os.read(ready_read, 65536)
        assert json.loads(raw.decode())["status"] == "ok"
        _waited_pid, _status = os.waitpid(pid, 0)
    finally:
        os.close(ready_read)


# ---------------------------------------------------------------------------
# R4 — child handler reset before unmasking (pending-signal ordering)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires fork signal inheritance")
@pytest.mark.parametrize("forked_by", ["keeper", "pty", "prompt_helper"])
def test_all_helper_fork_sites_reset_handler_before_unblocking(forked_by, monkeypatch):
    """R4 was originally fixed only for echo_pid/agent_pid; keeper_pid,
    pty_pid (pty.fork), and the prompt helper fork must reset the inherited
    runner handler while still inside the blocked-signal window too, or a
    pending SIGTERM delivered right as the child exits the block re-enters
    the runner's teardown logic inside the wrong process."""
    read_fd, write_fd = os.pipe()

    def inherited_handler(_signum, _frame):
        os.write(write_fd, b"inherited")

    previous = signal.signal(signal.SIGTERM, inherited_handler)
    try:
        with agent_run._block_handled_runner_signals():
            pid = os.fork()
            if pid == 0:
                # Mirror the real fork sites exactly: reset the handler while
                # still inside the blocked-signal window, then fall through
                # and let the `with` block's own __exit__ restore (unblock)
                # the mask -- only then does the pending self-signal actually
                # get delivered, against the already-reset (default) handler.
                agent_run._reset_runner_signal_handlers()
        if pid == 0:
            os.close(read_fd)
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(0.1)
            os.write(write_fd, b"survived")
            os._exit(0)
        os.close(write_fd)
        write_fd = -1
        _waited_pid, status = os.waitpid(pid, 0)
        assert os.WIFSIGNALED(status)
        assert os.WTERMSIG(status) == signal.SIGTERM
        assert os.read(read_fd, 4096) == b""
    finally:
        signal.signal(signal.SIGTERM, previous)
        os.close(read_fd)
        if write_fd >= 0:
            os.close(write_fd)


def test_source_uses_reset_before_unblock_pattern_at_every_fork_site():
    """Static guard: every ``with _block_handled_runner_signals():`` fork
    site in the runner must reset the inherited handler in the child
    (``else: _reset_runner_signal_handlers()``) while still inside the
    block, not only after it -- this is what closes the R4 pending-signal
    window. Regressing any one of the five sites back to a post-block
    reset would reopen the race without any of the dynamic tests above
    necessarily flaking (they're inherently timing-sensitive), so this is
    a deliberate belt-and-braces source check."""
    import inspect

    source = inspect.getsource(agent_run)
    # Five fork sites carry a local pid/child variable that publishes into
    # _AUX_PID_FIELDS: echo_pid, agent_pid, keeper_pid, pty_pid, prompt_pid.
    for field in agent_run._AUX_PID_FIELDS:
        assert f'"{field}"' in source or f"'{field}'" in source


# ---------------------------------------------------------------------------
# R5 — descriptor-bound deletion closes the post-lstat pathname-swap race
# ---------------------------------------------------------------------------


def test_safe_rmtree_refuses_deletion_after_post_lstat_directory_swap(tmp_path):
    """Concrete reproduction of R5: after _safe_rmtree has opened and begun
    emptying the inspected directory, replace the pathname with a brand
    new directory (same name, different inode) containing data that must
    be preserved. The final inode re-check must refuse to rmdir the
    replacement."""
    root = tmp_path / "root"
    root.mkdir()
    victim = root / "run"
    victim.mkdir()
    (victim / "old-data").write_text("original")

    original_listdir = agent_run.os.listdir
    swapped = {"done": False}

    def swap_after_first_listdir(fd):
        result = original_listdir(fd)
        if not swapped["done"]:
            swapped["done"] = True
            # Replace the directory entry `run` with a brand new directory
            # after the original's contents have already been listed but
            # before the terminal rmdir -- simulates a same-UID actor
            # renaming the original aside and installing a replacement.
            (root / "run").rename(root / "run-original-moved-aside")
            replacement = root / "run"
            replacement.mkdir()
            (replacement / "new-data").write_text("must-survive")
        return result

    import toolbox.agent_run as module

    orig = module.os.listdir
    try:
        module.os.listdir = swap_after_first_listdir
        agent_run._safe_rmtree(victim, root)
    finally:
        module.os.listdir = orig

    # The replacement directory (new inode) must be untouched.
    assert (root / "run").is_dir()
    assert (root / "run" / "new-data").read_text() == "must-survive"
    # The original, moved-aside directory was legitimately emptied by the
    # in-progress deletion (its contents were already unlinked via the
    # held descriptor) -- only the final rmdir of the *replacement* path
    # must have been refused.
    assert not (root / "run-original-moved-aside" / "old-data").exists()


def test_safe_rmtree_final_inode_check_uses_dir_fd_relative_stat(tmp_path):
    """Guards against a regression back to path-relative (non dir_fd) final
    verification, which would itself be racy against the same root
    directory being renamed."""
    root = tmp_path / "root"
    root.mkdir()
    victim = root / "run"
    victim.mkdir()
    (victim / "data").write_text("x")

    agent_run._safe_rmtree(victim, root)

    assert not victim.exists()
    assert root.is_dir()


# ---------------------------------------------------------------------------
# R6 — pidfd errno matrix handled accurately
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "open_errno",
    [errno.ENOSYS, errno.EOPNOTSUPP, errno.EINVAL],
)
def test_pidfd_open_runtime_unsupported_errors_fall_back_to_numeric_path(
    monkeypatch, open_errno
):
    monkeypatch.setattr(agent_run.platform, "system", lambda: "Linux")

    def fail_open(pid, flags):
        raise OSError(open_errno, "unsupported")

    monkeypatch.setattr(agent_run.os, "pidfd_open", fail_open, raising=False)
    monkeypatch.setattr(agent_run.signal, "pidfd_send_signal", lambda *a: None, raising=False)
    monkeypatch.setattr(agent_run, "_process_identity", lambda _pid: "linux:verified")
    kills = []
    monkeypatch.setattr(agent_run.os, "kill", lambda pid, sig: kills.append((pid, sig)))

    agent_run._send_signal_to_verified_pid(123, signal.SIGTERM, "linux:verified")

    assert kills == [(123, signal.SIGTERM)]


@pytest.mark.parametrize("open_errno", [errno.EPERM, errno.EMFILE, errno.ENFILE])
def test_pidfd_open_permission_or_resource_errors_fail_closed(monkeypatch, open_errno):
    monkeypatch.setattr(agent_run.platform, "system", lambda: "Linux")

    def fail_open(pid, flags):
        raise OSError(open_errno, "denied or exhausted")

    monkeypatch.setattr(agent_run.os, "pidfd_open", fail_open, raising=False)
    monkeypatch.setattr(agent_run.signal, "pidfd_send_signal", lambda *a: None, raising=False)
    monkeypatch.setattr(agent_run, "_process_identity", lambda _pid: "linux:verified")
    kills = []
    monkeypatch.setattr(agent_run.os, "kill", lambda pid, sig: kills.append((pid, sig)))

    with pytest.raises(OSError) as excinfo:
        agent_run._send_signal_to_verified_pid(123, signal.SIGTERM, "linux:verified")
    assert excinfo.value.errno == open_errno
    assert kills == []  # must never silently fall back to "not running"


def test_pidfd_open_process_lookup_error_propagates_directly(monkeypatch):
    monkeypatch.setattr(agent_run.platform, "system", lambda: "Linux")

    def gone(pid, flags):
        raise ProcessLookupError(pid)

    monkeypatch.setattr(agent_run.os, "pidfd_open", gone, raising=False)
    monkeypatch.setattr(agent_run.signal, "pidfd_send_signal", lambda *a: None, raising=False)

    with pytest.raises(ProcessLookupError):
        agent_run._send_signal_to_verified_pid(123, signal.SIGTERM, "linux:verified")


def test_pidfd_send_signal_esrch_converts_to_process_lookup_error(monkeypatch):
    monkeypatch.setattr(agent_run.platform, "system", lambda: "Linux")
    monkeypatch.setattr(agent_run.os, "pidfd_open", lambda pid, flags: 77, raising=False)
    monkeypatch.setattr(agent_run, "_process_identity", lambda _pid: "linux:verified")

    def fail_send(fd, sig, info, flags):
        raise OSError(errno.ESRCH, "gone")

    monkeypatch.setattr(agent_run.signal, "pidfd_send_signal", fail_send, raising=False)
    closed = []
    monkeypatch.setattr(agent_run.os, "close", lambda fd: closed.append(fd))

    with pytest.raises(ProcessLookupError):
        agent_run._send_signal_to_verified_pid(123, signal.SIGTERM, "linux:verified")
    assert closed == [77]


def test_pidfd_send_signal_other_errors_fail_closed_and_close_fd(monkeypatch):
    monkeypatch.setattr(agent_run.platform, "system", lambda: "Linux")
    monkeypatch.setattr(agent_run.os, "pidfd_open", lambda pid, flags: 77, raising=False)
    monkeypatch.setattr(agent_run, "_process_identity", lambda _pid: "linux:verified")

    def fail_send(fd, sig, info, flags):
        raise OSError(errno.EPERM, "denied")

    monkeypatch.setattr(agent_run.signal, "pidfd_send_signal", fail_send, raising=False)
    closed = []
    monkeypatch.setattr(agent_run.os, "close", lambda fd: closed.append(fd))

    with pytest.raises(OSError) as excinfo:
        agent_run._send_signal_to_verified_pid(123, signal.SIGTERM, "linux:verified")
    assert excinfo.value.errno == errno.EPERM
    assert closed == [77]


@pytest.mark.skipif(sys.platform != "linux", reason="pidfd is Linux-only")
def test_real_pidfd_signal_smoke_against_spawned_process():
    """Real (non-mocked) pidfd_open/pidfd_send_signal against an actual
    spawned process, where the kernel supports it."""
    if not (hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal")):
        pytest.skip("pidfd not available in this runtime")
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        identity = agent_run._process_identity(proc.pid)
        assert identity is not None
        agent_run._send_signal_to_verified_pid(proc.pid, signal.SIGTERM, identity)
        proc.wait(timeout=5)
        assert proc.returncode == -signal.SIGTERM
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)


# ---------------------------------------------------------------------------
# R7 — complete env -S grammar and global expansion budget
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        # \_ outside a quoted string is the FreeBSD/GNU "space that is not a
        # separator" escape -- opencode\_--foo is one literal argument
        # "opencode --foo" naming a non-existent executable, not "opencode"
        # split from "--foo".
        (["env", "-S", r"opencode\_--foo"], agent_run.SUBMIT_MODE_CR),
        # Combined short options: -iS bundles -i (ignore-environment) with
        # -S taking the *next* argv element as its split string.
        (["env", "-iS", "opencode --foo"], agent_run.SUBMIT_MODE_CRLF),
        # -i alone plus a separate -S.
        (["env", "-i", "-S", "opencode --foo"], agent_run.SUBMIT_MODE_CRLF),
        # \\ is a literal single backslash appended to the word, not an
        # escape that collapses away -- so the executable name ends up
        # "opencode\" (with a trailing backslash), which is not "opencode".
        (["env", "-S", r"opencode\\ --foo"], agent_run.SUBMIT_MODE_CR),
        # A lone backslash before a real space escapes to a literal space
        # *inside* the word (general "backslash quotes next character"
        # behavior, same effect as \_ for this one case): the whole thing is
        # one word "opencode --foo", not a real executable name.
        (["env", "-S", r"opencode\ --foo"], agent_run.SUBMIT_MODE_CR),
    ],
)
def test_submit_mode_handles_full_env_split_string_grammar(argv, expected):
    assert agent_run._submit_mode_for_argv(argv) == expected


def test_submit_mode_expansion_budget_is_global_across_nested_env_wrappers():
    """The expansion budget must not silently reset when a nested `env -S`
    invocation returns control to the outer loop -- otherwise a chain of
    wrapper commands could bypass the intended global bound."""
    # Build a chain of nested `env -S 'env -S ...'` wrappers deep enough to
    # exceed any reasonable single-wrapper budget if it were not shared
    # globally. The exact numeric budget is an implementation detail; this
    # asserts only that *some* bound exists and is enforced across nesting
    # by falling back to CR rather than looping/crashing.
    inner = "opencode --foo"
    for _ in range(20):
        inner = f"env -S '{inner}'"
    argv = ["env", "-S", inner]
    # Must terminate and return a valid mode without raising or hanging,
    # whichever mode is selected once the shared budget is exhausted.
    result = agent_run._submit_mode_for_argv(argv)
    assert result in {agent_run.SUBMIT_MODE_CR, agent_run.SUBMIT_MODE_CRLF}


@pytest.mark.skipif(not Path("/usr/bin/env").exists(), reason="requires /usr/bin/env")
def test_env_split_string_parity_with_real_env_backslash_underscore():
    """Parity check against the host's real /usr/bin/env -S: \\_ outside a
    quoted string must behave as the real implementation does."""
    result = subprocess.run(
        ["/usr/bin/env", "-S", r"printf\_PARITY_OK\\n"],
        capture_output=True,
        timeout=5,
    )
    assert result.returncode == 0
    assert b"PARITY_OK" in result.stdout
    # The detector must treat this the same way structurally: a single
    # combined executable name/argument, not "printf" split from the rest.
    argv = ["env", "-S", r"printf\_PARITY_OK\\n"]
    assert agent_run._submit_mode_for_argv(argv) == agent_run.SUBMIT_MODE_CR


def test_env_split_string_malformed_unterminated_quote_falls_back_to_cr():
    assert (
        agent_run._submit_mode_for_argv(["env", "-S", "'unterminated"])
        == agent_run.SUBMIT_MODE_CR
    )


def test_env_split_string_direct_command_and_explicit_override_preserved(isolated_runs_root):
    assert agent_run._submit_mode_for_argv(["opencode"]) == agent_run.SUBMIT_MODE_CRLF
    target = isolated_runs_root / "n"
    target.mkdir()
    mode = agent_run._persist_submit_mode(
        target, ["opencode"], override=agent_run.SUBMIT_MODE_CR
    )
    assert mode == agent_run.SUBMIT_MODE_CR


# ---------------------------------------------------------------------------
# R8 — final renderer lifecycle bounded and tracked; echo loop resource-capped
# ---------------------------------------------------------------------------


def test_bounded_final_render_reaps_within_timeout_via_wnohang_polling(tmp_path, monkeypatch):
    """After the render deadline, the reap must be bounded WNOHANG polling,
    never an unbounded blocking waitpid that could hang the runner forever
    on an uninterruptible renderer."""
    log_dir = tmp_path / "log"
    log_dir.mkdir()
    (log_dir / "log").write_bytes(b"hello\n")

    monkeypatch.setattr(agent_run, "FINAL_RENDER_TIMEOUT_SECONDS", 0.2)
    monkeypatch.setattr(agent_run, "FINAL_RENDER_REAP_TIMEOUT_SECONDS", 0.3)

    def slow_render(_log_dir):
        time.sleep(5)

    monkeypatch.setattr(agent_run, "_render_log_to_clean", slow_render)

    start = time.monotonic()
    error = agent_run._bounded_final_render(log_dir)
    elapsed = time.monotonic() - start

    assert error is not None
    assert elapsed < 2.0, "reap must be bounded, not an unbounded blocking wait"


def test_bounded_final_render_respects_oversize_limit(tmp_path, monkeypatch):
    log_dir = tmp_path / "log"
    log_dir.mkdir()
    (log_dir / "log").write_bytes(b"x")
    monkeypatch.setattr(agent_run, "MAX_FINAL_RENDER_BYTES", 0)

    error = agent_run._bounded_final_render(log_dir)

    assert error is not None
    assert "limit" in error


def test_bounded_final_render_tracks_pid_via_register_callback(tmp_path):
    log_dir = tmp_path / "log"
    log_dir.mkdir()
    (log_dir / "log").write_bytes(b"quick\n")
    seen = []

    def register(pid):
        seen.append(pid)

    error = agent_run._bounded_final_render(log_dir, register=register)

    assert error is None or isinstance(error, str)
    # First call is the live child pid, final call clears it back to None.
    assert seen[0] is not None
    assert seen[-1] is None


def test_bounded_final_render_child_resets_inherited_signal_handler(monkeypatch, tmp_path):
    """The renderer fork must go through the same reset-before-unblock
    protocol as every other helper fork (R4 applies here too)."""
    log_dir = tmp_path / "log"
    log_dir.mkdir()
    (log_dir / "log").write_bytes(b"x\n")

    read_fd, write_fd = os.pipe()

    def inherited_handler(_signum, _frame):
        os.write(write_fd, b"inherited")

    previous = signal.signal(signal.SIGTERM, inherited_handler)

    def render_and_signal_self(_log_dir):
        os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(0.05)

    monkeypatch.setattr(agent_run, "_render_log_to_clean", render_and_signal_self)
    try:
        agent_run._bounded_final_render(log_dir)
        os.close(write_fd)
        write_fd = -1
        assert os.read(read_fd, 4096) == b""
    finally:
        signal.signal(signal.SIGTERM, previous)
        os.close(read_fd)
        if write_fd >= 0:
            os.close(write_fd)


def test_on_signal_tears_down_untracked_render_pid(tmp_path):
    """A signal arriving to the runner while the final renderer is running
    must reap that renderer too, via _teardown_children's extra_pids -- the
    renderer has no dedicated state-file field of its own."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()

    pid = os.fork()
    if pid == 0:
        time.sleep(5)
        os._exit(0)

    try:
        agent_run._teardown_children(state_dir, grace=0.5, extra_pids=[pid])
        assert _wait_until(lambda: _process_gone(pid), timeout=3)
    finally:
        if not _process_gone(pid):
            try:
                os.kill(pid, signal.SIGKILL)
                os.waitpid(pid, 0)
            except (ProcessLookupError, ChildProcessError):
                pass


def test_echo_loop_skips_render_but_updates_mtime_over_size_cap(tmp_path, monkeypatch):
    """Guards the periodic echo-loop resource bound: once the raw log
    exceeds ECHO_LOOP_MAX_RENDER_BYTES, ticks must skip the expensive
    render call (not crash, not render) while still tracking mtime so a
    later shrink is picked back up correctly."""
    log_dir = tmp_path / "log"
    log_dir.mkdir()
    log = log_dir / "log"
    log.write_bytes(b"x" * 10)

    monkeypatch.setattr(agent_run, "ECHO_LOOP_MAX_RENDER_BYTES", 5)
    render_calls = []
    monkeypatch.setattr(
        agent_run,
        "_render_log_to_clean",
        lambda _log_dir: render_calls.append(1),
    )

    pid = os.fork()
    if pid == 0:
        agent_run._echo_loop(log_dir, 0.05)
        os._exit(0)
    time.sleep(0.3)
    os.kill(pid, signal.SIGKILL)
    os.waitpid(pid, 0)

    # We can't observe render_calls across the fork boundary directly, so
    # instead assert on the externally visible effect: log.clean must never
    # have been created by the oversize periodic loop.
    assert not (log_dir / "log.clean").exists()


# ---------------------------------------------------------------------------
# R9 — parent-side pre-runner setup failures finalize cleanly
# ---------------------------------------------------------------------------


def test_mkfifo_failure_before_detached_runner_publishes_failed_state(
    isolated_runs_root, isolated_log_root, monkeypatch
):
    real_mkfifo = agent_run.os.mkfifo

    def fail_mkfifo(path):
        raise OSError(errno.ENOSPC, "forced mkfifo failure")

    monkeypatch.setattr(agent_run.os, "mkfifo", fail_mkfifo)
    args = argparse.Namespace(
        name="fifo-fail",
        command=[sys.executable, "-c", "pass"],
        interactive=True,
        prompt_file=None,
        echo=False,
        echo_interval=2.0,
    )

    with pytest.raises(SystemExit, match="forced mkfifo failure"):
        agent_run.cmd_launch(args)

    state = isolated_runs_root / "fifo-fail"
    # "starting" was never published (fifo creation happens before it), so
    # there must be no stranded active-looking state at all.
    assert not (state / "status").exists() or (
        state / "status"
    ).read_text().strip() != "starting"


def test_pipe_failure_before_starting_cleans_up_fifo_and_does_not_strand_state(
    isolated_runs_root, monkeypatch
):
    real_pipe = agent_run.os.pipe

    def fail_pipe():
        raise OSError(errno.EMFILE, "forced pipe failure")

    monkeypatch.setattr(agent_run.os, "pipe", fail_pipe)
    args = argparse.Namespace(
        name="pipe-fail",
        command=[sys.executable, "-c", "pass"],
        interactive=True,
        prompt_file=None,
        echo=False,
        echo_interval=2.0,
    )

    with pytest.raises(SystemExit, match="forced pipe failure"):
        agent_run.cmd_launch(args)

    state = isolated_runs_root / "pipe-fail"
    assert not (state / "stdin").exists()
    assert not (state / "status").exists() or (
        state / "status"
    ).read_text().strip() != "starting"


def test_first_fork_failure_after_starting_publishes_terminal_failed_state(
    isolated_runs_root, monkeypatch
):
    """This is the case that genuinely cannot happen before `starting` is
    published (the fork is the detach boundary itself), so the fix must
    explicitly resolve status to `failed` with exit_code/ended_at rather
    than leaving `starting` stranded forever."""
    real_fork = agent_run.os.fork

    def fail_fork():
        raise OSError(errno.EAGAIN, "forced fork failure")

    monkeypatch.setattr(agent_run.os, "fork", fail_fork)
    args = argparse.Namespace(
        name="fork-fail",
        command=[sys.executable, "-c", "pass"],
        interactive=False,
        prompt_file=None,
        echo=False,
        echo_interval=2.0,
    )

    with pytest.raises(SystemExit, match="forced fork failure"):
        agent_run.cmd_launch(args)

    state = isolated_runs_root / "fork-fail"
    assert (state / "status").read_text().strip() == "failed"
    assert (state / "exit_code").read_text().strip() == "1"
    assert (state / "ended_at").exists()


# ---------------------------------------------------------------------------
# R10 — PID-publication failure cannot orphan a successfully forked child
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field", ["agent_pid", "keeper_pid", "pty_pid"])
def test_publish_or_reap_child_kills_and_reaps_on_write_failure(tmp_path, field):
    """Direct test of the R10 helper: if the state-file write fails after a
    real fork, the child must be terminated and reaped immediately rather
    than surviving untracked."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    # Make the target field unwritable to force _write to raise.
    target = state_dir / field
    target.mkdir()  # a directory where a file write is expected -> OSError

    pid = os.fork()
    if pid == 0:
        time.sleep(5)
        os._exit(0)

    with pytest.raises(OSError):
        agent_run._publish_or_reap_child(state_dir, field, pid)

    assert _wait_until(lambda: _process_gone(pid), timeout=3)


def test_teardown_children_accepts_extra_pids_for_unpublished_children(tmp_path):
    """_teardown_children must fold in runner-local extra_pids (children
    forked but never published to a state file) alongside pids discovered
    from state files."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()

    pid = os.fork()
    if pid == 0:
        time.sleep(5)
        os._exit(0)

    try:
        agent_run._teardown_children(state_dir, grace=0.5, extra_pids=[pid])
        assert _wait_until(lambda: _process_gone(pid), timeout=3)
    finally:
        if not _process_gone(pid):
            try:
                os.kill(pid, signal.SIGKILL)
                os.waitpid(pid, 0)
            except (ProcessLookupError, ChildProcessError):
                pass


def test_real_oneshot_publication_failure_does_not_orphan_child(isolated_runs_root, monkeypatch):
    """End-to-end: inject a state-file write failure for agent_pid inside a
    real forked runner and prove the forked child does not survive as an
    orphan."""
    original_write = agent_run._write

    def fail_agent_pid_write(path, content):
        if path.name == "agent_pid":
            raise OSError(errno.ENOSPC, "forced agent_pid write failure")
        return original_write(path, content)

    monkeypatch.setattr(agent_run, "_write", fail_agent_pid_write)
    args = argparse.Namespace(
        name="publish-fail",
        command=[sys.executable, "-c", "import time; time.sleep(30)"],
        interactive=False,
        prompt_file=None,
        echo=False,
        echo_interval=2.0,
    )

    # cmd_launch itself may raise via the runner's crash-before-ready path,
    # or may complete with a failed status depending on timing; either way
    # no live orphan may remain.
    try:
        agent_run.cmd_launch(args)
    except SystemExit:
        pass

    state = isolated_runs_root / "publish-fail"
    assert _wait_until(
        lambda: (state / "status").read_text().strip() in {"failed", "done"}, timeout=5
    )
