"""Focused tests for agent-run's per-agent interactive submission mode."""
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
from toolbox import agent_run_transcript


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["opencode"], agent_run.SUBMIT_MODE_CRLF),
        (["/opt/tools/opencode"], agent_run.SUBMIT_MODE_CRLF),
        (["env", "KEY=value", "opencode"], agent_run.SUBMIT_MODE_CRLF),
        (["/usr/bin/env", "-i", "KEY=value", "--", "/opt/opencode"], agent_run.SUBMIT_MODE_CRLF),
        (["command", "opencode"], agent_run.SUBMIT_MODE_CRLF),
        (["exec", "/opt/opencode"], agent_run.SUBMIT_MODE_CRLF),
        (["claude", "fix opencode integration"], agent_run.SUBMIT_MODE_CR),
        (["codex", "opencode"], agent_run.SUBMIT_MODE_CR),
        (["python", "opencode"], agent_run.SUBMIT_MODE_CR),
        (["env", "--chdir=/tmp", "opencode"], agent_run.SUBMIT_MODE_CRLF),
        (["env", "--chdir", "/tmp", "opencode"], agent_run.SUBMIT_MODE_CRLF),
        (["env", "-C/tmp", "opencode"], agent_run.SUBMIT_MODE_CRLF),
        (["env", "-u", "TOKEN", "opencode"], agent_run.SUBMIT_MODE_CRLF),
        (["env", "--unset=TOKEN", "opencode"], agent_run.SUBMIT_MODE_CRLF),
        (["env", "-S", "opencode --foo"], agent_run.SUBMIT_MODE_CRLF),
        (["env", "-S", "KEY=value opencode --foo"], agent_run.SUBMIT_MODE_CRLF),
        (["env", "-S", "-i opencode --foo"], agent_run.SUBMIT_MODE_CRLF),
        (["env", "-S", "-u TOKEN opencode --foo"], agent_run.SUBMIT_MODE_CRLF),
        (["env", "-S", "env -S 'KEY=value opencode --foo'"], agent_run.SUBMIT_MODE_CRLF),
        (["env", "--split-string", "opencode --foo"], agent_run.SUBMIT_MODE_CRLF),
        (["env", "--split-string=opencode --foo"], agent_run.SUBMIT_MODE_CRLF),
        (["env", "-Sopencode --foo"], agent_run.SUBMIT_MODE_CRLF),
        (["env", "-S", "'opencode' --foo"], agent_run.SUBMIT_MODE_CRLF),
        (["env", "-S", "claude opencode"], agent_run.SUBMIT_MODE_CR),
        (["env", "-S", "'unterminated"], agent_run.SUBMIT_MODE_CR),
        (["env", "-P", "name", "opencode"], agent_run.SUBMIT_MODE_CRLF),
        (["env", "--argv0=name", "opencode"], agent_run.SUBMIT_MODE_CRLF),
    ],
)
def test_submit_mode_for_argv_only_inspects_unambiguous_executable(argv, expected):
    assert agent_run._submit_mode_for_argv(argv) == expected


def test_persist_submit_mode_is_symbolic_and_uses_authoritative_argv(tmp_path):
    mode = agent_run._persist_submit_mode(tmp_path, ["/usr/local/bin/opencode", "mention"])

    assert mode == agent_run.SUBMIT_MODE_CRLF
    assert (tmp_path / "submit_mode").read_text() == "crlf\n"


def test_persist_submit_mode_honors_explicit_override(tmp_path):
    mode = agent_run._persist_submit_mode(
        tmp_path, ["opencode"], override=agent_run.SUBMIT_MODE_CR
    )

    assert mode == agent_run.SUBMIT_MODE_CR
    assert (tmp_path / "submit_mode").read_text() == "cr\n"


def test_missing_submit_mode_derives_legacy_default_from_argv(tmp_path):
    (tmp_path / "argv").write_text(json.dumps(["env", "--chdir=/tmp", "opencode"]))

    assert agent_run._submit_mode_from_state(tmp_path) == agent_run.SUBMIT_MODE_CRLF


def test_main_accepts_explicit_submit_mode_override(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        agent_run,
        "cmd_launch",
        lambda args: captured.update(vars(args)) or 0,
    )

    assert agent_run.main(["--submit-mode=cr", "run", "opencode"]) == 0
    assert captured["submit_mode"] == agent_run.SUBMIT_MODE_CR
    assert captured["command"] == ["opencode"]


@pytest.mark.parametrize("mode", ["", "lf", "auto"])
def test_main_rejects_invalid_submit_mode(mode):
    with pytest.raises(SystemExit, match="must be cr or crlf"):
        agent_run.main([f"--submit-mode={mode}", "run", "opencode"])


def _seed_live_interactive_run(root: Path, name: str, mode: str) -> tuple[Path, int]:
    state = root / name
    state.mkdir()
    (state / "interactive").write_text("1\n")
    (state / "pid").write_text("12345\n")
    (state / "submit_mode").write_text(mode + "\n")
    fifo = state / "stdin"
    os.mkfifo(fifo)
    return fifo, os.open(fifo, os.O_RDWR | os.O_NONBLOCK)


def _steer_args(name: str, *, esc: bool = False, raw: bool = False) -> argparse.Namespace:
    return argparse.Namespace(name=name, message=["hello"], esc=esc, raw=raw)


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (agent_run.SUBMIT_MODE_CR, b"hello\r"),
        (agent_run.SUBMIT_MODE_CRLF, b"hello\r\n"),
    ],
)
def test_steer_uses_persisted_submit_mode(
    isolated_runs_root, monkeypatch, mode, expected
):
    _fifo, reader = _seed_live_interactive_run(isolated_runs_root, "run", mode)
    monkeypatch.setattr(agent_run, "_pid_alive", lambda _pid: True)
    monkeypatch.setattr(agent_run.signal, "alarm", lambda _seconds: None)
    _witness_sequence(monkeypatch, [0, 1])
    try:
        assert agent_run.cmd_steer(_steer_args("run")) == 0
        assert os.read(reader, 4096) == expected
    finally:
        os.close(reader)


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (agent_run.SUBMIT_MODE_CR, b"\x1bhello\r"),
        (agent_run.SUBMIT_MODE_CRLF, b"\x1bhello\r\n"),
    ],
)
def test_steer_esc_reuses_persisted_submit_mode(
    isolated_runs_root, monkeypatch, mode, expected
):
    _fifo, reader = _seed_live_interactive_run(isolated_runs_root, "run", mode)
    monkeypatch.setattr(agent_run, "_pid_alive", lambda _pid: True)
    monkeypatch.setattr(agent_run.signal, "alarm", lambda _seconds: None)
    monkeypatch.setattr(agent_run.time, "sleep", lambda _seconds: None)
    _witness_sequence(monkeypatch, [0, 1])
    try:
        assert agent_run.cmd_steer(_steer_args("run", esc=True)) == 0
        assert os.read(reader, 4096) == expected
    finally:
        os.close(reader)


def test_steer_raw_is_verbatim_even_for_opencode_and_esc(isolated_runs_root, monkeypatch):
    _fifo, reader = _seed_live_interactive_run(
        isolated_runs_root, "run", agent_run.SUBMIT_MODE_CRLF
    )
    monkeypatch.setattr(agent_run, "_pid_alive", lambda _pid: True)
    monkeypatch.setattr(agent_run.signal, "alarm", lambda _seconds: None)
    try:
        assert agent_run.cmd_steer(_steer_args("run", esc=True, raw=True)) == 0
        assert os.read(reader, 4096) == b"hello"
    finally:
        os.close(reader)


def test_steer_unverified_exits_nonzero_and_writes_no_marker(
    isolated_runs_root, monkeypatch
):
    """An unreadable witness makes steer report failure, and the message text
    itself reaches the FIFO exactly once (the retry is the bare terminator)."""
    _fifo, reader = _seed_live_interactive_run(
        isolated_runs_root, "run", agent_run.SUBMIT_MODE_CR
    )
    monkeypatch.setattr(agent_run, "_pid_alive", lambda _pid: True)
    monkeypatch.setattr(agent_run.signal, "alarm", lambda _seconds: None)
    monkeypatch.setattr(agent_run, "SUBMISSION_RETRY_BACKOFF_SECONDS", 0.0)
    _witness_unreadable(monkeypatch)
    monkeypatch.setenv("AGENT_RUN_SUBMIT_VERIFY_TIMEOUT", "0.05")
    try:
        assert agent_run.cmd_steer(_steer_args("run")) == 1
        written = os.read(reader, 4096)
    finally:
        os.close(reader)
    assert written == b"hello\r\r"
    assert written.count(b"hello") == 1


def test_raw_interactive_launch_still_stamps_prompt_submitted(
    isolated_runs_root, isolated_log_root, tmp_path
):
    """An unmanaged launch stamps delivery after a successful prompt write."""
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("hello from a raw run\n")
    name = "raw-prompt-marker"
    args = argparse.Namespace(
        name=name,
        command=[sys.executable, "-c", "import sys; sys.stdin.read(1); import time; time.sleep(30)"],
        interactive=True,
        prompt_file=str(prompt),
    )

    assert agent_run.cmd_launch(args) == 0

    state = isolated_runs_root / name
    try:
        assert not (isolated_log_root / name / "session.json").exists()
        assert _wait_until(lambda: (state / "prompt_submitted").exists(), timeout=20.0), (
            "a raw run's prompt must still be marked submitted"
        )
        assert not (state / "prompt_unverified").exists()
    finally:
        pid_path = state / "pid"
        if pid_path.exists():
            try:
                os.kill(int(pid_path.read_text()), signal.SIGTERM)
            except (ProcessLookupError, ValueError):
                pass


def test_steer_on_a_raw_run_reports_unwitnessed_and_exits_zero(
    isolated_runs_root, monkeypatch
):
    """A raw run has never had a verifiable steer; verification must not
    start failing one."""
    _fifo, reader = _seed_live_interactive_run(
        isolated_runs_root, "run", agent_run.SUBMIT_MODE_CR
    )
    monkeypatch.setattr(agent_run, "_pid_alive", lambda _pid: True)
    monkeypatch.setattr(agent_run.signal, "alarm", lambda _seconds: None)
    try:
        rc = agent_run.cmd_steer(_steer_args("run"))
        assert rc == 0
        assert os.read(reader, 4096) == b"hello\r"
    finally:
        os.close(reader)


def test_unverified_prompt_is_not_classified_as_a_launch_failure(
    isolated_runs_root, isolated_log_root, tmp_path, monkeypatch
):
    """A run that received its prompt and later failed is a run failure. If
    prompt_unverified counted as never-submitted, an unverifiable witness
    would relabel every such failure as launch_failed."""
    fake_dir = tmp_path / "bin"
    fake_dir.mkdir()
    fake = fake_dir / "claude"
    # Exits non-zero past the launch grace window but inside the prompt
    # classification window, so only the prompt markers decide the status.
    fake.write_text("#!/bin/sh\nsleep 2\nexit 9\n")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", str(fake_dir) + ":" + os.environ.get("PATH", ""))
    monkeypatch.setattr(agent_run, "LAUNCH_GRACE_SECONDS", 0.5)
    monkeypatch.setattr(agent_run, "PROMPT_SUBMISSION_DELAY_SECONDS", 0.1)
    # Widens the classification window (grace + delay + verify budget) well
    # past the agent's exit without making the helper below wait for it.
    monkeypatch.setenv("AGENT_RUN_SUBMIT_VERIFY_TIMEOUT", "20")
    monkeypatch.setenv("AGENT_RUN_SUBMIT_ATTEMPTS", "1")
    # A managed run whose witness never verifies, resolved immediately:
    # prompt_unverified is written, prompt_submitted is not.
    monkeypatch.setattr(
        agent_run, "_submit_and_verify",
        lambda *_a, **_k: agent_run.SubmissionOutcome(
            False, 1, "keystroke", "witness_unreadable", delivered=True
        ),
    )

    prompt = tmp_path / "prompt.txt"
    prompt.write_text("do the thing\n")
    name = "unverified-not-launch-failed"
    ns = argparse.Namespace(
        name=name, command=[], interactive=True, prompt_file=str(prompt),
        submit_mode=None, idle_timeout=None,
        harness="claude", prompt=None, model=None, agent_mode=None,
        harness_args=[], permissions="bypass",
    )
    assert agent_run.cmd_launch(ns) == 0

    state = isolated_runs_root / name
    assert _wait_until(
        lambda: (state / "status").read_text().strip() in agent_run.TERMINAL_STATUSES,
        timeout=20.0,
    )
    assert (state / "prompt_unverified").exists()
    assert not (state / "prompt_submitted").exists()
    assert (state / "status").read_text().strip() == "failed", (
        "an unverified prompt means submitted-but-unconfirmed, not never-submitted"
    )
    assert agent_run.PROMPT_UNSUBMITTED_ERROR not in _read_launch_error(state)


def test_missing_prompt_marker_is_still_classified_as_a_launch_failure(
    isolated_runs_root, isolated_log_root, tmp_path, monkeypatch
):
    """The counterpart: neither marker present still means the agent never
    received its task, so the reclassification above must not swallow it."""
    fake_dir = tmp_path / "bin"
    fake_dir.mkdir()
    fake = fake_dir / "claude"
    fake.write_text("#!/bin/sh\nsleep 2\nexit 9\n")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", str(fake_dir) + ":" + os.environ.get("PATH", ""))
    monkeypatch.setattr(agent_run, "LAUNCH_GRACE_SECONDS", 0.5)
    monkeypatch.setattr(agent_run, "PROMPT_SUBMISSION_DELAY_SECONDS", 0.1)
    monkeypatch.setenv("AGENT_RUN_SUBMIT_VERIFY_TIMEOUT", "20")
    monkeypatch.setenv("AGENT_RUN_SUBMIT_ATTEMPTS", "1")
    # Neither marker is ever written: the submission never resolved at all.
    monkeypatch.setattr(agent_run, "_submit_and_verify", lambda *_a, **_k: None)

    prompt = tmp_path / "prompt.txt"
    prompt.write_text("do the thing\n")
    name = "no-marker-launch-failed"
    ns = argparse.Namespace(
        name=name, command=[], interactive=True, prompt_file=str(prompt),
        submit_mode=None, idle_timeout=None,
        harness="claude", prompt=None, model=None, agent_mode=None,
        harness_args=[], permissions="bypass",
    )
    assert agent_run.cmd_launch(ns) == 0

    state = isolated_runs_root / name
    assert _wait_until(
        lambda: (state / "status").read_text().strip() in agent_run.TERMINAL_STATUSES,
        timeout=20.0,
    )
    assert not (state / "prompt_unverified").exists()
    assert not (state / "prompt_submitted").exists()
    assert (state / "status").read_text().strip() == "launch_failed"
    assert agent_run.PROMPT_UNSUBMITTED_ERROR in _read_launch_error(state)


def _read_launch_error(state_dir: Path) -> str:
    try:
        return (state_dir / "launch_error").read_text()
    except OSError:
        return ""


def test_unexpected_interactive_select_error_marks_run_failed(
    isolated_runs_root, monkeypatch
):
    def fail_select(*_args, **_kwargs):
        raise OSError(errno.EBADF, "forced bad fd")

    monkeypatch.setattr(agent_run.select, "select", fail_select)
    args = argparse.Namespace(
        name="relay-error",
        command=[sys.executable, "-c", "import time; time.sleep(30)"],
        interactive=True,
        prompt_file=None,
    )

    assert agent_run.cmd_launch(args) == 0

    state = isolated_runs_root / "relay-error"
    assert _wait_until(lambda: (state / "status").read_text().strip() == "failed")
    assert (state / "exit_code").read_text().strip() == "1"
    log = (agent_run.LOG_ROOT / "relay-error" / "log").read_text()
    assert "forced bad fd" in log


def test_drain_pty_input_retries_partial_write_and_eagain(monkeypatch):
    payload = b"prompt bytes" + agent_run._submit_bytes(agent_run.SUBMIT_MODE_CRLF)
    delivered = bytearray()
    actions = iter([3, BlockingIOError(errno.EAGAIN, "backpressure"), 4, 999])

    def partial_write(_fd, data):
        action = next(actions)
        if isinstance(action, BaseException):
            raise action
        written = min(action, len(data))
        delivered.extend(data[:written])
        return written

    monkeypatch.setattr(agent_run.os, "write", partial_write)

    remaining = agent_run._drain_pty_input(42, payload)
    assert remaining == payload[3:]
    remaining = agent_run._drain_pty_input(42, remaining)
    assert remaining == b""
    assert bytes(delivered) == payload


@pytest.mark.parametrize("error", [errno.EIO, errno.EBADF, errno.EINVAL])
def test_drain_pty_input_drops_undeliverable_input(monkeypatch, error):
    def closed_pty(_fd, _data):
        raise OSError(error, "closed PTY")

    monkeypatch.setattr(agent_run.os, "write", closed_pty)

    assert agent_run._drain_pty_input(42, b"prompt bytes") == b""


def test_launch_fails_when_runner_setup_cannot_open_log(
    isolated_runs_root, isolated_log_root, monkeypatch
):
    log_path = isolated_log_root / "setup-failure" / "log"
    real_open = agent_run.os.open

    def fail_log_open(path, flags, mode=0o777):
        if Path(path) == log_path and flags & os.O_TRUNC:
            raise PermissionError("forced unwritable log")
        return real_open(path, flags, mode)

    monkeypatch.setattr(agent_run.os, "open", fail_log_open)
    args = argparse.Namespace(
        name="setup-failure",
        command=[sys.executable, "-c", "pass"],
        interactive=False,
        prompt_file=None,
    )

    with pytest.raises(SystemExit, match="forced unwritable log"):
        agent_run.cmd_launch(args)

    state = isolated_runs_root / "setup-failure"
    assert _wait_until(lambda: (state / "status").read_text().strip() == "failed")
    assert (state / "exit_code").read_text().strip() == "1"
    assert (state / "ended_at").is_file()


def _wait_until(predicate, timeout=5.0):
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


def test_sigterm_runner_reaps_one_shot_child(isolated_runs_root):
    env = os.environ.copy()
    launch = subprocess.run(
        [
            sys.executable,
            "-m",
            "toolbox.agent_run",
            "one-shot-term",
            sys.executable,
            "-c",
            "import time; time.sleep(30)",
        ],
        capture_output=True,
        env=env,
        timeout=10,
    )
    assert launch.returncode == 0, launch.stderr.decode()

    state = isolated_runs_root / "one-shot-term"
    runner_pid = int((state / "pid").read_text())
    identity = (state / "process_identity").read_text().strip()
    assert identity == agent_run._process_identity(runner_pid)
    assert _wait_until(lambda: (state / "agent_pid").exists())
    child_pid = int((state / "agent_pid").read_text())

    os.kill(runner_pid, signal.SIGTERM)

    assert _wait_until(lambda: (state / "status").read_text().strip() == "failed")
    assert (state / "exit_code").read_text().strip() == str(128 + signal.SIGTERM)
    assert _wait_until(lambda: _process_gone(child_pid))
    assert _wait_until(lambda: _process_gone(runner_pid))


def test_same_name_launches_are_serialized_without_state_clobber(
    isolated_runs_root, isolated_log_root
):
    """Two rapid launchers must not both replace the same run directory."""
    env = os.environ.copy()
    command = [
        sys.executable,
        "-m",
        "toolbox.agent_run",
        "same-name",
        sys.executable,
        "-c",
        "import time; time.sleep(30)",
    ]
    launches = [
        subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        for _ in range(2)
    ]
    results = [process.communicate(timeout=10) + (process.returncode,) for process in launches]

    try:
        return_codes = sorted(result[2] for result in results)
        assert return_codes == [0, 1]
        loser_output = b"".join(
            stdout + stderr for stdout, stderr, code in results if code != 0
        )
        assert b"still active" in loser_output

        state = isolated_runs_root / "same-name"
        assert (state / "status").read_text().strip() == "running"
        assert json.loads((state / "argv").read_text()) == command[4:]
        assert (isolated_runs_root / ".locks" / "same-name.lock").is_file()
        assert (isolated_log_root / "same-name" / "log").is_file()
    finally:
        pid_path = isolated_runs_root / "same-name" / "pid"
        if pid_path.exists():
            try:
                os.kill(int(pid_path.read_text()), signal.SIGTERM)
            except ProcessLookupError:
                pass


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires fork signal inheritance")
def test_forked_helper_resets_inherited_runner_signal_handler():
    """SIGTERM must use the default action in a helper fork.

    Before the fix, the inherited runner handler executes in this child and
    returns normally, so the helper writes the sentinel instead of terminating.
    """
    read_fd, write_fd = os.pipe()

    def inherited_handler(_signum, _frame):
        os.write(write_fd, b"inherited")

    previous = signal.signal(signal.SIGTERM, inherited_handler)
    try:
        pid = os.fork()
        if pid == 0:
            os.close(read_fd)
            agent_run._reset_runner_signal_handlers()
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


def test_block_handled_runner_signals_restores_exact_mask(monkeypatch):
    calls = []
    previous = {signal.SIGUSR1}
    monkeypatch.setattr(
        agent_run.signal,
        "pthread_sigmask",
        lambda how, mask: calls.append((how, set(mask))) or previous,
    )

    with agent_run._block_handled_runner_signals():
        pass

    assert calls == [
        (signal.SIG_BLOCK, set(agent_run._HANDLED_RUNNER_SIGNALS)),
        (signal.SIG_SETMASK, previous),
    ]


def test_teardown_never_signals_or_waits_for_self(tmp_path, monkeypatch):
    own_pid = os.getpid()
    (tmp_path / "keeper_pid").write_text(f"{own_pid}\n")
    (tmp_path / "pty_pid").write_text(f"{own_pid}\n")
    killed = []
    waited = []
    monkeypatch.setattr(agent_run.os, "kill", lambda pid, sig: killed.append((pid, sig)))
    monkeypatch.setattr(
        agent_run.os,
        "waitpid",
        lambda pid, options: waited.append((pid, options)) or (0, 0),
    )

    agent_run._teardown_children(tmp_path, grace=0)

    assert killed == []
    assert waited == []


def test_teardown_deduplicates_and_reaps_tracked_children(tmp_path, monkeypatch):
    child_pid = 4242
    (tmp_path / "keeper_pid").write_text(f"{child_pid}\n")
    (tmp_path / "pty_pid").write_text(f"{child_pid}\n")
    killed = []
    wait_results = iter([(0, 0), (child_pid, 0)])
    monkeypatch.setattr(agent_run.os, "getpid", lambda: 1111)
    monkeypatch.setattr(agent_run.os, "kill", lambda pid, sig: killed.append((pid, sig)))
    monkeypatch.setattr(agent_run.os, "waitpid", lambda _pid, _options: next(wait_results))

    agent_run._teardown_children(tmp_path, grace=0.1)

    assert killed == [(child_pid, signal.SIGTERM)]


# ---------------------------------------------------------------------------
# _submit_and_verify: the verified-submission witness + retry helper
# ---------------------------------------------------------------------------

def _witness_sequence(monkeypatch, counts: list) -> None:
    """Return successive witness counts, repeating the last after exhaustion."""
    calls: list = []

    def fake(state_dir, log_dir):
        def count() -> "int | None":
            calls.append((state_dir, log_dir))
            index = min(len(calls) - 1, len(counts) - 1)
            return counts[index]

        return agent_run._WitnessSource("fake", count)

    monkeypatch.setattr(agent_run, "_resolve_witness_source", fake)


def _witness_counter(monkeypatch, count: "callable") -> None:
    """Pin every _submit_and_verify call to a witness source whose
    user_count is *count*."""
    monkeypatch.setattr(
        agent_run, "_resolve_witness_source",
        lambda *_a: agent_run._WitnessSource("fake", count),
    )


def _witness_unreadable(monkeypatch) -> None:
    """Pin a witness source that exists but can never be read."""
    _witness_counter(monkeypatch, lambda: None)


def _no_transport_side_effects(monkeypatch) -> list:
    """Stub the keystroke transport as a no-op recording each attempt's
    payload, force keystroke transport (no opencode HTTP endpoint), and
    remove the inter-attempt backoff so retry tests do not sleep."""
    submissions: list = []
    monkeypatch.setattr(agent_run, "_opencode_http_endpoint", lambda *_a: None)
    monkeypatch.setattr(
        agent_run, "_submit_via_keystroke",
        lambda _state_dir, payload: submissions.append(payload),
    )
    monkeypatch.setattr(agent_run, "SUBMISSION_RETRY_BACKOFF_SECONDS", 0.0)
    return submissions


class _CoupledHarness:
    """A witness driven *by* the transport, so the count can only rise
    because a submission actually happened.

    An independent witness mock and transport mock let a success-path test
    pass on polling alone: the witness would rise on schedule whether or not
    _submit_and_verify ever called the transport. Here every accepted
    submission appends a user record, and the witness reports how many exist.

    accept_from bounds which attempts land: attempts before it raise or are
    silently swallowed, modelling a TUI that took the text and dropped the
    Enter. delivered records every payload the transport saw, landed only
    those that produced a record.
    """

    def __init__(self, monkeypatch, *, accept_from: int = 1, raise_before: bool = False):
        self.delivered: list = []
        self.landed: list = []
        self._accept_from = accept_from
        self._raise_before = raise_before
        monkeypatch.setattr(agent_run, "_opencode_http_endpoint", lambda *_a: None)
        monkeypatch.setattr(agent_run, "_submit_via_keystroke", self._submit)
        monkeypatch.setattr(
            agent_run, "_resolve_witness_source",
            lambda *_a: agent_run._WitnessSource("coupled", self.user_count),
        )
        monkeypatch.setattr(agent_run, "SUBMISSION_RETRY_BACKOFF_SECONDS", 0.0)

    def _submit(self, _state_dir, payload) -> None:
        attempt = len(self.delivered) + 1
        if attempt < self._accept_from:
            if self._raise_before:
                raise ConnectionResetError("transport failed")
            self.delivered.append(payload)  # written, but swallowed downstream
            return
        self.delivered.append(payload)
        self.landed.append(payload)

    def user_count(self) -> int:
        return len(self.landed)


def test_submit_and_verify_swallowed_input_exhausts_attempts(tmp_path, monkeypatch):
    """A flat witness exhausts max_attempts and reports a timeout."""
    submissions = _no_transport_side_effects(monkeypatch)
    _witness_sequence(monkeypatch, [0])  # baseline 0, every later read also 0

    outcome = agent_run._submit_and_verify(
        tmp_path, tmp_path, b"hello", submit_mode=agent_run.SUBMIT_MODE_CR,
        deadline_s=0.05, max_attempts=3,
    )

    assert outcome.verified is False
    assert outcome.attempts == 3
    assert outcome.transport == "keystroke"
    assert outcome.detail == "timeout"
    assert len(submissions) == 3


def test_submit_and_verify_verified_on_first_attempt(tmp_path, monkeypatch):
    """A witness that rises on the very first poll stops immediately: exactly
    one submission, verified True, attempts == 1."""
    submissions = _no_transport_side_effects(monkeypatch)
    _witness_sequence(monkeypatch, [0, 1])  # baseline 0, first poll sees 1

    outcome = agent_run._submit_and_verify(
        tmp_path, tmp_path, b"hello", submit_mode=agent_run.SUBMIT_MODE_CR,
        deadline_s=5.0, max_attempts=2,
    )

    assert outcome.verified is True
    assert outcome.attempts == 1
    assert outcome.detail is None
    assert len(submissions) == 1


def test_submit_and_verify_verified_on_retry(tmp_path, monkeypatch):
    """Flat through the entire first attempt's poll window and through the
    pre-resend re-check, then rises during the second attempt: two
    submissions, verified True, attempts == 2."""
    submissions = _no_transport_side_effects(monkeypatch)
    # baseline=0; attempt 1's poll and post-timeout re-check stay at 0, as
    # does attempt 2's pre-send re-check; attempt 2's first poll rises to 1.
    _witness_sequence(monkeypatch, [0, 0, 0, 0, 1])

    outcome = agent_run._submit_and_verify(
        tmp_path, tmp_path, b"hello", submit_mode=agent_run.SUBMIT_MODE_CR,
        deadline_s=0.05, max_attempts=2,
    )

    assert outcome.verified is True
    assert outcome.attempts == 2
    assert len(submissions) == 2


def test_submit_and_verify_late_lander_skips_second_submission(tmp_path, monkeypatch):
    """The witness rises exactly during the post-timeout re-check, not during
    any poll iteration: the outcome is verified and no second submission
    happens -- the duplicate-message race the re-check exists to close."""
    submissions = _no_transport_side_effects(monkeypatch)
    poll_calls = {"n": 0}

    def fake_witness():
        poll_calls["n"] += 1
        if poll_calls["n"] == 1:
            return 0  # baseline
        if poll_calls["n"] == 2:
            return 0  # the only poll iteration before the deadline elapses
        return 1  # the post-timeout re-check

    _witness_counter(monkeypatch, fake_witness)

    outcome = agent_run._submit_and_verify(
        tmp_path, tmp_path, b"hello", submit_mode=agent_run.SUBMIT_MODE_CR,
        deadline_s=0.05, max_attempts=3,
    )

    assert outcome.verified is True
    assert outcome.attempts == 1
    assert len(submissions) == 1


def test_submit_and_verify_witness_unreadable_never_resends_the_prompt_text(tmp_path, monkeypatch):
    """count_user_records raising TranscriptSourceError (via the real
    witness-source implementation) must degrade to verified=False with
    detail=witness_unreadable, never propagate.

    A witness that stays unreadable gives no evidence the submission failed
    to land, so nothing carrying the prompt text may be resent: that would
    risk a real duplicate prompt (messageID is not idempotent) for zero
    informational gain. The terminator-only keystroke retry is exempt --
    it carries no text and so cannot duplicate anything.
    """
    monkeypatch.setattr(agent_run, "_opencode_http_endpoint", lambda *_a: None)
    monkeypatch.setattr(
        agent_run, "_read_session_json",
        lambda _log_dir: {"session_id": "s1", "harness": "codex"},
    )

    def raise_source_error(_harness, _session_id, _cwd):
        raise agent_run_transcript.TranscriptSourceError("boom", code="store_missing")

    monkeypatch.setattr(
        agent_run._agent_run_transcript, "count_user_records", raise_source_error
    )
    submissions = []
    monkeypatch.setattr(
        agent_run, "_submit_via_keystroke",
        lambda _state_dir, payload: submissions.append(payload),
    )
    monkeypatch.setattr(agent_run, "SUBMISSION_RETRY_BACKOFF_SECONDS", 0.0)
    os.mkfifo(tmp_path / "stdin")

    outcome = agent_run._submit_and_verify(
        tmp_path, tmp_path, b"hello", submit_mode=agent_run.SUBMIT_MODE_CR,
        deadline_s=0.05, max_attempts=5,
    )

    cr = agent_run._submit_bytes(agent_run.SUBMIT_MODE_CR)
    assert outcome.verified is False
    assert outcome.detail == "witness_unreadable"
    # Attempt 1 sends the text; attempt 2 sends the terminator alone; the
    # loop stops rather than reaching attempt 3, which would resend the text.
    assert submissions == [b"hello" + cr, cr]
    assert sum(1 for payload in submissions if b"hello" in payload) == 1


def test_submit_and_verify_witness_unreadable_submits_once_for_a_texted_transport(
    tmp_path, monkeypatch
):
    """A transport with no terminator-only form (rpc here) has no
    non-duplicating retry available, so an unreadable witness stops it at
    exactly one submission."""
    monkeypatch.setattr(agent_run, "SUBMISSION_RETRY_BACKOFF_SECONDS", 0.0)
    _witness_unreadable(monkeypatch)
    calls = []

    outcome = agent_run._submit_and_verify(
        tmp_path, tmp_path, b"hello",
        deadline_s=0.05, max_attempts=5,
        submit=lambda text: calls.append(text),
    )

    assert outcome.verified is False
    assert outcome.detail == "witness_unreadable"
    assert outcome.attempts == 1
    assert len(calls) == 1


@pytest.mark.parametrize(
    ("opencode_endpoint", "expected_transport"),
    [
        ((41234, "ses_1"), "http"),
        (None, "keystroke"),
    ],
)
def test_submit_and_verify_transport_selection(
    tmp_path, monkeypatch, opencode_endpoint, expected_transport
):
    """opencode with a known port uses HTTP; anything else (claude, or
    opencode without a resolvable port/session) uses the keystroke FIFO."""
    monkeypatch.setattr(agent_run, "_opencode_http_endpoint", lambda *_a: opencode_endpoint)
    monkeypatch.setattr(agent_run, "_submit_via_http", lambda *_a, **_k: None)
    monkeypatch.setattr(agent_run, "_submit_via_keystroke", lambda *_a, **_k: None)
    _witness_sequence(monkeypatch, [0, 1])

    outcome = agent_run._submit_and_verify(
        tmp_path, tmp_path, b"hello", submit_mode=agent_run.SUBMIT_MODE_CR,
        deadline_s=5.0, max_attempts=1,
    )

    assert outcome.transport == expected_transport


# ---------------------------------------------------------------------------
# The witness predicate: one pinned source, user records only, fresh sessions
# ---------------------------------------------------------------------------

def test_witness_source_pinned_to_opencode_http_never_falls_back_to_the_store(
    tmp_path, monkeypatch
):
    """An opencode run whose HTTP endpoint stops answering must report an
    unreadable witness, not silently start reading the transcript store.

    The store counts a different thing on a different scale, so its number
    clears an HTTP baseline on sight -- which is exactly the false
    verification this pins against. The store here therefore returns a
    plausible larger count rather than raising: a raise would be indis-
    tinguishable from the unreadable-witness result under test.
    """
    monkeypatch.setattr(agent_run, "_opencode_http_endpoint", lambda *_a: (41234, "ses_1"))
    monkeypatch.setattr(agent_run, "_submit_via_http", lambda *_a, **_k: None)
    monkeypatch.setattr(
        agent_run, "_read_session_json",
        lambda _log_dir: {"session_id": "ses_1", "harness": "opencode"},
    )
    http_reads = {"n": 0}

    def flaky_http(_port, _session):
        http_reads["n"] += 1
        return 2 if http_reads["n"] == 1 else None  # baseline only, then dead

    monkeypatch.setattr(agent_run, "_opencode_http_message_count", flaky_http)
    store_reads = {"n": 0}

    def store_count(*_a, **_k):
        store_reads["n"] += 1
        return 340  # would tower over the HTTP baseline of 2

    monkeypatch.setattr(
        agent_run._agent_run_transcript, "count_user_records", store_count
    )

    outcome = agent_run._submit_and_verify(
        tmp_path, tmp_path, b"hello", deadline_s=0.05, max_attempts=1,
    )

    assert store_reads["n"] == 0, "a pinned HTTP witness must never read the store"
    assert outcome.verified is False
    assert outcome.detail == "witness_unreadable"


def test_witness_source_pinned_to_the_store_never_reaches_for_http(tmp_path, monkeypatch):
    """A source pinned to the transcript store must not start querying an
    HTTP endpoint that appeared mid-verification."""
    monkeypatch.setattr(agent_run, "_opencode_http_endpoint", lambda *_a: None)
    monkeypatch.setattr(
        agent_run, "_read_session_json",
        lambda _log_dir: {"session_id": "ses_1", "harness": "opencode"},
    )
    monkeypatch.setattr(agent_run, "_submit_via_keystroke", lambda *_a, **_k: None)
    monkeypatch.setattr(
        agent_run._agent_run_transcript, "count_user_records", lambda *_a: 4,
    )

    def http_would_verify(*_a, **_k):
        raise AssertionError("a pinned store witness must never query the HTTP endpoint")

    monkeypatch.setattr(agent_run, "_opencode_http_message_count", http_would_verify)

    outcome = agent_run._submit_and_verify(
        tmp_path, tmp_path, b"hello", deadline_s=0.05, max_attempts=1,
    )

    assert outcome.verified is False
    assert outcome.detail == "timeout"


def test_witness_counts_user_records_only_so_an_assistant_reply_cannot_verify(
    tmp_path, monkeypatch
):
    """A swallowed prompt whose session emits an assistant record must stay
    unverified: the witness reads count_user_records, not every record."""
    submissions = _no_transport_side_effects(monkeypatch)
    monkeypatch.setattr(
        agent_run, "_read_session_json",
        lambda _log_dir: {"session_id": "s1", "harness": "claude"},
    )
    # The full record count climbs from the assistant's own reply; the
    # user-record count does not move, because nothing was delivered.
    monkeypatch.setattr(
        agent_run._agent_run_transcript, "count_user_records",
        lambda *_a: 1,
    )

    outcome = agent_run._submit_and_verify(
        tmp_path, tmp_path, b"hello", submit_mode=agent_run.SUBMIT_MODE_CR,
        deadline_s=0.05, max_attempts=2,
    )

    assert outcome.verified is False
    assert outcome.detail == "timeout"
    assert len(submissions) == 2


def test_unreadable_baseline_cannot_verify_without_fresh_session(tmp_path, monkeypatch):
    """A steer runs against an established session that already holds user
    records, so an unreadable baseline followed by a readable nonzero count
    is pre-existing history, not proof this steer landed."""
    submissions = _no_transport_side_effects(monkeypatch)
    _witness_sequence(monkeypatch, [None, 7])

    outcome = agent_run._submit_and_verify(
        tmp_path, tmp_path, b"hello", submit_mode=agent_run.SUBMIT_MODE_CR,
        deadline_s=0.05, max_attempts=2, fresh_session=False,
    )

    assert outcome.verified is False
    assert outcome.detail == "witness_unreadable"
    # Attempt 2 is the terminator alone, which cannot duplicate anything;
    # the prompt text itself is sent exactly once.
    assert sum(1 for payload in submissions if b"hello" in payload) == 1


def test_unreadable_baseline_verifies_on_a_fresh_session(tmp_path, monkeypatch):
    """A launch mints its session, so it holds no records until this
    submission lands: the first readable nonzero user count is proof."""
    _no_transport_side_effects(monkeypatch)
    _witness_sequence(monkeypatch, [None, 1])

    outcome = agent_run._submit_and_verify(
        tmp_path, tmp_path, b"hello", submit_mode=agent_run.SUBMIT_MODE_CR,
        deadline_s=5.0, max_attempts=2, fresh_session=True,
    )

    assert outcome.verified is True
    assert outcome.attempts == 1


def test_resolve_witness_source_prefers_opencode_http_over_the_store(tmp_path):
    (tmp_path / "opencode_port").write_text("41234\n")
    (tmp_path / "session.json").write_text(
        json.dumps({"session_id": "ses_1", "harness": "opencode"})
    )

    source = agent_run._resolve_witness_source(tmp_path, tmp_path)

    assert source is not None
    assert source.name == "opencode_http"


def test_resolve_witness_source_names_the_harness_store_without_a_port(tmp_path):
    (tmp_path / "session.json").write_text(
        json.dumps({"session_id": "ses_1", "harness": "claude"})
    )

    source = agent_run._resolve_witness_source(tmp_path, tmp_path)

    assert source is not None
    assert source.name == "claude_transcript"


def test_resolve_witness_source_is_none_for_an_unmanaged_run(tmp_path):
    """No session.json means a raw run: there is no store to witness."""
    assert agent_run._resolve_witness_source(tmp_path, tmp_path) is None
    assert agent_run._resolve_witness_source(tmp_path, None) is None


def test_unmanaged_run_reports_unwitnessed_after_one_delivered_write(tmp_path, monkeypatch):
    """A raw run has no harness session, so there is nothing to verify. That
    is not a verification failure: report the write and do not retry."""
    submissions = _no_transport_side_effects(monkeypatch)

    outcome = agent_run._submit_and_verify(
        tmp_path, tmp_path, b"hello", submit_mode=agent_run.SUBMIT_MODE_CR,
        deadline_s=5.0, max_attempts=3,
    )

    assert outcome.verified is False
    assert outcome.detail == "unwitnessed"
    assert outcome.delivered is True
    assert len(submissions) == 1


def test_managed_run_with_a_failing_witness_is_not_reported_as_unwitnessed(
    tmp_path, monkeypatch
):
    """A managed run whose witness cannot be read must stay unverified and
    must not borrow the raw-run exemption."""
    _no_transport_side_effects(monkeypatch)
    _witness_unreadable(monkeypatch)

    outcome = agent_run._submit_and_verify(
        tmp_path, tmp_path, b"hello", submit_mode=agent_run.SUBMIT_MODE_CR,
        deadline_s=0.05, max_attempts=2,
    )

    assert outcome.verified is False
    assert outcome.detail == "witness_unreadable"


# ---------------------------------------------------------------------------
# Retry safety: attempt scoping, transport errors, keystroke framing
# ---------------------------------------------------------------------------

def test_readable_witness_flag_is_scoped_to_each_attempt(tmp_path, monkeypatch):
    """An attempt whose every read failed must stop the loop even when an
    earlier attempt read the witness fine: only evidence gathered during
    this attempt can justify one more submission."""
    submissions = _no_transport_side_effects(monkeypatch)
    reads = {"n": 0}

    def witness():
        reads["n"] += 1
        # Baseline and attempt 1 read 0; every read from attempt 2 on fails.
        return 0 if reads["n"] <= 3 else None

    _witness_counter(monkeypatch, witness)

    outcome = agent_run._submit_and_verify(
        tmp_path, tmp_path, b"hello", submit_mode=agent_run.SUBMIT_MODE_CR,
        deadline_s=0.05, max_attempts=5,
    )

    assert outcome.verified is False
    assert outcome.detail == "witness_unreadable"
    assert outcome.attempts == 2
    assert len(submissions) == 2


def test_transport_error_polls_the_witness_before_retrying(tmp_path, monkeypatch):
    """A connection that broke after the peer queued the request still
    delivers. Polling the witness after a transport error catches that and
    avoids a duplicate submission -- messageID is not idempotent."""
    calls = {"n": 0}

    def broken_then_unused(_text):
        calls["n"] += 1
        raise ConnectionResetError("reset after the request was queued")

    _witness_sequence(monkeypatch, [0, 1])
    monkeypatch.setattr(agent_run, "SUBMISSION_RETRY_BACKOFF_SECONDS", 0.0)

    outcome = agent_run._submit_and_verify(
        tmp_path, tmp_path, b"hello",
        deadline_s=5.0, max_attempts=3, submit=broken_then_unused,
    )

    assert outcome.verified is True
    assert outcome.attempts == 1
    assert calls["n"] == 1  # never resent: the witness proved it landed


def test_transport_error_still_retries_when_the_witness_stays_flat(tmp_path, monkeypatch):
    calls = {"n": 0}

    def flaky(_text):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionResetError("reset")

    _witness_sequence(monkeypatch, [0])  # flat forever
    monkeypatch.setattr(agent_run, "SUBMISSION_RETRY_BACKOFF_SECONDS", 0.0)

    outcome = agent_run._submit_and_verify(
        tmp_path, tmp_path, b"hello",
        deadline_s=0.05, max_attempts=2, submit=flaky,
    )

    assert outcome.verified is False
    assert calls["n"] == 2


def test_transport_errors_back_off_between_attempts(tmp_path, monkeypatch):
    """Without a backoff a 100ms blip burns every attempt instantly."""
    slept: list = []
    monkeypatch.setattr(agent_run.time, "sleep", lambda seconds: slept.append(seconds))
    _witness_sequence(monkeypatch, [0])

    def always_broken(_text):
        raise ConnectionResetError("reset")

    agent_run._submit_and_verify(
        tmp_path, tmp_path, b"hello",
        deadline_s=0.0, max_attempts=3, submit=always_broken,
    )

    backoffs = [s for s in slept if s >= agent_run.SUBMISSION_RETRY_BACKOFF_SECONDS]
    assert len(backoffs) == 2, f"expected one backoff before each retry, slept {slept}"
    assert all(s <= agent_run.SUBMISSION_RETRY_BACKOFF_MAX_SECONDS for s in backoffs)


def test_keystroke_retry_sends_the_terminator_alone_before_resending_text():
    """The failure a keystroke retry targets is 'TUI took the text, swallowed
    the Enter'. Attempt 2 must submit what is already buffered rather than
    doubling the prompt; only attempt 3 clears the composer and resends."""
    text = b"do the thing"
    cr = agent_run._submit_bytes(agent_run.SUBMIT_MODE_CR)

    assert agent_run._keystroke_payload(1, text, agent_run.SUBMIT_MODE_CR) == text + cr
    assert agent_run._keystroke_payload(2, text, agent_run.SUBMIT_MODE_CR) == cr
    third = agent_run._keystroke_payload(3, text, agent_run.SUBMIT_MODE_CR)
    assert third == agent_run._COMPOSER_RESET_BYTES + text + cr
    assert third.startswith(b"\x1b")


def test_keystroke_retry_payloads_reach_the_fifo_in_order(tmp_path, monkeypatch):
    submissions = _no_transport_side_effects(monkeypatch)
    _witness_sequence(monkeypatch, [0])

    agent_run._submit_and_verify(
        tmp_path, tmp_path, b"hello", submit_mode=agent_run.SUBMIT_MODE_CRLF,
        deadline_s=0.01, max_attempts=3,
    )

    crlf = agent_run._submit_bytes(agent_run.SUBMIT_MODE_CRLF)
    assert submissions == [
        b"hello" + crlf,
        crlf,
        agent_run._COMPOSER_RESET_BYTES + b"hello" + crlf,
    ]


def test_fifo_write_loops_until_every_byte_lands(monkeypatch):
    """os.write on a pipe is atomic only to PIPE_BUF; a short write past
    that would truncate the prompt or drop the terminator."""
    payload = b"a long prompt" + agent_run._submit_bytes(agent_run.SUBMIT_MODE_CR)
    delivered = bytearray()
    chunk_sizes = iter([4, 3, 999])

    def short_write(_fd, data):
        written = min(next(chunk_sizes), len(data))
        delivered.extend(data[:written])
        return written

    monkeypatch.setattr(agent_run.os, "write", short_write)

    agent_run._write_all_to_fifo(7, payload)

    assert bytes(delivered) == payload


def test_fifo_write_retries_eintr(monkeypatch):
    payload = b"prompt\r"
    delivered = bytearray()
    actions = iter([InterruptedError(), 999])

    def interrupted_write(_fd, data):
        action = next(actions)
        if isinstance(action, BaseException):
            raise action
        delivered.extend(data)
        return len(data)

    monkeypatch.setattr(agent_run.os, "write", interrupted_write)

    agent_run._write_all_to_fifo(7, payload)

    assert bytes(delivered) == payload


def test_submit_and_verify_rechecks_the_witness_immediately_before_a_resend(
    tmp_path, monkeypatch
):
    """A witness that rose during the inter-attempt gap must be seen before
    the transport is invoked again, not after another duplicate lands."""
    submissions = _no_transport_side_effects(monkeypatch)
    reads = {"n": 0}

    def witness():
        reads["n"] += 1
        # Baseline plus attempt 1's poll and post-timeout re-check read 0;
        # the pre-resend re-check is the first read to see the rise.
        return 0 if reads["n"] <= 3 else 1

    _witness_counter(monkeypatch, witness)

    outcome = agent_run._submit_and_verify(
        tmp_path, tmp_path, b"hello", submit_mode=agent_run.SUBMIT_MODE_CR,
        deadline_s=0.01, max_attempts=2,
    )

    assert outcome.verified is True
    assert len(submissions) == 1


def test_submit_and_verify_propagates_a_watchdog_timeout_error(tmp_path, monkeypatch):
    """TimeoutError subclasses OSError; swallowing it into transport_error
    would make cmd_steer's SIGALRM guard retry a stuck FIFO forever."""
    _witness_sequence(monkeypatch, [0])

    def alarm_fires(_text):
        raise TimeoutError("write timed out")

    with pytest.raises(TimeoutError):
        agent_run._submit_and_verify(
            tmp_path, tmp_path, b"hello",
            deadline_s=5.0, max_attempts=3, submit=alarm_fires,
        )


def test_submit_via_http_reports_a_socket_timeout_as_a_connection_error(monkeypatch):
    """A transport-level timeout must not surface as TimeoutError, which
    _submit_and_verify reserves for a caller's own watchdog."""
    import socket as socket_module

    def timing_out(*_a, **_k):
        raise TimeoutError("connect timed out")

    monkeypatch.setattr(socket_module, "create_connection", timing_out)

    with pytest.raises(ConnectionError):
        agent_run._submit_via_http(41234, "ses_1", "hello")


# ---------------------------------------------------------------------------
# Causally-coupled success paths: the witness rises only from a real submission
# ---------------------------------------------------------------------------

def test_coupled_first_attempt_verifies_off_its_own_submission(tmp_path, monkeypatch):
    """With the witness driven by the transport, verification can only come
    from the submission this call made -- not from polling long enough."""
    harness = _CoupledHarness(monkeypatch)

    outcome = agent_run._submit_and_verify(
        tmp_path, tmp_path, b"hello", submit_mode=agent_run.SUBMIT_MODE_CR,
        deadline_s=5.0, max_attempts=2,
    )

    assert outcome.verified is True
    assert outcome.attempts == 1
    assert harness.landed == [b"hello" + agent_run._submit_bytes(agent_run.SUBMIT_MODE_CR)]


def test_coupled_swallowed_first_attempt_verifies_off_the_retry(tmp_path, monkeypatch):
    """The TUI takes attempt 1's text and drops the Enter, so no record
    appears. Attempt 2 sends the terminator alone, which submits the buffered
    text -- and only then does the witness rise."""
    harness = _CoupledHarness(monkeypatch, accept_from=2)

    outcome = agent_run._submit_and_verify(
        tmp_path, tmp_path, b"hello", submit_mode=agent_run.SUBMIT_MODE_CR,
        deadline_s=0.05, max_attempts=2,
    )

    cr = agent_run._submit_bytes(agent_run.SUBMIT_MODE_CR)
    assert outcome.verified is True
    assert outcome.attempts == 2
    assert harness.delivered == [b"hello" + cr, cr]
    assert harness.landed == [cr], "the retry must not resend the prompt text"


def test_coupled_nothing_ever_lands_stays_unverified(tmp_path, monkeypatch):
    """The control for the two above: when no submission ever produces a
    record, no amount of polling verifies."""
    harness = _CoupledHarness(monkeypatch, accept_from=99)

    outcome = agent_run._submit_and_verify(
        tmp_path, tmp_path, b"hello", submit_mode=agent_run.SUBMIT_MODE_CR,
        deadline_s=0.05, max_attempts=3,
    )

    assert outcome.verified is False
    assert outcome.detail == "timeout"
    assert harness.landed == []
    assert len(harness.delivered) == 3


def test_coupled_transport_error_that_actually_landed_is_not_resent(tmp_path, monkeypatch):
    """A transport that raised after the peer took the request still produces
    a record. Polling the witness before retrying catches that."""
    landed: list = []
    monkeypatch.setattr(agent_run, "SUBMISSION_RETRY_BACKOFF_SECONDS", 0.0)
    monkeypatch.setattr(
        agent_run, "_resolve_witness_source",
        lambda *_a: agent_run._WitnessSource("coupled", lambda: len(landed)),
    )
    calls = {"n": 0}

    def queued_then_reset(text):
        calls["n"] += 1
        landed.append(text)  # the peer queued it...
        raise ConnectionResetError("...then the connection broke")

    outcome = agent_run._submit_and_verify(
        tmp_path, tmp_path, b"hello",
        deadline_s=5.0, max_attempts=3, submit=queued_then_reset,
    )

    assert outcome.verified is True
    assert calls["n"] == 1
    assert len(landed) == 1, "a submission that landed must not be duplicated"


# ---------------------------------------------------------------------------
# _submit_and_verify: submit= seam (codex's JSON-RPC transport)
# ---------------------------------------------------------------------------

def test_submit_and_verify_custom_submit_defaults_transport_to_rpc(tmp_path, monkeypatch):
    """A caller-supplied submit callable bypasses HTTP/keystroke entirely and
    reports transport="rpc" by default."""
    calls = []
    _witness_sequence(monkeypatch, [0, 1])

    outcome = agent_run._submit_and_verify(
        tmp_path, tmp_path, b"hello",
        deadline_s=5.0, max_attempts=1,
        submit=lambda text: calls.append(text),
    )

    assert outcome.verified is True
    assert outcome.transport == "rpc"
    assert calls == [b"hello"]


def test_submit_and_verify_custom_submit_witness_gated(tmp_path, monkeypatch):
    """A submit callable that always succeeds but whose witness never rises
    must not be reported as verified -- codex's turn/start ack is not proof
    of delivery, exactly like the FIFO/HTTP transports."""
    calls = []
    _witness_sequence(monkeypatch, [0])  # flat forever

    outcome = agent_run._submit_and_verify(
        tmp_path, tmp_path, b"hello",
        deadline_s=0.05, max_attempts=2,
        submit=lambda text: calls.append(text),
    )

    assert outcome.verified is False
    assert outcome.transport == "rpc"
    assert outcome.detail == "timeout"
    assert len(calls) == 2  # readable-but-flat witness still retries


def test_submit_and_verify_custom_submit_rejected_is_fast_and_non_retryable(
    tmp_path, monkeypatch
):
    """_SubmissionRejected from the submit callable (codex's turn_start_error
    fast-fail) returns immediately, without waiting out deadline_s or
    consuming a second attempt."""
    calls = []

    def rejecting_submit(text):
        calls.append(text)
        raise agent_run._SubmissionRejected("bad turn params")

    start = time.monotonic()
    outcome = agent_run._submit_and_verify(
        tmp_path, tmp_path, b"hello",
        deadline_s=30.0, max_attempts=3,
        submit=rejecting_submit,
    )
    elapsed = time.monotonic() - start

    assert outcome.verified is False
    assert outcome.attempts == 1
    assert outcome.detail == "rejected: bad turn params"
    assert len(calls) == 1
    assert elapsed < 5.0, f"rejection took {elapsed:.1f}s, expected an immediate return"


def test_submit_and_verify_custom_submit_transport_error_retries(tmp_path, monkeypatch):
    """A plain OSError from the submit callable is treated like a built-in
    transport failure: retried on the next attempt, not fatal. The retry
    happens only because the post-error witness poll stayed flat."""
    calls = {"n": 0}

    def flaky_submit(_text):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("connection reset")

    # Baseline 0; every read through the post-error poll and the pre-resend
    # re-check stays flat, so attempt 2 runs; then the witness rises.
    _witness_sequence(monkeypatch, [0, 0, 0, 0, 1])
    monkeypatch.setattr(agent_run, "SUBMISSION_RETRY_BACKOFF_SECONDS", 0.0)

    outcome = agent_run._submit_and_verify(
        tmp_path, tmp_path, b"hello",
        deadline_s=0.01, max_attempts=2,
        submit=flaky_submit,
    )

    assert outcome.verified is True
    assert outcome.attempts == 2
    assert calls["n"] == 2


def test_submit_via_http_is_fire_and_forget(monkeypatch):
    """The HTTP transport must never block reading the response: it writes
    the request and returns as soon as sendall completes, without calling
    recv/makefile/read on the socket."""
    import socket as socket_module

    sent = {}

    class _FakeSocket:
        def sendall(self, data):
            sent["data"] = data

        def close(self):
            sent["closed"] = True

        def recv(self, *_a, **_k):
            raise AssertionError("must not read the HTTP response")

        def makefile(self, *_a, **_k):
            raise AssertionError("must not read the HTTP response")

    monkeypatch.setattr(
        socket_module, "create_connection", lambda *_a, **_k: _FakeSocket()
    )

    agent_run._submit_via_http(41234, "ses_1", "hello there")

    assert sent.get("closed") is True
    assert b"POST /session/ses_1/message" in sent["data"]
    assert b"hello there" in sent["data"]


# ---------------------------------------------------------------------------
# steer exit codes (verified / unverified / --raw) and its write watchdog
# ---------------------------------------------------------------------------

def test_steer_times_out_instead_of_hanging_on_a_stuck_fifo(
    isolated_runs_root, monkeypatch
):
    """cmd_steer arms SIGALRM around the whole submit+verify call. Because
    TimeoutError subclasses OSError, _submit_and_verify must re-raise it
    rather than logging transport_error and retrying the same stuck FIFO."""
    _fifo, reader = _seed_live_interactive_run(
        isolated_runs_root, "run", agent_run.SUBMIT_MODE_CR
    )
    monkeypatch.setattr(agent_run, "_pid_alive", lambda _pid: True)
    monkeypatch.setattr(agent_run.signal, "alarm", lambda _seconds: None)
    _witness_sequence(monkeypatch, [0])
    sends = {"n": 0}

    def stuck_fifo(*_a, **_k):
        sends["n"] += 1
        raise TimeoutError("write timed out")

    monkeypatch.setattr(agent_run, "_submit_via_keystroke", stuck_fifo)
    try:
        with pytest.raises(SystemExit, match="timed out writing to FIFO"):
            agent_run.cmd_steer(_steer_args("run"))
    finally:
        os.close(reader)
    assert sends["n"] == 1, "a stuck FIFO must not be retried"


def test_steer_verified_prints_verified_and_exits_zero(isolated_runs_root, monkeypatch):
    _fifo, reader = _seed_live_interactive_run(
        isolated_runs_root, "run", agent_run.SUBMIT_MODE_CR
    )
    monkeypatch.setattr(agent_run, "_pid_alive", lambda _pid: True)
    monkeypatch.setattr(agent_run.signal, "alarm", lambda _seconds: None)
    _witness_sequence(monkeypatch, [0, 1])
    try:
        rc = agent_run.cmd_steer(_steer_args("run"))
        assert rc == 0
        os.read(reader, 4096)
    finally:
        os.close(reader)


def test_steer_unverified_exits_one_with_reason_on_stderr(
    isolated_runs_root, monkeypatch, capsys
):
    _fifo, reader = _seed_live_interactive_run(
        isolated_runs_root, "run", agent_run.SUBMIT_MODE_CR
    )
    monkeypatch.setattr(agent_run, "_pid_alive", lambda _pid: True)
    monkeypatch.setattr(agent_run.signal, "alarm", lambda _seconds: None)
    _witness_sequence(monkeypatch, [0])  # baseline 0, never rises
    try:
        rc = agent_run.cmd_steer(_steer_args("run"))
        assert rc == 1
        os.read(reader, 4096)
    finally:
        os.close(reader)
    captured = capsys.readouterr()
    assert "could not be verified" in captured.err


def test_steer_raw_skips_witness_polling_entirely(isolated_runs_root, monkeypatch):
    """--raw must not poll because raw bytes have no transcript record."""
    _fifo, reader = _seed_live_interactive_run(
        isolated_runs_root, "run", agent_run.SUBMIT_MODE_CR
    )
    monkeypatch.setattr(agent_run, "_pid_alive", lambda _pid: True)
    monkeypatch.setattr(agent_run.signal, "alarm", lambda _seconds: None)

    def fail_if_called(*_a, **_k):
        raise AssertionError("--raw must not poll the witness")

    monkeypatch.setattr(agent_run, "_resolve_witness_source", fail_if_called)
    try:
        rc = agent_run.cmd_steer(_steer_args("run", raw=True))
        assert rc == 0
        assert os.read(reader, 4096) == b"hello"
    finally:
        os.close(reader)


# ---------------------------------------------------------------------------
# Env override parsing: AGENT_RUN_SUBMIT_VERIFY_TIMEOUT / _ATTEMPTS
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("5", 5.0),
        ("2.5", 2.5),
        (None, agent_run.SUBMISSION_VERIFY_TIMEOUT_SECONDS),
        ("not-a-number", agent_run.SUBMISSION_VERIFY_TIMEOUT_SECONDS),
        ("0", agent_run.SUBMISSION_VERIFY_TIMEOUT_SECONDS),
        ("-1", agent_run.SUBMISSION_VERIFY_TIMEOUT_SECONDS),
        ("nan", agent_run.SUBMISSION_VERIFY_TIMEOUT_SECONDS),
        ("inf", agent_run.SUBMISSION_VERIFY_TIMEOUT_SECONDS),
    ],
)
def test_submission_verify_timeout_env_parsing(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv("AGENT_RUN_SUBMIT_VERIFY_TIMEOUT", raising=False)
    else:
        monkeypatch.setenv("AGENT_RUN_SUBMIT_VERIFY_TIMEOUT", raw)
    assert agent_run._submission_verify_timeout_seconds() == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("5", 5),
        (None, agent_run.SUBMISSION_MAX_ATTEMPTS),
        ("not-a-number", agent_run.SUBMISSION_MAX_ATTEMPTS),
        ("0", agent_run.SUBMISSION_MAX_ATTEMPTS),
        ("-1", agent_run.SUBMISSION_MAX_ATTEMPTS),
        ("1.5", agent_run.SUBMISSION_MAX_ATTEMPTS),
    ],
)
def test_submission_max_attempts_env_parsing(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv("AGENT_RUN_SUBMIT_ATTEMPTS", raising=False)
    else:
        monkeypatch.setenv("AGENT_RUN_SUBMIT_ATTEMPTS", raw)
    assert agent_run._submission_max_attempts() == expected


# ---------------------------------------------------------------------------
# opencode_port persistence + argv fallback for pre-existing runs
# ---------------------------------------------------------------------------

def test_opencode_http_endpoint_reads_persisted_port_file(tmp_path):
    (tmp_path / "opencode_port").write_text("41234\n")
    (tmp_path / "session.json").write_text(
        json.dumps({"session_id": "ses_1", "harness": "opencode"})
    )

    endpoint = agent_run._opencode_http_endpoint(tmp_path, tmp_path)

    assert endpoint == (41234, "ses_1")


def test_opencode_http_endpoint_falls_back_to_argv_port_for_legacy_state_dir(tmp_path):
    """A state dir predating opencode_port (only argv on disk) must still
    resolve the port by parsing --port out of the persisted argv."""
    (tmp_path / "argv").write_text(
        json.dumps(["opencode", "--port", "55001", "--session", "ses_2"])
    )
    (tmp_path / "session.json").write_text(
        json.dumps({"session_id": "ses_2", "harness": "opencode"})
    )
    assert not (tmp_path / "opencode_port").exists()

    endpoint = agent_run._opencode_http_endpoint(tmp_path, tmp_path)

    assert endpoint == (55001, "ses_2")


def test_opencode_http_endpoint_none_for_non_opencode_harness(tmp_path):
    (tmp_path / "opencode_port").write_text("41234\n")
    (tmp_path / "session.json").write_text(
        json.dumps({"session_id": "ses_1", "harness": "claude"})
    )

    assert agent_run._opencode_http_endpoint(tmp_path, tmp_path) is None


def test_opencode_http_endpoint_none_when_port_unresolvable(tmp_path):
    (tmp_path / "session.json").write_text(
        json.dumps({"session_id": "ses_1", "harness": "opencode"})
    )
    # No opencode_port file and no --port in argv.
    (tmp_path / "argv").write_text(json.dumps(["opencode", "--session", "ses_1"]))

    assert agent_run._opencode_http_endpoint(tmp_path, tmp_path) is None


def test_launch_persists_opencode_port_file(isolated_runs_root, isolated_log_root, monkeypatch):
    """cmd_launch writes state_dir/opencode_port for a managed interactive
    opencode run whose port was selected, so a later steer/verification can
    resolve it without re-parsing argv."""
    monkeypatch.setattr(agent_run, "_find_free_port", lambda: 47001)
    monkeypatch.setattr(
        agent_run, "_opencode_prefork_mint",
        lambda *a, **k: "ses_launched",
    )
    fake_dir = isolated_runs_root.parent / "bin"
    fake_dir.mkdir(exist_ok=True)
    fake = fake_dir / "opencode"
    fake.write_text("#!/bin/sh\nsleep 30\n")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", str(fake_dir) + ":" + os.environ.get("PATH", ""))

    name = "oc-port-persist"
    ns = argparse.Namespace(
        name=name, command=[], interactive=True, prompt_file=None,
        submit_mode=None, idle_timeout=None,
        harness="opencode", prompt=None, model=None, agent_mode=None,
        harness_args=[], permissions="bypass",
    )
    rc = agent_run.cmd_launch(ns)
    assert rc == 0
    state = isolated_runs_root / name
    try:
        assert _wait_until(lambda: (state / "opencode_port").exists())
        assert (state / "opencode_port").read_text().strip() == "47001"
    finally:
        pid_path = state / "pid"
        if pid_path.exists():
            try:
                os.kill(int(pid_path.read_text()), signal.SIGTERM)
            except (ProcessLookupError, ValueError):
                pass


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["opencode", "--port", "47001"], 47001),
        (["opencode", "--port=47001"], 47001),
        # harness_args are appended after the managed --port, so a caller's
        # override is what opencode actually binds: last one wins.
        (["opencode", "--port", "47001", "--port", "48002"], 48002),
        (["opencode", "--port", "47001", "--port=48002"], 48002),
        (["opencode", "--session", "ses_1"], None),
        (["opencode", "--port"], None),  # value missing entirely
        (["opencode", "--port", "not-a-number"], None),
    ],
)
def test_effective_argv_port_takes_the_last_port_flag(argv, expected):
    assert agent_run._effective_argv_port(argv) == expected


def test_launch_persists_the_effective_port_under_a_harness_arg_override(
    isolated_runs_root, isolated_log_root, monkeypatch
):
    """--harness-arg=--port makes opencode listen elsewhere; persisting the
    managed port would point the witness at a server nothing is running on."""
    monkeypatch.setattr(agent_run, "_find_free_port", lambda: 47001)
    monkeypatch.setattr(
        agent_run, "_opencode_prefork_mint",
        lambda *a, **k: "ses_override",
    )
    fake_dir = isolated_runs_root.parent / "bin"
    fake_dir.mkdir(exist_ok=True)
    fake = fake_dir / "opencode"
    fake.write_text("#!/bin/sh\nsleep 30\n")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", str(fake_dir) + ":" + os.environ.get("PATH", ""))

    name = "oc-port-override"
    ns = argparse.Namespace(
        name=name, command=[], interactive=True, prompt_file=None,
        submit_mode=None, idle_timeout=None,
        harness="opencode", prompt=None, model=None, agent_mode=None,
        harness_args=["--port", "48002"], permissions="bypass",
    )
    rc = agent_run.cmd_launch(ns)
    assert rc == 0
    state = isolated_runs_root / name
    try:
        assert _wait_until(lambda: (state / "opencode_port").exists())
        assert (state / "opencode_port").read_text().strip() == "48002"
    finally:
        pid_path = state / "pid"
        if pid_path.exists():
            try:
                os.kill(int(pid_path.read_text()), signal.SIGTERM)
            except (ProcessLookupError, ValueError):
                pass


def test_launch_resolves_to_failed_when_the_port_file_cannot_be_written(
    isolated_runs_root, isolated_log_root, monkeypatch
):
    """The opencode_port write happens after status=starting but before the
    fork, so a failure there must reach the same terminal-state cleanup a
    fork failure does rather than stranding the run in starting."""
    monkeypatch.setattr(agent_run, "_find_free_port", lambda: 47001)
    monkeypatch.setattr(
        agent_run, "_opencode_prefork_mint",
        lambda *a, **k: "ses_enospc",
    )
    fake_dir = isolated_runs_root.parent / "bin"
    fake_dir.mkdir(exist_ok=True)
    fake = fake_dir / "opencode"
    fake.write_text("#!/bin/sh\nsleep 30\n")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", str(fake_dir) + ":" + os.environ.get("PATH", ""))

    name = "oc-port-enospc"
    real_write = agent_run._write

    def fail_port_write(path, text):
        if path.name == "opencode_port":
            raise OSError(errno.ENOSPC, "no space left on device")
        return real_write(path, text)

    monkeypatch.setattr(agent_run, "_write", fail_port_write)

    ns = argparse.Namespace(
        name=name, command=[], interactive=True, prompt_file=None,
        submit_mode=None, idle_timeout=None,
        harness="opencode", prompt=None, model=None, agent_mode=None,
        harness_args=[], permissions="bypass",
    )
    with pytest.raises(SystemExit, match="failed to record opencode port"):
        agent_run.cmd_launch(ns)

    state = isolated_runs_root / name
    assert (state / "status").read_text().strip() == "failed"
    assert (state / "exit_code").read_text().strip() == "1"
    assert (state / "ended_at").is_file()
    assert not (state / "pid").exists()
