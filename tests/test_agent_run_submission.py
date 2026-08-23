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


def test_prompt_submission_writes_use_selected_sequence():
    assert agent_run._prompt_submission_writes(b"prompt", agent_run.SUBMIT_MODE_CRLF) == (
        b"prompt\r\n",
        b"\r\n",
    )
    assert agent_run._prompt_submission_writes(b"prompt", agent_run.SUBMIT_MODE_CR) == (
        b"prompt\r",
        b"\r",
    )


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


def _mock_witness_rises_on_first_poll(monkeypatch) -> None:
    """Baseline read returns 0; every subsequent read returns 1, so
    _submit_and_verify's very first post-submit poll sees the rise and
    returns immediately without sleeping through the real poll interval."""
    calls = {"n": 0}

    def fake_witness(_state_dir, _log_dir):
        calls["n"] += 1
        return 0 if calls["n"] == 1 else 1

    monkeypatch.setattr(agent_run, "_submission_witness_count", fake_witness)


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
    _mock_witness_rises_on_first_poll(monkeypatch)
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
    _mock_witness_rises_on_first_poll(monkeypatch)
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
    """The bug this feature closes: a steer that never lands must not report
    success. A permanently unreadable witness submits exactly once (no
    evidence to justify a retry) and the CLI reports failure."""
    _fifo, reader = _seed_live_interactive_run(
        isolated_runs_root, "run", agent_run.SUBMIT_MODE_CR
    )
    monkeypatch.setattr(agent_run, "_pid_alive", lambda _pid: True)
    monkeypatch.setattr(agent_run.signal, "alarm", lambda _seconds: None)
    monkeypatch.setattr(agent_run, "_submission_witness_count", lambda *_a: None)
    monkeypatch.setenv("AGENT_RUN_SUBMIT_VERIFY_TIMEOUT", "0.05")
    try:
        assert agent_run.cmd_steer(_steer_args("run")) == 1
        assert os.read(reader, 4096) == b"hello\r"
    finally:
        os.close(reader)


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

def _witness_sequence(monkeypatch, counts: list) -> list:
    """Patch _submission_witness_count to return successive values from
    *counts* (repeating the last value once exhausted) and return the list
    of (state_dir, log_dir) argument pairs it was called with, for callers
    that want to assert call counts."""
    calls: list = []

    def fake(state_dir, log_dir):
        calls.append((state_dir, log_dir))
        index = min(len(calls) - 1, len(counts) - 1)
        return counts[index]

    monkeypatch.setattr(agent_run, "_submission_witness_count", fake)
    return calls


def _no_transport_side_effects(monkeypatch) -> list:
    """Stub both transports as no-ops that just record how many times they
    were invoked, and force keystroke transport (no opencode HTTP endpoint)."""
    submissions: list = []
    monkeypatch.setattr(agent_run, "_opencode_http_endpoint", lambda *_a: None)
    monkeypatch.setattr(
        agent_run, "_submit_via_keystroke",
        lambda *_a, **_k: submissions.append(1),
    )
    return submissions


def test_submit_and_verify_swallowed_input_exhausts_attempts(tmp_path, monkeypatch):
    """Reproduces the production bug: transport accepts every write but the
    witness never rises. Exactly max_attempts submissions are made and the
    outcome reports unverified with a timeout reason."""
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
    """Flat through the entire first attempt's poll window, then rises during
    the second attempt: two submissions, verified True, attempts == 2."""
    submissions = _no_transport_side_effects(monkeypatch)
    # baseline=0; attempt 1 polls stay at 0 through its deadline and its
    # post-timeout re-check; attempt 2's first poll rises to 1.
    _witness_sequence(monkeypatch, [0, 0, 0, 1])

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

    def fake_witness(_state_dir, _log_dir):
        poll_calls["n"] += 1
        if poll_calls["n"] == 1:
            return 0  # baseline
        if poll_calls["n"] == 2:
            return 0  # the only poll iteration before the deadline elapses
        return 1  # the post-timeout re-check

    monkeypatch.setattr(agent_run, "_submission_witness_count", fake_witness)

    outcome = agent_run._submit_and_verify(
        tmp_path, tmp_path, b"hello", submit_mode=agent_run.SUBMIT_MODE_CR,
        deadline_s=0.05, max_attempts=3,
    )

    assert outcome.verified is True
    assert outcome.attempts == 1
    assert len(submissions) == 1


def test_submit_and_verify_witness_unreadable_submits_exactly_once(tmp_path, monkeypatch):
    """count_transcript raising TranscriptSourceError (via
    _submission_witness_count's real implementation) must degrade to
    verified=False with detail=witness_unreadable, never propagate.

    A witness that stays unreadable for the whole attempt gives no evidence
    the submission failed to land -- resubmitting risks a real duplicate
    prompt (messageID is not idempotent) for zero informational gain, so
    max_attempts is never consulted here: exactly one submission is made.
    """
    monkeypatch.setattr(agent_run, "_opencode_http_endpoint", lambda *_a: None)
    monkeypatch.setattr(
        agent_run, "_read_session_json",
        lambda _log_dir: {"session_id": "s1", "harness": "codex"},
    )

    def raise_source_error(_harness, _session_id, _cwd):
        raise agent_run_transcript.TranscriptSourceError("boom", code="store_missing")

    monkeypatch.setattr(
        agent_run._agent_run_transcript, "count_transcript", raise_source_error
    )
    submissions = []
    monkeypatch.setattr(
        agent_run, "_submit_via_keystroke",
        lambda *_a, **_k: submissions.append(1),
    )
    os.mkfifo(tmp_path / "stdin")

    outcome = agent_run._submit_and_verify(
        tmp_path, tmp_path, b"hello", submit_mode=agent_run.SUBMIT_MODE_CR,
        deadline_s=1.0, max_attempts=2,
    )

    assert outcome.verified is False
    assert outcome.detail == "witness_unreadable"
    assert outcome.attempts == 1
    assert len(submissions) == 1


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
    transport failure: retried on the next attempt, not fatal."""
    calls = {"n": 0}

    def flaky_submit(_text):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("connection reset")

    _witness_sequence(monkeypatch, [0, 1])

    outcome = agent_run._submit_and_verify(
        tmp_path, tmp_path, b"hello",
        deadline_s=1.0, max_attempts=2,
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
# steer exit codes (verified / unverified / --raw)
# ---------------------------------------------------------------------------

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
    """--raw must never call the witness: raw bytes have no transcript
    record to observe, and the brief requires zero polling for this path."""
    _fifo, reader = _seed_live_interactive_run(
        isolated_runs_root, "run", agent_run.SUBMIT_MODE_CR
    )
    monkeypatch.setattr(agent_run, "_pid_alive", lambda _pid: True)
    monkeypatch.setattr(agent_run.signal, "alarm", lambda _seconds: None)

    def fail_if_called(*_a, **_k):
        raise AssertionError("--raw must not poll the witness")

    monkeypatch.setattr(agent_run, "_submission_witness_count", fail_if_called)
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
