"""Focused tests for agent-run's per-agent interactive submission mode."""
from __future__ import annotations

import argparse
import os
import signal
import time
from pathlib import Path

import pytest

from toolbox import agent_run


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
        (["env", "--chdir=/tmp", "opencode"], agent_run.SUBMIT_MODE_CR),
    ],
)
def test_submit_mode_for_argv_only_inspects_unambiguous_executable(argv, expected):
    assert agent_run._submit_mode_for_argv(argv) == expected


def test_persist_submit_mode_is_symbolic_and_uses_authoritative_argv(tmp_path):
    mode = agent_run._persist_submit_mode(tmp_path, ["/usr/local/bin/opencode", "mention"])

    assert mode == agent_run.SUBMIT_MODE_CRLF
    assert (tmp_path / "submit_mode").read_text() == "crlf\n"


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
    try:
        assert agent_run.cmd_steer(_steer_args("run")) == 0
        assert os.read(reader, 4096) == expected
    finally:
        os.close(reader)


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (agent_run.SUBMIT_MODE_CR, b"\x1bhello\r\r"),
        (agent_run.SUBMIT_MODE_CRLF, b"\x1bhello\r\n\r\n"),
    ],
)
def test_steer_esc_reuses_persisted_submit_mode(
    isolated_runs_root, monkeypatch, mode, expected
):
    _fifo, reader = _seed_live_interactive_run(isolated_runs_root, "run", mode)
    monkeypatch.setattr(agent_run, "_pid_alive", lambda _pid: True)
    monkeypatch.setattr(agent_run.signal, "alarm", lambda _seconds: None)
    monkeypatch.setattr(agent_run.time, "sleep", lambda _seconds: None)
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


def test_teardown_never_signals_or_waits_for_self(tmp_path, monkeypatch):
    own_pid = os.getpid()
    (tmp_path / "keeper_pid").write_text(f"{own_pid}\n")
    (tmp_path / "echo_pid").write_text(f"{own_pid}\n")
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
    (tmp_path / "echo_pid").write_text(f"{child_pid}\n")
    killed = []
    wait_results = iter([(0, 0), (child_pid, 0)])
    monkeypatch.setattr(agent_run.os, "getpid", lambda: 1111)
    monkeypatch.setattr(agent_run.os, "kill", lambda pid, sig: killed.append((pid, sig)))
    monkeypatch.setattr(agent_run.os, "waitpid", lambda _pid, _options: next(wait_results))

    agent_run._teardown_children(tmp_path, grace=0.1)

    assert killed == [(child_pid, signal.SIGTERM)]
