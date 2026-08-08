from __future__ import annotations

import argparse
import errno
import io
import os
import signal
import struct
import sys
import time
from pathlib import Path

import pytest

from toolbox import agent_run


def _wait_until(predicate, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def test_pack_resize_uses_four_byte_network_order():
    assert agent_run._pack_resize(120, 42) == struct.pack(">HH", 120, 42)
    assert agent_run.RESIZE_RECORD_SIZE == 4


@pytest.mark.parametrize("cols,rows", [(0, 24), (80, 0), (-1, 24), (80, 65536)])
def test_pack_resize_rejects_invalid_dimensions(cols, rows):
    with pytest.raises(ValueError):
        agent_run._pack_resize(cols, rows)


def test_apply_resize_uses_rows_then_columns(monkeypatch):
    calls = []
    monkeypatch.setattr(
        agent_run.fcntl,
        "ioctl",
        lambda fd, request, payload: calls.append((fd, request, payload)),
    )

    agent_run._apply_resize(99, 120, 42)

    assert calls == [
        (99, agent_run.termios.TIOCSWINSZ, struct.pack("HHHH", 42, 120, 0, 0))
    ]


@pytest.mark.parametrize("error", [errno.EBADF, errno.EIO, errno.EINVAL, errno.ENOTTY])
def test_apply_resize_ignores_closed_or_non_tty_master(monkeypatch, error):
    def fail_ioctl(*_args):
        raise OSError(error, "expected test failure")

    monkeypatch.setattr(agent_run.fcntl, "ioctl", fail_ioctl)
    agent_run._apply_resize(99, 120, 42)


@pytest.mark.parametrize("error", [errno.EACCES, errno.EPERM])
def test_apply_resize_reraises_unexpected_errno(monkeypatch, error):
    def fail_ioctl(*_args):
        raise OSError(error, "expected test failure")

    monkeypatch.setattr(agent_run.fcntl, "ioctl", fail_ioctl)
    with pytest.raises(OSError):
        agent_run._apply_resize(99, 120, 42)


def test_drain_resize_records_preserves_partial_suffix(monkeypatch):
    applied = []
    monkeypatch.setattr(
        agent_run,
        "_apply_resize",
        lambda master_fd, cols, rows: applied.append((master_fd, cols, rows)),
    )
    record = struct.pack(">HH", 120, 42)

    suffix = agent_run._drain_resize_records(7, record[:2])
    assert suffix == record[:2]
    assert applied == []

    suffix = agent_run._drain_resize_records(7, suffix + record[2:] + struct.pack(">HH", 80, 24))
    assert suffix == b""
    assert applied == [(7, 120, 42), (7, 80, 24)]


def test_interactive_launch_creates_stdin_and_resize_fifos(isolated_runs_root):
    args = argparse.Namespace(
        name="resize-fifos",
        command=[sys.executable, "-c", "import time; time.sleep(30)"],
        interactive=True,
        prompt_file=None,
        echo=False,
        echo_interval=2.0,
        submit_mode=None,
    )
    assert agent_run.cmd_launch(args) == 0
    state = isolated_runs_root / "resize-fifos"
    try:
        assert _wait_until(lambda: (state / "status").read_text().strip() == "running")
        assert (state / "stdin").is_fifo()
        assert (state / "resize").is_fifo()
    finally:
        try:
            os.kill(int((state / "pid").read_text()), signal.SIGTERM)
        except ProcessLookupError:
            pass


def test_keeper_ack_eof_raises_instead_of_hanging(tmp_path):
    """If the keeper dies before acking (e.g. it failed to open one of the
    control FIFOs), _run_interactive must raise promptly rather than block
    forever on the subsequent blocking FIFO opens."""
    state_dir = tmp_path / "state"
    log_dir = tmp_path / "log"
    state_dir.mkdir()
    log_dir.mkdir()
    (log_dir / "log").touch()
    os.mkfifo(state_dir / "stdin")
    # Deliberately omit the "resize" FIFO so the keeper's open() fails and it
    # dies before writing its ack byte.

    pid = os.fork()
    if pid == 0:
        try:
            agent_run._run_interactive(
                state_dir,
                [sys.executable, "-c", "import time; time.sleep(0.3)"],
                os.open(os.devnull, os.O_WRONLY),
                lambda: None,
            )
            os._exit(0)
        except RuntimeError:
            os._exit(2)
        except BaseException:
            os._exit(3)

    deadline = time.monotonic() + 5.0
    waited_pid, status = 0, None
    while time.monotonic() < deadline:
        waited_pid, status = os.waitpid(pid, os.WNOHANG)
        if waited_pid == pid:
            break
        time.sleep(0.05)
    else:
        os.kill(pid, signal.SIGKILL)
        os.waitpid(pid, 0)
        pytest.fail("child hung instead of raising for a keeper that died before acking")

    assert os.WIFEXITED(status)
    assert os.WEXITSTATUS(status) == 2


def test_attach_rejects_noninteractive_run(isolated_runs_root):
    state = isolated_runs_root / "batch"
    state.mkdir()
    (state / "interactive").write_text("0\n")
    (state / "pid").write_text("123\n")

    with pytest.raises(SystemExit, match="not interactive"):
        agent_run.cmd_attach(argparse.Namespace(name="batch"))


def test_attach_rejects_dead_interactive_run(isolated_runs_root, monkeypatch):
    state = isolated_runs_root / "dead"
    state.mkdir()
    (state / "interactive").write_text("1\n")
    (state / "pid").write_text("123\n")
    monkeypatch.setattr(agent_run, "_pid_alive", lambda _pid: False)

    with pytest.raises(SystemExit, match="is not running"):
        agent_run.cmd_attach(argparse.Namespace(name="dead"))


def test_main_dispatches_attach(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        agent_run,
        "cmd_attach",
        lambda args: captured.update(name=args.name) or 0,
    )

    assert agent_run.main(["attach", "live-run"]) == 0
    assert captured == {"name": "live-run"}


def _seed_attachable_run(root, name: str) -> Path:
    state = root / name
    state.mkdir()
    (state / "interactive").write_text("1\n")
    (state / "pid").write_text(f"{os.getpid()}\n")
    os.mkfifo(state / "stdin")
    os.mkfifo(state / "resize")
    return state


def test_attach_resets_terminal_modes_on_external_sigint(
    isolated_runs_root, isolated_log_root, monkeypatch
):
    """Mirrors `tests/test_cli_clean.py::test_tail_resets_terminal_modes_after_ctrl_c`:
    attach replays raw PTY bytes to a real terminal exactly like `tail` does,
    so an external SIGINT (KeyboardInterrupt) must still run the same
    `_reset_terminal_modes()` cleanup and exit quietly with 128+SIGINT."""
    state = _seed_attachable_run(isolated_runs_root, "attach-sigint")
    (isolated_log_root / "attach-sigint").mkdir()
    (isolated_log_root / "attach-sigint" / "log").write_bytes(b"")
    stdin_reader = os.open(state / "stdin", os.O_RDONLY | os.O_NONBLOCK)
    resize_reader = os.open(state / "resize", os.O_RDONLY | os.O_NONBLOCK)

    class _FakeStdout:
        def __init__(self):
            self.buffer = io.BytesIO()

        def isatty(self) -> bool:
            return True

    stdout = _FakeStdout()
    monkeypatch.setattr(agent_run.sys, "stdout", stdout)
    monkeypatch.setattr(agent_run.sys.stdin, "fileno", lambda: 0)
    monkeypatch.setattr(agent_run, "_pid_alive", lambda _pid: True)
    monkeypatch.setattr(agent_run.termios, "tcgetattr", lambda _fd: ["saved"])
    monkeypatch.setattr(agent_run.tty, "setraw", lambda _fd: None)
    monkeypatch.setattr(agent_run.termios, "tcsetattr", lambda *_a: None)
    monkeypatch.setattr(
        agent_run.os, "get_terminal_size", lambda _fd: os.terminal_size((80, 24))
    )

    def fail_select(*_a, **_k):
        raise KeyboardInterrupt

    monkeypatch.setattr(agent_run.select, "select", fail_select)
    try:
        assert (
            agent_run.cmd_attach(argparse.Namespace(name="attach-sigint"))
            == 128 + agent_run.signal.SIGINT
        )
        assert stdout.buffer.getvalue() == agent_run._TERMINAL_MODE_RESET
    finally:
        os.close(stdin_reader)
        os.close(resize_reader)

