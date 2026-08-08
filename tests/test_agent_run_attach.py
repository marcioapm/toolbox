from __future__ import annotations

import argparse
import errno
import os
import signal
import struct
import sys
import time

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

