from __future__ import annotations

import argparse
import errno
import fcntl
import io
import os
import pty
import select
import signal
import struct
import subprocess
import sys
import termios
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


def _environment(state_root: Path, log_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["AGENT_RUN_STATE_DIR"] = str(state_root)
    env["AGENT_RUN_LOG_DIR"] = str(log_root)
    return env


def _launch_interactive(name: str, script: str, env: dict[str, str]) -> Path:
    result = subprocess.run(
        [sys.executable, "-m", "toolbox.agent_run", "-i", name, sys.executable, "-c", script],
        capture_output=True,
        env=env,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr.decode()
    state = Path(env["AGENT_RUN_STATE_DIR"]) / name
    assert _wait_until(lambda: (state / "status").read_text().strip() == "running")
    return state


def _spawn_attach(name: str, env: dict[str, str], rows: int, cols: int):
    master_fd, slave_fd = pty.openpty()
    fcntl.ioctl(slave_fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
    process = subprocess.Popen(
        [sys.executable, "-m", "toolbox.agent_run", "attach", name],
        stdin=slave_fd,
        stdout=slave_fd,
        stderr=slave_fd,
        env=env,
        start_new_session=True,
        close_fds=True,
    )
    os.close(slave_fd)
    return process, master_fd


def _read_until(fd: int, needle: bytes, timeout: float = 10.0) -> bytes:
    deadline = time.monotonic() + timeout
    output = bytearray()
    while time.monotonic() < deadline:
        readable, _, _ = select.select([fd], [], [], 0.1)
        if not readable:
            continue
        try:
            output.extend(os.read(fd, 4096))
        except OSError:
            break
        if needle in output:
            return bytes(output)
    raise AssertionError(f"did not receive {needle!r}; received {bytes(output)!r}")


def _poke_until_size(fd: int, probe: bytes, needle: bytes, timeout: float = 10.0) -> bytes:
    """Repeatedly write `probe` and read until `needle` appears.

    Used for terminal-size assertions where a single probe byte can race a
    resize record still in flight over the FIFO — retrying the probe until
    the reader's reported size matches converges regardless of how many
    stale reads (or echoed probe bytes) land first, without an arbitrary
    fixed sleep.
    """
    deadline = time.monotonic() + timeout
    output = bytearray()
    while time.monotonic() < deadline:
        os.write(fd, probe)
        readable, _, _ = select.select([fd], [], [], 0.2)
        if not readable:
            continue
        try:
            output.extend(os.read(fd, 4096))
        except OSError:
            break
        if needle in output:
            return bytes(output)
    raise AssertionError(f"did not receive {needle!r}; received {bytes(output)!r}")


def _stop_run(state: Path) -> None:
    try:
        os.kill(int((state / "pid").read_text()), signal.SIGTERM)
    except ProcessLookupError:
        return
    assert _wait_until(lambda: (state / "status").read_text().strip() != "running")


def _detach(process: subprocess.Popen, master_fd: int, timeout: float = 5.0) -> None:
    """Send Ctrl-C and wait for `attach` to exit, draining its PTY master
    the whole time.

    `attach`'s cleanup calls `termios.tcsetattr(..., TCSADRAIN, ...)`, which
    blocks until every byte it already wrote has actually been read by the
    other end of the PTY. Any output left unread from an earlier
    `_read_until`/`_poke_until_size` call (both stop as soon as their needle
    appears, not necessarily after draining everything queued) can fill the
    PTY's output buffer and wedge that tcsetattr forever unless something
    keeps reading — exactly what a real attached terminal emulator would do
    continuously.
    """
    try:
        if process.poll() is None:
            os.write(master_fd, b"\x03")
            deadline = time.monotonic() + timeout
            while process.poll() is None and time.monotonic() < deadline:
                readable, _, _ = select.select([master_fd], [], [], 0.1)
                if readable:
                    try:
                        os.read(master_fd, 65536)
                    except OSError:
                        break
            assert process.wait(timeout=max(0.0, deadline - time.monotonic())) == 0
    finally:
        os.close(master_fd)


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


def test_attach_relays_input_detaches_on_ctrl_c_and_restores_terminal(
    isolated_runs_root, isolated_log_root
):
    env = _environment(isolated_runs_root, isolated_log_root)
    script = """\
import os, sys, termios, tty, time
fd = sys.stdin.fileno()
saved = termios.tcgetattr(fd)
tty.setraw(fd)
print("READY", flush=True)
print(f"GOT:{os.read(fd, 1).hex()}", flush=True)
termios.tcsetattr(fd, termios.TCSADRAIN, saved)
time.sleep(30)
"""
    state = _launch_interactive("attach-input", script, env)
    process, master_fd = _spawn_attach("attach-input", env, rows=24, cols=80)
    try:
        original = termios.tcgetattr(master_fd)
        _read_until(master_fd, b"READY")
        os.write(master_fd, b"x")
        _read_until(master_fd, b"GOT:78")
        os.write(master_fd, b"\x03")
        assert process.wait(timeout=5) == 0
        assert termios.tcgetattr(master_fd) == original
        assert (state / "status").read_text().strip() == "running"
        assert agent_run._pid_alive(int((state / "pid").read_text()))
    finally:
        _detach(process, master_fd)
        _stop_run(state)


def test_attach_applies_initial_and_changed_terminal_size(
    isolated_runs_root, isolated_log_root
):
    env = _environment(isolated_runs_root, isolated_log_root)
    script = """\
import os, sys, termios, tty
fd = sys.stdin.fileno()
saved = termios.tcgetattr(fd)
tty.setraw(fd)
print("READY", flush=True)
while True:
    b = os.read(fd, 1)
    if not b:
        break
    size = os.get_terminal_size(fd)
    print(f"SIZE:{size.columns}x{size.lines}", flush=True)
termios.tcsetattr(fd, termios.TCSADRAIN, saved)
"""
    state = _launch_interactive("attach-size", script, env)
    process, master_fd = _spawn_attach("attach-size", env, rows=31, cols=101)
    try:
        _read_until(master_fd, b"READY")
        # The initial resize record races the probe keystroke over separate
        # FIFOs, so retry the probe until the reported size reflects it
        # rather than asserting on a single round trip.
        _poke_until_size(master_fd, b"s", b"SIZE:101x31")
        fcntl.ioctl(master_fd, termios.TIOCSWINSZ, struct.pack("HHHH", 47, 133, 0, 0))
        os.kill(process.pid, signal.SIGWINCH)
        _poke_until_size(master_fd, b"s", b"SIZE:133x47")
    finally:
        _detach(process, master_fd)
        _stop_run(state)


def test_multiple_attaches_share_output_and_last_resize_wins(
    isolated_runs_root, isolated_log_root
):
    # Two independently-spawned attach processes each send their own
    # "initial connect" resize record over the *same* shared resize FIFO,
    # so if they were spawned concurrently, which one is applied last would
    # be an unavoidable race. To keep the test deterministic, connect and
    # settle `first` on its own, THEN spawn `second` — at that point
    # `second` is the only source of a fresh resize record for its own
    # connect. Each size assertion also retries its probe keystroke (see
    # _poke_until_size) since a resize record can still race an individual
    # keystroke over the separate stdin/resize FIFOs.
    env = _environment(isolated_runs_root, isolated_log_root)
    script = """\
import os, sys, termios, tty
fd = sys.stdin.fileno()
saved = termios.tcgetattr(fd)
tty.setraw(fd)
print("READY", flush=True)
while True:
    b = os.read(fd, 1)
    if not b:
        break
    size = os.get_terminal_size(fd)
    print(f"SIZE:{size.columns}x{size.lines}", flush=True)
termios.tcsetattr(fd, termios.TCSADRAIN, saved)
"""
    state = _launch_interactive("attach-many", script, env)
    first, first_master = _spawn_attach("attach-many", env, rows=24, cols=80)
    try:
        _read_until(first_master, b"READY")
        _poke_until_size(first_master, b"s", b"SIZE:80x24")

        second, second_master = _spawn_attach("attach-many", env, rows=50, cols=120)
        try:
            # Before `attach` calls tty.setraw() on its own controlling
            # terminal, the PTY slave is still in canonical/echoing mode, so
            # a probe byte sent too early gets echoed back onto
            # second_master (and never reaches attach's stdin) instead of
            # being read by the wrapped script — wait for the replayed log
            # (containing "READY" from the very start, since attach opens
            # the log with no seek) to confirm attach has reached its main
            # loop before probing.
            _read_until(second_master, b"READY")
            _poke_until_size(second_master, b"s", b"SIZE:120x50")

            fcntl.ioctl(first_master, termios.TIOCSWINSZ, struct.pack("HHHH", 40, 90, 0, 0))
            os.kill(first.pid, signal.SIGWINCH)
            _poke_until_size(first_master, b"s", b"SIZE:90x40")
            _poke_until_size(second_master, b"s", b"SIZE:90x40")
        finally:
            _detach(second, second_master)
    finally:
        _detach(first, first_master)
        _stop_run(state)


def test_attach_drains_final_output_and_exits_zero(isolated_runs_root, isolated_log_root):
    env = _environment(isolated_runs_root, isolated_log_root)
    script = "import time; time.sleep(0.3); print('FINAL-ATTACH-OUTPUT', flush=True)"
    state = _launch_interactive("attach-final", script, env)
    process, master_fd = _spawn_attach("attach-final", env, rows=24, cols=80)
    try:
        _read_until(master_fd, b"FINAL-ATTACH-OUTPUT")
        assert process.wait(timeout=5) == 0
        assert _wait_until(lambda: (state / "status").read_text().strip() == "done")
    finally:
        if process.poll() is None:
            _detach(process, master_fd)
        else:
            os.close(master_fd)


