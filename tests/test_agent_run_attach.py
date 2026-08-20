"""Tests for `agent-run attach` and the interactive PTY runner backing it.

Run this file in isolation rather than concurrently with another copy of
the suite. The fixtures redirect state and log roots into tmp_path, but the
runs these tests launch are real processes reachable through the shared
/var/tmp/agent-runs tree by anything that scans it, so two suites running at
once can reap or terminate each other's runs and produce failures that do
not reproduce alone.
"""
from __future__ import annotations

import argparse
import contextlib
import errno
import fcntl
import io
import os
import pty
import random
import select
import signal
import struct
import subprocess
import sys
import termios
import textwrap
import threading
import time
from pathlib import Path
from unittest import mock

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


def _spawn_attach(name: str, env: dict[str, str], rows: int, cols: int, stderr=None):
    master_fd, slave_fd = pty.openpty()
    fcntl.ioctl(slave_fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
    process = subprocess.Popen(
        [sys.executable, "-m", "toolbox.agent_run", "attach", name],
        stdin=slave_fd,
        stdout=slave_fd,
        stderr=slave_fd if stderr is None else stderr,
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
    """Send Ctrl-D and wait for `attach` to exit, draining its PTY master
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
            os.write(master_fd, b"\x04")
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


def test_pack_resize_frames_records_with_a_magic_byte_version_and_checksum():
    assert agent_run._pack_resize(120, 42) == struct.pack(
        ">BBHHB",
        agent_run.RESIZE_RECORD_MAGIC,
        agent_run.RESIZE_RECORD_VERSION,
        120,
        42,
        agent_run._resize_checksum(120, 42),
    )
    assert agent_run.RESIZE_RECORD_SIZE == 7


@pytest.mark.parametrize(
    "cols,rows", [(0, 24), (80, 0), (-1, 24), (80, 65536), (2001, 24), (80, 2001)]
)
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
    record = agent_run._pack_resize(120, 42)

    suffix = agent_run._drain_resize_records(7, record[:2])
    assert suffix == record[:2]
    assert applied == []

    # A single record trickling in across two reads is still applied once
    # it's complete.
    suffix = agent_run._drain_resize_records(7, suffix + record[2:])
    assert suffix == b""
    assert applied == [(7, 120, 42)]


def test_drain_resize_records_coalesces_a_queued_batch_to_the_last(monkeypatch):
    # Multiple complete records arriving together in ONE call (e.g. a
    # window-drag burst that outran the reader) must apply only the final,
    # still-relevant size -- not every intermediate one, which would fire a
    # spurious SIGWINCH/repaint per discarded stale size.
    applied = []
    monkeypatch.setattr(
        agent_run,
        "_apply_resize",
        lambda master_fd, cols, rows: applied.append((master_fd, cols, rows)),
    )
    batch = (
        agent_run._pack_resize(100, 30)
        + agent_run._pack_resize(110, 32)
        + agent_run._pack_resize(120, 42)
    )

    suffix = agent_run._drain_resize_records(7, batch)

    assert suffix == b""
    assert applied == [(7, 120, 42)]


def test_drain_resize_records_batch_still_preserves_trailing_partial(monkeypatch):
    applied = []
    monkeypatch.setattr(
        agent_run,
        "_apply_resize",
        lambda master_fd, cols, rows: applied.append((master_fd, cols, rows)),
    )
    partial = agent_run._pack_resize(80, 24)[:2]
    batch = agent_run._pack_resize(100, 30) + agent_run._pack_resize(120, 42) + partial

    suffix = agent_run._drain_resize_records(7, batch)

    assert suffix == partial
    assert applied == [(7, 120, 42)]


def test_drain_resize_records_resyncs_past_a_truncated_record(monkeypatch):
    """A short write (or a second attach client interleaving) leaves a
    partial record in the stream. Without magic-byte framing every record
    after it decodes at the wrong offset, so the wrapped agent's PTY is
    resized to garbage dimensions from then on; the reader must skip to the
    next record boundary instead."""
    applied = []
    monkeypatch.setattr(
        agent_run,
        "_apply_resize",
        lambda master_fd, cols, rows: applied.append((master_fd, cols, rows)),
    )
    torn = agent_run._pack_resize(100, 30)[:3]

    suffix = agent_run._drain_resize_records(7, torn + agent_run._pack_resize(120, 42))

    assert suffix == b""
    assert applied == [(7, 120, 42)]


def test_drain_resize_records_discards_a_batch_with_no_record_boundary(monkeypatch):
    applied = []
    monkeypatch.setattr(
        agent_run,
        "_apply_resize",
        lambda master_fd, cols, rows: applied.append((master_fd, cols, rows)),
    )

    assert agent_run._drain_resize_records(7, b"\x00\x01\x02\x03") == b""
    assert applied == []


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
    (state / agent_run.RESIZE_PROTOCOL_MARKER).write_text(
        f"{agent_run.RESIZE_RECORD_VERSION}\n"
    )
    return state


def test_attach_requires_tty_stdin(isolated_runs_root, isolated_log_root, monkeypatch):
    """C1 regression: a non-TTY stdin (e.g. `attach x < /dev/null`, or any
    pipeline/CI invocation) must exit cleanly via sys.exit with the
    established `agent-run: ...` message style, not an unhandled
    termios.error traceback from tcgetattr()."""
    state = _seed_attachable_run(isolated_runs_root, "attach-notty")
    (isolated_log_root / "attach-notty").mkdir()
    (isolated_log_root / "attach-notty" / "log").write_bytes(b"")
    monkeypatch.setattr(agent_run.sys.stdin, "isatty", lambda: False)

    with pytest.raises(SystemExit, match="requires an interactive terminal"):
        agent_run.cmd_attach(argparse.Namespace(name="attach-notty"))


def test_attach_falls_back_when_resize_fifo_missing(isolated_runs_root, isolated_log_root, capsys):
    """Coverage for a run launched before this branch's resize-FIFO support:
    a state dir with a `stdin` FIFO but no `resize` FIFO must not block
    attach entirely. Keyboard passthrough and Ctrl-D detach still work;
    only resize forwarding is skipped, with a warning on stderr. (The
    subsequent SystemExit here is the unrelated no-tty check firing --
    confirms the resize check didn't sys.exit first.)"""
    state = isolated_runs_root / "pre-feature"
    state.mkdir()
    (state / "interactive").write_text("1\n")
    (state / "pid").write_text(f"{os.getpid()}\n")
    os.mkfifo(state / "stdin")
    (isolated_log_root / "pre-feature").mkdir()
    (isolated_log_root / "pre-feature" / "log").write_bytes(b"")

    with pytest.raises(SystemExit, match="requires an interactive terminal"):
        agent_run.cmd_attach(argparse.Namespace(name="pre-feature"))
    assert "no resize FIFO" in capsys.readouterr().err


def test_attach_end_to_end_without_resize_fifo(isolated_runs_root, isolated_log_root):
    """End-to-end: a session launched before resize-FIFO support (simulated
    by deleting the FIFO a real `-i` launch created) must still be
    attachable -- input relays through and Ctrl-D detaches cleanly -- just
    without resize forwarding."""
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
    state = _launch_interactive("attach-no-resize", script, env)
    os.remove(state / "resize")
    process, master_fd = _spawn_attach("attach-no-resize", env, rows=24, cols=80)
    try:
        _read_until(master_fd, b"READY")
        os.write(master_fd, b"x")
        _read_until(master_fd, b"GOT:78")
        os.write(master_fd, b"\x04")
        assert process.wait(timeout=5) == 0
        assert (state / "status").read_text().strip() == "running"
        assert agent_run._pid_alive(int((state / "pid").read_text()))
    finally:
        _detach(process, master_fd)
        _stop_run(state)


def test_attach_skips_resize_when_terminal_size_is_zero(
    isolated_runs_root, isolated_log_root, monkeypatch
):
    """C2 regression: os.get_terminal_size() can legitimately report 0x0
    (e.g. a pty.openpty() master whose winsize was never set, or a terminal
    mid-teardown). _pack_resize correctly rejects 0 as an out-of-range
    dimension, so cmd_attach must skip sending a resize that iteration
    instead of letting the ValueError escape as a traceback."""
    state = _seed_attachable_run(isolated_runs_root, "attach-zerosize")
    (isolated_log_root / "attach-zerosize").mkdir()
    (isolated_log_root / "attach-zerosize" / "log").write_bytes(b"")
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
    monkeypatch.setattr(agent_run.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(agent_run, "_pid_alive", lambda _pid: True)
    monkeypatch.setattr(agent_run.termios, "tcgetattr", lambda _fd: ["saved"])
    monkeypatch.setattr(agent_run.tty, "setraw", lambda _fd: None)
    monkeypatch.setattr(agent_run.termios, "tcsetattr", lambda *_a: None)
    monkeypatch.setattr(
        agent_run.os, "get_terminal_size", lambda _fd: os.terminal_size((0, 0))
    )

    def fail_select(*_a, **_k):
        raise KeyboardInterrupt

    monkeypatch.setattr(agent_run.select, "select", fail_select)
    try:
        # A 0x0 size must not raise ValueError out of _pack_resize before
        # reaching select() — if it did, the KeyboardInterrupt from
        # fail_select would never fire and this would propagate ValueError
        # instead of returning the expected exit status.
        assert (
            agent_run.cmd_attach(argparse.Namespace(name="attach-zerosize"))
            == 128 + agent_run.signal.SIGINT
        )
    finally:
        os.close(stdin_reader)
        os.close(resize_reader)


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
    monkeypatch.setattr(agent_run.sys.stdin, "isatty", lambda: True)
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


def test_attach_relays_input_detaches_on_ctrl_d_and_restores_terminal(
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
        os.write(master_fd, b"\x04")
        assert process.wait(timeout=5) == 0
        assert termios.tcgetattr(master_fd) == original
        assert (state / "status").read_text().strip() == "running"
        assert agent_run._pid_alive(int((state / "pid").read_text()))
    finally:
        _detach(process, master_fd)
        _stop_run(state)


def test_attach_ctrl_d_forwards_only_bytes_before_marker_within_one_chunk(
    isolated_runs_root, isolated_log_root
):
    """Direct coverage for the Task 2 byte-preservation fix.

    `test_attach_relays_input_detaches_on_ctrl_d_and_restores_terminal` only
    catches a leaked `\\x04` *indirectly*: its wrapped Python script happens
    to raise `KeyboardInterrupt` if a stray `\\x03` reaches it, flipping
    status to `failed` -- but that mechanism cannot catch a stray `\\x04`
    (EOF does not raise), so this test proves delivery directly. The
    wrapped agent logs every byte it reads, a single write of
    `b"ab\\x04cd"` is sent as one chunk (matching `cmd_attach`'s "scan each
    raw read chunk for `\\x04`" behavior), and the test asserts (1) the
    bytes before `\\x04` (0x61 'a', 0x62 'b') were delivered, (2) `\\x04`
    itself was not, (3) nothing after it in the same chunk (0x63 'c', 0x64
    'd') was either, and (4) the wrapped run stayed alive/running
    throughout — only `attach` detaches, never the run.

    Reads the persistent log file directly rather than `attach`'s own PTY
    output: `cmd_attach` returns immediately after forwarding the
    pre-`\\x04` bytes and detecting the detach marker, in the same loop
    iteration, without looping back to drain the log growth the wrapped
    agent's own `BYTE:..` writes produce — so by the time `attach` exits,
    those log lines have not necessarily been replayed onto its stdout yet.
    """
    env = _environment(isolated_runs_root, isolated_log_root)
    name = "attach-ctrld-truncate"
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
    print(f"BYTE:{b.hex()}", flush=True)
termios.tcsetattr(fd, termios.TCSADRAIN, saved)
"""
    state = _launch_interactive(name, script, env)
    process, master_fd = _spawn_attach(name, env, rows=24, cols=80)
    try:
        _read_until(master_fd, b"READY")
        os.write(master_fd, b"ab\x04cd")
        assert process.wait(timeout=5) == 0

        log_path = isolated_log_root / name / "log"
        assert _wait_until(lambda: b"BYTE:62" in log_path.read_bytes(), timeout=5)
        log = log_path.read_bytes()
        assert b"BYTE:61" in log  # 'a' delivered
        assert b"BYTE:62" in log  # 'b' delivered
        assert b"BYTE:04" not in log  # '\x04' itself never delivered
        assert b"BYTE:63" not in log  # 'c' (after \x04 in the chunk) dropped
        assert b"BYTE:64" not in log  # 'd' (after \x04 in the chunk) dropped

        assert (state / "status").read_text().strip() == "running"
        assert agent_run._pid_alive(int((state / "pid").read_text()))
    finally:
        _detach(process, master_fd)
        _stop_run(state)


@pytest.mark.parametrize(
    "ctrl_c_bytes",
    [
        b"ab\x03cd",  # raw Ctrl-C byte
        b"ab\x1b[27;5;99~cd",  # CSI modifyOtherKeys form
        b"ab\x1b[99;5ucd",  # kitty CSI-u form
    ],
    ids=["raw", "csi-modify-other-keys", "csi-kitty"],
)
def test_attach_forwards_ctrl_c_to_the_wrapped_agent_without_detaching(
    isolated_runs_root, isolated_log_root, ctrl_c_bytes
):
    """Ctrl-C is no longer the detach trigger: every byte, including a raw
    `\\x03` or a CSI-encoded Ctrl-C sequence, must reach the wrapped agent
    unmodified, and `attach` itself must stay attached (not exit) -- the
    mirror image of the Ctrl-D truncation test above."""
    env = _environment(isolated_runs_root, isolated_log_root)
    name = f"attach-ctrlc-passthrough-{ctrl_c_bytes.hex()}"
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
    print(f"BYTE:{b.hex()}", flush=True)
termios.tcsetattr(fd, termios.TCSADRAIN, saved)
"""
    state = _launch_interactive(name, script, env)
    process, master_fd = _spawn_attach(name, env, rows=24, cols=80)
    try:
        _read_until(master_fd, b"READY")
        os.write(master_fd, ctrl_c_bytes)

        log_path = isolated_log_root / name / "log"
        assert _wait_until(lambda: b"BYTE:64" in log_path.read_bytes(), timeout=5)
        log = log_path.read_bytes()
        for byte in ctrl_c_bytes:
            assert f"BYTE:{byte:02x}".encode() in log, (
                f"byte {byte:#04x} from {ctrl_c_bytes!r} was not forwarded"
            )

        assert process.poll() is None, "attach detached on Ctrl-C"
        assert (state / "status").read_text().strip() == "running"
        assert agent_run._pid_alive(int((state / "pid").read_text()))
    finally:
        _detach(process, master_fd)
        _stop_run(state)


def test_find_ctrl_d_trigger_recognizes_raw_byte():
    assert agent_run._find_ctrl_d_trigger(b"ab\x04cd") == (2, 3)


def test_find_ctrl_d_trigger_recognizes_iterm_csi_form():
    # What iTerm2 actually sends for Ctrl-D once Claude Code's TUI has
    # negotiated the "disambiguate escape codes" keyboard protocol: the
    # legacy xterm modifyOtherKeys form, ESC[27;<mod>;<codepoint>~, with
    # mod=5 (1 + Ctrl's bit 4) and codepoint=100 ('d').
    data = b"ab\x1b[27;5;100~cd"
    start, end = agent_run._find_ctrl_d_trigger(data)
    assert data[:start] == b"ab"
    assert data[end:] == b"cd"


def test_find_ctrl_d_trigger_recognizes_kitty_csi_u_form():
    data = b"ab\x1b[100;5ucd"
    start, end = agent_run._find_ctrl_d_trigger(data)
    assert data[:start] == b"ab"
    assert data[end:] == b"cd"


@pytest.mark.parametrize(
    "trigger",
    [
        b"\x1b[100;5:1u",  # explicit press event type
        b"\x1b[100;5:2u",  # repeat event type -- a held-down Ctrl-D
        b"\x1b[100;5:3u",  # release event type
        b"\x1b[100:65:67;5u",  # shifted + base-layout alt-key-codes
        b"\x1b[100:65;5u",  # a single alt-key-code
        b"\x1b[100;5;100u",  # "report associated text" field, one codepoint
        b"\x1b[100;5;100:100u",  # associated text, multiple codepoints
        b"\x1b[100:65:67;5:1;100u",  # alt-codes + event type + associated text together
    ],
)
def test_find_ctrl_d_trigger_recognizes_kitty_optional_subparameters(trigger):
    # Terminals implementing the kitty keyboard protocol may include these
    # optional colon-separated subparameters even when the client (Claude
    # Code) only requested the base "disambiguate escape codes" flag --
    # some terminals apply their own defaults regardless of exactly what
    # was requested. Detection must not depend on the client's requested
    # flags matching the terminal's actual behavior.
    data = b"ab" + trigger + b"cd"
    start, end = agent_run._find_ctrl_d_trigger(data)
    assert data[:start] == b"ab"
    assert data[end:] == b"cd"
    assert data[start:end] == trigger


@pytest.mark.parametrize(
    "trigger",
    [
        b"\x1b[100;5;u",  # empty "associated text" field
        b"\x1b[100;5:u",  # empty event-type subparameter
        b"\x1b[100;5:1;u",  # explicit event type, empty associated text
    ],
)
def test_find_ctrl_d_trigger_recognizes_kitty_forms_with_empty_subparameters(trigger):
    # Per ECMA-48/kitty-protocol grammar, an EMPTY parameter between two
    # separators means "use the default value" and is legal on the wire --
    # a terminal sending it is spec-compliant. Requiring at least one
    # digit in every subparameter slot would silently miss these forms,
    # reintroducing the exact leak-through-to-the-wrapped-agent bug this
    # detector exists to fix.
    data = b"ab" + trigger + b"cd"
    start, end = agent_run._find_ctrl_d_trigger(data)
    assert data[:start] == b"ab"
    assert data[end:] == b"cd"
    assert data[start:end] == trigger


def test_find_ctrl_d_trigger_returns_the_earliest_match_not_the_last():
    # A CSI Ctrl-D at an earlier offset, followed later in the SAME chunk
    # by a stray raw 0x04, must win -- picking the later trigger would
    # forward the earlier, intact CSI Ctrl-D sequence into the wrapped
    # agent's stdin as an unintended side effect of detaching.
    csi = b"\x1b[100;5u"
    data = b"ab" + csi + b"some intervening text" + b"\x04" + b"more"
    start, end = agent_run._find_ctrl_d_trigger(data)
    assert data[start:end] == csi
    assert start == 2


def test_find_ctrl_d_trigger_raw_byte_wins_when_it_is_earliest():
    # The inverse of the above: when the raw 0x04 genuinely comes first,
    # it must still be the one detected.
    data = b"ab\x04cd\x1b[100;5u"
    start, end = agent_run._find_ctrl_d_trigger(data)
    assert data[start:end] == b"\x04"
    assert start == 2


@pytest.mark.parametrize(
    "data",
    [
        b"\x1b[27;1;100~",  # mod=1 -> no modifiers held at all, not Ctrl-D
        b"\x1b[27;2;100~",  # mod=2 -> Shift only
        b"\x1b[27;3;100~",  # mod=3 -> Shift+Alt, still no Ctrl bit
        b"\x1b[100;1u",
        b"\x1b[100:65;1u",  # alt-key-code present, but still no Ctrl modifier
        b"\x1b[27;5;99~",  # codepoint 99 = 'c', not 'd'
        b"\x1b[99;5u",  # kitty form, wrong codepoint
    ],
)
def test_find_ctrl_d_trigger_ignores_non_ctrl_csi_forms(data):
    assert agent_run._find_ctrl_d_trigger(data) == (-1, -1)


def test_split_trailing_incomplete_escape_holds_back_in_flight_csi():
    forwardable, held = agent_run._split_trailing_incomplete_escape(b"ab\x1b[27;5")
    assert forwardable == b"ab"
    assert held == b"\x1b[27;5"


def test_split_trailing_incomplete_escape_releases_terminated_sequence():
    data = b"ab\x1b[27;5;100~"
    forwardable, held = agent_run._split_trailing_incomplete_escape(data)
    assert forwardable == data
    assert held == b""


def test_split_trailing_incomplete_escape_holds_back_lone_esc():
    forwardable, held = agent_run._split_trailing_incomplete_escape(b"ab\x1b")
    assert forwardable == b"ab"
    assert held == b"\x1b"


def test_split_trailing_incomplete_escape_releases_esc_without_bracket():
    # ESC not followed by '[' (e.g. an Alt-modified letter key) is not the
    # start of a CSI sequence -- never held back.
    data = b"ab\x1bc"
    forwardable, held = agent_run._split_trailing_incomplete_escape(data)
    assert forwardable == data
    assert held == b""


def test_split_trailing_incomplete_escape_forwards_an_over_long_unterminated_run():
    """An unterminated ESC run longer than any real escape sequence is
    forwarded, not dropped. Dropping it loses user input: the bytes are
    almost always paste payload that happens to contain an ESC, and
    discarding them silently truncates what the agent receives."""
    prefix = b"ab"
    data = prefix + b"\x1b[" + b"9" * agent_run._MAX_PENDING_ESCAPE_BYTES
    forwardable, held = agent_run._split_trailing_incomplete_escape(data)
    assert forwardable == data
    assert held == b""


def test_split_trailing_incomplete_escape_keeps_a_long_complete_sequence():
    """A complete sequence must be forwarded whole however long its tail
    is. Checking the length bound before completeness truncated the chunk
    at the last ESC, silently discarding everything after it."""
    data = b"ab\x1b[200~" + b"x" * (agent_run._MAX_PENDING_ESCAPE_BYTES * 4)
    forwardable, held = agent_run._split_trailing_incomplete_escape(data)
    assert forwardable == data
    assert held == b""


def test_split_trailing_incomplete_escape_ignores_an_earlier_terminated_escape():
    """`rfind` locates the last ESC, which may sit inside an already
    complete sequence. Only a genuinely unterminated trailing sequence is
    held back."""
    data = b"ab\x1b[200~cd\x1b[A"
    forwardable, held = agent_run._split_trailing_incomplete_escape(data)
    assert forwardable == data
    assert held == b""


def test_scan_local_input_forwards_a_whole_bracketed_paste_intact():
    """A paste arriving as one read must reach the agent byte-for-byte:
    losing the ESC[200~ marker drops the receiving TUI out of paste mode,
    so every embedded newline submits as a separate Enter."""
    payload = b"first line\nsecond line\n" * 100
    data = b"\x1b[200~" + payload + b"\x1b[201~"

    forwardable, held, detached, in_paste, _pb, _ps, _st = agent_run._scan_local_input_for_detach(data)

    assert forwardable == data
    assert held == b""
    assert not detached
    assert not in_paste


def test_scan_local_input_does_not_detach_on_ctrl_d_inside_a_paste():
    """A 0x04 byte in pasted content is data, not a keypress. Detaching on
    it drops the session mid-paste and forwards only the fragment before
    the byte."""
    data = b"\x1b[200~before\x04after\x1b[201~tail"

    forwardable, held, detached, in_paste, _pb, _ps, _st = agent_run._scan_local_input_for_detach(data)

    assert forwardable == data
    assert not detached
    assert not in_paste
    assert held == b""


def test_scan_local_input_does_not_detach_on_literal_csi_ctrl_d_inside_a_paste():
    data = b"\x1b[200~see \x1b[100;5u here\x1b[201~"

    forwardable, _held, detached, in_paste, _pb, _ps, _st = agent_run._scan_local_input_for_detach(data)

    assert forwardable == data
    assert not detached
    assert not in_paste


def test_scan_local_input_carries_paste_state_across_reads():
    """A paste larger than one 4096-byte read spans several calls; the
    in-paste flag has to persist or a 0x04 in a later chunk detaches."""
    first = b"\x1b[200~" + b"x" * 20
    forwardable, _held, detached, in_paste, _pb, _ps, _st = agent_run._scan_local_input_for_detach(first)
    assert forwardable == first
    assert not detached
    assert in_paste

    second = b"y\x04z\x1b[201~"
    forwardable, _held, detached, in_paste, _pb, _ps, _st = agent_run._scan_local_input_for_detach(
        second, in_paste
    )
    assert forwardable == second
    assert not detached
    assert not in_paste


def test_scan_local_input_detaches_again_once_the_paste_has_ended():
    data = b"\x1b[200~pasted\x1b[201~typed\x04rest"

    forwardable, _held, detached, in_paste, _pb, _ps, _st = agent_run._scan_local_input_for_detach(data)

    assert forwardable == b"\x1b[200~pasted\x1b[201~typed"
    assert detached
    assert not in_paste


def test_scan_local_input_detaches_on_ctrl_d_before_a_paste_starts():
    data = b"typed\x04\x1b[200~pasted\x1b[201~"

    forwardable, _held, detached, _in_paste, _pb, _ps, _st = agent_run._scan_local_input_for_detach(data)

    assert forwardable == b"typed"
    assert detached


@pytest.mark.parametrize(
    "trigger",
    [
        b"\x1b[27;1000;100~",  # modifier value past three digits
        b"\x1b[27;5;100:100~",  # subparameters on the codepoint
        b"\x1b[27:1;5;100~",  # subparameters on the leading 27
    ],
)
def test_find_ctrl_d_trigger_recognizes_wide_modifiers_and_subparameters(trigger):
    """Conforming encodings the detector must not miss: a modifier bitmask
    above 999 (many modifiers held at once) and colon subparameters on the
    modifyOtherKeys form's own parameters."""
    data = b"ab" + trigger + b"cd"
    start, end = agent_run._find_ctrl_d_trigger(data)
    assert data[start:end] == trigger


def test_attach_ctrl_d_via_csi_sequence_detaches_without_leaking(
    isolated_runs_root, isolated_log_root
):
    """Reproduces the real-world bug: iTerm2 (and any terminal honoring the
    kitty/CSI-u "disambiguate escape codes" keyboard protocol Claude Code's
    TUI requests) sends Ctrl-D as `ESC[27;5;100~`, not a raw `\\x04` byte. A
    detector that only looks for `\\x04` never fires, and the whole escape
    sequence leaks straight into the wrapped agent's stdin instead of
    detaching `attach` locally."""
    env = _environment(isolated_runs_root, isolated_log_root)
    name = "attach-ctrld-csi"
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
    print(f"BYTE:{b.hex()}", flush=True)
termios.tcsetattr(fd, termios.TCSADRAIN, saved)
"""
    state = _launch_interactive(name, script, env)
    process, master_fd = _spawn_attach(name, env, rows=24, cols=80)
    try:
        _read_until(master_fd, b"READY")
        os.write(master_fd, b"ab\x1b[27;5;100~cd")
        assert process.wait(timeout=5) == 0

        log_path = isolated_log_root / name / "log"
        assert _wait_until(lambda: b"BYTE:62" in log_path.read_bytes(), timeout=5)
        log = log_path.read_bytes()
        assert b"BYTE:61" in log  # 'a' delivered
        assert b"BYTE:62" in log  # 'b' delivered
        assert b"BYTE:1b" not in log  # the CSI sequence's ESC never delivered
        assert b"BYTE:63" not in log  # 'c' (after the CSI trigger) dropped
        assert b"BYTE:64" not in log  # 'd' (after the CSI trigger) dropped

        assert (state / "status").read_text().strip() == "running"
        assert agent_run._pid_alive(int((state / "pid").read_text()))
    finally:
        _detach(process, master_fd)
        _stop_run(state)


def test_attach_ctrl_d_via_kitty_extended_csi_form_detaches_without_leaking(
    isolated_runs_root, isolated_log_root
):
    """Terminals implementing the kitty keyboard protocol can send the
    optional alt-key-codes/event-type/associated-text subparameters even
    when Claude Code only requested the base disambiguation flag -- proves
    the whole attach pipeline (not just the unit-level regex) still
    detaches on one of those extended forms rather than only the minimal
    `ESC[100;5u`."""
    env = _environment(isolated_runs_root, isolated_log_root)
    name = "attach-ctrld-kitty-ext"
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
    print(f"BYTE:{b.hex()}", flush=True)
termios.tcsetattr(fd, termios.TCSADRAIN, saved)
"""
    state = _launch_interactive(name, script, env)
    process, master_fd = _spawn_attach(name, env, rows=24, cols=80)
    try:
        _read_until(master_fd, b"READY")
        # Alt-key-codes (65:67) + explicit press event type (:1) together.
        os.write(master_fd, b"ab\x1b[100:65:67;5:1ucd")
        assert process.wait(timeout=5) == 0

        log_path = isolated_log_root / name / "log"
        assert _wait_until(lambda: b"BYTE:62" in log_path.read_bytes(), timeout=5)
        log = log_path.read_bytes()
        assert b"BYTE:61" in log
        assert b"BYTE:62" in log
        assert b"BYTE:1b" not in log
        assert b"BYTE:63" not in log
        assert b"BYTE:64" not in log

        assert (state / "status").read_text().strip() == "running"
        assert agent_run._pid_alive(int((state / "pid").read_text()))
    finally:
        _detach(process, master_fd)
        _stop_run(state)


def test_attach_ctrl_d_via_csi_split_across_two_reads_still_detaches(
    isolated_runs_root, isolated_log_root
):
    """The CSI-u Ctrl-D trigger can legitimately arrive split across two
    local-terminal reads (e.g. a slow pty writer, or scheduler jitter). The
    held-back-prefix logic must reassemble it rather than either missing
    the detach or corrupting/duplicating forwarded bytes."""
    env = _environment(isolated_runs_root, isolated_log_root)
    name = "attach-ctrld-csi-split"
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
    print(f"BYTE:{b.hex()}", flush=True)
termios.tcsetattr(fd, termios.TCSADRAIN, saved)
"""
    state = _launch_interactive(name, script, env)
    process, master_fd = _spawn_attach(name, env, rows=24, cols=80)
    try:
        _read_until(master_fd, b"READY")
        os.write(master_fd, b"ab\x1b[27;5")
        time.sleep(0.01)
        os.write(master_fd, b";100~cd")
        assert process.wait(timeout=5) == 0

        log_path = isolated_log_root / name / "log"
        assert _wait_until(lambda: b"BYTE:62" in log_path.read_bytes(), timeout=5)
        log = log_path.read_bytes()
        assert b"BYTE:61" in log
        assert b"BYTE:62" in log
        assert b"BYTE:1b" not in log
        assert b"BYTE:63" not in log
        assert b"BYTE:64" not in log

        assert (state / "status").read_text().strip() == "running"
    finally:
        _detach(process, master_fd)
        _stop_run(state)


def test_attach_bare_escape_key_still_reaches_wrapped_agent(
    isolated_runs_root, isolated_log_root
):
    """A bare Escape keypress (used constantly in TUIs, including Claude
    Code's own, to cancel/back out) must still be forwarded — the
    CSI-sequence-holdback logic must not swallow it just because it starts
    with the same ESC byte a CSI sequence would."""
    env = _environment(isolated_runs_root, isolated_log_root)
    name = "attach-bare-esc"
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
    print(f"BYTE:{b.hex()}", flush=True)
termios.tcsetattr(fd, termios.TCSADRAIN, saved)
"""
    state = _launch_interactive(name, script, env)
    process, master_fd = _spawn_attach(name, env, rows=24, cols=80)
    try:
        _read_until(master_fd, b"READY")
        os.write(master_fd, b"\x1b")

        log_path = isolated_log_root / name / "log"
        assert _wait_until(lambda: b"BYTE:1b" in log_path.read_bytes(), timeout=5)
        assert (state / "status").read_text().strip() == "running"
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


def test_attach_is_listed_in_cli_help(capsys):
    assert agent_run.main(["--help"]) == 0
    assert "attach" in capsys.readouterr().out



def test_tail_non_tty_stdin_uses_keyboardinterrupt_fallback_unchanged(
    isolated_runs_root, isolated_log_root, monkeypatch
):
    """Piped/redirected stdin must skip the raw-mode/CSI-detection path
    entirely and keep relying on the existing KeyboardInterrupt-based exit
    -- confirms the TTY guard actually gates the new behavior."""
    run = isolated_runs_root / "tail-non-tty"
    run.mkdir()
    (run / "pid").write_text("123\n")
    log_dir = isolated_log_root / "tail-non-tty"
    log_dir.mkdir()
    (log_dir / "log").write_bytes(b"")

    class _FakeStdin:
        def isatty(self) -> bool:
            return False

    class _FakeStdout:
        def __init__(self):
            self.buffer = io.BytesIO()

    monkeypatch.setattr(agent_run.sys, "stdin", _FakeStdin())
    monkeypatch.setattr(agent_run.sys, "stdout", _FakeStdout())
    monkeypatch.setattr(agent_run, "_pid_alive", lambda _pid: True)
    monkeypatch.setattr(
        agent_run.time,
        "sleep",
        lambda _seconds: (_ for _ in ()).throw(KeyboardInterrupt),
    )
    monkeypatch.setattr(
        agent_run.select,
        "select",
        lambda *_a, **_k: pytest.fail("select() must not be called for non-TTY stdin"),
    )

    assert agent_run.cmd_tail(argparse.Namespace(name="tail-non-tty")) == 128 + signal.SIGINT


def _spawn_tail(name: str, env: dict[str, str], rows: int = 24, cols: int = 80, stderr=None):
    master_fd, slave_fd = pty.openpty()
    fcntl.ioctl(slave_fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
    process = subprocess.Popen(
        [sys.executable, "-m", "toolbox.agent_run", "tail", name],
        stdin=slave_fd,
        stdout=slave_fd,
        stderr=slave_fd if stderr is None else stderr,
        env=env,
        start_new_session=True,
        close_fds=True,
    )
    os.close(slave_fd)
    return process, master_fd


_BURST_SECONDS = 20.0
# The starvation bug defers input servicing until the whole log backlog has
# been replayed. At the client's 8192-byte-per-iteration read this backlog
# takes over ten seconds, against a detach budget measured from t0 = the
# instant \x04 is written.
_DETACH_BUDGET_SECONDS = 1.5
_BACKLOG_BYTES = 8 * 1024 * 1024
# Rate at which the test consumes the client's output, standing in for a
# terminal emulator that cannot repaint infinitely fast. It bounds how long
# replaying the backlog takes (~10s at this rate), so a client that replays
# it before servicing input misses the detach budget by a wide margin.
_DRAIN_BYTES_PER_READ = 8192
_DRAIN_INTERVAL_SECONDS = 0.01

# Dumps the backlog at full speed, then keeps the log growing at a bounded
# rate so the burst is still live at detach time without filling the disk.
_BURST_SCRIPT = f"""\
import sys, time
written = 0
while written < {_BACKLOG_BYTES}:
    sys.stdout.write("x" * 65536)
    sys.stdout.flush()
    written += 65536
deadline = time.monotonic() + {_BURST_SECONDS}
while time.monotonic() < deadline:
    sys.stdout.write("x" * 4096)
    sys.stdout.flush()
    time.sleep(0.01)
"""


def _drain_until_exit(process: subprocess.Popen, master_fd: int, deadline: float) -> None:
    """Keep reading `master_fd` until `process` exits or `deadline` passes.

    Draining must never stop: the client's own PTY output buffer is a
    limited kernel resource that a burst this size fills in milliseconds,
    and a full buffer blocks its `sys.stdout.buffer.write()` regardless of
    whether the fix under test works.
    """
    while process.poll() is None and time.monotonic() < deadline:
        readable, _, _ = select.select([master_fd], [], [], _DRAIN_INTERVAL_SECONDS)
        if not readable:
            continue
        try:
            os.read(master_fd, _DRAIN_BYTES_PER_READ)
            time.sleep(_DRAIN_INTERVAL_SECONDS)
        except OSError:
            # Closing the PTY slave (the client exiting) makes a subsequent
            # master read raise EIO before the child is reaped, so poll()
            # can still report None here. Leave the wait() to the caller.
            return


def _await_raw_mode(process: subprocess.Popen, master_fd: int) -> None:
    """Block until `attach` has actually put the PTY into raw mode.

    Checked by ECHO clearing, which is what `tty.setraw` does and what the
    caller depends on: setraw uses TCSAFLUSH, so anything written before it
    completes is discarded. Polls the condition explicitly rather than
    relying on the drain interval to be slower than attach's startup --
    that would make the wait a disguised sleep whose correctness depends on
    poll granularity.
    """
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if not termios.tcgetattr(master_fd)[3] & termios.ECHO:
            return
        if process.poll() is not None:
            raise AssertionError("attach exited before reaching raw mode")
        # Keep draining while waiting: attach's output write blocks once the
        # PTY buffer fills, which would stall the very startup we await.
        _drain_until_exit(process, master_fd, time.monotonic() + 0.05)
    raise AssertionError("attach never established raw mode")


def _await_backlog(log: Path) -> None:
    assert _wait_until(
        lambda: log.exists() and log.stat().st_size >= _BACKLOG_BYTES, timeout=15.0
    ), "wrapped agent never produced the backlog this test depends on"


def _assert_burst_still_live(log: Path) -> None:
    """The burst must still have been running when the client exited --
    otherwise the loop was never output-saturated and the test proves
    nothing about starvation."""
    before = log.stat().st_size
    time.sleep(0.3)
    assert log.stat().st_size > before, "burst had already finished on its own"


def test_attach_detaches_promptly_during_a_continuous_output_burst(
    isolated_runs_root, isolated_log_root
):
    """Regression test for the log-drain starvation bug: attach's loop used
    to `continue` straight back to another log read whenever output was
    pending, so a wrapped agent that never lets up starved keystroke
    forwarding, resize delivery, and Ctrl-D detection until the entire
    backlog had been replayed -- over ten seconds with the backlog here."""
    env = _environment(isolated_runs_root, isolated_log_root)
    name = "attach-burst-ctrld"
    state = _launch_interactive(name, _BURST_SCRIPT, env)
    log = isolated_log_root / name / "log"
    _await_backlog(log)
    process, master_fd = _spawn_attach(name, env, rows=24, cols=80)
    try:
        # tty.setraw uses TCSAFLUSH, which discards anything already queued
        # on stdin. Wait for raw mode to actually be established (ECHO off)
        # before writing \x04, or the byte is silently dropped.
        _await_raw_mode(process, master_fd)
        assert process.poll() is None, "attach exited before Ctrl-D was sent"

        t0 = time.monotonic()
        os.write(master_fd, b"\x04")
        _drain_until_exit(process, master_fd, t0 + _DETACH_BUDGET_SECONDS)
        # _drain_until_exit returns early on EIO, which the kernel raises as
        # soon as attach closes the PTY slave -- before the child is reaped,
        # so poll() can still be None. wait() out that window, then measure.
        assert process.wait(timeout=max(0.1, t0 + _DETACH_BUDGET_SECONDS - time.monotonic())) == 0
        elapsed = time.monotonic() - t0
        assert elapsed < _DETACH_BUDGET_SECONDS, f"detach took {elapsed:.2f}s"
        _assert_burst_still_live(log)
    finally:
        if process.poll() is None:
            process.terminate()
        os.close(master_fd)
        _stop_run(state)


def test_tail_exits_promptly_on_sigint_during_a_continuous_output_burst(
    isolated_runs_root, isolated_log_root
):
    """`tail` does not touch terminal modes, so Ctrl-C reaches it as a
    kernel-generated SIGINT rather than a byte its loop has to notice.
    Delivered directly here because the test PTY is not the subprocess's
    controlling terminal. It must still exit promptly mid-burst rather than
    deferring the interrupt until the backlog has drained."""
    env = _environment(isolated_runs_root, isolated_log_root)
    name = "tail-burst-sigint"
    state = _launch_interactive(name, _BURST_SCRIPT, env)
    log = isolated_log_root / name / "log"
    _await_backlog(log)
    process, master_fd = _spawn_tail(name, env)
    try:
        _drain_until_exit(process, master_fd, time.monotonic() + 0.5)
        assert process.poll() is None, "tail exited before SIGINT was sent"

        t0 = time.monotonic()
        process.send_signal(signal.SIGINT)
        _drain_until_exit(process, master_fd, t0 + _DETACH_BUDGET_SECONDS)
        assert (
            process.wait(timeout=max(0.1, t0 + _DETACH_BUDGET_SECONDS - time.monotonic()))
            == 128 + signal.SIGINT
        )
        elapsed = time.monotonic() - t0
        assert elapsed < _DETACH_BUDGET_SECONDS, f"exit took {elapsed:.2f}s"
        _assert_burst_still_live(log)
    finally:
        if process.poll() is None:
            process.terminate()
        os.close(master_fd)
        _stop_run(state)



def test_tail_leaves_terminal_modes_untouched(isolated_runs_root, isolated_log_root):
    """`tail` is read-only and must not put stdin into raw mode: a
    backgrounded `agent-run tail ... &` calling tcsetattr from a background
    process group gets SIGTTOU-stopped, and clearing OPOST staircases its
    own output for non-PTY runs."""
    env = _environment(isolated_runs_root, isolated_log_root)
    name = "tail-termios-untouched"
    # Log content to replay, so tail is past its first read (and so past any
    # mode change it would make) by the time the modes are compared.
    state = _launch_interactive(
        name, "import sys, time; sys.stdout.write('x' * 4096); sys.stdout.flush(); time.sleep(30)", env
    )
    process, master_fd = _spawn_tail(name, env)
    try:
        before = termios.tcgetattr(master_fd)
        # Wait until tail has actually replayed output: only then has it run
        # far enough to have changed terminal modes if it were going to.
        _read_until(master_fd, b"x" * 64, timeout=10.0)
        after = termios.tcgetattr(master_fd)
        assert after == before, "tail changed terminal modes"
        # ECHO and OPOST specifically: raw mode clears both, which is what
        # staircased tail's output and made a backgrounded tail SIGTTOU-stop.
        assert after[3] & termios.ECHO, "tail cleared ECHO"
        assert after[1] & termios.OPOST, "tail cleared OPOST"
        assert process.poll() is None
    finally:
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.terminate()
        os.close(master_fd)
        _stop_run(state)




@contextlib.contextmanager
def _fifo_readers(state: Path):
    """Hold the control FIFOs open for reading, as the runner's keeper does.
    Opening a FIFO write-only with no reader fails ENXIO."""
    fds = [os.open(str(state / fifo), os.O_RDWR) for fifo in ("stdin", "resize")]
    try:
        yield
    finally:
        for fd in fds:
            os.close(fd)


def _alive_until_first_loop_check():
    """`_pid_alive` reports True for the startup precheck, then False so
    cmd_attach's loop takes its normal agent-exited exit on the first pass."""
    calls = []

    def alive(_pid):
        calls.append(1)
        return len(calls) == 1

    return alive


def test_launch_unlinks_a_stale_fifo_before_recreating_it(
    isolated_runs_root, monkeypatch
):
    """_safe_rmtree returns without deleting when the state dir's inode no
    longer matches what was inspected, leaving the previous run's FIFOs in
    place. os.mkfifo then raises FileExistsError on relaunch unless the
    stale entry is unlinked first."""
    state = isolated_runs_root / "stale-fifo"
    state.mkdir()
    os.mkfifo(state / "stdin")
    os.mkfifo(state / "resize")
    # Stand in for the inode-mismatch/partial-teardown early return: the
    # directory (and its FIFOs) survives into the mkfifo path below.
    monkeypatch.setattr(agent_run, "_safe_rmtree", lambda *_a, **_k: None)

    assert (
        agent_run.cmd_launch(
            argparse.Namespace(
                name="stale-fifo",
                command=[sys.executable, "-c", "import time; time.sleep(30)"],
                interactive=True,
                prompt_file=None,
                echo=False,
                echo_interval=1.0,
                idle_timeout=None,
                submit_mode=None,
            )
        )
        == 0
    )
    try:
        # Reaching a running state at all is the assertion: without the
        # unlink, os.mkfifo raises FileExistsError and cmd_launch exits
        # "failed to create control fifo" instead.
        assert (state / "stdin").is_fifo()
        assert (state / "resize").is_fifo()
        assert _wait_until(
            lambda: (state / "status").read_text().strip() == "running"
        )
    finally:
        _stop_run(state)


def test_attach_survives_a_failing_tcsetattr_on_the_way_out(
    isolated_runs_root, isolated_log_root, monkeypatch
):
    """When the controlling terminal is gone (window closed, SSH dropped)
    the restoring `tcsetattr` raises. That must not abort cleanup before
    the DEC private modes the replayed PTY bytes enabled are reset, or the
    surviving terminal is left in alt-screen/mouse-tracking mode."""
    state = _seed_attachable_run(isolated_runs_root, "attach-tcsetattr-fails")
    (isolated_log_root / "attach-tcsetattr-fails").mkdir()
    (isolated_log_root / "attach-tcsetattr-fails" / "log").write_bytes(b"")

    def fail_tcsetattr(*_args):
        raise termios.error(errno.EBADF, "expected test failure")

    reset_calls = []
    monkeypatch.setattr(agent_run.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(agent_run.sys.stdin, "fileno", lambda: 0)
    monkeypatch.setattr(agent_run.termios, "tcgetattr", lambda _fd: "saved")
    monkeypatch.setattr(agent_run.termios, "tcsetattr", fail_tcsetattr)
    monkeypatch.setattr(agent_run.tty, "setraw", lambda _fd: None)
    monkeypatch.setattr(agent_run, "_reset_terminal_modes", lambda: reset_calls.append(1))
    monkeypatch.setattr(agent_run, "_pid_alive", _alive_until_first_loop_check())
    monkeypatch.setattr(agent_run.os, "get_terminal_size", lambda _fd: os.terminal_size((80, 24)))

    with _fifo_readers(state):
        assert agent_run.cmd_attach(argparse.Namespace(name="attach-tcsetattr-fails")) == 0
    assert reset_calls, "terminal-mode reset was skipped after tcsetattr failed"


@pytest.mark.parametrize("previous", [signal.SIG_DFL, signal.SIG_IGN, "handler"])
def test_attach_restores_the_previous_sigwinch_handler(
    isolated_runs_root, isolated_log_root, monkeypatch, previous
):
    """Whatever SIGWINCH disposition the caller had must come back. A
    `previous is not None` guard silently skips restoration for the
    SIG_DFL/None case, permanently leaving attach's own handler installed
    in whatever process called it."""
    state = _seed_attachable_run(isolated_runs_root, f"attach-winch-{previous}")
    (isolated_log_root / f"attach-winch-{previous}").mkdir()
    (isolated_log_root / f"attach-winch-{previous}" / "log").write_bytes(b"")
    monkeypatch.setattr(agent_run.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(agent_run.sys.stdin, "fileno", lambda: 0)
    monkeypatch.setattr(agent_run.termios, "tcgetattr", lambda _fd: "saved")
    monkeypatch.setattr(agent_run.termios, "tcsetattr", lambda *_a: None)
    monkeypatch.setattr(agent_run.tty, "setraw", lambda _fd: None)
    monkeypatch.setattr(agent_run, "_reset_terminal_modes", lambda: None)
    monkeypatch.setattr(agent_run, "_pid_alive", _alive_until_first_loop_check())
    monkeypatch.setattr(agent_run.os, "get_terminal_size", lambda _fd: os.terminal_size((80, 24)))

    if previous == "handler":
        previous = lambda _signum, _frame: None  # noqa: E731
    installed = signal.signal(signal.SIGWINCH, previous)
    try:
        with _fifo_readers(state):
            assert agent_run.cmd_attach(argparse.Namespace(name=state.name)) == 0
        assert signal.getsignal(signal.SIGWINCH) == previous
    finally:
        signal.signal(signal.SIGWINCH, installed)


def test_attach_restores_sigwinch_when_python_cannot_represent_the_handler(
    isolated_runs_root, isolated_log_root, monkeypatch
):
    """signal.signal returns None when the previous disposition was set
    outside Python and has no Python-level representation. Guarding
    restoration on `previous is not None` skips it entirely in that case,
    leaving attach's own handler installed in the calling process forever;
    SIG_DFL is the correct fallback because signal.signal rejects None."""
    state = _seed_attachable_run(isolated_runs_root, "attach-winch-none")
    (isolated_log_root / "attach-winch-none").mkdir()
    (isolated_log_root / "attach-winch-none" / "log").write_bytes(b"")
    monkeypatch.setattr(agent_run.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(agent_run.sys.stdin, "fileno", lambda: 0)
    monkeypatch.setattr(agent_run.termios, "tcgetattr", lambda _fd: "saved")
    monkeypatch.setattr(agent_run.termios, "tcsetattr", lambda *_a: None)
    monkeypatch.setattr(agent_run.tty, "setraw", lambda _fd: None)
    monkeypatch.setattr(agent_run, "_reset_terminal_modes", lambda: None)
    monkeypatch.setattr(agent_run, "_pid_alive", _alive_until_first_loop_check())
    monkeypatch.setattr(
        agent_run.os, "get_terminal_size", lambda _fd: os.terminal_size((80, 24))
    )

    restored = []
    real_signal = signal.signal

    def fake_signal(signum, handler):
        if signum != signal.SIGWINCH:
            return real_signal(signum, handler)
        restored.append(handler)
        return None

    monkeypatch.setattr(agent_run.signal, "signal", fake_signal)

    with _fifo_readers(state):
        assert agent_run.cmd_attach(argparse.Namespace(name="attach-winch-none")) == 0

    assert restored[-1] is signal.SIG_DFL, "SIGWINCH was left pointing at attach"


def test_append_bounded_discards_an_oversized_chunk_with_a_warning():
    """A paste past MAX_PTY_INPUT_BUFFER used to raise BufferError, dumping
    a traceback over the user's terminal mid-session. Report and drop the
    chunk instead."""
    warnings = []
    oversized = b"x" * (agent_run.MAX_PTY_INPUT_BUFFER + 1)

    result = agent_run._append_bounded(b"kept", oversized, warnings.append)

    assert result == b"kept"
    assert len(warnings) == 1
    assert str(agent_run.MAX_PTY_INPUT_BUFFER) in warnings[0]


def test_append_bounded_keeps_a_chunk_that_fits():
    warnings = []
    assert agent_run._append_bounded(b"ab", b"cd", warnings.append) == b"abcd"
    assert warnings == []


def test_flush_fifo_write_bounds_each_write_to_pipe_buf():
    """Writes larger than PIPE_BUF are not atomic, so two attach clients
    sharing the stdin FIFO would interleave mid-escape-sequence. Each write
    is capped to keep the kernel's atomicity guarantee."""
    read_fd, write_fd = os.pipe()
    try:
        os.set_blocking(write_fd, False)
        payload = b"z" * (agent_run._FIFO_ATOMIC_WRITE_BYTES * 3)

        written = agent_run._flush_fifo_write(write_fd, payload)

        assert written == agent_run._FIFO_ATOMIC_WRITE_BYTES
        assert len(os.read(read_fd, len(payload))) == written
    finally:
        os.close(read_fd)
        os.close(write_fd)


def test_flush_fifo_write_reports_zero_when_the_reader_is_backpressured():
    read_fd, write_fd = os.pipe()
    try:
        os.set_blocking(write_fd, False)
        while agent_run._flush_fifo_write(write_fd, b"z" * 4096):
            pass

        assert agent_run._flush_fifo_write(write_fd, b"more") == 0
    finally:
        os.close(read_fd)
        os.close(write_fd)


def test_drain_fifo_write_completes_a_short_write():
    """Bytes typed before Ctrl-D are already committed to the agent, so the
    detach path must keep writing past the first PIPE_BUF-sized chunk
    instead of dropping the remainder."""
    read_fd, write_fd = os.pipe()
    payload = b"z" * (agent_run._FIFO_ATOMIC_WRITE_BYTES * 2)
    received = bytearray()

    def reader():
        while len(received) < len(payload):
            received.extend(os.read(read_fd, 4096))

    try:
        os.set_blocking(write_fd, False)
        thread = threading.Thread(target=reader)
        thread.start()

        assert agent_run._drain_fifo_write(write_fd, payload) == b""
        thread.join(timeout=5)
        assert bytes(received) == payload
    finally:
        os.close(read_fd)
        os.close(write_fd)


def test_drain_fifo_write_gives_up_on_a_stalled_reader(monkeypatch):
    """Detach must never hang on a backpressured agent: undeliverable bytes
    are returned rather than blocking the exit path."""
    monkeypatch.setattr(agent_run, "_DETACH_FLUSH_TIMEOUT_SECONDS", 0.1)
    read_fd, write_fd = os.pipe()
    try:
        os.set_blocking(write_fd, False)
        while agent_run._flush_fifo_write(write_fd, b"z" * 4096):
            pass

        started = time.monotonic()
        assert agent_run._drain_fifo_write(write_fd, b"undeliverable") == b"undeliverable"
        assert time.monotonic() - started < 2.0
    finally:
        os.close(read_fd)
        os.close(write_fd)


def test_attach_forwards_a_large_paste_without_losing_bytes(
    isolated_runs_root, isolated_log_root
):
    """End-to-end guard against the two paste bugs together: a bracketed
    paste bigger than one 4096-byte read must reach the agent byte-for-byte,
    with the paste markers intact and no spurious detach on the 0x04 byte
    embedded in the payload."""
    env = _environment(isolated_runs_root, isolated_log_root)
    name = "attach-large-paste"
    payload = ("line-%04d\n" % 0).join(f"seg{i:04d}\x04" for i in range(400))
    pasted = "\x1b[200~" + payload + "\x1b[201~"
    # Raw mode on the agent's own PTY: without it the kernel treats the 0x04
    # bytes inside the paste as EOF (VEOF) and ICRNL rewrites the newlines,
    # so the comparison would measure the tty discipline, not the forwarding.
    script = """\
import os, sys, termios, tty
total = %d
fd = sys.stdin.fileno()
tty.setraw(fd)
seen = b""
sys.stdout.write("READY\\n")
sys.stdout.flush()
while len(seen) < total:
    chunk = os.read(fd, 65536)
    if not chunk:
        break
    seen += chunk
sys.stdout.write("GOT %%d %%d\\n" %% (len(seen), seen.count(b"\\x1b[200~")))
sys.stdout.flush()
import time; time.sleep(30)
""" % len(pasted.encode())
    state = _launch_interactive(name, script, env)
    process, master_fd = _spawn_attach(name, env, rows=24, cols=80)
    try:
        _read_until(master_fd, b"READY")
        for offset in range(0, len(pasted), 512):
            os.write(master_fd, pasted[offset : offset + 512].encode())
            time.sleep(0.01)
        output = _read_until(master_fd, b"GOT ", timeout=15.0)
        report = output.split(b"GOT ")[-1].split(b"\n")[0].split()
        assert int(report[0]) == len(pasted.encode()), "paste was truncated"
        assert int(report[1]) == 1, "paste-start marker was lost"
        assert process.poll() is None, "attach detached on a 0x04 inside the paste"
    finally:
        _detach(process, master_fd)
        _stop_run(state)


def test_tail_exits_instead_of_busy_spinning_when_its_terminal_peer_goes_away(
    isolated_runs_root, isolated_log_root
):
    """Closing the PTY master (terminal window shut, SSH dropped) makes
    every stdout write fail with EIO. `tail` used to keep looping on that,
    burning a core until the agent died; it must exit instead."""
    env = _environment(isolated_runs_root, isolated_log_root)
    name = "tail-peer-gone"
    state = _launch_interactive(name, _BURST_SCRIPT, env)
    _await_backlog(isolated_log_root / name / "log")
    # stderr on a pipe, not the PTY: the exit status and any traceback are
    # the whole point of this test, and a PTY-bound stderr dies with the
    # terminal we are about to close.
    process, master_fd = _spawn_tail(name, env, stderr=subprocess.PIPE)
    try:
        _drain_until_exit(process, master_fd, time.monotonic() + 0.5)
        assert process.poll() is None

        os.close(master_fd)
        master_fd = -1
        process.wait(timeout=5)
        stderr = process.stderr.read() if process.stderr else b""
        assert process.returncode == 0, (
            f"tail exited {process.returncode} on a dead terminal; "
            f"stderr={stderr[:400]!r}"
        )
        assert b"Traceback" not in stderr, f"tail dumped a traceback: {stderr[:400]!r}"
    finally:
        if process.poll() is None:
            process.terminate()
        if process.stderr:
            process.stderr.close()
        if master_fd != -1:
            os.close(master_fd)
        _stop_run(state)


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGHUP, signal.SIGQUIT])
def test_attach_restores_terminal_modes_on_signal_death(
    isolated_runs_root, isolated_log_root, sig
):
    """SIGTERM/SIGHUP/SIGQUIT kill the process outright, so neither the
    `finally` block nor the KeyboardInterrupt path runs. Without explicit
    handlers the terminal is left in raw mode with whatever DEC private
    modes the replayed PTY bytes enabled still on."""
    env = _environment(isolated_runs_root, isolated_log_root)
    name = f"attach-signal-{sig}"
    state = _launch_interactive(name, "import time; time.sleep(30)", env)
    process, master_fd = _spawn_attach(name, env, rows=24, cols=80)
    try:
        assert _wait_until(
            lambda: not (termios.tcgetattr(master_fd)[3] & termios.ECHO), timeout=5.0
        ), "attach never established raw mode"

        process.send_signal(sig)
        # Popen reports death by signal N as -N; the exit status must still
        # be signal death, not a swallowed normal exit.
        assert process.wait(timeout=5) == -sig
        assert _wait_until(
            lambda: bool(termios.tcgetattr(master_fd)[3] & termios.ECHO), timeout=5.0
        ), "terminal left in raw mode after signal death"
    finally:
        if process.poll() is None:
            process.terminate()
        os.close(master_fd)
        _stop_run(state)


def test_escape_hold_timeout_tolerates_ssh_latency():
    """A CSI Ctrl-D split across two reads is only reassembled if the first
    fragment is still held when the second arrives. At 50ms the hold
    expired under real SSH round-trip latency, releasing the fragment as
    literal input -- the exact leak-through the detector prevents."""
    assert agent_run._ESCAPE_HOLD_TIMEOUT_SECONDS >= 0.2


def test_release_expired_held_escape_keeps_the_prefix_before_the_deadline():
    held, deadline, released = agent_run._release_expired_held_escape(
        b"\x1b[27;5", time.monotonic() + agent_run._ESCAPE_HOLD_TIMEOUT_SECONDS
    )
    assert held == b"\x1b[27;5"
    assert deadline is not None
    assert released == b""


def test_release_expired_held_escape_releases_after_the_deadline():
    held, deadline, released = agent_run._release_expired_held_escape(
        b"\x1b", time.monotonic() - 0.001
    )
    assert held == b""
    assert deadline is None
    assert released == b"\x1b"


def test_drain_resize_records_recovers_from_a_replaced_partial_record(monkeypatch):
    """attach replaces a partially-written record when a new SIGWINCH
    arrives rather than completing it, so the orphaned prefix stays on the
    wire ahead of a full record. The reader must resynchronize and apply
    the newer size, not decode the fragment plus the next record's head."""
    applied = []
    monkeypatch.setattr(
        agent_run,
        "_apply_resize",
        lambda master_fd, cols, rows: applied.append((cols, rows)),
    )
    orphaned_prefix = agent_run._pack_resize(100, 30)[:2]

    leftover = agent_run._drain_resize_records(
        7, orphaned_prefix + agent_run._pack_resize(120, 42)
    )

    assert leftover == b""
    assert applied == [(120, 42)]


def test_detach_flushes_typed_input_through_to_the_agent(
    isolated_runs_root, isolated_log_root
):
    """Bytes typed before Ctrl-D are already committed to the agent, and a
    single PIPE_BUF-sized write is not enough for a large pending buffer.
    Drives the real attach loop and has the wrapped agent report how much of
    a >PIPE_BUF burst actually arrived."""
    env = _environment(isolated_runs_root, isolated_log_root)
    name = "attach-detach-flush"
    typed = b"t" * (agent_run._FIFO_ATOMIC_WRITE_BYTES * 3)
    script = """\
import os, sys, termios, tty
fd = sys.stdin.fileno()
tty.setraw(fd)
seen = b""
sys.stdout.write("READY\\n")
sys.stdout.flush()
while len(seen) < %d:
    chunk = os.read(fd, 65536)
    if not chunk:
        break
    seen += chunk
sys.stdout.write("GOT %%d\\n" %% len(seen))
sys.stdout.flush()
import time; time.sleep(30)
""" % len(typed)
    state = _launch_interactive(name, script, env)
    process, master_fd = _spawn_attach(name, env, rows=24, cols=80)
    try:
        _read_until(master_fd, b"READY")
        # Type a burst, then detach in the same breath: everything before
        # the \x04 must still reach the agent.
        os.write(master_fd, typed)
        time.sleep(0.3)
        os.write(master_fd, b"\x04")

        output = _read_until(master_fd, b"GOT ", timeout=15.0)
        got = int(output.split(b"GOT ")[-1].split(b"\n")[0])
        assert got == len(typed), f"detach delivered {got} of {len(typed)} typed bytes"
        assert process.wait(timeout=5) == 0
    finally:
        if process.poll() is None:
            process.terminate()
        os.close(master_fd)
        _stop_run(state)


def test_drain_fifo_write_keeps_writing_past_one_pipe_buf_chunk():
    """The detach flush must loop until the buffer is delivered. A single
    PIPE_BUF-sized write silently drops the rest of what the user typed --
    with a 3x PIPE_BUF buffer against a reader that drains in small steps,
    exactly 2/3 goes missing."""
    payload = b"p" * (agent_run._FIFO_ATOMIC_WRITE_BYTES * 3)
    read_fd, write_fd = os.pipe()
    received = bytearray()
    done = threading.Event()

    def slow_reader():
        # Small reads with pauses: the writer has to come back repeatedly
        # rather than emptying the pipe in one go.
        while len(received) < len(payload) and not done.is_set():
            readable, _, _ = select.select([read_fd], [], [], 0.05)
            if readable:
                received.extend(os.read(read_fd, 512))

    try:
        os.set_blocking(write_fd, False)
        # Pre-fill so the first write cannot take the whole payload.
        try:
            while True:
                os.write(write_fd, b"x" * 4096)
        except BlockingIOError:
            pass
        thread = threading.Thread(target=slow_reader, daemon=True)
        thread.start()

        undelivered = agent_run._drain_fifo_write(write_fd, payload)

        done.set()
        thread.join(timeout=5)
        assert undelivered == b"", (
            f"{len(undelivered)} of {len(payload)} bytes never written"
        )
    finally:
        done.set()
        os.close(read_fd)
        os.close(write_fd)


def test_drain_fifo_write_uses_its_full_budget_before_giving_up(monkeypatch):
    """A zero budget means the flush never really happens. The loop must
    keep trying for the configured window against a reader that is briefly
    backpressured."""
    payload = b"b" * (agent_run._FIFO_ATOMIC_WRITE_BYTES * 2)
    read_fd, write_fd = os.pipe()
    received = bytearray()
    done = threading.Event()

    def delayed_reader():
        # Nothing is readable for the first 0.4s: a flush that gives up
        # immediately delivers nothing at all.
        time.sleep(0.4)
        while len(received) < len(payload) and not done.is_set():
            readable, _, _ = select.select([read_fd], [], [], 0.05)
            if readable:
                received.extend(os.read(read_fd, 4096))

    try:
        os.set_blocking(write_fd, False)
        try:
            while True:
                os.write(write_fd, b"x" * 4096)
        except BlockingIOError:
            pass
        thread = threading.Thread(target=delayed_reader, daemon=True)
        thread.start()

        undelivered = agent_run._drain_fifo_write(write_fd, payload)

        done.set()
        thread.join(timeout=5)
        assert undelivered == b"", (
            f"gave up with {len(undelivered)} bytes undelivered before the budget"
        )
    finally:
        done.set()
        os.close(read_fd)
        os.close(write_fd)


def test_detach_warns_about_input_it_could_not_deliver(
    isolated_runs_root, isolated_log_root, monkeypatch, capsys
):
    """When the agent is not reading, detach must report the loss rather
    than discarding it silently. Exercises cmd_attach itself so the warning
    text, the byte count, and the call site are all covered."""
    state = _seed_attachable_run(isolated_runs_root, "attach-detach-warn")
    (isolated_log_root / "attach-detach-warn").mkdir()
    (isolated_log_root / "attach-detach-warn" / "log").write_bytes(b"")
    monkeypatch.setattr(agent_run, "_DETACH_FLUSH_TIMEOUT_SECONDS", 0.1)
    monkeypatch.setattr(agent_run.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(agent_run.sys.stdin, "fileno", lambda: 0)
    monkeypatch.setattr(agent_run.termios, "tcgetattr", lambda _fd: "saved")
    monkeypatch.setattr(agent_run.termios, "tcsetattr", lambda *_a: None)
    monkeypatch.setattr(agent_run.tty, "setraw", lambda _fd: None)
    monkeypatch.setattr(agent_run, "_reset_terminal_modes", lambda: None)
    monkeypatch.setattr(
        agent_run.os, "get_terminal_size", lambda _fd: os.terminal_size((80, 24))
    )
    monkeypatch.setattr(agent_run, "_pid_alive", lambda _pid: True)

    typed = b"q" * 4096
    reads = [typed + b"\x04"]

    def fake_read(fd, _n):
        return reads.pop(0) if reads else b""

    monkeypatch.setattr(agent_run.os, "read", fake_read)
    monkeypatch.setattr(
        agent_run.select, "select", lambda r, w, x, t=None: (list(r), [], [])
    )

    # Hold the FIFOs open, then fill the stdin FIFO so attach's flush cannot
    # deliver anything and has to report the shortfall.
    stdin_fd = os.open(str(state / "stdin"), os.O_RDWR | os.O_NONBLOCK)
    resize_fd = os.open(str(state / "resize"), os.O_RDWR | os.O_NONBLOCK)
    try:
        try:
            while True:
                os.write(stdin_fd, b"f" * 4096)
        except BlockingIOError:
            pass

        assert agent_run.cmd_attach(argparse.Namespace(name="attach-detach-warn")) == 0
    finally:
        os.close(stdin_fd)
        os.close(resize_fd)

    err = capsys.readouterr().err
    assert "undelivered" in err, f"detach did not report the loss: {err!r}"
    assert str(len(typed)) in err, f"warning omitted the byte count: {err!r}"


def test_attach_sends_one_resize_record_per_distinct_size(
    isolated_runs_root, isolated_log_root
):
    """SIGWINCH fires for reasons other than a real resize, and the PTY is
    last-writer-wins across attached clients, so resending an unchanged size
    drags every other client's view back to this one's. Reads the resize
    FIFO directly and counts records: redundant SIGWINCHes must add none.
    """
    state = _seed_attachable_run(isolated_runs_root, "attach-winsize-dedup")
    (isolated_log_root / "attach-winsize-dedup").mkdir()
    (isolated_log_root / "attach-winsize-dedup" / "log").write_bytes(b"")
    env = _environment(isolated_runs_root, isolated_log_root)

    # Hold both FIFOs open so attach can open them write-only, and read the
    # resize FIFO to count what the runner would have applied.
    stdin_fd = os.open(str(state / "stdin"), os.O_RDWR | os.O_NONBLOCK)
    resize_fd = os.open(str(state / "resize"), os.O_RDWR | os.O_NONBLOCK)
    master_fd, slave_fd = pty.openpty()
    fcntl.ioctl(slave_fd, termios.TIOCSWINSZ, struct.pack("HHHH", 24, 80, 0, 0))
    process = subprocess.Popen(
        [sys.executable, "-m", "toolbox.agent_run", "attach", "attach-winsize-dedup"],
        stdin=slave_fd, stdout=slave_fd, stderr=slave_fd,
        env=env, start_new_session=True, close_fds=True,
    )
    os.close(slave_fd)

    def drain_records() -> list:
        out = []
        while True:
            try:
                chunk = os.read(resize_fd, 4096)
            except BlockingIOError:
                break
            if not chunk:
                break
            for i in range(0, len(chunk) - agent_run.RESIZE_RECORD_SIZE + 1,
                           agent_run.RESIZE_RECORD_SIZE):
                _m, _v, c, r, _sum = struct.unpack(
                    agent_run.RESIZE_RECORD_FORMAT,
                    chunk[i:i + agent_run.RESIZE_RECORD_SIZE],
                )
                out.append((c, r))
        return out

    try:
        # The initial size is sent on connect; consume it before measuring.
        assert _wait_until(lambda: bool(drain_records()), timeout=10.0), (
            "attach never sent its initial size"
        )
        time.sleep(0.3)
        drain_records()

        # Redundant SIGWINCHes at an unchanged size must produce no records.
        for _ in range(3):
            os.kill(process.pid, signal.SIGWINCH)
            time.sleep(0.1)
        assert drain_records() == [], "resent an unchanged size"

        # A genuine change still propagates.
        fcntl.ioctl(master_fd, termios.TIOCSWINSZ, struct.pack("HHHH", 40, 100, 0, 0))
        os.kill(process.pid, signal.SIGWINCH)
        assert _wait_until(
            lambda: (100, 40) in drain_records(), timeout=10.0
        ), "a real resize after redundant SIGWINCHes was dropped"
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)
        os.close(master_fd)
        os.close(stdin_fd)
        os.close(resize_fd)


def test_attach_returns_zero_when_its_terminal_peer_goes_away(
    isolated_runs_root, isolated_log_root
):
    """attach writes replayed PTY bytes to a terminal that can vanish
    mid-session (window closed, SSH dropped). An unguarded write raises
    OSError/EIO out of the loop; the guarded write returns through the
    normal cleanup path so the exit status stays 0 rather than becoming a
    traceback."""
    env = _environment(isolated_runs_root, isolated_log_root)
    name = "attach-peer-gone"
    state = _launch_interactive(name, _BURST_SCRIPT, env)
    log = isolated_log_root / name / "log"
    _await_backlog(log)
    # stderr on a pipe: _spawn_attach's default points it at the PTY we are
    # about to close, so a traceback written there would be unobservable.
    process, master_fd = _spawn_attach(
        name, env, rows=24, cols=80, stderr=subprocess.PIPE
    )
    try:
        _await_raw_mode(process, master_fd)

        os.close(master_fd)
        master_fd = -1
        process.wait(timeout=10)
        stderr = process.stderr.read() if process.stderr else b""
        assert process.returncode == 0, (
            f"attach exited {process.returncode}; stderr={stderr[:400]!r}"
        )
        assert b"Traceback" not in stderr, f"attach dumped a traceback: {stderr[:400]!r}"
    finally:
        if process.poll() is None:
            process.terminate()
        if process.stderr:
            process.stderr.close()
        if master_fd != -1:
            os.close(master_fd)
        _stop_run(state)


def test_scan_local_input_gives_up_on_a_paste_that_never_closes():
    """An ESC[201~ that never arrives (paste aborted mid-flight, terminal
    reset, an emulator that drops the closer) must not latch the in-paste
    state on forever. Ctrl-D is the only documented way out of attach, so a
    stuck latch strands the user in a raw-mode terminal with no in-band
    escape and silently feeds the \\x04 to the agent instead."""
    _f, _h, detached, in_paste, paste_bytes, _ps, _st = agent_run._scan_local_input_for_detach(
        b"\x1b[200~" + b"x" * 64
    )
    assert in_paste and not detached

    # Push past the latch budget without ever sending the closing marker.
    _f, _h, detached, in_paste, paste_bytes, _ps, _st = agent_run._scan_local_input_for_detach(
        b"y" * (agent_run._MAX_PASTE_BYTES + 1), in_paste, paste_bytes
    )
    assert not in_paste, "in-paste latch never released without ESC[201~"

    _f, _h, detached, _ip, _pb, _ps, _st = agent_run._scan_local_input_for_detach(
        b"\x04", in_paste, paste_bytes
    )
    assert detached, "Ctrl-D detach permanently wedged by an unclosed paste"


def test_scan_local_input_keeps_a_genuine_large_paste_bracketed():
    """The latch budget must not cut a real paste short: a payload well
    under the bound stays in-paste, so an embedded 0x04 is still content."""
    payload = b"z" * (agent_run._MAX_PASTE_BYTES // 2)
    _f, _h, _d, in_paste, paste_bytes, _ps, _st = agent_run._scan_local_input_for_detach(
        b"\x1b[200~" + payload
    )
    assert in_paste

    _f, _h, detached, in_paste, _pb, _ps, _st = agent_run._scan_local_input_for_detach(
        b"\x04still pasting", in_paste, paste_bytes
    )
    assert not detached, "a genuine large paste was cut short by the latch"
    assert in_paste


@pytest.mark.parametrize("split", [1, 2, 3, 4, 5])
def test_release_or_resync_preserves_a_split_paste_start_marker(split):
    """ESC[200~ split across the hold deadline must still be recognized.
    Releasing the prefix as literal input destroys the marker: the next
    read starts mid-sequence, in_paste stays False, and any 0x04 in the
    payload detaches mid-paste -- reproducing round 1's Critical #2.

    Two mechanisms cover this between them: distinctive prefixes (ESC[2 and
    longer) are held past the ordinary deadline, and the two shorter ones
    are rejoined on the next read so Escape and the arrows stay responsive.
    """
    marker = agent_run._PASTE_START
    prefix, rest = marker[:split], marker[split:]
    payload = rest + b"pasted\x04text"

    held, _deadline, released = agent_run._release_expired_held_escape(
        prefix, time.monotonic() - 0.001
    )
    if not released:
        # Held past the deadline: the next read merges it back itself.
        assert held == prefix
        data = prefix + payload
    else:
        data, rejoined = agent_run._resync_paste_marker(payload, released)
        assert rejoined, "released prefix was not resynced"
        assert data.startswith(marker)

    _f, _h, detached, in_paste, _pb, _ps, _st = agent_run._scan_local_input_for_detach(data)
    assert in_paste, f"split at {split} lost the paste bracket"
    assert not detached, f"split at {split} caused a spurious mid-paste detach"


def test_release_expired_held_escape_still_frees_the_escape_key_promptly():
    """The paste-marker hold must not make Escape or the arrow keys feel
    stuck: neither is a distinctive marker prefix, so both release on the
    ordinary escape deadline."""
    for keystroke in (b"\x1b", b"\x1b["):
        _held, _deadline, released = agent_run._release_expired_held_escape(
            keystroke, time.monotonic() - 0.001
        )
        assert released == keystroke, f"{keystroke!r} was held past its deadline"


def test_resync_paste_marker_leaves_unrelated_input_alone():
    assert agent_run._resync_paste_marker(b"xyz", b"\x1b") == (b"xyz", False)
    assert agent_run._resync_paste_marker(b"Azz", b"\x1b[") == (b"Azz", False)
    assert agent_run._resync_paste_marker(b"abc", b"") == (b"abc", False)


@pytest.mark.parametrize(
    "err,peer_gone",
    [
        (errno.EPIPE, True),
        (errno.EIO, True),
        (errno.ENOSPC, False),
        (errno.EAGAIN, False),
        (errno.EFBIG, False),
    ],
)
def test_is_peer_gone_only_accepts_a_dead_terminal(err, peer_gone):
    """EPIPE/EIO mean the terminal went away. ENOSPC and EAGAIN are also
    OSError subclasses, but treating them as peer-gone exits 0 with
    silently truncated output where the shell expects a failure."""
    assert agent_run._is_peer_gone(OSError(err, "test")) is peer_gone


def test_tail_propagates_a_write_failure_that_is_not_a_dead_terminal(
    isolated_runs_root, isolated_log_root
):
    """`agent-run tail run > out` on a filling disk, or into a full
    non-blocking pipe, must not exit 0 with a truncated file. Round 1's
    peer-gone fix caught bare OSError, which swallowed both."""
    env = _environment(isolated_runs_root, isolated_log_root)
    name = "tail-write-fails"
    (isolated_runs_root / name).mkdir()
    (isolated_runs_root / name / "pid").write_text("1\n")
    (isolated_log_root / name).mkdir()
    (isolated_log_root / name / "log").write_bytes(b"x" * 3_000_000)

    read_fd, write_fd = os.pipe()
    try:
        os.set_blocking(write_fd, False)
        # Fill the pipe so every further write fails with EAGAIN.
        try:
            while True:
                os.write(write_fd, b"z" * 65536)
        except BlockingIOError:
            pass

        result = subprocess.run(
            [sys.executable, "-m", "toolbox.agent_run", "tail", name],
            stdout=write_fd,
            stderr=subprocess.PIPE,
            env=env,
            timeout=30,
        )
        assert result.returncode != 0, (
            "tail reported success despite failing to write its output"
        )
    finally:
        os.close(read_fd)
        os.close(write_fd)


def test_drain_resize_records_survives_a_167_column_terminal(monkeypatch):
    """0xA7 is both the framing magic byte and column 167, so at that width
    the magic reappears inside the payload. Anchoring on the magic alone
    mis-frames every orphan offset and never recovers, driving the wrapped
    agent's PTY to a garbage size permanently."""
    applied = []
    monkeypatch.setattr(
        agent_run, "_apply_resize", lambda fd, c, r: applied.append((c, r))
    )
    record = agent_run._pack_resize(167, 24)

    for orphan_len in range(1, agent_run.RESIZE_RECORD_SIZE):
        applied.clear()
        leftover = agent_run._drain_resize_records(7, record[:orphan_len] + record)
        assert applied == [(167, 24)], (
            f"orphan of {orphan_len}B mis-framed a 167-column record: {applied}"
        )
        assert leftover == b""


def test_drain_resize_records_never_applies_an_out_of_range_size(monkeypatch):
    """Adversarial bytes must be rejected, not decoded. Covers the in-range
    dimension check, which a mutation can otherwise remove undetected."""
    applied = []
    monkeypatch.setattr(
        agent_run, "_apply_resize", lambda fd, c, r: applied.append((c, r))
    )
    rng = random.Random(20260815)

    for _ in range(20000):
        blob = bytes(rng.getrandbits(8) for _ in range(rng.randint(1, 14)))
        applied.clear()
        agent_run._drain_resize_records(7, blob)
        for cols, rows in applied:
            assert 1 <= cols <= 0xFFFF and 1 <= rows <= 0xFFFF, (
                f"applied out-of-range size {(cols, rows)} from {blob.hex()}"
            )


@pytest.mark.parametrize("cols,rows", [(0, 24), (167, 0), (0, 0)])
def test_drain_resize_records_rejects_a_zero_dimension_record(monkeypatch, cols, rows):
    """A record carrying a zero dimension but an otherwise valid checksum
    must be rejected: _pack_resize refuses to build one, so seeing it means
    the stream is corrupt, and TIOCSWINSZ with a 0 axis blanks the agent's
    PTY."""
    applied = []
    monkeypatch.setattr(
        agent_run, "_apply_resize", lambda fd, c, r: applied.append((c, r))
    )
    forged = struct.pack(
        agent_run.RESIZE_RECORD_FORMAT,
        agent_run.RESIZE_RECORD_MAGIC,
        agent_run.RESIZE_RECORD_VERSION,
        cols,
        rows,
        agent_run._resize_checksum(cols, rows),
    )

    agent_run._drain_resize_records(7, forged)

    assert applied == [], f"applied a zero-dimension size {(cols, rows)}"


def test_drain_resize_records_rejects_what_a_one_byte_resync_would_accept(monkeypatch):
    """Differential guard on resync granularity: a validator that merely
    steps one byte at a time without checking the checksum accepts garbage
    windows. This is the concrete string that distinguishes them."""
    applied = []
    monkeypatch.setattr(
        agent_run, "_apply_resize", lambda fd, c, r: applied.append((c, r))
    )

    agent_run._drain_resize_records(
        7, bytes.fromhex("4601a718a70000180026a7180000a7a7ff00005018c6e1")
    )

    assert applied == [], f"accepted a mis-framed window: {applied}"


def test_valid_resize_record_rejects_a_shifted_window():
    """The checksum must be position-sensitive: an XOR that folds byte
    values together validates a window shifted inside a 167-column record."""
    record = agent_run._pack_resize(167, 24)
    doubled = record + record

    for offset in range(1, agent_run.RESIZE_RECORD_SIZE):
        window = doubled[offset : offset + agent_run.RESIZE_RECORD_SIZE]
        assert not agent_run._valid_resize_record(window), (
            f"window at offset {offset} validated: {window.hex(' ')}"
        )


def test_legitimate_resize_streams_always_apply_the_final_size(monkeypatch):
    """Whole-stream property: whatever the read chunking, the last complete
    record in a well-formed stream is the size that gets applied."""
    applied = []
    monkeypatch.setattr(
        agent_run, "_apply_resize", lambda fd, c, r: applied.append((c, r))
    )
    rng = random.Random(884422)

    for _ in range(2000):
        sizes = [
            (rng.randint(1, 300), rng.randint(1, 100))
            for _ in range(rng.randint(1, 4))
        ]
        stream = b"".join(agent_run._pack_resize(c, r) for c, r in sizes)
        applied.clear()
        buffered = b""
        pos = 0
        while pos < len(stream):
            step = rng.randint(1, 7)
            buffered = agent_run._drain_resize_records(
                7, buffered + stream[pos : pos + step]
            )
            pos += step
        assert applied[-1] == sizes[-1], f"final size wrong for {sizes}"


def test_flush_fifo_write_matches_the_platform_pipe_buf():
    """_FIFO_ATOMIC_WRITE_BYTES must track the real PIPE_BUF, not a
    hardcoded guess: the atomicity guarantee that keeps two attach clients
    from tearing each other's escape sequences only holds up to that size.
    """
    assert agent_run._FIFO_ATOMIC_WRITE_BYTES == select.PIPE_BUF

    read_fd, write_fd = os.pipe()
    try:
        expected = os.fpathconf(write_fd, "PC_PIPE_BUF")
        assert agent_run._FIFO_ATOMIC_WRITE_BYTES == expected
    finally:
        os.close(read_fd)
        os.close(write_fd)


def test_attach_resumes_a_short_stdin_fifo_write(monkeypatch):
    """A short write on the stdin FIFO must leave the remainder pending
    rather than dropping it; the next writable pass continues from there."""
    read_fd, write_fd = os.pipe()
    try:
        os.set_blocking(write_fd, False)
        payload = b"s" * (agent_run._FIFO_ATOMIC_WRITE_BYTES * 2)

        first = agent_run._flush_fifo_write(write_fd, payload)
        assert 0 < first <= agent_run._FIFO_ATOMIC_WRITE_BYTES
        remaining = payload[first:]
        assert remaining, "nothing left pending after a bounded write"

        drained = os.read(read_fd, len(payload))
        second = agent_run._flush_fifo_write(write_fd, remaining)
        assert second > 0, "pending remainder was never resumed"
        assert drained + os.read(read_fd, len(payload)) == payload[: first + second]
    finally:
        os.close(read_fd)
        os.close(write_fd)


def test_terminal_restored_on_signal_death_restores_on_normal_exit():
    """The context manager must put the previous dispositions back when the
    body returns normally, not only when a signal arrives -- otherwise it
    leaks its handlers into whatever runs next in the same process."""
    before = {
        sig: signal.getsignal(sig) for sig in agent_run._TERMINAL_DEATH_SIGNALS
    }
    calls = []

    with agent_run._terminal_restored_on_signal_death(lambda: calls.append(1)):
        for sig in agent_run._TERMINAL_DEATH_SIGNALS:
            assert signal.getsignal(sig) not in (before[sig], signal.SIG_DFL), (
                f"{sig} handler was not installed"
            )

    for sig in agent_run._TERMINAL_DEATH_SIGNALS:
        assert signal.getsignal(sig) == before[sig], f"{sig} not restored on exit"
    assert calls == [], "restore ran without a signal"


def test_terminal_restored_on_signal_death_survives_a_failing_restore():
    """An exception from the restore callback must not cancel the death
    signal: the process still has to die from it, or a SIGTERM silently
    becomes a no-op."""
    def boom():
        raise RuntimeError("restore failed")

    code = textwrap.dedent(
        """
        import os, signal, sys
        sys.path.insert(0, %r)
        from toolbox import agent_run
        def boom():
            raise RuntimeError("restore failed")
        with agent_run._terminal_restored_on_signal_death(boom):
            os.kill(os.getpid(), signal.SIGTERM)
            signal.pause()
        """
    ) % str(Path(agent_run.__file__).parent.parent)

    result = subprocess.run([sys.executable, "-c", code], capture_output=True, timeout=30)
    assert result.returncode == -signal.SIGTERM, (
        f"process survived SIGTERM (rc={result.returncode}); "
        f"stderr={result.stderr[:300]!r}"
    )


def test_tail_discards_unflushable_stdout_instead_of_exiting_120(
    isolated_runs_root, isolated_log_root
):
    """CPython flushes stdout again at interpreter shutdown. Without
    redirecting the dead fd first, that second failure turns tail's clean
    exit into status 120 with a message on stderr."""
    env = _environment(isolated_runs_root, isolated_log_root)
    name = "tail-shutdown-flush"
    state = _launch_interactive(name, _BURST_SCRIPT, env)
    _await_backlog(isolated_log_root / name / "log")
    process, master_fd = _spawn_tail(name, env, stderr=subprocess.PIPE)
    try:
        _drain_until_exit(process, master_fd, time.monotonic() + 0.5)
        os.close(master_fd)
        master_fd = -1
        process.wait(timeout=10)
        stderr = process.stderr.read() if process.stderr else b""
        assert process.returncode == 0, (
            f"exit {process.returncode}, stderr={stderr[:300]!r}"
        )
    finally:
        if process.poll() is None:
            process.terminate()
        if process.stderr:
            process.stderr.close()
        if master_fd != -1:
            os.close(master_fd)
        _stop_run(state)


def test_scan_local_input_gives_up_on_a_stalled_paste_after_a_timeout():
    """The byte budget alone does not recover the common abort, which sends
    almost no data: Ctrl-D mid-paste, a terminal reset, tmux forwarding the
    opener but dropping the closer. An idle bound does, and Ctrl-D is
    the only documented way out of attach."""
    _f, _h, _d, in_paste, paste_bytes, idle_since, started = (
        agent_run._scan_local_input_for_detach(b"\x1b[200~orphaned")
    )
    assert in_paste and idle_since is not None

    # Still inside the window: the paste is genuinely mid-flight.
    _f, _h, detached, in_paste, paste_bytes, idle_since, started = (
        agent_run._scan_local_input_for_detach(
            b"\x04more", in_paste, paste_bytes, idle_since, started
        )
    )
    assert not detached, "gave up on a paste that was still in flight"

    # Past the window with no closing marker.
    stale = idle_since - (agent_run._MAX_PASTE_IDLE_SECONDS + 1.0)
    _f, _h, detached, in_paste, _pb, _ps, _st = agent_run._scan_local_input_for_detach(
        b"\x04", in_paste, paste_bytes, stale, started
    )
    assert detached, "Ctrl-D still wedged after a stalled paste timed out"
    assert not in_paste


def test_scan_local_input_paste_timeout_measures_idleness_not_duration():
    """A large paste over a slow link trickles in for far longer than the
    idle bound. Timing from the opening marker expires the latch mid-stream,
    so the rest of the payload is rescanned as typed input and an embedded
    0x04 truncates it -- destroying a legitimate paste, which is the very
    failure the bracketing exists to prevent. Each arriving chunk must
    restart the idle window (the separate total-duration ceiling, which is
    not renewable, is what still bounds the latch overall)."""
    _f, _h, _d, in_paste, paste_bytes, idle_since, started = (
        agent_run._scan_local_input_for_detach(b"\x1b[200~start")
    )
    assert in_paste

    # Chunks keep arriving, each within the idle bound, but the paste as a
    # whole runs well past it.
    for _ in range(4):
        aged = idle_since - (agent_run._MAX_PASTE_IDLE_SECONDS * 0.5)
        _f, _h, detached, in_paste, paste_bytes, idle_since, started = (
            agent_run._scan_local_input_for_detach(
                b"payload\x04payload", in_paste, paste_bytes, aged, started
            )
        )
        assert not detached, "a slow but live paste was interrupted"
        assert in_paste, "the latch expired while payload was still arriving"

    # Only an actually idle gap releases it.
    stalled = idle_since - (agent_run._MAX_PASTE_IDLE_SECONDS + 1.0)
    _f, _h, detached, in_paste, _pb, _ps, _st = agent_run._scan_local_input_for_detach(
        b"\x04", in_paste, paste_bytes, stalled, started
    )
    assert detached, "Ctrl-D wedged after the paste genuinely stalled"


def test_attach_recovers_ctrl_d_after_an_aborted_paste(
    isolated_runs_root, isolated_log_root, monkeypatch
):
    """End-to-end: an ESC[200~ with no closing marker must not strand the
    user. The wrapped agent ignores SIGINT here so a forwarded \\x04 cannot
    kill it -- otherwise attach exits because the agent died, which looks
    like a working detach but is not one."""
    env = _environment(isolated_runs_root, isolated_log_root)
    name = "attach-aborted-paste"
    script = (
        "import signal, time\n"
        "signal.signal(signal.SIGINT, signal.SIG_IGN)\n"
        "while True: time.sleep(1)\n"
    )
    state = _launch_interactive(name, script, env)
    process, master_fd = _spawn_attach(name, env, rows=24, cols=80)
    try:
        _await_raw_mode(process, master_fd)
        os.write(master_fd, b"\x1b[200~orphaned text")

        # Inside the paste window Ctrl-D is content, so attach stays put.
        os.write(master_fd, b"\x04")
        time.sleep(0.5)
        assert process.poll() is None, "detached while a paste was in flight"

        # Past the window the latch releases and Ctrl-D works again.
        time.sleep(agent_run._MAX_PASTE_IDLE_SECONDS + 1.0)
        os.write(master_fd, b"\x04")
        deadline = time.monotonic() + 10.0
        while process.poll() is None and time.monotonic() < deadline:
            readable, _, _ = select.select([master_fd], [], [], 0.05)
            if readable:
                try:
                    os.read(master_fd, 65536)
                except OSError:
                    break
        assert process.wait(timeout=5) == 0, "Ctrl-D never detached after the abort"
        assert (state / "status").read_text().strip() == "running", (
            "the agent died instead of attach detaching"
        )
    finally:
        if process.poll() is None:
            process.terminate()
        os.close(master_fd)
        _stop_run(state)


def test_attach_does_not_detach_on_a_paste_whose_start_marker_was_split(
    isolated_runs_root, isolated_log_root
):
    """End-to-end mirror of the aborted-paste Critical. When ESC[200~ is
    split across a gap longer than the escape hold, the lone ESC is
    released as an ordinary keystroke; if the bytes that follow are not
    rejoined to it, the payload is scanned as typed input and any 0x04
    inside it detaches mid-paste. The agent ignores SIGINT so a forwarded
    \\x04 cannot end the session by killing it instead."""
    env = _environment(isolated_runs_root, isolated_log_root)
    name = "attach-split-start"
    script = (
        "import os, signal, sys, termios, tty\n"
        "signal.signal(signal.SIGINT, signal.SIG_IGN)\n"
        "fd = sys.stdin.fileno()\n"
        "tty.setraw(fd)\n"
        "sys.stdout.write('READY\\n')\n"
        "sys.stdout.flush()\n"
        "import time\n"
        "while True: time.sleep(1)\n"
    )
    state = _launch_interactive(name, script, env)
    process, master_fd = _spawn_attach(name, env, rows=24, cols=80)
    try:
        _read_until(master_fd, b"READY")
        # Split the marker across a gap longer than _ESCAPE_HOLD_TIMEOUT_SECONDS.
        os.write(master_fd, b"\x1b")
        time.sleep(agent_run._ESCAPE_HOLD_TIMEOUT_SECONDS + 0.15)
        os.write(master_fd, b"[200~payload\x04embedded\x1b[201~")
        time.sleep(1.0)

        assert process.poll() is None, (
            "attach detached on a 0x04 inside a paste whose start marker was split"
        )
    finally:
        if process.poll() is None:
            process.terminate()
        os.close(master_fd)
        _stop_run(state)


def _byte_accounting_script(total: int) -> str:
    """Agent that echoes back exactly what it received, so a test can assert
    on the bytes the wrapped agent got rather than on attach's internals."""
    return (
        "import os, signal, sys, tty\n"
        "signal.signal(signal.SIGINT, signal.SIG_IGN)\n"
        "fd = sys.stdin.fileno()\n"
        "tty.setraw(fd)\n"
        "sys.stdout.write('READY\\n')\n"
        "sys.stdout.flush()\n"
        "seen = b''\n"
        f"while len(seen) < {total}:\n"
        "    chunk = os.read(fd, 4096)\n"
        "    if not chunk:\n"
        "        break\n"
        "    seen += chunk\n"
        "sys.stdout.write('SEEN:' + seen.hex() + '\\n')\n"
        "sys.stdout.flush()\n"
        "import time\n"
        "while True: time.sleep(1)\n"
    )


@pytest.mark.parametrize(
    "first,second",
    [
        (b"\x1b", b"\x1b"),
        (b"\x1b", b"\x1b[A"),
        (b"\x1b[", b"\x1b[A"),
        (b"\x1b", b"ZZZ"),
        (b"\x1b[", b"ZZZ"),
    ],
)
def test_attach_deferred_escape_prefix_reaches_the_agent_intact(
    isolated_runs_root, isolated_log_root, first, second
):
    """Byte accounting across the release window: an escape prefix released
    by the hold timer must reach the agent whatever the next read looks
    like.

    The prefix used to be stashed and then dropped whenever the following
    read independently began with ESC, because the code inferred "the
    resync consumed it" from `data.startswith(prefix)` instead of from what
    the resync actually did. ESC then Escape/Up-arrow lost a byte; ESC[ then
    Up-arrow lost two. The gap here sits inside the loss window -- past the
    escape hold, below the point where the prefix is force-forwarded.
    """
    env = _environment(isolated_runs_root, isolated_log_root)
    name = f"attach-defer-{first.hex()}-{second.hex()}"
    expected = first + second
    state = _launch_interactive(
        name, _byte_accounting_script(len(expected)), env
    )
    process, master_fd = _spawn_attach(name, env, rows=24, cols=80)
    try:
        _read_until(master_fd, b"READY")
        os.write(master_fd, first)
        time.sleep(agent_run._ESCAPE_HOLD_TIMEOUT_SECONDS + 0.15)
        os.write(master_fd, second)

        output = _read_until(master_fd, b"SEEN:", timeout=10.0)
        seen = bytes.fromhex(
            output.split(b"SEEN:")[-1].split(b"\n")[0].decode().strip()
        )
        assert seen == expected, (
            f"sent {expected!r}, agent received {seen!r} "
            f"({len(expected) - len(seen)} bytes lost)"
        )
    finally:
        if process.poll() is None:
            process.terminate()
        os.close(master_fd)
        _stop_run(state)


def test_attach_deferred_escape_prefix_forwards_with_non_marker_followup(
    isolated_runs_root, isolated_log_root
):
    """The released-prefix branch must forward the prefix when the bytes
    that follow are not a paste marker. Discarding it instead (the mutation
    this pins) leaves the agent seeing `ZZZ` where the user typed
    Escape-then-ZZZ, which in a TUI is a different command entirely."""
    env = _environment(isolated_runs_root, isolated_log_root)
    name = "attach-defer-nonmarker"
    expected = b"\x1bZZZ"
    state = _launch_interactive(
        name, _byte_accounting_script(len(expected)), env
    )
    process, master_fd = _spawn_attach(name, env, rows=24, cols=80)
    try:
        _read_until(master_fd, b"READY")
        os.write(master_fd, b"\x1b")
        time.sleep(agent_run._ESCAPE_HOLD_TIMEOUT_SECONDS + 0.15)
        os.write(master_fd, b"ZZZ")

        output = _read_until(master_fd, b"SEEN:", timeout=10.0)
        seen = bytes.fromhex(
            output.split(b"SEEN:")[-1].split(b"\n")[0].decode().strip()
        )
        assert seen == expected, f"expected {expected!r}, agent received {seen!r}"
    finally:
        if process.poll() is None:
            process.terminate()
        os.close(master_fd)
        _stop_run(state)


def test_attach_bare_escape_reaches_the_agent_within_one_hold_window(
    isolated_runs_root, isolated_log_root
):
    """Escape is a cancel key in every TUI, so its latency is a felt
    property. Deferring the released prefix by an extra loop pass doubled it
    (roughly 310ms to 719ms); one hold window plus scheduling slack is the
    budget."""
    env = _environment(isolated_runs_root, isolated_log_root)
    name = "attach-esc-latency"
    state = _launch_interactive(name, _byte_accounting_script(1), env)
    process, master_fd = _spawn_attach(name, env, rows=24, cols=80)
    try:
        _read_until(master_fd, b"READY")
        started = time.monotonic()
        os.write(master_fd, b"\x1b")
        _read_until(master_fd, b"SEEN:", timeout=10.0)
        elapsed = time.monotonic() - started

        budget = agent_run._ESCAPE_HOLD_TIMEOUT_SECONDS * 2
        assert elapsed < budget, (
            f"bare Escape took {elapsed:.3f}s to reach the agent, over the "
            f"{budget:.3f}s budget"
        )
    finally:
        if process.poll() is None:
            process.terminate()
        os.close(master_fd)
        _stop_run(state)


def test_resync_paste_marker_reports_whether_it_consumed_the_prefix():
    """The caller has to know whether the prefix was rejoined, and cannot
    infer it from the returned data: a following read that independently
    starts with ESC looks identical to a rejoined one."""
    joined, consumed = agent_run._resync_paste_marker(b"[200~x", b"\x1b")
    assert consumed and joined == b"\x1b[200~x"

    # Independently starts with ESC, but is not a marker: not consumed.
    same, consumed = agent_run._resync_paste_marker(b"\x1b[A", b"\x1b")
    assert not consumed and same == b"\x1b[A"

    same, consumed = agent_run._resync_paste_marker(b"\x1b", b"\x1b")
    assert not consumed and same == b"\x1b"

    partial, consumed = agent_run._resync_paste_marker(b"[2", b"\x1b")
    assert consumed and partial == b"\x1b[2", "partial join was not recognized"


def test_resync_paste_marker_joins_a_still_incomplete_marker():
    """The partial-join branch: released + data is shorter than a full
    marker but is still a prefix of one, so it must be rejoined and held
    rather than treated as unrelated input."""
    for released, data, expected in (
        (b"\x1b", b"[", b"\x1b["),
        (b"\x1b", b"[20", b"\x1b[20"),
        (b"\x1b[", b"20", b"\x1b[20"),
        (b"\x1b[", b"2", b"\x1b[2"),
    ):
        joined, consumed = agent_run._resync_paste_marker(data, released)
        assert consumed, f"{released!r} + {data!r} was not rejoined"
        assert joined == expected


def test_valid_resize_record_rejects_a_foreign_format_version():
    """A record from a build with a different layout must not decode. Its
    payload bytes sit at different offsets, so reading them as dimensions
    drives the wrapped agent's terminal to a garbage size."""
    good = agent_run._pack_resize(80, 24)
    assert agent_run._valid_resize_record(good)

    foreign = bytes([good[0], agent_run.RESIZE_RECORD_VERSION + 1]) + good[2:]
    assert not agent_run._valid_resize_record(foreign)


def test_drain_resize_records_warns_and_skips_an_old_format_record(monkeypatch):
    """Mixed-version pairing: a runner from this build reading records
    written by the pre-version 5-byte format. Silently applying them
    (or silently dropping them) drives the terminal to 0x0 or leaves the
    user with no idea why resize stopped working -- warn and skip."""
    applied = []
    monkeypatch.setattr(
        agent_run, "_apply_resize", lambda fd, c, r: applied.append((c, r))
    )
    warnings = []

    # The original unversioned layout: magic, cols, rows, checksum.
    old_format = struct.pack(">BHHB", agent_run.RESIZE_RECORD_MAGIC, 100, 30, 0x5A)
    leftover = agent_run._drain_resize_records(
        7, old_format + old_format, warnings.append
    )

    assert applied == [], f"decoded an old-format record as {applied}"
    assert any("version" in w for w in warnings), (
        f"old-format records were dropped without a warning: {warnings}"
    )
    assert len(leftover) < agent_run.RESIZE_RECORD_SIZE


def test_drain_resize_records_still_applies_current_format_after_a_foreign_one(
    monkeypatch,
):
    """The version guard must skip only the foreign record. A current-format
    record behind it still has to apply, or one stray write from another
    build would wedge resize for the rest of the session."""
    applied = []
    monkeypatch.setattr(
        agent_run, "_apply_resize", lambda fd, c, r: applied.append((c, r))
    )
    foreign = struct.pack(
        agent_run.RESIZE_RECORD_FORMAT,
        agent_run.RESIZE_RECORD_MAGIC,
        agent_run.RESIZE_RECORD_VERSION + 7,
        100,
        30,
        0x11,
    )

    agent_run._drain_resize_records(
        7, foreign + agent_run._pack_resize(120, 42), lambda _m: None
    )

    assert applied == [(120, 42)], f"current-format record was lost: {applied}"


def test_attach_warns_and_keeps_going_when_the_runner_speaks_another_version(
    isolated_runs_root, isolated_log_root, capsys
):
    """A session launched by a build with a different resize format must
    still be attachable: keyboard passthrough and Ctrl-D detach are what
    matter, and forwarding records the runner cannot read would move its
    terminal to a garbage size. (The SystemExit is the unrelated no-tty
    check firing, which confirms the version check did not exit first.)"""
    state = _seed_attachable_run(isolated_runs_root, "attach-old-runner")
    (state / agent_run.RESIZE_PROTOCOL_MARKER).write_text("0\n")
    (isolated_log_root / "attach-old-runner").mkdir()
    (isolated_log_root / "attach-old-runner" / "log").write_bytes(b"")

    with pytest.raises(SystemExit, match="requires an interactive terminal"):
        agent_run.cmd_attach(argparse.Namespace(name="attach-old-runner"))
    assert "different resize record format" in capsys.readouterr().err


def test_attach_treats_a_missing_protocol_marker_as_a_mismatch(
    isolated_runs_root, isolated_log_root, capsys
):
    """A runner that predates resize versioning writes no marker and reads
    the old 5-byte record, so a versioned one is garbage to it. Absent must
    mean incompatible, not assumed-compatible."""
    state = isolated_runs_root / "attach-unversioned"
    state.mkdir()
    (state / "interactive").write_text("1\n")
    (state / "pid").write_text(f"{os.getpid()}\n")
    os.mkfifo(state / "stdin")
    os.mkfifo(state / "resize")
    (isolated_log_root / "attach-unversioned").mkdir()
    (isolated_log_root / "attach-unversioned" / "log").write_bytes(b"")

    with pytest.raises(SystemExit, match="requires an interactive terminal"):
        agent_run.cmd_attach(argparse.Namespace(name="attach-unversioned"))
    assert "different resize record format" in capsys.readouterr().err


def test_interactive_launch_publishes_the_resize_protocol_version(
    isolated_runs_root, isolated_log_root
):
    """attach reads this marker to decide whether its records are
    intelligible to the runner, so a real launch has to write it."""
    env = _environment(isolated_runs_root, isolated_log_root)
    state = _launch_interactive(
        "resize-version-marker", "import time\nwhile True: time.sleep(1)\n", env
    )
    try:
        marker = state / agent_run.RESIZE_PROTOCOL_MARKER
        assert _wait_until(lambda: marker.exists(), timeout=10.0), (
            "runner never published its resize protocol version"
        )
        assert marker.read_text().strip() == str(agent_run.RESIZE_RECORD_VERSION)
    finally:
        _stop_run(state)


@pytest.mark.parametrize("cols,rows", [(46, 24), (20, 100), (167, 24), (80, 24)])
def test_drain_resize_records_survives_an_orphan_at_every_offset(monkeypatch, cols, rows):
    """attach replaces a partially-written record rather than completing it,
    so an orphaned prefix sits on the wire ahead of a full record. Even
    checksum coefficients discarded the top bit of their byte, so whole
    families of dimensions -- 46x24 and 20x100 among them -- still framed a
    shifted window as valid."""
    applied = []
    monkeypatch.setattr(
        agent_run, "_apply_resize", lambda fd, c, r: applied.append((c, r))
    )
    record = agent_run._pack_resize(cols, rows)

    for orphan_len in range(1, agent_run.RESIZE_RECORD_SIZE):
        applied.clear()
        agent_run._drain_resize_records(7, record[:orphan_len] + record)
        assert applied == [(cols, rows)], (
            f"orphan of {orphan_len}B mis-framed {cols}x{rows}: {applied}"
        )


def test_drain_resize_records_never_misframes_across_a_dimension_sweep(monkeypatch):
    """Sweep rather than sample: the checksum collisions this pins were
    dimension-dependent, so a handful of hand-picked sizes cannot show they
    are gone."""
    applied = []
    monkeypatch.setattr(
        agent_run, "_apply_resize", lambda fd, c, r: applied.append((c, r))
    )
    misframed = []

    for cols in range(1, 400, 3):
        for rows in range(1, 200, 3):
            record = agent_run._pack_resize(cols, rows)
            for orphan_len in range(1, agent_run.RESIZE_RECORD_SIZE):
                applied.clear()
                agent_run._drain_resize_records(7, record[:orphan_len] + record)
                if applied != [(cols, rows)]:
                    misframed.append((cols, rows, orphan_len, list(applied)))

    assert not misframed, f"{len(misframed)} misframes, e.g. {misframed[:5]}"


def test_drain_resize_records_rejects_an_out_of_range_dimension(monkeypatch):
    """A checksum-valid record can still carry a nonsense size once framing
    has gone wrong. Terminals do not get near this bound, so anything past
    it is corruption rather than a size worth applying."""
    applied = []
    monkeypatch.setattr(
        agent_run, "_apply_resize", lambda fd, c, r: applied.append((c, r))
    )
    oversized = agent_run.MAX_TERMINAL_DIMENSION + 1
    forged = struct.pack(
        agent_run.RESIZE_RECORD_FORMAT,
        agent_run.RESIZE_RECORD_MAGIC,
        agent_run.RESIZE_RECORD_VERSION,
        oversized,
        24,
        agent_run._resize_checksum(oversized, 24),
    )

    agent_run._drain_resize_records(7, forged)

    assert applied == [], f"applied an out-of-range size: {applied}"


def test_drain_resize_records_rejects_adversarial_garbage(monkeypatch):
    """Random bytes must never decode into an applied resize. The previous
    reader accepted roughly 1 in 160 of these; 'resynchronizes safely' has
    to mean it applies nothing at all, not that it applies something
    in-range."""
    applied = []
    monkeypatch.setattr(
        agent_run, "_apply_resize", lambda fd, c, r: applied.append((c, r))
    )
    rng = random.Random(20260815)

    for _ in range(200000):
        blob = bytes(rng.getrandbits(8) for _ in range(rng.randint(1, 16)))
        agent_run._drain_resize_records(7, blob)

    assert applied == [], (
        f"{len(applied)} adversarial strings decoded as resizes, e.g. {applied[:5]}"
    )


def test_attach_does_not_interrupt_a_paste_that_trickles_in_slowly(
    isolated_runs_root, isolated_log_root
):
    """End-to-end for the idle-vs-duration bound. A large paste over a slow
    link (a loaded SSH session, a terminal trickling the payload) takes
    longer end to end than the latch budget while never actually going
    idle. Timing from the opening marker expires the latch mid-stream, so
    the rest of the payload is rescanned as typed input and the 0x04 bytes
    in it detach -- truncating a legitimate paste. The agent ignores SIGINT
    so a forwarded 0x04 cannot end the session by killing it instead."""
    env = _environment(isolated_runs_root, isolated_log_root)
    name = "attach-slow-paste"
    chunks = [f"chunk{i:03d}\x04".encode() for i in range(8)]
    pasted = b"\x1b[200~" + b"".join(chunks) + b"\x1b[201~"
    state = _launch_interactive(name, _byte_accounting_script(len(pasted)), env)
    process, master_fd = _spawn_attach(name, env, rows=24, cols=80)
    try:
        _read_until(master_fd, b"READY")
        # Total span well past the idle bound, every gap comfortably under it.
        gap = agent_run._MAX_PASTE_IDLE_SECONDS / 3
        os.write(master_fd, b"\x1b[200~")
        for chunk in chunks:
            time.sleep(gap)
            os.write(master_fd, chunk)
        time.sleep(gap)
        os.write(master_fd, b"\x1b[201~")

        output = _read_until(master_fd, b"SEEN:", timeout=20.0)
        seen = bytes.fromhex(
            output.split(b"SEEN:")[-1].split(b"\n")[0].decode().strip()
        )
        assert seen == pasted, (
            f"slow paste was truncated: sent {len(pasted)}B, agent got {len(seen)}B"
        )
        assert process.poll() is None, "attach detached inside a slow paste"
    finally:
        if process.poll() is None:
            process.terminate()
        os.close(master_fd)
        _stop_run(state)


def test_release_expired_held_escape_holds_a_marker_prefix_past_a_long_gap():
    """The extended paste-marker hold has to cover the whole window, not
    just the first fraction of it. A split ESC[200~ whose remainder arrives
    late is exactly the case the hold exists for; releasing the prefix
    early destroys the marker and the payload is scanned as typed input."""
    prefix = agent_run._PASTE_START[:4]
    for gap in (0.6, 1.0, agent_run._PASTE_MARKER_HOLD_SECONDS - 0.2):
        deadline = (
            time.monotonic() - gap + agent_run._ESCAPE_HOLD_TIMEOUT_SECONDS
        )
        held, _dl, released = agent_run._release_expired_held_escape(prefix, deadline)
        assert released == b"", f"marker prefix released after only {gap}s"
        assert held == prefix

    # Past the full window it does release, so the key can never wedge.
    expired = (
        time.monotonic()
        - agent_run._PASTE_MARKER_HOLD_SECONDS
        - 0.2
        + agent_run._ESCAPE_HOLD_TIMEOUT_SECONDS
    )
    _held, _dl, released = agent_run._release_expired_held_escape(prefix, expired)
    assert released == prefix, "marker prefix held past its own bound"


def test_attach_does_not_detach_on_a_paste_marker_split_by_a_long_gap(
    isolated_runs_root, isolated_log_root
):
    """The same boundary end to end, above the ~0.6s gap the suite used to
    stop at."""
    env = _environment(isolated_runs_root, isolated_log_root)
    name = "attach-split-long-gap"
    script = (
        "import signal, sys, time, tty\n"
        "signal.signal(signal.SIGINT, signal.SIG_IGN)\n"
        "tty.setraw(sys.stdin.fileno())\n"
        "sys.stdout.write('READY\\n')\n"
        "sys.stdout.flush()\n"
        "while True: time.sleep(1)\n"
    )
    state = _launch_interactive(name, script, env)
    process, master_fd = _spawn_attach(name, env, rows=24, cols=80)
    try:
        _read_until(master_fd, b"READY")
        os.write(master_fd, b"\x1b[2")
        time.sleep(agent_run._PASTE_MARKER_HOLD_SECONDS - 0.5)
        os.write(master_fd, b"00~payload\x04embedded\x1b[201~")
        time.sleep(1.0)

        assert process.poll() is None, (
            "detached on a 0x04 inside a paste whose marker was split by a long gap"
        )
    finally:
        if process.poll() is None:
            process.terminate()
        os.close(master_fd)
        _stop_run(state)


def test_attach_reports_input_it_could_not_deliver_when_the_agent_dies(
    isolated_runs_root, isolated_log_root, monkeypatch, capsys
):
    """Typed input still buffered when the wrapped agent stops reading must
    be reported with its byte count, not silently discarded -- the same
    guarantee the detach flush makes. An escape prefix released by the hold
    timer is the narrow case: it used to sit in a one-pass deferral where
    nothing reported it.

    Drives cmd_attach directly so the assertion is on the warning itself.
    An end-to-end variant cannot make this claim: attach's cleanup calls
    tcsetattr(TCSADRAIN), which blocks until the PTY master is drained, so
    such a test measures the drain rather than the report."""
    state = _seed_attachable_run(isolated_runs_root, "attach-death-report")
    (isolated_log_root / "attach-death-report").mkdir()
    (isolated_log_root / "attach-death-report" / "log").write_bytes(b"")
    monkeypatch.setattr(agent_run, "_DETACH_FLUSH_TIMEOUT_SECONDS", 0.1)
    monkeypatch.setattr(agent_run.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(agent_run.sys.stdin, "fileno", lambda: 0)
    monkeypatch.setattr(agent_run.termios, "tcgetattr", lambda _fd: "saved")
    monkeypatch.setattr(agent_run.termios, "tcsetattr", lambda *_a: None)
    monkeypatch.setattr(agent_run.tty, "setraw", lambda _fd: None)
    monkeypatch.setattr(agent_run, "_reset_terminal_modes", lambda: None)
    monkeypatch.setattr(
        agent_run.os, "get_terminal_size", lambda _fd: os.terminal_size((80, 24))
    )
    monkeypatch.setattr(agent_run, "_pid_alive", lambda _pid: True)

    # A bare ESC, then nothing: the hold expires, the prefix is released and
    # forwarded, and the detach that follows must account for it.
    typed = b"\x1b"
    reads = [typed, b"\x04"]

    def fake_read(_fd, _n):
        return reads.pop(0) if reads else b""

    monkeypatch.setattr(agent_run.os, "read", fake_read)
    # No fd is ever readable on the pass between the two reads, so the hold
    # timer expires and releases the prefix exactly as it would in practice.
    calls = {"n": 0}

    def fake_select(r, w, x, t=None):
        calls["n"] += 1
        time.sleep(agent_run._ESCAPE_HOLD_TIMEOUT_SECONDS + 0.05)
        readable = [] if calls["n"] == 2 else list(r)
        return readable, [], []

    monkeypatch.setattr(agent_run.select, "select", fake_select)

    stdin_fd = os.open(str(state / "stdin"), os.O_RDWR | os.O_NONBLOCK)
    resize_fd = os.open(str(state / "resize"), os.O_RDWR | os.O_NONBLOCK)
    try:
        # Fill the stdin FIFO so nothing attach forwards can be delivered.
        try:
            while True:
                os.write(stdin_fd, b"f" * 4096)
        except BlockingIOError:
            pass

        assert agent_run.cmd_attach(argparse.Namespace(name="attach-death-report")) == 0
    finally:
        os.close(stdin_fd)
        os.close(resize_fd)

    err = capsys.readouterr().err
    assert "undelivered" in err, f"undeliverable input was not reported: {err!r}"
    assert str(len(typed)) in err, f"warning omitted the byte count: {err!r}"


def test_terminal_restored_on_signal_death_reports_a_failing_restore():
    """A restore that raises leaves the terminal raw and possibly on the
    alt screen. The signal must still kill the process, but swallowing the
    failure silently leaves the user with a wedged terminal and no idea
    why -- or what to run to fix it."""
    code = textwrap.dedent(
        """
        import os, signal, sys
        sys.path.insert(0, %r)
        from toolbox import agent_run
        def boom():
            raise RuntimeError("restore failed")
        with agent_run._terminal_restored_on_signal_death(boom):
            os.kill(os.getpid(), signal.SIGTERM)
            signal.pause()
        """
    ) % str(Path(agent_run.__file__).parent.parent)

    result = subprocess.run([sys.executable, "-c", code], capture_output=True, timeout=30)

    assert result.returncode == -signal.SIGTERM, "process survived SIGTERM"
    assert b"could not restore terminal" in result.stderr, (
        f"restore failure was swallowed silently: {result.stderr!r}"
    )
    assert b"reset" in result.stderr, "no recovery hint for a wedged terminal"


def test_attach_does_not_duplicate_a_prefix_rejoined_across_three_reads(
    isolated_runs_root, isolated_log_root
):
    """A marker split into three reads makes the rejoined data itself still
    incomplete, so it is held back again with a prefix the agent has
    already received. The bytes past that prefix must be forwarded exactly
    once -- no loss, and no duplicated ESC that a TUI would read as a
    second Escape keypress."""
    env = _environment(isolated_runs_root, isolated_log_root)
    name = "attach-three-way-split"
    payload = b"pasted\x03text"
    expected = b"\x1b[200~" + payload + b"\x1b[201~"
    state = _launch_interactive(name, _byte_accounting_script(len(expected)), env)
    process, master_fd = _spawn_attach(name, env, rows=24, cols=80)
    try:
        _read_until(master_fd, b"READY")
        gap = agent_run._ESCAPE_HOLD_TIMEOUT_SECONDS + 0.15
        os.write(master_fd, b"\x1b")
        time.sleep(gap)
        os.write(master_fd, b"[2")
        time.sleep(gap)
        os.write(master_fd, b"00~" + payload + b"\x1b[201~")

        output = _read_until(master_fd, b"SEEN:", timeout=15.0)
        seen = bytes.fromhex(
            output.split(b"SEEN:")[-1].split(b"\n")[0].decode().strip()
        )
        assert seen == expected, f"expected {expected!r}, agent received {seen!r}"
    finally:
        if process.poll() is None:
            process.terminate()
        os.close(master_fd)
        _stop_run(state)


def test_scan_local_input_paste_latch_cannot_be_pinned_open_by_input():
    """The idle bound alone cannot end a paste that never stops receiving
    bytes, and inside a paste every byte is payload -- so the Ctrl-D a user
    presses to escape a stuck paste is itself what renews the window. With
    only an idle bound, input arriving more often than once per idle period
    pins the latch open for the rest of the session: pressing the escape
    key harder holds the door shut tighter.

    Detach must be reachable in bounded time at every arrival rate, which
    is the axis a single idle bound does not cover."""
    intervals = [
        agent_run._MAX_PASTE_IDLE_SECONDS / 8,
        agent_run._MAX_PASTE_IDLE_SECONDS / 2,
        agent_run._MAX_PASTE_IDLE_SECONDS * 0.98,
    ]
    ceiling = agent_run._MAX_PASTE_TOTAL_SECONDS
    for interval in intervals:
        clock = {"now": 1_000.0}
        with mock.patch.object(agent_run.time, "monotonic", lambda: clock["now"]):
            _f, _h, _d, in_paste, pb, idle, started = (
                agent_run._scan_local_input_for_detach(b"\x1b[200~orphaned")
            )
            assert in_paste

            detached = False
            elapsed = 0.0
            # Generous headroom over the ceiling: the point is that it ends,
            # not exactly when.
            while elapsed <= ceiling * 2:
                clock["now"] += interval
                elapsed += interval
                _f, _h, detached, in_paste, pb, idle, started = (
                    agent_run._scan_local_input_for_detach(
                        b"\x04", in_paste, pb, idle, started
                    )
                )
                if detached:
                    break

            assert detached, (
                f"Ctrl-D never reachable with input every {interval:.2f}s -- "
                f"the latch is pinned open by the user's own keypresses"
            )
            assert elapsed <= ceiling + interval + 1.0, (
                f"detach took {elapsed:.1f}s, past the {ceiling:.0f}s ceiling"
            )


def test_scan_local_input_paste_ceiling_is_not_renewable():
    """The ceiling must measure from the opening marker and never restart,
    or it is just a second idle bound and the pin-open case returns."""
    clock = {"now": 500.0}
    with mock.patch.object(agent_run.time, "monotonic", lambda: clock["now"]):
        _f, _h, _d, in_paste, pb, idle, started = (
            agent_run._scan_local_input_for_detach(b"\x1b[200~x")
        )
        opened_at = started
        for _ in range(20):
            clock["now"] += agent_run._MAX_PASTE_IDLE_SECONDS / 4
            _f, _h, _d, in_paste, pb, idle, started = (
                agent_run._scan_local_input_for_detach(
                    b"payload", in_paste, pb, idle, started
                )
            )
            assert in_paste
            assert started == opened_at, "the total-duration ceiling was renewed"


def test_scan_local_input_paste_ceiling_leaves_a_real_paste_alone():
    """The ceiling must not cut short a paste that could legitimately still
    be arriving: the byte budget at a plausibly slow link has to fit inside
    it, or the A-1 trickle case regresses."""
    slowest_usable_bytes_per_second = 15 * 1024
    seconds_for_a_full_payload = (
        agent_run._MAX_PASTE_BYTES / slowest_usable_bytes_per_second
    )
    assert agent_run._MAX_PASTE_TOTAL_SECONDS > seconds_for_a_full_payload, (
        "the ceiling cuts off a paste that is still within the byte budget"
    )


def test_pack_resize_is_never_called_with_an_out_of_range_size(monkeypatch):
    """attach reads its size from the environment, so a display wider than
    the record's range is a value to clamp, not a caller bug to raise on.
    Raising here unwinds the main loop, skips the finally: that restores the
    terminal, and leaves the shell raw -- the exact wedge B-F4 exists to
    prevent."""
    for cols, rows in (
        (agent_run.MAX_TERMINAL_DIMENSION + 1, 24),
        (80, agent_run.MAX_TERMINAL_DIMENSION + 1),
        (7680, 4320),
    ):
        clamped_cols = min(cols, agent_run.MAX_TERMINAL_DIMENSION)
        clamped_rows = min(rows, agent_run.MAX_TERMINAL_DIMENSION)
        # The clamped pair is what the sender must pass, and it must pack.
        record = agent_run._pack_resize(clamped_cols, clamped_rows)
        assert agent_run._valid_resize_record(record)


def test_attach_survives_a_terminal_wider_than_the_record_range(
    isolated_runs_root, isolated_log_root, monkeypatch, capsys
):
    """End-to-end for the same thing through cmd_attach: an oversized
    terminal must not take attach down. Before the clamp this raised
    ValueError out of the main loop."""
    state = _seed_attachable_run(isolated_runs_root, "attach-wide-term")
    (isolated_log_root / "attach-wide-term").mkdir()
    (isolated_log_root / "attach-wide-term" / "log").write_bytes(b"")
    monkeypatch.setattr(agent_run.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(agent_run.sys.stdin, "fileno", lambda: 0)
    monkeypatch.setattr(agent_run.termios, "tcgetattr", lambda _fd: "saved")
    monkeypatch.setattr(agent_run.termios, "tcsetattr", lambda *_a: None)
    monkeypatch.setattr(agent_run.tty, "setraw", lambda _fd: None)
    monkeypatch.setattr(agent_run, "_reset_terminal_modes", lambda: None)
    monkeypatch.setattr(agent_run, "_pid_alive", lambda _pid: True)
    oversized = agent_run.MAX_TERMINAL_DIMENSION + 1
    monkeypatch.setattr(
        agent_run.os,
        "get_terminal_size",
        lambda _fd: os.terminal_size((oversized, oversized)),
    )
    reads = [b"\x04"]
    monkeypatch.setattr(
        agent_run.os, "read", lambda _fd, _n: reads.pop(0) if reads else b""
    )
    monkeypatch.setattr(
        agent_run.select, "select", lambda r, w, x, t=None: (list(r), list(w), [])
    )

    stdin_fd = os.open(str(state / "stdin"), os.O_RDWR | os.O_NONBLOCK)
    resize_fd = os.open(str(state / "resize"), os.O_RDWR | os.O_NONBLOCK)
    try:
        rc = agent_run.cmd_attach(argparse.Namespace(name="attach-wide-term"))
        assert rc == 0, "attach died on an oversized terminal"
        # Whatever reached the FIFO must be a well-formed, in-range record.
        try:
            written = os.read(resize_fd, 4096)
        except BlockingIOError:
            written = b""
    finally:
        os.close(stdin_fd)
        os.close(resize_fd)

    assert "clamping" in capsys.readouterr().err, "clamp was silent"
    if written:
        assert agent_run._valid_resize_record(written[: agent_run.RESIZE_RECORD_SIZE])


@pytest.mark.parametrize(
    "raw", [b"\xff\xfe\n", b"\x80\x81\x82", b"1\xff\n", b"\xc3\x28"]
)
def test_resize_protocol_matches_survives_a_non_utf8_marker(tmp_path, raw):
    """A corrupt state file must not take attach down with a traceback.
    UnicodeDecodeError is a ValueError, not an OSError, so the original
    guard did not cover it."""
    (tmp_path / agent_run.RESIZE_PROTOCOL_MARKER).write_bytes(raw)
    assert agent_run._resize_protocol_matches(tmp_path) is False


def test_resize_protocol_matches_survives_an_unreadable_marker(tmp_path):
    marker = tmp_path / agent_run.RESIZE_PROTOCOL_MARKER
    marker.write_text("1\n")
    marker.chmod(0o000)
    try:
        assert agent_run._resize_protocol_matches(tmp_path) is False
    finally:
        marker.chmod(0o644)


def test_drain_resize_records_rejects_a_record_whose_checksum_is_wrong(monkeypatch):
    """The checksum has to be load-bearing on its own, not merely redundant
    with the magic-byte and boundary checks. A window with the right magic,
    the right version, in-range dimensions and a valid following boundary --
    everything except the checksum -- must still be rejected.

    This matters beyond coverage: it is the guard that keeps the framing
    reasoning honest for whoever later raises MAX_TERMINAL_DIMENSION, since
    the range check alone stops discriminating as that bound widens."""
    applied = []
    monkeypatch.setattr(
        agent_run, "_apply_resize", lambda fd, c, r: applied.append((c, r))
    )
    forged = struct.pack(
        agent_run.RESIZE_RECORD_FORMAT,
        agent_run.RESIZE_RECORD_MAGIC,
        agent_run.RESIZE_RECORD_VERSION,
        100,
        30,
        (agent_run._resize_checksum(100, 30) + 1) & 0xFF,
    )
    # Followed by a magic byte, so the boundary check cannot be what rejects it.
    agent_run._drain_resize_records(
        7, forged + bytes([agent_run.RESIZE_RECORD_MAGIC])
    )

    assert applied == [], f"a wrong checksum was accepted: {applied}"


def test_drain_resize_records_needs_the_magic_byte_check(monkeypatch):
    """Anchoring is three checks and each does real work. Without the magic
    byte, a window that satisfies the other two decodes at the wrong offset
    and applies a size nobody asked for."""
    applied = []
    monkeypatch.setattr(
        agent_run, "_apply_resize", lambda fd, c, r: applied.append((c, r))
    )

    agent_run._drain_resize_records(
        7, bytes.fromhex("50001fa7010101010150a7")
    )

    assert applied == [], f"applied a size from a non-anchored window: {applied}"


def test_drain_resize_records_accepts_a_record_carrying_a_magic_byte_payload(
    monkeypatch,
):
    """The complement: a legitimate record whose own payload/checksum bytes
    include 0xA7 must still be applied. _pack_resize(87, 24) ends in the
    magic byte, so it is the case where over-strict anchoring would drop
    real resizes."""
    applied = []
    monkeypatch.setattr(
        agent_run, "_apply_resize", lambda fd, c, r: applied.append((c, r))
    )
    record = agent_run._pack_resize(87, 24)
    assert bytes([agent_run.RESIZE_RECORD_MAGIC]) in record[1:], (
        "test premise: this record should contain the magic byte in its payload"
    )

    agent_run._drain_resize_records(7, record)

    assert applied == [(87, 24)], f"dropped a valid record: {applied}"


def test_resync_paste_marker_rejoins_a_split_paste_end_marker():
    """ESC[201~ splits across the hold exactly as ESC[200~ does. If the
    closer is not rejoined, in_paste never clears: the latch stays on and
    every subsequent Ctrl-D is treated as paste content instead of a
    detach."""
    marker = agent_run._PASTE_END
    for split in (1, 2):
        prefix, rest = marker[:split], marker[split:]
        joined, consumed = agent_run._resync_paste_marker(rest, prefix)
        assert consumed, f"ESC[201~ split at {split} was not rejoined"
        assert joined == marker


def test_split_paste_end_marker_still_closes_the_paste():
    """End to end through the scanner: a paste whose closer was split must
    leave in_paste False, so the next Ctrl-D detaches rather than being
    swallowed as payload."""
    marker = agent_run._PASTE_END
    for split in (1, 2):
        _f, _h, _d, in_paste, pb, idle, started = (
            agent_run._scan_local_input_for_detach(b"\x1b[200~payload")
        )
        assert in_paste

        prefix, rest = marker[:split], marker[split:]
        rejoined, consumed = agent_run._resync_paste_marker(rest, prefix)
        assert consumed
        _f, _h, _d, in_paste, pb, idle, started = (
            agent_run._scan_local_input_for_detach(
                rejoined, in_paste, pb, idle, started
            )
        )
        assert not in_paste, f"closer split at {split} left the latch on"

        _f, _h, detached, _ip, _pb, _idle, _st = (
            agent_run._scan_local_input_for_detach(
                b"\x04", in_paste, pb, idle, started
            )
        )
        assert detached, f"Ctrl-D swallowed after a closer split at {split}"


def test_attach_releases_the_already_sent_carry_after_the_rescan(
    isolated_runs_root, isolated_log_root
):
    """The three-read case pins the carry being held; this pins it being
    released. After a rescan resolves, ordinary typing must be forwarded in
    full -- a carry that never cleared would silently eat the first bytes
    of the next keystroke."""
    env = _environment(isolated_runs_root, isolated_log_root)
    name = "attach-carry-release"
    expected = b"\x1b[200~pasted\x1b[201~" + b"ABCDEF"
    state = _launch_interactive(name, _byte_accounting_script(len(expected)), env)
    process, master_fd = _spawn_attach(name, env, rows=24, cols=80)
    try:
        _read_until(master_fd, b"READY")
        gap = agent_run._ESCAPE_HOLD_TIMEOUT_SECONDS + 0.15
        os.write(master_fd, b"\x1b")
        time.sleep(gap)
        os.write(master_fd, b"[200~pasted\x1b[201~")
        time.sleep(gap)
        # Ordinary typing after the rescan resolved.
        os.write(master_fd, b"ABCDEF")

        output = _read_until(master_fd, b"SEEN:", timeout=15.0)
        seen = bytes.fromhex(
            output.split(b"SEEN:")[-1].split(b"\n")[0].decode().strip()
        )
        assert seen == expected, f"expected {expected!r}, agent received {seen!r}"
    finally:
        if process.poll() is None:
            process.terminate()
        os.close(master_fd)
        _stop_run(state)


def test_runner_warns_about_a_foreign_version_record(tmp_path):
    """The runner-side warning is what tells an operator why the terminal
    size stopped tracking when a client from another release writes to the
    resize FIFO. It goes to the log, which is where they look."""
    log_path = tmp_path / "log"
    log_fd = os.open(str(log_path), os.O_RDWR | os.O_CREAT, 0o600)
    try:
        messages = []

        def warn_resize(message: str) -> None:
            os.write(log_fd, f"agent-run: {message}\r\n".encode())
            messages.append(message)

        foreign = struct.pack(
            agent_run.RESIZE_RECORD_FORMAT,
            agent_run.RESIZE_RECORD_MAGIC,
            agent_run.RESIZE_RECORD_VERSION + 5,
            100,
            30,
            0x22,
        )
        agent_run._drain_resize_records(7, foreign, warn_resize)
    finally:
        os.close(log_fd)

    assert messages, "a foreign-version record was skipped without a warning"
    assert "version" in messages[0]
    written = log_path.read_bytes()
    assert b"agent-run:" in written and b"version" in written
