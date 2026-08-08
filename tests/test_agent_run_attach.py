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


def test_attach_rejects_run_missing_resize_fifo(isolated_runs_root, isolated_log_root):
    """Coverage for a run launched before this branch's resize-FIFO support:
    a state dir with a `stdin` FIFO but no `resize` FIFO must exit cleanly
    via the existing `if not resize_path.is_fifo()` check rather than
    reaching the terminal-relay loop."""
    state = isolated_runs_root / "pre-feature"
    state.mkdir()
    (state / "interactive").write_text("1\n")
    (state / "pid").write_text(f"{os.getpid()}\n")
    os.mkfifo(state / "stdin")
    (isolated_log_root / "pre-feature").mkdir()
    (isolated_log_root / "pre-feature" / "log").write_bytes(b"")

    with pytest.raises(SystemExit, match="no resize FIFO"):
        agent_run.cmd_attach(argparse.Namespace(name="pre-feature"))


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


def test_attach_ctrl_c_forwards_only_bytes_before_marker_within_one_chunk(
    isolated_runs_root, isolated_log_root
):
    """Direct coverage for the Task 2 byte-preservation fix.

    `test_attach_relays_input_detaches_on_ctrl_c_and_restores_terminal` only
    catches a leaked `\\x03` *indirectly*: its wrapped Python script happens
    to raise `KeyboardInterrupt` if a stray `\\x03` reaches it, flipping
    status to `failed`. A wrapped agent that ignores SIGINT (or doesn't run
    Python) would leak the same byte with nothing catching it. This test
    proves delivery directly: the wrapped agent logs every byte it reads, a
    single write of `b"ab\\x03cd"` is sent as one chunk (matching
    `cmd_attach`'s "scan each raw read chunk for `\\x03`" behavior), and the
    test asserts (1) the bytes before `\\x03` (0x61 'a', 0x62 'b') were
    delivered, (2) `\\x03` itself was not, (3) nothing after it in the same
    chunk (0x63 'c', 0x64 'd') was either, and (4) the wrapped run stayed
    alive/running throughout — only `attach` detaches, never the run.

    Reads the persistent log file directly rather than `attach`'s own PTY
    output: `cmd_attach` returns immediately after forwarding the
    pre-`\\x03` bytes and detecting the detach marker, in the same loop
    iteration, without looping back to drain the log growth the wrapped
    agent's own `BYTE:..` writes produce — so by the time `attach` exits,
    those log lines have not necessarily been replayed onto its stdout yet.
    """
    env = _environment(isolated_runs_root, isolated_log_root)
    name = "attach-ctrlc-truncate"
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
        os.write(master_fd, b"ab\x03cd")
        assert process.wait(timeout=5) == 0

        log_path = isolated_log_root / name / "log"
        assert _wait_until(lambda: b"BYTE:62" in log_path.read_bytes(), timeout=5)
        log = log_path.read_bytes()
        assert b"BYTE:61" in log  # 'a' delivered
        assert b"BYTE:62" in log  # 'b' delivered
        assert b"BYTE:03" not in log  # '\x03' itself never delivered
        assert b"BYTE:63" not in log  # 'c' (after \x03 in the chunk) dropped
        assert b"BYTE:64" not in log  # 'd' (after \x03 in the chunk) dropped

        assert (state / "status").read_text().strip() == "running"
        assert agent_run._pid_alive(int((state / "pid").read_text()))
    finally:
        _detach(process, master_fd)
        _stop_run(state)


def test_find_ctrl_c_trigger_recognizes_raw_byte():
    assert agent_run._find_ctrl_c_trigger(b"ab\x03cd") == (2, 3)


def test_find_ctrl_c_trigger_recognizes_iterm_csi_form():
    # What iTerm2 actually sends for Ctrl-C once Claude Code's TUI has
    # negotiated the "disambiguate escape codes" keyboard protocol: the
    # legacy xterm modifyOtherKeys form, ESC[27;<mod>;<codepoint>~, with
    # mod=5 (1 + Ctrl's bit 4) and codepoint=99 ('c').
    data = b"ab\x1b[27;5;99~cd"
    start, end = agent_run._find_ctrl_c_trigger(data)
    assert data[:start] == b"ab"
    assert data[end:] == b"cd"


def test_find_ctrl_c_trigger_recognizes_kitty_csi_u_form():
    data = b"ab\x1b[99;5ucd"
    start, end = agent_run._find_ctrl_c_trigger(data)
    assert data[:start] == b"ab"
    assert data[end:] == b"cd"


@pytest.mark.parametrize(
    "trigger",
    [
        b"\x1b[99;5:1u",  # explicit press event type
        b"\x1b[99;5:2u",  # repeat event type -- a held-down Ctrl-C
        b"\x1b[99;5:3u",  # release event type
        b"\x1b[99:65:67;5u",  # shifted + base-layout alt-key-codes
        b"\x1b[99:65;5u",  # a single alt-key-code
        b"\x1b[99;5;99u",  # "report associated text" field, one codepoint
        b"\x1b[99;5;99:100u",  # associated text, multiple codepoints
        b"\x1b[99:65:67;5:1;99u",  # alt-codes + event type + associated text together
    ],
)
def test_find_ctrl_c_trigger_recognizes_kitty_optional_subparameters(trigger):
    # Terminals implementing the kitty keyboard protocol may include these
    # optional colon-separated subparameters even when the client (Claude
    # Code) only requested the base "disambiguate escape codes" flag --
    # some terminals apply their own defaults regardless of exactly what
    # was requested. Detection must not depend on the client's requested
    # flags matching the terminal's actual behavior.
    data = b"ab" + trigger + b"cd"
    start, end = agent_run._find_ctrl_c_trigger(data)
    assert data[:start] == b"ab"
    assert data[end:] == b"cd"
    assert data[start:end] == trigger


@pytest.mark.parametrize(
    "data",
    [
        b"\x1b[27;1;99~",  # mod=1 -> no modifiers held at all, not Ctrl-C
        b"\x1b[27;2;99~",  # mod=2 -> Shift only
        b"\x1b[27;3;99~",  # mod=3 -> Shift+Alt, still no Ctrl bit
        b"\x1b[99;1u",
        b"\x1b[99:65;1u",  # alt-key-code present, but still no Ctrl modifier
        b"\x1b[27;5;100~",  # codepoint 100 = 'd', not 'c'
        b"\x1b[100;5u",  # kitty form, wrong codepoint
    ],
)
def test_find_ctrl_c_trigger_ignores_non_ctrl_csi_forms(data):
    assert agent_run._find_ctrl_c_trigger(data) == (-1, -1)


def test_split_trailing_incomplete_escape_holds_back_in_flight_csi():
    forwardable, held = agent_run._split_trailing_incomplete_escape(b"ab\x1b[27;5")
    assert forwardable == b"ab"
    assert held == b"\x1b[27;5"


def test_split_trailing_incomplete_escape_releases_terminated_sequence():
    data = b"ab\x1b[27;5;99~"
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


def test_split_trailing_incomplete_escape_bounds_pathological_sequences():
    data = b"\x1b[" + b"9" * agent_run._MAX_PENDING_ESCAPE_BYTES
    forwardable, held = agent_run._split_trailing_incomplete_escape(data)
    assert forwardable == data
    assert held == b""


def test_attach_ctrl_c_via_csi_sequence_detaches_without_leaking(
    isolated_runs_root, isolated_log_root
):
    """Reproduces the real-world bug: iTerm2 (and any terminal honoring the
    kitty/CSI-u "disambiguate escape codes" keyboard protocol Claude Code's
    TUI requests) sends Ctrl-C as `ESC[27;5;99~`, not a raw `\\x03` byte. A
    detector that only looks for `\\x03` never fires, and the whole escape
    sequence leaks straight into the wrapped agent's stdin instead of
    detaching `attach` locally."""
    env = _environment(isolated_runs_root, isolated_log_root)
    name = "attach-ctrlc-csi"
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
        os.write(master_fd, b"ab\x1b[27;5;99~cd")
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


def test_attach_ctrl_c_via_kitty_extended_csi_form_detaches_without_leaking(
    isolated_runs_root, isolated_log_root
):
    """Terminals implementing the kitty keyboard protocol can send the
    optional alt-key-codes/event-type/associated-text subparameters even
    when Claude Code only requested the base disambiguation flag -- proves
    the whole attach pipeline (not just the unit-level regex) still
    detaches on one of those extended forms rather than only the minimal
    `ESC[99;5u`."""
    env = _environment(isolated_runs_root, isolated_log_root)
    name = "attach-ctrlc-kitty-ext"
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
        os.write(master_fd, b"ab\x1b[99:65:67;5:1ucd")
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


def test_attach_ctrl_c_via_csi_split_across_two_reads_still_detaches(
    isolated_runs_root, isolated_log_root
):
    """The CSI-u Ctrl-C trigger can legitimately arrive split across two
    local-terminal reads (e.g. a slow pty writer, or scheduler jitter). The
    held-back-prefix logic must reassemble it rather than either missing
    the detach or corrupting/duplicating forwarded bytes."""
    env = _environment(isolated_runs_root, isolated_log_root)
    name = "attach-ctrlc-csi-split"
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
        os.write(master_fd, b";99~cd")
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


def _spawn_tail(name: str, env: dict[str, str], rows: int = 24, cols: int = 80):
    master_fd, slave_fd = pty.openpty()
    fcntl.ioctl(slave_fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
    process = subprocess.Popen(
        [sys.executable, "-m", "toolbox.agent_run", "tail", name],
        stdin=slave_fd,
        stdout=slave_fd,
        stderr=slave_fd,
        env=env,
        start_new_session=True,
        close_fds=True,
    )
    os.close(slave_fd)
    return process, master_fd


def _detach_tail(process: subprocess.Popen, master_fd: int, trigger: bytes, timeout: float = 5.0) -> None:
    """Send `trigger` (raw 0x03 or a CSI Ctrl-C form) to a `tail` subprocess
    and wait for it to exit, draining its PTY master the whole time so a
    real terminal's TCSADRAIN-equivalent cleanup can never wedge -- same
    rationale as `_detach` for `attach`.

    Retries the write: `tail` briefly buffers unread bytes in the kernel's
    raw-mode input queue until it reaches its select() loop and actually
    reads them, but there is no readiness marker to wait on (unlike
    `attach`, `tail` never prints anything of its own on startup), so a
    single early write can race that transition. Re-sending is harmless --
    once raw mode is active the first delivered trigger detaches
    immediately and every retry after that becomes a no-op against an
    already-exited process.
    """
    try:
        deadline = time.monotonic() + timeout
        while process.poll() is None and time.monotonic() < deadline:
            try:
                os.write(master_fd, trigger)
            except OSError:
                break
            readable, _, _ = select.select([master_fd], [], [], 0.1)
            if readable:
                try:
                    os.read(master_fd, 65536)
                except OSError:
                    break
        assert process.wait(timeout=max(0.0, deadline - time.monotonic())) == 128 + signal.SIGINT
    finally:
        os.close(master_fd)


def test_tail_raw_ctrl_c_exits_without_forwarding_anywhere(
    isolated_runs_root, isolated_log_root
):
    """A plain raw 0x03 must still stop `tail` once stdin is in raw mode
    (the kernel's own cooked-mode SIGINT generation is disabled by ISIG in
    raw mode, so `tail` must detect the byte itself)."""
    env = _environment(isolated_runs_root, isolated_log_root)
    name = "tail-ctrlc-raw"
    script = "import time; time.sleep(30)"
    state = _launch_interactive(name, script, env)
    process, master_fd = _spawn_tail(name, env)
    try:
        _detach_tail(process, master_fd, b"\x03")
        assert (state / "status").read_text().strip() == "running"
        assert agent_run._pid_alive(int((state / "pid").read_text()))
    finally:
        if process.poll() is None:
            process.terminate()
        _stop_run(state)


def test_tail_csi_ctrl_c_exits_cleanly(isolated_runs_root, isolated_log_root):
    """Reproduces the real-world bug: once the wrapped agent's TUI (e.g.
    Claude Code) negotiates the kitty/CSI-u keyboard protocol via output
    `tail` faithfully replays, a terminal honoring it (iTerm2, kitty, ...)
    sends Ctrl-C as `ESC[27;5;99~` instead of raw 0x03. The kernel's
    cooked-mode SIGINT generation never recognizes that sequence, so
    without active detection `tail` never exits on Ctrl-C at all -- the
    escape bytes just sit unread (and, in cooked mode, get locally echoed
    as garbage)."""
    env = _environment(isolated_runs_root, isolated_log_root)
    name = "tail-ctrlc-csi"
    script = "import time; time.sleep(30)"
    state = _launch_interactive(name, script, env)
    process, master_fd = _spawn_tail(name, env)
    try:
        _detach_tail(process, master_fd, b"\x1b[27;5;99~")
        assert (state / "status").read_text().strip() == "running"
        assert agent_run._pid_alive(int((state / "pid").read_text()))
    finally:
        if process.poll() is None:
            process.terminate()
        _stop_run(state)


def test_tail_drains_mouse_garbage_without_echoing_it(
    isolated_runs_root, isolated_log_root
):
    """Mouse-tracking escape sequences arriving on stdin while `tail` runs
    (because the wrapped TUI enabled mouse tracking and the real terminal
    is now sending motion events) must be silently discarded, not echoed
    onto the screen or left sitting in the terminal's input queue. Raw mode
    disables local echo; actively reading stdin drains them.

    Repeatedly re-sends the mouse report over ~1s rather than a single
    write immediately after spawn: `tail` has no readiness marker of its
    own (unlike `attach`, it prints nothing on startup), so a single early
    write could race the transition into raw mode. Re-sending is harmless
    here since every send is independently expected to be silently
    discarded."""
    env = _environment(isolated_runs_root, isolated_log_root)
    name = "tail-mouse-garbage"
    script = "import time; time.sleep(30)"
    state = _launch_interactive(name, script, env)
    process, master_fd = _spawn_tail(name, env)
    try:
        # A representative SGR mouse-motion report, distinct from any
        # Ctrl-C trigger form.
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            try:
                os.write(master_fd, b"\x1b[<35;10;5M")
            except OSError:
                break
            time.sleep(0.1)
        assert process.poll() is None  # tail must NOT have exited on this
    finally:
        _detach_tail(process, master_fd, b"\x03")
        _stop_run(state)


def test_tail_bare_escape_does_not_exit_or_hang(isolated_runs_root, isolated_log_root):
    """A bare Escape byte (no following CSI bytes) must be drained and
    ignored -- not mistaken for the start of a Ctrl-C sequence and held
    forever, and not confused with the detach trigger itself. See
    test_tail_drains_mouse_garbage_without_echoing_it for why this retries
    the write instead of sending once."""
    env = _environment(isolated_runs_root, isolated_log_root)
    name = "tail-bare-esc"
    script = "import time; time.sleep(30)"
    state = _launch_interactive(name, script, env)
    process, master_fd = _spawn_tail(name, env)
    try:
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            try:
                os.write(master_fd, b"\x1b")
            except OSError:
                break
            time.sleep(0.1)
        assert process.poll() is None
    finally:
        _detach_tail(process, master_fd, b"\x03")
        _stop_run(state)


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


