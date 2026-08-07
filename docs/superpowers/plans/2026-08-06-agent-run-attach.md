# `agent-run attach` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `agent-run attach <name>` so users can view and drive an existing interactive run from a local terminal, with terminal-size propagation to the run's PTY.

**Architecture:** Interactive runs receive a new `resize` FIFO alongside the existing `stdin` FIFO. The detached runner consumes fixed-size resize records and updates its PTY master. `attach` is a single-process raw-terminal client: it tails the log, relays local bytes through `stdin`, and sends the current terminal size initially and after local `SIGWINCH` events.

**Tech Stack:** Python 3.11 standard library (`fcntl`, `os`, `pty`, `select`, `signal`, `struct`, `termios`, `tty`), pytest, POSIX FIFOs, and PTYs.

## Global Constraints

- Preserve the detached lifecycle: detaching must not terminate or alter the wrapped agent.
- `agent-run tail` and `agent-run steer` retain their existing behavior.
- Multiple attaches are allowed; keystrokes are shared and the most recently delivered resize wins.
- Ctrl-C (`b"\x03"`) detaches locally and must never reach the wrapped agent through `attach`.
- All other input bytes, including escape sequences and control bytes, pass through unchanged.
- Always restore the attaching terminal's original termios attributes on normal detach, wrapped-agent exit, and exceptions.
- On wrapped-agent exit, drain final log output and return 0, matching `tail`.
- Keep the existing log-polling output model; do not add a subscriber daemon, socket, or new dependency.
- Support macOS/Darwin and POSIX platforms with FIFOs and PTYs.
- Do not push or open a pull request unless separately requested.
- `attach` replays raw PTY-captured bytes to a real terminal exactly like `tail`/`logs` do, so it must call the existing `_reset_terminal_modes()` (added in PR #12, `src/toolbox/agent_run.py:1335`) in its cleanup path, the same way `cmd_tail` does — otherwise DEC private modes (mouse tracking, alt-screen, bracketed paste) left on by the wrapped TUI leak into the attaching terminal after detach.

---

## File Structure

| File | Responsibility |
| --- | --- |
| `src/toolbox/agent_run.py` | Resize protocol, resize FIFO lifecycle, runner-side resize application, attach client, and CLI wiring. |
| `tests/test_agent_run_attach.py` | Unit coverage for protocol and validation plus PTY/FIFO integration coverage for attach behavior. |
| `README.md` | User-facing launch, attach, detach, and multi-attach usage. |

## Shared interfaces

Add these names to `src/toolbox/agent_run.py`:

```python
RESIZE_RECORD_FORMAT = ">HH"
RESIZE_RECORD_SIZE = struct.calcsize(RESIZE_RECORD_FORMAT)


def _pack_resize(cols: int, rows: int) -> bytes: ...
def _apply_resize(master_fd: int, cols: int, rows: int) -> None: ...
def _drain_resize_records(master_fd: int, buffered: bytes) -> bytes: ...
def _require_live_interactive_run(name: str) -> tuple[Path, int]: ...
def cmd_attach(args: argparse.Namespace) -> int: ...
```

`_pack_resize` accepts only dimensions in `1..65535` and emits `struct.pack(">HH", cols, rows)`. `_apply_resize` converts to the kernel's `rows, cols, xpixel, ypixel` order. `_drain_resize_records` calls `_apply_resize` for every complete four-byte record and returns an incomplete suffix unchanged.

---

### Task 1: Implement the resize protocol and runner relay

**Files:**
- Modify: `src/toolbox/agent_run.py` imports/constants near line 191-218, the interactive FIFO setup in `_cmd_launch_locked` near line 2495-2512, and the `_run_interactive` relay near line 3015-3256 (exact line numbers may have drifted further by the time this task starts — locate by the `import fcntl`/`MAX_PTY_INPUT_BUFFER` constants, the `fifo_path: Optional[Path] = None` line in `_cmd_launch_locked`, and `_drain_pty_input`/`_run_interactive` respectively, not by line number alone)
- Create: `tests/test_agent_run_attach.py`

**Consumes:** The existing interactive launch state directory and the `stdin` FIFO keeper design.

**Produces:** An interactive run has both `stdin` and `resize` FIFOs; its runner applies complete resize records to the PTY master, preserving partial records between `select()` iterations.

- [ ] **Step 1: Add failing unit tests for resize encoding and framing**

Create `tests/test_agent_run_attach.py` with:

```python
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
```

- [ ] **Step 2: Run the focused tests and confirm they fail**

Run: `pytest tests/test_agent_run_attach.py -v`

Expected: collection fails because the resize constants and helpers do not exist.

- [ ] **Step 3: Add the resize protocol helpers**

In `src/toolbox/agent_run.py`, add imports beside the existing standard-library imports:

```python
import struct
import termios
import tty
```

Add constants directly after `MAX_PTY_INPUT_BUFFER`:

```python
RESIZE_RECORD_FORMAT = ">HH"
RESIZE_RECORD_SIZE = struct.calcsize(RESIZE_RECORD_FORMAT)
```

Add these helpers immediately after `_drain_pty_input`:

```python
def _pack_resize(cols: int, rows: int) -> bytes:
    if not (1 <= cols <= 0xFFFF and 1 <= rows <= 0xFFFF):
        raise ValueError("terminal dimensions must be between 1 and 65535")
    return struct.pack(RESIZE_RECORD_FORMAT, cols, rows)


def _apply_resize(master_fd: int, cols: int, rows: int) -> None:
    try:
        fcntl.ioctl(
            master_fd,
            termios.TIOCSWINSZ,
            struct.pack("HHHH", rows, cols, 0, 0),
        )
    except OSError as exc:
        if exc.errno not in {errno.EBADF, errno.EIO, errno.EINVAL, errno.ENOTTY}:
            raise


def _drain_resize_records(master_fd: int, buffered: bytes) -> bytes:
    while len(buffered) >= RESIZE_RECORD_SIZE:
        record, buffered = buffered[:RESIZE_RECORD_SIZE], buffered[RESIZE_RECORD_SIZE:]
        cols, rows = struct.unpack(RESIZE_RECORD_FORMAT, record)
        _apply_resize(master_fd, cols, rows)
    return buffered
```

- [ ] **Step 4: Run the protocol tests and confirm they pass**

Run: `pytest tests/test_agent_run_attach.py -v`

Expected: PASS.

- [ ] **Step 5: Add a failing interactive-launch FIFO lifecycle test**

Append this test:

```python
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
```

- [ ] **Step 6: Run the lifecycle test and confirm it fails**

Run: `pytest tests/test_agent_run_attach.py::test_interactive_launch_creates_stdin_and_resize_fifos -v`

Expected: FAIL because `state/resize` does not exist.

- [ ] **Step 7: Create and retain the resize FIFO**

In `_cmd_launch_locked`, replace the single interactive `fifo_path` variable with an iterable of control paths. Create both before the readiness pipe and remove every created path if FIFO or readiness-pipe setup fails:

```python
fifo_paths: tuple[Path, ...] = ()
if args.interactive:
    fifo_paths = (d / "stdin", d / "resize")
    try:
        for fifo_path in fifo_paths:
            os.mkfifo(str(fifo_path))
    except OSError as exc:
        for fifo_path in fifo_paths:
            try:
                fifo_path.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                pass
        sys.exit(f"agent-run: failed to create control fifo: {exc}")
```

Keep state directories intact on normal run completion; existing reaping owns final removal.

In `_run_interactive`, rename `fifo_path` to `stdin_path` and define `resize_path = state_dir / "resize"`. In the keeper child open both paths with `O_RDWR`, acknowledge only after both opens succeed, then close both in its existing `finally` block. In the runner parent open both FIFOs read-only after the keeper acknowledgement.

Set `O_NONBLOCK` on `resize_fd` beside `master_fd` and `fifo_fd`. Initialize `buf_resize = b""`. Change the read set from `[master_fd, fifo_fd]` to `[master_fd, fifo_fd, resize_fd]`. On a readable resize fd, append `os.read(resize_fd, 4096)` to `buf_resize`, then assign:

```python
buf_resize = _drain_resize_records(master_fd, buf_resize)
```

Close `resize_fd` in the existing cleanup block before closing `master_fd`.

- [ ] **Step 8: Run resize lifecycle plus existing interactive regressions**

Run: `pytest tests/test_agent_run_attach.py -v && pytest tests/test_agent_run_submission.py tests/test_agent_run_r1_r10.py -v`

Expected: PASS. Existing steering and relay failure behavior remain unchanged.

- [ ] **Step 9: Commit the resize relay**

```bash
git add src/toolbox/agent_run.py tests/test_agent_run_attach.py
git commit -m "feat(agent-run): relay terminal resize events to the PTY"
```

### Task 2: Add the attach command and safe terminal relay

**Files:**
- Modify: `src/toolbox/agent_run.py` — `cmd_attach` goes immediately after `cmd_tail` (currently near line 1372-1417; locate by the `def cmd_tail` signature), the shared validation helper goes immediately before `cmd_steer` (currently near line 2100; locate by `def cmd_steer`), and the CLI parser/dispatch changes go in `_build_parser`/`main` (locate by `sub.add_parser("tail"` and `known_subcommands = {`)
- Modify: `tests/test_agent_run_attach.py`

**Consumes:** The `stdin` and `resize` FIFOs from Task 1, `_pack_resize`, and the existing log-following behavior.

**Produces:** `cmd_attach(args)` uses one `select()` loop to stream log output and relay input. It has no exclusivity lock and uses a bounded local input buffer.

- [ ] **Step 1: Add failing validation and CLI-dispatch tests**

Append:

```python
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
```

Add `import io` to this test file's imports if not already present from Task 1.

- [ ] **Step 2: Run the validation tests and confirm they fail**

Run: `pytest tests/test_agent_run_attach.py -k 'attach_rejects or main_dispatches_attach' -v`

Expected: collection fails because `cmd_attach` does not exist.

- [ ] **Step 3: Factor common live-interactive validation**

Add this helper immediately before `cmd_steer`:

```python
def _require_live_interactive_run(name: str) -> tuple[Path, int]:
    d = _require_state(name)
    if _read(d / "interactive") != "1":
        sys.exit(
            f"agent-run: '{name}' is not interactive. "
            f"Relaunch with: agent-run -i {name} <command...>"
        )
    pid = _require_positive_state_int(d, "pid", name)
    if not _pid_alive(pid):
        sys.exit(f"agent-run: '{name}' is not running")
    return d, pid
```

Refactor `cmd_steer` to call this helper after validating the name, preserving all its current FIFO-write behavior and error messages.

- [ ] **Step 4: Implement `cmd_attach`**

Add it immediately after `cmd_tail`, keeping log code close to the command it intentionally mirrors. `attach` replays raw PTY bytes to a real terminal exactly like `tail`/`logs` do, so it must call the existing `_reset_terminal_modes()` helper (added in PR #12, defined just above `cmd_logs`) in its cleanup path, and it must handle an external `KeyboardInterrupt` (e.g. `kill -INT` on the attach process, distinct from the in-band Ctrl-C byte handled below) the same quiet way `cmd_tail` now does — return `128 + signal.SIGINT` instead of a traceback. Use local nonblocking FIFO writers and bounded buffers so a wedged runner cannot freeze the attach terminal.

```python
def cmd_attach(args: argparse.Namespace) -> int:
    name = _validate_run_name(args.name)
    state_dir, pid = _require_live_interactive_run(name)
    log = _require_log(name)
    stdin_path = state_dir / "stdin"
    resize_path = state_dir / "resize"
    if not stdin_path.is_fifo():
        sys.exit(f"agent-run: no stdin FIFO at {stdin_path}")
    if not resize_path.is_fifo():
        sys.exit(f"agent-run: no resize FIFO at {resize_path}")

    local_fd = sys.stdin.fileno()
    saved_termios = termios.tcgetattr(local_fd)
    stdin_fd = os.open(str(stdin_path), os.O_WRONLY | os.O_NONBLOCK)
    resize_fd = os.open(str(resize_path), os.O_WRONLY | os.O_NONBLOCK)
    pending_input = b""
    pending_resize: Optional[bytes] = None
    resize_requested = True
    previous_winch = None

    try:
        tty.setraw(local_fd)

        def on_winch(_signum, _frame):
            nonlocal resize_requested
            resize_requested = True

        previous_winch = signal.signal(signal.SIGWINCH, on_winch)
        with log.open("rb") as log_file:
            while True:
                if resize_requested:
                    size = os.get_terminal_size(local_fd)
                    pending_resize = _pack_resize(size.columns, size.lines)
                    resize_requested = False

                writable = []
                if pending_input:
                    writable.append(stdin_fd)
                if pending_resize:
                    writable.append(resize_fd)
                readable, writable, _ = select.select([local_fd], writable, [], 0.2)

                chunk = log_file.read(8192)
                if chunk:
                    sys.stdout.buffer.write(chunk)
                    sys.stdout.buffer.flush()

                if not _pid_alive(pid):
                    time.sleep(0.1)
                    remaining = log_file.read()
                    if remaining:
                        sys.stdout.buffer.write(remaining)
                        sys.stdout.buffer.flush()
                    return 0

                if local_fd in readable:
                    data = os.read(local_fd, 4096)
                    if not data or b"\x03" in data:
                        return 0
                    if len(pending_input) + len(data) > MAX_PTY_INPUT_BUFFER:
                        raise BufferError(
                            f"attach input buffer exceeded {MAX_PTY_INPUT_BUFFER} bytes"
                        )
                    pending_input += data

                if stdin_fd in writable and pending_input:
                    try:
                        written = os.write(stdin_fd, pending_input)
                    except BlockingIOError:
                        written = 0
                    pending_input = pending_input[written:]

                if resize_fd in writable and pending_resize:
                    try:
                        os.write(resize_fd, pending_resize)
                    except BlockingIOError:
                        pass
                    else:
                        pending_resize = None
    except BrokenPipeError:
        return 0
    except KeyboardInterrupt:
        return 128 + signal.SIGINT
    finally:
        if previous_winch is not None:
            signal.signal(signal.SIGWINCH, previous_winch)
        os.close(stdin_fd)
        os.close(resize_fd)
        termios.tcsetattr(local_fd, termios.TCSADRAIN, saved_termios)
        _reset_terminal_modes()
```

Use a small local helper if needed to close fds conditionally when an `os.open` between the two opens fails. Do not suppress unexpected exceptions: the `finally` must restore the terminal and failure should otherwise remain visible.

Register the subcommand directly after `tail`:

```python
sp_attach = sub.add_parser(
    "attach",
    help="attach interactively (live keyboard + resize; Ctrl-C detaches)",
)
sp_attach.add_argument("name")
sp_attach.set_defaults(func=cmd_attach)
```

Add `"attach"` to `known_subcommands` in `main()`.

- [ ] **Step 5: Run the focused tests and confirm they pass**

Run: `pytest tests/test_agent_run_attach.py -k 'attach_rejects or main_dispatches_attach' -v && pytest tests/test_agent_run_submission.py -v`

Expected: PASS, including existing steering tests after validation factoring.

- [ ] **Step 6: Commit the attach command**

```bash
git add src/toolbox/agent_run.py tests/test_agent_run_attach.py
git commit -m "feat(agent-run): add interactive attach command"
```

### Task 3: Add end-to-end PTY attach coverage

**Files:**
- Modify: `tests/test_agent_run_attach.py`

**Consumes:** The `agent-run -i` launcher and the `attach` command from Tasks 1–2.

**Produces:** Regression coverage for input, Ctrl-C detach, terminal restoration, initial and changed size, concurrent attach last-resize-wins, and agent-exit log drain.

- [ ] **Step 1: Add real-process test helpers**

Add imports and helpers:

```python
import fcntl
import pty
import select
import subprocess
import termios
from pathlib import Path


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


def _stop_run(state: Path) -> None:
    try:
        os.kill(int((state / "pid").read_text()), signal.SIGTERM)
    except ProcessLookupError:
        return
    assert _wait_until(lambda: (state / "status").read_text().strip() != "running")


def _detach(process: subprocess.Popen[bytes], master_fd: int) -> None:
    try:
        if process.poll() is None:
            os.write(master_fd, b"\x03")
            assert process.wait(timeout=5) == 0
    finally:
        os.close(master_fd)
```

- [ ] **Step 2: Add input relay, Ctrl-C-detach, and terminal-restoration test**

```python
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
```

- [ ] **Step 3: Add initial-size and resize propagation test**

```python
def test_attach_applies_initial_and_changed_terminal_size(
    isolated_runs_root, isolated_log_root
):
    env = _environment(isolated_runs_root, isolated_log_root)
    script = """\
import os, sys, termios, tty
fd = sys.stdin.fileno()
saved = termios.tcgetattr(fd)
tty.setraw(fd)
for _ in range(2):
    os.read(fd, 1)
    size = os.get_terminal_size(fd)
    print(f"SIZE:{size.columns}x{size.lines}", flush=True)
termios.tcsetattr(fd, termios.TCSADRAIN, saved)
"""
    state = _launch_interactive("attach-size", script, env)
    process, master_fd = _spawn_attach("attach-size", env, rows=31, cols=101)
    try:
        os.write(master_fd, b"s")
        _read_until(master_fd, b"SIZE:101x31")
        fcntl.ioctl(master_fd, termios.TIOCSWINSZ, struct.pack("HHHH", 47, 133, 0, 0))
        os.kill(process.pid, signal.SIGWINCH)
        os.write(master_fd, b"s")
        _read_until(master_fd, b"SIZE:133x47")
    finally:
        _detach(process, master_fd)
        _stop_run(state)
```

- [ ] **Step 4: Add multi-attach last-resize-wins test**

```python
def test_multiple_attaches_share_output_and_last_resize_wins(
    isolated_runs_root, isolated_log_root
):
    env = _environment(isolated_runs_root, isolated_log_root)
    script = """\
import os, sys, termios, tty
fd = sys.stdin.fileno()
saved = termios.tcgetattr(fd)
tty.setraw(fd)
for _ in range(2):
    os.read(fd, 1)
    size = os.get_terminal_size(fd)
    print(f"SIZE:{size.columns}x{size.lines}", flush=True)
termios.tcsetattr(fd, termios.TCSADRAIN, saved)
"""
    state = _launch_interactive("attach-many", script, env)
    first, first_master = _spawn_attach("attach-many", env, rows=24, cols=80)
    second, second_master = _spawn_attach("attach-many", env, rows=50, cols=120)
    try:
        os.write(second_master, b"s")
        _read_until(second_master, b"SIZE:120x50")
        fcntl.ioctl(first_master, termios.TIOCSWINSZ, struct.pack("HHHH", 40, 90, 0, 0))
        os.kill(first.pid, signal.SIGWINCH)
        os.write(first_master, b"s")
        _read_until(first_master, b"SIZE:90x40")
        _read_until(second_master, b"SIZE:90x40")
    finally:
        _detach(first, first_master)
        _detach(second, second_master)
        _stop_run(state)
```

- [ ] **Step 5: Add final-output and wrapped-agent-exit test**

```python
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
```

- [ ] **Step 6: Run end-to-end and full regression suites**

Run: `pytest tests/test_agent_run_attach.py -v && pytest -v`

Expected: PASS. If an integration test is flaky, replace arbitrary sleeps with marker reads or `_wait_until` checks; do not lengthen global timeouts as the primary fix.

- [ ] **Step 7: Commit integration coverage**

```bash
git add tests/test_agent_run_attach.py
git commit -m "test(agent-run): cover interactive attach sessions"
```

### Task 4: Document the attach workflow

**Files:**
- Modify: `README.md:16` and add an `agent-run` section before `## gemini-image`
- Test: `tests/test_agent_run_attach.py`

**Consumes:** The public `agent-run attach <name>` CLI contract from Task 2.

**Produces:** Discoverable and accurate usage instructions, including the deliberate Ctrl-C and multi-attach semantics.

- [ ] **Step 1: Add a CLI-help test**

```python
def test_attach_is_listed_in_cli_help(capsys):
    assert agent_run.main(["--help"]) == 0
    assert "attach" in capsys.readouterr().out
```

- [ ] **Step 2: Run the help test**

Run: `pytest tests/test_agent_run_attach.py::test_attach_is_listed_in_cli_help -v`

Expected: PASS.

- [ ] **Step 3: Update README**

Change the table row at `README.md:16` to:

```markdown
| `agent-run` | Background wrapper for coding agents (Claude Code, Codex…) with interactive attach/steering + live log streaming | — |
```

Insert this section before `## gemini-image`:

```markdown
## agent-run

Launch a coding agent in a detached, interactive PTY:

```bash
agent-run -i review-agent claude
```

Attach from any terminal to see live output and drive it with normal
keystrokes:

```bash
agent-run attach review-agent
```

`attach` adopts the attached terminal's size immediately and after each
window resize. Press `Ctrl-C` to detach without stopping the agent. Multiple
attach sessions are allowed; keystrokes are shared and the latest resize
wins. Use `agent-run tail review-agent` for read-only output, or
`agent-run steer review-agent "message"` for one-shot prompt submission.
```

- [ ] **Step 4: Run final automated verification**

Run: `pytest tests/test_agent_run_attach.py -v && pytest -v && python -m toolbox.agent_run --help`

Expected: both pytest runs pass and help lists `attach` with Ctrl-C-detach wording.

- [ ] **Step 5: Commit documentation**

```bash
git add README.md tests/test_agent_run_attach.py
git commit -m "docs(agent-run): describe interactive attach workflow"
```

## Manual verification

- [ ] From a real terminal, run `agent-run -i smoke-test <interactive-command>`.
- [ ] Run `agent-run attach smoke-test`, send normal input, resize the terminal, and verify the wrapped command observes the new size.
- [ ] Press Ctrl-C; verify the attach client exits, restores the terminal, and `agent-run status smoke-test` remains running.
- [ ] Attach twice from differently sized terminals; verify the latest resize takes effect while both clients continue receiving output.
- [ ] Run `git status --short`; verify only planned source, test, documentation, and plan files changed.

## Plan self-review

- **Spec coverage:** Task 1 creates the dedicated resize FIFO and fixed `>HH` protocol; Task 2 adds raw keyboard relay, initial/SIGWINCH resize events, Ctrl-C local detach, log polling, validation, and parser dispatch; Task 3 proves live input/output, terminal restoration, resize behavior, concurrent attaches, and exit draining; Task 4 documents the workflow.
- **Scope:** The work stays within the existing `agent_run.py` state/FIFO architecture and adds no background output subscribers, attach ownership tracking, or dependencies.
- **Consistency:** Resize records are `(cols, rows)` in `>HH` and converted to `TIOCSWINSZ`'s `(rows, cols, 0, 0)`. Multiple writers rely on atomic four-byte FIFO records; `attach` input uses a bounded nonblocking buffer.
