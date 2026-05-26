#!/usr/bin/env python3
"""agent-run — Wrapper for coding agents (Claude Code, Codex, etc.).

Creates a run directory under /tmp/agent-runs/<name>/ with structured state
files so LLM orchestrators can poll safely without brittle process-poll loops,
and optionally allocates a real PTY so TUI agents behave as if attached to a
terminal (no more 0-CPU hangs from isatty() checks).

Usage::

    agent-run <name> <cmd...>            # non-interactive (one-shot)
    agent-run -i <name> <cmd...>         # interactive (PTY-wrapped, steerable)
    agent-run tail <name>                # follow log in real time
    agent-run logs <name> [N]            # last N lines (default 50)
    agent-run status <name>              # one-line status
    agent-run steer <name> <msg...>      # send text to agent stdin (needs -i)
    agent-run kill <name> [SIGNAL]       # TERM by default; 9/KILL if stuck
    agent-run list                       # list all runs

Files under /tmp/agent-runs/<name>/::

    status       running | done | failed
    exit_code    numeric exit code (after completion)
    pid          launcher pid
    pgid         process group id (kill target)
    pty_pid      PTY child pid (interactive only)
    keeper_pid   FIFO-keeper pid (interactive only)
    log          captured stdout+stderr (PTY-captured when interactive)
    command      pretty-printed launch command
    argv         JSON-encoded argv (authoritative form for replay)
    started_at   ISO-8601 UTC
    ended_at     ISO-8601 UTC (after completion)
    interactive  "1" if launched with -i, else "0"
    stdin        FIFO for steering (interactive only)
"""

from __future__ import annotations

import argparse
import errno
import fcntl
import json
import os
import pty
import select
import shlex
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence


RUNS_ROOT = Path(os.environ.get("AGENT_RUN_DIR", "/tmp/agent-runs"))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write(path: Path, text: str) -> None:
    path.write_text(text)


def _read(path: Path, default: str = "") -> str:
    try:
        return path.read_text().strip()
    except FileNotFoundError:
        return default


def _run_dir(name: str) -> Path:
    return RUNS_ROOT / name


def _require_run(name: str) -> Path:
    d = _run_dir(name)
    if not d.is_dir():
        sys.exit(f"agent-run: no run named '{name}' in {RUNS_ROOT}")
    return d


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but we can't signal it — still "alive" for our purposes.
        return True


def _pretty_command(argv: Sequence[str]) -> str:
    return " ".join(shlex.quote(a) for a in argv)


# ---------------------------------------------------------------------------
# list / status / logs / tail
# ---------------------------------------------------------------------------

def cmd_list(_args: argparse.Namespace) -> int:
    print(f"Active runs ({RUNS_ROOT}):")
    if not RUNS_ROOT.is_dir():
        print("  (none)")
        return 0
    runs = sorted(p for p in RUNS_ROOT.iterdir() if p.is_dir())
    if not runs:
        print("  (none)")
        return 0
    for d in runs:
        status = _read(d / "status", "unknown")
        pid = _read(d / "pid", "?")
        started = _read(d / "started_at", "?")
        lines = 0
        try:
            with (d / "log").open("rb") as f:
                lines = sum(1 for _ in f)
        except FileNotFoundError:
            pass
        interactive = _read(d / "interactive", "0")
        flag = " [interactive]" if interactive == "1" else ""
        print(f"  {d.name}: status={status} pid={pid} started={started} lines={lines}{flag}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    d = _require_run(args.name)
    name = d.name
    status = _read(d / "status", "unknown")
    pid = _read(d / "pid", "?")
    started = _read(d / "started_at", "?")
    ended = _read(d / "ended_at", "-")
    exit_code = _read(d / "exit_code", "-")
    interactive = _read(d / "interactive", "0")
    try:
        with (d / "log").open("rb") as f:
            lines = sum(1 for _ in f)
    except FileNotFoundError:
        lines = 0
    print(
        f"name={name} status={status} pid={pid} exit={exit_code} "
        f"started={started} ended={ended} lines={lines} interactive={interactive}"
    )
    return 0


def cmd_logs(args: argparse.Namespace) -> int:
    d = _require_run(args.name)
    n = max(1, args.n)
    log = d / "log"
    if not log.exists():
        return 0
    # Read tail-n efficiently for large logs.
    with log.open("rb") as f:
        f.seek(0, os.SEEK_END)
        end = f.tell()
        block = 8192
        data = b""
        pos = end
        while pos > 0 and data.count(b"\n") <= n:
            read_size = min(block, pos)
            pos -= read_size
            f.seek(pos)
            data = f.read(read_size) + data
    lines = data.splitlines()
    for line in lines[-n:]:
        try:
            sys.stdout.buffer.write(line + b"\n")
        except BrokenPipeError:
            break
    return 0


def cmd_tail(args: argparse.Namespace) -> int:
    d = _require_run(args.name)
    log = d / "log"
    log.touch(exist_ok=True)
    pid_raw = _read(d / "pid")
    try:
        pid = int(pid_raw) if pid_raw else None
    except ValueError:
        pid = None
    # Stream the whole file then tail until the agent dies or log stops growing.
    with log.open("rb") as f:
        while True:
            chunk = f.read(8192)
            if chunk:
                try:
                    sys.stdout.buffer.write(chunk)
                    sys.stdout.buffer.flush()
                except BrokenPipeError:
                    return 0
                continue
            # EOF. If agent dead, exit. Otherwise sleep and retry.
            if pid is not None and not _pid_alive(pid):
                # One more drain to catch final writes.
                time.sleep(0.1)
                remaining = f.read()
                if remaining:
                    try:
                        sys.stdout.buffer.write(remaining)
                        sys.stdout.buffer.flush()
                    except BrokenPipeError:
                        pass
                return 0
            time.sleep(0.2)


# ---------------------------------------------------------------------------
# clean (render PTY-captured logs into readable transcripts)
# ---------------------------------------------------------------------------

def _render_log(raw: bytes, width: int = 120, height: int = 60, history: int = 100000) -> str:
    """Render a raw PTY-captured log (with ANSI/Ink redraw artifacts) into a
    plain-text transcript by replaying the byte stream through a VT100
    emulator (pyte). Returns the deduplicated screen history + final visible
    viewport, joined with newlines.

    Pyte is loaded lazily so the rest of agent-run keeps working even if the
    extra is not installed (and so we get a clear error message when it is
    really needed).
    """
    try:
        import pyte  # type: ignore
    except ImportError:
        sys.exit(
            "agent-run: `pyte` is required for `clean` / --echo. "
            "Install with: pipx inject mmartins-toolbox pyte  (or uv tool install --with pyte ...)"
        )

    screen = pyte.HistoryScreen(width, height, history=history, ratio=0.5)
    stream = pyte.ByteStream(screen)
    stream.feed(raw)

    rows: List[str] = []
    # Past history rows that have scrolled off the top.
    for entry in screen.history.top:
        text = "".join(entry[col].data for col in sorted(entry)) if entry else ""
        rows.append(text.rstrip())
    # Currently-visible viewport.
    for row in screen.display:
        rows.append(row.rstrip())

    # Collapse adjacent duplicate lines (Ink redraws the same content many times).
    deduped: List[str] = []
    for line in rows:
        if not deduped or deduped[-1] != line:
            deduped.append(line)
    # Trim trailing empties.
    while deduped and not deduped[-1]:
        deduped.pop()
    return "\n".join(deduped) + "\n"


def _echo_loop(run_dir: "Path", interval: float) -> None:
    """Periodically render run_dir/log into run_dir/log.clean.

    Runs in a detached child for the lifetime of the agent. The parent's
    signal handler kills us on shutdown so we don't outlive the run. We
    only re-render when the raw log's mtime has changed, so a quiet run
    doesn't burn CPU.
    """
    log = run_dir / "log"
    clean = run_dir / "log.clean"
    last_mtime = -1.0
    # Soft cap: if pyte isn't installed, write a friendly stub and exit.
    try:
        import pyte  # noqa: F401  (just probe; real import is in _render_log)
    except ImportError:
        clean.write_text(
            "agent-run: --echo requested but `pyte` is not installed.\n"
            "Install with: pipx inject mmartins-toolbox pyte\n"
        )
        return
    while True:
        try:
            mtime = log.stat().st_mtime
        except FileNotFoundError:
            time.sleep(interval)
            continue
        if mtime != last_mtime:
            last_mtime = mtime
            try:
                raw = log.read_bytes()
                rendered = _render_log(raw)
                tmp = clean.with_suffix(".clean.tmp")
                tmp.write_text(rendered, encoding="utf-8")
                tmp.replace(clean)
            except Exception:
                # Don't crash the helper on transient render errors;
                # next tick may succeed.
                pass
        time.sleep(interval)


def cmd_clean(args: argparse.Namespace) -> int:
    d = _require_run(args.name)
    log = d / "log"
    if not log.is_file():
        sys.exit(f"agent-run: no log file for '{args.name}' at {log}")
    raw = log.read_bytes()
    rendered = _render_log(
        raw,
        width=args.width,
        height=args.height,
        history=args.history,
    )
    out_path = getattr(args, "out", None)
    if out_path:
        Path(out_path).write_text(rendered, encoding="utf-8")
        size = len(rendered.encode("utf-8"))
        sys.stderr.write(f"agent-run: wrote {size} bytes of cleaned transcript to {out_path}\n")
        return 0
    sys.stdout.write(rendered)
    return 0


# ---------------------------------------------------------------------------
# steer / kill
# ---------------------------------------------------------------------------

def cmd_steer(args: argparse.Namespace) -> int:
    d = _require_run(args.name)
    if _read(d / "interactive") != "1":
        sys.exit(
            f"agent-run: '{args.name}' is not interactive. "
            f"Relaunch with: agent-run -i {args.name} <command...>"
        )
    fifo = d / "stdin"
    if not fifo.is_fifo():
        sys.exit(f"agent-run: no stdin FIFO at {fifo}")
    pid_raw = _read(d / "pid")
    if not pid_raw or not _pid_alive(int(pid_raw)):
        sys.exit(f"agent-run: '{args.name}' is not running")
    msg = " ".join(args.message)
    if args.raw:
        # Caller knows what they want — send bytes verbatim.
        payload = msg
        esc_payload: Optional[bytes] = None
        send_separate_cr = False
    else:
        # PTY + raw-mode TUIs (Claude Code, Codex REPL) treat \r as Enter,
        # not \n. Send CR so the line is actually submitted.
        payload = msg + "\r"
        # --esc: send ESC first as its own write so the TUI has time to
        # exit generation mode before the new prompt+CR arrive. Sending ESC
        # + text in one chunk races the TUI's mode switch and the CR can
        # end up dropped while the input buffer is still being reset.
        esc_payload = b"\x1b" if args.esc else None
        send_separate_cr = args.esc
    data = payload.encode()
    # Write with a timeout guard: a healthy run has the keeper holding the
    # FIFO open for reading, so this returns immediately.
    def _alarm(_sig, _frame):
        raise TimeoutError("write timed out")
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(10)
    try:
        with fifo.open("wb") as f:
            if esc_payload is not None:
                f.write(esc_payload)
                f.flush()
                # Give the TUI ~600ms to register the interrupt, reset the
                # input buffer, and switch back to input mode before the new
                # prompt arrives.
                time.sleep(0.6)
            f.write(data)
            f.flush()
            if send_separate_cr:
                # Belt-and-braces: send a final CR as its own write after a
                # brief settle so the TUI is guaranteed to see Enter even if
                # it briefly flushed input while exiting generation mode.
                time.sleep(0.2)
                f.write(b"\r")
                f.flush()
    except TimeoutError:
        sys.exit("agent-run: steer timed out writing to FIFO — is the agent alive?")
    finally:
        signal.alarm(0)
    sent = len(data) + (len(esc_payload) if esc_payload else 0) + (1 if send_separate_cr else 0)
    print(f"agent-run: steered '{args.name}' ({sent} bytes)")
    return 0


def _signal_by_name(name: str) -> int:
    name = name.upper()
    if name.isdigit():
        return int(name)
    if not name.startswith("SIG"):
        name = "SIG" + name
    return getattr(signal, name)


def cmd_kill(args: argparse.Namespace) -> int:
    d = _require_run(args.name)
    try:
        sig = _signal_by_name(args.signal)
    except AttributeError:
        sys.exit(f"agent-run: unknown signal '{args.signal}'")
    pid_raw = _read(d / "pid")
    if not pid_raw:
        sys.exit(f"agent-run: no pid recorded for {args.name}")
    pid = int(pid_raw)
    if not _pid_alive(pid):
        print(f"agent-run: {args.name} is not running (pid {pid})")
        return 0
    pgid_raw = _read(d / "pgid")
    try:
        pgid = int(pgid_raw) if pgid_raw else None
    except ValueError:
        pgid = None
    # Prefer process-group kill (reaches agent + PTY wrapper + keeper).
    sent = False
    if pgid:
        try:
            os.killpg(pgid, sig)
            sent = True
        except (ProcessLookupError, PermissionError):
            pass
    if not sent:
        try:
            os.kill(pid, sig)
            sent = True
        except ProcessLookupError:
            pass
    # Belt-and-braces: also hit tracked aux pids.
    for aux in ("pty_pid", "keeper_pid"):
        raw = _read(d / aux)
        if raw:
            try:
                aux_pid = int(raw)
                if _pid_alive(aux_pid):
                    try:
                        os.kill(aux_pid, sig)
                    except ProcessLookupError:
                        pass
            except ValueError:
                pass
    sig_name = signal.Signals(sig).name
    print(f"agent-run: sent {sig_name} to {args.name} (pid={pid} pgid={pgid or '?'})")
    return 0


# ---------------------------------------------------------------------------
# launch + runner
# ---------------------------------------------------------------------------

def cmd_launch(args: argparse.Namespace) -> int:
    name: str = args.name
    argv: List[str] = list(args.command)
    if not argv:
        sys.exit("agent-run: missing command")
    prompt_file: Optional[str] = getattr(args, "prompt_file", None)
    if prompt_file and not Path(prompt_file).is_file():
        sys.exit(f"agent-run: prompt file not found: {prompt_file}")
    echo: bool = bool(getattr(args, "echo", False))
    echo_interval: float = float(getattr(args, "echo_interval", 2.0))
    d = _run_dir(name)
    # Reject if a previous run with the same name is still active.
    if d.is_dir():
        old_status = _read(d / "status")
        old_pid_raw = _read(d / "pid")
        if old_status == "running" and old_pid_raw:
            try:
                old_pid = int(old_pid_raw)
                if _pid_alive(old_pid):
                    sys.exit(
                        f"Error: Run '{name}' is still active (pid {old_pid}). "
                        f"Kill it first or use a different name."
                    )
            except ValueError:
                pass
        import shutil
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)

    _write(d / "command", _pretty_command(argv) + "\n")
    _write(d / "argv", json.dumps(argv))
    _write(d / "started_at", _now_iso() + "\n")
    _write(d / "status", "running\n")
    _write(d / "interactive", "1\n" if args.interactive else "0\n")
    (d / "log").touch()
    if prompt_file:
        # Snapshot the prompt-file path so introspection shows what was fed in.
        _write(d / "prompt_file", prompt_file + "\n")
    if echo:
        _write(d / "echo", f"{echo_interval}\n")

    if args.interactive:
        fifo = d / "stdin"
        if fifo.exists():
            fifo.unlink()
        os.mkfifo(str(fifo))

    # Double-fork to detach from the terminal and become our own session
    # leader. The grandchild runs the actual agent.
    parent_pid = os.getpid()
    r_ack, w_ack = os.pipe()
    child_pid = os.fork()
    if child_pid != 0:
        # Parent: wait for the grandchild to publish its pid, then return.
        os.close(w_ack)
        os.waitpid(child_pid, 0)  # reap the intermediate forker
        # Read a single byte as the "child is ready" signal.
        try:
            os.read(r_ack, 1)
        except OSError:
            pass
        os.close(r_ack)
        bg_pid_raw = _read(d / "pid")
        if not bg_pid_raw:
            sys.exit("agent-run: failed to start agent (no pid recorded)")
        bg_pid = int(bg_pid_raw)
        print(f"agent-run: started '{name}' (pid {bg_pid})")
        if args.interactive:
            print(f"agent-run: interactive — steer with: agent-run steer {name} '<message>'")
        print(f"agent-run: run_dir={d}")
        print(f"agent-run: poll:   agent-run status {name}")
        print(f"agent-run: logs:   agent-run tail {name}")
        return 0

    # Intermediate child: become session leader and fork once more.
    os.close(r_ack)
    os.setsid()
    grand = os.fork()
    if grand != 0:
        # Intermediate exits; parent's waitpid reaps it.
        os._exit(0)

    # Grandchild: actually run the agent.
    _runner(d, argv, args.interactive, w_ack, prompt_file, echo, echo_interval)
    return 0  # never reached


def _runner(
    run_dir: Path,
    argv: Sequence[str],
    interactive: bool,
    ready_fd: int,
    prompt_file: Optional[str] = None,
    echo: bool = False,
    echo_interval: float = 2.0,
) -> None:
    """Execute in the detached session-leader process.

    Writes pid/pgid then either execs the agent directly (non-interactive)
    or forks a PTY child and shuttles FIFO <-> PTY master <-> log (interactive).
    """
    my_pid = os.getpid()
    _write(run_dir / "pid", f"{my_pid}\n")
    # After setsid(), pid == pgid (we're the session & group leader).
    _write(run_dir / "pgid", f"{os.getpgid(my_pid)}\n")

    # Signal parent we're ready so it can print the launch banner.
    try:
        os.write(ready_fd, b".")
        os.close(ready_fd)
    except OSError:
        pass

    # Redirect stdio to /dev/null to fully detach (we write the log ourselves).
    devnull = os.open(os.devnull, os.O_RDWR)
    os.dup2(devnull, 0)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    if devnull > 2:
        os.close(devnull)

    log_fd = os.open(str(run_dir / "log"), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)

    def _finalize(code: int) -> None:
        if not (run_dir / "exit_code").exists():
            _write(run_dir / "exit_code", f"{code}\n")
            _write(run_dir / "ended_at", _now_iso() + "\n")
            _write(run_dir / "status", "done\n" if code == 0 else "failed\n")

    def _on_signal(signum: int, _frame) -> None:
        # Propagate to children, then finalize and exit.
        for aux in ("pty_pid", "keeper_pid", "echo_pid"):
            raw = _read(run_dir / aux)
            if raw:
                try:
                    os.kill(int(raw), signal.SIGTERM)
                except (ValueError, ProcessLookupError):
                    pass
        _finalize(128 + signum)
        os._exit(128 + signum)

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGHUP, _on_signal)

    # If --echo was requested, fork a background renderer that periodically
    # writes a cleaned transcript next to the raw log. Stays alive for the
    # whole run; the signal handler tears it down on shutdown.
    if echo:
        echo_pid = os.fork()
        if echo_pid == 0:
            try:
                _echo_loop(run_dir, echo_interval)
            finally:
                os._exit(0)
        _write(run_dir / "echo_pid", f"{echo_pid}\n")

    try:
        if interactive:
            exit_code = _run_interactive(run_dir, argv, log_fd, prompt_file)
        else:
            exit_code = _run_oneshot(run_dir, argv, log_fd, prompt_file)
    except Exception as exc:  # noqa: BLE001
        try:
            os.write(log_fd, f"\nagent-run: runner crashed: {exc!r}\n".encode())
        except OSError:
            pass
        exit_code = 1

    _finalize(exit_code)
    os._exit(exit_code)


def _run_oneshot(
    run_dir: Path,
    argv: Sequence[str],
    log_fd: int,
    prompt_file: Optional[str] = None,
) -> int:
    pid = os.fork()
    if pid == 0:
        # Child: stdin from prompt file (if provided) or /dev/null;
        # stdout/stderr to log.
        if prompt_file:
            try:
                stdin_fd = os.open(prompt_file, os.O_RDONLY)
            except OSError as exc:
                os.write(2, f"agent-run: cannot open prompt file: {exc}\n".encode())
                os._exit(127)
        else:
            stdin_fd = os.open(os.devnull, os.O_RDONLY)
        os.dup2(stdin_fd, 0)
        os.dup2(log_fd, 1)
        os.dup2(log_fd, 2)
        if stdin_fd > 2:
            os.close(stdin_fd)
        try:
            os.execvp(argv[0], list(argv))
        except OSError as exc:
            os.write(2, f"agent-run: exec failed: {exc}\n".encode())
            os._exit(127)
    _, status = os.waitpid(pid, 0)
    if os.WIFEXITED(status):
        return os.WEXITSTATUS(status)
    if os.WIFSIGNALED(status):
        return 128 + os.WTERMSIG(status)
    return 1


def _run_interactive(
    run_dir: Path,
    argv: Sequence[str],
    log_fd: int,
    prompt_file: Optional[str] = None,
) -> int:
    fifo_path = run_dir / "stdin"

    # Persistent keeper process: holds the FIFO open for writing so the reader
    # (the PTY runner) never sees EOF between steers. We fork a dedicated
    # child that blocks on a long sleep while holding the write end open.
    keeper_r, keeper_w = os.pipe()
    keeper_pid = os.fork()
    if keeper_pid == 0:
        os.close(keeper_r)
        # Open FIFO for writing (blocks until a reader appears, that's us below).
        # Use a background-safe open: O_RDWR avoids the reader-blocking behavior.
        fd = os.open(str(fifo_path), os.O_RDWR)
        # Ack and go to sleep.
        try:
            os.write(keeper_w, b".")
        finally:
            os.close(keeper_w)
        try:
            while True:
                time.sleep(3600)
        except (KeyboardInterrupt, SystemExit):
            pass
        finally:
            try:
                os.close(fd)
            except OSError:
                pass
        os._exit(0)

    os.close(keeper_w)
    # Wait for keeper to open the FIFO.
    try:
        os.read(keeper_r, 1)
    except OSError:
        pass
    os.close(keeper_r)
    _write(run_dir / "keeper_pid", f"{keeper_pid}\n")

    # Fork + PTY for the agent.
    pty_pid, master_fd = pty.fork()
    if pty_pid == 0:
        # Child: stdin/stdout/stderr are all the PTY slave. Exec.
        try:
            os.execvp(argv[0], list(argv))
        except OSError as exc:
            sys.stderr.write(f"agent-run: exec failed: {exc}\n")
            os._exit(127)
    _write(run_dir / "pty_pid", f"{pty_pid}\n")

    # If a prompt file was provided, fork a helper that waits for the TUI to
    # finish initializing (so the PTY is in raw mode and CR -> Enter), then
    # writes the prompt + CR to the FIFO so the agent receives it as if a
    # human had typed it. Same pattern as `agent-run steer`.
    if prompt_file:
        helper = os.fork()
        if helper == 0:
            # Detach from the parent's stdio. Errors are silent (no log to write to).
            try:
                # Wait a few seconds for the TUI to enable raw mode. Earlier
                # tests showed sub-3s delivery races ICRNL CR->LF translation.
                time.sleep(4)
                try:
                    data = Path(prompt_file).read_bytes()
                except OSError:
                    os._exit(0)
                # Submit with CR; trailing CR is unconditional so the agent
                # treats the file as a single Enter-terminated message. Send
                # a second separate CR after a brief settle so the TUI is
                # guaranteed to see Enter even if the first one races the
                # input-buffer being reset right after typing finishes.
                try:
                    fd = os.open(str(fifo_path), os.O_WRONLY)
                    try:
                        os.write(fd, data + b"\r")
                    finally:
                        os.close(fd)
                    time.sleep(0.5)
                    fd = os.open(str(fifo_path), os.O_WRONLY)
                    try:
                        os.write(fd, b"\r")
                    finally:
                        os.close(fd)
                except OSError:
                    pass
            finally:
                os._exit(0)
        # Parent (the runner) just continues; helper is detached.

    # Open FIFO read end (blocks until the keeper has opened for writing,
    # which it has by the time we got the ack).
    fifo_fd = os.open(str(fifo_path), os.O_RDONLY)
    # Make non-blocking for the select loop below? Keep blocking; we gate on select.

    # Make master non-blocking so reads don't stall when select lies briefly.
    flags = fcntl.fcntl(master_fd, fcntl.F_GETFL)
    fcntl.fcntl(master_fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
    flags = fcntl.fcntl(fifo_fd, fcntl.F_GETFL)
    fcntl.fcntl(fifo_fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

    exit_code: Optional[int] = None
    buf_in = b""
    while True:
        try:
            r, _, _ = select.select([master_fd, fifo_fd], [], [], 0.5)
        except (OSError, select.error) as exc:
            if isinstance(exc, OSError) and exc.errno == errno.EINTR:
                continue
            break

        if master_fd in r:
            try:
                data = os.read(master_fd, 4096)
            except BlockingIOError:
                data = b""
            except OSError:
                data = b""
            if data == b"":
                # Try to reap.
                try:
                    wpid, status = os.waitpid(pty_pid, os.WNOHANG)
                except ChildProcessError:
                    wpid = pty_pid
                    status = 0
                if wpid == pty_pid:
                    if os.WIFEXITED(status):
                        exit_code = os.WEXITSTATUS(status)
                    elif os.WIFSIGNALED(status):
                        exit_code = 128 + os.WTERMSIG(status)
                    else:
                        exit_code = 1
                    break
                # master closed but child still alive? unusual — loop again.
            else:
                try:
                    os.write(log_fd, data)
                except OSError:
                    pass

        if fifo_fd in r:
            try:
                chunk = os.read(fifo_fd, 4096)
            except BlockingIOError:
                chunk = b""
            except OSError:
                chunk = b""
            if chunk:
                buf_in += chunk
                try:
                    written = os.write(master_fd, buf_in)
                    buf_in = buf_in[written:]
                except OSError:
                    pass

        # Child may have exited without us seeing EOF (detached, etc.).
        try:
            wpid, status = os.waitpid(pty_pid, os.WNOHANG)
        except ChildProcessError:
            wpid = pty_pid
            status = 0
        if wpid == pty_pid:
            # Drain any remaining master output.
            try:
                while True:
                    data = os.read(master_fd, 4096)
                    if not data:
                        break
                    os.write(log_fd, data)
            except OSError:
                pass
            if os.WIFEXITED(status):
                exit_code = os.WEXITSTATUS(status)
            elif os.WIFSIGNALED(status):
                exit_code = 128 + os.WTERMSIG(status)
            else:
                exit_code = 1
            break

    # Clean up.
    try:
        os.kill(keeper_pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        os.close(fifo_fd)
    except OSError:
        pass
    try:
        os.close(master_fd)
    except OSError:
        pass
    try:
        os.waitpid(keeper_pid, 0)
    except ChildProcessError:
        pass
    return exit_code if exit_code is not None else 0


# ---------------------------------------------------------------------------
# CLI dispatch
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="agent-run",
        description="Coding agent wrapper with structured run tracking.",
    )
    sub = p.add_subparsers(dest="sub")

    sp_status = sub.add_parser("status", help="print one-line status")
    sp_status.add_argument("name")
    sp_status.set_defaults(func=cmd_status)

    sp_logs = sub.add_parser("logs", help="print last N lines of the log")
    sp_logs.add_argument("name")
    sp_logs.add_argument("n", nargs="?", type=int, default=50)
    sp_logs.set_defaults(func=cmd_logs)

    sp_tail = sub.add_parser("tail", help="follow log in real time (tail -f)")
    sp_tail.add_argument("name")
    sp_tail.set_defaults(func=cmd_tail)

    sp_clean = sub.add_parser(
        "clean",
        help="render PTY-captured TUI log into a readable transcript via pyte",
    )
    sp_clean.add_argument("name")
    sp_clean.add_argument(
        "-o",
        "--out",
        default=None,
        help="write the cleaned transcript to this file (default: stdout)",
    )
    sp_clean.add_argument(
        "--width",
        type=int,
        default=120,
        help="emulated terminal width in columns (default: 120)",
    )
    sp_clean.add_argument(
        "--height",
        type=int,
        default=60,
        help="emulated viewport height in rows (default: 60)",
    )
    sp_clean.add_argument(
        "--history",
        type=int,
        default=100000,
        help="scrollback line budget for the emulator (default: 100000)",
    )
    sp_clean.set_defaults(func=cmd_clean)

    sp_steer = sub.add_parser(
        "steer",
        help="send text to agent stdin (needs -i); auto-appends CR to submit",
    )
    sp_steer.add_argument("name")
    sp_steer.add_argument("message", nargs="+")
    sp_steer.add_argument(
        "--esc",
        action="store_true",
        help="prepend ESC to interrupt the running generation before sending",
    )
    sp_steer.add_argument(
        "--raw",
        action="store_true",
        help="send bytes verbatim — do not append CR or prepend ESC",
    )
    sp_steer.set_defaults(func=cmd_steer)

    sp_kill = sub.add_parser("kill", help="kill the agent (default SIGTERM)")
    sp_kill.add_argument("name")
    sp_kill.add_argument("signal", nargs="?", default="TERM")
    sp_kill.set_defaults(func=cmd_kill)

    sp_list = sub.add_parser("list", help="list all runs")
    sp_list.set_defaults(func=cmd_list)

    sp_help = sub.add_parser("help", help="show this help")
    sp_help.set_defaults(func=lambda _a: (p.print_help() or 0))

    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)

    # No args -> list runs.
    if not raw:
        return cmd_list(argparse.Namespace())

    # Top-level help.
    if raw[0] in ("-h", "--help"):
        _build_parser().print_help()
        return 0

    # Interactive flag consumed explicitly (may be before name).
    interactive = False
    prompt_file: Optional[str] = None
    echo: bool = False
    echo_interval: float = 2.0
    # Consume top-level flags (`-i`, `-f <path>`, `--echo[=interval]`) in
    # any order before the name.
    while raw:
        if raw[0] in ("-i", "--interactive"):
            interactive = True
            raw = raw[1:]
            continue
        if raw[0] in ("-f", "--prompt-file"):
            if len(raw) < 2:
                sys.exit("agent-run: -f/--prompt-file requires a path")
            prompt_file = raw[1]
            raw = raw[2:]
            continue
        if raw[0].startswith("--prompt-file="):
            prompt_file = raw[0].split("=", 1)[1]
            raw = raw[1:]
            continue
        if raw[0] == "--echo":
            echo = True
            raw = raw[1:]
            continue
        if raw[0].startswith("--echo="):
            echo = True
            try:
                echo_interval = float(raw[0].split("=", 1)[1])
            except ValueError:
                sys.exit("agent-run: --echo=<interval> needs a number (seconds)")
            raw = raw[1:]
            continue
        break

    # Try to dispatch a known subcommand; otherwise treat as launch.
    known_subcommands = {"status", "logs", "tail", "clean", "steer", "kill", "list", "help"}
    if raw and raw[0] in known_subcommands and not interactive and not prompt_file and not echo:
        # argparse handles these, including their own -h/--help.
        parser = _build_parser()
        args = parser.parse_args(raw)
        return int(args.func(args) or 0)

    if len(raw) < 2:
        _build_parser().print_help()
        return 2
    name, *command = raw
    # Basic validation of the name.
    if "/" in name or name.startswith("-"):
        sys.exit(f"agent-run: invalid name '{name}'")
    ns = argparse.Namespace(
        name=name,
        command=command,
        interactive=interactive,
        prompt_file=prompt_file,
        echo=echo,
        echo_interval=echo_interval,
    )
    return cmd_launch(ns)


if __name__ == "__main__":
    sys.exit(main())
