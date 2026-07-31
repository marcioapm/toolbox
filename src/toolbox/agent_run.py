#!/usr/bin/env python3
"""agent-run — Wrapper for coding agents (Claude Code, Codex, etc.).

Creates a run directory with structured state files so LLM orchestrators can
poll safely without brittle process-poll loops, and optionally allocates a
real PTY so TUI agents behave as if attached to a terminal (no more 0-CPU
hangs from isatty() checks).

Storage is split across two roots so a hard crash or reboot never loses a
log even though the ephemeral process state is gone:

    /tmp/agent-runs/<name>/       ephemeral process state (tmpfs on Linux —
                                   wiped on reboot, so "missing" unambiguously
                                   means "not running"). Override with
                                   AGENT_RUN_STATE_DIR.
    /var/tmp/agent-runs/<name>/   persistent log + prompt copy, survives
                                   reboot/crash. Override with
                                   AGENT_RUN_LOG_DIR. The log fd is opened
                                   here from the start — no copy-on-exit step
                                   that a crash could lose.

Usage::

    agent-run <name> <cmd...>            # non-interactive (one-shot)
    agent-run -i <name> <cmd...>         # interactive (PTY-wrapped, steerable)
    agent-run tail <name>                # follow log in real time
    agent-run logs <name> [N]            # last N lines (default 50)
    agent-run status <name>              # one-line status
    agent-run steer <name> <msg...>      # send text to agent stdin (needs -i)
    agent-run kill <name> [SIGNAL]       # TERM by default; 9/KILL if stuck
    agent-run list                       # list all runs

Ephemeral files under $AGENT_RUN_STATE_DIR/<name>/ (default /tmp/agent-runs)::

    status       running | done | failed
    exit_code    numeric exit code (after completion)
    pid          runner process id
    pgid         process group id (kill target)
    process_identity  platform-specific runner birth token (kill verification)
    agent_pid   one-shot command child pid (non-interactive only)
    pty_pid      PTY child pid (interactive only)
    keeper_pid   FIFO-keeper pid (interactive only)
    prompt_pid   initial-prompt helper pid (when -f and interactive)
    echo_pid     transcript renderer pid (when --echo)
    command      pretty-printed launch command
    argv         JSON-encoded argv (authoritative form for replay)
    submit_mode  cr | crlf (selected from argv for interactive submission)
    started_at   ISO-8601 UTC
    ended_at     ISO-8601 UTC (after completion)
    interactive  "1" if launched with -i, else "0"
    stdin        FIFO for steering (interactive only)

Persistent files under $AGENT_RUN_LOG_DIR/<name>/ (default /var/tmp/agent-runs)::

    log          captured stdout+stderr (PTY-captured when interactive)
    log.clean    rendered transcript (only when launched with --echo)
    prompt       copy of the -f/--prompt-file input, if one was given

`status` reports "not running (log preserved)" when the state dir is gone
but the log dir survived. `logs`/`tail`/`clean` always read from the log
dir, falling back to the old single-directory layout for runs started
before this split. Log dirs older than 21 days are pruned opportunistically
on `list`/launch.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import errno
import fcntl
import json
import math
import os
import pty
import re
import select
import shlex
import shutil
import signal
import stat as _stat_module
import subprocess
import sys
import threading
import time
import traceback
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence


STATE_ROOT = Path(os.environ.get("AGENT_RUN_STATE_DIR", "/tmp/agent-runs"))
LOG_ROOT = Path(os.environ.get("AGENT_RUN_LOG_DIR", "/var/tmp/agent-runs"))
PRUNE_AFTER_DAYS = 21
SUBMIT_MODE_CR = "cr"
SUBMIT_MODE_CRLF = "crlf"
MAX_PTY_INPUT_BUFFER = 1024 * 1024
MAX_FINAL_RENDER_BYTES = 16 * 1024 * 1024
FINAL_RENDER_TIMEOUT_SECONDS = 10.0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write(path: Path, text: str) -> None:
    """Atomically publish a state file so readers never see a partial value."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        tmp.write_text(text)
        tmp.replace(path)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def _read(path: Path, default: str = "") -> str:
    try:
        return path.read_text().strip()
    except FileNotFoundError:
        return default


_RUN_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _validate_run_name(name: str) -> str:
    """Validate a run name before it is ever used to build a filesystem path.

    Rejects empty, ``.``/``..``, path separators, NUL/control characters, and
    anything outside ``^[A-Za-z0-9][A-Za-z0-9._-]*$``. Returning the name
    unchanged lets callers write ``name = _validate_run_name(name)`` inline.

    This is the single choke point that keeps ``_state_dir``/``_log_dir`` (and
    the ``shutil.rmtree`` calls that consume them) from resolving to a root or
    parent directory and wiping unrelated state/logs.
    """
    if not name or name in (".", ".."):
        sys.exit(f"agent-run: invalid run name {name!r}")
    if "/" in name or "\\" in name:
        sys.exit(f"agent-run: invalid run name {name!r} (path separators not allowed)")
    if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in name):
        sys.exit(f"agent-run: invalid run name {name!r} (control characters not allowed)")
    if not _RUN_NAME_RE.match(name):
        sys.exit(
            f"agent-run: invalid run name {name!r} "
            f"(allowed: letters, digits, '.', '_', '-'; must start alnum)"
        )
    return name


def _state_dir(name: str) -> Path:
    return STATE_ROOT / name


def _log_dir(name: str) -> Path:
    return LOG_ROOT / name


def _safe_rmtree(candidate: Path, root: Path, expected: Optional[os.stat_result] = None) -> None:
    """Remove ``candidate`` only when it is a real, non-symlink direct child of ``root``.

    Uses ``lstat`` on the *original* pathname to refuse symlinks before any
    resolution occurs.  When ``expected`` is provided, its device/inode must
    still match immediately before deletion, preventing a prune inspection of
    one run directory from deleting a same-name replacement.
    """
    try:
        st = candidate.lstat()
    except OSError:
        return  # already gone — nothing to do
    if not _stat_module.S_ISDIR(st.st_mode):
        sys.exit(
            f"agent-run: refusing to delete {candidate} — not a real directory "
            f"(lstat mode {oct(st.st_mode)})"
        )
    if expected is not None and (st.st_dev, st.st_ino) != (expected.st_dev, expected.st_ino):
        return
    resolved_root = root.resolve()
    resolved = candidate.resolve()
    if resolved == resolved_root or resolved.parent != resolved_root:
        sys.exit(
            f"agent-run: refusing to delete {resolved} — not contained in {resolved_root}"
        )
    shutil.rmtree(resolved)


@contextmanager
def _launch_lock(name: str):
    """Serialize launch setup for one run name across processes.

    The lock lives under ``STATE_ROOT/.locks`` rather than inside the run
    directory, which launch may replace. It is held until runner readiness so
    a competing launch cannot pass the liveness check against partial state.
    """
    lock_dir = STATE_ROOT / ".locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"{name}.lock"
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield fd
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _log_file_for(name: str) -> Optional[Path]:
    """Resolve the persistent log for a run, preferring the new split
    layout ($AGENT_RUN_LOG_DIR/<name>/log) and falling back to the old
    single-directory layout ($AGENT_RUN_STATE_DIR/<name>/log) so in-flight
    runs started before this split remain readable."""
    new_log = _log_dir(name) / "log"
    if new_log.exists():
        return new_log
    old_log = _state_dir(name) / "log"
    if old_log.exists():
        return old_log
    return None


def _require_state(name: str) -> Path:
    d = _state_dir(name)
    if not d.is_dir():
        sys.exit(
            f"agent-run: no active run state for '{name}' in {STATE_ROOT} "
            f"(try 'agent-run status {name}' for a preserved log)"
        )
    return d


def _require_log(name: str) -> Path:
    log = _log_file_for(name)
    if log is not None:
        return log
    if _state_dir(name).is_dir() or _log_dir(name).is_dir():
        sys.exit(f"agent-run: no log file for '{name}' in {_log_dir(name)}")
    sys.exit(f"agent-run: no run named '{name}' in {STATE_ROOT} or {LOG_ROOT}")


def _known(name: str) -> bool:
    return _state_dir(name).is_dir() or _log_dir(name).is_dir()


def _prune_old_locks(max_age_days: int = PRUNE_AFTER_DAYS) -> None:  # noqa: ARG001
    """Per-name flock files are never removed.

    Each ``<name>.lock`` file under ``STATE_ROOT/.locks`` is the permanent,
    authoritative inode for its run name's serialization lock.  Unlinking it
    — even after acquiring the flock — creates an open-before-flock race: a
    launcher that already opened the file will hold a lock on the now-unlinked
    inode while a competing launcher creates and locks a *new* inode for the
    same pathname, breaking the serialization guarantee introduced by commit
    ``a18b720``.

    Zero-byte lock-file accumulation is intentional and safe: the directory
    grows by one tiny zero-byte file per distinct run name, and run names are
    validated to a safe character set by ``_validate_run_name``, so the attack
    surface is small.  If operator cleanup is ever needed it should be done
    out-of-band after confirming no launches are in flight.
    """
    # Intentionally a no-op: see docstring above.
    return


def _prune_old_logs(max_age_days: int = PRUNE_AFTER_DAYS) -> None:
    """Remove only stable, inactive, old real log directories.

    Pruning takes the permanent per-name launch lock before re-inspecting a
    candidate.  That serializes it with replacement launches and lets us reject
    live/unknown state before deleting.  An inode snapshot is verified again at
    deletion time so an inspected directory can never be swapped for a new run.
    """
    _prune_old_locks(max_age_days)
    if not LOG_ROOT.is_dir():
        return
    cutoff = time.time() - max_age_days * 86400
    try:
        candidates = list(LOG_ROOT.iterdir())
    except OSError:
        return
    for d in candidates:
        try:
            _validate_run_name(d.name)
            before = d.lstat()
        except (OSError, SystemExit):
            continue
        if not _stat_module.S_ISDIR(before.st_mode):
            continue
        try:
            initial_mtime = max(
                (f.stat().st_mtime for f in d.iterdir()), default=before.st_mtime
            )
        except OSError:
            continue
        if initial_mtime >= cutoff:
            continue
        with _launch_lock(d.name):
            try:
                current = d.lstat()
            except OSError:
                continue
            if not _stat_module.S_ISDIR(current.st_mode):
                continue
            if (current.st_dev, current.st_ino) != (before.st_dev, before.st_ino):
                continue
            state_dir = _state_dir(d.name)
            if state_dir.exists():
                # A state directory that is not conclusively terminal is live
                # or unverifiable; preserve its persistent log conservatively.
                status = _read(state_dir / "status")
                if status not in {"done", "failed"}:
                    continue
            try:
                mtime = max(
                    (f.stat().st_mtime for f in d.iterdir()), default=current.st_mtime
                )
            except OSError:
                continue
            if mtime >= cutoff:
                continue
            _safe_rmtree(d, LOG_ROOT, expected=current)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but we can't signal it — still "alive" for our purposes.
        return True


def _require_positive_state_int(state_dir: Path, field: str, name: str) -> int:
    """Read a positive integer control value or stop before process control."""
    raw = _read(state_dir / field)
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if not raw or value <= 0:
        sys.exit(
            f"agent-run: invalid {field} state for '{name}'; "
            "inspect status/logs and confirm the run is gone before removing stale state"
        )
    return value


def _process_identity(pid: int) -> Optional[str]:
    """Return a stable platform-specific birth token for ``pid``, if readable."""
    system = platform.system()
    if system == "Linux":
        try:
            stat_text = Path(f"/proc/{pid}/stat").read_text()
            _before, after = stat_text.rsplit(")", 1)
            fields = after.split()
            starttime = fields[19]
            if not starttime.isdigit():
                return None
        except (IndexError, OSError, ValueError):
            return None
        return f"linux:{starttime}"
    if system == "Darwin":
        try:
            result = subprocess.run(
                ["ps", "-p", str(pid), "-o", "lstart="],
                capture_output=True,
                check=True,
                text=True,
                timeout=2,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        start = result.stdout.strip()
        return f"darwin:{start}" if start else None
    return None


def _read_process_identity(state_dir: Path, name: str) -> str:
    token = _read(state_dir / "process_identity")
    if not token:
        sys.exit(
            f"agent-run: refusing to kill '{name}': no process identity recorded "
            "(legacy state); inspect status/logs and remove stale state only after confirming it is gone"
        )
    prefix, separator, value = token.partition(":")
    if prefix not in {"linux", "darwin"} or not separator or not value:
        sys.exit(f"agent-run: refusing to kill '{name}': corrupt process identity state")
    return token


def _pretty_command(argv: Sequence[str]) -> str:
    return " ".join(shlex.quote(a) for a in argv)


def _submit_mode_for_argv(argv: Sequence[str]) -> str:
    """Select the interactive Enter sequence from the launched argv.

    Only inspect the executable position, optionally beneath wrappers whose
    command structure is unambiguous. Never search arbitrary arguments: a
    prompt mentioning "opencode" must not change how another agent is driven.

    ``env -S``/``--split-string`` tokens are fed back through the same env
    option/assignment parser so that split strings containing ``env``
    assignments (``KEY=value``), ``-i``, ``-u TOKEN``, etc. before the
    executable are handled correctly.  A nesting counter (``_ENV_SPLIT_DEPTH``)
    caps recursion so a maliciously constructed or accidentally nested split
    string cannot loop indefinitely.
    """
    _ENV_SPLIT_MAX_DEPTH = 8

    command = list(argv)
    while command:
        executable = Path(command[0]).name
        if executable == "opencode":
            return SUBMIT_MODE_CRLF
        if executable in {"command", "exec"}:
            command = command[1:]
            continue
        if executable == "env":
            command = command[1:]
            split_depth = 0
            while command:
                argument = command[0]
                if argument == "--":
                    command = command[1:]
                    break
                if argument in {"-i", "--ignore-environment"} or (
                    not argument.startswith("-") and "=" in argument
                ):
                    command = command[1:]
                    continue
                # -S / --split-string / -S<arg>: split the value via shlex and
                # prepend the tokens back into the current command list so that
                # env options/assignments *within* the split string (e.g.
                # ``-S 'KEY=val opencode'``) are consumed by the same inner loop
                # rather than re-entered as the outer loop's executable.
                if argument in {"-S", "--split-string"}:
                    if len(command) < 2:
                        return SUBMIT_MODE_CR
                    try:
                        split_tokens = shlex.split(command[1])
                    except ValueError:
                        return SUBMIT_MODE_CR
                    split_depth += 1
                    if split_depth > _ENV_SPLIT_MAX_DEPTH:
                        return SUBMIT_MODE_CR
                    command = split_tokens + command[2:]
                    continue  # re-enter inner loop with expanded tokens
                if argument.startswith("--split-string="):
                    try:
                        split_tokens = shlex.split(argument.split("=", 1)[1])
                    except ValueError:
                        return SUBMIT_MODE_CR
                    split_depth += 1
                    if split_depth > _ENV_SPLIT_MAX_DEPTH:
                        return SUBMIT_MODE_CR
                    command = split_tokens + command[1:]
                    continue  # re-enter inner loop with expanded tokens
                if len(argument) > 2 and argument.startswith("-S"):
                    try:
                        split_tokens = shlex.split(argument[2:])
                    except ValueError:
                        return SUBMIT_MODE_CR
                    split_depth += 1
                    if split_depth > _ENV_SPLIT_MAX_DEPTH:
                        return SUBMIT_MODE_CR
                    command = split_tokens + command[1:]
                    continue  # re-enter inner loop with expanded tokens
                if argument in {"-u", "--unset", "-C", "--chdir", "-P", "--argv0"}:
                    # These options consume the following argument. If it is
                    # absent, env itself will fail; there is no executable to
                    # inspect, so leave the default as CR.
                    command = command[2:] if len(command) >= 2 else []
                    continue
                if argument.startswith(("--unset=", "--chdir=", "--argv0=")):
                    command = command[1:]
                    continue
                if len(argument) > 2 and argument[:2] in {"-u", "-C", "-P"}:
                    command = command[1:]
                    continue
                break
            continue
        break
    return SUBMIT_MODE_CR


def _submit_bytes(mode: str) -> bytes:
    """Return the selected Enter sequence; legacy/malformed state uses CR."""
    return b"\r\n" if mode == SUBMIT_MODE_CRLF else b"\r"


def _submit_mode_from_state(state_dir: Path) -> str:
    mode = _read(state_dir / "submit_mode")
    if mode in {SUBMIT_MODE_CR, SUBMIT_MODE_CRLF}:
        return mode
    # Upgrade path for runs created before submit_mode was persisted: recover
    # the authoritative argv JSON and apply current executable detection.
    try:
        argv = json.loads((state_dir / "argv").read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return SUBMIT_MODE_CR
    if isinstance(argv, list) and all(isinstance(arg, str) for arg in argv):
        return _submit_mode_for_argv(argv)
    return SUBMIT_MODE_CR


def _persist_submit_mode(
    state_dir: Path, argv: Sequence[str], override: Optional[str] = None
) -> str:
    """Persist a symbolic submission mode selected from override or argv."""
    mode = override or _submit_mode_for_argv(argv)
    _write(state_dir / "submit_mode", mode + "\n")
    return mode


def _prompt_submission_writes(data: bytes, mode: str) -> tuple[bytes, bytes]:
    """Return the two FIFO writes used to submit an initial prompt file."""
    submit = _submit_bytes(mode)
    return data + submit, submit


# ---------------------------------------------------------------------------
# list / status / logs / tail
# ---------------------------------------------------------------------------

def _log_line_count(log: Optional[Path]) -> int:
    if log is None:
        return 0
    try:
        with log.open("rb") as f:
            return sum(1 for _ in f)
    except FileNotFoundError:
        return 0


def cmd_list(_args: argparse.Namespace) -> int:
    _prune_old_logs()
    state_names = set()
    print(f"Live runs ({STATE_ROOT}):")
    if STATE_ROOT.is_dir():
        state_names = {
            p.name for p in STATE_ROOT.iterdir() if p.is_dir() and not p.name.startswith(".")
        }
    if not state_names:
        print("  (none)")
    else:
        for d in sorted(_state_dir(n) for n in state_names):
            status = _read(d / "status", "unknown")
            pid = _read(d / "pid", "?")
            started = _read(d / "started_at", "?")
            lines = _log_line_count(_log_file_for(d.name))
            interactive = _read(d / "interactive", "0")
            flag = " [interactive]" if interactive == "1" else ""
            print(f"  {d.name}: status={status} pid={pid} started={started} lines={lines}{flag}")

    log_only_names = set()
    if LOG_ROOT.is_dir():
        log_only_names = {p.name for p in LOG_ROOT.iterdir() if p.is_dir()} - state_names
    if log_only_names:
        print(f"Preserved logs, not running ({LOG_ROOT}):")
        for name in sorted(log_only_names):
            lines = _log_line_count(_log_file_for(name))
            print(f"  {name}: lines={lines}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    name = _validate_run_name(args.name)
    state_dir = _state_dir(name)
    log_dir = _log_dir(name)
    if not state_dir.is_dir() and not log_dir.is_dir():
        sys.exit(f"agent-run: no run named '{name}' in {STATE_ROOT} or {LOG_ROOT}")
    lines = _log_line_count(_log_file_for(name))
    if not state_dir.is_dir():
        print(f"name={name} status=not running (log preserved) lines={lines}")
        return 0
    d = state_dir
    status = _read(d / "status", "unknown")
    pid = _read(d / "pid", "?")
    started = _read(d / "started_at", "?")
    ended = _read(d / "ended_at", "-")
    exit_code = _read(d / "exit_code", "-")
    interactive = _read(d / "interactive", "0")
    print(
        f"name={name} status={status} pid={pid} exit={exit_code} "
        f"started={started} ended={ended} lines={lines} interactive={interactive}"
    )
    return 0


def cmd_logs(args: argparse.Namespace) -> int:
    log = _require_log(_validate_run_name(args.name))
    n = max(1, args.n)
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
    name = _validate_run_name(args.name)
    log = _require_log(name)
    pid_raw = _read(_state_dir(name) / "pid")
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
            # EOF. A preserved-log-only or otherwise non-live run has nothing
            # left to follow; after printing existing content, exit immediately.
            if pid is None:
                return 0
            if not _pid_alive(pid):
                # One more drain to catch final writes from a process that was
                # live when tail started.
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

# Fallback ANSI stripper used when pyte itself can't safely render a log
# (e.g. a RecursionError from a pathological escape-sequence pattern). Not
# as faithful as a real VT100 replay — no cursor-motion collapsing — but it
# never crashes, which is the point: the clean transcript is a convenience
# artifact and must never take the run down with it.
_OSC_RE = re.compile(rb"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")
_CSI_RE = re.compile(rb"\x1b\[[0-9;:?]*[ -/]*[@-~]")
_ESC_RE = re.compile(rb"\x1b.", re.DOTALL)
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Feeding pyte's coroutine-based FSM a pathological escape-sequence pattern
# can recurse past this many frames; give it plenty of headroom (restored
# immediately after) rather than tripping Python's conservative default.
_PYTE_RECURSION_LIMIT = 20000
# The C stack backing that higher Python recursion limit needs room too —
# run the feed in a worker thread with a generous stack so the process
# hits Python's RecursionError before the OS ever SIGSEGVs the thread.
_PYTE_THREAD_STACK_SIZE = 64 * 1024 * 1024


def _strip_ansi_fallback(raw: bytes) -> str:
    """Best-effort plain-text rendering: strip ANSI/OSC escape sequences
    with regexes instead of replaying them through a terminal emulator.
    Used when the pyte-based renderer fails outright."""
    text = _OSC_RE.sub(b"", raw)
    text = _CSI_RE.sub(b"", text)
    text = _ESC_RE.sub(b"", text)
    decoded = text.decode("utf-8", errors="replace")
    decoded = decoded.replace("\r\n", "\n").replace("\r", "\n")
    decoded = _CTRL_RE.sub("", decoded)

    deduped: List[str] = []
    for line in (ln.rstrip() for ln in decoded.split("\n")):
        if not deduped or deduped[-1] != line:
            deduped.append(line)
    while deduped and not deduped[-1]:
        deduped.pop()
    return "\n".join(deduped) + "\n"


def _feed_pyte(stream, raw: bytes) -> None:
    """Feed `raw` into a pyte ByteStream with a raised recursion limit and
    a worker thread sized for it, so pathological escape sequences hit a
    clean RecursionError instead of corrupting/crashing the interpreter."""
    error: List[BaseException] = []

    def _worker() -> None:
        try:
            stream.feed(raw)
        except Exception as exc:
            error.append(exc)
        except (KeyboardInterrupt, SystemExit) as exc:
            # Cross the worker-thread boundary so process-control exceptions
            # propagate on the caller's thread rather than being swallowed or
            # merely reported as an unhandled thread exception.
            error.append(exc)

    old_limit = sys.getrecursionlimit()
    old_stack_size = threading.stack_size()
    sys.setrecursionlimit(max(old_limit, _PYTE_RECURSION_LIMIT))
    try:
        threading.stack_size(_PYTE_THREAD_STACK_SIZE)
    except (ValueError, RuntimeError):
        pass  # platform doesn't support a custom stack size — best effort
    try:
        t = threading.Thread(target=_worker)
        t.start()
        t.join()
    finally:
        sys.setrecursionlimit(old_limit)
        try:
            threading.stack_size(old_stack_size)
        except (ValueError, RuntimeError):
            pass

    if error:
        raise error[0]


class RenderDependencyError(RuntimeError):
    """The optional terminal renderer is unavailable."""


_RENDER_DEPENDENCY_MESSAGE = (
    "agent-run: `pyte` is required for `clean` / --echo. "
    "Install with: pipx inject mmartins-toolbox pyte  (or uv tool install --with pyte ...)"
)


def _render_log(raw: bytes, width: int = 120, height: int = 60, history: int = 100000) -> str:
    """Render a raw PTY-captured log (with ANSI/Ink redraw artifacts) into a
    plain-text transcript by replaying the byte stream through a VT100
    emulator (pyte). Returns the deduplicated screen history + final visible
    viewport, joined with newlines.

    Pyte is loaded lazily so the rest of agent-run keeps working even if the
    extra is not installed (and so we get a clear error message when it is
    really needed). If pyte itself fails to render (e.g. a RecursionError
    triggered by a pathological escape-sequence pattern), this degrades to
    a best-effort ANSI-stripped plain-text render rather than crashing —
    the clean transcript is a convenience artifact and must never take the
    run down with it.
    """
    try:
        import pyte  # type: ignore
    except ImportError as exc:
        raise RenderDependencyError(_RENDER_DEPENDENCY_MESSAGE) from exc

    screen = pyte.HistoryScreen(width, height, history=history, ratio=0.5)
    stream = pyte.ByteStream(screen)
    try:
        _feed_pyte(stream, raw)
    except Exception:
        return _strip_ansi_fallback(raw)

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


def _render_log_to_clean(log_dir: Path) -> None:
    """Atomically render ``log`` to ``log.clean`` in ``log_dir``."""
    log = log_dir / "log"
    clean = log_dir / "log.clean"
    raw = log.read_bytes()
    rendered = _render_log(raw)
    tmp = clean.with_suffix(".clean.tmp")
    tmp.write_text(rendered, encoding="utf-8")
    tmp.replace(clean)


def _bounded_final_render(log_dir: Path) -> Optional[str]:
    """Best-effort final render with strict byte and wall-clock limits.

    Rendering occurs in a dedicated child so an uninterruptible or pathological
    renderer cannot hold the runner in a nonterminal state.  The child writes
    ``log.clean`` atomically through ``_render_log_to_clean``; the parent waits
    only ``FINAL_RENDER_TIMEOUT_SECONDS`` before killing it.  Returns an error
    description suitable for the raw log, or ``None`` on success.
    """
    try:
        size = (log_dir / "log").stat().st_size
    except OSError as exc:
        return f"cannot stat log: {exc}"
    if size > MAX_FINAL_RENDER_BYTES:
        return f"log is {size} bytes; final render limit is {MAX_FINAL_RENDER_BYTES} bytes"

    pid = os.fork()
    if pid == 0:
        try:
            _render_log_to_clean(log_dir)
        except BaseException:
            os._exit(1)
        os._exit(0)

    deadline = time.monotonic() + FINAL_RENDER_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        try:
            waited, status = os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            return "renderer child could not be reaped"
        if waited == pid:
            return None if os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0 else "renderer failed"
        time.sleep(0.05)

    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        os.waitpid(pid, 0)
    except ChildProcessError:
        pass
    return f"renderer exceeded {FINAL_RENDER_TIMEOUT_SECONDS:g}-second deadline"


def _echo_loop(log_dir: "Path", interval: float) -> None:
    """Periodically render log_dir/log into log_dir/log.clean.

    Runs in a detached child for the lifetime of the agent. The parent's
    signal handler kills us on shutdown so we don't outlive the run. We
    only re-render when the raw log's mtime has changed, so a quiet run
    doesn't burn CPU.
    """
    log = log_dir / "log"
    clean = log_dir / "log.clean"
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
            try:
                _render_log_to_clean(log_dir)
            except Exception:
                # Don't crash the helper on transient render errors;
                # next tick may succeed.
                pass
            else:
                # Failed renders must be retried even when the raw log's mtime
                # remains unchanged.
                last_mtime = mtime
        time.sleep(interval)


def cmd_clean(args: argparse.Namespace) -> int:
    log = _require_log(_validate_run_name(args.name))
    raw = log.read_bytes()
    try:
        rendered = _render_log(
            raw,
            width=args.width,
            height=args.height,
            history=args.history,
        )
    except RenderDependencyError as exc:
        sys.exit(str(exc))
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
    name = _validate_run_name(args.name)
    d = _require_state(name)
    if _read(d / "interactive") != "1":
        sys.exit(
            f"agent-run: '{name}' is not interactive. "
            f"Relaunch with: agent-run -i {name} <command...>"
        )
    pid = _require_positive_state_int(d, "pid", name)
    if not _pid_alive(pid):
        sys.exit(f"agent-run: '{args.name}' is not running")
    fifo = d / "stdin"
    if not fifo.is_fifo():
        sys.exit(f"agent-run: no stdin FIFO at {fifo}")
    msg = " ".join(args.message)
    if args.raw:
        # Caller knows what they want — send bytes verbatim.
        data = msg.encode()
        esc_payload: Optional[bytes] = None
        submit = b""
    else:
        # Most PTY + raw-mode TUIs use CR for Enter. OpenCode's Bubble Tea
        # TUI needs CRLF; the mode was selected from launch argv and persisted.
        submit = _submit_bytes(_submit_mode_from_state(d))
        data = msg.encode() + submit
        # --esc: send ESC first as its own write so the TUI has time to
        # exit generation mode before the new prompt + Enter arrive. Sending
        # ESC + text in one chunk races the TUI's mode switch and Enter can
        # end up dropped while the input buffer is still being reset.
        esc_payload = b"\x1b" if args.esc else None
    send_separate_submit = not args.raw and args.esc
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
            if send_separate_submit:
                # Belt-and-braces: send a final Enter as its own write after a
                # brief settle so the TUI is guaranteed to see it even if it
                # briefly flushed input while exiting generation mode.
                time.sleep(0.2)
                f.write(submit)
                f.flush()
    except TimeoutError:
        sys.exit("agent-run: steer timed out writing to FIFO — is the agent alive?")
    finally:
        signal.alarm(0)
    sent = len(data) + (len(esc_payload) if esc_payload else 0) + (len(submit) if send_separate_submit else 0)
    print(f"agent-run: steered '{args.name}' ({sent} bytes)")
    return 0


def _signal_by_name(name: str) -> int:
    name = name.upper()
    if name.isdigit():
        sig = int(name)
        # Reject signal 0 (process-existence probe, not a real signal) and
        # any value outside the platform's valid signal set.
        valid = signal.valid_signals() if hasattr(signal, "valid_signals") else range(1, 65)
        if sig not in valid:
            sys.exit(
                f"agent-run: invalid signal number {sig!r} — "
                "use a symbolic name (TERM, KILL, INT, …) or a valid signal number"
            )
        return sig
    if not name.startswith("SIG"):
        name = "SIG" + name
    try:
        return getattr(signal, name)
    except AttributeError:
        raise AttributeError(name)


def _send_signal_to_verified_pid(pid: int, sig: int, expected_identity: str) -> None:
    """Send ``sig`` to ``pid`` using the safest available mechanism.

    On Linux (kernel ≥ 5.1) we open a pidfd *before* re-verifying identity.
    Because a pidfd is bound to a specific process instance rather than a
    numeric PID, the signal is guaranteed to reach exactly that instance: if
    the process exits between identity verification and ``pidfd_send_signal``
    the kernel returns ``ESRCH`` instead of hitting a recycled PID.

    On Darwin (and Linux without pidfd support) we re-read the birth token
    immediately before ``os.kill`` to minimise — but not eliminate — the
    TOCTOU window.  This residual race is unavoidable on platforms without
    pidfd; it is documented here so callers understand the limitation.

    Raises ``ProcessLookupError`` if the process is gone, ``PermissionError``
    on permission failure, or ``RuntimeError`` if identity cannot be confirmed.
    """
    system = platform.system()

    if system == "Linux":
        # Try pidfd path first.
        pidfd_open = getattr(os, "pidfd_open", None)
        pidfd_send_signal = getattr(signal, "pidfd_send_signal", None)
        if pidfd_open is not None and pidfd_send_signal is not None:
            try:
                pfd = pidfd_open(pid, 0)
            except (ProcessLookupError, OSError):
                raise ProcessLookupError(pid)
            try:
                # Re-verify identity *with the pidfd open*: the pidfd keeps the
                # process table slot alive (kernel ref-counts it), so the PID
                # cannot be recycled underneath us.  Identity mismatch after a
                # successful pidfd_open is only possible if we opened the wrong
                # PID in a race — fail closed.
                current = _process_identity(pid)
                if current is None or current != expected_identity:
                    raise RuntimeError(
                        "process identity changed between verification and signal; refusing"
                    )
                pidfd_send_signal(pfd, sig, None, 0)
            finally:
                os.close(pfd)
            return
        # Fall through to the re-verified numeric path on old kernels.

    # Darwin and Linux without pidfd: re-verify identity as close as possible
    # to the actual signal call.  A process exit between re-verification and
    # os.kill produces a benign ESRCH; a PID recycle in that tiny window is
    # the residual TOCTOU race documented in the module docstring for Darwin.
    current = _process_identity(pid)
    if current is None or current != expected_identity:
        raise RuntimeError(
            "process identity changed between verification and signal; refusing"
        )
    os.kill(pid, sig)


def cmd_kill(args: argparse.Namespace) -> int:
    name = _validate_run_name(args.name)
    d = _require_state(name)
    try:
        sig = _signal_by_name(args.signal)
    except AttributeError:
        sys.exit(f"agent-run: unknown signal '{args.signal}'")
    if _read(d / "status") not in {"running", "starting"}:
        sys.exit(f"agent-run: refusing to kill '{name}': run is not marked running")
    pid = _require_positive_state_int(d, "pid", name)
    expected_identity = _read_process_identity(d, name)
    current_identity = _process_identity(pid)
    if current_identity is None:
        sys.exit(
            f"agent-run: refusing to kill '{name}': cannot verify current process identity"
        )
    if current_identity != expected_identity:
        sys.exit(
            f"agent-run: refusing to kill '{name}': process identity does not match recorded runner"
        )
    if not _pid_alive(pid):
        print(f"agent-run: {args.name} is not running (pid {pid})")
        return 0
    # Signal the verified runner PID directly.  The runner's own _on_signal
    # handler tears down its direct children (agent_pid, pty_pid, keeper_pid,
    # echo_pid) using waitpid-ownership checks, so we do not need to signal
    # aux PIDs externally.  killpg is not used: it is numeric and cannot be
    # made atomic with the group identity check.
    try:
        _send_signal_to_verified_pid(pid, sig, expected_identity)
    except ProcessLookupError:
        print(f"agent-run: {args.name} is not running (pid {pid})")
        return 0
    except PermissionError:
        sys.exit(f"agent-run: permission denied signaling '{name}' (pid {pid})")
    except RuntimeError as exc:
        sys.exit(f"agent-run: refusing to kill '{name}': {exc}")
    try:
        sig_name = signal.Signals(sig).name
    except ValueError:
        sig_name = str(sig)
    print(f"agent-run: sent {sig_name} to {args.name} (pid={pid})")
    return 0


# ---------------------------------------------------------------------------
# launch + runner
# ---------------------------------------------------------------------------

def cmd_launch(args: argparse.Namespace) -> int:
    name = _validate_run_name(args.name)
    # Pruning takes per-name locks itself.  Do it before acquiring this name's
    # lock so a stale log for this same name cannot self-deadlock on flock.
    _prune_old_logs()
    with _launch_lock(name) as lock_fd:
        return _cmd_launch_locked(args, name, lock_fd)


def _cmd_launch_locked(args: argparse.Namespace, name: str, lock_fd: int) -> int:
    """Perform launch setup while ``lock_fd`` serializes this run name."""
    argv: List[str] = list(args.command)
    if not argv:
        sys.exit("agent-run: missing command")
    prompt_file: Optional[str] = getattr(args, "prompt_file", None)
    if prompt_file and not Path(prompt_file).is_file():
        sys.exit(f"agent-run: prompt file not found: {prompt_file}")
    echo: bool = bool(getattr(args, "echo", False))
    echo_interval: float = float(getattr(args, "echo_interval", 2.0))
    d = _state_dir(name)
    log_d = _log_dir(name)
    # Reject if a previous run with the same name is still active.
    if d.is_dir():
        old_status = _read(d / "status")
        old_pid_raw = _read(d / "pid")
        if old_status in {"running", "starting"} and old_pid_raw:
            try:
                old_pid = int(old_pid_raw)
                if _pid_alive(old_pid):
                    sys.exit(
                        f"Error: Run '{name}' is still active (pid {old_pid}). "
                        f"Kill it first or use a different name."
                    )
            except ValueError:
                pass
        _safe_rmtree(d, STATE_ROOT)
    if log_d.is_dir():
        _safe_rmtree(log_d, LOG_ROOT)
    d.mkdir(parents=True, exist_ok=True)
    log_d.mkdir(parents=True, exist_ok=True)

    _write(d / "command", _pretty_command(argv) + "\n")
    _write(d / "argv", json.dumps(argv))
    submit_mode = _persist_submit_mode(
        d, argv, getattr(args, "submit_mode", None)
    )
    _write(d / "started_at", _now_iso() + "\n")
    _write(d / "status", "starting\n")
    _write(d / "interactive", "1\n" if args.interactive else "0\n")
    (log_d / "log").touch()
    if prompt_file:
        # Snapshot the prompt-file path so introspection shows what was fed in,
        # and copy the content into the persistent log dir for post-mortem
        # context (done synchronously here, before the fork, so a crash can't
        # lose it).
        _write(d / "prompt_file", prompt_file + "\n")
        try:
            shutil.copyfile(prompt_file, log_d / "prompt")
        except OSError:
            pass
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
        # Read the structured readiness result. EOF, malformed data, and an
        # explicit setup error all mean launch failed.
        try:
            ack_raw = os.read(r_ack, 65536)
        except OSError:
            ack_raw = b""
        os.close(r_ack)
        try:
            ack = json.loads(ack_raw.decode()) if ack_raw else {}
        except (UnicodeDecodeError, json.JSONDecodeError):
            ack = {}
        if ack.get("status") != "ok":
            error = str(ack.get("error") or "runner exited before readiness")
            if not (d / "exit_code").exists():
                _write(d / "exit_code", "1\n")
                _write(d / "ended_at", _now_iso() + "\n")
                _write(d / "status", "failed\n")
            sys.exit(f"agent-run: failed to start agent: {error}")
        bg_pid_raw = _read(d / "pid")
        if not bg_pid_raw:
            sys.exit("agent-run: failed to start agent (no pid recorded)")
        bg_pid = int(bg_pid_raw)
        print(f"agent-run: started '{name}' (pid {bg_pid})")
        if args.interactive:
            print(f"agent-run: interactive — steer with: agent-run steer {name} '<message>'")
        print(f"agent-run: state_dir={d}")
        print(f"agent-run: log_dir={log_d}")
        print(f"agent-run: poll:   agent-run status {name}")
        print(f"agent-run: logs:   agent-run tail {name}")
        return 0

    # Intermediate child: become session leader and fork once more. The launch
    # parent keeps the lock through readiness; detached descendants must not
    # inherit a reference that would hold it for the full run lifetime.
    os.close(lock_fd)
    os.close(r_ack)
    os.setsid()
    grand = os.fork()
    if grand != 0:
        # Intermediate exits; parent's waitpid reaps it.
        os._exit(0)

    # Grandchild: actually run the agent.
    _runner(
        d,
        log_d,
        argv,
        args.interactive,
        w_ack,
        prompt_file,
        submit_mode,
        echo,
        echo_interval,
    )
    return 0  # never reached


def _reset_runner_signal_handlers() -> None:
    """Remove signal handlers inherited from the runner after ``fork()``.

    Helper children must never execute the runner's teardown handler: it reads
    runner-owned pid files and may otherwise signal the helper itself. SIGTERM,
    SIGINT, and SIGHUP use their normal process defaults in every child; the
    exec'd agent can install its own handlers afterward.
    """
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, signal.SIG_DFL)


_HANDLED_RUNNER_SIGNALS = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)


@contextmanager
def _block_handled_runner_signals():
    """Atomically cover a helper fork and its PID-file publication.

    Blocking the runner's handled termination signals closes the interval in
    which a child exists but has not yet been recorded for
    ``_teardown_children``.  ``pthread_sigmask`` is inherited over fork; the
    context restores the exact prior mask in both parent and child before
    either resumes normal work.
    """
    pthread_sigmask = getattr(signal, "pthread_sigmask", None)
    if pthread_sigmask is None:
        yield
        return
    previous = pthread_sigmask(signal.SIG_BLOCK, _HANDLED_RUNNER_SIGNALS)
    try:
        yield
    finally:
        pthread_sigmask(signal.SIG_SETMASK, previous)


def _teardown_children(state_dir: Path, grace: float = 2.0) -> None:
    """SIGTERM (then SIGKILL after `grace` seconds) every child pid this
    runner recorded (agent_pid / pty_pid / keeper_pid / prompt_pid / echo_pid), reaping
    each one.

    Called from both the runner's signal handler and its crash `except` so
    a hard crash can never leave the launched agent or its helpers orphaned.
    Recorded pids are direct children of this process (forked or pty.fork()'d
    in `_runner`/`_run_interactive`), so waitpid is valid here.
    Stale or corrupt state is filtered so teardown never signals itself.
    """
    own_pid = os.getpid()
    pids = []
    seen = set()
    for aux in ("agent_pid", "pty_pid", "keeper_pid", "prompt_pid", "echo_pid"):
        raw = _read(state_dir / aux)
        if not raw:
            continue
        try:
            pid = int(raw)
        except ValueError:
            continue
        if pid <= 0 or pid == own_pid or pid in seen:
            continue
        seen.add(pid)
        try:
            waited_pid, _status = os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            continue
        if waited_pid == pid:
            continue
        pids.append(pid)
    if not pids:
        return

    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass

    remaining = set(pids)
    deadline = time.time() + grace
    while remaining and time.time() < deadline:
        for pid in list(remaining):
            try:
                wpid, _status = os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                remaining.discard(pid)
                continue
            if wpid == pid:
                remaining.discard(pid)
        if remaining:
            time.sleep(0.1)

    for pid in remaining:
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            os.waitpid(pid, 0)
        except ChildProcessError:
            pass


def _runner(
    state_dir: Path,
    log_dir: Path,
    argv: Sequence[str],
    interactive: bool,
    ready_fd: int,
    prompt_file: Optional[str] = None,
    submit_mode: str = SUBMIT_MODE_CR,
    echo: bool = False,
    echo_interval: float = 2.0,
) -> None:
    """Execute in the detached session-leader process.

    Writes pid/pgid then either execs the agent directly (non-interactive)
    or forks a PTY child and shuttles FIFO <-> PTY master <-> log (interactive).
    """
    my_pid = os.getpid()
    log_fd = -1

    def _finalize(code: int) -> None:
        if not (state_dir / "exit_code").exists():
            _write(state_dir / "exit_code", f"{code}\n")
            _write(state_dir / "ended_at", _now_iso() + "\n")
            _write(state_dir / "status", "done\n" if code == 0 else "failed\n")

    handling_signal = False

    def _on_signal(signum: int, _frame) -> None:
        nonlocal handling_signal
        # Block recursive delivery before touching children. A second signal
        # during teardown exits immediately instead of re-entering cleanup.
        if handling_signal:
            os._exit(128 + signum)
        handling_signal = True
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(sig, signal.SIG_IGN)
        _teardown_children(state_dir)
        _finalize(128 + signum)
        os._exit(128 + signum)

    try:
        runner_pgid = os.getpgid(my_pid)
        identity = _process_identity(my_pid)
        if identity is None:
            raise RuntimeError("cannot record runner process identity")
        _write(state_dir / "pid", f"{my_pid}\n")
        # After setsid(), pid == pgid (we're the session & group leader).
        _write(state_dir / "pgid", f"{runner_pgid}\n")
        _write(state_dir / "process_identity", identity + "\n")

        # Redirect stdio to /dev/null to fully detach (we write the log ourselves).
        devnull = os.open(os.devnull, os.O_RDWR)
        try:
            os.dup2(devnull, 0)
            os.dup2(devnull, 1)
            os.dup2(devnull, 2)
        finally:
            if devnull > 2:
                os.close(devnull)

        log_fd = os.open(
            str(log_dir / "log"), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644
        )

        signal.signal(signal.SIGTERM, _on_signal)
        signal.signal(signal.SIGINT, _on_signal)
        signal.signal(signal.SIGHUP, _on_signal)

        # If --echo was requested, fork a background renderer that periodically
        # writes a cleaned transcript next to the raw log. Stays alive for the
        # whole run; the signal handler tears it down on shutdown.
        if echo:
            with _block_handled_runner_signals():
                echo_pid = os.fork()
                if echo_pid != 0:
                    _write(state_dir / "echo_pid", f"{echo_pid}\n")
            if echo_pid == 0:
                _reset_runner_signal_handlers()
                try:
                    os.close(ready_fd)
                except OSError:
                    pass
                try:
                    _echo_loop(log_dir, echo_interval)
                finally:
                    os._exit(0)
    except Exception as exc:  # setup failed before readiness
        try:
            payload = json.dumps({"status": "error", "error": str(exc)}).encode()
            os.write(ready_fd, payload)
        except OSError:
            pass
        finally:
            try:
                os.close(ready_fd)
            except OSError:
                pass
        _teardown_children(state_dir)
        try:
            _finalize(1)
        except OSError:
            pass
        os._exit(1)

    # Acknowledge only after all required resources, handlers, and helpers are
    # ready. The parent treats EOF or an error payload as launch failure.
    try:
        os.write(ready_fd, b'{"status":"ok"}')
    except OSError:
        pass
    finally:
        try:
            os.close(ready_fd)
        except OSError:
            pass

    try:
        if interactive:
            exit_code = _run_interactive(
                state_dir, argv, log_fd, prompt_file, submit_mode
            )
        else:
            exit_code = _run_oneshot(state_dir, argv, log_fd, prompt_file)
    except Exception as exc:  # noqa: BLE001
        try:
            detail = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            os.write(
                log_fd,
                f"\nagent-run: runner crashed:\n{detail}".encode(errors="replace"),
            )
        except OSError:
            pass
        # A crash mid-run may have skipped the normal cleanup path (e.g. an
        # exception inside _run_interactive's select loop, before it reaches
        # its own "kill keeper_pid" tail), leaving the launched agent (and
        # keeper/echo helpers) orphaned. Never exit without reaping them.
        _teardown_children(state_dir)
        exit_code = 1

    # Persistent helpers (notably --echo) outlive the agent by design, so reap
    # them on successful completion as well as on crashes and external signals.
    _teardown_children(state_dir)
    # Publish the agent's terminal state *before* producing a convenience
    # transcript artifact.  A pathological renderer can never leave a dead
    # agent reported as starting/running.
    _finalize(exit_code)
    if echo:
        # The periodic renderer may never tick on a short run, or may have run
        # just before the agent's final output.  Render once in a bounded child
        # after stopping it.  The status is already terminal if this times out.
        render_error = _bounded_final_render(log_dir)
        if render_error:
            try:
                os.write(
                    log_fd,
                    f"\nagent-run: final echo render failed: {render_error}\n".encode(
                        errors="replace"
                    ),
                )
            except OSError:
                pass
    os._exit(exit_code)


def _run_oneshot(
    state_dir: Path,
    argv: Sequence[str],
    log_fd: int,
    prompt_file: Optional[str] = None,
) -> int:
    with _block_handled_runner_signals():
        pid = os.fork()
        if pid != 0:
            _write(state_dir / "agent_pid", f"{pid}\n")
            # Agent child exists: publish running now that it is controllable.
            _write(state_dir / "status", "running\n")
    if pid == 0:
        _reset_runner_signal_handlers()
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


def _drain_pty_input(master_fd: int, buffered: bytes) -> bytes:
    """Write buffered FIFO input to the PTY until empty or backpressured."""
    while buffered:
        try:
            written = os.write(master_fd, buffered)
        except BlockingIOError:
            break
        except OSError as exc:
            if exc.errno in {errno.EIO, errno.EBADF, errno.EINVAL}:
                return b""
            raise
        if written <= 0:
            break
        buffered = buffered[written:]
    return buffered


def _run_interactive(
    state_dir: Path,
    argv: Sequence[str],
    log_fd: int,
    prompt_file: Optional[str] = None,
    submit_mode: str = SUBMIT_MODE_CR,
) -> int:
    fifo_path = state_dir / "stdin"

    # Persistent keeper process: holds the FIFO open for writing so the reader
    # (the PTY runner) never sees EOF between steers. We fork a dedicated
    # child that blocks on a long sleep while holding the write end open.
    keeper_r, keeper_w = os.pipe()
    with _block_handled_runner_signals():
        keeper_pid = os.fork()
        if keeper_pid != 0:
            _write(state_dir / "keeper_pid", f"{keeper_pid}\n")
    if keeper_pid == 0:
        _reset_runner_signal_handlers()
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

    # Fork + PTY for the agent.
    with _block_handled_runner_signals():
        pty_pid, master_fd = pty.fork()
        if pty_pid != 0:
            _write(state_dir / "pty_pid", f"{pty_pid}\n")
    if pty_pid == 0:
        _reset_runner_signal_handlers()
        # Child: stdin/stdout/stderr are all the PTY slave. Exec.
        try:
            os.execvp(argv[0], list(argv))
        except OSError as exc:
            sys.stderr.write(f"agent-run: exec failed: {exc}\n")
            os._exit(127)
    # Full interactive relay is now established: PTY forked, keeper holds FIFO
    # open, FIFO ready.  Publish running only now that the run is controllable.
    _write(state_dir / "status", "running\n")

    # If a prompt file was provided, fork a helper that waits for the TUI to
    # finish initializing (so the PTY is in raw mode and Enter is recognized),
    # then writes the prompt + selected Enter sequence to the FIFO so the agent
    # human had typed it. Same pattern as `agent-run steer`.
    if prompt_file:
        with _block_handled_runner_signals():
            helper = os.fork()
            if helper != 0:
                _write(state_dir / "prompt_pid", f"{helper}\n")
        if helper == 0:
            _reset_runner_signal_handlers()
            # Detach from the parent's stdio. Errors are silent (no log to write to).
            try:
                # Wait a few seconds for the TUI to enable raw mode. Earlier
                # tests showed sub-3s delivery races ICRNL CR->LF translation.
                time.sleep(4)
                try:
                    data = Path(prompt_file).read_bytes()
                except OSError:
                    os._exit(0)
                # Submit with the launch-selected Enter sequence. Trailing
                # Enter is unconditional so the agent treats the file as a
                # single submitted message. Send a second separate Enter
                # after a brief settle so the TUI is guaranteed to see it
                # even if the first one races input-buffer reset.
                submit_writes = _prompt_submission_writes(data, submit_mode)
                try:
                    fd = os.open(str(fifo_path), os.O_WRONLY)
                    try:
                        os.write(fd, submit_writes[0])
                    finally:
                        os.close(fd)
                    time.sleep(0.5)
                    fd = os.open(str(fifo_path), os.O_WRONLY)
                    try:
                        os.write(fd, submit_writes[1])
                    finally:
                        os.close(fd)
                except OSError:
                    pass
            finally:
                os._exit(0)

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
            writable = [master_fd] if buf_in else []
            r, w, _ = select.select([master_fd, fifo_fd], writable, [], 0.5)
        except (OSError, select.error) as exc:
            if isinstance(exc, OSError) and exc.errno == errno.EINTR:
                continue
            raise

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
                if len(buf_in) + len(chunk) > MAX_PTY_INPUT_BUFFER:
                    raise BufferError(
                        f"PTY input buffer exceeded {MAX_PTY_INPUT_BUFFER} bytes"
                    )
                buf_in += chunk

        if master_fd in w and buf_in:
            buf_in = _drain_pty_input(master_fd, buf_in)

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

def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be at least 0")
    return parsed


def _positive_finite_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be finite and greater than 0")
    return parsed


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
    sp_logs.add_argument("n", nargs="?", type=_positive_int, default=50)
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
        type=_positive_int,
        default=120,
        help="emulated terminal width in columns (default: 120)",
    )
    sp_clean.add_argument(
        "--height",
        type=_positive_int,
        default=60,
        help="emulated viewport height in rows (default: 60)",
    )
    sp_clean.add_argument(
        "--history",
        type=_nonnegative_int,
        default=100000,
        help="scrollback line budget for the emulator (default: 100000)",
    )
    sp_clean.set_defaults(func=cmd_clean)

    sp_steer = sub.add_parser(
        "steer",
        help="send text to agent stdin (needs -i); auto-appends the run's submission sequence",
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
    submit_mode: Optional[str] = None
    # Consume top-level flags (`-i`, `-f <path>`, `--echo[=interval]`,
    # `--submit-mode=cr|crlf`) in any order before the name.
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
            value = raw[0].split("=", 1)[1]
            try:
                echo_interval = _positive_finite_float(value)
            except argparse.ArgumentTypeError as exc:
                sys.exit(f"agent-run: --echo interval {exc}")
            raw = raw[1:]
            continue
        if raw[0].startswith("--submit-mode="):
            submit_mode = raw[0].split("=", 1)[1]
            if submit_mode not in {SUBMIT_MODE_CR, SUBMIT_MODE_CRLF}:
                sys.exit("agent-run: --submit-mode must be cr or crlf")
            raw = raw[1:]
            continue
        break

    # Try to dispatch a known subcommand; otherwise treat as launch.
    known_subcommands = {"status", "logs", "tail", "clean", "steer", "kill", "list", "help"}
    if (
        raw
        and raw[0] in known_subcommands
        and not interactive
        and not prompt_file
        and not echo
        and submit_mode is None
    ):
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
        submit_mode=submit_mode,
    )
    return cmd_launch(ns)


if __name__ == "__main__":
    sys.exit(main())
