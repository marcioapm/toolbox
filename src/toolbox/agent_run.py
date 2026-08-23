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
    /var/tmp/agent-runs/<name>/tmp/  per-run scratch dir (mode 0700), on the
                                   same persistent disk as the log. Exported
                                   as both TMPDIR and BUN_TMPDIR into the
                                   launched command's environment (and
                                   therefore all of its descendants) —
                                   BUN_TMPDIR because Bun (which OpenCode
                                   runs on) does not consult TMPDIR for its
                                   own scratch space — so agents/tools that
                                   dump large scratch data into $TMPDIR — e.g.
                                   OpenCode's bundled JDTLS, which leaks
                                   multi-hundred-MB Eclipse workspaces under
                                   mkdtemp() and never cleans them up — land
                                   on disk under this run's own directory
                                   instead of silently filling up a shared,
                                   possibly RAM-backed /tmp. The scratch dir
                                    is NOT deleted when the run ends —
                                    postmortem artifacts matter — `agent-run
                                    reap` removes it alongside a terminal
                                    state dir, and also removes an aged
                                    orphaned scratch after a reboot wiped
                                    its ephemeral state. Relaunching the
                                    same name replaces its whole log dir
                                    (including the prior scratch).

Usage::

    agent-run [flags] <name> -- <cmd...>  # recommended: explicit separator
    agent-run <name> <cmd...>            # non-interactive (one-shot)
    agent-run -i <name> -- <cmd...>      # interactive (PTY-wrapped, steerable)
    agent-run --idle-timeout N <name> -- <cmd...>  # self-terminate after N idle seconds
    agent-run --cwd <dir> <name> -- <cmd...>  # run the command in <dir> (managed mode too)
    agent-run --worktree DIR --worktree-base REF <name> -- <cmd...>
                                          # create a linked worktree at DIR, branched from
                                          # REF, and run the command there (managed mode too)
    agent-run --harness claude|opencode|codex   # managed mode (no trailing --)
              [--prompt <text> | --prompt-file <path>]
              [-i] [--model <model>] [--agent-mode <name>]
              [--harness-arg <flag>]...
              <name>
    agent-run attach <name>              # live keyboard + resize passthrough (Ctrl-C detaches)
    agent-run tail <name>                # follow log in real time
    agent-run logs <name> [--tail N | --head N]  # last/first N lines (default --tail 50)
    agent-run logs <name> --plain        # ANSI-stripped (default is raw bytes)
    agent-run logs <name> --clean        # pyte-rendered transcript, slow on large logs
    agent-run transcript <name> [--tail N | --head N] [--json]  # harness's own conversation record
                                          # (managed-mode runs only; see "Transcripts" below)
    agent-run status <name>              # one-line status
    agent-run watch <name> [--json] [--repo PATH]  # stateless fact snapshot for pollers
    agent-run steer <name> <msg...>      # send text to agent stdin (needs -i)
    agent-run kill <name> [SIGNAL]       # TERM by default; see "kill" below
    agent-run list [--all] [--status S] [--include-logs]  # list runs; defaults to non-terminal only
    agent-run reap [--dry-run] [--idle-hours N] [--min-age-hours N] [--name NAME]
                    [--include-logs] [--log-min-age-hours N]
                    [--orphan-processes] [--orphan-min-age-hours N]
    agent-run du [--by-run] [--top N] [--bytes|--json]  # disk usage per status or per run

Managed mode (--harness) builds the harness command itself and records the
session id deterministically: claude via --session-id (push), opencode via
POST /session mint-then-attach, codex via app-server thread/start. In managed
mode there is no trailing command — --harness and a trailing -- are mutually
exclusive. Raw passthrough mode is unchanged and unaffected.

Everything before "--" is an agent-run flag or the run name; everything
after "--" is the launch command verbatim — no subcommand dispatch,
leading-dash tokens accepted. Omitting "--" works for plain commands;
a flag typed after the name is rejected with an error naming "--".

`kill` sends TERM, INT, or HUP straight to the identity-verified runner
process, which catches it and runs its own teardown (kill/reap the
workload, publish terminal state). Signals the runner cannot catch —
notably KILL — are never sent to it directly: an uncatchable signal would
skip that teardown entirely and orphan the running agent while state kept
reporting "running". Instead `agent-run kill <name> KILL` first sends the
runner a regular TERM and waits a bounded window for its normal teardown;
only if the runner is still alive after that does it re-verify the
runner's identity and its recorded children's parentage, KILL each
directly, and publish terminal state itself. Signals outside
{TERM, INT, HUP, KILL} are rejected rather than forwarded, since an
unhandled one could bypass cleanup the same way a raw KILL would.

A run's `status` moves from `starting` (published before the detached
runner exists) to `running` (published only once the runner has actually
become controllable — child spawned/exec'd for one-shot, or PTY, FIFO, and
keeper ready for interactive) to a terminal `done`/`failed`. Any failure
before a status file exists, or before the detached runner takes over,
still resolves synchronously to `failed` rather than leaving `starting`
stranded with no process behind it. `agent-run reap` can additionally move
a stale `running` run to `died` (pid gone) or `killed` (idle-killed); both
are conclusively terminal, exactly like `done`/`failed`, and become
eligible for garbage collection once old enough (see `reap` below).

Ephemeral files under $AGENT_RUN_STATE_DIR/<name>/ (default /tmp/agent-runs)::

    status       starting | running | done | failed | launch_failed | died | killed
                 (`watch` additionally reports two values computed rather
                 than persisted here: `stalled` — pid alive, log idle past
                 the threshold; `unverified` — pid alive but its identity
                 could not be confirmed against a recorded launch token)
    exit_code    numeric exit code (after completion)
    pid          runner process id
    pgid         process group id (also the `kill --force` fallback target)
    launch_error first output line of a run that failed at launch
    prompt_submitted  written once an interactive prompt file was delivered
    idle_timeout      idle-timeout seconds this run was launched with
    idle_timeout_fired  idle seconds measured when the watchdog fired
    process_identity  platform-specific runner birth token (kill verification)
    agent_pid   one-shot command child pid (non-interactive only)
    pty_pid      PTY child pid (interactive only)
    keeper_pid   FIFO-keeper pid (interactive only)
    prompt_pid   initial-prompt helper pid (when -f and interactive)
    command      pretty-printed launch command
    argv         JSON-encoded argv (authoritative form for replay)
    submit_mode  cr | crlf (selected from argv for interactive submission)
    started_at   ISO-8601 UTC
    ended_at     ISO-8601 UTC (after completion)
    interactive  "1" if launched with -i, else "0"
    cwd          absolute working directory the command runs in: --cwd when
                 given, otherwise the launch-time working directory (used by
                 `watch` to locate the repo for git facts; may be absent on
                 legacy runs)
    launch_head  full git commit hash HEAD pointed to at launch, if `cwd` was
                 a git repo (used by `watch` to count commits made during the
                 run without trusting commit timestamps; absent when `cwd`
                 wasn't a repo, and on legacy runs)
    stdin        FIFO for steering / attach keyboard input (interactive only)
    resize       FIFO for attach terminal-resize records (interactive only)
    reap_reason  set by `agent-run reap` when it changes status (died/killed)
    tmp_dir      absolute path to this run's scratch dir (see below)
    worktree_created  absolute path of a linked worktree this launch created via
                 `--worktree`; absent for a run launched without `--worktree`
                 or attached to a pre-existing one via `--worktree-reuse`.
                 Observability only -- `reap`'s removal policy does not
                 consult this file

Persistent files under $AGENT_RUN_LOG_DIR/<name>/ (default /var/tmp/agent-runs)::

    log          captured stdout+stderr (PTY-captured when interactive)
    prompt       copy of the -f/--prompt-file input, if one was given
    session.json session attribution (managed mode only): session_id, harness,
                 acquisition, confidence, observed_at; absent for raw runs
    run.json     immutable launch facts + exit facts (all modes): name, argv,
                 command, cwd, started_at, harness, interactive, model,
                 agent_mode; augmented with ended_at, exit_code, status on exit
    hooks.jsonl  append-only log written by `agent-run hook`; one JSON object
                 per line (event, kind, timestamp, harness, message excerpt,
                 payload excerpt). kind normalises event/payload into
                 turn_complete/permission_required/session_start/other across
                 harnesses; message is the last assistant turn, when the
                 payload carries one. Absent until a hook fires; agent-run
                 never writes it itself. See "Harness hook integration" below.
    session-acquire.log  diagnostic log for session acquisition (managed mode)
    tmp/         per-run scratch dir exported as TMPDIR and BUN_TMPDIR (see
                 above); removed only by `agent-run reap`, never on normal
                 run exit

Transcripts. `agent-run transcript <name>` reads the harness's own
conversation record instead of reconstructing one from the PTY log: it looks
up session_id + harness in session.json and reads the matching store
directly (opencode: SQLite at ~/.local/share/opencode/opencode.db; claude:
~/.claude/projects/<mangled-cwd>/<session_id>.jsonl, including its
subagents/ subdirectory; codex: ~/.codex/sessions/**/rollout-*-<session_id>
.jsonl). Managed-mode only -- a raw run (`agent-run NAME -- <cmd>`) has no
session.json and therefore no transcript; use `logs --clean` for those.
Every store is opened read-only. Unparseable individual records are skipped
and their count is reported on stderr; a missing or unusable store is an
error. See toolbox/agent_run_transcript.py for the reader implementations.

Harness hook integration. Configure these yourself; agent-run never edits a
harness config file:

    claude (~/.claude/settings.json):
        {"hooks": {
          "Stop": [{"matcher": "", "hooks": [{"type": "command",
            "command": "agent-run hook stop"}]}],
          "PermissionRequest": [{"matcher": "", "hooks": [{"type": "command",
            "command": "agent-run hook permission-request"}]}]}}

    codex (~/.codex/config.toml):
        notify = ["agent-run", "hook", "turn-complete"]

        codex reports only turn completion this way: notify emits a single
        agent-turn-complete event. Its richer hook engine, which does have a
        PermissionRequest event, requires --dangerously-bypass-hook-trust,
        which `codex app-server` -- the binary managed mode drives -- does not
        accept, so permission_required is unavailable for managed codex runs.

    opencode (~/.config/opencode/plugin/agent-run-hook.js; opencode plugins
    are JavaScript, so the plugin shells out. The export must be a named
    factory returning an object of handlers -- `export default {onEvent}` is
    silently never invoked):
        import { execFileSync } from "child_process"

        export const AgentRunHook = async () => ({
          event: async ({ event }) => {
            const kind = event.type === "session.idle" ? "session-idle"
              : event.type.startsWith("permission.") ? "permission-request"
              : null
            if (!kind) return
            // Array argv: no shell, so event contents cannot inject.
            execFileSync("agent-run", ["hook", kind, "--json",
              JSON.stringify(event)])
          },
        })

The runner exports AGENT_RUN_NAME, AGENT_RUN_STATE_DIR, and AGENT_RUN_LOG_DIR
into the agent's environment, so a hook need not know how the run was
launched. `agent-run hook` resolves the run from --name, else AGENT_RUN_NAME,
else by matching its own process ancestry against a recorded pid/pgid, and
exits 0 silently when none of those resolve.

`status` reports "not running (log preserved)" when the state dir is gone
but the log dir survived. `logs` and `tail` always read from the log dir,
falling back to the old single-directory layout for runs started
before this split. Log dirs older than 21 days are pruned opportunistically
on `list`/launch; that whole-log-dir prune can remove any surviving `tmp/`
scratch directory too (for example when reap's min-age threshold is configured
longer than the prune interval, or a prior reap never saw the state dir).

Each run name is serialized by a permanent per-name lock file under
$AGENT_RUN_STATE_DIR/.locks/<name>.lock; these lock files are never pruned
(only the run's own state/log directories are), so the lock inode a launch
acquires is always the same one a concurrent launch or prune of that name
contends on. A launcher may die before its detached runner finishes
publishing identity and readiness, but the runner holds its own inherited
copy of the same lock fd until then, so the name stays serialized either
way.

Kill-time process identity verification (a platform-specific "birth
token": /proc/<pid>/stat start time on Linux, `ps -o lstart=` on Darwin)
closes the ordinary PID-recycling window, and Linux additionally binds the
signal to a pidfd opened before the final check. Darwin has no pidfd
equivalent: the gap between the last identity re-check and the actual
`os.kill()` call is a real, if extremely narrow, TOCTOU window where a
recycled PID could theoretically receive a signal meant for its prior
occupant. This residual race is accepted, not fully closed, on Darwin.

`agent-run reap` does three things, all reported (and, without --dry-run,
performed) in one invocation:

   1. Stale-"running" reconciliation: a "running" run whose pid is missing,
      malformed, or gone is marked `died`; a "running" run whose pid is alive
      but whose log has been idle longer than the idle threshold
      (--idle-hours, or AGENT_RUN_IDLE_KILL_HOURS, default 24h) is idle-killed
      and marked `killed`, via the same identity-verified `_force_kill`
      escalation `agent-run kill <name> KILL` uses.
   2. Terminal-state garbage collection: runs whose status is conclusively
      terminal — `done`, `failed`, `launch_failed`, `died`, or `killed` — and whose state dir
      has not been modified in longer than a separate, more conservative age
      threshold (--min-age-hours, or AGENT_RUN_MIN_AGE_HOURS (preferred) /
      AGENT_RUN_REAP_MIN_AGE_HOURS (compatible alias), default 168h/7 days)
      have their ephemeral state dir *and* their persistent scratch dir
      ($AGENT_RUN_LOG_DIR/<name>/tmp) removed. State-less orphaned scratch
      dirs left after a reboot are independently collected once their own
      contents are old enough. The persistent `log` and `prompt` files
      are never touched by this step — with `--include-logs`
      (see below), preserved logs get their own separate, independently
      thresholded pass instead. A `stalled` run (alive pid, idle
      log — see `_effective_status`) is not terminal and is never
      garbage-collected by this pass; only reconciliation in step 1 can
      eventually turn it into `killed`, which becomes GC-eligible only on a
      later reap. Unknown/legacy/corrupt statuses are shown separately by
      `list`, left untouched by default, and can only be collected with the
      explicit `reap --force-unknown` opt-in.
   3. Every destructive state-backed action re-verifies identity/inode state
      immediately before mutation (per-name `_launch_lock`, `_safe_rmtree`'s
      root-contained, inode-reverified deletion, `_pid_alive`/
      `_process_identity`), so a run that raced into life or was replaced
      between listing and action is never touched. State deletion first
      atomically renames its dir to a dot-prefixed sentinel, so an interrupted
      deletion is self-identifying and resumed by the next reap rather than
      stranding a partially-deleted run. --dry-run performs the same read-only
      eligibility checks as a real reap and prints only actions it would take,
      without writing or deleting anything.
   4. Orphan-process termination (--orphan-processes, off by default):
      terminate live runners with no state dir — invisible to passes 1-3.
      Candidates come from argv parsing (strict basename match) filtered by
      _find_orphan_runners' safety rules; identity captured at discovery is
      re-verified immediately before every signal, since with no state dir
      PID reuse is the only hazard left to defend against. See README.

`agent-run reap --include-logs` additionally garbage-collects whole
preserved-log-only run directories under $AGENT_RUN_LOG_DIR (state dir
already gone) once their newest recursive mtime is older than
--log-min-age-hours (default 21 days). Deliberately independent of
--min-age-hours; see ``--log-min-age-hours`` help. Off by default; a run
with a live state dir is never touched regardless of age.

`agent-run reap --include-worktrees` additionally removes a terminal run's
recorded launch `cwd` when — and only when — that directory is a *linked*
git worktree older than --worktree-min-age-hours (default 7 days, its own
independent threshold). A non-git directory is never removed: a run's cwd
is frequently a real project checkout or $HOME. Removal goes through `git
worktree remove` against the owning repository so the parent's
.git/worktrees/<name> admin entry is unregistered too. Uncommitted changes,
untracked files, commits absent from every remote, a live run sharing the
directory, and any symlinked path component are all refusals; only
--force-dirty overrides the unsaved-work refusals. A worktree shared by
several terminal runs is removed once. Off by default.

`agent-run list` defaults to showing only runs whose effective status is
*not* conclusively terminal (`starting`, `running`, `stalled`; a `died` or
`killed` run is terminal and hidden by default, matching reap's own
definition of terminal). Pass `--status done,failed,died,killed,...` (or
any comma-separated subset of statuses) or `--all` to see terminal runs
too. Scripts that previously scraped every line under "Live runs" should
add `--all` if they relied on terminal runs being listed there; the
heading text also now reflects the actual filter in effect rather than
unconditionally claiming everything shown is live. Preserved-log-only runs
(state dir gone) are hidden by default; pass `--include-logs` (or set
AGENT_RUN_LIST_INCLUDE_LOGS=1) to show them — orthogonal to `--all`/
`--status`, which govern only the state-backed sections. When hidden logs
exist and are not shown, a one-line hint is printed to stderr, never
stdout, so `agent-run list | grep ...` stays honest.

`agent-run du` reports disk usage per effective status (or per run with
`--by-run`), including preserved logs, and never mutates anything — no
locks, no heal, no prune. With `--worktrees` it additionally sizes each
run's recorded launch `cwd` when that directory is a *linked* git worktree,
in its own WORKTREE column; a worktree shared by several runs is
deduplicated by realpath and charged once, so TOTAL counts every byte
exactly once. Off by default because those trees are typically much larger
than STATE_ROOT and LOG_ROOT combined. See its own `--help` for `--top`,
`--bytes`, and `--json`.
"""

from __future__ import annotations

import argparse
import codecs
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
import socket
import stat as _stat_module
import struct
import subprocess
import sys
import termios
import threading
import time
import traceback
import tty
import unicodedata
import urllib.error
import urllib.request
import uuid
import platform
from wcwidth import wcwidth
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Iterable, List, NamedTuple, NoReturn, Optional, Sequence, Tuple
from toolbox import __version__ as TOOLBOX_VERSION
from toolbox import agent_run_transcript as _agent_run_transcript


# Absolutised against the invocation directory at import: --cwd chdirs the
# launcher, and a relative root would otherwise re-base into the target
# directory, creating run state (and running _prune_old_logs deletions) inside
# it rather than under the configured root.
STATE_ROOT = Path(os.environ.get("AGENT_RUN_STATE_DIR", "/tmp/agent-runs")).absolute()
LOG_ROOT = Path(os.environ.get("AGENT_RUN_LOG_DIR", "/var/tmp/agent-runs")).absolute()
PRUNE_AFTER_DAYS = 21
SUBMIT_MODE_CR = "cr"
SUBMIT_MODE_CRLF = "crlf"
MAX_PTY_INPUT_BUFFER = 1024 * 1024
# Each resize record is framed by a magic first byte, tagged with a format
# version, and closed by a checksum over the payload. The resize FIFO has no
# per-writer arbitration, so a torn or replaced record can otherwise desync
# the stream permanently -- and the magic byte alone is not enough, because
# 0xA7 is also an ordinary column count (167) that reappears inside the
# payload at that width. The checksum makes a mis-anchored boundary
# detectable rather than silently decoding as garbage dimensions.
#
# The version byte keeps a mixed-version pairing (a session launched by one
# release, attached by another) from decoding the other side's layout as
# dimensions. Bump it whenever the record layout or the checksum changes.
RESIZE_RECORD_MAGIC = 0xA7
RESIZE_RECORD_VERSION = 1
RESIZE_RECORD_FORMAT = ">BBHHB"
RESIZE_RECORD_SIZE = struct.calcsize(RESIZE_RECORD_FORMAT)
# Name of the state-dir marker a runner writes to advertise which resize
# record version it can read, so attach can detect the mismatch before
# writing records the runner would silently discard.
RESIZE_PROTOCOL_MARKER = "resize_protocol"
# Upper bound on a dimension carried in a resize record. Terminals do not
# get near this, so anything above it is corrupt framing that happened to
# satisfy the checksum rather than a real size worth applying.
MAX_TERMINAL_DIMENSION = 2000
# Per-line and total byte caps for `logs` output. A single TUI line
# can be megabytes long (one \n per session, many \r redraw frames), so a line
# count alone does not bound output. The primary consumer, the threadctl drift
# check, reads at most 32 000 bytes and forwards the last 4 000 characters to an
# LLM; anything above 32 KiB total is discarded before use.
LOGS_MAX_LINE_BYTES = 8 * 1024
LOGS_MAX_TOTAL_BYTES = 32 * 1024

# Terminal geometry used for the PTY at launch and as the render fallback
# when a log's run.json has no `terminal` field.
#
# A pty.fork()ed child inherits a 0x0 winsize, so without applying this at
# launch each agent falls back to a size of its own choosing -- OpenCode
# picks 80x24 -- that a renderer cannot observe and does not match. run.json
# records these values, so absolute cursor positions replay into the
# coordinate system they were computed for.
_LAUNCH_TERMINAL_COLS = 120
_LAUNCH_TERMINAL_ROWS = 40
_RENDER_LOG_DEFAULT_WIDTH = _LAUNCH_TERMINAL_COLS
_RENDER_LOG_DEFAULT_HEIGHT = _LAUNCH_TERMINAL_ROWS
_RENDER_LOG_DEFAULT_HISTORY = 2048

# Statuses that are conclusively terminal: the run will never transition
# again on its own. "done"/"failed"/"launch_failed" are published by the
# runner itself; "died"/"killed" are published only by reap/heal
# reconciliation (a dead pid, or a completed idle-kill escalation) — never
# by the runner, since a runner that can still write state files is by
# definition not dead. This set is deliberately narrow: any other status
# (including "stalled", which still has a live, signal-verified pid) is
# treated as non-terminal and is never a target for `agent-run reap`'s
# terminal-state garbage collection.
TERMINAL_STATUSES = frozenset({"done", "failed", "launch_failed", "died", "killed"})

# Recognized non-terminal statuses: "starting"/"running" are raw states the
# runner itself writes; "stalled" is a *computed* status from
# _effective_status (never persisted to disk) for an alive-but-idle runner.
# Any status outside TERMINAL_STATUSES | KNOWN_NONTERMINAL_STATUSES —
# "unknown" (missing/empty status file), a legacy value like "dead" that
# nothing in this repo currently writes, or outright corrupt/garbage text —
# is neither of the above and is treated as unrecognized: never GC'd
# implicitly, never shown under "Live runs", and given its own bucket by
# `list` plus an explicit opt-in escape hatch in `reap` (see
# `--force-unknown`) so an operator has a supported way to clear it instead
# of `rm -rf` by hand.
KNOWN_NONTERMINAL_STATUSES = frozenset({"starting", "running", "stalled"})


def _status_bucket(status: str) -> str:
    """Classify an effective/raw status into "terminal", "live", or
    "unrecognized" for `list` and `reap` GC eligibility decisions."""
    if status in TERMINAL_STATUSES:
        return "terminal"
    if status in KNOWN_NONTERMINAL_STATUSES:
        return "live"
    return "unrecognized"


# Idle-stall threshold: a "running" run whose log file hasn't been touched in
# this many seconds is considered "stalled" by _effective_status(), and a
# candidate for idle-killing by `agent-run reap`.
# Parsed defensively so a typo in the env var doesn't crash every subcommand.
# Values must be finite and positive: `float("nan")` and `float("inf")` both
# parse without raising, and `age < nan` is always False in IEEE-754, so an
# unchecked nan/negative here would silently disable the idle-kill/GC gate
# instead of falling back to the safe default.
def _positive_finite_hours(raw: str, env_name: str, default_hours: float) -> float:
    try:
        value = float(raw)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(raw)
    except ValueError:
        print(
            f"agent-run: warning: {env_name}={raw!r} is not a finite, positive "
            f"number of hours; using default {default_hours:g}h",
            file=sys.stderr,
        )
        return default_hours * 3600
    return value * 3600


def _parse_idle_stall_seconds() -> float:
    raw = os.environ.get("AGENT_RUN_IDLE_KILL_HOURS", "24")
    return _positive_finite_hours(raw, "AGENT_RUN_IDLE_KILL_HOURS", 24.0)


IDLE_STALL_SECONDS: float = _parse_idle_stall_seconds()

# `watch` is stateless, so "growing" means "written to recently" rather than
# "bytes increased since the previous poll": a recency description, not a
# liveness verdict.
WATCH_LOG_GROWING_MAX_AGE_SECONDS: float = 60.0

# A log mtime further than this into the future means the host clock and the
# log's clock disagree (ssh to a skewed host, NFS server-side timestamps),
# not that the log is fresh. Clamping such a delta to 0.0 would report a
# long-dead log as freshly growing — the wrong direction for a contract that
# must fail toward firing — so it degrades mtime_age_s to null instead.
# Smaller negative deltas are ordinary jitter and still clamp to 0.
WATCH_LOG_FUTURE_MTIME_TOLERANCE_SECONDS: float = 2.0

WATCH_TAIL_LINES: int = 200
WATCH_REPEAT_THRESHOLD: int = 3
# Consumers see a line silently clipped at this width, so it is contract.
WATCH_ERROR_LINE_MAX_CHARS: int = 200
WATCH_GIT_SUBPROCESS_TIMEOUT_SECONDS: float = 2.5
# A supervisor polls N runs inside a 30s cycle, so one repo's git calls must
# not each spend up to the per-call timeout; this bounds their total.
WATCH_GIT_TOTAL_BUDGET_SECONDS: float = 5.0
WATCH_TAIL_READ_BLOCK_BYTES: int = 65536
# Cap how far the tail scan walks back, however few newlines it has found.
WATCH_TAIL_MAX_BYTES: int = 256 * 1024
# Line counting is skipped above this threshold and log.lines is set to null.
# 16 MiB covers a full day of typical PTY capture at ~180 bytes/s.
WATCH_LINE_COUNT_MAX_BYTES: int = 16 * 1024 * 1024

# Scratch-scan bounds. The scan walks a working directory including its
# gitignored paths, so nothing about the tree is under our control; each bound
# caps a different way that walk can run long.
# Regular files stat()ed before truncating.
WATCH_SCRATCH_MAX_FILES: int = 2000
# Directory entries visited whatever their type, so that a wide directory of
# non-files (symlinks, FIFOs, empty sub-dirs) is bounded too — the file cap
# alone never fires there.
WATCH_SCRATCH_MAX_ENTRIES: int = 10_000
# Max directory depth to descend into below the working directory root.
WATCH_SCRATCH_MAX_DEPTH: int = 8
# Wall-clock budget for the entire scan, checked between filesystem calls.
# The budget is cooperative, not hard: a single blocking scandir or stat on a
# slow/hostile mount can exceed it without interruption.  A 30s watchdog poll
# runs five serial watches plus a 5s aggregate git budget, so 0.5s per scan
# keeps scratch under 2% of the cycle per watched run even over SSH.
# Uses time.monotonic() to avoid NTP jumps skewing deadline arithmetic.
# The outer deadline check is pinned by test_outer_deadline_check_fires.
WATCH_SCRATCH_BUDGET_SECONDS: float = 0.5
# "Recent" window for files_modified_recent, matching the log growing window
# so both facts describe freshness the same way.
WATCH_SCRATCH_RECENT_SECONDS: float = 60.0
# Directory names pruned from every descent level — heavy generated trees
# that an agent never writes to and that would dominate the file count.
WATCH_SCRATCH_PRUNE_DIRS: frozenset[str] = frozenset({
    ".git", "node_modules", "__pycache__", ".venv",
    "target", "dist", "build",
})

# Git env vars that redirect a command at a different repo/index than the one
# named on the command line; a poll must not inherit these from its caller.
WATCH_GIT_ENV_VARS_TO_STRIP: frozenset[str] = frozenset(
    {
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_INDEX_FILE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_COMMON_DIR",
        "GIT_NAMESPACE",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_CEILING_DIRECTORIES",
        "GIT_REPLACE_REF_BASE",
        # GIT_GRAFT_FILE points at a caller-controlled file that silently
        # rewrites parent pointers, making commits_since_start wrong.
        "GIT_GRAFT_FILE",
    }
)

# Anchored at (stripped) line start, optionally after a `path:`/`prefix:`
# token, with word boundaries — a bare substring match fires on lint/test
# noise like "0 errors, 0 warnings" or a table header "| Error | Count |".
_WATCH_ERROR_LINE_RE = re.compile(r"(?i)^\s*(?:\S+:\s*)?(error|traceback|exception|failed)\b")
# A count of 0 next to the matched word (either order, plural or not) is a
# negation, not an occurrence: "0 errors", "0 failed", "errors: 0".
_WATCH_ERROR_ZERO_COUNT_RE = re.compile(r"(?i)\b0\s+(?:error|fail)\w*\b|\b(?:error|fail)\w*:\s*0\b")
_WATCH_READ_LINE_RE = re.compile(r"\bRead\s+(\S+)")
# A `Read` target must look like a path (contains `/` or `.`), so prose like
# "Please Read carefully" doesn't parse "carefully" as a file.
_WATCH_READ_PATH_LIKE_RE = re.compile(r"[/.]")
# CSI sequences (`\x1b[...<final>`) and other two-byte escapes (`\x1b` plus
# one byte in `@`-`Z` or `\`-`_`), e.g. from colour codes in a TUI capture.
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b[@-Z\\-_]")


# Terminal-state garbage-collection age threshold: how long a run must have
# been sitting in a conclusively terminal status (see TERMINAL_STATUSES)
# before `agent-run reap` removes its ephemeral state dir and persistent
# scratch dir. Deliberately much more conservative than the idle-kill
# threshold above — that one decides when to stop a *live* run; this one
# decides when it's safe to delete the last record of a run that already
# stopped, so it defaults to a full week rather than a day. Parsed
# defensively for the same reason as AGENT_RUN_IDLE_KILL_HOURS. Note there
# is deliberately no module-level constant mirroring IDLE_STALL_SECONDS
# here: `cmd_reap` calls this parser directly at call time (so `--min-age-hours`
# and the env var both work), and a stray module-level call would just
# duplicate the warning below on every subcommand invocation for nothing.
def _parse_reap_min_age_seconds() -> float:
    # AGENT_RUN_MIN_AGE_HOURS is the concise, reap-specific alias introduced
    # after AGENT_RUN_REAP_MIN_AGE_HOURS; retain the latter for compatibility.
    env_name = (
        "AGENT_RUN_MIN_AGE_HOURS"
        if "AGENT_RUN_MIN_AGE_HOURS" in os.environ
        else "AGENT_RUN_REAP_MIN_AGE_HOURS"
    )
    raw = os.environ.get(env_name, "168")
    return _positive_finite_hours(raw, env_name, 168.0)


# Preserved-log GC threshold; deliberately independent of the state-dir
# threshold (see --log-min-age-hours help). Parsed per call so flag and env
# both take effect per invocation.
def _parse_log_min_age_seconds() -> float:
    raw = os.environ.get("AGENT_RUN_LOG_MIN_AGE_HOURS", str(PRUNE_AFTER_DAYS * 24))
    return _positive_finite_hours(raw, "AGENT_RUN_LOG_MIN_AGE_HOURS", PRUNE_AFTER_DAYS * 24.0)


# Orphan-process discovery threshold: minimum age for a process to be
# considered a reap candidate.  Independent of the two thresholds above —
# those govern state-dir and log-dir GC; this one governs live process
# discovery and defaults to 24 h so recently started (possibly still
# mid-launch) runners are never candidates.
def _parse_orphan_min_age_seconds() -> float:
    raw = os.environ.get("AGENT_RUN_ORPHAN_MIN_AGE_HOURS", "24")
    return _positive_finite_hours(raw, "AGENT_RUN_ORPHAN_MIN_AGE_HOURS", 24.0)


# Linked-worktree GC threshold, independent of every threshold above:
# removing a worktree destroys a working tree rather than bookkeeping, so it
# defaults to the same conservative week the state-dir threshold uses and is
# tuned separately from it.
def _parse_worktree_min_age_seconds() -> float:
    raw = os.environ.get("AGENT_RUN_WORKTREE_MIN_AGE_HOURS", "168")
    return _positive_finite_hours(raw, "AGENT_RUN_WORKTREE_MIN_AGE_HOURS", 168.0)


def _parse_reap_max_seconds() -> float:
    """Wall-clock budget for one reap invocation; default 600 s (10 min)."""
    raw = os.environ.get("AGENT_RUN_REAP_MAX_SECONDS", "600")
    try:
        v = float(raw)
        if v > 0 and math.isfinite(v):
            return v
    except (ValueError, TypeError):
        pass
    return 600.0


# ---------------------------------------------------------------------------
# Orphan-process discovery helpers
# ---------------------------------------------------------------------------

class _ProcEntry(NamedTuple):
    """Snapshot of one entry from the OS process table.

    ``start_time`` is an epoch float (seconds since the Unix epoch) used
    only for age comparisons — it does not need sub-second precision.
    ``None`` means the start time could not be determined; such entries are
    skipped by ``_find_orphan_runners`` with reason ``start_time_unknown``
    rather than being treated as ancient (fail-closed, not fail-open).
    ``identity`` mirrors the format used by ``_process_identity``: a
    stable platform-specific birth token that lets callers detect PID
    recycling between discovery and any later action.
    ``pgid`` is the process group ID, used to skip processes in the
    reaper's own process group (``pgid == self_pgid``).
    """
    pid: int
    ppid: int
    uid: int
    argv: List[str]
    start_time: Optional[float]  # epoch seconds; None = unknown
    identity: str                # "linux:<starttime>" or "darwin:<lstart>"
    pgid: int = 0                # process group ID; 0 means unknown


class _OrphanCandidate(NamedTuple):
    """A runner process judged eligible for orphan handling."""
    pid: int
    name: str
    start_time: Optional[float]  # epoch seconds; None if indeterminate
    identity: str


class _OrphanSkip(NamedTuple):
    """A process that was examined but excluded, with a machine-readable reason."""
    pid: int
    reason: str
    detail: str = ""


def _scan_process_table_linux() -> List[_ProcEntry]:
    """Read /proc to build a process table of current-uid entries.

    Tolerates races: a pid that vanishes between directory listing and
    reading its files is silently skipped.  Only processes owned by the
    current uid are returned.
    """
    my_uid = os.getuid()
    entries: List[_ProcEntry] = []
    try:
        pids = [int(p) for p in os.listdir("/proc") if p.isdigit()]
    except OSError:
        return entries

    # /proc/uptime and SC_CLK_TCK are the same for every pid in the snapshot;
    # read them once so the per-pid loop does not pay repeated open/read/close
    # and sysconf calls (~1000 on a loaded host).  A failure degrades every
    # entry to start_time=None (fail-closed).
    try:
        uptime_s = float(Path("/proc/uptime").read_text().split()[0])
        boot_epoch: Optional[float] = time.time() - uptime_s
        hz: Optional[float] = float(os.sysconf("SC_CLK_TCK"))
    except (OSError, ValueError, AttributeError):
        boot_epoch = None
        hz = None

    for pid in pids:
        try:
            st = os.stat(f"/proc/{pid}")
            if st.st_uid != my_uid:
                continue
            # NUL-separated argv from /proc/<pid>/cmdline; empty means kernel thread.
            raw = Path(f"/proc/{pid}/cmdline").read_bytes()
            if not raw:
                continue
            argv = raw.rstrip(b"\x00").split(b"\x00")
            argv_str = [a.decode("utf-8", errors="replace") for a in argv]

            # Parse /proc/<pid>/stat: ppid is field 4 (0-indexed 3),
            # starttime is field 22 (0-indexed 21), both after the comm field.
            fields = _proc_stat_fields(pid)
            if fields is None or len(fields) <= 19:
                continue
            ppid = int(fields[1]) if fields[1].lstrip("-").isdigit() else 0
            pgid = int(fields[2]) if fields[2].lstrip("-").isdigit() else 0
            starttime_ticks = int(fields[19]) if fields[19].isdigit() else 0

            # Convert kernel jiffies to epoch seconds using the pre-read boot_epoch.
            if boot_epoch is not None and hz is not None:
                start_time: Optional[float] = boot_epoch + starttime_ticks / hz
            else:
                start_time = None

            identity = f"linux:{starttime_ticks}"
            entries.append(_ProcEntry(
                pid=pid, ppid=ppid, uid=my_uid,
                argv=argv_str, start_time=start_time, identity=identity,
                pgid=pgid,
            ))
        except (OSError, ValueError):
            continue
    return entries


def _darwin_lstart_normalize(raw: str) -> str:
    """Collapse internal whitespace runs in a Darwin ``lstart`` string to a
    single space.

    ``ps -axo lstart=`` collapses whitespace when splitting columns (single
    space between tokens), while ``ps -p PID -o lstart=`` preserves the
    fixed-width right-padding that the kernel uses for single-digit days
    (e.g. ``Mon Jun  8`` — two spaces between month and day).  Calling this
    on both sides before building the ``darwin:<lstart>`` identity token
    makes the two forms compare equal regardless of day width.
    """
    return " ".join(raw.split())


def _scan_process_table_darwin() -> List[_ProcEntry]:
    """Query the macOS process table via ``ps``.

    Uses ``ps -axo`` to enumerate all uid-visible processes, then filters
    to the current uid.  The ``lstart`` field is the same token used by
    ``_process_identity`` on Darwin, so identity values are comparable.
    Tolerates ps failure: returns an empty list rather than raising.
    """
    my_uid = os.getuid()
    entries: List[_ProcEntry] = []
    try:
        # pid, ppid, pgid, uid, lstart (5 whitespace-separated tokens), and the full command.
        # LC_ALL=C forces ASCII month/day names regardless of the operator's
        # locale; a non-C locale (e.g. fr_FR.UTF-8) causes ps to emit localised
        # month names that strptime cannot parse with the fixed "%b" directive.
        env_c = {**os.environ, "LC_ALL": "C"}
        result = subprocess.run(
            ["ps", "-axo", "pid=,ppid=,pgid=,uid=,lstart=,command="],
            capture_output=True, text=True, timeout=10, env=env_c,
        )
    except (OSError, subprocess.SubprocessError):
        return entries

    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            parts = line.split(None, 4)
            if len(parts) < 5:
                continue
            pid = int(parts[0])
            ppid = int(parts[1])
            pgid = int(parts[2])
            uid = int(parts[3])
            if uid != my_uid:
                continue
            # lstart: 5 whitespace-separated tokens "Www Mmm DD HH:MM:SS YYYY".
            # split(None, 5) separates these 5 tokens from the command.
            rest = parts[4]
            lstart_parts = rest.split(None, 5)
            if len(lstart_parts) < 6:
                continue
            lstart = " ".join(lstart_parts[:5])
            command_str = lstart_parts[5]
        except (ValueError, IndexError):
            continue

        # Parse the command string into argv tokens.  ps reconstructs argv
        # by joining with spaces, so we cannot perfectly recover arguments
        # that contained spaces; shlex.split is the best available heuristic.
        try:
            argv_str = shlex.split(command_str)
        except ValueError:
            argv_str = command_str.split()
        if not argv_str:
            continue

        # Convert lstart to an epoch float for age comparisons.
        # lstart is wall-clock local time; a naive datetime.timestamp()
        # interprets it in the host's local timezone, matching what ps printed.
        # LC_ALL=C on the ps invocation guarantees ASCII day/month names, so
        # strptime always matches "%a %b %d %H:%M:%S %Y" regardless of locale.
        # A parse failure leaves start_time as None; _find_orphan_runners treats
        # None as unknown and emits start_time_unknown (fail-closed, not open).
        start_time: Optional[float]
        try:
            # "Www Mmm DD HH:MM:SS YYYY" — strptime consumes leading zeros.
            start_time = datetime.strptime(lstart, "%a %b %d %H:%M:%S %Y").timestamp()
        except ValueError:
            start_time = None

        identity = f"darwin:{_darwin_lstart_normalize(lstart)}"
        entries.append(_ProcEntry(
            pid=pid, ppid=ppid, uid=uid,
            argv=argv_str, start_time=start_time, identity=identity,
            pgid=pgid,
        ))
    return entries


def _scan_process_table() -> List[_ProcEntry]:
    """Return a snapshot of the OS process table, current uid only.

    Platform dispatch: Linux reads /proc; Darwin uses ``ps``.  A pid that
    vanishes mid-scan is silently skipped rather than raising.
    """
    system = platform.system()
    if system == "Linux":
        return _scan_process_table_linux()
    if system == "Darwin":
        return _scan_process_table_darwin()
    return []


def _argv_is_agent_run_runner(argv: Sequence[str]) -> bool:
    """Return True only when ``argv`` invokes the agent-run entry point.

    Matches two forms:
      - A direct invocation where argv[0]'s basename is ``agent-run``.
      - A Python-interpreter invocation where some argv element's basename
        is ``agent-run`` (e.g. ``python /usr/local/bin/agent-run <args>``).

    Matching is on path *components*, not substrings: a process whose argv
    merely mentions ``agent-run`` in an argument value — the prototypical
    false positive being ``bash -lc "cat /var/tmp/agent-runs/foo/log"`` —
    returns False.  The basename check is the key gate: only a token whose
    final path component is exactly ``agent-run`` qualifies, so ``grep
    agent-run``, an editor with the file open, and shell commands that
    expand the string as data are all excluded.
    """
    if not argv:
        return False

    if Path(argv[0]).name == "agent-run":
        return True

    # Python-interpreter invocation: skip interpreter and its own flags
    # until the first non-flag positional is found; that positional is
    # the script path.
    interpreter_name = Path(argv[0]).name
    if interpreter_name.startswith("python"):
        i = 1
        while i < len(argv):
            token = argv[i]
            if not token.startswith("-"):
                # First positional: this is the script.
                return Path(token).name == "agent-run"
            # Python flags that consume their value as the next argv token.
            # -W and -X each take one value argument; -c and -m end the
            # flag sequence but their operands are not a script path.
            if token in ("-W", "-X"):
                i += 2
                continue
            if token in ("-c", "-m"):
                # Command string or module name follows; no script path.
                return False
            # All other single-dash flags (-u, -O, -B, -d, -E, -i, …) are
            # boolean; skip just this token.
            i += 1
        return False

    return False


def _run_name_from_argv(argv: Sequence[str]) -> Optional[str]:
    """Extract the run name from a runner's argv, or None if indeterminate.

    Calls ``_parse_launch_argv`` so this path and the launch path always
    agree on flag parsing, including the ``--`` separator form.  Returns
    None when:
      - ``_parse_launch_argv`` raises ``_LaunchArgvError``
      - the argv belongs to a subcommand (reap, list, …) rather than a launch
      - the recovered name fails ``_validate_run_name``
      - no name was found (too few arguments)
    Never guesses.
    """
    # Strip the entry-point token(s) from argv before parsing.  A direct
    # invocation has one leading token (agent-run); a python-wrapped
    # invocation has two or more (python /path/agent-run).  We drop
    # everything up to and including the agent-run token.
    rest = list(argv)
    found = False
    for i, token in enumerate(rest):
        if Path(token).name == "agent-run":
            rest = rest[i + 1:]
            found = True
            break
    if not found:
        return None

    try:
        parsed = _parse_launch_argv(rest)
    except _LaunchArgvError:
        return None

    # A subcommand invocation (agent-run reap, agent-run list, …) is not a
    # launch; it has no run name to recover.
    if parsed.subcommand_tokens is not None:
        return None

    name = parsed.name
    if not name:
        return None

    # Guard against invalid names reaching filesystem paths.
    try:
        _validate_run_name(name)
    except SystemExit:
        return None

    return name


def _find_orphan_runners(
    table: Sequence[_ProcEntry],
    *,
    min_age_seconds: float,
    now: float,
    self_pid: int,
    self_pgid: int,
    target_name: Optional[str] = None,
) -> tuple[List[_OrphanCandidate], List[_OrphanSkip]]:
    """Classify each _ProcEntry as an orphan candidate or a skip.

    A candidate satisfies all of:
      - argv identifies it as an agent-run runner (_argv_is_agent_run_runner)
      - run name recovers (_run_name_from_argv) and passes _validate_run_name
      - no state dir exists for that name (_path_entry_exists returns False)
      - process age >= min_age_seconds
      - not self_pid, not any ancestor of self_pid, not in self_pgid, not pid 1
      - uid matches the current uid (guaranteed by _scan_process_table, checked here too)
      - matches target_name when given

    Every rejected entry is returned as an _OrphanSkip with a distinct reason.
    Processes that do not look like agent-run runners at all are silently
    excluded (they are noise, not skips).
    """
    my_uid = os.getuid()

    # Build a pid→ppid map for ancestor-chain traversal.
    ppid_map: dict[int, int] = {e.pid: e.ppid for e in table}

    def _ancestor_pids(pid: int) -> set[int]:
        """Walk ppid chain up to pid 1 or a cycle; return all ancestor pids."""
        seen: set[int] = set()
        current = ppid_map.get(pid)
        while current is not None and current not in seen and current > 1:
            seen.add(current)
            current = ppid_map.get(current)
        return seen

    ancestors = _ancestor_pids(self_pid)

    candidates: List[_OrphanCandidate] = []
    skips: List[_OrphanSkip] = []

    for entry in table:
        if not _argv_is_agent_run_runner(entry.argv):
            continue  # not an agent-run runner — ignore entirely

        pid = entry.pid

        if pid == 1:
            skips.append(_OrphanSkip(pid=pid, reason="pid_1"))
            continue
        if entry.uid != my_uid:
            skips.append(_OrphanSkip(pid=pid, reason="foreign_uid", detail=str(entry.uid)))
            continue
        if pid == self_pid:
            skips.append(_OrphanSkip(pid=pid, reason="self"))
            continue
        if pid in ancestors:
            skips.append(_OrphanSkip(pid=pid, reason="ancestor"))
            continue
        if entry.pgid == self_pgid:
            skips.append(_OrphanSkip(pid=pid, reason="same_pgid"))
            continue

        name = _run_name_from_argv(entry.argv)
        if name is None:
            skips.append(_OrphanSkip(pid=pid, reason="name_unrecoverable"))
            continue

        if target_name is not None and name != target_name:
            skips.append(_OrphanSkip(pid=pid, reason="name_mismatch", detail=name))
            continue

        # Ambiguity guard: ps joins argv with spaces, so shlex.split and plain
        # str.split() may recover different names when the command string contains
        # shell metacharacters (quotes, backslashes).  Differing results indicate
        # that the recovered name is not unique and the candidate is unsafe to act on.
        # (Space-in-path mis-parses are caught below by _log_dir corroboration.)
        command_str = " ".join(entry.argv)
        try:
            plain_argv = command_str.split()
        except Exception:
            plain_argv = []
        plain_name = _run_name_from_argv(plain_argv)
        if plain_name != name:
            skips.append(_OrphanSkip(pid=pid, reason="argv_ambiguous", detail=name))
            continue

        # Log-dir corroboration: every real agent-run runner creates LOG_ROOT/<name>
        # before writing any state.  Requiring the directory to exist — and to be a
        # real directory, not a symlink — rejects the entire mis-parse class from S1:
        # a shifted name like "prompt.md" has no LOG_ROOT entry, so it is never a
        # candidate regardless of the state-dir outcome.
        # ps argv reconstruction is lossy; the log dir is the filesystem artefact
        # that corroborates the recovered name without re-parsing the command string.
        log_d = _log_dir(name)
        try:
            lst = log_d.lstat()
            if not _stat_module.S_ISDIR(lst.st_mode):
                skips.append(_OrphanSkip(pid=pid, reason="log_dir_missing", detail=name))
                continue
        except FileNotFoundError:
            skips.append(_OrphanSkip(pid=pid, reason="log_dir_missing", detail=name))
            continue

        # State-root guard: a runner started with AGENT_RUN_STATE_DIR=/other/path
        # records state there, not under our STATE_ROOT.  The state-dir check below
        # asks the wrong question unless both roots agree.  Read the process env
        # to verify; if unreadable (or not Linux), skip conservatively.
        runner_root = _runner_state_root(pid)
        if runner_root is None:
            skips.append(_OrphanSkip(pid=pid, reason="state_root_unreadable"))
            continue
        if runner_root.resolve() != STATE_ROOT.resolve():
            skips.append(_OrphanSkip(pid=pid, reason="foreign_state_root",
                                     detail=str(runner_root)))
            continue

        # State-dir check: a process with a live state dir is tracked by
        # reap's existing passes, not this one.
        if _path_entry_exists(_state_dir(name)):
            skips.append(_OrphanSkip(pid=pid, reason="state_dir_exists", detail=name))
            continue

        # Age gate: too-young processes are excluded so a runner still
        # mid-launch (state dir creation in progress) is not a candidate.
        # start_time=None means the parse failed; treat as unknown rather than
        # as infinitely old — any uncertainty skips the candidate (fail-closed).
        if entry.start_time is None:
            skips.append(_OrphanSkip(pid=pid, reason="start_time_unknown"))
            continue
        age = now - entry.start_time
        if age < min_age_seconds:
            skips.append(_OrphanSkip(
                pid=pid, reason="too_young",
                detail=f"age={age:.1f}s min={min_age_seconds:.1f}s",
            ))
            continue

        candidates.append(_OrphanCandidate(
            pid=pid, name=name,
            start_time=entry.start_time,
            identity=entry.identity,
        ))

    return candidates, skips


# Launch-grace window: how many seconds after exec a non-zero exit still reads
# as argv validation or a missing binary rather than work that ran and failed.
def _positive_finite_seconds(raw: str, label: str, default_seconds: float) -> float:
    try:
        value = float(raw)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(raw)
    except ValueError:
        print(
            f"agent-run: warning: {label}={raw!r} is not a finite, positive "
            f"number of seconds; using default {default_seconds:g}s",
            file=sys.stderr,
        )
        return default_seconds
    return value


def _parse_launch_grace_seconds() -> float:
    raw = os.environ.get("AGENT_RUN_LAUNCH_GRACE_SECS", "10")
    return _positive_finite_seconds(raw, "AGENT_RUN_LAUNCH_GRACE_SECS", 10.0)


LAUNCH_GRACE_SECONDS: float = _parse_launch_grace_seconds()

# How much of the log to scan for the diagnostic first line, and how much of
# that line to keep.
LAUNCH_ERROR_SCAN_BYTES = 65536
LAUNCH_ERROR_MAX_CHARS = 500

PROMPT_UNSUBMITTED_ERROR = "agent exited before its prompt file was submitted"

# The interactive prompt helper waits this long for the TUI to enter raw mode
# before submitting; it also bounds how late an unsubmitted prompt still counts
# as a launch failure.
PROMPT_SUBMISSION_DELAY_SECONDS = 4.0


# Off by default, unlike the thresholds above. An invalid flag value is a hard
# launch-time error: a launcher that believes idle-timeout is active must not
# silently run without it.
def _parse_idle_timeout_flag(raw: str) -> float:
    return _positive_finite_float(raw)


def _idle_timeout_env_seconds() -> Optional[float]:
    """Parse AGENT_RUN_IDLE_TIMEOUT_SECS, defaulting to off (None) — both when
    unset and when the value fails validation, since idle-timeout being
    disabled is this feature's fail-safe direction."""
    raw = os.environ.get("AGENT_RUN_IDLE_TIMEOUT_SECS")
    if raw is None:
        return None
    try:
        value = float(raw)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(raw)
    except ValueError:
        print(
            f"agent-run: warning: AGENT_RUN_IDLE_TIMEOUT_SECS={raw!r} is not a "
            "finite, positive number of seconds; idle-timeout stays off",
            file=sys.stderr,
        )
        return None
    return value


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
    # A poll loop must degrade on an unreadable state file, not crash: a
    # directory in place of a file, mode 000, or non-UTF-8 bytes are all
    # observation failures, not process failures.
    try:
        return path.read_text().strip()
    except (FileNotFoundError, OSError, UnicodeError):
        return default


def _safe_int(value: str) -> Optional[int]:
    """Parse ``value`` as an int, or ``None`` for anything that isn't one.

    Use this rather than an ``isdigit()`` guard around ``int()``:
    ``str.isdigit()`` accepts non-ASCII digits like ``"\u00b2"`` that
    ``int()`` then rejects.
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


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


def _open_real_directory(path: Path) -> int:
    """Open a directory without following its final path component."""
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    try:
        return os.open(path, flags)
    except FileNotFoundError:
        raise
    except OSError as exc:
        sys.exit(f"agent-run: refusing to use {path} — not a real directory ({exc})")


def _safe_rmtree(candidate: Path, root: Path, expected: Optional[os.stat_result] = None) -> None:
    """Remove a direct-child directory without following a mutable pathname.

    The directory is opened relative to an opened root with ``O_NOFOLLOW`` and
    recursively emptied through directory descriptors.  Consequently a rename
    plus symlink replacement after inspection cannot redirect deletion to a
    sibling (the final ``rmdir`` is also relative to the original root fd).
    """
    # pathlib never resolves ".." lexically, so ``(root / "..").parent`` is
    # ``root`` itself — the parent check below would accept it and then
    # ``os.open("..", dir_fd=root_fd)`` would walk straight out of root.
    # Reject any non-plain name before it ever reaches dir_fd-relative opens.
    if candidate.name in ("", ".", "..") or candidate.parent != root:
        sys.exit(f"agent-run: refusing to delete {candidate} — not a direct child of {root}")
    root_fd = child_fd = -1
    try:
        root_fd = _open_real_directory(root)
        flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
        try:
            child_fd = os.open(candidate.name, flags, dir_fd=root_fd)
        except FileNotFoundError:
            return
        except OSError as exc:
            sys.exit(f"agent-run: refusing to delete {candidate} — not a real directory ({exc})")
        st = os.fstat(child_fd)
        if not _stat_module.S_ISDIR(st.st_mode):
            sys.exit(f"agent-run: refusing to delete {candidate} — not a real directory")
        if expected is not None and (st.st_dev, st.st_ino) != (expected.st_dev, expected.st_ino):
            return

        def remove_tree(fd: int) -> None:
            for entry in os.listdir(fd):
                entry_st = os.stat(entry, dir_fd=fd, follow_symlinks=False)
                if _stat_module.S_ISDIR(entry_st.st_mode):
                    nested = os.open(entry, flags, dir_fd=fd)
                    try:
                        remove_tree(nested)
                    finally:
                        os.close(nested)
                    os.rmdir(entry, dir_fd=fd)
                else:
                    os.unlink(entry, dir_fd=fd)

        remove_tree(child_fd)
        # Re-verify the pathname still names the inode we opened and emptied.
        # A rename-then-replace between the open above and here would
        # otherwise let a path-relative rmdir remove an unrelated directory
        # that happens to share the name.
        try:
            final_st = os.stat(candidate.name, dir_fd=root_fd, follow_symlinks=False)
        except FileNotFoundError:
            return
        if (final_st.st_dev, final_st.st_ino) != (st.st_dev, st.st_ino):
            return  # candidate was replaced after inspection — do not touch it
        os.rmdir(candidate.name, dir_fd=root_fd)
    except FileNotFoundError:
        return
    finally:
        if child_fd >= 0:
            os.close(child_fd)
        if root_fd >= 0:
            os.close(root_fd)


# Reserved dot-prefixed sentinel name used by _crash_safe_rmtree. Every scan
# in this module that walks STATE_ROOT (both reap loops, _opportunistic_heal,
# cmd_list) already skips dot-prefixed entries, so a sentinel left behind by
# an interrupted deletion never reappears as a phantom "live" run — it just
# sits there, invisible to everything except _reap_stale_sentinels, until a
# later reap finishes deleting it.
_SENTINEL_PREFIX = ".reaping-"


def _sentinel_name(name: str) -> str:
    return f"{_SENTINEL_PREFIX}{name}.{os.getpid()}.{time.time_ns()}"


def _crash_safe_rmtree(candidate: Path, root: Path, expected: os.stat_result) -> None:
    """Crash-safely remove *candidate*, a direct child of *root*.

    ``_safe_rmtree`` empties a directory entry-by-entry with no ordering
    discipline: if the process is killed (OOM, timer TimeoutStopSec,
    operator Ctrl-C) partway through, the directory can survive with a
    partial file set and, critically, no ``status`` file — which makes it
    permanently un-recognizable to reap's own terminal-status check (see
    H4) and (before this fix) get misclassified as "live" by ``list``.

    This makes deletion self-identifying and resumable: the directory is
    first renamed, atomically, to a reserved sentinel name under the same
    root, and only then emptied and removed. A crash between the rename and
    the final ``rmdir`` leaves a dot-prefixed sentinel — already invisible
    to every scan in this module — rather than a stranded, misclassified
    original name. ``_reap_stale_sentinels`` (called at the top of every
    ``reap`` invocation) finishes deleting any sentinel left behind by a
    prior interrupted run.

    Raises ``SystemExit`` under the same conditions ``_safe_rmtree`` does
    (candidate not a real directory, inode mismatch after the rename) —
    callers should catch this exactly as they would for a direct
    ``_safe_rmtree`` call.
    """
    sentinel = root / _sentinel_name(candidate.name)
    root_fd = -1
    try:
        # Rename via one opened parent directory. This preserves the direct-
        # child containment rule even if the textual root pathname is swapped
        # after our caller's inspection, and expected= still makes the final
        # descriptor-relative delete refuse a replaced candidate.
        root_fd = _open_real_directory(root)
        os.rename(candidate.name, sentinel.name, src_dir_fd=root_fd, dst_dir_fd=root_fd)
    except FileNotFoundError:
        return
    finally:
        if root_fd >= 0:
            os.close(root_fd)
    _safe_rmtree(sentinel, root, expected=expected)


def _reap_stale_sentinels(root: Path) -> int:
    """Finish deleting any sentinel directories left behind by a
    ``_crash_safe_rmtree`` call that was interrupted mid-deletion.

    Called at the top of every ``reap`` invocation so a crash during one
    reap is cleaned up transparently by the next one, without requiring any
    manual operator intervention. Returns the count cleared.
    """
    if not root.is_dir():
        return 0
    try:
        candidates = list(root.iterdir())
    except OSError:
        return 0
    cleared = 0
    for d in candidates:
        if not d.name.startswith(_SENTINEL_PREFIX):
            continue
        try:
            st = d.lstat()
        except OSError:
            continue
        if not _stat_module.S_ISDIR(st.st_mode):
            continue
        try:
            _safe_rmtree(d, root, expected=st)
        except SystemExit as exc:
            print(f"agent-run: warning: cannot resume interrupted deletion {d}: {exc}", file=sys.stderr)
            continue
        cleared += 1
    return cleared


@contextmanager
def _worktree_publication_lock(*, exclusive: bool):
    """Serialize launch-state publication with destructive worktree removal."""
    lock_dir = STATE_ROOT / ".locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / "worktree-publication.lock"
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        yield fd
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


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
    runs started before this split remain readable.

    Never raises: like ``Path.is_dir()``, ``Path.exists()`` propagates
    ``OSError`` (e.g. an unreadable parent directory) instead of returning
    ``False``, and this is reachable from ``cmd_status``, which has no
    enclosing never-raise guard."""
    for log in (_log_dir(name) / "log", _state_dir(name) / "log"):
        try:
            if log.exists():
                return log
        except OSError:
            continue
    return None


@contextmanager
def _watch_open_validated_log(path: Optional[Path]):
    """Open *path* and validate it is a regular file on the resulting
    descriptor, yielding a binary file object — or ``None`` if *path* is
    unset, cannot be opened, or does not stat as a regular file once open.

    Validation is on the descriptor, not the pathname: the inode a path
    names can be replaced (rotation, an agent recreating its log, a hostile
    swap) between a check and a later open, whereas a descriptor always
    refers to the inode it was opened against. ``O_NONBLOCK`` keeps the open
    itself safe against a FIFO with no writer, and is cleared again so a
    regular-file read behaves normally. Never raises: any ``OSError``
    degrades to yielding ``None``, and the fd is closed on every path.
    """
    fd: Optional[int] = None
    f = None
    try:
        if path is not None:
            try:
                fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
                if _stat_module.S_ISREG(os.fstat(fd).st_mode):
                    try:
                        flags = fcntl.fcntl(fd, fcntl.F_GETFL)
                        fcntl.fcntl(fd, fcntl.F_SETFL, flags & ~os.O_NONBLOCK)
                    except OSError:
                        pass
                    f = os.fdopen(fd, "rb")
                    fd = None  # ownership transferred to f
            except OSError:
                pass
        yield f
    finally:
        if f is not None:
            f.close()
        elif fd is not None:
            os.close(fd)


def _path_entry_exists(path: Path) -> bool:
    """Does a directory entry exist, without following a possible symlink?"""
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def _is_dir_safe(path: Path) -> bool:
    """Like ``Path.is_dir()``, but never raises.

    ``Path.is_dir()`` propagates ``OSError`` (e.g. ``PermissionError`` from
    an unreadable parent directory) instead of returning ``False``. The
    existence checks gating `cmd_watch`/`cmd_status` must degrade rather
    than crash, so an unresolvable run name stays the only non-zero exit.
    """
    try:
        return path.is_dir()
    except OSError:
        return False


# Tri-state values returned by _probe_dir_state.
_DIR_PRESENT = "present"        # a directory is there
_DIR_MISSING = "missing"        # nothing there, or there but not a directory
_DIR_UNSTATABLE = "unstatable"  # stat raised OSError (e.g. unreadable parent)


def _probe_dir_state(path: Path) -> str:
    """Return a tri-state describing *path*: present, missing, or unstatable.

    ``Path.is_dir()`` answers ``False`` both for "absent" and for "cannot be
    interrogated" (an unreadable parent raises on some platforms and returns
    ``False`` on others), while ``os.stat()`` raises ``OSError`` for the
    latter.  Callers that must not read "cannot stat" as "gone" use this
    probe rather than ``Path.is_dir()`` or ``_is_dir_safe()``.

    ``ELOOP`` (self-referential symlink) counts as missing: no directory
    could ever exist at that path.
    """
    try:
        st = os.stat(path)
        return _DIR_PRESENT if _stat_module.S_ISDIR(st.st_mode) else _DIR_MISSING
    except FileNotFoundError:
        return _DIR_MISSING
    except OSError as exc:
        return _DIR_MISSING if exc.errno == errno.ELOOP else _DIR_UNSTATABLE


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
        initial_mtime = _newest_mtime(d)
        if initial_mtime is None:
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
                # Uses the same TERMINAL_STATUSES set reap's own GC pass
                # uses, so this 21-day log prune and reap's (much shorter,
                # by default) terminal-state GC never disagree about which
                # runs are "done with".
                status = _read(state_dir / "status")
                if status not in TERMINAL_STATUSES:
                    continue
            mtime = _newest_mtime(d)
            if mtime is None:
                continue
            if mtime >= cutoff:
                continue
            _safe_rmtree(d, LOG_ROOT, expected=current)


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
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


def _proc_stat_fields(pid: int) -> Optional[List[str]]:
    """Whitespace-split fields of ``/proc/<pid>/stat`` after the comm field
    (so a comm containing spaces or ')' cannot shift the indices), or
    ``None`` if it cannot be read or parsed. Linux only."""
    try:
        _before, after = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)
        return after.split()
    except (IndexError, OSError, ValueError):
        return None


def _ps_field(pid: int, fmt: str) -> Optional[str]:
    """One ``ps -o <fmt>=`` field for ``pid``, or ``None`` if ps fails or
    prints nothing."""
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", f"{fmt}="],
            capture_output=True,
            check=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _process_identity(pid: int) -> Optional[str]:
    """Return a stable platform-specific birth token for ``pid``, if readable."""
    if os.environ.get("AGENT_RUN_TEST_SLOW_IDENTITY"):
        # Test-only hook, inert unless this env var is explicitly set: widens
        # the real window between a runner's fork and its state publication
        # so tests exercising R3 (launcher-death-before-publication) can
        # deterministically hit that race instead of relying on scheduling
        # luck. Never set in normal operation.
        time.sleep(1.5)
    system = platform.system()
    if system == "Linux":
        fields = _proc_stat_fields(pid)
        if fields is None or len(fields) <= 19 or not fields[19].isdigit():
            return None
        return f"linux:{fields[19]}"
    if system == "Darwin":
        start = _ps_field(pid, "lstart")
        return f"darwin:{_darwin_lstart_normalize(start)}" if start else None
    return None


def _runner_state_root(pid: int) -> Optional[Path]:
    """Read ``AGENT_RUN_STATE_DIR`` from the environment of ``pid``.

    A runner started with a non-default state root records state there, not
    under the reaper's ``STATE_ROOT``.  Requiring the two to match before
    treating a process as an orphan prevents the reaper from killing a live,
    tracked run just because it used a different root.

    Returns the resolved ``Path`` of the runner's state root, or ``None`` if
    the environment cannot be read (caller must skip the candidate — fail-closed).

    On Linux the process's NUL-separated ``/proc/<pid>/environ`` is parsed
    directly.  If ``AGENT_RUN_STATE_DIR`` is absent the runner uses the same
    default as the reaper (``STATE_ROOT``), so ``STATE_ROOT`` is returned.
    ``/proc/<pid>/environ`` takes the target's ``mmap_lock`` (the same
    uninterruptible-read hazard as ``/proc/<pid>/cmdline``; see P5 in
    ``findings-performance.md``).  This read runs inside ``_find_orphan_runners``
    during discovery — before the per-candidate budget checks in the orphan
    processing loop — so a single process stuck in D state can block discovery
    for the duration of its stall.  The overall ``reap_budget`` enforced in
    ``cmd_reap`` covers only the post-discovery action phase and does not bound
    this read.

    Darwin does not expose another process's environment through an
    unprivileged boundary-preserving API, so it returns ``None``. Destructive
    orphan discovery must skip a candidate whose state root is unknown.
    """
    if platform.system() != "Linux":
        return None
    try:
        raw = Path(f"/proc/{pid}/environ").read_bytes()
    except OSError:
        return None
    for kv in raw.split(b"\x00"):
        if kv.startswith(b"AGENT_RUN_STATE_DIR="):
            val = kv[len(b"AGENT_RUN_STATE_DIR="):]
            try:
                return Path(val.decode("utf-8", errors="strict"))
            except (UnicodeDecodeError, ValueError):
                return None
    # Variable absent: runner uses the compiled-in default.
    return STATE_ROOT


def _watch_pid_is_zombie(pid: int) -> bool:
    """Is ``pid`` currently in zombie state (exited, not yet reaped)?

    A zombie's pid entry and its ``_process_identity`` birth token both
    survive until the parent reaps it, so ``os.kill(pid, 0)`` succeeding and
    the identity token matching are not evidence the process is still doing
    work. Watch-local only: callers outside `watch`'s liveness path must not
    use this.

    Any unreadable/failed probe or an unsupported platform returns
    ``False`` — never invent a zombie verdict from a failed probe.
    """
    system = platform.system()
    if system == "Linux":
        fields = _proc_stat_fields(pid)
        return bool(fields) and fields[0] == "Z"
    if system == "Darwin":
        state = _ps_field(pid, "stat")
        return bool(state) and state[0] == "Z"
    return False


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


def _mark_died(state_dir: Path, reason: str) -> None:
    """Write status=died, ended_at, and reap_reason to *state_dir*.

    Idempotent: ended_at is only written if not already present.
    """
    _mark_terminal(state_dir, "died", reason)


def _mark_terminal(state_dir: Path, status: str, reason: str) -> None:
    """Write ``status``, ``ended_at`` (idempotently), and ``reap_reason``.

    Shared by the dead-pid ("died") and idle-kill ("killed") reconciliation
    paths, which previously each hand-rolled the identical three-write
    sequence and had already begun to drift (see quality finding F3).
    ``ended_at`` is only written if not already present, matching
    ``_mark_died``'s original idempotence: a run that already recorded when
    it actually stopped (e.g. ``_force_kill`` writing terminal state
    directly) must not have that timestamp overwritten by a later
    reconciliation pass — doing so would reset the reap min-age clock and
    reopen the same-pass double-action hole this fix closes (see H1).

    Also mirrors the terminal facts to run.json in the persistent log dir so
    the reboot-durable record reflects abnormal exits, not just clean ones
    from _finalize. Failure to write run.json never affects the run.
    """
    _write(state_dir / "status", status + "\n")
    ended_at_path = state_dir / "ended_at"
    if not ended_at_path.exists():
        ended_at = _now_iso()
        _write(ended_at_path, ended_at + "\n")
    else:
        try:
            ended_at = ended_at_path.read_text().strip()
        except OSError:
            ended_at = _now_iso()
    _write(state_dir / "reap_reason", reason + "\n")
    # Mirror to run.json; exit_code is read from the state dir if present so the
    # same record used by _finalize is reflected here when available.
    try:
        exit_code_raw = (state_dir / "exit_code").read_text().strip()
        exit_code: Optional[int] = int(exit_code_raw)
    except (OSError, ValueError):
        exit_code = None
    log_dir = LOG_ROOT / state_dir.name
    if log_dir.is_dir():
        run_json_data: dict = {"status": status, "ended_at": ended_at}
        if exit_code is not None:
            run_json_data["exit_code"] = exit_code
        _write_run_json(log_dir, run_json_data)


def _newest_mtime(d: Path) -> Optional[float]:
    """Newest mtime across the direct (non-recursive) children of ``d``, or
    ``d``'s own mtime if it has none. ``None`` if ``d`` cannot be stat'd.

    Shared by ``_terminal_state_age_seconds`` and ``_prune_old_logs`` (see
    quality finding F4), which each independently computed this.
    """
    try:
        return max(
            (f.stat().st_mtime for f in d.iterdir()),
            default=d.stat().st_mtime,
        )
    except FileNotFoundError:
        return None
    except OSError as exc:
        print(
            f"agent-run: warning: {d.name}: cannot determine age ({exc})",
            file=sys.stderr,
        )
        return None


# Returned by _newest_mtime_recursive when an OSError cut the walk short;
# distinct from None (walk completed, directory genuinely empty).
_WALK_INCOMPLETE: object = object()


def _newest_mtime_recursive(
    d: Path,
    cutoff: Optional[float] = None,
    *,
    skip_top_dir_mtime: bool = False,
    strict: bool = False,
) -> Optional[float]:
    """Newest mtime anywhere below ``d`` (including ``d`` itself unless
    ``skip_top_dir_mtime`` is set), without following symlinks.

    Returns ``None`` when the walk completed and no entries were found (the
    directory is genuinely empty).  Returns the ``_WALK_INCOMPLETE`` sentinel
    when an ``OSError`` prevented the walk from completing — the caller must
    not apply a fallback mtime in that case, since the directory may contain
    fresh files that simply could not be enumerated.

    When ``skip_top_dir_mtime`` is ``True`` the root directory's own mtime
    is excluded from the result.  Pass 2.5 uses this when checking a log dir
    immediately after pass 2 may have deleted its ``tmp/`` subdirectory:
    removing ``tmp/`` bumps ``log_d``'s mtime to now, which would make a
    30-day-old log appear brand-new and defer its collection by another
    ``log_min_age_threshold``.  Ignoring the container's own mtime breaks
    that cycle; children's mtimes still reflect actual content age.

    When ``cutoff`` is given, returns as soon as any entry's mtime exceeds it
    — the exact maximum is irrelevant to the caller once it is known that
    something is newer than ``cutoff``.  At steady state (almost every
    candidate is younger than the reap threshold) this reduces an O(N-entries)
    walk to O(1).  When the directory is genuinely old the full walk runs,
    which is correct — that directory is about to be deleted anyway.

    When ``strict`` is ``True``, any ``scandir`` or ``stat`` failure returns
    ``_WALK_INCOMPLETE`` immediately.  Use this when a partially-observed tree
    must not be treated as old: a recent file in an unreadable subtree would
    otherwise be invisible, making a live worktree appear idle.

    Symlinks are never followed (``entry.stat(follow_symlinks=False)``), so a
    malicious scratch tree cannot redirect the walk outside its log directory.
    """
    try:
        top_st = os.stat(d, follow_symlinks=False)
        if skip_top_dir_mtime:
            newest: Optional[float] = None
        else:
            newest = top_st.st_mtime
            if cutoff is not None and newest > cutoff:
                return newest
    except OSError:
        return _WALK_INCOMPLETE  # type: ignore[return-value]

    walk_error = False
    stack: List[str] = [str(d)]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as it:
                entries = list(it)
        except OSError:
            if strict:
                return _WALK_INCOMPLETE  # type: ignore[return-value]
            walk_error = True
            continue
        for entry in entries:
            try:
                st = entry.stat(follow_symlinks=False)
            except OSError:
                if strict:
                    return _WALK_INCOMPLETE  # type: ignore[return-value]
                continue
            mtime = st.st_mtime
            if newest is None or mtime > newest:
                newest = mtime
            if cutoff is not None and newest > cutoff:
                return newest
            if _stat_module.S_ISDIR(st.st_mode):
                stack.append(entry.path)
    if walk_error and newest is None:
        return _WALK_INCOMPLETE  # type: ignore[return-value]
    return newest


def _dir_size_bytes(d: Path, *, exclude: Optional[Path] = None,
                    excludes: Sequence[Path] = ()) -> int:
    """Apparent size (sum of ``st_size``) of every regular file at or below
    ``d``, in bytes, excluding anything at or below ``exclude`` or ``excludes``.
    The singular argument keeps log/scratch callers concise; the sequence is
    used when several separately charged roots are nested in one worktree.
    Uses lstat (``entry.stat(follow_symlinks=False)``), so neither the walk
    interior nor ``d`` itself can be a symlink that redirects the count
    outside ``d``. Tolerates races (a file vanishing between scan and stat)
    by skipping the entry rather than raising; an unreadable or non-real-
    directory ``d`` returns 0.

    Shared by `reap --include-logs` (per-candidate size in its report line)
    and `du` (per-group/per-run totals), so both report the same notion of
    "size" for a log directory.  Callers that need completeness information
    should use ``_dir_size_bytes_complete``.
    """
    size, _ = _dir_size_bytes_complete(d, exclude=exclude, excludes=excludes)
    return size


def _dir_size_bytes_complete(d: Path, *, exclude: Optional[Path] = None,
                             excludes: Sequence[Path] = ()) -> Tuple[int, bool]:
    """Like ``_dir_size_bytes`` but also returns whether the walk was complete.

    Returns ``(size, complete)`` where ``complete`` is ``True`` when every
    entry was successfully read.  A permission error on the root directory
    returns ``(0, False)``; a permission error on a subtree directory returns
    ``(partial, False)``.  Both cases mean the reported size is a lower bound.
    """
    try:
        top_st = d.lstat()
    except FileNotFoundError:
        # Missing directory is not an error: a run may have no scratch dir.
        return 0, True
    except OSError:
        return 0, False
    if not _stat_module.S_ISDIR(top_st.st_mode):
        return 0, True
    total = 0
    complete = True
    excluded = {p.resolve() for p in excludes}
    if exclude is not None:
        excluded.add(exclude.resolve())
    stack = [d]
    while stack:
        current = stack.pop()
        if current.resolve() in excluded:
            continue
        try:
            entries = list(os.scandir(current))
        except OSError:
            complete = False
            continue
        for entry in entries:
            try:
                if entry.is_dir(follow_symlinks=False):
                    stack.append(Path(entry.path))
                elif entry.is_file(follow_symlinks=False):
                    st = entry.stat(follow_symlinks=False)
                    total += st.st_size
            except OSError:
                complete = False
                continue
    return total, complete


_SIZE_UNITS = ("B", "K", "M", "G", "T", "P")


def _human_size(n: int) -> str:
    """Render a byte count as e.g. ``1.2G``, ``340M``, ``12K``, ``0B`` —
    binary (1024-based) units, one decimal place above the byte unit."""
    size = float(n)
    for unit in _SIZE_UNITS:
        if size < 1024 or unit == _SIZE_UNITS[-1]:
            return f"{n}{unit}" if unit == "B" else f"{size:.1f}{unit}"
        size /= 1024


def _terminal_state_age_seconds(state_dir: Path) -> Optional[float]:
    """How long a conclusively-terminal run has been sitting idle, for the
    reap min-age GC threshold.

    Prefers the parsed ``ended_at`` timestamp (the moment the run actually
    became terminal) over the state dir's mtime, since the dir's mtime can
    be bumped by unrelated later writes (e.g. a subsequent ``reap_reason``
    write) that must not reset the GC clock. Falls back to the newest mtime
    across the state dir's own files if ``ended_at`` is missing or
    unparsable (legacy/corrupt state — a warning is printed so this is
    visible rather than silently switching to a weaker age source), and
    finally to the directory's own mtime if the dir is empty. Returns
    ``None`` only if the dir cannot be stat'd at all (e.g. removed
    concurrently).

    A parsed ``ended_at`` that is in the future (NTP step, clock-skewed
    host, VM snapshot restore) is clamped to age 0 with a warning rather
    than yielding a negative age: fail-safe here (age 0 never clears a
    positive min-age threshold) without an unexplained negative number in
    diagnostic output.
    """
    ended_raw = _read(state_dir / "ended_at")
    if ended_raw:
        try:
            parsed = datetime.strptime(ended_raw, "%Y-%m-%dT%H:%M:%SZ").replace(
                tzinfo=timezone.utc
            )
            age = time.time() - parsed.timestamp()
            if age < 0:
                print(
                    f"agent-run: warning: {state_dir.name} ended_at "
                    f"({ended_raw!r}) is in the future; treating age as 0",
                    file=sys.stderr,
                )
                return 0.0
            return age
        except ValueError:
            print(
                f"agent-run: warning: {state_dir.name} ended_at "
                f"({ended_raw!r}) is malformed; falling back to file mtime "
                "for its GC age",
                file=sys.stderr,
            )
    mtime = _newest_mtime(state_dir)
    if mtime is None:
        return None
    return time.time() - mtime


def _gc_pid_is_live_runner(state_dir: Path, pid: int) -> bool:
    """Is ``pid`` still evidence that the recorded runner is alive, for
    reap's terminal-state GC liveness check?

    Every other process-control site in this module (``cmd_kill``, reap's
    own idle-kill path, ``_force_kill``, ``_send_signal_to_verified_pid``)
    pairs ``_pid_alive`` with a ``process_identity`` check before treating a
    live pid as evidence about a *specific* process. GC previously trusted a
    bare pid alone (M1): after a reboot, pids restart from low numbers and a
    terminal run's recorded pid frequently collides with an unrelated live
    process, producing a permanent, self-repeating false-positive refusal
    that leaks that run's state/scratch forever (the same leak class as H3).

    When ``process_identity`` was recorded and the live process's identity
    can be read and does not match, the pid has been recycled — safe to
    treat as not-the-runner, so GC may proceed. In every other case (no
    identity recorded — legacy state; identity unreadable; identity
    matches) this conservatively reports "still live", preserving the
    fail-safe direction (a spurious skip, never a spurious delete).
    """
    recorded = _read(state_dir / "process_identity")
    if not recorded:
        return True  # legacy state, no identity to compare — stay conservative
    live = _process_identity(pid)
    if live is None:
        return True  # can't verify — stay conservative
    return live == recorded


def _gc_status_eligible(status: str, *, force_unknown: bool) -> bool:
    """Whether a persisted status is eligible for terminal-state GC."""
    return status in TERMINAL_STATUSES or (
        force_unknown and _status_bucket(status) == "unrecognized"
    )


# Sentinel returned by _gc_live_runner_pid for "pid file unreadable" —
# never a real pid (pids are positive), so callers can tell it apart from
# both a verified-live pid and None ("safe to GC").
PID_UNREADABLE = -1


def _gc_live_runner_pid(state_dir: Path) -> Optional[int]:
    """Return the recorded runner PID if it still blocks GC, or ``None`` if
    the run is safe to garbage-collect.

    The recorded pid itself is returned when it is verified alive; when the
    pid file cannot be read at all, ``PID_UNREADABLE`` is returned instead —
    a distinct non-``None`` value so callers refuse GC the same way they
    would for a confirmed-live pid, without claiming to know which pid.

    GC's fail-safe direction is a spurious skip, never a spurious delete —
    the opposite of watch/status's — so the pid file is read strictly here
    rather than through the lenient shared `_read`: an unreadable pid file
    (mode 000, a directory in its place) must block GC exactly like a
    confirmed-live pid, not read as "no pid" and let GC proceed.
    """
    try:
        raw = (state_dir / "pid").read_text().strip()
    except FileNotFoundError:
        return None
    except (OSError, UnicodeError):
        return PID_UNREADABLE
    pid = _safe_int(raw)
    if pid is not None and _pid_alive(pid) and _gc_pid_is_live_runner(state_dir, pid):
        return pid
    return None


def _effective_status(state_dir: Path, idle_threshold: Optional[float] = None) -> str:
    """Compute a display status for a run — read-only, no mutations.

    Rules (applied when the raw status is "running" or "starting"):
    - "starting" with no pid published yet → "starting" (normal transient
      window before the detached runner takes over; not evidence of death)
    - pid missing (raw == "running") or dead → "died"
    - pid alive but log idle > threshold (raw == "running") → "stalled"
    - otherwise                              → raw

    ``idle_threshold`` defaults to the module-level ``IDLE_STALL_SECONDS``
    (which is itself overridable via ``AGENT_RUN_IDLE_KILL_HOURS``).
    """
    raw = _read(state_dir / "status", "unknown")
    if raw not in {"running", "starting"}:
        return raw

    # Check pid liveness.
    pid_raw = _read(state_dir / "pid")
    if not pid_raw:
        # A "starting" run has not published a pid yet -- that's a normal
        # transient window, not evidence of death.
        return raw if raw == "starting" else "died"
    try:
        pid = int(pid_raw)
    except ValueError:
        return "died"
    if not _pid_alive(pid):
        return "died"

    if raw == "starting":
        return "starting"

    # raw == "running" and pid is alive — check log freshness.
    if idle_threshold is None:
        idle_threshold = IDLE_STALL_SECONDS
    log = _log_file_for(state_dir.name)
    if log is not None:
        try:
            mtime = log.stat().st_mtime
            if time.time() - mtime > idle_threshold:
                return "stalled"
        except OSError:
            pass

    return "running"


def _watch_effective_status(state_dir: Path, idle_threshold: Optional[float] = None) -> str:
    """`watch`'s liveness resolution: pid-identity verification layered on
    top of ``_effective_status``'s raw/pid/idle result, leaving that shared
    function's behaviour for ``cmd_status``/``cmd_logs``/``cmd_list`` alone.

    When the resolved status is "running" or "stalled" (both imply an alive
    pid), the live process's identity is compared against the token recorded
    at launch: a mismatch means the pid was recycled by an unrelated process
    and the run is reported "died"; a live pid whose identity cannot be
    confirmed at all (no token recorded, or the token is unreadable) is
    reported "unverified" rather than "running".

    ``_gc_pid_is_live_runner`` treats that same unverifiable case as
    conservatively still-live, because GC's safe direction is never deleting
    state out from under a run it can't rule out. `watch`'s safe direction is
    the opposite — an unconfirmable pid must never be reported as a healthy
    "running" run — so the two intentionally diverge on this case.

    A zombie pid is rejected as "died" before the identity comparison even
    runs: a zombie's pid and birth token both survive until reaped, so a
    matching identity is not evidence of liveness here.
    """
    status = _effective_status(state_dir, idle_threshold)
    if status not in {"starting", "running", "stalled"}:
        return status
    pid = _safe_int(_read(state_dir / "pid"))
    if pid is None:
        return "unverified"
    if _watch_pid_is_zombie(pid):
        return "died"
    recorded = _read(state_dir / "process_identity")
    if not recorded:
        return "unverified"
    live = _process_identity(pid)
    if live is None:
        return "unverified"
    if live != recorded:
        return "died"
    return status


# Status string emitted by `watch` when the state dir is gone but the log
# survived. Not a member of TERMINAL_STATUSES: that set is reap's GC
# vocabulary, and this string is watch-only.
WATCH_STATUS_LOG_PRESERVED = "not running (log preserved)"


def _watch_is_terminal(status: str) -> bool:
    """Single source of truth for the `terminal` field in the `watch`
    payload, so it cannot drift from `status`.

    Deliberately watch-local rather than `status in TERMINAL_STATUSES`:
    that set is reap's GC vocabulary and must stay narrow so reap never
    implicitly deletes state for an unrecognized status. `watch` has the
    opposite obligation — a poller must never wait forever on a status that
    will never change — so anything not known to be pending, including
    "unknown" and any other unrecognized string, is terminal here even
    though reap still refuses to touch it.
    """
    return status not in (KNOWN_NONTERMINAL_STATUSES | {"unverified"})


def _opportunistic_heal(state_root: Optional[Path] = None) -> None:
    """Mark clearly-dead (pid gone) "running"/"starting" runs as "died".

    Deliberately NEVER idle-kills — that is only done by ``cmd_reap``.
    Called from read commands (``cmd_list``) and launch so stale state
    from crashed runs gets cleaned up passively, just like ``_prune_old_logs``.
    """
    root = state_root if state_root is not None else STATE_ROOT
    if not root.is_dir():
        return
    try:
        candidates = list(root.iterdir())
    except OSError:
        return
    for d in candidates:
        if not d.is_dir():
            continue
        try:
            raw = _read(d / "status")
            if raw not in {"running", "starting"}:
                continue
            pid_raw = _read(d / "pid")
            if not pid_raw:
                if raw == "starting":
                    # Pid not published yet -- normal transient window before
                    # the detached runner takes over; not evidence of death.
                    continue
                # "running" with no pid recorded → mark died.
                _mark_died(d, "no pid recorded")
                continue
            try:
                pid = int(pid_raw)
            except ValueError:
                # Unparseable pid → mark died so it doesn't stay running forever.
                _mark_died(d, f"invalid pid: {pid_raw!r}")
                continue
            if not _pid_alive(pid):
                _mark_died(d, f"pid {pid} no longer alive")
        except OSError:
            continue


def _pretty_command(argv: Sequence[str]) -> str:
    return " ".join(shlex.quote(a) for a in argv)


def _submit_mode_for_argv(argv: Sequence[str]) -> str:
    """Select CRLF only when the unambiguous wrapped executable is OpenCode.

    ``env -S`` uses a small BSD/GNU-compatible lexer: whitespace separates
    words, quotes group words, backslash quotes the next character, and ``\_``
    represents a literal space.  Expansion has one budget for the complete
    wrapper walk, including nested ``env`` invocations.
    """
    def split_env_string(value: str) -> Optional[list[str]]:
        words: list[str] = []
        word: list[str] = []
        quote: Optional[str] = None
        escaped = False
        for char in value:
            if escaped:
                word.append(" " if char == "_" else char)
                escaped = False
            elif char == "\\":
                escaped = True
            elif quote:
                if char == quote:
                    quote = None
                else:
                    word.append(char)
            elif char in {"'", '"'}:
                quote = char
            elif char.isspace():
                if word:
                    words.append("".join(word))
                    word = []
            else:
                word.append(char)
        if escaped or quote:
            return None
        if word:
            words.append("".join(word))
        return words

    budget = 8
    command = list(argv)
    while command:
        executable = Path(command[0]).name
        if executable == "opencode":
            return SUBMIT_MODE_CRLF
        if executable in {"command", "exec"}:
            command = command[1:]
            continue
        if executable != "env":
            break
        command = command[1:]
        while command:
            argument = command[0]
            if argument == "--":
                command = command[1:]
                break
            # BSD env accepts combined short flags (notably -iS).  -S consumes
            # either its suffix or the next argv element as its split string.
            if argument.startswith("-") and not argument.startswith("--") and len(argument) > 2 and "S" in argument[1:]:
                before, split_value = argument[1:].split("S", 1)
                if any(flag not in {"i", "v"} for flag in before):
                    break
                if split_value:
                    command = ["-S", split_value] + command[1:]
                else:
                    command = ["-S"] + command[1:]
                continue
            if argument in {"-S", "--split-string"} or argument.startswith("--split-string="):
                if argument.startswith("--split-string="):
                    value, rest = argument.split("=", 1)[1], command[1:]
                elif len(command) >= 2:
                    value, rest = command[1], command[2:]
                else:
                    return SUBMIT_MODE_CR
                budget -= 1
                tokens = split_env_string(value)
                if budget < 0 or tokens is None:
                    return SUBMIT_MODE_CR
                command = tokens + rest
                continue
            if argument in {"-i", "--ignore-environment"} or (not argument.startswith("-") and "=" in argument):
                command = command[1:]
                continue
            if argument in {"-u", "--unset", "-C", "--chdir", "-P", "--argv0"}:
                command = command[2:] if len(command) >= 2 else []
                continue
            if argument.startswith(("--unset=", "--chdir=", "--argv0=")) or (len(argument) > 2 and argument[:2] in {"-u", "-C", "-P"}):
                command = command[1:]
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
    except (FileNotFoundError, OSError, UnicodeError):
        return 0


# Valid tokens for `list --status`: the recognized terminal/non-terminal
# statuses, plus the literal "unknown" _effective_status can return for a
# missing/empty status file. Deliberately does NOT include arbitrary legacy
# values like "dead" — see KNOWN_NONTERMINAL_STATUSES's docstring; those are
# unrecognized by design and always surface in the separate "Unrecognized"
# section below rather than being individually targetable by a typo-prone
# free-text token.
_VALID_LIST_STATUS_TOKENS = TERMINAL_STATUSES | KNOWN_NONTERMINAL_STATUSES | {"unknown"}

_ENV_BOOL_TRUE = {"1", "true", "yes"}
_ENV_BOOL_FALSE = {"0", "false", "no", ""}


def _env_bool_default(env_name: str) -> bool:
    """Parse an opt-in boolean env var (``1``/``true``/``yes``, case-
    insensitive; ``0``/``false``/``no``/unset for off), warning to stderr
    on an unrecognized value and falling back to False."""
    raw = os.environ.get(env_name)
    if raw is None:
        return False
    normalized = raw.strip().lower()
    if normalized in _ENV_BOOL_TRUE:
        return True
    if normalized in _ENV_BOOL_FALSE:
        return False
    print(
        f"agent-run: warning: {env_name}={raw!r} is not one of "
        "1/true/yes/0/false/no; using default",
        file=sys.stderr,
    )
    return False


def _real_subdir_names(root: Path) -> set:
    """Names of ``root``'s real subdirectories, excluding dot-prefixed names.

    ``lstat`` rather than ``is_dir()`` so a symlink is never followed and can
    never present itself as a run. An entry that cannot be lstat'd (raced away)
    is skipped; an ``OSError`` from the iteration itself propagates, so callers
    that must tolerate an unreadable root wrap the call.
    """
    names = set()
    for p in root.iterdir():
        if p.name.startswith("."):
            continue
        try:
            st = p.lstat()
        except OSError:
            continue
        if _stat_module.S_ISDIR(st.st_mode):
            names.add(p.name)
    return names


def cmd_list(args: argparse.Namespace) -> int:
    _prune_old_logs()
    _opportunistic_heal()

    show_all: bool = bool(getattr(args, "all", False))
    status_filter_raw: Optional[str] = getattr(args, "status", None)
    # Explicit --include-logs always wins over the env default; the flag's
    # own default is False, so falsy-but-explicit and unset are
    # indistinguishable here — acceptable since the env var only ever raises
    # the effective default, never lowers an explicit flag.
    include_logs: bool = bool(getattr(args, "include_logs", False)) or _env_bool_default(
        "AGENT_RUN_LIST_INCLUDE_LOGS"
    )
    if not show_all and status_filter_raw is None:
        list_default = os.environ.get("AGENT_RUN_LIST_DEFAULT", "live").strip().lower()
        if list_default not in {"live", "all"}:
            print(
                f"agent-run: warning: AGENT_RUN_LIST_DEFAULT={list_default!r} "
                "must be 'live' or 'all'; using live",
                file=sys.stderr,
            )
        else:
            show_all = list_default == "all"
    status_filter: Optional[set] = None
    if status_filter_raw:
        if show_all:
            sys.exit("agent-run: --status and --all are mutually exclusive")
        status_filter = {s.strip() for s in status_filter_raw.split(",") if s.strip()}
        if not status_filter:
            sys.exit("agent-run: --status requires at least one status")
        unrecognized_tokens = status_filter - _VALID_LIST_STATUS_TOKENS
        if unrecognized_tokens:
            sys.exit(
                "agent-run: --status: unrecognized value(s) "
                f"{','.join(sorted(unrecognized_tokens))}; valid values: "
                f"{','.join(sorted(_VALID_LIST_STATUS_TOKENS))}"
            )

    def _visible(status: str, bucket: str) -> bool:
        if status_filter is not None:
            return status in status_filter
        if show_all:
            return bucket != "unrecognized"
        # Default: show only live (starting/running/stalled) runs — "Live
        # runs" should actually mean live. Conclusively terminal runs are
        # hidden (pass --all or --status to see them); unrecognized statuses
        # are never folded in here either way — they always get their own
        # section below so they can't silently masquerade as live (see M4)
        # or silently vanish.
        return bucket == "live"

    if status_filter is not None:
        heading = f"Runs (status={','.join(sorted(status_filter))}) ({STATE_ROOT}):"
    elif show_all:
        heading = f"All runs ({STATE_ROOT}):"
    else:
        heading = f"Live runs, non-terminal only ({STATE_ROOT}):"
        # The "how to see more" hint goes to stderr, not stdout: it used to
        # be baked into the stdout heading, which meant `agent-run list |
        # grep killed` (or `grep done`) got a false-positive match on the
        # literal word "killed"/"done" in the heading even when the run set
        # was completely empty. Keeping status tokens out of stdout headings
        # entirely keeps scripted greps honest.
        print(
            "agent-run: pass --all to include done/failed/launch_failed/died/killed",
            file=sys.stderr,
        )

    state_names = set()
    print(heading)
    if STATE_ROOT.is_dir():
        state_names = _real_subdir_names(STATE_ROOT)
    shown = 0
    shown_names = set()
    unrecognized_rows: List[tuple] = []
    if state_names:
        for d in sorted(_state_dir(n) for n in state_names):
            status = _effective_status(d)
            bucket = _status_bucket(status)
            if bucket == "unrecognized":
                unrecognized_rows.append((d, status))
            if not _visible(status, bucket):
                continue
            shown += 1
            shown_names.add(d.name)
            pid = _read(d / "pid", "?")
            started = _read(d / "started_at", "?")
            lines = _log_line_count(_log_file_for(d.name))
            interactive = _read(d / "interactive", "0")
            flag = " [interactive]" if interactive == "1" else ""
            print(f"  {d.name}: status={status} pid={pid} started={started} lines={lines}{flag}")
    if shown == 0:
        print("  (none)")

    # Unrecognized statuses (missing/empty status file, legacy values like
    # "dead" that nothing in this repo writes, corrupt state) get their own
    # honest heading instead of being silently hidden or shown as "live" —
    # see M4. Always printed when non-empty, regardless of --all/--status,
    # the same way "Preserved logs" below always is; skip a name already
    # printed above (only possible if the caller explicitly matched it via
    # --status unknown).
    unrecognized_to_show = [
        (d, status) for d, status in unrecognized_rows if d.name not in shown_names
    ]
    if unrecognized_to_show:
        print(f"Unrecognized / needs attention ({STATE_ROOT}):")
        for d, status in sorted(unrecognized_to_show):
            pid = _read(d / "pid", "?")
            started = _read(d / "started_at", "?")
            lines = _log_line_count(_log_file_for(d.name))
            print(f"  {d.name}: status={status} pid={pid} started={started} lines={lines}")

    log_only_names = set()
    if LOG_ROOT.is_dir():
        log_only_names = _real_subdir_names(LOG_ROOT) - state_names
    if include_logs:
        if log_only_names:
            # Preserved-log-only runs (state dir already gone) are already
            # unambiguously and correctly labeled as "not running" under
            # their own heading. Shown only when requested — see the hint
            # printed below when they're hidden.
            print(f"Preserved logs, not running ({LOG_ROOT}):")
            for name in sorted(log_only_names):
                lines = _log_line_count(_log_file_for(name))
                print(f"  {name}: lines={lines}")
    elif log_only_names:
        # Hint goes to stderr, same reasoning as the --all hint above: a
        # count/word in a stdout heading would give `list | grep` a
        # false-positive match even when nothing was actually shown.
        print(
            f"agent-run: {len(log_only_names)} preserved log(s) hidden; "
            "pass --include-logs to show them",
            file=sys.stderr,
        )
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    name = _validate_run_name(args.name)
    state_dir = _state_dir(name)
    log_dir = _log_dir(name)
    if not _is_dir_safe(state_dir) and not _is_dir_safe(log_dir):
        sys.exit(f"agent-run: no run named '{name}' in {STATE_ROOT} or {LOG_ROOT}")
    lines = _log_line_count(_log_file_for(name))
    if not _is_dir_safe(state_dir):
        # The state dir is volatile; a launch failure recorded alongside the
        # log outlives it and is exactly what a post-reboot status needs.
        launch_error = _read(_log_dir(name) / "launch_error")
        suffix = f" launch_error={launch_error!r}" if launch_error else ""
        print(f"name={name} status=not running (log preserved) lines={lines}{suffix}")
        return 0
    d = state_dir
    status = _effective_status(d)
    pid = _read(d / "pid", "?")
    started = _read(d / "started_at", "?")
    ended = _read(d / "ended_at", "-")
    exit_code = _read(d / "exit_code", "-")
    interactive = _read(d / "interactive", "0")
    launch_error = _read(d / "launch_error") or _read(_log_dir(name) / "launch_error")
    suffix = f" launch_error={launch_error!r}" if launch_error else ""
    worktree_created = _read(d / "worktree_created")
    if worktree_created:
        suffix += f" worktree_created={worktree_created!r}"
    print(
        f"name={name} status={status} pid={pid} exit={exit_code} "
        f"started={started} ended={ended} lines={lines} interactive={interactive}"
        f"{suffix}"
    )
    return 0


def _watch_parse_iso(raw: Optional[str]) -> Optional[datetime]:
    if not raw:
        return None
    try:
        return datetime.strptime(raw, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


class _WatchGitOutcome(NamedTuple):
    """Result of one bounded git command, including operator-visible failure detail."""

    stdout: Optional[str]
    error: Optional[str]
    stderr: Optional[str] = None

    @property
    def error_detail(self) -> str:
        if self.stderr:
            return f"{self.error}: {self.stderr.strip()}"
        return self.error or "git_failed"


# "No commits yet" has no single stable stderr wording across the git
# commands/versions this module runs, so that case is instead disambiguated
# with a follow-up refs probe (see ``_watch_git_facts_checked``).
_WATCH_GIT_NOT_A_REPO_STDERR = "not a git repository"


def _watch_run_git_checked(
    repo: Path,
    git_args: Sequence[str],
    timeout: float = WATCH_GIT_SUBPROCESS_TIMEOUT_SECONDS,
) -> _WatchGitOutcome:
    """Run one bounded, lock-free git read, reporting *why* a failed read
    failed rather than collapsing every cause to ``None``."""
    # Strip inherited GIT_* pointers so the query can't be silently redirected
    # at a different repo/index, and block config that runs external programs.
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in WATCH_GIT_ENV_VARS_TO_STRIP and not k.startswith("GIT_CONFIG")
    }
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    env["GIT_CONFIG_GLOBAL"] = "/dev/null"
    env["GIT_NO_REPLACE_OBJECTS"] = "1"
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "--no-optional-locks",
                "-c",
                "core.fsmonitor=false",
                "-c",
                "core.hooksPath=/dev/null",
                "-c",
                "diff.external=",
                "-c",
                "filter.allowProcess=false",
                "-c",
                "filter.required=false",
                "-c",
                "core.pager=cat",
                *git_args,
            ],
            capture_output=True,
            timeout=timeout,
            check=True,
            env=env,
            stdin=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        return _WatchGitOutcome(None, "git_missing")
    except subprocess.TimeoutExpired:
        return _WatchGitOutcome(None, "timeout")
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or b"").decode("utf-8", errors="replace")
        if _WATCH_GIT_NOT_A_REPO_STDERR in stderr.lower():
            return _WatchGitOutcome(None, "not_a_repo", stderr)
        return _WatchGitOutcome(None, "git_failed", stderr)
    except (OSError, subprocess.SubprocessError) as exc:
        return _WatchGitOutcome(None, "git_failed", str(exc))
    return _WatchGitOutcome(result.stdout.decode("utf-8", errors="surrogateescape"), None)


_WATCH_SHORTSTAT_RES = (
    re.compile(r"(\d+) files? changed"),
    re.compile(r"(\d+) insertions?\(\+\)"),
    re.compile(r"(\d+) deletions?\(-\)"),
)


def _watch_parse_shortstat(text: str) -> tuple[int, int, int]:
    """``files_changed, insertions, deletions`` from ``diff --shortstat``
    output; each counter absent from the text is 0."""
    counts = tuple(
        int(m.group(1)) if (m := pattern.search(text)) else 0
        for pattern in _WATCH_SHORTSTAT_RES
    )
    files, insertions, deletions = counts
    return files, insertions, deletions


class _WatchGitFactsResult(NamedTuple):
    """``facts`` and ``git_error`` are never both set: the watch contract's
    top-level ``git_error`` discriminator must say *why* ``git`` is null
    rather than collapsing "cannot observe" and "observed nothing" into the
    same null the consumer already treats as "no facts"."""

    facts: Optional[dict]
    git_error: Optional[str]


def _watch_git_facts_checked(repo: Path, launch_head: Optional[str]) -> _WatchGitFactsResult:
    """Read-only git facts for ``repo``, plus why they're missing on failure
    — not a git repo, git not installed, a wedged command hitting the
    bounded timeout, a brand-new repo with no commits yet, or a repo mid-
    rebase/mid-merge with an unreadable index — never raises. Reads only,
    and ``--no-optional-locks`` keeps it from contending with an agent's own
    git operations.

    ``commits_since_start`` is ``rev-list <launch_head>..HEAD --count``:
    which commit HEAD is, not when git says it was committed, since the
    committing process controls those timestamps and can backdate them. It
    is null, not 0, when ``launch_head`` itself is unknown (``cwd`` wasn't a
    git repo at launch, or it is a legacy run without the field) — the count
    is genuinely undetermined, not zero.

    ``untracked_files`` counts working-tree files git doesn't track yet
    (``status --porcelain``'s ``??`` entries) — real agent output the
    tracked-file-only diff counters would otherwise miss. Ignored files are
    excluded from every counter: an agent's own build/dependency output is
    not progress, and ``.gitignore`` states what that output is.

    ``toplevel`` is the work-tree root git resolved for this observation, or
    ``None`` if that call failed. It may be an **enclosing** repo when the
    observed path is not itself a repository — the consumer compares it
    against the repo it expects.
    """
    if not repo.is_dir():
        return _WatchGitFactsResult(None, "no_repo_path")

    deadline = time.monotonic() + WATCH_GIT_TOTAL_BUDGET_SECONDS

    def run_git(git_args: Sequence[str]) -> _WatchGitOutcome:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return _WatchGitOutcome(None, "timeout")
        return _watch_run_git_checked(
            repo, git_args, timeout=min(WATCH_GIT_SUBPROCESS_TIMEOUT_SECONDS, max(0.05, remaining))
        )

    head_outcome = run_git(["rev-parse", "HEAD"])
    if head_outcome.stdout is None:
        # A non-zero exit is ambiguous between "not a repo" (already
        # classified) and "repo exists, zero commits yet", and git's stderr
        # wording for the latter isn't stable across commands/versions. An
        # unborn HEAD and a dangling/corrupt HEAD are both unresolvable, but
        # only the former has no refs at all, so ``no_commits`` requires
        # that absence — otherwise repo damage would read as the one
        # discriminator value callers treat as benign.
        if head_outcome.error == "git_failed":
            refs_outcome = run_git(["rev-list", "--all", "--count"])
            if refs_outcome.stdout is not None and refs_outcome.stdout.strip() == "0":
                return _WatchGitFactsResult(None, "no_commits")
        return _WatchGitFactsResult(None, head_outcome.error)
    head_oid = head_outcome.stdout.strip()

    # Mandatory: the stat-cache-driven reads below (status, diff) can skip
    # re-reading a tracked file's blob when its mtime still matches the
    # index, so a missing or corrupt object below HEAD would otherwise go
    # undetected and this function would return fully populated facts.
    fsck_outcome = run_git(["fsck", "--connectivity-only", "--no-progress"])
    if fsck_outcome.stdout is None:
        return _WatchGitFactsResult(None, fsck_outcome.error)

    porcelain_outcome = run_git(["status", "--porcelain", "-z", "--untracked-files=all"])
    # --untracked-files=all disables git's directory-collapse optimisation, so a
    # large untracked tree (node_modules/ not yet gitignored) can expand output
    # ~10x inside the shared git budget; the failure mode is git_error "timeout".
    if porcelain_outcome.stdout is None:
        return _WatchGitFactsResult(None, porcelain_outcome.error)
    porcelain = porcelain_outcome.stdout
    # A rename/copy (`R`/`C` in XY) emits two NUL-separated records: the status
    # record, then a bare origin path. Consuming the origin path keeps a path
    # literally starting with "?? " from being counted as untracked.
    records = [r for r in porcelain.split("\0") if r]
    untracked_files = 0
    skip_next = False
    for record in records:
        if skip_next:
            skip_next = False
            continue
        xy = record[:2] if len(record) >= 2 else ""
        if xy[0:1] in ("R", "C"):
            skip_next = True
        if record.startswith("?? "):
            untracked_files += 1
    shortstat_outcome = run_git(["diff", head_oid, "--shortstat"])
    if shortstat_outcome.stdout is None:
        return _WatchGitFactsResult(None, shortstat_outcome.error)
    files_changed, insertions, deletions = _watch_parse_shortstat(shortstat_outcome.stdout)

    commits_since_start: Optional[int] = None
    if launch_head is not None:
        count_outcome = run_git(["rev-list", "--count", f"{launch_head}..{head_oid}"])
        if count_outcome.stdout is None:
            return _WatchGitFactsResult(None, count_outcome.error)
        try:
            commits_since_start = int(count_outcome.stdout.strip())
        except ValueError:
            return _WatchGitFactsResult(None, "git_failed")

    last_commit_outcome = run_git(["log", "-1", "--format=%ct", head_oid])
    if last_commit_outcome.stdout is None:
        return _WatchGitFactsResult(None, last_commit_outcome.error)
    last_commit_age_s: Optional[float] = None
    stripped = last_commit_outcome.stdout.strip()
    if stripped:
        try:
            last_commit_age_s = max(0.0, time.time() - int(stripped))
        except ValueError:
            last_commit_age_s = None

    # Reported, not enforced: git discovery walks upward to the nearest
    # enclosing repo, so a legitimate subdirectory of its own repo and a
    # plain directory nested inside an unrelated repo are indistinguishable
    # by path alone. The consumer knows which repo a run belongs to.
    toplevel: Optional[str] = None
    toplevel_outcome = run_git(["rev-parse", "--show-toplevel"])
    if toplevel_outcome.stdout is not None:
        toplevel = str(Path(toplevel_outcome.stdout.strip()).resolve())

    final_head = run_git(["rev-parse", "HEAD"])
    if final_head.stdout is None:
        return _WatchGitFactsResult(None, final_head.error)
    if final_head.stdout.strip() != head_oid:
        return _WatchGitFactsResult(None, "changed_during_observation")

    return _WatchGitFactsResult(
        {
            "head": head_oid[:7],
            "dirty": bool(porcelain.strip()),
            "files_changed": files_changed,
            "insertions": insertions,
            "deletions": deletions,
            "untracked_files": untracked_files,
            "commits_since_start": commits_since_start,
            "last_commit_age_s": last_commit_age_s,
            "toplevel": toplevel,
        },
        None,
    )


def _watch_normalize_lines(lines: List[str]) -> List[str]:
    """ANSI-strip each tail line. The source is a raw PTY capture of a TUI,
    so colour codes are otherwise indistinguishable from real content to a
    line-oriented scan. Same-line ``\\r`` redraws are already separate lines
    here: the tail scan splits with ``str.splitlines()``, which breaks on CR
    as well as LF."""
    return [_ANSI_ESCAPE_RE.sub("", line) for line in lines]


def _watch_top_repeated(counts: dict) -> Optional[tuple[str, int]]:
    """The most-repeated key and its count, or ``None`` unless it recurs at
    least ``WATCH_REPEAT_THRESHOLD`` times. Shared by both "repeated"
    signals so they answer the same question rather than one picking
    encounter order and the other the loudest."""
    top = max(counts, key=lambda key: counts[key], default=None)
    if top is None or counts[top] < WATCH_REPEAT_THRESHOLD:
        return None
    return top, counts[top]


def _watch_repeated_error(lines: List[str]) -> Optional[dict]:
    """The most-repeated normalized tail line matching an error signature,
    as ``{"line": ..., "count": ...}`` with the line truncated to
    ``WATCH_ERROR_LINE_MAX_CHARS``, or ``None``. A description of what was
    observed, not a verdict."""
    counts: dict = {}
    for line in lines:
        trimmed = line.strip()
        if not _WATCH_ERROR_LINE_RE.match(trimmed):
            continue
        if _WATCH_ERROR_ZERO_COUNT_RE.search(trimmed):
            continue
        counts[line] = counts.get(line, 0) + 1
    top = _watch_top_repeated(counts)
    if top is None:
        return None
    return {"line": top[0][:WATCH_ERROR_LINE_MAX_CHARS], "count": top[1]}


def _watch_read_signals(lines: List[str]) -> tuple[int, Optional[dict]]:
    """Distinct ``Read <path>`` targets in the normalized tail, and the
    most-repeated one."""
    counts: dict = {}
    for line in lines:
        m = _WATCH_READ_LINE_RE.search(line)
        if not m:
            continue
        path = m.group(1)
        if not _WATCH_READ_PATH_LIKE_RE.search(path):
            continue
        counts[path] = counts.get(path, 0) + 1
    top = _watch_top_repeated(counts)
    return len(counts), None if top is None else {"path": top[0], "count": top[1]}


def _watch_validate_repo_arg(repo_arg: Optional[str]) -> Optional[str]:
    """Validate and resolve an explicit ``--repo`` value before it is ever
    used to locate git facts.

    An empty/whitespace-only value is rejected rather than treated as
    absent: ``--repo ""`` is a caller mistake (an unset shell variable
    interpolated blank), and falling back to the run's recorded launch
    ``cwd`` would hide it. Rejecting mutates nothing — this runs before any
    file is touched.

    Resolved so a relative ``--repo`` is echoed back as the absolute path
    git actually read, matching the always-absolute recorded launch
    ``cwd``."""
    if repo_arg is None:
        return None
    if not repo_arg.strip():
        sys.exit("agent-run: --repo must not be empty or whitespace-only")
    return str(Path(repo_arg).resolve())


# ---------------------------------------------------------------------------
# agent-run hook — harness hook receiver
# ---------------------------------------------------------------------------

# Hook events retained per run. Further events are dropped. Advisory under
# concurrency: N racing writers may overshoot by up to N records. A missing,
# non-integer, or non-positive AGENT_RUN_HOOK_MAX_EVENTS falls back to 1000 —
# the hook must not fail to start over a malformed environment.
_HOOK_MAX_EVENTS_ENV: Optional[int] = _safe_int(
    os.environ.get("AGENT_RUN_HOOK_MAX_EVENTS", "")
)
# Upper bound on the retained-event cap. The cap check reads
# (cap + 1) * _HOOK_MAX_LINE_BYTES bytes, so an unclamped value makes that size
# unrepresentable and the read raises where cmd_hook's mandatory success would
# hide it. At this limit the window is ~8 GiB, far beyond any real hook volume.
_HOOK_MAX_EVENTS_LIMIT: int = 1_000_000

AGENT_RUN_HOOK_MAX_EVENTS: int = (
    min(_HOOK_MAX_EVENTS_ENV, _HOOK_MAX_EVENTS_LIMIT)
    if _HOOK_MAX_EVENTS_ENV and _HOOK_MAX_EVENTS_ENV > 0
    else 1000
)

# Stdin payload cap. More than this is transcript content, not a signal.
_HOOK_STDIN_MAX_BYTES: int = 64 * 1024

# Wall-clock ceiling on the stdin read, so a harness that spawns the hook with
# stdin held open by a silent writer still gets its turn back.
_HOOK_STDIN_TIMEOUT_SECONDS: float = 2.0

# A single os.write() to an O_APPEND fd on a local regular file is atomic
# against other appenders regardless of size: POSIX makes the offset-update
# and the write one operation for O_APPEND, so no writer can observe or
# produce a torn line. PIPE_BUF (512 on macOS, 4096 on Linux) governs pipes
# and FIFOs, not regular files, and does not bound this. Measured directly:
# 40 concurrent processes x 25 lines each, every line checked for exact
# length and writer identity, 0 torn lines at 512, 4096, 8192 and 16384
# bytes. This guarantee is specific to local filesystems — NFS does not
# make O_APPEND atomic, so AGENT_RUN_LOG_DIR must not be a network mount.
_HOOK_MAX_LINE_BYTES: int = 8192

# Event name cap, in JSON-encoded bytes rather than characters: ensure_ascii
# escapes one astral char to 12 bytes, so a character cap bounds nothing.
_HOOK_EVENT_MAX_BYTES: int = 64

# Read ceiling for hooks.jsonl. Beyond this the tail is ignored rather than
# buffered, so watch stays bounded on a file an agent has been appending to.
# Deliberately not equal to AGENT_RUN_HOOK_MAX_EVENTS * _HOOK_MAX_LINE_BYTES
# (1000 * 8192 = 8 MB) so a raised event cap does not silently collide with
# this bound.
_HOOKS_READ_MAX_BYTES: int = 32 * 1024 * 1024

# Per-field payload cap, so one event cannot become a transcript store.
# Payload fields are context, not the point of the record — message is.
_HOOK_PAYLOAD_TRUNCATE_CHARS: int = 200

# The assistant's last turn is what makes a wake actionable; it gets its own,
# much larger budget than payload fields and is clipped last, not uniformly.
_HOOK_MESSAGE_MAX_CHARS: int = 4096

# Ancestry walk depth. A hook runs as a grandchild of the agent at worst;
# 10 hops covers wrapper shells without walking to init on a broken ppid chain.
_HOOK_ANCESTRY_MAX_HOPS: int = 10


def _hook_resolve_run(name_override: Optional[str]) -> Optional[tuple[str, str]]:
    """Return (run_name, resolved_by) for this hook invocation, or None.

    Tries --name, then AGENT_RUN_NAME, then the caller's process ancestry; the
    name must be valid and have an existing log dir. An explicitly-set but
    unresolvable name returns None rather than falling through to ancestry, so
    a stale AGENT_RUN_NAME cannot misattribute events to a neighbouring run.
    Never raises.
    """
    if name_override:
        return _hook_check_name(name_override, "name")

    env_name = os.environ.get("AGENT_RUN_NAME", "").strip()
    if env_name:
        return _hook_check_name(env_name, "env")

    ancestors = _hook_ancestor_pids()
    if not ancestors:
        return None

    # setsid makes the runner its own session leader: sid == pgid == pid, so a
    # hook in the run's session matches on pgid even with no ppid link left.
    try:
        my_sid: Optional[int] = os.getsid(0)
    except OSError:
        my_sid = None

    try:
        names = _real_subdir_names(STATE_ROOT)
    except OSError:
        return None

    sid_matches: list[str] = []
    for name in sorted(names):
        if _hook_check_name(name, "ancestry") is None:
            continue
        entry = STATE_ROOT / name
        pids = {
            _safe_int(_read(entry / f))
            for f in ("pid", "pgid", "pty_pid")
        }
        if pids & ancestors:
            return name, "ancestry"  # direct ancestry beats a session match
        if my_sid is not None and _safe_int(_read(entry / "pgid")) == my_sid:
            sid_matches.append(name)

    # Two runs in one session cannot be told apart; attribute to neither.
    if len(sid_matches) == 1:
        return sid_matches[0], "ancestry"
    return None


def _hook_check_name(name: str, resolved_by: str) -> Optional[tuple[str, str]]:
    """Return (name, resolved_by) if name is a valid run with a log dir."""
    try:
        _validate_run_name(name)
    except SystemExit:
        return None
    return (name, resolved_by) if _log_dir(name).is_dir() else None


def _hook_ancestor_pids() -> set[int]:
    """Pids of this process's ancestors, up to _HOOK_ANCESTRY_MAX_HOPS hops.

    Reads /proc on Linux and falls back to `ps` elsewhere. Stops at pid 1 or at
    the first unreadable link; a partial chain still resolves nearer ancestors.
    """
    pids: set[int] = set()
    pid = os.getppid()
    for _ in range(_HOOK_ANCESTRY_MAX_HOPS):
        if pid <= 1:
            break
        pids.add(pid)
        fields = _proc_stat_fields(pid)
        parent = _safe_int(fields[1]) if fields else _safe_int(_ps_field(pid, "ppid") or "")
        if parent is None:
            break
        pid = parent
    return pids


def _hook_parse_json(text: str) -> Any:
    """Parse text as JSON, or return {"raw": excerpt} if it is not JSON.

    RecursionError is a decode failure like any other here: json.loads recurses
    per nesting level and blows the stack well before it runs out of input.
    """
    try:
        return json.loads(text)
    except (ValueError, RecursionError):
        return {"raw": text[:_HOOK_PAYLOAD_TRUNCATE_CHARS]}


def _hook_read_stdin() -> bytes:
    """Read up to _HOOK_STDIN_MAX_BYTES from stdin, or b"" if it is a tty.

    Returns within _HOOK_STDIN_TIMEOUT_SECONDS even if a writer holds the pipe
    open without sending EOF: the deadline is re-checked before every select,
    so total wall time is bounded regardless of how the writer chunks its data.
    """
    try:
        if sys.stdin is None or sys.stdin.isatty():
            return b""
        fd = sys.stdin.fileno()
    except (OSError, AttributeError, ValueError):
        return b""

    deadline = time.monotonic() + _HOOK_STDIN_TIMEOUT_SECONDS
    chunks: list[bytes] = []
    total = 0
    while total < _HOOK_STDIN_MAX_BYTES:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            ready, _, _ = select.select([fd], [], [], remaining)
            if not ready:
                break
            chunk = os.read(fd, min(65536, _HOOK_STDIN_MAX_BYTES - total))
        except (BlockingIOError, InterruptedError):
            continue
        except (OSError, ValueError):
            break
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
    return b"".join(chunks)


def _hook_read_payload(
    json_flag: Optional[str],
    extra_argv: list[str],
) -> tuple[Any, str]:
    """Return (payload, source) for the harness that invoked this hook.

    Each harness delivers its payload differently: opencode via --json, codex
    as argv[1], claude on stdin. Tries them in that order and falls back to
    ({}, "none"). Non-JSON input becomes {"raw": excerpt} rather than an error.
    """
    if json_flag is not None:
        return _hook_parse_json(json_flag), "flag"

    if extra_argv and extra_argv[0][:1] in ("{", "["):
        return _hook_parse_json(extra_argv[0]), "argv"

    raw = _hook_read_stdin()
    if not raw:
        return {}, "none"
    return _hook_parse_json(raw.decode("utf-8", errors="replace").strip()), "stdin"


def _hook_extract_message(payload: Any) -> Optional[str]:
    """Return the assistant's last turn from payload, or None.

    Checks claude's last_assistant_message, codex's last-assistant-message,
    then opencode's properties.message / properties.text, in that order.
    A present but non-string value is skipped, not raised on. The result is
    clipped to _HOOK_MESSAGE_MAX_CHARS encoded bytes so one message cannot
    by itself force the payload out of the line budget.
    """
    if not isinstance(payload, dict):
        return None
    for key in ("last_assistant_message", "last-assistant-message"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return _json_clip(value, _HOOK_MESSAGE_MAX_CHARS)
    properties = payload.get("properties")
    if isinstance(properties, dict):
        for key in ("message", "text"):
            value = properties.get(key)
            if isinstance(value, str) and value:
                return _json_clip(value, _HOOK_MESSAGE_MAX_CHARS)
    return None


def _hook_truncate_payload(payload: Any) -> tuple[Any, bool]:
    """Return (truncated_payload, was_truncated) with top-level values capped.

    Caps each top-level string to _HOOK_PAYLOAD_TRUNCATE_CHARS characters and
    clips strings inside top-level lists by the same limit. Nested content
    beyond one level deep is not traversed; bulk nesting is caught by the byte
    budget in _hook_encode_line. Non-dict payloads are returned unchanged.
    """
    if not isinstance(payload, dict):
        return payload, False
    result = {}
    changed = False
    for k, v in payload.items():
        if isinstance(v, str) and len(v) > _HOOK_PAYLOAD_TRUNCATE_CHARS:
            result[k] = v[:_HOOK_PAYLOAD_TRUNCATE_CHARS]
            changed = True
        elif isinstance(v, list):
            clipped = []
            list_changed = False
            for item in v:
                if isinstance(item, str) and len(item) > _HOOK_PAYLOAD_TRUNCATE_CHARS:
                    clipped.append(item[:_HOOK_PAYLOAD_TRUNCATE_CHARS])
                    list_changed = True
                else:
                    clipped.append(item)
            result[k] = clipped
            if list_changed:
                changed = True
        else:
            result[k] = v
    return result, changed


def _json_clip(text: str, max_bytes: int) -> str:
    """Longest prefix of text whose JSON string encoding fits in max_bytes.

    Measures the encoding, not the input: under ensure_ascii one astral
    character expands to a 12-byte surrogate-pair escape, so a character count
    bounds nothing. Excludes the surrounding quotes from the budget.
    """
    def encoded_len(s: str) -> int:
        return len(json.dumps(s, ensure_ascii=True)) - 2

    if encoded_len(text) <= max_bytes:
        return text
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if encoded_len(text[:mid]) <= max_bytes:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo]


def _hook_encode_line(record: dict) -> bytes:
    """Encode record as one JSONL line, always at most _HOOK_MAX_LINE_BYTES.

    Priority ladder, each rung re-checking the bound and stopping as soon as
    the line fits:

        1. as-given
        2. shrink payload only (descending byte budget, floor 8) — message
           untouched, so a long transcript_path cannot eat the message budget
        3. drop payload to {} — message still untouched
        4. clip message progressively (halve from _HOOK_MESSAGE_MAX_CHARS,
           floor 64 bytes)
        5. drop message, keep the rest of the envelope

    Rungs 4-5 are unreachable from cmd_hook: _cmd_hook_inner clips event to
    _HOOK_EVENT_MAX_BYTES and message to _HOOK_MESSAGE_MAX_CHARS first, so the
    worst-case rung-3 envelope is 4414 bytes against an 8192 budget. They bound
    a caller that bypasses that clipping, and are kept for that reason.

    The empty-payload, no-message envelope is the proven floor: every other
    field is a short fixed-vocabulary scalar, and `event` is clipped to
    _HOOK_EVENT_MAX_BYTES by _cmd_hook_inner before this call. allow_nan=False
    makes NaN/Infinity a ValueError, which falls through to the next rung so
    emitted lines are valid RFC-8259 JSON. A non-JSON-native value raises
    TypeError instead, which encode() does not catch; cmd_hook's blanket
    handler absorbs it, preserving the exit-0 guarantee.
    """
    def encode(rec: dict) -> bytes:
        try:
            return (
                json.dumps(rec, ensure_ascii=True, separators=(",", ":"),
                           allow_nan=False) + "\n"
            ).encode()
        except (ValueError, OverflowError):
            return None  # type: ignore[return-value]

    line = encode(record)
    if line is not None and len(line) <= _HOOK_MAX_LINE_BYTES:
        return line

    payload = record.get("payload")
    if isinstance(payload, dict):
        # Descending per-field budget: halve from the initial cap until the
        # line fits, floor at 8 bytes per field. message is not part of this
        # candidate, so it rides along unclipped at every step.
        budget = _HOOK_PAYLOAD_TRUNCATE_CHARS
        while budget >= 8:
            candidate: dict[str, Any] = {}
            for k, v in payload.items():
                if isinstance(v, str):
                    candidate[k] = _json_clip(v, budget)
                elif isinstance(v, list):
                    candidate[k] = [
                        _json_clip(item, budget) if isinstance(item, str) else item
                        for item in v
                    ]
                else:
                    candidate[k] = v
            line = encode({**record, "payload": candidate, "payload_truncated": True})
            if line is not None and len(line) <= _HOOK_MAX_LINE_BYTES:
                return line
            budget //= 2

    # Payload dropped entirely; message is still whatever _cmd_hook_inner set.
    base = {**record, "payload": {}, "payload_truncated": True}
    line = encode(base)
    if line is not None and len(line) <= _HOOK_MAX_LINE_BYTES:
        return line

    message = base.get("message")
    if isinstance(message, str) and message:
        # Clip the message itself: it is the last field worth keeping, so it
        # is shrunk rather than dropped until the floor is reached.
        clip_budget = _HOOK_MESSAGE_MAX_CHARS
        while clip_budget >= 64:
            clipped_message = _json_clip(message, clip_budget)
            candidate = {**base, "message": clipped_message, "message_truncated": True}
            line = encode(candidate)
            if line is not None and len(line) <= _HOOK_MAX_LINE_BYTES:
                return line
            clip_budget //= 2

    # Backstop: drop message entirely. The bare envelope is proven to fit.
    line = encode({**base, "message": None})
    if line is None:
        # Reached when a non-message envelope field is non-finite: NaN/Inf in
        # payload is resolved at rung 3, but in an envelope field it survives
        # to here and makes encode() return None. Emit a minimal safe record.
        line = encode({
            "event": _json_clip(str(record.get("event", "")), _HOOK_EVENT_MAX_BYTES),
            "at": record.get("at", ""),
            "payload": {},
            "payload_truncated": True,
            "message": None,
        })
    # Final unconditional bound check: a line over budget would break append
    # atomicity for every concurrent writer. Truncation at this point means
    # the caller passed an un-clipped event; drop rather than corrupt the log.
    if line is None or len(line) > _HOOK_MAX_LINE_BYTES:
        return b""
    return line


def _hooks_read_tail(log_dir: Path, max_bytes: int) -> tuple[Optional[bytes], bool]:
    """Read at most max_bytes from the tail of log_dir/hooks.jsonl.

    Returns (data, truncated). truncated is True when the file exceeds max_bytes
    and only the tail was read. The first partial line in a truncated read is
    discarded so all returned lines are complete records.
    """
    try:
        dir_fd = os.open(str(log_dir), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError:
        return None, False
    try:
        fd = os.open(
            "hooks.jsonl",
            os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
            dir_fd=dir_fd,
        )
    except OSError:
        return None, False
    finally:
        os.close(dir_fd)
    try:
        if not _stat_module.S_ISREG(os.fstat(fd).st_mode):
            os.close(fd)
            return None, False
        os.set_blocking(fd, True)
        with open(fd, "rb", closefd=True) as fh:
            size = fh.seek(0, 2)  # seek to end to get file size
            if size <= max_bytes:
                fh.seek(0)
                return fh.read(max_bytes), False
            # Tail read: seek back max_bytes from end, drop the first partial line.
            fh.seek(size - max_bytes)
            raw = fh.read(max_bytes)
            newline = raw.find(b"\n")
            if newline >= 0:
                raw = raw[newline + 1:]
            return raw, True
    except OSError:
        try:
            os.close(fd)
        except OSError:
            pass
        return None, False


_HOOK_OPEN_ENOENT_ATTEMPTS: int = 5


def _hook_open_append(dir_fd: int) -> int:
    """Open hooks.jsonl for atomic append relative to dir_fd.

    Concurrent creators racing O_CREAT can each get ENOENT even though the
    directory exists and the flags ask for creation. The condition is transient
    and confined to creation. Since cmd_hook must return 0 on every path, an
    unretried failure here is indistinguishable from a successful append and
    silently drops the record it was called to persist.

    Retries ENOENT only. Every other errno propagates on the first attempt, and
    O_NOFOLLOW/O_NONBLOCK apply to each attempt, so a symlink or FIFO planted
    between retries is refused exactly as on the first open.
    """
    flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK
    for attempt in range(_HOOK_OPEN_ENOENT_ATTEMPTS):
        try:
            return os.open("hooks.jsonl", flags, 0o600, dir_fd=dir_fd)
        except FileNotFoundError:
            if attempt == _HOOK_OPEN_ENOENT_ATTEMPTS - 1:
                raise
    raise AssertionError("unreachable: loop returns or raises")


def _hook_append_record(log_dir: Path, record: dict) -> None:
    """Append one record to log_dir/hooks.jsonl, or drop it at the event cap.

    O_NOFOLLOW refuses a symlink at the path and fstat refuses any non-regular
    file that replaced it after the cap check, so neither can redirect the
    write. A single os.write of at most _HOOK_MAX_LINE_BYTES to the O_APPEND
    fd on a local regular file does not interleave with concurrent hooks —
    the atomicity is O_APPEND's offset+write guarantee, not a PIPE_BUF limit.
    Never raises; errors go to stderr.
    """
    # One line-length of headroom past the cap's worth of max-size lines. A
    # window of exactly AGENT_RUN_HOOK_MAX_EVENTS * _HOOK_MAX_LINE_BYTES holds
    # that many maximal lines with no slack, so crossing it makes
    # _hooks_read_tail discard the partial leading line and report at most
    # AGENT_RUN_HOOK_MAX_EVENTS - 1 — below the cap forever, disabling it.
    try:
        existing, truncated = _hooks_read_tail(
            log_dir, (AGENT_RUN_HOOK_MAX_EVENTS + 1) * _HOOK_MAX_LINE_BYTES
        )
    except (OverflowError, ValueError) as exc:
        # An unrepresentable read size means the cap cannot be decided. Say so
        # and drop this record: cmd_hook must return 0 either way, so an
        # exception here would be indistinguishable from a successful append.
        print(f"agent-run hook: cannot size the event-cap read: {exc}", file=sys.stderr)
        return
    # A truncated read means the file is larger than the cap could ever need,
    # which is itself proof of being at the cap; counting is only meaningful
    # when the whole relevant region was read.
    if truncated:
        return
    if existing is not None and sum(
        1 for ln in existing.split(b"\n") if ln.strip()
    ) >= AGENT_RUN_HOOK_MAX_EVENTS:
        return

    line = _hook_encode_line(record)
    try:
        dir_fd = os.open(str(log_dir), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            fd = _hook_open_append(dir_fd)
            try:
                if not _stat_module.S_ISREG(os.fstat(fd).st_mode):
                    return
                written = os.write(fd, line)
                if written != len(line):
                    # Terminate the partial line so readers see one bad record
                    # rather than a splice of this one and the next.
                    os.write(fd, b"\n")
                    print(
                        f"agent-run hook: short write ({written}/{len(line)} bytes)",
                        file=sys.stderr,
                    )
            finally:
                os.close(fd)
        finally:
            os.close(dir_fd)
    except OSError as exc:
        print(f"agent-run hook: cannot write hooks.jsonl: {exc}", file=sys.stderr)


def cmd_hook(args: argparse.Namespace) -> int:
    """Record a harness hook notification in the run's hooks.jsonl.

    Always returns 0: a claude Stop hook that exits non-zero is read by the
    harness as a decision to block, so a broken hook would stall the session.
    Diagnostics go to stderr only — codex warns on notify-program stdout. See
    the module docstring for per-harness configuration.
    """
    try:
        return _cmd_hook_inner(args)
    except Exception as exc:  # noqa: BLE001
        print(f"agent-run hook: unexpected error: {exc}", file=sys.stderr)
        return 0


def _hook_detect_harness(payload: Any) -> Optional[str]:
    """Infer harness from payload key presence; returns None when unrecognised.

    Detection is best-effort key presence, not authoritative: an agent that
    controls hooks.jsonl can forge this field by adding keys from another
    harness's shape. Use the `source` field (flag/argv/stdin) as a
    cross-check when attribution matters — it reflects the delivery mechanism
    the hook binary observed, which the agent cannot choose independently.
    """
    if not isinstance(payload, dict):
        return None
    if "session_id" in payload and "hook_event_name" in payload:
        return "claude"
    if "type" in payload and "thread-id" in payload:
        return "codex"
    if "type" in payload and "properties" in payload:
        return "opencode"
    return None


_HOOK_KIND_TURN_COMPLETE_EVENTS = frozenset(
    {"stop", "turn-complete", "turn_complete", "session-idle", "session_idle", "idle"}
)
_HOOK_KIND_SESSION_START_EVENTS = frozenset({"session-start", "session_start"})

# Only events naming a PENDING request. A resolved permission is not a reason
# to wake anyone, so it is deliberately absent.
_HOOK_KIND_PERMISSION_EVENTS = frozenset(
    {
        "permission",
        "permission-request",
        "permission_request",
        "permission-asked",
        "permission_asked",
        "permission-required",
        "permission_required",
    }
)


def _hook_canonical_kind(harness: Optional[str], event: str, payload: Any) -> str:
    """Return one of turn_complete/permission_required/session_start/other.

    threadctl must consume this without knowing which harness ran or what a
    user named their hook. Payload evidence is checked first, per harness,
    because it cannot be renamed by a user's harness config the way the
    event argv string can; the event string is only a fallback for payloads
    that carry no recognisable shape.
    """
    if isinstance(payload, dict):
        if harness == "claude":
            hook_event_name = payload.get("hook_event_name")
            if hook_event_name == "Stop":
                return "turn_complete"
            # Notification is claude's generic attention nag and carries no
            # evidence a permission is pending, so it is not mapped here.
            if hook_event_name == "PermissionRequest":
                return "permission_required"
            if hook_event_name == "SessionStart":
                return "session_start"
        elif harness == "codex":
            if payload.get("type") == "agent-turn-complete":
                return "turn_complete"
            hook_event_name = payload.get("hook_event_name")
            if hook_event_name == "Stop":
                return "turn_complete"
            if hook_event_name == "PermissionRequest":
                return "permission_required"
            if hook_event_name == "SessionStart":
                return "session_start"
        elif harness == "opencode":
            event_type = payload.get("type")
            if event_type == "session.idle":
                return "turn_complete"
            # Exact match, not a "permission." prefix: permission.replied
            # reports a permission already answered, which is not pending.
            if event_type == "permission.asked":
                return "permission_required"
            if event_type == "session.created":
                return "session_start"

    if event in _HOOK_KIND_TURN_COMPLETE_EVENTS:
        return "turn_complete"
    if event in _HOOK_KIND_PERMISSION_EVENTS:
        return "permission_required"
    if event in _HOOK_KIND_SESSION_START_EVENTS:
        return "session_start"
    return "other"


def _cmd_hook_inner(args: argparse.Namespace) -> int:
    resolved = _hook_resolve_run(getattr(args, "name", None))
    if resolved is None:
        return 0
    name, resolved_by = resolved

    payload, source = _hook_read_payload(
        getattr(args, "json_payload", None),
        list(getattr(args, "extra", []) or []),
    )
    truncated_payload, field_truncated = _hook_truncate_payload(payload)
    # Kind derivation reads the un-clipped event string and the untruncated
    # payload: both carry more signal than what ends up in the stored record.
    normalized_event = args.event.lower().encode("utf-8", "replace").decode("utf-8")
    harness = _hook_detect_harness(payload)
    record: dict[str, Any] = {
        # Clipped in encoded bytes so the envelope alone can never exceed
        # _HOOK_MAX_LINE_BYTES and cost the write its atomicity.
        "event": _json_clip(normalized_event, _HOOK_EVENT_MAX_BYTES),
        "at": _now_iso(),
        # harness is inferred from payload key presence; it is attacker-declared
        # when the agent controls hooks.jsonl, not an authoritative fact.
        "harness": harness,
        # kind normalises event across harnesses and user-renamed hook argv;
        # payload evidence is preferred over the possibly-misleading event.
        "kind": _hook_canonical_kind(harness, normalized_event, payload),
        "message": _hook_extract_message(payload),
        "source": source,
        "pid": os.getpid(),
        "resolved_by": resolved_by,
        "payload": truncated_payload,
    }
    if field_truncated:
        record["payload_truncated"] = True
    _hook_append_record(_log_dir(name), record)
    return 0


def _read_hooks_jsonl(log_dir: Path) -> Optional[dict]:
    """Summarise hooks.jsonl, or return None if there is no readable one.

    Total by contract, and the single place that guarantees it: `watch` calls
    this while reporting a run's liveness, and an exception propagating from
    here would land in cmd_watch's handler as status "unknown", which
    _watch_is_terminal treats as terminal — a live run reported dead.
    """
    try:
        return _hooks_summary(log_dir)
    except Exception as exc:  # noqa: BLE001
        print(f"agent-run: cannot summarise hooks.jsonl: {exc}", file=sys.stderr)
        return None


def _hooks_summary(log_dir: Path) -> Optional[dict]:
    """Count and aggregate hooks.jsonl records; skip lines that are not JSON
    objects. None when the file is absent or not a readable regular file.

    Reads the tail of the file so `last` and `last_event_age_s` describe the
    newest records even when the file exceeds _HOOKS_READ_MAX_BYTES. truncated
    is True when the file was larger than the read cap and only the tail was
    parsed; a consumer can use it to tell a complete aggregate from a partial one.
    """
    raw, truncated = _hooks_read_tail(log_dir, _HOOKS_READ_MAX_BYTES)
    if raw is None:
        return None

    records: list[dict] = []
    for line in raw.split(b"\n"):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except (ValueError, RecursionError):
            continue
        if isinstance(obj, dict):
            records.append(obj)

    if not records:
        return {"count": 0, "last": None, "last_event_age_s": None,
                "events": {}, "kinds": {}, "at_cap": False, "truncated": truncated}

    last = records[-1]

    def _clean(s: Any, max_chars: int) -> Optional[str]:
        # Envelope fields come from a file any process can append to, so a
        # non-string here is data, not a bug: summarise it as None rather
        # than raising and losing the whole aggregate. The bound applies here
        # rather than at write time because this dict is the watch contract: a
        # supervisor polls it on a timer and renders it, so one oversized line
        # in hooks.jsonl must not become an oversized field in every poll.
        if not isinstance(s, str):
            return None
        return s.encode("utf-8", "replace").decode("utf-8")[:max_chars]

    raw_message = last.get("message")
    message = _clean(raw_message, _HOOK_MESSAGE_MAX_CHARS)
    last_summary = {
        "event": _clean(last.get("event"), _HOOK_EVENT_MAX_BYTES),
        "at": _clean(last.get("at"), _HOOK_EVENT_MAX_BYTES),
        "harness": _clean(last.get("harness"), _HOOK_EVENT_MAX_BYTES),
        "kind": _clean(last.get("kind"), _HOOK_EVENT_MAX_BYTES),
        "message": message,
        "message_clipped": (
            isinstance(raw_message, str) and len(message or "") < len(raw_message)
        ),
    }

    last_event_age_s: Optional[float] = None
    at_str = last.get("at")
    if isinstance(at_str, str) and at_str:
        try:
            at_dt = datetime.strptime(at_str, "%Y-%m-%dT%H:%M:%SZ").replace(
                tzinfo=timezone.utc
            )
            last_event_age_s = max(
                0.0, (datetime.now(timezone.utc) - at_dt).total_seconds()
            )
        except (ValueError, TypeError):
            pass

    event_counts: dict[str, int] = {}
    kind_counts: dict[str, int] = {}
    for r in records:
        # Counter keys are bounded for the same reason as last_summary: they
        # are attacker-chosen strings that ship in every watch poll.
        ev = _clean(r.get("event"), _HOOK_EVENT_MAX_BYTES)
        if ev is not None:
            event_counts[ev] = event_counts.get(ev, 0) + 1
        kind = _clean(r.get("kind"), _HOOK_EVENT_MAX_BYTES)
        if kind is not None:
            kind_counts[kind] = kind_counts.get(kind, 0) + 1

    return {
        "count": len(records),
        "last": last_summary,
        "last_event_age_s": last_event_age_s,
        "events": event_counts,
        "kinds": kind_counts,
        # at_cap reflects whether the parsed record count meets the cap constant
        # as seen by the watcher's process; hooks were written under the hook
        # process's cap. Only reliable when both share the same environment.
        "at_cap": len(records) >= AGENT_RUN_HOOK_MAX_EVENTS,
        "truncated": truncated,
    }


def cmd_watch(args: argparse.Namespace) -> int:
    """Emit a stateless, read-only JSON fact snapshot for one run.

    Pure observation: no escalation policy, no thresholds-as-actions, no
    network, no model calls, no mutations. Exit codes carry three distinct
    meanings a poller must be able to tell apart:

        0  a valid contract was printed. This includes the case where
           observation itself failed unexpectedly — the contract still came
           out, with ``observation_error`` set instead of null.
        1  the tool itself failed before it could print anything (e.g. an
           invalid run name) — nothing observed, nothing printed.
        2  the run name is unresolvable: no state dir and no log dir exist
           for it, stdout empty. This is the one code a poller should treat
           as "stop polling this name".

    Every other failure mode (dead run, missing ``cwd`` file, git repo
    unreadable, log unreadable) degrades individual fields to ``null``
    rather than raising or exiting non-zero, so a poller can call this
    every 30s forever. ``cmd_status`` is unaffected by this split — it
    keeps exiting 1 for an unresolvable name.

    ``status`` comes from ``_watch_effective_status``, which adds pid-
    identity verification: an alive pid whose recorded launch identity does
    not match the live process's is a recycled pid and reports "died"; an
    alive pid whose identity cannot be confirmed at all reports
    "unverified" rather than "running". ``terminal`` is exactly derivable
    from ``status`` via ``_watch_is_terminal``.

    ``observation_error`` is ``null`` on every normal path; when the
    top-level guard fires it holds the exception class name plus a
    truncated message, with every other field at its null/unknown value.

    ``git_error`` is ``null`` whenever ``git`` is populated, and otherwise
    exactly one of ``"no_repo_path"``, ``"not_a_repo"``, ``"no_commits"``,
    ``"timeout"``, ``"git_missing"``, ``"git_failed"``, or
    ``"changed_during_observation"`` — a poller that fails
    toward escalating on unknown state must be able to tell "cannot observe
    this repo" (alarming) apart from "observed it, there is nothing there"
    (benign), which a bare ``git: null`` cannot express on its own. See
    ``_watch_git_facts_checked`` for the ``git`` object's keys.

    ``signals.repeated_error`` and ``signals.top_repeated_read`` share one
    winner rule: the most-repeated qualifying line/path in the log tail, or
    ``null`` if nothing repeats at least ``WATCH_REPEAT_THRESHOLD`` times.

    ``log.lines`` is ``null`` when the log exceeds ``WATCH_LINE_COUNT_MAX_BYTES``
    (16 MiB); ``log.bytes`` is always populated and gives the true file size.
    """
    name = _validate_run_name(args.name)
    repo_arg = _watch_validate_repo_arg(getattr(args, "repo", None))
    as_json = bool(args.json)
    state_dir = _state_dir(name)
    log_dir = _log_dir(name)

    state_dir_probe = _probe_dir_state(state_dir)
    log_dir_probe = _probe_dir_state(log_dir)

    if state_dir_probe == _DIR_MISSING and log_dir_probe == _DIR_MISSING:
        # Both paths are conclusively absent: no run by this name exists.
        # Distinct from observation breaking; empty stdout, exit 2.
        print(f"agent-run: no run named '{name}' in {STATE_ROOT} or {LOG_ROOT}", file=sys.stderr)
        return 2

    if state_dir_probe == _DIR_UNSTATABLE or log_dir_probe == _DIR_UNSTATABLE:
        # At least one path cannot be statted: the run may be live but
        # unreadable.  Emit the degraded contract so a poller never reads
        # "can't observe" as "no such run", then exit 0.
        message = (
            f"cannot stat {'state' if state_dir_probe == _DIR_UNSTATABLE else 'log'} "
            f"directory for '{name}' — run may be live but is unreadable"
        )
        payload = _watch_payload(name, _now_iso(), "unknown", observation_error=message)
        _watch_emit(
            payload,
            as_json,
            f"name={name} status={payload['status']} observation_error={message!r}",
        )
        return 0

    # Everything below observes best-effort facts about a run this process
    # does not control; an unanticipated failure must still print the full
    # contract and exit 0, so a poller can never mistake "observation broke"
    # for "no such run".
    try:
        return _cmd_watch_observe(name, state_dir, log_dir, repo_arg, as_json)
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"[:WATCH_ERROR_LINE_MAX_CHARS]
        payload = _watch_payload(name, _now_iso(), "unknown", observation_error=message)
        _watch_emit(
            payload,
            as_json,
            f"name={name} status={payload['status']} observation_error={message!r}",
        )
        return 0


def _watch_emit(payload: dict, as_json: bool, human: str) -> None:
    """Print the contract as JSON, or *human* as the one-line form."""
    print(json.dumps(payload) if as_json else human)


class _ScratchScanError(Exception):
    """Aborts a scratch scan with a categorical *code*.

    The code is a fixed string, never ``str(exc)``: the scratch facts are
    rendered into Discord and persisted, and an OSError's text carries the
    path it failed on.
    """

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _scratch_unknown(code: str) -> dict:
    """Scratch facts saying "unknown, because *code*".

    Every decision field is null rather than 0 so a poller cannot read a
    failed scan as observed inactivity.
    """
    return {
        "newest_mtime_age_s": None,
        "files_modified_recent": None,
        "scanned": None,
        "truncated": False,
        "error": code,
    }


def _watch_scratch_facts(working_dir: Optional[str]) -> dict:
    """Scan *working_dir* for recent file-write activity, gitignored paths included.

    Read-only review runs write their findings to gitignored scratch (e.g.
    ``.taskdocs/probe/``, ``.taskdocs/findings-*.md``) and produce no git
    facts at all, so git-only evidence treats them as stalled.  This scan
    gives the watchdog a signal that is blind to git status.

    Bounded by WATCH_SCRATCH_MAX_FILES, WATCH_SCRATCH_MAX_ENTRIES,
    WATCH_SCRATCH_MAX_DEPTH and WATCH_SCRATCH_BUDGET_SECONDS so it stays cheap
    on every poll over SSH; WATCH_SCRATCH_PRUNE_DIRS trees are skipped.

    Returns a dict with keys:
      newest_mtime_age_s    – age of the most-recently-modified file (float|null)
      files_modified_recent – files modified within WATCH_SCRATCH_RECENT_SECONDS (int|null)
      scanned               – files actually examined (int|null)
      truncated             – True when a bound was hit before the scan finished (bool)
      error                 – categorical reason the result is unknown (str|null)

    On truncation the result is asymmetric by design:
      files_modified_recent >= 1  – sound positive evidence (activity was seen
                                    within the root tree; fd-based descent with
                                    O_NOFOLLOW prevents counting outside files);
                                    kept with truncated=True and error=None.
      files_modified_recent == 0  – absence of evidence, not evidence of absence;
                                    degraded to null fields, truncated=True, and a
                                    categorical code (file_limit/entry_limit/
                                    timeout/depth_limit).  See test
                                    test_truncated_zero_degrades_to_null_with_code.

    Any hard error yields ``_scratch_unknown``: null counts, never a confident 0.
    """
    if not working_dir:
        return _scratch_unknown("no_working_dir")

    root = Path(working_dir)
    now = time.monotonic()
    deadline = now + WATCH_SCRATCH_BUDGET_SECONDS

    try:
        if not root.is_dir():
            return _scratch_unknown("not_a_directory")
    except OSError:
        return _scratch_unknown("stat_error")

    scan_start = time.time()  # wall-clock epoch for mtime comparisons only

    newest_mtime: Optional[float] = None  # epoch seconds of the newest file seen
    files_modified_recent: int = 0
    scanned: int = 0
    entries_visited: int = 0
    truncated: bool = False
    truncation_code: str = ""  # set alongside truncated=True; one of the categorical codes

    # Iterative walk using opened directory descriptors so descent cannot follow
    # a symlink that replaced a queued directory between classification and
    # descent.  O_NOFOLLOW|O_DIRECTORY causes os.open to fail (ENOTDIR/ELOOP)
    # when the target is a symlink, preventing both outside-positive injection
    # and active-subtree hiding.  The queue stores (fd, depth) tuples; open fds
    # are bounded by the queue, not by total entries seen.
    #
    # Fall-back (platforms without O_NOFOLLOW or O_DIRECTORY, or where
    # os.open does not accept dir_fd): revalidate directory identity with
    # lstat(st_dev, st_ino) immediately before descent and fail unknown on any
    # mismatch.  This narrows but does not fully close the race: a swap that
    # completes between the lstat and the scandir call cannot be detected.
    _DIR_OPEN_FLAGS = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_DIRECTORY", 0)
    _USE_FD_DESCENT = (
        hasattr(os, "O_NOFOLLOW")
        and hasattr(os, "O_DIRECTORY")
        and os.open in os.supports_dir_fd
    )

    try:
        root_fd = os.open(str(root), _DIR_OPEN_FLAGS)
    except OSError:
        return _scratch_unknown("stat_error")

    # Queue stores (fd, depth).  Each fd is open and will be closed after
    # its contents are scanned (or when the queue is drained on early exit).
    queue: list[tuple[int, int]] = [(root_fd, 0)]

    def _drain_queue(q: list) -> None:
        """Close every fd remaining in the queue without raising."""
        for _fd, _ in q:
            try:
                os.close(_fd)
            except OSError:
                pass
        q.clear()

    try:
        while queue:
            # Budget is cooperative: checked between filesystem calls, not
            # during them.  A single blocking scandir/stat can exceed it on
            # slow or hostile mounts.  See WATCH_SCRATCH_BUDGET_SECONDS.
            if time.monotonic() >= deadline:
                truncated = True
                truncation_code = "timeout"
                break

            current_fd, depth = queue.pop()

            # Stream the scandir iterator, never list() it, and re-check the
            # deadline per entry: on slow storage the enumeration itself is
            # what runs long, and only a per-entry check bounds it.
            try:
                with os.scandir(current_fd) as it:
                    for entry in it:
                        if time.monotonic() >= deadline:
                            truncated = True
                            truncation_code = "timeout"
                            break

                        entries_visited += 1
                        if entries_visited >= WATCH_SCRATCH_MAX_ENTRIES:
                            truncated = True
                            truncation_code = "entry_limit"
                            break

                        # One no-follow stat classifies the entry and fetches the
                        # mtime in a single syscall.  DirEntry.stat(follow_symlinks=
                        # False) is cached by the OS after the first call, so this
                        # is cost-neutral relative to two predicate calls.  Any
                        # OSError (including a file that vanished during
                        # classification) is caught here, preventing the false/false
                        # predicate path that silently treated disappearing entries
                        # as ignorable special files.
                        try:
                            entry_stat = entry.stat(follow_symlinks=False)
                        except OSError:
                            raise _ScratchScanError("stat_error")

                        st_mode = entry_stat.st_mode
                        if _stat_module.S_ISDIR(st_mode):
                            if entry.name not in WATCH_SCRATCH_PRUNE_DIRS:
                                if depth < WATCH_SCRATCH_MAX_DEPTH:
                                    # Open child fd relative to current_fd (openat
                                    # semantics): O_NOFOLLOW rejects a symlink that
                                    # replaced the queued directory after classification,
                                    # failing with ENOTDIR or ELOOP; other OSErrors
                                    # (EACCES, EIO, ENOENT) are scan failures.
                                    # On the fallback path, revalidate identity (st_dev,
                                    # st_ino) before opening; a mismatch means the
                                    # directory was replaced but the recheck cannot
                                    # detect a swap between lstat and scandir.
                                    if _USE_FD_DESCENT:
                                        try:
                                            child_fd = os.open(
                                                entry.name,
                                                _DIR_OPEN_FLAGS,
                                                dir_fd=current_fd,
                                            )
                                        except OSError as _open_err:
                                            # ENOTDIR/ELOOP: O_NOFOLLOW rejected a
                                            # symlink — directory identity changed.
                                            # Other errors: permission, I/O, gone.
                                            if _open_err.errno in (errno.ENOTDIR, errno.ELOOP):
                                                raise _ScratchScanError("stat_error")
                                            raise _ScratchScanError("scan_error")
                                    else:
                                        child_path = Path(entry.path)
                                        try:
                                            recheck = os.lstat(str(child_path))
                                        except OSError:
                                            raise _ScratchScanError("stat_error")
                                        if (
                                            recheck.st_dev != entry_stat.st_dev
                                            or recheck.st_ino != entry_stat.st_ino
                                        ):
                                            raise _ScratchScanError("stat_error")
                                        try:
                                            child_fd = os.open(str(child_path), _DIR_OPEN_FLAGS)
                                        except OSError:
                                            raise _ScratchScanError("scan_error")
                                    queue.append((child_fd, depth + 1))
                                else:
                                    # Non-pruned subtree skipped solely for depth:
                                    # unseen files may be recent, so mark truncated.
                                    truncated = True
                                    truncation_code = "depth_limit"
                            continue

                        # Only regular files contribute to the activity signal.
                        if not _stat_module.S_ISREG(st_mode):
                            continue

                        mtime = entry_stat.st_mtime

                        # NaN and infinities cannot represent a real timestamp:
                        # NaN makes every ordered comparison false (max(0.0, NaN)
                        # returns 0.0); -inf reads as infinitely old; +inf reaches
                        # clock_skew only incidentally.  Reject all three uniformly
                        # so hostile or faulty FUSE/NFS stat data cannot manufacture
                        # a confident zero.
                        if not math.isfinite(mtime):
                            raise _ScratchScanError("invalid_mtime")

                        scanned += 1
                        if newest_mtime is None or mtime > newest_mtime:
                            newest_mtime = mtime
                        age = scan_start - mtime
                        if age < -WATCH_LOG_FUTURE_MTIME_TOLERANCE_SECONDS:
                            # mtime materially in the future: clock disagreement on the
                            # filesystem server.  A freshly-written file on NFS can show
                            # age < 0 because the server clock leads the client.  Beyond
                            # WATCH_LOG_FUTURE_MTIME_TOLERANCE_SECONDS this is unsound to
                            # ignore: degrade to null rather than report a confident zero
                            # (matching the tolerance the log path already applies).
                            raise _ScratchScanError("clock_skew")
                        # age is >= -TOLERANCE here; minor jitter (age < 0 within tolerance)
                        # is treated as 0, which is within the recent window.
                        if age < WATCH_SCRATCH_RECENT_SECONDS:
                            files_modified_recent += 1

                        if scanned >= WATCH_SCRATCH_MAX_FILES:
                            truncated = True
                            truncation_code = "file_limit"
                            break

            finally:
                # current_fd was opened before being pushed; close it now
                # regardless of how the scan ended (normal, break, exception).
                try:
                    os.close(current_fd)
                except OSError:
                    pass

            if truncated and truncation_code != "depth_limit":
                # depth_limit only truncates the queue; continue scanning
                # remaining queued directories unless another bound fires.
                break

    except _ScratchScanError as err:
        _drain_queue(queue)
        return _scratch_unknown(err.code)
    except OSError:
        _drain_queue(queue)
        # scandir() failed on the root or a discovered subdirectory (removed,
        # permission denied, I/O error). Partial observations are unreliable.
        return _scratch_unknown("scan_error")

    # Drain any remaining queued fds (e.g. after a truncation break above).
    _drain_queue(queue)

    if newest_mtime is None:
        newest_mtime_age_s: Optional[float] = None
    else:
        delta = scan_start - newest_mtime
        # Clamp minor future jitter (within WATCH_LOG_FUTURE_MTIME_TOLERANCE_SECONDS)
        # to 0.0; material clock skew was already caught per-file above.
        newest_mtime_age_s = max(0.0, delta)

    # Asymmetric truncation contract: files_modified_recent >= 1 is sound
    # positive evidence even under truncation — fd-based descent with
    # O_NOFOLLOW|O_DIRECTORY confines the scan to the root tree so outside
    # files cannot be counted.  A zero under truncation is absence of evidence,
    # not evidence of absence — degrade to null fields plus a categorical code.
    # Tests: test_truncated_zero_degrades_to_null_with_code (zero-degradation),
    # test_queued_dir_replaced_by_symlink_outside_injection (confinement positive),
    # test_queued_dir_replaced_by_symlink_hides_activity (confinement zero).
    if truncated and files_modified_recent == 0:
        result = _scratch_unknown(truncation_code)
        result["truncated"] = True
        return result

    return {
        "newest_mtime_age_s": newest_mtime_age_s,
        "files_modified_recent": files_modified_recent,
        "scanned": scanned,
        "truncated": truncated,
        "error": None,
    }


def _transcript_unknown(code: str, submitted_age_s: Optional[float] = None) -> dict:
    """Return unavailable facts; an unknown entry count is never zero."""
    return {
        "available": False,
        "entries": None,
        "newest_age_s": None,
        "submitted_age_s": submitted_age_s,
        "error": code,
    }


def _read_prompt_submitted_sentinel(path: Path) -> Optional[tuple[str, float]]:
    """Read bounded regular-file content and mtime from one descriptor."""
    max_bytes = 256  # generous for either on-disk format: an ISO-8601 timestamp or the bare "1" sentinel
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return None
    try:
        st = os.fstat(fd)
        if not _stat_module.S_ISREG(st.st_mode) or st.st_size > max_bytes:
            return None
        data = os.read(fd, max_bytes)
    except OSError:
        return None
    finally:
        os.close(fd)
    try:
        return data.decode("utf-8").strip(), st.st_mtime
    except UnicodeError:
        return None


def _watch_prompt_submitted_age_s(state_dir: Optional[Path], interactive: Optional[bool]) -> Optional[float]:
    """Return interactive prompt age from ISO content or legacy sentinel mtime."""
    if interactive is not True or state_dir is None:
        return None
    read = _read_prompt_submitted_sentinel(state_dir / "prompt_submitted")
    if read is None:
        return None
    raw, mtime = read
    if not raw:
        return None
    now = datetime.now(timezone.utc)
    parsed = _watch_parse_iso(raw)
    if parsed is not None:
        return max(0.0, (now - parsed).total_seconds())
    if raw != "1":
        return None
    return max(0.0, now.timestamp() - mtime)


def _watch_transcript_facts(
    session_data: Optional[dict],
    cwd: Optional[str],
    state_dir: Optional[Path] = None,
    interactive: Optional[bool] = None,
) -> dict:
    """Return transcript facts, degrading every store failure to unavailable."""
    submitted_age_s = _watch_prompt_submitted_age_s(state_dir, interactive)

    if not isinstance(session_data, dict):
        return _transcript_unknown("no_session_json", submitted_age_s)
    session_id = session_data.get("session_id")
    harness = session_data.get("harness")
    if not isinstance(session_id, str) or not session_id or not isinstance(harness, str):
        return _transcript_unknown("no_session_id", submitted_age_s)

    try:
        count, newest_iso = _agent_run_transcript.count_transcript(harness, session_id, cwd)
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise TypeError(f"count_transcript returned a non-count value: {count!r}")
        if newest_iso is not None and not isinstance(newest_iso, str):
            raise TypeError(f"count_transcript returned a non-timestamp newest value: {newest_iso!r}")
        newest_age_s: Optional[float] = None
        newest_dt = _watch_parse_iso(newest_iso)
        if newest_dt is not None:
            newest_age_s = max(0.0, (datetime.now(timezone.utc) - newest_dt).total_seconds())
        return {
            "available": True,
            "entries": count,
            "newest_age_s": newest_age_s,
            "submitted_age_s": submitted_age_s,
            "error": None,
        }
    except _agent_run_transcript.TranscriptSourceError as exc:
        return _transcript_unknown(exc.code, submitted_age_s)
    except Exception:
        return _transcript_unknown("store_unreadable", submitted_age_s)


def _watch_payload(name: str, observed_at: str, status: str, **fields) -> dict:
    """Build the watch contract with every field at its null/unknown default,
    overridden by *fields*. The key set is fixed here so it cannot vary
    between the normal, missing-state-dir and observation-error branches, and
    ``terminal`` is always derived from ``status`` rather than passed in.

    ``session`` is a new additive field (spec §7): a dict when session.json is
    present in the run's log dir, null otherwise. Existing keys are unchanged.
    """
    payload = {
        "schema": "agent-run.watch.v2",
        "agent_run_version": TOOLBOX_VERSION,
        "name": name,
        "observed_at": observed_at,
        "status": status,
        "exit_code": None,
        "pid": None,
        "interactive": None,
        "started_at": None,
        "ended_at": None,
        "elapsed_s": None,
        "terminal": _watch_is_terminal(status),
        "launch_error": None,
        "log": None,
        "repo": None,
        "git": None,
        "git_error": None,
        "signals": {
            "repeated_error": None,
            "distinct_files_read": 0,
            "top_repeated_read": None,
        },
        "observation_error": None,
        "session": None,
        "scratch": _scratch_unknown("not_observed"),
        "hooks": None,
        "transcript": _transcript_unknown("not_observed"),
    }
    unknown = set(fields) - set(payload)
    if unknown:
        raise KeyError(f"not watch contract fields: {sorted(unknown)}")
    payload.update(fields)
    return payload


def _watch_read_cwd_file(path: Path) -> str:
    """Read the recorded launch ``cwd``, taking only the first line.
    ``cmd_launch`` writes it as a single line, but by the time ``watch``
    reads it a state file is untrusted input: a multi-line value must not be
    echoed verbatim into the contract's ``repo`` field."""
    raw = _read(path)
    return raw.splitlines()[0] if raw else ""


def _watch_repo_git(repo: Optional[str], launch_head: Optional[str]) -> _WatchGitFactsResult:
    """Git facts for the run's repo, or the ``no_repo_path`` discriminator
    when no repo path is known at all."""
    if not repo:
        return _WatchGitFactsResult(None, "no_repo_path")
    return _watch_git_facts_checked(Path(repo), launch_head)


def _cmd_watch_observe(
    name: str, state_dir: Path, log_dir: Path, repo_arg: Optional[str], as_json: bool
) -> int:
    """Build and print the watch contract. May raise; ``cmd_watch`` is the
    never-raise boundary that catches it."""
    observed_at = _now_iso()
    # Only the path is resolved here; regular-file validation happens on the
    # open descriptor in _watch_open_validated_log, because the inode a path
    # names can change between this line and the actual read.
    log = _log_file_for(name)

    def observed(
        repo: Optional[str],
        launch_head: Optional[str],
        cwd: Optional[str],
        session_data: Optional[dict],
        interactive: Optional[bool],
    ) -> dict:
        git, git_error = _watch_repo_git(repo, launch_head)
        log_facts, signals = _watch_log_observation(log)
        return {
            "launch_error": (
                _read(state_dir / "launch_error") or _read(log_dir / "launch_error") or None
            ),
            "log": log_facts,
            "repo": repo,
            "git": git,
            "git_error": git_error,
            "signals": signals,
            "scratch": _watch_scratch_facts(cwd),
            # Match cmd_transcript's persistent cwd source for Claude lookup.
            "transcript": _watch_transcript_facts(
                session_data, _run_json_cwd(log_dir), state_dir, interactive
            ),
        }

    state_dir_state = _probe_dir_state(state_dir)

    if state_dir_state == _DIR_UNSTATABLE:
        # The state dir may well be there; we just cannot tell. Any status
        # emitted here would look terminal and hide a still-live run, so
        # raise into cmd_watch's never-raise guard, which emits the contract
        # with observation_error set instead.
        raise PermissionError(
            f"state directory for '{name}' is unreadable — "
            "run may still be live; cannot determine status"
        )

    if state_dir_state == _DIR_MISSING:
        # State dir gone, log survived: cmd_status's "not running (log
        # preserved)" case, usually a reboot. The `cwd` state file went with
        # it, so git facts need an explicit --repo and every process fact is
        # unknowable.
        session_data = _read_session_json(log_dir)
        payload = _watch_payload(
            name, observed_at, WATCH_STATUS_LOG_PRESERVED,
            **observed(repo_arg, None, repo_arg, session_data, None),
            session=session_data,
            hooks=_read_hooks_jsonl(log_dir),
        )
        _watch_emit(
            payload,
            as_json,
            f"name={name} status={payload['status']} lines={(payload['log'] or {}).get('lines')}",
        )
        return 0

    status = _watch_effective_status(state_dir)
    terminal = _watch_is_terminal(status)
    pid = _safe_int(_read(state_dir / "pid"))
    if pid is not None and pid <= 0:
        pid = None
    started_raw = _read(state_dir / "started_at") or None
    ended_raw = _read(state_dir / "ended_at") or None
    started_dt = _watch_parse_iso(started_raw)
    ended_dt = _watch_parse_iso(ended_raw)
    interactive_raw = _read(state_dir / "interactive")

    elapsed_s: Optional[float] = None
    if started_dt is not None:
        end_ref = ended_dt if (terminal and ended_dt is not None) else datetime.now(timezone.utc)
        elapsed_s = max(0.0, (end_ref - started_dt).total_seconds())

    repo_str = repo_arg or (_watch_read_cwd_file(state_dir / "cwd") or None)
    session_data = _read_session_json(log_dir)
    recorded_cwd = _watch_read_cwd_file(state_dir / "cwd") or None
    interactive = interactive_raw == "1" if interactive_raw in {"0", "1"} else None
    payload = _watch_payload(
        name,
        observed_at,
        status,
        exit_code=_safe_int(_read(state_dir / "exit_code")),
        pid=pid,
        interactive=interactive,
        started_at=started_raw,
        ended_at=ended_raw,
        elapsed_s=elapsed_s,
        session=session_data,
        hooks=_read_hooks_jsonl(log_dir),
        # repo_str is for git-fact attribution; recorded_cwd is for scratch so
        # that --repo (correcting which repo to inspect) does not silently move
        # the scratch scan off the run's launch directory.
        **observed(repo_str, _read(state_dir / "launch_head") or None, recorded_cwd, session_data, interactive),
    )
    _watch_emit(
        payload,
        as_json,
        f"name={name} status={status} pid={pid} elapsed_s={elapsed_s} terminal={terminal} "
        f"log_bytes={(payload['log'] or {}).get('bytes')} repo={repo_str}",
    )
    return 0


def _watch_log_facts_from_file(f, log: Path, st: os.stat_result) -> dict:
    delta = time.time() - st.st_mtime
    if delta < -WATCH_LOG_FUTURE_MTIME_TOLERANCE_SECONDS:
        mtime_age_s: Optional[float] = None
        growing = False
    else:
        mtime_age_s = max(0.0, delta)
        growing = mtime_age_s < WATCH_LOG_GROWING_MAX_AGE_SECONDS
    lines: Optional[int] = None
    if st.st_size <= WATCH_LINE_COUNT_MAX_BYTES:
        f.seek(0)
        remaining = st.st_size
        lines = 0
        last = b""
        while remaining:
            chunk = f.read(min(WATCH_TAIL_READ_BLOCK_BYTES, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            lines += chunk.count(b"\n")
            last = chunk[-1:]
        if st.st_size and last != b"\n":
            lines += 1
    return {
        "path": str(log),
        "bytes": st.st_size,
        "lines": lines,
        "mtime_age_s": mtime_age_s,
        "growing": growing,
    }


def _watch_tail_lines_from_file(f, end: int, n: int) -> List[str]:
    chunks: List[bytes] = []
    pos = end
    newline_count = 0
    bytes_read = 0
    while pos > 0 and newline_count <= n and bytes_read < WATCH_TAIL_MAX_BYTES:
        read_size = min(WATCH_TAIL_READ_BLOCK_BYTES, pos, WATCH_TAIL_MAX_BYTES - bytes_read)
        pos -= read_size
        f.seek(pos)
        chunk = f.read(read_size)
        newline_count += chunk.count(b"\n")
        bytes_read += len(chunk)
        chunks.append(chunk)
    data = b"".join(reversed(chunks))
    return data.decode("utf-8", errors="replace").splitlines()[-n:]


def _watch_log_observation(log: Optional[Path]) -> tuple[Optional[dict], dict]:
    with _watch_open_validated_log(log) as f:
        if f is None or log is None:
            return None, _watch_signals_from_lines([])
        try:
            st = os.fstat(f.fileno())
            facts = _watch_log_facts_from_file(f, log, st)
            lines = _watch_normalize_lines(
                _watch_tail_lines_from_file(f, st.st_size, WATCH_TAIL_LINES)
            )
        except OSError:
            return None, _watch_signals_from_lines([])
    return facts, _watch_signals_from_lines(lines)


def _watch_signals_from_lines(lines: List[str]) -> dict:
    distinct_files_read, top_repeated_read = _watch_read_signals(lines)
    return {
        "repeated_error": _watch_repeated_error(lines),
        "distinct_files_read": distinct_files_read,
        "top_repeated_read": top_repeated_read,
    }


# Logs are raw PTY captures of the wrapped TUI agent, which routinely enable
# terminal modes (mouse tracking, focus reporting, bracketed paste, hidden
# cursor) that only the live TUI's own shutdown sequence turns back off.
# Replaying those bytes onto the real terminal via `logs`/`tail` leaves the
# mode enabled after we're done printing, so e.g. mouse movement afterward
# shows up as garbage escape sequences. Force every such mode off (best
# effort; harmless if the mode was never on) once we stop writing to stdout.
_TERMINAL_MODE_RESET = (
    b"\x1b[0m\x1b[r\x1b[?7h"
    b"\x1b[?1000l\x1b[?1002l\x1b[?1003l\x1b[?1006l\x1b[?1015l"
    b"\x1b[?1004l\x1b[?2004l\x1b[?2031l\x1b[?25h"
    b"\x1b[?1049l\x1b[?1047l\x1b[?47l"
)


def _reset_terminal_modes() -> None:
    try:
        if not sys.stdout.isatty():
            return
        sys.stdout.buffer.write(_TERMINAL_MODE_RESET)
        sys.stdout.buffer.flush()
    except (AttributeError, OSError, ValueError):
        pass


def _discard_unflushable_stdout() -> None:
    """Point stdout at /dev/null after its real destination has died.

    CPython flushes stdout during interpreter shutdown. When the terminal is
    already gone that flush fails again, past any handler we control, and
    turns an otherwise clean exit into status 120 plus a stderr message.
    """
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
    except OSError:
        return
    try:
        os.dup2(devnull, sys.stdout.fileno())
    except (OSError, ValueError):
        pass
    finally:
        os.close(devnull)


def _is_peer_gone(exc: OSError) -> bool:
    """True when ``exc`` means the terminal on the other end is gone.

    EPIPE is a closed pipe reader; EIO is a PTY whose master has closed.
    Nothing else qualifies: ENOSPC (full disk) and EAGAIN (full non-blocking
    pipe) are also OSError subclasses, and treating them as peer-gone would
    exit 0 with silently truncated output where the shell expects a
    failure. Those propagate instead."""
    return exc.errno in (errno.EPIPE, errno.EIO)


_TERMINAL_DEATH_SIGNALS = (signal.SIGTERM, signal.SIGHUP, signal.SIGQUIT)


def _report_from_signal_handler(message: str) -> None:
    """Write ``message`` to stderr from inside a signal handler without
    ever blocking.

    A plain write is not safe here: if stderr is a pipe whose reader has
    stopped consuming, it blocks until drained, and a handler that hangs is
    worse than the silence it was added to replace -- the process never
    reaches the SIG_DFL re-raise and never dies. Write once to the raw fd
    with O_NONBLOCK set, restore the flag, and accept a truncated or
    dropped message as the cost of always returning."""
    try:
        fd = sys.stderr.fileno()
    except (AttributeError, ValueError, OSError):
        return
    try:
        flags = fcntl.fcntl(fd, fcntl.F_GETFL)
    except OSError:
        return
    try:
        fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
        os.write(fd, message.encode("utf-8", "replace"))
    except (OSError, ValueError):
        pass
    finally:
        try:
            fcntl.fcntl(fd, fcntl.F_SETFL, flags)
        except OSError:
            pass


@contextmanager
def _terminal_restored_on_signal_death(restore):
    """Run ``restore`` before dying from SIGTERM/SIGHUP/SIGQUIT.

    Those three default to killing the process outright, so neither the
    ``finally`` block nor the KeyboardInterrupt path ever runs and the
    terminal is left however the command left it -- raw mode for `attach`,
    and for both commands whatever DEC private modes (alt-screen, mouse
    tracking) the replayed PTY bytes turned on. The handler restores, puts
    the default disposition back and re-raises, so the exit status still
    reports death by that signal.
    """

    def handle(signum, _frame):
        # A failing restore must not cancel the signal: the process still
        # has to die from it, or a SIGTERM silently becomes exit 1 and the
        # caller's kill looks like it did nothing. It is still reported --
        # the terminal is left raw and/or on the alt screen, and the user
        # needs to know to run `reset`.
        try:
            restore()
        except Exception as exc:
            _report_from_signal_handler(
                f"agent-run: could not restore terminal modes ({exc}) -- "
                "run 'reset' if the terminal misbehaves\r\n"
            )
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    previous = {}
    for signum in _TERMINAL_DEATH_SIGNALS:
        # ValueError when not on the main thread; OSError for a signal this
        # platform refuses to trap. Either way the command still works, it
        # just loses this cleanup path.
        try:
            previous[signum] = signal.signal(signum, handle)
        except (OSError, ValueError):
            pass
    try:
        yield
    finally:
        for signum, handler in previous.items():
            try:
                signal.signal(signum, handler)
            except (OSError, ValueError):
                pass


def _tail_bytes(f: BinaryIO, n: int, *, keep_delimiters: bool = False) -> List[bytes]:
    """Return the last ``n`` lines of ``f``, walking backward from EOF.

    Reads 8 KiB blocks in reverse until n+1 newlines are seen or byte 0 is
    reached, collecting chunks in a list and joining once; repeated ``bytes``
    concatenation would be O(size**2). A log containing fewer than n+1
    newlines is therefore read in full — a single-line multi-megabyte log
    costs one whole-file read, and the caller's byte budget, not this
    function, is what bounds the emitted output.

    Lines are split on ``\\n`` only. With ``keep_delimiters`` false (the
    default), a trailing ``\\r`` is stripped from each segment -- terminal
    logs use ``\\r`` as an in-line cursor-home for progress redraws, not a
    line terminator -- and no terminator is added back. With
    ``keep_delimiters`` true, every segment keeps its original trailing
    ``\\n`` (or ``\\r\\n``) intact, and an unterminated final segment at EOF
    is returned with none added, so concatenating the result reproduces the
    corresponding tail of ``f`` byte for byte. Returns [] for n < 1.
    """
    if n < 1:
        return []
    f.seek(0, os.SEEK_END)
    pos = f.tell()
    chunks: List[bytes] = []
    newline_count = 0
    while pos > 0 and newline_count <= n:
        read_size = min(8192, pos)
        pos -= read_size
        f.seek(pos)
        chunk = f.read(read_size)
        chunks.append(chunk)
        newline_count += chunk.count(b"\n")
    pieces = b"".join(reversed(chunks)).split(b"\n")
    had_final_newline = pieces[-1] == b""
    if had_final_newline:
        pieces = pieces[:-1]  # trailing empty segment from a file-final \n
    if keep_delimiters:
        if had_final_newline:
            lines = [piece + b"\n" for piece in pieces]
        else:
            lines = [piece + b"\n" for piece in pieces[:-1]]
            if pieces:
                lines.append(pieces[-1])
    else:
        lines = [piece.rstrip(b"\r") for piece in pieces]
    return lines[-n:]


def _head_bytes(f: BinaryIO, n: int, *, keep_delimiters: bool = False) -> List[bytes]:
    """Return the first ``n`` lines of ``f``, scanning forward in 8 KiB reads.

    Each chunk is scanned once for ``\\n`` rather than re-scanning accumulated
    bytes, so cost is O(bytes read). Reading stops after the n-th complete line
    or at EOF; an unterminated final line at EOF counts as a line. A log whose
    first n lines span the whole file — again, the single-line multi-megabyte
    case — is read in full, and the caller's byte budget bounds the output.

    Same line grammar and ``keep_delimiters`` semantics as ``_tail_bytes``.
    Returns [] for n < 1.
    """
    if n < 1:
        return []
    lines: List[bytes] = []
    pending: List[bytes] = []  # bytes of the current incomplete line
    while len(lines) < n:
        chunk = f.read(8192)
        if not chunk:
            if pending:
                unterminated = b"".join(pending)
                lines.append(unterminated if keep_delimiters else unterminated.rstrip(b"\r"))
            break
        start = 0
        while start < len(chunk) and len(lines) < n:
            nl = chunk.find(b"\n", start)
            if nl == -1:
                pending.append(chunk[start:])
                start = len(chunk)
            else:
                end = nl + 1 if keep_delimiters else nl
                pending.append(chunk[start:end])
                terminated = b"".join(pending)
                lines.append(terminated if keep_delimiters else terminated.rstrip(b"\r"))
                pending = []
                start = nl + 1
    return lines


def _slice_str_lines(text: str, n: int, *, from_end: bool) -> str:
    """Return the last (``from_end``) or first ``n`` lines of ``text``, newline-terminated.

    Splits on ``\\n`` only so that ``\\r``, ``\\f``, ``\\v``, U+2028, and other
    Unicode line-break characters occurring inside a rendered transcript are
    neither treated as separators nor rewritten. Returns ``""`` for empty
    ``text`` or n < 1.
    """
    if n < 1 or not text:
        return ""
    parts = text.split("\n")
    if parts[-1] == "":
        parts = parts[:-1]  # trailing empty part from a final \n
    selected = parts[-n:] if from_end else parts[:n]
    return "".join(line + "\n" for line in selected)


# Terminal-control sequences are invisible in a terminal but consume tokens and
# corrupt matching for the LLM consumers that read `logs --plain` output.
_LOGS_OSC_RE = re.compile(rb"\x1b\].*?(?:\x07|\x1b\\)", re.DOTALL)
_LOGS_PARTIAL_OSC_HEAD_RE = re.compile(rb"\x1b\].*\Z", re.DOTALL)
_LOGS_PARTIAL_OSC_TAIL_RE = re.compile(rb"\A[^\x07]*(?:\x07|\x1b\\)", re.DOTALL)
# Parameter bytes are 0x30-0x3F: digits, `;`, `:`, and the private-parameter
# bytes `<=>?` (ECMA-48 5.4c). `\x1b[>4;1m` (xterm's modifyOtherKeys query
# reply) is a representative sequence using `>`.
_LOGS_CSI_RE = re.compile(rb"\x1b\[[0-9;:<=>?]*[ -/]*[@-~]")
_LOGS_ESC_RE = re.compile(rb"\x1b.", re.DOTALL)


def _strip_ansi_bytes(data: bytes) -> bytes:
    """Remove ANSI CSI, OSC, and bare ESC sequences from a byte string."""
    data = _LOGS_OSC_RE.sub(b"", data)
    data = _LOGS_PARTIAL_OSC_HEAD_RE.sub(b"", data)
    if b"\x1b]" not in data:
        data = _LOGS_PARTIAL_OSC_TAIL_RE.sub(b"", data)
    data = _LOGS_CSI_RE.sub(b"", data)
    return _LOGS_ESC_RE.sub(b"", data)


_TRUNCATION_MARKER_STR = "[agent-run: output truncated]\n"
_TRUNCATION_MARKER_BYTES = _TRUNCATION_MARKER_STR.encode("utf-8")

# OSC (ESC ]), DCS (ESC P), PM (ESC ^), and APC (ESC _) open a control string
# terminated by BEL or ST (ESC \). A byte-cut line can end between the
# introducer and its terminator; the receiving terminal then keeps consuming
# following bytes -- including the truncation marker -- as string payload.
_STRING_CONTROL_TERMINATED_RE = re.compile(rb"\x1b[\]P^_].*?(?:\x07|\x1b\\)", re.DOTALL)
_STRING_CONTROL_INTRODUCER_RE = re.compile(rb"\x1b[\]P^_]")
_STRING_TERMINATOR = b"\x1b\\"


def _close_unterminated_string_control(data: bytes) -> bytes:
    """Close a control string a byte cut left open, or drop a dangling
    introducer/ESC a cut left with no evidence of its own content.

    Every complete control string is removed first, so a surviving introducer
    in the remainder never reached its own terminator. One with at least one
    payload byte after it is known to be open -- the cut landed inside real
    content -- and gets ST (ESC \\) appended to close it without altering
    that content. An introducer with no payload byte after it, or a bare
    trailing ESC, asserts nothing about what the source would have sent next;
    appending ST there would invent an empty payload the source never wrote,
    so both are dropped instead.
    """
    remainder = _STRING_CONTROL_TERMINATED_RE.sub(b"", data)
    match = _STRING_CONTROL_INTRODUCER_RE.search(remainder)
    if match:
        if match.end() < len(remainder):
            return data + _STRING_TERMINATOR
        return data[: len(data) - (len(remainder) - match.start())]
    if data.endswith(b"\x1b"):
        return data[:-1]
    return data


def _cut_with_closure(content: bytes, cap: int) -> "tuple[bytes, bool]":
    """Cut ``content`` to at most ``cap`` bytes, accounting for a closing
    terminator ``_close_unterminated_string_control`` may append.

    A first cut at ``cap`` is closed; if that closure appended ST (ESC \\),
    the result is ``len(_STRING_TERMINATOR)`` bytes over budget, so the cut
    point is moved back by that many bytes and closed again. Closing can only
    append those same two bytes or drop a dangling ESC/introducer, never grow
    the result further, so the second pass lands at or under ``cap``. Returns
    (result, was_cut).
    """
    if len(content) <= cap:
        return content, False
    cut = content[:cap]
    closed = _close_unterminated_string_control(cut)
    if len(closed) > cap:
        reserve = len(_STRING_TERMINATOR)
        cut = content[: max(cap - reserve, 0)]
        closed = _close_unterminated_string_control(cut)
    return closed, True


def _budget_bytes_lines(
    lines: "Iterable[bytes]", *, keep_delimiters: bool = False
) -> "tuple[bytes, bool]":
    """Apply LOGS_MAX_LINE_BYTES / LOGS_MAX_TOTAL_BYTES to raw log lines.

    With ``keep_delimiters`` false (the default, used by ``--plain``), each
    input ``line`` is content with no terminator; LOGS_MAX_LINE_BYTES caps
    that content, and this function appends the ``\\n`` each emitted line
    is billed for. With ``keep_delimiters`` true (used by raw mode), each
    input ``line`` already carries whatever terminator
    ``_tail_bytes``/``_head_bytes`` preserved -- ``\\n``, ``\\r\\n``, or none
    for an unterminated final segment -- and LOGS_MAX_LINE_BYTES caps the
    whole segment; nothing is added.

    LOGS_MAX_TOTAL_BYTES bounds every byte this function emits: a content cut
    leaves one byte of headroom for the synthesized newline, reaching the
    total stops emission before starting a line with no headroom left, and
    the per-line cap folds into the same bound via ``min(LOGS_MAX_LINE_BYTES,
    content_room)`` so one call to ``_cut_with_closure`` accounts for both.
    Any ST a cut needs (see ``_close_unterminated_string_control``) is billed
    inside that same cap; it never pushes emitted bytes past either limit.
    The truncation marker the caller appends on a true return is not
    accounted here and falls outside both caps. Returns (output, truncated).
    """
    out_parts: "list[bytes]" = []
    total = 0
    truncated = False
    for line in lines:
        remaining = LOGS_MAX_TOTAL_BYTES - total
        content_room = remaining if keep_delimiters else remaining - 1
        if content_room < 0:
            truncated = True
            break
        cap = min(LOGS_MAX_LINE_BYTES, content_room)
        line, was_cut = _cut_with_closure(line, cap)
        if was_cut:
            truncated = True
        if keep_delimiters:
            out_parts.append(line)
            total += len(line)
        else:
            out_parts.append(line + b"\n")
            total += len(line) + 1
    return b"".join(out_parts), truncated


def _budget_str(text: str) -> "tuple[str, bool]":
    """Apply LOGS_MAX_LINE_BYTES / LOGS_MAX_TOTAL_BYTES to a rendered transcript.

    Counterpart of ``_budget_bytes_lines`` for the ``str`` produced by
    ``_render_log``. Kept separate rather than merged: caps here are measured on
    the UTF-8 encoding and a cut that lands mid-code-point is dropped by
    ``errors="ignore"``, so an accepted line can bill fewer bytes than the cap it
    was cut at. Routing this through the bytes version would change that
    accounting and therefore the truncation point.

    LOGS_MAX_LINE_BYTES caps line content; LOGS_MAX_TOTAL_BYTES bounds every
    byte this function emits, including the appended ``\\n``, so content is
    accepted only up to one byte less than the remaining total. The
    truncation marker the caller appends on a true return is not accounted
    here and falls outside both caps.

    Returns (output, truncated); output stays newline-terminated.
    """
    stripped = text.rstrip("\n")
    lines = stripped.split("\n") if stripped else []
    out_parts: "list[str]" = []
    total = 0
    truncated = False
    for line in lines:
        line_bytes = line.encode("utf-8")
        if len(line_bytes) > LOGS_MAX_LINE_BYTES:
            line_bytes = line_bytes[:LOGS_MAX_LINE_BYTES]
            line = line_bytes.decode("utf-8", errors="ignore")
            truncated = True
        remaining = LOGS_MAX_TOTAL_BYTES - total
        content_room = remaining - 1
        if content_room < 0:
            truncated = True
            break
        if len(line_bytes) > content_room:
            line_bytes = line_bytes[:content_room]
            line = line_bytes.decode("utf-8", errors="ignore")
            truncated = True
        out_parts.append(line + "\n")
        total += len(line.encode("utf-8")) + 1
    return "".join(out_parts), truncated


def cmd_logs(args: argparse.Namespace) -> int:
    log = _require_log(_validate_run_name(args.name))
    clean = bool(getattr(args, "clean", False))
    try:
        if clean:
            # Dumb and synchronous: read the whole log, render it inline
            # through pyte, then slice. A byte-sliced fragment starts
            # mid-escape-sequence, so head/tail apply to the rendered text.
            with _watch_open_validated_log(log) as f:
                if f is None:
                    sys.exit(f"agent-run: cannot open log for '{args.name}' (not a regular file)")
                raw = f.read()
            try:
                width, height = _resolved_default_geometry(log.parent)
                rendered = _render_log(
                    raw, width=width, height=height,
                    history=_RENDER_LOG_DEFAULT_HISTORY,
                    resizes=_read_resize_timeline(log.parent),
                )
            except RenderDependencyError as exc:
                sys.exit(str(exc))
            if args.head is not None:
                rendered = _slice_str_lines(rendered, args.head, from_end=False)
            else:
                rendered = _slice_str_lines(rendered, args.tail if args.tail is not None else 50, from_end=True)
            budgeted, truncated = _budget_str(rendered)
            if truncated:
                budgeted += _TRUNCATION_MARKER_STR
            try:
                sys.stdout.buffer.write(budgeted.encode("utf-8"))
            except BrokenPipeError:
                pass
            return 0

        plain = bool(getattr(args, "plain", False))
        with _watch_open_validated_log(log) as f:
            if f is None:
                sys.exit(f"agent-run: cannot open log for '{args.name}' (not a regular file)")
            if args.head is not None:
                lines = _head_bytes(f, args.head, keep_delimiters=not plain)
            else:
                lines = _tail_bytes(
                    f, args.tail if args.tail is not None else 50, keep_delimiters=not plain
                )
        # Default is raw bytes (a human piping to a terminal gets the drawing
        # back, `\r` mid-line and all, since it is how the PTY drove a
        # progress redraw); --plain strips ANSI and normalizes line endings
        # for grepping/piping/agent consumption. The byte budget applies
        # either way, so raw mode spends part of its budget on escape
        # sequences rather than visible text.
        if plain:
            lines = (_strip_ansi_bytes(line) for line in lines)
        out, truncated = _budget_bytes_lines(lines, keep_delimiters=not plain)
        try:
            sys.stdout.buffer.write(out)
            if truncated:
                sys.stdout.buffer.write(_TRUNCATION_MARKER_BYTES)
        except BrokenPipeError:
            pass
    finally:
        _reset_terminal_modes()
    return 0


def cmd_transcript(args: argparse.Namespace) -> int:
    """Print a run's harness transcript.

    A readable empty store exits zero with blank stdout. Missing session
    metadata or an unavailable store exits nonzero.
    """

    transcript = _agent_run_transcript

    name = _validate_run_name(args.name)
    log_dir = _log_dir(name)
    if not _known(name):
        sys.exit(f"agent-run: no run named '{name}' in {STATE_ROOT} or {LOG_ROOT}")
    session = _read_session_json(log_dir)
    if session is None:
        sys.exit(
            f"agent-run: no transcript for '{name}' (raw run, no session.json) -- "
            f"relaunch under --harness <name> for a transcript, or use 'agent-run logs {name} --clean'"
        )
    session_id = session.get("session_id")
    harness = session.get("harness")
    if not isinstance(session_id, str) or not session_id or not isinstance(harness, str):
        sys.exit(
            f"agent-run: no transcript for '{name}' (session id was never acquired) -- "
            f"relaunch under --harness <name> for a transcript, or use 'agent-run logs {name} --clean'"
        )

    cwd = _run_json_cwd(log_dir)
    try:
        entries, skipped = transcript.read_transcript(harness, session_id, cwd)
    except transcript.TranscriptSourceError as exc:
        sys.exit(f"agent-run: {exc}")

    if not entries:
        store = {
            "opencode": transcript.OPENCODE_DB_PATH,
            "claude": transcript.CLAUDE_PROJECTS_DIR,
            "codex": transcript.CODEX_SESSIONS_DIR,
        }.get(harness, "?")
        skipped_note = f", {skipped} record(s) skipped as unparseable" if skipped else ""
        print(
            f"agent-run: transcript for '{name}' ({harness}, session {session_id}) is empty "
            f"(store: {store}{skipped_note})",
            file=sys.stderr,
        )
        return 0

    if args.head is not None:
        entries = entries[: args.head]
    else:
        entries = entries[-(args.tail if args.tail is not None else 50):]

    if getattr(args, "json", False):
        out = "".join(json.dumps(e.to_dict()) + "\n" for e in entries)
        try:
            sys.stdout.buffer.write(out.encode("utf-8"))
        except BrokenPipeError:
            pass
    else:
        rendered = transcript.render_text(entries)
        budgeted, truncated = _budget_str(rendered)
        if truncated:
            budgeted += _TRUNCATION_MARKER_STR
        try:
            sys.stdout.buffer.write(budgeted.encode("utf-8"))
        except BrokenPipeError:
            pass

    if skipped:
        print(f"agent-run: transcript: skipped {skipped} unparseable record(s)", file=sys.stderr)
    return 0


def cmd_tail(args: argparse.Namespace) -> int:
    name = _validate_run_name(args.name)
    log = _require_log(name)
    pid_raw = _read(_state_dir(name) / "pid")
    try:
        pid = int(pid_raw) if pid_raw else None
    except ValueError:
        pid = None

    def emit(data: bytes) -> bool:
        """Write replayed log bytes to stdout. False means the terminal peer
        is gone (closed window, dropped SSH: EIO on a PTY, EPIPE on a pipe)
        and tail should exit rather than keep looping against a dead fd.

        Any other OSError -- a full disk, a full non-blocking pipe --
        propagates: exiting 0 with silently truncated output would hide a
        real write failure that the shell expects to see as a failure."""
        try:
            sys.stdout.buffer.write(data)
            sys.stdout.buffer.flush()
        except OSError as exc:
            if not _is_peer_gone(exc):
                raise
            _discard_unflushable_stdout()
            return False
        return True

    # Stream the whole file then tail until the agent dies or log stops growing.
    # Ctrl-C is the normal way to stop following (same as `tail -f`): catch it
    # here so it exits quietly with the conventional 128+SIGINT status instead
    # of dumping a traceback, while still running the terminal-mode reset.
    try:
        with _terminal_restored_on_signal_death(_reset_terminal_modes):
            with log.open("rb") as f:
                while True:
                    chunk = f.read(8192)
                    if chunk:
                        if not emit(chunk):
                            return 0
                        continue
                    # EOF. A preserved-log-only or otherwise non-live run has
                    # nothing left to follow; after printing existing content,
                    # exit immediately.
                    if pid is None:
                        return 0
                    if not _pid_alive(pid):
                        # One more drain to catch final writes from a process
                        # that was live when tail started.
                        time.sleep(0.1)
                        remaining = f.read()
                        if remaining:
                            emit(remaining)
                        return 0
                    time.sleep(0.2)
    except KeyboardInterrupt:
        return 128 + signal.SIGINT
    finally:
        _reset_terminal_modes()


# Terminals that negotiate the "disambiguate escape codes" keyboard protocol
# encode Ctrl-C as a CSI sequence instead of the raw 0x03 byte -- Claude
# Code's TUI requests this protocol (`ESC[>1u`) on startup, and terminals
# that honor it (iTerm2, kitty, wezterm, ghostty, ...) then send every
# modified/disambiguated key, including Ctrl-C, as an escape sequence rather
# than a legacy control byte. A plain search for 0x03 misses this entirely,
# so the keystroke leaks straight through attach into the wrapped agent's
# stdin instead of detaching. Recognize both encodings: the legacy xterm
# "modifyOtherKeys" form (`ESC[27;<mod>;99~`) and the kitty-native CSI-u
# form (`ESC[99;<mod>u`). <mod> is 1 + a bitmask of held modifiers
# (Shift=1, Alt=2, Ctrl=4, ...); 99 is the codepoint for 'c'. Only trigger
# when the Ctrl bit is set.
#
# Every parameter accepts colon-separated subparameters (alternate key
# codes, press/repeat/release event type, associated text) and an
# unbounded modifier value: terminals emit those even when the client
# requested only the base protocol flag, and an empty subparameter means
# "use the default" per ECMA-48. A stricter match would silently miss
# conforming input and reintroduce the leak-through this detector prevents.
_CTRL_C_CSI_RE = re.compile(
    rb"\x1b\[(?:27(?::\d*)*;(\d+)(?::\d*)*;99(?::\d*)*~"
    rb"|99(?::\d*)*;(\d+)(?::\d*)*(?:;[\d:]*)?u)"
)

# Bracketed-paste delimiters. Everything between them is pasted content,
# never keystrokes, so no detach trigger inside them counts.
_PASTE_START = b"\x1b[200~"
_PASTE_END = b"\x1b[201~"

# Upper bound on a single bracketed paste before the in-paste state gives
# up and treats input as typed again. Without it, an ESC[201~ that never
# arrives (paste aborted, terminal reset, emulator drops the closer) makes
# Ctrl-C unreachable for the rest of the attach session -- and Ctrl-C is
# the only documented way out. Far above any realistic paste, so a genuine
# one is never cut short.
_MAX_PASTE_BYTES = 8 * 1024 * 1024

# Idle bound on the same state: how long the in-paste latch may sit with no
# further payload before giving up. A terminal writes a paste as a
# continuous burst, so a gap this long with no closing marker means the
# paste is never going to finish -- and unlike the byte budget, this
# recovers an aborted paste that sent very little data, which is the
# common case (Ctrl-C mid-paste, tmux dropping the closer).
#
# Idle rather than total duration: a large paste over a slow link trickles
# in for far longer than this, and expiring mid-stream would rescan the
# remaining payload as typed input, so an embedded 0x03 would truncate it.
_MAX_PASTE_IDLE_SECONDS = 5.0

# Ceiling on the whole latch, measured from the opening marker regardless of
# how busy the stream stays. The idle bound alone cannot end a paste that
# never stops receiving bytes -- and inside a paste every byte is payload by
# definition, so the Ctrl-C a user presses to escape a stuck paste is itself
# what renews the idle window. Without this, input arriving more often than
# once per _MAX_PASTE_IDLE_SECONDS pins the latch open for the rest of the
# session and pressing the escape key harder only holds it tighter.
#
# Anchored to the byte budget rather than guessed: _MAX_PASTE_BYTES at
# ~15 KB/s -- slower than any link that would still be usable
# interactively -- arrives in about nine minutes, so a paste that could
# legitimately complete within the byte cap always fits inside this. Past it
# the payload could not have been a real paste, and an escape the user can
# actually reach matters more than bracketing a paste that will never end.
_MAX_PASTE_TOTAL_SECONDS = 600.0

# How long a partial paste marker is held waiting for its remaining bytes.
# Longer than _ESCAPE_HOLD_TIMEOUT_SECONDS because releasing one of these
# as literal input destroys the marker outright, whereas releasing an
# ordinary escape prefix merely forwards a keystroke slightly late.
_PASTE_MARKER_HOLD_SECONDS = 2.0

# Longest trailing unterminated CSI prefix held back waiting for its
# terminator. Comfortably above any real single-keystroke sequence
# (including the kitty form's full subparameter set), so input is never
# withheld indefinitely by bytes that will never resolve into one.
_MAX_PENDING_ESCAPE_BYTES = 64

# How long a possibly-incomplete escape sequence is held back waiting for
# more bytes before being flushed through as literal input. A CSI sequence
# split across reads must not be torn, but a bare ESC keypress (Alt-prefix,
# or Escape itself, which TUIs use to cancel) never gets follow-up bytes.
# Sized to absorb SSH round-trip latency and a loaded scheduler -- too
# short and a split CSI Ctrl-C leaks into the agent instead of detaching --
# while staying below the point where a solitary Escape feels delayed.
_ESCAPE_HOLD_TIMEOUT_SECONDS = 0.3

# FIFO writes of at most PIPE_BUF bytes are atomic, so bounding each write
# keeps concurrent attach clients from interleaving bytes mid-sequence.
_FIFO_ATOMIC_WRITE_BYTES = getattr(select, "PIPE_BUF", 4096)


def _find_ctrl_c_trigger(data: bytes) -> tuple:
    """Return the ``(start, end)`` byte range of the earliest Ctrl-C detach
    trigger in ``data``, or ``(-1, -1)`` if there is none. Recognizes the
    raw 0x03 byte and the CSI forms matched by _CTRL_C_CSI_RE.

    Earliest wins because every byte before the trigger has already been
    committed to forwarding: picking a later match would forward an
    earlier, intact trigger sequence into the agent's stdin as a side
    effect of detaching."""
    best_start = -1
    best_end = -1
    raw_idx = data.find(b"\x03")
    if raw_idx != -1:
        best_start, best_end = raw_idx, raw_idx + 1
    for m in _CTRL_C_CSI_RE.finditer(data):
        if not (int(m.group(1) or m.group(2)) - 1) & 4:
            continue
        # finditer runs left to right, so this first Ctrl-held match is
        # already the earliest CSI trigger; only the raw byte can precede it.
        if best_start == -1 or m.start() < best_start:
            best_start, best_end = m.start(), m.end()
        break
    return best_start, best_end


def _split_trailing_incomplete_escape(data: bytes) -> tuple:
    """Split ``data`` into ``(forwardable, held_back)`` so a CSI sequence
    straddling two reads from the local terminal is never torn: a trailing
    ESC that has not yet received its 0x40-0x7E terminator is held back to
    be prepended to the next read.

    Only a genuinely unterminated trailing sequence is held. A complete
    sequence is forwarded whatever its length, and an over-long
    unterminated run (past _MAX_PENDING_ESCAPE_BYTES, so not a real
    keystroke) is forwarded rather than held -- dropping local input is
    never an acceptable outcome, since the bytes in question are usually a
    paste payload rather than an escape sequence at all."""
    esc_at = data.rfind(b"\x1b")
    if esc_at == -1:
        return data, b""
    tail = data[esc_at:]
    if len(tail) == 1:
        return data[:esc_at], tail
    if tail[1:2] != b"[":
        return data, b""
    # CSI parameter/intermediate bytes are 0x20-0x3F; a byte in 0x40-0x7E
    # terminates the sequence.
    if any(0x40 <= byte <= 0x7E for byte in tail[2:]):
        return data, b""
    if len(tail) > _MAX_PENDING_ESCAPE_BYTES:
        return data, b""
    return data[:esc_at], tail


def _is_paste_marker_prefix(data: bytes) -> bool:
    """True when ``data`` could still grow into ``ESC[200~``/``ESC[201~``
    and is long enough to be distinctive.

    Releasing such a prefix as literal input destroys the marker: the next
    read then starts mid-sequence, so the paste bracket is never recognized
    and the payload is scanned for detach triggers as if it were typed.

    ``ESC`` and ``ESC[`` are excluded deliberately. Both are far more often
    the Escape key or an arrow key -- neither of which gets follow-up bytes
    -- than the first bytes of a split marker, and holding them for the
    longer paste window would make those keys feel stuck. From ``ESC[2`` on
    no competing single keystroke exists, so the extended hold is
    unambiguous. The two excluded split points are covered instead by
    _resync_paste_marker on the following read.
    """
    if len(data) < 3:
        return False
    return any(
        marker.startswith(data) for marker in (_PASTE_START, _PASTE_END)
    )


def _resync_paste_marker(data: bytes, released: bytes) -> tuple:
    """Re-attach ``released`` to ``data`` when the two together start a
    paste marker that the hold timer split apart. Returns
    ``(new_data, consumed)`` -- ``consumed`` is True only when ``released``
    was actually rejoined, so the caller knows whether it still owes those
    bytes to the agent.

    The flag has to be reported rather than inferred from ``new_data``: a
    following read that independently starts with ESC looks exactly like a
    rejoined prefix, and treating it as one drops the held byte outright.

    ``ESC`` and ``ESC[`` are released after the ordinary escape timeout so
    Escape and the arrow keys stay responsive. When the bytes that follow
    turn out to complete a paste marker, the marker has to be reassembled
    before scanning or the payload is treated as typed input -- which is
    what makes a `0x03` inside it detach mid-paste."""
    if not released or not data:
        return data, False
    joined = released + data
    for marker in (_PASTE_START, _PASTE_END):
        if joined.startswith(marker) or marker.startswith(joined):
            return joined, True
    return data, False


def _release_expired_held_escape(
    held_escape: bytes, held_escape_deadline: Optional[float]
) -> tuple:
    """If ``held_escape`` has sat unresolved past ``held_escape_deadline``
    with no follow-up bytes, release it. Returns
    ``(new_held_escape, new_held_escape_deadline, released_bytes)`` --
    ``released_bytes`` is empty unless the deadline has actually passed.

    A prefix that could still become a bracketed-paste marker is held past
    the deadline rather than released, up to _PASTE_MARKER_HOLD_SECONDS:
    the terminal writes those markers in one burst, so a partial one means
    the rest is in flight, and releasing it is what turns a split marker
    into a spurious mid-paste detach."""
    if not held_escape or held_escape_deadline is None:
        return held_escape, held_escape_deadline, b""
    now = time.monotonic()
    if now < held_escape_deadline:
        return held_escape, held_escape_deadline, b""
    if _is_paste_marker_prefix(held_escape) and now < (
        held_escape_deadline - _ESCAPE_HOLD_TIMEOUT_SECONDS + _PASTE_MARKER_HOLD_SECONDS
    ):
        return held_escape, held_escape_deadline, b""
    return b"", None, held_escape


def _scan_local_input_for_detach(
    data: bytes,
    in_paste: bool = False,
    paste_bytes: int = 0,
    paste_idle_since: Optional[float] = None,
    paste_started: Optional[float] = None,
) -> tuple:
    """Given raw bytes read from the local terminal (with any previously
    held-back escape prefix already merged in by the caller), decide what to
    forward. Returns ``(forwardable, new_held_escape, detached,
    new_in_paste, new_paste_bytes, new_paste_idle_since,
    new_paste_started)``.

    Detach triggers are recognized only outside a bracketed paste: between
    ``ESC[200~`` and ``ESC[201~`` the terminal is transmitting pasted
    content, so a 0x03 byte or the literal text of a CSI Ctrl-C sequence is
    data, not a keypress, and must be forwarded like any other byte. On a
    trigger, ``forwardable`` is everything strictly before it -- neither the
    trigger nor anything racing in behind it is forwarded.

    The in-paste state is bounded three ways, because an ``ESC[201~`` that
    never arrives would otherwise make Ctrl-C unreachable for the rest of
    the session and Ctrl-C is the only documented way out.

    ``paste_bytes`` caps total payload. ``paste_idle_since`` caps the gap
    since payload last arrived, tracking idleness rather than total duration
    so a large payload trickling in over a slow link -- which can
    legitimately take minutes -- is never cut off mid-stream.
    ``paste_started`` caps the latch's whole lifetime: the idle bound alone
    cannot end a paste that keeps receiving bytes, and since every byte
    inside a paste is payload, a user pressing Ctrl-C to escape a stuck
    paste renews the idle window with each press. The ceiling is what makes
    the escape reachable in bounded time no matter how busy the stream is."""
    forwardable, held_escape = _split_trailing_incomplete_escape(data)
    now = time.monotonic()
    if in_paste and (
        (
            paste_idle_since is not None
            and now - paste_idle_since > _MAX_PASTE_IDLE_SECONDS
        )
        or (
            paste_started is not None
            and now - paste_started > _MAX_PASTE_TOTAL_SECONDS
        )
    ):
        in_paste, paste_bytes = False, 0
        paste_idle_since, paste_started = None, None
    pos = 0
    while pos < len(forwardable):
        if in_paste:
            end = forwardable.find(_PASTE_END, pos)
            if end != -1:
                pos = end + len(_PASTE_END)
                in_paste, paste_bytes = False, 0
                paste_idle_since, paste_started = None, None
                continue
            paste_bytes += len(forwardable) - pos
            # Payload arrived, so the paste is still live: restart the idle
            # window rather than letting it run from the opening marker.
            # paste_started deliberately keeps its original value -- the
            # ceiling must not be renewable, or it would be a second idle
            # bound rather than a bound on the whole latch.
            paste_idle_since = now
            if paste_bytes <= _MAX_PASTE_BYTES:
                break
            # Latch expired: fall through and rescan the tail as typed
            # input so the detach key works again.
            in_paste, paste_bytes = False, 0
            paste_idle_since, paste_started = None, None
            continue
        paste_at = forwardable.find(_PASTE_START, pos)
        trigger_at, _trigger_end = _find_ctrl_c_trigger(forwardable[pos:])
        if trigger_at != -1 and (paste_at == -1 or pos + trigger_at < paste_at):
            return (
                forwardable[: pos + trigger_at],
                b"",
                True,
                in_paste,
                paste_bytes,
                paste_idle_since,
                paste_started,
            )
        if paste_at == -1:
            break
        pos = paste_at + len(_PASTE_START)
        in_paste, paste_bytes = True, 0
        paste_idle_since, paste_started = now, now
    return (
        forwardable,
        held_escape,
        False,
        in_paste,
        paste_bytes,
        paste_idle_since,
        paste_started,
    )


def _append_bounded(pending: bytes, extra: bytes, warn) -> bytes:
    """Append ``extra`` to ``pending``, discarding it and reporting via
    ``warn`` if it would push the buffer past MAX_PTY_INPUT_BUFFER.

    The bound exists because a paste larger than the wrapped agent can
    consume would otherwise grow this buffer without limit. Dropping the
    overflowing chunk with a visible warning beats an unhandled exception
    dumping a traceback over the user's terminal mid-session."""
    if len(pending) + len(extra) > MAX_PTY_INPUT_BUFFER:
        warn(
            f"input buffer full ({MAX_PTY_INPUT_BUFFER} bytes) -- "
            f"discarded {len(extra)} bytes of input"
        )
        return pending
    return pending + extra


def _flush_fifo_write(fd: int, data: bytes) -> int:
    """Write up to _FIFO_ATOMIC_WRITE_BYTES of ``data`` to a non-blocking
    FIFO and return the byte count actually written.

    Bounding each write to PIPE_BUF keeps the write itself atomic, so two
    attach clients cannot tear each other apart *within* one write. It is
    not a whole-keystroke guarantee: a pending buffer larger than PIPE_BUF
    is delivered in several writes, and another client can interleave
    between them, so a long escape sequence can still be split. Concurrent
    typing from multiple clients is documented as unreliable for exactly
    this reason.
    """
    try:
        return os.write(fd, data[:_FIFO_ATOMIC_WRITE_BYTES])
    except BlockingIOError:
        return 0


_DETACH_FLUSH_TIMEOUT_SECONDS = 1.0


def _drain_fifo_write(fd: int, data: bytes) -> bytes:
    """Write all of ``data`` to a non-blocking FIFO before detaching,
    returning whatever could not be delivered within
    _DETACH_FLUSH_TIMEOUT_SECONDS.

    The bytes typed before Ctrl-C are already committed to the agent, so a
    single PIPE_BUF-sized write is not enough -- but the reader may be
    backpressured, and detach must not hang waiting for it."""
    deadline = time.monotonic() + _DETACH_FLUSH_TIMEOUT_SECONDS
    while data:
        written = _flush_fifo_write(fd, data)
        if written:
            data = data[written:]
            continue
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        select.select([], [fd], [], remaining)
    return data


@contextmanager
def _attach_presence_lock(state_dir: Path):
    """Serialize presence registration, the emptiness test, and the restore
    write for one run.

    Without it a client can register between another's unlink and its
    directory scan, so the departing client sends the launch geometry after
    the new client has already published its own size and the final geometry
    becomes scheduling-dependent.
    """
    lock_path = state_dir / "attached.lock"
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield fd
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(fd)


def _register_attach_presence(state_dir: Path) -> Optional[int]:
    """Create ``<state_dir>/attached/<pid>`` and hold an exclusive lock on it
    for the lifetime of this attach, returning the open descriptor.

    The lock, not the file, is what marks this client present: the kernel
    releases it when the process exits for any reason, so a client killed with
    SIGKILL leaves a file that other clients can identify as stale rather than
    a marker that suppresses the detach restore forever. A PID is only a
    filename here, so PID reuse cannot make a dead client look alive.

    Returns ``None`` when the marker cannot be created or locked; the client
    then attaches normally but is not counted, which at worst skips a restore.
    """
    try:
        presence_dir = state_dir / "attached"
        presence_dir.mkdir(exist_ok=True)
        fd = os.open(presence_dir / str(os.getpid()), os.O_RDWR | os.O_CREAT, 0o600)
    except OSError:
        return None
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        return None
    return fd


def _attach_presence_is_empty(state_dir: Path, own_pid: int) -> bool:
    """True when no live client other than ``own_pid`` holds a presence marker.

    A marker whose lock can be taken belongs to a process that has exited
    without unlinking it; it is removed here rather than left to suppress every
    future restore.
    """
    presence_dir = state_dir / "attached"
    try:
        entries = list(presence_dir.iterdir())
    except OSError:
        return False
    for entry in entries:
        if entry.name == str(own_pid):
            continue
        try:
            fd = os.open(entry, os.O_RDWR)
        except OSError:
            continue
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return False  # still held: that client is alive
        else:
            try:
                entry.unlink()
            except OSError:
                pass
        finally:
            os.close(fd)
    return True


def _deregister_attach_presence_and_restore(
    name: str, state_dir: Path, marker_fd: Optional[int], resize_fd: Optional[int]
) -> None:
    """Drop this client's presence marker and, if it was the last live one,
    restore the run's PTY to its launch-time geometry.

    A resize is last-writer-wins and out-of-band (TIOCSWINSZ + SIGWINCH): once
    every client has detached, the PTY otherwise stays at whatever size the last
    one left it, and every byte the wrapped agent writes afterward is at a
    geometry nobody recorded. One final resize record carrying run.json's
    ``terminal`` geometry closes that gap; a run launched by an older build has
    no recorded geometry, so nothing is sent.

    The whole sequence runs under ``_attach_presence_lock`` so a client
    registering concurrently either counts as present here or publishes its own
    size after this restore, never both.
    """
    if marker_fd is None:
        return
    own_pid = os.getpid()
    try:
        with _attach_presence_lock(state_dir):
            try:
                (state_dir / "attached" / str(own_pid)).unlink()
            except OSError:
                pass
            os.close(marker_fd)
            if resize_fd is None or not _attach_presence_is_empty(state_dir, own_pid):
                return
            geometry = _read_run_terminal_geometry(_log_dir(name))
            if geometry is None:
                return
            try:
                _drain_fifo_write(resize_fd, _pack_resize(*geometry))
            except OSError:
                pass
    except OSError:
        try:
            os.close(marker_fd)
        except OSError:
            pass


def cmd_attach(args: argparse.Namespace) -> int:
    name = _validate_run_name(args.name)
    state_dir, pid = _require_live_interactive_run(name)
    log = _require_log(name)
    stdin_path = state_dir / "stdin"
    resize_path = state_dir / "resize"
    if not stdin_path.is_fifo():
        sys.exit(f"agent-run: no stdin FIFO at {stdin_path}")
    # A session launched before resize support has no resize FIFO. Fall
    # back to keyboard-only passthrough rather than refusing to attach to
    # an otherwise-live run.
    has_resize = resize_path.is_fifo()
    if not has_resize:
        print(
            f"agent-run: no resize FIFO at {resize_path} (session predates "
            "resize support) -- attaching without resize forwarding",
            file=sys.stderr,
        )
    elif not _resize_protocol_matches(state_dir):
        # A runner from another release reads a different record layout and
        # would discard (or worse, mis-decode) what this client writes.
        has_resize = False
        print(
            "agent-run: this session's runner speaks a different resize "
            f"record format (this build speaks version {RESIZE_RECORD_VERSION}) "
            "-- attaching without resize forwarding",
            file=sys.stderr,
        )

    if not sys.stdin.isatty():
        sys.exit(
            "agent-run: attach requires an interactive terminal "
            "(use 'agent-run tail' instead)"
        )
    local_fd = sys.stdin.fileno()
    saved_termios = termios.tcgetattr(local_fd)
    try:
        stdin_fd = os.open(str(stdin_path), os.O_WRONLY | os.O_NONBLOCK)
    except OSError as exc:
        sys.exit(f"agent-run: failed to open stdin FIFO at {stdin_path}: {exc}")
    resize_fd: Optional[int] = None
    if has_resize:
        try:
            resize_fd = os.open(str(resize_path), os.O_WRONLY | os.O_NONBLOCK)
        except OSError as exc:
            os.close(stdin_fd)
            sys.exit(f"agent-run: failed to open resize FIFO at {resize_path}: {exc}")
    # Presence registration: once every client has detached, the last one
    # restores the run's PTY to its launch geometry (see
    # _deregister_attach_presence_and_restore). Registering under the presence
    # lock keeps it ordered against a concurrent last-detach, and the try/finally
    # below drops the marker on Ctrl-C, an error, and normal exit alike.
    with _attach_presence_lock(state_dir):
        attach_marker_fd = _register_attach_presence(state_dir)
    pending_input = b""
    pending_resize = b""
    resize_requested = True
    last_sent_size: Optional[tuple] = None
    winch_installed = False
    previous_winch = None
    held_escape = b""
    held_escape_deadline: Optional[float] = None
    in_paste = False
    paste_bytes = 0
    paste_idle_since: Optional[float] = None
    paste_started: Optional[float] = None
    # A released escape prefix is forwarded immediately and kept here only
    # so the next read can be rescanned as the paste marker it may complete;
    # already_sent counts the leading bytes of that rescan the agent has
    # therefore already received.
    resync_prefix = b""
    resync_deadline = 0.0
    already_sent = 0

    def restore_terminal() -> None:
        try:
            termios.tcsetattr(local_fd, termios.TCSADRAIN, saved_termios)
        except termios.error:
            # The controlling terminal itself is already gone (window closed,
            # SSH dropped): tcsetattr on the now-invalid fd raises, and must
            # not stop _reset_terminal_modes from running -- otherwise DEC
            # private modes the replayed PTY bytes turned on (alt-screen,
            # mouse tracking) stay stuck enabled.
            pass
        _reset_terminal_modes()

    def emit(data: bytes) -> bool:
        """Write replayed PTY bytes to the local terminal. False means the
        terminal peer is gone (EIO on a closed PTY, EPIPE on a pipe), so
        attach should return through its normal cleanup rather than let the
        exception unwind past the terminal-restore ordering in `finally`.

        Discards whatever is still buffered on the way out: CPython flushes
        stdout again at interpreter shutdown, and a second failure there
        turns a clean exit into status 120 with a message on stderr.

        Any other OSError -- a full disk, a full non-blocking pipe --
        propagates rather than being mistaken for a dead terminal."""
        try:
            sys.stdout.buffer.write(data)
            sys.stdout.buffer.flush()
        except OSError as exc:
            if not _is_peer_gone(exc):
                raise
            _discard_unflushable_stdout()
            return False
        return True

    def warn(message: str) -> None:
        # CRLF because the local terminal is in raw mode: a bare LF would
        # leave the cursor in the current column.
        try:
            sys.stderr.write(f"agent-run: {message}\r\n")
            sys.stderr.flush()
        except (OSError, ValueError):
            pass

    # attach replays raw PTY bytes to a real terminal exactly like tail/logs
    # do (see _reset_terminal_modes above), so it must reset DEC private
    # modes on the way out, and an external SIGINT (kill -INT on this
    # process, distinct from the in-band Ctrl-C byte handled below) must
    # exit quietly with the conventional 128+SIGINT status instead of a
    # traceback.
    try:
        with _terminal_restored_on_signal_death(restore_terminal):
            tty.setraw(local_fd)

            def on_winch(_signum, _frame):
                nonlocal resize_requested
                resize_requested = True

            previous_winch = signal.signal(signal.SIGWINCH, on_winch)
            winch_installed = True
            with log.open("rb") as log_file:
                while True:
                    # Drain the log at full speed while output is pending, but
                    # still fall through to service local input, resize, and
                    # the liveness check every iteration (via a zero-length
                    # select() when more log data is queued). A bare
                    # `continue` back to the log read would let a wrapped
                    # agent emitting output continuously starve keystroke
                    # forwarding, Ctrl-C detach, and resize delivery for as
                    # long as the burst lasts.
                    chunk = log_file.read(8192)
                    if chunk and not emit(chunk):
                        return 0

                    if has_resize and resize_requested:
                        resize_requested = False
                        size = os.get_terminal_size(local_fd)
                        # A 0x0 size is a legitimate transient read (a PTY
                        # whose winsize was never set, a terminal mid-
                        # teardown), not an error -- _pack_resize rejects 0 as
                        # out of range, so skip this iteration's resize
                        # rather than crashing. A partially-written record is
                        # replaced rather than completed: only the latest size
                        # matters, and the reader resynchronizes on the next
                        # record's magic byte, discarding the orphaned prefix.
                        #
                        # Only send when this client's size actually changed.
                        # SIGWINCH fires for reasons other than a real resize,
                        # and with several clients attached the PTY is
                        # last-writer-wins: a redundant send would drag every
                        # other client's view back to this one's size.
                        #
                        # Dimensions past the record's range are clamped, not
                        # rejected: this size comes from the environment (a
                        # very wide display, a tmux pane on an ultrawide), so
                        # raising here would take attach down mid-session and
                        # leave the terminal raw. A clamped size is slightly
                        # wrong; a dead attach is unusable.
                        if size.columns and size.lines and (size.columns, size.lines) != last_sent_size:
                            last_sent_size = (size.columns, size.lines)
                            cols = min(size.columns, MAX_TERMINAL_DIMENSION)
                            rows = min(size.lines, MAX_TERMINAL_DIMENSION)
                            if (cols, rows) != (size.columns, size.lines):
                                warn(
                                    f"terminal is {size.columns}x{size.lines}; "
                                    f"clamping the reported size to "
                                    f"{cols}x{rows}"
                                )
                            pending_resize = _pack_resize(cols, rows)

                    want_write = []
                    if pending_input:
                        want_write.append(stdin_fd)
                    if resize_fd is not None and pending_resize:
                        want_write.append(resize_fd)
                    select_timeout = 0.0 if chunk else 0.2
                    if held_escape_deadline is not None:
                        select_timeout = max(
                            0.0, min(select_timeout, held_escape_deadline - time.monotonic())
                        )
                    readable, writable, _ = select.select(
                        [local_fd], want_write, [], select_timeout
                    )

                    if not _pid_alive(pid):
                        time.sleep(0.1)
                        remaining = log_file.read()
                        if remaining:
                            emit(remaining)
                        return 0

                    if local_fd not in readable:
                        held_escape, held_escape_deadline, released = (
                            _release_expired_held_escape(held_escape, held_escape_deadline)
                        )
                        if released:
                            # Forward straight away -- the prefix is a real
                            # keystroke and withholding it costs latency and
                            # risks losing it if the agent dies next. Keep a
                            # copy only so the following read can be scanned
                            # as the paste marker it may complete.
                            pending_input = _append_bounded(
                                pending_input, released[already_sent:], warn
                            )
                            already_sent = 0
                            resync_prefix = released
                            resync_deadline = (
                                time.monotonic() + _ESCAPE_HOLD_TIMEOUT_SECONDS
                            )

                    # Nothing arrived in time to complete a marker, so the
                    # prefix stands on its own; it is already forwarded.
                    if resync_prefix and time.monotonic() >= resync_deadline:
                        resync_prefix = b""

                    if local_fd in readable:
                        data = os.read(local_fd, 4096)
                        if not data:
                            return 0
                        if held_escape:
                            data = held_escape + data
                            held_escape = b""
                            held_escape_deadline = None
                        elif resync_prefix:
                            data, rejoined = _resync_paste_marker(data, resync_prefix)
                            if rejoined:
                                # Scanned as one sequence, but the prefix has
                                # already gone to the agent: forward only the
                                # bytes past it.
                                already_sent = len(resync_prefix)
                            resync_prefix = b""
                        (
                            forwardable,
                            held_escape,
                            detached,
                            in_paste,
                            paste_bytes,
                            paste_idle_since,
                            paste_started,
                        ) = _scan_local_input_for_detach(
                            data, in_paste, paste_bytes, paste_idle_since, paste_started
                        )
                        new_bytes = forwardable[already_sent:]
                        # Whatever the scan held back still carries the tail of
                        # an already-forwarded prefix when the rejoined marker
                        # is not complete yet.
                        already_sent = max(0, already_sent - len(forwardable))
                        if detached:
                            pending_input = _append_bounded(pending_input, new_bytes, warn)
                            undelivered = _drain_fifo_write(stdin_fd, pending_input)
                            if undelivered:
                                warn(
                                    f"detaching with {len(undelivered)} bytes of typed "
                                    "input undelivered (agent not reading)"
                                )
                            return 0
                        if held_escape:
                            held_escape_deadline = (
                                time.monotonic() + _ESCAPE_HOLD_TIMEOUT_SECONDS
                            )
                        pending_input = _append_bounded(pending_input, new_bytes, warn)

                    if stdin_fd in writable and pending_input:
                        pending_input = pending_input[_flush_fifo_write(stdin_fd, pending_input):]

                    if resize_fd is not None and resize_fd in writable and pending_resize:
                        pending_resize = pending_resize[
                            _flush_fifo_write(resize_fd, pending_resize):
                        ]
    except BrokenPipeError:
        return 0
    except KeyboardInterrupt:
        return 128 + signal.SIGINT
    finally:
        if winch_installed:
            # SIG_DFL rather than the raw return value when SIGWINCH had no
            # Python-level handler: signal.signal returns None for that and
            # rejects None as a new handler.
            signal.signal(
                signal.SIGWINCH,
                previous_winch if previous_winch is not None else signal.SIG_DFL,
            )
        os.close(stdin_fd)
        # Deregister before closing resize_fd: restoring the launch geometry
        # (when this is the last live client) writes through it.
        _deregister_attach_presence_and_restore(
            name, state_dir, attach_marker_fd, resize_fd
        )
        if resize_fd is not None:
            os.close(resize_fd)
        restore_terminal()


# ---------------------------------------------------------------------------
# `logs --clean` (render a PTY-captured log into a readable transcript)
# ---------------------------------------------------------------------------

# Fallback ANSI stripper used when pyte itself can't safely render a log
# (e.g. a RecursionError from a pathological escape-sequence pattern). Not
# as faithful as a real VT100 replay — no cursor-motion collapsing — but it
# provides readable output when emulation fails.
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
    "agent-run: `pyte` is required for `logs --clean`. "
    "Install with: pipx inject mmartins-toolbox pyte  (or uv tool install --with pyte ...)"
)


def _new_pyte_screen(pyte, width: int, height: int, history: int):
    """Create a HistoryScreen that ignores unsupported zero-width controls.

    Pyte 0.8 stops an entire plain-text draw batch at those characters.  Skipping
    each unsupported character keeps replay independent of feed boundaries.

    LNM (Line New Mode) is enabled so bare LF moves the cursor to column 0 as
    well as down one row.  Without it, a one-shot (non-PTY) run's bare-LF line
    endings leave the cursor column unchanged and every subsequent line is
    indented by the length of the previous one — a staircase artifact.  PTY
    output carries CRLF and renders identically with LNM on.
    """
    from pyte import modes as mo

    class SafeHistoryScreen(pyte.HistoryScreen):
        def draw(self, data: str) -> None:
            data = data.translate(self.g1_charset if self.charset else self.g0_charset)
            for char in data:
                char_width = wcwidth(char)
                if self.cursor.x == self.columns:
                    if mo.DECAWM in self.mode:
                        self.dirty.add(self.cursor.y)
                        self.carriage_return()
                        self.linefeed()
                    elif char_width > 0:
                        self.cursor.x -= char_width
                if mo.IRM in self.mode and char_width > 0:
                    self.insert_characters(char_width)
                line = self.buffer[self.cursor.y]
                if char_width == 1:
                    line[self.cursor.x] = self.cursor.attrs._replace(data=char)
                elif char_width == 2:
                    line[self.cursor.x] = self.cursor.attrs._replace(data=char)
                    if self.cursor.x + 1 < self.columns:
                        line[self.cursor.x + 1] = self.cursor.attrs._replace(data="")
                elif char_width == 0 and unicodedata.combining(char):
                    if self.cursor.x:
                        last = line[self.cursor.x - 1]
                        line[self.cursor.x - 1] = last._replace(
                            data=unicodedata.normalize("NFC", last.data + char)
                        )
                    elif self.cursor.y:
                        last = self.buffer[self.cursor.y - 1][self.columns - 1]
                        self.buffer[self.cursor.y - 1][self.columns - 1] = last._replace(
                            data=unicodedata.normalize("NFC", last.data + char)
                        )
                elif char_width < 0 or char_width == 0:
                    continue
                if char_width > 0:
                    self.cursor.x = min(self.cursor.x + char_width, self.columns)
            self.dirty.add(self.cursor.y)

    screen = SafeHistoryScreen(width, height, history=history, ratio=0.5)
    screen.set_mode(mo.LNM)
    return screen


def _screen_lines(screen) -> "Iterable[str]":
    """Yield scrollback's sparse cell maps, then the rendered viewport."""
    for entry in screen.history.top:
        yield ("".join(entry[col].data for col in sorted(entry)) if entry else "").rstrip()
    for row in screen.display:
        yield row.rstrip()


def _serialize_screen(screen) -> str:
    """Serialize a pyte HistoryScreen as a readable transcript."""
    deduped: List[str] = []
    for line in _screen_lines(screen):
        if not deduped or deduped[-1] != line:
            deduped.append(line)
    while deduped and not deduped[-1]:
        deduped.pop()
    return "\n".join(deduped) + "\n"


def _bounded_int(value, lower: int, upper: int) -> bool:
    """True when ``value`` is an ``int`` (not a ``bool``) within the closed
    range ``lower..upper``.

    ``bool`` subclasses ``int`` and ``True == 1``, so an ``isinstance`` check
    would admit a boolean as a one-cell dimension or a zero offset.
    """
    return type(value) is int and lower <= value <= upper


def _validate_resize_timeline(
    resizes: Optional[Sequence[dict]], raw_len: int
) -> List[Tuple[int, int, int]]:
    """Filter a raw resize timeline to well-formed, strictly increasing,
    in-bounds ``(offset, cols, rows)`` tuples, in the input's order.

    Each record is validated independently: a malformed shape, a
    non-integer field, a dimension outside ``1..MAX_TERMINAL_DIMENSION``, an
    offset before the last accepted one, or an offset past the end of ``raw``
    is dropped rather than raised, so a corrupt timeline degrades to fewer
    resize points instead of an exception. Records sharing an offset coalesce
    to the last valid one.
    """
    if not resizes:
        return []
    accepted: List[Tuple[int, int, int]] = []
    last_offset = -1
    for record in resizes:
        try:
            offset = record["offset"]
            cols = record["cols"]
            rows = record["rows"]
        except (TypeError, KeyError):
            continue
        if not (
            _bounded_int(offset, max(last_offset, 0), raw_len)
            and _bounded_int(cols, 1, MAX_TERMINAL_DIMENSION)
            and _bounded_int(rows, 1, MAX_TERMINAL_DIMENSION)
        ):
            continue
        if accepted and offset == last_offset:
            # Several resizes can be applied with no log output between them
            # (attach startup then detach restore, two clients, a drag burst).
            # TIOCSWINSZ is last-writer-wins, so the final geometry at an
            # offset is the one that was in effect for the bytes after it.
            accepted[-1] = (offset, cols, rows)
            continue
        accepted.append((offset, cols, rows))
        last_offset = offset
    return accepted


def _resize_screen(screen, cols: int, rows: int) -> None:
    """Resize ``screen`` to ``cols`` x ``rows`` and clamp the cursor into it.

    ``pyte.Screen.resize`` clips cells at the right and rows at the top but
    leaves the cursor where it was, so a cursor beyond the new bounds keeps
    drawing into cells outside ``screen.columns``/``screen.lines`` that
    ``_serialize_screen`` never reads: on a 10-column screen resized to 5, the
    next character lands at column 10 and disappears.

    ``cursor.x == columns`` is not out of bounds -- it is the DECAWM
    pending-wrap state that ``draw`` tests for -- so x clamps to ``columns``
    rather than ``columns - 1``, preserving a pending wrap across the resize.
    ``savepoints`` need no clamping: ``restore_cursor`` bounds the restored
    position itself.

    Replay across a resize is approximate, and this clamp does not make it
    faithful. ``pyte.Screen.resize`` clips where a terminal reflows: on a width
    reduction the cells beyond the new width are dropped instead of wrapping to
    the next row, and on a height reduction the clipped rows are deleted rather
    than scrolled into ``history.top``. Measured against tmux at 10 columns
    shrunk to 5, a terminal shows ``12345 / 67890 / X`` where replay produces
    ``12345 / X``.

    What the clamp fixes is narrower and worth having: without it the cursor
    keeps its old coordinates, so subsequent draws land in cells outside
    ``screen.columns``/``screen.lines`` that ``_serialize_screen`` never reads
    and the output disappears entirely.
    """
    screen.resize(rows, cols)
    screen.cursor.x = min(screen.cursor.x, screen.columns)
    screen.cursor.y = min(screen.cursor.y, screen.lines - 1)


def _render_log(
    raw: bytes,
    width: int = _RENDER_LOG_DEFAULT_WIDTH,
    height: int = _RENDER_LOG_DEFAULT_HEIGHT,
    history: int = _RENDER_LOG_DEFAULT_HISTORY,
    resizes: Optional[Sequence[dict]] = None,
) -> str:
    """Render a raw PTY-captured log (with ANSI/Ink redraw artifacts) into a
    plain-text transcript by replaying the byte stream through a VT100
    emulator (pyte). Returns the deduplicated screen history + final visible
    viewport, joined with newlines.

    ``resizes`` is an optional timeline of ``{"offset", "cols", "rows"}``
    records sorted by offset -- a resize is out-of-band (TIOCSWINSZ +
    SIGWINCH), so it never appears in ``raw`` itself, and the recorded
    offset is the only way to know where the geometry changed. ``raw`` is
    fed in segments split at each valid offset, calling
    ``screen.resize(rows, cols)`` between segments -- pyte's own argument
    order is (lines, columns), the reverse of this function's. With
    ``resizes`` absent or empty the render is byte-identical to feeding
    ``raw`` whole.

    Pyte is loaded lazily so the rest of agent-run keeps working even if the
    extra is not installed (and so we get a clear error message when it is
    really needed). If pyte itself fails to render (e.g. a RecursionError
    triggered by a pathological escape-sequence pattern), this degrades to
    a best-effort ANSI-stripped plain-text render.
    """
    try:
        import pyte  # type: ignore
    except ImportError as exc:
        raise RenderDependencyError(_RENDER_DEPENDENCY_MESSAGE) from exc

    screen = _new_pyte_screen(pyte, width, height, history)
    stream = pyte.ByteStream(screen)
    timeline = _validate_resize_timeline(resizes, len(raw))
    try:
        start = 0
        for offset, cols, rows in timeline:
            _feed_pyte(stream, raw[start:offset])
            _resize_screen(screen, cols, rows)
            start = offset
        _feed_pyte(stream, raw[start:])
    except Exception:
        return _strip_ansi_fallback(raw)

    return _serialize_screen(screen)


def _read_run_terminal_geometry(log_dir: Path) -> Optional[Tuple[int, int]]:
    """Read the launch-time PTY geometry from run.json's ``terminal`` field.

    Returns ``None`` -- meaning "use the render defaults" -- when run.json
    is absent, unreadable, not valid JSON, or ``terminal`` is malformed;
    every log from before this field existed falls back the same way.
    """
    try:
        data = json.loads((log_dir / "run.json").read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    terminal = data.get("terminal")
    if not isinstance(terminal, dict):
        return None
    cols = terminal.get("cols")
    rows = terminal.get("rows")
    if not (
        _bounded_int(cols, 1, MAX_TERMINAL_DIMENSION)
        and _bounded_int(rows, 1, MAX_TERMINAL_DIMENSION)
    ):
        return None
    return (cols, rows)


def _read_resize_timeline(log_dir: Path) -> List[dict]:
    """Read ``resizes.jsonl`` into a list of raw JSON-object records.

    A line that fails to parse, or does not decode as a JSON object, is
    skipped individually; a missing, unreadable, or non-UTF-8 file yields an
    empty timeline (``UnicodeDecodeError`` is a ``ValueError`` subclass).
    Never raises -- ``_validate_resize_timeline`` still screens every field
    of every returned record before it is trusted.
    """
    try:
        text = (log_dir / "resizes.jsonl").read_text(encoding="utf-8")
    except (OSError, ValueError):
        return []
    records: List[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def _resolved_default_geometry(log_dir: Path) -> Tuple[int, int]:
    """Return the recorded launch geometry, or the render fallback."""
    return _read_run_terminal_geometry(log_dir) or (
        _RENDER_LOG_DEFAULT_WIDTH, _RENDER_LOG_DEFAULT_HEIGHT
    )


# ---------------------------------------------------------------------------
# linked git worktree detection (shared by du and reap)
# ---------------------------------------------------------------------------

# Classification of a run's recorded launch ``cwd``. Only _WORKTREE_LINKED is
# ever acted on: a linked worktree is a disposable checkout whose repository
# data lives in another directory that is not being touched. Every other
# value — including every git failure — means "leave this directory alone".
_WORKTREE_LINKED = "linked"
_WORKTREE_MAIN = "main"
_WORKTREE_BARE = "bare"
_WORKTREE_NOT_A_REPO = "not_a_repo"
_WORKTREE_UNKNOWN = "unknown"


class _WorktreeInfo(NamedTuple):
    """``kind`` is one of the ``_WORKTREE_*`` constants. ``common_dir`` is the
    owning repository's absolute ``--git-common-dir``, populated only for
    ``_WORKTREE_LINKED``: ``git worktree remove`` must run against it so the
    parent repo's ``.git/worktrees/<name>`` admin entry is unregistered along
    with the checkout."""

    kind: str
    common_dir: Optional[str] = None
    detail: Optional[str] = None


def _dir_identity(path: Path) -> Optional[Tuple[int, int]]:
    """``(st_dev, st_ino)`` of ``path`` if it is a real directory (lstat, so a
    symlink never qualifies), else ``None``."""
    try:
        st = path.lstat()
    except OSError:
        return None
    return (st.st_dev, st.st_ino) if _stat_module.S_ISDIR(st.st_mode) else None


def _worktree_resolve_cwd(raw: str) -> Optional[Path]:
    """Canonicalize a recorded launch ``cwd`` to an absolute, symlink-free
    path naming a real directory, or ``None``."""
    if not raw:
        return None
    if not Path(raw).is_absolute():
        return None
    try:
        resolved = Path(os.path.realpath(raw))
    except (OSError, ValueError):
        return None
    if not resolved.is_absolute() or _dir_identity(resolved) is None:
        return None
    return resolved


def _worktree_classify(path: Path) -> _WorktreeInfo:
    """Classify ``path`` as a linked worktree, a main worktree, a bare repo,
    not a repo, or unknown. Read-only: only ``rev-parse`` plumbing runs, so
    nothing is created, pruned or garbage-collected in the inspected repo.

    A linked worktree's ``--git-dir`` is ``<common>/worktrees/<name>`` while
    its ``--git-common-dir`` is the owning repo's git dir; for a main worktree
    the two are the same path. ``--path-format=absolute`` (git >= 2.31) makes
    that comparison meaningful; an older git fails the whole ``rev-parse``,
    which is reported as ``_WORKTREE_UNKNOWN`` rather than guessed at.

    Every failure mode — git not installed, a timeout, unparseable output,
    an unreadable directory — yields ``_WORKTREE_UNKNOWN``. No caller may read
    that as permission to delete.
    """
    if _dir_identity(path) is None:
        return _WorktreeInfo(_WORKTREE_UNKNOWN, detail="not a real directory")

    # `--show-toplevel` is deliberately absent from this call: it aborts the
    # whole rev-parse in a bare repository, which would mask the bare case as
    # a generic failure. It is asked for separately, only when needed.
    probe = _watch_run_git_checked(path, [
        "rev-parse", "--path-format=absolute",
        "--is-bare-repository", "--is-inside-work-tree",
        "--git-dir", "--git-common-dir",
    ])
    if probe.stdout is None:
        if probe.error == "not_a_repo":
            return _WorktreeInfo(_WORKTREE_NOT_A_REPO, detail="not a git repository")
        return _WorktreeInfo(_WORKTREE_UNKNOWN, detail=probe.error or "git_failed")

    lines = [ln.strip() for ln in probe.stdout.splitlines()]
    if len(lines) != 4:
        return _WorktreeInfo(_WORKTREE_UNKNOWN, detail="unexpected rev-parse output")
    is_bare, inside_work_tree, git_dir, common_dir = lines
    if is_bare == "true":
        return _WorktreeInfo(_WORKTREE_BARE, detail="bare repository")
    if inside_work_tree != "true":
        # Inside a .git directory, or a git version answering something other
        # than true/false. Neither is a work tree this may act on.
        return _WorktreeInfo(_WORKTREE_UNKNOWN, detail="not inside a work tree")
    if not git_dir or not common_dir:
        return _WorktreeInfo(_WORKTREE_UNKNOWN, detail="empty rev-parse path")
    if git_dir == common_dir:
        return _WorktreeInfo(_WORKTREE_MAIN, common_dir=common_dir)

    top_probe = _watch_run_git_checked(
        path, ["rev-parse", "--path-format=absolute", "--show-toplevel"]
    )
    if top_probe.stdout is None:
        return _WorktreeInfo(_WORKTREE_UNKNOWN, detail=top_probe.error or "git_failed")
    toplevel = top_probe.stdout.strip()
    top_identity = _dir_identity(Path(toplevel)) if toplevel else None
    if top_identity is None or top_identity != _dir_identity(path):
        # A subdirectory of a linked worktree reports the same pair of git
        # dirs as the worktree's top. `git worktree remove` only accepts the
        # top, and sizing a subdirectory would account for part of a tree that
        # another candidate already covers in full.
        return _WorktreeInfo(_WORKTREE_UNKNOWN, detail="not the top of its worktree")
    return _WorktreeInfo(_WORKTREE_LINKED, common_dir=common_dir)


def _worktree_run_cwd(name: str) -> Optional[Path]:
    """Resolved launch directory recorded for run ``name``, or ``None`` when
    the run has no readable ``cwd`` state file or it no longer names a real
    directory."""
    return _worktree_resolve_cwd(_watch_read_cwd_file(_state_dir(name) / "cwd"))


def _worktree_du_root(cwd: Path) -> Optional[Path]:
    """Linked-worktree root for disk-accounting of ``cwd``, or ``None``.

    ``_worktree_classify`` returns UNKNOWN for a subdirectory of a linked
    worktree, because ``git worktree remove`` requires the top-level path.
    For accounting the top-level root is correct — charging a subdirectory
    would count part of a tree that another run sharing the same worktree
    root already covers.  This function resolves the actual top-level so runs
    launched from a subdirectory still contribute their worktree bytes.

    Returns ``None`` when ``cwd`` is not inside any linked worktree, or when
    any git query fails (fail-safe: unclassifiable paths contribute 0 bytes
    rather than blocking the entire ``du`` pass).
    """
    probe = _watch_run_git_checked(cwd, [
        "rev-parse", "--path-format=absolute",
        "--is-bare-repository", "--is-inside-work-tree",
        "--git-dir", "--git-common-dir",
    ])
    if probe.stdout is None:
        return None
    lines = [ln.strip() for ln in probe.stdout.splitlines()]
    if len(lines) != 4:
        return None
    is_bare, inside_work_tree, git_dir, common_dir = lines
    if is_bare == "true" or inside_work_tree != "true":
        return None
    if not git_dir or not common_dir or git_dir == common_dir:
        return None
    top_probe = _watch_run_git_checked(
        cwd, ["rev-parse", "--path-format=absolute", "--show-toplevel"]
    )
    if top_probe.stdout is None:
        return None
    toplevel_str = top_probe.stdout.strip()
    if not toplevel_str:
        return None
    toplevel = Path(toplevel_str)
    top_identity = _dir_identity(toplevel)
    if top_identity is None:
        return None
    # Verify the recorded cwd is at or below this toplevel.
    try:
        cwd.relative_to(toplevel)
    except ValueError:
        return None
    # Classify the toplevel itself to confirm it is a registered linked worktree.
    if _worktree_classify(toplevel).kind != _WORKTREE_LINKED:
        return None
    return toplevel


# `git worktree remove` unlinks the whole checkout, so it needs far more than
# the plumbing-read budget a rev-parse gets; a multi-gigabyte tree on a slow
# disk would otherwise time out mid-deletion.
WORKTREE_REMOVE_TIMEOUT_SECONDS: float = 300.0


def _worktree_symlink_free_reason(raw: str) -> Optional[str]:
    """``None`` when ``raw`` is an absolute path whose every component is an
    lstat-confirmed real directory, else the reason it is not.

    Deletion never follows a mutable pathname: if any component is a symlink,
    the directory eventually removed is not the one that was inspected. This
    mirrors the lstat-only guards the LOG_ROOT passes apply before deleting.
    """
    if not raw:
        return "no cwd recorded"
    path = Path(raw)
    if not path.is_absolute():
        return f"recorded cwd {raw!r} is not absolute"
    for component in [path, *path.parents]:
        if _dir_identity(component) is None:
            return f"path component {component} is not a real directory"
    return None


class _WorktreeStateScan(NamedTuple):
    entries: Optional[List[Path]]
    error: Optional[str]


def _worktree_state_scan() -> _WorktreeStateScan:
    """Return every real named state directory, or one fail-closed error."""
    try:
        root_st = STATE_ROOT.lstat()
        if not _stat_module.S_ISDIR(root_st.st_mode):
            return _WorktreeStateScan(None, f"state root is missing or not a real directory: {STATE_ROOT}")
        entries = sorted(STATE_ROOT.iterdir())
    except OSError as exc:
        return _WorktreeStateScan(None, f"cannot read state root {STATE_ROOT}: {exc}")

    states: List[Path] = []
    for d in entries:
        if d.name.startswith("."):
            continue
        try:
            _validate_run_name(d.name)
            st = d.lstat()
        except SystemExit:
            return _WorktreeStateScan(None, f"invalid state directory name {d.name!r}")
        except OSError as exc:
            return _WorktreeStateScan(None, f"cannot inspect state directory {d}: {exc}")
        if not _stat_module.S_ISDIR(st.st_mode):
            return _WorktreeStateScan(None, f"state entry is not a real directory: {d}")
        states.append(d)
    return _WorktreeStateScan(states, None)


def _worktree_state_has_live_runner(d: Path) -> Tuple[Optional[bool], Optional[str]]:
    """Determine whether a state directory's recorded runner is still alive.

    A missing pid file with no other evidence is not scan-fatal: a legacy
    state dir without a pid should not abort the whole pass.  An unreadable
    pid file (mode 000, a directory in its place) is ambiguity, not absence:
    it blocks GC the same way a live pid does, mirroring _gc_live_runner_pid.
    """
    try:
        pid_raw = (d / "pid").read_text().strip()
    except FileNotFoundError:
        return False, None
    except (OSError, UnicodeError) as exc:
        return None, f"run {d.name} has unreadable pid file: {exc}"
    if not pid_raw:
        return False, None
    try:
        pid = int(pid_raw)
    except ValueError:
        return None, f"run {d.name} has malformed pid {pid_raw!r}"
    if pid <= 0:
        return None, f"run {d.name} has invalid pid {pid}"
    if not _pid_alive(pid):
        return False, None
    recorded = _read(d / "process_identity")
    if not recorded:
        return None, f"run {d.name} has live pid {pid} but no process identity"
    current = _process_identity(pid)
    if current is None:
        return None, f"run {d.name} has live pid {pid} with unverifiable identity"
    return current == recorded, None


class _WorktreeLiveCwds(NamedTuple):
    paths: Optional[List[Path]]
    error: Optional[str]
    unresolved: Tuple[str, ...] = ()


def _worktree_live_run_cwds(state_entries: List[Path]) -> _WorktreeLiveCwds:
    """Resolve every cwd that is live by status, process identity, or ambiguity.

    An unresolvable cwd is scan-fatal only with evidence of a live runner: a
    pid that is alive and identity-matched may hold any directory, so nothing
    may be deleted. Ambiguity — a malformed pid, or a live pid whose identity
    cannot be verified — is reported by ``_worktree_state_has_live_runner`` and
    aborts the scan before the cwd is read.

    An entry with no readable pid, or a pid that is not alive, has no process
    to protect anything and no recorded path to overlap a candidate. Its name
    is returned in ``unresolved`` and the scan continues, so one legacy state
    directory missing ``status``/``pid``/``cwd`` cannot disable the whole pass.
    """
    live: List[Path] = []
    unresolved: List[str] = []
    for d in state_entries:
        status = _effective_status(d)
        process_live, process_error = _worktree_state_has_live_runner(d)
        if process_error is not None:
            return _WorktreeLiveCwds(None, process_error)
        protects_cwd = status not in TERMINAL_STATUSES or process_live is True
        if not protects_cwd:
            continue
        raw = _watch_read_cwd_file(d / "cwd")
        resolved = _worktree_resolve_cwd(raw)
        if resolved is None:
            if process_live is True:
                return _WorktreeLiveCwds(None, f"cannot resolve cwd for live run {d.name}")
            unresolved.append(d.name)
            continue
        live.append(resolved)
    return _WorktreeLiveCwds(live, None, tuple(unresolved))


def _worktree_empty_tree_oid(path: Path) -> Tuple[Optional[str], Optional[str]]:
    """Object id of the empty tree in ``path``'s repository, for ``--attr-source``.

    Hashing ``/dev/null`` as a tree is a pure computation: it writes no object,
    reads no attributes, and yields the hash-algorithm-correct id for both sha1
    and sha256 repositories.
    """
    probe = _watch_run_git_checked(path, ["hash-object", "-t", "tree", "/dev/null"])
    if probe.stdout is None:
        return None, f"cannot compute empty tree object ({probe.error_detail})"
    oid = probe.stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{40,64}", oid):
        return None, f"unexpected empty tree object {oid!r}"
    return oid, None


def _worktree_content_reason(path: Path) -> Optional[str]:
    """Return a refusal for tracked, untracked, or ignored worktree content.

    ``ls-files --modified`` compares working-tree files through the clean
    filter whenever the index stat cache is inconclusive, so a repository
    supplying ``filter.<driver>.clean`` in ``.gitattributes`` would execute its
    command under this inspection. ``--attr-source`` pointed at the empty tree
    makes every path attribute-free, so no filter driver is ever selected;
    see ``_watch_run_git_checked`` for hook and fsmonitor hardening.
    """
    empty_tree, oid_error = _worktree_empty_tree_oid(path)
    if oid_error is not None:
        return oid_error
    assert empty_tree is not None
    attr_free = ["--attr-source", empty_tree]
    checks = (
        (["ls-files", "--modified", "--deleted", "-z"], "modified or deleted tracked file(s)"),
        (["ls-files", "--others", "--exclude-standard", "-z"], "untracked file(s)"),
        (["ls-files", "--others", "--ignored", "--exclude-standard", "-z"], "ignored file(s) or directory content"),
    )
    for args, label in checks:
        outcome = _watch_run_git_checked(path, [*attr_free, *args])
        if outcome.stdout is None:
            return f"cannot inspect {label} ({outcome.error_detail})"
        count = len([entry for entry in outcome.stdout.split("\0") if entry])
        if count:
            return f"{count} {label}"
    return None


def _worktree_unpushed_reason(path: Path) -> Optional[str]:
    """Refusal reason when HEAD is not reachable from any local remote-tracking ref.

    Uses ``git rev-list --count HEAD --not --remotes``, which compares HEAD
    against ``refs/remotes/*`` in the local repository.  It does NOT contact
    remote servers and does NOT prune stale tracking refs.  A branch that was
    deleted on the remote but whose local tracking ref has not been pruned
    (e.g. via ``git fetch --prune``) still shows count=0, so the commit would
    not block removal even though it is no longer on any actual remote.

    This is a deliberate gap: any network call in a GC path can hang, require
    authentication, or fail in ways that make deletion decisions depend on
    connectivity.  The documented contract is "commits not reachable from any
    local remote-tracking ref".  Callers that need current remote state should
    run ``git fetch --prune`` before invoking reap.
    """
    unpushed = _watch_run_git_checked(path, ["rev-list", "--count", "HEAD", "--not", "--remotes"])
    if unpushed.stdout is None:
        return f"cannot count unpushed commits ({unpushed.error_detail})"
    count = _safe_int(unpushed.stdout.strip())
    if count is None:
        return "unparseable unpushed commit count"
    if count > 0:
        return f"{count} commit(s) not on any remote-tracking ref"
    return None


def _worktree_registered_roots(info: _WorktreeInfo) -> Tuple[Optional[List[Path]], Optional[str]]:
    """Enumerate all registered roots owned by the candidate repository."""
    if not info.common_dir:
        return None, "owning repository is unknown"
    outcome = _watch_run_git_checked(
        Path(info.common_dir), ["worktree", "list", "--porcelain", "-z"]
    )
    if outcome.stdout is None:
        return None, f"cannot enumerate registered worktrees ({outcome.error_detail})"
    roots: List[Path] = []
    for field in outcome.stdout.split("\0"):
        if not field:
            continue
        if field.startswith("worktree "):
            raw = field[len("worktree "):]
            resolved = _worktree_resolve_cwd(raw)
            if resolved is None:
                return None, f"cannot resolve registered worktree {raw!r}"
            roots.append(resolved)
    if not roots:
        return None, "worktree list returned no roots"
    return roots, None


def _worktree_nested_reason(info: _WorktreeInfo, path: Path) -> Optional[str]:
    roots, error = _worktree_registered_roots(info)
    if error is not None:
        return error
    assert roots is not None
    for root in roots:
        if root != path and _path_is_within(root, path):
            return f"registered worktree {root} is nested inside the candidate"
    return None


def _has_surrogate(s: str) -> bool:
    """True if ``s`` contains any UTF-16 surrogate code point (U+D800–U+DFFF).

    ``_watch_run_git_checked`` decodes git output with ``errors="surrogateescape"``,
    which maps each non-UTF-8 byte to a private surrogate (U+DC80–U+DCFF).
    A surrogate in a path component means the original byte sequence was not
    valid UTF-8: the path cannot be safely round-tripped through Python's str
    layer without re-encoding via ``os.fsencode``.  Any such path must be
    treated as unresolvable to avoid acting on a mangled name.
    """
    return any("\ud800" <= c <= "\udfff" for c in s)


def _worktree_submodule_reason(path: Path) -> Optional[str]:
    """Return a refusal if any initialized submodule exists in the worktree.

    An initialized submodule is a mode-160000 gitlink in the index whose
    checkout directory contains a `.git` file pointing into the parent
    repository's modules store.  It is absent from both ``ls-files --others``
    queries (gitlinks are tracked, not untracked) and invisible to
    ``_worktree_foreign_nested_reason``.  Under ``--force-dirty`` the content
    checks are skipped entirely; this check runs unconditionally so neither
    code path reaches ``git worktree remove --force`` with an initialized
    submodule checkout present.

    A deinitialized submodule has a mode-160000 index entry but an empty
    checkout directory (no `.git` file or directory); removing it loses nothing
    irreplaceable and is permitted.  The test is whether the checkout directory
    has a `.git` entry, not merely whether the index entry exists.

    Uses NUL-delimited output for path safety.  Fails closed on command
    failure or malformed output.  A `.git` lstat error other than
    ``FileNotFoundError`` also fails closed.  A non-UTF-8 submodule path
    (surrogate in the decoded string) is treated as unresolvable and fails
    closed rather than acting on a mangled name.
    """
    outcome = _watch_run_git_checked(path, ["ls-files", "--stage", "-z"])
    if outcome.stdout is None:
        return f"cannot enumerate index entries ({outcome.error_detail})"
    for entry in outcome.stdout.split("\0"):
        if not entry:
            continue
        # Stage output format: "<mode> <object> <stage>\t<path>"
        fields = entry.split("\t", 1)
        if len(fields) != 2:
            return f"malformed ls-files --stage output: {entry!r}"
        meta = fields[0].split()
        if not meta:
            return f"malformed ls-files --stage output: {entry!r}"
        if meta[0] != "160000":
            continue
        submodule_path = fields[1]
        if _has_surrogate(submodule_path):
            return f"submodule path contains non-UTF-8 bytes: {submodule_path!r}"
        checkout = path / submodule_path
        git_entry = checkout / ".git"
        try:
            git_entry.lstat()
            # .git present: initialized checkout with submodule data.
            return f"initialized submodule at {submodule_path}"
        except FileNotFoundError:
            # No .git entry: deinitialized (empty directory), safe to remove.
            continue
        except OSError as exc:
            return f"cannot check submodule at {submodule_path}: {exc}"
    return None


def _worktree_bare_repo_check(candidate_dir: Path) -> Optional[str]:
    """Return a refusal if ``candidate_dir`` is a bare git repository.

    Called for directories that ``ls-files`` descended into (emitting individual
    files) rather than reporting as a boundary (slash-terminated entry).  Git
    descends into bare repositories because they have no ``.git`` marker to stop
    traversal; those files therefore appear as ordinary untracked paths rather
    than a single ``bare.git/`` entry.

    Uses ``git rev-parse --is-bare-repository`` via the hardened subprocess
    wrapper to remain consistent with the rest of the classification machinery.
    A non-zero exit (not a repository) is silently ignored; any other failure
    or a ``true`` answer is a refusal.
    """
    probe = _watch_run_git_checked(candidate_dir, [
        "rev-parse", "--is-bare-repository",
    ])
    if probe.stdout is None:
        if probe.error == "not_a_repo":
            return None
        return f"cannot check bare-repository status of {candidate_dir} ({probe.error_detail})"
    if probe.stdout.strip() == "true":
        return f"nested bare git repository at {candidate_dir}"
    return None


def _worktree_foreign_nested_reason(path: Path) -> Optional[str]:
    """Detect git repository roots structurally nested inside the candidate directory.

    ``_worktree_nested_reason`` only queries the candidate's own repository, so
    a linked worktree or bare repository of a different repository is invisible
    to it.  This check uses ``git ls-files --others`` to enumerate untracked and
    ignored paths, exploiting two complementary mechanisms:

    Non-bare repositories: git halts at the repository boundary and emits a
    single slash-terminated directory entry (e.g. ``nested.git/``); ``lstat``
    on ``.git`` inside confirms the boundary.

    Bare repositories: git has no ``.git`` marker to stop traversal and
    descends into the bare repo, emitting individual files such as
    ``precious.git/HEAD`` and ``precious.git/config``.  The trailing-``/``
    filter skips all of these.  The second pass below collects all unique
    directory prefixes from non-slash entries and calls
    ``_worktree_bare_repo_check`` on each.  This is bounded because git only
    descends into directories it cannot identify as repositories (i.e. bare
    repos and plain untracked dirs); the rev-parse check is fast (one read).
    Prefixes already confirmed as non-repositories by the slash scan are skipped.

    Two ls-files passes cover both gitignored and non-gitignored content:
    - ``--others --exclude-standard``: untracked content outside ignored dirs.
    - ``--others --ignored --exclude-standard``: repos inside ``node_modules/``
      or ``.venv/`` appear as a directory entry, not as individual files.

    A path entry containing non-UTF-8 bytes is decoded with surrogate escapes
    by ``_watch_run_git_checked``; ``_has_surrogate`` detects such entries and
    fails closed rather than acting on a mangled name.

    Called unconditionally, including under ``--force-dirty``.
    """
    checks = (
        ["ls-files", "--others", "--exclude-standard", "-z"],
        ["ls-files", "--others", "--ignored", "--exclude-standard", "-z"],
    )
    seen_dirs: set[str] = set()
    # All unique directory prefixes from non-slash entries, to probe for bare repos.
    bare_candidates: set[str] = set()
    for git_args in checks:
        outcome = _watch_run_git_checked(path, git_args)
        if outcome.stdout is None:
            return f"cannot enumerate untracked paths ({outcome.error_detail})"
        for entry in outcome.stdout.split("\0"):
            if not entry:
                continue
            if _has_surrogate(entry):
                # Non-UTF-8 byte in path: cannot safely resolve the name.
                return f"untracked path contains non-UTF-8 bytes: {entry!r}"
            if entry.endswith("/"):
                if entry in seen_dirs:
                    continue
                seen_dirs.add(entry)
                # Trailing / marks a git repository boundary; verify via lstat.
                candidate_git = path / entry.rstrip("/") / ".git"
                try:
                    candidate_git.lstat()
                    return f"nested git repository at {path / entry.rstrip('/')}"
                except FileNotFoundError:
                    pass
                except OSError as exc:
                    return f"cannot check .git in {path / entry.rstrip('/')}: {exc}"
            else:
                # No slash terminator: git descended into this directory, which
                # cannot be a non-bare repo (git would have halted there).
                # Collect every directory prefix of this entry so nested bare
                # repos at any depth (e.g. vendor/cached.git/HEAD) are found.
                parts = entry.split("/")
                for depth in range(1, len(parts)):
                    prefix = "/".join(parts[:depth])
                    if prefix and prefix + "/" not in seen_dirs:
                        bare_candidates.add(prefix)

    for prefix in bare_candidates:
        candidate_dir = path / prefix
        reason = _worktree_bare_repo_check(candidate_dir)
        if reason is not None:
            return reason
    return None


def _worktree_activity_age_seconds(path: Path) -> Tuple[Optional[float], Optional[str]]:
    """Age of the newest recursively observed filesystem mtime in a worktree.

    Uses strict mode so any unreadable subtree returns an error rather than a
    stale timestamp: a recent file hidden by a permission error must not make
    the worktree appear idle.
    """
    newest = _newest_mtime_recursive(path, strict=True)
    if newest is None or newest is _WALK_INCOMPLETE:
        return None, "cannot determine worktree filesystem activity"
    return max(0.0, time.time() - newest), None


def _worktree_remove(info: _WorktreeInfo, path: Path, *, force: bool) -> Optional[str]:
    """Remove the linked worktree at ``path`` via ``git worktree remove``,
    returning ``None`` on success or the failure reason.

    The command runs against the owning repository's ``--git-common-dir`` so
    the ``.git/worktrees/<name>`` administrative entry is unregistered along
    with the checkout; an ``rmtree`` would leave that registration stranded.
    ``force`` is passed only under ``reap --force-dirty``: without it git
    itself refuses a worktree with modified or untracked files, which is a
    second line of defence behind ``_worktree_unsaved_work_reason``.
    """
    if not info.common_dir:
        return "owning repository is unknown"
    args = ["worktree", "remove", *(["--force"] if force else []), str(path)]
    outcome = _watch_run_git_checked(
        Path(info.common_dir), args, timeout=WORKTREE_REMOVE_TIMEOUT_SECONDS
    )
    if outcome.stdout is None:
        return f"git worktree remove failed ({outcome.error_detail})"
    return None


# `git worktree add` clones the base tree into a fresh checkout, so it can
# take as long as an initial clone on a slow disk; reuse the same generous
# budget as removal rather than the plumbing-read timeout.
WORKTREE_ADD_TIMEOUT_SECONDS: float = 300.0


def _worktree_usage_error(message: str) -> NoReturn:
    """Exit 2 for a malformed ``--worktree*`` invocation.

    Distinct from the plain ``sys.exit(...)`` (exit 1) used elsewhere in this
    module for runtime/preflight failures: a bad combination of flags is a
    usage error in the conventional Unix sense, not a failure of an
    otherwise-valid request.
    """
    print(f"agent-run: {message}", file=sys.stderr)
    raise SystemExit(2)


def _worktree_repo_from_invocation_cwd() -> Optional[str]:
    """Absolute toplevel of the repo containing the process's current
    directory, or ``None`` if it is not inside a git repository.

    Must be called before ``_apply_launch_cwd`` chdirs the process: this is
    the "invocation cwd", not wherever ``--cwd``/``--worktree`` later move to.
    """
    outcome = _watch_run_git_checked(
        Path.cwd(), ["rev-parse", "--path-format=absolute", "--show-toplevel"]
    )
    if outcome.stdout is None:
        return None
    toplevel = outcome.stdout.strip()
    return toplevel or None


class _CreatedWorktree(NamedTuple):
    """A linked worktree (and possibly a branch) this invocation created,
    kept only for rollback if launch fails before the run is published."""

    path: Path
    repo: Path
    branch: str
    branch_created: bool
    # OID the branch was created at, when branch_created is True. Rollback
    # deletes the branch only if it still points here, so a branch moved by
    # the workload (or by anything else) in the interim is never discarded.
    branch_oid: Optional[str] = None


def _worktree_branch_checked_out(repo: Path, branch: str) -> Tuple[Optional[bool], Optional[str]]:
    """Whether ``refs/heads/<branch>`` is checked out in any worktree
    registered to ``repo``. Returns ``(None, error)`` when this cannot be
    determined; callers must treat that as "assume checked out" and refuse
    to delete. ``git branch -D`` itself refuses a checked-out branch and is
    the actual protection; this function exists only to produce a clear
    diagnostic before attempting the delete, not to gate correctness.
    """
    outcome = _watch_run_git_checked(repo, ["worktree", "list", "--porcelain", "-z"])
    if outcome.stdout is None:
        return None, f"cannot enumerate worktrees ({outcome.error_detail})"
    target_ref = f"refs/heads/{branch}"
    for field in outcome.stdout.split("\0"):
        if field.startswith("branch ") and field[len("branch "):] == target_ref:
            return True, None
    return False, None


def _worktree_delete_branch_if_unchanged(repo: Path, branch: str, expected_oid: str) -> Optional[str]:
    """Delete ``refs/heads/<branch>`` iff it still points at ``expected_oid``
    and no registered worktree has it checked out. Returns ``None`` on
    success or if the branch is already gone, else the reason it was left in
    place.

    The OID check guards against a branch the workload (or anything else)
    moved: it is compared here, then the deletion is ``git branch -D``, not
    ``git update-ref -d``, because git itself refuses ``branch -D`` against
    a branch any worktree has checked out -- closing the gap between the
    checked-out scan below and the delete, which a scan-then-delete of our
    own could never do atomically. ``_worktree_branch_checked_out`` is kept
    as a pre-check purely for a clear diagnostic; ``branch -D``'s own refusal
    is what actually protects a concurrent checkout.

    Residual: an unrelated actor deleting and recreating ``branch`` at
    ``expected_oid`` between the OID check and the delete below is
    indistinguishable from this invocation's own branch never having moved,
    so ``branch -D`` would delete a branch this invocation no longer owns.
    Accepted rather than closed -- see H2 in the fix-round report.
    """
    current = _watch_run_git_checked(repo, ["rev-parse", "--verify", "--quiet", f"refs/heads/{branch}"])
    if current.stdout is None:
        return None  # already gone: nothing left to roll back
    if current.stdout.strip() != expected_oid:
        return f"branch {branch!r} no longer points at the commit this invocation created; leaving it in place"
    checked_out, err = _worktree_branch_checked_out(repo, branch)
    if err is not None:
        return f"cannot confirm branch {branch!r} is unused: {err}"
    if checked_out:
        return f"branch {branch!r} is checked out in a worktree; refusing to delete"
    outcome = _watch_run_git_checked(repo, ["branch", "-D", branch])
    if outcome.stdout is None:
        return outcome.error_detail or "git branch -D failed"
    return None


def _create_launch_worktree(args: argparse.Namespace) -> Optional[_CreatedWorktree]:
    """No-op unless ``--worktree`` is set. Otherwise validates the full
    ``--worktree*`` flag set, creates (or attaches to) a linked worktree, and
    points ``args.cwd`` at it so ``_apply_launch_cwd`` needs no worktree
    awareness of its own.

    Every check here runs against the owning repo via ``git -C``, never by
    chdir'ing into ``args.worktree`` — which may not exist yet — and runs
    before any mutation, so a rejected request leaves the filesystem
    untouched. Returns ``None`` when nothing was created by this call: either
    ``--worktree`` was absent, or ``--worktree-reuse`` attached to an
    already-existing worktree directory. The non-``None`` case is exactly
    "this invocation is the one that must clean it up on a later failure".

    Rollback (``_rollback_launch_worktree``) is best-effort, not crash-atomic:
    a SIGKILL of this process between ``git worktree add``/``git branch``
    succeeding and the caller publishing run state leaves a linked worktree,
    a `.git/worktrees/<name>` admin entry, and possibly a branch that no
    Python code ever runs again to attribute or remove. Nothing in this
    module journals worktree creation intent durably enough to survive that;
    recovering from it is a manual `git worktree remove` / `git branch -D`
    against the orphaned path this invocation would have printed.

    ``--worktree-reuse`` validation (registered root, branch, detached-HEAD
    checks) is a snapshot taken at preflight, not a lease: nothing here
    prevents an external ``git switch``/``git worktree remove`` against the
    same directory between this check and the runner forking. It is
    advisory against concurrent external mutation, not a guarantee of
    isolation -- a dedicated worktree (the default, non-reuse path) is what
    actually isolates a run from concurrent activity elsewhere.
    """
    worktree_raw: Optional[str] = getattr(args, "worktree", None)
    reuse: bool = bool(getattr(args, "worktree_reuse", False))
    if reuse and not worktree_raw:
        _worktree_usage_error("--worktree-reuse requires --worktree")
    if not worktree_raw:
        # --worktree-base/-branch/-repo are meaningless without --worktree;
        # accepting them silently would launch in the invocation cwd despite
        # apparently-configured worktree flags, masking a missing --worktree.
        for flag, dest in (
            ("--worktree-base", "worktree_base"),
            ("--worktree-branch", "worktree_branch"),
            ("--worktree-repo", "worktree_repo"),
        ):
            if getattr(args, dest, None):
                _worktree_usage_error(f"{flag} requires --worktree")
        return None
    if getattr(args, "cwd", None):
        _worktree_usage_error("--worktree and --cwd are mutually exclusive")
    base: Optional[str] = getattr(args, "worktree_base", None)
    if not base:
        _worktree_usage_error("--worktree-base is required with --worktree")

    # Anchored at the invocation cwd: _apply_launch_cwd has not run yet, so
    # a relative DIR means relative to where agent-run was actually typed.
    worktree_dir = Path(os.path.expanduser(worktree_raw))
    if not worktree_dir.is_absolute():
        worktree_dir = Path.cwd() / worktree_dir

    repo_raw: Optional[str] = getattr(args, "worktree_repo", None)
    if repo_raw:
        repo = Path(os.path.expanduser(repo_raw))
        if not repo.is_absolute():
            repo = Path.cwd() / repo
        if not repo.is_dir():
            sys.exit(f"agent-run: --worktree-repo does not exist or is not a directory: {repo_raw}")
    else:
        discovered = _worktree_repo_from_invocation_cwd()
        if discovered is None:
            _worktree_usage_error(
                "--worktree requires --worktree-repo: the invocation directory "
                "is not inside a git repository"
            )
        repo = Path(discovered)

    branch: str = getattr(args, "worktree_branch", None) or args.name
    # _validate_run_name's whitelist (alnum start, then [A-Za-z0-9._-]) admits
    # strings git rejects as ref names outright ("..", trailing ".", trailing
    # ".lock"), and --worktree-branch is user-supplied independent of the run
    # name entirely — so branch legality is git's call, not re-derived here.
    branch_check = _watch_run_git_checked(repo, ["check-ref-format", "--branch", branch])
    if branch_check.stdout is None:
        sys.exit(f"agent-run: --worktree-branch {branch!r} is not a valid git branch name")

    base_check = _watch_run_git_checked(repo, ["rev-parse", "--verify", "--quiet", f"{base}^{{commit}}"])
    if base_check.stdout is None:
        sys.exit(f"agent-run: --worktree-base {base!r} does not resolve to a commit in {repo}")
    # Frozen to the OID resolved here, not the symbolic ref: a concurrent ref
    # update between this check and worktree creation must not silently
    # change which commit the new branch is built on, and rollback needs an
    # exact OID to attribute a branch to this invocation (see below).
    base_oid = base_check.stdout.strip()

    branch_ref_check = _watch_run_git_checked(repo, ["rev-parse", "--verify", "--quiet", f"refs/heads/{branch}"])
    branch_exists = branch_ref_check.stdout is not None
    if branch_exists and not reuse:
        sys.exit(
            f"agent-run: branch {branch!r} already exists in {repo}; "
            f"pass --worktree-reuse to attach to it"
        )

    dir_present = _path_entry_exists(worktree_dir)
    if dir_present and not reuse:
        sys.exit(f"agent-run: --worktree directory already exists: {worktree_dir}")

    if dir_present:
        # --worktree-reuse opts into attaching, but only to a worktree this
        # exact repo already owns — never to an unrelated directory or a
        # worktree belonging to some other repo, which would be the same
        # class of cross-contamination bug this feature exists to prevent.
        info = _worktree_classify(worktree_dir)
        common_check = _watch_run_git_checked(repo, ["rev-parse", "--path-format=absolute", "--git-common-dir"])
        owning_common = common_check.stdout.strip() if common_check.stdout is not None else None
        owning_identity = _dir_identity(Path(owning_common)) if owning_common else None
        candidate_identity = _dir_identity(Path(info.common_dir)) if info.common_dir else None
        if (
            info.kind != _WORKTREE_LINKED
            or owning_identity is None
            or candidate_identity is None
            or owning_identity != candidate_identity
        ):
            sys.exit(
                f"agent-run: --worktree {worktree_dir} exists but is not a linked "
                f"worktree of {repo}; refusing to attach"
            )
        # A recursively copied worktree keeps a .git file pointing at the
        # original repo, so the checks above still pass, but the copy is not
        # registered and `git worktree remove` would refuse it. Require DIR's
        # filesystem identity to match one of the repo's registered roots —
        # the same guard `reap` already applies before touching a worktree it
        # did not itself create (see _worktree_registered_roots).
        registered_roots, reg_error = _worktree_registered_roots(info)
        if reg_error is not None:
            sys.exit(
                f"agent-run: cannot verify --worktree {worktree_dir} is a registered "
                f"worktree of {repo}: {reg_error}"
            )
        assert registered_roots is not None
        dir_identity = _dir_identity(worktree_dir)
        if not any(_dir_identity(root) == dir_identity for root in registered_roots):
            sys.exit(
                f"agent-run: --worktree {worktree_dir} is not a registered worktree "
                f"of {repo}; refusing to attach"
            )
        # The reuse target must already be on the requested branch: attaching
        # while silently running on whatever branch happens to be checked out
        # is exactly the cross-contamination this feature exists to prevent.
        # A detached HEAD is rejected outright rather than guessed at.
        # Comparison uses the full ref, not --short: --short returns the
        # shortest *unambiguous* name, which is "heads/<branch>" whenever a
        # tag shares the branch's name, breaking equality against a bare
        # branch name for no reason visible to the operator. Same form
        # _worktree_branch_checked_out already uses for this reason.
        branch_probe = _watch_run_git_checked(worktree_dir, ["symbolic-ref", "--quiet", "HEAD"])
        current_ref = branch_probe.stdout.strip() if branch_probe.stdout is not None else None
        if current_ref is None:
            sys.exit(
                f"agent-run: --worktree {worktree_dir} has a detached HEAD; "
                f"refusing to attach with --worktree-reuse"
            )
        target_ref = f"refs/heads/{branch}"
        if current_ref != target_ref:
            current_branch = current_ref.removeprefix("refs/heads/")
            sys.exit(
                f"agent-run: --worktree {worktree_dir} is on branch {current_branch!r}, "
                f"not {branch!r}; refusing to attach with --worktree-reuse"
            )
        args.cwd = str(worktree_dir)
        return None  # attached as-is; this invocation created nothing

    if branch_exists:
        # --worktree-reuse attaching to a pre-existing branch: check it out
        # rather than creating it.
        add_args = ["worktree", "add", str(worktree_dir), branch]
        outcome = _watch_run_git_checked(repo, add_args, timeout=WORKTREE_ADD_TIMEOUT_SECONDS)
        if outcome.stdout is None:
            sys.exit(f"agent-run: git worktree add failed for {worktree_dir}: {outcome.error_detail}")
        branch_created = False
    else:
        # Branch creation and worktree creation are split into two operations
        # instead of one `git worktree add -b`, whose partial result cannot
        # be attributed: `worktree add -b` can create the branch and then
        # fail preparing the directory, permanently orphaning that branch.
        # `git branch <name> <oid>` is atomic (creates exactly refs/heads/
        # <name> = <oid>, or fails outright), so a subsequent worktree-add
        # failure can be attributed and rolled back precisely.
        branch_create = _watch_run_git_checked(repo, ["branch", branch, base_oid])
        if branch_create.stdout is None:
            sys.exit(f"agent-run: failed to create branch {branch!r}: {branch_create.error_detail}")
        add_args = ["worktree", "add", str(worktree_dir), branch]
        outcome = _watch_run_git_checked(repo, add_args, timeout=WORKTREE_ADD_TIMEOUT_SECONDS)
        if outcome.stdout is None:
            reason = _worktree_delete_branch_if_unchanged(repo, branch, base_oid)
            if reason is not None:
                print(
                    f"agent-run: warning: git worktree add failed and branch "
                    f"{branch!r} could not be rolled back: {reason}",
                    file=sys.stderr,
                )
            sys.exit(f"agent-run: git worktree add failed for {worktree_dir}: {outcome.error_detail}")
        branch_created = True

    args.cwd = str(worktree_dir)
    # Recorded on args, not returned separately, so _cmd_launch_locked (which
    # never sees the _CreatedWorktree rollback handle) can still write the
    # durable state-dir marker. Resolved through realpath to match how
    # <state_dir>/cwd is recorded post-chdir (Path.cwd() after os.chdir
    # resolves symlinks), so the two files agree on macOS's /var -> /private/var
    # and similar cases.
    args._worktree_created_path = os.path.realpath(worktree_dir)
    return _CreatedWorktree(
        worktree_dir, repo, branch, branch_created=branch_created,
        branch_oid=base_oid if branch_created else None,
    )


def _rollback_launch_worktree(created: _CreatedWorktree) -> None:
    """Best-effort teardown of a worktree (and branch, if freshly created)
    this invocation made, called only when launch fails before ``os.fork()``
    -- no runner process exists yet, so nothing can own the tree. Never
    raises: failure here must not mask the launch failure that triggered it,
    so every problem is reported to stderr and swallowed.
    """
    info = _worktree_classify(created.path)
    if info.kind != _WORKTREE_LINKED:
        print(
            f"agent-run: warning: cannot roll back {created.path}: "
            f"not a linked worktree ({info.detail})",
            file=sys.stderr,
        )
        return
    reason = _worktree_remove(info, created.path, force=False)
    if reason is not None:
        print(f"agent-run: warning: failed to roll back worktree {created.path}: {reason}", file=sys.stderr)
        return  # worktree still checked out; deleting its branch would fail anyway
    if created.branch_created:
        reason = _worktree_delete_branch_if_unchanged(
            created.repo, created.branch, created.branch_oid or ""
        )
        if reason is not None:
            print(
                f"agent-run: warning: failed to remove branch {created.branch!r} "
                f"during rollback: {reason}",
                file=sys.stderr,
            )


class _WorktreeCandidate(NamedTuple):
    """One deduplicated worktree candidate attributed across all run state."""

    resolved: Path
    raw: str
    names: List[str]
    age_seconds: Optional[float]
    identity_fd: int


class _WorktreeCandidates(NamedTuple):
    candidates: List[_WorktreeCandidate]
    skipped: List[str]


def _worktree_collect_candidates(
    state_entries: List[Path], selected_names: Optional[set[str]]
) -> _WorktreeCandidates:
    """Group terminal run cwd records using all-state safety attribution."""
    by_path: dict[str, _WorktreeCandidate] = {}
    skipped: List[str] = []
    for d in state_entries:
        if _read(d / "status") not in TERMINAL_STATUSES:
            continue
        raw = _watch_read_cwd_file(d / "cwd")
        resolved = _worktree_resolve_cwd(raw)
        if resolved is None:
            if selected_names is None or d.name in selected_names:
                skipped.append(f"run {d.name} has an unresolvable cwd")
            continue
        if _dir_identity(resolved) is None:
            if selected_names is None or d.name in selected_names:
                skipped.append(f"run {d.name} cwd is not a real directory")
            continue
        age = _terminal_state_age_seconds(d)
        key = str(resolved)
        if key not in by_path:
            # O_DIRECTORY | O_NOFOLLOW: atomically refuse symlinks and non-dirs.
            # The open fd pins the inode so its number cannot be recycled while
            # we hold it; samestat at revalidation then confirms the path still
            # names the same kernel object.  -1 signals a failed open and is
            # treated as a refusal in _worktree_candidate_refusal.
            try:
                identity_fd = os.open(
                    str(resolved), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
                )
            except OSError:
                if selected_names is None or d.name in selected_names:
                    skipped.append(f"run {d.name} cwd cannot be opened for identity pin")
                continue
            by_path[key] = _WorktreeCandidate(
                resolved, raw, [d.name], age, identity_fd,
            )
            continue
        existing = by_path[key]
        merged_age = None if age is None or existing.age_seconds is None else min(
            existing.age_seconds, age
        )
        by_path[key] = existing._replace(
            names=[*existing.names, d.name],
            age_seconds=merged_age,
        )

    candidates = [
        by_path[key]
        for key in sorted(by_path)
        if selected_names is None or selected_names.intersection(by_path[key].names)
    ]
    return _WorktreeCandidates(candidates, skipped)


def _path_is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _worktree_candidate_refusal(
    cand: _WorktreeCandidate,
    state_entries: List[Path],
    *,
    force_dirty: bool,
    min_age_threshold: float,
    reconciled_cwds: List[Path],
    unresolved_cwds: set[str],
) -> Tuple[Optional[str], Optional[_WorktreeInfo], Optional[float]]:
    """Run every deletion refusal check against one current filesystem view."""
    for reconciled in reconciled_cwds:
        if _path_is_within(reconciled, cand.resolved):
            return "a sharing run was reconciled in this invocation", None, None
    if cand.age_seconds is None or cand.age_seconds < min_age_threshold:
        return "youngest sharing run is below the age threshold", None, None
    # Every terminal entry that resolves to this candidate must clear the age
    # threshold independently — including entries whose name is in cand.names.
    # cand.age_seconds is the frozen minimum from collection time; re-reading
    # each entry's age here handles two cases:
    #   • entries absent from cand.names (became terminal after collection);
    #   • same-name entries where the state directory was replaced between
    #     collection and this scan (a new run generation with its own fresh
    #     ended_at that cand.age_seconds does not reflect).
    # Inode identity is not a reliable generation marker on ext4, which reuses
    # inodes immediately after directory removal.  Fresh age is always
    # computable, so it is always checked.
    for d in state_entries:
        if _effective_status(d) not in TERMINAL_STATUSES:
            continue
        raw = _watch_read_cwd_file(d / "cwd")
        run_resolved = _worktree_resolve_cwd(raw)
        if run_resolved is None or not _path_is_within(run_resolved, cand.resolved):
            continue
        late_age = _terminal_state_age_seconds(d)
        if late_age is None:
            return f"run {d.name} has unparseable age", None, None
        if late_age < min_age_threshold:
            return "youngest sharing run is below the age threshold", None, None
    # Verify the candidate path still names the same directory object opened at
    # collection.  fstat on the pinned fd and lstat on the path must agree: if
    # the original directory was removed and another was created at the same
    # path between collection and this scan, the new directory has a different
    # inode — guaranteed because the fd holds the original inode's refcount
    # above zero, preventing its recycling on any filesystem.
    try:
        fd_st = os.fstat(cand.identity_fd)
        path_st = cand.resolved.lstat()
    except OSError:
        return "candidate path identity unreadable after collection", None, None
    if not os.path.samestat(fd_st, path_st):
        return "candidate path identity changed after collection", None, None
    symlink_reason = _worktree_symlink_free_reason(cand.raw)
    if symlink_reason is not None:
        return symlink_reason, None, None

    info = _worktree_classify(cand.resolved)
    if info.kind != _WORKTREE_LINKED:
        return f"not a linked worktree ({info.detail or info.kind})", None, None
    # Require the candidate to be a registered worktree root in its own repository.
    # A recursively copied worktree keeps a .git file from the original, so
    # _worktree_classify returns LINKED, but the copy is not registered and
    # git worktree remove would refuse it.  Verify by filesystem identity: a
    # registered root with the same (st_dev, st_ino) as the candidate is
    # unambiguously this path.
    registered_roots, reg_error = _worktree_registered_roots(info)
    if reg_error is not None:
        return reg_error, None, None
    assert registered_roots is not None
    cand_identity = _dir_identity(cand.resolved)
    if not any(_dir_identity(r) == cand_identity for r in registered_roots):
        return "not a registered worktree root", None, None
    nested = _worktree_nested_reason(info, cand.resolved)
    if nested is not None:
        return nested, None, None
    foreign_nested = _worktree_foreign_nested_reason(cand.resolved)
    if foreign_nested is not None:
        return foreign_nested, None, None
    submodule = _worktree_submodule_reason(cand.resolved)
    if submodule is not None:
        return submodule, None, None

    live = _worktree_live_run_cwds(state_entries)
    if live.error is not None:
        return f"liveness scan failed: {live.error}", None, None
    assert live.paths is not None
    unresolved_cwds.update(live.unresolved)
    for live_cwd in live.paths:
        if _path_is_within(live_cwd, cand.resolved):
            return f"a live run is using this directory ({live_cwd})", None, None

    activity_age, activity_error = _worktree_activity_age_seconds(cand.resolved)
    if activity_error is not None:
        return activity_error, None, None
    assert activity_age is not None
    effective_age = min(cand.age_seconds, activity_age)
    if effective_age < min_age_threshold:
        return "worktree filesystem activity is below the age threshold", None, effective_age

    if not force_dirty:
        content = _worktree_content_reason(cand.resolved)
        if content is not None:
            return f"{content} — use --force-dirty to override", None, effective_age
    unpushed = _worktree_unpushed_reason(cand.resolved)
    if unpushed is not None:
        return unpushed, None, effective_age
    return None, info, effective_age


def _worktree_gc_pass(
    *,
    collected: Optional[_WorktreeCandidates],
    scan_error: Optional[str],
    dry_run: bool,
    force_dirty: bool,
    min_age_threshold: float,
    reconciled_cwds: List[Path],
    budget_expired,
) -> Tuple[int, int, int]:
    """Remove linked worktrees only after an all-state fail-closed scan."""
    removed = skipped = deferred = 0
    if scan_error is not None:
        print(f"  [worktree]: skipped: {scan_error}")
        return 0, 1, 0
    assert collected is not None
    for reason in collected.skipped:
        print(f"  [worktree]: skipped: {reason}")
        skipped += 1

    unresolved_cwds: set[str] = set()
    for cand in collected.candidates:
        if budget_expired():
            deferred += 1
            os.close(cand.identity_fd)
            continue
        label = f"  [worktree] {cand.resolved}"
        shared = f" (runs: {', '.join(cand.names)})" if len(cand.names) > 1 else ""

        try:
            with _worktree_publication_lock(exclusive=True):
                final_scan = _worktree_state_scan()
                if final_scan.error is not None:
                    print(f"{label}: skipped: {final_scan.error}{shared}")
                    skipped += 1
                    continue
                assert final_scan.entries is not None
                refusal, info, effective_age = _worktree_candidate_refusal(
                    cand,
                    final_scan.entries,
                    force_dirty=force_dirty,
                    min_age_threshold=min_age_threshold,
                    reconciled_cwds=reconciled_cwds,
                    unresolved_cwds=unresolved_cwds,
                )
                if refusal is not None:
                    print(f"{label}: skipped: {refusal}{shared}")
                    skipped += 1
                    continue
                assert info is not None and effective_age is not None
                age_h = effective_age / 3600
                if dry_run:
                    print(
                        f"{label}: linked worktree (conservative_age={age_h:.1f}h) "
                        f"[dry-run]{shared}"
                    )
                    removed += 1
                    continue
                # Validation is the final step immediately preceding git worktree
                # remove.  An external process can still create ignored content in
                # the interval between the last ls-files check (inside
                # _worktree_candidate_refusal above) and the remove call below; git
                # removes ignored files without --force, so this window cannot be
                # fully closed while using git worktree remove.  The exclusive
                # publication lock prevents concurrent agent-run publication but
                # does not fence non-agent processes.
                failure = _worktree_remove(info, cand.resolved, force=force_dirty)
                if failure is not None:
                    print(f"{label}: skipped: {failure}")
                    skipped += 1
                    continue
                print(
                    f"{label}: linked worktree (conservative_age={age_h:.1f}h) "
                    f"[removing]{shared}"
                )
            removed += 1
        finally:
            os.close(cand.identity_fd)
    if unresolved_cwds:
        names = ", ".join(sorted(unresolved_cwds))
        print(
            f"  [worktree]: skipped: {len(unresolved_cwds)} run(s) with no live runner "
            f"and an unresolvable cwd: {names}"
        )
        skipped += 1
    return removed, skipped, deferred


def cmd_reap(args: argparse.Namespace) -> int:
    """Reconcile stale ``running`` state, idle-kill lingering processes, and
    garbage-collect old terminal-state runs, state-less scratch dirs, and
    (with ``--include-logs``) preserved log dirs.

    The opt-in passes — ``--include-logs``, ``--orphan-processes``,
    ``--include-worktrees`` — can be enabled together with ``--all``, which
    changes no age threshold and overrides no refusal.

    The pass is deliberately split into reconciliation, state-backed GC,
    preserved-log GC, and orphan-scratch GC. Every destructive state-backed
    action is guarded by a name lock plus inode/status/pid revalidation;
    every scratch/log action also requires lstat-confirmed real directory
    components so a symlink can never redirect deletion outside LOG_ROOT.
    """
    dry_run: bool = args.dry_run
    idle_hours: Optional[float] = getattr(args, "idle_hours", None)
    min_age_hours: Optional[float] = getattr(args, "min_age_hours", None)
    log_min_age_hours: Optional[float] = getattr(args, "log_min_age_hours", None)
    target_name: Optional[str] = getattr(args, "name", None)
    force_unknown: bool = bool(getattr(args, "force_unknown", False))
    # --all turns on every opt-in pass and nothing else: force_unknown and
    # force_dirty override refusals rather than enable a pass, so they stay
    # exactly as the caller set them.
    all_passes: bool = bool(getattr(args, "all", False))
    include_logs: bool = all_passes or bool(getattr(args, "include_logs", False))
    orphan_processes: bool = all_passes or bool(getattr(args, "orphan_processes", False))
    orphan_min_age_hours: Optional[float] = getattr(args, "orphan_min_age_hours", None)
    include_worktrees: bool = all_passes or bool(getattr(args, "include_worktrees", False))
    worktree_min_age_hours: Optional[float] = getattr(args, "worktree_min_age_hours", None)
    force_dirty: bool = bool(getattr(args, "force_dirty", False))
    if target_name is not None:
        target_name = _validate_run_name(target_name)

    # Re-read the env at call time so --idle-hours / env override both work.
    idle_threshold: float = (
        idle_hours * 3600 if idle_hours is not None else _parse_idle_stall_seconds()
    )
    min_age_threshold: float = (
        min_age_hours * 3600 if min_age_hours is not None else _parse_reap_min_age_seconds()
    )
    # Independent of min_age_threshold by design; see --log-min-age-hours help.
    log_min_age_threshold: float = (
        log_min_age_hours * 3600
        if log_min_age_hours is not None
        else _parse_log_min_age_seconds()
    )
    # Independent of GC thresholds: governs live process age, not artifact
    # retention. Parsed even when --orphan-processes is absent so the flag
    # is always validated; see --orphan-min-age-hours help.
    orphan_min_age_threshold: float = (
        orphan_min_age_hours * 3600
        if orphan_min_age_hours is not None
        else _parse_orphan_min_age_seconds()
    )
    # Independent of every threshold above: governs destruction of a working
    # tree, not of bookkeeping. Parsed even when --include-worktrees is
    # absent so the flag is always validated; see --worktree-min-age-hours help.
    worktree_min_age_threshold: float = (
        worktree_min_age_hours * 3600
        if worktree_min_age_hours is not None
        else _parse_worktree_min_age_seconds()
    )
    max_seconds_arg: Optional[float] = getattr(args, "max_seconds", None)
    reap_budget: float = (
        max_seconds_arg if max_seconds_arg is not None else _parse_reap_max_seconds()
    )
    reap_start = time.monotonic()

    # A previous SIGKILL can leave a dot-prefixed deletion sentinel. Finish it
    # before scanning named runs; each failure remains per-sentinel so one
    # poisoned leftover cannot wedge every later reap invocation.
    resumed_count = 0
    if not dry_run:
        resumed_count = _reap_stale_sentinels(STATE_ROOT)
        if LOG_ROOT.is_dir():
            resumed_count += _reap_stale_sentinels(LOG_ROOT)

    candidates: List[Path] = []
    if STATE_ROOT.is_dir():
        if target_name is not None:
            candidates = [_state_dir(target_name)]
        else:
            try:
                candidates = sorted(STATE_ROOT.iterdir())
            except OSError as exc:
                print(f"reap: cannot read state root: {exc}")
                return 1
    elif target_name is None and not LOG_ROOT.is_dir():
        print("reap: no state or log root, nothing to do.")
        return 0

    died_count = 0
    killed_count = 0
    skipped_count = 0
    gc_count = 0
    orphaned_scratch_count = 0
    logs_collected_count = 0
    gc_skipped_count = 0
    orphan_procs_killed = 0
    orphan_procs_skipped = 0
    worktrees_removed = 0
    worktrees_skipped = 0
    deferred_count = 0
    found_target = False
    reconciled_this_pass = set()
    # Resolved launch directories of runs this invocation moved to a terminal
    # status, captured while their cwd record is still readable. The worktree
    # pass refuses any candidate containing one: _mark_terminal preserves a
    # stale ended_at, so such a run can immediately look old enough to collect.
    worktree_reconciled_cwds: List[Path] = []

    def _record_reconciled_cwd(state_dir: Path) -> None:
        resolved = _worktree_resolve_cwd(_watch_read_cwd_file(state_dir / "cwd"))
        if resolved is not None:
            worktree_reconciled_cwds.append(resolved)
    # Names whose whole log dir was collected (or reported under --dry-run) by
    # pass 2.5 in this invocation.  Pass 3 skips these to avoid counting the
    # same run's scratch as both a collected log and an orphaned scratch dir.
    logs_acted_on: set[str] = set()

    # Filter the state candidates once for both passes (quality F2/C5), and
    # validate names before *any* path construction/mutation (M2). A targeted
    # invocation gets a direct O(1) candidate rather than scanning the root.
    state_candidates: List[Path] = []
    for d in candidates:
        if d.name.startswith("."):
            continue
        try:
            _validate_run_name(d.name)
            before = d.lstat()
        except (OSError, SystemExit):
            continue
        if not _stat_module.S_ISDIR(before.st_mode):
            continue
        state_candidates.append(d)
        found_target = True

    # Capture worktree attribution before routine state GC removes cwd records.
    worktree_candidates: Optional[_WorktreeCandidates] = None
    worktree_scan_error: Optional[str] = None
    worktree_state_names: set[str] = set()
    if include_worktrees:
        worktree_scan = _worktree_state_scan()
        worktree_scan_error = worktree_scan.error
        if worktree_scan.entries is not None:
            selected = {target_name} if target_name is not None else None
            worktree_candidates = _worktree_collect_candidates(worktree_scan.entries, selected)
            worktree_state_names = {
                name for candidate in worktree_candidates.candidates for name in candidate.names
            }

    # Pass 1: stale-running reconciliation and identity-verified idle kill.
    for d in state_candidates:
        if time.monotonic() - reap_start > reap_budget:
            deferred_count += 1
            continue
        name = d.name
        raw_status = _read(d / "status")
        if raw_status != "running":
            continue

        pid_raw = _read(d / "pid")
        if not pid_raw:
            print(f"  {name}: dead (no pid recorded) [{'dry-run' if dry_run else 'marking died'}]")
            _record_reconciled_cwd(d)
            if not dry_run:
                _mark_died(d, "no pid recorded")
                reconciled_this_pass.add(name)
            died_count += 1
            continue

        try:
            pid = int(pid_raw)
        except ValueError:
            print(f"  {name}: dead (invalid pid {pid_raw!r}) [{'dry-run' if dry_run else 'marking died'}]")
            _record_reconciled_cwd(d)
            if not dry_run:
                _mark_died(d, f"invalid pid: {pid_raw!r}")
                reconciled_this_pass.add(name)
            died_count += 1
            continue

        if not _pid_alive(pid):
            print(f"  {name}: dead pid={pid} [{'dry-run' if dry_run else 'marking died'}]")
            _record_reconciled_cwd(d)
            if not dry_run:
                _mark_died(d, f"pid {pid} no longer alive")
                reconciled_this_pass.add(name)
            died_count += 1
            continue

        log = _log_file_for(name)
        idle_secs: Optional[float] = None
        if log is not None:
            try:
                idle_secs = time.time() - log.stat().st_mtime
            except OSError:
                pass

        if idle_secs is None or idle_secs <= idle_threshold:
            skipped_count += 1
            continue

        idle_h = idle_secs / 3600
        reason = f"idle>{idle_h:.1f}h"
        print(f"  {name}: idle pid={pid} idle={idle_h:.1f}h [{'dry-run' if dry_run else 'killing'}]")
        # A run this invocation would reconcile keeps its worktree in both
        # modes, so --dry-run predicts the same worktree outcome as a real pass.
        _record_reconciled_cwd(d)
        if dry_run:
            killed_count += 1
            continue

        # Identity check: verify the live pid matches what our own verified
        # process_identity token recorded at launch, to avoid signalling a
        # recycled PID — reuses the same identity primitive cmd_kill uses.
        recorded_identity = _read(d / "process_identity")
        if not recorded_identity:
            print(f"  {name}: skipped: no process identity recorded (legacy state)")
            skipped_count += 1
            continue
        live_identity = _process_identity(pid)
        if live_identity is None or live_identity != recorded_identity:
            print(
                f"  {name}: skipped: pid identity unverified "
                f"(recorded={recorded_identity!r}, live={live_identity!r})"
            )
            skipped_count += 1
            continue

        try:
            _force_kill(name, d, pid, recorded_identity)
        except SystemExit as exc:
            print(f"  {name}: skipped: {exc}")
            skipped_count += 1
            continue
        try:
            os.waitpid(pid, os.WNOHANG)
        except (ChildProcessError, OSError):
            pass
        if not _pid_alive(pid):
            _mark_terminal(d, "killed", reason)
            reconciled_this_pass.add(name)
            killed_count += 1
        else:
            print(
                f"  {name}: kill_failed — process pid={pid} still alive after "
                "force-kill; leaving status as published by force-kill"
            )
            skipped_count += 1

    # Pass 2: collect terminal state dirs older than the threshold. The
    # explicit set is structural protection for H1: an old ended_at left by a
    # crash between finalization writes cannot make a run reconciled in pass 1
    # eligible for deletion in pass 2 of the same invocation.
    for d in state_candidates:
        if time.monotonic() - reap_start > reap_budget:
            deferred_count += 1
            continue
        name = d.name
        if name in reconciled_this_pass:
            continue
        if include_worktrees and name in worktree_state_names:
            continue
        try:
            before = d.lstat()
        except OSError:
            continue
        if not _stat_module.S_ISDIR(before.st_mode):
            continue

        raw_status = _read(d / "status")
        if not _gc_status_eligible(raw_status, force_unknown=force_unknown):
            continue

        age_secs = _terminal_state_age_seconds(d)
        if age_secs is None or age_secs < min_age_threshold:
            continue
        age_h = age_secs / 3600

        # These read-only checks intentionally happen before the dry-run
        # branch, so the preview does not claim an action a real invocation
        # would refuse (quality C1). The normal path repeats them under the
        # per-name lock to close the actual scan-to-delete race.
        status_now = _read(d / "status")
        if not _gc_status_eligible(status_now, force_unknown=force_unknown):
            gc_skipped_count += 1
            continue
        live_pid = _gc_live_runner_pid(d)
        if live_pid is not None:
            print(
                f"  {name}: skipped: status={status_now or 'unknown'} but pid={live_pid} "
                "is the recorded live runner — refusing to remove"
            )
            gc_skipped_count += 1
            continue

        # Validate both log path components using lstat (never is_dir(),
        # which follows symlinks) before touching scratch. A malformed/symlink
        # scratch is a per-run refusal: retain the state too, emit an
        # actionable warning, and continue with later candidates (C2/H2/D1).
        log_d = _log_dir(name)
        scratch_dir = log_d / "tmp"
        try:
            log_st = log_d.lstat()
        except FileNotFoundError:
            log_st = None
        except OSError as exc:
            print(f"  {name}: gc skipped: cannot inspect log dir {log_d}: {exc}")
            gc_skipped_count += 1
            continue
        if log_st is not None:
            if not _stat_module.S_ISDIR(log_st.st_mode):
                print(f"  {name}: gc skipped: log path is not a real directory; leaving state and scratch")
                gc_skipped_count += 1
                continue
            try:
                scratch_st = scratch_dir.lstat()
            except FileNotFoundError:
                scratch_st = None
            except OSError as exc:
                print(f"  {name}: gc skipped: cannot inspect scratch path {scratch_dir}: {exc}")
                gc_skipped_count += 1
                continue
            if scratch_st is not None and not _stat_module.S_ISDIR(scratch_st.st_mode):
                print(f"  {name}: gc skipped: scratch path is not a real directory; leaving state and scratch")
                gc_skipped_count += 1
                continue

        action = "force-unknown" if raw_status not in TERMINAL_STATUSES else "terminal"
        print(
            f"  {name}: {action} (status={raw_status or 'unknown'}, age={age_h:.1f}h) "
            f"[{'dry-run' if dry_run else 'removing state+scratch'}]"
        )
        if dry_run:
            gc_count += 1
            continue

        # Re-verify under the per-name lock: serializes with relaunch and
        # rejects an inode/status/pid replacement before either directory is
        # deleted. Scratch deletion uses an expected lstat snapshot too.
        with _launch_lock(name):
            try:
                current = d.lstat()
            except OSError:
                gc_skipped_count += 1
                continue
            if not _stat_module.S_ISDIR(current.st_mode) or (
                (current.st_dev, current.st_ino) != (before.st_dev, before.st_ino)
            ):
                gc_skipped_count += 1
                continue
            status_current = _read(d / "status")
            if not _gc_status_eligible(status_current, force_unknown=force_unknown):
                gc_skipped_count += 1
                continue
            live_pid = _gc_live_runner_pid(d)
            if live_pid is not None:
                print(
                    f"  {name}: skipped: status={status_current or 'unknown'} but pid={live_pid} "
                    "is the recorded live runner — refusing to remove"
                )
                gc_skipped_count += 1
                continue
            try:
                # Re-inspect log and scratch under the same name lock, closing
                # the scan-to-delete window for both components. Never use
                # symlink-following Path.is_dir() here (C2).
                if _path_entry_exists(log_d):
                    log_current = log_d.lstat()
                    if not _stat_module.S_ISDIR(log_current.st_mode):
                        raise SystemExit("log path is not a real directory")
                    try:
                        scratch_current = scratch_dir.lstat()
                    except FileNotFoundError:
                        scratch_current = None
                    if scratch_current is not None:
                        if not _stat_module.S_ISDIR(scratch_current.st_mode):
                            raise SystemExit("scratch path is not a real directory")
                        _safe_rmtree(scratch_dir, log_d, expected=scratch_current)
                _crash_safe_rmtree(d, STATE_ROOT, expected=current)
            except SystemExit as exc:
                print(f"  {name}: gc skipped: {exc}")
                gc_skipped_count += 1
                continue
            gc_count += 1

    # Log/scratch passes share one scan of LOG_ROOT. A targeted --name has
    # one direct candidate; untargeted scans read LOG_ROOT once.
    log_candidates: List[Path] = []
    if LOG_ROOT.is_dir():
        if target_name is not None:
            log_candidates = [_log_dir(target_name)]
        else:
            try:
                log_candidates = sorted(LOG_ROOT.iterdir())
            except OSError as exc:
                print(f"reap: cannot read log root: {exc}")
                log_candidates = []

    # Pre-validate log candidates: filter out dot-prefix names, invalid names,
    # non-directories, and entries that cannot be stat'd.  The lstat snapshot
    # here is a pre-scan only; both pass 2.5 and pass 3 re-lstat under their
    # own per-name lock before any deletion.
    validated_logs: List[tuple[Path, os.stat_result]] = []
    for _ld in log_candidates:
        if _ld.name.startswith("."):
            continue
        try:
            _validate_run_name(_ld.name)
            _ld_st = _ld.lstat()
        except (OSError, SystemExit):
            continue
        if _stat_module.S_ISDIR(_ld_st.st_mode):
            validated_logs.append((_ld, _ld_st))

    # Pass 2.5 (--include-logs): whole-log-dir GC for preserved-log-only
    # runs (state dir gone). Runs after pass 2 so a run whose state dir pass 2
    # just removed in this same invocation can become log-only and eligible
    # here too. Runs before pass 3 (orphan scratch): a log dir removed whole
    # here makes log_d.lstat() in pass 3 raise FileNotFoundError, which that
    # loop already treats as "nothing to do" — no double counting, no error
    # spam over an already-gone path.
    if include_logs:
        for log_d, log_before in validated_logs:
            if time.monotonic() - reap_start > reap_budget:
                deferred_count += 1
                continue
            name = log_d.name
            if _path_entry_exists(_state_dir(name)):
                continue  # state-backed — pass 2 owns this run, never race it
            newest = _newest_mtime_recursive(
                log_d,
                cutoff=time.time() - log_min_age_threshold,
                skip_top_dir_mtime=True,
            )
            if newest is _WALK_INCOMPLETE:
                continue  # age unknowable; retain
            if newest is None:
                # Genuinely empty: judge by the container's own mtime, which
                # only moves when entries are added or removed.
                newest = log_before.st_mtime
            age_secs = time.time() - newest
            if age_secs < log_min_age_threshold:
                continue
            age_h = age_secs / 3600
            if dry_run:
                print(
                    f"  {name}: preserved log (age={age_h:.1f}h) "
                    "[dry-run]"
                )
                logs_collected_count += 1
                logs_acted_on.add(name)
                continue
            size = _dir_size_bytes(log_d)
            print(
                f"  {name}: preserved log (age={age_h:.1f}h, size={_human_size(size)}) "
                "[removing log]"
            )
            with _launch_lock(name):
                try:
                    if _path_entry_exists(_state_dir(name)):
                        continue
                    log_current = log_d.lstat()
                    if not _stat_module.S_ISDIR(log_current.st_mode):
                        raise SystemExit("log path is not a real directory")
                    if (log_current.st_dev, log_current.st_ino) != (
                        log_before.st_dev, log_before.st_ino
                    ):
                        continue
                    newest_current = _newest_mtime_recursive(
                        log_d,
                        cutoff=time.time() - log_min_age_threshold,
                        skip_top_dir_mtime=True,
                    )
                    if newest_current is _WALK_INCOMPLETE:
                        continue
                    if newest_current is None:
                        newest_current = log_current.st_mtime
                    if time.time() - newest_current < log_min_age_threshold:
                        continue
                    _crash_safe_rmtree(log_d, LOG_ROOT, expected=log_before)
                except (OSError, SystemExit) as exc:
                    print(f"  {name}: gc skipped: {exc}")
                    gc_skipped_count += 1
                    continue
                logs_collected_count += 1
                logs_acted_on.add(name)

    # Pass 3: a reboot wipes STATE_ROOT (normally tmpfs) while persistent
    # LOG_ROOT/<name>/tmp survives. Sweep those orphaned scratch dirs once
    # their own recursive contents have been quiet for the GC threshold (H3).
    for log_d, _log_before in validated_logs:
        if time.monotonic() - reap_start > reap_budget:
            deferred_count += 1
            continue
        name = log_d.name
        # Skip names pass 2.5 already acted on (or reported under --dry-run):
        # the whole log dir was collected, so its scratch is no longer orphaned.
        # Without this guard, --dry-run double-counts the same run as both a
        # collected log and an orphaned scratch.
        if name in logs_acted_on:
            continue
        scratch_dir = log_d / "tmp"
        try:
            scratch_before = scratch_dir.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            print(f"  {name}: gc skipped: cannot inspect orphan scratch {scratch_dir}: {exc}")
            gc_skipped_count += 1
            continue
        if not _stat_module.S_ISDIR(scratch_before.st_mode):
            print(f"  {name}: gc skipped: orphan scratch path is not a real directory")
            gc_skipped_count += 1
            continue
        # State can have been recreated after pass 2, so only act when there
        # is no state path at all; a symlink/non-dir is also treated as present
        # conservatively rather than following it.
        if _path_entry_exists(_state_dir(name)):
            continue
        newest = _newest_mtime_recursive(scratch_dir, cutoff=time.time() - min_age_threshold)
        if newest is None or newest is _WALK_INCOMPLETE or time.time() - newest < min_age_threshold:
            continue
        age_h = (time.time() - newest) / 3600
        print(
            f"  {name}: orphaned scratch (age={age_h:.1f}h) "
            f"[{'dry-run' if dry_run else 'removing scratch'}]"
        )
        if dry_run:
            orphaned_scratch_count += 1
            continue
        with _launch_lock(name):
            try:
                if _path_entry_exists(_state_dir(name)):
                    continue
                log_current = log_d.lstat()
                scratch_current = scratch_dir.lstat()
                if not _stat_module.S_ISDIR(log_current.st_mode):
                    raise SystemExit("log path is not a real directory")
                if not _stat_module.S_ISDIR(scratch_current.st_mode):
                    raise SystemExit("orphan scratch path is not a real directory")
                if (scratch_current.st_dev, scratch_current.st_ino) != (
                    scratch_before.st_dev, scratch_before.st_ino
                ):
                    continue
                newest_current = _newest_mtime_recursive(
                    scratch_dir, cutoff=time.time() - min_age_threshold
                )
                if (newest_current is None or newest_current is _WALK_INCOMPLETE
                        or time.time() - newest_current < min_age_threshold):
                    continue
                _safe_rmtree(scratch_dir, log_d, expected=scratch_before)
            except (OSError, SystemExit) as exc:
                print(f"  {name}: gc skipped: {exc}")
                gc_skipped_count += 1
                continue
            orphaned_scratch_count += 1

    # Pass 4 (--orphan-processes): find and terminate live agent-run runner
    # processes that have no state dir — invisible to all prior passes because
    # they hold no entry in STATE_ROOT.  Runs last so a state dir removed in
    # pass 2 of this same invocation is correctly seen as absent by discovery.
    if orphan_processes:
        now = time.time()
        proc_table = _scan_process_table()
        orphan_candidates, orphan_skips = _find_orphan_runners(
            proc_table,
            min_age_seconds=orphan_min_age_threshold,
            now=now,
            self_pid=os.getpid(),
            self_pgid=os.getpgid(0),
            target_name=target_name,
        )

        for skip in orphan_skips:
            reason = skip.reason
            detail = f" ({skip.detail})" if skip.detail else ""
            print(f"  [orphan] pid={skip.pid}: skipped: {reason}{detail}")
            orphan_procs_skipped += 1

        # Collects (cand, pgid, use_group, age_h) for every candidate that
        # received SIGTERM, for phases 2 and 3 below.
        _term_targets: List[Tuple[Any, Optional[int], bool, float]] = []

        def _orphan_skip(cand, age_h: float, why: str) -> None:
            nonlocal orphan_procs_skipped
            print(
                f"  [orphan] {cand.name} pid={cand.pid} age={age_h:.1f}h: "
                f"skipped: {why}"
            )
            orphan_procs_skipped += 1

        for cand in orphan_candidates:
            if time.monotonic() - reap_start > reap_budget:
                deferred_count += 1
                continue
            age_h = (now - cand.start_time) / 3600

            # Dry-run: report the action without acquiring the lock or creating
            # any files under STATE_ROOT.  All other passes skip on dry_run before
            # locking; pass 4 must do the same so a preview run neither serialises
            # against a concurrent launch nor creates .locks entries.
            if dry_run:
                print(
                    f"  [orphan] {cand.name} pid={cand.pid} age={age_h:.1f}h: "
                    "dry-run (would TERM)"
                )
                orphan_procs_killed += 1
                continue

            # Re-verify under the per-name lock: serializes with a concurrent
            # launch that may have just created a state dir for this name.
            with _launch_lock(cand.name):
                if _path_entry_exists(_state_dir(cand.name)):
                    _orphan_skip(cand, age_h, "state dir appeared after discovery")
                    continue

                # PID-reuse is the central hazard: there is no state dir, so there
                # is no recorded process_identity to compare against — the identity
                # captured at discovery is the only compensating control.  Re-read
                # it immediately before signalling and abort on any mismatch.
                live_identity = _process_identity(cand.pid)
                if live_identity is None or live_identity != cand.identity:
                    _orphan_skip(
                        cand, age_h,
                        f"identity changed (expected={cand.identity!r}, "
                        f"live={live_identity!r})",
                    )
                    continue

                if not _pid_alive(cand.pid):
                    _orphan_skip(cand, age_h, "process gone before action")
                    continue

                # Determine process-group scope.  The runner calls setsid() so its
                # pid == pgid when it is a session leader.  Only kill the group when
                # getpgid confirms this; otherwise signal the pid alone.
                try:
                    pgid = os.getpgid(cand.pid)
                    use_group = pgid == cand.pid
                except OSError:
                    use_group = False
                    pgid = None

                pgid_note = f" pgid={pgid}" if use_group and pgid is not None else " (pid only — pgid mismatch)"

                # Phase 1: SIGTERM under the lock, then release to let grace run
                # concurrently with other candidates.  For the pid-only case,
                # route through _send_signal_to_verified_pid so Linux gets the
                # pidfd guarantee (signal bound before identity re-read, eliminating
                # the PID-reuse window for that platform).  Group signals have no
                # pidfd equivalent; use os.killpg directly for those.
                print(
                    f"  [orphan] {cand.name} pid={cand.pid} age={age_h:.1f}h: "
                    f"TERM{pgid_note}"
                )
                try:
                    if use_group and pgid is not None:
                        os.killpg(pgid, signal.SIGTERM)
                    else:
                        _send_signal_to_verified_pid(cand.pid, signal.SIGTERM, cand.identity)
                except (ProcessLookupError, OSError, RuntimeError):
                    orphan_procs_skipped += 1
                    continue

                # Record enough to complete phases 2 and 3.
                _term_targets.append((cand, pgid, use_group, age_h))

        # Phase 2: one shared grace window covering all candidates that received
        # SIGTERM.  O(5 s) total instead of O(5 s × N).
        if _term_targets:
            deadline = time.time() + ORPHAN_KILL_GRACE_SECONDS
            while time.time() < deadline:
                if all(not _pid_alive(c.pid) for c, *_ in _term_targets):
                    break
                time.sleep(KILL_POLL_INTERVAL_SECONDS)

        # Phase 3: re-verify identity and SIGKILL whoever survived the grace window.
        for cand, pgid, use_group, age_h in _term_targets:
            pgid_note = f" pgid={pgid}" if use_group and pgid is not None else " (pid only — pgid mismatch)"
            if not _pid_alive(cand.pid):
                print(f"  [orphan] {cand.name} pid={cand.pid}: terminated after TERM")
                orphan_procs_killed += 1
                continue
            # Re-take the per-name lock before KILL; mirrors the phase-1 lock so
            # a concurrent launch between TERM and KILL cannot create a state dir
            # undetected.  Re-verify identity here too — PID reuse can happen in
            # the grace window, and we must not kill an unrelated process.
            with _launch_lock(cand.name):
                current = _process_identity(cand.pid)
                if current is None or current != cand.identity:
                    print(
                        f"  [orphan] {cand.name} pid={cand.pid}: "
                        "skipped KILL: identity changed after TERM"
                    )
                    orphan_procs_skipped += 1
                    continue

                print(
                    f"  [orphan] {cand.name} pid={cand.pid}: "
                    f"KILL after grace{pgid_note}"
                )
                try:
                    if use_group and pgid is not None:
                        os.killpg(pgid, signal.SIGKILL)
                    else:
                        _send_signal_to_verified_pid(cand.pid, signal.SIGKILL, cand.identity)
                except (ProcessLookupError, OSError, RuntimeError):
                    pass
                orphan_procs_killed += 1

    # Worktree deletion runs last so its long removal timeout cannot starve
    # routine state, log, scratch, or orphan-process cleanup.
    if include_worktrees:
        wt_removed, wt_skipped, wt_deferred = _worktree_gc_pass(
            collected=worktree_candidates,
            scan_error=worktree_scan_error,
            dry_run=dry_run,
            force_dirty=force_dirty,
            min_age_threshold=worktree_min_age_threshold,
            reconciled_cwds=worktree_reconciled_cwds,
            budget_expired=lambda: time.monotonic() - reap_start > reap_budget,
        )
        worktrees_removed += wt_removed
        worktrees_skipped += wt_skipped
        deferred_count += wt_deferred

    if target_name and not found_target:
        if _known(target_name):
            print(f"reap: '{target_name}' has no ephemeral state (log preserved); nothing to reconcile")
        else:
            print(f"reap: no such run '{target_name}' in {STATE_ROOT}")

    if deferred_count:
        print(
            f"reap: budget exhausted ({reap_budget:.0f}s); deferred {deferred_count} candidate(s) — "
            "next invocation resumes"
        )

    prefix = "[dry-run] " if dry_run else ""
    print(
        f"{prefix}reap done: died={died_count} killed={killed_count} "
        f"skipped={skipped_count} collected={gc_count} "
        f"orphaned_scratch={orphaned_scratch_count} gc_skipped={gc_skipped_count} "
        f"resumed={resumed_count} logs_collected={logs_collected_count} "
        f"orphan_procs_killed={orphan_procs_killed} orphan_procs_skipped={orphan_procs_skipped} "
        f"worktrees_removed={worktrees_removed} worktrees_skipped={worktrees_skipped} "
        f"deferred={deferred_count}"
    )
    return 0


# ---------------------------------------------------------------------------
# du
# ---------------------------------------------------------------------------

# Recognized rollup group keys. A state-backed run whose effective status is
# not one of these falls into "unrecognized" rather than silently inventing a
# new bucket per legacy/corrupt value.
_DU_STATUS_GROUPS = frozenset({"running", "starting", "stalled", "done",
                               "failed", "launch_failed", "died", "killed"})
_DU_UNRECOGNIZED_GROUP = "unrecognized"
_DU_PRESERVED_LOG_GROUP = "preserved-log-only"


class _DuRow(NamedTuple):
    """One accounting row: a single run (``--by-run``) or an aggregated
    group. ``count`` is the number of runs folded into the row (1 for a
    per-run row). Log bytes exclude the ``tmp/`` scratch subtree — total is
    state + log + scratch + worktree, so nothing is double-counted."""

    key: str  # run name (--by-run) or group/status label (rollup)
    count: int
    state_bytes: int
    log_bytes: int
    scratch_bytes: int
    worktree_bytes: int = 0
    complete: bool = True  # False when any walk encountered a permission error

    @property
    def total_bytes(self) -> int:
        return self.state_bytes + self.log_bytes + self.scratch_bytes + self.worktree_bytes


def _du_charge_worktrees(rows: List[_DuRow]) -> List[_DuRow]:
    """Charge each linked worktree once while excluding separately charged roots.

    For deletion, only the worktree's exact top-level path is acceptable (see
    ``_worktree_classify``).  For accounting, a run launched from a subdirectory
    of a linked worktree should still contribute the whole worktree's bytes:
    ``_worktree_du_root`` resolves ``--show-toplevel`` so the correct root is
    charged regardless of how deep the recorded ``cwd`` is.

    Excluding all of ``STATE_ROOT`` or ``LOG_ROOT`` would drop unrecognized
    content (e.g. ``.locks``, sentinels, invalid-name entries) from the total,
    because those bytes appear neither in the STATE/LOG/SCRATCH columns nor in
    the worktree charge.  Instead, only the per-run directories actually charged
    to a row are excluded, so any remainder in those roots falls through to the
    worktree charge and is counted exactly once.
    """
    owners: dict[str, int] = {}
    roots: dict[str, Path] = {}
    root_cache: dict[str, Optional[Path]] = {}
    for index, row in enumerate(rows):
        cwd = _worktree_run_cwd(row.key)
        if cwd is None:
            continue
        cwd_key = str(cwd)
        if cwd_key not in root_cache:
            root_cache[cwd_key] = _worktree_du_root(cwd)
        root = root_cache[cwd_key]
        if root is None:
            continue
        key = str(root)
        owners.setdefault(key, index)
        roots[key] = root

    charges = [0] * len(rows)
    incomplete_owners: set[int] = set()
    accounted = list(roots.values())
    for key, owner in owners.items():
        root = roots[key]
        nested = [other for other in accounted if other != root and _path_is_within(other, root)]
        # Exclude only the per-run state and log directories that are already
        # charged to a row, not the entire configured roots.
        charged_excludes: List[Path] = []
        for configured in (STATE_ROOT, LOG_ROOT):
            if not _path_is_within(configured.resolve(), root):
                continue
            for row in rows:
                for per_run_dir in (_state_dir(row.key), _log_dir(row.key)):
                    per_run_resolved = per_run_dir.resolve()
                    if _path_is_within(per_run_resolved, root):
                        charged_excludes.append(per_run_resolved)
        size, complete = _dir_size_bytes_complete(root, excludes=[*nested, *charged_excludes])
        charges[owner] = size
        if not complete:
            incomplete_owners.add(owner)
    return [
        row._replace(
            worktree_bytes=charges[index],
            complete=row.complete and index not in incomplete_owners,
        )
        for index, row in enumerate(rows)
    ]


def _du_collect_rows() -> List[_DuRow]:
    """One row per run, read-only: no locks, no heal, no prune, no mutation.

    Sizes are apparent size (sum of ``st_size``), matching ``_dir_size_bytes``
    used elsewhere for the same reason (`reap --include-logs`'s report
    line) — labelled explicitly in ``cmd_du``'s own header so the number's
    meaning isn't ambiguous with on-disk block usage.

    Each run's recorded launch ``cwd`` is sized too when it is a linked git
    worktree; those trees are typically far larger than STATE_ROOT and
    LOG_ROOT combined and walking them dominates the command's runtime.
    """
    state_names: set = set()
    if STATE_ROOT.is_dir():
        try:
            state_names = _real_subdir_names(STATE_ROOT)
        except OSError:
            state_names = set()
    log_names: set = set()
    if LOG_ROOT.is_dir():
        try:
            log_names = _real_subdir_names(LOG_ROOT)
        except OSError:
            log_names = set()

    rows: List[_DuRow] = []
    for name in sorted(state_names | log_names):
        try:
            _validate_run_name(name)
        except SystemExit:
            continue
        has_state = name in state_names
        has_log = name in log_names
        state_dir = _state_dir(name) if has_state else None
        log_dir = _log_dir(name) if has_log else None
        scratch_dir = log_dir / "tmp" if log_dir is not None else None

        # Assign bytes with deterministic column precedence: state, then log
        # (excluding scratch), then scratch.  Each real path is charged at most
        # once; the log column excludes any path already charged as state so
        # equal or nested roots are not double-counted.  Track completeness
        # per-column; any walk error marks the whole row incomplete.
        row_complete = True
        state_bytes, s_complete = (
            _dir_size_bytes_complete(state_dir) if state_dir is not None else (0, True)
        )
        row_complete = row_complete and s_complete
        log_excludes: List[Path] = []
        if scratch_dir is not None:
            log_excludes.append(scratch_dir)
        if state_dir is not None and log_dir is not None:
            log_excludes.append(state_dir)
        log_bytes, l_complete = (
            _dir_size_bytes_complete(log_dir, excludes=log_excludes)
            if log_dir is not None else (0, True)
        )
        row_complete = row_complete and l_complete
        scratch_bytes, sc_complete = (
            _dir_size_bytes_complete(scratch_dir) if scratch_dir is not None else (0, True)
        )
        row_complete = row_complete and sc_complete
        rows.append(_DuRow(name, 1, state_bytes, log_bytes, scratch_bytes, complete=row_complete))
    return _du_charge_worktrees(rows)


def _du_run_group(name: str) -> str:
    """Rollup group key for run ``name`` — its effective status, or
    ``preserved-log-only`` if no state dir remains."""
    state_dir = _state_dir(name)
    if not _is_dir_safe(state_dir):
        return _DU_PRESERVED_LOG_GROUP
    status = _effective_status(state_dir)
    return status if status in _DU_STATUS_GROUPS else _DU_UNRECOGNIZED_GROUP


def _du_aggregate_groups(run_rows: List[_DuRow]) -> List[_DuRow]:
    """Fold per-run rows into one row per rollup group."""
    totals: dict = {}
    for row in run_rows:
        group = _du_run_group(row.key)
        state_b, log_b, scratch_b, worktree_b, n, complete = totals.get(
            group, (0, 0, 0, 0, 0, True)
        )
        totals[group] = (
            state_b + row.state_bytes,
            log_b + row.log_bytes,
            scratch_b + row.scratch_bytes,
            worktree_b + row.worktree_bytes,
            n + 1,
            complete and row.complete,
        )
    return [
        _DuRow(group, n, state_b, log_b, scratch_b, worktree_b, complete=complete)
        for group, (state_b, log_b, scratch_b, worktree_b, n, complete) in totals.items()
    ]


def _du_sort_rows(rows: List[_DuRow]) -> List[_DuRow]:
    return sorted(rows, key=lambda r: (-r.total_bytes, r.key))


def _du_split_top(rows: List[_DuRow], top: Optional[int]) -> tuple[List[_DuRow], List[_DuRow]]:
    """Sort ``rows`` and split at ``top``, returning (shown, omitted).
    When ``top`` is None or covers all rows, omitted is empty."""
    sorted_rows = _du_sort_rows(rows)
    if top is not None and len(sorted_rows) > top:
        return sorted_rows[:top], sorted_rows[top:]
    return sorted_rows, []


def _du_fmt(n: int, *, as_bytes: bool) -> str:
    return str(n) if as_bytes else _human_size(n)


def _du_print_table(
    rows: List[_DuRow],
    total: _DuRow,
    *,
    label_header: str,
    as_bytes: bool,
    top: Optional[int],
) -> None:
    """Print one plain-aligned-columns table (no Markdown) plus a TOTAL row.
    ``total`` is passed in rather than recomputed so it always reflects
    every run even when ``rows`` was truncated by ``--top``."""
    count_header = "RUNS"
    shown, omitted = _du_split_top(rows, top)

    columns = [label_header, count_header, "STATE", "LOG", "SCRATCH", "WORKTREE", "TOTAL"]
    widths = [max(len(columns[0]), *(len(r.key) for r in shown)) if shown else len(columns[0]), len(columns[1])]

    def size_fields(r: _DuRow) -> str:
        sizes = [r.state_bytes, r.log_bytes, r.scratch_bytes, r.worktree_bytes, r.total_bytes]
        return "  ".join(f"{_du_fmt(b, as_bytes=as_bytes):>10}" for b in sizes)

    header_sizes = "  ".join(f"{c:>10}" for c in columns[2:])
    print(f"{columns[0]:<{widths[0]}}  {columns[1]:>{widths[1]}}  {header_sizes}")
    for r in shown:
        print(f"{r.key:<{widths[0]}}  {r.count:>{widths[1]}}  {size_fields(r)}")
    if omitted:
        omitted_total = sum(r.total_bytes for r in omitted)
        print(
            f"... {len(omitted)} more row(s) omitted by --top "
            f"(sum {_du_fmt(omitted_total, as_bytes=as_bytes)}; TOTAL below covers all runs)"
        )
    print(f"{total.key:<{widths[0]}}  {total.count:>{widths[1]}}  {size_fields(total)}")


def _du_row_to_dict(r: _DuRow) -> dict:
    d = {
        "runs": r.count,
        "state_bytes": r.state_bytes,
        "log_bytes": r.log_bytes,
        "scratch_bytes": r.scratch_bytes,
        "worktree_bytes": r.worktree_bytes,
        "total_bytes": r.total_bytes,
    }
    if not r.complete:
        d["complete"] = False
    return d


def cmd_du(args: argparse.Namespace) -> int:
    """Disk usage per effective status (default) or per run (``--by-run``),
    including preserved logs. Strictly read-only: no locks, no
    ``_opportunistic_heal``, no ``_prune_old_logs``, no mutation of any kind
    — only ``os.scandir``/``stat`` reads via ``_dir_size_bytes``, plus the
    ``git rev-parse`` reads that classify each launch ``cwd``, which neither
    create nor prune anything in the inspected repository.
    """
    as_bytes: bool = bool(getattr(args, "bytes", False))
    as_json: bool = bool(getattr(args, "json", False))
    by_run: bool = bool(getattr(args, "by_run", False))
    top: Optional[int] = getattr(args, "top", None)
    if as_bytes and as_json:
        # --json already emits exact integers; combining with --bytes would
        # either be a silent no-op or need to change --json's shape, both
        # confusing. Reject rather than guess which the caller meant.
        sys.exit("agent-run: --bytes has no effect with --json, which always emits exact integers")

    run_rows = _du_collect_rows()
    all_rows = run_rows if by_run else _du_aggregate_groups(run_rows)
    total_complete = all(r.complete for r in all_rows)
    total = _DuRow(
        "TOTAL",
        sum(r.count for r in all_rows),
        sum(r.state_bytes for r in all_rows),
        sum(r.log_bytes for r in all_rows),
        sum(r.scratch_bytes for r in all_rows),
        sum(r.worktree_bytes for r in all_rows),
        complete=total_complete,
    )

    if as_json:
        shown, omitted = _du_split_top(all_rows, top)
        payload = {
            "state_root": str(STATE_ROOT),
            "log_root": str(LOG_ROOT),
            # Names the attribution rule the numbers were produced under, so a
            # consumer never has to infer it from the values.
            "worktree_attribution": "charged once to the first run sharing the worktree",
            "total": _du_row_to_dict(total),
        }
        if by_run:
            payload["runs"] = [{"name": r.key, **_du_row_to_dict(r)} for r in shown]
        else:
            payload["groups"] = {r.key: _du_row_to_dict(r) for r in shown}
        if omitted:
            payload["omitted"] = {
                "count": len(omitted),
                "total_bytes": sum(r.total_bytes for r in omitted),
            }
        print(json.dumps(payload))
        return 0

    size_label = "bytes" if as_bytes else "human-readable (binary, 1024-based)"
    print(
        f"agent-run du: apparent size (st_size), {size_label}, "
        f"STATE_ROOT={STATE_ROOT} LOG_ROOT={LOG_ROOT}"
    )
    print(
        "worktrees: linked git worktrees at each run's recorded cwd, "
        "deduplicated by realpath and charged once to the first run sharing them"
    )
    if not total_complete:
        incomplete = [r for r in all_rows if not r.complete]
        print(
            f"agent-run du: WARNING: {len(incomplete)} row(s) have incomplete walk(s) "
            "due to permission errors; totals shown are lower bounds"
        )
    _du_print_table(
        all_rows,
        total,
        label_header="NAME" if by_run else "STATUS",
        as_bytes=as_bytes,
        top=top,
    )
    return 0

    size_label = "bytes" if as_bytes else "human-readable (binary, 1024-based)"
    print(
        f"agent-run du: apparent size (st_size), {size_label}, "
        f"STATE_ROOT={STATE_ROOT} LOG_ROOT={LOG_ROOT}"
    )
    print(
        "worktrees: linked git worktrees at each run's recorded cwd, "
        "deduplicated by realpath and charged once to the first run sharing them"
    )
    _du_print_table(
        all_rows,
        total,
        label_header="NAME" if by_run else "STATUS",
        as_bytes=as_bytes,
        top=top,
    )
    return 0


# ---------------------------------------------------------------------------
# steer / kill
# ---------------------------------------------------------------------------

def _require_live_interactive_run(name: str) -> tuple[Path, int]:
    d = _require_state(name)
    if _read(d / "interactive") != "1":
        # Keyed on the interactive state file, not the FIFO's existence: a
        # one-shot run may still have a FIFO but nothing reading it.
        sys.exit(
            f"agent-run: '{name}' was launched one-shot (not interactive) and cannot be steered. "
            f"Relaunch with -i or --harness ... -i to make it steerable."
        )
    pid = _require_positive_state_int(d, "pid", name)
    if not _pid_alive(pid):
        sys.exit(f"agent-run: '{name}' is not running")
    return d, pid


def _reject_raw_steer_for_codex(name: str) -> None:
    """The codex app-server adapter dispatches on the submit terminator, which
    --raw omits, so the bytes would sit in its buffer forever and steer would
    still exit 0. Fail loudly instead of silently losing the message."""
    try:
        run_json_data = json.loads((_log_dir(name) / "run.json").read_text())
    except (OSError, json.JSONDecodeError, ValueError):
        run_json_data = {}
    if isinstance(run_json_data, dict) and run_json_data.get("harness") == "codex":
        sys.exit(
            "agent-run: '--raw' is not supported for managed codex runs; "
            "the codex app-server adapter requires a newline terminator to dispatch input. "
            "Use plain 'steer' (without --raw) instead."
        )


def cmd_steer(args: argparse.Namespace) -> int:
    name = _validate_run_name(args.name)
    d, pid = _require_live_interactive_run(name)
    if args.raw:
        _reject_raw_steer_for_codex(name)
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
        # Try pidfd path first.  Only an unavailable pidfd implementation falls
        # back; permission/resource failures are real operational errors.
        pidfd_open = getattr(os, "pidfd_open", None)
        pidfd_send_signal = getattr(signal, "pidfd_send_signal", None)
        if pidfd_open is not None and pidfd_send_signal is not None:
            try:
                pfd = pidfd_open(pid, 0)
            except ProcessLookupError:
                raise
            except OSError as exc:
                if exc.errno not in {errno.ENOSYS, errno.EOPNOTSUPP, errno.EINVAL}:
                    raise
            else:
                try:
                    current = _process_identity(pid)
                    if current is None or current != expected_identity:
                        raise RuntimeError("process identity changed between verification and signal; refusing")
                    try:
                        pidfd_send_signal(pfd, sig, None, 0)
                    except OSError as exc:
                        if exc.errno == errno.ESRCH:
                            raise ProcessLookupError(pid) from exc
                        raise
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


def _pid_parent_pid(pid: int) -> Optional[int]:
    """Best-effort parent-pid lookup for ``pid``, used only to confirm that a
    recorded child pid is still the runner's actual child immediately before
    force-killing it (see ``_force_kill``)."""
    system = platform.system()
    if system == "Linux":
        try:
            stat_text = Path(f"/proc/{pid}/stat").read_text()
            _before, after = stat_text.rsplit(")", 1)
            fields = after.split()
            ppid = fields[1]
            return int(ppid) if ppid.lstrip("-").isdigit() else None
        except (IndexError, OSError, ValueError):
            return None
    if system == "Darwin":
        try:
            result = subprocess.run(
                ["ps", "-p", str(pid), "-o", "ppid="],
                capture_output=True,
                check=True,
                text=True,
                timeout=2,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        raw = result.stdout.strip()
        return int(raw) if raw.lstrip("-").isdigit() else None
    return None


def _run_is_terminal(state_dir: Path) -> bool:
    return _read(state_dir / "status") in TERMINAL_STATUSES


KILL_ESCALATION_TIMEOUT_SECONDS = 8.0
KILL_POLL_INTERVAL_SECONDS = 0.2
KILL_CHILD_REAP_TIMEOUT_SECONDS = 3.0

# Grace period for orphan-process SIGTERM before escalating to SIGKILL.
# Shorter than KILL_ESCALATION_TIMEOUT_SECONDS (8 s) because orphan runners
# have no state dir — there is no teardown work to protect, and a hanging
# process denies a slot indefinitely.
ORPHAN_KILL_GRACE_SECONDS = 5.0


def _force_kill(name: str, state_dir: Path, pid: int, expected_identity: str) -> None:
    """Force-terminate a run without orphaning its workload or leaving a
    false ``running`` state.

    SIGKILL cannot be delivered straight to the runner: it cannot be caught,
    so the runner's own teardown handler -- which kills/reaps the workload
    and publishes terminal state -- would never run, orphaning the agent
    process tree while state still claims the run is active. Instead this
    first sends the runner an authenticated, handled SIGTERM (the same
    request a plain ``kill`` sends) and gives it a bounded window to run its
    normal teardown. Only if the runner is still alive after that window --
    a wedged supervisor, not merely a slow one -- does this escalate to
    directly terminating the runner and any workload pids it had recorded,
    and it then publishes terminal state itself so a forced kill can never
    leave the run reporting ``running``.
    """
    try:
        _send_signal_to_verified_pid(pid, signal.SIGTERM, expected_identity)
    except ProcessLookupError:
        pass
    except (PermissionError, RuntimeError) as exc:
        sys.exit(f"agent-run: refusing to force-kill '{name}': {exc}")

    deadline = time.time() + KILL_ESCALATION_TIMEOUT_SECONDS
    while time.time() < deadline:
        if not _pid_alive(pid) and _run_is_terminal(state_dir):
            print(f"agent-run: {name} terminated cleanly (pid={pid})")
            return
        time.sleep(KILL_POLL_INTERVAL_SECONDS)

    # The runner did not finish its own teardown in time. While it is still
    # the live, verified owner of its recorded children, a recorded pid
    # cannot have been recycled out from under it -- a live, non-reaping
    # parent keeps its dead children as zombies, never reusable -- so
    # re-confirm parentage one more time and kill each directly, then
    # force-kill the runner itself, mirroring what its own teardown handler
    # would have done had SIGTERM reached it in time.
    verified_children = []
    for field in _AUX_PID_FIELDS:
        raw = _read(state_dir / field)
        if not raw:
            continue
        try:
            child_pid = int(raw)
        except ValueError:
            continue
        if child_pid <= 0 or not _pid_alive(child_pid):
            continue
        if _pid_parent_pid(child_pid) != pid:
            continue
        verified_children.append(child_pid)
        try:
            os.kill(child_pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass

    current_identity = _process_identity(pid)
    if current_identity is not None and current_identity == expected_identity:
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass

    reap_deadline = time.time() + KILL_CHILD_REAP_TIMEOUT_SECONDS
    survivors = [p for p in verified_children if _pid_alive(p)]
    while survivors and time.time() < reap_deadline:
        time.sleep(KILL_POLL_INTERVAL_SECONDS)
        survivors = [p for p in survivors if _pid_alive(p)]

    if not _run_is_terminal(state_dir):
        ended_at = _now_iso()
        _write(state_dir / "exit_code", f"{128 + signal.SIGKILL}\n")
        _write(state_dir / "ended_at", ended_at + "\n")
        _write(state_dir / "status", "failed\n")
        # Mirror to run.json: a SIGKILL'd runner never runs _finalize.
        log_dir = LOG_ROOT / state_dir.name
        if log_dir.is_dir():
            _write_run_json(log_dir, {
                "ended_at": ended_at,
                "exit_code": 128 + signal.SIGKILL,
                "status": "failed",
            })

    if survivors:
        print(
            f"agent-run: force-killed '{name}' (pid={pid}); "
            f"{len(survivors)} descendant process(es) did not exit "
            "(likely blocked in uninterruptible I/O)"
        )
    else:
        print(f"agent-run: force-killed unresponsive '{name}' (pid={pid})")


def _publish_forced_kill(state_dir: Path, reason: str) -> None:
    """Record a SIGKILL-terminated run as ``killed``: an uncatchable signal
    leaves the runner unable to publish its own terminal state."""
    if _run_is_terminal(state_dir):
        return
    if not (state_dir / "exit_code").exists():
        _write(state_dir / "exit_code", f"{128 + signal.SIGKILL}\n")
    _mark_terminal(state_dir, "killed", reason)


# `kill --force` cannot rule out PID reuse: this path has no birth token.
_FORCED_UNVERIFIED_NOTE = "forced — no recorded process identity to verify"


def _recorded_pgid(state_dir: Path) -> Optional[int]:
    """The recorded process group id, if present and a plausible positive
    group id — used only by the ``--force`` legacy-identity kill fallback."""
    raw = _read(state_dir / "pgid")
    try:
        pgid = int(raw) if raw else None
    except ValueError:
        return None
    return pgid if pgid is not None and pgid > 0 else None


def _force_kill_legacy(name: str, state_dir: Path, pid: int, sig: int) -> None:
    """``kill --force`` fallback for state dirs with no recorded
    ``process_identity`` (pre-dates identity tracking, or it failed to
    read at launch time).

    There is no birth token to verify, so this cannot close the PID-reuse
    TOCTOU window the identity-verified paths do — the caller's liveness
    check immediately before this call, and confinement to the run's own
    recorded process group, are the only remaining safeguards. Targets the
    recorded pgid (matching the documented manual workaround, ``kill
    -TERM -- -<pgid>``) so descendants are reached the same way normal
    teardown would reach them; falls back to the bare pid if no pgid was
    recorded. TERM/INT/HUP are single unverified signals, trusting a
    still-alive runner's own handler exactly like the verified path does. KILL
    cannot be caught, so — mirroring ``_force_kill`` — this also publishes
    terminal state itself after a bounded wait, since an unverifiable
    runner killed outright will never publish it.
    """
    pgid = _recorded_pgid(state_dir)
    if pgid is not None and pgid == os.getpgrp():
        sys.exit(
            f"agent-run: refusing to force-kill '{name}': caller shares the "
            f"run's process group ({pgid}) and would signal itself"
        )
    if sig == signal.SIGKILL and pgid is not None:
        try:
            live_pgid = os.getpgid(pid)
        except ProcessLookupError:
            # Already gone: fall through to the ProcessLookupError handler below.
            pass
        except OSError as exc:
            sys.exit(f"agent-run: refusing to force-kill '{name}': cannot verify process group ({exc})")
        else:
            if live_pgid != pgid:
                sys.exit(
                    f"agent-run: refusing to force-kill '{name}': recorded pgid {pgid} "
                    f"does not contain pid {pid} (current pgid {live_pgid})"
                )
    try:
        # Only the uncatchable SIGKILL targets the group: a catchable signal
        # sent group-wide reaches the runner's children directly, racing the
        # runner's own handler and producing non-deterministic terminal state.
        if sig == signal.SIGKILL and pgid is not None:
            os.killpg(pgid, sig)
        else:
            os.kill(pid, sig)
    except ProcessLookupError:
        print(f"agent-run: {name} is not running (pid {pid})")
        return
    except PermissionError:
        sys.exit(f"agent-run: permission denied signaling '{name}' (pid {pid})")

    if sig != signal.SIGKILL:
        sig_name = signal.Signals(sig).name
        print(
            f"agent-run: sent {sig_name} cleanup request to {name} (pid={pid}, "
            f"{_FORCED_UNVERIFIED_NOTE})"
        )
        return

    deadline = time.time() + KILL_ESCALATION_TIMEOUT_SECONDS
    while time.time() < deadline and _pid_alive(pid):
        time.sleep(KILL_POLL_INTERVAL_SECONDS)
    _publish_forced_kill(state_dir, f"kill --force ({_FORCED_UNVERIFIED_NOTE})")
    print(
        f"agent-run: force-killed '{name}' (pid={pid}, {_FORCED_UNVERIFIED_NOTE})"
    )


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
    force = bool(getattr(args, "force", False))
    recorded_identity = _read(d / "process_identity")
    # --force overrides every unconfirmable case: no birth token, an unreadable
    # one, or a mismatch. It previously applied only to the first.
    legacy_force = force and not recorded_identity
    if force and recorded_identity:
        current_identity = _process_identity(pid)
        if current_identity is None or current_identity != recorded_identity.strip():
            print(
                f"agent-run: '{name}' identity does not verify; proceeding because "
                "--force was given",
                file=sys.stderr,
            )
            legacy_force = True
    if not legacy_force:
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
    # Cleanup is implemented by the runner's catchable handlers. Only signals
    # it can catch (TERM/INT/HUP) are sent straight to it; anything else
    # could bypass cleanup entirely and is rejected up front. KILL is
    # special-cased below to a bounded escalation instead of a raw signal,
    # since it cannot be caught and would otherwise orphan the workload.
    supported = {signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGKILL}
    if sig not in supported:
        sys.exit("agent-run: only TERM, INT, HUP, or KILL are supported; arbitrary signals can bypass cleanup")
    if legacy_force:
        _force_kill_legacy(name, d, pid, sig)
        return 0
    if sig == signal.SIGKILL:
        _force_kill(name, d, pid, expected_identity)
        return 0
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
    print(f"agent-run: sent {sig_name} cleanup request to {args.name} (pid={pid})")
    return 0


# ---------------------------------------------------------------------------
# launch + runner
# ---------------------------------------------------------------------------


def cmd_launch(args: argparse.Namespace) -> int:
    name = _validate_run_name(args.name)
    # A relative --prompt-file names a file the caller typed, so it is anchored
    # to the invocation directory before --cwd moves the process; the recorded
    # <state_dir>/prompt_file is then absolute with or without --cwd.
    prompt_file: Optional[str] = getattr(args, "prompt_file", None)
    if prompt_file:
        args.prompt_file = os.path.abspath(prompt_file)
    # Pruning takes per-name locks itself.  Do it before acquiring this name's
    # lock so a stale log for this same name cannot self-deadlock on flock.
    _prune_old_logs()
    # The shared publication lock must be acquired before entering the launch
    # cwd: a concurrent reaper holds the exclusive lock during its final scan,
    # so any run visible to that scan is protected.  Acquiring shared here
    # ensures the reaper's exclusive-lock attempt blocks until this run's cwd
    # and status are both published.
    with _worktree_publication_lock(exclusive=False):
        # Worktree creation must happen inside this same shared lock: the
        # lock exists precisely so nothing visible to a concurrent reaper's
        # final exclusive-lock scan is half-created, and a freshly `git
        # worktree add`-ed directory with no run state yet is exactly that.
        created = _create_launch_worktree(args)
        # chdir next: every path below (state, logs, git facts, the launched
        # command) must observe one single effective working directory.
        _apply_launch_cwd(args)
        if created is None:
            with _launch_lock(name) as lock_fd:
                return _cmd_launch_locked(args, name, lock_fd)
        # Roll back a worktree this call created only while args has no
        # record of a process that could hold it as cwd.
        # _worktree_process_started is set immediately before the first
        # such process is created; once set, this process can no longer
        # prove that process (or whatever it has itself forked) is dead,
        # so rollback is refused rather than raced. _cmd_launch_locked
        # signals failure with sys.exit (SystemExit), not a return value,
        # so the guard must catch BaseException to see it; the original
        # failure is always re-raised, even when rollback runs and itself
        # fails.
        try:
            with _launch_lock(name) as lock_fd:
                rc = _cmd_launch_locked(args, name, lock_fd)
        except BaseException:
            if getattr(args, "_worktree_process_started", False):
                try:
                    repo_q = shlex.quote(str(created.repo))
                    cleanup_cmds = f"git -C {repo_q} worktree remove {shlex.quote(str(created.path))}"
                    if created.branch_created:
                        cleanup_cmds += (
                            f" ; git -C {repo_q} branch -D -- {shlex.quote(created.branch)}"
                        )
                    print(
                        f"agent-run: warning: launch for '{name}' failed after starting "
                        f"the runner; leaving worktree in place: {created.path} "
                        f"(branch {created.branch!r}). Once you have confirmed no "
                        f"process is still using it, clean up by hand: {cleanup_cmds}",
                        file=sys.stderr,
                    )
                except BaseException:
                    pass
                raise
            try:
                _rollback_launch_worktree(created)
            except BaseException as rollback_exc:  # noqa: BLE001
                print(
                    f"agent-run: warning: worktree rollback itself failed: {rollback_exc}",
                    file=sys.stderr,
                )
            raise
        return rc


def _apply_launch_cwd(args: argparse.Namespace) -> None:
    """Enter the ``--cwd`` directory, before any run state exists.

    ``~`` is expanded (argparse does not).  The chdir is the sole validation:
    checking exists/is_dir first would judge a path that could be replaced
    before the chdir landed, and reports nothing the chdir does not report
    itself.  An unusable DIR exits non-zero here, before the state dir is
    created, so no run is left at status=starting.

    ``PWD``/``OLDPWD`` are re-pointed afterwards because ``os.chdir`` changes
    only the kernel cwd, leaving the launched command an environment that
    disagrees with its real working directory.
    """
    requested: Optional[str] = getattr(args, "cwd", None)
    if not requested:
        return
    try:
        os.chdir(os.path.expanduser(requested))
    except NotADirectoryError:
        sys.exit(f"agent-run: --cwd is not a directory: {requested}")
    except FileNotFoundError:
        sys.exit(f"agent-run: --cwd directory does not exist: {requested}")
    except OSError as exc:
        sys.exit(f"agent-run: cannot enter --cwd {requested}: {exc}")
    except ValueError as exc:
        # Embedded NUL: os.chdir rejects the string before issuing a syscall.
        sys.exit(f"agent-run: invalid --cwd {requested!r}: {exc}")
    os.environ["PWD"] = os.getcwd()
    os.environ.pop("OLDPWD", None)


def _cmd_launch_locked(args: argparse.Namespace, name: str, lock_fd: int) -> int:
    """Perform launch setup while ``lock_fd`` serializes this run name."""
    # Managed mode builds its own argv below; raw mode takes args.command verbatim.
    harness: Optional[str] = getattr(args, "harness", None)
    is_managed = harness is not None
    argv: List[str] = [] if is_managed else list(args.command)
    if not is_managed and not argv:
        sys.exit("agent-run: missing command")
    prompt_file: Optional[str] = getattr(args, "prompt_file", None)
    if prompt_file and not Path(prompt_file).is_file():
        sys.exit(f"agent-run: prompt file not found: {prompt_file}")
    enable_planning: bool = bool(getattr(args, "enable_planning", False))
    enable_questions: bool = bool(getattr(args, "enable_questions", False))
    # _parse_launch_argv rejects this too; repeated here because cmd_launch is
    # also called with a hand-built Namespace, which bypasses the parser.
    if harness == "codex" and enable_planning:
        sys.exit(
            "agent-run: --enable-planning is unsupported for --harness codex; "
            "the codex app-server path does not expose plan mode"
        )
    _opportunistic_heal()
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
    # Durable record of "agent-run created this worktree" (vs. a human having
    # made it, or --worktree-reuse attaching to one that already existed),
    # for observability only -- reap's removal policy reads none of this and
    # its age/cleanliness gates are unchanged either way.
    if getattr(args, "_worktree_created_path", None):
        _write(d / "worktree_created", args._worktree_created_path + "\n")

    # Per-run scratch dir, on the same persistent disk as the log (not the
    # possibly-tmpfs/RAM-backed STATE_ROOT). Created eagerly, before the
    # fork, so a launch that fails before the runner even starts still
    # leaves no half-created scratch dir with nothing to point at it — and
    # so its path can be recorded in state up front for tooling to find.
    # Mode 0700: the launched command's own temp files, potentially
    # containing sensitive scratch data, should not be world/group readable.
    # Do not use mkdir(..., exist_ok=True) then chmod: besides the umask-
    # widened window between those calls, that silently reuses an attacker- or
    # other-user-owned pre-existing path. The normal launch path just removed
    # log_d above, so the exclusive mkdir succeeds; an unexpected collision is
    # explicitly lstat/owner checked before reuse (L1).
    scratch_dir = log_d / "tmp"
    try:
        os.mkdir(scratch_dir, 0o700)
    except FileExistsError:
        try:
            scratch_st = scratch_dir.lstat()
        except OSError as exc:
            sys.exit(f"agent-run: cannot inspect existing scratch dir {scratch_dir}: {exc}")
        if not _stat_module.S_ISDIR(scratch_st.st_mode) or scratch_st.st_uid != os.getuid():
            sys.exit(f"agent-run: refusing to reuse scratch dir {scratch_dir}")
    # mkdir's mode is umask-masked, so narrow it explicitly; after the owner
    # check above this is either our newly-created directory or an owned reuse.
    os.chmod(scratch_dir, 0o700)
    _write(d / "tmp_dir", str(scratch_dir) + "\n")

    # Create resources that can fail (fifo, pipe) before publishing
    # "starting" wherever possible: a failure here means the run never
    # appeared active, so there is no stranded state to clean up.
    fifo_paths: tuple[Path, ...] = ()
    if args.interactive:
        fifo_paths = (d / "stdin", d / "resize")
        try:
            for fifo_path in fifo_paths:
                # _safe_rmtree can leave the old state dir, and so an old
                # FIFO, in place on a partial/early-return path. Unlink
                # first or os.mkfifo raises FileExistsError on relaunch.
                if fifo_path.exists():
                    fifo_path.unlink()
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
    try:
        r_ack, w_ack = os.pipe()
    except OSError as exc:
        for fifo_path in fifo_paths:
            try:
                fifo_path.unlink()
            except OSError:
                pass
        sys.exit(f"agent-run: failed to create readiness pipe: {exc}")

    # Written before "starting" is published: Path.cwd() raises if the launch
    # directory is gone, and that must not happen once the run already looks
    # active with nothing behind it. _apply_launch_cwd has already entered any
    # --cwd, so this reads the one effective directory in both cases.
    cwd: Optional[str]
    try:
        cwd = str(Path.cwd())
    except OSError:
        cwd = None
    if cwd is not None:
        _write(d / "cwd", cwd + "\n")
        # Best-effort, only when cwd is itself a git repo: `watch` counts
        # commits from HEAD movement rather than commit timestamps, which
        # the committing process controls and can backdate.
        head_outcome = _watch_run_git_checked(Path(cwd), ["rev-parse", "HEAD"])
        if head_outcome.stdout is not None:
            _write(d / "launch_head", head_outcome.stdout.strip() + "\n")
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
    idle_timeout: Optional[float] = getattr(args, "idle_timeout", None)
    if idle_timeout is not None:
        # Persisted so introspection can tell whether a running run is guarded
        # by a watchdog at all, and post-mortem can reconstruct the launch.
        _write(d / "idle_timeout", f"{idle_timeout}\n")

    # Managed mode: build argv, acquire the session id, write session.json.
    # All acquisition happens pre-fork so the runner's argv is complete at fork
    # time and session.json exists before status=running is published.
    acquire_log = log_d / "session-acquire.log"
    managed_prompt: Optional[str] = None
    managed_model: Optional[str] = None
    managed_harness_args: List[str] = []
    managed_agent_mode: Optional[str] = None
    opencode_extra_agent_names: set[str] = set()
    if is_managed:
        opencode_port: Optional[int] = None
        managed_prompt = getattr(args, "prompt", None)
        managed_model = getattr(args, "model", None)
        managed_agent_mode = getattr(args, "agent_mode", None)
        managed_harness_args = getattr(args, "harness_args", [])
        managed_session_id: Optional[str] = None

        # Every agent name that could be selected at run time and could carry a
        # per-agent allow in an external config source; passed to
        # _opencode_policy_config so each one gets an explicit deny block.
        if harness == "opencode":
            opencode_extra_agent_names = (
                _opencode_agent_names_from_harness_args(managed_harness_args)
                | _opencode_collect_config_agent_names(cwd)
            )

        # Materialise an inline prompt into a file so every delivery path
        # (stdin for one-shot, FIFO for interactive) reads from prompt_file.
        if managed_prompt and not prompt_file:
            try:
                prompt_path = log_d / "prompt"
                prompt_path.write_bytes(managed_prompt.encode("utf-8"))
                prompt_file = str(prompt_path)
            except OSError as exc:
                _acquire_log_write(acquire_log, f"could not write prompt file: {exc}")

        if harness == "claude":
            # Push acquisition: agent-run mints the UUID4 and claude is told to
            # use it, so the id is known before exec.
            managed_session_id = str(uuid.uuid4())
            _record_session(log_d, acquire_log, "claude", managed_session_id, "pushed", "certain")

        elif harness == "opencode":
            # Mint-then-attach, identical in both modes: pick a free port, start a
            # temporary opencode there, poll /global/health, POST /session, kill the
            # temporary process. The id is known before exec, so --session goes into
            # the real argv. Only interactive keeps the port (the TUI serves its HTTP
            # API on it); one-shot's `opencode run` does not take --port.
            try:
                opencode_port = _find_free_port()
                _acquire_log_write(acquire_log, f"opencode port={opencode_port} selected")
            except RuntimeError as exc:
                _record_session(log_d, acquire_log, "opencode", None, "missing", "missing", str(exc))
                opencode_port = None

            if opencode_port is not None:
                managed_session_id = _opencode_prefork_mint(
                    opencode_port, name, cwd or str(Path.cwd()), acquire_log, state_dir=d,
                    enable_planning=enable_planning,
                    enable_questions=enable_questions,
                    opencode_agent_mode=managed_agent_mode,
                    extra_agent_names=opencode_extra_agent_names,
                    launch_args=args,
                )
                if managed_session_id:
                    _record_session(log_d, acquire_log, "opencode", managed_session_id,
                                    "minted", "certain")
                else:
                    _record_session(log_d, acquire_log, "opencode", None, "missing", "missing",
                                    "prefork mint failed (health poll or POST /session)")
            if not args.interactive:
                opencode_port = None

        if harness == "codex":
            # Both modes run entirely through app-server JSON-RPC. thread/start
            # mints the id post-fork in the runner, so session.json is written
            # there, not here. This argv is recorded for postmortem, never exec'd.
            _acquire_log_write(
                acquire_log,
                f"codex {'interactive' if args.interactive else 'one-shot'}: "
                "using app-server for mint+run",
            )
            argv = ["codex", "app-server"]
        else:
            managed_permissions: str = getattr(args, "permissions", _PERMISSIONS_BYPASS)
            argv = _build_managed_argv(
                harness,
                interactive=args.interactive,
                # Interactive opencode delivers the prompt post-attach via the
                # FIFO; --prompt alongside --session is silently swallowed.
                prompt=None if args.interactive else managed_prompt,
                prompt_file=None if args.interactive else prompt_file,
                model=managed_model,
                agent_mode=managed_agent_mode,
                session_id=managed_session_id,
                harness_args=managed_harness_args,
                opencode_port=opencode_port,
                permissions=managed_permissions,
                enable_planning=enable_planning,
                enable_questions=enable_questions,
            )

    _write(d / "command", _pretty_command(argv) + "\n")
    _write(d / "argv", json.dumps(argv))
    submit_mode = _persist_submit_mode(
        d, argv, getattr(args, "submit_mode", None)
    )

    # Copy the launch facts into the persistent log dir so postmortem survives a
    # reboot that wipes the ephemeral /tmp state. Never a liveness signal.
    launch_run_json: dict = {
        "name": name,
        "argv": argv,
        "command": _pretty_command(argv),
        "cwd": cwd,
        "started_at": (_read(d / "started_at") or "").strip() or None,
        "interactive": args.interactive,
        "harness": harness,
        "agent_run_version": TOOLBOX_VERSION,
        "terminal": {"cols": _LAUNCH_TERMINAL_COLS, "rows": _LAUNCH_TERMINAL_ROWS},
    }
    if is_managed:
        launch_run_json["model"] = getattr(args, "model", None)
        launch_run_json["agent_mode"] = getattr(args, "agent_mode", None)
        launch_run_json["enable_planning"] = enable_planning
        launch_run_json["enable_questions"] = enable_questions
    try:
        _write_run_json(log_d, launch_run_json)
    except Exception:  # noqa: BLE001
        pass  # never fail the run

    # Double-fork to detach from the terminal and become our own session
    # leader. The grandchild runs the actual agent.
    parent_pid = os.getpid()
    # Set immediately before the first process that can hold this worktree
    # as cwd; cmd_launch never rolls back once this is set.
    args._worktree_process_started = True
    try:
        child_pid = os.fork()
    except OSError as exc:
        # "starting" is already published with no runner behind it: unlike
        # the fifo/pipe failures above, this must explicitly resolve to a
        # terminal state rather than leaving a phantom active run.
        os.close(r_ack)
        os.close(w_ack)
        _write(d / "exit_code", "1\n")
        _write(d / "ended_at", _now_iso() + "\n")
        _write(d / "status", "failed\n")
        sys.exit(f"agent-run: failed to start agent: {exc}")
    if child_pid != 0:
        # Parent: wait for the grandchild to publish its pid, then return.
        os.close(w_ack)
        os.waitpid(child_pid, 0)  # reap the intermediate forker
        # Read the structured readiness result. EOF, malformed data, and an
        # explicit setup error all mean launch failed. None of these outcomes
        # -- nor an interrupt anywhere in this block -- triggers rollback:
        # the fork already happened, so failure here is reported (and, for a
        # worktree this launch created, warned about) but the tree is left
        # in place.
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
        print(f"agent-run: tmp_dir={scratch_dir}")
        print(f"agent-run: poll:   agent-run status {name}")
        print(f"agent-run: logs:   agent-run tail {name}")
        return 0

    # Intermediate child: fork once more. Keep the
    # inherited lock descriptor: flock ownership survives launcher death until
    # the runner has published identity and resolved readiness.
    os.close(r_ack)
    grand = os.fork()
    if grand != 0:
        # Intermediate exits; parent's waitpid reaps it.
        os._exit(0)

    # Grandchild: detach as the session/process-group leader and run the agent.
    os.setsid()
    _runner(
        d,
        log_d,
        argv,
        args.interactive,
        w_ack,
        lock_fd,
        prompt_file,
        submit_mode,
        tmp_dir=scratch_dir,
        idle_timeout=getattr(args, "idle_timeout", None),
        managed_harness=harness,
        codex_appserver_args=(
            _codex_policy_args(enable_questions=enable_questions) + managed_harness_args
            if harness == "codex" else None
        ),
        managed_prompt=managed_prompt if is_managed else None,
        managed_model=managed_model,
        enable_planning=enable_planning,
        enable_questions=enable_questions,
        opencode_agent_mode=managed_agent_mode,
        extra_agent_names=opencode_extra_agent_names,
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


_AUX_PID_FIELDS = (
    "agent_pid", "pty_pid", "keeper_pid", "prompt_pid", "watchdog_pid", "appserver_pid",
    # echo_pid/render_pid: state-file names for a runner's echo/render helper
    # children. _force_kill and _watchdog_escalate discover children solely
    # through this tuple, so a state directory carrying either name must
    # still resolve to a pid teardown signals, even though no runner in this
    # tree writes them.
    "echo_pid", "render_pid",
)


# Distinguishes a watchdog kill from an ordinary SIGTERM in `_finalize`.
IDLE_TIMEOUT_MARKER = "idle_timeout_fired"


def _parse_watchdog_startup_grace_seconds() -> float:
    raw = os.environ.get("AGENT_RUN_WATCHDOG_STARTUP_GRACE_SECS", "300")
    return _positive_finite_seconds(
        raw, "AGENT_RUN_WATCHDOG_STARTUP_GRACE_SECS", 300.0
    )


def _idle_timeout_reason(state_dir: Path) -> str:
    """Human-readable `reap_reason` for a watchdog kill, using the idle seconds
    the watchdog recorded in its marker."""
    recorded = _read(state_dir / IDLE_TIMEOUT_MARKER).strip()
    measured = f"idle>{recorded}s" if recorded else "idle"
    return f"{measured} (idle-timeout watchdog)"


def _watchdog_escalate(state_dir: Path, runner_pid: int, runner_identity: str) -> None:
    """SIGKILL the runner and its recorded children after SIGTERM was ignored.

    The watchdog is a sibling of those processes rather than their parent, so
    it can signal but never reap them. Terminal state is published here because
    a runner killed outright never publishes its own.
    """
    if _process_identity(runner_pid) != runner_identity:
        return
    own_pid = os.getpid()
    for field in _AUX_PID_FIELDS:
        raw = _read(state_dir / field)
        if not raw:
            continue
        try:
            pid = int(raw)
        except ValueError:
            continue
        if pid <= 0 or pid == own_pid:
            continue
        if _pid_parent_pid(pid) != runner_pid:
            continue
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    try:
        _send_signal_to_verified_pid(runner_pid, signal.SIGKILL, runner_identity)
    except (OSError, RuntimeError):
        return
    _publish_forced_kill(state_dir, _idle_timeout_reason(state_dir))


def _idle_watchdog_loop(
    state_dir: Path,
    log_dir: Path,
    runner_pid: int,
    runner_identity: str,
    idle_timeout: float,
    initial_log_stat: Optional[tuple[int, int, int, int]] = None,
) -> None:
    """Terminate ``runner_pid`` once the run log has stopped growing for
    ``idle_timeout`` seconds.

    An agent that stops producing output but never exits would otherwise hold
    the run at "running" indefinitely. Idleness must hold on both the wall
    clock and a monotonic clock, so an NTP step or a suspend/resume cannot
    manufacture or mask it. Enforcement is suppressed until the agent's first
    output, bounded by ``AGENT_RUN_WATCHDOG_STARTUP_GRACE_SECS``: before that
    first byte the log still carries the launcher's own ``touch`` timestamp, so
    measuring against it would kill an agent that is merely slow to start.

    Termination is delegated to the runner's SIGTERM handler so teardown and
    status publication follow the single existing path; SIGKILL escalation
    covers a runner that cannot service the signal.
    """
    log_path = log_dir / "log"
    poll = max(1.0, min(idle_timeout / 4.0, 30.0))
    startup_grace = _parse_watchdog_startup_grace_seconds()
    started_monotonic = time.monotonic()
    last_change_monotonic = started_monotonic
    last_signature = initial_log_stat
    saw_output = False

    while True:
        time.sleep(poll)
        current_identity = _process_identity(runner_pid)
        if current_identity is not None and current_identity != runner_identity:
            # PID reuse: a different process holds this pid, so stop guarding.
            return
        if current_identity is None and not _pid_alive(runner_pid):
            # Identity probe unavailable, but liveness confirms the runner is gone.
            return
        now_monotonic = time.monotonic()
        try:
            log_stat = log_path.stat()
            signature: Optional[tuple[int, int, int, int]] = (
                log_stat.st_dev, log_stat.st_ino, log_stat.st_size, log_stat.st_mtime_ns
            )
            mtime: Optional[float] = log_stat.st_mtime
        except OSError:
            # A log that cannot be stat'd is not producing output. Keep the run
            # guarded rather than looping inertly, but only once enough time has
            # passed that a transient failure cannot trigger a kill.
            mtime = None
            if now_monotonic - last_change_monotonic < max(idle_timeout, startup_grace):
                continue
        if mtime is not None:
            if last_signature is None:
                last_signature = signature
            elif signature != last_signature:
                last_signature = signature
                last_change_monotonic = now_monotonic
                saw_output = True
            idle_wall = time.time() - mtime
            idle_monotonic = now_monotonic - last_change_monotonic
            if min(idle_wall, idle_monotonic) < idle_timeout:
                continue
            if not saw_output and now_monotonic - started_monotonic < startup_grace:
                continue
        idle_seconds = now_monotonic - last_change_monotonic
        break

    # The marker only labels the kill; a failed write must never stop it, or a
    # full state dir would silently disable the watchdog entirely.
    try:
        _write(state_dir / IDLE_TIMEOUT_MARKER, f"{idle_seconds:.0f}\n")
    except OSError:
        pass
    try:
        _send_signal_to_verified_pid(runner_pid, signal.SIGTERM, runner_identity)
    except (OSError, RuntimeError):
        return
    deadline = time.monotonic() + KILL_ESCALATION_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if not _pid_alive(runner_pid):
            return
        time.sleep(KILL_POLL_INTERVAL_SECONDS)
    _watchdog_escalate(state_dir, runner_pid, runner_identity)


def _publish_or_reap_child(state_dir: Path, field: str, pid: int) -> None:
    """Persist a forked child's pid to ``field``; if that write fails, the
    child is untrackable by ``_teardown_children`` (which only reads state
    files), so terminate and reap it immediately here instead of letting it
    survive unaccounted for after the runner reports setup failure.
    """
    try:
        _write(state_dir / field, f"{pid}\n")
    except OSError:
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            os.waitpid(pid, 0)
        except ChildProcessError:
            pass
        raise


def _teardown_children(
    state_dir: Path, grace: float = 2.0, extra_pids: Optional[Iterable[int]] = None
) -> None:
    """SIGTERM (then SIGKILL after `grace` seconds) every child pid this
    runner recorded (agent_pid / pty_pid / keeper_pid / prompt_pid /
    watchdog_pid), reaping each one.

    ``extra_pids`` covers children forked but not yet (or never) published to
    a state file — e.g. the state-file write itself failed. Publication is
    not atomic with the fork, so relying solely on state files would orphan
    a child whose write failed; callers pass their locally tracked pid set
    here so teardown can still find and reap it.

    Called from both the runner's signal handler and its crash `except` so
    a hard crash can never leave the launched agent or its helpers orphaned.
    Recorded pids are direct children of this process (forked or pty.fork()'d
    in `_runner`/`_run_interactive`), so waitpid is valid here.
    Stale or corrupt state is filtered so teardown never signals itself.
    """
    own_pid = os.getpid()
    pids = []
    seen = set()
    candidates: List[int] = []
    for aux in _AUX_PID_FIELDS:
        raw = _read(state_dir / aux)
        if not raw:
            continue
        try:
            candidates.append(int(raw))
        except ValueError:
            continue
    if extra_pids:
        candidates.extend(extra_pids)
    for pid in candidates:
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


def _codex_policy_args(*, enable_questions: bool) -> List[str]:
    """Return the `-c` overrides carrying managed policy into codex app-server.

    The value must be a struct literal; codex rejects a bare boolean here.
    Plan mode has no codex app-server equivalent, so only questions are set —
    --enable-planning is rejected before launch.

    These args are prepended before any caller-supplied managed_harness_args so
    that codex's last-occurrence-wins layering lets a caller's -c override ours.
    """
    enabled = "true" if enable_questions else "false"
    return ["-c", f"tools.experimental_request_user_input={{enabled={enabled}}}"]


def _opencode_agent_names_from_harness_args(harness_args: Sequence[str]) -> set[str]:
    """Return every agent name selected via --agent in a harness_args sequence.

    Accepts both the ``--agent NAME`` and ``--agent=NAME`` spellings.
    """
    names: set[str] = set()
    for i, token in enumerate(harness_args):
        if token.startswith("--agent="):
            name = token[len("--agent="):]
        elif token == "--agent":
            following = harness_args[i + 1:i + 2]
            # A flag-form token after --agent is the next option, not a name.
            name = following[0] if following and not following[0].startswith("-") else ""
        else:
            continue
        if name:
            names.add(name)
    return names


def _opencode_collect_config_agent_names(cwd: Optional[str]) -> set[str]:
    """Return every agent name defined in OpenCode's external config sources.

    Reads the project ``opencode.json`` in ``cwd``, the user config at
    ``$XDG_CONFIG_HOME/opencode/opencode.json`` (or ``$HOME/.config/opencode/
    opencode.json`` when XDG_CONFIG_HOME is unset), and the file named by
    ``OPENCODE_CONFIG``. Sources are parsed independently, so an unreadable or
    malformed one does not hide the agents defined in the others.
    """
    names: set[str] = set()

    def _agent_names_from_file(path: str) -> set[str]:
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8", errors="replace"))
            if isinstance(data, dict) and isinstance(data.get("agent"), dict):
                return {k for k in data["agent"] if isinstance(k, str) and k}
        except (OSError, json.JSONDecodeError, ValueError):
            pass
        return set()

    if cwd:
        names |= _agent_names_from_file(os.path.join(cwd, "opencode.json"))

    xdg_cfg = os.environ.get("XDG_CONFIG_HOME")
    home = os.environ.get("HOME")
    if xdg_cfg:
        names |= _agent_names_from_file(os.path.join(xdg_cfg, "opencode", "opencode.json"))
    elif home:
        names |= _agent_names_from_file(
            os.path.join(home, ".config", "opencode", "opencode.json")
        )

    oc_cfg = os.environ.get("OPENCODE_CONFIG")
    if oc_cfg:
        names |= _agent_names_from_file(oc_cfg)

    return names


def _opencode_policy_config(
    existing: Optional[str],
    *,
    enable_planning: bool,
    enable_questions: bool,
    opencode_agent_mode: Optional[str] = None,
    extra_agent_names: Optional[Iterable[str]] = None,
) -> str:
    """Merge the managed permission policy into an OPENCODE_CONFIG_CONTENT value.

    Sets the three policy keys (question, plan_enter, plan_exit) at global scope
    and inside a per-agent block for every agent that could be selected at run
    time. OpenCode loads config sources in order — user/XDG, project
    opencode.json, OPENCODE_CONFIG, then OPENCODE_CONFIG_CONTENT — and merges
    per-agent blocks last-match-wins, so a per-agent allow from any other source
    defeats a global-only deny but not a per-agent deny placed here.

    ``opencode_agent_mode`` is the agent named by ``--agent-mode``;
    ``extra_agent_names`` covers agents named by ``--harness-arg --agent`` and
    agents defined in the external config sources.

    Any other config the caller already had survives untouched. Unparseable or
    non-object input is replaced rather than raising — the policy must apply even
    if the inherited value is junk.

    Ordering asymmetry vs claude/codex: opencode policy is delivered via env,
    not argv, so --harness-arg has no effect on the three policy keys
    (question, plan_enter, plan_exit).  A caller-supplied OPENCODE_CONFIG_CONTENT
    in the parent environment is read here as the base but its policy keys are
    always overwritten by the managed values.  Use --enable-planning or
    --enable-questions to relax the policy; there is no per-key override path.
    """
    try:
        config = json.loads(existing) if existing else {}
    except json.JSONDecodeError:
        if existing:
            print(
                f"agent-run: warning: OPENCODE_CONFIG_CONTENT is not valid JSON; "
                f"managed policy applied to empty config. "
                f"first 200 chars: {existing[:200]!r}",
                file=sys.stderr,
            )
        config = {}
    if not isinstance(config, dict):
        if existing:
            print(
                f"agent-run: warning: OPENCODE_CONFIG_CONTENT parsed to a non-object "
                f"({type(config).__name__}); managed policy applied to empty config.",
                file=sys.stderr,
            )
        config = {}

    policy_values = {
        "question": "allow" if enable_questions else "deny",
        "plan_enter": "allow" if enable_planning else "deny",
        "plan_exit": "allow" if enable_planning else "deny",
    }

    # Set at global scope.
    permission = config.get("permission")
    if not isinstance(permission, dict):
        permission = {}
        config["permission"] = permission
    permission.update(policy_values)

    # "build" is OpenCode's built-in primary agent and the default when no
    # --agent is given, so it always needs a block.
    target_agents: set[str] = {"build"}
    if opencode_agent_mode:
        target_agents.add(opencode_agent_mode)
    for name in extra_agent_names or ():
        if isinstance(name, str) and name:
            target_agents.add(name)

    agents = config.get("agent")
    if not isinstance(agents, dict):
        agents = {}
        config["agent"] = agents

    # A target agent gets a block created if it has none; an agent block the
    # caller already had gets the policy merged into its existing permission
    # block. Either way OPENCODE_CONFIG_CONTENT is loaded last, so these values
    # are the last match and win over allows from the other sources.
    for agent_name in target_agents | set(agents):
        required = agent_name in target_agents
        agent_cfg = agents.get(agent_name)
        if not isinstance(agent_cfg, dict):
            if not required:
                continue
            agent_cfg = agents[agent_name] = {}
        agent_perm = agent_cfg.get("permission")
        if not isinstance(agent_perm, dict):
            if not required:
                continue
            agent_perm = agent_cfg["permission"] = {}
        agent_perm.update(policy_values)

    return json.dumps(config, separators=(",", ":"))


def _runner(
    state_dir: Path,
    log_dir: Path,
    argv: Sequence[str],
    interactive: bool,
    ready_fd: int,
    lock_fd: int = -1,
    prompt_file: Optional[str] = None,
    submit_mode: str = SUBMIT_MODE_CR,
    tmp_dir: Optional[Path] = None,
    idle_timeout: Optional[float] = None,
    managed_harness: Optional[str] = None,
    codex_appserver_args: Optional[List[str]] = None,
    managed_prompt: Optional[str] = None,
    managed_model: Optional[str] = None,
    enable_planning: bool = False,
    enable_questions: bool = False,
    opencode_agent_mode: Optional[str] = None,
    extra_agent_names: Optional[Iterable[str]] = None,
) -> None:
    """Execute in the detached session-leader process.

    Writes pid/pgid then either execs the agent directly (non-interactive), forks
    a PTY child and shuttles FIFO <-> PTY master <-> log (interactive), or drives
    codex over app-server JSON-RPC (managed_harness == "codex"). Any prompt has
    already been materialised to prompt_file by the launcher.

    Must only be called in the post-setsid grandchild. This function mutates
    process-global os.environ (OPENCODE_CONFIG_CONTENT, TMPDIR, BUN_TMPDIR) and
    terminates via os._exit — calling it pre-fork leaks env mutations into the
    parent and any concurrent agent-run invocations sharing that process.
    """
    my_pid = os.getpid()
    log_fd = -1
    # Set immediately before the agent is exec'd, so `_finalize` can measure how
    # long it survived. None until then: a failure during runner setup is not a
    # launch failure of the agent itself.
    agent_started_monotonic: Optional[float] = None
    # True only once the agent process returned its own exit status. Signal
    # teardown and runner crashes finalize through other paths and must stay
    # plain `failed`, however early they happen.
    agent_exited_naturally = False

    def _record_launch_error(message: str) -> None:
        """Persist a launch diagnostic to both the volatile state dir and the
        durable log dir, so it survives state-dir GC and reboots."""
        payload = message[:LAUNCH_ERROR_MAX_CHARS] + "\n"
        for target in (state_dir / "launch_error", log_dir / "launch_error"):
            try:
                _write(target, payload)
            except OSError:
                pass

    def _capture_launch_error() -> None:
        """Persist the agent's first non-empty output line so `status` can state
        why the launch failed without the operator opening the log."""
        try:
            with (log_dir / "log").open("rb") as handle:
                raw = handle.read(LAUNCH_ERROR_SCAN_BYTES)
        except OSError:
            return
        for line in _strip_ansi_fallback(raw).splitlines():
            stripped = line.strip()
            if stripped:
                _record_launch_error(stripped)
                return

    def _prompt_was_required_but_unsubmitted() -> bool:
        """True when an interactive prompt file never reached the agent. The
        submission helper publishes its marker only after both FIFO writes.

        Bounded by the helper's own delay plus the launch grace: past that, the
        run may have been steered manually and done real work, which must not
        be recorded as a launch failure.
        """
        if not (interactive and prompt_file is not None):
            return False
        if (state_dir / "prompt_submitted").exists():
            return False
        if agent_started_monotonic is None:
            return False
        elapsed = time.monotonic() - agent_started_monotonic
        return elapsed <= LAUNCH_GRACE_SECONDS + PROMPT_SUBMISSION_DELAY_SECONDS

    def _finalize(code: int) -> None:
        if not (state_dir / "exit_code").exists():
            ended_at = _now_iso()
            _write(state_dir / "exit_code", f"{code}\n")
            _write(state_dir / "ended_at", ended_at + "\n")
            status = "done" if code == 0 else "failed"
            # A watchdog kill is a deliberate termination, not a crash, and
            # carries the same reap_reason every other `killed` producer writes.
            if code != 0 and (state_dir / IDLE_TIMEOUT_MARKER).exists():
                status = "killed"
                _write(state_dir / "reap_reason", _idle_timeout_reason(state_dir) + "\n")
            # A non-zero exit within the grace window is argv validation or a
            # missing binary, not work that ran; an unsubmitted prompt means the
            # agent never received its task. Both are launch failures. Codes at
            # or above 128 encode a terminating signal, which is a kill.
            elif code != 0 and code < 128 and agent_exited_naturally and agent_started_monotonic is not None:
                if time.monotonic() - agent_started_monotonic <= LAUNCH_GRACE_SECONDS:
                    status = "launch_failed"
                    _capture_launch_error()
                elif _prompt_was_required_but_unsubmitted():
                    status = "launch_failed"
                    _record_launch_error(PROMPT_UNSUBMITTED_ERROR)
            _write(state_dir / "status", status + "\n")
            # Mirror exit facts to the persistent log dir so they survive a
            # reboot that wipes the ephemeral state dir.
            _write_run_json(log_dir, {
                "ended_at": ended_at,
                "exit_code": code,
                "status": status,
            })

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
        if tmp_dir is not None:
            # Export before any child is forked/exec'd below: os.environ is
            # process-wide, so every subsequent fork/exec — including the
            # Darwin _process_identity() implementation's `ps` subprocess —
            # inherits this run's disk-backed scratch dir. Argv is
            # intentionally left untouched; only the environment carries
            # this. BUN_TMPDIR is set alongside TMPDIR, same value, same
            # condition: Bun (which OpenCode runs on) does not consult
            # TMPDIR for its own scratch space, so a Bun-based agent would
            # otherwise still spill into the shared system temp despite
            # TMPDIR being redirected. Both point at the same one scratch
            # dir — no second directory is created.
            os.environ["TMPDIR"] = str(tmp_dir)
            os.environ["BUN_TMPDIR"] = str(tmp_dir)
        if managed_harness == "opencode":
            # Same rationale as TMPDIR above: the runner process exists solely
            # for this run, so exporting here scopes the policy to the agent
            # and every helper it forks, without touching shared config files.
            os.environ["OPENCODE_CONFIG_CONTENT"] = _opencode_policy_config(
                os.environ.get("OPENCODE_CONFIG_CONTENT"),
                enable_planning=enable_planning,
                enable_questions=enable_questions,
                opencode_agent_mode=opencode_agent_mode,
                extra_agent_names=extra_agent_names,
            )

        # Let harness hooks find this run without out-of-band configuration.
        # setdefault on the roots so a hook inherits the roots this run was
        # launched with rather than the process defaults.
        os.environ["AGENT_RUN_NAME"] = state_dir.name
        os.environ.setdefault("AGENT_RUN_STATE_DIR", str(STATE_ROOT))
        os.environ.setdefault("AGENT_RUN_LOG_DIR", str(LOG_ROOT))

        runner_pgid = os.getpgid(my_pid)
        # pid is written before process identity is computed: identity may
        # shell out (a `ps` subprocess on Darwin, see _process_identity) and
        # take up to a couple of seconds, and a launcher racing to terminate
        # this runner before it owns the worktree can only find it via pid.
        _write(state_dir / "pid", f"{my_pid}\n")
        # _cmd_launch_locked called setsid() before entering _runner, so
        # pid == pgid here (this process is the session and group leader).
        _write(state_dir / "pgid", f"{runner_pgid}\n")
        identity = _process_identity(my_pid)
        if identity is None:
            raise RuntimeError("cannot record runner process identity")
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

        # Opt-in stall guard: a sibling child watching log mtime, so a wedged
        # agent reaches a terminal status instead of running forever.
        if idle_timeout is not None:
            try:
                initial = os.fstat(log_fd)
                initial_log_stat = (
                    initial.st_dev, initial.st_ino, initial.st_size, initial.st_mtime_ns
                )
            except OSError:
                initial_log_stat = None
            with _block_handled_runner_signals():
                watchdog_pid = os.fork()
                if watchdog_pid != 0:
                    _publish_or_reap_child(state_dir, "watchdog_pid", watchdog_pid)
                else:
                    _reset_runner_signal_handlers()
            if watchdog_pid == 0:
                _reset_runner_signal_handlers()
                try:
                    os.close(ready_fd)
                except OSError:
                    pass
                try:
                    _idle_watchdog_loop(
                        state_dir, log_dir, my_pid, identity, idle_timeout, initial_log_stat
                    )
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

    ready_sent = False

    def _ready() -> None:
        nonlocal ready_sent
        if ready_sent:
            return
        os.write(ready_fd, b'{"status":"ok"}')
        os.close(ready_fd)
        if lock_fd >= 0:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)
        ready_sent = True

    try:
        agent_started_monotonic = time.monotonic()
        if managed_harness == "codex":
            # Codex never execs an argv: both modes drive `codex app-server`
            # over JSON-RPC. The grandchild inherits cwd from the launcher.
            try:
                cwd_str = os.getcwd()
            except OSError:
                cwd_str = "/"
            codex_runner = (
                _run_managed_interactive_codex_appserver if interactive
                else _run_managed_oneshot_codex_appserver
            )
            exit_code = codex_runner(
                state_dir, log_dir,
                prompt=managed_prompt,
                prompt_file=prompt_file,
                cwd=cwd_str,
                harness_args=codex_appserver_args or [],
                log_fd=log_fd,
                ready=_ready,
                acquire_log=log_dir / "session-acquire.log",
                model=managed_model,
            )
        elif interactive:
            # Managed claude/opencode use the same PTY path as raw runs: the
            # session id is already in argv, and the prompt is delivered through
            # the FIFO after the TUI comes up.
            exit_code = _run_interactive(
                state_dir, argv, log_fd, _ready, prompt_file, submit_mode, log_dir
            )
        else:
            exit_code = _run_oneshot(state_dir, argv, log_fd, _ready, prompt_file)
        agent_exited_naturally = True
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
        # keeper helpers) orphaned. Never exit without reaping them.
        _teardown_children(state_dir)
        if not ready_sent:
            # Setup failed before the run became controllable (e.g. FIFO
            # open, PTY fork, or fcntl raised): the launcher is still
            # blocked reading ready_fd expecting a structured result, not a
            # closed pipe it must interpret as an ambiguous EOF. Tell it
            # explicitly so it reports failure instead of racing a false
            # "started" success.
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
                if lock_fd >= 0:
                    try:
                        fcntl.flock(lock_fd, fcntl.LOCK_UN)
                        os.close(lock_fd)
                    except OSError:
                        pass
            ready_sent = True
        exit_code = 1

    # Persistent helpers outlive the agent by design, so reap them on
    # successful completion as well as on crashes and external signals.
    _teardown_children(state_dir)
    _finalize(exit_code)
    os._exit(exit_code)


def _run_oneshot(
    state_dir: Path,
    argv: Sequence[str],
    log_fd: int,
    ready: callable,
    prompt_file: Optional[str] = None,
) -> int:
    with _block_handled_runner_signals():
        pid = os.fork()
        if pid != 0:
            _publish_or_reap_child(state_dir, "agent_pid", pid)
            _write(state_dir / "status", "running\n")
            ready()
        else:
            _reset_runner_signal_handlers()
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


def _resize_checksum(cols: int, rows: int) -> int:
    """Checksum byte closing a resize record.

    Distinguishes a correctly-anchored record from a coincidental magic
    byte inside another record's payload -- 0xA7 is also column 167, so at
    that width the magic reappears in the payload and magic-alone anchoring
    mis-frames. Each byte is weighted by position, so a window shifted
    within a record produces a different sum rather than an XOR that folds
    back to the same value.

    The weights are odd: an even one is singular modulo 256, so it discards
    the top bit of its byte and lets a shifted window collide with a valid
    record for whole families of dimensions."""
    total = RESIZE_RECORD_MAGIC * 31 + RESIZE_RECORD_VERSION
    for weight, byte in enumerate(
        (cols >> 8, cols & 0xFF, rows >> 8, rows & 0xFF), start=1
    ):
        total = (total * 31 + byte * (2 * weight - 1)) & 0xFF
    return total


def _pack_resize(cols: int, rows: int) -> bytes:
    if not (1 <= cols <= MAX_TERMINAL_DIMENSION and 1 <= rows <= MAX_TERMINAL_DIMENSION):
        raise ValueError(
            f"terminal dimensions must be between 1 and {MAX_TERMINAL_DIMENSION}"
        )
    return struct.pack(
        RESIZE_RECORD_FORMAT,
        RESIZE_RECORD_MAGIC,
        RESIZE_RECORD_VERSION,
        cols,
        rows,
        _resize_checksum(cols, rows),
    )


def _apply_resize(master_fd: int, cols: int, rows: int) -> bool:
    """Set the PTY window size, returning whether the ioctl succeeded.

    Never raises: resizing is auxiliary, and a failure here must not end a run
    that is otherwise healthy, nor escape at launch before the child pid has
    been published and become reapable.

    The return value gates the resize timeline: recording a geometry the kernel
    rejected would make replay change size at an offset where the live PTY never
    did.
    """
    try:
        fcntl.ioctl(
            master_fd,
            termios.TIOCSWINSZ,
            struct.pack("HHHH", rows, cols, 0, 0),
        )
    except OSError:
        return False
    return True


_LOG_WRITE_FAILED_MESSAGE = (
    "log write failed; the capture is truncated and resize offsets can no "
    "longer locate a geometry change, so the resize timeline stops here"
)


def _write_all_to_log(log_fd: int, data: bytes) -> "tuple[int, bool]":
    """Write all of ``data`` to ``log_fd``, returning (bytes written, complete).

    A short write would otherwise drop the tail of a chunk, and every later
    resize offset would name a position in a truncated capture rather than in
    the byte stream the child emitted. EINTR retries; any other OSError stops
    the loop and reports the write as incomplete, since the unwritten suffix has
    already been consumed from the master and cannot be recovered.
    """
    written = 0
    while written < len(data):
        try:
            written += os.write(log_fd, data[written:])
        except InterruptedError:
            continue
        except OSError:
            return written, False
    return written, True


# Ceiling on one pre-resize drain. A child writing as fast as the runner reads
# never lets the master reach EAGAIN, and the relay is single-threaded: an
# unbounded drain would starve the resize it is preparing for and stop
# servicing stdin and child exit. Past this the boundary is approximate, which
# is the same guarantee the ioctl itself gives against a concurrent writer.
DRAIN_BEFORE_RESIZE_MAX_BYTES = 4 * 1024 * 1024
DRAIN_BEFORE_RESIZE_MAX_SECONDS = 0.25


def _drain_master_to_log(master_fd: int, log_fd: int) -> "tuple[int, bool]":
    """Read the PTY master toward EAGAIN, writing to ``log_fd``.

    Returns (bytes written, capture intact). Used before applying a resize so
    output the child has already produced is logged under the geometry it was
    drawn for.

    The drain is bounded by bytes and elapsed time: reaching either leaves the
    remaining output to the ordinary relay path, so the recorded offset lands
    slightly early rather than the run stalling. A closed master or read error
    ends the drain; the caller's ordinary read path handles reaping.

    ``capture intact`` is False when a log write failed, meaning bytes consumed
    from the master were lost. The caller must stop recording resize offsets:
    they index the captured log, so a gap makes every later offset meaningless.
    """
    total = 0
    deadline = time.monotonic() + DRAIN_BEFORE_RESIZE_MAX_SECONDS
    while total < DRAIN_BEFORE_RESIZE_MAX_BYTES and time.monotonic() < deadline:
        try:
            data = os.read(master_fd, 65536)
        except InterruptedError:
            continue  # the read was interrupted, not exhausted
        except (BlockingIOError, OSError):
            break
        if not data:
            break
        written, complete = _write_all_to_log(log_fd, data)
        total += written
        if not complete:
            return total, False
    return total, True


def _resize_record_pending(buffered: bytes) -> bool:
    """True when ``buffered`` holds at least one complete, valid resize record.

    Lets the relay skip the pre-resize drain for a batch that carries only a
    partial or malformed record, which applies no resize.
    """
    while len(buffered) >= RESIZE_RECORD_SIZE:
        if _valid_resize_record(buffered[:RESIZE_RECORD_SIZE]):
            return True
        buffered = buffered[1:]
    return False


def _drain_resize_records(
    master_fd: int, buffered: bytes, warn=None, on_applied=None
) -> bytes:
    """Apply the LAST complete resize record found in ``buffered``,
    discarding earlier complete records in the same batch, and return any
    trailing partial record for the next call.

    A single ``os.read(resize_fd, 4096)`` can return many queued records at
    once -- dragging a window edge emits a burst of intermediate sizes
    faster than the runner's select() loop drains them. Applying each would
    issue a TIOCSWINSZ ioctl, and therefore a SIGWINCH to the wrapped
    agent, per already-stale intermediate size: one full TUI repaint each
    instead of one repaint at the size that actually matters.

    Bytes that don't parse as a well-formed record are skipped one at a
    time until one does. Without that, a torn write, a replaced record, or
    a second attach client writing concurrently desyncs the stream
    permanently and every subsequent record decodes as garbage dimensions.
    Validation is the checksum plus in-range dimensions, not the magic byte
    alone: 0xA7 is also column 167, so at that width the magic reappears
    inside the payload and anchoring on it alone mis-frames the record.

    A record whose magic byte is right but whose version is not this
    build's is reported through ``warn`` and skipped rather than decoded:
    an attach client from another release writes a different layout, and
    reading its bytes as dimensions drives the agent's terminal to a
    garbage size.

    ``on_applied(cols, rows)``, when given, runs after ``_apply_resize``
    succeeds, so a caller can record the resize timeline without this
    function knowing anything about logs or offsets."""
    last_record = None
    foreign_version = None
    while len(buffered) >= RESIZE_RECORD_SIZE:
        candidate = buffered[:RESIZE_RECORD_SIZE]
        rest = buffered[RESIZE_RECORD_SIZE:]
        if _valid_resize_record(candidate) and _resize_record_prefix_possible(
            rest, allow_empty=True
        ):
            last_record = candidate
            buffered = rest
            continue
        if (
            candidate[0] == RESIZE_RECORD_MAGIC
            and candidate[1] != RESIZE_RECORD_VERSION
        ):
            foreign_version = candidate[1]
        # Advance a single byte: a candidate boundary can be any offset, so
        # every offset has to be tried. Skipping ahead to the next magic
        # byte would be equivalent here only because a record must start
        # with one -- it is not a weaker check, just a faster path this
        # code does not need.
        buffered = buffered[1:]
    # Keep only a trailing run that could still complete into a record.
    while buffered and not _resize_record_prefix_possible(buffered):
        buffered = buffered[1:]
    if foreign_version is not None and last_record is None and warn is not None:
        warn(
            f"ignoring resize record with unsupported format version "
            f"{foreign_version} (this build speaks version "
            f"{RESIZE_RECORD_VERSION}); terminal size not updated"
        )
    if last_record is not None:
        _magic, _version, cols, rows, _sum = struct.unpack(
            RESIZE_RECORD_FORMAT, last_record
        )
        if _apply_resize(master_fd, cols, rows):
            if on_applied is not None:
                on_applied(cols, rows)
        elif warn is not None:
            # Common during teardown once the master is closed, so this is a
            # note rather than a fault; the timeline deliberately gets no record.
            warn(f"terminal resize to {cols}x{rows} was refused; size unchanged")
    return buffered


def _valid_resize_record(candidate: bytes) -> bool:
    if len(candidate) != RESIZE_RECORD_SIZE or candidate[0] != RESIZE_RECORD_MAGIC:
        return False
    _magic, version, cols, rows, checksum = struct.unpack(
        RESIZE_RECORD_FORMAT, candidate
    )
    if version != RESIZE_RECORD_VERSION:
        return False
    if not (1 <= cols <= MAX_TERMINAL_DIMENSION and 1 <= rows <= MAX_TERMINAL_DIMENSION):
        return False
    return checksum == _resize_checksum(cols, rows)


def _resize_record_prefix_possible(buffered: bytes, allow_empty: bool = False) -> bool:
    """True when ``buffered`` could still be the start of a valid record
    once more bytes arrive -- i.e. it begins with the magic byte.

    ``allow_empty`` accepts an exhausted buffer as well, which is what the
    boundary check after a candidate record needs: whatever follows a real
    record is either nothing yet or the next record's magic byte."""
    if not buffered:
        return allow_empty
    return buffered[:1] == bytes([RESIZE_RECORD_MAGIC])


def _resize_protocol_matches(state_dir: Path) -> bool:
    """True when this build can exchange resize records with the runner
    that owns ``state_dir``.

    An absent marker means a runner that predates resize versioning. Those
    read the original unversioned 5-byte record, so a versioned one is
    garbage to them -- treat it as a mismatch rather than assuming
    compatibility, since silently driving the agent's terminal to a bogus
    size is worse than forwarding no resizes at all.

    An unreadable or non-UTF-8 marker is a mismatch for the same reason:
    whatever wrote those bytes is not a runner speaking this format, and a
    corrupt state file must not take attach down with a traceback."""
    try:
        recorded = (state_dir / RESIZE_PROTOCOL_MARKER).read_text().strip()
    except (OSError, ValueError):
        return False
    return recorded == str(RESIZE_RECORD_VERSION)


def _fork_fifo_keeper(state_dir: Path) -> int:
    """Fork a child that holds the run's control FIFOs open for writing, and
    return its pid.

    Without it the FIFO reader sees EOF whenever no steer is in flight. The child
    opens O_RDWR (which never blocks), acks over a pipe so the caller knows the
    write end is held before opening the read end, then sleeps until reaped.

    Both the stdin and resize FIFOs are held: attach writes resize records to
    the second one, and a reader that saw EOF there would stop tracking the
    client's terminal size for the rest of the session.
    """
    stdin_path = state_dir / "stdin"
    resize_path = state_dir / "resize"
    keeper_r, keeper_w = os.pipe()
    with _block_handled_runner_signals():
        keeper_pid = os.fork()
        if keeper_pid != 0:
            _publish_or_reap_child(state_dir, "keeper_pid", keeper_pid)
        else:
            _reset_runner_signal_handlers()
    if keeper_pid == 0:
        _reset_runner_signal_handlers()
        os.close(keeper_r)
        # Open FIFOs for writing (blocks until a reader appears, that's us below).
        # Use a background-safe open: O_RDWR avoids the reader-blocking behavior.
        #
        # Both opens are guarded: this keeper is a real fork() of the
        # runner, sharing its Python call stack, not an exec'd child. An
        # unguarded OSError here (ENOENT from a racing FIFO removal,
        # EMFILE/ENFILE under fd pressure) would propagate straight up
        # that shared stack into _runner's own `except Exception` handler
        # -- but running *inside this keeper process*, not the real
        # runner. That handler releases the per-name launch lock
        # (`fcntl.flock(lock_fd, LOCK_UN)`) on the assumption it's the
        # runner cleaning up after itself; here it would release the lock
        # prematurely, before the real runner has reached readiness,
        # letting a concurrent relaunch/reap act on state that's still
        # mid-setup. Fail this process directly instead: os._exit without
        # acking leaves keeper_r's read at the parent returning empty,
        # which is already handled below as "keeper failed to open
        # control FIFOs" -- the intended, single, unambiguous failure
        # path.
        try:
            fd = os.open(str(stdin_path), os.O_RDWR)
            resize_fd_keeper = os.open(str(resize_path), os.O_RDWR)
        except OSError:
            os._exit(1)
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
            try:
                os.close(resize_fd_keeper)
            except OSError:
                pass
        os._exit(0)

    os.close(keeper_w)
    # Wait for keeper to open the FIFOs. An empty read means the keeper died
    # (e.g. failed to open one of the control FIFOs) before acking; proceeding
    # would hang forever on the blocking FIFO opens below.
    try:
        ack = os.read(keeper_r, 1)
    except OSError:
        ack = b""
    os.close(keeper_r)
    if not ack:
        raise RuntimeError("keeper failed to open control FIFOs")
    return keeper_pid


def _reap_fifo_keeper(keeper_pid: Optional[int]) -> None:
    """Terminate and reap the FIFO keeper child."""
    if keeper_pid is None:
        return
    try:
        os.kill(keeper_pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        os.waitpid(keeper_pid, 0)
    except ChildProcessError:
        pass


def _open_fifo_reader(fifo_path: Path) -> int:
    """Open the FIFO read end non-blocking; reads are gated by select."""
    fifo_fd = os.open(str(fifo_path), os.O_RDONLY)
    fcntl.fcntl(fifo_fd, fcntl.F_SETFL, fcntl.fcntl(fifo_fd, fcntl.F_GETFL) | os.O_NONBLOCK)
    return fifo_fd


def _run_interactive(
    state_dir: Path,
    argv: Sequence[str],
    log_fd: int,
    ready: callable,
    prompt_file: Optional[str] = None,
    submit_mode: str = SUBMIT_MODE_CR,
    log_dir: Optional[Path] = None,
) -> int:
    stdin_path = state_dir / "stdin"
    resize_path = state_dir / "resize"
    keeper_pid = _fork_fifo_keeper(state_dir)

    # Fork + PTY for the agent.
    with _block_handled_runner_signals():
        pty_pid, master_fd = pty.fork()
        if pty_pid != 0:
            # Publish first: an exception before this strands a live child that
            # teardown has no record of. Sizing the PTY afterwards still beats
            # the child to its first TIOCGWINSZ in practice, and the ioctl
            # raises SIGWINCH, so an agent that already read 0x0 re-queries.
            _publish_or_reap_child(state_dir, "pty_pid", pty_pid)
            _apply_resize(master_fd, _LAUNCH_TERMINAL_COLS, _LAUNCH_TERMINAL_ROWS)
        else:
            _reset_runner_signal_handlers()
    if pty_pid == 0:
        _reset_runner_signal_handlers()
        # Child: stdin/stdout/stderr are all the PTY slave. Exec.
        try:
            os.execvp(argv[0], list(argv))
        except OSError as exc:
            # Unbuffered, matching the one-shot path: os._exit does not flush, and
            # this is the line _capture_launch_error reports for a failed exec.
            os.write(2, f"agent-run: exec failed: {exc}\n".encode())
            os._exit(127)
    # Full interactive relay is now established: PTY forked and keeper holds
    # FIFO open.  Finish the descriptor setup before declaring readiness.
    # If a prompt file was provided, fork a helper that waits for the TUI to
    # finish initializing (so the PTY is in raw mode and Enter is recognized),
    # then writes the prompt + selected Enter sequence to the FIFO so the agent
    # human had typed it. Same pattern as `agent-run steer`.
    if prompt_file:
        with _block_handled_runner_signals():
            helper = os.fork()
            if helper != 0:
                _publish_or_reap_child(state_dir, "prompt_pid", helper)
            else:
                _reset_runner_signal_handlers()
        if helper == 0:
            _reset_runner_signal_handlers()
            # Detach from the parent's stdio. Errors are silent (no log to write to).
            try:
                # Wait a few seconds for the TUI to enable raw mode. Earlier
                # tests showed sub-3s delivery races ICRNL CR->LF translation.
                time.sleep(PROMPT_SUBMISSION_DELAY_SECONDS)
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
                    fd = os.open(str(stdin_path), os.O_WRONLY)
                    try:
                        os.write(fd, submit_writes[0])
                    finally:
                        os.close(fd)
                    time.sleep(0.5)
                    fd = os.open(str(stdin_path), os.O_WRONLY)
                    try:
                        os.write(fd, submit_writes[1])
                    finally:
                        os.close(fd)
                    # Marks delivery for `_finalize`; absence means the agent
                    # died before its task ever reached it.
                    _write(state_dir / "prompt_submitted", _now_iso() + "\n")
                except OSError:
                    pass
            finally:
                os._exit(0)

    # Open FIFO read ends (blocks until the keeper has opened for writing,
    # which it has by the time we got the ack). Both are non-blocking; the
    # select loop below gates every read.
    fifo_fd = _open_fifo_reader(stdin_path)
    resize_fd = _open_fifo_reader(resize_path)

    # Make master non-blocking so reads don't stall when select lies briefly.
    flags = fcntl.fcntl(master_fd, fcntl.F_GETFL)
    fcntl.fcntl(master_fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
    # Advertise the resize record layout this runner reads, so an attach
    # client from a different release can warn instead of writing records
    # this loop would discard.
    _write(state_dir / RESIZE_PROTOCOL_MARKER, f"{RESIZE_RECORD_VERSION}\n")
    _write(state_dir / "status", "running\n")
    ready()

    def warn_resize(message: str) -> None:
        # The runner's stdout/stderr is the log file, which is where an
        # operator looks when the terminal size stops tracking.
        try:
            os.write(log_fd, f"agent-run: {message}\r\n".encode())
        except OSError:
            pass

    log_bytes_written = 0
    # Cleared when a log write fails: recorded offsets index the captured log,
    # so a gap makes every later offset meaningless.
    log_capture_intact = True

    def record_resize(cols: int, rows: int) -> None:
        nonlocal log_capture_intact
        # Resizes are out-of-band (TIOCSWINSZ + SIGWINCH): no byte of the
        # ioctl reaches the log, so the offset it took effect at has to be
        # recorded here, not recovered from the stream later. Bytes at or
        # after log_bytes_written are the first the new geometry applies to.
        if log_dir is None or not log_capture_intact:
            return
        record = json.dumps(
            {"offset": log_bytes_written, "cols": cols, "rows": rows}
        )
        try:
            with (log_dir / "resizes.jsonl").open("a", encoding="utf-8") as f:
                f.write(record + "\n")
        except OSError:
            pass

    exit_code: Optional[int] = None
    buf_in = b""
    buf_resize = b""
    while True:
        try:
            writable = [master_fd] if buf_in else []
            r, w, _ = select.select([master_fd, fifo_fd, resize_fd], writable, [], 0.5)
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
                written, complete = _write_all_to_log(log_fd, data)
                log_bytes_written += written
                if not complete and log_capture_intact:
                    log_capture_intact = False
                    warn_resize(_LOG_WRITE_FAILED_MESSAGE)

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

        if resize_fd in r:
            try:
                chunk = os.read(resize_fd, 4096)
            except BlockingIOError:
                chunk = b""
            except OSError:
                chunk = b""
            if chunk:
                buf_resize += chunk
            if _resize_record_pending(buf_resize):
                # Everything the child has already produced belongs to the old
                # geometry, but a single 4096-byte read leaves the rest queued
                # on the master to be logged after the resize offset. Drain
                # toward EAGAIN first so the recorded offset names a real
                # boundary. Output generated concurrently between this drain
                # and the ioctl is an unavoidable PTY race: the child writes
                # without coordinating with the resize.
                drained, drain_intact = _drain_master_to_log(master_fd, log_fd)
                log_bytes_written += drained
                if not drain_intact and log_capture_intact:
                    log_capture_intact = False
                    warn_resize(_LOG_WRITE_FAILED_MESSAGE)
            buf_resize = _drain_resize_records(
                master_fd, buf_resize, warn_resize, record_resize
            )

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
        os.close(fifo_fd)
    except OSError:
        pass
    try:
        os.close(resize_fd)
    except OSError:
        pass
    try:
        os.close(master_fd)
    except OSError:
        pass
    _reap_fifo_keeper(keeper_pid)
    return exit_code if exit_code is not None else 0


# ---------------------------------------------------------------------------
# Managed mode — harness-specific command building and session acquisition
# ---------------------------------------------------------------------------

# Valid harness names for --harness.
MANAGED_HARNESSES: frozenset[str] = frozenset({"claude", "opencode", "codex"})

# How long to wait for opencode's HTTP API to report healthy, and how often to ask.
_OPENCODE_HEALTH_TIMEOUT = 30.0
_OPENCODE_HEALTH_POLL_INTERVAL = 0.25

# Timeout for the codex app-server initialize + thread/start handshake (seconds).
_CODEX_APPSERVER_TIMEOUT = 20.0

# Cap on the rendered thread/start error text stored in session.json's reason field.
_THREAD_START_ERROR_REASON_MAX = 300

_CODEX_AUTH_PATH = Path.home() / ".codex" / "auth.json"


def _codex_subprocess_env() -> Optional[dict]:
    """Return an environment for codex app-server subprocesses, or None to inherit.

    Codex reads its API key from the variable named by config.env_key (typically
    OPENAI_API_KEY). agent-run double-forks into a detached runner, so a launching
    shell that never exported it leaves the subprocess with nothing and codex dies
    with "Missing environment variable". Read the key from codex's own credential
    store instead, leaving every other inherited variable intact. ~/.codex/auth.json
    is opened read-only and only when the variable is unset.
    """
    key_name = "OPENAI_API_KEY"
    if os.environ.get(key_name):
        return None
    try:
        auth = json.loads(_CODEX_AUTH_PATH.read_text())
        api_key = auth.get(key_name)
        if api_key and isinstance(api_key, str):
            env = dict(os.environ)
            env[key_name] = api_key
            return env
    except (OSError, json.JSONDecodeError, TypeError):
        pass
    return None


def _find_free_port() -> int:
    """Ask the kernel for a free loopback port.

    The port is released immediately on return; opencode binds it moments later,
    so a narrow race is possible but unavoidable without holding the fd open through
    exec. Raises RuntimeError if the kernel cannot allocate a port.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]
    except OSError as exc:
        raise RuntimeError(f"no free TCP port: {exc}") from exc


def _opencode_health_poll(port: int, timeout: float, acquire_log: Path) -> bool:
    """Poll GET /global/health until {"healthy":true} or timeout.

    Diagnostic messages go to acquire_log only, never to the PTY log.
    Returns True when healthy, False on timeout.
    """
    url = f"http://127.0.0.1:{port}/global/health"
    deadline = time.monotonic() + timeout
    attempt = 0
    while time.monotonic() < deadline:
        attempt += 1
        try:
            with urllib.request.urlopen(url, timeout=2.0) as resp:
                body = resp.read().decode("utf-8", errors="replace")
                data = json.loads(body)
                if data.get("healthy") is True:
                    _acquire_log_write(acquire_log, f"health ok after {attempt} attempt(s)")
                    return True
        except (urllib.error.URLError, json.JSONDecodeError, OSError):
            pass
        time.sleep(_OPENCODE_HEALTH_POLL_INTERVAL)
    _acquire_log_write(acquire_log, f"health poll timed out after {timeout:.1f}s ({attempt} attempt(s))")
    return False


def _opencode_mint_session(
    port: int, title: str, expected_cwd: str, acquire_log: Path
) -> Optional[str]:
    """POST /session to mint a new opencode session, then verify server identity.

    Returns the session id if and only if the server's response confirms the
    session directory matches expected_cwd — proving the responder is the process
    we started in that directory, not a foreign server that raced to bind the port.
    Returns None on any mismatch or failure; caller must degrade to missing.
    """
    url = f"http://127.0.0.1:{port}/session"
    payload = json.dumps({"title": title}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10.0) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            data = json.loads(body)
            session_id = data.get("id")
            if not (session_id and isinstance(session_id, str)):
                _acquire_log_write(acquire_log, f"POST /session returned no id: {body[:200]}")
                return None
            # Verify that the returned session belongs to our process: opencode
            # sets the session's directory to its own cwd, so a match proves the
            # responder was started in the same directory as this launch.
            session_dir = data.get("directory", "")
            if os.path.realpath(session_dir) != os.path.realpath(expected_cwd):
                _acquire_log_write(
                    acquire_log,
                    f"POST /session identity check failed: session directory "
                    f"{session_dir!r} != expected cwd {expected_cwd!r}; "
                    "a foreign opencode server may own the port — degrading to missing",
                )
                return None
            _acquire_log_write(acquire_log, f"minted session id={session_id!r} directory={session_dir!r}")
            return session_id
    except (urllib.error.URLError, json.JSONDecodeError, OSError) as exc:
        _acquire_log_write(acquire_log, f"POST /session failed: {exc}")
        return None


def _opencode_prefork_mint(
    port: int,
    run_name: str,
    cwd: str,
    acquire_log: Path,
    state_dir: Optional[Path] = None,
    enable_planning: bool = False,
    enable_questions: bool = False,
    opencode_agent_mode: Optional[str] = None,
    extra_agent_names: Optional[Iterable[str]] = None,
    launch_args: Optional[argparse.Namespace] = None,
) -> Optional[str]:
    """Start a temporary opencode process, mint a session, return the session id.

    Launches `opencode --port <port> --auto` so its HTTP API is reachable, polls
    until healthy, POSTs /session, then kills the temporary process. The minted
    session is registered in opencode's database and is continued when the real
    invocation runs with --session <id>. Returns None if any step fails; the
    caller must degrade to confidence=missing.

    state_dir lets the signal handler resolve the phantom status=starting run
    that would otherwise be left behind if the launcher is killed mid-poll.

    launch_args, when given, is the same Namespace cmd_launch reads for
    ``_worktree_process_started``. It is set immediately before the ``Popen``
    below -- the temporary process's cwd is the launch worktree -- and is
    cleared only when ``Popen`` itself raises, proving no child was created.
    Reaping the direct child does not prove its descendants are dead: a
    descendant inherits the worktree as cwd and can outlive that reap, so a
    successful wait must never clear the mark.
    """
    mark_was_set = bool(getattr(launch_args, "_worktree_process_started", False))
    if launch_args is not None:
        launch_args._worktree_process_started = True
    mint_env = dict(os.environ)
    mint_env["OPENCODE_CONFIG_CONTENT"] = _opencode_policy_config(
        mint_env.get("OPENCODE_CONFIG_CONTENT"),
        enable_planning=enable_planning,
        enable_questions=enable_questions,
        opencode_agent_mode=opencode_agent_mode,
        extra_agent_names=extra_agent_names,
    )
    try:
        proc = subprocess.Popen(
            ["opencode", "--port", str(port), "--auto"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=mint_env,
            cwd=cwd,
        )
    except OSError as exc:
        _acquire_log_write(acquire_log, f"could not start opencode for mint: {exc}")
        if launch_args is not None and not mark_was_set:
            launch_args._worktree_process_started = False
        return None

    # SIGTERM/SIGINT/SIGHUP (the same set _runner's _on_signal handles) must kill
    # the mint process and write a terminal status before re-raising: a run stuck
    # at status=starting with no pid is never healed or reaped, so watch reports
    # terminal:false forever and the run name stays occupied.
    _mint_proc_ref = [proc]
    _handled_signals = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)
    _orig_handlers = [signal.getsignal(sig) for sig in _handled_signals]

    def _restore_handlers() -> None:
        for sig, orig in zip(_handled_signals, _orig_handlers):
            signal.signal(sig, orig)

    def _mint_cleanup_handler(signum, frame):
        p = _mint_proc_ref[0]
        if p is not None:
            try:
                p.terminate()
            except OSError:
                pass
            try:
                p.wait(timeout=3.0)
            except Exception:  # noqa: BLE001
                try:
                    p.kill()
                except OSError:
                    pass
        if state_dir is not None:
            try:
                _write(state_dir / "exit_code", "1\n")
                _write(state_dir / "ended_at", _now_iso() + "\n")
                _write(state_dir / "status", "failed\n")
            except OSError:
                pass
        # Re-raise so the caller's normal signal handling takes over.
        _restore_handlers()
        os.kill(os.getpid(), signum)

    try:
        for sig in _handled_signals:
            signal.signal(sig, _mint_cleanup_handler)

        if not _opencode_health_poll(port, _OPENCODE_HEALTH_TIMEOUT, acquire_log):
            _acquire_log_write(acquire_log, "opencode server did not become healthy for mint")
            return None
        return _opencode_mint_session(port, run_name, cwd, acquire_log)
    finally:
        _restore_handlers()
        _mint_proc_ref[0] = None
        try:
            proc.terminate()
            proc.wait(timeout=5.0)
        except BaseException:
            try:
                proc.kill()
            except BaseException:
                pass
            raise


def _acquire_log_write(path: Path, message: str) -> None:
    """Append a timestamped line to the session acquisition diagnostic log."""
    try:
        with path.open("a") as f:
            f.write(f"{_now_iso()} {message}\n")
    except OSError:
        pass


def _write_session_json(log_dir: Path, data: dict) -> None:
    """Atomically write session.json into log_dir.

    Uses a pid+nanosecond-unique temp file in the same directory so
    os.replace is atomic. On write failure, unlinks the temp and re-raises;
    callers that must never propagate should catch OSError themselves.
    """
    path = log_dir / "session.json"
    tmp = log_dir / f".session.{os.getpid()}.{time.time_ns()}.tmp"
    try:
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(path)
    except OSError:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _read_session_json(log_dir: Path) -> Optional[dict]:
    """Read session.json from log_dir, or None if absent, malformed, or non-object."""
    try:
        data = json.loads((log_dir / "session.json").read_text(errors="replace"))
        return data if isinstance(data, dict) else None
    except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError):
        return None


def _run_json_cwd(log_dir: Path) -> Optional[str]:
    """Read the launch ``cwd`` recorded in run.json, or None if absent,
    malformed, or not a string. Used to locate a claude session's project
    directory (see ``_claude_session_path``) without depending on the
    ephemeral state dir's ``cwd`` file, which a preserved-log-only run no
    longer has."""
    try:
        data = json.loads((log_dir / "run.json").read_text())
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    cwd = data.get("cwd") if isinstance(data, dict) else None
    return cwd if isinstance(cwd, str) else None


def _write_run_json(log_dir: Path, data: dict) -> None:
    """Merge data into run.json in log_dir, atomically. Never raises.

    run.json records immutable launch facts (name, argv, command, cwd,
    started_at, harness, interactive) at launch and exit facts (ended_at,
    exit_code, status) from _finalize, hence the read-merge-replace. It lives in
    the persistent log dir so postmortem survives a reboot that wipes the
    ephemeral /tmp state; the /tmp files are still written exactly as before and
    run.json is never a liveness signal.
    """
    path = log_dir / "run.json"
    try:
        existing: dict = json.loads(path.read_text()) if path.exists() else {}
    except Exception:  # noqa: BLE001
        existing = {}
    existing.update(data)
    tmp = log_dir / f".run.{os.getpid()}.{time.time_ns()}.tmp"
    try:
        tmp.write_text(json.dumps(existing, indent=2))
        tmp.replace(path)
    except OSError:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def _appserver_split_lines(buf: bytes) -> tuple[list[bytes], bytes]:
    """Split buf into complete newline-terminated lines plus the remainder."""
    lines: list[bytes] = []
    while b"\n" in buf:
        nl = buf.index(b"\n")
        lines.append(buf[:nl])
        buf = buf[nl + 1:]
    return lines, buf


def _appserver_read_lines(out_fd: int, buf: bytes, timeout: float) -> tuple[list[bytes], bytes, bool]:
    """Drain all available bytes from raw fd out_fd, return (complete_lines, remainder, eof).

    If timeout > 0, waits up to that many seconds for data to arrive. Then
    drains all bytes currently available in the kernel buffer so frames are
    never stranded due to a BufferedReader/select mismatch.

    eof is True when the write end of the pipe has been closed. It is a distinct
    return value because ([], buf) alone cannot tell "closed" from "nothing yet",
    and a closed pipe fd stays permanently readable — callers that cannot
    distinguish the two busy-loop at 100% CPU. Break the read loop when eof.
    """
    if timeout > 0:
        try:
            ready, _, _ = select.select([out_fd], [], [], timeout)
        except OSError:
            ready = []
        if not ready:
            lines, buf = _appserver_split_lines(buf)
            return lines, buf, False

    # Drain all available bytes without blocking further.
    eof = False
    while True:
        try:
            chunk = os.read(out_fd, 65536)
        except BlockingIOError:
            break
        except OSError:
            eof = True
            break
        if not chunk:
            # os.read returns b"" when the write end of the pipe is closed.
            eof = True
            break
        buf += chunk
        try:
            more, _, _ = select.select([out_fd], [], [], 0)
        except OSError:
            break
        if not more:
            break

    lines, buf = _appserver_split_lines(buf)
    return lines, buf, eof


def _managed_prompt_text(prompt: Optional[str], prompt_file: Optional[str]) -> str:
    """Resolve the managed-mode prompt, preferring prompt_file when both are set."""
    if prompt_file:
        try:
            return Path(prompt_file).read_text(errors="replace")
        except OSError as exc:
            return f"(prompt file unreadable: {exc})"
    return prompt or ""


def _log_write(log_fd: int, data: bytes) -> None:
    """Write agent output to the run log, ignoring write failures."""
    try:
        os.write(log_fd, data)
    except OSError:
        pass


class _CodexAppServer:
    """JSON-RPC client for one ``codex app-server`` subprocess.

    Shared by the managed-codex one-shot and interactive runners: both need the
    same initialize/thread-start handshake, the same non-blocking frame reader,
    and the same pid publication and teardown. The app-server must stay alive
    across ``thread/start`` and ``turn/start`` because ``thread/start`` only
    allocates the id — the rollout file is not written until the turn runs.

    JSON-RPC chatter goes to session-acquire.log only, never the PTY log.
    """

    def __init__(self, state_dir: Path, log_dir: Path, acquire_log: Path, tag: str) -> None:
        self.state_dir = state_dir
        self.log_dir = log_dir
        self.acquire_log = acquire_log
        self.tag = tag
        self.proc: Optional[subprocess.Popen] = None
        self.out_fd = -1
        self.thread_id: Optional[str] = None
        self._buf = b""
        self._rpc_id = 0

    def log(self, message: str) -> None:
        _acquire_log_write(self.acquire_log, f"{self.tag}: {message}")

    def _record_missing(self, reason: str, ready: "callable") -> None:
        """Degrade to confidence=missing and let the run proceed to a terminal state."""
        _record_session(self.log_dir, self.acquire_log, "codex", None, "missing", "missing", reason)
        _write(self.state_dir / "status", "running\n")
        ready()

    def start(self, harness_args: List[str], cwd: str, ready: "callable") -> bool:
        """Spawn the app-server and publish its pid. False means acquisition failed."""
        try:
            self.proc = subprocess.Popen(
                ["codex", "app-server"] + list(harness_args),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                cwd=cwd,
                env=_codex_subprocess_env(),
            )
        except OSError as exc:
            self.log(f"launch failed: {exc}")
            self._record_missing(f"codex app-server launch failed: {exc}", ready)
            return False
        # appserver_pid must land in the state dir immediately: it is the only
        # channel _teardown_children, _force_kill and _watchdog_escalate read,
        # and the runner may be killed before any finally block runs.
        # _publish_or_reap_child kills and reaps the child if the write fails.
        try:
            _publish_or_reap_child(self.state_dir, "appserver_pid", self.proc.pid)
        except OSError:
            self._record_missing("could not publish app-server pid", ready)
            return False
        # Raw fd, non-blocking: a BufferedReader's userspace buffer would make
        # select report not-ready with frames already pending.
        self.out_fd = self.proc.stdout.fileno()
        fcntl.fcntl(
            self.out_fd, fcntl.F_SETFL,
            fcntl.fcntl(self.out_fd, fcntl.F_GETFL) | os.O_NONBLOCK,
        )
        return True

    def send(self, obj: dict) -> None:
        self.proc.stdin.write((json.dumps(obj) + "\n").encode())
        self.proc.stdin.flush()

    def call(self, method: str, params: dict, rpc_id: Optional[int] = None) -> int:
        """Send a request and return its rpc id."""
        if rpc_id is None:
            self._rpc_id += 1
            rpc_id = self._rpc_id
        self.send({"jsonrpc": "2.0", "id": rpc_id, "method": method, "params": params})
        return rpc_id

    def next_id(self) -> int:
        self._rpc_id += 1
        return self._rpc_id

    def read_frames(self, timeout: float) -> tuple[list[dict], bool, bool]:
        """Return (decoded frames, whether any line arrived, eof).

        ``saw_lines`` is reported separately from the frame list so callers can
        tell "nothing arrived" from "a frame arrived but did not decode".
        """
        lines, self._buf, eof = _appserver_read_lines(self.out_fd, self._buf, timeout)
        frames: list[dict] = []
        for line_bytes in lines:
            try:
                frames.append(json.loads(line_bytes.decode("utf-8", errors="replace")))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
        return frames, bool(lines), eof

    def mint_thread(self, cwd: str, model: Optional[str] = None) -> Optional[str]:
        """Run initialize + thread/start and write session.json. Returns the thread id.

        ``model`` is forwarded verbatim in thread/start params when given; omitted
        entirely (not sent as null) when None, so codex falls back to whatever
        model_provider/model the operator's ~/.codex/config.toml selects.
        """
        self.call("initialize", {
            "clientInfo": {"name": "codex_exec", "title": "agent-run", "version": "0"},
        })
        self.send({"jsonrpc": "2.0", "method": "initialized", "params": {}})
        thread_start_params: dict = {"cwd": cwd}
        if model is not None:
            thread_start_params["model"] = model
        thread_rpc_id = self.call("thread/start", thread_start_params)

        thread_start_error: Optional[dict] = None
        deadline = time.monotonic() + _CODEX_APPSERVER_TIMEOUT
        while time.monotonic() < deadline and self.thread_id is None:
            if self.proc.poll() is not None:
                break
            frames, _, eof = self.read_frames(0.2)
            for msg in frames:
                if msg.get("id") != thread_rpc_id:
                    continue
                if "result" in msg:
                    result = msg["result"]
                    if not isinstance(result, dict):
                        self.log(f"unexpected thread/start result type: {type(result)}")
                        break
                    thread = result.get("thread") or result
                    if not isinstance(thread, dict):
                        self.log(f"unexpected thread object type: {type(thread)}")
                        break
                    self.thread_id = thread.get("id") or thread.get("sessionId")
                    if self.thread_id:
                        model_suffix = f" model={model!r}" if model is not None else ""
                        self.log(
                            f"minted thread_id={self.thread_id!r} "
                            f"rollout={thread.get('path', '?')!r}{model_suffix}"
                        )
                        _record_session(
                            self.log_dir, self.acquire_log, "codex",
                            self.thread_id, "minted", "certain",
                        )
                        break
                elif "error" in msg:
                    self.log(f"thread/start error: {msg['error']}")
                    thread_start_error = msg["error"]
                    break
            if thread_start_error is not None or eof:
                break

        if self.thread_id is None:
            if thread_start_error is not None:
                reason = f"thread/start error: {thread_start_error}".replace("\n", " ")
                if len(reason) > _THREAD_START_ERROR_REASON_MAX:
                    reason = reason[:_THREAD_START_ERROR_REASON_MAX - 3] + "..."
            else:
                self.log("thread/start failed or timed out")
                reason = "thread/start failed or timed out"
            _record_session(self.log_dir, self.acquire_log, "codex", None, "missing", "missing",
                            reason)
        return self.thread_id

    def start_turn(self, text: str, rpc_id: Optional[int] = None) -> int:
        return self.call("turn/start", {
            "threadId": self.thread_id,
            "input": [{"type": "text", "text": text}],
        }, rpc_id=rpc_id)

    def steer_turn(self, text: str, expected_turn_id: str, rpc_id: Optional[int] = None) -> int:
        return self.call("turn/steer", {
            "threadId": self.thread_id,
            "expectedTurnId": expected_turn_id,
            "input": [{"type": "text", "text": text}],
        }, rpc_id=rpc_id)

    def close(self) -> None:
        if self.proc is None:
            return
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        try:
            self.proc.terminate()
            self.proc.wait(timeout=5.0)
        except Exception:  # noqa: BLE001
            try:
                self.proc.kill()
            except Exception:  # noqa: BLE001
                pass
        # Drop the marker once reaped so it cannot name a recycled pid.
        try:
            (self.state_dir / "appserver_pid").unlink(missing_ok=True)
        except OSError:
            pass


def _publish_running(state_dir: Path, ready: "callable") -> None:
    """Publish status=running and signal readiness if it has not happened yet."""
    status_path = state_dir / "status"
    if not status_path.exists() or status_path.read_text().strip() == "starting":
        _write(status_path, "running\n")
        ready()


def _turn_delta_text(msg: dict) -> Optional[str]:
    """Return the agent text of an item/agentMessage/delta frame, if it is one."""
    if msg.get("method") != "item/agentMessage/delta":
        return None
    delta = (msg.get("params") or {}).get("delta")
    return delta if isinstance(delta, str) and delta else None


def _result_turn_id(msg: dict) -> Optional[str]:
    """Extract result.turn.id from an rpc response, or None if not a turn result."""
    result = msg.get("result")
    if not isinstance(result, dict):
        return None
    turn = result.get("turn")
    if not isinstance(turn, dict):
        return None
    return turn.get("id")


def _run_managed_oneshot_codex_appserver(
    state_dir: Path,
    log_dir: Path,
    prompt: Optional[str],
    prompt_file: Optional[str],
    cwd: str,
    harness_args: List[str],
    log_fd: int,
    ready: "callable",
    acquire_log: Path,
    model: Optional[str] = None,
) -> int:
    """One-shot codex run: mint a thread, run one turn, stream its text to the log.

    Agent text from item/agentMessage/delta is written to log_fd so tail, watch
    and the idle timeout observe incremental progress. session.json is written
    after thread/start, before the first model call.
    """
    server = _CodexAppServer(state_dir, log_dir, acquire_log, "codex app-server")
    if not server.start(harness_args, cwd, ready):
        return 1

    exit_code = 1
    try:
        thread_id = server.mint_thread(cwd, model=model)

        _write(state_dir / "status", "running\n")
        ready()
        # The prompt goes out via turn/start below; mark it now so _finalize does
        # not misclassify a codex failure as a launch failure.
        _write(state_dir / "prompt_submitted", "1\n")
        if thread_id is None:
            return 1

        turn_rpc_id = server.start_turn(_managed_prompt_text(prompt, prompt_file))

        turn_done = False
        deadline = time.monotonic() + 600  # hard cap for a one-shot turn
        while time.monotonic() < deadline and not turn_done:
            frames, saw_lines, eof = server.read_frames(0.2)
            for msg in frames:
                delta = _turn_delta_text(msg)
                if delta is not None:
                    _log_write(log_fd, delta.encode("utf-8", errors="replace"))
                elif msg.get("method") == "turn/completed":
                    turn_done = True
                    turn_status = ((msg.get("params") or {}).get("turn") or {}).get("status", "")
                    exit_code = 0 if turn_status == "completed" else 1
                    server.log(f"turn completed status={turn_status!r}")
                    _log_write(log_fd, b"\n")
                elif msg.get("id") == turn_rpc_id and "error" in msg:
                    server.log(f"turn/start error: {msg['error']}")
                    turn_done = True
                    exit_code = 1
            # Drain before checking process exit, otherwise frames still buffered
            # in the pipe after the app-server exits are discarded.
            if eof or (not saw_lines and server.proc.poll() is not None):
                break

        if not turn_done:
            server.log("turn did not complete (timeout or process died)")
            exit_code = 1

    except OSError as exc:
        server.log(f"I/O error: {exc}")
        _publish_running(state_dir, ready)
        exit_code = 1
    finally:
        server.close()

    return exit_code


def _split_steer_message(buf: bytes) -> tuple[Optional[bytes], bytes]:
    r"""Split off one steer message at the rightmost terminator in buf.

    Each ``cmd_steer`` write is one logical message ended by the run's submit
    sequence (\r\n, \n or \r). Splitting at the *last* terminator keeps a
    multi-line steer as a single message instead of one per embedded newline.
    Returns (message, remainder); message is None when no terminator is present.
    """
    last_crlf = buf.rfind(b"\r\n")
    last_lf = buf.rfind(b"\n")
    last_cr = buf.rfind(b"\r")
    if last_crlf >= 0 and last_crlf >= max(last_lf, last_cr):
        return buf[:last_crlf], buf[last_crlf + 2:]
    if last_lf >= 0 and last_lf >= last_cr:
        return buf[:last_lf], buf[last_lf + 1:]
    if last_cr >= 0:
        return buf[:last_cr], buf[last_cr + 1:]
    return None, buf


def _run_managed_interactive_codex_appserver(
    state_dir: Path,
    log_dir: Path,
    prompt: Optional[str],
    prompt_file: Optional[str],
    cwd: str,
    harness_args: List[str],
    log_fd: int,
    ready: "callable",
    acquire_log: Path,
    model: Optional[str] = None,
) -> int:
    """Interactive codex run: one long-lived app-server relaying steer input.

    After mint_thread and the initial turn/start, selects on the app-server's
    stdout and the run's FIFO. FIFO text becomes turn/steer while a turn is
    running and turn/start when the agent is idle; agent text deltas stream to
    log_fd. The keeper child holds the FIFO write end open so the read end never
    sees EOF between steers — the same mechanism _run_interactive uses, without
    a PTY.
    """
    server = _CodexAppServer(state_dir, log_dir, acquire_log, "codex app-server interactive")
    if not server.start(harness_args, cwd, ready):
        return 1

    exit_code = 1
    # None means the agent is idle between turns.
    active_turn_id: Optional[str] = None
    # Rpc ids of turn/steer calls awaiting a response, so errors can be matched.
    pending_steer_rpc_ids: set = set()

    keeper_pid = _fork_fifo_keeper(state_dir)
    fifo_fd = _open_fifo_reader(state_dir / "stdin")

    try:
        thread_id = server.mint_thread(cwd, model=model)
        if thread_id is None:
            _write(state_dir / "status", "running\n")
            ready()
            return 1

        turn_rpc_id = server.start_turn(_managed_prompt_text(prompt, prompt_file))

        _write(state_dir / "status", "running\n")
        ready()
        # The prompt went out via turn/start; mark it so _finalize does not
        # misclassify a failure as launch_failed.
        _write(state_dir / "prompt_submitted", "1\n")

        steer_buf = b""
        # Distinguishes a clean app-server exit from a mid-session transport failure.
        had_completed_turn = False
        session_deadline = time.monotonic() + 86400

        while time.monotonic() < session_deadline:
            if server.proc.poll() is not None:
                break

            try:
                readable, _, _ = select.select([server.out_fd, fifo_fd], [], [], 0.3)
            except (OSError, select.error) as exc:
                if isinstance(exc, OSError) and exc.errno == errno.EINTR:
                    continue
                break

            if server.out_fd in readable:
                frames, _, eof = server.read_frames(0)
                for msg in frames:
                    msg_id = msg.get("id")

                    delta = _turn_delta_text(msg)
                    if delta is not None:
                        _log_write(log_fd, delta.encode("utf-8", errors="replace"))

                    elif msg.get("method") == "turn/completed":
                        turn = (msg.get("params") or {}).get("turn") or {}
                        completed_id = turn.get("id")
                        # Clear on any completion, including when the server omits
                        # the turn id, so the next steer cannot carry a stale
                        # expectedTurnId.
                        if active_turn_id and completed_id in (active_turn_id, None):
                            active_turn_id = None
                        turn_status = turn.get("status", "")
                        server.log(f"turn completed id={completed_id!r} status={turn_status!r}")
                        if turn_status == "completed":
                            exit_code = 0
                            had_completed_turn = True
                        _log_write(log_fd, b"\n")

                    elif "error" in msg and msg_id is not None:
                        server.log(f"rpc error id={msg_id} error={msg['error']!r}")
                        if msg_id in pending_steer_rpc_ids:
                            # The steer text is already consumed from the FIFO and
                            # cannot be resent; clear the turn so the next one starts
                            # fresh and record the loss.
                            pending_steer_rpc_ids.discard(msg_id)
                            active_turn_id = None
                            server.log("steer rejected — turn already completed; steer was lost")

                    elif msg_id == turn_rpc_id and "result" in msg:
                        new_turn_id = _result_turn_id(msg)
                        if new_turn_id:
                            active_turn_id = new_turn_id
                            server.log(f"turn started id={active_turn_id!r}")
                        # Later responses are matched by their own rpc ids.
                        turn_rpc_id = -1

                    elif msg_id in pending_steer_rpc_ids and "result" in msg:
                        pending_steer_rpc_ids.discard(msg_id)
                        new_turn_id = _result_turn_id(msg)
                        if new_turn_id:
                            active_turn_id = new_turn_id
                            server.log(f"turn id updated to {active_turn_id!r}")
                # Stdout EOF means the app-server closed its write end; stop
                # selecting on it — continuing would busy-loop at 100% CPU.
                if eof:
                    break

            if fifo_fd in readable:
                try:
                    chunk = os.read(fifo_fd, 4096)
                except (BlockingIOError, OSError):
                    chunk = b""
                steer_buf += chunk
                while steer_buf:
                    msg_bytes, steer_buf = _split_steer_message(steer_buf)
                    if msg_bytes is None:
                        break
                    steer_text = msg_bytes.decode("utf-8", errors="replace").strip()
                    if not steer_text:
                        continue
                    steer_rpc_id = server.next_id()
                    if active_turn_id is not None:
                        pending_steer_rpc_ids.add(steer_rpc_id)
                        server.steer_turn(steer_text, active_turn_id, rpc_id=steer_rpc_id)
                        server.log(
                            f"turn/steer expectedTurnId={active_turn_id!r} "
                            f"text={steer_text[:80]!r}"
                        )
                    else:
                        turn_rpc_id = steer_rpc_id
                        server.start_turn(steer_text, rpc_id=steer_rpc_id)
                        server.log(f"turn/start (steer idle) text={steer_text[:80]!r}")

        # Any non-zero app-server exit is a transport failure even after a
        # completed turn: the session is gone and further steers cannot land.
        # Only a clean exit after a completed turn is a normal outcome.
        rc = server.proc.poll()
        if rc is not None and (rc != 0 or not had_completed_turn):
            exit_code = 1
        server.log(f"session loop ended exit_code={exit_code}")

    except OSError as exc:
        server.log(f"I/O error: {exc}")
        _publish_running(state_dir, ready)
        exit_code = 1
    finally:
        try:
            os.close(fifo_fd)
        except OSError:
            pass
        _reap_fifo_keeper(keeper_pid)
        server.close()

    return exit_code


# Valid values for the --permissions managed-mode flag.
_PERMISSIONS_BYPASS = "bypass"
_PERMISSIONS_PROMPT = "prompt"
_VALID_PERMISSIONS: frozenset[str] = frozenset({_PERMISSIONS_BYPASS, _PERMISSIONS_PROMPT})


def _build_managed_argv(
    harness: str,
    *,
    interactive: bool,
    prompt: Optional[str],
    prompt_file: Optional[str] = None,
    model: Optional[str],
    agent_mode: Optional[str],
    session_id: Optional[str],
    harness_args: List[str],
    opencode_port: Optional[int] = None,
    permissions: str = _PERMISSIONS_BYPASS,
    enable_planning: bool = False,
    enable_questions: bool = False,
) -> List[str]:
    """Build the argv for the given harness and mode. Returns [] for codex.

    An inline prompt becomes a positional argument only when prompt_file is
    unset: with a prompt_file, _run_oneshot opens it as the agent's stdin, so a
    positional would send the text twice. harness_args are appended last, after
    every agent-run-managed flag, so callers can override.

    Codex is not built here — both its modes drive `codex app-server` over
    JSON-RPC rather than exec'ing an argv.

    permissions controls whether unattended-operation flags are added:
    "bypass" (default) appends --permission-mode bypassPermissions for claude
    and --auto for interactive opencode; "prompt" omits them so the harness's
    own permission UI is used.
    """
    argv: List[str] = []

    if harness == "claude":
        argv.append("claude")
        if model:
            argv.extend(["--model", model])
        if session_id:
            argv.extend(["--session-id", session_id])
        # bypassPermissions makes unattended operation possible. Omitted when
        # --permissions prompt so the harness's own permission UI is used instead.
        if permissions == _PERMISSIONS_BYPASS:
            if not interactive:
                argv.extend(["--print", "--permission-mode", "bypassPermissions"])
                if prompt and not prompt_file:
                    argv.append(prompt)
            else:
                argv.extend(["--permission-mode", "bypassPermissions"])
        else:
            if not interactive:
                argv.append("--print")
                if prompt and not prompt_file:
                    argv.append(prompt)
        # --disallowedTools is variadic: claude consumes every bare (non-flag)
        # token that follows it as a tool name.  Two hazards must both be
        # avoided:
        #   1. The inline prompt positional must precede every deny flag, or
        #      claude would swallow it as a tool name.
        #   2. harness_args must NOT immediately follow a --disallowedTools
        #      block, or a bare caller token would be silently absorbed into
        #      the deny list.
        # Solution: place harness_args after the prompt but before the deny
        # block.  The deny flags that follow terminate the preceding variadic
        # with a flag-form token, so no bare harness_arg can be swallowed.
        # --disallowedTools is additive (not last-wins), so positioning it
        # last does not affect the user's ability to override other flags.
        argv.extend(harness_args)
        if not enable_planning:
            argv.extend(["--disallowedTools", "EnterPlanMode", "--disallowedTools", "ExitPlanMode"])
        if not enable_questions:
            argv.extend(["--disallowedTools", "AskUserQuestion"])

    elif harness == "opencode":
        argv.append("opencode")
        if model:
            argv.extend(["-m", model])
        if agent_mode:
            argv.extend(["--agent", agent_mode])
        if not interactive:
            argv.append("run")
            if session_id:
                argv.extend(["--session", session_id])
            if prompt and not prompt_file:
                argv.append(prompt)
        else:
            # Bare TUI attached to the pre-minted session. --port keeps the HTTP
            # API reachable. --auto approves permissions unattended; omitted when
            # --permissions prompt so the harness's own permission UI is used.
            # NEVER add --prompt here: --session silently swallows it, which is
            # how 24 runs were lost.
            if opencode_port is not None:
                argv.extend(["--port", str(opencode_port)])
            if session_id:
                argv.extend(["--session", session_id])
            if permissions == _PERMISSIONS_BYPASS:
                argv.append("--auto")
        argv.extend(harness_args)

    return argv


def _record_session(
    log_dir: Path,
    acquire_log: Path,
    harness: str,
    session_id: Optional[str],
    acquisition: str,
    confidence: str,
    reason: Optional[str] = None,
) -> None:
    """Write session.json recording how the session id was obtained.

    Never raises: acquisition failure must not affect the run (spec §6).
    acquisition is "pushed", "minted", "reported", or "missing"; confidence is
    "certain" or "missing". There is no heuristic tier — every path is
    structurally certain or genuinely missing.
    """
    data: dict = {
        "session_id": session_id,
        "harness": harness,
        "acquisition": acquisition,
        "confidence": confidence,
        "observed_at": _now_iso(),
    }
    if reason is not None:
        data["reason"] = reason
    _acquire_log_write(
        acquire_log,
        f"{harness} {acquisition}/{confidence} id={session_id!r}"
        + (f" reason={reason}" if reason else ""),
    )
    try:
        _write_session_json(log_dir, data)
    except OSError as exc:
        _acquire_log_write(acquire_log, f"could not write session.json: {exc}")


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
        # Raw-mode usage shown first so the existing contract is prominent;
        # managed-mode options follow in the argument group below.
        usage=(
            "agent-run [flags] NAME -- <cmd...>\n"
            "       agent-run [flags] NAME <cmd...>\n"
            "       agent-run --harness claude|opencode|codex [options] NAME\n"
            "       agent-run {status,watch,logs,transcript,tail,attach,steer,kill,list,reap,du,help} ..."
        ),
    )
    sub = p.add_subparsers(dest="sub")

    # Managed-mode options are parsed by _parse_launch_argv, not by this parser.
    # They are registered here with default=SUPPRESS so they appear in --help
    # without injecting phantom keys into the Namespace on subcommand dispatch.
    mg = p.add_argument_group(
        "managed mode (--harness claude|opencode|codex)",
        "agent-run builds the launch command itself; requires --prompt or --prompt-file",
    )
    # --cwd applies to raw and managed launches alike, so it is registered on the
    # top-level parser rather than in the managed-mode group.
    p.add_argument(
        "--cwd",
        metavar="DIR",
        default=argparse.SUPPRESS,
        help="run the launched command with DIR as its working directory; DIR may be "
        "relative and may start with '~', is resolved to an absolute path with symlinks "
        "collapsed, and is entered before any run state is published so <state_dir>/cwd "
        "records it; available in both managed and raw mode. A relative command path is "
        "resolved by the command itself and so is relative to DIR, but a relative "
        "-f/--prompt-file names a file the caller typed and stays relative to the "
        "invocation directory",
    )
    p.add_argument(
        "--worktree",
        metavar="DIR",
        default=argparse.SUPPRESS,
        help="create a linked git worktree at DIR and launch into it, setting --cwd to "
        "DIR; mutually exclusive with --cwd. Requires --worktree-base. DIR must not "
        "already exist unless --worktree-reuse is given",
    )
    p.add_argument(
        "--worktree-base",
        metavar="REF",
        default=argparse.SUPPRESS,
        help="base ref the new branch starts from; required with --worktree. Deriving "
        "this from the invocation checkout's HEAD is deliberately unsupported: an agent "
        "launched from a stale or mid-work checkout would otherwise silently branch off "
        "whatever happened to be checked out",
    )
    p.add_argument(
        "--worktree-branch",
        metavar="NAME",
        default=argparse.SUPPRESS,
        help="branch to create for --worktree (default: the run name)",
    )
    p.add_argument(
        "--worktree-repo",
        metavar="PATH",
        default=argparse.SUPPRESS,
        help="owning repository for --worktree (default: the repository containing the "
        "invocation directory)",
    )
    p.add_argument(
        "--worktree-reuse",
        action="store_true",
        default=argparse.SUPPRESS,
        help="attach to an existing worktree directory (of the same --worktree-repo) or "
        "an existing --worktree-branch instead of failing; without this flag, an "
        "existing DIR or branch is always a hard failure, never a silent attach. "
        "Validation runs once at preflight and is advisory: it does not prevent a "
        "concurrent external git switch/worktree remove against DIR before the agent "
        "starts. Use a dedicated (non-reuse) --worktree for isolation guarantees",
    )
    mg.add_argument(
        "--harness",
        metavar="claude|opencode|codex",
        default=argparse.SUPPRESS,
        help="select managed mode and harness; mutually exclusive with a trailing '-- <cmd>'",
    )
    mg.add_argument(
        "--prompt",
        metavar="TEXT",
        default=argparse.SUPPRESS,
        help="inline prompt text (mutually exclusive with --prompt-file)",
    )
    mg.add_argument(
        "-f",
        "--prompt-file",
        metavar="PATH",
        default=argparse.SUPPRESS,
        help="read the prompt from a file (mutually exclusive with --prompt); "
        "also available in raw mode",
    )
    mg.add_argument(
        "--permissions",
        metavar="bypass|prompt",
        default=argparse.SUPPRESS,
        help="permission mode: 'bypass' (default) appends --permission-mode bypassPermissions "
        "/ --auto to the harness command; 'prompt' omits those flags so the harness's own "
        "permission UI is used",
    )
    mg.add_argument(
        "--model",
        metavar="MODEL",
        default=argparse.SUPPRESS,
        help="model string forwarded verbatim to the harness",
    )
    mg.add_argument(
        "--agent-mode",
        metavar="NAME",
        default=argparse.SUPPRESS,
        help="harness agent/mode name (e.g. opencode --agent build); forwarded as "
        "--agent to the harness",
    )
    mg.add_argument(
        "--harness-arg",
        metavar="FLAG",
        default=argparse.SUPPRESS,
        help="pass FLAG verbatim to the harness command; repeatable. "
        "For claude and codex, values are appended after agent-run's own managed-mode args "
        "so they can override injected flags (codex uses last-occurrence-wins for -c). "
        "For opencode, policy is delivered via OPENCODE_CONFIG_CONTENT (env) and is not "
        "overridable this way; --harness-arg still passes flags to the opencode process "
        "but cannot override the three managed policy keys (question, plan_enter, plan_exit).",
    )
    mg.add_argument(
        "--enable-planning",
        action="store_true",
        default=argparse.SUPPRESS,
        help="allow planning in managed mode (not supported by codex app-server)",
    )
    mg.add_argument(
        "--enable-questions",
        action="store_true",
        default=argparse.SUPPRESS,
        help="allow interactive questions in managed mode",
    )

    sp_status = sub.add_parser("status", help="print one-line status")
    sp_status.add_argument("name")
    sp_status.set_defaults(func=cmd_status)

    sp_watch = sub.add_parser(
        "watch",
        help="print a stateless read-only JSON fact snapshot (log, git, "
        "signals) for a poller to apply its own policy to",
    )
    sp_watch.add_argument("name")
    sp_watch.add_argument(
        "--json", action="store_true", help="emit the full JSON fact object"
    )
    sp_watch.add_argument(
        "--repo",
        default=None,
        help="repo path for git facts; defaults to the run's recorded launch "
        "cwd, else no git facts",
    )
    sp_watch.set_defaults(func=cmd_watch)

    sp_logs = sub.add_parser(
        "logs",
        help="print last N lines of the log (default --tail 50)",
        allow_abbrev=False,
    )
    sp_logs.add_argument("name")
    logs_slice = sp_logs.add_mutually_exclusive_group()
    logs_slice.add_argument(
        "--tail",
        type=_positive_int,
        default=None,
        metavar="N",
        help="print the last N lines (default when neither flag is given: 50)",
    )
    logs_slice.add_argument(
        "--head",
        type=_positive_int,
        default=None,
        metavar="N",
        help="print the first N lines; reads forward until N newlines found or EOF",
    )
    logs_mode = sp_logs.add_mutually_exclusive_group()
    logs_mode.add_argument(
        "--plain",
        action="store_true",
        help="ANSI-stripped output, for grepping/piping/agent consumption",
    )
    logs_mode.add_argument(
        "--clean",
        action="store_true",
        help="render the log through pyte and print the transcript; slow on a large log",
    )
    sp_logs.set_defaults(func=cmd_logs)

    sp_transcript = sub.add_parser(
        "transcript",
        help="print the harness's own conversation record (needs --harness at launch)",
        allow_abbrev=False,
    )
    sp_transcript.add_argument("name")
    transcript_slice = sp_transcript.add_mutually_exclusive_group()
    transcript_slice.add_argument(
        "--tail",
        type=_positive_int,
        default=None,
        metavar="N",
        help="print the last N entries (default when neither flag is given: 50)",
    )
    transcript_slice.add_argument(
        "--head",
        type=_positive_int,
        default=None,
        metavar="N",
        help="print the first N entries",
    )
    sp_transcript.add_argument(
        "--json",
        action="store_true",
        help="emit one JSON object per entry instead of plain text",
    )
    sp_transcript.set_defaults(func=cmd_transcript)

    sp_tail = sub.add_parser("tail", help="follow log in real time (tail -f)")
    sp_tail.add_argument("name")
    sp_tail.set_defaults(func=cmd_tail)

    sp_attach = sub.add_parser(
        "attach",
        help="attach interactively (live keyboard + resize; Ctrl-C detaches, exit 0)",
        description=(
            "Attach a real terminal to an interactive run: keystrokes are "
            "forwarded to the agent's stdin and terminal resizes are relayed "
            "to its PTY. Ctrl-C detaches without stopping the agent and exits "
            "0. Several clients may attach at once; output is mirrored to all "
            "of them, but only one should type at a time -- concurrent "
            "keyboard input from multiple clients is not reliable: each "
            "write is atomic up to PIPE_BUF, but a longer burst is split "
            "across writes that another client can interleave between, so "
            "an escape sequence can still tear. The PTY size is last-writer-"
            "wins: each client sends its size on connect and whenever that "
            "size changes, so the most recent resize governs for everyone. "
            "Attaching from inside another "
            "attach session shares one terminal between two raw-mode owners: "
            "the inner session restores the outer session's raw mode on exit, "
            "so exit them in reverse order or the terminal is left raw."
        ),
    )
    sp_attach.add_argument("name")
    sp_attach.set_defaults(func=cmd_attach)

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
    sp_kill.add_argument(
        "--force",
        action="store_true",
        help="kill a run whose state dir records no process identity, using its "
        "recorded process group; cannot verify PID reuse",
    )
    sp_kill.set_defaults(func=cmd_kill)

    sp_list = sub.add_parser(
        "list",
        help="list runs (defaults to non-terminal only; AGENT_RUN_LIST_DEFAULT=all overrides; see --all/--status)",
    )
    group_list = sp_list.add_mutually_exclusive_group()
    group_list.add_argument(
        "--all",
        action="store_true",
        default=False,
        help="show every recognized run, including conclusively terminal ones "
        "(done/failed/died/killed) that are hidden by default",
    )
    group_list.add_argument(
        "--status",
        default=None,
        metavar="STATUS[,STATUS...]",
        help="show only runs whose effective status is in this comma-separated "
        "list (e.g. --status running,stalled); overrides the default filter",
    )
    # Deliberately not in group_list: orthogonal to --all/--status, both of
    # which govern only the state-backed "Live runs"/"Unrecognized" sections.
    # --all alone must still hide preserved logs.
    sp_list.add_argument(
        "--include-logs",
        action="store_true",
        default=False,
        help="also show preserved-log-only runs (state dir gone); hidden by "
        "default, or set AGENT_RUN_LIST_INCLUDE_LOGS=1",
    )
    sp_list.set_defaults(func=cmd_list)

    sp_reap = sub.add_parser(
        "reap",
        help="reconcile stale status, idle-kill lingering runs, "
        "garbage-collect old terminal-state run dirs, collect preserved logs "
        "(--include-logs), terminate orphan processes (--orphan-processes), "
        "and remove linked worktrees (--include-worktrees); --all enables all "
        "three opt-in passes without overriding any safety refusal",
    )
    sp_reap.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="report actions without mutating any state",
    )
    sp_reap.add_argument(
        "--all",
        action="store_true",
        default=False,
        help="enable every opt-in pass: --include-logs, --orphan-processes "
        "and --include-worktrees.  Each pass keeps its own age threshold "
        "(--min-age-hours, --log-min-age-hours, --orphan-min-age-hours, "
        "--worktree-min-age-hours) unchanged.  DOES NOT imply --force-dirty "
        "or --force-unknown: those override safety refusals (unpushed or "
        "uncommitted work, unclassifiable status) rather than enable a pass, "
        "and must always be requested explicitly.  Combining --all with an "
        "individual pass flag is redundant, not an error",
    )
    sp_reap.add_argument(
        "--idle-hours",
        type=_positive_finite_float,
        default=None,
        metavar="N",
        help="override idle threshold (hours, float, must be finite and > 0); "
        "default from AGENT_RUN_IDLE_KILL_HOURS or 24h",
    )
    sp_reap.add_argument(
        "--min-age-hours",
        type=_positive_finite_float,
        default=None,
        metavar="N",
        help="override the terminal-state GC age threshold (hours, float, "
        "must be finite and > 0); a done/failed/died/killed run must be at "
        "least this old before its state dir and scratch dir are removed; "
        "default from AGENT_RUN_MIN_AGE_HOURS (preferred), "
        "AGENT_RUN_REAP_MIN_AGE_HOURS (compatible alias), or 168h (7 days)",
    )
    sp_reap.add_argument(
        "--force-unknown",
        action="store_true",
        default=False,
        help="also collect old unrecognized/legacy/corrupt statuses; still refuses "
        "a live or unverifiable recorded runner.  Off by default and never "
        "implied by --all or any other flag",
    )
    sp_reap.add_argument(
        "--name",
        default=None,
        metavar="NAME",
        help="reap only this specific run",
    )
    sp_reap.add_argument(
        "--include-logs",
        action="store_true",
        default=False,
        help="also garbage-collect preserved log directories (state dir "
        "already gone) once older than --log-min-age-hours; without this "
        "flag (or --all) preserved logs are never touched by reap",
    )
    sp_reap.add_argument(
        "--log-min-age-hours",
        type=_positive_finite_float,
        default=None,
        metavar="N",
        help="override the preserved-log GC age threshold used by "
        "--include-logs (hours, float, must be finite and > 0); independent "
        "of --min-age-hours, since a preserved log is the artifact an "
        "operator wanted to keep, not disposable state bookkeeping; default "
        "from AGENT_RUN_LOG_MIN_AGE_HOURS, or PRUNE_AFTER_DAYS*24 "
        "(21 days), matching the existing whole-log-dir prune",
    )
    sp_reap.add_argument(
        "--orphan-processes",
        action="store_true",
        default=False,
        help="terminate live agent-run runner processes that have no state "
        "directory — i.e. runs that exist as live processes but whose "
        "ephemeral state record (in $AGENT_RUN_STATE_DIR) is gone.  This "
        "kills processes agent-run has no state record for, selected by "
        "argv parsing; it is opt-in for that reason.  Identity captured at "
        "discovery is re-verified immediately before every signal; any "
        "ambiguity aborts the candidate instead of sending a signal.  Only "
        "processes older than --orphan-min-age-hours are eligible.  Also "
        "enabled by --all.",
    )
    sp_reap.add_argument(
        "--orphan-min-age-hours",
        type=_positive_finite_float,
        default=None,
        metavar="N",
        help="minimum process age for --orphan-processes candidates (hours, "
        "float, must be finite and > 0); independent of --min-age-hours and "
        "--log-min-age-hours; default from AGENT_RUN_ORPHAN_MIN_AGE_HOURS "
        "or 24h.  Parsed and validated even when --orphan-processes is absent.",
    )
    sp_reap.add_argument(
        "--include-worktrees",
        action="store_true",
        default=False,
        help="also remove the launch directory of a terminal run when that "
        "directory is a LINKED GIT WORKTREE, once older than "
        "--worktree-min-age-hours. A non-git directory is NEVER removed: a "
        "run's cwd is frequently a real project checkout or $HOME, so main "
        "worktrees, bare repos, non-git directories, and anything git cannot "
        "classify are all refused, as is any path with a symlinked "
        "component. Removal goes through 'git worktree remove' against the "
        "owning repository, so the parent's .git/worktrees/<name> admin entry "
        "is unregistered too. A worktree with modified/deleted tracked files, "
        "ordinary untracked files, ignored files or directory content, or "
        "commits not reachable from any local remote-tracking ref is refused. "
        "NOTE: stale tracking refs (remote branch deleted but not pruned locally) "
        "are not detected; run 'git fetch --prune' beforehand if current remote "
        "state matters. --force-dirty overrides only the "
        "tracked, untracked, and ignored content checks; unpushed commits and "
        "every structural, liveness, scan, identity, nested-worktree, and git "
        "failure remain refusals. A worktree still used by a live run is always "
        "refused, whatever its age. A worktree shared by several terminal runs "
        "is removed once. Off by "
        "default; also enabled by --all",
    )
    sp_reap.add_argument(
        "--worktree-min-age-hours",
        type=_positive_finite_float,
        default=None,
        metavar="N",
        help="override the linked-worktree GC age threshold used by "
        "--include-worktrees (hours, float, must be finite and > 0), measured "
        "from the younger of the youngest terminal run and the newest recursive "
        "filesystem mtime under the worktree. Directory/file mtimes are only a "
        "conservative activity signal and can be reset externally; independent of "
        "--min-age-hours, --log-min-age-hours and --orphan-min-age-hours, "
        "since it governs destruction of a working tree rather than of "
        "bookkeeping; default from AGENT_RUN_WORKTREE_MIN_AGE_HOURS or 168h "
        "(7 days).  Parsed and validated even when --include-worktrees is absent",
    )
    sp_reap.add_argument(
        "--force-dirty",
        action="store_true",
        default=False,
        help="with --include-worktrees, override only refusals for modified or "
        "deleted tracked files, ordinary untracked files, and ignored files or "
        "directory content. It never overrides unpushed commits, classification, "
        "live use, nested worktrees, symlink/path identity, state-scan failures, "
        "or git failures. This destroys local content irreversibly; off by "
        "default and never implied by --all or any other flag",
    )
    sp_reap.add_argument(
        "--max-seconds",
        type=_positive_finite_float,
        default=None,
        metavar="N",
        dest="max_seconds",
        help="soft candidate-admission budget for reap (seconds, float, must be "
        "finite and > 0); an in-progress scan, lock wait, or kill may overrun it; "
        "when exceeded, remaining candidates in any pass "
        "are deferred and a count is printed; exit 0 so the next timer tick resumes. "
        "Default from AGENT_RUN_REAP_MAX_SECONDS, or 600 s (10 min, comfortably "
        "under the 30-minute systemd timer period).",
    )
    sp_reap.set_defaults(func=cmd_reap)

    sp_du = sub.add_parser(
        "du",
        help="disk usage per effective status (or per run with --by-run), "
        "including preserved logs and linked-worktree launch dirs; "
        "strictly read-only",
        description="Disk usage per effective status (or per run with "
        "--by-run), including preserved logs and each run's recorded launch "
        "cwd when that directory is a linked git worktree (WORKTREE column, "
        "worktree_bytes under --json). Linked worktrees are usually far "
        "larger than STATE_ROOT and LOG_ROOT combined, so walking them "
        "dominates this command's runtime. Main worktrees, bare repos, "
        "non-git directories, and any directory git cannot classify "
        "contribute 0. A worktree shared by several runs is deduplicated by "
        "realpath and charged once, to the first run sharing it (the others "
        "show 0), so TOTAL counts every byte exactly once. Strictly "
        "read-only: worktree detection runs git rev-parse only, never gc or "
        "worktree prune.",
    )
    sp_du.add_argument(
        "--by-run",
        action="store_true",
        default=False,
        help="one row per run instead of the per-status rollup",
    )
    sp_du.add_argument(
        "--top",
        type=_positive_int,
        default=None,
        metavar="N",
        help="show only the N largest rows; TOTAL still covers every run",
    )
    sp_du.add_argument(
        "--bytes",
        action="store_true",
        default=False,
        help="print exact integer byte counts instead of human-readable sizes",
    )
    sp_du.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="emit a machine-readable object instead of a table; always uses "
        "exact integers, so --bytes is rejected alongside it",
    )
    sp_du.set_defaults(func=cmd_du)

    sp_hook = sub.add_parser(
        "hook",
        help="receive a harness hook notification and record it in hooks.jsonl (always exits 0)",
    )
    sp_hook.add_argument(
        "event",
        help="hook event name (e.g. stop, turn-complete, session-idle); normalised to lowercase",
    )
    sp_hook.add_argument(
        "extra",
        nargs="*",
        help="optional extra positional: if the first looks like JSON, treated as the "
        "harness payload (codex notify shape passes JSON as argv[1])",
    )
    sp_hook.add_argument(
        "--name",
        default=None,
        metavar="RUN",
        help="resolve against this run name; overrides AGENT_RUN_NAME and ancestry walk",
    )
    sp_hook.add_argument(
        "--json",
        dest="json_payload",
        default=None,
        metavar="JSON",
        help="harness payload as a JSON string (opencode plugin path); "
        "takes priority over positional argv and stdin",
    )
    sp_hook.set_defaults(func=cmd_hook)

    sp_help = sub.add_parser("help", help="show this help")
    sp_help.set_defaults(func=lambda _a: (p.print_help() or 0))

    return p


class _LaunchArgvError(ValueError):
    """Raised by _parse_launch_argv for any argv-level parse failure.

    Carries the exact message string main() should pass to sys.exit, preserving
    byte-identical error text without coupling the pure parser to sys.exit.
    """


class _LaunchArgv(NamedTuple):
    """Parsed result of agent-run's top-level argv — flags, name, and command.

    ``subcommand_tokens`` is non-None when the first non-flag token is a known
    subcommand and no launch flags were supplied; main() delegates to argparse
    in that case and ignores all other fields.

    Managed-mode fields (harness is not None):
      harness          — "claude" | "opencode" | "codex"
      prompt           — inline prompt string (mutually exclusive with prompt_file)
      model            — model string forwarded to the harness
      agent_mode       — harness agent/mode name (opencode --agent)
      harness_args     — extra raw args forwarded verbatim after harness's own args
      enable_planning  — allow the harness planning capability where supported
      enable_questions — allow the harness interactive-question capability
    """
    interactive: bool
    prompt_file: Optional[str]
    submit_mode: Optional[str]
    idle_timeout: Optional[float]
    name: str
    command: List[str]
    subcommand_tokens: Optional[List[str]]
    # Working directory for the launched command; None means inherit.
    cwd: Optional[str] = None
    # --worktree* fields; all None/False when --worktree is absent. Mutual
    # exclusion with cwd and the --worktree-base requirement are enforced by
    # _create_launch_worktree (at cmd_launch time, exit 2), not here: this
    # parser is pure and does no filesystem/git access.
    worktree: Optional[str] = None
    worktree_base: Optional[str] = None
    worktree_branch: Optional[str] = None
    worktree_repo: Optional[str] = None
    worktree_reuse: bool = False
    # Managed-mode fields; all None/empty for raw runs.
    harness: Optional[str] = None
    prompt: Optional[str] = None
    model: Optional[str] = None
    agent_mode: Optional[str] = None
    harness_args: Tuple[str, ...] = ()
    # "bypass" (default) appends --permission-mode bypassPermissions / --auto.
    # "prompt" omits those flags so the harness's own permission UI is used.
    permissions: str = _PERMISSIONS_BYPASS
    enable_planning: bool = False
    enable_questions: bool = False


_KNOWN_SUBCOMMANDS: frozenset[str] = frozenset({
    "status", "watch", "logs", "transcript", "tail", "attach", "steer", "kill",
    "list", "reap", "du", "hook", "help",
})

# Managed-mode flags that take a value. Each accepts "--flag value" and
# "--flag=value"; --harness-arg accumulates, the rest keep the last value.
_MANAGED_VALUE_FLAGS: Tuple[str, ...] = (
    "--harness", "--prompt", "--model", "--agent-mode", "--harness-arg",
    "--permissions",
)


def _parse_launch_argv(raw: Sequence[str]) -> _LaunchArgv:
    """Parse agent-run's own flags, the run name, and the launch command.

    Pure: no sys.exit, no printing, no env reads, no filesystem access, no
    globals.  All parse failures raise _LaunchArgvError with the verbatim
    message main() should pass to sys.exit.

    Handles all current flag forms in any order before the name:
      -i / --interactive
      -f X / --prompt-file X / --prompt-file=X
      --submit-mode=cr|crlf
      --idle-timeout N / --idle-timeout=N
      the managed-mode flags in _MANAGED_VALUE_FLAGS
      --cwd <dir>  (working directory for the launched command)
      --worktree DIR --worktree-base REF [--worktree-branch NAME]
        [--worktree-repo PATH] [--worktree-reuse]
        (create or attach to a linked worktree; --worktree-base required;
        mutually exclusive with --cwd; see _create_launch_worktree)

    Preserves the -- separator semantics: name must precede --, everything
    after -- is taken verbatim.  Without --, a leading-dash token immediately
    after the name is rejected.

    When the first non-flag token is a known subcommand and no launch flags
    were set, returns with subcommand_tokens set to the remaining argv so
    main() can delegate to argparse; all other fields hold zero values in that
    case.

    Managed mode (--harness) and raw mode (trailing command) are mutually
    exclusive: supplying both raises _LaunchArgvError.
    """
    tokens = list(raw)
    interactive = False
    prompt_file: Optional[str] = None
    submit_mode: Optional[str] = None
    idle_timeout: Optional[float] = None
    harness: Optional[str] = None
    prompt: Optional[str] = None
    model: Optional[str] = None
    agent_mode: Optional[str] = None
    harness_args: List[str] = []
    permissions: str = _PERMISSIONS_BYPASS
    cwd: Optional[str] = None
    worktree: Optional[str] = None
    worktree_base: Optional[str] = None
    worktree_branch: Optional[str] = None
    worktree_repo: Optional[str] = None
    worktree_reuse = False
    enable_planning = False
    enable_questions = False

    # Consume flags in any order before the name.
    while tokens:
        if tokens[0] in ("-i", "--interactive"):
            interactive = True
            tokens = tokens[1:]
            continue
        if tokens[0] in ("-f", "--prompt-file"):
            if len(tokens) < 2:
                raise _LaunchArgvError("agent-run: -f/--prompt-file requires a path")
            prompt_file = tokens[1]
            tokens = tokens[2:]
            continue
        if tokens[0].startswith("--prompt-file="):
            prompt_file = tokens[0].split("=", 1)[1]
            tokens = tokens[1:]
            continue
        if tokens[0].startswith("--submit-mode="):
            submit_mode = tokens[0].split("=", 1)[1]
            if submit_mode not in {SUBMIT_MODE_CR, SUBMIT_MODE_CRLF}:
                raise _LaunchArgvError("agent-run: --submit-mode must be cr or crlf")
            tokens = tokens[1:]
            continue
        if tokens[0] == "--idle-timeout":
            if len(tokens) < 2:
                raise _LaunchArgvError("agent-run: --idle-timeout requires a value in seconds")
            try:
                idle_timeout = _parse_idle_timeout_flag(tokens[1])
            except argparse.ArgumentTypeError as exc:
                raise _LaunchArgvError(f"agent-run: --idle-timeout {exc}") from exc
            tokens = tokens[2:]
            continue
        if tokens[0].startswith("--idle-timeout="):
            try:
                idle_timeout = _parse_idle_timeout_flag(tokens[0].split("=", 1)[1])
            except argparse.ArgumentTypeError as exc:
                raise _LaunchArgvError(f"agent-run: --idle-timeout {exc}") from exc
            tokens = tokens[1:]
            continue
        if tokens[0] == "--enable-planning":
            enable_planning = True
            tokens = tokens[1:]
            continue
        if tokens[0] == "--enable-questions":
            enable_questions = True
            tokens = tokens[1:]
            continue
        # Managed-mode flags (see _MANAGED_VALUE_FLAGS).
        managed_flag = None
        for name_ in _MANAGED_VALUE_FLAGS:
            if tokens[0] == name_:
                if len(tokens) < 2:
                    raise _LaunchArgvError(f"agent-run: {name_} requires a value")
                managed_flag, value, tokens = name_, tokens[1], tokens[2:]
                break
            if tokens[0].startswith(name_ + "="):
                managed_flag, value, tokens = name_, tokens[0].split("=", 1)[1], tokens[1:]
                break
        if managed_flag is not None:
            if managed_flag == "--harness":
                if value not in MANAGED_HARNESSES:
                    raise _LaunchArgvError(
                        f"agent-run: --harness {value!r} is not valid; "
                        f"choose one of: {', '.join(sorted(MANAGED_HARNESSES))}"
                    )
                harness = value
            elif managed_flag == "--prompt":
                prompt = value
            elif managed_flag == "--model":
                model = value
            elif managed_flag == "--agent-mode":
                agent_mode = value
            elif managed_flag == "--harness-arg":
                harness_args = harness_args + [value]
            elif managed_flag == "--permissions":
                if value not in _VALID_PERMISSIONS:
                    raise _LaunchArgvError(
                        f"agent-run: --permissions {value!r} is not valid; "
                        f"choose one of: {', '.join(sorted(_VALID_PERMISSIONS))}"
                    )
                permissions = value
            continue
        # --cwd applies to both managed and raw launches; the value is validated
        # and entered by cmd_launch, not here (this parser touches no filesystem).
        # Both spellings share one emptiness check so neither form can accept "".
        if tokens[0] == "--cwd" or tokens[0].startswith("--cwd="):
            if tokens[0] == "--cwd":
                cwd, tokens = (tokens[1] if len(tokens) > 1 else ""), tokens[2:]
            else:
                cwd, tokens = tokens[0].split("=", 1)[1], tokens[1:]
            if not cwd:
                raise _LaunchArgvError("agent-run: --cwd requires a directory")
            continue
        # --worktree* flags are collected here but validated (mutual exclusion
        # with --cwd, --worktree-base required, filesystem/git checks) by
        # _create_launch_worktree at cmd_launch time — this parser does no
        # filesystem or git access.
        if tokens[0] == "--worktree-reuse":
            worktree_reuse = True
            tokens = tokens[1:]
            continue
        worktree_flag = None
        for name_, dest in (
            ("--worktree", "worktree"),
            ("--worktree-base", "worktree_base"),
            ("--worktree-branch", "worktree_branch"),
            ("--worktree-repo", "worktree_repo"),
        ):
            if tokens[0] == name_:
                if len(tokens) < 2:
                    raise _LaunchArgvError(f"agent-run: {name_} requires a value")
                if tokens[1].startswith("--"):
                    raise _LaunchArgvError(
                        f"agent-run: {name_} requires a value, got the flag {tokens[1]!r}"
                    )
                worktree_flag, value, tokens = dest, tokens[1], tokens[2:]
                break
            if tokens[0].startswith(name_ + "="):
                worktree_flag, value, tokens = dest, tokens[0].split("=", 1)[1], tokens[1:]
                break
        if worktree_flag is not None:
            if not value:
                raise _LaunchArgvError(f"agent-run: --{worktree_flag.replace('_', '-')} requires a value")
            if worktree_flag == "worktree":
                worktree = value
            elif worktree_flag == "worktree_base":
                worktree_base = value
            elif worktree_flag == "worktree_branch":
                worktree_branch = value
            elif worktree_flag == "worktree_repo":
                worktree_repo = value
            continue
        break

    # After the flag loop, reject managed-only flags on raw launches.
    if harness is None and (
        prompt is not None or model is not None or agent_mode is not None
        or bool(harness_args) or permissions != _PERMISSIONS_BYPASS
        or enable_planning or enable_questions
    ):
        raise _LaunchArgvError(
            "agent-run: --prompt/--model/--agent-mode/--harness-arg/--permissions/"
            "--enable-planning/--enable-questions require --harness <claude|opencode|codex>"
        )

    # A bare "--" before any name has no run to attach the command to.
    if tokens and tokens[0] == "--":
        raise _LaunchArgvError(
            "agent-run: the run name must appear before '--'; shape is "
            "'agent-run [flags] NAME -- <command> [args...]'"
        )

    # No launch flags set and first token is a known subcommand: delegate to
    # argparse.  A run may not be named after a subcommand, so a "--" after a
    # subcommand name is still part of that subcommand's own argv.
    any_launch_flag = (
        interactive or prompt_file or submit_mode is not None
        or idle_timeout is not None or harness is not None or prompt is not None
        or model is not None or agent_mode is not None or cwd is not None
        or worktree is not None or worktree_base is not None
        or worktree_branch is not None or worktree_repo is not None or worktree_reuse
        or bool(harness_args) or permissions != _PERMISSIONS_BYPASS
        or enable_planning or enable_questions
    )
    if tokens and tokens[0] in _KNOWN_SUBCOMMANDS and not any_launch_flag:
        return _LaunchArgv(
            interactive=False, prompt_file=None,
            submit_mode=None, idle_timeout=None, name="", command=[],
            subcommand_tokens=tokens,
        )

    if len(tokens) < 1 or (len(tokens) < 2 and harness is None):
        # Signal main() to print help; name/command are meaningless here.
        return _LaunchArgv(
            interactive=interactive, prompt_file=prompt_file,
            submit_mode=submit_mode,
            idle_timeout=idle_timeout, name="", command=[],
            subcommand_tokens=None, cwd=cwd,
            worktree=worktree, worktree_base=worktree_base,
            worktree_branch=worktree_branch, worktree_repo=worktree_repo,
            worktree_reuse=worktree_reuse,
            harness=harness, prompt=prompt, model=model,
            agent_mode=agent_mode, harness_args=harness_args,
        )

    # Managed mode: the name is the only remaining token — agent-run builds the
    # command itself, so there is no trailing command to parse.
    if harness is not None:
        name = tokens[0] if tokens else ""
        rest = tokens[1:]
        if not name or "/" in name or name.startswith("-"):
            raise _LaunchArgvError(f"agent-run: invalid name '{name}'")
        if rest and rest[0] == "--":
            raise _LaunchArgvError(
                "agent-run: --harness and a trailing '-- <command>' are mutually exclusive; "
                "in managed mode agent-run builds the command itself"
            )
        if rest:
            raise _LaunchArgvError(
                f"agent-run: unexpected tokens after run name in managed mode: "
                f"{rest!r}; use --harness-arg to pass extra flags to the harness"
            )
        if prompt and prompt_file:
            raise _LaunchArgvError("agent-run: --prompt and --prompt-file are mutually exclusive")
        if not prompt and not prompt_file:
            raise _LaunchArgvError(
                "agent-run: managed mode requires exactly one of --prompt <text> or --prompt-file <path>"
            )
        # Everything below runs before any run state exists, so a rejected launch
        # cannot strand a phantom run at status=starting.
        for ha in harness_args:
            flag = ha.split("=", 1)[0]
            if flag in {"--session", "-s", "--session-id", "--port", "--prompt"}:
                raise _LaunchArgvError(
                    f"agent-run: --harness-arg {flag!r} is managed internally; "
                    f"use the corresponding agent-run flag instead"
                )
        # Caller args are appended after the managed --permission-mode
        # bypassPermissions, so a later --permission-mode plan wins and starts
        # claude in plan mode without ever hitting the EnterPlanMode/ExitPlanMode
        # denies. Both spellings are rejected.
        if harness == "claude" and not enable_planning:
            for i, token in enumerate(harness_args):
                if token == "--permission-mode=plan":
                    raise _LaunchArgvError(
                        "agent-run: --harness-arg --permission-mode=plan selects Claude plan mode "
                        "directly, bypassing the planning deny policy; use --enable-planning to "
                        "relax the policy instead"
                    )
                if token == "--permission-mode" and harness_args[i + 1:i + 2] == ["plan"]:
                    raise _LaunchArgvError(
                        "agent-run: --harness-arg --permission-mode --harness-arg plan selects "
                        "Claude plan mode directly, bypassing the planning deny policy; use "
                        "--enable-planning to relax the policy instead"
                    )
        if enable_planning and harness == "codex":
            raise _LaunchArgvError(
                "agent-run: --enable-planning is unsupported for --harness codex; "
                "the codex app-server path does not expose plan mode"
            )
        return _LaunchArgv(
            interactive=interactive,
            prompt_file=prompt_file,
            submit_mode=submit_mode,
            idle_timeout=idle_timeout,
            name=name,
            command=[],
            subcommand_tokens=None,
            cwd=cwd,
            worktree=worktree,
            worktree_base=worktree_base,
            worktree_branch=worktree_branch,
            worktree_repo=worktree_repo,
            worktree_reuse=worktree_reuse,
            harness=harness,
            prompt=prompt,
            model=model,
            agent_mode=agent_mode,
            harness_args=tuple(harness_args),
            permissions=permissions,
            enable_planning=enable_planning,
            enable_questions=enable_questions,
        )

    # Raw mode.
    name, *rest = tokens
    if not name or "/" in name or name.startswith("-"):
        raise _LaunchArgvError(f"agent-run: invalid name '{name}'")

    if rest and rest[0] == "--":
        # Everything after "--" is the launch command verbatim — leading-dash
        # tokens and further "--" tokens included; no subcommand dispatch.
        command = rest[1:]
        if not command:
            raise _LaunchArgvError(
                "agent-run: empty command after '--'; provide a command to "
                "launch: 'agent-run [flags] NAME -- <command> [args...]'"
            )
    else:
        command = rest
        if command and command[0].startswith("-"):
            # A flag-like token after the name without "--" would silently
            # become argv[0] and exec would fail with an opaque ENOENT.
            raise _LaunchArgvError(
                f"agent-run: {command[0]!r} looks like an agent-run flag, not "
                "part of the launch command; agent-run flags (-i, -f, "
                "--submit-mode, --idle-timeout) must precede the run name, or "
                "separate them from the launch command with '--': "
                "'agent-run [flags] NAME -- <command> [args...]'"
            )

    return _LaunchArgv(
        interactive=interactive,
        prompt_file=prompt_file,
        submit_mode=submit_mode,
        idle_timeout=idle_timeout,
        name=name,
        command=command,
        subcommand_tokens=None,
        cwd=cwd,
        worktree=worktree,
        worktree_base=worktree_base,
        worktree_branch=worktree_branch,
        worktree_repo=worktree_repo,
        worktree_reuse=worktree_reuse,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)

    # No args -> list runs.
    if not raw:
        return cmd_list(argparse.Namespace())

    # Top-level help.
    if raw[0] in ("-h", "--help"):
        _build_parser().print_help()
        return 0

    try:
        parsed = _parse_launch_argv(raw)
    except _LaunchArgvError as exc:
        sys.exit(str(exc))

    # Subcommand dispatch: argparse handles help for each subcommand.
    if parsed.subcommand_tokens is not None:
        parser = _build_parser()
        try:
            args = parser.parse_args(parsed.subcommand_tokens)
        except SystemExit as exc:
            if parsed.subcommand_tokens[0] == "watch" and exc.code == 2:
                raise SystemExit(1) from exc
            if parsed.subcommand_tokens[0] == "hook":
                # argparse exits 2 on bad argv, but a misconfigured hook must
                # still not look like a blocking decision to the harness.
                return 0
            raise
        return int(args.func(args) or 0)

    if not parsed.name:
        _build_parser().print_help()
        return 2

    # Managed-mode fields are always present and are None/empty for raw runs;
    # cmd_launch keys off `harness` alone.
    ns = argparse.Namespace(
        name=parsed.name,
        command=parsed.command,
        interactive=parsed.interactive,
        prompt_file=parsed.prompt_file,
        submit_mode=parsed.submit_mode,
        idle_timeout=parsed.idle_timeout if parsed.idle_timeout is not None else _idle_timeout_env_seconds(),
        harness=parsed.harness,
        prompt=parsed.prompt,
        model=parsed.model,
        agent_mode=parsed.agent_mode,
        harness_args=list(parsed.harness_args),
        permissions=parsed.permissions,
        cwd=parsed.cwd,
        worktree=parsed.worktree,
        worktree_base=parsed.worktree_base,
        worktree_branch=parsed.worktree_branch,
        worktree_repo=parsed.worktree_repo,
        worktree_reuse=parsed.worktree_reuse,
        enable_planning=parsed.enable_planning,
        enable_questions=parsed.enable_questions,
    )
    return cmd_launch(ns)


if __name__ == "__main__":
    sys.exit(main())
