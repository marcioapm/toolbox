# `agent-run attach` — interactive attach to a running agent

Status: approved
Date: 2026-08-06

## Problem

`agent-run -i` launches an agent in a detached PTY, and today the only ways
to interact with it are:

- `agent-run tail <name>` — read-only, follows the log file.
- `agent-run steer <name> '<message>'` — one-shot write into the run's
  `stdin` FIFO.

There is no way to drive the agent live, keystroke-by-keystroke, as if it
were running in your own terminal. Users currently have to tail in one
terminal and send discrete `steer` commands from another, which is clunky
for anything interactive (TUIs, confirmation prompts, arrow-key navigation).

Separately, the PTY the agent runs in is never resized after launch — it's
stuck at whatever size `pty.fork()` defaulted to, regardless of the size of
whatever terminal is watching it. `agent-run tail` cannot fix this on its
own: it only reads the log file and has no channel to tell the runner
process (which owns the PTY master fd) about a new size.

## Goal

Add `agent-run attach <name>`: a command that puts the local terminal into
raw mode, relays local keystrokes into the run live, streams the run's
output live, and propagates local terminal resizes into the run's PTY —
without disturbing the run's existing detached lifecycle (it must survive
after every attach session ends).

## Non-goals

- Replacing `steer` or `tail` — both keep working exactly as they do today.
- Session exclusivity/locking — multiple simultaneous attaches are allowed.
- A push-based output channel — attach reuses `tail`'s log-polling approach.
- Propagating the wrapped agent's exit code out of `attach`.

## Design

### 1. New state: resize FIFO

Interactive launches gain a second FIFO, `state_dir/resize`, created next to
the existing `stdin` FIFO at launch time (unconditionally, whether or not
anything ever attaches). The existing keeper process — which already opens
`stdin` with `O_RDWR` so its own reader-side open never blocks waiting for a
writer — additionally opens `resize` the same way, and holds it open for the
lifetime of the run. This guarantees `resize` is always immediately writable
without needing a reader to show up first, exactly like `stdin` today.
Cleanup follows the same path as `stdin` (removed in `_launch`'s error/
cleanup handling, referenced in the same places `fifo_path` is).

### 2. Runner changes (`_run_interactive`)

The runner opens `resize` read-only, sets it `O_NONBLOCK` (matching how
`master_fd`/`fifo_fd` are already configured), and adds it as a third fd to
the existing `select()` loop.

Resize messages are a fixed 4-byte record: `struct.pack('>HH', cols, rows)`.
Fixed-size and small enough to be written atomically per POSIX `PIPE_BUF`
guarantees, so concurrent writers from multiple attached clients never
interleave a torn record — whoever wrote last simply wins, which is exactly
the desired "last resize wins" semantics with no extra coordination needed.

On a complete 4-byte read, the runner calls `fcntl.ioctl(master_fd,
termios.TIOCSWINSZ, struct.pack('HHHH', rows, cols, 0, 0))`. The kernel
delivers `SIGWINCH` to the PTY's foreground process group itself once the
size changes — the runner does not need to signal `pty_pid` manually.

Partial reads (fewer than 4 bytes available) are buffered and re-assembled
across loop iterations, the same pattern `_drain_pty_input`/`buf_in` already
use for FIFO input.

### 3. New `cmd_attach`

```
agent-run attach <name>
```

Preconditions mirror `cmd_steer`: run must exist, be interactive
(`interactive` state file == `"1"`), and alive (`pid` state file points at a
live process). Exits with a clear error otherwise, matching `steer`'s
existing error messages/style.

On start:

1. Save current termios state for the local stdin fd; put it into raw mode
   (`tty.setraw`). Register a `finally`/`atexit`-style restore so the
   terminal is never left raw on any exit path (normal detach, agent exit,
   uncaught exception, signal).
2. Open the run's `stdin` and `resize` FIFOs for writing (`O_WRONLY`) —
   these succeed immediately since the keeper holds each open for reading.
3. Install a `SIGWINCH` handler that computes the local terminal size via
   `os.get_terminal_size()` and writes a resize record to the `resize`
   FIFO. Fire it once synchronously immediately after attaching (not just
   on the first real resize event), so the PTY is sized correctly for the
   very first frame instead of waiting for the user to touch their window.
4. Run a `select()` loop, structurally mirroring `_run_interactive`'s own
   loop, over:
   - local stdin fd (readable → new input) — the same run/wait posture
     `_run_interactive` already uses for its own fds.
   - a bounded poll timeout, during which the log file is read from its
     current offset the same way `cmd_tail` does today (open once, `read()`
     in a loop, sleep-poll on EOF while the agent PID is alive).

Input handling: bytes read from local stdin are inspected for `\x03`
(Ctrl-C). If seen, `attach` treats it purely as a local detach signal — it
is *not* forwarded into the run's `stdin` FIFO. All other bytes (including
every other control character, e.g. Ctrl-D, Ctrl-Z, arrow-key escape
sequences) are forwarded verbatim into `stdin`, unbuffered, so the attached
session behaves like a real terminal in front of the agent.

Output handling: identical to `cmd_tail`'s existing loop — stream new log
bytes to local stdout as they appear.

Exit conditions:
- Local Ctrl-C detected → restore termios, exit 0. The run keeps running
  untouched.
- Agent process exits while attached (detected via `_pid_alive(pid)`,
  same check `cmd_tail` uses) → drain remaining log output, restore
  termios, exit 0. This matches `cmd_tail`'s current behavior exactly; the
  wrapped agent's actual exit code is not propagated.

### 4. Multi-attach

No locking, no exclusivity check. Any number of `attach` processes may run
concurrently against the same run, each independently:
- polling the log file for output (already safe for concurrent readers —
  `tail` does this today),
- opening `stdin`/`resize` for writing and sending its own local events.

Whichever attach process last wrote a resize record has the PTY sized to
its terminal; whichever last wrote input bytes is effectively "driving."
This is a deliberate, simple last-write-wins policy — no session takeover
protocol, no primary/secondary bookkeeping.

### 5. CLI wiring

New subparser, alongside `tail`/`steer`:

```python
sp_attach = sub.add_parser("attach", help="attach interactively (live keyboard + resize)")
sp_attach.add_argument("name")
sp_attach.set_defaults(func=cmd_attach)
```

Added to `known_subcommands` in the same place `"tail"` and `"steer"` are
listed.

## Error handling

- `resize` FIFO missing or not a FIFO on attach → clear error, same style
  as `cmd_steer`'s `"agent-run: no stdin FIFO at {fifo}"` check.
- Writes to `stdin`/`resize` during attach use the same short-timeout guard
  pattern `cmd_steer` uses around its FIFO write (`SIGALRM`-based), so a
  wedged run doesn't hang the attach session indefinitely on input.
- `ioctl(TIOCSWINSZ)` failures in the runner are caught and ignored (best
  effort — a failed resize should never take the run down), consistent with
  how the runner already treats PTY I/O errors as non-fatal where safe.

## Testing

- Unit-level: resize record framing/parsing (pack/unpack round-trip,
  partial-read reassembly across loop iterations) and that a complete
  record results in the expected `TIOCSWINSZ` ioctl call.
- Integration (matching the style of `tests/test_agent_run_*`, likely a new
  `tests/test_agent_run_attach.py`, using a pty/pexpect-driven harness):
  - Launch an interactive run, attach, send keystrokes, confirm they reach
    the wrapped agent (observable via its output).
  - Confirm Ctrl-C detaches the local session without killing the run or
    reaching the agent.
  - Confirm a `SIGWINCH` on the attaching terminal results in the wrapped
    agent observing the new `$COLUMNS`/`$LINES` (or `stty size`).
  - Confirm two concurrent attaches can both send input and both receive
    output, and that the PTY size ends up matching whichever attached last.
  - Confirm attach exits cleanly (and restores terminal mode) when the
    wrapped agent exits on its own while attached.
