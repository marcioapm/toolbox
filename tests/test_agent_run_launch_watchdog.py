"""Tests for launch-failure classification, the idle watchdog, and forced kills."""
from __future__ import annotations

import argparse
import os
import signal
import time
from pathlib import Path

import pytest

from toolbox import agent_run


def _seed_run(root: Path, name: str = "run", **files: str) -> Path:
    state = root / name
    state.mkdir()
    (state / "status").write_text("running\n")
    (state / "pid").write_text("123\n")
    (state / "pgid").write_text("123\n")
    for field, value in files.items():
        (state / field).write_text(value)
    return state


def _alive_for(count: int):
    """A `_pid_alive` stub that reports the runner alive `count` times, then
    dead, so a watchdog loop under test terminates on its own."""
    calls = {"n": 0}

    def _alive(_pid: int) -> bool:
        calls["n"] += 1
        return calls["n"] <= count

    return _alive


def _run_watchdog(state: Path, log_dir: Path, idle_timeout: float) -> None:
    agent_run._idle_watchdog_loop(state, log_dir, 4242, "linux:runner", idle_timeout)


# --- idle watchdog ---------------------------------------------------------


def test_watchdog_terminates_a_run_whose_log_stopped_growing(
    isolated_runs_root, isolated_log_root, monkeypatch
):
    state = _seed_run(isolated_runs_root)
    log_dir = isolated_log_root / "run"
    log_dir.mkdir()
    log = log_dir / "log"
    log.write_text("started\n")
    old = time.time() - 600
    os.utime(log, (old, old))

    signals: list[tuple[int, int]] = []
    monkeypatch.setenv("AGENT_RUN_WATCHDOG_STARTUP_GRACE_SECS", "0.5")
    monkeypatch.setattr(agent_run, "_pid_alive", lambda _pid: True)
    monkeypatch.setattr(agent_run, "_process_identity", lambda _pid: "linux:runner")
    monkeypatch.setattr(agent_run, "_send_signal_to_verified_pid", lambda pid, sig, _identity: signals.append((pid, sig)))
    monkeypatch.setattr(agent_run.os, "kill", lambda pid, sig: signals.append((pid, sig)))
    monkeypatch.setattr(agent_run, "KILL_ESCALATION_TIMEOUT_SECONDS", 0.1)
    monkeypatch.setattr(agent_run, "_watchdog_escalate", lambda *_args: None)

    _run_watchdog(state, log_dir, 1.0)

    assert (4242, signal.SIGTERM) in signals
    assert (state / agent_run.IDLE_TIMEOUT_MARKER).exists()


def test_watchdog_does_not_fire_before_first_output_within_startup_grace(
    isolated_runs_root, isolated_log_root, monkeypatch
):
    """A slow-starting agent has produced no output, so the log still carries
    the launcher's own touch timestamp. Measuring idleness against it would
    kill a run that was never idle."""
    state = _seed_run(isolated_runs_root)
    log_dir = isolated_log_root / "run"
    log_dir.mkdir()
    log = log_dir / "log"
    log.touch()
    stale = time.time() - 3600
    os.utime(log, (stale, stale))

    signals: list[tuple[int, int]] = []
    monkeypatch.setenv("AGENT_RUN_WATCHDOG_STARTUP_GRACE_SECS", "600")
    # Three polls with the runner alive, then the runner dies (identity probe
    # returns None and pid_alive returns False), ending the loop cleanly.
    identities = iter(["linux:runner"] * 3 + [None])
    monkeypatch.setattr(agent_run, "_process_identity", lambda _pid: next(identities, None))
    monkeypatch.setattr(agent_run, "_pid_alive", lambda _pid: False)
    monkeypatch.setattr(agent_run.os, "kill", lambda pid, sig: signals.append((pid, sig)))

    _run_watchdog(state, log_dir, 1.0)

    assert signals == []
    assert not (state / agent_run.IDLE_TIMEOUT_MARKER).exists()


def test_watchdog_does_not_fire_while_the_log_keeps_growing(
    isolated_runs_root, isolated_log_root, monkeypatch
):
    state = _seed_run(isolated_runs_root)
    log_dir = isolated_log_root / "run"
    log_dir.mkdir()
    log = log_dir / "log"
    log.write_text("tick\n")

    signals: list[tuple[int, int]] = []
    identity_calls = 0

    def _identity(_pid: int):
        nonlocal identity_calls
        identity_calls += 1
        if identity_calls > 3:
            return None
        with log.open("a") as handle:
            handle.write("tick\n")
        return "linux:runner"

    monkeypatch.setenv("AGENT_RUN_WATCHDOG_STARTUP_GRACE_SECS", "0.5")
    monkeypatch.setattr(agent_run, "_process_identity", _identity)
    # When identity returns None, _pid_alive is the fallback.  Returning False
    # exits the loop cleanly without sending any signal.
    monkeypatch.setattr(agent_run, "_pid_alive", lambda _pid: False)
    monkeypatch.setattr(agent_run.os, "kill", lambda pid, sig: signals.append((pid, sig)))

    _run_watchdog(state, log_dir, 1.0)

    assert signals == []


def test_watchdog_still_signals_when_the_marker_write_fails(
    isolated_runs_root, isolated_log_root, monkeypatch
):
    """A full or vanished state dir must not silently disable the watchdog:
    the marker only labels the kill, it does not authorize it."""
    state = _seed_run(isolated_runs_root)
    log_dir = isolated_log_root / "run"
    log_dir.mkdir()
    log = log_dir / "log"
    log.write_text("started\n")
    old = time.time() - 600
    os.utime(log, (old, old))

    signals: list[tuple[int, int]] = []

    def _refuse_write(*_args, **_kwargs):
        raise OSError("no space left on device")

    monkeypatch.setenv("AGENT_RUN_WATCHDOG_STARTUP_GRACE_SECS", "0.5")
    monkeypatch.setattr(agent_run, "_pid_alive", lambda _pid: True)
    monkeypatch.setattr(agent_run, "_process_identity", lambda _pid: "linux:runner")
    monkeypatch.setattr(agent_run, "_send_signal_to_verified_pid", lambda pid, sig, _identity: signals.append((pid, sig)))
    monkeypatch.setattr(agent_run, "_write", _refuse_write)
    monkeypatch.setattr(agent_run.os, "kill", lambda pid, sig: signals.append((pid, sig)))
    monkeypatch.setattr(agent_run, "KILL_ESCALATION_TIMEOUT_SECONDS", 0.1)
    monkeypatch.setattr(agent_run, "_watchdog_escalate", lambda *_args: None)

    _run_watchdog(state, log_dir, 1.0)

    assert (4242, signal.SIGTERM) in signals


def test_watchdog_escalates_when_the_runner_ignores_sigterm(
    isolated_runs_root, isolated_log_root, monkeypatch
):
    state = _seed_run(isolated_runs_root)
    log_dir = isolated_log_root / "run"
    log_dir.mkdir()
    log = log_dir / "log"
    log.write_text("started\n")
    old = time.time() - 600
    os.utime(log, (old, old))

    escalated: list[int] = []
    monkeypatch.setenv("AGENT_RUN_WATCHDOG_STARTUP_GRACE_SECS", "0.5")
    monkeypatch.setattr(agent_run, "_pid_alive", lambda _pid: True)
    monkeypatch.setattr(agent_run, "_process_identity", lambda _pid: "linux:runner")
    monkeypatch.setattr(agent_run, "_send_signal_to_verified_pid", lambda *_args: None)
    monkeypatch.setattr(agent_run.os, "kill", lambda _pid, _sig: None)
    monkeypatch.setattr(agent_run, "KILL_ESCALATION_TIMEOUT_SECONDS", 0.1)
    monkeypatch.setattr(
        agent_run, "_watchdog_escalate", lambda _state, pid, _identity: escalated.append(pid)
    )

    _run_watchdog(state, log_dir, 1.0)

    assert escalated == [4242]


def test_watchdog_escalation_publishes_killed_with_a_reason(
    isolated_runs_root, monkeypatch
):
    state = _seed_run(isolated_runs_root)
    (state / agent_run.IDLE_TIMEOUT_MARKER).write_text("42\n")
    monkeypatch.setattr(agent_run, "_process_identity", lambda _pid: "linux:runner")
    monkeypatch.setattr(agent_run, "_pid_parent_pid", lambda _pid: 4242)
    monkeypatch.setattr(agent_run, "_send_signal_to_verified_pid", lambda *_args: None)

    agent_run._watchdog_escalate(state, 4242, "linux:runner")

    assert (state / "status").read_text().strip() == "killed"
    assert "42" in (state / "reap_reason").read_text()
    assert (state / "exit_code").read_text().strip() == str(128 + signal.SIGKILL)


def test_watchdog_escalation_does_not_publish_when_identity_changed(
    isolated_runs_root, monkeypatch
):
    """A mismatched identity at escalation time means the runner is gone and a
    different process has the pid.  No terminal state must be published and no
    signal must be sent."""
    state = _seed_run(isolated_runs_root)
    (state / agent_run.IDLE_TIMEOUT_MARKER).write_text("42\n")
    signals_sent = []
    monkeypatch.setattr(agent_run, "_process_identity", lambda _pid: "linux:replacement")
    monkeypatch.setattr(agent_run, "_send_signal_to_verified_pid",
                        lambda *_args: signals_sent.append(_args))

    agent_run._watchdog_escalate(state, 4242, "linux:runner")

    assert not (state / "status").exists() or (state / "status").read_text().strip() == "running"
    assert signals_sent == [], "must not signal a recycled pid"


def test_watchdog_escalation_does_not_signal_aux_pid_with_wrong_parent(
    isolated_runs_root, monkeypatch
):
    """An aux pid whose parent is not the runner must not be signalled; pid
    reuse could have placed an unrelated process in the recorded slot."""
    state = _seed_run(isolated_runs_root)
    (state / agent_run.IDLE_TIMEOUT_MARKER).write_text("5\n")
    # Write a fake pty_pid into state.
    (state / "pty_pid").write_text("9999\n")
    os_kills = []
    monkeypatch.setattr(agent_run, "_process_identity", lambda _pid: "linux:runner")
    monkeypatch.setattr(agent_run, "_pid_parent_pid", lambda _pid: 1)  # not the runner
    monkeypatch.setattr(agent_run, "_send_signal_to_verified_pid", lambda *_args: None)
    monkeypatch.setattr(agent_run.os, "kill", lambda pid, sig: os_kills.append((pid, sig)))

    agent_run._watchdog_escalate(state, 4242, "linux:runner")

    assert 9999 not in [p for p, _ in os_kills], "unrelated aux pid must not be signalled"


def test_watchdog_escalation_reaps_legacy_echo_and_render_pid(
    isolated_runs_root, monkeypatch
):
    """A runner started by a pre-upgrade build can still have echo_pid/
    render_pid files on disk even though no current runner writes them;
    a mixed-version escalation must still discover and kill them."""
    state = _seed_run(isolated_runs_root)
    (state / agent_run.IDLE_TIMEOUT_MARKER).write_text("5\n")
    (state / "echo_pid").write_text("7001\n")
    (state / "render_pid").write_text("7002\n")
    os_kills = []
    monkeypatch.setattr(agent_run, "_process_identity", lambda _pid: "linux:runner")
    monkeypatch.setattr(agent_run, "_pid_parent_pid", lambda _pid: 4242)
    monkeypatch.setattr(agent_run, "_send_signal_to_verified_pid", lambda *_args: None)
    monkeypatch.setattr(agent_run.os, "kill", lambda pid, sig: os_kills.append((pid, sig)))

    agent_run._watchdog_escalate(state, 4242, "linux:runner")

    assert (7001, signal.SIGKILL) in os_kills
    assert (7002, signal.SIGKILL) in os_kills


# --- forced kill -----------------------------------------------------------


def test_force_kill_legacy_refuses_when_the_caller_shares_the_process_group(
    isolated_runs_root, monkeypatch
):
    """An agent that kills its own run would otherwise signal itself before any
    terminal state is published."""
    state = _seed_run(isolated_runs_root)
    (state / "pgid").write_text(f"{os.getpgrp()}\n")
    monkeypatch.setattr(
        agent_run.os, "killpg", lambda *_args: pytest.fail("signalled own group")
    )

    with pytest.raises(SystemExit, match="shares the run's process group"):
        agent_run._force_kill_legacy("run", state, 123, signal.SIGKILL)


def test_force_kill_legacy_sends_catchable_signals_to_the_runner_alone(
    isolated_runs_root, monkeypatch
):
    """Group-wide catchable signals reach the runner's children directly and
    race its own handler, so only SIGKILL targets the group."""
    state = _seed_run(isolated_runs_root)
    (state / "pgid").write_text("999\n")
    sent: list[tuple[int, int]] = []
    monkeypatch.setattr(
        agent_run.os, "killpg", lambda *_args: pytest.fail("used killpg for TERM")
    )
    monkeypatch.setattr(agent_run.os, "kill", lambda pid, sig: sent.append((pid, sig)))

    agent_run._force_kill_legacy("run", state, 123, signal.SIGTERM)

    assert sent == [(123, signal.SIGTERM)]


def test_force_kill_legacy_refuses_mismatched_recorded_group(
    isolated_runs_root, monkeypatch
):
    state = _seed_run(isolated_runs_root)
    (state / "pgid").write_text("999\n")
    monkeypatch.setattr(agent_run.os, "getpgid", lambda _pid: 998)
    monkeypatch.setattr(
        agent_run.os, "killpg", lambda *_args: pytest.fail("signalled unrelated group")
    )

    with pytest.raises(SystemExit, match="does not contain pid"):
        agent_run._force_kill_legacy("run", state, 123, signal.SIGKILL)


def test_force_kill_legacy_esrch_from_getpgid_is_not_a_refusal(
    isolated_runs_root, monkeypatch
):
    """ESRCH from getpgid means the process already exited; fall through to the
    ProcessLookupError handler rather than refusing with exit code 1.  A run
    whose pid died in the check window must not be permanently stuck in
    'running' with no way to force-kill it."""
    state = _seed_run(isolated_runs_root)
    (state / "pgid").write_text("999\n")
    monkeypatch.setattr(
        agent_run.os, "getpgid", lambda _pid: (_ for _ in ()).throw(ProcessLookupError(3, "No such process"))
    )
    monkeypatch.setattr(
        agent_run.os, "killpg", lambda _pgid, _sig: (_ for _ in ()).throw(ProcessLookupError(3, "No such process"))
    )
    monkeypatch.setattr(agent_run.os, "kill", lambda _pid, _sig: None)

    # Must not raise SystemExit; should return normally (process is gone).
    agent_run._force_kill_legacy("run", state, 123, signal.SIGKILL)


def test_watchdog_stops_when_runner_identity_changes(
        isolated_runs_root, isolated_log_root, monkeypatch
):
    state = _seed_run(isolated_runs_root)
    log_dir = isolated_log_root / "run"
    log_dir.mkdir()
    (log_dir / "log").write_text("started\n")
    monkeypatch.setattr(agent_run.time, "sleep", lambda _seconds: None)
    # _pid_alive returns True so the loop cannot exit on liveness alone —
    # only a changed identity can terminate it.
    monkeypatch.setattr(agent_run, "_pid_alive", lambda _pid: True)
    monkeypatch.setattr(agent_run, "_process_identity", lambda _pid: "linux:replacement")
    monkeypatch.setattr(
        agent_run, "_send_signal_to_verified_pid", lambda *_args: pytest.fail("signalled replacement")
    )

    _run_watchdog(state, log_dir, 1.0)

    assert not (state / agent_run.IDLE_TIMEOUT_MARKER).exists()


def test_watchdog_transient_identity_probe_failure_does_not_abandon_run(
    isolated_runs_root, isolated_log_root, monkeypatch
):
    """A one-shot None from _process_identity must not exit the watchdog when
    _pid_alive confirms the runner is still alive.

    With the old code (`if _process_identity(pid) != identity: return`),
    `None != identity` causes an immediate exit on poll 1.  With the fix, a
    None result falls back to _pid_alive; the loop continues when _pid_alive
    is True, and only exits on a subsequent poll when _pid_alive is False."""
    state = _seed_run(isolated_runs_root)
    log_dir = isolated_log_root / "run"
    log_dir.mkdir()
    log = log_dir / "log"
    log.write_text("started\n")
    # Backdate so idle_wall > idle_timeout and the idle check can fire,
    # but the startup-grace window has already elapsed.
    old = time.time() - 600
    os.utime(log, (old, old))

    poll_count = {"n": 0}
    monkeypatch.setattr(agent_run.time, "sleep", lambda _: None)

    def _identity(_pid):
        poll_count["n"] += 1
        # Always returns None — identity probe is permanently unavailable.
        return None

    alive_count = {"n": 0}

    def _alive(_pid):
        alive_count["n"] += 1
        # First call (poll 1 fallback): runner still alive.
        # Second call (poll 2 fallback): runner gone.
        return alive_count["n"] <= 1

    monkeypatch.setenv("AGENT_RUN_WATCHDOG_STARTUP_GRACE_SECS", "0.5")
    monkeypatch.setattr(agent_run, "_process_identity", _identity)
    monkeypatch.setattr(agent_run, "_pid_alive", _alive)
    monkeypatch.setattr(
        agent_run, "_send_signal_to_verified_pid", lambda *_args: None
    )
    monkeypatch.setattr(agent_run, "KILL_ESCALATION_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(agent_run, "_watchdog_escalate", lambda *_args: None)

    _run_watchdog(state, log_dir, 1.0)

    # Must have gone through at least 2 identity probes (2 polls).  With the
    # old code (`None != identity` → return), poll_count["n"] would be 1.
    assert poll_count["n"] >= 2, (
        f"watchdog exited after {poll_count['n']} poll(s); "
        "a None identity must not cause immediate exit when _pid_alive is True"
    )


def test_watchdog_recognizes_output_before_first_poll(
    isolated_runs_root, isolated_log_root, monkeypatch
):
    state = _seed_run(isolated_runs_root)
    log_dir = isolated_log_root / "run"
    log_dir.mkdir()
    log = log_dir / "log"
    log.write_text("output\n")
    old = time.time() - 600
    os.utime(log, (old, old))
    current = log.stat()
    baseline = (current.st_dev, current.st_ino, 0, current.st_mtime_ns - 1)
    sent = []
    monkeypatch.setenv("AGENT_RUN_WATCHDOG_STARTUP_GRACE_SECS", "600")
    monkeypatch.setattr(agent_run, "_process_identity", lambda _pid: "linux:runner")
    monkeypatch.setattr(
        agent_run, "_send_signal_to_verified_pid", lambda pid, sig, _identity: sent.append((pid, sig))
    )
    monkeypatch.setattr(agent_run, "KILL_ESCALATION_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(agent_run, "_watchdog_escalate", lambda *_args: None)
    # Without the initial_log_stat seed the loop never records first output, so
    # the 600 s startup grace suppresses the kill forever.  Bound the poll count
    # so that regression fails fast instead of hanging; the fixed path needs ~2.
    real_sleep = time.sleep
    polls = {"n": 0}

    def bounded_sleep(seconds):
        polls["n"] += 1
        assert polls["n"] <= 10, "watchdog did not terminate; startup grace never released"
        real_sleep(seconds)

    monkeypatch.setattr(agent_run.time, "sleep", bounded_sleep)

    agent_run._idle_watchdog_loop(
        state, log_dir, 4242, "linux:runner", 1.0, baseline
    )

    assert (4242, signal.SIGTERM) in sent


# --- launch diagnostics ----------------------------------------------------


def test_status_surfaces_launch_error(isolated_runs_root, isolated_log_root, capsys):
    state = _seed_run(isolated_runs_root)
    (state / "status").write_text("launch_failed\n")
    (state / "exit_code").write_text("1\n")
    (state / "launch_error").write_text("Error: you must provide a message\n")
    (isolated_log_root / "run").mkdir()
    (isolated_log_root / "run" / "log").write_text("Error: you must provide a message\n")

    agent_run.cmd_status(argparse.Namespace(name="run"))

    out = capsys.readouterr().out
    assert "status=launch_failed" in out
    assert "you must provide a message" in out


def test_status_reads_the_durable_launch_error_when_state_is_gone(
    isolated_runs_root, isolated_log_root, capsys
):
    """The state dir is volatile; the log dir outlives reap GC and reboots."""
    log_dir = isolated_log_root / "run"
    log_dir.mkdir()
    (log_dir / "log").write_text("boom\n")
    (log_dir / "launch_error").write_text("agent-run: exec failed\n")

    agent_run.cmd_status(argparse.Namespace(name="run"))

    out = capsys.readouterr().out
    assert "log preserved" in out
    assert "exec failed" in out


# --- status vocabulary and validators --------------------------------------


@pytest.mark.parametrize(
    "status", ["done", "failed", "launch_failed", "died", "killed"]
)
def test_run_is_terminal_covers_every_terminal_status(isolated_runs_root, status):
    state = _seed_run(isolated_runs_root)
    (state / "status").write_text(status + "\n")
    assert agent_run._run_is_terminal(state) is True


@pytest.mark.parametrize("status", ["running", "starting", "stalled"])
def test_run_is_terminal_rejects_live_statuses(isolated_runs_root, status):
    state = _seed_run(isolated_runs_root)
    (state / "status").write_text(status + "\n")
    assert agent_run._run_is_terminal(state) is False


@pytest.mark.parametrize("raw", ["nan", "inf", "-inf", "-5", "0", "junk"])
def test_idle_timeout_flag_rejects_non_positive_and_non_finite(raw):
    with pytest.raises(argparse.ArgumentTypeError):
        agent_run._parse_idle_timeout_flag(raw)


def test_idle_timeout_env_defaults_to_off_on_invalid_value(monkeypatch):
    monkeypatch.setenv("AGENT_RUN_IDLE_TIMEOUT_SECS", "nan")
    assert agent_run._idle_timeout_env_seconds() is None


def test_launch_grace_falls_back_to_the_default_on_invalid_value(monkeypatch, capsys):
    monkeypatch.setenv("AGENT_RUN_LAUNCH_GRACE_SECS", "-1")
    assert agent_run._parse_launch_grace_seconds() == 10.0
    assert "not a finite, positive" in capsys.readouterr().err


def test_flags_after_the_run_name_are_rejected_before_forking(monkeypatch):
    monkeypatch.setattr(
        agent_run, "cmd_launch", lambda *_args: pytest.fail("launched with a flag as argv[0]")
    )
    with pytest.raises(SystemExit, match="must precede the run name"):
        agent_run.main(["myrun", "-f", "brief.md", "echo", "hi"])
