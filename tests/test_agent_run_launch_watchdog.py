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
    agent_run._idle_watchdog_loop(state, log_dir, 4242, idle_timeout)


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
    monkeypatch.setattr(agent_run, "_pid_alive", _alive_for(3))
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
    alive = _alive_for(3)

    def _growing(pid: int) -> bool:
        with log.open("a") as handle:
            handle.write("tick\n")
        return alive(pid)

    monkeypatch.setenv("AGENT_RUN_WATCHDOG_STARTUP_GRACE_SECS", "0.5")
    monkeypatch.setattr(agent_run, "_pid_alive", _growing)
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
    monkeypatch.setattr(agent_run.os, "kill", lambda _pid, _sig: None)
    monkeypatch.setattr(agent_run, "KILL_ESCALATION_TIMEOUT_SECONDS", 0.1)
    monkeypatch.setattr(
        agent_run, "_watchdog_escalate", lambda _state, pid: escalated.append(pid)
    )

    _run_watchdog(state, log_dir, 1.0)

    assert escalated == [4242]


def test_watchdog_escalation_publishes_killed_with_a_reason(
    isolated_runs_root, monkeypatch
):
    state = _seed_run(isolated_runs_root)
    (state / agent_run.IDLE_TIMEOUT_MARKER).write_text("42\n")
    monkeypatch.setattr(agent_run.os, "kill", lambda _pid, _sig: None)

    agent_run._watchdog_escalate(state, 4242)

    assert (state / "status").read_text().strip() == "killed"
    assert "42" in (state / "reap_reason").read_text()
    assert (state / "exit_code").read_text().strip() == str(128 + signal.SIGKILL)


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
