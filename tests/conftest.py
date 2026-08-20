"""Shared pytest fixtures for the toolbox test suite."""
from __future__ import annotations

import os
import signal
import time
from pathlib import Path

import pytest


FIXTURES = Path(__file__).parent / "fixtures"


def _ancestor_pids() -> set:
    """This process's ancestor pids, stopping at init, a lookup failure, or a
    cycle."""
    from toolbox import agent_run

    seen = set()
    current = os.getppid()
    while current and current > 1 and current not in seen:
        seen.add(current)
        try:
            current = agent_run._pid_parent_pid(current)
        except Exception:
            break
    return seen


@pytest.fixture
def fixtures_dir() -> Path:
    """Path to the bundled real-Claude log captures used as test inputs."""
    return FIXTURES


@pytest.fixture
def moon_log_bytes(fixtures_dir: Path) -> bytes:
    """A real PTY-captured Claude Code v2.1.x -i session that answered a
    prompt about the moon. Contains Ink TUI redraws, ANSI escapes, OSC
    title sequences, and a known assistant response substring
    (\"craters\", \"tides\")."""
    return (fixtures_dir / "claude_moon_tui.log").read_bytes()


@pytest.fixture
def print_log_bytes(fixtures_dir: Path) -> bytes:
    """A `claude --print` capture: 15 bytes, plain text, no ANSI."""
    return (fixtures_dir / "claude_print.log").read_bytes()


@pytest.fixture
def isolated_runs_root(request, tmp_path, monkeypatch) -> Path:
    """Point agent-run's state/log roots at fresh temp dirs so tests don't
    collide with real /tmp/agent-runs/ or /var/tmp/agent-runs/. Reaches into
    both the env vars (which the CLI reads at import time) and the
    module-level constants.

    Registers a finalizer that kills every harness process still alive in the
    state root after the test ends — both pass and fail paths — so an
    interrupted or slow test cannot leave orphaned processes behind.
    """
    state = tmp_path / "agent-runs-state"
    logs = tmp_path / "agent-runs-log"
    state.mkdir()
    logs.mkdir()
    monkeypatch.setenv("AGENT_RUN_STATE_DIR", str(state))
    monkeypatch.setenv("AGENT_RUN_LOG_DIR", str(logs))
    # The module captured these at import time; patch them too.
    from toolbox import agent_run
    monkeypatch.setattr(agent_run, "STATE_ROOT", state)
    monkeypatch.setattr(agent_run, "LOG_ROOT", logs)

    own_pid = os.getpid()
    own_ancestors = _ancestor_pids()

    def _reap_all() -> None:
        """Kill every harness runner pid still recorded in the state root.

        Only targets runs whose status is non-terminal (running or starting) —
        tests that use the pid file for other purposes (e.g. steer tests that
        write the test process's own pid) are not affected. Never signals the
        current process, one of its ancestors, or pid 0/negatives.
        """
        if not state.is_dir():
            return
        for run_dir in state.iterdir():
            if not run_dir.is_dir():
                continue
            # Only reap runs still in a non-terminal state.
            try:
                run_status = (run_dir / "status").read_text().strip()
            except OSError:
                continue
            if run_status not in {"running", "starting"}:
                continue
            pid_file = run_dir / "pid"
            if not pid_file.exists():
                continue
            try:
                pid = int(pid_file.read_text().strip())
            except (ValueError, OSError):
                continue
            # An ancestry-resolution test may legitimately record an ancestor
            # pid, and under CI the immediate parent is the step's shell.
            if pid <= 0 or pid == own_pid or pid in own_ancestors:
                continue
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.kill(pid, sig)
                except (ProcessLookupError, PermissionError):
                    break
                except OSError:
                    break
                # Brief wait for SIGTERM to take effect before escalating.
                if sig == signal.SIGTERM:
                    time.sleep(0.3)

    request.addfinalizer(_reap_all)
    return state


@pytest.fixture
def isolated_log_root(isolated_runs_root) -> Path:
    """The persistent log root paired with isolated_runs_root."""
    from toolbox import agent_run
    return agent_run.LOG_ROOT
