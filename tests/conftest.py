"""Shared pytest fixtures for the toolbox test suite."""
from __future__ import annotations

import os
from pathlib import Path

import pytest


FIXTURES = Path(__file__).parent / "fixtures"


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
def isolated_runs_root(tmp_path, monkeypatch) -> Path:
    """Point agent-run's runs-root at a fresh temp dir so tests don't
    collide with real /tmp/agent-runs/. Reaches into both the env var
    (which the CLI reads at import time) and the module-level constant."""
    runs = tmp_path / "agent-runs"
    runs.mkdir()
    monkeypatch.setenv("AGENT_RUN_DIR", str(runs))
    # The module captured RUNS_ROOT at import time; patch it too.
    from toolbox import agent_run
    monkeypatch.setattr(agent_run, "RUNS_ROOT", runs)
    return runs
