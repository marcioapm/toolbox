"""Shared pytest fixtures for the toolbox test suite."""
from __future__ import annotations

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
    ("craters", "tides")."""
    return (fixtures_dir / "claude_moon_tui.log").read_bytes()


@pytest.fixture
def print_log_bytes(fixtures_dir: Path) -> bytes:
    """A `claude --print` capture: 15 bytes, plain text, no ANSI."""
    return (fixtures_dir / "claude_print.log").read_bytes()
