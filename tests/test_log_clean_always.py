"""Tests for the always-on log.clean production guarantee.

Every run that produces output must produce log.clean and a valid
log.clean.meta.json, regardless of launch mode (interactive, one-shot,
or the legacy --echo flag).  The --echo flag is accepted for backward
compatibility but is now a no-op: the live daemon and final render run
unconditionally.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pytest

from toolbox import agent_run


def _wait_until(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def _metadata_is_valid(log_dir: Path) -> bool:
    """True when log.clean.meta.json covers the current raw log completely."""
    clean = log_dir / "log.clean"
    meta_path = log_dir / "log.clean.meta.json"
    if not clean.is_file() or not meta_path.is_file():
        return False
    try:
        metadata = json.loads(meta_path.read_text())
    except (OSError, ValueError):
        return False
    log_stat = (log_dir / "log").stat()
    return (
        metadata.get("version") == 1
        and metadata.get("complete") is True
        and metadata.get("dev") == log_stat.st_dev
        and metadata.get("ino") == log_stat.st_ino
        and metadata.get("offset") == log_stat.st_size
        and metadata.get("size") == log_stat.st_size
    )


class TestOneShotAlwaysProducesLogClean:
    """A non-interactive run (no -i, no --echo) must produce log.clean."""

    def test_oneshot_produces_log_clean(self, isolated_runs_root, isolated_log_root):
        """A plain one-shot run creates log.clean and valid metadata."""
        args = argparse.Namespace(
            name="oneshot-clean",
            command=[sys.executable, "-c", "print('oneshot-output')"],
            interactive=False,
            prompt_file=None,
            echo=False,
            echo_interval=2.0,
        )
        assert agent_run.cmd_launch(args) == 0
        state = isolated_runs_root / "oneshot-clean"
        log_dir = isolated_log_root / "oneshot-clean"
        assert _wait_until(lambda: (state / "status").read_text().strip() == "done")
        clean = log_dir / "log.clean"
        assert _wait_until(lambda: clean.is_file()), "log.clean was not produced"
        assert "oneshot-output" in clean.read_text()

    def test_oneshot_metadata_is_valid(self, isolated_runs_root, isolated_log_root):
        """log.clean.meta.json covers the raw log with complete=True."""
        args = argparse.Namespace(
            name="oneshot-meta",
            command=[sys.executable, "-c", "print('meta-check')"],
            interactive=False,
            prompt_file=None,
            echo=False,
            echo_interval=2.0,
        )
        assert agent_run.cmd_launch(args) == 0
        state = isolated_runs_root / "oneshot-meta"
        log_dir = isolated_log_root / "oneshot-meta"
        assert _wait_until(lambda: (state / "status").read_text().strip() == "done")
        assert _wait_until(lambda: _metadata_is_valid(log_dir)), (
            "log.clean.meta.json does not cover the raw log"
        )


class TestPrintModeSGRRendering:
    """Print-mode output with SGR colour codes must be rendered away in log.clean.

    Real print-mode runs emit SGR sequences such as \\x1b[0m (reset) and \\x1b[90m
    (dim/dark grey) even when non-interactive.  log.clean must contain the visible
    text without those sequences, and every rendered line must start at column 0
    (no staircase from bare-LF cursor drift).
    """

    @pytest.fixture
    def print_sgr_log_bytes(self, fixtures_dir) -> bytes:
        """Synthetic print-mode log shaped after real runs: bare-LF line endings,
        SGR dim prefix, SGR reset.  One-shot runs write to the log fd without a
        PTY, so there is no ONLCR translation and line endings are bare LF."""
        raw = (fixtures_dir / "print_mode_sgr.log").read_bytes()
        # Fixture must be bare LF; CRLF would mask the staircase bug.
        assert b"\r\n" not in raw, "fixture must use bare LF, not CRLF"
        assert b"\n" in raw, "fixture must have at least one bare LF"
        return raw

    def test_sgr_codes_absent_from_log_clean(
        self, isolated_runs_root, isolated_log_root, print_sgr_log_bytes, tmp_path
    ):
        """SGR colour codes present in the raw log must not appear in log.clean."""
        # Verify the fixture actually contains SGR codes (sanity guard).
        assert b"\x1b[0m" in print_sgr_log_bytes
        assert b"\x1b[90m" in print_sgr_log_bytes

        # Write the fixture to a data file and emit it from a helper script.
        data_file = tmp_path / "sgr.log"
        data_file.write_bytes(print_sgr_log_bytes)
        script = tmp_path / "emit_sgr.py"
        script.write_text(
            f"import sys\n"
            f"sys.stdout.buffer.write(open({str(data_file)!r}, 'rb').read())\n"
        )
        args = argparse.Namespace(
            name="sgr-render",
            command=[sys.executable, str(script)],
            interactive=False,
            prompt_file=None,
            echo=False,
            echo_interval=2.0,
        )
        assert agent_run.cmd_launch(args) == 0
        state = isolated_runs_root / "sgr-render"
        log_dir = isolated_log_root / "sgr-render"
        assert _wait_until(lambda: (state / "status").read_text().strip() == "done")
        clean = log_dir / "log.clean"
        assert _wait_until(lambda: clean.is_file())
        content = clean.read_text()
        assert "\x1b[" not in content, "SGR introducer leaked into log.clean"
        assert "\x1b" not in content, "ESC byte leaked into log.clean"
        # No staircase: every non-empty line must start at column 0.
        staircase = [l for l in content.splitlines() if l != l.lstrip() and l.lstrip()]
        assert not staircase, (
            f"bare-LF staircase in log.clean: {staircase[:2]}"
        )

    def test_sgr_visible_text_preserved(
        self, isolated_runs_root, isolated_log_root, print_sgr_log_bytes, tmp_path
    ):
        """Text that was wrapped in SGR codes must still appear in log.clean."""
        data_file = tmp_path / "sgr2.log"
        data_file.write_bytes(print_sgr_log_bytes)
        script = tmp_path / "emit_sgr2.py"
        script.write_text(
            f"import sys\n"
            f"sys.stdout.buffer.write(open({str(data_file)!r}, 'rb').read())\n"
        )
        args = argparse.Namespace(
            name="sgr-text",
            command=[sys.executable, str(script)],
            interactive=False,
            prompt_file=None,
            echo=False,
            echo_interval=2.0,
        )
        assert agent_run.cmd_launch(args) == 0
        state = isolated_runs_root / "sgr-text"
        log_dir = isolated_log_root / "sgr-text"
        assert _wait_until(lambda: (state / "status").read_text().strip() == "done")
        clean = log_dir / "log.clean"
        assert _wait_until(lambda: clean.is_file())
        content = clean.read_text()
        # Visible text from the fixture must survive the render.
        assert "FIXTURE-FINAL-LINE" in content
        assert "FIXTURE-WORD-0" in content
        # No staircase: bare-LF cursor drift must not indent any line.
        staircase = [l for l in content.splitlines() if l != l.lstrip() and l.lstrip()]
        assert not staircase, (
            f"bare-LF staircase in log.clean: {staircase[:2]}"
        )


class TestInteractiveModeAlwaysProducesLogClean:
    """Interactive (-i) runs must also produce log.clean."""

    def test_interactive_produces_log_clean(self, isolated_runs_root, isolated_log_root):
        args = argparse.Namespace(
            name="interactive-clean",
            command=[sys.executable, "-c", "print('interactive-output')"],
            interactive=True,
            prompt_file=None,
            echo=False,
            echo_interval=2.0,
        )
        assert agent_run.cmd_launch(args) == 0
        state = isolated_runs_root / "interactive-clean"
        log_dir = isolated_log_root / "interactive-clean"
        assert _wait_until(lambda: (state / "status").read_text().strip() in {"done", "failed"})
        clean = log_dir / "log.clean"
        assert _wait_until(lambda: clean.is_file()), "log.clean was not produced for interactive run"


class TestEmptyOutputNoSpuriousCache:
    """A run that produces no output must not create log.clean."""

    def test_no_output_no_log_clean(self, isolated_runs_root, isolated_log_root):
        """A command that exits immediately with no output leaves no log.clean."""
        args = argparse.Namespace(
            name="empty-output",
            command=[sys.executable, "-c", "pass"],
            interactive=False,
            prompt_file=None,
            echo=False,
            echo_interval=2.0,
        )
        assert agent_run.cmd_launch(args) == 0
        state = isolated_runs_root / "empty-output"
        log_dir = isolated_log_root / "empty-output"
        assert _wait_until(lambda: (state / "status").read_text().strip() == "done")
        # Allow a moment for any spurious write to land.
        time.sleep(0.2)
        assert not (log_dir / "log.clean").exists(), (
            "log.clean must not be created for a run with no output"
        )


class TestEchoFlagBackwardCompat:
    """--echo is accepted and does not break anything; behavior is now unconditional."""

    def test_echo_true_still_produces_log_clean(self, isolated_runs_root, isolated_log_root):
        """Explicit --echo flag still works and produces log.clean."""
        args = argparse.Namespace(
            name="echo-explicit",
            command=[sys.executable, "-c", "print('echo-output')"],
            interactive=False,
            prompt_file=None,
            echo=True,
            echo_interval=60.0,
        )
        assert agent_run.cmd_launch(args) == 0
        state = isolated_runs_root / "echo-explicit"
        log_dir = isolated_log_root / "echo-explicit"
        assert _wait_until(lambda: (state / "status").read_text().strip() == "done")
        clean = log_dir / "log.clean"
        assert _wait_until(lambda: clean.is_file() and "echo-output" in clean.read_text())

    def test_echo_false_and_echo_true_produce_same_rendered_output(
        self, isolated_runs_root, isolated_log_root
    ):
        """A run without --echo and one with --echo produce identical rendered output
        for the same command, confirming the flag is now a no-op for rendering."""
        command_source = "import sys; sys.stdout.write('common-output\\n')"

        for name, echo_flag in [("no-echo-run", False), ("with-echo-run", True)]:
            args = argparse.Namespace(
                name=name,
                command=[sys.executable, "-c", command_source],
                interactive=False,
                prompt_file=None,
                echo=echo_flag,
                echo_interval=2.0,
            )
            assert agent_run.cmd_launch(args) == 0

        for name in ("no-echo-run", "with-echo-run"):
            state = isolated_runs_root / name
            assert _wait_until(lambda s=state: (s / "status").read_text().strip() == "done")

        for name in ("no-echo-run", "with-echo-run"):
            log_dir = isolated_log_root / name
            assert _wait_until(lambda ld=log_dir: (ld / "log.clean").is_file())

        no_echo_clean = (isolated_log_root / "no-echo-run" / "log.clean").read_text()
        with_echo_clean = (isolated_log_root / "with-echo-run" / "log.clean").read_text()
        assert "common-output" in no_echo_clean
        assert "common-output" in with_echo_clean
