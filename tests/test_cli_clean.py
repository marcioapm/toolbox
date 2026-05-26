"""Integration tests for the `agent-run clean` CLI subcommand.

These don't spawn real Claude — they seed a run directory with a captured
fixture log, then invoke `cmd_clean` directly to verify the CLI plumbing
(file lookup, --out, --width, missing-run errors) works end-to-end."""
from __future__ import annotations

import argparse
import sys

import pytest

from toolbox import agent_run


def _make_run(runs_root, name: str, log_bytes: bytes, interactive: bool = True):
    """Seed a minimal run directory the way agent-run would."""
    d = runs_root / name
    d.mkdir()
    (d / "log").write_bytes(log_bytes)
    (d / "status").write_text("done\n")
    (d / "pid").write_text("0\n")
    (d / "started_at").write_text("2026-05-26T18:00:00Z\n")
    (d / "interactive").write_text("1\n" if interactive else "0\n")
    return d


class TestCmdCleanStdout:
    def test_renders_real_moon_log_to_stdout(
        self, isolated_runs_root, moon_log_bytes, capsys
    ):
        _make_run(isolated_runs_root, "moon", moon_log_bytes)
        args = argparse.Namespace(
            name="moon", out=None, width=120, height=60, history=100000
        )
        rc = agent_run.cmd_clean(args)
        captured = capsys.readouterr()
        assert rc == 0
        # The moon paragraph keywords must be present.
        assert "craters" in captured.out
        assert "tides" in captured.out
        assert "●" in captured.out
        # No ANSI cruft.
        assert "\x1b" not in captured.out


class TestCmdCleanFileOut:
    def test_writes_to_file_with_dash_o(
        self, isolated_runs_root, moon_log_bytes, tmp_path, capsys
    ):
        _make_run(isolated_runs_root, "moon", moon_log_bytes)
        out_file = tmp_path / "moon.txt"
        args = argparse.Namespace(
            name="moon",
            out=str(out_file),
            width=120,
            height=60,
            history=100000,
        )
        rc = agent_run.cmd_clean(args)
        captured = capsys.readouterr()
        assert rc == 0
        # stdout should be silent on -o (the "wrote N bytes" line goes to stderr).
        assert captured.out == ""
        # File exists and contains the expected content.
        text = out_file.read_text(encoding="utf-8")
        assert "craters" in text
        assert "tides" in text
        # stderr gets the diagnostic.
        assert "wrote" in captured.err
        assert str(out_file) in captured.err


class TestCmdCleanErrors:
    def test_missing_run_exits(self, isolated_runs_root):
        args = argparse.Namespace(
            name="does-not-exist", out=None, width=120, height=60, history=100000
        )
        with pytest.raises(SystemExit):
            agent_run.cmd_clean(args)

    def test_run_without_log_exits(self, isolated_runs_root):
        # Create the dir but no log file.
        (isolated_runs_root / "no-log").mkdir()
        (isolated_runs_root / "no-log" / "status").write_text("done\n")
        args = argparse.Namespace(
            name="no-log", out=None, width=120, height=60, history=100000
        )
        with pytest.raises(SystemExit):
            agent_run.cmd_clean(args)


class TestCmdCleanCustomSizing:
    def test_custom_width_changes_wrapping(
        self, isolated_runs_root, moon_log_bytes, capsys
    ):
        _make_run(isolated_runs_root, "moon", moon_log_bytes)
        # Render twice with very different widths; the line-wrapping pattern
        # should differ even though the keywords remain present.
        args_wide = argparse.Namespace(
            name="moon", out=None, width=200, height=60, history=100000
        )
        agent_run.cmd_clean(args_wide)
        wide = capsys.readouterr().out

        args_narrow = argparse.Namespace(
            name="moon", out=None, width=60, height=60, history=100000
        )
        agent_run.cmd_clean(args_narrow)
        narrow = capsys.readouterr().out

        # Keywords survive both renderings.
        assert "craters" in wide
        assert "craters" in narrow
        # The two transcripts are not identical (different wrapping).
        assert wide != narrow
