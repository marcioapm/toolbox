"""Integration tests for the `agent-run clean` CLI subcommand.

These don't spawn real Claude — they seed a run directory with a captured
fixture log, then invoke `cmd_clean` directly to verify the CLI plumbing
(file lookup, --out, --width, missing-run errors) works end-to-end."""
from __future__ import annotations

import argparse
import builtins
import io
import signal
import sys

import pytest

from toolbox import agent_run


def _make_run(runs_root, name: str, log_bytes: bytes, interactive: bool = True, old_layout: bool = False):
    """Seed a run the way agent-run would: ephemeral state under
    runs_root/<name>/, persistent log under LOG_ROOT/<name>/log. Pass
    old_layout=True to seed the pre-split single-directory layout instead
    (log alongside state), to exercise the fallback path."""
    from toolbox import agent_run

    d = runs_root / name
    d.mkdir()
    (d / "status").write_text("done\n")
    (d / "pid").write_text("0\n")
    (d / "started_at").write_text("2026-05-26T18:00:00Z\n")
    (d / "interactive").write_text("1\n" if interactive else "0\n")
    if old_layout:
        (d / "log").write_bytes(log_bytes)
    else:
        log_d = agent_run.LOG_ROOT / name
        log_d.mkdir(parents=True, exist_ok=True)
        (log_d / "log").write_bytes(log_bytes)
    return d


class _FakeStdout:
    def __init__(self, tty: bool):
        self.buffer = io.BytesIO()
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


class _BrokenPipeBuffer:
    def write(self, _data: bytes) -> int:
        raise BrokenPipeError

    def flush(self) -> None:
        raise BrokenPipeError


class _BrokenPipeStdout:
    buffer = _BrokenPipeBuffer()

    def isatty(self) -> bool:
        return True


def test_tail_prints_preserved_log_and_exits(
    isolated_runs_root, isolated_log_root, monkeypatch
):
    log_dir = isolated_log_root / "preserved"
    log_dir.mkdir()
    (log_dir / "log").write_bytes(b"preserved output\n")
    stdout = _FakeStdout(tty=True)
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(
        agent_run.time,
        "sleep",
        lambda _seconds: pytest.fail("tail slept despite no live PID"),
    )

    assert agent_run.cmd_tail(argparse.Namespace(name="preserved")) == 0
    assert stdout.buffer.getvalue() == b"preserved output\n" + agent_run._TERMINAL_MODE_RESET


def test_logs_resets_terminal_modes_on_tty(isolated_runs_root, monkeypatch):
    _make_run(isolated_runs_root, "tty", b"captured output\n")
    stdout = _FakeStdout(tty=True)
    monkeypatch.setattr(sys, "stdout", stdout)

    assert agent_run.cmd_logs(argparse.Namespace(name="tty", n=1)) == 0
    assert stdout.buffer.getvalue() == b"captured output\n" + agent_run._TERMINAL_MODE_RESET


def test_logs_does_not_reset_terminal_modes_when_not_a_tty(
    isolated_runs_root, monkeypatch
):
    _make_run(isolated_runs_root, "pipe", b"captured output\n")
    stdout = _FakeStdout(tty=False)
    monkeypatch.setattr(sys, "stdout", stdout)

    assert agent_run.cmd_logs(argparse.Namespace(name="pipe", n=1)) == 0
    assert stdout.buffer.getvalue() == b"captured output\n"


def test_tail_resets_terminal_modes_after_ctrl_c(isolated_runs_root, monkeypatch):
    run = _make_run(isolated_runs_root, "following", b"")
    (run / "pid").write_text("123\n")
    stdout = _FakeStdout(tty=True)
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(agent_run, "_pid_alive", lambda _pid: True)
    monkeypatch.setattr(
        agent_run.time,
        "sleep",
        lambda _seconds: (_ for _ in ()).throw(KeyboardInterrupt),
    )

    assert agent_run.cmd_tail(argparse.Namespace(name="following")) == 128 + signal.SIGINT
    assert stdout.buffer.getvalue() == agent_run._TERMINAL_MODE_RESET


def test_logs_survives_broken_stdout(isolated_runs_root, monkeypatch):
    _make_run(isolated_runs_root, "broken", b"captured output\n")
    monkeypatch.setattr(sys, "stdout", _BrokenPipeStdout())

    assert agent_run.cmd_logs(argparse.Namespace(name="broken", n=1)) == 0


def test_reset_survives_closed_stdout(monkeypatch):
    stdout = _FakeStdout(tty=True)
    stdout.buffer.close()
    monkeypatch.setattr(sys, "stdout", stdout)

    agent_run._reset_terminal_modes()


class TestCliNumericValidation:
    @pytest.mark.parametrize(
        "argv",
        [
            ["logs", "run", "0"],
            ["logs", "run", "-1"],
            ["clean", "run", "--width", "0"],
            ["clean", "run", "--height", "-1"],
            ["clean", "run", "--history", "-1"],
        ],
    )
    def test_argparse_rejects_out_of_range_integers(self, argv, capsys):
        with pytest.raises(SystemExit) as exc_info:
            agent_run._build_parser().parse_args(argv)

        assert exc_info.value.code == 2
        assert "must be at least" in capsys.readouterr().err

    def test_history_zero_is_valid(self):
        args = agent_run._build_parser().parse_args(
            ["clean", "run", "--width", "1", "--height", "1", "--history", "0"]
        )
        assert (args.width, args.height, args.history) == (1, 1, 0)

    @pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "-inf"])
    def test_echo_interval_must_be_finite_and_positive(self, value):
        with pytest.raises(SystemExit, match="finite and greater than 0"):
            agent_run.main([f"--echo={value}", "run", "true"])

    def test_positive_echo_interval_is_accepted(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(
            agent_run,
            "cmd_launch",
            lambda args: captured.update(vars(args)) or 0,
        )

        assert agent_run.main(["--echo=0.25", "run", "true"]) == 0
        assert captured["echo_interval"] == 0.25


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

    def test_missing_pyte_exits_with_dependency_diagnostic(
        self, isolated_runs_root, monkeypatch
    ):
        _make_run(isolated_runs_root, "missing-pyte", b"output\n")
        real_import = builtins.__import__

        def no_pyte(name, *args, **kwargs):
            if name == "pyte":
                raise ImportError("forced missing pyte")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", no_pyte)
        args = argparse.Namespace(
            name="missing-pyte", out=None, width=120, height=60, history=100000
        )

        with pytest.raises(SystemExit, match="pyte.*required"):
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
