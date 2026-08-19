"""Integration tests for the `agent-run clean` CLI subcommand.

These don't spawn real Claude — they seed a run directory with a captured
fixture log, then invoke `cmd_clean` directly to verify the CLI plumbing
(file lookup, --out, --width, missing-run errors) works end-to-end."""
from __future__ import annotations

import argparse
import builtins
import io
import json
import os
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

    assert agent_run.cmd_logs(argparse.Namespace(name="tty", tail=1, head=None)) == 0
    assert stdout.buffer.getvalue() == b"captured output\n" + agent_run._TERMINAL_MODE_RESET


def test_logs_does_not_reset_terminal_modes_when_not_a_tty(
    isolated_runs_root, monkeypatch
):
    _make_run(isolated_runs_root, "pipe", b"captured output\n")
    stdout = _FakeStdout(tty=False)
    monkeypatch.setattr(sys, "stdout", stdout)

    assert agent_run.cmd_logs(argparse.Namespace(name="pipe", tail=1, head=None)) == 0
    assert stdout.buffer.getvalue() == b"captured output\n"


def test_tail_resets_terminal_modes_after_ctrl_c(isolated_runs_root, monkeypatch):
    """Exercises the non-TTY fallback path explicitly (sys.stdin.isatty
    forced False) rather than relying on the real invoking environment's
    stdin happening to not be a TTY (e.g. under default pytest capture) --
    `tail` never touches terminal modes, so this is the only path: the
    KeyboardInterrupt handler must still run the DEC private-mode reset
    that the replayed PTY bytes made necessary."""
    run = _make_run(isolated_runs_root, "following", b"")
    (run / "pid").write_text("123\n")
    stdout = _FakeStdout(tty=True)
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(agent_run.sys.stdin, "isatty", lambda: False)
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

    assert agent_run.cmd_logs(argparse.Namespace(name="broken", tail=1, head=None)) == 0


def test_reset_survives_closed_stdout(monkeypatch):
    stdout = _FakeStdout(tty=True)
    stdout.buffer.close()
    monkeypatch.setattr(sys, "stdout", stdout)

    agent_run._reset_terminal_modes()


class TestCliNumericValidation:
    @pytest.mark.parametrize(
        "argv",
        [
            ["logs", "run", "--tail", "0"],
            ["logs", "run", "--tail", "-1"],
            ["logs", "run", "--head", "0"],
            ["logs", "run", "--head", "-1"],
            ["clean", "run", "--width", "0"],
            ["clean", "run", "--height", "-1"],
            ["clean", "run", "--history", "-1"],
            ["clean", "run", "--tail", "0"],
            ["clean", "run", "--tail", "-1"],
            ["clean", "run", "--head", "0"],
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


# ---------------------------------------------------------------------------
# logs --tail / --head
# ---------------------------------------------------------------------------

def _make_log_run(runs_root, name: str, log_bytes: bytes):
    """Seed a log-only run (no state dir needed for logs command)."""
    from toolbox import agent_run as ar
    log_d = ar.LOG_ROOT / name
    log_d.mkdir(parents=True, exist_ok=True)
    (log_d / "log").write_bytes(log_bytes)
    # State dir required for _require_log fallback path resolution.
    state_d = runs_root / name
    state_d.mkdir(parents=True, exist_ok=True)
    (state_d / "status").write_text("done\n")
    return log_d


class TestLogsFlags:
    def test_tail_n_returns_last_n_lines(self, isolated_runs_root, monkeypatch, capsys):
        log_content = "\n".join(f"line{i}" for i in range(1, 21)) + "\n"
        _make_log_run(isolated_runs_root, "tailrun", log_content.encode())
        stdout = _FakeStdout(tty=False)
        monkeypatch.setattr(sys, "stdout", stdout)
        rc = agent_run.cmd_logs(argparse.Namespace(name="tailrun", tail=5, head=None))
        assert rc == 0
        output = stdout.buffer.getvalue().decode()
        lines = output.splitlines()
        assert lines == [f"line{i}" for i in range(16, 21)]

    def test_head_n_returns_first_n_lines(self, isolated_runs_root, monkeypatch):
        log_content = "\n".join(f"line{i}" for i in range(1, 21)) + "\n"
        _make_log_run(isolated_runs_root, "headrun", log_content.encode())
        stdout = _FakeStdout(tty=False)
        monkeypatch.setattr(sys, "stdout", stdout)
        rc = agent_run.cmd_logs(argparse.Namespace(name="headrun", tail=None, head=5))
        assert rc == 0
        output = stdout.buffer.getvalue().decode()
        lines = output.splitlines()
        assert lines == [f"line{i}" for i in range(1, 6)]

    def test_default_is_tail_50(self, isolated_runs_root, monkeypatch):
        # 60 lines; default (no flag) should return last 50.
        log_content = "\n".join(f"L{i:03d}" for i in range(1, 61)) + "\n"
        _make_log_run(isolated_runs_root, "defrun", log_content.encode())
        stdout = _FakeStdout(tty=False)
        monkeypatch.setattr(sys, "stdout", stdout)
        rc = agent_run.cmd_logs(argparse.Namespace(name="defrun", tail=None, head=None))
        assert rc == 0
        lines = stdout.buffer.getvalue().decode().splitlines()
        assert len(lines) == 50
        assert lines[0] == "L011"  # lines 11-60 are the last 50

    def test_stale_positional_form_exits_nonzero(self, capsys):
        """agent-run logs <name> 200 (old positional) must be rejected non-zero via main()."""
        with pytest.raises(SystemExit) as exc_info:
            agent_run.main(["logs", "myrun", "200"])
        assert exc_info.value.code != 0

    @pytest.mark.parametrize(
        "flag, fill",
        [("tail", b"x"), ("head", b"y")],
    )
    def test_single_line_8mb_log_is_byte_bounded(
        self, isolated_runs_root, monkeypatch, flag, fill
    ):
        """--tail/--head 1 on an 8 MB single-line log must emit bounded output.

        The motivating wtgc5 case: one \\n means one "line", so the line count
        cannot bound output and neither can the reader, which must read the whole
        file to find that line. The byte budget is the only bound, and the
        truncation marker must tell the consumer the output is partial.
        """
        file_size = 8 * 1024 * 1024 + 1  # 8 MB payload + trailing \n
        log_content = fill * (file_size - 1) + b"\n"
        name = f"bigrun_{flag}"
        _make_log_run(isolated_runs_root, name, log_content)
        stdout = _FakeStdout(tty=False)
        monkeypatch.setattr(sys, "stdout", stdout)
        rc = agent_run.cmd_logs(
            argparse.Namespace(
                name=name,
                tail=1 if flag == "tail" else None,
                head=1 if flag == "head" else None,
            )
        )
        assert rc == 0
        out = stdout.buffer.getvalue()
        assert len(out) <= agent_run.LOGS_MAX_TOTAL_BYTES + 100, (
            f"--{flag} output {len(out)} bytes exceeds byte budget"
        )
        assert b"[agent-run: output truncated]" in out, "truncation marker missing"
        assert fill in out, "budgeted output must retain log content"

    def test_tail_stops_reverse_walk_before_start_of_file(self, isolated_runs_root, monkeypatch):
        """_tail_bytes stops the reverse walk once n+1 newlines are seen.

        With 10 000 short lines, --tail 50 must satisfy its newline count within
        the first few 8 KiB blocks read backward from EOF and never reach byte 0.
        This bounds only the many-newline shape; a log with fewer than n+1
        newlines is still read in full, and the byte budget is what bounds
        output there (see test_single_line_8mb_log_tail_1).
        """
        log_content = (b"line\n" * 10000)  # 50 KB, 10 000 newlines
        file_size = len(log_content)
        _make_log_run(isolated_runs_root, "bigrun_tail_reads", log_content)
        read_total = [0]

        original_open = agent_run._watch_open_validated_log

        from contextlib import contextmanager

        class CountingFile:
            def __init__(self, f):
                self._f = f
            def read(self, n=-1):
                data = self._f.read(n)
                read_total[0] += len(data)
                return data
            def seek(self, *a): return self._f.seek(*a)
            def tell(self): return self._f.tell()

        @contextmanager
        def patched_open(path):
            with original_open(path) as f:
                if f is not None:
                    yield CountingFile(f)
                else:
                    yield None

        monkeypatch.setattr(agent_run, "_watch_open_validated_log", patched_open)
        stdout = _FakeStdout(tty=False)
        monkeypatch.setattr(sys, "stdout", stdout)
        agent_run.cmd_logs(argparse.Namespace(name="bigrun_tail_reads", tail=50, head=None))
        assert read_total[0] < file_size, (
            f"read {read_total[0]} bytes, expected < {file_size} (whole file)"
        )

    def test_tail_and_head_return_different_content(self, isolated_runs_root, monkeypatch):
        """--tail N and --head N must return different content on a multi-section log."""
        # Distinct markers at head and tail so swapping the helpers would flip content.
        lines = [b"HEAD-MARKER"] + [b"middle"] * 20 + [b"TAIL-MARKER"]
        log_content = b"\n".join(lines) + b"\n"
        _make_log_run(isolated_runs_root, "content_run", log_content)

        stdout_tail = _FakeStdout(tty=False)
        monkeypatch.setattr(sys, "stdout", stdout_tail)
        agent_run.cmd_logs(argparse.Namespace(name="content_run", tail=3, head=None))
        tail_out = stdout_tail.buffer.getvalue()

        stdout_head = _FakeStdout(tty=False)
        monkeypatch.setattr(sys, "stdout", stdout_head)
        agent_run.cmd_logs(argparse.Namespace(name="content_run", tail=None, head=3))
        head_out = stdout_head.buffer.getvalue()

        assert b"TAIL-MARKER" in tail_out, "--tail must include tail of file"
        assert b"HEAD-MARKER" not in tail_out, "--tail must not include head of file"
        assert b"HEAD-MARKER" in head_out, "--head must include head of file"
        assert b"TAIL-MARKER" not in head_out, "--head must not include tail of file"

    @pytest.mark.parametrize(
        "argv",
        [
            pytest.param(["logs", "run", "--tail", "5", "--head", "3"], id="logs-tail-and-head"),
            pytest.param(["logs", "run", "--tail", "0"], id="logs-tail-zero"),
            pytest.param(["logs", "run", "--head", "0"], id="logs-head-zero"),
            pytest.param(["clean", "run", "--tail", "5", "--head", "3"], id="clean-tail-and-head"),
            pytest.param(["clean", "run", "--tail", "0"], id="clean-tail-zero"),
        ],
    )
    def test_rejected_slice_argv_exits_2(self, argv):
        """Mutually exclusive --tail/--head and non-positive N are argparse usage errors."""
        with pytest.raises(SystemExit) as exc_info:
            agent_run._build_parser().parse_args(argv)
        assert exc_info.value.code == 2

    def test_ansi_stripped_on_logs_output(self, isolated_runs_root, monkeypatch):
        """ANSI and OSC sequences in the log must be stripped from logs output."""
        ansi_line = b"\x1b[2J\x1b[H\x1b]0;TITLE\x07visible"
        log_content = ansi_line + b"\n"
        _make_log_run(isolated_runs_root, "ansirun", log_content)
        stdout = _FakeStdout(tty=False)
        monkeypatch.setattr(sys, "stdout", stdout)
        rc = agent_run.cmd_logs(argparse.Namespace(name="ansirun", tail=1, head=None))
        assert rc == 0
        out = stdout.buffer.getvalue()
        assert b"\x1b" not in out, "ANSI escape sequences must be stripped"
        assert b"visible" in out, "visible content must be preserved"
    def test_osc_fragments_are_removed_from_logs_output(self, isolated_runs_root, monkeypatch):
        log_content = b"\x1b]0;hidden\nmetadata\x07visible\n"
        _make_log_run(isolated_runs_root, "oscfrag", log_content)
        stdout = _FakeStdout(tty=False)
        monkeypatch.setattr(sys, "stdout", stdout)
        assert agent_run.cmd_logs(argparse.Namespace(name="oscfrag", tail=2, head=None)) == 0
        out = stdout.buffer.getvalue()
        assert b"hidden" not in out
        assert b"metadata" not in out
        assert b"visible" in out
        assert b"hidden" not in agent_run._strip_ansi_bytes(b"\x1b]0;hidden")


class TestCleanSlicing:
    def test_tail_n_bounds_stdout(self, isolated_runs_root, moon_log_bytes, capsys):
        _make_run(isolated_runs_root, "moon_tail", moon_log_bytes)
        args = argparse.Namespace(
            name="moon_tail", out=None, width=120, height=60, history=100000, tail=3, head=None
        )
        rc = agent_run.cmd_clean(args)
        assert rc == 0
        out = capsys.readouterr().out
        assert len(out.splitlines()) == 3

    def test_head_n_bounds_stdout(self, isolated_runs_root, moon_log_bytes, capsys):
        _make_run(isolated_runs_root, "moon_head", moon_log_bytes)
        args = argparse.Namespace(
            name="moon_head", out=None, width=120, height=60, history=100000, tail=None, head=3
        )
        rc = agent_run.cmd_clean(args)
        assert rc == 0
        out = capsys.readouterr().out
        assert len(out.splitlines()) == 3

    def test_tail_n_bounds_file_out(self, isolated_runs_root, moon_log_bytes, tmp_path, capsys):
        _make_run(isolated_runs_root, "moon_tail_file", moon_log_bytes)
        out_file = tmp_path / "sliced.txt"
        args = argparse.Namespace(
            name="moon_tail_file",
            out=str(out_file),
            width=120,
            height=60,
            history=100000,
            tail=3,
            head=None,
        )
        rc = agent_run.cmd_clean(args)
        assert rc == 0
        text = out_file.read_text(encoding="utf-8")
        assert len(text.splitlines()) == 3
        # Byte count on stderr matches the bytes actually written.
        stderr = capsys.readouterr().err
        written = len(text.encode("utf-8"))
        assert str(written) in stderr

    def test_no_flag_is_byte_identical_to_unsliced(
        self, isolated_runs_root, moon_log_bytes, capsys
    ):
        """clean with neither --tail nor --head must produce the same output as before."""
        _make_run(isolated_runs_root, "moon_unsliced", moon_log_bytes)
        args_sliced = argparse.Namespace(
            name="moon_unsliced", out=None, width=120, height=60, history=100000,
            tail=None, head=None,
        )
        agent_run.cmd_clean(args_sliced)
        unsliced_out = capsys.readouterr().out

        # Render again without the new attributes at all (simulates pre-change callers).
        args_old_style = argparse.Namespace(
            name="moon_unsliced", out=None, width=120, height=60, history=100000
        )
        agent_run.cmd_clean(args_old_style)
        old_out = capsys.readouterr().out

        assert unsliced_out == old_out


class TestCleanByteBudget:
    """cmd_clean applies LOGS_MAX_LINE_BYTES / LOGS_MAX_TOTAL_BYTES to output.

    The rendered transcript can reach hundreds of KB on real runs. Both the
    stdout and -o paths must be bounded identically to cmd_logs, with a
    visible truncation marker when the budget fires.
    """

    def test_stdout_budget_fires_on_oversized_render(
        self, isolated_runs_root, monkeypatch, capsys
    ):
        """A rendered transcript larger than LOGS_MAX_TOTAL_BYTES is truncated
        on the stdout path and the truncation marker is emitted."""
        oversized = "A" * (agent_run.LOGS_MAX_TOTAL_BYTES + 1) + "\n"
        monkeypatch.setattr(agent_run, "_render_log", lambda *_a, **_kw: oversized)
        _make_log_run(isolated_runs_root, "budget_stdout", b"x\n")
        args = argparse.Namespace(
            name="budget_stdout", out=None, width=120, height=60, history=100000,
            tail=None, head=None,
        )
        rc = agent_run.cmd_clean(args)
        assert rc == 0
        out = capsys.readouterr().out
        # Output must be bounded.
        assert len(out.encode("utf-8")) <= agent_run.LOGS_MAX_TOTAL_BYTES + len(
            agent_run._TRUNCATION_MARKER_STR.encode("utf-8")
        ) + 10
        # Truncation marker must be present.
        assert agent_run._TRUNCATION_MARKER_STR in out

    def test_file_out_budget_fires_on_oversized_render(
        self, isolated_runs_root, monkeypatch, tmp_path, capsys
    ):
        """A rendered transcript larger than LOGS_MAX_TOTAL_BYTES is truncated
        on the -o path; the written file includes the truncation marker."""
        oversized = "B" * (agent_run.LOGS_MAX_TOTAL_BYTES + 1) + "\n"
        monkeypatch.setattr(agent_run, "_render_log", lambda *_a, **_kw: oversized)
        _make_log_run(isolated_runs_root, "budget_fileout", b"x\n")
        out_file = tmp_path / "budget.txt"
        args = argparse.Namespace(
            name="budget_fileout",
            out=str(out_file),
            width=120,
            height=60,
            history=100000,
            tail=None,
            head=None,
        )
        rc = agent_run.cmd_clean(args)
        assert rc == 0
        text = out_file.read_text(encoding="utf-8")
        data = text.encode("utf-8")
        assert len(data) <= agent_run.LOGS_MAX_TOTAL_BYTES + len(
            agent_run._TRUNCATION_MARKER_STR.encode("utf-8")
        ) + 10
        assert agent_run._TRUNCATION_MARKER_STR in text
        # Byte count in stderr message must match the file size.
        stderr = capsys.readouterr().err
        assert str(len(data)) in stderr

    def test_small_render_passes_through_untruncated(
        self, isolated_runs_root, monkeypatch, capsys
    ):
        """A rendered transcript well within budget emits no truncation marker."""
        small = "line one\nline two\nline three\n"
        monkeypatch.setattr(agent_run, "_render_log", lambda *_a, **_kw: small)
        _make_log_run(isolated_runs_root, "budget_small", b"x\n")
        args = argparse.Namespace(
            name="budget_small", out=None, width=120, height=60, history=100000,
            tail=None, head=None,
        )
        rc = agent_run.cmd_clean(args)
        assert rc == 0
        out = capsys.readouterr().out
        assert out == small
        assert agent_run._TRUNCATION_MARKER_STR not in out


class TestCleanCache:
    """cmd_clean reads log.clean when metadata proves it covers the raw log.

    Reuse requires a geometry match (width/height/history equal the daemon's
    defaults) and metadata whose dev/ino/offset/size match the current raw log;
    log.clean mtime does not enter the decision. The fallback to raw rendering is
    still load-bearing for older run directories that predate the always-on renderer.
    """

    def _make_cached_run(
        self,
        runs_root,
        name: str,
        clean_text,
        log_mtime: float,
        clean_mtime: float,
    ):
        """Seed a run with log, and log.clean unless clean_text is None, at set mtimes."""
        log_d = _make_log_run(runs_root, name, b"raw content\n")
        os.utime(log_d / "log", (log_mtime, log_mtime))
        if clean_text is not None:
            clean_path = log_d / "log.clean"
            clean_path.write_text(clean_text, encoding="utf-8")
            os.utime(clean_path, (clean_mtime, clean_mtime))
            log_stat = (log_d / "log").stat()
            (log_d / "log.clean.meta.json").write_text(json.dumps({
                "version": 1, "dev": log_stat.st_dev, "ino": log_stat.st_ino,
                "offset": log_stat.st_size, "size": log_stat.st_size, "complete": True,
                "width": agent_run._RENDER_LOG_DEFAULT_WIDTH,
                "height": agent_run._RENDER_LOG_DEFAULT_HEIGHT,
                "history": agent_run._RENDER_LOG_DEFAULT_HISTORY,
                "updated_at": 0,
            }))
        return log_d

    @pytest.mark.parametrize(
        "case, clean_text, gap, width_delta, expect_render",
        [
            pytest.param("hit", "cached content\n", 0.0, 0, False, id="complete-cache-reused"),
            pytest.param("absent", None, 0.0, 0, True, id="no-cache-renders"),
            pytest.param("stale", "cached content\n", 3600.0, 0, False, id="mtime-does-not-control-cache"),
            pytest.param("geometry", "cached content\n", 0.0, 1, True, id="non-default-geometry-renders"),
        ],
    )
    def test_cache_reuse_decision(
        self, isolated_runs_root, monkeypatch, capsys,
        case, clean_text, gap, width_delta, expect_render,
    ):
        """log.clean is reused only when metadata covers the raw log and geometry
        matches.

        The ``stale`` case gives log.clean an mtime an hour behind log to prove
        mtime is not part of the decision; a non-default geometry means the
        cached render does not match what was asked for, forcing a raw render.
        """
        now = 1_000_000.0

        render_called = []

        def fake_render(*_a, **_kw):
            render_called.append(1)
            return "freshly rendered\n"

        monkeypatch.setattr(agent_run, "_render_log", fake_render)
        self._make_cached_run(
            isolated_runs_root, f"cache_{case}",
            clean_text=clean_text,
            log_mtime=now,
            clean_mtime=now - gap,
        )
        args = argparse.Namespace(
            name=f"cache_{case}", out=None,
            width=agent_run._RENDER_LOG_DEFAULT_WIDTH + width_delta,
            height=agent_run._RENDER_LOG_DEFAULT_HEIGHT,
            history=agent_run._RENDER_LOG_DEFAULT_HISTORY,
            tail=None, head=None,
        )

        assert agent_run.cmd_clean(args) == 0
        out = capsys.readouterr().out
        if expect_render:
            assert render_called, "expected a raw render, but log.clean was reused"
            assert "freshly rendered" in out
            assert "cached content" not in out
        else:
            assert not render_called, "expected log.clean reuse, but _render_log ran"
            assert "cached content" in out
