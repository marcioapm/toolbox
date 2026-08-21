"""Tests for `agent-run logs`'s byte budgeting and raw-mode byte fidelity:
`_tail_bytes`/`_head_bytes` keep_delimiters, `_budget_bytes_lines`,
`_budget_str`, and `_close_unterminated_string_control`."""
from __future__ import annotations

import argparse
import io

import pytest

from toolbox import agent_run
from toolbox.agent_run import (
    LOGS_MAX_LINE_BYTES,
    LOGS_MAX_TOTAL_BYTES,
    _budget_bytes_lines,
    _budget_str,
    _close_unterminated_string_control,
    _head_bytes,
    _tail_bytes,
)


class TestTailHeadKeepDelimiters:
    """With keep_delimiters=True, concatenating the returned segments must
    reproduce the corresponding slice of the source bytes exactly."""

    @pytest.mark.parametrize(
        "data",
        [
            b"a\nb\nc\n",
            b"a\nb\nc",  # unterminated final line
            b"a\r\nb\r\nc\r\n",  # CRLF, must be preserved (not stripped)
            b"only one line, no newline at all",
            b"",
        ],
    )
    def test_tail_keep_delimiters_round_trips_whole_file(self, data):
        assert b"".join(_tail_bytes(io.BytesIO(data), 1000, keep_delimiters=True)) == data

    @pytest.mark.parametrize(
        "data",
        [
            b"a\nb\nc\n",
            b"a\nb\nc",
            b"a\r\nb\r\nc\r\n",
            b"only one line, no newline at all",
            b"",
        ],
    )
    def test_head_keep_delimiters_round_trips_whole_file(self, data):
        assert b"".join(_head_bytes(io.BytesIO(data), 1000, keep_delimiters=True)) == data

    def test_tail_keep_delimiters_preserves_mid_line_cr(self):
        data = b"progress\rprogress2\rdone\n"
        assert _tail_bytes(io.BytesIO(data), 1, keep_delimiters=True) == [data]

    def test_head_keep_delimiters_preserves_mid_line_cr(self):
        data = b"progress\rprogress2\rdone\n"
        assert _head_bytes(io.BytesIO(data), 1, keep_delimiters=True) == [data]

    def test_tail_default_still_strips_trailing_cr(self):
        data = b"a\r\nb\r\n"
        assert _tail_bytes(io.BytesIO(data), 10) == [b"a", b"b"]

    def test_head_default_still_strips_trailing_cr(self):
        data = b"a\r\nb\r\n"
        assert _head_bytes(io.BytesIO(data), 10) == [b"a", b"b"]

    def test_tail_keep_delimiters_selects_last_n_lines(self):
        data = b"a\nb\nc\nd\n"
        assert _tail_bytes(io.BytesIO(data), 2, keep_delimiters=True) == [b"c\n", b"d\n"]

    def test_head_keep_delimiters_selects_first_n_lines(self):
        data = b"a\nb\nc\nd\n"
        assert _head_bytes(io.BytesIO(data), 2, keep_delimiters=True) == [b"a\n", b"b\n"]


class TestCloseUnterminatedStringControl:
    def test_open_osc_gets_st_appended(self):
        data = b"\x1b]0;title-in-progress"
        out = _close_unterminated_string_control(data)
        assert out == data + b"\x1b\\"

    def test_terminated_osc_is_left_alone(self):
        data = b"\x1b]0;title\x07visible text"
        assert _close_unterminated_string_control(data) == data

    def test_terminated_with_st_is_left_alone(self):
        data = b"\x1b]0;title\x1b\\visible text"
        assert _close_unterminated_string_control(data) == data

    def test_no_control_sequence_is_left_alone(self):
        assert _close_unterminated_string_control(b"plain text") == b"plain text"

    @pytest.mark.parametrize("introducer", [b"\x1b]", b"\x1bP", b"\x1b^", b"\x1b_"])
    def test_every_string_control_type_is_closed(self, introducer):
        data = introducer + b"payload with no terminator"
        assert _close_unterminated_string_control(data) == data + b"\x1b\\"


class TestBudgetBytesLinesTotalCap:
    def test_default_mode_output_never_exceeds_total_cap(self):
        lines = [b"x" * LOGS_MAX_LINE_BYTES for _ in range(5)]
        out, truncated = _budget_bytes_lines(lines)
        assert len(out) <= LOGS_MAX_TOTAL_BYTES
        assert truncated

    def test_keep_delimiters_mode_output_never_exceeds_total_cap(self):
        lines = [b"x" * LOGS_MAX_LINE_BYTES + b"\n" for _ in range(5)]
        out, truncated = _budget_bytes_lines(lines, keep_delimiters=True)
        assert len(out) <= LOGS_MAX_TOTAL_BYTES
        assert truncated

    def test_single_overlong_line_content_stays_within_line_cap(self):
        out, truncated = _budget_bytes_lines([b"y" * (LOGS_MAX_LINE_BYTES + 1000)])
        # out is content + one synthesized "\n"; content itself must not
        # exceed LOGS_MAX_LINE_BYTES.
        assert len(out) - 1 == LOGS_MAX_LINE_BYTES
        assert truncated

    def test_exact_fit_is_not_marked_truncated(self):
        # Lines short enough to stay under the per-line cap, sized so content
        # plus one newline each lands exactly at the total cap.
        line = b"z" * 8000
        n = LOGS_MAX_TOTAL_BYTES // (len(line) + 1)
        lines = [line] * n
        remainder = LOGS_MAX_TOTAL_BYTES - n * (len(line) + 1)
        if remainder:
            lines.append(b"w" * (remainder - 1))
        out, truncated = _budget_bytes_lines(lines)
        assert len(out) == LOGS_MAX_TOTAL_BYTES
        assert not truncated


class TestBudgetStrTotalCap:
    def test_output_never_exceeds_total_cap_in_utf8_bytes(self):
        text = ("x" * LOGS_MAX_LINE_BYTES + "\n") * 5
        out, truncated = _budget_str(text)
        assert len(out.encode("utf-8")) <= LOGS_MAX_TOTAL_BYTES
        assert truncated


class TestTruncationInsideOscStringIsClosed:
    """F1: a line cut inside an OSC introducer must not swallow the
    truncation marker as OSC payload."""

    def test_budget_bytes_lines_closes_open_osc_before_truncation_marker(self):
        line = b"\x1b]0;" + b"x" * LOGS_MAX_LINE_BYTES  # cut lands inside the OSC string
        out, truncated = _budget_bytes_lines([line])
        assert truncated
        # The emitted segment must end with a string terminator, not raw
        # payload -- otherwise a real terminal reading `out +
        # TRUNCATION_MARKER_BYTES` would swallow the marker as OSC text.
        content = out.rstrip(b"\n")
        assert content.endswith(b"\x1b\\")

    def test_marker_remains_visible_after_the_appended_terminator(self):
        from toolbox.agent_run import _TRUNCATION_MARKER_BYTES

        line = b"\x1b]0;" + b"x" * LOGS_MAX_LINE_BYTES
        out, truncated = _budget_bytes_lines([line])
        assert truncated
        full = out + _TRUNCATION_MARKER_BYTES
        # Simulate a minimal OSC-aware terminal: everything from the last
        # unterminated introducer to the next ST/BEL is swallowed. The
        # appended ST must close the string before the marker starts, so the
        # marker text is not consumed as payload.
        marker_start = full.index(_TRUNCATION_MARKER_BYTES)
        preceding = full[:marker_start]
        introducer_pos = preceding.rfind(b"\x1b]")
        terminator_pos = preceding.find(b"\x1b\\", introducer_pos)
        assert terminator_pos != -1 and terminator_pos < marker_start

    def test_raw_mode_keep_delimiters_also_closes_open_osc(self):
        line = b"\x1b]0;" + b"x" * LOGS_MAX_LINE_BYTES + b"\n"
        out, truncated = _budget_bytes_lines([line], keep_delimiters=True)
        assert truncated
        assert out.rstrip(b"\n").endswith(b"\x1b\\")


class TestCmdLogsRawModeByteFidelity:
    """`agent-run logs` default (raw) mode must reproduce the underlying log
    bytes exactly, including mid-line `\\r` redraws and CRLF."""

    def _run(self, log_root, name, data, capsysbinary):
        log_dir = log_root / name
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "log").write_bytes(data)
        rc = agent_run.cmd_logs(
            argparse.Namespace(name=name, tail=None, head=None, plain=False, clean=False)
        )
        assert rc == 0
        return capsysbinary.readouterr().out

    def test_mid_line_cr_and_trailing_newline_preserved(
        self, isolated_runs_root, isolated_log_root, capsysbinary
    ):
        data = b"progress\rprogress2\rprogress3\ndone\n"
        out = self._run(isolated_log_root, "rawfidelity", data, capsysbinary)
        assert out == data

    def test_unterminated_final_line_preserved(
        self, isolated_runs_root, isolated_log_root, capsysbinary
    ):
        data = b"line one\nline two, no trailing newline"
        out = self._run(isolated_log_root, "rawfidelity2", data, capsysbinary)
        assert out == data


class TestCmdLogsCleanSurvivesCorruptResizeTimeline:
    """F2: invalid UTF-8 in resizes.jsonl must not raise out of `logs
    --clean` -- _read_resize_timeline degrades to an empty timeline and the
    render falls back to the run's resolved default geometry."""

    def test_corrupt_resizes_jsonl_yields_empty_timeline_and_clean_succeeds(
        self, isolated_runs_root, isolated_log_root, capsysbinary
    ):
        log_dir = isolated_log_root / "cleancorrupt"
        log_dir.mkdir(parents=True)
        (log_dir / "log").write_bytes(b"hello world\r\n")
        (log_dir / "resizes.jsonl").write_bytes(b"\xff\xfe")

        assert agent_run._read_resize_timeline(log_dir) == []

        rc = agent_run.cmd_logs(
            argparse.Namespace(name="cleancorrupt", tail=None, head=None, plain=False, clean=True)
        )
        assert rc == 0
        out = capsysbinary.readouterr().out
        assert b"hello world" in out


class TestCmdLogsPlainByteIdentity:
    """F6/F7 regression: `--plain` must stay byte-identical to the
    pre-existing (753924d) behavior -- ANSI-stripped, `\\r`-stripped lines,
    one `\\n` per emitted line, content capped at LOGS_MAX_TOTAL_BYTES."""

    def test_plain_output_matches_strip_then_budget_by_hand(
        self, isolated_runs_root, isolated_log_root, capsysbinary
    ):
        data = (
            b"progress\rdone\x1b[31mred\x1b[0mtext\n"
            b"line two\r\n"
            b"line three, no trailing newline"
        )
        log_dir = isolated_log_root / "plainfidelity"
        log_dir.mkdir(parents=True)
        (log_dir / "log").write_bytes(data)

        rc = agent_run.cmd_logs(
            argparse.Namespace(name="plainfidelity", tail=1000, head=None, plain=True, clean=False)
        )
        assert rc == 0
        out = capsysbinary.readouterr().out

        expected_lines = [
            agent_run._strip_ansi_bytes(line)
            for line in _tail_bytes(io.BytesIO(data), 1000)
        ]
        expected, expected_truncated = _budget_bytes_lines(expected_lines)
        assert not expected_truncated
        assert out == expected

    def test_plain_content_never_exceeds_total_cap_on_a_huge_log(
        self, isolated_runs_root, isolated_log_root, capsysbinary
    ):
        from toolbox.agent_run import _TRUNCATION_MARKER_BYTES

        data = (b"x" * 8000 + b"\n") * 10
        log_dir = isolated_log_root / "plaincap"
        log_dir.mkdir(parents=True)
        (log_dir / "log").write_bytes(data)

        rc = agent_run.cmd_logs(
            argparse.Namespace(name="plaincap", tail=1000, head=None, plain=True, clean=False)
        )
        assert rc == 0
        out = capsysbinary.readouterr().out
        assert out.endswith(_TRUNCATION_MARKER_BYTES)
        content = out[: -len(_TRUNCATION_MARKER_BYTES)]
        assert len(content) == LOGS_MAX_TOTAL_BYTES
