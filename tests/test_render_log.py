"""Tests for toolbox.agent_run._render_log (the PTY -> readable transcript
renderer used by `agent-run clean` and `--echo`)."""
from __future__ import annotations

import re

import pytest

from toolbox.agent_run import _render_log


# ---------------------------------------------------------------------------
# Synthetic byte streams — exercise the cleaner in isolation.
# ---------------------------------------------------------------------------

class TestRenderLogSynthetic:
    def test_empty_input_returns_empty_or_minimal(self):
        out = _render_log(b"")
        # An empty byte stream may still produce a single newline; either is fine.
        assert out.strip() == ""

    def test_plain_ascii_passes_through(self):
        out = _render_log(b"hello world\r\n")
        assert "hello world" in out

    def test_strips_csi_ansi_codes(self):
        # Bold red text, then reset
        raw = b"\x1b[1;31mHELLO\x1b[0m world\r\n"
        out = _render_log(raw)
        assert "HELLO world" in out
        # The literal escape introducer must not survive.
        assert "\x1b[" not in out
        assert "[1;31m" not in out

    def test_strips_osc_title_sequences(self):
        # OSC 0 (set window title), BEL-terminated
        raw = b"\x1b]0;My Title\x07visible\r\n"
        out = _render_log(raw)
        assert "visible" in out
        assert "My Title" not in out  # window title is metadata, not content
        assert "\x1b]" not in out

    def test_ink_cursor_right_yields_spaces_between_words(self):
        """Claude Code's Ink TUI emits ESC[1C between words instead of typing
        a space. Pyte's terminal model handles that by leaving the cell at
        that column empty (which renders as a space when joined)."""
        # 'foo' + CursorRight(1) + 'bar' rendered into a single row should
        # produce 'foo bar' (a space at column 3 between foo and bar).
        raw = b"foo\x1b[1Cbar\r\n"
        out = _render_log(raw)
        # Either 'foo bar' (with the implicit space) or 'foo' and 'bar'
        # on consecutive cells separated by one column. Accept any rendering
        # that doesn't smush them together.
        assert "foobar" not in out, "cursor-right should not be eaten"
        assert re.search(r"foo\s+bar", out), f"expected 'foo<spaces>bar' in: {out!r}"
    def test_dedupes_adjacent_identical_lines(self):
        # Real PTY output uses CR+LF (\r\n) so the cursor returns to column
        # 0 each line. Without CR, pyte leaves the cursor where it was,
        # producing a 'staircase' of indented lines that look different.
        # Real Claude logs always include CR; use the same here.
        raw = (b"loading\r\n" * 6) + b"done\r\n"
        out = _render_log(raw, history=100)
        # All six 'loading's collapse to one (or at most a few).
        loading_count = out.count("loading")
        assert loading_count < 6, (
            f"expected adjacent-identical dedup; saw {loading_count} 'loading' lines:\n{out}"
        )
        assert "done" in out

    def test_returns_trailing_newline(self):
        out = _render_log(b"hi\r\n")
        assert out.endswith("\n")


# ---------------------------------------------------------------------------
# Real-world fixtures — capture the actual failure modes that motivated this
# whole effort (Ink TUI redraws on Claude Code).
# ---------------------------------------------------------------------------

class TestRenderLogRealClaude:
    def test_moon_paragraph_is_readable(self, moon_log_bytes):
        """The fixture is a real Claude Code -i session answering a prompt
        about the moon mentioning craters and tides. The cleaned output
        must contain that paragraph in continuous, human-readable form."""
        out = _render_log(moon_log_bytes)
        # Key keywords from the actual response.
        assert "craters" in out, "expected the assistant's 'craters' in cleaned output"
        assert "tides" in out, "expected the assistant's 'tides' in cleaned output"
        # The assistant bullet should be present.
        assert "●" in out

    def test_moon_response_words_have_spaces(self, moon_log_bytes):
        """Regression check: Ink emits ESC[1C between every word. The
        cleaned output must put visible whitespace between them, not
        runtogetherslikethis."""
        out = _render_log(moon_log_bytes)
        # 'Earth's only natural satellite' is a phrase from the assistant
        # reply — verify it's spaced correctly.
        assert re.search(r"Earth'?s\s+only\s+natural\s+satellite", out, re.IGNORECASE), (
            f"expected proper word spacing; got snippet around the phrase: "
            f"{[ln for ln in out.splitlines() if 'satellite' in ln.lower()][:3]}"
        )

    def test_no_raw_escape_codes_leak_through(self, moon_log_bytes):
        out = _render_log(moon_log_bytes)
        assert "\x1b" not in out, "raw ESC byte leaked into cleaned output"

    def test_no_carriage_returns_or_bel(self, moon_log_bytes):
        out = _render_log(moon_log_bytes)
        # The renderer should normalise control bytes away.
        assert "\r" not in out
        assert "\x07" not in out

    def test_user_prompt_preserved(self, moon_log_bytes):
        """The user's typed message should also appear in the transcript
        (not just the assistant's reply)."""
        out = _render_log(moon_log_bytes)
        assert "moon" in out.lower()

    def test_plain_print_log_passes_through(self, print_log_bytes):
        """A `claude --print` capture has no ANSI at all — should round-trip."""
        assert b"FIXTURE-WORD-7" in print_log_bytes  # sanity
        out = _render_log(print_log_bytes)
        assert "FIXTURE-WORD-7" in out


# ---------------------------------------------------------------------------
# Sizing knobs — width/history affect long-line wrapping & deep transcripts.
# ---------------------------------------------------------------------------

class TestRenderLogSizing:
    def test_wide_width_keeps_long_lines_intact(self):
        long = b"x" * 200 + b"\r\n"
        out = _render_log(long, width=240)
        # Should appear as one continuous run of x's somewhere.
        assert "x" * 200 in out

    def test_narrow_width_truncates_unwrapped_long_line(self):
        # Pyte by default does NOT autowrap a long unbroken stream past
        # the viewport width — trailing bytes after the last column are
        # dropped. This is acceptable for our use case because real
        # Claude output is always \n-terminated between paragraphs. The
        # test pins down current behavior so a future autowrap toggle
        # would be a deliberate decision, not a silent regression.
        long = b"x" * 200 + b"\r\n"
        out = _render_log(long, width=50)
        # No 200-char run survives at width 50.
        assert "x" * 200 not in out
        # ~50 x's do (the viewport width).
        assert "x" * 50 in out

    def test_wrapped_via_explicit_newlines(self):
        # When the source contains \r\n, lines flow across multiple rows
        # cleanly regardless of width.
        raw = (b"abc\r\n" * 5)
        out = _render_log(raw, width=20)
        assert out.count("abc") >= 1  # dedup may collapse to one
        # No truncation: every emitted 'abc' is intact.
        assert "ab\n" not in out
