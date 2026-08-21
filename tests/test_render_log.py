"""Tests for toolbox.agent_run._render_log (the PTY -> readable transcript
renderer used by `agent-run logs --clean`)."""
from __future__ import annotations

import re

import pytest

from toolbox import agent_run
from toolbox.agent_run import (
    _render_log,
    _serialize_screen,
)


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

    def test_bare_lf_does_not_staircase(self):
        """A one-shot run writes bare LF (no PTY, no ONLCR translation).
        Without LNM enabled on the screen, each bare LF advances the cursor
        row but leaves the column unchanged, so every subsequent line is
        indented by the character count of the previous line.  LNM must be
        set so bare LF resets the cursor to column 0."""
        # Three lines of different lengths to make the staircase visible.
        raw = (
            b"\x1b[90mprefix\x1b[0m short\n"
            b"\x1b[90mprefix\x1b[0m much-longer-line-here\n"
            b"\x1b[90mprefix\x1b[0m end\n"
        )
        out = _render_log(raw)
        lines = [l for l in out.splitlines() if l.strip()]
        assert lines, "rendered output must not be empty"
        staircase = [l for l in lines if l != l.lstrip()]
        assert not staircase, (
            f"bare-LF staircase: {staircase}; full output: {out!r}"
        )
        # Visible content must be preserved.
        assert "short" in out
        assert "much-longer-line-here" in out
        assert "end" in out

    def test_bare_lf_and_crlf_render_identically(self):
        """LNM on means bare LF and CRLF produce the same rendered output."""
        content = b"line-one content\nline-two content\nline-three content\n"
        crlf_content = content.replace(b"\n", b"\r\n")
        assert _render_log(content) == _render_log(crlf_content)

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


# ---------------------------------------------------------------------------
# Geometry-aware replay — resizes are out-of-band (TIOCSWINSZ + SIGWINCH) and
# never appear in the byte stream, so _render_log must apply a recorded
# offset/cols/rows timeline by resizing the emulated screen mid-replay.
# ---------------------------------------------------------------------------

class TestRenderLogResizeTimeline:
    def test_timeline_splits_replay_and_matches_manual_resize(self):
        import pyte

        raw = (
            b"x" * 60 + b"\r\n"
            + b"y" * 60 + b"\r\n"
            + b"z" * 60 + b"\r\n"
        )
        split_a = len(b"x" * 60 + b"\r\n")
        split_b = split_a + len(b"y" * 60 + b"\r\n")
        resizes = [
            {"offset": split_a, "cols": 40, "rows": 20},
            {"offset": split_b, "cols": 100, "rows": 30},
        ]

        out = _render_log(raw, resizes=resizes)

        screen = agent_run._new_pyte_screen(
            pyte, agent_run._RENDER_LOG_DEFAULT_WIDTH,
            agent_run._RENDER_LOG_DEFAULT_HEIGHT, agent_run._RENDER_LOG_DEFAULT_HISTORY,
        )
        stream = pyte.ByteStream(screen)
        agent_run._feed_pyte(stream, raw[:split_a])
        screen.resize(20, 40)
        agent_run._feed_pyte(stream, raw[split_a:split_b])
        screen.resize(30, 100)
        agent_run._feed_pyte(stream, raw[split_b:])
        expected = _serialize_screen(screen)

        assert out == expected

    def test_pyte_resize_argument_order_is_lines_then_columns(self):
        """Regression pin: pyte.Screen.resize(lines, columns) -- the reverse
        of this module's own (cols, rows) convention. Getting the order
        backwards silently swaps width and height instead of raising."""
        raw = b"x" * 60 + b"\r\n" + b"after resize\r\n"
        split = len(b"x" * 60 + b"\r\n")
        # A narrow-but-tall resize (10 cols, 40 rows): if _render_log passed
        # (cols, rows) into pyte's (lines, columns) signature, the emulator
        # would end up 40 columns wide, so this 12-char line would survive
        # intact instead of being cut to 10 characters.
        out = _render_log(raw, resizes=[{"offset": split, "cols": 10, "rows": 40}])
        assert "after resize" not in out
        assert "after re" in out

    @pytest.mark.parametrize(
        "bad_record",
        [
            pytest.param("not-a-dict", id="not-a-mapping"),
            pytest.param({"cols": 80, "rows": 24}, id="missing-offset"),
            pytest.param({"offset": 3, "rows": 24}, id="missing-cols"),
            pytest.param({"offset": 3, "cols": 80}, id="missing-rows"),
            pytest.param({"offset": "3", "cols": 80, "rows": 24}, id="non-integer-offset"),
            pytest.param({"offset": 3, "cols": "80", "rows": 24}, id="non-integer-cols"),
            pytest.param({"offset": 3, "cols": 80, "rows": 24.5}, id="non-integer-rows"),
            pytest.param({"offset": 3, "cols": 0, "rows": 24}, id="cols-zero"),
            pytest.param({"offset": 3, "cols": 80, "rows": 0}, id="rows-zero"),
            pytest.param(
                {"offset": 3, "cols": 80, "rows": 3000}, id="rows-out-of-range"
            ),
            pytest.param(
                {"offset": 3, "cols": 3000, "rows": 24}, id="cols-out-of-range"
            ),
            pytest.param({"offset": -1, "cols": 80, "rows": 24}, id="negative-offset"),
            pytest.param({"offset": 999999, "cols": 80, "rows": 24}, id="offset-past-eof"),
            pytest.param({"offset": 3, "cols": True, "rows": 24}, id="bool-cols"),
            pytest.param({"offset": 3, "cols": 80, "rows": True}, id="bool-rows"),
            pytest.param({"offset": True, "cols": 80, "rows": 24}, id="bool-offset"),
        ],
    )
    def test_malformed_or_out_of_range_record_is_skipped_without_raising(self, bad_record):
        raw = b"hello world\r\n"
        # A single bad record must simply be dropped, leaving the render
        # identical to no timeline at all -- never an exception.
        out = _render_log(raw, resizes=[bad_record])
        assert out == _render_log(raw)

    def test_backwards_offset_is_skipped(self):
        raw = b"a" * 20 + b"\r\n" + b"b" * 20 + b"\r\n"
        first_split = len(b"a" * 20 + b"\r\n")
        resizes = [
            {"offset": first_split, "cols": 40, "rows": 20},
            # An offset before the last accepted one cannot describe a segment
            # boundary in a forward replay.
            {"offset": 1, "cols": 10, "rows": 5},
        ]
        out = _render_log(raw, resizes=resizes)
        expected = _render_log(raw, resizes=[{"offset": first_split, "cols": 40, "rows": 20}])
        assert out == expected

    def test_absent_timeline_reproduces_current_render(self, moon_log_bytes):
        assert _render_log(moon_log_bytes, resizes=None) == _render_log(moon_log_bytes)

    def test_empty_timeline_reproduces_current_render(self, moon_log_bytes):
        assert _render_log(moon_log_bytes, resizes=[]) == _render_log(moon_log_bytes)

    def test_validate_resize_timeline_helper_directly(self):
        raw_len = 100
        resizes = [
            {"offset": 10, "cols": 80, "rows": 24},
            {"offset": 5, "cols": 80, "rows": 24},  # non-monotonic, dropped
            {"offset": 200, "cols": 80, "rows": 24},  # past EOF, dropped
            "garbage",  # malformed shape, dropped
            {"offset": 50, "cols": 80, "rows": 24},
        ]
        assert agent_run._validate_resize_timeline(resizes, raw_len) == [
            (10, 80, 24),
            (50, 80, 24),
        ]


# ---------------------------------------------------------------------------
# Incremental pyte feed — every partition must produce the same transcript as
# a whole-file feed under the renderer's shared screen semantics.
# ---------------------------------------------------------------------------

class TestIncrementalFeed:
    @pytest.mark.parametrize(
        "raw",
        [
            b"A\x01BBBB\r\n",
            "A\ufe0fBBBB\r\n".encode(),
            "A\u200bBBBB\r\n".encode(),
            b"A\x85BBBB\r\n",
            b"A\x1b[31mB\x1b[0m\r\n",
        ],
    )
    def test_every_split_matches_whole_render(self, raw):
        import pyte

        whole = _render_log(raw)
        for cut in range(len(raw) + 1):
            screen = agent_run._new_pyte_screen(pyte, 120, 60, 100000)
            stream = pyte.ByteStream(screen)
            agent_run._feed_pyte(stream, raw[:cut])
            agent_run._feed_pyte(stream, raw[cut:])
            assert _serialize_screen(screen) == whole, (raw, cut)

    def test_serialize_screen_matches_render_log(self, moon_log_bytes):
        import pyte

        screen = agent_run._new_pyte_screen(pyte, 120, 60, 100000)
        stream = pyte.ByteStream(screen)
        agent_run._feed_pyte(stream, moon_log_bytes)
        assert _serialize_screen(screen) == _render_log(moon_log_bytes)


# ---------------------------------------------------------------------------
# RecursionError hardening — pyte's coroutine-based FSM can recurse past the
# interpreter's limit on certain pathological ANSI/Ink redraw patterns.
# _render_log must degrade to a plain-text ANSI-stripped rendering instead of
# propagating the crash — the clean transcript is a convenience artifact, never
# a run-ending hazard. The exact recursing byte pattern is not portably
# reproducible, so the failure is simulated at the ByteStream.feed seam.
# ---------------------------------------------------------------------------

class TestRenderLogRecursionHardening:
    @pytest.mark.parametrize("exception", [KeyboardInterrupt(), SystemExit(7)])
    def test_process_control_exceptions_propagate(self, monkeypatch, exception):
        import pyte

        def _boom(self, data):
            raise exception

        monkeypatch.setattr(pyte.ByteStream, "feed", _boom)

        with pytest.raises(type(exception)):
            _render_log(b"control exception\r\n")

    def test_recursion_error_falls_back_to_plain_text(self, monkeypatch):
        import pyte

        def _boom(self, data):
            raise RecursionError("maximum recursion depth exceeded")

        monkeypatch.setattr(pyte.ByteStream, "feed", _boom)
        raw = b"\x1b[1;31mHELLO\x1b[0m world\r\n"
        out = _render_log(raw)
        assert isinstance(out, str)
        # Fallback still strips the escape codes and keeps the visible text.
        assert "HELLO world" in out
        assert "\x1b" not in out

    def test_other_pyte_exceptions_also_fall_back(self, monkeypatch):
        """Not just RecursionError — any pyte-internal failure must degrade
        gracefully rather than crash the run."""
        import pyte

        def _boom(self, data):
            raise ValueError("simulated pyte internal failure")

        monkeypatch.setattr(pyte.ByteStream, "feed", _boom)
        raw = b"plain text, no escapes\r\n"
        out = _render_log(raw)
        assert "plain text, no escapes" in out

    def test_fallback_dedupes_and_terminates_with_newline(self, monkeypatch):
        import pyte

        def _boom(self, data):
            raise RecursionError("maximum recursion depth exceeded")

        monkeypatch.setattr(pyte.ByteStream, "feed", _boom)
        raw = (b"loading\r\n" * 6) + b"done\r\n"
        out = _render_log(raw)
        assert out.endswith("\n")
        assert out.count("loading") < 6
        assert "done" in out

    def test_deeply_nested_pathological_escape_blob_does_not_raise(self):
        """A known pathological pattern: thousands of chained cursor-motion
        and combining-character escape sequences with no plain-text runs in
        between, modeling the kind of Ink redraw storm that triggered the
        production RecursionErrors. Whether or not this specific blob
        actually recurses pyte's FSM on the installed pyte version, the
        contract holds either way: _render_log must return a string, never
        raise."""
        blob = (b"\x1b[1C\x1b[1D\x1b[s\x1b[u" * 50_000) + b"survived\r\n"
        out = _render_log(blob)
        assert isinstance(out, str)
        assert "survived" in out


class TestResizeCursorClamping:
    """pyte's Screen.resize clips cells but leaves the cursor where it was, so
    a resize during replay must clamp it back into the new screen."""

    @staticmethod
    def _render_at(width, height, payload):
        return agent_run._render_log(payload, width=width, height=height)

    def test_cursor_past_new_right_edge_keeps_drawing_on_screen(self):
        """A cursor left beyond the new width would otherwise write into
        columns _serialize_screen never emits, silently dropping output."""
        rendered = agent_run._render_log(
            b"1234567890X\r\n", width=10, height=4,
            resizes=[{"offset": 10, "cols": 5, "rows": 4}],
        )
        assert rendered == self._render_at(5, 4, b"12345\r\nX\r\n")

    def test_cursor_below_new_bottom_keeps_drawing_on_screen(self):
        head = b"".join(f"line{i}\r\n".encode() for i in range(8))
        rendered = agent_run._render_log(
            head + b"AFTER\r\n", width=20, height=10,
            resizes=[{"offset": len(head), "cols": 20, "rows": 3}],
        )
        assert "AFTER" in rendered

    def test_growing_does_not_move_a_cursor_already_in_range(self):
        rendered = agent_run._render_log(
            b"abcDEF\r\n", width=10, height=4,
            resizes=[{"offset": 3, "cols": 40, "rows": 12}],
        )
        assert rendered == "abcDEF\n"

    def test_pending_wrap_survives_a_resize(self):
        """cursor.x == columns is DECAWM's pending-wrap state, not an
        out-of-range cursor, so clamping must not pull it back a column."""
        pyte = pytest.importorskip("pyte")
        screen = agent_run._new_pyte_screen(pyte, 5, 4, 2048)
        agent_run._feed_pyte(pyte.ByteStream(screen), b"12345")
        assert screen.cursor.x == screen.columns
        agent_run._resize_screen(screen, 5, 4)
        assert screen.cursor.x == screen.columns

    def test_height_shrink_drops_clipped_rows_rather_than_scrolling_them(self):
        """Documents pyte's model, which a terminal does not share: rows
        clipped by a height reduction are deleted, not moved into
        history.top, so replay across a shrink loses them."""
        head = b"".join(f"line{i}\r\n".encode() for i in range(8))
        rendered = agent_run._render_log(
            head + b"AFTER\r\n", width=20, height=10,
            resizes=[{"offset": len(head), "cols": 20, "rows": 3}],
        )
        assert "line0" not in rendered
        assert rendered != self._render_at(20, 3, head + b"AFTER\r\n")


class TestResizeTimelineCoalescing:
    def test_records_sharing_an_offset_coalesce_to_the_last(self):
        """TIOCSWINSZ is last-writer-wins, and several resizes can be applied
        with no log output between them (attach startup then detach restore,
        two clients, a drag burst)."""
        timeline = agent_run._validate_resize_timeline(
            [{"offset": 0, "cols": 5, "rows": 5},
             {"offset": 0, "cols": 20, "rows": 5}], 30,
        )
        assert timeline == [(0, 20, 5)]

    def test_a_coalesced_record_still_rejects_an_earlier_offset(self):
        timeline = agent_run._validate_resize_timeline(
            [{"offset": 10, "cols": 5, "rows": 5},
             {"offset": 3, "cols": 20, "rows": 5}], 30,
        )
        assert timeline == [(10, 5, 5)]

    def test_an_invalid_record_does_not_replace_a_valid_one_at_the_offset(self):
        timeline = agent_run._validate_resize_timeline(
            [{"offset": 4, "cols": 8, "rows": 6},
             {"offset": 4, "cols": 0, "rows": 6}], 30,
        )
        assert timeline == [(4, 8, 6)]


class TestSerializeScreenExtraction:
    """_screen_lines was extracted from _serialize_screen. _serialize_screen
    feeds the rendered transcript, so a behaviour change here corrupts it."""

    @staticmethod
    def _reference(screen):
        """The algorithm as it stood before the extraction."""
        rows = []
        for entry in screen.history.top:
            rows.append(
                ("".join(entry[col].data for col in sorted(entry)) if entry else "").rstrip()
            )
        for row in screen.display:
            rows.append(row.rstrip())
        deduped = []
        for line in rows:
            if not deduped or deduped[-1] != line:
                deduped.append(line)
        while deduped and not deduped[-1]:
            deduped.pop()
        return "\n".join(deduped) + "\n"

    @pytest.mark.parametrize("raw", [
        b"",
        b"plain\r\n",
        b"   \r\n\r\n  x  \r\n",
        "wide \u5e7f\u5473 chars\r\n".encode(),
        "combining a\u0301e\u0301\r\n".encode(),
        b"".join(b"scroll %d\r\n" % i for i in range(300)),
        b"\x1b[?1049h\x1b[2J\x1b[1;1Hframe\r\n",
    ])
    def test_matches_the_pre_extraction_algorithm(self, raw):
        pyte = pytest.importorskip("pyte")
        screen = agent_run._new_pyte_screen(pyte, 40, 8, 64)
        agent_run._feed_pyte(pyte.ByteStream(screen), raw)
        assert agent_run._serialize_screen(screen) == self._reference(screen)
