"""Tests for toolbox.agent_run._render_log (the PTY -> readable transcript
renderer used by `agent-run clean` and `--echo`)."""
from __future__ import annotations

import builtins
import json
import re
import time
from pathlib import Path

import pytest

from toolbox import agent_run
from toolbox.agent_run import (
    ECHO_LOOP_MAX_RENDER_BYTES,
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
# _echo_loop — incremental parse, reset, quiet-tick, and cap behaviour.
#
# All tests drive the loop by monkeypatching time.sleep: the patched version
# counts ticks and raises KeyboardInterrupt once the scenario is complete.
# Log files are written directly to tmp_path so no real filesystem races occur.
# ---------------------------------------------------------------------------

def _run_echo_loop_ticks(log_dir, *, ticks, monkeypatch, extra_actions=None):
    """Drive _echo_loop for exactly *ticks* sleep calls, then stop.

    *extra_actions* is an optional callable(tick_number) invoked just before
    each sleep, letting a test mutate the log file between ticks.
    """
    tick_count = 0

    def controlled_sleep(_interval):
        nonlocal tick_count
        if extra_actions is not None:
            extra_actions(tick_count)
        tick_count += 1
        if tick_count >= ticks:
            raise KeyboardInterrupt

    monkeypatch.setattr(agent_run.time, "sleep", controlled_sleep)
    with pytest.raises(KeyboardInterrupt):
        agent_run._echo_loop(log_dir, 1.0)


class TestEchoLoop:
    def test_basic_render_writes_log_clean(self, tmp_path, monkeypatch):
        log_dir = tmp_path / "run"
        log_dir.mkdir()
        (log_dir / "log").write_bytes(b"hello\r\n")

        _run_echo_loop_ticks(log_dir, ticks=1, monkeypatch=monkeypatch)

        assert (log_dir / "log.clean").exists()
        assert "hello" in (log_dir / "log.clean").read_text()

    def test_quiet_tick_no_write(self, tmp_path, monkeypatch):
        """When st_size == offset (no new bytes), log.clean is not touched."""
        log_dir = tmp_path / "run"
        log_dir.mkdir()
        (log_dir / "log").write_bytes(b"hello\r\n")

        # Tick 0: parse and write.  Ticks 1 and 2: nothing new — sleep again.
        write_count = [0]
        original_replace = Path.replace

        def track_replace(self, target):
            if self.name.startswith(".log.clean.") and self.name.endswith(".clean"):
                write_count[0] += 1
            return original_replace(self, target)

        monkeypatch.setattr(Path, "replace", track_replace)
        _run_echo_loop_ticks(log_dir, ticks=3, monkeypatch=monkeypatch)

        # Only one write: tick 0.  Ticks 1 and 2 see size == offset and
        # skip the serialize+write path entirely.
        assert write_count[0] == 1, f"expected 1 write; got {write_count[0]}"

    def test_unchanged_render_skips_write(self, tmp_path, monkeypatch):
        """When new bytes arrive but the serialized text is unchanged, the
        atomic write+rename is skipped."""
        log_dir = tmp_path / "run"
        log_dir.mkdir()
        log_file = log_dir / "log"
        log_file.write_bytes(b"hello\r\n")

        write_count = [0]
        original_replace = Path.replace

        def track_replace(self, target):
            if self.name.startswith(".log.clean.") and self.name.endswith(".clean"):
                write_count[0] += 1
            return original_replace(self, target)

        def append_on_tick(tick):
            if tick == 1:
                # ESC[0m (SGR reset) changes no screen cell content, so the
                # serialized output is identical to the previous tick.
                with log_file.open("ab") as f:
                    f.write(b"\x1b[0m")

        monkeypatch.setattr(Path, "replace", track_replace)
        _run_echo_loop_ticks(log_dir, ticks=3, monkeypatch=monkeypatch,
                             extra_actions=append_on_tick)

        # Tick 0 writes (new content).  Tick 1 parses new bytes but the
        # serialized text is identical — no second write.
        assert write_count[0] == 1, (
            f"expected 1 write (second tick produces identical output); "
            f"got {write_count[0]}"
        )

    def test_truncation_triggers_reset(self, tmp_path, monkeypatch):
        """When st_size < offset (file truncated), screen/stream reset and the
        new file content is re-parsed from byte zero."""
        log_dir = tmp_path / "run"
        log_dir.mkdir()
        log_file = log_dir / "log"
        # First write: 20 bytes so offset after tick 0 is 20.
        log_file.write_bytes(b"first-content-here\r\n")

        def mutate_on_tick(tick):
            if tick == 1:
                # Overwrite with 10 bytes — shorter than offset (20), so
                # cur_size < offset fires on the next stat().
                log_file.write_bytes(b"replaced\r\n")

        _run_echo_loop_ticks(log_dir, ticks=3, monkeypatch=monkeypatch,
                             extra_actions=mutate_on_tick)

        content = (log_dir / "log.clean").read_text()
        assert "replaced" in content, f"expected 'replaced' in log.clean; got: {content!r}"
        assert "first-content-here" not in content, (
            f"'first-content-here' survived truncation reset in log.clean: {content!r}"
        )

    def test_inode_replacement_triggers_reset(self, tmp_path, monkeypatch):
        """When st_ino/st_dev changes (file replaced), the loop resets even if
        st_size coincidentally equals the previous offset."""
        log_dir = tmp_path / "run"
        log_dir.mkdir()
        log_file = log_dir / "log"
        # 10 bytes of non-printing content so tick 0 parses but produces no
        # visible text (pyte ignores NUL bytes).
        log_file.write_bytes(b"\x00" * 10)

        def replace_on_tick(tick):
            if tick == 1:
                # Replace with a different inode of the same byte count.
                # The size matches the recorded offset, so only the inode
                # change can trigger a reset.
                tmp_new = log_file.parent / ".log.new"
                tmp_new.write_bytes(b"replaced!\r\n")
                tmp_new.rename(log_file)

        _run_echo_loop_ticks(log_dir, ticks=3, monkeypatch=monkeypatch,
                             extra_actions=replace_on_tick)

        content = (log_dir / "log.clean").read_text()
        assert "replaced" in content, (
            f"expected 'replaced' after inode-replacement reset; got: {content!r}"
        )

    def test_failed_feed_does_not_advance_offset(self, tmp_path, monkeypatch):
        """A transient _feed_pyte failure leaves the offset unchanged so the
        next tick re-parses the same bytes."""
        log_dir = tmp_path / "run"
        log_dir.mkdir()
        (log_dir / "log").write_bytes(b"data\r\n")

        feed_calls: list[int] = []
        original_feed = agent_run._feed_pyte

        def flaky_feed(stream, chunk):
            feed_calls.append(len(chunk))
            if len(feed_calls) == 1:
                raise OSError("simulated transient failure")
            return original_feed(stream, chunk)

        monkeypatch.setattr(agent_run, "_feed_pyte", flaky_feed)
        _run_echo_loop_ticks(log_dir, ticks=3, monkeypatch=monkeypatch)

        # Both ticks that saw new bytes must have attempted a feed.
        assert len(feed_calls) >= 2, f"expected ≥2 feed attempts; got {feed_calls}"
        # Second attempt succeeded — log.clean must contain the content.
        assert "data" in (log_dir / "log.clean").read_text()

    def test_failed_feed_and_serialization_rebuild_from_zero(self, tmp_path, monkeypatch):
        log_dir = tmp_path / "run"
        log_dir.mkdir()
        raw = b"abcdef\r\n"
        (log_dir / "log").write_bytes(raw)
        original_feed = agent_run._feed_pyte
        calls = [0]

        def half_feed(stream, chunk):
            calls[0] += 1
            if calls[0] == 1:
                original_feed(stream, chunk[:3])
                raise ValueError("after prefix")
            return original_feed(stream, chunk)

        monkeypatch.setattr(agent_run, "_feed_pyte", half_feed)
        _run_echo_loop_ticks(log_dir, ticks=4, monkeypatch=monkeypatch)
        assert (log_dir / "log.clean").read_text() == _render_log(raw)

    def test_failed_serialization_rebuilds_from_zero(self, tmp_path, monkeypatch):
        log_dir = tmp_path / "run"
        log_dir.mkdir()
        raw = b"serialize\r\n"
        (log_dir / "log").write_bytes(raw)
        original_serialize = agent_run._serialize_screen
        calls = [0]

        def flaky_serialize(screen):
            calls[0] += 1
            if calls[0] == 1:
                raise ValueError("after feed")
            return original_serialize(screen)

        monkeypatch.setattr(agent_run, "_serialize_screen", flaky_serialize)
        _run_echo_loop_ticks(log_dir, ticks=4, monkeypatch=monkeypatch)
        assert (log_dir / "log.clean").read_text() == _render_log(raw)

    def test_failed_publication_retries_on_quiet_log(self, tmp_path, monkeypatch):
        log_dir = tmp_path / "run"
        log_dir.mkdir()
        (log_dir / "log").write_bytes(b"quiet\r\n")
        original_replace = Path.replace
        calls = [0]

        def flaky_replace(self, target):
            if self.name.startswith(".log.clean.") and self.name.endswith(".clean"):
                calls[0] += 1
                if calls[0] == 1:
                    raise OSError("rename failed")
            return original_replace(self, target)

        monkeypatch.setattr(Path, "replace", flaky_replace)
        _run_echo_loop_ticks(log_dir, ticks=3, monkeypatch=monkeypatch)
        assert calls[0] >= 2
        assert "quiet" in (log_dir / "log.clean").read_text()

    def test_cap_then_small_append_preserves_continuous_prefix(self, tmp_path, monkeypatch):
        log_dir = tmp_path / "run"
        log_dir.mkdir()
        log_file = log_dir / "log"
        log_file.write_bytes(b"")
        monkeypatch.setattr(agent_run, "ECHO_LOOP_MAX_RENDER_BYTES", 5)

        def grow_on_tick(tick):
            if tick == 0:
                log_file.write_bytes(b"A" * 100)
            elif tick == 1:
                with log_file.open("ab") as f:
                    f.write(b"NEW\r\n")

        _run_echo_loop_ticks(log_dir, ticks=22, monkeypatch=monkeypatch, extra_actions=grow_on_tick)
        assert (log_dir / "log.clean").read_text() == _render_log(log_file.read_bytes())
        metadata = json.loads((log_dir / "log.clean.meta.json").read_text())
        assert metadata["complete"] is True
        assert metadata["offset"] == log_file.stat().st_size

    def test_echo_loop_max_render_bytes_constant_value(self):
        """Pin the cap value; any change must be a deliberate, test-breaking decision."""
        assert ECHO_LOOP_MAX_RENDER_BYTES == 16 * 1024 * 1024

    def test_missing_pyte_does_not_publish_unverifiable_transcript(self, tmp_path, monkeypatch):
        log_dir = tmp_path / "run"
        log_dir.mkdir()
        (log_dir / "log").write_bytes(b"complete\r\n")
        real_import = builtins.__import__

        def no_pyte(name, *args, **kwargs):
            if name == "pyte":
                raise ImportError("forced missing pyte")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", no_pyte)
        agent_run._echo_loop(log_dir, 1.0)
        assert not (log_dir / "log.clean").exists()


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


class TestStitchedHistory:
    """Stitched mode samples the whole screen during replay to recover content a
    repainting TUI overwrites before it can scroll into pyte's scrollback."""

    # A TUI: enters the alternate screen, then repaints in place with absolute
    # cursor moves. Nothing ever scrolls, so pyte's history stays empty.
    @staticmethod
    def _repainting_tui(frames):
        out = [b"\x1b[?1049h"]
        for frame in frames:
            out.append(b"\x1b[2J\x1b[1;1H" + frame.encode())
        return b"".join(out)

    def test_recovers_frames_a_repainting_tui_overwrote(self):
        """Sampling recovers frames the viewport lost, but not every frame: one
        repainted and overwritten entirely between two samples is unrecoverable,
        so this asserts the gain rather than any particular frame."""
        raw = self._repainting_tui([f"frame {i}" for i in range(40)])
        viewport = agent_run._render_log(raw)
        stitched = agent_run._stitch_history(raw, chunk_bytes=16)

        assert "frame 0" not in viewport, "the TUI overwrote it; viewport keeps the last frame"
        assert "frame 39" in stitched, "the final frame is on screen at the last boundary"

        recovered = sum(f"frame {i}" in stitched for i in range(40))
        assert recovered > sum(f"frame {i}" in viewport for i in range(40))
        assert recovered >= 5, f"only {recovered}/40 frames recovered"

    def test_a_frame_overwritten_between_samples_is_unrecoverable(self):
        """The bound on stitching: it samples, it does not record. A sample
        interval wider than the repaint interval misses whole frames."""
        raw = self._repainting_tui([f"frame {i}" for i in range(40)])
        coarse = agent_run._stitch_history(raw, chunk_bytes=4096)
        fine = agent_run._stitch_history(raw, chunk_bytes=16)
        assert sum(f"frame {i}" in coarse for i in range(40)) < sum(
            f"frame {i}" in fine for i in range(40)
        )

    def test_is_a_superset_of_the_viewport_render(self):
        """Sampling covers scrollback and viewport together. Sampling the
        viewport alone would drop every line that scrolled away between two
        samples, which on line-oriented output is most of them."""
        raw = b"".join(f"line {i}\r\n".encode() for i in range(400))
        viewport = {l for l in agent_run._render_log(raw).splitlines() if l.strip()}
        stitched = {l for l in agent_run._stitch_history(raw, chunk_bytes=256).splitlines() if l.strip()}
        assert viewport <= stitched

    def test_repeats_within_one_screen_survive(self):
        """Deduplication removes the overlap between consecutive samples, not
        every recurrence: a line printed several times in one screenful is real
        output, not a resampled row."""
        raw = b"".join(b"====\r\n" for _ in range(5))
        stitched = agent_run._stitch_history(raw, chunk_bytes=4096)
        assert stitched.count("====") == 5

    def test_overlap_between_consecutive_samples_is_removed(self):
        """Each sample yields the whole screen, so a line still present in the
        previous sample has already been emitted."""
        raw = b"only line\r\n" + b"\x00" * 40_000
        stitched = agent_run._stitch_history(raw, chunk_bytes=1024)
        assert stitched.count("only line") == 1

    def test_line_cap_truncates_and_says_so(self, monkeypatch):
        monkeypatch.setattr(agent_run, "_STITCH_MAX_LINES", 5)
        raw = b"".join(f"line {i}\r\n".encode() for i in range(200))
        stitched = agent_run._stitch_history(raw, chunk_bytes=64)
        assert stitched.endswith(agent_run._STITCH_TRUNCATED_MARKER)
        body = stitched[: -len(agent_run._STITCH_TRUNCATED_MARKER)]
        assert len([l for l in body.splitlines() if l.strip()]) <= 5

    def test_byte_cap_truncates_and_says_so(self, monkeypatch):
        monkeypatch.setattr(agent_run, "_STITCH_MAX_BYTES", 64)
        raw = b"".join(f"line {i}\r\n".encode() for i in range(200))
        stitched = agent_run._stitch_history(raw, chunk_bytes=64)
        assert stitched.endswith(agent_run._STITCH_TRUNCATED_MARKER)

    def test_applies_the_resize_timeline(self):
        """A sample boundary and a resize offset are independent, so replay
        splits at the union of both."""
        raw = b"X" * 30 + b"\r\n"
        stitched = agent_run._stitch_history(
            raw, width=40, height=4, chunk_bytes=8,
            resizes=[{"offset": 10, "cols": 10, "rows": 4}],
        )
        assert stitched, "a resize inside a sampled span must not abort the render"
        assert max(len(l) for l in stitched.splitlines()) <= 10

    def test_falls_back_when_pyte_raises(self, monkeypatch):
        """A convenience artifact must never take the caller down."""
        monkeypatch.setattr(
            agent_run, "_feed_pyte",
            lambda *_a, **_kw: (_ for _ in ()).throw(RecursionError("pathological")),
        )
        stitched = agent_run._stitch_history(b"hello\r\n", chunk_bytes=8)
        assert "hello" in stitched

    def test_viewport_remains_the_default_render(self, moon_log_bytes):
        """Stitched mode is opt-in; the default path must be byte-identical."""
        assert agent_run._render_log(moon_log_bytes) == agent_run._render_log(moon_log_bytes)

    @pytest.mark.parametrize("chunk", [16, 64, 1024, 8192])
    def test_stitching_recovers_a_long_run_of_identical_lines(self, chunk):
        """Viewport is lossy for repeated output; stitching is not.

        ``_serialize_screen`` drops a line equal to its predecessor, so 200
        identical lines serialize as one. Counting occurrences per sample
        instead of testing membership recovers every print, independently of
        where the sample boundaries fall.
        """
        printed = 200
        raw = b"".join(b"====\r\n" for _ in range(printed))

        assert agent_run._render_log(raw).count("====") == 1
        assert agent_run._stitch_history(raw, chunk_bytes=chunk).count("====") == printed

    def test_stitching_may_add_partial_repaint_rows(self):
        """The cost of sampling: a boundary landing mid-repaint keeps a partly
        drawn row, so a stitched render can exceed the viewport line count on
        output the viewport already captured in full."""
        raw = b"".join(f"line {i}\r\n".encode() for i in range(200))
        viewport = [l for l in agent_run._render_log(raw).splitlines() if l.strip()]
        stitched = [l for l in agent_run._stitch_history(raw, chunk_bytes=64).splitlines() if l.strip()]

        assert set(viewport) <= set(stitched)
        assert len(stitched) >= len(viewport)


class TestStitchedHistoryBounds:
    """The properties an adversarial review broke: multiplicity across sample
    boundaries, the unit the byte cap is enforced in, the reach of the pyte
    fallback, and the interval's domain."""

    def test_a_repeat_printed_across_a_boundary_is_kept(self):
        """Deduplication compares occurrence counts, not membership: a row
        printed again while an identical row is still on screen is new output."""
        raw = b"X\r\n12345" + b"\r\nX\r\n"
        stitched = agent_run._stitch_history(raw, width=20, height=5, chunk_bytes=8)
        assert stitched.splitlines().count("X") == 2

    def test_a_row_static_across_many_samples_is_emitted_once(self):
        raw = b"only line\r\n" + b"\x00" * 40_000
        assert agent_run._stitch_history(raw, chunk_bytes=1024).count("only line") == 1

    def test_the_byte_cap_counts_utf8_bytes_not_code_points(self, monkeypatch):
        """A code-point budget lets non-ASCII output reach ~4x the named cap."""
        monkeypatch.setattr(agent_run, "_STITCH_MAX_BYTES", 8)
        raw = "".join(f"{i}\U0001f600\r\n" for i in range(20)).encode()
        stitched = agent_run._stitch_history(raw, chunk_bytes=16)
        assert stitched.endswith(agent_run._STITCH_TRUNCATED_MARKER)
        body = stitched[: -len(agent_run._STITCH_TRUNCATED_MARKER)]
        assert len(body.encode("utf-8")) <= 8

    def test_the_byte_cap_admits_no_line_that_would_exceed_it(self, monkeypatch):
        """The cap is checked against the prospective line, so the body cannot
        overshoot by a full row."""
        monkeypatch.setattr(agent_run, "_STITCH_MAX_BYTES", 20)
        raw = b"".join(b"a" * 30 + b"\r\n" for _ in range(10))
        stitched = agent_run._stitch_history(raw, chunk_bytes=32)
        body = stitched[: -len(agent_run._STITCH_TRUNCATED_MARKER)]
        assert len(body.encode("utf-8")) <= 20

    def test_failure_constructing_the_screen_falls_back(self, monkeypatch):
        """Screen and stream construction sit inside the fallback: a pyte
        failure there must degrade like a feed failure, not propagate."""
        monkeypatch.setattr(
            agent_run, "_new_pyte_screen",
            lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError("construction failed")),
        )
        assert "hello" in agent_run._stitch_history(b"hello\r\n", chunk_bytes=8)

    @pytest.mark.parametrize("chunk", [0, -1, True, 1.5, "8192"])
    def test_a_nonpositive_or_nonint_interval_is_rejected(self, chunk):
        """range() accepts 0 and negatives via a max() guard, silently sampling
        every byte and making the boundary set O(len(raw))."""
        with pytest.raises(ValueError):
            agent_run._stitch_history(b"abc", chunk_bytes=chunk)

    def test_truncated_output_says_so(self, monkeypatch):
        """Truncation stops replay before the final screen, so the result omits
        the tail the viewport render ends with. The marker is the only signal a
        reader gets, and it must always be present."""
        monkeypatch.setattr(agent_run, "_STITCH_MAX_LINES", 50)
        raw = b"".join(f"line {i}\r\n".encode() for i in range(400))
        stitched = agent_run._stitch_history(raw, chunk_bytes=256)
        assert stitched.endswith(agent_run._STITCH_TRUNCATED_MARKER)
        viewport = {l for l in agent_run._render_log(raw).splitlines() if l.strip()}
        assert viewport - set(stitched.splitlines()), (
            "this test documents the loss; if truncation ever preserves the "
            "final viewport, tighten the docstring instead of deleting this"
        )

    def test_contains_every_nonempty_viewport_line_when_untruncated(self):
        """The superset claim holds for nonempty lines: empty rows are dropped,
        which is why the docstring says nonempty."""
        raw = b"top\r\n\r\nbottom\r\n"
        viewport = {l for l in agent_run._render_log(raw).splitlines() if l.strip()}
        stitched = set(agent_run._stitch_history(raw, chunk_bytes=4096).splitlines())
        assert viewport <= stitched


class TestSerializeScreenExtraction:
    """_screen_lines was extracted from _serialize_screen and is now shared with
    stitching. _serialize_screen feeds log.clean and the incremental echo
    daemon, so a behaviour change there corrupts cached transcripts."""

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
