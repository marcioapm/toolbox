"""Tests for toolbox.agent_run._render_log (the PTY -> readable transcript
renderer used by `agent-run clean` and `--echo`)."""
from __future__ import annotations

import builtins
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
# Incremental pyte feed — _serialize_screen produces the same output as a
# single whole-file _render_log call when fed the same bytes in chunks.
# Uses the real moon fixture, which contains Ink TUI redraws, cursor motions,
# OSC title sequences, and escape sequences that span natural chunk boundaries.
# ---------------------------------------------------------------------------

class TestIncrementalFeed:
    def test_chunked_feed_byte_identical_to_whole_file(self, moon_log_bytes):
        """Feeding the real TUI fixture in 64 KiB chunks into a persistent
        pyte screen produces output byte-identical to a single whole-file
        _render_log call.  Confirms that ByteStream correctly buffers partial
        escape sequences across chunk boundaries."""
        import pyte

        whole = _render_log(moon_log_bytes)

        chunk_size = 64 * 1024
        screen = pyte.HistoryScreen(
            agent_run._RENDER_LOG_DEFAULT_WIDTH,
            agent_run._RENDER_LOG_DEFAULT_HEIGHT,
            history=agent_run._RENDER_LOG_DEFAULT_HISTORY,
            ratio=0.5,
        )
        stream = pyte.ByteStream(screen)
        for i in range(0, len(moon_log_bytes), chunk_size):
            agent_run._feed_pyte(stream, moon_log_bytes[i : i + chunk_size])

        chunked = _serialize_screen(screen)
        assert chunked == whole, (
            f"chunked output ({len(chunked)} bytes) differs from whole-file "
            f"output ({len(whole)} bytes)"
        )

    def test_serialize_screen_matches_render_log(self, moon_log_bytes):
        """_serialize_screen on a freshly-fed screen equals _render_log output."""
        import pyte

        screen = pyte.HistoryScreen(
            agent_run._RENDER_LOG_DEFAULT_WIDTH,
            agent_run._RENDER_LOG_DEFAULT_HEIGHT,
            history=agent_run._RENDER_LOG_DEFAULT_HISTORY,
            ratio=0.5,
        )
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
            if self.name == "log.clean.tmp":
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
            if self.name == "log.clean.tmp":
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

    def test_per_tick_delta_cap_writes_warning(self, tmp_path, monkeypatch):
        """When the new-bytes delta in one tick exceeds ECHO_LOOP_MAX_RENDER_BYTES,
        log.clean receives a warning string rather than a silent skip."""
        log_dir = tmp_path / "run"
        log_dir.mkdir()
        log_file = log_dir / "log"
        log_file.write_bytes(b"")

        # Override the cap to a tiny value so the test does not need a 16 MiB file.
        monkeypatch.setattr(agent_run, "ECHO_LOOP_MAX_RENDER_BYTES", 5)

        def grow_on_tick(tick):
            if tick == 0:
                log_file.write_bytes(b"A" * 100)  # 100 > 5-byte cap

        _run_echo_loop_ticks(log_dir, ticks=2, monkeypatch=monkeypatch,
                             extra_actions=grow_on_tick)

        content = (log_dir / "log.clean").read_text()
        assert "ECHO_LOOP_MAX_RENDER_BYTES" in content or "skipped" in content.lower(), (
            f"expected a cap-exceeded warning in log.clean; got: {content!r}"
        )

    def test_echo_loop_max_render_bytes_constant_value(self):
        """Pin the cap value; any change must be a deliberate, test-breaking decision."""
        assert ECHO_LOOP_MAX_RENDER_BYTES == 16 * 1024 * 1024

    def test_missing_pyte_writes_diagnostic_stub(self, tmp_path, monkeypatch):
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

        stub = (log_dir / "log.clean").read_text()
        assert "--echo requested" in stub
        assert "pyte" in stub


# ---------------------------------------------------------------------------
# RecursionError hardening — pyte's coroutine-based FSM can recurse past the
# interpreter's limit on certain pathological ANSI/Ink redraw patterns (this
# crashed 101/212 recent agent-run sessions in production). _render_log must
# degrade to a plain-text ANSI-stripped rendering instead of propagating the
# crash — the clean transcript is a convenience artifact, never a run-ending
# hazard. We can't reliably reproduce the exact pyte-recursing byte pattern
# in a portable unit test, so we simulate the failure at the seam
# (ByteStream.feed) and assert the fallback kicks in and returns a string.
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
