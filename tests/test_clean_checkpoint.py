"""Checkpoint-resume rendering: stream grounding, state round-trip, cache resume."""
import json
import os
from pathlib import Path

import pytest

pyte = pytest.importorskip("pyte")

from toolbox import agent_run as ar


W = ar._RENDER_LOG_DEFAULT_WIDTH
H = ar._RENDER_LOG_DEFAULT_HEIGHT
HIST = ar._RENDER_LOG_DEFAULT_HISTORY

# Screen attributes the checkpoint serializes, and those it deliberately omits.
# Anything pyte exposes outside both sets is unaccounted for: a hand-written
# dump would silently stop covering it and resumed renders would diverge from
# full ones with no error raised.
CHECKPOINTED = {
    "buffer", "cursor", "charset", "g0_charset", "g1_charset", "tabstops",
    "savepoints", "margins", "mode", "saved_columns", "title", "icon_name",
    "history",
}
EXCLUDED = {
    "columns": "geometry, carried in cache metadata and compared before load",
    "lines": "geometry, carried in cache metadata and compared before load",
    "dirty": "repaint bookkeeping; load marks the whole viewport dirty",
}

# Payloads whose parser state a byte-level scanner is apt to model incorrectly:
# three-byte ESC commands, ESC inside an OSC payload, C1 controls decoded from
# UTF-8, malformed UTF-8, and multi-byte characters.
GROUNDING_PAYLOADS = [
    pytest.param(b"hello \x1b[31mred\x1b[0m text plain\r\n", id="csi-sgr"),
    pytest.param("h\u00e9llo w\u00f6rld \u2014 dash \u2713 check\r\n".encode("utf-8"),
                 id="multibyte-utf8"),
    pytest.param(b"\x1b]0;title\x07plain after osc\r\n", id="osc-bel"),
    pytest.param(b"A\x1b]0;x\x1b\x07BAD\x07Z", id="osc-containing-esc-bel"),
    pytest.param(b"A\x1b]0;x\x1b\\Z", id="osc-string-terminator"),
    pytest.param(b"A\x1b#8Z", id="esc-hash-decaln"),
    pytest.param(b"A\x1b(0qZ", id="esc-charset-g0"),
    pytest.param(b"A\x1b)Bq\x0eZ", id="esc-charset-g1"),
    pytest.param(b"A\x1b%GZ", id="esc-percent-utf8"),
    pytest.param(b"A\x1b%@\xe2\x80\x94Z", id="esc-percent-no-utf8"),
    pytest.param(b"A\xc2\x9b31mX", id="c1-csi"),
    pytest.param(b"A\xc2\x9d0;abcDEF\x07Z", id="c1-osc"),
    pytest.param(b"A\x1b]0;x\xc2\x9cZ", id="c1-string-terminator"),
    pytest.param(b"A\xffB", id="malformed-lead-byte"),
    pytest.param(b"A\xc0\xafB", id="overlong-encoding"),
    pytest.param(b"A\x80B", id="lone-continuation-byte"),
    pytest.param(b"line one\r\nline two\r\n\x1b[2Kerased\r\n", id="erase-in-line"),
]


def _screen():
    return ar._new_pyte_screen(pyte, W, H, HIST)


def _render(raw: bytes) -> str:
    return ar._render_log(raw, width=W, height=H, history=HIST)


def _feed_prefix(raw: bytes, cut: int):
    """Feed raw[:cut] through one stream and report the screen and grounding."""
    screen = _screen()
    stream = pyte.ByteStream(screen)
    ar._feed_pyte(stream, raw[:cut])
    return screen, ar._ckpt_stream_is_grounded(stream)


def _publish_checkpoint(log: Path, raw: bytes, *, offset_delta: int = 0):
    """Publish transcript, metadata and checkpoint for a grounded prefix of raw,
    then grow the log to its full length as a live run would.

    Returns (clean, cut) or None when no grounded cut exists in the payload.
    """
    clean = log.with_name("log.clean")
    for cut in range(len(raw) // 2, 0, -1):
        screen, grounded = _feed_prefix(raw, cut)
        if not grounded:
            continue
        log.write_bytes(raw[:cut])
        ar._publish_clean(clean, ar._serialize_screen(screen),
                          stat_result=os.stat(log), offset=cut, complete=True)
        ar._ckpt_publish(clean, pyte, screen, offset=cut + offset_delta)
        with log.open("ab") as handle:
            handle.write(raw[cut:])
        return clean, cut
    return None


def test_screen_state_surface_is_fully_accounted_for():
    probe = pyte.HistoryScreen(W, H, history=HIST, ratio=0.5)
    actual = {k for k in vars(probe) if not k.startswith("__")}
    unknown = actual - (CHECKPOINTED | set(EXCLUDED))
    assert not unknown, (
        f"pyte exposes unaccounted screen state {sorted(unknown)}; serialize it "
        f"in _ckpt_dump_state/_ckpt_load_state or add it to EXCLUDED with a reason"
    )
    assert not (CHECKPOINTED | set(EXCLUDED)) - actual


def test_fingerprint_is_stable_across_calls():
    assert ar._ckpt_surface_fingerprint(pyte) == ar._ckpt_surface_fingerprint(pyte)


def test_load_rejects_foreign_fingerprint():
    state = ar._ckpt_dump_state(pyte, _screen())
    state["fingerprint"] = "0" * 16
    with pytest.raises(ValueError):
        ar._ckpt_load_state(pyte, _screen(), state)


def test_load_rejects_unknown_version():
    state = ar._ckpt_dump_state(pyte, _screen())
    state["version"] = ar._CKPT_VERSION + 1
    with pytest.raises(ValueError):
        ar._ckpt_load_state(pyte, _screen(), state)


@pytest.mark.parametrize("payload", GROUNDING_PAYLOADS)
def test_grounded_split_points_render_identically(payload):
    """Wherever the stream reports itself grounded, a checkpoint taken there must
    reproduce the one-pass render.

    Cuts inside an escape sequence or a multi-byte character are the dangerous
    ones: a fresh ByteStream loses both the parser FSM and the incremental UTF-8
    decoder, so grounding must reject them.
    """
    expected = _render(payload)
    for cut in range(1, len(payload) + 1):
        screen, grounded = _feed_prefix(payload, cut)
        if not grounded:
            continue
        state = json.loads(json.dumps(ar._ckpt_dump_state(pyte, screen)))
        restored = _screen()
        ar._ckpt_load_state(pyte, restored, state)
        ar._feed_pyte(pyte.ByteStream(restored), payload[cut:])
        assert ar._serialize_screen(restored) == expected, f"cut={cut}"


@pytest.mark.parametrize("payload", GROUNDING_PAYLOADS)
def test_every_payload_has_at_least_one_grounded_cut(payload):
    """Grounding must not reject every offset: a payload that never grounds would
    stall checkpoint publication for the rest of the run."""
    assert any(_feed_prefix(payload, cut)[1] for cut in range(1, len(payload) + 1))


@pytest.mark.parametrize("pending", [b"\x1b", b"\x1b[", b"\x1b[31", b"\x1b[31;1",
                                     b"\x1b#", b"\x1b(", b"\x1b%", b"\x1b]0;partial"])
def test_stream_is_not_grounded_mid_sequence(pending):
    assert not _feed_prefix(b"hello " + pending, len(b"hello " + pending))[1]


@pytest.mark.parametrize("partial", [b"\xe2", b"\xe2\x80"])
def test_stream_is_not_grounded_mid_utf8_character(partial):
    assert not _feed_prefix(b"ab" + partial, len(b"ab" + partial))[1]


def test_stream_is_not_grounded_after_utf8_mode_is_disabled():
    """ESC % @ clears ByteStream.use_utf8, which lives on the stream and is not
    carried in the checkpoint."""
    assert not _feed_prefix(b"A\x1b%@B", 6)[1]


def test_history_top_window_survives_deque_rollover():
    """history.top is a bounded deque: once full it evicts as it appends, so the
    published window must be the current contents rather than an append log."""
    screen = ar._new_pyte_screen(pyte, 20, 3, 5)
    ar._feed_pyte(pyte.ByteStream(screen),
                  b"".join(f"row{i}\r\n".encode() for i in range(40)))
    assert len(screen.history.top) == screen.history.top.maxlen

    state = json.loads(json.dumps(ar._ckpt_dump_state(pyte, screen)))
    restored = ar._new_pyte_screen(pyte, 20, 3, 5)
    ar._ckpt_load_state(pyte, restored, state)
    assert ar._serialize_screen(restored) == ar._serialize_screen(screen)


def test_read_clean_cache_resumes_from_published_checkpoint(tmp_path):
    raw = b"".join(f"line {i}\r\n".encode() for i in range(200))
    raw += b"\x1b[32mgreen tail\x1b[0m\r\n"
    log = tmp_path / "log"
    assert _publish_checkpoint(log, raw) is not None
    assert ar._read_clean_cache(log, W, H, HIST) == _render(raw)


def test_read_clean_cache_falls_back_when_checkpoint_is_corrupt(tmp_path):
    raw = b"".join(f"line {i}\r\n".encode() for i in range(50))
    log = tmp_path / "log"
    clean, _cut = _publish_checkpoint(log, raw)
    ar._ckpt_state_path(clean).write_text("{not json", encoding="utf-8")
    assert ar._read_clean_cache(log, W, H, HIST) is None


@pytest.mark.parametrize("body", ["1", '"text"', "null", "[]"])
def test_read_clean_cache_falls_back_on_non_object_checkpoint(tmp_path, body):
    """Valid JSON of the wrong shape is a cache miss, not an AttributeError."""
    raw = b"".join(f"line {i}\r\n".encode() for i in range(50))
    log = tmp_path / "log"
    clean, _cut = _publish_checkpoint(log, raw)
    ar._ckpt_state_path(clean).write_text(body, encoding="utf-8")
    assert ar._read_clean_cache(log, W, H, HIST) is None


def test_read_clean_cache_falls_back_on_non_object_buffer_row(tmp_path):
    raw = b"".join(f"line {i}\r\n".encode() for i in range(50))
    log = tmp_path / "log"
    clean, _cut = _publish_checkpoint(log, raw)
    state = json.loads(ar._ckpt_state_path(clean).read_text(encoding="utf-8"))
    state["buffer"] = {"0": 1}
    ar._ckpt_state_path(clean).write_text(json.dumps(state), encoding="utf-8")
    assert ar._read_clean_cache(log, W, H, HIST) is None


def test_read_clean_cache_rejects_checkpoint_for_another_offset(tmp_path):
    raw = b"".join(f"line {i}\r\n".encode() for i in range(50))
    log = tmp_path / "log"
    assert _publish_checkpoint(log, raw, offset_delta=1) is not None
    assert ar._read_clean_cache(log, W, H, HIST) is None


def test_ckpt_discard_removes_the_checkpoint(tmp_path):
    clean = tmp_path / "log.clean"
    ar._ckpt_publish(clean, pyte, _screen(), offset=0)
    assert ar._ckpt_state_path(clean).exists()
    ar._ckpt_discard(clean)
    assert not ar._ckpt_state_path(clean).exists()
