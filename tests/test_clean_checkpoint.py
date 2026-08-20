"""Checkpoint-resume rendering: grounded offsets, state round-trip, cache resume."""
import json
import os

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


def _screen():
    return ar._new_pyte_screen(pyte, W, H, HIST)


def _render(raw: bytes) -> str:
    return ar._render_log(raw, width=W, height=H, history=HIST)


def _render_via_checkpoint(raw: bytes, cut: int) -> str:
    """Render raw by publishing a checkpoint at the grounded offset at or before
    ``cut``, restoring it into a new screen, and feeding the remainder."""
    ground = ar._grounded_offset(raw[:cut])
    first = _screen()
    ar._feed_pyte(pyte.ByteStream(first), raw[:ground])

    state = json.loads(json.dumps(ar._ckpt_dump_state(pyte, first)))
    history_top = json.loads(json.dumps(
        [ar._ckpt_dump_row(row) for row in first.history.top]
    ))

    second = _screen()
    ar._ckpt_load_state(pyte, second, state, history_top)
    ar._feed_pyte(pyte.ByteStream(second), raw[ground:])
    return ar._serialize_screen(second)


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
    screen = _screen()
    state = ar._ckpt_dump_state(pyte, screen)
    state["fingerprint"] = "0" * 16
    with pytest.raises(ValueError):
        ar._ckpt_load_state(pyte, _screen(), state, [])


def test_load_rejects_unknown_version():
    screen = _screen()
    state = ar._ckpt_dump_state(pyte, screen)
    state["version"] = ar._CKPT_VERSION + 1
    with pytest.raises(ValueError):
        ar._ckpt_load_state(pyte, _screen(), state, [])


@pytest.mark.parametrize("payload", [
    b"hello \x1b[31mred\x1b[0m text plain\r\n",
    "h\u00e9llo w\u00f6rld \u2014 em dash and \u2713 check\r\n".encode("utf-8"),
    b"\x1b]0;title\x07plain after osc\r\n",
    b"line one\r\nline two\r\n\x1b[2Kerased\r\n",
])
def test_every_split_point_renders_identically(payload):
    """A checkpoint at any cut must reproduce the one-pass render.

    Cuts landing inside an escape sequence or a multi-byte character are the
    interesting ones: a fresh ByteStream loses both the escape FSM and the
    incremental UTF-8 decoder, so the grounded offset must back away from them.
    """
    expected = _render(payload)
    for cut in range(1, len(payload) + 1):
        assert _render_via_checkpoint(payload, cut) == expected, f"cut={cut}"


@pytest.mark.parametrize("partial", [b"\x1b", b"\x1b[", b"\x1b[31", b"\x1b[31;1"])
def test_grounded_offset_backs_away_from_partial_csi(partial):
    """A CSI is complete only once its final byte (0x40-0x7E) arrives; until
    then the whole sequence must stay unfed."""
    prefix = b"hello "
    assert ar._grounded_offset(prefix + partial) == len(prefix)


def test_grounded_offset_accepts_complete_csi():
    data = b"hello \x1b[31m"
    assert ar._grounded_offset(data) == len(data)
    assert ar._grounded_offset(data + b"x") == len(data) + 1


def test_grounded_offset_backs_away_from_partial_utf8():
    data = "ab\u2014".encode("utf-8")  # em dash is three bytes
    assert ar._grounded_offset(data[:-1]) == 2
    assert ar._grounded_offset(data) == len(data)


def test_grounded_offset_ignores_esc_inside_osc_payload():
    """An OSC string terminator contains ESC, so a backward search for the last
    ESC would mistake a completed OSC for a pending sequence."""
    data = b"\x1b]0;t\x1b\\after"
    assert ar._grounded_offset(data) == len(data)


def test_read_clean_cache_resumes_from_published_checkpoint(tmp_path):
    raw = b"".join(f"line {i}\r\n".encode() for i in range(200))
    raw += b"\x1b[32mgreen tail\x1b[0m\r\n"
    log = tmp_path / "log"
    clean = tmp_path / "log.clean"

    cut = len(raw) // 2
    ground = ar._grounded_offset(raw[:cut])
    screen = _screen()
    ar._feed_pyte(pyte.ByteStream(screen), raw[:ground])
    log.write_bytes(raw[:ground])
    ar._publish_clean(clean, ar._serialize_screen(screen),
                      stat_result=os.stat(log), offset=ground, complete=True)
    ar._ckpt_publish(clean, pyte, screen, history_written=0, offset=ground)
    with log.open("ab") as handle:
        handle.write(raw[ground:])

    assert ar._read_clean_cache(log, W, H, HIST) == _render(raw)


def test_read_clean_cache_falls_back_when_checkpoint_is_corrupt(tmp_path):
    raw = b"".join(f"line {i}\r\n".encode() for i in range(50))
    log = tmp_path / "log"
    clean = tmp_path / "log.clean"

    ground = ar._grounded_offset(raw[: len(raw) // 2])
    screen = _screen()
    ar._feed_pyte(pyte.ByteStream(screen), raw[:ground])
    log.write_bytes(raw[:ground])
    ar._publish_clean(clean, ar._serialize_screen(screen),
                      stat_result=os.stat(log), offset=ground, complete=True)
    ar._ckpt_publish(clean, pyte, screen, history_written=0, offset=ground)
    with log.open("ab") as handle:
        handle.write(raw[ground:])

    ar._ckpt_state_path(clean).write_text("{not json", encoding="utf-8")
    assert ar._read_clean_cache(log, W, H, HIST) is None


def test_read_clean_cache_rejects_checkpoint_for_another_offset(tmp_path):
    raw = b"".join(f"line {i}\r\n".encode() for i in range(50))
    log = tmp_path / "log"
    clean = tmp_path / "log.clean"

    ground = ar._grounded_offset(raw[: len(raw) // 2])
    screen = _screen()
    ar._feed_pyte(pyte.ByteStream(screen), raw[:ground])
    log.write_bytes(raw[:ground])
    ar._publish_clean(clean, ar._serialize_screen(screen),
                      stat_result=os.stat(log), offset=ground, complete=True)
    ar._ckpt_publish(clean, pyte, screen, history_written=0, offset=ground + 1)
    with log.open("ab") as handle:
        handle.write(raw[ground:])

    assert ar._read_clean_cache(log, W, H, HIST) is None


def test_ckpt_discard_removes_both_artifacts(tmp_path):
    clean = tmp_path / "log.clean"
    screen = _screen()
    ar._ckpt_publish(clean, pyte, screen, history_written=0, offset=0)
    assert ar._ckpt_state_path(clean).exists()
    ar._ckpt_discard(clean)
    assert not ar._ckpt_state_path(clean).exists()
    assert not ar._ckpt_history_path(clean).exists()


def test_history_file_is_append_only(tmp_path):
    """Scrolled-off lines are immutable, so publication appends rather than
    rewriting a file that grows with run length."""
    clean = tmp_path / "log.clean"
    screen = _screen()
    ar._feed_pyte(pyte.ByteStream(screen), b"".join(
        f"row {i}\r\n".encode() for i in range(H + 40)
    ))
    written = ar._ckpt_publish(clean, pyte, screen, history_written=0, offset=1)
    assert written > 0
    first = ar._ckpt_history_path(clean).read_bytes()

    ar._feed_pyte(pyte.ByteStream(screen), b"".join(
        f"more {i}\r\n".encode() for i in range(20)
    ))
    ar._ckpt_publish(clean, pyte, screen, history_written=written, offset=2)
    second = ar._ckpt_history_path(clean).read_bytes()
    assert second.startswith(first)
    assert len(second) > len(first)
