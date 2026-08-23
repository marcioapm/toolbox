"""Tests for `agent-run transcript`: the per-harness readers in
agent_run_transcript.py and the cmd_transcript CLI command in agent_run.py.

Every reader is tested against a small fixture store built here (a
temporary SQLite db for opencode, a few JSONL files for claude/codex) --
never against the real per-machine stores. Covers: missing session.json,
unknown harness, missing store file, a corrupt record mid-file, an empty
session, and a tool call with a large output.
"""
from __future__ import annotations

import argparse
import contextlib
import inspect
import json
import os
import sqlite3
import time
import tracemalloc
from pathlib import Path
from typing import Optional

import pytest

from toolbox import agent_run
from toolbox import agent_run_transcript as transcript


ROOT_SKIP = pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0,
    reason="root ignores permission bits",
)


# ---------------------------------------------------------------------------
# fixture builders
# ---------------------------------------------------------------------------

def _make_opencode_db(path: Path) -> None:
    """Minimal `part`/`message` tables matching the real opencode schema
    (only the columns this reader touches), including the `part_session_idx`
    index `_count_opencode` relies on to avoid a full table scan."""
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            "create table message (id text primary key, session_id text, "
            "time_created integer, data text)"
        )
        conn.execute(
            "create table part (id text primary key, message_id text, "
            "session_id text, time_created integer, data text)"
        )
        conn.execute("create index part_session_idx on part (session_id)")
        conn.commit()
    finally:
        conn.close()


def _opencode_insert(
    path: Path, *, message_id: str, session_id: str, role: str, time_created: int, parts: list
) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            "insert into message (id, session_id, time_created, data) values (?, ?, ?, ?)",
            (
                message_id,
                session_id,
                time_created,
                json.dumps({"role": role, "time": {"created": time_created}}),
            ),
        )
        for i, part_data in enumerate(parts):
            conn.execute(
                "insert into part (id, message_id, session_id, time_created, data) values (?, ?, ?, ?, ?)",
                (f"{message_id}-part{i}", message_id, session_id, time_created + i, json.dumps(part_data)),
            )
        conn.commit()
    finally:
        conn.close()


def _write_jsonl(path: Path, records: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


@contextlib.contextmanager
def _unreadable_dir(path: Path):
    """Make an existing directory genuinely unreadable (mode 000) for the
    duration of the block, restoring it in a `finally` so test cleanup
    (tmp_path removal) still works. Exercises the real permission-denied
    path the production code must handle, not a simulated one."""
    path.chmod(0o000)
    try:
        yield
    finally:
        path.chmod(0o755)


# ---------------------------------------------------------------------------
# opencode reader
# ---------------------------------------------------------------------------

class TestOpencodeReader:
    def test_reads_text_and_tool_parts_in_order(self, tmp_path, monkeypatch):
        db = tmp_path / "opencode.db"
        _make_opencode_db(db)
        monkeypatch.setattr(transcript, "OPENCODE_DB_PATH", db)

        _opencode_insert(
            db,
            message_id="msg1",
            session_id="ses_1",
            role="user",
            time_created=1000,
            parts=[{"type": "text", "text": "do the thing"}],
        )
        _opencode_insert(
            db,
            message_id="msg2",
            session_id="ses_1",
            role="assistant",
            time_created=2000,
            parts=[
                {"type": "text", "text": "ok, doing it"},
                {
                    "type": "tool",
                    "tool": "bash",
                    "state": {
                        "status": "completed",
                        "title": "echo hi",
                        "input": {"command": "echo hi"},
                        "output": "hi\n",
                        "time": {"start": 2001, "end": 2002},
                    },
                },
            ],
        )

        entries, skipped = transcript.read_transcript("opencode", "ses_1", None)
        assert skipped == 0
        assert [e.type for e in entries] == ["user", "assistant", "tool"]
        assert entries[0].text == "do the thing"
        assert entries[2].tool_name == "bash"
        assert entries[2].tool_summary == "echo hi"
        assert entries[2].tool_output == "hi\n"
        assert entries[2].tool_status == "completed"

    def test_only_returns_rows_for_the_requested_session(self, tmp_path, monkeypatch):
        db = tmp_path / "opencode.db"
        _make_opencode_db(db)
        monkeypatch.setattr(transcript, "OPENCODE_DB_PATH", db)
        _opencode_insert(
            db, message_id="m1", session_id="ses_a", role="user", time_created=1,
            parts=[{"type": "text", "text": "for a"}],
        )
        _opencode_insert(
            db, message_id="m2", session_id="ses_b", role="user", time_created=2,
            parts=[{"type": "text", "text": "for b"}],
        )
        entries, _ = transcript.read_transcript("opencode", "ses_a", None)
        assert [e.text for e in entries] == ["for a"]

    def test_empty_session_returns_no_entries(self, tmp_path, monkeypatch):
        db = tmp_path / "opencode.db"
        _make_opencode_db(db)
        monkeypatch.setattr(transcript, "OPENCODE_DB_PATH", db)
        entries, skipped = transcript.read_transcript("opencode", "ses_nonexistent", None)
        assert entries == []
        assert skipped == 0

    def test_missing_db_file_raises_source_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr(transcript, "OPENCODE_DB_PATH", tmp_path / "does-not-exist.db")
        with pytest.raises(transcript.TranscriptSourceError):
            transcript.read_transcript("opencode", "ses_1", None)

    @ROOT_SKIP
    def test_unreadable_parent_dir_raises_store_unreadable_not_missing(self, tmp_path, monkeypatch):
        """A permission-denied parent directory makes the db file
        genuinely unstat'able -- distinct from it simply not existing --
        and must not collapse into `store_missing`, which a supervisor
        would otherwise read as a routine empty session."""
        parent = tmp_path / "opencode-data"
        parent.mkdir()
        db = parent / "opencode.db"
        _make_opencode_db(db)
        monkeypatch.setattr(transcript, "OPENCODE_DB_PATH", db)
        with _unreadable_dir(parent):
            with pytest.raises(transcript.TranscriptSourceError) as exc_info:
                transcript.read_transcript("opencode", "ses_1", None)
        assert exc_info.value.code == "store_unreadable"
        assert exc_info.value.__cause__ is not None

    def test_corrupt_row_mid_session_is_skipped_not_raised(self, tmp_path, monkeypatch):
        db = tmp_path / "opencode.db"
        _make_opencode_db(db)
        monkeypatch.setattr(transcript, "OPENCODE_DB_PATH", db)
        conn = sqlite3.connect(db)
        try:
            conn.execute(
                "insert into message (id, session_id, time_created, data) values (?, ?, ?, ?)",
                ("m1", "ses_1", 1, json.dumps({"role": "user", "time": {"created": 1}})),
            )
            # Not valid JSON: the reader must skip this row and keep going.
            conn.execute(
                "insert into part (id, message_id, session_id, time_created, data) values (?, ?, ?, ?, ?)",
                ("p1", "m1", "ses_1", 1, "{not json"),
            )
            conn.commit()
        finally:
            conn.close()
        _opencode_insert(
            db, message_id="m2", session_id="ses_1", role="user", time_created=2,
            parts=[{"type": "text", "text": "still readable"}],
        )
        entries, skipped = transcript.read_transcript("opencode", "ses_1", None)
        assert skipped == 1
        assert [e.text for e in entries] == ["still readable"]

    def test_truncated_database_file_raises_source_error(self, tmp_path, monkeypatch):
        # Not a sqlite file at all: opens (sqlite3.connect never touches the
        # disk until the first query), but the first query must fail as a
        # store-level error, not silently return zero rows.
        db = tmp_path / "garbage.db"
        db.write_bytes(b"not a sqlite database")
        monkeypatch.setattr(transcript, "OPENCODE_DB_PATH", db)
        with pytest.raises(transcript.TranscriptSourceError):
            transcript.read_transcript("opencode", "ses_1", None)

    def test_held_exclusive_lock_fails_fast_not_after_default_busy_timeout(self, tmp_path, monkeypatch):
        db = tmp_path / "opencode.db"
        _make_opencode_db(db)
        monkeypatch.setattr(transcript, "OPENCODE_DB_PATH", db)
        locker = sqlite3.connect(db)
        locker.execute("begin exclusive")
        locker.execute("insert into message (id, session_id, time_created, data) values ('x', 'ses_1', 1, '{}')")
        try:
            start = time.monotonic()
            with pytest.raises(transcript.TranscriptSourceError, match="lock"):
                transcript.read_transcript("opencode", "ses_1", None)
            assert time.monotonic() - start < 1.0
        finally:
            locker.rollback()
            locker.close()

    def test_operational_error_without_lock_or_busy_wording_reports_unreadable_not_locked(
        self, tmp_path, monkeypatch
    ):
        # A store missing the `part` table raises OperationalError with a
        # "no such table" message -- an OperationalError that has nothing
        # to do with a held write lock. It must not be reported as one.
        db = tmp_path / "opencode.db"
        conn = sqlite3.connect(db)
        try:
            conn.execute(
                "create table message (id text primary key, session_id text, "
                "time_created integer, data text)"
            )
            conn.commit()
        finally:
            conn.close()
        monkeypatch.setattr(transcript, "OPENCODE_DB_PATH", db)
        with pytest.raises(transcript.TranscriptSourceError) as exc_info:
            transcript.read_transcript("opencode", "ses_1", None)
        message = str(exc_info.value)
        assert "unreadable" in message
        assert "locked" not in message
        assert "no such table" in message

    def test_large_tool_output_is_bounded(self, tmp_path, monkeypatch):
        db = tmp_path / "opencode.db"
        _make_opencode_db(db)
        monkeypatch.setattr(transcript, "OPENCODE_DB_PATH", db)
        huge_output = "\n".join(f"line {i}" for i in range(3000))
        _opencode_insert(
            db, message_id="m1", session_id="ses_1", role="assistant", time_created=1,
            parts=[
                {
                    "type": "tool",
                    "tool": "read",
                    "state": {"status": "completed", "title": "big.txt", "output": huge_output},
                }
            ],
        )
        entries, _ = transcript.read_transcript("opencode", "ses_1", None)
        assert len(entries) == 1
        out = entries[0].tool_output
        assert out is not None
        assert len(out.splitlines()) <= transcript.TOOL_OUTPUT_MAX_LINES + 1  # +1 for the marker
        assert "truncated" in out
        assert "line 0" in out
        assert "line 2999" not in out


# ---------------------------------------------------------------------------
# count_transcript
# ---------------------------------------------------------------------------

class TestCountTranscript:
    def test_opencode_count_query_uses_the_session_index_not_a_full_scan(self, tmp_path, monkeypatch):
        """Regression guard for the index this counter relies on: fixtures
        must carry `part_session_idx` so a scan-plan regression fails this
        test instead of silently degrading to an unbounded per-poll table
        scan against opencode.db (shared by every session on the machine)."""
        db = tmp_path / "opencode.db"
        _make_opencode_db(db)
        monkeypatch.setattr(transcript, "OPENCODE_DB_PATH", db)
        _opencode_insert(
            db, message_id="m1", session_id="ses_1", role="user", time_created=1000,
            parts=[{"type": "text", "text": "hi"}],
        )
        conn = sqlite3.connect(db)
        try:
            plan = conn.execute(
                "explain query plan select count(*), max(time_created) from part where session_id = ?",
                ("ses_1",),
            ).fetchall()
        finally:
            conn.close()
        plan_text = " ".join(str(cell) for row in plan for cell in row).lower()
        assert "using index part_session_idx" in plan_text, f"expected an index scan, got plan: {plan}"

    def test_opencode_counts_raw_part_rows_and_finds_newest_time(self, tmp_path, monkeypatch):
        db = tmp_path / "opencode.db"
        _make_opencode_db(db)
        monkeypatch.setattr(transcript, "OPENCODE_DB_PATH", db)
        _opencode_insert(
            db, message_id="m1", session_id="ses_1", role="user", time_created=1000,
            parts=[{"type": "text", "text": "hi"}],
        )
        _opencode_insert(
            db, message_id="m2", session_id="ses_1", role="assistant", time_created=2000,
            parts=[{"type": "text", "text": "ok"}, {"type": "step-start"}],
        )
        count, newest = transcript.count_transcript("opencode", "ses_1", None)
        # Two messages, three parts total (one is bookkeeping and would be
        # filtered by read_transcript, but the raw count still includes it).
        assert count == 3
        assert newest is not None
        entries, _ = transcript.read_transcript("opencode", "ses_1", None)
        assert count >= len(entries)

    def test_opencode_absent_session_counts_zero_with_no_newest(self, tmp_path, monkeypatch):
        db = tmp_path / "opencode.db"
        _make_opencode_db(db)
        monkeypatch.setattr(transcript, "OPENCODE_DB_PATH", db)
        count, newest = transcript.count_transcript("opencode", "ses_nonexistent", None)
        assert count == 0
        assert newest is None

    def test_opencode_missing_store_raises_source_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr(transcript, "OPENCODE_DB_PATH", tmp_path / "does-not-exist.db")
        with pytest.raises(transcript.TranscriptSourceError) as exc_info:
            transcript.count_transcript("opencode", "ses_1", None)
        assert exc_info.value.code == "store_missing"

    def test_opencode_held_lock_raises_locked_code(self, tmp_path, monkeypatch):
        db = tmp_path / "opencode.db"
        _make_opencode_db(db)
        monkeypatch.setattr(transcript, "OPENCODE_DB_PATH", db)
        locker = sqlite3.connect(db)
        locker.execute("begin exclusive")
        locker.execute("insert into message (id, session_id, time_created, data) values ('x', 'ses_1', 1, '{}')")
        try:
            with pytest.raises(transcript.TranscriptSourceError) as exc_info:
                transcript.count_transcript("opencode", "ses_1", None)
            assert exc_info.value.code == "store_locked"
        finally:
            locker.rollback()
            locker.close()

    def test_opencode_corrupt_database_raises_unreadable_code(self, tmp_path, monkeypatch):
        db = tmp_path / "garbage.db"
        db.write_bytes(b"not a sqlite database")
        monkeypatch.setattr(transcript, "OPENCODE_DB_PATH", db)
        with pytest.raises(transcript.TranscriptSourceError) as exc_info:
            transcript.count_transcript("opencode", "ses_1", None)
        assert exc_info.value.code == "store_unreadable"

    def test_unknown_harness_raises_unknown_harness_code(self):
        with pytest.raises(transcript.TranscriptSourceError) as exc_info:
            transcript.count_transcript("some-future-harness", "ses_1", None)
        assert exc_info.value.code == "unknown_harness"

    @pytest.mark.parametrize(
        "bad_session_id",
        [
            pytest.param("../../escaped-session", id="parent_traversal"),
            pytest.param("wild*card", id="glob_star"),
            pytest.param("br[ack]et", id="glob_bracket"),
            pytest.param("question?mark", id="glob_question"),
            pytest.param("has/slash", id="path_separator"),
            pytest.param("has\\backslash", id="backslash"),
            pytest.param("has\x00nul", id="embedded_nul"),
        ],
    )
    def test_unsafe_session_id_rejected_before_path_construction_claude(
        self, tmp_path, monkeypatch, bad_session_id
    ):
        monkeypatch.setattr(transcript, "CLAUDE_PROJECTS_DIR", tmp_path)
        with pytest.raises(transcript.TranscriptSourceError) as exc_info:
            transcript.count_transcript("claude", bad_session_id, "/Users/x/proj")
        assert exc_info.value.code == "store_unreadable"

    @pytest.mark.parametrize(
        "bad_session_id",
        [
            pytest.param("../../escaped-session", id="parent_traversal"),
            pytest.param("wild*card", id="glob_star"),
            pytest.param("br[ack]et", id="glob_bracket"),
            pytest.param("has/slash", id="path_separator"),
        ],
    )
    def test_unsafe_session_id_rejected_before_path_construction_codex(
        self, tmp_path, monkeypatch, bad_session_id
    ):
        monkeypatch.setattr(transcript, "CODEX_SESSIONS_DIR", tmp_path)
        with pytest.raises(transcript.TranscriptSourceError) as exc_info:
            transcript.count_transcript("codex", bad_session_id, None)
        assert exc_info.value.code == "store_unreadable"

    def test_claude_count_matches_read_transcript_entry_count(self, tmp_path, monkeypatch):
        monkeypatch.setattr(transcript, "CLAUDE_PROJECTS_DIR", tmp_path)
        cwd = "/Users/x/proj"
        session_id = "sess-count"
        path = tmp_path / "-Users-x-proj" / f"{session_id}.jsonl"
        _write_jsonl(
            path,
            [
                {
                    "type": "user", "sessionId": session_id, "timestamp": "2026-08-16T00:00:00Z",
                    "message": {"role": "user", "content": "hi"},
                },
                {
                    "type": "assistant", "sessionId": session_id, "timestamp": "2026-08-16T00:00:01Z",
                    "message": {"role": "assistant", "content": "hello back"},
                },
            ],
        )
        count, newest = transcript.count_transcript("claude", session_id, cwd)
        entries, _ = transcript.read_transcript("claude", session_id, cwd)
        assert count == len(entries) == 2
        assert newest == "2026-08-16T00:00:01Z"

    def test_claude_count_converts_non_utc_offset_to_true_utc_instant(self, tmp_path, monkeypatch):
        """A record timestamp carrying a non-UTC offset must be converted,
        not relabeled: `+14:00`'s wall-clock digits are 14 hours ahead of
        the same instant in UTC, so naively appending `Z` would report a
        newest time 14 hours in the future."""
        monkeypatch.setattr(transcript, "CLAUDE_PROJECTS_DIR", tmp_path)
        cwd = "/Users/x/proj"
        session_id = "sess-offset"
        path = tmp_path / "-Users-x-proj" / f"{session_id}.jsonl"
        _write_jsonl(
            path,
            [
                {
                    "type": "user", "sessionId": session_id, "timestamp": "2026-08-16T12:00:00+14:00",
                    "message": {"role": "user", "content": "hi"},
                },
            ],
        )
        count, newest = transcript.count_transcript("claude", session_id, cwd)
        assert count == 1
        assert newest == "2026-08-15T22:00:00Z"  # same instant, converted to UTC

    def test_claude_count_converts_negative_offset_to_true_utc_instant(self, tmp_path, monkeypatch):
        monkeypatch.setattr(transcript, "CLAUDE_PROJECTS_DIR", tmp_path)
        cwd = "/Users/x/proj"
        session_id = "sess-offset-neg"
        path = tmp_path / "-Users-x-proj" / f"{session_id}.jsonl"
        _write_jsonl(
            path,
            [
                {
                    "type": "user", "sessionId": session_id, "timestamp": "2026-08-16T12:00:00-05:00",
                    "message": {"role": "user", "content": "hi"},
                },
            ],
        )
        count, newest = transcript.count_transcript("claude", session_id, cwd)
        assert count == 1
        assert newest == "2026-08-16T17:00:00Z"  # same instant, converted to UTC

    def test_codex_count_converts_non_utc_offset_to_true_utc_instant(self, tmp_path, monkeypatch):
        monkeypatch.setattr(transcript, "CODEX_SESSIONS_DIR", tmp_path)
        session_id = "offset-session"
        path = tmp_path / "2026" / "08" / "19" / f"rollout-2026-08-19T00-00-00-{session_id}.jsonl"
        _write_jsonl(
            path,
            [
                {
                    "timestamp": "2026-08-19T12:00:00+14:00", "type": "response_item",
                    "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]},
                },
            ],
        )
        count, newest = transcript.count_transcript("codex", session_id, None)
        assert count == 1
        assert newest == "2026-08-18T22:00:00Z"

    def test_claude_missing_store_raises_store_missing_code(self, tmp_path, monkeypatch):
        monkeypatch.setattr(transcript, "CLAUDE_PROJECTS_DIR", tmp_path)
        with pytest.raises(transcript.TranscriptSourceError) as exc_info:
            transcript.count_transcript("claude", "no-such-session", "/nonexistent/cwd")
        assert exc_info.value.code == "store_missing"

    def test_codex_count_matches_read_transcript_entry_count(self, tmp_path, monkeypatch):
        monkeypatch.setattr(transcript, "CODEX_SESSIONS_DIR", tmp_path)
        session_id = "count-session"
        path = tmp_path / "2026" / "08" / "19" / f"rollout-2026-08-19T00-00-00-{session_id}.jsonl"
        _write_jsonl(
            path,
            [
                {
                    "timestamp": "2026-08-19T00:00:00Z", "type": "response_item",
                    "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]},
                },
                {
                    "timestamp": "2026-08-19T00:00:01Z", "type": "response_item",
                    "payload": {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "hey"}]},
                },
            ],
        )
        count, newest = transcript.count_transcript("codex", session_id, None)
        entries, _ = transcript.read_transcript("codex", session_id, None)
        assert count == len(entries) == 2
        assert newest == "2026-08-19T00:00:01Z"

    def test_codex_missing_store_raises_store_missing_code(self, tmp_path, monkeypatch):
        monkeypatch.setattr(transcript, "CODEX_SESSIONS_DIR", tmp_path)
        with pytest.raises(transcript.TranscriptSourceError) as exc_info:
            transcript.count_transcript("codex", "no-such-session", None)
        assert exc_info.value.code == "store_missing"

    @ROOT_SKIP
    def test_claude_unreadable_glob_directory_raises_store_unreadable_not_missing(
        self, tmp_path, monkeypatch
    ):
        root = tmp_path / "projects"
        root.mkdir()
        monkeypatch.setattr(transcript, "CLAUDE_PROJECTS_DIR", root)
        with _unreadable_dir(root):
            with pytest.raises(transcript.TranscriptSourceError) as exc_info:
                transcript.count_transcript("claude", "some-session", "/nonexistent/cwd")
        assert exc_info.value.code == "store_unreadable"
        assert exc_info.value.__cause__ is not None

    @ROOT_SKIP
    def test_codex_unreadable_sessions_dir_raises_store_unreadable_not_missing(
        self, tmp_path, monkeypatch
    ):
        root = tmp_path / "sessions"
        root.mkdir()
        monkeypatch.setattr(transcript, "CODEX_SESSIONS_DIR", root)
        with _unreadable_dir(root):
            with pytest.raises(transcript.TranscriptSourceError) as exc_info:
                transcript.count_transcript("codex", "some-session", None)
        assert exc_info.value.code == "store_unreadable"
        assert exc_info.value.__cause__ is not None

    def test_claude_skipped_only_file_raises_store_unreadable_not_zero(self, tmp_path, monkeypatch):
        """Every record in the file is a user/assistant record with a
        non-dict `message` -- wholly malformed, not merely a session with
        no conversation. `count_transcript` must not report this as a
        trustworthy 0."""
        monkeypatch.setattr(transcript, "CLAUDE_PROJECTS_DIR", tmp_path)
        session_id = "sess-all-skipped"
        path = tmp_path / "-Users-x-proj" / f"{session_id}.jsonl"
        _write_jsonl(
            path,
            [
                {"type": "user", "sessionId": session_id, "message": "not-a-dict"},
                {"type": "assistant", "sessionId": session_id, "message": None},
            ],
        )
        with pytest.raises(transcript.TranscriptSourceError) as exc_info:
            transcript.count_transcript("claude", session_id, "/Users/x/proj")
        assert exc_info.value.code == "store_unreadable"

    def test_claude_partially_skipped_file_still_counts_the_readable_records(
        self, tmp_path, monkeypatch
    ):
        """skipped > 0 alongside a nonzero count is a normal partial-parse
        outcome, not a store failure."""
        monkeypatch.setattr(transcript, "CLAUDE_PROJECTS_DIR", tmp_path)
        session_id = "sess-partial-skip"
        path = tmp_path / "-Users-x-proj" / f"{session_id}.jsonl"
        _write_jsonl(
            path,
            [
                {"type": "user", "sessionId": session_id, "message": "not-a-dict"},
                {
                    "type": "user", "sessionId": session_id, "timestamp": "2026-08-16T00:00:00Z",
                    "message": {"role": "user", "content": "hi"},
                },
            ],
        )
        count, newest = transcript.count_transcript("claude", session_id, "/Users/x/proj")
        assert count == 1
        assert newest == "2026-08-16T00:00:00Z"

    def test_claude_direct_path_all_mismatched_records_raises_store_unreadable(
        self, tmp_path, monkeypatch
    ):
        """(a) A direct-path (cwd-derived) candidate whose every record
        names a different session is a conflicting candidate, not a
        verified empty store -- must not report a false `entries: 0`."""
        monkeypatch.setattr(transcript, "CLAUDE_PROJECTS_DIR", tmp_path)
        session_id = "sess-direct-all-mismatched"
        path = tmp_path / "-Users-x-proj" / f"{session_id}.jsonl"
        _write_jsonl(
            path,
            [
                {
                    "type": "user", "sessionId": "someone-elses-session",
                    "timestamp": "2026-08-16T00:00:00Z",
                    "message": {"role": "user", "content": "not mine"},
                },
                {
                    "type": "assistant", "sessionId": "someone-elses-session",
                    "timestamp": "2026-08-16T00:00:01Z",
                    "message": {"role": "assistant", "content": "also not mine"},
                },
            ],
        )
        with pytest.raises(transcript.TranscriptSourceError) as exc_info:
            transcript.count_transcript("claude", session_id, "/Users/x/proj")
        assert exc_info.value.code == "store_unreadable"

    def test_claude_direct_path_mismatched_plus_ignorable_metadata_raises_store_unreadable(
        self, tmp_path, monkeypatch
    ):
        """(b) Mismatched conversation records plus otherwise-ignorable
        non-conversation metadata (a record type this reader always
        skips) is still a conflicting candidate, not a valid empty
        store."""
        monkeypatch.setattr(transcript, "CLAUDE_PROJECTS_DIR", tmp_path)
        session_id = "sess-direct-mismatched-plus-meta"
        path = tmp_path / "-Users-x-proj" / f"{session_id}.jsonl"
        _write_jsonl(
            path,
            [
                {"type": "queue-operation", "sessionId": session_id, "op": "noop"},
                {
                    "type": "user", "sessionId": "someone-elses-session",
                    "timestamp": "2026-08-16T00:00:00Z",
                    "message": {"role": "user", "content": "not mine"},
                },
            ],
        )
        with pytest.raises(transcript.TranscriptSourceError) as exc_info:
            transcript.count_transcript("claude", session_id, "/Users/x/proj")
        assert exc_info.value.code == "store_unreadable"

    def test_claude_direct_path_one_match_among_mismatched_records_counts_the_match(
        self, tmp_path, monkeypatch
    ):
        """(c) A direct-path candidate carrying one genuine record for this
        session alongside records for other sessions returns the matching
        count -- the file is this session's transcript, just not the only
        one that ever wrote to this location."""
        monkeypatch.setattr(transcript, "CLAUDE_PROJECTS_DIR", tmp_path)
        session_id = "sess-direct-one-match"
        path = tmp_path / "-Users-x-proj" / f"{session_id}.jsonl"
        _write_jsonl(
            path,
            [
                {
                    "type": "user", "sessionId": "someone-elses-session",
                    "timestamp": "2026-08-16T00:00:00Z",
                    "message": {"role": "user", "content": "not mine"},
                },
                {
                    "type": "user", "sessionId": session_id,
                    "timestamp": "2026-08-16T00:00:01Z",
                    "message": {"role": "user", "content": "mine"},
                },
            ],
        )
        count, newest = transcript.count_transcript("claude", session_id, "/Users/x/proj")
        assert count == 1
        assert newest == "2026-08-16T00:00:01Z"

    def test_claude_direct_path_empty_file_is_a_valid_zero(self, tmp_path, monkeypatch):
        """A genuinely empty direct-path file (no records at all) is a
        trustworthy zero, not a conflict -- distinguishes 'nothing here
        yet' from 'something else is here'."""
        monkeypatch.setattr(transcript, "CLAUDE_PROJECTS_DIR", tmp_path)
        session_id = "sess-direct-empty"
        path = tmp_path / "-Users-x-proj" / f"{session_id}.jsonl"
        _write_jsonl(path, [])
        count, newest = transcript.count_transcript("claude", session_id, "/Users/x/proj")
        assert count == 0
        assert newest is None

    def test_claude_direct_path_metadata_only_file_is_a_valid_zero(self, tmp_path, monkeypatch):
        """A direct-path file holding only recognized non-conversation
        metadata (no user/assistant records at all) is a valid zero, not
        a conflict."""
        monkeypatch.setattr(transcript, "CLAUDE_PROJECTS_DIR", tmp_path)
        session_id = "sess-direct-metadata-only"
        path = tmp_path / "-Users-x-proj" / f"{session_id}.jsonl"
        _write_jsonl(
            path,
            [
                {"type": "queue-operation", "sessionId": session_id, "op": "noop"},
                {"type": "mode", "sessionId": session_id, "mode": "plan"},
            ],
        )
        count, newest = transcript.count_transcript("claude", session_id, "/Users/x/proj")
        assert count == 0
        assert newest is None

    def test_codex_skipped_only_file_raises_store_unreadable_not_zero(self, tmp_path, monkeypatch):
        """Every `response_item` in the file has a non-dict `payload` --
        wholly malformed, not merely an empty session."""
        monkeypatch.setattr(transcript, "CODEX_SESSIONS_DIR", tmp_path)
        session_id = "sess-codex-all-skipped"
        path = tmp_path / "2026" / "08" / "19" / f"rollout-2026-08-19T00-00-00-{session_id}.jsonl"
        _write_jsonl(
            path,
            [
                {"timestamp": "2026-08-19T00:00:00Z", "type": "response_item", "payload": "not-a-dict"},
                {"timestamp": "2026-08-19T00:00:01Z", "type": "response_item", "payload": None},
            ],
        )
        with pytest.raises(transcript.TranscriptSourceError) as exc_info:
            transcript.count_transcript("codex", session_id, None)
        assert exc_info.value.code == "store_unreadable"

    def test_codex_partially_skipped_file_still_counts_the_readable_records(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(transcript, "CODEX_SESSIONS_DIR", tmp_path)
        session_id = "sess-codex-partial-skip"
        path = tmp_path / "2026" / "08" / "19" / f"rollout-2026-08-19T00-00-00-{session_id}.jsonl"
        _write_jsonl(
            path,
            [
                {"timestamp": "2026-08-19T00:00:00Z", "type": "response_item", "payload": "not-a-dict"},
                {
                    "timestamp": "2026-08-19T00:00:01Z", "type": "response_item",
                    "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]},
                },
            ],
        )
        count, newest = transcript.count_transcript("codex", session_id, None)
        assert count == 1
        assert newest == "2026-08-19T00:00:01Z"


# ---------------------------------------------------------------------------
# counters stream, not materialize
# ---------------------------------------------------------------------------

class TestCountersStream:
    def test_iter_jsonl_objects_is_a_generator_not_a_list_builder(self):
        assert inspect.isgeneratorfunction(transcript._iter_jsonl_objects)

    def test_claude_count_peak_memory_bounded_by_largest_line_not_file_size(
        self, tmp_path, monkeypatch
    ):
        """A counter that materializes every parsed record before
        classifying it (the old `_read_jsonl_objects`-backed
        implementation) has peak traced memory that scales with file
        size. One that streams and discards each record after
        classification does not: peak memory stays close to one line's
        size regardless of how many lines the file holds."""
        monkeypatch.setattr(transcript, "CLAUDE_PROJECTS_DIR", tmp_path)
        session_id = "sess-streaming-claude"
        path = tmp_path / "-Users-x-proj" / f"{session_id}.jsonl"
        record_count = 20_000
        filler = "x" * 200  # ~250 bytes/record once JSON-encoded

        def _records():
            for i in range(record_count):
                yield {
                    "type": "user", "sessionId": session_id,
                    "timestamp": "2026-08-16T00:00:00Z",
                    "message": {"role": "user", "content": f"{filler}-{i}"},
                }

        _write_jsonl(path, _records())
        file_size = path.stat().st_size
        assert file_size > 4_000_000  # sanity: this is a genuinely large file

        tracemalloc.start()
        try:
            tracemalloc.clear_traces()
            count, _ = transcript.count_transcript("claude", session_id, "/Users/x/proj")
            _current, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        assert count == record_count
        # Bounded by a small constant multiple of one line's size, not by
        # file size: a materializing implementation would peak near
        # file_size; a streaming one stays orders of magnitude below it.
        assert peak < file_size // 4, (
            f"peak traced memory {peak} bytes is too close to file size {file_size} bytes "
            "-- counter appears to materialize the whole file rather than stream it"
        )

    def test_codex_count_peak_memory_bounded_by_largest_line_not_file_size(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(transcript, "CODEX_SESSIONS_DIR", tmp_path)
        session_id = "sess-streaming-codex"
        path = tmp_path / "2026" / "08" / "19" / f"rollout-2026-08-19T00-00-00-{session_id}.jsonl"
        record_count = 20_000
        filler = "x" * 200

        def _records():
            for i in range(record_count):
                yield {
                    "timestamp": "2026-08-19T00:00:00Z", "type": "response_item",
                    "payload": {
                        "type": "message", "role": "user",
                        "content": [{"type": "input_text", "text": f"{filler}-{i}"}],
                    },
                }

        _write_jsonl(path, _records())
        file_size = path.stat().st_size
        assert file_size > 4_000_000

        tracemalloc.start()
        try:
            tracemalloc.clear_traces()
            count, _ = transcript.count_transcript("codex", session_id, None)
            _current, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        assert count == record_count
        assert peak < file_size // 4, (
            f"peak traced memory {peak} bytes is too close to file size {file_size} bytes "
            "-- counter appears to materialize the whole file rather than stream it"
        )


# ---------------------------------------------------------------------------
# claude reader
# ---------------------------------------------------------------------------

class TestClaudeReader:
    def test_reads_user_assistant_and_tool_use_with_matched_result(self, tmp_path, monkeypatch):
        monkeypatch.setattr(transcript, "CLAUDE_PROJECTS_DIR", tmp_path)
        cwd = "/Users/x/proj"
        session_id = "sess-abc"
        path = tmp_path / "-Users-x-proj" / f"{session_id}.jsonl"
        _write_jsonl(
            path,
            [
                {
                    "type": "user",
                    "sessionId": session_id,
                    "timestamp": "2026-08-16T00:00:00Z",
                    "message": {"role": "user", "content": "please run echo"},
                },
                {
                    "type": "assistant",
                    "sessionId": session_id,
                    "timestamp": "2026-08-16T00:00:01Z",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": "running it"},
                            {
                                "type": "tool_use",
                                "id": "toolu_1",
                                "name": "Bash",
                                "input": {"command": "echo hi", "description": "Echo hi"},
                            },
                        ],
                    },
                },
                {
                    "type": "user",
                    "sessionId": session_id,
                    "timestamp": "2026-08-16T00:00:02Z",
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "tool_use_id": "toolu_1",
                                "type": "tool_result",
                                "content": [{"type": "text", "text": "hi\n"}],
                            }
                        ],
                    },
                },
            ],
        )
        entries, skipped = transcript.read_transcript("claude", session_id, cwd)
        assert skipped == 0
        assert [e.type for e in entries] == ["user", "assistant", "tool"]
        assert entries[1].text == "running it"
        assert entries[2].tool_name == "Bash"
        assert entries[2].tool_summary == "Echo hi"
        assert entries[2].tool_output == "hi\n"
        assert entries[2].tool_status == "completed"

    def test_falls_back_to_glob_when_cwd_path_misses(self, tmp_path, monkeypatch):
        monkeypatch.setattr(transcript, "CLAUDE_PROJECTS_DIR", tmp_path)
        session_id = "sess-glob"
        path = tmp_path / "-some-other-dir" / f"{session_id}.jsonl"
        _write_jsonl(
            path,
            [
                {
                    "type": "user",
                    "sessionId": session_id,
                    "timestamp": "2026-08-16T00:00:00Z",
                    "message": {"role": "user", "content": "hello"},
                }
            ],
        )
        # cwd points somewhere that does not exist on disk.
        entries, _ = transcript.read_transcript("claude", session_id, "/nonexistent/cwd")
        assert [e.text for e in entries] == ["hello"]

    def test_glob_fallback_with_two_colliding_filenames_picks_matching_content(self, tmp_path, monkeypatch):
        monkeypatch.setattr(transcript, "CLAUDE_PROJECTS_DIR", tmp_path)
        session_id = "wanted-sid"
        _write_jsonl(
            tmp_path / "-proj-a" / f"{session_id}.jsonl",
            [
                {
                    "type": "user",
                    "sessionId": "SOMEONE-ELSE",
                    "timestamp": "2026-08-16T00:00:00Z",
                    "message": {"role": "user", "content": "foreign content"},
                }
            ],
        )
        _write_jsonl(
            tmp_path / "-proj-b" / f"{session_id}.jsonl",
            [
                {
                    "type": "user",
                    "sessionId": session_id,
                    "timestamp": "2026-08-16T00:00:00Z",
                    "message": {"role": "user", "content": "correct content"},
                }
            ],
        )
        entries, _ = transcript.read_transcript("claude", session_id, "/nonexistent/cwd")
        assert [e.text for e in entries] == ["correct content"]

    def test_glob_fallback_with_genuine_ambiguity_fails_loudly(self, tmp_path, monkeypatch):
        monkeypatch.setattr(transcript, "CLAUDE_PROJECTS_DIR", tmp_path)
        session_id = "wanted-sid-ambiguous"
        record = {
            "type": "user",
            "sessionId": session_id,
            "timestamp": "2026-08-16T00:00:00Z",
            "message": {"role": "user", "content": "hi"},
        }
        path_a = tmp_path / "-proj-a" / f"{session_id}.jsonl"
        path_b = tmp_path / "-proj-b" / f"{session_id}.jsonl"
        _write_jsonl(path_a, [record])
        _write_jsonl(path_b, [record])
        with pytest.raises(transcript.TranscriptSourceError) as exc_info:
            transcript.read_transcript("claude", session_id, "/nonexistent/cwd")
        message = str(exc_info.value)
        assert str(path_a) in message
        assert str(path_b) in message

    def test_main_file_record_with_mismatched_session_id_excluded(self, tmp_path, monkeypatch):
        monkeypatch.setattr(transcript, "CLAUDE_PROJECTS_DIR", tmp_path)
        session_id = "sess-main-mismatch"
        cwd = "/Users/x/proj"
        path = tmp_path / "-Users-x-proj" / f"{session_id}.jsonl"
        _write_jsonl(
            path,
            [
                {
                    "type": "user",
                    "sessionId": session_id,
                    "timestamp": "2026-08-16T00:00:00Z",
                    "message": {"role": "user", "content": "mine"},
                },
                {
                    "type": "assistant",
                    "sessionId": "some-other-session",
                    "timestamp": "2026-08-16T00:00:01Z",
                    "message": {"role": "assistant", "content": "not mine"},
                },
            ],
        )
        entries, _ = transcript.read_transcript("claude", session_id, cwd)
        assert [e.text for e in entries] == ["mine"]

    def test_missing_store_raises_source_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr(transcript, "CLAUDE_PROJECTS_DIR", tmp_path)
        with pytest.raises(transcript.TranscriptSourceError):
            transcript.read_transcript("claude", "no-such-session", "/nowhere")

    @ROOT_SKIP
    def test_unreadable_cwd_candidate_check_raises_store_unreadable(self, tmp_path, monkeypatch):
        """A permission-denied path component under the cwd-derived
        candidate's directory (stat failure) must surface as a read
        failure, not fall through to the glob path and possibly report
        absence."""
        monkeypatch.setattr(transcript, "CLAUDE_PROJECTS_DIR", tmp_path)
        session_id = "sess-unreadable-candidate"
        project_dir = tmp_path / "-Users-x-proj"
        project_dir.mkdir()
        with _unreadable_dir(project_dir):
            with pytest.raises(transcript.TranscriptSourceError) as exc_info:
                transcript.read_transcript("claude", session_id, "/Users/x/proj")
        assert exc_info.value.code == "store_unreadable"
        assert exc_info.value.__cause__ is not None

    @ROOT_SKIP
    def test_unreadable_glob_directory_raises_store_unreadable_not_missing(self, tmp_path, monkeypatch):
        """CLAUDE_PROJECTS_DIR itself unreadable (permission denied on
        scandir) is not evidence the session is absent -- it must not
        collapse into store_missing."""
        root = tmp_path / "projects"
        root.mkdir()
        monkeypatch.setattr(transcript, "CLAUDE_PROJECTS_DIR", root)
        session_id = "sess-unreadable-glob"
        with _unreadable_dir(root):
            with pytest.raises(transcript.TranscriptSourceError) as exc_info:
                transcript.read_transcript("claude", session_id, None)
        assert exc_info.value.code == "store_unreadable"
        assert exc_info.value.__cause__ is not None

    @ROOT_SKIP
    def test_unreadable_subagents_dir_check_raises_store_unreadable(self, tmp_path, monkeypatch):
        """A readable main file whose session directory (housing
        subagents/) cannot be stat'd must not silently report only the
        main file's entries -- the subagent tree is part of this
        session's transcript."""
        monkeypatch.setattr(transcript, "CLAUDE_PROJECTS_DIR", tmp_path)
        session_id = "sess-unreadable-subagents"
        cwd = "/Users/x/proj"
        path = tmp_path / "-Users-x-proj" / f"{session_id}.jsonl"
        _write_jsonl(
            path,
            [
                {
                    "type": "user", "sessionId": session_id, "timestamp": "2026-08-16T00:00:00Z",
                    "message": {"role": "user", "content": "main turn"},
                }
            ],
        )
        session_dir = path.parent / path.stem
        session_dir.mkdir()
        with _unreadable_dir(session_dir):
            with pytest.raises(transcript.TranscriptSourceError) as exc_info:
                transcript.read_transcript("claude", session_id, cwd)
        assert exc_info.value.code == "store_unreadable"
        assert exc_info.value.__cause__ is not None

    @ROOT_SKIP
    def test_unreadable_subagents_glob_raises_store_unreadable(self, tmp_path, monkeypatch):
        """The subagents/ directory itself stat'able (mode allows lookup)
        but not listable (execute-only) must still surface as a read
        failure."""
        monkeypatch.setattr(transcript, "CLAUDE_PROJECTS_DIR", tmp_path)
        session_id = "sess-unreadable-subagents-glob"
        cwd = "/Users/x/proj"
        path = tmp_path / "-Users-x-proj" / f"{session_id}.jsonl"
        _write_jsonl(
            path,
            [
                {
                    "type": "user", "sessionId": session_id, "timestamp": "2026-08-16T00:00:00Z",
                    "message": {"role": "user", "content": "main turn"},
                }
            ],
        )
        subagents_dir = path.parent / path.stem / "subagents"
        subagents_dir.mkdir(parents=True)
        subagents_dir.chmod(0o111)  # traversable (stat/is_dir succeeds) but not listable
        try:
            with pytest.raises(transcript.TranscriptSourceError) as exc_info:
                transcript.read_transcript("claude", session_id, cwd)
        finally:
            subagents_dir.chmod(0o755)
        assert exc_info.value.code == "store_unreadable"
        assert exc_info.value.__cause__ is not None

    def test_empty_session_file_returns_no_entries(self, tmp_path, monkeypatch):
        monkeypatch.setattr(transcript, "CLAUDE_PROJECTS_DIR", tmp_path)
        session_id = "sess-empty"
        path = tmp_path / "-Users-x-proj" / f"{session_id}.jsonl"
        _write_jsonl(path, [])
        entries, skipped = transcript.read_transcript("claude", session_id, "/Users/x/proj")
        assert entries == []
        assert skipped == 0

    def test_corrupt_record_mid_file_is_skipped(self, tmp_path, monkeypatch):
        monkeypatch.setattr(transcript, "CLAUDE_PROJECTS_DIR", tmp_path)
        session_id = "sess-corrupt"
        path = tmp_path / "-Users-x-proj" / f"{session_id}.jsonl"
        path.parent.mkdir(parents=True)
        lines = [
            json.dumps(
                {
                    "type": "user",
                    "sessionId": session_id,
                    "timestamp": "2026-08-16T00:00:00Z",
                    "message": {"role": "user", "content": "first"},
                }
            ),
            "{not json at all",
            json.dumps(
                {
                    "type": "user",
                    "sessionId": session_id,
                    "timestamp": "2026-08-16T00:00:01Z",
                    "message": {"role": "user", "content": "second"},
                }
            ),
        ]
        path.write_text("\n".join(lines) + "\n")
        entries, skipped = transcript.read_transcript("claude", session_id, "/Users/x/proj")
        assert skipped == 1
        assert [e.text for e in entries] == ["first", "second"]

    def test_subagent_transcript_included_and_labeled_when_session_matches(self, tmp_path, monkeypatch):
        monkeypatch.setattr(transcript, "CLAUDE_PROJECTS_DIR", tmp_path)
        session_id = "sess-parent"
        base = tmp_path / "-Users-x-proj"
        _write_jsonl(
            base / f"{session_id}.jsonl",
            [
                {
                    "type": "user",
                    "sessionId": session_id,
                    "timestamp": "2026-08-16T00:00:00Z",
                    "message": {"role": "user", "content": "main turn"},
                }
            ],
        )
        _write_jsonl(
            base / session_id / "subagents" / "agent-xyz.jsonl",
            [
                {
                    "type": "assistant",
                    "sessionId": session_id,
                    "timestamp": "2026-08-16T00:00:01Z",
                    "message": {"role": "assistant", "content": "subagent turn"},
                }
            ],
        )
        entries, _ = transcript.read_transcript("claude", session_id, "/Users/x/proj")
        assert [e.text for e in entries] == ["main turn", "subagent turn"]
        assert entries[1].subagent == "agent-xyz"

    def test_subagent_record_with_mismatched_session_id_excluded(self, tmp_path, monkeypatch):
        monkeypatch.setattr(transcript, "CLAUDE_PROJECTS_DIR", tmp_path)
        session_id = "sess-parent2"
        base = tmp_path / "-Users-x-proj"
        _write_jsonl(base / f"{session_id}.jsonl", [])
        _write_jsonl(
            base / session_id / "subagents" / "agent-abc.jsonl",
            [
                {
                    "type": "assistant",
                    "sessionId": "a-different-session",
                    "timestamp": "2026-08-16T00:00:01Z",
                    "message": {"role": "assistant", "content": "not this session"},
                }
            ],
        )
        entries, _ = transcript.read_transcript("claude", session_id, "/Users/x/proj")
        assert entries == []

    def test_main_and_subagent_entries_merged_by_timestamp(self, tmp_path, monkeypatch):
        monkeypatch.setattr(transcript, "CLAUDE_PROJECTS_DIR", tmp_path)
        session_id = "sess-interleave"
        base = tmp_path / "-Users-x-proj"

        def rec(role, text, ts):
            return {
                "type": role,
                "sessionId": session_id,
                "timestamp": ts,
                "message": {"role": role, "content": text},
            }

        # Main stream spans the whole run; the subagent runs in the middle
        # of it, so a naive concatenation would put every subagent entry
        # after every main entry despite the subagent finishing first.
        _write_jsonl(
            base / f"{session_id}.jsonl",
            [
                rec("user", "main-1", "2026-08-16T00:00:00Z"),
                rec("assistant", "main-2", "2026-08-16T00:00:10Z"),
                rec("assistant", "main-3", "2026-08-16T00:00:30Z"),
            ],
        )
        _write_jsonl(
            base / session_id / "subagents" / "agent-mid.jsonl",
            [
                rec("assistant", "sub-1", "2026-08-16T00:00:05Z"),
                rec("assistant", "sub-2", "2026-08-16T00:00:15Z"),
            ],
        )
        entries, _ = transcript.read_transcript("claude", session_id, "/Users/x/proj")
        assert [e.text for e in entries] == ["main-1", "sub-1", "main-2", "sub-2", "main-3"]

    def test_mixed_naive_and_aware_timestamps_merge_without_raising(self, tmp_path, monkeypatch):
        monkeypatch.setattr(transcript, "CLAUDE_PROJECTS_DIR", tmp_path)
        session_id = "sess-mixed-tz"
        base = tmp_path / "-Users-x-proj"

        def rec(role, text, ts):
            record = {"type": role, "sessionId": session_id, "message": {"role": role, "content": text}}
            if ts is not None:
                record["timestamp"] = ts
            return record

        # "aware" carries a Z offset; "naive" carries none, a store quirk
        # that must not crash the sort comparing the two.
        _write_jsonl(
            base / f"{session_id}.jsonl",
            [
                rec("user", "aware", "2026-01-01T00:00:00Z"),
                rec("user", "naive", "2026-01-01T00:00:01"),
            ],
        )
        _write_jsonl(
            base / session_id / "subagents" / "agent-mid.jsonl",
            [rec("assistant", "sub", "2026-01-01T00:00:00.500000Z")],
        )
        entries, _ = transcript.read_transcript("claude", session_id, "/Users/x/proj")
        assert [e.text for e in entries] == ["aware", "sub", "naive"]

    def test_equal_timestamps_across_files_keep_deterministic_order(self, tmp_path, monkeypatch):
        monkeypatch.setattr(transcript, "CLAUDE_PROJECTS_DIR", tmp_path)
        session_id = "sess-tie"
        base = tmp_path / "-Users-x-proj"

        def rec(role, text, ts):
            return {
                "type": role,
                "sessionId": session_id,
                "timestamp": ts,
                "message": {"role": role, "content": text},
            }

        same_ts = "2026-08-16T00:00:00Z"
        _write_jsonl(base / f"{session_id}.jsonl", [rec("user", "main-a", same_ts), rec("user", "main-b", same_ts)])
        _write_jsonl(
            base / session_id / "subagents" / "agent-tie.jsonl",
            [rec("assistant", "sub-a", same_ts), rec("assistant", "sub-b", same_ts)],
        )
        entries, _ = transcript.read_transcript("claude", session_id, "/Users/x/proj")
        # Same across both runs: file discovery order (main first, then
        # subagents sorted by filename), then position within the file.
        assert [e.text for e in entries] == ["main-a", "main-b", "sub-a", "sub-b"]

    def test_file_with_no_timestamps_keeps_internal_order(self, tmp_path, monkeypatch):
        monkeypatch.setattr(transcript, "CLAUDE_PROJECTS_DIR", tmp_path)
        session_id = "sess-notime"
        base = tmp_path / "-Users-x-proj"
        records = [
            {
                "type": "user",
                "sessionId": session_id,
                "message": {"role": "user", "content": "first"},
            },
            {
                "type": "assistant",
                "sessionId": session_id,
                "message": {"role": "assistant", "content": "second"},
            },
            {
                "type": "user",
                "sessionId": session_id,
                "message": {"role": "user", "content": "third"},
            },
        ]
        _write_jsonl(base / f"{session_id}.jsonl", records)
        entries, _ = transcript.read_transcript("claude", session_id, "/Users/x/proj")
        assert [e.text for e in entries] == ["first", "second", "third"]

    def test_untimestamped_entry_inherits_the_nearer_neighbour_not_always_preceding(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(transcript, "CLAUDE_PROJECTS_DIR", tmp_path)
        session_id = "sess-nearest"
        base = tmp_path / "-Users-x-proj"

        def rec(role, text, ts=None):
            record = {"type": role, "sessionId": session_id, "message": {"role": role, "content": text}}
            if ts is not None:
                record["timestamp"] = ts
            return record

        # Main file: prev(t=0) is one position from "gapX" and two from
        # "missing"; nxt(t=30) is one position from "missing". A
        # cross-file entry sits between them at t=20. Correct nearest-
        # neighbour inheritance puts "missing" next to nxt (after t5);
        # inheriting only ever from the preceding entry (the old bug)
        # would instead sort it with prev, before t5.
        _write_jsonl(
            base / f"{session_id}.jsonl",
            [
                rec("user", "prev", "2026-08-16T00:00:00Z"),
                rec("user", "gapX"),
                rec("user", "missing"),
                rec("assistant", "nxt", "2026-08-16T00:00:30Z"),
            ],
        )
        _write_jsonl(
            base / session_id / "subagents" / "agent-t5.jsonl",
            [rec("assistant", "t5", "2026-08-16T00:00:20Z")],
        )
        entries, _ = transcript.read_transcript("claude", session_id, "/Users/x/proj")
        assert [e.text for e in entries] == ["prev", "gapX", "t5", "missing", "nxt"]

    def test_large_tool_output_is_bounded(self, tmp_path, monkeypatch):
        monkeypatch.setattr(transcript, "CLAUDE_PROJECTS_DIR", tmp_path)
        session_id = "sess-bigout"
        path = tmp_path / "-Users-x-proj" / f"{session_id}.jsonl"
        huge = "\n".join(f"row {i}" for i in range(5000))
        _write_jsonl(
            path,
            [
                {
                    "type": "assistant",
                    "sessionId": session_id,
                    "timestamp": "2026-08-16T00:00:00Z",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {"type": "tool_use", "id": "t1", "name": "Read", "input": {"filePath": "/big.txt"}}
                        ],
                    },
                },
                {
                    "type": "user",
                    "sessionId": session_id,
                    "timestamp": "2026-08-16T00:00:01Z",
                    "message": {
                        "role": "user",
                        "content": [
                            {"tool_use_id": "t1", "type": "tool_result", "content": [{"type": "text", "text": huge}]}
                        ],
                    },
                },
            ],
        )
        entries, _ = transcript.read_transcript("claude", session_id, "/Users/x/proj")
        tool_entries = [e for e in entries if e.type == "tool"]
        assert len(tool_entries) == 1
        assert "truncated" in tool_entries[0].tool_output
        assert "row 4999" not in tool_entries[0].tool_output


# ---------------------------------------------------------------------------
# codex reader
# ---------------------------------------------------------------------------

class TestCodexReader:
    def test_reads_message_reasoning_and_function_call_with_matched_output(self, tmp_path, monkeypatch):
        monkeypatch.setattr(transcript, "CODEX_SESSIONS_DIR", tmp_path)
        session_id = "019d0341-7dd0-7a90-ac00-41507ae74d8f"
        path = tmp_path / "2026" / "08" / "19" / f"rollout-2026-08-19T00-00-00-{session_id}.jsonl"
        records = [
            {
                "timestamp": "2026-08-19T00:00:00Z",
                "type": "response_item",
                "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "say hi"}]},
            },
            {
                "timestamp": "2026-08-19T00:00:01Z",
                "type": "response_item",
                "payload": {"type": "reasoning", "summary": ["thinking about it"]},
            },
            {
                "timestamp": "2026-08-19T00:00:02Z",
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "name": "exec_command",
                    "call_id": "call_1",
                    "arguments": json.dumps({"cmd": "echo hi", "workdir": "/tmp"}),
                },
            },
            {
                "timestamp": "2026-08-19T00:00:03Z",
                "type": "response_item",
                "payload": {"type": "function_call_output", "call_id": "call_1", "output": "hi\n"},
            },
            {
                "timestamp": "2026-08-19T00:00:04Z",
                "type": "response_item",
                "payload": {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "hi"}]},
            },
            # event_msg records duplicate response_item content and must be skipped.
            {"timestamp": "2026-08-19T00:00:04Z", "type": "event_msg", "payload": {"type": "agent_message", "message": "hi"}},
        ]
        _write_jsonl(path, records)

        entries, skipped = transcript.read_transcript("codex", session_id, None)
        assert skipped == 0
        assert [e.type for e in entries] == ["user", "reasoning", "tool", "assistant"]
        assert entries[2].tool_name == "exec_command"
        assert entries[2].tool_summary == "echo hi"
        assert entries[2].tool_output == "hi\n"
        assert entries[2].tool_status == "completed"

    def test_developer_role_messages_excluded(self, tmp_path, monkeypatch):
        monkeypatch.setattr(transcript, "CODEX_SESSIONS_DIR", tmp_path)
        session_id = "dev-role-session"
        path = tmp_path / "2026" / "08" / "19" / f"rollout-2026-08-19T00-00-00-{session_id}.jsonl"
        records = [
            {
                "timestamp": "2026-08-19T00:00:00Z",
                "type": "response_item",
                "payload": {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "<permissions>"}]},
            },
            {
                "timestamp": "2026-08-19T00:00:01Z",
                "type": "response_item",
                "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]},
            },
        ]
        _write_jsonl(path, records)
        entries, _ = transcript.read_transcript("codex", session_id, None)
        assert [e.text for e in entries] == ["hi"]

    def test_injected_context_wrapper_suppressed_genuine_message_preserved(self, tmp_path, monkeypatch):
        monkeypatch.setattr(transcript, "CODEX_SESSIONS_DIR", tmp_path)
        session_id = "injected-context-session"
        path = tmp_path / "2026" / "08" / "19" / f"rollout-2026-08-19T00-00-00-{session_id}.jsonl"

        def msg_record(text, ts):
            return {
                "timestamp": ts,
                "type": "response_item",
                "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]},
            }

        real_boilerplate = "<environment_context>\n  <cwd>/tmp</cwd>\n</environment_context>"
        starts_with_tag_but_not_wrapped = "<environment_context> is a tag codex uses, and here's my question"
        merely_mentions_tag = "Why does codex inject <environment_context> into every session?"
        records = [
            msg_record(real_boilerplate, "2026-08-19T00:00:00Z"),
            msg_record(starts_with_tag_but_not_wrapped, "2026-08-19T00:00:01Z"),
            msg_record(merely_mentions_tag, "2026-08-19T00:00:02Z"),
        ]
        _write_jsonl(path, records)
        entries, _ = transcript.read_transcript("codex", session_id, None)
        assert [e.text for e in entries] == [starts_with_tag_but_not_wrapped, merely_mentions_tag]

    def test_consecutive_wrappers_in_one_message_both_suppressed(self, tmp_path, monkeypatch):
        monkeypatch.setattr(transcript, "CODEX_SESSIONS_DIR", tmp_path)
        session_id = "double-wrapper-session"
        path = tmp_path / "2026" / "08" / "19" / f"rollout-2026-08-19T00-00-00-{session_id}.jsonl"

        def msg_record(text, ts):
            return {
                "timestamp": ts,
                "type": "response_item",
                "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]},
            }

        # Real codex sessions can put both wrappers in one message: it
        # starts with <environment_context> and ends with
        # </user_instructions>, so a check requiring one tag pair to span
        # the whole message misses this case.
        double_wrapper = (
            "<environment_context>\n  <cwd>/tmp</cwd>\n</environment_context>\n"
            "<user_instructions>\nBe concise.\n</user_instructions>"
        )
        wrapper_then_real_text = double_wrapper + "\nplease also fix the bug in foo.py"
        records = [
            msg_record(double_wrapper, "2026-08-19T00:00:00Z"),
            msg_record(wrapper_then_real_text, "2026-08-19T00:00:01Z"),
        ]
        _write_jsonl(path, records)
        entries, _ = transcript.read_transcript("codex", session_id, None)
        assert [e.text for e in entries] == [wrapper_then_real_text]

    def test_missing_store_raises_source_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr(transcript, "CODEX_SESSIONS_DIR", tmp_path)
        with pytest.raises(transcript.TranscriptSourceError):
            transcript.read_transcript("codex", "no-such-session", None)

    @ROOT_SKIP
    def test_unreadable_sessions_dir_raises_store_unreadable_not_missing(self, tmp_path, monkeypatch):
        """CODEX_SESSIONS_DIR itself unreadable (permission-denied scandir)
        is not evidence the session is absent."""
        root = tmp_path / "sessions"
        root.mkdir()
        monkeypatch.setattr(transcript, "CODEX_SESSIONS_DIR", root)
        with _unreadable_dir(root):
            with pytest.raises(transcript.TranscriptSourceError) as exc_info:
                transcript.read_transcript("codex", "some-session", None)
        assert exc_info.value.code == "store_unreadable"
        assert exc_info.value.__cause__ is not None

    def test_empty_session_file_returns_no_entries(self, tmp_path, monkeypatch):
        monkeypatch.setattr(transcript, "CODEX_SESSIONS_DIR", tmp_path)
        session_id = "empty-session"
        path = tmp_path / "2026" / "08" / "19" / f"rollout-2026-08-19T00-00-00-{session_id}.jsonl"
        _write_jsonl(path, [])
        entries, skipped = transcript.read_transcript("codex", session_id, None)
        assert entries == []
        assert skipped == 0

    def test_corrupt_record_mid_file_is_skipped(self, tmp_path, monkeypatch):
        monkeypatch.setattr(transcript, "CODEX_SESSIONS_DIR", tmp_path)
        session_id = "corrupt-session"
        path = tmp_path / "2026" / "08" / "19" / f"rollout-2026-08-19T00-00-00-{session_id}.jsonl"
        good1 = json.dumps(
            {
                "timestamp": "2026-08-19T00:00:00Z",
                "type": "response_item",
                "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "first"}]},
            }
        )
        good2 = json.dumps(
            {
                "timestamp": "2026-08-19T00:00:01Z",
                "type": "response_item",
                "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "second"}]},
            }
        )
        path.parent.mkdir(parents=True)
        path.write_text(good1 + "\n" + "not { json\n" + good2 + "\n")
        entries, skipped = transcript.read_transcript("codex", session_id, None)
        assert skipped == 1
        assert [e.text for e in entries] == ["first", "second"]

    def test_large_tool_output_is_bounded(self, tmp_path, monkeypatch):
        monkeypatch.setattr(transcript, "CODEX_SESSIONS_DIR", tmp_path)
        session_id = "bigout-session"
        path = tmp_path / "2026" / "08" / "19" / f"rollout-2026-08-19T00-00-00-{session_id}.jsonl"
        huge = "\n".join(f"l{i}" for i in range(2000))
        records = [
            {
                "timestamp": "2026-08-19T00:00:00Z",
                "type": "response_item",
                "payload": {"type": "function_call", "name": "exec_command", "call_id": "c1", "arguments": "{\"cmd\":\"cat big\"}"},
            },
            {
                "timestamp": "2026-08-19T00:00:01Z",
                "type": "response_item",
                "payload": {"type": "function_call_output", "call_id": "c1", "output": huge},
            },
        ]
        _write_jsonl(path, records)
        entries, _ = transcript.read_transcript("codex", session_id, None)
        assert len(entries) == 1
        assert "truncated" in entries[0].tool_output
        assert "l1999" not in entries[0].tool_output


# ---------------------------------------------------------------------------
# dispatch / unknown harness
# ---------------------------------------------------------------------------

def test_unrecognised_harness_raises_source_error():
    with pytest.raises(transcript.TranscriptSourceError):
        transcript.read_transcript("gemini-cli", "some-session", None)


# ---------------------------------------------------------------------------
# cmd_transcript CLI integration
# ---------------------------------------------------------------------------

class TestCmdTranscript:
    def _args(self, name, **overrides):
        base = dict(name=name, tail=None, head=None, json=False)
        base.update(overrides)
        return argparse.Namespace(**base)

    def test_no_run_named_exits_nonzero(self, isolated_runs_root, isolated_log_root):
        with pytest.raises(SystemExit):
            agent_run.cmd_transcript(self._args("nosuchrun"))

    def test_raw_run_with_no_session_json_exits_with_two_alternatives(
        self, isolated_runs_root, isolated_log_root
    ):
        name = "rawrun"
        (isolated_runs_root / name).mkdir(parents=True)
        (isolated_runs_root / name / "status").write_text("done\n")
        (isolated_log_root / name).mkdir(parents=True)
        with pytest.raises(SystemExit) as exc_info:
            agent_run.cmd_transcript(self._args(name))
        message = str(exc_info.value)
        assert "--harness" in message
        assert "logs" in message and "--clean" in message

    def test_session_json_with_missing_session_id_exits_with_two_alternatives(
        self, isolated_runs_root, isolated_log_root
    ):
        name = "missingid"
        (isolated_runs_root / name).mkdir(parents=True)
        (isolated_runs_root / name / "status").write_text("done\n")
        log_dir = isolated_log_root / name
        log_dir.mkdir(parents=True)
        agent_run._write_session_json(
            log_dir,
            {"session_id": None, "harness": "opencode", "acquisition": "missing", "confidence": "missing"},
        )
        with pytest.raises(SystemExit) as exc_info:
            agent_run.cmd_transcript(self._args(name))
        message = str(exc_info.value)
        assert "--harness" in message

    def test_unknown_harness_in_session_json_exits_nonzero(
        self, isolated_runs_root, isolated_log_root
    ):
        name = "weirdharness"
        (isolated_runs_root / name).mkdir(parents=True)
        (isolated_runs_root / name / "status").write_text("done\n")
        log_dir = isolated_log_root / name
        log_dir.mkdir(parents=True)
        agent_run._write_session_json(
            log_dir,
            {
                "session_id": "abc",
                "harness": "some-future-harness",
                "acquisition": "minted",
                "confidence": "certain",
            },
        )
        with pytest.raises(SystemExit):
            agent_run.cmd_transcript(self._args(name))

    def test_missing_store_exits_nonzero(self, isolated_runs_root, isolated_log_root, monkeypatch, tmp_path):
        monkeypatch.setattr(transcript, "OPENCODE_DB_PATH", tmp_path / "no.db")
        name = "opencoderun"
        (isolated_runs_root / name).mkdir(parents=True)
        (isolated_runs_root / name / "status").write_text("done\n")
        log_dir = isolated_log_root / name
        log_dir.mkdir(parents=True)
        agent_run._write_session_json(
            log_dir,
            {"session_id": "ses_1", "harness": "opencode", "acquisition": "minted", "confidence": "certain"},
        )
        with pytest.raises(SystemExit):
            agent_run.cmd_transcript(self._args(name))

    def test_renders_plain_text(
        self, isolated_runs_root, isolated_log_root, monkeypatch, tmp_path, capsysbinary
    ):
        db = tmp_path / "opencode.db"
        _make_opencode_db(db)
        monkeypatch.setattr(transcript, "OPENCODE_DB_PATH", db)
        _opencode_insert(
            db, message_id="m1", session_id="ses_ok", role="user", time_created=1000,
            parts=[{"type": "text", "text": "hello there"}],
        )

        name = "goodrun"
        (isolated_runs_root / name).mkdir(parents=True)
        (isolated_runs_root / name / "status").write_text("done\n")
        log_dir = isolated_log_root / name
        log_dir.mkdir(parents=True)
        agent_run._write_session_json(
            log_dir,
            {"session_id": "ses_ok", "harness": "opencode", "acquisition": "minted", "confidence": "certain"},
        )

        rc = agent_run.cmd_transcript(self._args(name))
        assert rc == 0
        out = capsysbinary.readouterr().out.decode("utf-8")
        assert "hello there" in out
        assert "user:" in out

    def test_json_output_emits_one_object_per_line(
        self, isolated_runs_root, isolated_log_root, monkeypatch, tmp_path, capsysbinary
    ):
        db = tmp_path / "opencode.db"
        _make_opencode_db(db)
        monkeypatch.setattr(transcript, "OPENCODE_DB_PATH", db)
        _opencode_insert(
            db, message_id="m1", session_id="ses_json", role="user", time_created=1000,
            parts=[{"type": "text", "text": "first"}],
        )
        _opencode_insert(
            db, message_id="m2", session_id="ses_json", role="assistant", time_created=2000,
            parts=[{"type": "text", "text": "second"}],
        )

        name = "jsonrun"
        (isolated_runs_root / name).mkdir(parents=True)
        (isolated_runs_root / name / "status").write_text("done\n")
        log_dir = isolated_log_root / name
        log_dir.mkdir(parents=True)
        agent_run._write_session_json(
            log_dir,
            {"session_id": "ses_json", "harness": "opencode", "acquisition": "minted", "confidence": "certain"},
        )

        rc = agent_run.cmd_transcript(self._args(name, json=True))
        assert rc == 0
        out = capsysbinary.readouterr().out.decode("utf-8")
        lines = [line for line in out.splitlines() if line.strip()]
        assert len(lines) == 2
        parsed = [json.loads(line) for line in lines]
        assert parsed[0]["text"] == "first"
        assert parsed[1]["text"] == "second"
        assert parsed[0]["harness"] == "opencode"

    def test_tail_and_head_slice_entries_not_bytes(
        self, isolated_runs_root, isolated_log_root, monkeypatch, tmp_path, capsysbinary
    ):
        db = tmp_path / "opencode.db"
        _make_opencode_db(db)
        monkeypatch.setattr(transcript, "OPENCODE_DB_PATH", db)
        for i in range(10):
            _opencode_insert(
                db, message_id=f"m{i}", session_id="ses_slice", role="user", time_created=1000 + i,
                parts=[{"type": "text", "text": f"entry {i}"}],
            )
        name = "slicerun"
        (isolated_runs_root / name).mkdir(parents=True)
        (isolated_runs_root / name / "status").write_text("done\n")
        log_dir = isolated_log_root / name
        log_dir.mkdir(parents=True)
        agent_run._write_session_json(
            log_dir,
            {"session_id": "ses_slice", "harness": "opencode", "acquisition": "minted", "confidence": "certain"},
        )

        rc = agent_run.cmd_transcript(self._args(name, head=3, json=True))
        assert rc == 0
        out = capsysbinary.readouterr().out.decode("utf-8")
        parsed = [json.loads(line) for line in out.splitlines() if line.strip()]
        assert [p["text"] for p in parsed] == ["entry 0", "entry 1", "entry 2"]

        rc = agent_run.cmd_transcript(self._args(name, tail=2, json=True))
        assert rc == 0
        out = capsysbinary.readouterr().out.decode("utf-8")
        parsed = [json.loads(line) for line in out.splitlines() if line.strip()]
        assert [p["text"] for p in parsed] == ["entry 8", "entry 9"]

    def test_skipped_records_reported_on_stderr(
        self, isolated_runs_root, isolated_log_root, monkeypatch, tmp_path, capsysbinary
    ):
        db = tmp_path / "opencode.db"
        _make_opencode_db(db)
        monkeypatch.setattr(transcript, "OPENCODE_DB_PATH", db)
        conn = sqlite3.connect(db)
        try:
            conn.execute(
                "insert into message (id, session_id, time_created, data) values (?, ?, ?, ?)",
                ("m1", "ses_skip", 1, json.dumps({"role": "user", "time": {"created": 1}})),
            )
            conn.execute(
                "insert into part (id, message_id, session_id, time_created, data) values (?, ?, ?, ?, ?)",
                ("p1", "m1", "ses_skip", 1, "{not json"),
            )
            conn.commit()
        finally:
            conn.close()
        # A record the reader can use, alongside the unparseable one above:
        # this run must still succeed and merely report the skip count.
        _opencode_insert(
            db, message_id="m2", session_id="ses_skip", role="user", time_created=2,
            parts=[{"type": "text", "text": "still readable"}],
        )

        name = "skiprun"
        (isolated_runs_root / name).mkdir(parents=True)
        (isolated_runs_root / name / "status").write_text("done\n")
        log_dir = isolated_log_root / name
        log_dir.mkdir(parents=True)
        agent_run._write_session_json(
            log_dir,
            {"session_id": "ses_skip", "harness": "opencode", "acquisition": "minted", "confidence": "certain"},
        )

        rc = agent_run.cmd_transcript(self._args(name))
        assert rc == 0
        captured = capsysbinary.readouterr()
        stderr = captured.err.decode("utf-8")
        assert "skipped 1" in stderr

    def test_empty_but_valid_store_exits_zero_with_stderr_note_and_no_stdout(
        self, isolated_runs_root, isolated_log_root, monkeypatch, tmp_path, capsysbinary
    ):
        db = tmp_path / "opencode.db"
        _make_opencode_db(db)
        monkeypatch.setattr(transcript, "OPENCODE_DB_PATH", db)

        name = "emptyrun"
        (isolated_runs_root / name).mkdir(parents=True)
        (isolated_runs_root / name / "status").write_text("done\n")
        log_dir = isolated_log_root / name
        log_dir.mkdir(parents=True)
        agent_run._write_session_json(
            log_dir,
            {"session_id": "ses_gone", "harness": "opencode", "acquisition": "minted", "confidence": "certain"},
        )

        rc = agent_run.cmd_transcript(self._args(name))
        assert rc == 0
        captured = capsysbinary.readouterr()
        assert captured.out == b""
        message = captured.err.decode("utf-8")
        assert name in message
        assert "opencode" in message
        assert "ses_gone" in message
        assert str(db) in message

    def test_skipped_only_store_exits_zero_with_skipped_count_on_stderr(
        self, isolated_runs_root, isolated_log_root, monkeypatch, tmp_path, capsysbinary
    ):
        db = tmp_path / "opencode.db"
        _make_opencode_db(db)
        monkeypatch.setattr(transcript, "OPENCODE_DB_PATH", db)
        conn = sqlite3.connect(db)
        try:
            conn.execute(
                "insert into message (id, session_id, time_created, data) values (?, ?, ?, ?)",
                ("m1", "ses_onlyskip", 1, json.dumps({"role": "user", "time": {"created": 1}})),
            )
            conn.execute(
                "insert into part (id, message_id, session_id, time_created, data) values (?, ?, ?, ?, ?)",
                ("p1", "m1", "ses_onlyskip", 1, "{not json"),
            )
            conn.commit()
        finally:
            conn.close()

        name = "onlyskiprun"
        (isolated_runs_root / name).mkdir(parents=True)
        (isolated_runs_root / name / "status").write_text("done\n")
        log_dir = isolated_log_root / name
        log_dir.mkdir(parents=True)
        agent_run._write_session_json(
            log_dir,
            {"session_id": "ses_onlyskip", "harness": "opencode", "acquisition": "minted", "confidence": "certain"},
        )

        rc = agent_run.cmd_transcript(self._args(name))
        assert rc == 0
        captured = capsysbinary.readouterr()
        assert captured.out == b""
        message = captured.err.decode("utf-8")
        assert "1" in message
        assert "skip" in message.lower()


# ---------------------------------------------------------------------------
# render_text
# ---------------------------------------------------------------------------

class TestRenderText:
    def test_renders_chronological_speaker_lines_and_tool_output(self):
        entries = [
            transcript.TranscriptEntry(type="user", harness="opencode", text="hi", time="2026-08-16T00:00:00Z"),
            transcript.TranscriptEntry(
                type="tool",
                harness="opencode",
                tool_name="bash",
                tool_summary="echo hi",
                tool_output="hi\n",
                tool_status="completed",
            ),
        ]
        text = transcript.render_text(entries)
        lines = text.splitlines()
        assert any("user:" in line and "hi" in line for line in lines)
        assert any("bash" in line and "echo hi" in line for line in lines)
        assert any(line.strip() == "hi" for line in lines)
