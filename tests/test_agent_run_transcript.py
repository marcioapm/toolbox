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
import json
import sqlite3
import time
from pathlib import Path

import pytest

from toolbox import agent_run
from toolbox import agent_run_transcript as transcript


# ---------------------------------------------------------------------------
# fixture builders
# ---------------------------------------------------------------------------

def _make_opencode_db(path: Path) -> None:
    """Minimal `part`/`message` tables matching the real opencode schema
    (only the columns this reader touches)."""
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

    def test_missing_store_raises_source_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr(transcript, "CODEX_SESSIONS_DIR", tmp_path)
        with pytest.raises(transcript.TranscriptSourceError):
            transcript.read_transcript("codex", "no-such-session", None)

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
