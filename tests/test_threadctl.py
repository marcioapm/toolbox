"""Tests for toolbox.threadctl: SQLite core (WAL/busy_timeout, schema),
bind/touch/set/add-pr/rm-pr/primary-pr/show/list, icon + title rendering,
staleness, GitHub polling + cycle, Discord writes + rename reconciliation,
and migration from the old thread-state registry.json. All Discord/GitHub
calls are mocked — no live API calls."""
from __future__ import annotations

import json
import sqlite3

import pytest

from toolbox import threadctl as tc


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "sub" / "state.db"


def run_cli(argv, db_path, capsys):
    full = [argv[0], *argv[1:], "--db", str(db_path)]
    code = tc.main(full)
    out = capsys.readouterr()
    return code, out.out, out.err


def _bind(db_path, capsys, thread_id="t1", title="Experiment recompute", repo="absmartly/abs"):
    return run_cli(["bind", thread_id, "--title", title, "--repo", repo], db_path, capsys)


# ---------------------------------------------------------------------------
# database: connection, schema, WAL
# ---------------------------------------------------------------------------

class TestDatabaseBasics:
    def test_connect_creates_parent_dirs_and_schema(self, db_path):
        conn = tc.connect_db(db_path)
        try:
            assert db_path.exists()
            tables = {
                r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
            assert {"threads", "prs", "history", "meta"} <= tables
        finally:
            conn.close()

    def test_wal_mode_enabled(self, db_path):
        conn = tc.connect_db(db_path)
        try:
            mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
            assert mode.lower() == "wal"
        finally:
            conn.close()

    def test_busy_timeout_set(self, db_path):
        conn = tc.connect_db(db_path)
        try:
            timeout = conn.execute("PRAGMA busy_timeout").fetchone()[0]
            assert timeout > 0
        finally:
            conn.close()

    def test_db_env_var_resolution(self, tmp_path, monkeypatch, capsys):
        env_db = tmp_path / "envstate.db"
        monkeypatch.setenv("THREADCTL_DB", str(env_db))
        code = tc.main(["bind", "t1", "--title", "x", "--repo", "absmartly/abs"])
        assert code == 0
        assert env_db.exists()

    def test_explicit_db_flag_wins_over_env(self, tmp_path, monkeypatch, capsys):
        env_db = tmp_path / "envstate.db"
        flag_db = tmp_path / "flagstate.db"
        monkeypatch.setenv("THREADCTL_DB", str(env_db))
        code, out, err = run_cli(["bind", "t1", "--title", "x", "--repo", "absmartly/abs"], flag_db, capsys)
        assert code == 0
        assert flag_db.exists()
        assert not env_db.exists()


# ---------------------------------------------------------------------------
# bind
# ---------------------------------------------------------------------------

class TestBind:
    def test_bind_creates_thread(self, db_path, capsys):
        code, out, err = _bind(db_path, capsys)
        assert code == 0
        conn = tc.connect_db(db_path)
        try:
            row = tc.get_thread(conn, "t1")
            assert row["title"] == "Experiment recompute"
            assert row["owner"] == "absmartly"
            assert row["repo"] == "absmartly/abs"
            assert row["status"] == "active"
            assert row["question"] == 0
        finally:
            conn.close()

    def test_bind_records_history(self, db_path, capsys):
        _bind(db_path, capsys)
        conn = tc.connect_db(db_path)
        try:
            rows = conn.execute("SELECT * FROM history WHERE thread_id='t1'").fetchall()
            assert len(rows) == 1
            assert rows[0]["kind"] == "bind"
        finally:
            conn.close()

    def test_bind_rejects_invalid_repo(self, db_path, capsys):
        code, out, err = run_cli(["bind", "t1", "--title", "x", "--repo", "not-a-repo"], db_path, capsys)
        assert code == 2
        assert "invalid --repo" in err

    def test_bind_truncates_long_title(self, db_path, capsys):
        long_title = "x" * 60
        code, out, err = run_cli(["bind", "t1", "--title", long_title, "--repo", "absmartly/abs"], db_path, capsys)
        assert code == 0
        conn = tc.connect_db(db_path)
        try:
            row = tc.get_thread(conn, "t1")
            assert len(row["title"]) == 48
            assert row["title"].endswith("…")
        finally:
            conn.close()
        assert "truncated" in err

    def test_rebind_is_destructive_fresh_start(self, db_path, capsys):
        _bind(db_path, capsys)
        conn = tc.connect_db(db_path)
        try:
            with conn:
                conn.execute(
                    "INSERT INTO prs (thread_id, repo, number, is_primary, created_at) VALUES (?, ?, ?, 1, ?)",
                    ("t1", "absmartly/abs", 100, tc._now_iso()),
                )
                conn.execute("UPDATE threads SET status='paused', question=1, pin_message_id='999' WHERE thread_id='t1'")
        finally:
            conn.close()

        code, out, err = run_cli(["bind", "t1", "--title", "New work", "--repo", "absmartly/other"], db_path, capsys)
        assert code == 0
        assert "WARNING" in err
        assert "fresh start" in err

        conn = tc.connect_db(db_path)
        try:
            row = tc.get_thread(conn, "t1")
            assert row["title"] == "New work"
            assert row["repo"] == "absmartly/other"
            assert row["status"] == "active"
            assert row["question"] == 0
            prs = tc.get_prs(conn, "t1")
            assert prs == []
            history = conn.execute(
                "SELECT * FROM history WHERE thread_id='t1' ORDER BY id"
            ).fetchall()
            assert len(history) == 2
            assert history[1]["kind"] == "bind"
            assert "discarded" in history[1]["text"].lower() or "100" in history[1]["text"]
        finally:
            conn.close()

    def test_rebind_keeps_thread_row_and_history(self, db_path, capsys):
        _bind(db_path, capsys)
        run_cli(["touch", "t1"], db_path, capsys)
        _bind(db_path, capsys, title="Rebound")
        conn = tc.connect_db(db_path)
        try:
            history = conn.execute("SELECT * FROM history WHERE thread_id='t1'").fetchall()
            # bind, touch, bind (rebind) — history survives across rebind
            assert len(history) == 3
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# touch
# ---------------------------------------------------------------------------

class TestTouch:
    def test_touch_updates_last_touch_at(self, db_path, capsys, monkeypatch):
        _bind(db_path, capsys)
        conn = tc.connect_db(db_path)
        try:
            before = tc.get_thread(conn, "t1")["last_touch_at"]
        finally:
            conn.close()

        monkeypatch.setattr(tc, "_now_iso", lambda: "2026-07-28T12:00:00Z")
        code, out, err = run_cli(["touch", "t1"], db_path, capsys)
        assert code == 0

        conn = tc.connect_db(db_path)
        try:
            after = tc.get_thread(conn, "t1")["last_touch_at"]
        finally:
            conn.close()
        assert after == "2026-07-28T12:00:00Z"

    def test_touch_does_not_clear_question_flag(self, db_path, capsys):
        _bind(db_path, capsys)
        run_cli(["set", "t1", "--question"], db_path, capsys)
        run_cli(["touch", "t1"], db_path, capsys)
        conn = tc.connect_db(db_path)
        try:
            assert tc.get_thread(conn, "t1")["question"] == 1
        finally:
            conn.close()

    def test_touch_unbound_thread_errors(self, db_path, capsys):
        code, out, err = run_cli(["touch", "nope"], db_path, capsys)
        assert code == 1
        assert "no such thread" in err

    def test_touch_records_note_in_history(self, db_path, capsys):
        _bind(db_path, capsys)
        run_cli(["touch", "t1", "--note", "still working"], db_path, capsys)
        conn = tc.connect_db(db_path)
        try:
            rows = conn.execute("SELECT * FROM history WHERE thread_id='t1' AND kind='touch'").fetchall()
            assert len(rows) == 1
            assert rows[0]["text"] == "still working"
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# set
# ---------------------------------------------------------------------------

class TestSet:
    def test_set_title(self, db_path, capsys):
        _bind(db_path, capsys)
        run_cli(["set", "t1", "--title", "New title"], db_path, capsys)
        conn = tc.connect_db(db_path)
        try:
            assert tc.get_thread(conn, "t1")["title"] == "New title"
        finally:
            conn.close()

    def test_set_status(self, db_path, capsys):
        _bind(db_path, capsys)
        code, out, err = run_cli(["set", "t1", "--status", "paused"], db_path, capsys)
        assert code == 0
        conn = tc.connect_db(db_path)
        try:
            assert tc.get_thread(conn, "t1")["status"] == "paused"
        finally:
            conn.close()

    def test_set_invalid_status_rejected(self, db_path, capsys):
        _bind(db_path, capsys)
        with pytest.raises(SystemExit) as exc_info:
            run_cli(["set", "t1", "--status", "blocked"], db_path, capsys)
        assert exc_info.value.code == 2

    def test_set_question_raises_flag(self, db_path, capsys):
        _bind(db_path, capsys)
        run_cli(["set", "t1", "--question"], db_path, capsys)
        conn = tc.connect_db(db_path)
        try:
            assert tc.get_thread(conn, "t1")["question"] == 1
        finally:
            conn.close()

    def test_set_answered_clears_flag(self, db_path, capsys):
        _bind(db_path, capsys)
        run_cli(["set", "t1", "--question"], db_path, capsys)
        run_cli(["set", "t1", "--answered"], db_path, capsys)
        conn = tc.connect_db(db_path)
        try:
            assert tc.get_thread(conn, "t1")["question"] == 0
        finally:
            conn.close()

    def test_set_question_and_answered_together_rejected(self, db_path, capsys):
        _bind(db_path, capsys)
        code, out, err = run_cli(["set", "t1", "--question", "--answered"], db_path, capsys)
        assert code == 2

    def test_set_unbound_thread_errors(self, db_path, capsys):
        code, out, err = run_cli(["set", "nope", "--status", "paused"], db_path, capsys)
        assert code == 1

    def test_set_updates_last_touch_at(self, db_path, capsys, monkeypatch):
        _bind(db_path, capsys)
        monkeypatch.setattr(tc, "_now_iso", lambda: "2026-07-28T13:00:00Z")
        run_cli(["set", "t1", "--status", "paused"], db_path, capsys)
        conn = tc.connect_db(db_path)
        try:
            assert tc.get_thread(conn, "t1")["last_touch_at"] == "2026-07-28T13:00:00Z"
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# PR ref parsing
# ---------------------------------------------------------------------------

class TestParsePrRef:
    def test_bare_number(self):
        assert tc.parse_pr_ref("4615", "absmartly/abs") == ("absmartly/abs", 4615)

    def test_hash_number(self):
        assert tc.parse_pr_ref("#4615", "absmartly/abs") == ("absmartly/abs", 4615)

    def test_repo_qualified(self):
        assert tc.parse_pr_ref("absmartly/other#4615", "absmartly/abs") == ("absmartly/other", 4615)

    def test_full_url(self):
        assert tc.parse_pr_ref("https://github.com/absmartly/abs/pull/4615") == ("absmartly/abs", 4615)

    def test_no_default_repo_returns_none(self):
        assert tc.parse_pr_ref("4615") == (None, 4615)

    def test_invalid_ref_raises(self):
        with pytest.raises(ValueError):
            tc.parse_pr_ref("not-a-ref")


# ---------------------------------------------------------------------------
# add-pr / rm-pr / primary-pr
# ---------------------------------------------------------------------------

class TestAddPr:
    def test_add_pr_bare_number_resolves_thread_repo(self, db_path, capsys):
        _bind(db_path, capsys)
        code, out, err = run_cli(["add-pr", "t1", "4615"], db_path, capsys)
        assert code == 0
        conn = tc.connect_db(db_path)
        try:
            prs = tc.get_prs(conn, "t1")
            assert len(prs) == 1
            assert prs[0]["repo"] == "absmartly/abs"
            assert prs[0]["number"] == 4615
            assert prs[0]["is_primary"] == 1  # first PR is auto-primary
        finally:
            conn.close()

    def test_add_pr_rejects_mismatched_owner(self, db_path, capsys):
        _bind(db_path, capsys)
        code, out, err = run_cli(["add-pr", "t1", "marcioapm/toolbox#5"], db_path, capsys)
        assert code == 2
        assert "owner" in err

    def test_add_pr_explicit_primary_flag(self, db_path, capsys):
        _bind(db_path, capsys)
        run_cli(["add-pr", "t1", "4615"], db_path, capsys)
        run_cli(["add-pr", "t1", "4620", "--primary"], db_path, capsys)
        conn = tc.connect_db(db_path)
        try:
            prs = {p["number"]: p for p in tc.get_prs(conn, "t1")}
            assert prs[4620]["is_primary"] == 1
            assert prs[4615]["is_primary"] == 0
        finally:
            conn.close()

    def test_add_pr_second_pr_not_auto_primary(self, db_path, capsys):
        _bind(db_path, capsys)
        run_cli(["add-pr", "t1", "4615"], db_path, capsys)
        run_cli(["add-pr", "t1", "4620"], db_path, capsys)
        conn = tc.connect_db(db_path)
        try:
            prs = {p["number"]: p for p in tc.get_prs(conn, "t1")}
            assert prs[4615]["is_primary"] == 1
            assert prs[4620]["is_primary"] == 0
        finally:
            conn.close()

    def test_add_pr_idempotent(self, db_path, capsys):
        _bind(db_path, capsys)
        run_cli(["add-pr", "t1", "4615"], db_path, capsys)
        code, out, err = run_cli(["add-pr", "t1", "4615"], db_path, capsys)
        assert code == 0
        conn = tc.connect_db(db_path)
        try:
            assert len(tc.get_prs(conn, "t1")) == 1
        finally:
            conn.close()

    def test_add_pr_unbound_thread_errors(self, db_path, capsys):
        code, out, err = run_cli(["add-pr", "nope", "4615"], db_path, capsys)
        assert code == 1


class TestRmPr:
    def test_rm_pr_removes(self, db_path, capsys):
        _bind(db_path, capsys)
        run_cli(["add-pr", "t1", "4615"], db_path, capsys)
        code, out, err = run_cli(["rm-pr", "t1", "4615"], db_path, capsys)
        assert code == 0
        conn = tc.connect_db(db_path)
        try:
            assert tc.get_prs(conn, "t1") == []
        finally:
            conn.close()

    def test_rm_pr_reassigns_primary(self, db_path, capsys):
        _bind(db_path, capsys)
        run_cli(["add-pr", "t1", "4615"], db_path, capsys)
        run_cli(["add-pr", "t1", "4620"], db_path, capsys)
        run_cli(["rm-pr", "t1", "4615"], db_path, capsys)
        conn = tc.connect_db(db_path)
        try:
            prs = tc.get_prs(conn, "t1")
            assert len(prs) == 1
            assert prs[0]["number"] == 4620
            assert prs[0]["is_primary"] == 1
        finally:
            conn.close()

    def test_rm_pr_not_tracked_errors(self, db_path, capsys):
        _bind(db_path, capsys)
        code, out, err = run_cli(["rm-pr", "t1", "9999"], db_path, capsys)
        assert code == 1


class TestPrimaryPr:
    def test_primary_pr_switches(self, db_path, capsys):
        _bind(db_path, capsys)
        run_cli(["add-pr", "t1", "4615"], db_path, capsys)
        run_cli(["add-pr", "t1", "4620"], db_path, capsys)
        run_cli(["primary-pr", "t1", "4620"], db_path, capsys)
        conn = tc.connect_db(db_path)
        try:
            prs = {p["number"]: p for p in tc.get_prs(conn, "t1")}
            assert prs[4620]["is_primary"] == 1
            assert prs[4615]["is_primary"] == 0
        finally:
            conn.close()

    def test_primary_pr_unmatched_errors(self, db_path, capsys):
        _bind(db_path, capsys)
        run_cli(["add-pr", "t1", "4615"], db_path, capsys)
        code, out, err = run_cli(["primary-pr", "t1", "9999"], db_path, capsys)
        assert code == 1


# ---------------------------------------------------------------------------
# show / list
# ---------------------------------------------------------------------------

class TestShowList:
    def test_show_unbound_thread(self, db_path, capsys):
        code, out, err = run_cli(["show", "nope"], db_path, capsys)
        assert code == 1

    def test_show_json_shape(self, db_path, capsys):
        _bind(db_path, capsys)
        run_cli(["add-pr", "t1", "4615"], db_path, capsys)
        code, out, err = run_cli(["show", "t1", "--json"], db_path, capsys)
        assert code == 0
        data = json.loads(out)
        assert data["threadId"] == "t1"
        assert data["repo"] == "absmartly/abs"
        assert len(data["prs"]) == 1
        assert data["prs"][0]["number"] == 4615

    def test_list_empty(self, db_path, capsys):
        code, out, err = run_cli(["list"], db_path, capsys)
        assert code == 0
        assert "no threads" in out

    def test_list_json(self, db_path, capsys):
        _bind(db_path, capsys, thread_id="t1")
        _bind(db_path, capsys, thread_id="t2", title="Other")
        code, out, err = run_cli(["list", "--json"], db_path, capsys)
        assert code == 0
        data = json.loads(out)
        assert {d["threadId"] for d in data} == {"t1", "t2"}
