"""Tests for toolbox.threadctl: SQLite core (WAL/busy_timeout, schema),
bind/touch/set/add-pr/rm-pr/primary-pr/show/list, icon + title rendering,
staleness, GitHub polling + cycle, Discord writes + rename reconciliation,
and migration from the old thread-state registry.json. All Discord/GitHub
calls are mocked — no live API calls."""
from __future__ import annotations

import http.client
import json
import socket
import sqlite3
import ssl
import urllib.error
from datetime import datetime, timedelta, timezone

import pytest

from toolbox import threadctl as tc

# Captured before any test can monkeypatch tc._resolve_discord_token (the
# safety fixture below stubs it for every other test), so the dedicated
# token-resolution tests can still exercise the real implementation.
_real_resolve_discord_token = tc._resolve_discord_token

# Same idea for _discord_transport: the safety fixture below replaces it
# with a forbidding stub for every other test, but the dedicated transport
# tests need the real implementation to verify its exception handling.
_real_discord_transport = tc._discord_transport


@pytest.fixture(autouse=True)
def _no_live_discord_calls(monkeypatch):
    """Hard safety net: never resolve a real Discord token or hit the real
    HTTP transport, even if a test forgets to stub them explicitly. Real
    Discord threads are registered against this bot — a stray call burns a
    real rate-limit slot."""
    monkeypatch.setattr(tc, "_resolve_discord_token", lambda *a, **k: None)

    def _forbidden_transport(*args, **kwargs):
        raise AssertionError("test attempted a live Discord transport call — stub `transport` explicitly")

    monkeypatch.setattr(tc, "_discord_transport", _forbidden_transport)


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

    def test_busy_timeout_is_30_seconds(self, db_path):
        """CRITICAL 3: raised from 5000ms to 30000ms — 5s was easily
        exceeded by an agent command contending with the cycle's `gh`/
        Discord I/O, surfacing a raw `database is locked` traceback."""
        conn = tc.connect_db(db_path)
        try:
            timeout = conn.execute("PRAGMA busy_timeout").fetchone()[0]
            assert timeout == tc.DB_BUSY_TIMEOUT_MS
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
# CRITICAL 3: locked-DB retry for agent commands, incremental cycle commits
# ---------------------------------------------------------------------------

class TestRetryOnLocked:
    """Agent-facing write commands must retry a couple of times on
    `sqlite3.OperationalError: database is locked/busy` and, failing that,
    print a clean one-line message — never a raw traceback. A failed
    `touch` used to mean a thread silently going 💤 while an agent was
    actively working, the exact false signal threadctl exists to
    eliminate."""

    def test_retries_and_succeeds_after_transient_lock(self):
        calls = {"n": 0}

        @tc._retry_on_locked(retries=3, backoff_s=0)
        def flaky(args):
            calls["n"] += 1
            if calls["n"] < 2:
                raise sqlite3.OperationalError("database is locked")
            return 0

        assert flaky(None) == 0
        assert calls["n"] == 2

    def test_gives_up_after_retries_with_clean_message_not_traceback(self, capsys):
        @tc._retry_on_locked(retries=2, backoff_s=0)
        def always_locked(args):
            raise sqlite3.OperationalError("database is locked")

        code = always_locked(None)
        assert code == 1
        err = capsys.readouterr().err
        assert "database busy" in err
        assert "Traceback" not in err

    def test_non_lock_operational_errors_are_not_swallowed(self):
        @tc._retry_on_locked(retries=3, backoff_s=0)
        def other_error(args):
            raise sqlite3.OperationalError("no such table: threads")

        with pytest.raises(sqlite3.OperationalError):
            other_error(None)

    def test_touch_command_is_wrapped_with_retry(self, db_path, capsys):
        assert tc.cmd_touch.__wrapped__ is not None


class TestCycleIncrementalCommits:
    """CRITICAL 3: the cycle must not hold the SQLite write lock across all
    of its network I/O — `_poll_due_prs` commits per repo-batch and
    `reconcile_discord` commits per thread, instead of one `with conn:`
    wrapping the entire cycle. Verified by asserting a concurrent writer on
    a second connection succeeds *during* a slow step of the cycle."""

    def test_recompute_desired_names_skips_update_when_unchanged(self, db_path, capsys, monkeypatch):
        """CRITICAL 3 also calls out: recompute_desired_names rewrote all
        25 thread rows every minute regardless of change. Now it must skip
        the UPDATE (and updated_at bump) when desired_name is unchanged."""
        _bind(db_path, capsys)
        conn = tc.connect_db(db_path)
        try:
            with conn:
                conn.execute(
                    "UPDATE threads SET desired_name='🟢 Experiment recompute', updated_at='2020-01-01T00:00:00Z' "
                    "WHERE thread_id='t1'"
                )
            now = datetime(2026, 7, 28, 12, 0, tzinfo=timezone.utc)
            tc.recompute_desired_names(conn, now, 30)
            conn.commit()
            thread = tc.get_thread(conn, "t1")
            # updated_at must NOT have been touched — the UPDATE was skipped.
            assert thread["updated_at"] == "2020-01-01T00:00:00Z"
        finally:
            conn.close()

    def test_recompute_desired_names_writes_when_changed(self, db_path, capsys):
        _bind(db_path, capsys)
        conn = tc.connect_db(db_path)
        try:
            with conn:
                conn.execute(
                    "UPDATE threads SET desired_name='stale value', updated_at='2020-01-01T00:00:00Z' "
                    "WHERE thread_id='t1'"
                )
            now = datetime(2026, 7, 28, 12, 0, tzinfo=timezone.utc)
            tc.recompute_desired_names(conn, now, 30)
            conn.commit()
            thread = tc.get_thread(conn, "t1")
            assert thread["desired_name"] != "stale value"
            assert thread["updated_at"] != "2020-01-01T00:00:00Z"
        finally:
            conn.close()

    def test_poll_due_prs_commits_between_repo_batches(self, db_path, capsys, monkeypatch):
        """Poll two repos; _poll_due_prs must commit after each repo's
        batch rather than holding one transaction across the whole poll
        (CRITICAL 3) — verified by checking both threads' PR state landed
        even though the poll processes repos one at a time."""
        _bind(db_path, capsys, thread_id="t1", title="One", repo="absmartly/abs")
        run_cli(["add-pr", "t1", "1"], db_path, capsys)
        _bind(db_path, capsys, thread_id="t2", title="Two", repo="marcioapm/toolbox")
        run_cli(["add-pr", "t2", "1"], db_path, capsys)

        now = datetime(2026, 7, 28, 12, 0, tzinfo=timezone.utc)

        def runner(cmd, timeout):
            if cmd[:2] == ["gh", "auth"]:
                return __import__("subprocess").CompletedProcess(cmd, 0, "token\n", "")
            payload = _graphql_payload(pr_1=_github_pr(ci="FAILURE"))
            return __import__("subprocess").CompletedProcess(cmd, 0, json.dumps(payload), "")

        conn = tc.connect_db(db_path)
        try:
            refreshed, errors = tc._poll_due_prs(
                conn, now, {"pr_recheck_min_s": 150, "max_pr_checks_per_cycle": 40}, runner, tc.DEFAULT_ACCOUNTS
            )
            assert refreshed == 2
            assert errors == 0
        finally:
            conn.close()

        # A brand new connection (simulating a concurrent agent command)
        # must see both repos' writes — proving each batch was committed,
        # not held in one uncommitted transaction for the whole poll.
        conn2 = tc.connect_db(db_path)
        try:
            assert tc.get_prs(conn2, "t1")[0]["ci"] == "FAILURE"
            assert tc.get_prs(conn2, "t2")[0]["ci"] == "FAILURE"
        finally:
            conn2.close()


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


# ---------------------------------------------------------------------------
# per-PR icon derivation
# ---------------------------------------------------------------------------

def _pr(number=1, repo=None, primary=False, state="OPEN", ci=None, review=None, mergeable=None, draft=False):
    return {
        "repo": repo, "number": number, "primary": primary, "state": state,
        "ci": ci, "review": review, "mergeable": mergeable, "draft": draft,
    }


class TestPrIcon:
    def test_merged(self):
        assert tc.pr_icon(_pr(state="MERGED")) == "🟣"

    def test_closed_not_merged(self):
        assert tc.pr_icon(_pr(state="CLOSED")) == "⚪"

    def test_open_ci_failure(self):
        assert tc.pr_icon(_pr(ci="FAILURE")) == "🔴"

    def test_open_ci_pending(self):
        assert tc.pr_icon(_pr(ci="PENDING")) == "🟡"

    def test_approved_but_conflicting(self):
        assert tc.pr_icon(_pr(ci="SUCCESS", review="APPROVED", mergeable="CONFLICTING")) == "❗"

    def test_approved_and_mergeable_and_green(self):
        assert tc.pr_icon(_pr(ci="SUCCESS", review="APPROVED", mergeable="MERGEABLE")) == "✅"

    def test_green_ci_not_approved_is_in_review(self):
        assert tc.pr_icon(_pr(ci="SUCCESS", review=None)) == "🔵"

    def test_green_ci_changes_requested_is_in_review(self):
        assert tc.pr_icon(_pr(ci="SUCCESS", review="CHANGES_REQUESTED")) == "🔵"

    def test_draft_no_ci_is_active(self):
        assert tc.pr_icon(_pr(ci=None, review=None, draft=True)) == "🟢"

    def test_ci_failure_beats_approved_mergeable(self):
        # CI failure is checked before the approved+mergeable branch.
        assert tc.pr_icon(_pr(ci="FAILURE", review="APPROVED", mergeable="MERGEABLE")) == "🔴"


# ---------------------------------------------------------------------------
# worst-wins thread icon priority
# ---------------------------------------------------------------------------

class TestWorstIcon:
    def test_priority_order_matches_spec(self):
        # ❓ > 🔴 > ❗ > 🟡 > 🔵 > 🟢 > ✅ > 🟣 > ⚪
        assert tc.ICON_PRIORITY == ["❓", "🔴", "❗", "🟡", "🔵", "🟢", "✅", "🟣", "⚪"]

    @pytest.mark.parametrize("icons,expected", [
        (["🔴", "🟢"], "🔴"),
        (["✅", "🟢"], "🟢"),  # 🟢 outranks ✅ -- intentional, not a bug
        (["⚪", "✅"], "✅"),
        (["🟣", "⚪"], "🟣"),
        (["❓", "🔴"], "❓"),
        (["🟢"], "🟢"),
    ])
    def test_worst_of_set(self, icons, expected):
        assert tc.worst_icon(icons) == expected

    def test_question_flag_always_wins(self):
        thread = {"status": "active", "question": True}
        prs = [_pr(state="OPEN", ci="FAILURE")]
        assert tc.effective_thread_icon(thread, prs) == "❓"

    def test_thread_icon_is_worst_of_status_and_open_prs(self):
        thread = {"status": "active", "question": False}
        prs = [
            _pr(number=1, state="OPEN", ci="SUCCESS", review="APPROVED", mergeable="MERGEABLE"),
            _pr(number=2, state="OPEN", ci="FAILURE"),
        ]
        assert tc.effective_thread_icon(thread, prs) == "🔴"

    def test_closed_prs_excluded_from_worst_wins(self):
        thread = {"status": "active", "question": False}
        prs = [_pr(number=1, state="CLOSED"), _pr(number=2, state="MERGED")]
        # Only thread status (active -> 🟢) counts; closed/merged PRs are not OPEN.
        assert tc.effective_thread_icon(thread, prs) == "🟢"

    def test_no_prs_uses_status_icon(self):
        assert tc.effective_thread_icon({"status": "paused", "question": False}, []) == "⚪"

    def test_merged_status_icon(self):
        assert tc.effective_thread_icon({"status": "merged", "question": False}, []) == "🟣"


# ---------------------------------------------------------------------------
# staleness + exemptions
# ---------------------------------------------------------------------------

class TestStaleness:
    def test_old_touch_is_stale_for_non_exempt_icon(self):
        from datetime import datetime, timedelta, timezone
        now = datetime(2026, 7, 28, 12, 0, 0, tzinfo=timezone.utc)
        old = (now - timedelta(minutes=31)).strftime("%Y-%m-%dT%H:%M:%SZ")
        assert tc.is_stale("🟢", old, now=now) is True

    def test_recent_touch_not_stale(self):
        from datetime import datetime, timedelta, timezone
        now = datetime(2026, 7, 28, 12, 0, 0, tzinfo=timezone.utc)
        recent = (now - timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M:%SZ")
        assert tc.is_stale("🟢", recent, now=now) is False

    def test_exactly_at_threshold_not_stale(self):
        from datetime import datetime, timedelta, timezone
        now = datetime(2026, 7, 28, 12, 0, 0, tzinfo=timezone.utc)
        edge = (now - timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
        assert tc.is_stale("🟢", edge, now=now) is False

    @pytest.mark.parametrize("icon", sorted(tc.STALE_EXEMPT_ICONS))
    def test_exempt_icons_never_stale(self, icon):
        from datetime import datetime, timedelta, timezone
        now = datetime(2026, 7, 28, 12, 0, 0, tzinfo=timezone.utc)
        ancient = (now - timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
        assert tc.is_stale(icon, ancient, now=now) is False

    @pytest.mark.parametrize("icon", ["🟢", "🔴", "🟡"])
    def test_non_exempt_icons_can_go_stale(self, icon):
        from datetime import datetime, timedelta, timezone
        now = datetime(2026, 7, 28, 12, 0, 0, tzinfo=timezone.utc)
        ancient = (now - timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
        assert tc.is_stale(icon, ancient, now=now) is True

    def test_custom_stale_after_min(self):
        from datetime import datetime, timedelta, timezone
        now = datetime(2026, 7, 28, 12, 0, 0, tzinfo=timezone.utc)
        old = (now - timedelta(minutes=10)).strftime("%Y-%m-%dT%H:%M:%SZ")
        assert tc.is_stale("🟢", old, now=now, stale_after_min=5) is True
        assert tc.is_stale("🟢", old, now=now, stale_after_min=15) is False


# ---------------------------------------------------------------------------
# title rendering
# ---------------------------------------------------------------------------

class TestRenderTitle:
    def test_example_from_spec(self):
        thread = {"title": "Experiment recompute", "status": "active", "question": False, "repo": "absmartly/abs"}
        prs = [
            _pr(number=4615, repo="absmartly/abs", primary=True, ci="FAILURE"),
            _pr(number=4620, repo="absmartly/abs", ci="PENDING"),
            _pr(number=4631, repo="absmartly/abs", ci="SUCCESS", review="APPROVED", mergeable="MERGEABLE"),
        ]
        title = tc.render_title(thread, prs, stale=False)
        assert title == "🔴 Experiment recompute · 🔴#4615 🟡#4620 ✅#4631"

    def test_no_open_prs_omits_pr_segment(self):
        thread = {"title": "mSPRT", "status": "active", "question": False, "repo": "absmartly/abs"}
        assert tc.render_title(thread, [], stale=False) == "🟢 mSPRT"

    def test_stale_suffix_appended_last(self):
        thread = {"title": "mSPRT", "status": "active", "question": False, "repo": "absmartly/abs"}
        prs = [_pr(number=1, repo="absmartly/abs", ci="SUCCESS")]
        title = tc.render_title(thread, prs, stale=True)
        assert title.endswith("💤")
        # 🔵 (in-review PR) worst-wins over 🟢 (active thread status)
        assert title == "🔵 mSPRT · 🔵#1 💤"

    def test_bare_pr_number_when_repo_matches_thread(self):
        thread = {"title": "x", "status": "active", "question": False, "repo": "absmartly/abs"}
        prs = [_pr(number=5, repo="absmartly/abs", ci="SUCCESS")]
        title = tc.render_title(thread, prs, stale=False)
        assert "#5" in title
        assert "abs#5" not in title

    def test_repo_qualified_pr_number_when_different_repo(self):
        thread = {"title": "x", "status": "active", "question": False, "repo": "absmartly/abs"}
        prs = [_pr(number=5, repo="absmartly/other", ci="SUCCESS")]
        title = tc.render_title(thread, prs, stale=False)
        assert "other#5" in title

    def test_primary_pr_first_then_ascending(self):
        thread = {"title": "x", "status": "active", "question": False, "repo": "absmartly/abs"}
        prs = [
            _pr(number=10, repo="absmartly/abs", ci="SUCCESS"),
            _pr(number=3, repo="absmartly/abs", primary=True, ci="SUCCESS"),
            _pr(number=7, repo="absmartly/abs", ci="SUCCESS"),
        ]
        title = tc.render_title(thread, prs, stale=False)
        # primary (#3) first, then ascending: #7, #10
        pr_part = title.split(" · ", 1)[1]
        assert pr_part.index("#3") < pr_part.index("#7") < pr_part.index("#10")

    def test_only_open_prs_shown(self):
        thread = {"title": "x", "status": "active", "question": False, "repo": "absmartly/abs"}
        prs = [
            _pr(number=1, repo="absmartly/abs", state="MERGED"),
            _pr(number=2, repo="absmartly/abs", state="OPEN", ci="SUCCESS"),
        ]
        title = tc.render_title(thread, prs, stale=False)
        assert "#1" not in title
        assert "#2" in title

    def test_title_never_exceeds_100_chars_with_many_prs(self):
        thread = {"title": "x" * 48, "status": "active", "question": False, "repo": "absmartly/abs"}
        prs = [_pr(number=1000 + i, repo="absmartly/abs", ci="SUCCESS") for i in range(20)]
        title = tc.render_title(thread, prs, stale=False)
        assert len(title) <= 100

    def test_overflow_truncates_from_right_and_appends_plus_n(self):
        thread = {"title": "Experiment recompute pipeline", "status": "active", "question": False, "repo": "absmartly/abs"}
        prs = [_pr(number=4000 + i, repo="absmartly/abs", ci="SUCCESS") for i in range(20)]
        title = tc.render_title(thread, prs, stale=False)
        assert len(title) <= 100
        assert "+" in title

    def test_icon_and_title_never_truncated_only_pr_list(self):
        thread = {"title": "y" * 48, "status": "active", "question": False, "repo": "absmartly/abs"}
        prs = [_pr(number=5000 + i, repo="absmartly/abs", ci="SUCCESS") for i in range(30)]
        title = tc.render_title(thread, prs, stale=False)
        assert title.startswith("🔵 " + "y" * 48)
        assert len(title) <= 100

    def test_stale_suffix_survives_overflow_truncation(self):
        thread = {"title": "Experiment recompute pipeline", "status": "active", "question": False, "repo": "absmartly/abs"}
        prs = [_pr(number=4000 + i, repo="absmartly/abs", ci="SUCCESS") for i in range(20)]
        title = tc.render_title(thread, prs, stale=True)
        assert title.endswith("💤")
        assert len(title) <= 100


# ---------------------------------------------------------------------------
# GitHub GraphQL polling + database-only cycle
# ---------------------------------------------------------------------------


def _graphql_payload(**prs):
    return {"data": {"repository": prs}}


def _github_pr(
    state="OPEN", ci="SUCCESS", review="APPROVED", mergeable="MERGEABLE",
    draft=False, title="Implement thing", url="https://github.com/absmartly/abs/pull/1",
):
    """Builds one `pullRequest` node using the verified-correct real shape
    (commits -> nodes -> commit -> statusCheckRollup{state, contexts}; see
    ADDENDUM.md / tests/fixtures/graphql_*.json) — for cycle-level tests
    below that only care about downstream behaviour (staggering,
    terminal-freeze, desired-name persistence), not the GraphQL contract
    itself. The contract itself is exercised directly against RECORDED
    fixtures in TestGithubPolling, per CRITICAL 1: the old version of this
    helper invented the same nonexistent `statusCheckRollup { nodes {...} }`
    shape the code emitted, so 100% of polling tests passed against a
    schema that doesn't exist — this helper must never again drift from
    what GitHub actually returns."""
    rollup = None if ci == "NONE" else {
        "state": ci,
        "contexts": {"nodes": [
            {"__typename": "CheckRun", "conclusion": ci, "status": "COMPLETED", "name": "build"},
        ]},
    }
    return {
        "state": state,
        "mergeable": mergeable,
        "reviewDecision": review,
        "isDraft": draft,
        "title": title,
        "url": url,
        "commits": {"nodes": [{"commit": {"statusCheckRollup": rollup}}]},
    }


class TestGithubPolling:
    def test_fetches_batched_graphql_query_against_recorded_fixture(self, monkeypatch, fixtures_dir):
        """Drives fetch_repo_pr_states from a REAL recorded GraphQL response
        (absmartly/abs PRs 4615, 4547, 4620 — see ADDENDUM.md), not a
        hand-invented shape. This is exactly the trap CRITICAL 1 warns
        about: the old `_github_pr()` fixture invented the same
        nonexistent `statusCheckRollup { nodes {...} }` shape the code
        emitted, so 100% of polling tests passed against a schema that
        does not exist."""
        monkeypatch.setattr(tc, "_now_iso", lambda: "2026-07-28T12:00:00Z")
        payload = json.loads((fixtures_dir / "graphql_open_varied.json").read_text())
        calls = []

        def runner(cmd, timeout):
            calls.append(cmd)
            if cmd[:2] == ["gh", "auth"]:
                assert cmd == ["gh", "auth", "token", "--user", "marcio-absmartly"]
                return __import__("subprocess").CompletedProcess(cmd, 0, "tok\n", "")
            return __import__("subprocess").CompletedProcess(cmd, 0, json.dumps(payload), "")

        states, error, graphql_errors = tc.fetch_repo_pr_states(
            "absmartly/abs", [4615, 4547, 4620], tc.DEFAULT_ACCOUNTS, 20, runner
        )

        assert error is None
        assert graphql_errors == []
        assert states[4615]["ci"] == "SUCCESS"
        assert states[4615]["review"] == "APPROVED"
        assert states[4615]["mergeable"] == "MERGEABLE"
        assert states[4547]["ci"] == "FAILURE"
        assert states[4547]["review"] == "REVIEW_REQUIRED"
        assert states[4620]["ci"] == "SUCCESS"
        assert states[4615]["checked_at"] == "2026-07-28T12:00:00Z"
        assert calls[1][:2] == ["env", "GH_TOKEN=tok"]
        query = next(part for part in calls[1] if part.startswith("query="))
        assert "pullRequest(number: 4615)" in query
        assert "pullRequest(number: 4547)" in query
        # The verified-correct shape (CRITICAL 1): commits(last:1) ->
        # statusCheckRollup { state contexts(last:100) { ... } }, NOT the
        # nonexistent `statusCheckRollup { nodes {...} }`.
        assert "commits(last: 1)" in query
        assert "contexts(last: 100)" in query
        assert "... on CheckRun" in query
        assert "... on StatusContext" in query
        assert "statusCheckRollup { nodes" not in query
        assert not any(call[:3] == ["gh", "auth", "switch"] for call in calls)

    def test_ci_derivation_from_real_rollup_shapes(self, fixtures_dir):
        """`_derive_ci` against real `statusCheckRollup` objects extracted
        from recorded fixtures (not hand-invented dicts) — `rollup.state`
        is SUCCESS/FAILURE/PENDING and is the primary, simplest-correct
        source of truth (ADDENDUM.md)."""
        open_varied = json.loads((fixtures_dir / "graphql_open_varied.json").read_text())
        terminal = json.loads((fixtures_dir / "graphql_terminal_and_missing.json").read_text())
        toolbox_pr7 = json.loads((fixtures_dir / "graphql_toolbox_pr7.json").read_text())

        def rollup_of(payload, alias):
            node = payload["data"]["repository"][alias]
            return node["commits"]["nodes"][0]["commit"]["statusCheckRollup"]

        assert tc._derive_ci(rollup_of(open_varied, "pr_4615")) == "SUCCESS"
        assert tc._derive_ci(rollup_of(open_varied, "pr_4547")) == "FAILURE"
        assert tc._derive_ci(rollup_of(open_varied, "pr_4620")) == "SUCCESS"
        assert tc._derive_ci(rollup_of(terminal, "pr_4720")) == "SUCCESS"
        assert tc._derive_ci(rollup_of(terminal, "pr_4716")) == "FAILURE"
        assert tc._derive_ci(rollup_of(toolbox_pr7, "pr_7")) == "FAILURE"

    def test_ci_derivation_none_rollup_means_no_checks_configured(self):
        assert tc._derive_ci(None) == "NONE"

    def test_ci_derivation_falls_back_to_contexts_for_unrecognised_rollup_state(self):
        # Defensive fallback: a rollup.state GitHub might add that we don't
        # map yet still derives correctly from the CheckRun/StatusContext
        # union in `contexts`.
        rollup = {
            "state": "SOMETHING_NEW",
            "contexts": {"nodes": [
                {"__typename": "CheckRun", "conclusion": "FAILURE", "status": "COMPLETED", "name": "build"},
                {"__typename": "StatusContext", "state": "SUCCESS", "context": "CodeRabbit"},
            ]},
        }
        assert tc._derive_ci(rollup) == "FAILURE"

    def test_graphql_failure_is_returned_not_raised(self):
        def runner(cmd, timeout):
            return __import__("subprocess").CompletedProcess(cmd, 1, "", "bad credentials")

        states, error, graphql_errors = tc.fetch_repo_pr_states("unknown/repo", [1], {}, 20, runner)
        assert states is None
        assert error == "bad credentials"
        assert graphql_errors == []

    def test_malformed_graphql_repository_is_returned_not_raised(self):
        def runner(cmd, timeout):
            return __import__("subprocess").CompletedProcess(
                cmd, 0, json.dumps({"data": {"repository": None}}), ""
            )

        states, error, graphql_errors = tc.fetch_repo_pr_states("unknown/repo", [1], {}, 20, runner)
        assert states is None
        assert "invalid GraphQL JSON" in error

    def test_partial_failure_missing_pr_does_not_wipe_sibling_pr_state(self, fixtures_dir):
        """A missing PR (pr_9999999 -> null alias + a top-level NOT_FOUND
        error) must not prevent the other two good aliases (pr_4720
        MERGED, pr_4716 CLOSED) in the SAME response from coming back
        usable — verified against the real recorded response
        (ADDENDUM.md behaviours #2/#3).

        R2-4: `gh api graphql` exits 1 (not 0) whenever top-level `errors`
        are present, even though `data` is fully usable — this is real
        `gh` behaviour (ADDENDUM.md, verified live), not an invented one.
        returncode=0 here would not exercise the actual regression."""
        payload = json.loads((fixtures_dir / "graphql_terminal_and_missing.json").read_text())

        def runner(cmd, timeout):
            return __import__("subprocess").CompletedProcess(cmd, 1, json.dumps(payload), "")

        states, error, graphql_errors = tc.fetch_repo_pr_states(
            "absmartly/abs", [4720, 4716, 9999999], {}, 20, runner
        )
        assert error is None
        assert states[4720] is not None
        assert states[4720]["state"] == "MERGED"
        assert states[4716] is not None
        assert states[4716]["state"] == "CLOSED"
        assert states[9999999] is None
        assert len(graphql_errors) == 1
        assert "9999999" in graphql_errors[0]

    def test_merged_pr_mergeable_unknown_is_preserved_not_treated_as_conflicting(self, fixtures_dir):
        """Behaviour #5: MERGED PRs report mergeable=UNKNOWN — this must
        pass through as-is, never be coerced into CONFLICTING."""
        payload = json.loads((fixtures_dir / "graphql_terminal_and_missing.json").read_text())

        def runner(cmd, timeout):
            return __import__("subprocess").CompletedProcess(cmd, 0, json.dumps(payload), "")

        states, error, _ = tc.fetch_repo_pr_states("absmartly/abs", [4720], {}, 20, runner)
        assert states[4720]["mergeable"] == "UNKNOWN"
        assert states[4720]["state"] == "MERGED"
        pr = {**states[4720], "primary": False, "number": 4720, "draft": False}
        # MERGED is terminal and checked first in pr_icon — mergeable being
        # unexpected/falsy must not fall through to the CONFLICTING branch.
        assert tc.pr_icon(pr) == "🟣"

    def test_trailing_non_json_text_after_gh_stdout_parses_cleanly(self, fixtures_dir):
        """`gh` appends a human-readable error line AFTER the JSON body
        when a query aliases a nonexistent PR (captured verbatim from a
        real invocation in raw_gh_stdout.txt). A naive `json.loads` raises
        `JSONDecodeError: Extra data` on this; the parser must tolerate
        it (behaviour #4).

        R2-4: this exact invocation is the one Rax verified live —
        `gh api graphql` exited 1 while stdout (this raw fixture) still
        carried both good `data` and the top-level `errors`. returncode=0
        would not be the real `gh` behaviour and would not exercise the
        bug (fetch_repo_pr_states used to bail out on non-zero returncode
        before ever looking at stdout)."""
        raw = (fixtures_dir / "graphql_terminal_and_missing.raw_gh_stdout.txt").read_text()
        assert "gh: Could not resolve to a PullRequest" in raw  # sanity: the trap really is there
        with pytest.raises(json.JSONDecodeError):
            json.loads(raw)  # the naive approach really does fail

        def runner(cmd, timeout):
            return __import__("subprocess").CompletedProcess(cmd, 1, raw, "")

        states, error, graphql_errors = tc.fetch_repo_pr_states(
            "absmartly/abs", [4720, 4716, 9999999], {}, 20, runner
        )
        assert error is None
        assert states[4720]["state"] == "MERGED"
        assert states[9999999] is None
        assert len(graphql_errors) == 1

    def test_returncode_1_does_not_discard_good_data_end_to_end(self, db_path, capsys, monkeypatch, fixtures_dir):
        """R2-4 end-to-end regression, driven exactly as Rax reproduced it
        live: `gh api graphql` exits 1 (a NOT_FOUND alias in the same
        batch) while stdout still carries usable data for the OTHER PRs.
        Before the fix, `_poll_due_prs` (via `fetch_repo_pr_states`
        bailing out on `returncode != 0` before parsing stdout) discarded
        the entire repo batch every cycle: `checked_at` for the good PRs
        stayed NULL forever, so they were re-fetched (and re-discarded)
        every single cycle. This test fails against d062d5d and passes
        after the fix: the good PR (#4720) must come out of the cycle with
        state populated and `checked_at` set (not NULL, not perpetually
        due), while only the missing PR (#9999999) is reported as an
        error."""
        _bind(db_path, capsys, thread_id="t1", repo="absmartly/abs")
        run_cli(["add-pr", "t1", "4720"], db_path, capsys)
        run_cli(["add-pr", "t1", "9999999"], db_path, capsys)

        raw = (fixtures_dir / "graphql_terminal_and_missing.raw_gh_stdout.txt").read_text()
        monkeypatch.setattr(tc, "_now_iso", lambda: "2026-07-28T12:00:00Z")

        def runner(cmd, timeout):
            # Real `gh` behaviour: exit 1 whenever top-level `errors` are
            # present, even though `data` is fully usable for the other
            # aliases (ADDENDUM.md, verified live).
            return __import__("subprocess").CompletedProcess(cmd, 1, raw, "")

        result = tc.run_cycle(db_path, runner=runner, token=None, transport=lambda *a, **k: None, dry_run=True)
        assert result["refreshed"] == 1  # only #4720 is in the fixture's usable data
        assert result["errors"] == 1  # #9999999 alone is reported as unusable

        conn = tc.connect_db(db_path)
        try:
            pr_4720 = conn.execute("SELECT * FROM prs WHERE number=4720").fetchone()
            assert pr_4720["state"] == "MERGED"
            pr_missing = conn.execute("SELECT * FROM prs WHERE number=9999999").fetchone()
            assert pr_missing["state"] is None
            history_kinds = [
                r["text"] for r in conn.execute(
                    "SELECT text FROM history WHERE thread_id='t1' AND kind='error'"
                )
            ]
            assert any("9999999" in (t or "") for t in history_kinds)
        finally:
            conn.close()

        # MERGED is a fresh terminal transition, so spec's "one more poll"
        # rule deliberately leaves checked_at NULL after this first cycle
        # (see _just_became_terminal) — that's correct, not the bug. Run a
        # second cycle: THIS is where the pre-fix code would have kept
        # discarding the whole batch (including #4720) forever, because
        # `fetch_repo_pr_states` bailed out on returncode=1 before parsing
        # stdout on every single cycle. After the fix, the second poll
        # confirms the transition and freezes checked_at for good.
        result2 = tc.run_cycle(db_path, runner=runner, token=None, transport=lambda *a, **k: None, dry_run=True)
        assert result2["refreshed"] == 1
        conn = tc.connect_db(db_path)
        try:
            pr_4720 = conn.execute("SELECT * FROM prs WHERE number=4720").fetchone()
            assert pr_4720["state"] == "MERGED"
            assert pr_4720["checked_at"] is not None  # now frozen, not stuck NULL forever
        finally:
            conn.close()

    def test_toolbox_pr7_real_fixture_open_failing_ci(self, fixtures_dir):
        """marcioapm/toolbox#7 — OPEN, reviewDecision null (never
        requested review), rollup FAILURE (one pytest matrix leg failed,
        the rest cancelled)."""
        payload = json.loads((fixtures_dir / "graphql_toolbox_pr7.json").read_text())

        def runner(cmd, timeout):
            return __import__("subprocess").CompletedProcess(cmd, 0, json.dumps(payload), "")

        states, error, _ = tc.fetch_repo_pr_states("marcioapm/toolbox", [7], {}, 20, runner)
        assert error is None
        assert states[7]["state"] == "OPEN"
        assert states[7]["ci"] == "FAILURE"
        assert states[7]["review"] == "NONE"  # reviewDecision null -> our NONE default


    def test_cycle_staggers_oldest_prs_and_honours_cap(self, db_path, capsys, monkeypatch):
        _bind(db_path, capsys)
        for number in (1, 2, 3):
            run_cli(["add-pr", "t1", str(number)], db_path, capsys)
        conn = tc.connect_db(db_path)
        try:
            with conn:
                conn.execute("UPDATE prs SET checked_at='2026-07-28T10:00:00Z' WHERE number=1")
                conn.execute("UPDATE prs SET checked_at='2026-07-28T11:00:00Z' WHERE number=2")
                conn.execute("UPDATE prs SET checked_at='2026-07-28T11:59:00Z' WHERE number=3")
        finally:
            conn.close()
        now = __import__("datetime").datetime(2026, 7, 28, 12, 0, tzinfo=__import__("datetime").timezone.utc)
        monkeypatch.setattr(tc, "_utcnow", lambda: now)
        monkeypatch.setattr(tc, "_now_iso", lambda: "2026-07-28T12:00:00Z")
        fetched = []

        def runner(cmd, timeout):
            if cmd[:2] == ["gh", "auth"]:
                return __import__("subprocess").CompletedProcess(cmd, 0, "token\n", "")
            query = next(part for part in cmd if part.startswith("query="))
            fetched.extend(int(number) for number in __import__("re").findall(r"pullRequest\(number: (\d+)\)", query))
            payload = _graphql_payload(
                **{f"pr_{n}": _github_pr(url=f"https://github.com/absmartly/abs/pull/{n}") for n in fetched}
            )
            return __import__("subprocess").CompletedProcess(cmd, 0, json.dumps(payload), "")

        result = tc.run_cycle(
            db_path,
            runner=runner,
            config={"stale_after_min": 30, "pr_recheck_min_s": 150, "max_pr_checks_per_cycle": 2},
        )
        assert result["refreshed"] == 2
        assert result["errors"] == 0
        assert result["threads"] == 1
        assert fetched == [1, 2]

    def test_terminal_pr_is_confirmed_once_then_frozen(self, db_path, capsys, monkeypatch):
        _bind(db_path, capsys)
        run_cli(["add-pr", "t1", "1"], db_path, capsys)
        now = __import__("datetime").datetime(2026, 7, 28, 12, 0, tzinfo=__import__("datetime").timezone.utc)
        monkeypatch.setattr(tc, "_utcnow", lambda: now)
        monkeypatch.setattr(tc, "_now_iso", lambda: "2026-07-28T12:00:00Z")
        calls = []

        def runner(cmd, timeout):
            if cmd[:2] == ["gh", "auth"]:
                return __import__("subprocess").CompletedProcess(cmd, 0, "token\n", "")
            calls.append(cmd)
            return __import__("subprocess").CompletedProcess(
                cmd, 0, json.dumps(_graphql_payload(pr_1=_github_pr(state="MERGED"))), ""
            )

        assert tc.run_cycle(db_path, runner=runner)["refreshed"] == 1
        conn = tc.connect_db(db_path)
        try:
            assert tc.get_prs(conn, "t1")[0]["checked_at"] is None
        finally:
            conn.close()
        assert tc.run_cycle(db_path, runner=runner)["refreshed"] == 1
        assert tc.run_cycle(db_path, runner=runner)["refreshed"] == 0
        assert len(calls) == 2

    def test_terminal_pr_is_not_polled_again_after_confirmation(self, db_path, capsys, monkeypatch):
        _bind(db_path, capsys)
        run_cli(["add-pr", "t1", "1"], db_path, capsys)
        conn = tc.connect_db(db_path)
        try:
            with conn:
                conn.execute(
                    "UPDATE prs SET state='MERGED', checked_at='2026-07-28T10:00:00Z' WHERE number=1"
                )
        finally:
            conn.close()
        now = __import__("datetime").datetime(2026, 7, 28, 12, 0, tzinfo=__import__("datetime").timezone.utc)
        monkeypatch.setattr(tc, "_utcnow", lambda: now)

        def runner(cmd, timeout):
            raise AssertionError("terminal PR must not be fetched again")

        assert tc.run_cycle(db_path, runner=runner)["refreshed"] == 0

    def test_cycle_persists_desired_name_from_refreshed_pr_state(self, db_path, capsys, monkeypatch):
        _bind(db_path, capsys)
        run_cli(["add-pr", "t1", "1"], db_path, capsys)
        now = __import__("datetime").datetime(2026, 7, 28, 12, 0, tzinfo=__import__("datetime").timezone.utc)
        monkeypatch.setattr(tc, "_utcnow", lambda: now)
        monkeypatch.setattr(tc, "_now_iso", lambda: "2026-07-28T12:00:00Z")

        def runner(cmd, timeout):
            if cmd[:2] == ["gh", "auth"]:
                return __import__("subprocess").CompletedProcess(cmd, 0, "token\n", "")
            return __import__("subprocess").CompletedProcess(
                cmd, 0, json.dumps(_graphql_payload(pr_1=_github_pr(ci="FAILURE"))), ""
            )

        assert tc.run_cycle(db_path, runner=runner)["refreshed"] == 1
        conn = tc.connect_db(db_path)
        try:
            thread = tc.get_thread(conn, "t1")
            assert thread["desired_name"] == "🔴 Experiment recompute · 🔴#1"
            assert thread["applied_name"] is None
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Discord: transport network-error handling (CRITICAL 2)
# ---------------------------------------------------------------------------

class TestDiscordTransportNetworkErrors:
    """`_discord_transport` used to only catch `HTTPError` (a well-formed
    4xx/5xx response). A transient network blip — DNS failure, connection
    reset, read timeout, TLS error — raised straight out of the transport,
    which used to escape `reconcile_discord` mid-cycle and roll back the
    whole `with conn:` transaction, discarding DB state for renames that
    had ALREADY succeeded on Discord for threads processed earlier in the
    same cycle. This is what actually generates the duplicate-rename bug:
    the next cycle sees `applied_name` reverted and re-issues the PATCH."""

    def test_urlerror_returns_synthetic_response_not_raises(self, monkeypatch):
        def fake_urlopen(req, timeout):
            raise urllib.error.URLError("nodename nor servname provided")

        monkeypatch.setattr(tc.urllib.request, "urlopen", fake_urlopen)
        resp = _real_discord_transport("PATCH", "https://discord.com/api/v10/channels/1", "tok", {"name": "x"}, 15)
        assert resp.status == 0
        assert resp.data is not None

    def test_socket_timeout_returns_synthetic_response_not_raises(self, monkeypatch):
        def fake_urlopen(req, timeout):
            raise socket.timeout("timed out")

        monkeypatch.setattr(tc.urllib.request, "urlopen", fake_urlopen)
        resp = _real_discord_transport("PATCH", "https://discord.com/api/v10/channels/1", "tok", {"name": "x"}, 15)
        assert resp.status == 0

    def test_ssl_error_returns_synthetic_response_not_raises(self, monkeypatch):
        def fake_urlopen(req, timeout):
            raise ssl.SSLError("bad handshake")

        monkeypatch.setattr(tc.urllib.request, "urlopen", fake_urlopen)
        resp = _real_discord_transport("PATCH", "https://discord.com/api/v10/channels/1", "tok", {"name": "x"}, 15)
        assert resp.status == 0

    def test_httperror_still_handled_as_before(self, monkeypatch):
        def fake_urlopen(req, timeout):
            raise urllib.error.HTTPError("url", 429, "rate limited", {}, __import__("io").BytesIO(b'{"retry_after": 5}'))

        monkeypatch.setattr(tc.urllib.request, "urlopen", fake_urlopen)
        resp = _real_discord_transport("PATCH", "https://discord.com/api/v10/channels/1", "tok", {"name": "x"}, 15)
        assert resp.status == 429
        assert resp.data == {"retry_after": 5}


class _FakeHTTPResponse:
    """Minimal context-manager stand-in for the object `urlopen()` returns,
    so R2-3 tests can control exactly what `.read()` does/raises without
    touching real sockets."""

    def __init__(self, status=200, read_result=None, read_raises=None):
        self.status = status
        self._read_result = read_result
        self._read_raises = read_raises

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        if self._read_raises is not None:
            raise self._read_raises
        return self._read_result


class TestDiscordTransportReadAndParseErrors:
    """R2-3 (regression of C2, ROUND2-FINDINGS.md): the old exception
    handling covered only what `urlopen()` itself could raise. `resp.read()`
    and `json.loads(body)` sat OUTSIDE any try/except, so exceptions that
    occur precisely AFTER Discord has already applied the write —
    non-JSON 2xx bodies, truncated reads, connection resets mid-read —
    used to escape uncaught and trigger the R2-1 rollback of DB state for
    a write that had, in fact, already succeeded."""

    def test_non_json_2xx_body_returns_synthetic_response_not_raises(self, monkeypatch):
        """A proxy/CDN interstitial can return HTTP 200 with an HTML body
        instead of JSON. `json.loads` on that raises `JSONDecodeError` (a
        ValueError) — this used to be completely uncaught."""
        def fake_urlopen(req, timeout):
            return _FakeHTTPResponse(status=200, read_result=b"<html>upstream proxy error</html>")

        monkeypatch.setattr(tc.urllib.request, "urlopen", fake_urlopen)
        resp = _real_discord_transport("PATCH", "https://discord.com/api/v10/channels/1", "tok", {"name": "x"}, 15)
        assert resp.status == 0
        assert resp.data is not None

    def test_incomplete_read_returns_synthetic_response_not_raises(self, monkeypatch):
        """`http.client.IncompleteRead` — the connection dropped mid-body,
        after Discord's server had already processed the write."""
        def fake_urlopen(req, timeout):
            return _FakeHTTPResponse(
                status=200, read_raises=http.client.IncompleteRead(b"partial")
            )

        monkeypatch.setattr(tc.urllib.request, "urlopen", fake_urlopen)
        resp = _real_discord_transport("PATCH", "https://discord.com/api/v10/channels/1", "tok", {"name": "x"}, 15)
        assert resp.status == 0

    def test_connection_reset_during_read_returns_synthetic_response_not_raises(self, monkeypatch):
        """`ConnectionResetError` is an `OSError` subclass, NOT a
        `URLError` — the old except tuple did not cover it."""
        def fake_urlopen(req, timeout):
            return _FakeHTTPResponse(status=200, read_raises=ConnectionResetError("connection reset by peer"))

        monkeypatch.setattr(tc.urllib.request, "urlopen", fake_urlopen)
        resp = _real_discord_transport("PATCH", "https://discord.com/api/v10/channels/1", "tok", {"name": "x"}, 15)
        assert resp.status == 0

    def test_httperror_non_json_body_returns_synthetic_response_not_raises(self, monkeypatch):
        """Same treatment for `HTTPError.read()` at the error path: a
        non-JSON error body (e.g. an HTML 502 page from a CDN in front of
        Discord) must not raise out of the transport."""
        def fake_urlopen(req, timeout):
            raise urllib.error.HTTPError(
                "url", 502, "bad gateway", {}, __import__("io").BytesIO(b"<html>bad gateway</html>")
            )

        monkeypatch.setattr(tc.urllib.request, "urlopen", fake_urlopen)
        resp = _real_discord_transport("PATCH", "https://discord.com/api/v10/channels/1", "tok", {"name": "x"}, 15)
        assert resp.status == 502
        assert resp.data is None

    def test_httperror_read_incomplete_read_returns_synthetic_response_not_raises(self, monkeypatch):
        class BoomOnReadHTTPError(urllib.error.HTTPError):
            def read(self, *a, **k):
                raise http.client.IncompleteRead(b"partial")

        def fake_urlopen(req, timeout):
            raise BoomOnReadHTTPError("url", 500, "server error", {}, __import__("io").BytesIO(b""))

        monkeypatch.setattr(tc.urllib.request, "urlopen", fake_urlopen)
        resp = _real_discord_transport("PATCH", "https://discord.com/api/v10/channels/1", "tok", {"name": "x"}, 15)
        assert resp.status == 500
        assert resp.data is None


class TestReconcileDiscordCommitsPerThread:
    """CRITICAL 2: reconcile_discord must commit per-thread, so one
    thread's Discord failure doesn't roll back another thread's already-
    successful DB writes in the same cycle. Simulates the reviewer's
    reproduction: t1's pin+rename succeed, t2's rename explodes with an
    unexpected exception — t1's state must survive."""

    def test_one_threads_exception_does_not_roll_back_another_threads_success(self, db_path, capsys, monkeypatch):
        _bind(db_path, capsys, thread_id="t1", title="Thread one")
        run_cli(["add-pr", "t1", "1"], db_path, capsys)
        _bind(db_path, capsys, thread_id="t2", title="Thread two")
        run_cli(["add-pr", "t2", "1"], db_path, capsys)

        conn = tc.connect_db(db_path)
        try:
            with conn:
                conn.execute("UPDATE threads SET desired_name='🟢 t1 new' WHERE thread_id='t1'")
                conn.execute("UPDATE threads SET desired_name='🟢 t2 new' WHERE thread_id='t2'")
        finally:
            conn.close()

        now = datetime(2026, 7, 28, 12, 0, tzinfo=timezone.utc)

        class BoomTransport:
            def __call__(self, method, url, token, payload, timeout):
                if "/channels/t2" in url and method == "PATCH":
                    raise RuntimeError("simulated unexpected crash")
                if method == "POST":
                    return tc.DiscordResponse(201, {"id": "555"})
                return tc.DiscordResponse(200, {"id": "555"})

        conn = tc.connect_db(db_path)
        try:
            pins_changed, renames_applied, errors = tc.reconcile_discord(
                conn, now, {"rename_min_interval_s": 330}, "tok", BoomTransport(), dry_run=False
            )
            assert errors >= 1
            t1 = tc.get_thread(conn, "t1")
            assert t1["applied_name"] == "🟢 t1 new"
            t2 = tc.get_thread(conn, "t2")
            # t2's own crash means its rename did NOT apply, but that must
            # not undo t1's already-successful rename above.
            assert t2["applied_name"] != "🟢 t2 new"
        finally:
            conn.close()


class TestReconcileDiscordPinSurvivesRenameCrash:
    """R2-1 (regression of C2, ROUND2-FINDINGS.md): `reconcile_pin` stages
    `pin_message_id` but the old code did NOT commit it before
    `reconcile_rename` went and did its own network PATCH for the SAME
    thread. If the rename raised, the shared try/except rolled back the
    whole transaction — discarding the pin write that had ALREADY
    succeeded on Discord. DB ends with pin_message_id=None while Discord
    has a real pinned message, so the next cycle posts a SECOND pinned
    message. Fix: reconcile_pin's commit happens before reconcile_rename
    starts its network I/O; only the interaction that actually crashes
    gets rolled back."""

    def test_pin_write_survives_same_threads_rename_crash(self, db_path, capsys):
        _bind(db_path, capsys, thread_id="t1", title="Thread one")
        run_cli(["add-pr", "t1", "1"], db_path, capsys)
        conn = tc.connect_db(db_path)
        try:
            with conn:
                conn.execute("UPDATE threads SET desired_name='🟢 t1 new' WHERE thread_id='t1'")
        finally:
            conn.close()

        now = datetime(2026, 7, 28, 12, 0, tzinfo=timezone.utc)

        class BoomOnRenameTransport:
            """Pin creation succeeds; the rename PATCH for the SAME thread
            raises an unexpected exception — exactly the reviewer's
            reproduction of R2-1."""

            def __call__(self, method, url, token, payload, timeout):
                if method == "PATCH" and "/channels/t1" in url and "/messages/" not in url:
                    raise RuntimeError("simulated unexpected crash mid-rename")
                if method == "POST":
                    return tc.DiscordResponse(201, {"id": "555"})
                return tc.DiscordResponse(200, {"id": "555"})

        conn = tc.connect_db(db_path)
        try:
            pins_changed, renames_applied, errors = tc.reconcile_discord(
                conn, now, {"rename_min_interval_s": 330}, "tok", BoomOnRenameTransport(), dry_run=False
            )
            assert errors >= 1
            assert renames_applied == 0
            t1 = tc.get_thread(conn, "t1")
            # The pin write ALREADY succeeded on Discord (POST + PUT both
            # returned 2xx) before the rename crashed — it must be
            # committed and visible, not rolled back.
            assert t1["pin_message_id"] == "555"
        finally:
            conn.close()

        # Re-run the cycle's pin step with a transport that would fail the
        # test if a SECOND pin message gets created — proving the DB
        # genuinely retained the first pin and won't re-create it.
        conn = tc.connect_db(db_path)
        try:

            def forbid_second_create(method, url, token, payload, timeout):
                if method == "POST":
                    raise AssertionError("must not create a second pinned message")
                return tc.DiscordResponse(200, {"id": "555"})

            thread = tc.get_thread(conn, "t1")
            prs = [tc._pr_to_dict(p) for p in tc.get_prs(conn, "t1")]
            applied, error = tc.reconcile_pin(conn, thread, prs, "tok", forbid_second_create, dry_run=False)
            conn.commit()
            assert error is None
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Discord: token resolution
# ---------------------------------------------------------------------------

class TestDiscordTokenResolution:
    def test_env_var_wins(self, tmp_path, monkeypatch):
        monkeypatch.setenv("THREADCTL_DISCORD_TOKEN", "env-tok")
        assert _real_resolve_discord_token(tmp_path / "missing.toml", tmp_path / "missing.json") == "env-tok"

    def test_config_toml_used_when_no_env(self, tmp_path, monkeypatch):
        monkeypatch.delenv("THREADCTL_DISCORD_TOKEN", raising=False)
        cfg = tmp_path / "config.toml"
        cfg.write_text('[discord]\ntoken = "cfg-tok"\n')
        assert _real_resolve_discord_token(cfg, tmp_path / "missing.json") == "cfg-tok"

    def test_openclaw_fallback_used_last(self, tmp_path, monkeypatch):
        monkeypatch.delenv("THREADCTL_DISCORD_TOKEN", raising=False)
        oc = tmp_path / "openclaw.json"
        oc.write_text(json.dumps({"channels": {"discord": {"token": "oc-tok"}}}))
        assert _real_resolve_discord_token(tmp_path / "missing.toml", oc) == "oc-tok"

    def test_returns_none_when_nothing_configured(self, tmp_path, monkeypatch):
        monkeypatch.delenv("THREADCTL_DISCORD_TOKEN", raising=False)
        assert _real_resolve_discord_token(tmp_path / "missing.toml", tmp_path / "missing.json") is None

    def test_config_toml_wins_over_openclaw_fallback(self, tmp_path, monkeypatch):
        monkeypatch.delenv("THREADCTL_DISCORD_TOKEN", raising=False)
        cfg = tmp_path / "config.toml"
        cfg.write_text('[discord]\ntoken = "cfg-tok"\n')
        oc = tmp_path / "openclaw.json"
        oc.write_text(json.dumps({"channels": {"discord": {"token": "oc-tok"}}}))
        assert _real_resolve_discord_token(cfg, oc) == "cfg-tok"


# ---------------------------------------------------------------------------
# Discord: pinned message reconciliation
# ---------------------------------------------------------------------------

class FakeTransport:
    """Records calls and returns scripted DiscordResponse objects keyed by
    (method, url-prefix), falling back to a 200 no-op response."""

    def __init__(self, script=None):
        self.script = script or {}
        self.calls = []

    def __call__(self, method, url, token, payload, timeout):
        self.calls.append((method, url, token, payload))
        for (m, prefix), resp in self.script.items():
            if method == m and url.startswith(prefix):
                return resp
        return tc.DiscordResponse(200, {"id": "999"})


class TestReconcilePin:
    def test_creates_and_pins_message_on_first_pr(self, db_path, capsys):
        _bind(db_path, capsys)
        run_cli(["add-pr", "t1", "1"], db_path, capsys)
        conn = tc.connect_db(db_path)
        try:
            thread = tc.get_thread(conn, "t1")
            prs = [tc._pr_to_dict(p) for p in tc.get_prs(conn, "t1")]
            transport = FakeTransport({("POST", f"{tc.DISCORD_API}/channels/t1/messages"): tc.DiscordResponse(201, {"id": "555"})})
            applied, error = tc.reconcile_pin(conn, thread, prs, "tok", transport, dry_run=False)
            conn.commit()
            assert error is None
            assert applied is True
            assert [c[:2] for c in transport.calls] == [
                ("POST", f"{tc.DISCORD_API}/channels/t1/messages"),
                ("PUT", f"{tc.DISCORD_API}/channels/t1/pins/555"),
            ]
            assert tc.get_thread(conn, "t1")["pin_message_id"] == "555"
        finally:
            conn.close()

    def test_edits_in_place_when_content_changed(self, db_path, capsys):
        _bind(db_path, capsys)
        run_cli(["add-pr", "t1", "1"], db_path, capsys)
        conn = tc.connect_db(db_path)
        try:
            with conn:
                conn.execute("UPDATE threads SET pin_message_id='555' WHERE thread_id='t1'")
                tc._set_meta(conn, "pin_body:t1", "stale old body")
            thread = tc.get_thread(conn, "t1")
            prs = [tc._pr_to_dict(p) for p in tc.get_prs(conn, "t1")]
            transport = FakeTransport()
            applied, error = tc.reconcile_pin(conn, thread, prs, "tok", transport, dry_run=False)
            conn.commit()
            assert error is None
            assert applied is True
            assert transport.calls[0][:2] == ("PATCH", f"{tc.DISCORD_API}/channels/t1/messages/555")
        finally:
            conn.close()

    def test_skips_edit_when_content_unchanged(self, db_path, capsys):
        _bind(db_path, capsys)
        run_cli(["add-pr", "t1", "1"], db_path, capsys)
        conn = tc.connect_db(db_path)
        try:
            thread = tc.get_thread(conn, "t1")
            prs = [tc._pr_to_dict(p) for p in tc.get_prs(conn, "t1")]
            body = tc.render_pin_body(dict(thread), prs)
            with conn:
                conn.execute("UPDATE threads SET pin_message_id='555' WHERE thread_id='t1'")
                tc._set_meta(conn, "pin_body:t1", body)
            thread = tc.get_thread(conn, "t1")
            transport = FakeTransport()
            applied, error = tc.reconcile_pin(conn, thread, prs, "tok", transport, dry_run=False)
            assert error is None
            assert applied is False
            assert transport.calls == []
        finally:
            conn.close()

    def test_unpins_and_forgets_when_no_prs_remain(self, db_path, capsys):
        _bind(db_path, capsys)
        conn = tc.connect_db(db_path)
        try:
            with conn:
                conn.execute("UPDATE threads SET pin_message_id='555' WHERE thread_id='t1'")
                tc._set_meta(conn, "pin_body:t1", "old body")
            thread = tc.get_thread(conn, "t1")
            transport = FakeTransport()
            applied, error = tc.reconcile_pin(conn, thread, [], "tok", transport, dry_run=False)
            conn.commit()
            assert error is None
            assert applied is True
            assert transport.calls[0][:2] == ("DELETE", f"{tc.DISCORD_API}/channels/t1/pins/555")
            assert tc.get_thread(conn, "t1")["pin_message_id"] is None
            assert tc._get_meta(conn, "pin_body:t1") is None
        finally:
            conn.close()

    def test_dry_run_makes_no_discord_calls(self, db_path, capsys):
        _bind(db_path, capsys)
        run_cli(["add-pr", "t1", "1"], db_path, capsys)
        conn = tc.connect_db(db_path)
        try:
            thread = tc.get_thread(conn, "t1")
            prs = [tc._pr_to_dict(p) for p in tc.get_prs(conn, "t1")]
            transport = FakeTransport()
            applied, error = tc.reconcile_pin(conn, thread, prs, "tok", transport, dry_run=True)
            assert applied is False
            assert error is None
            assert transport.calls == []
        finally:
            conn.close()

    def test_missing_token_makes_no_discord_calls(self, db_path, capsys):
        _bind(db_path, capsys)
        run_cli(["add-pr", "t1", "1"], db_path, capsys)
        conn = tc.connect_db(db_path)
        try:
            thread = tc.get_thread(conn, "t1")
            prs = [tc._pr_to_dict(p) for p in tc.get_prs(conn, "t1")]
            transport = FakeTransport()
            applied, error = tc.reconcile_pin(conn, thread, prs, None, transport, dry_run=False)
            assert applied is False
            assert error is None
            assert transport.calls == []
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Discord: rename reconciliation
# ---------------------------------------------------------------------------

class TestReconcileRename:
    def _thread_row(self, db_path, capsys, **overrides):
        _bind(db_path, capsys)
        conn = tc.connect_db(db_path)
        try:
            if overrides:
                set_clause = ", ".join(f"{k}=?" for k in overrides)
                with conn:
                    conn.execute(f"UPDATE threads SET {set_clause} WHERE thread_id='t1'", tuple(overrides.values()))
            return tc.get_thread(conn, "t1")
        finally:
            conn.close()

    def test_skips_when_desired_equals_applied(self, db_path, capsys):
        thread = self._thread_row(db_path, capsys, desired_name="🟢 x", applied_name="🟢 x")
        transport = FakeTransport()
        now = datetime(2026, 7, 28, 12, 0, tzinfo=timezone.utc)
        applied, error = tc.reconcile_rename(None, thread, now, {"rename_min_interval_s": 330}, "tok", transport, dry_run=False)
        assert applied is False
        assert error is None
        assert transport.calls == []

    def test_skips_when_desired_is_none(self, db_path, capsys):
        thread = self._thread_row(db_path, capsys)
        transport = FakeTransport()
        now = datetime(2026, 7, 28, 12, 0, tzinfo=timezone.utc)
        applied, error = tc.reconcile_rename(None, thread, now, {"rename_min_interval_s": 330}, "tok", transport, dry_run=False)
        assert applied is False
        assert transport.calls == []

    def test_skips_within_min_interval(self, db_path, capsys):
        thread = self._thread_row(
            db_path, capsys,
            desired_name="🟢 new", applied_name="🟢 old",
            last_rename_at="2026-07-28T11:58:00Z",
        )
        transport = FakeTransport()
        now = datetime(2026, 7, 28, 12, 0, tzinfo=timezone.utc)
        applied, error = tc.reconcile_rename(None, thread, now, {"rename_min_interval_s": 330}, "tok", transport, dry_run=False)
        assert applied is False
        assert transport.calls == []

    def test_renames_after_min_interval_elapsed(self, db_path, capsys):
        thread = self._thread_row(
            db_path, capsys,
            desired_name="🟢 new", applied_name="🟢 old",
            last_rename_at="2026-07-28T11:00:00Z",
        )
        conn = tc.connect_db(db_path)
        try:
            transport = FakeTransport()
            now = datetime(2026, 7, 28, 12, 0, tzinfo=timezone.utc)
            applied, error = tc.reconcile_rename(conn, thread, now, {"rename_min_interval_s": 330}, "tok", transport, dry_run=False)
            conn.commit()
            assert error is None
            assert applied is True
            assert transport.calls == [("PATCH", f"{tc.DISCORD_API}/channels/t1", "tok", {"name": "🟢 new"})]
            updated = tc.get_thread(conn, "t1")
            assert updated["applied_name"] == "🟢 new"
            assert updated["last_rename_at"] is not None
        finally:
            conn.close()

    def test_skips_within_backoff_window(self, db_path, capsys):
        thread = self._thread_row(
            db_path, capsys,
            desired_name="🟢 new", applied_name="🟢 old",
            rename_backoff_until="2026-07-28T12:05:00Z",
        )
        transport = FakeTransport()
        now = datetime(2026, 7, 28, 12, 0, tzinfo=timezone.utc)
        applied, error = tc.reconcile_rename(None, thread, now, {"rename_min_interval_s": 330}, "tok", transport, dry_run=False)
        assert applied is False
        assert transport.calls == []

    def test_429_sets_backoff_and_records_history(self, db_path, capsys):
        thread = self._thread_row(db_path, capsys, desired_name="🟢 new", applied_name="🟢 old")
        conn = tc.connect_db(db_path)
        try:
            transport = FakeTransport({
                ("PATCH", f"{tc.DISCORD_API}/channels/t1"): tc.DiscordResponse(429, {"retry_after": 42.5}),
            })
            now = datetime(2026, 7, 28, 12, 0, tzinfo=timezone.utc)
            applied, error = tc.reconcile_rename(conn, thread, now, {"rename_min_interval_s": 330}, "tok", transport, dry_run=False)
            conn.commit()
            assert applied is False
            assert error is None
            updated = tc.get_thread(conn, "t1")
            assert updated["applied_name"] == "🟢 old"
            expected_backoff = (now + timedelta(seconds=42.5)).strftime("%Y-%m-%dT%H:%M:%SZ")
            assert updated["rename_backoff_until"] == expected_backoff
            kinds = [r["kind"] for r in conn.execute("SELECT kind FROM history WHERE thread_id='t1'")]
            assert "rename" in kinds
        finally:
            conn.close()

    def test_dry_run_makes_no_rename_call(self, db_path, capsys):
        thread = self._thread_row(db_path, capsys, desired_name="🟢 new", applied_name="🟢 old")
        transport = FakeTransport()
        now = datetime(2026, 7, 28, 12, 0, tzinfo=timezone.utc)
        applied, error = tc.reconcile_rename(None, thread, now, {"rename_min_interval_s": 330}, "tok", transport, dry_run=True)
        assert applied is False
        assert transport.calls == []

    def test_missing_token_makes_no_rename_call(self, db_path, capsys):
        thread = self._thread_row(db_path, capsys, desired_name="🟢 new", applied_name="🟢 old")
        transport = FakeTransport()
        now = datetime(2026, 7, 28, 12, 0, tzinfo=timezone.utc)
        applied, error = tc.reconcile_rename(None, thread, now, {"rename_min_interval_s": 330}, None, transport, dry_run=False)
        assert applied is False
        assert transport.calls == []

    def test_http_400_sets_last_rename_at(self, db_path, capsys):
        """R2-2 (critical): a 400 (or any non-429, non-200) response must
        still stamp last_rename_at so the coalescing window applies to the
        NEXT attempt. Before the fix, only the success path wrote
        last_rename_at, so a 400 left it NULL and the identical PATCH
        re-fired 60s later instead of waiting the full 330s."""
        thread = self._thread_row(db_path, capsys, desired_name="🟢 new", applied_name="🟢 old")
        conn = tc.connect_db(db_path)
        try:
            transport = FakeTransport({
                ("PATCH", f"{tc.DISCORD_API}/channels/t1"): tc.DiscordResponse(400, {"message": "bad request"}),
            })
            now = datetime(2026, 7, 28, 12, 0, tzinfo=timezone.utc)
            applied, error = tc.reconcile_rename(conn, thread, now, {"rename_min_interval_s": 330}, "tok", transport, dry_run=False)
            conn.commit()
            assert applied is False
            assert error is not None
            updated = tc.get_thread(conn, "t1")
            assert updated["last_rename_at"] is not None
            assert updated["applied_name"] == "🟢 old"  # not confirmed, must not change
        finally:
            conn.close()

    def test_status_zero_network_blip_sets_last_rename_at(self, db_path, capsys):
        """R2-2: a synthetic status=0 response (a network blip that may
        have hit AFTER Discord already applied the rename — see
        `_discord_transport`) must also stamp last_rename_at. Otherwise a
        read timeout right after a real rename triggers a second PATCH at
        60s instead of waiting the full 330s — guaranteeing two renames
        inside the rate-limit window."""
        thread = self._thread_row(db_path, capsys, desired_name="🟢 new", applied_name="🟢 old")
        conn = tc.connect_db(db_path)
        try:
            transport = FakeTransport({
                ("PATCH", f"{tc.DISCORD_API}/channels/t1"): tc.DiscordResponse(0, {"error": "timed out"}),
            })
            now = datetime(2026, 7, 28, 12, 0, tzinfo=timezone.utc)
            applied, error = tc.reconcile_rename(conn, thread, now, {"rename_min_interval_s": 330}, "tok", transport, dry_run=False)
            conn.commit()
            assert applied is False
            assert error is not None
            updated = tc.get_thread(conn, "t1")
            assert updated["last_rename_at"] is not None
        finally:
            conn.close()

    def test_500_response_sets_last_rename_at(self, db_path, capsys):
        thread = self._thread_row(db_path, capsys, desired_name="🟢 new", applied_name="🟢 old")
        conn = tc.connect_db(db_path)
        try:
            transport = FakeTransport({
                ("PATCH", f"{tc.DISCORD_API}/channels/t1"): tc.DiscordResponse(500, None),
            })
            now = datetime(2026, 7, 28, 12, 0, tzinfo=timezone.utc)
            applied, error = tc.reconcile_rename(conn, thread, now, {"rename_min_interval_s": 330}, "tok", transport, dry_run=False)
            conn.commit()
            updated = tc.get_thread(conn, "t1")
            assert updated["last_rename_at"] is not None
        finally:
            conn.close()

    def test_400_then_five_more_cycles_issues_only_one_patch_call(self, db_path, capsys):
        """The regression that matters most (ROUND2-TASK.md testing bar):
        after an HTTP 400, simulate N=5 more cycle passes at 60s
        intervals (the old broken re-fire cadence) and assert the PATCH
        call count. Before the fix this issued 6 PATCHes total (one per
        cycle, every cycle, forever, because last_rename_at never got
        set); after the fix, only the FIRST attempt reaches the network —
        every subsequent cycle within the 330s coalescing window is
        skipped because last_rename_at is now populated."""
        thread = self._thread_row(db_path, capsys, desired_name="🟢 new", applied_name="🟢 old")
        conn = tc.connect_db(db_path)
        try:
            transport = FakeTransport({
                ("PATCH", f"{tc.DISCORD_API}/channels/t1"): tc.DiscordResponse(400, {"message": "bad request"}),
            })
            base = datetime(2026, 7, 28, 12, 0, tzinfo=timezone.utc)
            # Cycle 0 (the initial failing attempt) plus 5 more cycles,
            # each 60s apart — matching the old broken re-fire cadence
            # described in ROUND2-FINDINGS.md ("5 cycles issued 6
            # PATCHes").
            for i in range(6):
                now = base + timedelta(seconds=60 * i)
                thread = tc.get_thread(conn, "t1")
                tc.reconcile_rename(conn, thread, now, {"rename_min_interval_s": 330}, "tok", transport, dry_run=False)
                conn.commit()
            patch_calls = [c for c in transport.calls if c[0] == "PATCH" and c[1] == f"{tc.DISCORD_API}/channels/t1"]
            assert len(patch_calls) == 1
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Discord: full cycle wiring (reconcile_discord + run_cycle)
# ---------------------------------------------------------------------------

class TestCycleDiscordWiring:
    def test_cycle_creates_pin_and_renames_thread_end_to_end(self, db_path, capsys, monkeypatch):
        _bind(db_path, capsys)
        run_cli(["add-pr", "t1", "1"], db_path, capsys)
        now = datetime(2026, 7, 28, 12, 0, tzinfo=timezone.utc)
        monkeypatch.setattr(tc, "_utcnow", lambda: now)
        monkeypatch.setattr(tc, "_now_iso", lambda: "2026-07-28T12:00:00Z")

        def runner(cmd, timeout):
            if cmd[:2] == ["gh", "auth"]:
                return __import__("subprocess").CompletedProcess(cmd, 0, "token\n", "")
            return __import__("subprocess").CompletedProcess(
                cmd, 0, json.dumps(_graphql_payload(pr_1=_github_pr(ci="FAILURE"))), ""
            )

        transport = FakeTransport({
            ("POST", f"{tc.DISCORD_API}/channels/t1/messages"): tc.DiscordResponse(201, {"id": "555"}),
        })
        result = tc.run_cycle(db_path, runner=runner, token="tok", transport=transport)
        assert result["pins_changed"] == 1
        assert result["renames_applied"] == 1
        assert result["errors"] == 0

        conn = tc.connect_db(db_path)
        try:
            thread = tc.get_thread(conn, "t1")
            assert thread["pin_message_id"] == "555"
            assert thread["applied_name"] == "🔴 Experiment recompute · 🔴#1"
        finally:
            conn.close()

    def test_dry_run_leaves_applied_name_and_pin_untouched(self, db_path, capsys, monkeypatch):
        _bind(db_path, capsys)
        run_cli(["add-pr", "t1", "1"], db_path, capsys)
        now = datetime(2026, 7, 28, 12, 0, tzinfo=timezone.utc)
        monkeypatch.setattr(tc, "_utcnow", lambda: now)
        monkeypatch.setattr(tc, "_now_iso", lambda: "2026-07-28T12:00:00Z")

        def runner(cmd, timeout):
            if cmd[:2] == ["gh", "auth"]:
                return __import__("subprocess").CompletedProcess(cmd, 0, "token\n", "")
            return __import__("subprocess").CompletedProcess(
                cmd, 0, json.dumps(_graphql_payload(pr_1=_github_pr(ci="FAILURE"))), ""
            )

        transport = FakeTransport()
        result = tc.run_cycle(db_path, runner=runner, token="tok", transport=transport, dry_run=True)
        assert result["pins_changed"] == 0
        assert result["renames_applied"] == 0
        assert transport.calls == []

        conn = tc.connect_db(db_path)
        try:
            thread = tc.get_thread(conn, "t1")
            assert thread["desired_name"] == "🔴 Experiment recompute · 🔴#1"
            assert thread["applied_name"] is None
            assert thread["pin_message_id"] is None
        finally:
            conn.close()

    def test_no_op_rename_is_skipped_across_full_cycle(self, db_path, capsys, monkeypatch):
        _bind(db_path, capsys)
        run_cli(["add-pr", "t1", "1"], db_path, capsys)
        conn = tc.connect_db(db_path)
        try:
            with conn:
                conn.execute(
                    "UPDATE threads SET applied_name='🟢 Experiment recompute · 🟢#1', pin_message_id='555' "
                    "WHERE thread_id='t1'"
                )
                tc._set_meta(conn, "pin_body:t1", "irrelevant")
        finally:
            conn.close()
        now = datetime(2026, 7, 28, 12, 0, tzinfo=timezone.utc)
        monkeypatch.setattr(tc, "_utcnow", lambda: now)
        monkeypatch.setattr(tc, "_now_iso", lambda: "2026-07-28T12:00:00Z")

        def runner(cmd, timeout):
            if cmd[:2] == ["gh", "auth"]:
                return __import__("subprocess").CompletedProcess(cmd, 0, "token\n", "")
            return __import__("subprocess").CompletedProcess(
                cmd, 0, json.dumps(_graphql_payload(pr_1=_github_pr(ci="NONE"))), ""
            )

        transport = FakeTransport()
        result = tc.run_cycle(db_path, runner=runner, token="tok", transport=transport)
        rename_calls = [c for c in transport.calls if c[0] == "PATCH" and c[1] == f"{tc.DISCORD_API}/channels/t1"]
        assert rename_calls == []
        assert result["renames_applied"] == 0


# ---------------------------------------------------------------------------
# migrate: import from the old thread-state registry.json
# ---------------------------------------------------------------------------

def _registry(**threads):
    return {"schemaVersion": 5, "accounts": {}, "threads": threads}


def _old_thread(
    thread_id="t1", title="Experiment recompute", repo="absmartly/abs", status="active",
    prs=None, created_at="2026-07-01T00:00:00Z", updated_at="2026-07-27T09:00:00Z",
    card_message_id=None, discord_name=None,
):
    return {
        "threadId": thread_id,
        "title": title,
        "description": "",
        "repo": repo,
        "status": status,
        "tags": [],
        "links": [],
        "notes": "",
        "prs": prs or [],
        "branches": [],
        "runs": [],
        "lastActivityAt": None,
        "lastActivitySource": None,
        "updates": [],
        "cardMessageId": card_message_id,
        "parentChannelId": None,
        "discordName": discord_name,
        "createdAt": created_at,
        "updatedAt": updated_at,
    }


def _old_pr(number=1, repo="absmartly/abs", primary=True, state="OPEN", ci="SUCCESS", title="Fix thing", url="https://github.com/absmartly/abs/pull/1"):
    return {
        "repo": repo,
        "number": number,
        "primary": primary,
        "prState": {
            "state": state, "ci": ci, "title": title, "url": url,
            "mergedAt": None, "checkedAt": "2026-07-27T08:00:00Z", "stale": False, "error": None,
        },
    }


class TestMigrate:
    def test_imports_thread_with_pr_and_discord_identity(self, db_path, tmp_path, capsys):
        registry = _registry(t1=_old_thread(
            prs=[_old_pr()], card_message_id="555", discord_name="🟢 Experiment recompute · 🔵#1",
        ))
        src = tmp_path / "registry.json"
        src.write_text(json.dumps(registry))

        code, out, err = run_cli(["migrate", "--from", str(src)], db_path, capsys)
        assert code == 0
        assert "migrated 1 thread" in out

        conn = tc.connect_db(db_path)
        try:
            thread = tc.get_thread(conn, "t1")
            assert thread["title"] == "Experiment recompute"
            assert thread["owner"] == "absmartly"
            assert thread["repo"] == "absmartly/abs"
            assert thread["status"] == "active"
            assert thread["last_touch_at"] == "2026-07-27T09:00:00Z"
            assert thread["pin_message_id"] == "555"
            assert thread["applied_name"] == "🟢 Experiment recompute · 🔵#1"

            prs = tc.get_prs(conn, "t1")
            assert len(prs) == 1
            assert prs[0]["number"] == 1
            assert prs[0]["is_primary"] == 1
            assert prs[0]["state"] == "OPEN"
            assert prs[0]["ci"] == "SUCCESS"
        finally:
            conn.close()

    def test_title_over_48_chars_is_truncated(self, db_path, tmp_path, capsys):
        long_title = "x" * 60
        registry = _registry(t1=_old_thread(title=long_title))
        src = tmp_path / "registry.json"
        src.write_text(json.dumps(registry))
        run_cli(["migrate", "--from", str(src)], db_path, capsys)
        conn = tc.connect_db(db_path)
        try:
            assert len(tc.get_thread(conn, "t1")["title"]) <= tc.THREAD_TITLE_MAX
        finally:
            conn.close()

    @pytest.mark.parametrize("old_status,new_status", [
        ("active", "active"), ("blocked", "active"), ("review", "active"),
        ("merged", "merged"), ("closed", "closed"), ("done", "closed"), ("paused", "paused"),
    ])
    def test_status_mapping(self, db_path, tmp_path, capsys, old_status, new_status):
        registry = _registry(t1=_old_thread(status=old_status))
        src = tmp_path / "registry.json"
        src.write_text(json.dumps(registry))
        run_cli(["migrate", "--from", str(src)], db_path, capsys)
        conn = tc.connect_db(db_path)
        try:
            assert tc.get_thread(conn, "t1")["status"] == new_status
        finally:
            conn.close()

    def test_last_touch_at_comes_from_updated_at_so_idle_threads_show_stale(self, db_path, tmp_path, capsys, monkeypatch):
        registry = _registry(t1=_old_thread(updated_at="2026-01-01T00:00:00Z"))
        src = tmp_path / "registry.json"
        src.write_text(json.dumps(registry))
        run_cli(["migrate", "--from", str(src)], db_path, capsys)

        now = datetime(2026, 7, 28, 12, 0, tzinfo=timezone.utc)
        monkeypatch.setattr(tc, "_utcnow", lambda: now)
        conn = tc.connect_db(db_path)
        try:
            thread = dict(tc.get_thread(conn, "t1"))
            icon = tc.effective_thread_icon(thread, [])
            assert tc.is_stale(icon, thread["last_touch_at"], now, 30) is True
        finally:
            conn.close()

    def test_registry_file_untouched_after_migration(self, db_path, tmp_path, capsys):
        registry = _registry(t1=_old_thread())
        src = tmp_path / "registry.json"
        original = json.dumps(registry)
        src.write_text(original)
        run_cli(["migrate", "--from", str(src)], db_path, capsys)
        assert src.read_text() == original

    def test_importing_all_25_threads_from_a_realistic_registry(self, db_path, tmp_path, capsys):
        threads = {f"t{i}": _old_thread(thread_id=f"t{i}", title=f"Thread {i}") for i in range(25)}
        registry = _registry(**threads)
        src = tmp_path / "registry.json"
        src.write_text(json.dumps(registry))
        code, out, err = run_cli(["migrate", "--from", str(src)], db_path, capsys)
        assert code == 0
        assert "migrated 25 thread" in out
        conn = tc.connect_db(db_path)
        try:
            assert len(tc.get_all_threads(conn)) == 25
        finally:
            conn.close()

    def test_already_bound_thread_is_skipped_not_clobbered(self, db_path, tmp_path, capsys):
        _bind(db_path, capsys, thread_id="t1", title="Existing live thread")
        registry = _registry(t1=_old_thread(title="Old registry title"))
        src = tmp_path / "registry.json"
        src.write_text(json.dumps(registry))
        code, out, err = run_cli(["migrate", "--from", str(src)], db_path, capsys)
        assert code == 0
        assert "migrated 0 thread" in out
        assert "skipped 1" in err
        conn = tc.connect_db(db_path)
        try:
            assert tc.get_thread(conn, "t1")["title"] == "Existing live thread"
        finally:
            conn.close()

    def test_missing_registry_file_reports_error(self, db_path, tmp_path, capsys):
        code, out, err = run_cli(["migrate", "--from", str(tmp_path / "nope.json")], db_path, capsys)
        assert code != 0
        assert "nope.json" in err

    def test_invalid_json_reports_error(self, db_path, tmp_path, capsys):
        src = tmp_path / "registry.json"
        src.write_text("not json")
        code, out, err = run_cli(["migrate", "--from", str(src)], db_path, capsys)
        assert code != 0
        assert "not valid JSON" in err

    def test_thread_with_no_prs_and_no_discord_identity_migrates_cleanly(self, db_path, tmp_path, capsys):
        registry = _registry(t1=_old_thread(card_message_id=None, discord_name=None))
        src = tmp_path / "registry.json"
        src.write_text(json.dumps(registry))
        code, out, err = run_cli(["migrate", "--from", str(src)], db_path, capsys)
        assert code == 0
        conn = tc.connect_db(db_path)
        try:
            thread = tc.get_thread(conn, "t1")
            assert thread["pin_message_id"] is None
            assert thread["applied_name"] is None
            assert tc.get_prs(conn, "t1") == []
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# HIGH 4: migration with null/invalid repo must not crash the cycle
# ---------------------------------------------------------------------------

class TestMigrateInvalidRepo:
    """RAX VERIFIED: 11 of 25 live threads have `"repo": null`. The old
    `_migrate_thread` did `repo = entry.get("repo") or ""`, silently
    storing owner='' repo=''. `_check_pr_owner` then compared ''=='' and
    passed, `add-pr` could insert a PR with repo='', and `_graphql_query`'s
    `''.split('/', 1)` raised uncaught — which, per the old CRITICAL 2
    all-in-one-transaction bug, rolled back polling+renaming for ALL 25
    threads because of one bad row."""

    def test_null_repo_is_imported_as_null_not_empty_string(self, db_path, tmp_path, capsys):
        registry = _registry(t1=_old_thread(repo=None))
        src = tmp_path / "registry.json"
        src.write_text(json.dumps(registry))
        run_cli(["migrate", "--from", str(src)], db_path, capsys)
        conn = tc.connect_db(db_path)
        try:
            thread = tc.get_thread(conn, "t1")
            assert thread["repo"] is None
            assert thread["owner"] is None
        finally:
            conn.close()

    def test_invalid_repo_shape_is_also_imported_as_null(self, db_path, tmp_path, capsys):
        registry = _registry(t1=_old_thread(repo="not-a-valid-repo-shape"))
        src = tmp_path / "registry.json"
        src.write_text(json.dumps(registry))
        run_cli(["migrate", "--from", str(src)], db_path, capsys)
        conn = tc.connect_db(db_path)
        try:
            thread = tc.get_thread(conn, "t1")
            assert thread["repo"] is None
            assert thread["owner"] is None
        finally:
            conn.close()

    def test_migrate_prints_loud_summary_of_threads_needing_manual_bind(self, db_path, tmp_path, capsys):
        registry = _registry(
            t1=_old_thread(thread_id="t1", repo=None),
            t2=_old_thread(thread_id="t2", repo="absmartly/abs"),
            t3=_old_thread(thread_id="t3", repo=None),
        )
        src = tmp_path / "registry.json"
        src.write_text(json.dumps(registry))
        code, out, err = run_cli(["migrate", "--from", str(src)], db_path, capsys)
        assert code == 0
        assert "WARNING" in err
        assert "t1" in err
        assert "t3" in err
        assert "bind --repo" in err or "bind" in err

    def test_valid_repo_thread_unaffected_by_others_invalid(self, db_path, tmp_path, capsys):
        registry = _registry(
            bad=_old_thread(thread_id="bad", repo=None),
            good=_old_thread(thread_id="good", repo="absmartly/abs"),
        )
        src = tmp_path / "registry.json"
        src.write_text(json.dumps(registry))
        run_cli(["migrate", "--from", str(src)], db_path, capsys)
        conn = tc.connect_db(db_path)
        try:
            assert tc.get_thread(conn, "good")["repo"] == "absmartly/abs"
        finally:
            conn.close()

    def test_pr_with_no_usable_repo_is_skipped_not_inserted(self, db_path, tmp_path, capsys):
        registry = _registry(t1=_old_thread(repo=None, prs=[_old_pr(repo=None)]))
        src = tmp_path / "registry.json"
        src.write_text(json.dumps(registry))
        run_cli(["migrate", "--from", str(src)], db_path, capsys)
        conn = tc.connect_db(db_path)
        try:
            assert tc.get_prs(conn, "t1") == []
        finally:
            conn.close()

    def test_bind_repairs_a_null_repo_thread(self, db_path, tmp_path, capsys):
        registry = _registry(t1=_old_thread(repo=None))
        src = tmp_path / "registry.json"
        src.write_text(json.dumps(registry))
        run_cli(["migrate", "--from", str(src)], db_path, capsys)
        code, out, err = run_cli(
            ["bind", "t1", "--title", "Fixed", "--repo", "absmartly/abs"], db_path, capsys
        )
        assert code == 0
        conn = tc.connect_db(db_path)
        try:
            thread = tc.get_thread(conn, "t1")
            assert thread["repo"] == "absmartly/abs"
            assert thread["owner"] == "absmartly"
        finally:
            conn.close()


class TestPollDuePrsSkipsInvalidRepo:
    """HIGH 4, poller side: a PR row with a NULL/invalid repo (e.g. a
    migrated thread nobody has `bind --repo`'d yet) must be skipped and
    recorded to history, never crash `_graphql_query`'s `repo.split('/')`
    and, via that, the whole cycle."""

    def test_null_repo_pr_is_skipped_and_recorded_not_raised(self, db_path, capsys, monkeypatch):
        """A PR row with an invalid repo (empty string — the shape the old
        code used to silently store; `prs.repo` stays NOT NULL, so this is
        the realistic "bad data slipped in" case, e.g. from a pre-fix DB)
        must not crash `_graphql_query`'s `repo.split('/')` and, via that,
        the whole cycle."""
        _bind(db_path, capsys, thread_id="t1", repo="absmartly/abs")
        conn = tc.connect_db(db_path)
        try:
            with conn:
                conn.execute(
                    "INSERT INTO prs (thread_id, repo, number, is_primary, created_at) VALUES (?, ?, ?, 0, ?)",
                    ("t1", "", 1, tc._now_iso()),
                )
        finally:
            conn.close()

        now = datetime(2026, 7, 28, 12, 0, tzinfo=timezone.utc)

        def runner(cmd, timeout):
            raise AssertionError("must not call gh for a PR with no usable repo")

        conn = tc.connect_db(db_path)
        try:
            refreshed, errors = tc._poll_due_prs(
                conn, now, {"pr_recheck_min_s": 150, "max_pr_checks_per_cycle": 40}, runner, tc.DEFAULT_ACCOUNTS
            )
            assert refreshed == 0
            assert errors == 1
            history = conn.execute("SELECT * FROM history WHERE thread_id='t1' AND kind='error'").fetchall()
            assert len(history) == 1
            assert "invalid repo" in history[0]["text"]
        finally:
            conn.close()

    def test_valid_repo_pr_on_other_thread_still_polled_when_sibling_has_invalid_repo(
        self, db_path, capsys, monkeypatch
    ):
        _bind(db_path, capsys, thread_id="good", repo="absmartly/abs")
        run_cli(["add-pr", "good", "1"], db_path, capsys)

        _bind(db_path, capsys, thread_id="bad", repo="absmartly/abs")
        conn = tc.connect_db(db_path)
        try:
            with conn:
                # Force an invalid-repo PR directly — simulates bad data
                # that slipped in from a pre-fix DB or a future code path
                # that doesn't validate as strictly as add-pr/migrate now do.
                conn.execute(
                    "INSERT INTO prs (thread_id, repo, number, is_primary, created_at) VALUES (?, ?, ?, 0, ?)",
                    ("bad", "", 1, tc._now_iso()),
                )
        finally:
            conn.close()

        now = datetime(2026, 7, 28, 12, 0, tzinfo=timezone.utc)

        def runner(cmd, timeout):
            if cmd[:2] == ["gh", "auth"]:
                return __import__("subprocess").CompletedProcess(cmd, 0, "token\n", "")
            payload = _graphql_payload(pr_1=_github_pr(ci="SUCCESS"))
            return __import__("subprocess").CompletedProcess(cmd, 0, json.dumps(payload), "")

        conn = tc.connect_db(db_path)
        try:
            refreshed, errors = tc._poll_due_prs(
                conn, now, {"pr_recheck_min_s": 150, "max_pr_checks_per_cycle": 40}, runner, tc.DEFAULT_ACCOUNTS
            )
            assert refreshed == 1  # good/1 still polled
            assert errors == 1     # bad/1 recorded as an error, not raised
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# HIGH 5: pinned message length cap + mention sanitisation
# ---------------------------------------------------------------------------

class TestRenderPinBodyCapAndSanitisation:
    def test_body_under_cap_is_unchanged(self):
        thread = {"title": "Small thread"}
        prs = [{"repo": "absmartly/abs", "number": 1, "primary": True, "state": "OPEN",
                "ci": "SUCCESS", "review": "APPROVED", "mergeable": "MERGEABLE",
                "title": "A normal PR title", "url": "https://github.com/absmartly/abs/pull/1"}]
        body = tc.render_pin_body(thread, prs)
        assert len(body) <= tc.DISCORD_MESSAGE_MAX
        assert "A normal PR title" in body

    def test_many_prs_is_capped_at_discord_message_max(self):
        """LLM Proxy-style thread: enough PRs (14+) to exceed Discord's
        2000-char message limit if rendered unconditionally."""
        thread = {"title": "LLM Proxy"}
        prs = [
            {
                "repo": "absmartly/abs", "number": 1000 + i, "primary": (i == 0), "state": "OPEN",
                "ci": "SUCCESS", "review": "APPROVED", "mergeable": "MERGEABLE",
                "title": "A reasonably long pull request title describing the change in detail " * 2,
                "url": f"https://github.com/absmartly/abs/pull/{1000 + i}",
            }
            for i in range(30)
        ]
        body = tc.render_pin_body(thread, prs)
        assert len(body) <= tc.DISCORD_MESSAGE_MAX

    def test_capped_body_notes_how_many_were_omitted(self):
        thread = {"title": "Many PRs"}
        prs = [
            {
                "repo": "absmartly/abs", "number": 2000 + i, "primary": False, "state": "OPEN",
                "ci": "SUCCESS", "review": "APPROVED", "mergeable": "MERGEABLE",
                "title": "x" * 150, "url": f"https://github.com/absmartly/abs/pull/{2000 + i}",
            }
            for i in range(40)
        ]
        body = tc.render_pin_body(thread, prs)
        assert len(body) <= tc.DISCORD_MESSAGE_MAX
        assert "omitted" in body

    def test_everyone_mention_in_pr_title_is_neutralised(self):
        thread = {"title": "x"}
        prs = [{"repo": "absmartly/abs", "number": 1, "primary": True, "state": "OPEN",
                "ci": "SUCCESS", "review": "APPROVED", "mergeable": "MERGEABLE",
                "title": "@everyone please review", "url": "https://github.com/absmartly/abs/pull/1"}]
        body = tc.render_pin_body(thread, prs)
        assert "@everyone" not in body
        assert "everyone" in body  # the visible text survives, just defused

    def test_here_mention_in_pr_title_is_neutralised(self):
        thread = {"title": "x"}
        prs = [{"repo": "absmartly/abs", "number": 1, "primary": True, "state": "OPEN",
                "ci": "SUCCESS", "review": "APPROVED", "mergeable": "MERGEABLE",
                "title": "@here fix this now", "url": "https://github.com/absmartly/abs/pull/1"}]
        body = tc.render_pin_body(thread, prs)
        assert "@here" not in body

    def test_role_mention_in_pr_title_is_neutralised(self):
        thread = {"title": "x"}
        prs = [{"repo": "absmartly/abs", "number": 1, "primary": True, "state": "OPEN",
                "ci": "SUCCESS", "review": "APPROVED", "mergeable": "MERGEABLE",
                "title": "ping <@&123456789012345678> please", "url": "https://github.com/absmartly/abs/pull/1"}]
        body = tc.render_pin_body(thread, prs)
        assert "<@&123456789012345678>" not in body

    def test_user_mention_in_pr_title_is_neutralised(self):
        thread = {"title": "x"}
        prs = [{"repo": "absmartly/abs", "number": 1, "primary": True, "state": "OPEN",
                "ci": "SUCCESS", "review": "APPROVED", "mergeable": "MERGEABLE",
                "title": "cc <@123456789012345678>", "url": "https://github.com/absmartly/abs/pull/1"}]
        body = tc.render_pin_body(thread, prs)
        assert "<@123456789012345678>" not in body

    def test_thread_title_mention_is_also_neutralised(self):
        thread = {"title": "@everyone read this"}
        body = tc.render_pin_body(thread, [])
        assert "@everyone" not in body

    def test_mention_in_middle_of_normal_title_still_defused(self):
        thread = {"title": "x"}
        prs = [{"repo": "absmartly/abs", "number": 1, "primary": True, "state": "OPEN",
                "ci": "SUCCESS", "review": "APPROVED", "mergeable": "MERGEABLE",
                "title": "fix bug reported by @everyone in standup", "url": "https://github.com/absmartly/abs/pull/1"}]
        body = tc.render_pin_body(thread, prs)
        assert "@everyone" not in body
        assert "fix bug reported by" in body
