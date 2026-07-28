"""Tests for toolbox.threadctl: SQLite core (WAL/busy_timeout, schema),
bind/touch/set/add-pr/rm-pr/primary-pr/show/list, icon + title rendering,
staleness, GitHub polling + cycle, Discord writes + rename reconciliation,
and migration from the old thread-state registry.json. All Discord/GitHub
calls are mocked — no live API calls."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from toolbox import threadctl as tc

# Captured before any test can monkeypatch tc._resolve_discord_token (the
# safety fixture below stubs it for every other test), so the dedicated
# token-resolution tests can still exercise the real implementation.
_real_resolve_discord_token = tc._resolve_discord_token


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
    nodes = [] if ci == "NONE" else [{"conclusion": ci, "status": None}]
    return {
        "state": state,
        "mergeable": mergeable,
        "reviewDecision": review,
        "isDraft": draft,
        "title": title,
        "url": url,
        "statusCheckRollup": {"nodes": nodes},
    }


class TestGithubPolling:
    def test_fetches_one_batched_graphql_query_with_account_token(self, monkeypatch):
        monkeypatch.setattr(tc, "_now_iso", lambda: "2026-07-28T12:00:00Z")
        calls = []

        def runner(cmd, timeout):
            calls.append(cmd)
            if cmd[:2] == ["gh", "auth"]:
                assert cmd == ["gh", "auth", "token", "--user", "marcio-absmartly"]
                return __import__("subprocess").CompletedProcess(cmd, 0, "tok\n", "")
            payload = _graphql_payload(pr_1=_github_pr(), pr_2=_github_pr(ci="PENDING"))
            return __import__("subprocess").CompletedProcess(cmd, 0, json.dumps(payload), "")

        states, error = tc.fetch_repo_pr_states(
            "absmartly/abs", [1, 2], tc.DEFAULT_ACCOUNTS, 20, runner
        )

        assert error is None
        assert states[1]["ci"] == "SUCCESS"
        assert states[2]["ci"] == "PENDING"
        assert states[1]["review"] == "APPROVED"
        assert states[1]["checked_at"] == "2026-07-28T12:00:00Z"
        assert calls[1][:2] == ["env", "GH_TOKEN=tok"]
        query = next(part for part in calls[1] if part.startswith("query="))
        assert "pullRequest(number: 1)" in query
        assert "pullRequest(number: 2)" in query
        assert not any(call[:3] == ["gh", "auth", "switch"] for call in calls)

    @pytest.mark.parametrize("rollup,expected", [
        ([{"conclusion": "FAILURE", "status": None}], "FAILURE"),
        ([{"conclusion": None, "status": "IN_PROGRESS"}], "PENDING"),
        ([], "NONE"),
    ])
    def test_ci_derivation(self, rollup, expected):
        assert tc._derive_ci(rollup) == expected

    def test_graphql_failure_is_returned_not_raised(self):
        def runner(cmd, timeout):
            return __import__("subprocess").CompletedProcess(cmd, 1, "", "bad credentials")

        states, error = tc.fetch_repo_pr_states("unknown/repo", [1], {}, 20, runner)
        assert states is None
        assert error == "bad credentials"

    def test_malformed_graphql_repository_is_returned_not_raised(self):
        def runner(cmd, timeout):
            return __import__("subprocess").CompletedProcess(
                cmd, 0, json.dumps({"data": {"repository": None}}), ""
            )

        states, error = tc.fetch_repo_pr_states("unknown/repo", [1], {}, 20, runner)
        assert states is None
        assert "invalid GraphQL JSON" in error

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
