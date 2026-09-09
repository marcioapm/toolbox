"""Tests for opencode_gc: session GC and page reclamation for opencode's DB.

Run: pytest tests/test_opencode_gc.py -q

This deletes irreplaceable history, so the tests that matter are negative:
a live session is never touched, an old parent of a live session is never
touched, a too-small retention is refused, and a dry run never writes.

Every test builds a real SQLite database with the same schema shape as the
live one -- in particular `event_sequence` with NO foreign key to `session`,
which is the whole reason a naive session delete leaks event rows.
"""
from __future__ import annotations

import sqlite3
import time

import pytest

from toolbox import opencode_gc


DAY_MS = 86_400_000


def _make_db(path, *, auto_vacuum=2):
    conn = sqlite3.connect(path, isolation_level=None)
    conn.execute(f"PRAGMA auto_vacuum={auto_vacuum}")
    conn.executescript(
        """
        CREATE TABLE session (
            id text PRIMARY KEY,
            parent_id text,
            time_created integer NOT NULL,
            time_updated integer NOT NULL
        );
        CREATE TABLE message (
            id text PRIMARY KEY,
            session_id text NOT NULL REFERENCES session(id) ON DELETE CASCADE,
            data text NOT NULL
        );
        CREATE TABLE part (
            id text PRIMARY KEY,
            message_id text NOT NULL REFERENCES message(id) ON DELETE CASCADE,
            session_id text NOT NULL,
            data text NOT NULL
        );
        -- Deliberately NO foreign key to session: this mirrors the live schema
        -- and is why deleting a session alone strands its events.
        CREATE TABLE event_sequence (
            aggregate_id text PRIMARY KEY,
            seq integer NOT NULL
        );
        CREATE TABLE event (
            id text PRIMARY KEY,
            aggregate_id text NOT NULL
                REFERENCES event_sequence(aggregate_id) ON DELETE CASCADE,
            data text NOT NULL
        );
        """
    )
    conn.execute("VACUUM")
    return conn


def _add_session(conn, sid, *, age_days, parent=None, events=3, messages=2):
    now_ms = int(time.time() * 1000)
    t = now_ms - int(age_days * DAY_MS)
    conn.execute(
        "INSERT INTO session (id, parent_id, time_created, time_updated) VALUES (?,?,?,?)",
        (sid, parent, t, t),
    )
    conn.execute("INSERT INTO event_sequence (aggregate_id, seq) VALUES (?,?)", (sid, events))
    for i in range(events):
        conn.execute(
            "INSERT INTO event (id, aggregate_id, data) VALUES (?,?,?)",
            (f"{sid}-e{i}", sid, "x" * 64),
        )
    for m in range(messages):
        mid = f"{sid}-m{m}"
        conn.execute(
            "INSERT INTO message (id, session_id, data) VALUES (?,?,?)", (mid, sid, "y" * 64)
        )
        conn.execute(
            "INSERT INTO part (id, message_id, session_id, data) VALUES (?,?,?,?)",
            (f"{mid}-p0", mid, sid, "z" * 64),
        )


@pytest.fixture()
def db(tmp_path):
    path = tmp_path / "opencode.db"
    conn = _make_db(path)
    yield path, conn
    conn.close()


def _counts(conn):
    return {
        t: conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
        for t in ("session", "message", "part", "event", "event_sequence")
    }


class TestExpirySelection:
    def test_old_sessions_expire_and_fresh_ones_do_not(self, db):
        path, conn = db
        _add_session(conn, "old", age_days=30)
        _add_session(conn, "fresh", age_days=1)
        cutoff = int((time.time() - 5 * 86400) * 1000)
        expired, kept = opencode_gc.expired_session_ids(conn, cutoff)
        assert expired == ["old"]
        assert kept == 0

    def test_boundary_is_strict(self, db):
        path, conn = db
        _add_session(conn, "just_inside", age_days=4.9)
        _add_session(conn, "just_outside", age_days=5.1)
        cutoff = int((time.time() - 5 * 86400) * 1000)
        expired, _ = opencode_gc.expired_session_ids(conn, cutoff)
        assert expired == ["just_outside"]

    def test_old_parent_of_live_child_is_kept(self, db):
        """session.parent_id has no FK, so deleting the parent would leave the
        live child pointing at a row that no longer exists."""
        path, conn = db
        _add_session(conn, "parent", age_days=30)
        _add_session(conn, "child", age_days=1, parent="parent")
        cutoff = int((time.time() - 5 * 86400) * 1000)
        expired, kept = opencode_gc.expired_session_ids(conn, cutoff)
        assert expired == []
        assert kept == 1

    def test_whole_old_chain_is_collected(self, db):
        path, conn = db
        _add_session(conn, "gp", age_days=40)
        _add_session(conn, "p", age_days=35, parent="gp")
        _add_session(conn, "c", age_days=30, parent="p")
        cutoff = int((time.time() - 5 * 86400) * 1000)
        expired, kept = opencode_gc.expired_session_ids(conn, cutoff)
        assert expired == ["c", "gp", "p"]
        assert kept == 0

    def test_grandparent_of_live_grandchild_is_kept(self, db):
        path, conn = db
        _add_session(conn, "gp", age_days=40)
        _add_session(conn, "p", age_days=35, parent="gp")
        _add_session(conn, "live", age_days=1, parent="p")
        cutoff = int((time.time() - 5 * 86400) * 1000)
        expired, kept = opencode_gc.expired_session_ids(conn, cutoff)
        assert expired == []
        assert kept == 2

    def test_parent_cycle_does_not_hang(self, db):
        """Corrupt data must not spin the ancestor walk forever."""
        path, conn = db
        _add_session(conn, "a", age_days=30)
        _add_session(conn, "b", age_days=1, parent="a")
        conn.execute("UPDATE session SET parent_id='b' WHERE id='a'")
        cutoff = int((time.time() - 5 * 86400) * 1000)
        expired, _ = opencode_gc.expired_session_ids(conn, cutoff)
        assert "a" not in expired


class TestDeletion:
    def test_events_go_too(self, db):
        """The whole point: event rows key on aggregate_id with no FK to
        session, so a session-only delete would strand every one of them."""
        path, conn = db
        _add_session(conn, "old", age_days=30, events=5, messages=2)
        _add_session(conn, "keep", age_days=1, events=4, messages=1)
        before = _counts(conn)
        assert before["event"] == 9

        opencode_gc.delete_sessions(conn, ["old"], batch=200, deadline=None)

        after = _counts(conn)
        assert after["session"] == 1
        assert after["event"] == 4, "old session's events must be gone"
        assert after["event_sequence"] == 1
        assert after["message"] == 1
        assert after["part"] == 1

    def test_no_orphans_left_behind(self, db):
        path, conn = db
        _add_session(conn, "a", age_days=30)
        _add_session(conn, "b", age_days=30)
        opencode_gc.delete_sessions(conn, ["a", "b"], batch=1, deadline=None)
        for table, column in opencode_gc.CHILD_TABLES:
            assert conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 0

    def test_live_session_rows_survive(self, db):
        path, conn = db
        _add_session(conn, "old", age_days=30)
        _add_session(conn, "live", age_days=1)
        opencode_gc.delete_sessions(conn, ["old"], batch=200, deadline=None)
        assert conn.execute(
            "SELECT count(*) FROM event WHERE aggregate_id='live'"
        ).fetchone()[0] == 3
        assert conn.execute("SELECT id FROM session").fetchall() == [("live",)]

    def test_batching_deletes_everything(self, db):
        path, conn = db
        ids = [f"s{i}" for i in range(7)]
        for sid in ids:
            _add_session(conn, sid, age_days=30)
        counts = opencode_gc.delete_sessions(conn, ids, batch=2, deadline=None)
        assert counts["session"] == 7
        assert _counts(conn)["event"] == 0

    def test_deadline_stops_early_without_corruption(self, db):
        path, conn = db
        ids = [f"s{i}" for i in range(6)]
        for sid in ids:
            _add_session(conn, sid, age_days=30)
        # Already expired: the first batch check trips immediately.
        opencode_gc.delete_sessions(conn, ids, batch=2, deadline=time.monotonic() - 1)
        c = _counts(conn)
        # Nothing half-deleted: every surviving session keeps all its rows.
        assert c["session"] * 3 == c["event"]
        assert c["session"] * 2 == c["message"]


class TestCounting:
    def test_dry_run_counts_match_what_apply_deletes(self, db):
        path, conn = db
        _add_session(conn, "old", age_days=30, events=5, messages=3)
        predicted = opencode_gc.count_rows_for(conn, ["old"], 200)
        actual = opencode_gc.delete_sessions(conn, ["old"], batch=200, deadline=None)
        assert predicted == actual


class TestIncrementalVacuum:
    def test_releases_pages_when_incremental(self, db):
        path, conn = db
        for i in range(60):
            _add_session(conn, f"s{i}", age_days=30, events=40, messages=10)
        opencode_gc.delete_sessions(
            conn, [f"s{i}" for i in range(60)], batch=200, deadline=None
        )
        assert conn.execute("PRAGMA freelist_count").fetchone()[0] > 0
        released = opencode_gc.run_incremental_vacuum(conn, pages=None, deadline=None)
        assert released > 0
        assert conn.execute("PRAGMA freelist_count").fetchone()[0] == 0

    def test_respects_page_cap(self, db):
        path, conn = db
        for i in range(60):
            _add_session(conn, f"s{i}", age_days=30, events=40, messages=10)
        opencode_gc.delete_sessions(
            conn, [f"s{i}" for i in range(60)], batch=200, deadline=None
        )
        free_before = conn.execute("PRAGMA freelist_count").fetchone()[0]
        released = opencode_gc.run_incremental_vacuum(conn, pages=1, deadline=None)
        assert 0 < released < free_before

    def test_noop_when_nothing_freed(self, db):
        path, conn = db
        _add_session(conn, "live", age_days=1)
        assert opencode_gc.run_incremental_vacuum(conn, pages=None, deadline=None) == 0

    def test_enable_is_a_noop_when_already_incremental(self, db):
        path, conn = db
        stats = opencode_gc.read_stats(conn)
        assert stats.auto_vacuum == 2
        notes = opencode_gc.enable_incremental_vacuum(conn, path, stats)
        assert "already INCREMENTAL" in notes[0]

    def test_enable_switches_from_none(self, tmp_path):
        path = tmp_path / "none.db"
        conn = _make_db(path, auto_vacuum=0)
        try:
            _add_session(conn, "s", age_days=30)
            stats = opencode_gc.read_stats(conn)
            assert stats.auto_vacuum == 0
            opencode_gc.enable_incremental_vacuum(conn, path, stats)
            assert opencode_gc.read_stats(conn).auto_vacuum == 2
        finally:
            conn.close()

    def test_enable_refuses_without_free_space(self, tmp_path):
        """A VACUUM that runs out of disk mid-rewrite is worse than not starting.

        Uses an auto_vacuum=0 database on purpose: with an already-INCREMENTAL
        one the function returns before the free-space guard, so the test would
        pass without exercising it at all.
        """
        path = tmp_path / "huge.db"
        conn = _make_db(path, auto_vacuum=0)
        try:
            _add_session(conn, "s", age_days=30)
            stats = opencode_gc.read_stats(conn)
            # Claim a database far larger than any filesystem here.
            huge = opencode_gc.DbStats(
                page_size=stats.page_size,
                page_count=10 ** 9,
                freelist_count=stats.freelist_count,
                auto_vacuum=0,
            )
            with pytest.raises(RuntimeError, match="needs"):
                opencode_gc.enable_incremental_vacuum(conn, path, huge)
            # The refusal must not have switched the mode on the way out.
            assert opencode_gc.read_stats(conn).auto_vacuum == 0
        finally:
            conn.close()


class TestStats:
    def test_reports_sizes_and_mode(self, db):
        path, conn = db
        stats = opencode_gc.read_stats(conn)
        assert stats.total_bytes == stats.page_size * stats.page_count
        assert stats.auto_vacuum_name == "INCREMENTAL"
