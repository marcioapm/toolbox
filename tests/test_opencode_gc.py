"""Tests for opencode_gc: session GC and page reclamation for opencode's DB.

Run: pytest tests/test_opencode_gc.py -q

This deletes irreplaceable history, so the tests that matter are negative:
a live session is never touched, an old parent of a live session is never
touched, a session that came back to life mid-run is never touched, a
too-small retention is refused, and a dry run never writes.

Every test builds a real SQLite database with the same schema shape as the
live one -- in particular `event_sequence` with NO foreign key to `session`,
which is the whole reason a naive session delete leaks event rows.
"""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import time
from pathlib import Path

import pytest

from toolbox import opencode_gc


DAY_MS = 86_400_000

SCHEMA = """
        CREATE TABLE session (
            id text PRIMARY KEY,
            parent_id text,
            time_created integer NOT NULL,
            time_updated integer {updated_null}
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


def _make_db(path, *, auto_vacuum=2, wal=False, nullable_time_updated=False):
    conn = sqlite3.connect(path, isolation_level=None)
    conn.execute(f"PRAGMA auto_vacuum={auto_vacuum}")
    if wal:
        conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(
        SCHEMA.format(updated_null="" if nullable_time_updated else "NOT NULL")
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
    _add_rows(conn, sid, events=events, messages=messages)


def _add_rows(conn, sid, *, events=3, messages=2):
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


def _cutoff(days=5):
    return int((time.time() - days * 86400) * 1000)


@pytest.fixture()
def db(tmp_path):
    path = tmp_path / "opencode.db"
    conn = _make_db(path)
    yield path, conn
    conn.close()


TABLES = ("session", "message", "part", "event", "event_sequence")


def _counts(conn):
    return {t: conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in TABLES}


def _snapshot(path, tables=TABLES):
    """Every row of the requested tables."""
    conn = sqlite3.connect(path)
    try:
        return {t: sorted(conn.execute(f"SELECT * FROM {t}").fetchall()) for t in tables}
    finally:
        conn.close()


def _delete(conn, ids, *, cutoff=None, batch=200, deadline=None):
    return opencode_gc.delete_sessions(
        conn, ids, cutoff_ms=_cutoff() if cutoff is None else cutoff,
        batch=batch, deadline=deadline,
    )


def _dangling_sessions(conn):
    return conn.execute(
        "SELECT id FROM session WHERE parent_id IS NOT NULL "
        "AND parent_id NOT IN (SELECT id FROM session)"
    ).fetchall()


def _run_json_cli(monkeypatch, capsys, *args):
    monkeypatch.setattr(opencode_gc.sys, "argv", ["opencode-gc", *args, "--json"])
    return opencode_gc.main(), json.loads(capsys.readouterr().out)


def _counting_clock():
    """Advances one second per read, so a deadline trips at an exact batch
    instead of at whatever the wall clock happens to do on a loaded machine."""
    ticks = iter(range(10_000))
    return lambda: float(next(ticks))


class TestExpirySelection:
    def test_old_sessions_expire_and_fresh_ones_do_not(self, db):
        path, conn = db
        _add_session(conn, "old", age_days=30)
        _add_session(conn, "fresh", age_days=1)
        sel = opencode_gc.expired_session_ids(conn, _cutoff())
        assert sel.deletable == ["old"]
        assert sel.kept_live_descendant == 0

    def test_boundary_is_exactly_strict(self, db):
        """One millisecond either side of the cutoff, and the cutoff itself.

        4.9-vs-5.1-day fixtures pass under both `<` and `<=`, so they do not
        test the boundary at all; a separate time.time() per session also
        drifts the effective cutoff between fixture and assertion.

        Mutation: in select_expired, `t < cutoff_ms` -> `t <= cutoff_ms`.
        'at' is then expired and the assertion fails.
        """
        path, conn = db
        cutoff = int((time.time() - 5 * 86400) * 1000)
        for sid, t in (("before", cutoff - 1), ("at", cutoff), ("after", cutoff + 1)):
            conn.execute(
                "INSERT INTO session (id, parent_id, time_created, time_updated) "
                "VALUES (?,NULL,?,?)", (sid, t, t),
            )
            _add_rows(conn, sid)

        sel = opencode_gc.expired_session_ids(conn, cutoff)

        assert sel.deletable == ["before"], "only strictly-older-than-cutoff may expire"

    def test_old_parent_of_live_child_is_kept(self, db):
        """session.parent_id has no FK, so deleting the parent would leave the
        live child pointing at a row that no longer exists."""
        path, conn = db
        _add_session(conn, "parent", age_days=30)
        _add_session(conn, "child", age_days=1, parent="parent")
        sel = opencode_gc.expired_session_ids(conn, _cutoff())
        assert sel.deletable == []
        assert sel.kept_live_descendant == 1

    def test_whole_old_chain_is_collected(self, db):
        path, conn = db
        _add_session(conn, "gp", age_days=40)
        _add_session(conn, "p", age_days=35, parent="gp")
        _add_session(conn, "c", age_days=30, parent="p")
        sel = opencode_gc.expired_session_ids(conn, _cutoff())
        assert sorted(sel.deletable) == ["c", "gp", "p"]
        assert sel.kept_live_descendant == 0

    def test_grandparent_of_live_grandchild_is_kept(self, db):
        path, conn = db
        _add_session(conn, "gp", age_days=40)
        _add_session(conn, "p", age_days=35, parent="gp")
        _add_session(conn, "live", age_days=1, parent="p")
        sel = opencode_gc.expired_session_ids(conn, _cutoff())
        assert sel.deletable == []
        assert sel.kept_live_descendant == 2

    def test_parent_cycle_does_not_hang(self, db):
        """Corrupt data must not spin the ancestor walk forever."""
        path, conn = db
        _add_session(conn, "a", age_days=30)
        _add_session(conn, "b", age_days=1, parent="a")
        conn.execute("UPDATE session SET parent_id='b' WHERE id='a'")
        sel = opencode_gc.expired_session_ids(conn, _cutoff())
        assert "a" not in sel.deletable


class TestNullTimeUpdated:
    """A NULL time_updated is an unknown age. Guessing that unknown means
    expired makes a destructive tool delete on no evidence at all.

    Mutation: in select_expired, `old = {sid for sid, t in updated.items()
    if t is not None and t < cutoff_ms}` -> `... if (t or 0) < cutoff_ms}`
    (the original code). Both tests below fail.
    """

    @pytest.fixture()
    def nulldb(self, tmp_path):
        path = tmp_path / "null.db"
        conn = _make_db(path, nullable_time_updated=True)
        yield path, conn
        conn.close()

    def test_unknown_age_session_is_not_selected(self, nulldb):
        path, conn = nulldb
        _add_session(conn, "unknown", age_days=30)
        conn.execute("UPDATE session SET time_updated=NULL WHERE id='unknown'")
        sel = opencode_gc.expired_session_ids(conn, _cutoff())
        assert sel.deletable == []
        assert sel.kept_unknown_age == 1

    def test_unknown_age_session_and_its_rows_survive_a_delete(self, nulldb):
        path, conn = nulldb
        _add_session(conn, "unknown", age_days=30, events=4, messages=2)
        _add_session(conn, "old", age_days=30)
        conn.execute("UPDATE session SET time_updated=NULL WHERE id='unknown'")

        sel = opencode_gc.expired_session_ids(conn, _cutoff())
        _delete(conn, sel.deletable)

        assert conn.execute("SELECT id FROM session").fetchall() == [("unknown",)]
        assert conn.execute(
            "SELECT count(*) FROM event WHERE aggregate_id='unknown'"
        ).fetchone()[0] == 4
        assert conn.execute(
            "SELECT count(*) FROM message WHERE session_id='unknown'"
        ).fetchone()[0] == 2

    def test_unknown_age_child_protects_its_old_parent(self, nulldb):
        """Mutation: same as above -- with NULL read as 0 the child is itself
        expired, so nothing protects the parent and both are selected."""
        path, conn = nulldb
        _add_session(conn, "parent", age_days=30)
        _add_session(conn, "child", age_days=30, parent="parent")
        conn.execute("UPDATE session SET time_updated=NULL WHERE id='child'")
        sel = opencode_gc.expired_session_ids(conn, _cutoff())
        assert sel.deletable == []


class TestDeletionOrder:
    """Batches commit separately, so a parent deleted in an earlier
    transaction than its child leaves a dangling parent_id if the run then
    stops. Order must be deepest-descendant-first.
    """

    def test_children_are_ordered_before_their_parents(self, db):
        path, conn = db
        _add_session(conn, "a-gp", age_days=40)
        _add_session(conn, "z-p", age_days=35, parent="a-gp")
        _add_session(conn, "m-c", age_days=30, parent="z-p")
        sel = opencode_gc.expired_session_ids(conn, _cutoff())
        pos = {sid: i for i, sid in enumerate(sel.deletable)}
        assert pos["m-c"] < pos["z-p"] < pos["a-gp"]

    def test_batch_boundary_never_strands_a_child(self, db):
        """Mutation: `ordered, cyclic = _order_descendant_first(...)` ->
        `ordered, cyclic = sorted(old - protected), []` (lexicographic, the
        original behaviour). 'a-gp' then sorts first, commits in batch 1,
        and 'm-c'/'z-p' survive pointing at a deleted parent.
        """
        path, conn = db
        _add_session(conn, "a-gp", age_days=40)
        _add_session(conn, "z-p", age_days=35, parent="a-gp")
        _add_session(conn, "m-c", age_days=30, parent="z-p")
        sel = opencode_gc.expired_session_ids(conn, _cutoff())

        # One batch commits, then stop: exactly the crash/deadline window.
        first = sel.deletable[:1]
        _delete(conn, first, batch=1)

        dangling = _dangling_sessions(conn)
        assert dangling == [], f"surviving session points at a deleted parent: {dangling}"
        assert conn.execute("SELECT count(*) FROM session").fetchone()[0] == 2

    def test_deadline_between_batches_leaves_no_dangling_parent(self, db):
        """A real stop: some batches commit, then the deadline trips. This is
        the crash/deadline window the docstring promises is safe.

        Mutation: same `_order_descendant_first` -> `sorted(...)` swap.
        Lexicographic order deletes 's0' (the root) first and leaves its
        children behind.
        """
        path, conn = db
        _add_session(conn, "s0", age_days=40)
        for i in range(1, 5):
            _add_session(conn, f"s{i}", age_days=40 - i, parent=f"s{i - 1}")
        sel = opencode_gc.expired_session_ids(conn, _cutoff())

        outcome = opencode_gc.delete_sessions(
            conn, sel.deletable, cutoff_ms=_cutoff(), batch=1,
            deadline=1.5, clock=_counting_clock(),
        )

        assert outcome.rows["session"] == 2, "the run must stop part-way through"
        dangling = _dangling_sessions(conn)
        assert dangling == []

    def test_a_parent_with_several_old_children_waits_for_all_of_them(self, db):
        """Linear chains cannot distinguish counting children from merely
        flagging that one exists: with one child per parent, `+= 1` and `= 1`
        are the same statement. A parent with three children tells them apart.

        Mutation: in _order_descendant_first, `pending_children[parent] += 1`
        -> `pending_children[parent] = 1`. The parent is then released after
        its first child, and the ids below make the heap pop it before the
        other two.

        The ids are chosen so lexicographic order is actively harmful: the
        parent 'a-par' sorts before every child, so a wrong count surfaces it
        from the heap immediately.
        """
        path, conn = db
        _add_session(conn, "a-par", age_days=40)
        for kid in ("b-kid", "c-kid", "d-kid"):
            _add_session(conn, kid, age_days=30, parent="a-par")
        sel = opencode_gc.expired_session_ids(conn, _cutoff())

        pos = {sid: i for i, sid in enumerate(sel.deletable)}
        assert set(pos) == {"a-par", "b-kid", "c-kid", "d-kid"}
        assert pos["a-par"] == 3, "the parent must come after all three children"

        # And the ordering must survive a real interrupted run, not just an
        # index check: one batch commits, then the run stops.
        _delete(conn, sel.deletable[:1], batch=1)
        dangling = _dangling_sessions(conn)
        assert dangling == [], f"surviving session points at a deleted parent: {dangling}"

    def test_a_shared_grandparent_waits_for_every_branch(self, db):
        """Two branches of different depths under one root. A miscount frees
        the root as soon as the shallower branch drains.

        Mutation: same `+= 1` -> `= 1` swap.
        """
        path, conn = db
        _add_session(conn, "a-root", age_days=50)
        _add_session(conn, "b-shallow", age_days=45, parent="a-root")
        _add_session(conn, "c-mid", age_days=44, parent="a-root")
        _add_session(conn, "d-deep", age_days=40, parent="c-mid")
        sel = opencode_gc.expired_session_ids(conn, _cutoff())

        pos = {sid: i for i, sid in enumerate(sel.deletable)}
        assert pos["d-deep"] < pos["c-mid"] < pos["a-root"]
        assert pos["b-shallow"] < pos["a-root"]

        # Stop after every batch but the last: no prefix may strand a child.
        for stop in range(1, len(sel.deletable)):
            probe = _make_db(path.with_name(f"probe{stop}.db"))
            try:
                _add_session(probe, "a-root", age_days=50)
                _add_session(probe, "b-shallow", age_days=45, parent="a-root")
                _add_session(probe, "c-mid", age_days=44, parent="a-root")
                _add_session(probe, "d-deep", age_days=40, parent="c-mid")
                _delete(probe, sel.deletable[:stop], batch=1)
                dangling = _dangling_sessions(probe)
                assert dangling == [], f"stopping after {stop} batch(es) stranded {dangling}"
            finally:
                probe.close()

    def test_cyclic_component_is_retained_not_reordered(self, db):
        """Two old sessions pointing at each other have no safe order.

        Mutation: in _order_descendant_first, return `sorted(deletable), []`
        instead of `(ordered, sorted(deletable - set(ordered)))`. The cycle is
        then deleted one id per batch, stranding the other half.
        """
        path, conn = db
        _add_session(conn, "x", age_days=30)
        _add_session(conn, "y", age_days=30, parent="x")
        conn.execute("UPDATE session SET parent_id='y' WHERE id='x'")
        _add_session(conn, "plain", age_days=30)

        sel = opencode_gc.expired_session_ids(conn, _cutoff())
        assert sel.deletable == ["plain"]
        assert sel.kept_parent_cycle == 2

        _delete(conn, sel.deletable, batch=1)
        assert sorted(r[0] for r in conn.execute("SELECT id FROM session")) == ["x", "y"]


class TestConcurrentRevival:
    """The database is live. Selection and deletion are different instants,
    and BEGIN IMMEDIATE only serialises writers from the lock onward -- it
    does not make an earlier SELECT current.

    Mutation for both tests: in delete_sessions, replace
    `doomed = [sid for sid in chunk if sid in still]` with `doomed = chunk`
    (the original behaviour). Both fail: the revived session is destroyed.
    """

    @pytest.fixture()
    def live(self, tmp_path):
        path = tmp_path / "live.db"
        setup = _make_db(path, wal=True)
        setup.close()
        gc_conn = opencode_gc.connect(path, read_only=False, timeout_s=5)
        app = sqlite3.connect(path, isolation_level=None, timeout=5)
        app.execute("PRAGMA busy_timeout=5000")
        yield path, gc_conn, app
        gc_conn.close()
        app.close()

    def test_session_touched_after_selection_survives(self, live):
        path, gc_conn, app = live
        _add_session(app, "s", age_days=30, events=4)
        cutoff = _cutoff()

        selected = opencode_gc.expired_session_ids(gc_conn, cutoff).deletable
        assert selected == ["s"], "the test must start from a genuinely expired session"

        # opencode writes to the session between selection and deletion.
        app.execute(
            "UPDATE session SET time_updated=? WHERE id='s'", (int(time.time() * 1000),)
        )

        outcome = opencode_gc.delete_sessions(
            gc_conn, selected, cutoff_ms=cutoff, batch=200, deadline=None
        )

        assert outcome.rows["session"] == 0
        assert outcome.skipped == ["s"]
        assert app.execute("SELECT id FROM session").fetchall() == [("s",)]
        assert app.execute(
            "SELECT count(*) FROM event WHERE aggregate_id='s'"
        ).fetchone()[0] == 4

    def test_live_child_created_after_selection_protects_its_parent(self, live):
        path, gc_conn, app = live
        _add_session(app, "p", age_days=30, events=4)
        cutoff = _cutoff()

        selected = opencode_gc.expired_session_ids(gc_conn, cutoff).deletable
        assert selected == ["p"]

        # opencode forks a subagent session under the expired parent.
        now = int(time.time() * 1000)
        app.execute(
            "INSERT INTO session (id, parent_id, time_created, time_updated) "
            "VALUES ('kid','p',?,?)", (now, now),
        )

        outcome = opencode_gc.delete_sessions(
            gc_conn, selected, cutoff_ms=cutoff, batch=200, deadline=None
        )

        assert outcome.rows["session"] == 0
        assert sorted(r[0] for r in app.execute("SELECT id FROM session")) == ["kid", "p"]
        assert app.execute(
            "SELECT count(*) FROM event WHERE aggregate_id='p'"
        ).fetchone()[0] == 4


class TestMidRunGraphChanges:
    """The batch a transaction deletes must come from the graph under its own
    write lock, not from an order computed before it.

    Intersecting a pre-computed chunk with a freshly-computed eligibility set
    keeps membership current but keeps the *stale* ordering and batch
    boundaries, so the descendant-first guarantee only holds for the topology
    that existed at selection time.

    Mutation for every test in this class: in delete_sessions, restore the
    round-2 body of the loop --

        chunk = <the next `batch` ids of session_ids in order>
        still = set(expired_session_ids(conn, cutoff_ms).deletable)
        doomed = [sid for sid in chunk if sid in still]

    i.e. iterate `for start in range(0, len(session_ids), batch)` and drop the
    `restrict_to=pending` recomputation.
    """

    @pytest.fixture()
    def live(self, tmp_path):
        path = tmp_path / "midrun.db"
        setup = _make_db(path, wal=True)
        setup.close()
        gc_conn = opencode_gc.connect(path, read_only=False, timeout_s=5)
        app = sqlite3.connect(str(path), isolation_level=None, timeout=5)
        app.execute("PRAGMA busy_timeout=5000")
        yield path, gc_conn, app
        gc_conn.close()
        app.close()

    def test_old_child_inserted_after_selection_protects_its_parent(self, live):
        """A session that was never a candidate is retained whatever its age,
        so it still shields the ancestors it acquires.

        No deadline and no interruption: the parent is simply deleted while an
        old child points at it, and the child is left dangling. Membership
        revalidation alone cannot see this -- the new child is itself expired,
        so the parent stays in the fresh eligibility set.
        """
        path, gc_conn, app = live
        _add_session(app, "par", age_days=30, events=4)
        cutoff = _cutoff()
        selected = opencode_gc.expired_session_ids(gc_conn, cutoff).deletable
        assert selected == ["par"]

        # opencode forks a subagent under the expired parent and backdates it,
        # or a clock correction lands: either way the child is old and new.
        old = int((time.time() - 30 * 86400) * 1000)
        app.execute(
            "INSERT INTO session (id, parent_id, time_created, time_updated) "
            "VALUES ('kid','par',?,?)", (old, old),
        )

        outcome = opencode_gc.delete_sessions(
            gc_conn, selected, cutoff_ms=cutoff, batch=10, deadline=None
        )

        assert _dangling_sessions(app) == [], "'kid' must not outlive its parent"
        assert sorted(r[0] for r in app.execute("SELECT id FROM session")) == ["kid", "par"]
        assert outcome.rows["session"] == 0
        assert outcome.skipped == ["par"]
        assert app.execute(
            "SELECT count(*) FROM event WHERE aggregate_id='par'"
        ).fetchone()[0] == 4

    def test_concurrent_reparent_is_reordered_before_the_next_batch(self, live):
        """Two unrelated candidates become parent and child mid-run. The
        stale order deletes the new parent first and strands the child.
        """
        path, gc_conn, app = live
        _add_session(app, "aaa", age_days=30)
        _add_session(app, "bbb", age_days=31)
        cutoff = _cutoff()
        selected = opencode_gc.expired_session_ids(gc_conn, cutoff).deletable
        assert selected == ["aaa", "bbb"], "the stale order must put the parent first"

        app.execute("UPDATE session SET parent_id='aaa' WHERE id='bbb'")

        # batch=1 with a deadline that trips after the first commit: exactly
        # the crash window where a stranded child becomes permanent.
        outcome = opencode_gc.delete_sessions(
            gc_conn, selected, cutoff_ms=cutoff, batch=1,
            deadline=0.5, clock=_counting_clock(),
        )

        assert outcome.rows["session"] == 1, "the run must stop after one batch"
        assert _dangling_sessions(app) == []
        surviving = sorted(r[0] for r in app.execute("SELECT id FROM session"))
        assert surviving == ["aaa"], "the child must be deleted before its new parent"

    def test_a_session_revived_mid_run_still_protects_its_ancestors(self, live):
        """The chain is fully expired at selection; the leaf is then touched.
        The whole chain above it must be spared, not merely the leaf.
        """
        path, gc_conn, app = live
        _add_session(app, "gp", age_days=40)
        _add_session(app, "p", age_days=35, parent="gp")
        _add_session(app, "c", age_days=30, parent="p")
        cutoff = _cutoff()
        selected = opencode_gc.expired_session_ids(gc_conn, cutoff).deletable
        assert sorted(selected) == ["c", "gp", "p"]

        app.execute(
            "UPDATE session SET time_updated=? WHERE id='c'", (int(time.time() * 1000),)
        )

        outcome = opencode_gc.delete_sessions(
            gc_conn, selected, cutoff_ms=cutoff, batch=1, deadline=None
        )

        assert outcome.rows["session"] == 0
        assert sorted(r[0] for r in app.execute("SELECT id FROM session")) == \
            ["c", "gp", "p"]
        assert sorted(outcome.skipped) == ["c", "gp", "p"]
        assert outcome.remaining == 0

    def test_a_cycle_formed_mid_run_is_retained(self, live):
        """A parent_id cycle appearing after selection has no safe order, so
        both members must survive rather than be deleted in the stale one.
        """
        path, gc_conn, app = live
        _add_session(app, "x", age_days=30)
        _add_session(app, "y", age_days=31, parent="x")
        _add_session(app, "plain", age_days=32)
        cutoff = _cutoff()
        selected = opencode_gc.expired_session_ids(gc_conn, cutoff).deletable
        assert sorted(selected) == ["plain", "x", "y"]

        app.execute("UPDATE session SET parent_id='y' WHERE id='x'")

        outcome = opencode_gc.delete_sessions(
            gc_conn, selected, cutoff_ms=cutoff, batch=1, deadline=None
        )

        assert sorted(r[0] for r in app.execute("SELECT id FROM session")) == ["x", "y"]
        assert outcome.rows["session"] == 1
        assert sorted(outcome.skipped) == ["x", "y"]
        assert outcome.remaining == 0


class TestRolledBackWorkIsNotReported:
    """Counts describe committed, irreversible work -- that is the contract an
    operator uses to decide what to restore. A DELETE whose transaction rolls
    back deleted nothing.

    Mutation: in delete_sessions, accumulate straight into `outcome.rows[...]`
    at each DELETE instead of into the transaction-local `committed` dict.
    Every test here fails: the rolled-back child rows are reported as gone.
    """

    def _abort_on(self, conn, table):
        conn.execute(
            f"CREATE TRIGGER abort_{table} BEFORE DELETE ON {table} "
            f"BEGIN SELECT RAISE(ABORT,'boom'); END"
        )

    def test_counts_exclude_a_rolled_back_batch(self, db):
        path, conn = db
        _add_session(conn, "old", age_days=30, events=3, messages=2)
        # The child deletes succeed, then the session delete aborts: the
        # window where counts and disk can disagree.
        self._abort_on(conn, "session")
        before = _counts(conn)

        outcome = _delete(conn, ["old"], batch=1)

        assert outcome.failure is not None and "boom" in outcome.failure
        assert _counts(conn) == before, "the rollback must restore every row"
        assert outcome.rows == {
            "part": 0, "message": 0, "event": 0, "event_sequence": 0, "session": 0,
        }

    def test_the_exact_pre_run_snapshot_is_restored(self, tmp_path):
        """A real RAISE(ABORT) after child DELETEs have run, asserting the
        rollback path itself: rows restored byte for byte, no transaction left
        open, and the write lock released to another connection.

        Mutation: replace _rollback_quietly's body with `pass` (a no-op). The
        aborted statement leaves the transaction open, so the snapshot below
        still differs and the second connection cannot take BEGIN IMMEDIATE.
        """
        path = tmp_path / "rollback.db"
        conn = _make_db(path, wal=True)
        try:
            _add_session(conn, "old", age_days=30, events=3, messages=2)
            _add_session(conn, "other", age_days=31, events=2, messages=1)
            self._abort_on(conn, "session")
            before = _snapshot(path)

            outcome = _delete(conn, ["old", "other"], batch=1)

            assert outcome.failure is not None and "boom" in outcome.failure
            assert _snapshot(path) == before, "rollback must restore every row"
            assert conn.in_transaction is False, "no transaction may be left open"
            assert outcome.rows["session"] == 0
            assert sum(outcome.rows.values()) == 0, \
                "nothing committed, so nothing may be reported as deleted"

            # The write lock must be free: a stuck transaction would block
            # opencode itself indefinitely.
            other = sqlite3.connect(str(path), isolation_level=None, timeout=0)
            try:
                other.execute("PRAGMA busy_timeout=0")
                other.execute("BEGIN IMMEDIATE")
                other.execute("ROLLBACK")
            finally:
                other.close()
        finally:
            conn.close()

    def test_an_earlier_committed_batch_is_still_reported(self, db):
        """The counts must be neither overstated nor lost: work that did
        commit stays reported when a later batch rolls back.
        """
        path, conn = db
        _add_session(conn, "aaa", age_days=30, events=3, messages=2)
        _add_session(conn, "bbb", age_days=31, events=3, messages=2)
        # Abort only once 'aaa' is already gone.
        conn.execute(
            "CREATE TRIGGER abort_second BEFORE DELETE ON session "
            "WHEN OLD.id='bbb' BEGIN SELECT RAISE(ABORT,'boom'); END"
        )

        outcome = _delete(conn, ["aaa", "bbb"], batch=1)

        assert outcome.failure is not None
        on_disk = _counts(conn)
        assert on_disk["session"] == 1, "exactly one session delete committed"
        assert outcome.rows["session"] == 1
        # 'bbb' still has all of its rows: its batch rolled back entirely.
        assert outcome.rows["event"] == 3
        assert on_disk["event"] == 3
        assert outcome.rows["message"] == 2
        assert outcome.rows["part"] == 2


    def test_a_failing_commit_reports_nothing_as_deleted(self, tmp_path):
        """The DELETEs all succeed and only COMMIT fails -- a deferred foreign
        key is checked at commit time, not at the statement. Merging the
        counts anywhere before COMMIT returns is therefore not equivalent to
        merging after it.

        Mutation: in delete_sessions, move the
        `for table, n in committed.items(): outcome.rows[table] += n` merge to
        just above `conn.execute("COMMIT")`. Every DELETE has run by then, so
        the counts are merged and the failed COMMIT leaves them standing.
        """
        path = tmp_path / "commitfail.db"
        conn = _make_db(path)
        try:
            _add_session(conn, "old", age_days=30, events=3, messages=2)
            # An audit row that must always point at a live session, checked
            # at COMMIT rather than at the DELETE.
            conn.execute(
                "CREATE TABLE audit (id integer PRIMARY KEY, session_id text NOT NULL "
                "REFERENCES session(id) DEFERRABLE INITIALLY DEFERRED)"
            )
            conn.execute("INSERT INTO audit VALUES (1,'old')")
            conn.execute("PRAGMA foreign_keys=ON")
            before = _counts(conn)

            outcome = _delete(conn, ["old"], batch=1)

            assert outcome.failure is not None, "the COMMIT must have failed"
            assert "FOREIGN KEY" in outcome.failure
            assert _counts(conn) == before, "the rollback must restore every row"
            assert sum(outcome.rows.values()) == 0, (
                "the transaction never committed, so no row may be reported "
                f"as deleted: {outcome.rows}"
            )
        finally:
            conn.close()


class TestRevalidationHoldsTheWriteLock:
    """Re-reading the graph before BEGIN IMMEDIATE is not revalidation: the
    window between the read and the lock is exactly the window the read exists
    to close. The competing writer must be shut out for the whole of it.

    Mutation: in delete_sessions, hoist the `expired_session_ids(...)` call to
    just before `conn.execute("BEGIN IMMEDIATE")` (reverting the round-1 fix).
    The writer below then succeeds during revalidation instead of being
    blocked, and the assertion on `blocked` fails.
    """

    def test_a_competing_writer_cannot_write_while_the_graph_is_re_read(self, tmp_path):
        path = tmp_path / "interleave.db"
        setup = _make_db(path, wal=True)
        _add_session(setup, "s", age_days=30)
        setup.close()

        # timeout=0: a blocked writer reports SQLITE_BUSY immediately instead
        # of waiting, so the test observes the lock rather than a delay.
        writer = sqlite3.connect(str(path), isolation_level=None, timeout=0)
        writer.execute("PRAGMA busy_timeout=0")

        attempts = []

        class WritesDuringRevalidation(sqlite3.Connection):
            def execute(self, sql, *a):
                if sql.startswith("SELECT id, parent_id, time_updated"):
                    # Revalidation is running. If it is genuinely under the
                    # write lock, this cannot get in.
                    try:
                        writer.execute("BEGIN IMMEDIATE")
                        writer.execute(
                            "UPDATE session SET time_updated=? WHERE id='s'",
                            (int(time.time() * 1000),),
                        )
                        writer.execute("COMMIT")
                        attempts.append("wrote")
                    except sqlite3.OperationalError as exc:
                        attempts.append(f"blocked: {exc}")
                return super().execute(sql, *a)

        gc_conn = sqlite3.connect(
            str(path), isolation_level=None, timeout=0,
            factory=WritesDuringRevalidation,
        )
        gc_conn.execute("PRAGMA busy_timeout=0")
        try:
            cutoff = _cutoff()
            outcome = opencode_gc.delete_sessions(
                gc_conn, ["s"], cutoff_ms=cutoff, batch=1, deadline=None
            )

            assert attempts, "revalidation must have run at least once"
            blocked = [a for a in attempts if a.startswith("blocked")]
            assert blocked == attempts, (
                "a writer got in while the graph was being re-read, so the "
                f"revalidation is not under the write lock: {attempts}"
            )
            # The write never landed, so the session was still expired and dies.
            assert outcome.rows["session"] == 1
            assert outcome.failure is None
        finally:
            gc_conn.close()
            writer.close()


class TestDeletion:
    def test_events_go_too(self, db):
        """The whole point: event rows key on aggregate_id with no FK to
        session, so a session-only delete would strand every one of them."""
        path, conn = db
        _add_session(conn, "old", age_days=30, events=5, messages=2)
        _add_session(conn, "keep", age_days=1, events=4, messages=1)
        before = _counts(conn)
        assert before["event"] == 9

        _delete(conn, ["old"])

        after = _counts(conn)
        assert after["session"] == 1
        assert after["event"] == 4, "old session's events must be gone"
        assert after["event_sequence"] == 1
        assert after["message"] == 1
        assert after["part"] == 1

    def test_no_orphans_left_behind(self, db):
        """Mutation: remove ("event_sequence", "aggregate_id") from
        CHILD_TABLES. The stranded event_sequence row is then found here,
        because the table list below is the schema's, not production's.
        """
        path, conn = db
        _add_session(conn, "a", age_days=30)
        _add_session(conn, "b", age_days=30)
        _delete(conn, ["a", "b"], batch=1)
        # Enumerated independently of CHILD_TABLES: deriving them from the
        # production constant would let dropping a table from that constant
        # also drop the assertion that its rows were cleaned.
        leftovers = {
            "session": conn.execute("SELECT count(*) FROM session").fetchone()[0],
            "message": conn.execute("SELECT count(*) FROM message").fetchone()[0],
            "part": conn.execute("SELECT count(*) FROM part").fetchone()[0],
            "event": conn.execute("SELECT count(*) FROM event").fetchone()[0],
            "event_sequence":
                conn.execute("SELECT count(*) FROM event_sequence").fetchone()[0],
        }
        assert leftovers == {
            "session": 0, "message": 0, "part": 0, "event": 0, "event_sequence": 0,
        }

    def test_live_session_rows_survive(self, db):
        path, conn = db
        _add_session(conn, "old", age_days=30)
        _add_session(conn, "live", age_days=1)
        _delete(conn, ["old"])
        assert conn.execute(
            "SELECT count(*) FROM event WHERE aggregate_id='live'"
        ).fetchone()[0] == 3
        assert conn.execute("SELECT id FROM session").fetchall() == [("live",)]

    def test_a_live_id_passed_in_is_refused(self, db):
        """delete_sessions authorises against cutoff_ms, not its argument."""
        path, conn = db
        _add_session(conn, "live", age_days=1)
        outcome = _delete(conn, ["live"])
        assert outcome.rows["session"] == 0
        assert outcome.skipped == ["live"]
        assert conn.execute("SELECT id FROM session").fetchall() == [("live",)]

    def test_batching_deletes_everything(self, db):
        path, conn = db
        ids = [f"s{i}" for i in range(7)]
        for sid in ids:
            _add_session(conn, sid, age_days=30)
        outcome = _delete(conn, ids, batch=2)
        assert outcome.rows["session"] == 7
        assert _counts(conn)["event"] == 0


class TestCounting:
    def test_dry_run_counts_match_what_apply_deletes(self, db):
        path, conn = db
        _add_session(conn, "old", age_days=30, events=5, messages=3)
        predicted = opencode_gc.count_rows_for(conn, ["old"], 200)
        actual = _delete(conn, ["old"]).rows
        assert predicted == actual


class TestIncrementalVacuum:
    def test_releases_pages_when_incremental(self, db):
        path, conn = db
        for i in range(60):
            _add_session(conn, f"s{i}", age_days=30, events=40, messages=10)
        _delete(conn, [f"s{i}" for i in range(60)])
        assert conn.execute("PRAGMA freelist_count").fetchone()[0] > 0
        vac = opencode_gc.run_incremental_vacuum(conn, pages=None, deadline=None)
        assert vac.released > 0
        assert vac.deadline_reached is False
        assert vac.remaining == 0
        assert conn.execute("PRAGMA freelist_count").fetchone()[0] == 0

    def test_respects_page_cap(self, db):
        """`pages=1` must release one page, not merely fewer than all of them.

        Mutation: in run_incremental_vacuum, `budget = before if pages is None
        else min(pages, before)` -> `budget = before`. The old assertion
        (`0 < released < free_before`) passed under any over-release; this one
        pins the cap and checks the freelist moved by exactly that much.
        """
        path, conn = db
        for i in range(60):
            _add_session(conn, f"s{i}", age_days=30, events=40, messages=10)
        _delete(conn, [f"s{i}" for i in range(60)])
        free_before = conn.execute("PRAGMA freelist_count").fetchone()[0]
        assert free_before > 1, "the fixture must leave more pages than the cap"

        vac = opencode_gc.run_incremental_vacuum(conn, pages=1, deadline=None)

        assert vac.released == 1, "the cap is one page, not 'fewer than all'"
        free_after = conn.execute("PRAGMA freelist_count").fetchone()[0]
        assert free_before - free_after == vac.released, (
            "the reported count must match the pages that actually left the "
            f"freelist: {free_before} -> {free_after}, reported {vac.released}"
        )

    def test_noop_when_nothing_freed(self, db):
        path, conn = db
        _add_session(conn, "live", age_days=1)
        vac = opencode_gc.run_incremental_vacuum(conn, pages=None, deadline=None)
        assert vac.released == 0
        assert vac.deadline_reached is False

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

    def test_enable_from_full_needs_no_vacuum(self, tmp_path):
        """FULL -> INCREMENTAL is a header change. Demanding VACUUM space for
        it would refuse a conversion that costs nothing.

        Mutation: delete the `if stats.auto_vacuum == 1:` branch in
        enable_incremental_vacuum. The free-space guard then rejects the
        oversized DbStats and the RuntimeError propagates.
        """
        path = tmp_path / "full.db"
        conn = _make_db(path, auto_vacuum=1)
        try:
            _add_session(conn, "s", age_days=30)
            stats = opencode_gc.read_stats(conn)
            assert stats.auto_vacuum == 1
            # Larger than any filesystem: a VACUUM-requiring path must refuse.
            huge = opencode_gc.DbStats(
                page_size=stats.page_size, page_count=10 ** 9,
                freelist_count=stats.freelist_count, auto_vacuum=1,
            )
            notes = opencode_gc.enable_incremental_vacuum(conn, path, huge)
            assert "no VACUUM needed" in notes[0]
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

    def test_guard_demands_the_documented_two_copies(self, tmp_path, monkeypatch):
        """SQLite documents VACUUM as needing up to twice the file size. A
        database with 1.5x its size free must be refused; 2.5x proceeds.

        Mutation: `VACUUM_COPY_FACTOR = 2` -> `1` (or the pre-review 1.1
        multiplier). The 1.5x case then passes the guard and nothing raises.
        """
        path = tmp_path / "guard.db"
        conn = _make_db(path, auto_vacuum=0)
        try:
            _add_session(conn, "s", age_days=30)
            real = opencode_gc.read_stats(conn)
            # 40 GB claimed, so the proportional reserve dominates the floor
            # and the assertion is about the copy factor, not the reserve.
            stats = opencode_gc.DbStats(
                page_size=4096, page_count=10 ** 7, freelist_count=0, auto_vacuum=0
            )
            payload, _, _ = opencode_gc.vacuum_space_plan(path.resolve(), stats)
            assert payload * opencode_gc.VACUUM_RESERVE_FRACTION > \
                opencode_gc.MIN_VACUUM_RESERVE_BYTES

            def fake_usage(free):
                return lambda _p: type("U", (), {"free": free})()

            monkeypatch.setattr(opencode_gc.shutil, "disk_usage", fake_usage(int(payload * 1.5)))
            with pytest.raises(RuntimeError, match="needs"):
                opencode_gc.enable_incremental_vacuum(conn, path, stats)
            assert opencode_gc.read_stats(conn).auto_vacuum == 0

            monkeypatch.setattr(opencode_gc.shutil, "disk_usage", fake_usage(int(payload * 2.5)))
            opencode_gc.enable_incremental_vacuum(conn, path, stats)
            assert opencode_gc.read_stats(conn).auto_vacuum == 2
            assert real.auto_vacuum == 0
        finally:
            conn.close()

    def test_guard_counts_the_wal(self, tmp_path):
        """An uncheckpointed WAL is data VACUUM has to copy too.

        Mutation: drop the `+ wal` term from vacuum_space_plan's payload.
        """
        path = tmp_path / "wal.db"
        conn = _make_db(path, auto_vacuum=0, wal=True)
        try:
            for i in range(200):
                _add_session(conn, f"s{i}", age_days=30, events=20, messages=5)
            wal_bytes = path.with_name(path.name + "-wal").stat().st_size
            assert wal_bytes > 0, "fixture must leave an uncheckpointed WAL"
            stats = opencode_gc.read_stats(conn)
            payload, _, _ = opencode_gc.vacuum_space_plan(path.resolve(), stats)
            assert payload >= stats.total_bytes + wal_bytes
        finally:
            conn.close()


class TestVacuumGuardBehaviour:
    """The guard is only worth having if it actually refuses. Every test here
    drives enable_incremental_vacuum end to end with a mocked filesystem and
    asserts on whether the conversion happened, not on the arithmetic.
    """

    def _db(self, tmp_path, name):
        path = tmp_path / name
        conn = _make_db(path, auto_vacuum=0)
        _add_session(conn, "s", age_days=30)
        return path, conn

    def _free(self, monkeypatch, free_bytes):
        monkeypatch.setattr(
            opencode_gc.shutil, "disk_usage",
            lambda _p: type("U", (), {"free": int(free_bytes)})(),
        )

    def _stat_devices(self, monkeypatch, *, tmp_dev, db_dev, tmp_dir):
        """Report distinct st_dev for SQLite's temp dir and the DB's dir."""
        real_stat = opencode_gc.os.stat
        tmp_real = real_stat(tmp_dir)

        def fake_stat(p, *a, **kw):
            st = real_stat(p, *a, **kw)
            dev = tmp_dev if os.path.samestat(st, tmp_real) else db_dev
            return type(st)((st.st_mode, st.st_ino, dev, st.st_nlink, st.st_uid,
                             st.st_gid, st.st_size, int(st.st_atime),
                             int(st.st_mtime), int(st.st_ctime)))

        monkeypatch.setattr(opencode_gc.os, "stat", fake_stat)

    @pytest.mark.parametrize("page_count,dominant", [
        # 40 GB: 5% (2 GB) dwarfs the 64 MiB floor.
        (10 ** 7, "fraction"),
        # 4 MB: 5% is 200 KB, so the 64 MiB floor is what applies.
        (10 ** 3, "floor"),
    ])
    def test_threshold_is_exact_on_both_reserve_branches(
        self, tmp_path, monkeypatch, page_count, dominant
    ):
        """One byte below the documented requirement must refuse; the
        requirement itself must proceed. Both reserve branches are covered, so
        neither can be neutralised on its own.

        Mutations, each killed by one parametrisation or the other:
          * `MIN_VACUUM_RESERVE_BYTES = 64 * 1024 ** 2` -> `0`
            (the 'floor' case then accepts one byte too few);
          * `VACUUM_RESERVE_FRACTION = 0.05` -> `0`
            (the 'fraction' case then accepts one byte too few);
          * `reserve = max(...)` -> `reserve = 0` -- both cases accept;
          * `VACUUM_COPY_FACTOR = 2` -> `1` -- both accept far too little.
        """
        path, conn = self._db(tmp_path, f"reserve-{page_count}.db")
        try:
            stats = opencode_gc.DbStats(
                page_size=4096, page_count=page_count, freelist_count=0, auto_vacuum=0
            )
            payload, need_db_fs, _ = opencode_gc.vacuum_space_plan(path.resolve(), stats)
            reserve = need_db_fs - opencode_gc.VACUUM_COPY_FACTOR * payload
            if dominant == "fraction":
                assert reserve > opencode_gc.MIN_VACUUM_RESERVE_BYTES
            else:
                assert reserve == opencode_gc.MIN_VACUUM_RESERVE_BYTES

            self._free(monkeypatch, need_db_fs - 1)
            with pytest.raises(RuntimeError, match="needs"):
                opencode_gc.enable_incremental_vacuum(conn, path, stats)
            assert opencode_gc.read_stats(conn).auto_vacuum == 0, \
                "a refused conversion must not switch the mode"

            self._free(monkeypatch, need_db_fs)
            opencode_gc.enable_incremental_vacuum(conn, path, stats)
            assert opencode_gc.read_stats(conn).auto_vacuum == 2
        finally:
            conn.close()

    def test_a_separate_temp_filesystem_must_also_have_room(self, tmp_path, monkeypatch):
        """SQLite writes its VACUUM copy under SQLITE_TMPDIR. When that is a
        different filesystem, having room beside the database proves nothing.

        Mutation: delete the `if os.stat(tmp_dir).st_dev !=
        os.stat(db.parent).st_dev:` check and its body. The tiny temp
        filesystem below is then never consulted and nothing raises.
        """
        path, conn = self._db(tmp_path, "tmpfs.db")
        tmp_dir = tmp_path / "sqlite-tmp"
        tmp_dir.mkdir()
        monkeypatch.setenv("SQLITE_TMPDIR", str(tmp_dir))
        try:
            stats = opencode_gc.DbStats(
                page_size=4096, page_count=10 ** 7, freelist_count=0, auto_vacuum=0
            )
            _, need_db_fs, need_tmp_fs = opencode_gc.vacuum_space_plan(
                path.resolve(), stats
            )
            # Plenty beside the database, one byte too little on the temp fs.
            monkeypatch.setattr(
                opencode_gc.shutil, "disk_usage",
                lambda p: type("U", (), {
                    "free": need_tmp_fs - 1 if Path(p) == tmp_dir else need_db_fs * 4
                })(),
            )
            self._stat_devices(monkeypatch, tmp_dev=11, db_dev=22, tmp_dir=tmp_dir)

            with pytest.raises(RuntimeError, match="temp"):
                opencode_gc.enable_incremental_vacuum(conn, path, stats)
            assert opencode_gc.read_stats(conn).auto_vacuum == 0
        finally:
            conn.close()

    def test_the_same_filesystem_is_not_charged_twice(self, tmp_path, monkeypatch):
        """When the temp dir shares the database's filesystem, the 2x figure
        already covers the copy. Charging a third copy would refuse a VACUUM
        that fits.

        Mutation: replace the `st_dev` comparison with `if True:`. The temp
        check then runs against the same filesystem and, given exactly
        need_db_fs free, refuses a conversion that should proceed.
        """
        path, conn = self._db(tmp_path, "samefs.db")
        tmp_dir = tmp_path / "sqlite-tmp"
        tmp_dir.mkdir()
        monkeypatch.setenv("SQLITE_TMPDIR", str(tmp_dir))
        try:
            stats = opencode_gc.DbStats(
                page_size=4096, page_count=10 ** 7, freelist_count=0, auto_vacuum=0
            )
            _, need_db_fs, need_tmp_fs = opencode_gc.vacuum_space_plan(
                path.resolve(), stats
            )
            # Exactly the documented two copies: enough to proceed, and -- as
            # the assertion below records -- not enough for a spurious third.
            assert need_db_fs < need_tmp_fs * 2
            self._free(monkeypatch, need_db_fs)
            self._stat_devices(monkeypatch, tmp_dev=33, db_dev=33, tmp_dir=tmp_dir)

            opencode_gc.enable_incremental_vacuum(conn, path, stats)
            assert opencode_gc.read_stats(conn).auto_vacuum == 2
        finally:
            conn.close()


class TestVacuumDeadline:
    """A reclamation pass cut short by --max-seconds has left pages in the
    file. Reporting that as a complete run tells an operator the disk was
    reclaimed when it was not.
    """

    def test_outcome_reports_the_unreleased_remainder(self, db):
        """Mutation: in run_incremental_vacuum, drop the
        `outcome.deadline_reached = True` assignment (the round-2 bare
        `break`). remaining is then reported without the flag that explains it.
        """
        path, conn = db
        for i in range(60):
            _add_session(conn, f"s{i}", age_days=30, events=40, messages=10)
        _delete(conn, [f"s{i}" for i in range(60)])
        free = conn.execute("PRAGMA freelist_count").fetchone()[0]
        assert free > 0

        # The deadline is already past on the first check, so nothing is
        # released and the whole freelist is still eligible.
        vac = opencode_gc.run_incremental_vacuum(
            conn, pages=None, deadline=-1.0, clock=_counting_clock()
        )

        assert vac.deadline_reached is True
        assert vac.released == 0
        assert vac.target == free
        assert vac.remaining == free
        assert conn.execute("PRAGMA freelist_count").fetchone()[0] == free

    def test_cli_exits_three_when_reclamation_is_cut_short(self, tmp_path, monkeypatch, capsys):
        """Mutation: in main, drop the `res.incomplete = True` inside
        `if vac.remaining:`. rc falls back to 0, so automation is told a run
        that left the file oversized finished cleanly.
        """
        path = tmp_path / "vacdeadline.db"
        conn = _make_db(path)
        for i in range(60):
            _add_session(conn, f"s{i}", age_days=30, events=40, messages=10)
        conn.close()

        real_vacuum = opencode_gc.run_incremental_vacuum
        monkeypatch.setattr(
            opencode_gc, "run_incremental_vacuum",
            lambda c, **kw: real_vacuum(
                c, **{**kw, "deadline": -1.0, "clock": _counting_clock()}
            ),
        )
        rc, payload = _run_json_cli(monkeypatch, capsys, "--db", str(path), "--apply")

        assert rc == 3, "an unfinished reclamation must not share the clean-run status"
        assert payload["incomplete"] is True
        assert payload["vacuum_deadline_reached"] is True
        assert payload["pages_reclaimable_remaining"] > 0
        assert payload["errors"] == [], "a deadline is not an error"
        # The deletions themselves did finish; only the reclamation did not.
        assert payload["sessions_deleted"] == 60
        assert len(_snapshot(path)["session"]) == 0


class TestVacuumRemainderIsHonest:
    """`remaining` is what an operator and their automation read to decide
    whether the disk is reclaimed. It must count pages still in the file,
    whatever stopped the pass -- a budget, a deadline or a stall.

    Conflating "what this run was allowed to do" with "what is reclaimable"
    made a capped run report a full freelist as fully reclaimed. Every
    assertion here is checked against `PRAGMA freelist_count` on the real
    database rather than against the message text.
    """

    def _freed(self, conn, sessions=120):
        for i in range(sessions):
            _add_session(conn, f"s{i}", age_days=30, events=40, messages=10)
        _delete(conn, [f"s{i}" for i in range(sessions)])
        free = conn.execute("PRAGMA freelist_count").fetchone()[0]
        assert free > 5, f"the fixture must free more pages than the cap: {free}"
        return free

    def test_a_capped_run_reports_the_pages_it_did_not_release(self, db):
        """The reproduced blocker: `--vacuum-pages` bounds the work, not the
        freelist. `remaining` must equal what is genuinely still on disk.

        Mutation (V-budget-is-target): in run_incremental_vacuum, collapse the
        split back to `target = before if pages is None else min(pages, before)`
        with the loop bounded by `target`. remaining becomes 0 while the
        freelist below is still full, and both assertions fail.
        """
        path, conn = db
        free_before = self._freed(conn)

        vac = opencode_gc.run_incremental_vacuum(conn, pages=5, deadline=None)

        free_after = conn.execute("PRAGMA freelist_count").fetchone()[0]
        assert vac.released == 5
        assert free_before - free_after == 5, "exactly the budget must have moved"
        assert free_after > 0, "the fixture must leave pages behind, or nothing is proven"
        assert vac.target == free_before, \
            "target is what was reclaimable, not what this run was allowed to do"
        assert vac.remaining == free_after, (
            f"remaining must be the pages genuinely still in the file: "
            f"reported {vac.remaining}, freelist holds {free_after}"
        )
        assert vac.deadline_reached is False
        assert vac.stalled is False

    def test_an_uncapped_run_that_finishes_reports_nothing_remaining(self, db):
        """The complement: a genuinely complete pass must not now claim to be
        incomplete, or --vacuum-pages' honesty would cost every clean run its
        exit status.

        Mutation (V-remaining-always): make `remaining` return `self.target`
        unconditionally. This test fails while the capped one still passes.
        """
        path, conn = db
        self._freed(conn)

        vac = opencode_gc.run_incremental_vacuum(conn, pages=None, deadline=None)

        assert conn.execute("PRAGMA freelist_count").fetchone()[0] == 0
        assert vac.remaining == 0
        assert vac.released == vac.target
        assert vac.stalled is False

    def test_the_cli_exits_three_after_a_capped_run(self, tmp_path, monkeypatch, capsys):
        """End to end through argument parsing, which no --vacuum-pages test
        previously reached at all.

        Mutation (V-budget-is-target): remaining is 0, so `if vac.remaining:`
        never fires, rc is 0 and incomplete is false -- while the freelist
        asserted below is still full.
        """
        path = tmp_path / "capped.db"
        conn = _make_db(path)
        for i in range(120):
            _add_session(conn, f"s{i}", age_days=30, events=40, messages=10)
        conn.close()

        rc, payload = _run_json_cli(
            monkeypatch, capsys,
            "--db", str(path), "--apply", "--vacuum-pages", "5",
        )

        probe = sqlite3.connect(str(path))
        try:
            left = probe.execute("PRAGMA freelist_count").fetchone()[0]
        finally:
            probe.close()

        assert left > 0, "the fixture must leave pages on the freelist"
        assert rc == 3, \
            "a capped run left pages in the file and must not share the clean status"
        assert payload["incomplete"] is True
        assert payload["pages_released"] == 5
        assert payload["pages_reclaimable_remaining"] == left
        assert payload["vacuum_deadline_reached"] is False
        assert payload["vacuum_stalled"] is False
        assert payload["errors"] == [], "a page budget is not an error"
        # The deletions themselves completed.
        assert payload["sessions_deleted"] == 120
        assert len(_snapshot(path)["session"]) == 0

    def test_re_running_after_a_cap_reclaims_the_rest(self, tmp_path, monkeypatch, capsys):
        """"Re-run to continue" is the instruction the note gives, so it has
        to be true: a second capped run must release more pages and lower the
        remainder, and an uncapped one must finish and exit 0.

        Mutation (V-budget-is-target): the first run reports remaining=0 and
        rc=0, so the assertion that it was incomplete fails before the
        continuation is even exercised.
        """
        path = tmp_path / "resume.db"
        conn = _make_db(path)
        for i in range(120):
            _add_session(conn, f"s{i}", age_days=30, events=40, messages=10)
        conn.close()

        def run(*extra):
            return _run_json_cli(
                monkeypatch, capsys, "--db", str(path), "--apply", *extra
            )

        rc1, first = run("--vacuum-pages", "5")
        assert rc1 == 3 and first["pages_reclaimable_remaining"] > 0

        rc2, second = run("--vacuum-pages", "5")
        assert rc2 == 3
        assert second["pages_released"] == 5
        assert second["pages_reclaimable_remaining"] == \
            first["pages_reclaimable_remaining"] - 5, \
            "each capped run must genuinely advance the remainder"

        rc3, third = run()
        assert rc3 == 0, "an uncapped run must finish and report a clean status"
        assert third["pages_reclaimable_remaining"] == 0
        probe = sqlite3.connect(str(path))
        try:
            assert probe.execute("PRAGMA freelist_count").fetchone()[0] == 0
        finally:
            probe.close()


class TestVacuumStall:
    """`incremental_vacuum` returning without moving a page is what stops the
    loop from spinning forever. Nothing exercised it, and reaching it used to
    rewrite `target` down to `released` -- so a stalled pass reported
    remaining=0 and claimed it had reclaimed everything available.

    A database with auto_vacuum=NONE reproduces it honestly: the pragma is a
    documented no-op there, so the freelist genuinely cannot shrink.
    """

    def _stalled_db(self, path, sessions=60):
        conn = _make_db(path, auto_vacuum=0)
        for i in range(sessions):
            _add_session(conn, f"s{i}", age_days=30, events=40, messages=10)
        _delete(conn, [f"s{i}" for i in range(sessions)])
        free = conn.execute("PRAGMA freelist_count").fetchone()[0]
        assert free > 0, "the fixture must leave a freelist that cannot shrink"
        return conn, free

    @staticmethod
    def _bounded_clock(limit=10):
        """A clock that aborts the pass after `limit` iterations.

        Every test in this class drives a freelist that cannot shrink, which
        is precisely the input the stall guard exists to stop looping on. If
        the guard is gone they would spin forever and the suite would hang
        rather than fail, so each one carries its own bound: the deadline is
        set far in the future, making this purely an iteration counter.
        """
        reads = []

        def clock():
            reads.append(1)
            if len(reads) > limit:
                raise AssertionError(
                    f"run_incremental_vacuum did not terminate within {limit} passes"
                )
            return 0.0

        return clock, reads

    def test_a_stalled_pass_terminates(self, tmp_path):
        """The branch's actual job. Without it the loop never exits, because
        `released < budget` stays true forever with released stuck at 0.

        Termination is proven from inside the pass with a bounded clock rather
        than by wall-clock timeout: a test that hangs cannot report anything.

        Mutation (V-no-stall-guard): delete the `if progressed <= 0:` block.
        The loop then spins on a freelist that never shrinks; the bounded
        clock raises, so this fails instead of the suite hanging.
        """
        conn, free = self._stalled_db(tmp_path / "stall.db")
        try:
            clock, reads = self._bounded_clock()

            vac = opencode_gc.run_incremental_vacuum(
                conn, pages=None, deadline=1e18, clock=clock
            )

            assert len(reads) <= 2, f"a stall must be detected at once, took {len(reads)}"
            assert vac.stalled is True
            assert vac.released == 0
            assert conn.execute("PRAGMA freelist_count").fetchone()[0] == free
        finally:
            conn.close()

    def test_a_stalled_pass_does_not_claim_the_pages_were_reclaimed(self, tmp_path):
        """The misreport: `target` must stay at the real freelist size, so
        `remaining` still counts the pages sitting in the file.

        Mutation (V-stall-lowers-target): restore `outcome.target = released`
        in the stall branch. target and remaining both collapse to 0 while the
        freelist below is untouched.
        """
        conn, free = self._stalled_db(tmp_path / "stallreport.db")
        try:
            clock, _ = self._bounded_clock()
            vac = opencode_gc.run_incremental_vacuum(
                conn, pages=None, deadline=1e18, clock=clock
            )

            still_free = conn.execute("PRAGMA freelist_count").fetchone()[0]
            assert still_free == free, "the fixture's freelist must be unchanged"
            assert vac.target == free
            assert vac.remaining == still_free, (
                "a stalled pass left every page in the file and must say so: "
                f"reported {vac.remaining}, freelist holds {still_free}"
            )
        finally:
            conn.close()

    def test_the_cli_reports_a_stall_as_incomplete(self, tmp_path, monkeypatch, capsys):
        """A stalled run leaves the file oversized, so it must not exit 0.

        The database is auto_vacuum=NONE, which main() normally routes away
        from reclamation entirely; read_stats is stubbed to report INCREMENTAL
        so the pass actually runs and stalls, which is the only way to reach
        this branch through the CLI.

        Mutation (V-stall-lowers-target): remaining collapses to 0, so
        `if vac.remaining:` never fires and rc is 0 while the freelist
        asserted below is still full.
        """
        path = tmp_path / "stallcli.db"
        conn = _make_db(path, auto_vacuum=0)
        for i in range(60):
            _add_session(conn, f"s{i}", age_days=30, events=40, messages=10)
        conn.close()

        real_read_stats = opencode_gc.read_stats

        def as_incremental(c):
            stats = real_read_stats(c)
            return opencode_gc.DbStats(
                page_size=stats.page_size, page_count=stats.page_count,
                freelist_count=stats.freelist_count, auto_vacuum=2,
            )

        clock, _ = self._bounded_clock()
        real_vacuum = opencode_gc.run_incremental_vacuum
        monkeypatch.setattr(opencode_gc, "read_stats", as_incremental)
        monkeypatch.setattr(
            opencode_gc, "run_incremental_vacuum",
            lambda c, **kw: real_vacuum(
                c, **{**kw, "deadline": 1e18, "clock": clock}
            ),
        )
        rc, payload = _run_json_cli(monkeypatch, capsys, "--db", str(path), "--apply")

        probe = sqlite3.connect(str(path))
        try:
            left = probe.execute("PRAGMA freelist_count").fetchone()[0]
        finally:
            probe.close()

        assert left > 0, "the fixture must stall with pages still on the freelist"
        assert rc == 3, "a stalled run left the file oversized and must say so"
        assert payload["incomplete"] is True
        assert payload["vacuum_stalled"] is True
        assert payload["vacuum_deadline_reached"] is False
        assert payload["pages_released"] == 0
        assert payload["pages_reclaimable_remaining"] == left
        assert payload["errors"] == []
        # The deletions are unaffected: failing to reclaim is not failing to
        # delete.
        assert payload["sessions_deleted"] == 60
        assert len(_snapshot(path)["session"]) == 0

    def test_a_freelist_that_grows_mid_pass_is_not_negative_progress(self, tmp_path):
        """opencode deleting rows during the pass grows the freelist, so
        `before - now` goes negative. That is a stall, not a negative release
        count, and `released` must never go below zero.

        Mutation (V-released-unclamped): `released = max(0, before - now)` ->
        `released = before - now`. released is then negative and remaining
        exceeds the freelist, overstating what is left to reclaim.
        """
        path = tmp_path / "grow.db"
        setup = _make_db(path, wal=True)
        for i in range(200):
            _add_session(setup, f"s{i}", age_days=30, events=40, messages=10)
        _delete(setup, [f"s{i}" for i in range(60)])
        setup.close()

        app = sqlite3.connect(str(path), isolation_level=None, timeout=5)

        class GrowsTheFreelistMidPass(sqlite3.Connection):
            passes = 0

            def execute(self, sql, *a):
                if sql.startswith("PRAGMA incremental_vacuum"):
                    GrowsTheFreelistMidPass.passes += 1
                    if GrowsTheFreelistMidPass.passes == 1:
                        # A concurrent prune frees far more than this pass
                        # can release.
                        app.execute(
                            "DELETE FROM event WHERE aggregate_id IN "
                            "(SELECT id FROM session LIMIT 100)"
                        )
                return super().execute(sql, *a)

        gc_conn = sqlite3.connect(
            str(path), isolation_level=None, timeout=5,
            factory=GrowsTheFreelistMidPass,
        )
        try:
            before = gc_conn.execute("PRAGMA freelist_count").fetchone()[0]
            clock, _ = self._bounded_clock()
            vac = opencode_gc.run_incremental_vacuum(
                gc_conn, pages=None, deadline=1e18, clock=clock
            )
            after = gc_conn.execute("PRAGMA freelist_count").fetchone()[0]

            assert after > before, "the fixture must grow the freelist mid-pass"
            assert vac.released >= 0, \
                f"a growing freelist is not a negative release: {vac.released}"
            assert vac.remaining <= after, (
                "remaining may not exceed the pages actually on the freelist: "
                f"reported {vac.remaining}, freelist holds {after}"
            )
            assert vac.stalled is True
        finally:
            gc_conn.close()
            app.close()


class TestChildTableOrder:
    """CHILD_TABLES deletes children before parents so every intermediate
    state of the transaction is referentially valid.

    The live schema's `ON DELETE CASCADE` actions mask this: under them either
    order commits, which is why the property was previously asserted nowhere.
    A plain `REFERENCES` with no action -- one migration away, and what
    event_sequence would become if the cascade were ever dropped -- makes the
    wrong order fail immediately under `PRAGMA foreign_keys=ON`.
    """

    NO_CASCADE_SCHEMA = """
        CREATE TABLE session (
            id text PRIMARY KEY,
            parent_id text,
            time_created integer NOT NULL,
            time_updated integer NOT NULL
        );
        CREATE TABLE message (
            id text PRIMARY KEY,
            session_id text NOT NULL REFERENCES session(id),
            data text NOT NULL
        );
        CREATE TABLE part (
            id text PRIMARY KEY,
            message_id text NOT NULL REFERENCES message(id),
            session_id text NOT NULL,
            data text NOT NULL
        );
        CREATE TABLE event_sequence (
            aggregate_id text PRIMARY KEY,
            seq integer NOT NULL
        );
        CREATE TABLE event (
            id text PRIMARY KEY,
            aggregate_id text NOT NULL REFERENCES event_sequence(aggregate_id),
            data text NOT NULL
        );
    """

    def _db(self, path):
        conn = sqlite3.connect(str(path), isolation_level=None)
        conn.executescript(self.NO_CASCADE_SCHEMA)
        _add_session(conn, "old", age_days=30, events=3, messages=2)
        conn.execute("PRAGMA foreign_keys=ON")
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        return conn

    def test_the_documented_order_deletes_cleanly_without_cascades(self, tmp_path):
        """Mutation (O-child-tables-reversed): `CHILD_TABLES` ->
        `list(reversed(CHILD_TABLES))`. event_sequence is then deleted before
        the event rows referencing it, and the batch fails with a FOREIGN KEY
        error instead of committing.
        """
        conn = self._db(tmp_path / "nocascade.db")
        try:
            outcome = _delete(conn, ["old"], batch=1)

            assert outcome.failure is None, (
                "children must be deleted before their parents: "
                f"{outcome.failure}"
            )
            assert outcome.rows["session"] == 1
            for table in ("session", "message", "part", "event", "event_sequence"):
                assert conn.execute(
                    f"SELECT count(*) FROM {table}"
                ).fetchone()[0] == 0, f"{table} must be empty"
        finally:
            conn.close()

    def test_the_reverse_order_really_would_fail(self, tmp_path):
        """The fixture's own proof: without this, the test above would pass on
        a schema where order is irrelevant and assert nothing.

        No mutation -- this test exists to make the previous one meaningful,
        and asserts on the schema fixture rather than on production code.
        """
        conn = self._db(tmp_path / "reverse.db")
        try:
            with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
                conn.execute("BEGIN IMMEDIATE")
                for table, column in reversed(opencode_gc.CHILD_TABLES):
                    conn.execute(f"DELETE FROM {table} WHERE {column}='old'")
                conn.execute("DELETE FROM session WHERE id='old'")
                conn.execute("COMMIT")
        finally:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            conn.close()


class TestUriEscaping:
    """`file:{db}?mode=rw` reparses a legitimate filename as URI syntax, so
    `--db 'victim?.db'` passes is_file() and then deletes from `victim`. Each
    candidate below carries a distinct marker session, so the test proves
    *which* database was opened rather than merely that one opened.

    Mutation for all of these: in connect(), replace the body with
    `conn = sqlite3.connect(f"file:{Path(db).resolve()}?mode=" + ("ro" if
    read_only else "rw"), uri=True, timeout=timeout_s, isolation_level=None)`.
    The '?' and '#' cases then read the decoy's marker, and '%41' reads the
    percent-decoded neighbour's.
    """

    @pytest.fixture()
    def decoys(self, tmp_path):
        """A directory where a naive URI parse lands on a different file."""
        made = {}
        for name, marker in (
            ("victim", "DECOY_TRUNCATED"),      # what 'victim?.db' truncates to
            ("victimA.db", "DECOY_PERCENT"),    # what 'victim%41.db' decodes to
            ("victim?.db", "TARGET_Q"),
            ("victim#1.db", "TARGET_HASH"),
            ("victim%41.db", "TARGET_PERCENT"),
            ("vic tim.db", "TARGET_SPACE"),
        ):
            path = tmp_path / name
            conn = _make_db(path)
            _add_session(conn, marker, age_days=30)
            conn.close()
            made[name] = (path, marker)
        return made

    @pytest.mark.parametrize(
        "name", ["victim?.db", "victim#1.db", "victim%41.db", "vic tim.db"]
    )
    @pytest.mark.parametrize("read_only", [True, False])
    def test_connect_opens_the_named_file(self, decoys, name, read_only):
        path, marker = decoys[name]
        conn = opencode_gc.connect(path, read_only=read_only)
        try:
            assert conn.execute("SELECT id FROM session").fetchall() == [(marker,)]
        finally:
            conn.close()

    def test_apply_deletes_from_the_named_file_only(self, decoys, monkeypatch):
        """The full blast radius: --apply on 'victim?.db' must not touch
        'victim'."""
        path, marker = decoys["victim?.db"]
        decoy_path, decoy_marker = decoys["victim"]

        monkeypatch.setattr(
            opencode_gc.sys, "argv", ["opencode-gc", "--db", str(path), "--apply"]
        )
        assert opencode_gc.main() == 0

        assert _snapshot(path)["session"] == [], "the named database must be pruned"
        assert [r[0] for r in _snapshot(decoy_path)["session"]] == [decoy_marker], \
            "the decoy database must be untouched"


class TestInterruptedRunReporting:
    """Committed batches are irreversible. A failure or deadline afterwards
    must still yield counts and an incomplete marker, never a traceback.

    The failure here is a genuine SQLITE_BUSY: a second connection takes the
    write lock before one of the batches, which is exactly what opencode does
    to this tool in production. No exception is fabricated.
    """

    def _locking_conn(self, path, blocker, *, before_batch):
        """A connection whose `before_batch`-th BEGIN IMMEDIATE loses the race
        for the write lock."""

        class LosesTheLock(sqlite3.Connection):
            begins = 0

            def execute(self, sql, *a):
                if sql == "BEGIN IMMEDIATE":
                    LosesTheLock.begins += 1
                    if LosesTheLock.begins == before_batch:
                        blocker.execute("BEGIN IMMEDIATE")
                        blocker.execute(
                            "INSERT INTO session (id, parent_id, time_created, "
                            "time_updated) VALUES ('blocker',NULL,1,1)"
                        )
                return super().execute(sql, *a)

        conn = sqlite3.connect(
            str(path), isolation_level=None, timeout=0, factory=LosesTheLock
        )
        conn.execute("PRAGMA busy_timeout=0")
        return conn

    def test_lock_failure_after_a_committed_batch_reports_what_was_destroyed(self, tmp_path):
        """Mutation: in delete_sessions, move `conn.execute("BEGIN IMMEDIATE")`
        back outside the `try`, or re-`raise` from the
        `except (sqlite3.Error, ...)` handler. The SQLITE_BUSY then escapes as
        a traceback and the committed counts are lost.
        """
        path = tmp_path / "busy.db"
        setup = _make_db(path, wal=True)
        for i in range(6):
            _add_session(setup, f"s{i}", age_days=30)
        setup.close()

        blocker = sqlite3.connect(str(path), isolation_level=None, timeout=0)
        gc_conn = self._locking_conn(path, blocker, before_batch=3)
        try:
            cutoff = _cutoff()
            sel = opencode_gc.expired_session_ids(gc_conn, cutoff)
            assert len(sel.deletable) == 6

            outcome = opencode_gc.delete_sessions(
                gc_conn, sel.deletable, cutoff_ms=cutoff, batch=1, deadline=None
            )

            assert outcome.failure is not None
            assert "locked" in outcome.failure.lower()
            assert outcome.incomplete is True
            # Two batches committed before the lock was lost; the reported
            # count must match what is genuinely gone.
            assert outcome.rows["session"] == 2
            assert outcome.remaining == 4
            blocker.execute("ROLLBACK")
            surviving = {r[0] for r in gc_conn.execute("SELECT id FROM session")}
            assert len(surviving) == 4
            assert len(sel.deletable) - outcome.rows["session"] == len(surviving)
        finally:
            gc_conn.close()
            blocker.close()

    def test_cli_reports_committed_counts_and_exits_nonzero(self, tmp_path, monkeypatch, capsys):
        """Mutation: same as above. main() dies with a traceback and emits no
        JSON at all, so the json.loads below raises.
        """
        path = tmp_path / "busycli.db"
        setup = _make_db(path, wal=True)
        for i in range(6):
            _add_session(setup, f"s{i}", age_days=30)
        setup.close()

        blocker = sqlite3.connect(str(path), isolation_level=None, timeout=0)
        gc_conn = self._locking_conn(path, blocker, before_batch=2)
        monkeypatch.setattr(opencode_gc, "connect", lambda *a, **kw: gc_conn)
        try:
            rc, payload = _run_json_cli(
                monkeypatch, capsys, "--db", str(path), "--apply", "--batch", "1"
            )

            assert rc == 1
            assert payload["incomplete"] is True
            assert payload["sessions_deleted"] == 1
            assert any("locked" in e.lower() for e in payload["errors"])
            assert payload["sessions_remaining"] == 5
        finally:
            blocker.execute("ROLLBACK")
            blocker.close()
        # The reported count must match the database, not the intent.
        assert len(_snapshot(path)["session"]) == 5

    def test_deadline_is_not_reported_as_clean_success(self, tmp_path, monkeypatch, capsys):
        """A run that stops on --max-seconds leaves eligible sessions behind.

        Mutation: in main, delete the two lines assigning
        `res.incomplete = outcome.incomplete` and
        `res.deadline_reached = outcome.deadline_reached`. rc returns to 0 and
        the flags go false, so a caller cannot tell the run was truncated.
        """
        path = tmp_path / "deadline.db"
        conn = _make_db(path)
        for i in range(5):
            _add_session(conn, f"s{i}", age_days=30)
        conn.close()

        real_delete = opencode_gc.delete_sessions
        monkeypatch.setattr(
            opencode_gc, "delete_sessions",
            lambda c, ids, **kw: real_delete(
                c, ids, **{**kw, "deadline": 1.5, "clock": _counting_clock()}
            ),
        )
        rc, payload = _run_json_cli(
            monkeypatch, capsys, "--db", str(path), "--apply", "--batch", "1"
        )

        assert rc == 3, "a truncated run must not share the complete-run status"
        assert payload["incomplete"] is True
        assert payload["deadline_reached"] is True
        assert payload["sessions_deleted"] == 2
        assert payload["sessions_remaining"] == 3
        assert len(_snapshot(path)["session"]) == 3


class TestInterruptDuringDeletion:
    """Ctrl-C is how an operator stops a multi-hour prune, and by then batches
    have committed irreversibly. Every point in the loop must route the
    interrupt into the outcome rather than let it escape: a traceback exits
    130, a status _status() never produces, and carries none of the counts.
    """

    def _interrupt_clock(self, after):
        """A clock that raises KeyboardInterrupt on its `after`-th read.

        The deadline check is the loop's one clock() call per batch, so this
        lands the interrupt exactly there.
        """
        reads = []

        def clock():
            reads.append(1)
            if len(reads) == after:
                raise KeyboardInterrupt()
            return 0.0

        return clock

    def test_an_interrupt_at_the_deadline_check_keeps_the_committed_counts(self, db):
        """The reproduced blocker: batch 1 commits, then Ctrl-C lands on the
        deadline check before batch 2.

        Mutation (D-ctrlc-deadline-outside): in delete_sessions, move the
        `if deadline is not None and clock() > deadline:` check back above
        `try:`. The KeyboardInterrupt then escapes delete_sessions entirely --
        no DeleteOutcome is returned, so the sessions destroyed by batch 1 are
        reported nowhere.
        """
        path, conn = db
        for sid in ("aaa", "bbb", "ccc", "ddd"):
            _add_session(conn, sid, age_days=30, events=3, messages=2)

        outcome = opencode_gc.delete_sessions(
            conn, ["aaa", "bbb", "ccc", "ddd"], cutoff_ms=_cutoff(), batch=1,
            deadline=1.0, clock=self._interrupt_clock(2),
        )

        assert outcome.failure is not None
        assert "KeyboardInterrupt" in outcome.failure
        assert outcome.incomplete is True
        # Exactly one batch committed, and the count says so.
        assert outcome.rows["session"] == 1
        assert outcome.rows["event"] == 3
        assert outcome.rows["message"] == 2
        surviving = sorted(r[0] for r in conn.execute("SELECT id FROM session"))
        assert surviving == ["bbb", "ccc", "ddd"], \
            "the interrupt must not have destroyed a second batch"
        assert outcome.remaining == 3
        assert conn.in_transaction is False

    def test_an_interrupt_before_the_first_batch_reports_zero_not_a_traceback(self, db):
        """Nothing has committed yet, so the counts are zero -- but the
        outcome must still exist, and the database must be untouched.

        Mutation (D-ctrlc-deadline-outside): same move. The interrupt escapes
        and there is no outcome to assert on at all.
        """
        path, conn = db
        _add_session(conn, "old", age_days=30)
        before = _counts(conn)

        outcome = opencode_gc.delete_sessions(
            conn, ["old"], cutoff_ms=_cutoff(), batch=1,
            deadline=1.0, clock=self._interrupt_clock(1),
        )

        assert "KeyboardInterrupt" in outcome.failure
        assert sum(outcome.rows.values()) == 0
        assert outcome.remaining == 1
        assert _counts(conn) == before

    def test_an_interrupt_just_after_commit_still_reports_that_batch(self, tmp_path):
        """The narrowest window: COMMIT has returned and the rows are gone,
        but the counts have not been merged yet. An interrupt there must not
        lose the batch it describes.

        The interrupt is raised from the connection itself, on the statement
        immediately after the COMMIT of the first batch -- the BEGIN IMMEDIATE
        of the second -- because that is the first thing the loop executes
        once the merge has happened.

        Mutation (D-ctrlc-merge-outside): in delete_sessions, move the
        `outcome.rows.update(...)` merge and the two `pending` updates below
        the `except` clauses, back outside the `try`. The interrupt then skips
        the merge and reports 0 sessions deleted while one is genuinely gone.
        """
        path = tmp_path / "postcommit.db"
        setup = _make_db(path, wal=True)
        for sid in ("aaa", "bbb"):
            _add_session(setup, sid, age_days=30, events=3, messages=2)
        setup.close()

        class InterruptsAfterFirstCommit(sqlite3.Connection):
            commits = 0

            def execute(self, sql, *a):
                result = super().execute(sql, *a)
                if sql == "COMMIT":
                    InterruptsAfterFirstCommit.commits += 1
                    if InterruptsAfterFirstCommit.commits == 1:
                        raise KeyboardInterrupt()
                return result

        conn = sqlite3.connect(
            str(path), isolation_level=None, timeout=5,
            factory=InterruptsAfterFirstCommit,
        )
        try:
            outcome = opencode_gc.delete_sessions(
                conn, ["aaa", "bbb"], cutoff_ms=_cutoff(), batch=1, deadline=None
            )

            assert "KeyboardInterrupt" in outcome.failure
            assert outcome.incomplete is True
            gone = 2 - conn.execute("SELECT count(*) FROM session").fetchone()[0]
            assert gone == 1, "exactly one batch must have committed"
            assert outcome.rows["session"] == gone, (
                "the COMMIT returned, so its rows are durable and must be "
                f"reported: {outcome.rows}"
            )
            assert outcome.rows["event"] == 3
        finally:
            conn.close()

    def test_an_interrupt_on_the_merge_itself_does_not_escape(self, tmp_path, monkeypatch):
        """The last unguarded window the reviewer named: COMMIT has returned
        and the bookkeeping that records it has not finished. An interrupt
        delivered between those bytecodes must still come back as an outcome.

        The interrupt is raised from `outcome.rows.update` itself -- the exact
        statement -- because that is the only way to land in a window a few
        bytecodes wide deterministically. The batch it describes is lost
        either way (the counts never got written), but the earlier batches'
        are not, and neither is the report.

        Mutation (D-ctrlc-merge-outside): in delete_sessions, move the merge,
        the `skipped` extend, the `pending` update and the two `break`s below
        the `except` clauses, back outside the `try`. The KeyboardInterrupt
        then escapes delete_sessions and this raises instead of asserting.
        """
        path = tmp_path / "mergeint.db"
        setup = _make_db(path, wal=True)
        for sid in ("aaa", "bbb", "ccc"):
            _add_session(setup, sid, age_days=30, events=3, messages=2)
        setup.close()

        class InterruptsOnSecondMerge(dict):
            merges = 0

            def update(self, *a, **kw):
                InterruptsOnSecondMerge.merges += 1
                if InterruptsOnSecondMerge.merges == 2:
                    raise KeyboardInterrupt()
                return super().update(*a, **kw)

        real_outcome = opencode_gc.DeleteOutcome

        def wrapping_outcome(*, rows):
            return real_outcome(rows=InterruptsOnSecondMerge(rows))

        monkeypatch.setattr(opencode_gc, "DeleteOutcome", wrapping_outcome)

        conn = sqlite3.connect(str(path), isolation_level=None, timeout=5)
        try:
            outcome = opencode_gc.delete_sessions(
                conn, ["aaa", "bbb", "ccc"], cutoff_ms=_cutoff(), batch=1, deadline=None
            )

            assert "KeyboardInterrupt" in outcome.failure, \
                "an interrupt on the merge must be reported, not raised"
            assert outcome.incomplete is True
            # Batch 1's merge ran; batch 2 committed but was interrupted
            # before its counts landed.
            assert outcome.rows["session"] == 1
            assert conn.execute("SELECT count(*) FROM session").fetchone()[0] == 1
            assert conn.in_transaction is False
        finally:
            conn.close()

    def test_the_cli_reports_an_interrupt_instead_of_exiting_130(
        self, tmp_path, monkeypatch, capsys
    ):
        """End to end: Ctrl-C during --apply must produce the JSON report and
        a classified status, not a traceback.

        Mutation (D-ctrlc-deadline-outside): the interrupt escapes
        delete_sessions and then main(), so no JSON is printed at all and the
        json.loads below raises.
        """
        path = tmp_path / "ctrlc.db"
        conn = _make_db(path)
        for i in range(4):
            _add_session(conn, f"s{i}", age_days=30)
        conn.close()

        reads = []

        def interrupting_clock():
            reads.append(1)
            if len(reads) == 3:
                raise KeyboardInterrupt()
            return 0.0

        real_delete = opencode_gc.delete_sessions
        monkeypatch.setattr(
            opencode_gc, "delete_sessions",
            lambda c, ids, **kw: real_delete(
                c, ids, **{**kw, "deadline": 1.0, "clock": interrupting_clock}
            ),
        )
        rc, payload = _run_json_cli(
            monkeypatch, capsys, "--db", str(path), "--apply", "--batch", "1"
        )

        assert rc == 1, "an interrupted run must map to a documented status"
        assert payload["incomplete"] is True
        assert any("KeyboardInterrupt" in e for e in payload["errors"])
        assert payload["sessions_deleted"] == 2
        assert payload["sessions_remaining"] == 2
        # The report must match the database, not the intent.
        assert len(_snapshot(path)["session"]) == 2

    def test_an_interrupt_closing_the_database_still_reports(
        self, tmp_path, monkeypatch, capsys
    ):
        """In WAL mode close() checkpoints, which on a 76 GB file is minutes
        of IO an operator can interrupt -- with the deletions already
        committed.

        Mutation (M-close-sqlite-only): in main's `finally`, narrow the
        handler back to `except sqlite3.Error`. The KeyboardInterrupt escapes
        the finally, no report is printed, and json.loads gets nothing.
        """
        path = tmp_path / "closeint.db"
        setup = _make_db(path, wal=True)
        for i in range(3):
            _add_session(setup, f"s{i}", age_days=30)
        setup.close()

        class InterruptsOnClose(sqlite3.Connection):
            def close(self):
                super().close()
                raise KeyboardInterrupt()

        gc_conn = sqlite3.connect(
            str(path), isolation_level=None, timeout=5, factory=InterruptsOnClose
        )
        monkeypatch.setattr(opencode_gc, "connect", lambda *a, **kw: gc_conn)
        rc, payload = _run_json_cli(monkeypatch, capsys, "--db", str(path), "--apply")

        assert rc == 1
        assert payload["incomplete"] is True
        assert any("KeyboardInterrupt" in e and "closing" in e
                   for e in payload["errors"])
        assert payload["sessions_deleted"] == 3
        assert len(_snapshot(path)["session"]) == 0

    def test_an_interrupt_during_reclamation_still_reports_the_deletions(
        self, tmp_path, monkeypatch, capsys
    ):
        """incremental_vacuum runs for as long as the freelist is large, so an
        interrupt lands there too -- with every deletion already committed.
        Failing to reclaim is not failing to delete, and the counts must
        survive it.

        This pins the reclamation handler main() already has, alongside the
        close() one widened above: both sit after the rows are gone, and
        catching the interrupt in one but not the other leaves the same hole.

        Mutation (M-reclaim-sqlite-only): in main, narrow the reclamation
        handler from `except (sqlite3.Error, KeyboardInterrupt, OSError)` to
        `except sqlite3.Error`. The interrupt escapes main(), no JSON is
        printed, and json.loads below raises.
        """
        path = tmp_path / "vacint.db"
        conn = _make_db(path)
        for i in range(20):
            _add_session(conn, f"s{i}", age_days=30, events=10, messages=4)
        conn.close()

        def interrupting_vacuum(c, **kw):
            raise KeyboardInterrupt()

        monkeypatch.setattr(
            opencode_gc, "run_incremental_vacuum", interrupting_vacuum
        )
        rc, payload = _run_json_cli(monkeypatch, capsys, "--db", str(path), "--apply")

        assert rc == 1
        assert payload["incomplete"] is True
        assert any("KeyboardInterrupt" in e and "reclamation" in e
                   for e in payload["errors"])
        # The deletions themselves finished and are still fully reported.
        assert payload["sessions_deleted"] == 20
        assert payload["rows_deleted"]["event"] == 200
        assert len(_snapshot(path)["session"]) == 0

    def test_an_interrupt_measuring_the_file_still_reports_the_deletions(
        self, tmp_path, monkeypatch, capsys
    ):
        """on_disk_bytes() stats files after close, once the rows are gone.

        Mutation (M-measure-oserror-only): in main, narrow the handler around
        `res.bytes_after = on_disk_bytes(db_path)` back to `except OSError`.
        The interrupt escapes and no report is printed.
        """
        path = tmp_path / "sizeint.db"
        conn = _make_db(path)
        for i in range(3):
            _add_session(conn, f"s{i}", age_days=30)
        conn.close()

        real_on_disk = opencode_gc.on_disk_bytes
        calls = []

        def interrupting_on_disk(p):
            calls.append(1)
            if len(calls) >= 2:
                raise KeyboardInterrupt()
            return real_on_disk(p)

        monkeypatch.setattr(opencode_gc, "on_disk_bytes", interrupting_on_disk)
        rc, payload = _run_json_cli(monkeypatch, capsys, "--db", str(path), "--apply")

        assert rc == 1
        assert payload["incomplete"] is True
        assert any("KeyboardInterrupt" in e and "measuring" in e
                   for e in payload["errors"])
        assert payload["sessions_deleted"] == 3
        assert len(_snapshot(path)["session"]) == 0


class TestNonNumericTimeUpdated:
    """`time_updated integer` is an affinity, not a constraint: SQLite stores
    a value it cannot convert verbatim, and Python reads it back as `str` or
    `bytes`. Comparing that against the cutoff raises TypeError from under the
    write lock, after earlier batches have committed irreversibly.

    Conservative, like the NULL policy: an age that cannot be read is an
    unknown age, and unknown ages are never deleted.

    Mutation for every test here (S-nonnumeric-compared): in select_expired,
    revert to `unknown = {sid for sid, t in updated.items() if t is None}` /
    `old = {... if t is not None and t < cutoff_ms}`.
    """

    @pytest.fixture()
    def textdb(self, tmp_path):
        path = tmp_path / "text.db"
        conn = _make_db(path)
        yield path, conn
        conn.close()

    def _add_text_age(self, conn, sid, value, *, parent=None):
        conn.execute(
            "INSERT INTO session (id, parent_id, time_created, time_updated) "
            "VALUES (?,?,?,?)", (sid, parent, 0, value),
        )
        _add_rows(conn, sid)
        stored = conn.execute(
            "SELECT typeof(time_updated) FROM session WHERE id=?", (sid,)
        ).fetchone()[0]
        assert stored == "text", (
            f"the fixture must actually store non-numeric text, got {stored!r}"
        )

    def test_a_text_age_is_kept_as_unknown_rather_than_raising(self, textdb):
        path, conn = textdb
        self._add_text_age(conn, "garbled", "not-a-timestamp")
        _add_session(conn, "old", age_days=30)

        sel = opencode_gc.expired_session_ids(conn, _cutoff())

        assert sel.deletable == ["old"], "the unreadable age must not be deletable"
        assert sel.kept_unknown_age == 1

    def test_a_text_age_survives_a_real_delete(self, textdb):
        """The selection is one thing; the revalidation under the write lock
        runs the same comparison on every session in the graph, including ones
        this run never selected.
        """
        path, conn = textdb
        _add_session(conn, "old", age_days=30, events=3, messages=2)
        self._add_text_age(conn, "garbled", "not-a-timestamp")

        outcome = _delete(conn, ["old"], batch=1)

        assert outcome.failure is None, (
            f"an unreadable age must not fail the run: {outcome.failure}"
        )
        assert outcome.rows["session"] == 1
        assert [r[0] for r in conn.execute("SELECT id FROM session")] == ["garbled"]
        assert conn.execute(
            "SELECT count(*) FROM event WHERE aggregate_id='garbled'"
        ).fetchone()[0] == 3

    def test_a_text_age_appearing_mid_run_does_not_lose_the_counts(self, tmp_path):
        """The row need not exist when verify_usable() probes, so the write
        lock is where it is met. One batch has committed by then, and the
        TypeError is not in delete_sessions' enumerated except list: it would
        take the `except BaseException` arm, roll back, and re-raise, losing
        every count.
        """
        path = tmp_path / "midrun-text.db"
        setup = _make_db(path, wal=True)
        for sid in ("aaa", "bbb"):
            _add_session(setup, sid, age_days=30, events=3, messages=2)
        setup.close()

        gc_conn = opencode_gc.connect(path, read_only=False, timeout_s=5)
        app = sqlite3.connect(str(path), isolation_level=None, timeout=5)
        try:
            cutoff = _cutoff()
            selected = opencode_gc.expired_session_ids(gc_conn, cutoff).deletable
            assert sorted(selected) == ["aaa", "bbb"]

            # opencode -- or a corruption, or a schema migration -- writes a
            # non-numeric time_updated after selection.
            app.execute(
                "INSERT INTO session (id, parent_id, time_created, time_updated) "
                "VALUES ('garbled',NULL,0,'not-a-timestamp')"
            )

            outcome = opencode_gc.delete_sessions(
                gc_conn, selected, cutoff_ms=cutoff, batch=1, deadline=None
            )

            assert outcome.failure is None, (
                f"the unreadable age must not abort the run: {outcome.failure}"
            )
            assert outcome.rows["session"] == 2
            assert [r[0] for r in app.execute("SELECT id FROM session")] == ["garbled"]
        finally:
            gc_conn.close()
            app.close()

    def test_a_text_aged_child_protects_its_old_parent(self, textdb):
        """Unknown age joins the same policy as NULL, so it shields ancestors
        exactly as a live session does.
        """
        path, conn = textdb
        _add_session(conn, "parent", age_days=30)
        self._add_text_age(conn, "child", "not-a-timestamp", parent="parent")

        sel = opencode_gc.expired_session_ids(conn, _cutoff())

        assert sel.deletable == []
        assert sel.kept_live_descendant == 1

    def test_a_float_age_is_still_a_readable_age(self, textdb):
        """REAL is numeric and comparable, so it is an age like any other --
        the unknown-age policy must not swallow it. INTEGER affinity only
        rewrites a float it can convert losslessly, so a fractional
        millisecond survives as `real`.

        Mutation (S-float-unknown): in select_expired, narrow the isinstance
        test to `isinstance(t, int)`. The old float-aged session is then
        counted as unknown age and never collected.
        """
        path, conn = textdb
        cutoff = _cutoff()
        conn.execute(
            "INSERT INTO session (id, parent_id, time_created, time_updated) "
            "VALUES ('floaty',NULL,0,?)", (cutoff - 1000.5,),
        )
        _add_rows(conn, "floaty")
        assert conn.execute(
            "SELECT typeof(time_updated) FROM session WHERE id='floaty'"
        ).fetchone()[0] == "real", "the fixture must actually store a REAL"

        sel = opencode_gc.expired_session_ids(conn, cutoff)

        assert sel.deletable == ["floaty"], "a REAL age is readable and expired"
        assert sel.kept_unknown_age == 0


class TestDryRunIsReadOnlyAtTheHandle:
    """The dry-run snapshot being unchanged proves the dry-run branch happens
    not to write; it does not prove the handle could not. `mode=ro` is the
    last backstop against a dry run mutating an irreplaceable store, so it is
    asserted directly.
    """

    @pytest.fixture()
    def populated(self, tmp_path):
        path = tmp_path / "ro.db"
        conn = _make_db(path)
        _add_session(conn, "old", age_days=30)
        _add_session(conn, "live", age_days=1)
        conn.close()
        return path

    def test_a_dry_run_opens_the_database_read_only(self, populated, monkeypatch):
        """Mutation (C-dry-run-read-write): in main,
        `connect(db_path, read_only=not args.apply)` -> `read_only=False`.
        The recorded flag is then False and this fails, where
        test_dry_run_changes_nothing passes on a read-write handle.
        """
        seen = []
        real_connect = opencode_gc.connect

        def spy(db, *, read_only, **kw):
            seen.append(read_only)
            return real_connect(db, read_only=read_only, **kw)

        monkeypatch.setattr(opencode_gc, "connect", spy)
        monkeypatch.setattr(
            opencode_gc.sys, "argv", ["opencode-gc", "--db", str(populated)]
        )

        assert opencode_gc.main() == 0
        assert seen == [True], \
            f"a dry run must open a read-only handle, got read_only={seen}"

    def test_apply_opens_the_database_read_write(self, populated, monkeypatch):
        """The complement: hard-coding read_only=True would make --apply a
        no-op that still exits 0.

        Mutation (C-apply-read-only): same line -> `read_only=True`. The
        deletes then raise 'attempt to write a readonly database' and the
        session survives.
        """
        seen = []
        real_connect = opencode_gc.connect

        def spy(db, *, read_only, **kw):
            seen.append(read_only)
            return real_connect(db, read_only=read_only, **kw)

        monkeypatch.setattr(opencode_gc, "connect", spy)
        monkeypatch.setattr(
            opencode_gc.sys, "argv", ["opencode-gc", "--db", str(populated), "--apply"]
        )

        assert opencode_gc.main() == 0
        assert seen == [False]
        assert [r[0] for r in _snapshot(populated)["session"]] == ["live"]

    def test_a_read_only_handle_refuses_every_write_this_tool_makes(self, populated):
        """What read_only=True actually buys: the statements the apply path
        would run are refused by SQLite itself, whatever the calling code does.

        Mutation (C-connect-ro-dropped): in connect(), drop the `?mode=ro`
        from the read-only URI. Every statement below then succeeds and the
        rows are gone.
        """
        before = _snapshot(populated)
        conn = opencode_gc.connect(populated, read_only=True)
        try:
            # A read must still work -- a dry run has to count rows.
            assert conn.execute("SELECT count(*) FROM session").fetchone()[0] == 2

            for sql in (
                "DELETE FROM session WHERE id='old'",
                "DELETE FROM event WHERE aggregate_id='old'",
                "UPDATE session SET time_updated=0 WHERE id='live'",
                "PRAGMA incremental_vacuum(1)",
            ):
                with pytest.raises(sqlite3.OperationalError, match="readonly"):
                    conn.execute(sql)
        finally:
            conn.close()
        assert _snapshot(populated) == before


class TestDeadlineBoundary:
    """`--max-seconds` is a bound on when new batches may start. Which side of
    it the boundary instant falls on is a one-line decision that no other test
    exercises.
    """

    def _clock_reading(self, *values):
        """A clock returning exactly these values, one per read."""
        it = iter(values)
        return lambda: next(it)

    def test_a_batch_starting_exactly_at_the_deadline_still_runs(self, db):
        """`clock() > deadline`, not `>=`: the deadline is the last instant a
        batch may start, and a run that stops at it reports itself incomplete
        and leaves rows behind for no reason.

        Mutation (D5b-deadline-ge): `clock() > deadline` -> `>=`. The batch
        below is then never attempted: deadline_reached is True, nothing is
        deleted, and the CLI would exit 3 on a run that had work it could
        have done.
        """
        path, conn = db
        _add_session(conn, "old", age_days=30)

        outcome = opencode_gc.delete_sessions(
            conn, ["old"], cutoff_ms=_cutoff(), batch=1,
            deadline=10.0, clock=self._clock_reading(10.0),
        )

        assert outcome.deadline_reached is False, \
            "a batch starting exactly at the deadline is within the bound"
        assert outcome.rows["session"] == 1
        assert outcome.remaining == 0
        assert conn.execute("SELECT count(*) FROM session").fetchone()[0] == 0

    def test_a_batch_one_tick_past_the_deadline_does_not_run(self, db):
        """The other side of the same boundary: past the bound, no new batch.

        Mutation (D5b-deadline-never): `clock() > deadline` -> `False`. The
        run then ignores --max-seconds entirely and deletes.
        """
        path, conn = db
        _add_session(conn, "old", age_days=30)
        before = _counts(conn)

        outcome = opencode_gc.delete_sessions(
            conn, ["old"], cutoff_ms=_cutoff(), batch=1,
            deadline=10.0, clock=self._clock_reading(10.001),
        )

        assert outcome.deadline_reached is True
        assert outcome.rows["session"] == 0
        assert outcome.remaining == 1
        assert _counts(conn) == before


class TestTheLoopAlwaysTerminates:
    """`delete_sessions` loops until `pending` is empty, and every iteration
    must shrink it. A batch that deletes nothing and drops nothing spins
    forever, taking and releasing the write lock at full speed against a live
    database -- a hang, and a hang is the one outcome a test suite cannot
    report: it never gets to fail.

    So the loop refuses to iterate without progress, and these tests assert
    both halves: the normal path drops ineligible candidates and finishes, and
    a batch that somehow makes no progress stops with a failure instead of
    looping. Termination is proven from inside the run, with a bounded clock,
    rather than left to a wall-clock timeout.
    """

    def _bounded(self, conn, session_ids, *, cutoff, batch, limit):
        """Run delete_sessions, failing if the loop exceeds `limit` batches.

        The counter is driven by the injected `clock`, which the loop reads
        exactly once per iteration. Raising from it goes through the guarded
        region like any other interrupt, so the outcome still comes back and
        the bound is enforced rather than merely observed.
        """
        iterations = []

        def counting_clock():
            iterations.append(1)
            if len(iterations) > limit:
                raise MemoryError(f"loop exceeded {limit} batches")
            return 0.0

        outcome = opencode_gc.delete_sessions(
            conn, session_ids, cutoff_ms=cutoff, batch=batch,
            # A deadline far in the future: the check never trips, so the
            # clock is read purely as an iteration counter.
            deadline=1e18, clock=counting_clock,
        )
        return outcome, len(iterations)

    def test_ineligible_candidates_are_dropped_so_the_run_ends(self, db):
        """Every candidate is ineligible under the lock, so no batch can
        delete anything. The loop must still end, by dropping them from
        `pending`.

        Mutation (D2-ineligible-empty): `ineligible = pending - set(eligible)`
        -> `ineligible = set()`. Nothing is deleted and nothing is dropped, so
        the batch makes no progress; the loop stops on that rather than
        spinning, and `failure` below is set instead of None. Before the
        no-progress guard existed this mutant hung the whole suite -- it never
        reached a failing assertion at all.
        """
        path, conn = db
        # Live sessions: delete_sessions authorises against cutoff_ms, so
        # none of these is eligible however they were passed in.
        for sid in ("aaa", "bbb", "ccc"):
            _add_session(conn, sid, age_days=1)
        before = _counts(conn)

        outcome, batches = self._bounded(
            conn, ["aaa", "bbb", "ccc"], cutoff=_cutoff(), batch=1, limit=20
        )

        assert outcome.failure is None, (
            f"the loop did not terminate on its own: {outcome.failure}"
        )
        assert batches <= 2, (
            f"one batch suffices to drop every ineligible candidate, took {batches}"
        )
        assert sorted(outcome.skipped) == ["aaa", "bbb", "ccc"]
        assert outcome.remaining == 0
        assert _counts(conn) == before, "nothing was eligible, so nothing may go"

    def test_a_cycle_among_the_candidates_does_not_spin(self, db):
        """A parent_id cycle is ineligible for a different reason -- no safe
        order rather than the wrong age -- and reaches the same drop.

        Mutation (D2-ineligible-empty): same. 'plain' is deleted, then the two
        cyclic ids stay in `pending` with nothing to delete: the no-progress
        guard stops the run and `failure` is set.
        """
        path, conn = db
        _add_session(conn, "x", age_days=30)
        _add_session(conn, "y", age_days=30, parent="x")
        conn.execute("UPDATE session SET parent_id='y' WHERE id='x'")
        _add_session(conn, "plain", age_days=30)

        outcome, batches = self._bounded(
            conn, ["plain", "x", "y"], cutoff=_cutoff(), batch=1, limit=20
        )

        assert outcome.failure is None, (
            f"the loop did not terminate on its own: {outcome.failure}"
        )
        assert batches <= 3
        assert outcome.rows["session"] == 1
        assert sorted(outcome.skipped) == ["x", "y"]
        assert outcome.remaining == 0
        assert sorted(r[0] for r in conn.execute("SELECT id FROM session")) == ["x", "y"]

    def test_every_batch_makes_progress_on_a_mixed_workload(self, db):
        """Deletable and ineligible candidates together, one session per
        batch: the loop must take at most one iteration per candidate plus a
        final empty one, never more.

        Mutation (D2-ineligible-empty): the three live ids are never dropped,
        so the batch after the last deletion makes no progress and the run
        stops with a failure instead of finishing clean.
        """
        path, conn = db
        old = [f"old{i}" for i in range(4)]
        for sid in old:
            _add_session(conn, sid, age_days=30)
        for sid in ("live0", "live1", "live2"):
            _add_session(conn, sid, age_days=1)

        outcome, batches = self._bounded(
            conn, old + ["live0", "live1", "live2"],
            cutoff=_cutoff(), batch=1, limit=20,
        )

        assert outcome.failure is None, (
            f"the loop did not terminate on its own: {outcome.failure}"
        )
        assert batches <= len(old) + 1, (
            f"{len(old)} deletable candidates at batch=1 need at most "
            f"{len(old) + 1} iterations, took {batches}"
        )
        assert outcome.rows["session"] == 4
        assert sorted(outcome.skipped) == ["live0", "live1", "live2"]
        assert outcome.remaining == 0
        assert sorted(r[0] for r in conn.execute("SELECT id FROM session")) == \
            ["live0", "live1", "live2"]

    def test_a_batch_that_makes_no_progress_stops_instead_of_looping(self, db):
        """The guard itself, driven directly: a graph where the candidates are
        neither deleted nor dropped is what a progress bug looks like from
        inside the loop, and the run must end saying so.

        Constructed by stubbing `expired_session_ids` to report every pending
        candidate as eligible -- so nothing is dropped -- while putting a
        session that is not pending first, so the `batch`-sized slice deletes
        none of them. The real function cannot produce that, because
        `restrict_to=pending` bounds what it returns; the guard exists for the
        case that invariant is broken, and asserting it needs the case built.

        Mutation (D2-no-progress-guard-dropped): delete the
        `if len(pending) == remaining_before:` block. The loop then iterates
        on a `pending` that never shrinks; the bounded clock stops it, so
        `failure` reads "MemoryError: loop exceeded 20 batches" and the
        assertions below fail rather than the suite hanging.
        """
        path, conn = db
        for sid in ("aaa", "bbb"):
            _add_session(conn, sid, age_days=30)
        before = _counts(conn)

        import unittest.mock

        # 'zzz' is no session at all, so the batch deletes nothing, while
        # 'aaa'/'bbb' being eligible means neither is dropped as ineligible.
        stuck = opencode_gc.Selection(deletable=["zzz", "aaa", "bbb"])
        with unittest.mock.patch.object(
            opencode_gc, "expired_session_ids", return_value=stuck
        ):
            outcome, batches = self._bounded(
                conn, ["aaa", "bbb"], cutoff=_cutoff(), batch=1, limit=20
            )

        assert outcome.failure is not None, "a stuck loop must be reported"
        assert "no progress" in outcome.failure
        assert batches <= 2, f"the guard must stop on the first stuck batch, took {batches}"
        assert outcome.remaining == 2, "the candidates it could not resolve are still owed"
        assert outcome.incomplete is True
        assert _counts(conn) == before, "a stuck batch must not have deleted anything"
        assert conn.in_transaction is False


class TestArgumentBounds:
    """Invalid numerics must be refused, not silently turned into "no limit"."""

    @pytest.fixture()
    def anydb(self, tmp_path):
        path = tmp_path / "args.db"
        conn = _make_db(path)
        _add_session(conn, "old", age_days=30)
        conn.close()
        return path

    @pytest.mark.parametrize("flag,value", [
        ("--max-seconds", "nan"),
        ("--max-seconds", "-1"),
        ("--vacuum-pages", "0"),
        ("--vacuum-pages", "-1"),
        ("--retention-days", "nan"),
    ])
    def test_invalid_bounds_are_refused_without_touching_the_db(
        self, anydb, monkeypatch, flag, value
    ):
        """Mutation: drop the `not math.isfinite(args.max_seconds) or
        args.max_seconds < 0` check (and likewise the --vacuum-pages check).
        The NaN/negative cases then run to completion instead of exiting 2,
        having silently disabled the deadline or the page cap.
        """
        before = _snapshot(anydb)
        monkeypatch.setattr(
            opencode_gc.sys, "argv",
            ["opencode-gc", "--db", str(anydb), "--apply", flag, value],
        )
        with pytest.raises(SystemExit) as exc:
            opencode_gc.main()
        assert exc.value.code == 2
        assert _snapshot(anydb) == before

    def test_max_seconds_zero_means_no_limit_and_completes(self, anydb, monkeypatch):
        """The documented escape hatch must actually run to completion."""
        monkeypatch.setattr(
            opencode_gc.sys, "argv",
            ["opencode-gc", "--db", str(anydb), "--apply", "--max-seconds", "0"],
        )
        assert opencode_gc.main() == 0
        assert _snapshot(anydb)["session"] == []


class TestCli:
    """Nothing above goes through argument parsing, so an argument-wiring or
    mode-inversion bug would pass every other test in this file.
    """

    def _run(self, monkeypatch, argv):
        monkeypatch.setattr(opencode_gc.sys, "argv", ["opencode-gc", *argv])
        return opencode_gc.main()

    @pytest.fixture()
    def populated(self, tmp_path):
        path = tmp_path / "cli.db"
        conn = _make_db(path)
        _add_session(conn, "old", age_days=30, events=4, messages=2)
        _add_session(conn, "live", age_days=1, events=3, messages=2)
        _add_session(conn, "oldparent", age_days=40)
        _add_session(conn, "livekid", age_days=1, parent="oldparent")
        conn.close()
        return path

    def test_dry_run_changes_nothing(self, populated, monkeypatch, capsys):
        """Mutation: in main, `conn = connect(args.db, read_only=not args.apply)`
        stays, but change `if args.apply:` before delete_sessions to
        `if True:`. The dry run then writes and the snapshot differs.
        """
        before = _snapshot(populated)
        rc = self._run(monkeypatch, ["--db", str(populated)])
        assert rc == 0
        assert _snapshot(populated) == before
        assert "DRY RUN" in capsys.readouterr().out

    def test_apply_deletes_only_eligible_sessions(self, populated, monkeypatch):
        """Mutation: `if args.apply:` -> `if not args.apply:` around the
        delete/count branch. 'old' then survives.
        """
        rc = self._run(monkeypatch, ["--db", str(populated), "--apply"])
        assert rc == 0
        after = _snapshot(populated)
        assert sorted(r[0] for r in after["session"]) == ["live", "livekid", "oldparent"]
        assert {r[1] for r in after["event"]} == {"live", "livekid", "oldparent"}
        assert {r[1] for r in after["message"]} == {"live", "livekid", "oldparent"}

    def test_retention_floor_is_refused_and_nothing_is_touched(self, populated, monkeypatch):
        """Mutation: `args.retention_days < MIN_RETENTION_DAYS` ->
        `args.retention_days < 0`. The run then proceeds and deletes.
        """
        before = _snapshot(populated)
        with pytest.raises(SystemExit) as exc:
            self._run(monkeypatch, ["--db", str(populated), "--retention-days", "0.5", "--apply"])
        assert exc.value.code == 2
        assert _snapshot(populated) == before

    def test_missing_database_is_reported_not_created(self, tmp_path, monkeypatch):
        missing = tmp_path / "nope.db"
        assert self._run(monkeypatch, ["--db", str(missing)]) == 2
        assert not missing.exists()

    def test_apply_on_a_missing_database_creates_nothing(self, tmp_path, monkeypatch, capsys):
        """The destructive path of the same guard. sqlite3.connect() creates
        the file it cannot find, so without the check --apply on a typo'd
        --db would silently make an empty database at that path -- and report
        a clean run against it.

        Mutation (G-no-is-file-guard): in main, delete the
        `if not args.db.is_file():` block. rc becomes 0, and the file below
        exists.

        The dry-run half is covered above; this is the half that writes.
        """
        missing = tmp_path / "typo.db"

        rc = self._run(monkeypatch, ["--db", str(missing), "--apply"])

        assert rc == 2
        assert "no database at" in capsys.readouterr().err
        assert not missing.exists(), \
            "a mistyped --db must not leave a new database behind"
        assert list(tmp_path.iterdir()) == [], \
            "no sidecar (-wal, -shm) may be created either"

    def test_failed_mode_conversion_aborts_before_deleting(self, tmp_path, monkeypatch, capsys):
        """--apply --enable-incremental-vacuum must not destroy history when it
        cannot enable reclamation: that is the worst of both outcomes.

        Mutation: in main's `except (RuntimeError, sqlite3.Error)` handler,
        replace `return _report(args, res)` with `pass`. The run then deletes
        'old' while still exiting nonzero.
        """
        path = tmp_path / "nofree.db"
        conn = _make_db(path, auto_vacuum=0)
        _add_session(conn, "old", age_days=30)
        _add_session(conn, "live", age_days=1)
        conn.close()
        before = _snapshot(path)

        monkeypatch.setattr(
            opencode_gc.shutil, "disk_usage", lambda _p: type("U", (), {"free": 0})()
        )
        rc = self._run(
            monkeypatch, ["--db", str(path), "--apply", "--enable-incremental-vacuum"]
        )
        assert rc == 1
        assert "refusing to delete anything" in capsys.readouterr().err
        assert _snapshot(path) == before

    def test_batch_is_clamped_before_any_deletion(self, populated, monkeypatch, capsys):
        """A --batch above SQLite's parameter limit must not fail after
        earlier batches have already committed.

        Mutation: `batch = min(args.batch, var_limit)` -> `batch = args.batch`.
        The first DELETE raises sqlite3.OperationalError instead of succeeding.
        """
        rc = self._run(
            monkeypatch, ["--db", str(populated), "--apply", "--batch", "10000000"]
        )
        assert rc == 0
        assert "clamped" in capsys.readouterr().out
        assert sorted(r[0] for r in _snapshot(populated)["session"]) == [
            "live", "livekid", "oldparent",
        ]

    def test_json_reports_cutoff_and_keep_reasons(self, populated, monkeypatch, capsys):
        _, payload = _run_json_cli(monkeypatch, capsys, "--db", str(populated))
        assert payload["dry_run"] is True
        assert payload["sessions_expired"] == 1
        assert payload["sessions_kept_live_descendant"] == 1
        assert payload["cutoff_ms"] > 0


class TestStats:
    def test_reported_sizes_are_the_real_file_sizes(self, tmp_path, monkeypatch, capsys):
        """The old assertion restated the total_bytes property and never
        touched reporting. Check the JSON the CLI actually emits against the
        database and -wal files on disk.

        Mutation: in main, `res.bytes_before = on_disk_bytes(args.db)` ->
        `res.bytes_before = stats.total_bytes`. bytes_before then excludes the
        WAL and no longer matches the measured file sizes.
        """
        path = tmp_path / "sizes.db"
        conn = _make_db(path, wal=True)
        for i in range(80):
            _add_session(conn, f"s{i}", age_days=30, events=20, messages=5)
        # Left open: closing checkpoints and removes the WAL, and the point
        # here is that an uncheckpointed WAL is counted.
        wal_path = path.with_name(path.name + "-wal")
        assert wal_path.is_file(), "fixture must leave an uncheckpointed WAL to measure"
        db_bytes = path.stat().st_size
        wal_bytes = wal_path.stat().st_size
        assert wal_bytes > 0

        try:
            rc, payload = _run_json_cli(monkeypatch, capsys, "--db", str(path))
            assert rc == 0

            assert payload["bytes_before"] == db_bytes + wal_bytes
            assert payload["auto_vacuum_before"] == "INCREMENTAL"
            assert payload["bytes_reclaimed"] == 0, "a dry run reclaims nothing"
        finally:
            conn.close()

    def test_reclaimed_bytes_are_measured_not_asserted(self, tmp_path, monkeypatch, capsys):
        """Released pages and reclaimed bytes are different quantities.

        In WAL mode a long-lived reader defers the checkpoint that truncates
        the file, so pages leave the freelist while the bytes are still on
        disk -- here the footprint even grows, by the WAL the deletes wrote.
        Reporting page arithmetic as "freed" would claim space the filesystem
        has not got back. A reader is held open for exactly that reason;
        without one, page arithmetic and the real file size happen to agree
        and the assertion would prove nothing.

        Mutation: `res.bytes_after = on_disk_bytes(args.db)` ->
        `res.bytes_after = res.bytes_before - res.pages_released * 4096`.
        bytes_reclaimed then reports pages that are still on disk.
        """
        path = tmp_path / "reclaim.db"
        conn = _make_db(path, wal=True)
        for i in range(120):
            _add_session(conn, f"s{i}", age_days=30, events=30, messages=8)
        conn.close()

        # A concurrent reader, i.e. opencode itself, blocking the checkpoint.
        reader = sqlite3.connect(str(path), isolation_level=None)
        reader.execute("BEGIN")
        reader.execute("SELECT count(*) FROM event").fetchone()
        try:
            before = opencode_gc.on_disk_bytes(path)
            rc, payload = _run_json_cli(monkeypatch, capsys, "--db", str(path), "--apply")
            assert rc == 0

            after = opencode_gc.on_disk_bytes(path)
            assert payload["pages_released"] > 0, "pages must actually be released"
            assert payload["bytes_after"] == after
            assert payload["bytes_reclaimed"] == max(0, before - after)

            # The blocked checkpoint means the footprint has not shrunk at all
            # -- here it has grown by the WAL the deletes themselves wrote.
            assert after >= before
            assert payload["bytes_reclaimed"] == 0, (
                "no byte left the disk, so none may be reported as reclaimed"
            )
            assert any("has not shrunk yet" in n for n in payload["notes"])
        finally:
            reader.close()

    def test_on_disk_bytes_includes_the_wal(self, tmp_path):
        """Reclaimed bytes must come from real file sizes, not page arithmetic.

        Mutation: drop the WAL from on_disk_bytes' loop.
        """
        path = tmp_path / "sizes.db"
        conn = _make_db(path, wal=True)
        try:
            for i in range(100):
                _add_session(conn, f"s{i}", age_days=30, events=20, messages=5)
            wal = path.with_name(path.name + "-wal").stat().st_size
            assert wal > 0
            assert opencode_gc.on_disk_bytes(path) == path.stat().st_size + wal
        finally:
            conn.close()

    def test_a_symlinked_database_measures_the_target_wal(self, tmp_path, monkeypatch, capsys):
        """connect() opens the resolved file, so the sidecar probes must
        resolve too: for `alias.db -> target.db` there is no `alias.db-wal`,
        and probing one silently reports a WAL of zero bytes.

        Mutation: in main, `res.bytes_before = on_disk_bytes(db_path)` ->
        `on_disk_bytes(args.db)` (the unresolved path). bytes_before then
        omits the whole WAL and no longer matches the files on disk.
        """
        target = tmp_path / "target.db"
        conn = _make_db(target, wal=True)
        for i in range(80):
            _add_session(conn, f"s{i}", age_days=30, events=20, messages=5)
        alias = tmp_path / "alias.db"
        alias.symlink_to(target)
        wal_path = target.with_name(target.name + "-wal")
        assert wal_path.is_file()
        assert not alias.with_name(alias.name + "-wal").exists(), \
            "the alias must have no -wal of its own, or the probe cannot go wrong"
        expected = target.stat().st_size + wal_path.stat().st_size

        try:
            rc, payload = _run_json_cli(monkeypatch, capsys, "--db", str(alias))
            assert rc == 0

            assert payload["bytes_before"] == expected
            # The report still names what the operator typed.
            assert payload["db"] == str(alias)
        finally:
            conn.close()


class TestSqliteTempDirSearchOrder:
    """The guard measures free space wherever SQLite will actually write its
    VACUUM copy. https://sqlite.org/tempfiles.html gives the unix order:
    SQLITE_TMPDIR, TMPDIR, /var/tmp, /usr/tmp, /tmp, then the current
    directory -- first one that exists and is writable and searchable.

    Python's tempfile.gettempdir() applies a different policy and caches its
    answer, so on a host where /var/tmp and /tmp are separate filesystems it
    can name the wrong one and approve a VACUUM that fills the real one.
    """

    def test_sqlite_tmpdir_wins(self, tmp_path, monkeypatch):
        chosen = tmp_path / "chosen"
        chosen.mkdir()
        other = tmp_path / "other"
        other.mkdir()
        monkeypatch.setenv("SQLITE_TMPDIR", str(chosen))
        monkeypatch.setenv("TMPDIR", str(other))
        assert opencode_gc._sqlite_temp_dir() == chosen

    def test_tmpdir_is_used_when_sqlite_tmpdir_is_unset(self, tmp_path, monkeypatch):
        """Mutation: drop `os.environ.get("TMPDIR")` from the candidate list.
        The function then skips straight to /var/tmp and returns the wrong
        filesystem.
        """
        chosen = tmp_path / "chosen"
        chosen.mkdir()
        monkeypatch.delenv("SQLITE_TMPDIR", raising=False)
        monkeypatch.setenv("TMPDIR", str(chosen))
        assert opencode_gc._sqlite_temp_dir() == chosen

    def test_a_nonexistent_sqlite_tmpdir_falls_through(self, tmp_path, monkeypatch):
        """SQLite skips a candidate that does not exist rather than failing on
        it, and so must the guard.

        Mutation: `if path.is_dir() and os.access(...)` -> `return path` on the
        first candidate (the old `SQLITE_TMPDIR or gettempdir()` behaviour).
        The nonexistent directory is then returned and stat() raises
        FileNotFoundError out of enable_incremental_vacuum.
        """
        fallback = tmp_path / "fallback"
        fallback.mkdir()
        monkeypatch.setenv("SQLITE_TMPDIR", str(tmp_path / "does-not-exist"))
        monkeypatch.setenv("TMPDIR", str(fallback))
        assert opencode_gc._sqlite_temp_dir() == fallback

    def test_an_unwritable_candidate_is_skipped(self, tmp_path, monkeypatch):
        """Mutation: drop `and os.access(path, os.W_OK | os.X_OK)` from the
        test. The unwritable directory is then chosen, and the space check
        runs against a filesystem SQLite cannot use.
        """
        unwritable = tmp_path / "unwritable"
        unwritable.mkdir(mode=0o500)
        fallback = tmp_path / "fallback"
        fallback.mkdir()
        monkeypatch.setenv("SQLITE_TMPDIR", str(unwritable))
        monkeypatch.setenv("TMPDIR", str(fallback))
        try:
            assert opencode_gc._sqlite_temp_dir() == fallback
        finally:
            unwritable.chmod(0o700)

    def test_an_unsearchable_candidate_is_skipped(self, tmp_path, monkeypatch):
        """SQLite requires the directory be searchable as well as writable, and
        0o600 is writable but not. Without this, the X_OK half of the check is
        unpinned: the 0o500 case above fails W_OK too, so dropping X_OK alone
        changes nothing there.

        Mutation (T4-no-exec-bit-check): `os.access(path, os.W_OK | os.X_OK)`
        -> `os.access(path, os.W_OK)`. The unsearchable directory is then
        chosen, and the guard measures a filesystem SQLite cannot write its
        VACUUM copy to.
        """
        unsearchable = tmp_path / "unsearchable"
        unsearchable.mkdir(mode=0o600)
        fallback = tmp_path / "fallback"
        fallback.mkdir()
        assert os.access(unsearchable, os.W_OK), \
            "the fixture must be writable, or W_OK alone would reject it"
        assert not os.access(unsearchable, os.X_OK), \
            "the fixture must not be searchable, or there is nothing to pin"
        monkeypatch.setenv("SQLITE_TMPDIR", str(unsearchable))
        monkeypatch.setenv("TMPDIR", str(fallback))
        try:
            assert opencode_gc._sqlite_temp_dir() == fallback
        finally:
            unsearchable.chmod(0o700)

    def test_the_current_directory_is_the_last_resort(self, tmp_path, monkeypatch):
        """SQLite falls back to the current directory when no named candidate
        qualifies, and so must the guard: refusing outright would block a
        conversion SQLite would have completed.

        Asserted by using the fallback, not by neutralising it -- the cwd is
        moved to a known directory and the returned path must resolve there.

        Mutation (T5-no-cwd-fallback): drop `candidates.append(".")`. Every
        named candidate is missing here, so _sqlite_temp_dir raises
        RuntimeError instead of returning the cwd.
        """
        monkeypatch.delenv("SQLITE_TMPDIR", raising=False)
        monkeypatch.delenv("TMPDIR", raising=False)
        missing = tmp_path / "absent"
        monkeypatch.setattr(
            opencode_gc, "SQLITE_TEMP_DIR_CANDIDATES", (str(missing),)
        )
        cwd = tmp_path / "cwd"
        cwd.mkdir()
        monkeypatch.chdir(cwd)

        chosen = opencode_gc._sqlite_temp_dir()

        assert chosen.resolve() == cwd.resolve(), (
            "with no named candidate usable, SQLite would write its VACUUM "
            f"copy in the current directory; the guard chose {chosen}"
        )

    def test_documented_order_is_followed_when_the_env_is_empty(self, tmp_path, monkeypatch):
        """With no environment override, /var/tmp is preferred over /tmp --
        SQLite's own precedence, and the two are separate filesystems on the
        hosts this tool targets.

        The expectation is written out literally rather than derived from
        SQLITE_TEMP_DIR_CANDIDATES: computing it from the constant under test
        would reverse with the constant and assert nothing.

        Mutation: `SQLITE_TEMP_DIR_CANDIDATES = ("/var/tmp", "/usr/tmp",
        "/tmp")` -> `("/tmp", "/usr/tmp", "/var/tmp")`. '/tmp' is then
        returned where SQLite would have used '/var/tmp'.
        """
        monkeypatch.delenv("SQLITE_TMPDIR", raising=False)
        monkeypatch.delenv("TMPDIR", raising=False)

        # Stand-ins for the real system directories, so the assertion does not
        # depend on which of them happens to exist on the host.
        var_tmp = tmp_path / "var-tmp"
        usr_tmp = tmp_path / "usr-tmp"
        plain_tmp = tmp_path / "plain-tmp"
        for d in (var_tmp, usr_tmp, plain_tmp):
            d.mkdir()
        monkeypatch.setattr(
            opencode_gc, "SQLITE_TEMP_DIR_CANDIDATES",
            (str(var_tmp), str(usr_tmp), str(plain_tmp)),
        )

        assert opencode_gc._sqlite_temp_dir() == var_tmp, \
            "the first candidate in SQLite's order must win"

    def test_the_real_candidate_list_is_sqlites_documented_one(self):
        """The order the production constant actually carries.

        Mutation: same reversal. https://sqlite.org/tempfiles.html gives
        /var/tmp before /usr/tmp before /tmp, and on a host where /var/tmp and
        /tmp are separate filesystems the difference decides which filesystem
        the free-space guard measures.
        """
        assert opencode_gc.SQLITE_TEMP_DIR_CANDIDATES == \
            ("/var/tmp", "/usr/tmp", "/tmp")

    def test_a_later_candidate_is_used_when_an_earlier_one_is_missing(
        self, tmp_path, monkeypatch
    ):
        """Mutation: same reversal, and also `if path.is_dir() and
        os.access(...)` -> `return path` unconditionally.
        """
        monkeypatch.delenv("SQLITE_TMPDIR", raising=False)
        monkeypatch.delenv("TMPDIR", raising=False)
        present = tmp_path / "present"
        present.mkdir()
        monkeypatch.setattr(
            opencode_gc, "SQLITE_TEMP_DIR_CANDIDATES",
            (str(tmp_path / "absent"), str(present)),
        )
        assert opencode_gc._sqlite_temp_dir() == present

    def test_an_unusable_temp_dir_refuses_instead_of_raising_oserror(
        self, tmp_path, monkeypatch, capsys
    ):
        """A nonexistent SQLITE_TMPDIR used to reach os.stat() and escape as
        FileNotFoundError. It must be the documented conversion refusal, and
        nothing may be deleted.

        The refusal must come from _sqlite_temp_dir() finding no usable
        candidate at all, so every candidate -- including the "." fallback --
        is pointed at a directory that does not exist.

        Mutation: in _sqlite_temp_dir, replace the closing `raise
        RuntimeError(...)` with `return Path(candidates[0])`. The nonexistent
        directory is then handed to enable_incremental_vacuum, whose os.stat
        raises FileNotFoundError -- and main()'s handler catches RuntimeError
        and sqlite3.Error, not OSError, so the run dies with a traceback and
        prints no refusal at all.
        """
        path = tmp_path / "badtmp.db"
        conn = _make_db(path, auto_vacuum=0)
        _add_session(conn, "old", age_days=30)
        _add_session(conn, "live", age_days=1)
        conn.close()
        before = _snapshot(path)

        missing = tmp_path / "nope"
        assert not missing.exists()
        monkeypatch.setattr(
            opencode_gc, "SQLITE_TEMP_DIR_CANDIDATES", (str(missing / "3"),)
        )
        monkeypatch.setenv("SQLITE_TMPDIR", str(missing / "1"))
        monkeypatch.setenv("TMPDIR", str(missing / "2"))
        # The final "." candidate: run from a directory that has been removed
        # is not portable, so make Path(".") fail the is_dir() test instead.
        real_is_dir = opencode_gc.Path.is_dir
        monkeypatch.setattr(
            opencode_gc.Path, "is_dir",
            lambda self: False if str(self) == "." else real_is_dir(self),
        )
        monkeypatch.setattr(
            opencode_gc.sys, "argv",
            ["opencode-gc", "--db", str(path), "--apply", "--enable-incremental-vacuum"],
        )

        rc = opencode_gc.main()

        assert rc == 1
        err = capsys.readouterr().err
        assert "refusing to delete anything" in err
        assert "temporary directories" in err
        assert _snapshot(path) == before, "a refused conversion must delete nothing"

    def test_an_unreadable_temp_dir_is_a_refusal_not_a_traceback(
        self, tmp_path, monkeypatch, capsys
    ):
        """Measuring the temp filesystem is a separate syscall at a later
        instant than choosing it, and it can fail on its own -- an unmounted
        or stale network filesystem answers `is_dir()` and then refuses to be
        measured.

        Mutation: in enable_incremental_vacuum, remove the `try/except OSError`
        around the temp-dir stat and disk_usage. main()'s handler catches
        RuntimeError and sqlite3.Error, not OSError, so the run dies with a
        traceback instead of the documented refusal.
        """
        path = tmp_path / "statfail.db"
        conn = _make_db(path, auto_vacuum=0)
        _add_session(conn, "old", age_days=30)
        conn.close()
        before = _snapshot(path)

        usable = tmp_path / "usable"
        usable.mkdir()
        monkeypatch.setenv("SQLITE_TMPDIR", str(usable))

        # A different device, so the temp filesystem is measured at all.
        real_stat = opencode_gc.os.stat
        usable_real = real_stat(usable)

        def fake_stat(p, *a, **kw):
            st = real_stat(p, *a, **kw)
            dev = 77 if os.path.samestat(st, usable_real) else 88
            return type(st)((st.st_mode, st.st_ino, dev, st.st_nlink, st.st_uid,
                             st.st_gid, st.st_size, int(st.st_atime),
                             int(st.st_mtime), int(st.st_ctime)))

        def failing_usage(p):
            if Path(p) == usable:
                raise OSError("stale NFS file handle")
            return type("U", (), {"free": 1 << 60})()

        monkeypatch.setattr(opencode_gc.os, "stat", fake_stat)
        monkeypatch.setattr(opencode_gc.shutil, "disk_usage", failing_usage)
        monkeypatch.setattr(
            opencode_gc.sys, "argv",
            ["opencode-gc", "--db", str(path), "--apply", "--enable-incremental-vacuum"],
        )

        rc = opencode_gc.main()

        assert rc == 1
        err = capsys.readouterr().err
        assert "refusing to delete anything" in err
        assert "stale NFS file handle" in err
        assert _snapshot(path) == before, "a refused conversion must delete nothing"


class TestUnusableDatabase:
    """A corrupt file, or a valid SQLite file that is not opencode's, must be
    classified before any destructive work rather than surfacing as a
    traceback from the middle of the run.
    """

    def _run(self, monkeypatch, path, *extra):
        monkeypatch.setattr(
            opencode_gc.sys, "argv", ["opencode-gc", "--db", str(path), *extra]
        )
        return opencode_gc.main()

    def test_a_file_that_is_not_sqlite_is_refused(self, tmp_path, monkeypatch, capsys):
        """Mutation: delete the `verify_usable(conn)` call in main. The
        DatabaseError escapes from the first query as a traceback instead.
        """
        path = tmp_path / "junk.db"
        path.write_bytes(b"this is not a database at all")

        rc = self._run(monkeypatch, path, "--apply")

        assert rc == 2
        assert "cannot use" in capsys.readouterr().err
        assert path.read_bytes() == b"this is not a database at all", \
            "the file must be left exactly as it was"

    def test_a_foreign_sqlite_database_is_refused(self, tmp_path, monkeypatch, capsys):
        """Valid SQLite, but not opencode's schema: deleting from it would be
        deleting from the wrong database entirely.

        Mutation: same as above.
        """
        path = tmp_path / "foreign.db"
        conn = sqlite3.connect(str(path), isolation_level=None)
        conn.execute("CREATE TABLE unrelated (x integer)")
        conn.execute("INSERT INTO unrelated VALUES (1)")
        conn.close()
        before = _snapshot(path, ["unrelated"])

        rc = self._run(monkeypatch, path, "--apply")

        assert rc == 2
        assert "not an opencode database" in capsys.readouterr().err
        assert _snapshot(path, ["unrelated"]) == before

    def test_a_database_missing_one_child_table_is_refused(
        self, tmp_path, monkeypatch, capsys
    ):
        """Discovering a missing table mid-run would leave a half-pruned
        database: the session rows gone, their events stranded.

        The missing table is named as a *table*, not as a missing column on a
        table that is not there -- the column probe would otherwise subsume
        this check and the table list could be reduced to ["session"] with no
        test noticing.

        Mutation: in verify_usable, `required = ["session"] + [t for t, _ in
        CHILD_TABLES]` -> `required = ["session"]`. The refusal then comes
        from the column probe with a different message, so the assertion on
        "table(s)" fails.
        """
        path = tmp_path / "partial.db"
        conn = _make_db(path)
        _add_session(conn, "old", age_days=30)
        conn.execute("DROP TABLE event")
        conn.close()
        surviving = _snapshot(path, ["session", "event_sequence"])

        rc = self._run(monkeypatch, path, "--apply")

        assert rc == 2
        err = capsys.readouterr().err
        assert "table(s) event are missing" in err, (
            "a missing table must be reported as a missing table, before any "
            f"column probe: {err.strip()}"
        )
        assert _snapshot(path, ["session", "event_sequence"]) == surviving, \
            "nothing may be deleted from a database we cannot fully prune"

    def test_a_table_present_but_empty_of_its_keyed_column_is_refused(
        self, tmp_path, monkeypatch, capsys
    ):
        """The complement: the table exists, so only the column probe can
        catch it.

        Mutation: in verify_usable, drop the per-column PRAGMA table_info
        loop. The run then reaches the DELETE and fails mid-batch.
        """
        path = tmp_path / "nocolumn.db"
        conn = _make_db(path)
        _add_session(conn, "old", age_days=30)
        # 'event' still exists, but no longer keys on aggregate_id.
        conn.execute("ALTER TABLE event RENAME COLUMN aggregate_id TO agg")
        conn.close()
        before = _snapshot(path, ["session", "event"])

        rc = self._run(monkeypatch, path, "--apply")

        assert rc == 2
        assert "event.aggregate_id" in capsys.readouterr().err
        assert _snapshot(path, ["session", "event"]) == before

    def test_a_session_table_missing_parent_id_is_refused(
        self, tmp_path, monkeypatch, capsys
    ):
        """The whole live-descendant protection keys on parent_id. A session
        table without it cannot be pruned safely at all.

        Mutation: in verify_usable, drop the per-column PRAGMA table_info
        loop. The run then reaches the selection query and raises.
        """
        path = tmp_path / "nocol.db"
        conn = _make_db(path)
        _add_session(conn, "old", age_days=30)
        conn.execute("ALTER TABLE session DROP COLUMN parent_id")
        conn.close()
        before = _snapshot(path, ["session"])

        rc = self._run(monkeypatch, path, "--apply")

        assert rc == 2
        assert "parent_id" in capsys.readouterr().err
        assert _snapshot(path, ["session"]) == before

    def test_a_healthy_database_is_not_refused(self, tmp_path, monkeypatch):
        """The guard must not reject the databases it exists to protect.

        Mutation: in verify_usable, `if missing:` -> `if True:`. A perfectly
        good database is then refused and nothing is ever collected.
        """
        path = tmp_path / "healthy.db"
        conn = _make_db(path)
        _add_session(conn, "old", age_days=30)
        _add_session(conn, "live", age_days=1)
        conn.close()

        assert self._run(monkeypatch, path, "--apply") == 0
        assert [r[0] for r in _snapshot(path)["session"]] == ["live"]


class TestPostDeletionErrorsAreReported:
    """Once rows are gone, nothing may escape as a traceback: an operator
    needs the committed counts to decide what to restore, and a stack trace
    carries none of them.
    """

    def test_a_failure_after_deletion_still_reports_the_counts(
        self, tmp_path, monkeypatch, capsys
    ):
        """A SQLite error raised after the deletes commit -- here from the
        post-deletion read_stats() that decides whether to reclaim.

        Mutation: in main, move the `read_stats(conn)` / reclamation block
        back outside the `try` that routes post-deletion failures into the
        report (the round-2 arrangement, where only run_incremental_vacuum was
        guarded). main() then dies with a traceback, so json.loads below gets
        no output at all and the counts are lost.
        """
        path = tmp_path / "poststats.db"
        conn = _make_db(path)
        for i in range(3):
            _add_session(conn, f"s{i}", age_days=30)
        conn.close()

        real_read_stats = opencode_gc.read_stats
        calls = []

        def failing_read_stats(c):
            calls.append(1)
            # The first call is the pre-deletion one; the second is the
            # post-deletion check, by which point rows are already gone.
            if len(calls) >= 2:
                raise sqlite3.OperationalError("disk I/O error")
            return real_read_stats(c)

        monkeypatch.setattr(opencode_gc, "read_stats", failing_read_stats)
        rc, payload = _run_json_cli(monkeypatch, capsys, "--db", str(path), "--apply")

        assert rc == 1
        assert payload["incomplete"] is True
        assert any("disk I/O error" in e for e in payload["errors"])
        # The counts must describe the rows that really went.
        assert payload["sessions_deleted"] == 3
        assert len(_snapshot(path)["session"]) == 0
        assert payload["rows_deleted"]["event"] == 9

    def test_a_failing_close_after_deletion_is_reported_not_raised(
        self, tmp_path, monkeypatch, capsys
    ):
        """conn.close() sits after the deletes and can fail on its own.

        Mutation: in main's `finally`, replace the guarded close with a bare
        `conn.close()`. The error escapes and the report is never printed.
        """
        path = tmp_path / "postclose.db"
        setup = _make_db(path)
        for i in range(3):
            _add_session(setup, f"s{i}", age_days=30)
        setup.close()

        class FailsToClose(sqlite3.Connection):
            def close(self):
                super().close()
                raise sqlite3.OperationalError("close failed")

        gc_conn = sqlite3.connect(
            str(path), isolation_level=None, timeout=5, factory=FailsToClose
        )
        monkeypatch.setattr(opencode_gc, "connect", lambda *a, **kw: gc_conn)
        rc, payload = _run_json_cli(monkeypatch, capsys, "--db", str(path), "--apply")

        assert rc == 1
        assert payload["incomplete"] is True
        assert any("close failed" in e for e in payload["errors"])
        assert payload["sessions_deleted"] == 3
        assert len(_snapshot(path)["session"]) == 0

    def test_a_failing_size_measurement_after_deletion_is_reported(
        self, tmp_path, monkeypatch, capsys
    ):
        """on_disk_bytes() runs after close, once the rows are irreversibly
        gone; an OSError there must not bury the counts.

        Mutation: in main, drop the try/except around
        `res.bytes_after = on_disk_bytes(db_path)`. The OSError propagates and
        no report is printed.
        """
        path = tmp_path / "postsize.db"
        conn = _make_db(path)
        for i in range(3):
            _add_session(conn, f"s{i}", age_days=30)
        conn.close()

        real_on_disk = opencode_gc.on_disk_bytes
        calls = []

        def failing_on_disk(p):
            calls.append(1)
            if len(calls) >= 2:
                raise OSError("stat failed")
            return real_on_disk(p)

        monkeypatch.setattr(opencode_gc, "on_disk_bytes", failing_on_disk)
        rc, payload = _run_json_cli(monkeypatch, capsys, "--db", str(path), "--apply")

        assert rc == 1
        assert payload["incomplete"] is True
        assert any("stat failed" in e for e in payload["errors"])
        assert payload["sessions_deleted"] == 3
        assert len(_snapshot(path)["session"]) == 0


# ---------------------------------------------------------------------------
# Operational hygiene: WAL checkpointing, the opportunistic rebuild, a deletion
# loop that yields the write lock, and the child-table list that makes deleting
# with foreign keys off equivalent to deleting with the cascades on.
# ---------------------------------------------------------------------------

# The live schema's shape as of 2026-09-12, enumerated from
# `SELECT sql FROM sqlite_master` on a real store rather than assumed: every
# table whose foreign key names session(id), plus the two event tables that key
# on a session id with no foreign key at all. Columns are trimmed to the ones
# this tool reads; the FK graph is not.
LIVE_SCHEMA = """
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
    CREATE TABLE todo (
        session_id text NOT NULL REFERENCES session(id) ON DELETE CASCADE,
        content text NOT NULL,
        position integer NOT NULL,
        PRIMARY KEY (session_id, position)
    );
    CREATE TABLE session_message (
        id text PRIMARY KEY,
        session_id text NOT NULL REFERENCES session(id) ON DELETE CASCADE,
        data text NOT NULL
    );
    CREATE TABLE session_input (
        id text PRIMARY KEY,
        session_id text NOT NULL REFERENCES session(id) ON DELETE CASCADE,
        prompt text NOT NULL
    );
    CREATE TABLE session_share (
        session_id text PRIMARY KEY REFERENCES session(id) ON DELETE CASCADE,
        secret text NOT NULL
    );
    CREATE TABLE session_context_epoch (
        session_id text PRIMARY KEY REFERENCES session(id) ON DELETE CASCADE,
        baseline text NOT NULL
    );
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

# Every table in LIVE_SCHEMA that a deleted session must take with it, listed
# independently of CHILD_TABLES so that dropping one from production does not
# also drop the assertion that its rows were cleaned.
LIVE_CHILD_TABLES = (
    "message", "part", "todo", "session_message", "session_input",
    "session_share", "session_context_epoch", "event", "event_sequence",
)


def _make_live_db(path, *, wal=True):
    """A database with the live FK graph. auto_vacuum is set BEFORE the first
    table: set afterwards it is silently discarded and reads back 0, which
    would leave every reclamation test exercising nothing."""
    conn = sqlite3.connect(str(path), isolation_level=None)
    conn.execute("PRAGMA auto_vacuum=2")
    if wal:
        conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(LIVE_SCHEMA)
    assert conn.execute("PRAGMA auto_vacuum").fetchone()[0] == 2, \
        "the fixture must really be INCREMENTAL, or it tests nothing"
    return conn


def _add_live_session(conn, sid, *, age_days, parent=None, events=3, messages=2,
                      todos=2):
    t = int(time.time() * 1000) - int(age_days * DAY_MS)
    conn.execute(
        "INSERT INTO session (id, parent_id, time_created, time_updated) "
        "VALUES (?,?,?,?)", (sid, parent, t, t),
    )
    conn.execute("INSERT INTO event_sequence VALUES (?,?)", (sid, events))
    for i in range(events):
        conn.execute("INSERT INTO event VALUES (?,?,?)", (f"{sid}-e{i}", sid, "x" * 64))
    for m in range(messages):
        mid = f"{sid}-m{m}"
        conn.execute("INSERT INTO message VALUES (?,?,?)", (mid, sid, "y" * 64))
        conn.execute(
            "INSERT INTO part VALUES (?,?,?,?)", (f"{mid}-p0", mid, sid, "z" * 64)
        )
        conn.execute(
            "INSERT INTO session_message VALUES (?,?,?)", (f"{mid}-sm", sid, "w" * 64)
        )
    for n in range(todos):
        conn.execute("INSERT INTO todo VALUES (?,?,?)", (sid, f"todo {n}", n))
    conn.execute("INSERT INTO session_input VALUES (?,?,?)", (f"{sid}-in", sid, "ask"))
    conn.execute("INSERT INTO session_share VALUES (?,?)", (sid, "secret"))
    conn.execute("INSERT INTO session_context_epoch VALUES (?,?)", (sid, "base"))


def _live_counts(conn):
    return {
        t: conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
        for t in ("session",) + LIVE_CHILD_TABLES
    }


class TestChildTablesCoverTheLiveSchema:
    """CHILD_TABLES being complete is what makes deleting with
    `PRAGMA foreign_keys` off -- SQLite's default, and what this tool runs
    under -- equivalent to deleting with the cascades on.

    It was not complete. `todo`, `session_message`, `session_input` and
    `session_share` all carry `session_id ... ON DELETE CASCADE` in the live
    schema and were absent from the list, so every session the tool deleted
    orphaned their rows silently. `session_context_epoch` is a fifth.
    """

    def test_every_session_child_in_the_live_schema_is_covered(self, tmp_path):
        """The regression that a future opencode migration would reintroduce.

        Mutation: drop ("todo", "session_id") -- or any other entry -- from
        CHILD_TABLES. `todo` is then a session child production does not know
        about, and this fails naming it.
        """
        conn = _make_live_db(tmp_path / "cover.db")
        try:
            children = opencode_gc.session_child_tables(conn)
            # The fixture must really contain the tables at issue, or the
            # set-difference below is trivially empty and proves nothing.
            assert {"todo", "session_message", "session_input", "session_share",
                    "session_context_epoch", "message"} <= children

            known = {t for t, _ in opencode_gc.CHILD_TABLES}
            assert children - known == set(), (
                "these tables reference session(id) but are not in "
                f"CHILD_TABLES, so their rows are orphaned: {children - known}"
            )
        finally:
            conn.close()

    def test_a_new_session_child_is_refused_rather_than_orphaned(self, tmp_path):
        """A schema gaining a session child this tool does not know about must
        stop the run, not prune around it.

        Mutation: delete the `uncovered` block from verify_usable. The run then
        proceeds and deletes sessions whose `bookmark` rows survive them.
        """
        path = tmp_path / "future.db"
        conn = _make_live_db(path)
        conn.execute(
            "CREATE TABLE bookmark (id text PRIMARY KEY, "
            "session_id text NOT NULL REFERENCES session(id) ON DELETE CASCADE)"
        )
        _add_live_session(conn, "old", age_days=30)
        conn.execute("INSERT INTO bookmark VALUES ('b1','old')")
        conn.close()

        with pytest.raises(RuntimeError, match="bookmark"):
            probe = opencode_gc.connect(path, read_only=True)
            try:
                opencode_gc.verify_usable(probe)
            finally:
                probe.close()

    def test_the_cli_refuses_an_unknown_child_before_deleting_anything(
        self, tmp_path, monkeypatch, capsys
    ):
        path = tmp_path / "future_cli.db"
        conn = _make_live_db(path)
        conn.execute(
            "CREATE TABLE bookmark (id text PRIMARY KEY, "
            "session_id text NOT NULL REFERENCES session(id) ON DELETE CASCADE)"
        )
        _add_live_session(conn, "old", age_days=30)
        conn.execute("INSERT INTO bookmark VALUES ('b1','old')")
        conn.close()

        monkeypatch.setattr(
            opencode_gc.sys, "argv",
            ["opencode-gc", "--db", str(path), "--apply"],
        )
        assert opencode_gc.main() == 2
        assert "bookmark" in capsys.readouterr().err
        survivor = sqlite3.connect(str(path))
        try:
            assert survivor.execute("SELECT count(*) FROM session").fetchone()[0] == 1
        finally:
            survivor.close()

    def test_deleting_a_session_leaves_no_row_in_any_child(self, tmp_path):
        """The orphan itself, not the table list. Every child is counted from
        the schema, so removing a table from CHILD_TABLES fails here.

        Mutation: drop ("todo", "session_id") from CHILD_TABLES -> todo keeps
        2 rows for a session that no longer exists.
        """
        path = tmp_path / "orphans.db"
        conn = _make_live_db(path)
        try:
            _add_live_session(conn, "old", age_days=30)
            _add_live_session(conn, "live", age_days=1)
            assert _live_counts(conn)["todo"] == 4

            outcome = opencode_gc.delete_sessions(
                conn, ["old"], cutoff_ms=_cutoff(), batch=25, deadline=None,
            )
            assert outcome.failure is None

            for table in LIVE_CHILD_TABLES:
                column = "aggregate_id" if table.startswith("event") else "session_id"
                orphans = conn.execute(
                    f"SELECT count(*) FROM {table} WHERE {column} "
                    "NOT IN (SELECT id FROM session)"
                ).fetchone()[0]
                assert orphans == 0, f"{table} kept {orphans} orphaned row(s)"
            # And the live session is untouched.
            assert conn.execute(
                "SELECT count(*) FROM todo WHERE session_id='live'"
            ).fetchone()[0] == 2
        finally:
            conn.close()

    def test_the_counts_report_every_child_table(self, tmp_path):
        """A dry run's estimate must cover the same tables the apply path
        deletes, or it understates what is about to be destroyed."""
        path = tmp_path / "counts.db"
        conn = _make_live_db(path)
        try:
            _add_live_session(conn, "old", age_days=30)
            predicted = opencode_gc.count_rows_for(conn, ["old"], 25)
            actual = opencode_gc.delete_sessions(
                conn, ["old"], cutoff_ms=_cutoff(), batch=25, deadline=None,
            ).rows
            assert predicted == actual
            assert predicted["todo"] == 2
            assert predicted["session_share"] == 1
        finally:
            conn.close()

    def test_a_store_without_the_newer_tables_still_prunes(self, tmp_path):
        """opencode added several of these tables recently and this tool has to
        keep working against a store written by an older build: the optional
        ones are deleted from when present and ignored when absent.

        Mutation: make every CHILD_TABLES entry required in verify_usable. The
        old-shaped database below is then refused outright.
        """
        path = tmp_path / "older.db"
        conn = _make_db(path)
        try:
            _add_session(conn, "old", age_days=30)
            opencode_gc.verify_usable(conn)
            outcome = opencode_gc.delete_sessions(
                conn, ["old"], cutoff_ms=_cutoff(), batch=25, deadline=None,
            )
            assert outcome.failure is None
            assert outcome.rows["session"] == 1
            assert "todo" not in outcome.rows
            assert _counts(conn) == {
                "session": 0, "message": 0, "part": 0, "event": 0,
                "event_sequence": 0,
            }
        finally:
            conn.close()


class TestCheckpointModeSelection:
    """TRUNCATE and RESTART wait for readers and block writers while they hold
    the WAL; PASSIVE never blocks. opencode instances are writers, and a writer
    that exhausts its own busy_timeout behind us dies with "Failed to execute
    statement", so the mode is chosen from whether anything else holds the
    database -- never by preference.
    """

    @pytest.mark.parametrize(
        "holders, expected",
        [
            ([], "TRUNCATE"),           # determined, and nobody: safe to block
            (["123"], "PASSIVE"),       # somebody is attached
            (["123", "456"], "PASSIVE"),
            (None, "PASSIVE"),          # undeterminable is not idle
        ],
    )
    def test_only_a_database_known_to_be_idle_gets_a_blocking_mode(
        self, holders, expected
    ):
        """Mutation: `return "TRUNCATE" if holders == [] else "PASSIVE"` ->
        `... if not holders else ...`. The None case then selects TRUNCATE,
        which is the fail-open this whole design exists to prevent.
        """
        assert opencode_gc.checkpoint_mode_for(holders) == expected

    def test_the_cli_uses_passive_while_something_holds_the_database(
        self, tmp_path, monkeypatch, capsys
    ):
        """Mutation: in main, `checkpoint_wal(conn, checkpoint_mode_for(holders))`
        -> `checkpoint_wal(conn, "TRUNCATE")`. The reported mode is then
        TRUNCATE with a holder attached.
        """
        path = tmp_path / "busy.db"
        conn = _make_live_db(path)
        for i in range(4):
            _add_live_session(conn, f"s{i}", age_days=30)
        conn.close()

        monkeypatch.setattr(opencode_gc, "db_holders", lambda db: ["4242"])
        rc, payload = _run_json_cli(
            monkeypatch, capsys, "--db", str(path), "--apply", "--batch-sleep-ms", "0"
        )

        assert rc == 0
        assert payload["wal_checkpoint_mode"] == "PASSIVE"
        assert payload["holders_before"] == ["4242"]

    def test_the_cli_uses_truncate_only_when_nothing_holds_the_database(
        self, tmp_path, monkeypatch, capsys
    ):
        path = tmp_path / "idle.db"
        conn = _make_live_db(path)
        for i in range(4):
            _add_live_session(conn, f"s{i}", age_days=30)
        conn.close()

        monkeypatch.setattr(opencode_gc, "db_holders", lambda db: [])
        rc, payload = _run_json_cli(
            monkeypatch, capsys, "--db", str(path), "--apply", "--batch-sleep-ms", "0"
        )

        assert rc == 0
        assert payload["wal_checkpoint_mode"] == "TRUNCATE"

    def test_undeterminable_holders_downgrade_the_checkpoint_and_say_so(
        self, tmp_path, monkeypatch, capsys
    ):
        """launchd and systemd hand a job a bare environment and macOS keeps
        lsof in /usr/sbin, so this is the expected failure, not an exotic one.
        """
        path = tmp_path / "unknown.db"
        conn = _make_live_db(path)
        _add_live_session(conn, "old", age_days=30)
        conn.close()

        monkeypatch.setattr(opencode_gc, "db_holders", lambda db: None)
        rc, payload = _run_json_cli(
            monkeypatch, capsys, "--db", str(path), "--apply", "--batch-sleep-ms", "0"
        )

        assert rc == 0
        assert payload["wal_checkpoint_mode"] == "PASSIVE"
        assert payload["holders_before"] is None
        assert any("could not be determined" in n for n in payload["notes"])


class TestCheckpointActuallyFoldsTheWal:
    """opencode-gc never checkpointed. In WAL mode a page released by
    incremental_vacuum does not leave the file until a checkpoint runs, and an
    uncheckpointed WAL is itself on the disk -- 15.28 GiB of it on vibes.

    Every test here holds a second connection open for the duration, and must:
    SQLite checkpoints and deletes the WAL when the LAST connection closes, so
    against a database nothing else has open the file is folded whether this
    tool checkpoints or not, and the assertions would pass on a build that
    never checkpoints at all. An attached connection is also the only case that
    matters in production, because opencode is the thing attached. Verified:
    with one idle connection open, our close() leaves the 8,689,112-byte WAL
    untouched, and the explicit checkpoint takes it to 0.
    """

    def _populated(self, path, sessions=40):
        conn = _make_live_db(path)
        for i in range(sessions):
            _add_live_session(conn, f"s{i}", age_days=30, events=40, messages=6)
        conn.close()

    @pytest.fixture()
    def attached(self):
        """An idle connection, as opencode would be between requests. It holds
        no read transaction, so it defers nothing -- it only stops SQLite from
        folding the WAL for us on close."""
        opened = []

        def attach(path):
            conn = sqlite3.connect(str(path), isolation_level=None)
            conn.execute("SELECT count(*) FROM session").fetchone()
            opened.append(conn)
            return conn

        yield attach
        for conn in opened:
            conn.close()

    def test_an_idle_run_leaves_no_wal_behind(
        self, tmp_path, monkeypatch, capsys, attached
    ):
        """Mutation: delete the `checkpoint_wal(conn, checkpoint_mode_for(...))`
        call from main. The 8 MB the deletes wrote to the WAL is then still on
        disk when the run reports itself finished.
        """
        path = tmp_path / "fold.db"
        self._populated(path)
        attached(path)
        wal = path.with_name(path.name + "-wal")
        monkeypatch.setattr(opencode_gc, "db_holders", lambda db: [])

        rc, payload = _run_json_cli(
            monkeypatch, capsys, "--db", str(path), "--apply", "--batch-sleep-ms", "0"
        )

        assert rc == 0
        assert payload["sessions_deleted"] == 40
        assert payload["wal_pages_checkpointed"] > 0
        assert wal.exists(), "the attached connection must keep the WAL in place"
        assert wal.stat().st_size == 0, (
            "a TRUNCATE checkpoint on an idle database must empty the WAL, "
            f"but {wal.stat().st_size} bytes remain"
        )
        assert payload["wal_bytes_after"] == 0

    def test_the_file_actually_shrinks_when_nothing_defers_the_checkpoint(
        self, tmp_path, monkeypatch, capsys, attached
    ):
        """The point of the whole exercise: released pages become bytes the
        filesystem has back, which is what bytes_before/bytes_after measures.

        Mutation: delete the checkpoint call from main. pages_released stays
        positive while bytes_reclaimed collapses to 0 and the run starts
        emitting the "has not shrunk yet" note.
        """
        path = tmp_path / "shrink.db"
        self._populated(path, sessions=60)
        attached(path)
        before = opencode_gc.on_disk_bytes(path)
        monkeypatch.setattr(opencode_gc, "db_holders", lambda db: [])

        rc, payload = _run_json_cli(
            monkeypatch, capsys, "--db", str(path), "--apply", "--batch-sleep-ms", "0"
        )

        assert rc == 0
        assert payload["pages_released"] > 0
        assert payload["bytes_reclaimed"] > 0, (
            "with nothing deferring the checkpoint the released pages must "
            "leave the disk"
        )
        assert opencode_gc.on_disk_bytes(path) < before
        assert not any("has not shrunk yet" in n for n in payload["notes"])

    def test_a_blocking_checkpoint_never_waits_out_the_busy_timeout(self, tmp_path):
        """checkpoint_mode_for can only act on a snapshot of who holds the
        database. A reader attaching between that snapshot and the call would
        otherwise turn a checkpoint chosen as safe into a stall as long as the
        connection's busy_timeout -- measured at 31.85s against a 30s timeout,
        versus 0.0s with the timeout suspended.

        Mutation: drop the `PRAGMA busy_timeout=0` / restore pair from
        checkpoint_wal. This then takes the full timeout below and fails.
        """
        path = tmp_path / "raced.db"
        conn = _make_live_db(path)
        for i in range(30):
            _add_live_session(conn, f"s{i}", age_days=30, events=40, messages=6)
        conn.close()

        # The reader takes its snapshot BEFORE the writer appends, so the
        # checkpoint genuinely cannot complete and must give up rather than
        # wait. A reader that attached afterwards would not block it at all,
        # and the test would pass on a stalling implementation.
        reader = sqlite3.connect(str(path), isolation_level=None)
        reader.execute("BEGIN")
        reader.execute("SELECT count(*) FROM event").fetchone()

        writer = opencode_gc.connect(path, read_only=False, timeout_s=30.0)
        try:
            writer.execute("BEGIN IMMEDIATE")
            writer.execute("DELETE FROM event WHERE aggregate_id='s0'")
            writer.execute("COMMIT")

            started = time.monotonic()
            outcome = opencode_gc.checkpoint_wal(writer, "TRUNCATE")
            elapsed = time.monotonic() - started

            assert outcome.busy is True, (
                "the reader holds an older snapshot, so this checkpoint cannot "
                "complete -- if it did, the fixture proves nothing"
            )
            assert elapsed < 5.0, (
                f"a blocked checkpoint must give up at once, took {elapsed:.1f}s"
            )
            # The timeout the rest of the tool relies on is restored.
            assert writer.execute("PRAGMA busy_timeout").fetchone()[0] == 30000
        finally:
            writer.close()
            reader.close()

    def test_a_busy_checkpoint_is_reported_not_hidden(
        self, tmp_path, monkeypatch, capsys
    ):
        """A checkpoint that could not fold the whole WAL is a fact the
        operator needs: the file is larger than the page counts imply."""
        path = tmp_path / "deferred.db"
        self._populated(path, sessions=30)

        reader = sqlite3.connect(str(path), isolation_level=None)
        reader.execute("BEGIN")
        reader.execute("SELECT count(*) FROM event").fetchone()
        try:
            # Claim the database is idle so a blocking mode is selected; the
            # reader then makes it genuinely busy.
            monkeypatch.setattr(opencode_gc, "db_holders", lambda db: [])
            rc, payload = _run_json_cli(
                monkeypatch, capsys, "--db", str(path), "--apply",
                "--batch-sleep-ms", "0",
            )

            assert payload["wal_checkpoint_mode"] == "TRUNCATE"
            assert payload["wal_checkpoint_busy"] is True
            assert any("could not fold the whole WAL" in n for n in payload["notes"])
            # Not an error: the frames fold on a later run.
            assert rc in (0, 3)
            assert payload["errors"] == []
        finally:
            reader.close()

    def test_our_connection_bounds_the_wal_it_may_leave_behind(self, tmp_path):
        """journal_size_limit is per-connection: it constrains our own
        transactions and does not touch opencode's connections.

        Mutation: drop the `PRAGMA journal_size_limit` from connect(). The
        read-write handle then reports -1, i.e. unlimited.
        """
        path = tmp_path / "jsl.db"
        conn = _make_live_db(path)
        _add_live_session(conn, "old", age_days=30)
        conn.close()

        ours = opencode_gc.connect(path, read_only=False)
        try:
            assert ours.execute("PRAGMA journal_size_limit").fetchone()[0] == \
                opencode_gc.JOURNAL_SIZE_LIMIT_BYTES
        finally:
            ours.close()

        # A connection opencode would make is unaffected.
        theirs = sqlite3.connect(str(path), isolation_level=None)
        try:
            assert theirs.execute("PRAGMA journal_size_limit").fetchone()[0] == -1
        finally:
            theirs.close()


class TestHolderDetectionFailsClosed:
    """`db_holders` returns None when it CANNOT determine who holds the file,
    which is deliberately distinct from `[]`, "determined: nobody".

    The rebuild destroys the file it swaps, so a missing lsof reading as idle
    would swap it out from under live writers. launchd and systemd start jobs
    with a bare environment and macOS keeps lsof in /usr/sbin, so this is the
    expected failure mode -- the existing com.marcioapm.agent-run-reap.plist
    already carries an explicit PATH for exactly this reason.
    """

    def _fake_lsof(self, monkeypatch, behaviour):
        calls = []

        def run(argv, **kwargs):
            calls.append(argv)
            return behaviour(argv, **kwargs)

        monkeypatch.setattr(opencode_gc.subprocess, "run", run)
        return calls

    def test_a_missing_lsof_is_unknown_not_empty(self, tmp_path, monkeypatch):
        """Mutation: in db_holders, `return None` -> `continue` (or return the
        pids gathered so far). A host without lsof then reports an idle
        database and every guard downstream opens up.
        """
        def missing(argv, **kwargs):
            raise FileNotFoundError(2, "No such file or directory", "lsof")

        self._fake_lsof(monkeypatch, missing)
        assert opencode_gc.db_holders(tmp_path / "any.db") is None

    def test_a_timed_out_lsof_is_unknown_not_empty(self, tmp_path, monkeypatch):
        def slow(argv, **kwargs):
            raise opencode_gc.subprocess.TimeoutExpired(argv, 30)

        self._fake_lsof(monkeypatch, slow)
        assert opencode_gc.db_holders(tmp_path / "any.db") is None

    def test_no_output_is_determined_nobody(self, tmp_path, monkeypatch):
        """The complement: an lsof that ran and found nothing must give the
        empty list, or the rebuild could never run at all."""
        def empty(argv, **kwargs):
            return subprocess.CompletedProcess(argv, 1, stdout="", stderr="")

        self._fake_lsof(monkeypatch, empty)
        assert opencode_gc.db_holders(tmp_path / "any.db") == []

    def test_our_own_pid_is_not_a_holder(self, tmp_path, monkeypatch):
        """We hold the database ourselves whenever we have it open, and
        counting that would make every rebuild skip forever."""
        me = str(os.getpid())

        def me_and_another(argv, **kwargs):
            return subprocess.CompletedProcess(
                argv, 0, stdout=f"{me}\n5150\n", stderr=""
            )

        self._fake_lsof(monkeypatch, me_and_another)
        assert opencode_gc.db_holders(tmp_path / "any.db") == ["5150"]

    def test_the_sidecars_are_examined_too(self, tmp_path, monkeypatch):
        """A process can hold the -wal or -shm without the main file, and it
        is just as much a writer that a swap would strand."""
        seen = []

        def per_path(argv, **kwargs):
            seen.append(argv[-1])
            pid = "777" if argv[-1].endswith("-wal") else ""
            return subprocess.CompletedProcess(argv, 0, stdout=pid, stderr="")

        self._fake_lsof(monkeypatch, per_path)
        db = tmp_path / "side.db"
        assert opencode_gc.db_holders(db) == ["777"]
        assert seen == [str(db), f"{db}-wal", f"{db}-shm"]

    def test_a_real_lsof_sees_a_real_open_handle(self, tmp_path):
        """No mocking: if lsof is genuinely available, an open connection must
        show up and a closed one must not. Otherwise every test above is
        asserting against a fake whose shape was never checked.
        """
        if shutil.which("lsof") is None:
            pytest.skip("lsof is not on PATH")
        path = tmp_path / "real.db"
        conn = _make_live_db(path)
        _add_live_session(conn, "old", age_days=30)

        held = opencode_gc.db_holders(path)
        conn.close()
        released = opencode_gc.db_holders(path)

        # Our own pid is excluded, and this process is the one holding it, so
        # the observable claim is that neither reading invents a stranger.
        assert held is not None and released is not None
        assert str(os.getpid()) not in held
        assert released == []


GIB = 1024 ** 3


class TestRebuildGuards:
    """`VACUUM INTO` is the only pass that shrinks an already-oversized file,
    and the only destructive one. The rebuild is safe; the SWAP is what loses
    data, so the guards around it are the whole safety argument.
    """

    def _store(self, path, *, sessions=40, wal=True):
        conn = _make_live_db(path, wal=wal)
        for i in range(sessions):
            _add_live_session(conn, f"s{i}", age_days=30, events=30, messages=5)
        conn.close()
        return path

    def _stats(self, path):
        conn = opencode_gc.connect(path, read_only=True)
        try:
            return opencode_gc.read_stats(conn)
        finally:
            conn.close()

    def _rebuild(self, path, monkeypatch, *, holders, max_seconds=60.0,
                 min_free=0, late=..., clock=time.monotonic):
        """Run a rebuild with a scripted holder sequence: `holders` for the
        pre-flight check, `late` for the pre-swap re-check."""
        answers = [holders] if late is ... else [holders, late]
        calls = []

        def fake(db):
            calls.append(db)
            return answers[min(len(calls) - 1, len(answers) - 1)]

        monkeypatch.setattr(opencode_gc, "db_holders", fake)
        outcome = opencode_gc.rebuild_database(
            path, self._stats(path), max_seconds=max_seconds,
            min_free_bytes=min_free, clock=clock,
        )
        return outcome, calls

    def test_an_idle_database_is_rebuilt_and_shrinks(self, tmp_path, monkeypatch):
        """The positive case, so every refusal below is a refusal of something
        that would otherwise have worked."""
        path = self._store(tmp_path / "rebuild.db")
        # Free the pages first, so there is genuinely something to compact.
        conn = opencode_gc.connect(path, read_only=False)
        opencode_gc.delete_sessions(
            conn, [f"s{i}" for i in range(30)], cutoff_ms=_cutoff(), batch=25,
            deadline=None,
        )
        conn.close()
        before = path.stat().st_size

        outcome, _ = self._rebuild(path, monkeypatch, holders=[], late=[])

        assert outcome.failure is None, outcome.failure
        assert outcome.skipped is None, outcome.skipped
        assert outcome.completed is True
        assert path.stat().st_size < before, "a rebuild must shrink the file"

        check = sqlite3.connect(str(path))
        try:
            assert check.execute("PRAGMA quick_check").fetchone()[0] == "ok"
            assert check.execute("PRAGMA auto_vacuum").fetchone()[0] == 2
            assert check.execute("SELECT count(*) FROM session").fetchone()[0] == 10
        finally:
            check.close()

    def test_a_held_database_is_not_rebuilt(self, tmp_path, monkeypatch):
        """Mutation: drop the `if holders:` branch from rebuild_database. The
        swap then runs while another process is writing to the old inode.
        """
        path = self._store(tmp_path / "held.db")
        before = path.read_bytes()

        outcome, _ = self._rebuild(path, monkeypatch, holders=["991", "992"])

        assert outcome.completed is False
        assert outcome.attempted is False
        assert "2 process(es) hold the database" in outcome.skipped
        assert outcome.failure is None
        assert path.read_bytes() == before, "the database must be untouched"

    def test_undeterminable_holders_refuse_the_rebuild(self, tmp_path, monkeypatch):
        """Mutation: `if holders is None:` -> `if False:`. `None` then falls
        through to `if holders:`, which is falsey, and a host whose lsof is
        missing rebuilds and swaps blind.
        """
        path = self._store(tmp_path / "blind.db")
        before = path.read_bytes()

        outcome, _ = self._rebuild(path, monkeypatch, holders=None)

        assert outcome.completed is False
        assert outcome.attempted is False
        assert "cannot determine who holds the database" in outcome.skipped
        assert path.read_bytes() == before

    def test_a_holder_appearing_during_the_rebuild_discards_the_copy(
        self, tmp_path, monkeypatch
    ):
        """The window this re-check exists to close. A process that opens the
        database while VACUUM INTO runs keeps writing to the OLD inode, and
        os.replace strands those writes on an unlinked inode.

        Mutation: delete the `late = db_holders(db)` block. The swap then
        happens regardless and the original inode is replaced.
        """
        path = self._store(tmp_path / "late.db")
        before = path.read_bytes()

        outcome, calls = self._rebuild(
            path, monkeypatch, holders=[], late=["8080"]
        )

        assert len(calls) == 2, "holders must be re-checked immediately before the swap"
        assert outcome.completed is False
        assert outcome.attempted is True, "the rebuild itself must have run"
        assert "1 process(es) attached while it ran" in outcome.skipped
        assert "strand their writes" in outcome.skipped
        assert path.read_bytes() == before, "the original file must survive intact"
        assert not path.with_name(path.name + opencode_gc.REBUILD_SUFFIX).exists(), \
            "the discarded copy must not be left on disk"

    def test_holders_becoming_undeterminable_mid_rebuild_discards_the_copy(
        self, tmp_path, monkeypatch
    ):
        path = self._store(tmp_path / "lateblind.db")
        before = path.read_bytes()

        outcome, _ = self._rebuild(path, monkeypatch, holders=[], late=None)

        assert outcome.completed is False
        assert "became undeterminable" in outcome.skipped
        assert path.read_bytes() == before
        assert not path.with_name(path.name + opencode_gc.REBUILD_SUFFIX).exists()

    def test_the_swap_replaces_the_inode_and_clears_the_sidecars(
        self, tmp_path, monkeypatch
    ):
        """The -wal and -shm describe the file being replaced. Left in place,
        the new database is opened against a WAL for the old one.

        Mutation: delete the sidecar unlink loop before os.replace. The stale
        -wal and -shm then survive the swap.

        The connection stays open ACROSS the rebuild on purpose: SQLite folds
        and deletes the WAL when the last connection closes, so a fixture that
        closed it first would have no sidecars left to clear and would pass
        against an implementation that clears nothing.
        """
        path = self._store(tmp_path / "sidecars.db")
        attached = sqlite3.connect(str(path), isolation_level=None)
        attached.execute("SELECT count(*) FROM session").fetchone()
        try:
            writer = sqlite3.connect(str(path), isolation_level=None)
            writer.execute("BEGIN")
            writer.execute("DELETE FROM event WHERE aggregate_id='s0'")
            writer.execute("COMMIT")
            writer.close()

            wal = path.with_name(path.name + "-wal")
            shm = path.with_name(path.name + "-shm")
            assert wal.exists() and wal.stat().st_size > 0, \
                "the fixture must really leave a WAL, or the assertion is vacuous"
            assert shm.exists(), "the fixture must really leave a -shm"
            old_inode = path.stat().st_ino

            outcome, _ = self._rebuild(path, monkeypatch, holders=[], late=[])

            assert outcome.completed is True
            assert path.stat().st_ino != old_inode, "the file must be replaced"
            assert not wal.exists(), \
                "a WAL describing the old inode must not survive the swap"
            assert not shm.exists()
        finally:
            attached.close()

    def test_a_corrupt_copy_is_never_swapped_in(self, tmp_path, monkeypatch):
        """Mutation: delete the `_verify_rebuilt` call. A copy that fails
        quick_check is then swapped over a healthy database.
        """
        path = self._store(tmp_path / "corrupt.db")
        before = path.read_bytes()
        monkeypatch.setattr(
            opencode_gc, "_verify_rebuilt",
            lambda target: "the rebuilt copy failed quick_check: wrecked",
        )

        outcome, _ = self._rebuild(path, monkeypatch, holders=[], late=[])

        assert outcome.completed is False
        assert "failed quick_check" in outcome.failure
        assert outcome.skipped is None, "a broken copy is a failure, not a skip"
        assert path.read_bytes() == before
        assert not path.with_name(path.name + opencode_gc.REBUILD_SUFFIX).exists()

    def test_a_copy_that_lost_incremental_vacuum_is_rejected(self, tmp_path):
        """auto_vacuum lives in the file header. A copy that came out NONE
        could never reclaim in place again, and only another full rebuild
        would restore it.

        Mutation: drop the `auto_vacuum != 2` branch from _verify_rebuilt.
        """
        plain = tmp_path / "plain.db"
        conn = sqlite3.connect(str(plain), isolation_level=None)
        conn.execute("CREATE TABLE t(a)")
        conn.execute("INSERT INTO t VALUES (1)")
        conn.close()
        assert sqlite3.connect(str(plain)).execute(
            "PRAGMA auto_vacuum"
        ).fetchone()[0] == 0, "the fixture must really be auto_vacuum=NONE"

        problem = opencode_gc._verify_rebuilt(plain)
        assert problem is not None
        assert "not INCREMENTAL" in problem

        # And a sound INCREMENTAL copy passes, or the check above would be
        # satisfied by a function that rejects everything.
        good = tmp_path / "good.db"
        conn = _make_live_db(good, wal=False)
        conn.close()
        assert opencode_gc._verify_rebuilt(good) is None


class TestRebuildIsBounded:
    """The 2026-09-12 incident: a `VACUUM INTO` on a 99 GB store ran 23 minutes
    in uninterruptible disk sleep, wrote a 39.6 GB temp copy and drove the disk
    from 88% to 93% before it was killed by hand. `--max-seconds` never applied
    to it, because VACUUM INTO is a single uninterruptible SQLite call.

    The only thing that can bound it is `set_progress_handler`: SQLite calls
    the handler every N virtual-machine instructions and a non-zero return
    aborts the statement.
    """

    def _big_store(self, path, sessions=200):
        """Large enough that the rebuild takes many progress callbacks, so an
        abort has somewhere to land rather than finishing first."""
        conn = _make_live_db(path, wal=False)
        conn.execute("BEGIN")
        for i in range(sessions):
            _add_live_session(conn, f"s{i}", age_days=30, events=60, messages=10)
        conn.execute("COMMIT")
        conn.close()
        return path

    def _stats(self, path):
        conn = opencode_gc.connect(path, read_only=True)
        try:
            return opencode_gc.read_stats(conn)
        finally:
            conn.close()

    def test_an_exception_from_a_progress_handler_cannot_be_caught(self, tmp_path):
        """The reason the abort reason is recorded in a cell and 1 returned,
        rather than raised.

        Verified on CPython 3.14: an exception raised inside a progress handler
        is DISCARDED, and the statement resurfaces as a generic
        `sqlite3.OperationalError: interrupted`. An `except MyAbortError`
        clause is dead code that can never match, so a deliberate abort would
        be misreported as a failure and spam a 5-minute timer with false
        alarms. This test pins that platform behaviour: if a future CPython
        propagates the original exception, the design can be simplified -- and
        until then it must not be.
        """
        path = tmp_path / "handler.db"
        conn = sqlite3.connect(str(path), isolation_level=None)
        conn.execute("CREATE TABLE t(a)")
        conn.execute("BEGIN")
        conn.executemany("INSERT INTO t VALUES (?)", [(i,) for i in range(20000)])
        conn.execute("COMMIT")

        class Deliberate(Exception):
            pass

        state = {"raised": False}

        def handler():
            if not state["raised"]:
                state["raised"] = True
                raise Deliberate("abort")
            return 0

        conn.set_progress_handler(handler, 5)
        try:
            with pytest.raises(sqlite3.OperationalError, match="interrupted"):
                conn.execute("SELECT sum(a * a) FROM t WHERE a % 3 = 0").fetchone()
        finally:
            conn.set_progress_handler(None, 0)
            conn.close()
        assert state["raised"] is True, "the handler must actually have raised"

    def test_the_wall_clock_cap_aborts_the_rebuild(self, tmp_path, monkeypatch):
        """Mutation: delete the `conn.set_progress_handler(guard, ...)` line.
        The rebuild then runs to completion however long it takes, which is
        precisely the 23-minute runaway.
        """
        path = self._big_store(tmp_path / "slow.db")
        monkeypatch.setattr(opencode_gc, "db_holders", lambda db: [])

        # A clock that leaps past the cap on its third reading: the deadline is
        # computed from the first, and the guard trips on a later one.
        ticks = iter([0.0, 0.0, 0.0, 10_000.0] + [10_000.0] * 100_000)
        outcome = opencode_gc.rebuild_database(
            path, self._stats(path), max_seconds=5.0, min_free_bytes=0,
            clock=lambda: next(ticks),
        )

        assert outcome.completed is False
        assert outcome.failure is None, (
            "a guard-triggered abort is a deliberate skip, not a failure: "
            f"{outcome.failure}"
        )
        assert outcome.skipped is not None
        assert "wall-clock cap" in outcome.skipped

    def test_an_aborted_rebuild_leaves_no_orphan_copy(self, tmp_path, monkeypatch):
        """SQLite leaves the partial copy behind on an aborted VACUUM INTO
        (verified: the output file exists after `interrupted`), so without the
        unlink a 39 GB orphan accumulates on every abort -- every 5 minutes, on
        the disk the abort fired to protect.

        Mutation: delete the `_unlink_quietly(target)` from the
        `except (sqlite3.Error, OSError)` handler in rebuild_database.
        """
        path = self._big_store(tmp_path / "orphan.db")
        monkeypatch.setattr(opencode_gc, "db_holders", lambda db: [])
        target = path.with_name(path.name + opencode_gc.REBUILD_SUFFIX)

        ticks = iter([0.0, 0.0, 0.0, 10_000.0] + [10_000.0] * 100_000)
        outcome = opencode_gc.rebuild_database(
            path, self._stats(path), max_seconds=5.0, min_free_bytes=0,
            clock=lambda: next(ticks),
        )

        assert outcome.skipped is not None and "wall-clock cap" in outcome.skipped
        assert not target.exists(), (
            "SQLite leaves the partial copy behind; it must be unlinked or it "
            "accumulates once per abort"
        )

    def test_the_disk_floor_aborts_a_rebuild_that_is_eating_the_disk(
        self, tmp_path, monkeypatch
    ):
        """The other half of the incident: the temp copy drove the disk from
        88% to 93%. The floor is re-checked from inside the statement, not just
        before it, because the space disappears while it runs.

        Mutation: delete the `free < min_free_bytes` branch from _rebuild_guard.
        The rebuild then runs on regardless of how little disk is left.
        """
        path = self._big_store(tmp_path / "floor.db")
        monkeypatch.setattr(opencode_gc, "db_holders", lambda db: [])

        # Roomy for the pre-flight check, then the disk "fills" underneath it.
        readings = iter([opencode_gc.shutil.disk_usage(tmp_path).free] + [1] * 100_000)
        monkeypatch.setattr(
            opencode_gc.shutil, "disk_usage", lambda p: _Usage(next(readings))
        )

        outcome = opencode_gc.rebuild_database(
            path, self._stats(path), max_seconds=600.0, min_free_bytes=10 * GIB,
        )

        assert outcome.completed is False
        assert outcome.failure is None
        assert "free space fell to" in outcome.skipped
        assert not path.with_name(path.name + opencode_gc.REBUILD_SUFFIX).exists()

    def test_the_preflight_demands_room_for_the_copy_AND_the_floor(
        self, tmp_path, monkeypatch
    ):
        """`free - needed >= floor`, not `free >= needed`: a rebuild that fits
        exactly still drives the filesystem to the edge while it runs.

        Mutation: `if free - needed < min_free_bytes:` -> `if free < needed:`.
        The 'fits, but eats the floor' case below is then accepted.
        """
        path = self._big_store(tmp_path / "preflight.db", sessions=40)
        monkeypatch.setattr(opencode_gc, "db_holders", lambda db: [])
        stats = self._stats(path)
        needed = int(opencode_gc.live_bytes(stats) * opencode_gc.REBUILD_COPY_HEADROOM)
        floor = 10 * GIB

        # Enough for the copy, but not enough to leave the floor intact.
        monkeypatch.setattr(
            opencode_gc.shutil, "disk_usage", lambda p: _Usage(needed + floor - 1)
        )
        refused = opencode_gc.rebuild_database(
            path, stats, max_seconds=600.0, min_free_bytes=floor
        )
        assert refused.completed is False
        assert refused.attempted is False
        assert "must leave" in refused.skipped

        # One byte more and it proceeds, so the refusal above is the boundary
        # and not a blanket rejection.
        monkeypatch.setattr(
            opencode_gc.shutil, "disk_usage", lambda p: _Usage(needed + floor)
        )
        allowed = opencode_gc.rebuild_database(
            path, stats, max_seconds=600.0, min_free_bytes=floor
        )
        assert allowed.completed is True, allowed.skipped or allowed.failure

    def test_a_guard_abort_is_a_skip_and_exits_zero(
        self, tmp_path, monkeypatch, capsys
    ):
        """A guard that fired did its job: the database is untouched and a
        later run may succeed. Reporting it as an error would page someone
        every five minutes for a working safety valve.

        Mutation: in main, `res.errors.append(...)` for `rb.skipped` instead of
        `res.notes.append(...)`. The exit status then becomes 1.
        """
        path = self._big_store(tmp_path / "skipzero.db", sessions=40)
        monkeypatch.setattr(opencode_gc, "db_holders", lambda db: ["31337"])

        rc, payload = _run_json_cli(
            monkeypatch, capsys, "--db", str(path), "--apply", "--rebuild",
            "--batch-sleep-ms", "0", "--retention-days", "9999",
        )

        assert rc == 0, "a skipped rebuild is not an error"
        assert payload["errors"] == []
        assert payload["rebuild_completed"] is False
        assert "hold the database" in payload["rebuild_skipped"]
        assert any("--rebuild did not run" in n for n in payload["notes"])


class _Usage:
    """shutil.disk_usage's shape, for tests that script free space."""

    def __init__(self, free):
        self.total = free * 4
        self.used = self.total - free
        self.free = free


class TestDeletionYieldsTheWriteLock:
    """WAL lets readers run concurrently, but opencode instances are WRITERS
    and writers serialise: a concurrent writer that exhausts its own
    busy_timeout waiting on our transaction fails with "Failed to execute
    statement". Smaller batches with a pause between them give it a window.
    """

    def _store(self, path, sessions=10):
        conn = _make_live_db(path)
        for i in range(sessions):
            _add_live_session(conn, f"s{i}", age_days=30)
        conn.close()
        return [f"s{i}" for i in range(sessions)]

    def test_the_pause_happens_between_batches_and_not_inside_one(self, tmp_path):
        """A sleep inside the transaction would hold the write lock for longer,
        which is the opposite of the intent.

        Mutation: move the `sleep(...)` call above `_commit_batch(conn)`. The
        recorded in_transaction flags below then become True.
        """
        path = tmp_path / "yield.db"
        ids = self._store(path, sessions=10)
        conn = opencode_gc.connect(path, read_only=False)
        naps = []
        try:
            outcome = opencode_gc.delete_sessions(
                conn, ids, cutoff_ms=_cutoff(), batch=3, deadline=None,
                sleep_ms=25, sleep=lambda s: naps.append((s, conn.in_transaction)),
            )
            assert outcome.rows["session"] == 10
            assert len(naps) == 3, (
                "four batches of three: a pause after each but the last, "
                f"got {len(naps)}"
            )
            assert all(seconds == 0.025 for seconds, _ in naps)
            assert not any(in_txn for _, in_txn in naps), (
                "the pause must not happen while the write lock is held"
            )
        finally:
            conn.close()

    def test_no_pause_is_taken_after_the_final_batch(self, tmp_path):
        """Sleeping once the work is done just delays the report."""
        path = tmp_path / "last.db"
        ids = self._store(path, sessions=4)
        conn = opencode_gc.connect(path, read_only=False)
        naps = []
        try:
            opencode_gc.delete_sessions(
                conn, ids, cutoff_ms=_cutoff(), batch=4, deadline=None,
                sleep_ms=50, sleep=naps.append,
            )
            assert naps == [], "a single batch clears everything; nothing to yield to"
        finally:
            conn.close()

    def test_the_pause_can_be_disabled(self, tmp_path):
        """Mutation: `if sleep_ms:` -> `if True:`. sleep(0.0) is then called
        and this fails."""
        path = tmp_path / "nosleep.db"
        ids = self._store(path, sessions=6)
        conn = opencode_gc.connect(path, read_only=False)
        naps = []
        try:
            outcome = opencode_gc.delete_sessions(
                conn, ids, cutoff_ms=_cutoff(), batch=2, deadline=None,
                sleep_ms=0, sleep=naps.append,
            )
            assert outcome.rows["session"] == 6
            assert naps == []
        finally:
            conn.close()

    def test_a_checkpoint_runs_between_batches(self, tmp_path):
        """Each batch's WAL frames are folded back as it goes, rather than
        accumulating until the end.

        Mutation: delete the `checkpoint_wal(conn, "PASSIVE")` call from the
        loop. `checkpoints` is then 0.
        """
        path = tmp_path / "ckpt.db"
        conn = _make_live_db(path)
        for i in range(9):
            _add_live_session(conn, f"s{i}", age_days=30, events=40, messages=6)
        conn.close()
        # Keeps SQLite from folding the WAL for us behind the test's back.
        attached = sqlite3.connect(str(path), isolation_level=None)
        attached.execute("SELECT count(*) FROM session").fetchone()

        gc_conn = opencode_gc.connect(path, read_only=False)
        try:
            outcome = opencode_gc.delete_sessions(
                gc_conn, [f"s{i}" for i in range(9)], cutoff_ms=_cutoff(),
                batch=3, deadline=None, sleep_ms=0,
            )
            assert outcome.rows["session"] == 9
            assert outcome.checkpoints == 2, (
                "three batches means a checkpoint after the first two, "
                f"got {outcome.checkpoints}"
            )
            assert outcome.pages_checkpointed > 0, (
                "the checkpoints must have folded real WAL frames"
            )
        finally:
            gc_conn.close()
            attached.close()

    def test_the_between_batch_checkpoint_is_never_a_blocking_mode(self, tmp_path):
        """TRUNCATE between every chunk is what turned the standalone script
        into the most contended thing on the box, and killed an opencode run
        with "Failed to execute statement".

        Mutation: `checkpoint_wal(conn, "PASSIVE")` -> `..., "TRUNCATE")` in
        the deletion loop.
        """
        path = tmp_path / "mode.db"
        ids = self._store(path, sessions=6)
        conn = opencode_gc.connect(path, read_only=False)
        modes = []
        real = opencode_gc.checkpoint_wal
        try:
            opencode_gc.checkpoint_wal = lambda c, mode: (
                modes.append(mode) or real(c, mode)
            )
            opencode_gc.delete_sessions(
                conn, ids, cutoff_ms=_cutoff(), batch=2, deadline=None, sleep_ms=0,
            )
        finally:
            opencode_gc.checkpoint_wal = real
            conn.close()

        assert modes, "the loop must checkpoint between batches"
        assert set(modes) == {"PASSIVE"}, (
            f"only PASSIVE may run between batches, saw {sorted(set(modes))}"
        )

    def test_a_concurrent_writer_wins_the_lock_during_the_pause(self, tmp_path):
        """The behaviour the pause exists for, with a writer that is genuinely
        contending: it has a short busy_timeout and would fail if our loop
        never let go.

        Both processes are live for the whole exchange -- a "concurrency" test
        whose other party has already exited proves nothing.

        Mutation: `sleep_ms=0` in the call below, i.e. no pause. The writer
        below then has no window and its INSERT raises SQLITE_BUSY.
        """
        path = tmp_path / "contend.db"
        ids = self._store(path, sessions=12)

        rival = sqlite3.connect(str(path), isolation_level=None, timeout=0)
        rival.execute("PRAGMA busy_timeout=0")
        landed = []
        failures = []

        def rival_write(_seconds):
            # Called from inside the pause, i.e. exactly when the lock is free.
            try:
                rival.execute("BEGIN IMMEDIATE")
                rival.execute(
                    "INSERT INTO session (id, parent_id, time_created, "
                    f"time_updated) VALUES ('r{len(landed)}',NULL,1,1)"
                )
                rival.execute("COMMIT")
                landed.append(1)
            except sqlite3.OperationalError as exc:
                failures.append(str(exc))

        conn = opencode_gc.connect(path, read_only=False)
        try:
            outcome = opencode_gc.delete_sessions(
                conn, ids, cutoff_ms=_cutoff(), batch=3, deadline=None,
                sleep_ms=1, sleep=rival_write,
            )
        finally:
            conn.close()

        assert outcome.rows["session"] == 12, "our own work must still finish"
        assert failures == [], f"the rival writer must not be starved: {failures}"
        assert len(landed) == 3, "the rival must have taken the lock in each pause"
        try:
            assert rival.execute(
                "SELECT count(*) FROM session WHERE id LIKE 'r%'"
            ).fetchone()[0] == 3
        finally:
            rival.close()


class TestNewFlagsAndDefaults:
    """The existing surface is additive-only, and the new flags have to be
    reachable and validated."""

    @pytest.fixture()
    def anydb(self, tmp_path):
        path = tmp_path / "flags.db"
        conn = _make_live_db(path)
        _add_live_session(conn, "old", age_days=30)
        conn.close()
        return path

    def test_help_lists_the_new_flags_and_keeps_the_old_ones(self, monkeypatch, capsys):
        monkeypatch.setattr(opencode_gc.sys, "argv", ["opencode-gc", "--help"])
        with pytest.raises(SystemExit) as exit_info:
            opencode_gc.main()
        assert exit_info.value.code == 0

        out = capsys.readouterr().out
        for flag in ("--rebuild", "--rebuild-max-seconds", "--rebuild-min-free-gib",
                     "--batch-sleep-ms"):
            assert flag in out, f"{flag} must be documented"
        for flag in ("--retention-days", "--apply", "--batch", "--max-seconds",
                     "--vacuum-pages", "--no-vacuum", "--enable-incremental-vacuum",
                     "--json", "--db"):
            assert flag in out, f"the existing flag {flag} must still be offered"

    def test_retention_defaults_to_four_days(self, anydb, monkeypatch, capsys):
        """Márcio's call; the cutoff is what actually decides, so it is what is
        asserted rather than the parser's default string."""
        rc, payload = _run_json_cli(monkeypatch, capsys, "--db", str(anydb))
        assert rc == 0
        assert payload["retention_days"] == 4.0
        expected = (time.time() - 4 * 86400) * 1000
        assert abs(payload["cutoff_ms"] - expected) < 5000

    def test_the_retention_floor_still_applies(self, anydb, monkeypatch, capsys):
        monkeypatch.setattr(
            opencode_gc.sys, "argv",
            ["opencode-gc", "--db", str(anydb), "--retention-days", "0.5"],
        )
        with pytest.raises(SystemExit) as exit_info:
            opencode_gc.main()
        assert exit_info.value.code == 2
        assert "--retention-days" in capsys.readouterr().err

    def test_the_batch_default_is_the_gentler_one(self, anydb, monkeypatch, capsys):
        seen = {}
        real = opencode_gc.delete_sessions

        def spy(conn, ids, **kw):
            seen.update(kw)
            return real(conn, ids, **kw)

        monkeypatch.setattr(opencode_gc, "delete_sessions", spy)
        _run_json_cli(monkeypatch, capsys, "--db", str(anydb), "--apply")

        assert seen["batch"] == opencode_gc.DEFAULT_BATCH == 25
        assert seen["sleep_ms"] == opencode_gc.DEFAULT_BATCH_SLEEP_MS == 1000

    @pytest.mark.parametrize(
        "flag, value",
        [
            ("--batch-sleep-ms", "-1"),
            ("--rebuild-max-seconds", "0"),      # unbounded is the incident
            ("--rebuild-max-seconds", "-5"),
            ("--rebuild-max-seconds", "nan"),
            ("--rebuild-min-free-gib", "-1"),
            ("--rebuild-min-free-gib", "nan"),
        ],
    )
    def test_invalid_new_bounds_are_refused_without_touching_the_db(
        self, anydb, monkeypatch, capsys, flag, value
    ):
        """Mutation: drop the `--rebuild-max-seconds must be finite and > 0`
        check. `0` is then accepted and the guard's deadline is `now`, or
        worse, never trips.
        """
        before = _snapshot(anydb, tables=("session",))
        monkeypatch.setattr(
            opencode_gc.sys, "argv",
            ["opencode-gc", "--db", str(anydb), "--apply", flag, value],
        )
        with pytest.raises(SystemExit) as exit_info:
            opencode_gc.main()
        assert exit_info.value.code == 2
        assert flag in capsys.readouterr().err
        assert _snapshot(anydb, tables=("session",)) == before

    def test_rebuild_needs_apply(self, anydb, monkeypatch, capsys):
        """A dry run must never rewrite the file, whatever else is asked for."""
        before = anydb.read_bytes()
        rc, payload = _run_json_cli(
            monkeypatch, capsys, "--db", str(anydb), "--rebuild"
        )
        assert rc == 0
        assert payload["rebuild_attempted"] is False
        assert any("--rebuild needs --apply" in n for n in payload["notes"])
        assert anydb.read_bytes() == before

    def test_a_dry_run_asks_nothing_of_lsof(self, anydb, monkeypatch, capsys):
        """A dry run holds a read-only handle and neither checkpoints nor
        rebuilds, so spawning lsof for it is pure cost."""
        calls = []
        monkeypatch.setattr(
            opencode_gc, "db_holders", lambda db: calls.append(db) or []
        )
        rc, payload = _run_json_cli(monkeypatch, capsys, "--db", str(anydb))
        assert rc == 0
        assert calls == []
        assert payload["holders_before"] is None


class TestRebuildEndToEnd:
    """The CLI path: prune, reclaim, checkpoint, then rebuild -- which is the
    order that makes a store too big to rebuild before a prune small enough to
    rebuild after one. macmini pre-prune needed ~64 GiB for a plain VACUUM
    against 14 GiB free and was correctly refused; post-prune live is 6.2 GiB,
    so VACUUM INTO needs ~6.5 GiB against 13 GiB. It fits.
    """

    def test_prune_then_rebuild_actually_shrinks_the_file(
        self, tmp_path, monkeypatch, capsys
    ):
        """Mutation: drop the `--rebuild` handling from main. The file then
        stays at its pre-prune size, because incremental reclamation alone
        cannot return a freelist this large in one pass.
        """
        path = tmp_path / "e2e.db"
        conn = _make_live_db(path)
        conn.execute("BEGIN")
        for i in range(150):
            _add_live_session(conn, f"s{i}", age_days=30, events=60, messages=8)
        for i in range(5):
            _add_live_session(conn, f"live{i}", age_days=0.5, events=10, messages=2)
        conn.execute("COMMIT")
        conn.close()
        before = path.stat().st_size

        monkeypatch.setattr(opencode_gc, "db_holders", lambda db: [])
        rc, payload = _run_json_cli(
            monkeypatch, capsys, "--db", str(path), "--apply", "--rebuild",
            "--batch-sleep-ms", "0", "--rebuild-min-free-gib", "0",
        )

        assert rc == 0, payload["errors"]
        assert payload["sessions_deleted"] == 150
        assert payload["rebuild_completed"] is True
        assert payload["rebuild_skipped"] is None

        after = path.stat().st_size
        assert after < before, f"the file must shrink: {before} -> {after}"
        assert payload["bytes_after"] < payload["bytes_before"]

        # The survivors and their children are all still there and intact.
        check = sqlite3.connect(str(path))
        try:
            assert check.execute("PRAGMA quick_check").fetchone()[0] == "ok"
            assert check.execute("PRAGMA auto_vacuum").fetchone()[0] == 2
            assert [r[0] for r in check.execute(
                "SELECT id FROM session ORDER BY id"
            )] == [f"live{i}" for i in range(5)]
            assert check.execute("SELECT count(*) FROM todo").fetchone()[0] == 10
            assert check.execute("SELECT count(*) FROM event").fetchone()[0] == 50
        finally:
            check.close()

    def test_an_existing_run_without_rebuild_is_unchanged(
        self, tmp_path, monkeypatch, capsys
    ):
        """The whole change is additive: without --rebuild nothing rewrites."""
        path = tmp_path / "norebuild.db"
        conn = _make_live_db(path)
        for i in range(20):
            _add_live_session(conn, f"s{i}", age_days=30)
        conn.close()
        inode = path.stat().st_ino

        monkeypatch.setattr(opencode_gc, "db_holders", lambda db: [])
        rc, payload = _run_json_cli(
            monkeypatch, capsys, "--db", str(path), "--apply", "--batch-sleep-ms", "0"
        )

        assert rc == 0
        assert payload["sessions_deleted"] == 20
        assert payload["rebuild_attempted"] is False
        assert payload["rebuild_completed"] is False
        assert path.stat().st_ino == inode, "the file must not be replaced"

    def test_a_rebuild_is_skipped_on_a_non_incremental_store(
        self, tmp_path, monkeypatch, capsys
    ):
        """A copy inherits the source's auto_vacuum, so rebuilding a NONE store
        would produce a NONE store that still cannot reclaim in place."""
        path = tmp_path / "none.db"
        conn = sqlite3.connect(str(path), isolation_level=None)
        conn.executescript(LIVE_SCHEMA)
        assert conn.execute("PRAGMA auto_vacuum").fetchone()[0] == 0
        _add_live_session(conn, "old", age_days=30)
        conn.close()

        monkeypatch.setattr(opencode_gc, "db_holders", lambda db: [])
        rc, payload = _run_json_cli(
            monkeypatch, capsys, "--db", str(path), "--apply", "--rebuild",
            "--batch-sleep-ms", "0",
        )

        assert rc == 0
        assert payload["rebuild_completed"] is False
        assert "auto_vacuum is NONE" in payload["rebuild_skipped"]
        assert payload["errors"] == []


class TestTodoOrphanProvenance:
    """Which mechanism strands `todo` rows, and which does not.

    A live host showed `todo` = 551 orphans while `event`, `event_sequence`,
    `message`, `part` and `session_message` were all 0. The two candidate
    causes behave differently, and the difference is testable:

      * a delete run with `PRAGMA foreign_keys=ON` that touches only `session`
        (and the event tables) cascades, and takes `todo` with it;
      * a delete run with foreign keys OFF -- SQLite's default, and what
        opencode's own session deletes use -- strands every cascade child.

    `message.session_id` and `todo.session_id` carry the SAME
    `ON DELETE CASCADE` constraint, so any mechanism that orphaned one would
    have orphaned the other. The host's `message` orphan count was 0, which
    places the `todo` orphans before the FK-enabled prune rather than in it.
    """

    def test_a_cascade_prune_does_not_orphan_todo_rows(self, tmp_path):
        path = tmp_path / "cascade.db"
        conn = _make_live_db(path, wal=False)
        _add_live_session(conn, "old", age_days=30)
        conn.close()

        pruner = sqlite3.connect(str(path), isolation_level=None)
        try:
            pruner.execute("PRAGMA foreign_keys = ON")
            assert pruner.execute("PRAGMA foreign_keys").fetchone()[0] == 1, \
                "the fixture must really have cascades enabled"
            assert pruner.execute("SELECT count(*) FROM todo").fetchone()[0] == 2

            pruner.execute("BEGIN IMMEDIATE")
            pruner.execute("DELETE FROM event WHERE aggregate_id='old'")
            pruner.execute("DELETE FROM event_sequence WHERE aggregate_id='old'")
            pruner.execute("DELETE FROM session WHERE id='old'")
            pruner.execute("COMMIT")

            for table in ("todo", "message", "session_message", "session_input",
                          "session_share", "session_context_epoch"):
                assert pruner.execute(
                    f"SELECT count(*) FROM {table}"
                ).fetchone()[0] == 0, f"the cascade must have taken {table}"
        finally:
            pruner.close()

    def test_a_session_delete_with_foreign_keys_off_orphans_every_child(
        self, tmp_path
    ):
        """SQLite's default, and what opencode's own session deletes run under.
        This is the mechanism that produces the orphans, and the reason this
        module deletes every table explicitly instead of trusting cascade."""
        path = tmp_path / "nofk.db"
        conn = _make_live_db(path, wal=False)
        _add_live_session(conn, "old", age_days=30)
        conn.close()

        opencode_like = sqlite3.connect(str(path), isolation_level=None)
        try:
            assert opencode_like.execute("PRAGMA foreign_keys").fetchone()[0] == 0
            opencode_like.execute("DELETE FROM session WHERE id='old'")

            assert opencode_like.execute(
                "SELECT count(*) FROM todo"
            ).fetchone()[0] == 2, "with cascades off the todo rows are stranded"
            assert opencode_like.execute(
                "SELECT count(*) FROM message"
            ).fetchone()[0] == 2, (
                "message carries the same ON DELETE CASCADE as todo, so any "
                "mechanism that orphans one orphans the other"
            )
        finally:
            opencode_like.close()

    def test_this_tool_orphans_neither_whatever_the_pragma_says(self, tmp_path):
        """The explicit deletes do not depend on the pragma being set either
        way, which is the property that makes them correct under SQLite's
        default."""
        path = tmp_path / "explicit.db"
        conn = _make_live_db(path, wal=False)
        try:
            _add_live_session(conn, "old", age_days=30)
            assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 0

            opencode_gc.delete_sessions(
                conn, ["old"], cutoff_ms=_cutoff(), batch=25, deadline=None,
            )

            assert _live_counts(conn) == {
                t: 0 for t in ("session",) + LIVE_CHILD_TABLES
            }
        finally:
            conn.close()


class TestRebuildFailureIsReportedNotRaised:
    """The rebuild runs after the deletes are already committed, so nothing it
    does may escape as a traceback: an operator deciding what to restore needs
    the counts far more than a stack trace.
    """

    def _store(self, path, sessions=30):
        conn = _make_live_db(path)
        for i in range(sessions):
            _add_live_session(conn, f"s{i}", age_days=30)
        conn.close()
        return path

    def test_an_interrupt_during_the_rebuild_still_reports_the_deletions(
        self, tmp_path, monkeypatch, capsys
    ):
        """Ctrl-C is how an operator stops a long rebuild, and by then the
        prune has already committed.

        Mutation: drop the `except (KeyboardInterrupt, OSError)` around
        `_run_rebuild` in main. The interrupt then escapes and the committed
        deletions are reported nowhere.
        """
        path = self._store(tmp_path / "interrupt.db")

        def interrupting(db, stats, **kw):
            raise KeyboardInterrupt()

        monkeypatch.setattr(opencode_gc, "db_holders", lambda db: [])
        monkeypatch.setattr(opencode_gc, "rebuild_database", interrupting)

        rc, payload = _run_json_cli(
            monkeypatch, capsys, "--db", str(path), "--apply", "--rebuild",
            "--batch-sleep-ms", "0",
        )

        assert rc == 1
        assert payload["sessions_deleted"] == 30, \
            "the committed deletions must still be reported"
        assert any("KeyboardInterrupt" in e for e in payload["errors"])
        assert len(_snapshot(path, tables=("session",))["session"]) == 0

    def test_an_interrupt_inside_vacuum_into_removes_the_partial_copy(
        self, tmp_path, monkeypatch
    ):
        """A BaseException is not a guard abort and must propagate, but the
        partial copy still has to go: SQLite leaves it behind either way.

        Mutation: delete the `except BaseException: _unlink_quietly(target);
        raise` clause. The partial copy then survives the interrupt.
        """
        path = self._store(tmp_path / "ctrlc.db")
        target = path.with_name(path.name + opencode_gc.REBUILD_SUFFIX)
        monkeypatch.setattr(opencode_gc, "db_holders", lambda db: [])

        real_connect = opencode_gc.connect

        class InterruptsTheVacuum(sqlite3.Connection):
            def execute(self, sql, *a):
                if sql.startswith("VACUUM INTO"):
                    # Leave a partial copy behind exactly as SQLite does on an
                    # aborted VACUUM INTO, then interrupt.
                    target.write_bytes(b"partial")
                    raise KeyboardInterrupt()
                return super().execute(sql, *a)

        def interrupting_connect(db, *, read_only, **kw):
            if read_only:
                return real_connect(db, read_only=read_only, **kw)
            return sqlite3.connect(
                str(Path(db).resolve()), isolation_level=None,
                factory=InterruptsTheVacuum,
            )

        conn = real_connect(path, read_only=True)
        try:
            stats = opencode_gc.read_stats(conn)
        finally:
            conn.close()

        monkeypatch.setattr(opencode_gc, "connect", interrupting_connect)
        with pytest.raises(KeyboardInterrupt):
            opencode_gc.rebuild_database(
                path, stats, max_seconds=60.0, min_free_bytes=0
            )

        assert not target.exists(), \
            "the partial copy must not survive an interrupt"
