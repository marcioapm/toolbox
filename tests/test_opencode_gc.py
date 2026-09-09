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

import os
import sqlite3
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


def _snapshot(path):
    """Every row of every table, so a test can assert nothing at all changed."""
    conn = sqlite3.connect(path)
    try:
        return {t: sorted(conn.execute(f"SELECT * FROM {t}").fetchall()) for t in TABLES}
    finally:
        conn.close()


def _snapshot_of(path, tables):
    """Every row of the named tables, for databases whose shape is not the
    standard fixture's."""
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

        dangling = conn.execute(
            "SELECT s.id FROM session s WHERE s.parent_id IS NOT NULL "
            "AND s.parent_id NOT IN (SELECT id FROM session)"
        ).fetchall()
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
        dangling = conn.execute(
            "SELECT id FROM session WHERE parent_id IS NOT NULL "
            "AND parent_id NOT IN (SELECT id FROM session)"
        ).fetchall()
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
        dangling = conn.execute(
            "SELECT id FROM session WHERE parent_id IS NOT NULL "
            "AND parent_id NOT IN (SELECT id FROM session)"
        ).fetchall()
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
                dangling = probe.execute(
                    "SELECT id FROM session WHERE parent_id IS NOT NULL "
                    "AND parent_id NOT IN (SELECT id FROM session)"
                ).fetchall()
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

    def _dangling(self, conn):
        return conn.execute(
            "SELECT id FROM session WHERE parent_id IS NOT NULL "
            "AND parent_id NOT IN (SELECT id FROM session)"
        ).fetchall()

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

        assert self._dangling(app) == [], "'kid' must not outlive its parent"
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
        assert self._dangling(app) == []
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

        Mutation: in run_incremental_vacuum, `target = before if pages is None
        else min(pages, before)` -> `target = before`. The old assertion
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
        `if vac.deadline_reached:`. rc falls back to 0, so automation is told
        a run that left the file oversized finished cleanly.
        """
        import json

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
        monkeypatch.setattr(
            opencode_gc.sys, "argv",
            ["opencode-gc", "--db", str(path), "--apply", "--json"],
        )
        rc = opencode_gc.main()
        payload = json.loads(capsys.readouterr().out)

        assert rc == 3, "an unfinished reclamation must not share the clean-run status"
        assert payload["incomplete"] is True
        assert payload["vacuum_deadline_reached"] is True
        assert payload["pages_reclaimable_remaining"] > 0
        assert payload["errors"] == [], "a deadline is not an error"
        # The deletions themselves did finish; only the reclamation did not.
        assert payload["sessions_deleted"] == 60
        assert len(_snapshot(path)["session"]) == 0


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
        import json

        path = tmp_path / "busycli.db"
        setup = _make_db(path, wal=True)
        for i in range(6):
            _add_session(setup, f"s{i}", age_days=30)
        setup.close()

        blocker = sqlite3.connect(str(path), isolation_level=None, timeout=0)
        gc_conn = self._locking_conn(path, blocker, before_batch=2)
        monkeypatch.setattr(opencode_gc, "connect", lambda *a, **kw: gc_conn)
        monkeypatch.setattr(
            opencode_gc.sys, "argv",
            ["opencode-gc", "--db", str(path), "--apply", "--batch", "1", "--json"],
        )
        try:
            rc = opencode_gc.main()
            payload = json.loads(capsys.readouterr().out)

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
        import json

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
        monkeypatch.setattr(
            opencode_gc.sys, "argv",
            ["opencode-gc", "--db", str(path), "--apply", "--batch", "1", "--json"],
        )
        rc = opencode_gc.main()
        payload = json.loads(capsys.readouterr().out)

        assert rc == 3, "a truncated run must not share the complete-run status"
        assert payload["incomplete"] is True
        assert payload["deadline_reached"] is True
        assert payload["sessions_deleted"] == 2
        assert payload["sessions_remaining"] == 3
        assert len(_snapshot(path)["session"]) == 3


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
        import json

        self._run(monkeypatch, ["--db", str(populated), "--json"])
        payload = json.loads(capsys.readouterr().out)
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
        import json

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
            monkeypatch.setattr(
                opencode_gc.sys, "argv", ["opencode-gc", "--db", str(path), "--json"]
            )
            assert opencode_gc.main() == 0
            payload = json.loads(capsys.readouterr().out)

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
        import json

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
            monkeypatch.setattr(
                opencode_gc.sys, "argv",
                ["opencode-gc", "--db", str(path), "--apply", "--json"],
            )
            assert opencode_gc.main() == 0
            payload = json.loads(capsys.readouterr().out)

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
        import json

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
            monkeypatch.setattr(
                opencode_gc.sys, "argv", ["opencode-gc", "--db", str(alias), "--json"]
            )
            assert opencode_gc.main() == 0
            payload = json.loads(capsys.readouterr().out)

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

    def test_documented_order_is_followed_when_the_env_is_empty(self, monkeypatch):
        """With no environment override the first existing, writable entry of
        SQLite's own list wins -- not whatever tempfile caches.

        Mutation: `SQLITE_TEMP_DIR_CANDIDATES = ("/var/tmp", "/usr/tmp", "/tmp")`
        -> `("/tmp", "/usr/tmp", "/var/tmp")`. On this host both exist, so the
        wrong one is returned.
        """
        monkeypatch.delenv("SQLITE_TMPDIR", raising=False)
        monkeypatch.delenv("TMPDIR", raising=False)
        expected = next(
            (Path(c) for c in opencode_gc.SQLITE_TEMP_DIR_CANDIDATES
             if Path(c).is_dir() and os.access(c, os.W_OK | os.X_OK)),
            None,
        )
        assert expected is not None, "this host must have one of SQLite's temp dirs"
        assert opencode_gc._sqlite_temp_dir() == expected

    def test_an_unusable_temp_dir_refuses_instead_of_raising_oserror(
        self, tmp_path, monkeypatch, capsys
    ):
        """A nonexistent SQLITE_TMPDIR used to reach os.stat() and escape as
        FileNotFoundError. It must be the documented conversion refusal, and
        nothing may be deleted.

        Mutation: in enable_incremental_vacuum, remove the `except OSError`
        around the temp-dir stat and restore the bare
        `os.stat(tmp_dir).st_dev` call. main()'s handler catches RuntimeError
        and sqlite3.Error, not OSError, so the run dies with a traceback.
        """
        path = tmp_path / "badtmp.db"
        conn = _make_db(path, auto_vacuum=0)
        _add_session(conn, "old", age_days=30)
        _add_session(conn, "live", age_days=1)
        conn.close()
        before = _snapshot(path)

        # Every candidate unusable, so the failure is unambiguous.
        monkeypatch.setattr(
            opencode_gc, "SQLITE_TEMP_DIR_CANDIDATES", (str(tmp_path / "nope-3"),)
        )
        monkeypatch.setenv("SQLITE_TMPDIR", str(tmp_path / "nope-1"))
        monkeypatch.setenv("TMPDIR", str(tmp_path / "nope-2"))
        # The final "." fallback must not rescue the lookup either.
        monkeypatch.setattr(opencode_gc.os, "access", lambda p, mode: False)
        monkeypatch.setattr(
            opencode_gc.sys, "argv",
            ["opencode-gc", "--db", str(path), "--apply", "--enable-incremental-vacuum"],
        )

        rc = opencode_gc.main()

        assert rc == 1
        err = capsys.readouterr().err
        assert "refusing to delete anything" in err
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
        before = _snapshot_of(path, ["unrelated"])

        rc = self._run(monkeypatch, path, "--apply")

        assert rc == 2
        assert "not an opencode database" in capsys.readouterr().err
        assert _snapshot_of(path, ["unrelated"]) == before

    def test_a_database_missing_one_child_table_is_refused(
        self, tmp_path, monkeypatch, capsys
    ):
        """Discovering a missing table mid-run would leave a half-pruned
        database: the session rows gone, their events stranded.

        Mutation: in verify_usable, `required = ["session"] + [t for t, _ in
        CHILD_TABLES]` -> `required = ["session"]`. The run then starts and
        dies partway through the first batch.
        """
        path = tmp_path / "partial.db"
        conn = _make_db(path)
        _add_session(conn, "old", age_days=30)
        conn.execute("DROP TABLE event")
        conn.close()
        surviving = _snapshot_of(path, ["session", "event_sequence"])

        rc = self._run(monkeypatch, path, "--apply")

        assert rc == 2
        assert "event" in capsys.readouterr().err
        assert _snapshot_of(path, ["session", "event_sequence"]) == surviving, \
            "nothing may be deleted from a database we cannot fully prune"

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
        before = _snapshot_of(path, ["session"])

        rc = self._run(monkeypatch, path, "--apply")

        assert rc == 2
        assert "parent_id" in capsys.readouterr().err
        assert _snapshot_of(path, ["session"]) == before

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
        import json

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
        monkeypatch.setattr(
            opencode_gc.sys, "argv",
            ["opencode-gc", "--db", str(path), "--apply", "--json"],
        )

        rc = opencode_gc.main()
        payload = json.loads(capsys.readouterr().out)

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
        import json

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
        monkeypatch.setattr(
            opencode_gc.sys, "argv",
            ["opencode-gc", "--db", str(path), "--apply", "--json"],
        )

        rc = opencode_gc.main()
        payload = json.loads(capsys.readouterr().out)

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
        import json

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
        monkeypatch.setattr(
            opencode_gc.sys, "argv",
            ["opencode-gc", "--db", str(path), "--apply", "--json"],
        )

        rc = opencode_gc.main()
        payload = json.loads(capsys.readouterr().out)

        assert rc == 1
        assert payload["incomplete"] is True
        assert any("stat failed" in e for e in payload["errors"])
        assert payload["sessions_deleted"] == 3
        assert len(_snapshot(path)["session"]) == 0
