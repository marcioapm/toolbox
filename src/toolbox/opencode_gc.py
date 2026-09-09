#!/usr/bin/env python3
"""Garbage-collect old opencode sessions and hand the freed pages back to the OS.

opencode's SQLite store is append-forever: it never prunes finished sessions.
Measured on the vibes host 2026-09-09: 76.6 GB for 3,164 sessions --
2.30M `event` rows, 611k `part`, 145k `message` -- growing ~6 GB/day and the
single largest consumer of the root filesystem.

WHAT DELETING A SESSION DOES NOT DO
-----------------------------------
The obvious approach (`opencode session delete`, or `DELETE FROM session`) is
not enough, and quietly so. The FK graph is:

    message.session_id       -> session.id                  ON DELETE CASCADE
    part.message_id          -> message.id                  ON DELETE CASCADE
    event.aggregate_id       -> event_sequence.aggregate_id ON DELETE CASCADE
    event_sequence           -> (nothing)

`event_sequence` has NO foreign key to `session`. Its `aggregate_id` merely
happens to equal a session id -- verified exhaustively on the live DB:
3,165/3,165 event_sequence rows and 2,301,539/2,301,539 event rows key on an
existing session id. So deleting a session leaves every one of its event rows
behind: the bulk of the file, orphaned and unreachable.

`PRAGMA foreign_keys` is also OFF by default in SQLite (confirmed: 0), so the
cascades above do not even fire unless enabled. This module therefore deletes
every table explicitly, in dependency order, instead of trusting cascade.

THE DATABASE IS LIVE WHILE THIS RUNS
------------------------------------
opencode may be writing to the same file. Selecting expired sessions and then
deleting them are two different points in time, and `BEGIN IMMEDIATE` only
serialises writers from the moment it takes the lock -- it does not make an
earlier `SELECT` current. A session that became active in between would still
be destroyed, and an order computed earlier describes a topology that may have
changed shape since. Every transaction therefore re-reads the whole `session`
graph under its own write lock and recomputes eligibility, the
descendant-first order and the batch boundary from it; the selection pass
bounds which sessions may be considered, not which are deleted or when.

WHY INCREMENTAL VACUUM AND NOT VACUUM
-------------------------------------
A plain `VACUUM` copies the database to a temporary file and then overwrites
the original, so SQLite documents it as needing roughly twice the file size in
free space. On vibes that is ~153 GB against 41 GB free -- it would fail after
hours of IO. `PRAGMA auto_vacuum=2` (INCREMENTAL) instead lets
`PRAGMA incremental_vacuum(N)` return freed pages to the filesystem in bounded
chunks, with no rewrite and no large temp file.

vibes already runs auto_vacuum=2, so deletes there are immediately reclaimable.
Switching FULL -> INCREMENTAL is a header change and needs no VACUUM; only
NONE -> INCREMENTAL requires the full rewrite, which is what
--enable-incremental-vacuum does, and only after a free-space check.

REPORTING AN INTERRUPTED RUN
----------------------------
Committed batches cannot be undone. A deadline, a lock error or an I/O failure
part-way through therefore still produces a full result -- committed counts,
`incomplete: true`, and the number of eligible sessions left -- instead of a
traceback that tells automation nothing about what was destroyed. Exit status
is 0 for a complete run, 3 for one stopped early, 1 for one that errored.
"""
from __future__ import annotations

import argparse
import heapq
import json
import math
import os
import shutil
import sqlite3
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

DEFAULT_DB = Path.home() / ".local/share/opencode/opencode.db"

# Deletion order matters: children before parents, so an interrupted run can
# never strand rows whose parent is already gone.
#   (table, column holding the session id)
CHILD_TABLES = [
    ("part", "session_id"),
    ("message", "session_id"),
    ("event", "aggregate_id"),
    ("event_sequence", "aggregate_id"),
]

AUTO_VACUUM_NAMES = {0: "NONE", 1: "FULL", 2: "INCREMENTAL"}

# A retention floor under a day is almost certainly a typo, and this deletes
# irreplaceable history. Refuse rather than honour it.
MIN_RETENTION_DAYS = 1.0

# https://sqlite.org/lang_vacuum.html: VACUUM copies into a temporary database
# and then overwrites the original under a rollback journal, so it "can require
# up to twice as much temporary disk space as the original file".
VACUUM_COPY_FACTOR = 2
# Never drive the filesystem to zero; a VACUUM that fits exactly still competes
# with every other writer on the host.
VACUUM_RESERVE_FRACTION = 0.05
MIN_VACUUM_RESERVE_BYTES = 64 * 1024 ** 2


@dataclass
class DbStats:
    page_size: int
    page_count: int
    freelist_count: int
    auto_vacuum: int

    @property
    def total_bytes(self) -> int:
        return self.page_size * self.page_count

    @property
    def free_bytes(self) -> int:
        return self.page_size * self.freelist_count

    @property
    def auto_vacuum_name(self) -> str:
        return AUTO_VACUUM_NAMES.get(self.auto_vacuum, str(self.auto_vacuum))


@dataclass
class Selection:
    """Sessions eligible for deletion, plus why the rest were spared."""

    deletable: list[str]
    kept_live_descendant: int = 0
    kept_unknown_age: int = 0
    kept_parent_cycle: int = 0


@dataclass
class DeleteOutcome:
    rows: dict = field(default_factory=dict)
    # Candidates found ineligible under a write lock -- revived, sheltering a
    # live descendant, or cyclic in the graph at that moment.
    skipped: list = field(default_factory=list)
    # Candidates never attempted, because the run stopped first.
    remaining: int = 0
    deadline_reached: bool = False
    # The failure that stopped the run, if any. Whatever committed before it
    # is already durable, so `rows` is still authoritative for those batches.
    failure: Optional[str] = None

    @property
    def incomplete(self) -> bool:
        return self.failure is not None or self.deadline_reached


@dataclass
class Result:
    db: str
    dry_run: bool
    retention_days: float
    cutoff_ms: int = 0
    sessions_expired: int = 0
    sessions_deleted: int = 0
    sessions_skipped_revalidation: int = 0
    sessions_remaining: int = 0
    sessions_kept_live_descendant: int = 0
    sessions_kept_unknown_age: int = 0
    sessions_kept_parent_cycle: int = 0
    rows_deleted: dict = field(default_factory=dict)
    pages_released: int = 0
    pages_reclaimable_remaining: int = 0
    vacuum_deadline_reached: bool = False
    # Real on-disk footprint (main database + WAL), not page arithmetic: a
    # released page is not a reclaimed byte until the file actually shrinks.
    bytes_before: int = 0
    bytes_after: int = 0
    auto_vacuum_before: str = ""
    auto_vacuum_after: str = ""
    # True when work that was eligible did not happen: a deadline, or a
    # failure after earlier batches had already committed irreversibly.
    incomplete: bool = False
    deadline_reached: bool = False
    errors: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    @property
    def bytes_reclaimed(self) -> int:
        return max(0, self.bytes_before - self.bytes_after)


def connect(db: Path, *, read_only: bool, timeout_s: float = 30.0) -> sqlite3.Connection:
    """Open `db` -- that exact file, whatever its name contains.

    Interpolating a path into a `file:` URI lets a legitimate filename be
    reparsed as URI syntax: `victim?.db` opens `victim`, `a#b.db` opens `a`,
    and `x%41.db` opens `xA.db`. For a tool that deletes, opening a different
    database than the one whose existence was checked is the entire blast
    radius aimed at the wrong target. The read-write path therefore uses a
    plain filename with no URI parsing at all, and the read-only path uses a
    properly escaped URI because `mode=ro` has no non-URI equivalent.
    """
    resolved = Path(db).resolve()
    if read_only:
        conn = sqlite3.connect(
            resolved.as_uri() + "?mode=ro", uri=True, timeout=timeout_s, isolation_level=None
        )
    else:
        conn = sqlite3.connect(
            str(resolved), uri=False, timeout=timeout_s, isolation_level=None
        )
    try:
        conn.execute(f"PRAGMA busy_timeout={int(timeout_s * 1000)}")
    except Exception:
        conn.close()
        raise
    return conn


def read_stats(conn: sqlite3.Connection) -> DbStats:
    def one(pragma: str) -> int:
        return int(conn.execute(f"PRAGMA {pragma}").fetchone()[0])

    return DbStats(
        page_size=one("page_size"),
        page_count=one("page_count"),
        freelist_count=one("freelist_count"),
        auto_vacuum=one("auto_vacuum"),
    )


def on_disk_bytes(db: Path) -> int:
    """Main database plus WAL. The -shm file is scratch and carries no data."""
    total = 0
    for path in (db, db.with_name(db.name + "-wal")):
        try:
            total += path.stat().st_size
        except OSError:
            pass
    return total


def _order_descendant_first(
    deletable: set[str], parent_of: dict[str, Optional[str]]
) -> tuple[list[str], list[str]]:
    """Order so every session precedes its ancestors; return (ordered, cyclic).

    Batching commits each chunk separately, so a parent deleted before its
    child leaves the child with a dangling `parent_id` if the run then stops.
    Sessions in a `parent_id` cycle have no such order and are returned
    unordered for the caller to retain.
    """
    pending_children = {sid: 0 for sid in deletable}
    for sid in deletable:
        parent = parent_of.get(sid)
        if parent in pending_children:
            pending_children[parent] += 1

    ready = [sid for sid, n in pending_children.items() if n == 0]
    heapq.heapify(ready)
    ordered: list[str] = []
    while ready:
        sid = heapq.heappop(ready)
        ordered.append(sid)
        parent = parent_of.get(sid)
        if parent in pending_children:
            pending_children[parent] -= 1
            if pending_children[parent] == 0:
                heapq.heappush(ready, parent)

    return ordered, sorted(deletable - set(ordered))


def select_expired(
    rows: list[tuple], cutoff_ms: int, *, restrict_to: Optional[set[str]] = None
) -> Selection:
    """Sessions whose entire subtree is older than the cutoff, deepest first.

    `rows` is (id, parent_id, time_updated) for the whole `session` table.

    `session.parent_id` has no foreign key, so deleting a parent whose child is
    still active would leave the child pointing at a row that no longer exists.
    A session is therefore only expired when it and every descendant are.

    A NULL `time_updated` is an unknown age, not an infinite one: such a
    session is kept and counted rather than treated as older than the cutoff.

    `restrict_to` narrows the candidates to that set. Every other session in
    `rows` is then retained whatever its age, and still protects its ancestors
    -- which is what makes a mid-run recomputation safe: a session that
    appeared after the candidates were chosen was never authorised for
    deletion, so it counts as a live descendant even when it is old.
    """
    updated = {r[0]: r[2] for r in rows}
    parent_of: dict[str, Optional[str]] = {r[0]: r[1] for r in rows}

    unknown = {sid for sid, t in updated.items() if t is None}
    old = {sid for sid, t in updated.items() if t is not None and t < cutoff_ms}
    if restrict_to is not None:
        old &= restrict_to

    # Walk up from every live session, protecting its whole ancestor chain.
    protected: set[str] = set()
    for sid in updated:
        if sid in old:
            continue
        cur = parent_of.get(sid)
        seen = {sid}
        while cur is not None and cur not in seen:
            seen.add(cur)
            protected.add(cur)
            cur = parent_of.get(cur)

    ordered, cyclic = _order_descendant_first(old - protected, parent_of)
    return Selection(
        deletable=ordered,
        kept_live_descendant=len(old & protected),
        kept_unknown_age=len(unknown),
        kept_parent_cycle=len(cyclic),
    )


def expired_session_ids(
    conn: sqlite3.Connection, cutoff_ms: int, *, restrict_to: Optional[set[str]] = None
) -> Selection:
    rows = conn.execute("SELECT id, parent_id, time_updated FROM session").fetchall()
    return select_expired(rows, cutoff_ms, restrict_to=restrict_to)


def delete_sessions(
    conn: sqlite3.Connection,
    session_ids: list[str],
    *,
    cutoff_ms: int,
    batch: int,
    deadline: Optional[float],
    clock=time.monotonic,
) -> DeleteOutcome:
    """Delete sessions and all their rows, children first, in batches.

    Batched so a single statement never holds the write lock across millions of
    rows while opencode itself is running.

    `session_ids` only bounds the work: it is the set of sessions this run is
    *allowed* to consider, and nothing more. Every batch is chosen from
    scratch inside its own transaction, under the write lock, from the
    `session` graph as it exists at that moment -- eligibility, the
    descendant-first order, and the batch boundary alike. An order computed
    before the lock describes a topology that may no longer exist: a session
    reparented, or a new old child inserted under a candidate, changes which
    deletions strand a row. Only ids in `session_ids` are ever deleted; every
    other session in the graph is treated as retained whatever its age and
    protects its whole ancestor chain, so a session that appeared after
    selection can never be deleted and always shields its parents.

    `clock` is the monotonic source the deadline is compared against.

    A batch that fails is rolled back, but every batch committed before it is
    already durable and irreversible. The failure is therefore recorded on the
    outcome rather than raised, so the caller can report exactly what was
    destroyed instead of losing the counts to a traceback. Row counts are
    accumulated per transaction and merged into `rows` only once its COMMIT
    has returned: a rolled-back DELETE undid its rows, and reporting them as
    destroyed would misdirect the very recovery the counts exist to inform.
    """
    outcome = DeleteOutcome(rows={t: 0 for t, _ in CHILD_TABLES} | {"session": 0})
    order_index = {sid: i for i, sid in enumerate(session_ids)}
    pending = set(session_ids)

    while pending:
        if deadline is not None and clock() > deadline:
            outcome.deadline_reached = True
            break

        # BEGIN IMMEDIATE is inside the guarded region: it is the statement
        # most likely to fail, with SQLITE_BUSY, when opencode holds the write
        # lock -- and by then earlier batches have already committed.
        began = False
        committed = {t: 0 for t, _ in CHILD_TABLES} | {"session": 0}
        try:
            conn.execute("BEGIN IMMEDIATE")
            began = True
            eligible = expired_session_ids(conn, cutoff_ms, restrict_to=pending).deletable
            doomed = eligible[:batch]
            # Anything left over is ineligible in the graph under this lock:
            # revived, newly sheltering a live descendant, or cyclic.
            ineligible = pending - set(eligible)
            if doomed:
                marks = ",".join("?" * len(doomed))
                for table, column in CHILD_TABLES:
                    cur = conn.execute(
                        f"DELETE FROM {table} WHERE {column} IN ({marks})", doomed
                    )
                    committed[table] += max(0, cur.rowcount)
                cur = conn.execute(f"DELETE FROM session WHERE id IN ({marks})", doomed)
                committed["session"] += max(0, cur.rowcount)
            conn.execute("COMMIT")
        except (sqlite3.Error, KeyboardInterrupt, MemoryError, OSError) as exc:
            if began:
                _rollback_quietly(conn)
            outcome.failure = f"{type(exc).__name__}: {exc}"
            break
        except BaseException:
            if began:
                _rollback_quietly(conn)
            raise
        # Past COMMIT: the deletes are durable, so their counts are now facts.
        for table, n in committed.items():
            outcome.rows[table] += n
        outcome.skipped.extend(sorted(ineligible, key=order_index.__getitem__))
        pending -= set(doomed)
        pending -= ineligible

    outcome.remaining = len(pending)
    return outcome


def _rollback_quietly(conn: sqlite3.Connection) -> None:
    """Abandon the open transaction. A rollback that itself fails must not mask
    the original error, and there is nothing further to do about it."""
    try:
        conn.execute("ROLLBACK")
    except sqlite3.Error:
        pass


def count_rows_for(conn: sqlite3.Connection, session_ids: list[str], batch: int) -> dict[str, int]:
    """Rows a real run would delete, as of right now.

    A point-in-time estimate: the apply path re-checks eligibility under its own
    write lock, so a concurrently-revived session is counted here and spared
    there."""
    counts: dict[str, int] = {t: 0 for t, _ in CHILD_TABLES}
    counts["session"] = len(session_ids)
    for start in range(0, len(session_ids), batch):
        chunk = session_ids[start:start + batch]
        marks = ",".join("?" * len(chunk))
        for table, column in CHILD_TABLES:
            counts[table] += conn.execute(
                f"SELECT count(*) FROM {table} WHERE {column} IN ({marks})", chunk
            ).fetchone()[0]
    return counts


@dataclass
class VacuumOutcome:
    """What one reclamation pass actually managed to hand back.

    `released < target` with `deadline_reached` set is an incomplete run:
    pages that were eligible are still in the file, and re-running continues.
    """

    released: int = 0
    target: int = 0
    deadline_reached: bool = False

    @property
    def remaining(self) -> int:
        return max(0, self.target - self.released)


def run_incremental_vacuum(
    conn: sqlite3.Connection, *, pages: Optional[int], deadline: Optional[float],
    clock=time.monotonic,
) -> VacuumOutcome:
    """Release freed pages back to the filesystem.

    Bounded per call so a huge freelist cannot block the database for minutes.
    In WAL mode the released pages only leave the file once a checkpoint runs,
    which a long-lived reader can defer, so the count is of logical pages --
    not a promise of bytes off the disk.

    Stopping on the deadline is reported rather than swallowed: exiting 0 with
    a still-oversized file tells an operator the reclamation finished when it
    did not.
    """
    before = int(conn.execute("PRAGMA freelist_count").fetchone()[0])
    if before == 0:
        return VacuumOutcome()
    step = 2000
    released = 0
    target = before if pages is None else min(pages, before)
    outcome = VacuumOutcome(target=target)
    while released < target:
        if deadline is not None and clock() > deadline:
            outcome.deadline_reached = True
            break
        conn.execute(f"PRAGMA incremental_vacuum({min(step, target - released)})")
        now = int(conn.execute("PRAGMA freelist_count").fetchone()[0])
        progressed = before - now - released
        released = before - now
        outcome.released = released
        if progressed <= 0:
            # No page moved: the freelist is as small as this pass can make
            # it, so the target is unreachable rather than merely unfinished.
            outcome.target = released
            break
    return outcome


def vacuum_space_plan(db: Path, stats: DbStats) -> tuple[int, int, int]:
    """Free bytes a full VACUUM needs: (payload, on the DB's fs, on the temp fs).

    The payload is everything that has to be copied -- the database and its
    uncheckpointed WAL. The database's filesystem must hold the rewritten copy
    and the rollback journal of the overwrite; SQLite's temp filesystem, when
    it is a different one, must hold one copy.
    """
    wal = 0
    try:
        wal = db.with_name(db.name + "-wal").stat().st_size
    except OSError:
        pass
    payload = stats.total_bytes + wal
    reserve = max(MIN_VACUUM_RESERVE_BYTES, int(payload * VACUUM_RESERVE_FRACTION))
    return payload, VACUUM_COPY_FACTOR * payload + reserve, payload + reserve


# https://sqlite.org/tempfiles.html: on unix SQLite tries each of these in
# turn and uses the first that exists and is writable and searchable. Asking
# Python's tempfile instead answers a different question -- it applies its own
# policy and caches the result -- and on a host where /var/tmp and /tmp are
# separate filesystems that means measuring free space on the wrong one.
SQLITE_TEMP_DIR_CANDIDATES = ("/var/tmp", "/usr/tmp", "/tmp")


def _sqlite_temp_dir() -> Path:
    """Where SQLite will put the VACUUM copy, in SQLite's own search order.

    Raises RuntimeError when no candidate qualifies: the free-space guard
    cannot be evaluated at all then, and a destructive rewrite must not start
    on an unchecked filesystem.
    """
    candidates = []
    for value in (os.environ.get("SQLITE_TMPDIR"), os.environ.get("TMPDIR")):
        if value:
            candidates.append(value)
    candidates.extend(SQLITE_TEMP_DIR_CANDIDATES)
    # SQLite falls back to the current directory when nothing else qualifies.
    candidates.append(".")

    for candidate in candidates:
        path = Path(candidate)
        try:
            if path.is_dir() and os.access(path, os.W_OK | os.X_OK):
                return path
        except OSError:
            continue
    raise RuntimeError(
        "none of SQLite's temporary directories "
        f"({', '.join(str(c) for c in candidates)}) exists and is writable, "
        "so the space a VACUUM needs cannot be checked. Set SQLITE_TMPDIR to a "
        "writable directory on a filesystem with room for one copy."
    )


def enable_incremental_vacuum(conn: sqlite3.Connection, db: Path, stats: DbStats) -> list[str]:
    """Switch the DB to auto_vacuum=INCREMENTAL.

    FULL -> INCREMENTAL is a header change. NONE -> INCREMENTAL needs a full
    VACUUM, so it raises RuntimeError unless the filesystems involved have the
    space SQLite documents for one, because a VACUUM that runs out mid-rewrite
    is far worse than not starting.
    """
    if stats.auto_vacuum == 2:
        return ["auto_vacuum already INCREMENTAL; nothing to change"]

    if stats.auto_vacuum == 1:
        conn.execute("PRAGMA auto_vacuum=2")
        if int(conn.execute("PRAGMA auto_vacuum").fetchone()[0]) != 2:
            raise RuntimeError("PRAGMA auto_vacuum=2 did not take effect on a FULL database")
        return ["auto_vacuum FULL -> INCREMENTAL (header change, no VACUUM needed)"]

    db = Path(db).resolve()
    payload, need_db_fs, need_tmp_fs = vacuum_space_plan(db, stats)
    free = shutil.disk_usage(db.parent).free
    if free < need_db_fs:
        raise RuntimeError(
            f"VACUUM needs ~{need_db_fs:,} bytes free on {db.parent} to rewrite a "
            f"{payload:,} byte database (+WAL); only {free:,} available. "
            "Delete rows first, or move the database to a larger filesystem."
        )
    tmp_dir = _sqlite_temp_dir()
    try:
        different_fs = os.stat(tmp_dir).st_dev != os.stat(db.parent).st_dev
        tmp_free = shutil.disk_usage(tmp_dir).free if different_fs else None
    except OSError as exc:
        # The temp filesystem could not be measured, so the space a VACUUM
        # needs there is unknown. Refuse rather than rewrite on an unchecked
        # filesystem: an interrupted VACUUM is worse than a skipped one.
        raise RuntimeError(
            f"cannot inspect SQLite's temp directory {tmp_dir}: {exc}. "
            "Set SQLITE_TMPDIR to a readable directory with room for one copy."
        ) from exc
    if different_fs and tmp_free < need_tmp_fs:
        raise RuntimeError(
            f"VACUUM needs ~{need_tmp_fs:,} bytes free on SQLite's temp "
            f"filesystem {tmp_dir}; only {tmp_free:,} available. "
            "Set SQLITE_TMPDIR to a larger filesystem."
        )

    conn.execute("PRAGMA auto_vacuum=2")
    conn.execute("VACUUM")
    return [f"auto_vacuum {stats.auto_vacuum_name} -> INCREMENTAL (full VACUUM run)"]


def verify_usable(conn: sqlite3.Connection) -> None:
    """Confirm this is readable SQLite carrying the schema we delete from.

    Raises RuntimeError otherwise. Reaching the deletion loop and discovering
    there mid-run that a table is missing would leave a half-pruned database
    and a traceback; every table this tool touches is therefore probed first,
    while nothing has been written.
    """
    required = ["session"] + [t for t, _ in CHILD_TABLES]
    try:
        present = {
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
    except sqlite3.DatabaseError as exc:
        raise RuntimeError(f"not a readable SQLite database: {exc}") from exc

    missing = [t for t in required if t not in present]
    if missing:
        raise RuntimeError(
            f"not an opencode database: table(s) {', '.join(missing)} are missing"
        )

    # A table can exist without the columns this tool keys on.
    for table, column in [("session", "id"), ("session", "parent_id"),
                          ("session", "time_updated")] + list(CHILD_TABLES):
        try:
            cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        except sqlite3.DatabaseError as exc:
            raise RuntimeError(f"cannot read the schema of {table}: {exc}") from exc
        if column not in cols:
            raise RuntimeError(
                f"not an opencode database: {table}.{column} is missing"
            )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--db", type=Path, default=DEFAULT_DB,
                    help=f"opencode database (default {DEFAULT_DB})")
    ap.add_argument("--retention-days", type=float, default=5.0,
                    help="keep sessions updated within this many days (default 5)")
    ap.add_argument("--apply", action="store_true",
                    help="actually delete; without this the run is a dry run")
    ap.add_argument("--batch", type=int, default=200,
                    help="sessions per transaction (default 200; clamped to "
                         "SQLite's bound-variable limit)")
    ap.add_argument("--max-seconds", type=float, default=600.0,
                    help="stop starting new batches after this long (default 600); "
                         "0 means no limit. Not a bound on total runtime: a batch "
                         "or a VACUUM already in flight runs to completion")
    ap.add_argument("--vacuum-pages", type=int, default=None,
                    help="cap pages released per run, >= 1 "
                         "(default: the whole freelist)")
    ap.add_argument("--no-vacuum", action="store_true",
                    help="delete rows but do not release pages")
    ap.add_argument("--enable-incremental-vacuum", action="store_true",
                    help="switch auto_vacuum to INCREMENTAL. From FULL this is a "
                         "header change; from NONE it needs one full VACUUM and "
                         f"~{VACUUM_COPY_FACTOR}x the database size free")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if not math.isfinite(args.retention_days) or args.retention_days < MIN_RETENTION_DAYS:
        ap.error(f"--retention-days must be finite and >= {MIN_RETENTION_DAYS}")
    if args.batch < 1:
        ap.error("--batch must be >= 1")
    # NaN compares false against every bound, so an unchecked NaN would silently
    # disable the deadline instead of limiting it.
    if not math.isfinite(args.max_seconds) or args.max_seconds < 0:
        ap.error("--max-seconds must be finite and >= 0 (0 means no limit)")
    if args.vacuum_pages is not None and args.vacuum_pages < 1:
        ap.error("--vacuum-pages must be >= 1; omit it to release the whole freelist")
    if not args.db.is_file():
        print(f"opencode-gc: no database at {args.db}", file=sys.stderr)
        return 2
    # connect() opens the resolved file, so every sidecar probe (-wal, the
    # containing filesystem) must resolve too: for `alias.db -> target.db`,
    # `alias.db-wal` is a path that simply does not exist. Only the report
    # keeps the name the operator typed.
    db_path = args.db.resolve()

    deadline = time.monotonic() + args.max_seconds if args.max_seconds > 0 else None
    res = Result(db=str(args.db), dry_run=not args.apply,
                 retention_days=args.retention_days)

    try:
        conn = connect(args.db, read_only=not args.apply)
    except (sqlite3.Error, RuntimeError, ValueError) as exc:
        print(f"opencode-gc: cannot open {args.db}: {exc}", file=sys.stderr)
        return 2
    # sqlite3.connect() does not touch the file, so a corrupt or foreign
    # database is only discovered on first use. Find out now, before anything
    # is deleted, and report it as an unusable input rather than a traceback.
    try:
        verify_usable(conn)
    except RuntimeError as exc:
        conn.close()
        print(f"opencode-gc: cannot use {args.db}: {exc}", file=sys.stderr)
        return 2
    try:
        # Clamp before anything is deleted: a batch wider than SQLite's
        # parameter limit must not fail after earlier batches have committed.
        var_limit = conn.getlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER)
        batch = min(args.batch, var_limit)
        if batch < args.batch:
            res.notes.append(f"--batch {args.batch} clamped to {batch} "
                             "(SQLITE_LIMIT_VARIABLE_NUMBER)")

        stats = read_stats(conn)
        res.bytes_before = on_disk_bytes(db_path)
        res.auto_vacuum_before = res.auto_vacuum_after = stats.auto_vacuum_name

        if args.enable_incremental_vacuum:
            if not args.apply:
                res.notes.append(
                    f"would switch auto_vacuum {stats.auto_vacuum_name} -> INCREMENTAL"
                    if stats.auto_vacuum != 2 else
                    "auto_vacuum already INCREMENTAL; nothing to change"
                )
            else:
                try:
                    res.notes.extend(enable_incremental_vacuum(conn, db_path, stats))
                    res.auto_vacuum_after = read_stats(conn).auto_vacuum_name
                except (RuntimeError, sqlite3.Error) as exc:
                    # Deleting anyway would destroy history and still leave the
                    # freed pages unreclaimable: the worst of both outcomes.
                    res.errors.append(f"{exc}; refusing to delete anything")
                    res.bytes_after = on_disk_bytes(db_path)
                    return _report(args, res)

        res.cutoff_ms = cutoff_ms = int((time.time() - args.retention_days * 86400) * 1000)
        selection = expired_session_ids(conn, cutoff_ms)
        res.sessions_expired = len(selection.deletable)
        res.sessions_kept_live_descendant = selection.kept_live_descendant
        res.sessions_kept_unknown_age = selection.kept_unknown_age
        res.sessions_kept_parent_cycle = selection.kept_parent_cycle

        if selection.deletable:
            if args.apply:
                outcome = delete_sessions(
                    conn, selection.deletable,
                    cutoff_ms=cutoff_ms, batch=batch, deadline=deadline,
                )
                res.rows_deleted = outcome.rows
                res.sessions_deleted = outcome.rows.get("session", 0)
                res.sessions_skipped_revalidation = len(outcome.skipped)
                res.sessions_remaining = outcome.remaining
                res.incomplete = outcome.incomplete
                res.deadline_reached = outcome.deadline_reached
                if outcome.failure:
                    res.errors.append(
                        f"stopped after {outcome.rows.get('session', 0)} committed "
                        f"session deletion(s): {outcome.failure}"
                    )
                if outcome.deadline_reached:
                    res.notes.append(
                        f"--max-seconds reached; {outcome.remaining} eligible "
                        "session(s) were not attempted. Re-run to continue."
                    )
            else:
                res.rows_deleted = count_rows_for(conn, selection.deletable, batch)

        # Past this point rows may already be gone. Nothing that follows is
        # allowed to escape as a traceback: an operator deciding what to
        # restore needs the committed counts far more than a stack trace.
        try:
            # Reclaiming is still worth doing after a partial delete: the pages
            # freed by the batches that did commit are already on the freelist.
            if args.apply and not args.no_vacuum:
                current = read_stats(conn)
                if current.auto_vacuum == 2:
                    vac = run_incremental_vacuum(
                        conn, pages=args.vacuum_pages, deadline=deadline
                    )
                    res.pages_released = vac.released
                    res.pages_reclaimable_remaining = vac.remaining
                    res.vacuum_deadline_reached = vac.deadline_reached
                    if vac.deadline_reached:
                        # Pages that were eligible are still in the file, so
                        # the file is larger than the run implies.
                        res.incomplete = True
                        res.notes.append(
                            f"--max-seconds reached during page reclamation; "
                            f"{vac.remaining:,} of {vac.target:,} page(s) were not "
                            "released. Re-run to continue."
                        )
                elif current.freelist_count:
                    res.notes.append(
                        f"{current.freelist_count:,} pages are free but auto_vacuum is "
                        f"{current.auto_vacuum_name}; they stay in the file. Re-run with "
                        "--enable-incremental-vacuum to reclaim future deletes."
                    )
        except (sqlite3.Error, KeyboardInterrupt, OSError) as exc:
            # The deletes are already committed and must still be reported;
            # failing to reclaim is not failing to delete.
            res.incomplete = True
            res.errors.append(f"page reclamation stopped: "
                              f"{type(exc).__name__}: {exc}")
    finally:
        try:
            conn.close()
        except sqlite3.Error as exc:
            res.incomplete = True
            res.errors.append(f"closing the database failed: "
                              f"{type(exc).__name__}: {exc}")

    try:
        res.bytes_after = on_disk_bytes(db_path)
    except OSError as exc:
        res.incomplete = True
        res.errors.append(f"measuring the database failed: "
                          f"{type(exc).__name__}: {exc}")
    if res.pages_released and res.bytes_reclaimed == 0:
        res.notes.append(
            f"{res.pages_released:,} pages were released but the file has not "
            "shrunk yet; a WAL checkpoint is pending, typically because another "
            "reader is still holding the database open."
        )
    return _report(args, res)


def _report(args, res: Result) -> int:
    """Print the result and map it to an exit status.

    0 = complete, 1 = an error occurred, 3 = no error but eligible work was
    left undone (a deadline). Automation must be able to tell "finished" from
    "stopped part-way with rows already destroyed".
    """
    if args.json:
        print(json.dumps(asdict(res) | {"bytes_reclaimed": res.bytes_reclaimed}, indent=2))
        return _status(res)

    mode = "APPLIED" if args.apply else "DRY RUN"
    if res.incomplete:
        mode += " (INCOMPLETE)"
    gib = res.bytes_reclaimed / 1024 ** 3
    print(f"[opencode-gc] {mode}: retention={res.retention_days}d "
          f"sessions_expired={res.sessions_expired} "
          f"sessions_deleted={res.sessions_deleted} "
          f"reclaimed={gib:.2f}GiB auto_vacuum={res.auto_vacuum_after}")
    if res.rows_deleted:
        verb = "deleted" if args.apply else "would delete (estimate at this instant)"
        rows = " ".join(f"{k}={v:,}" for k, v in sorted(res.rows_deleted.items()))
        print(f"    {verb}: {rows}")
        if not args.apply:
            print(f"    cutoff: sessions not updated since epoch ms {res.cutoff_ms}")
    if res.incomplete and args.apply and res.sessions_remaining:
        print(f"    INCOMPLETE: {res.sessions_remaining} eligible session(s) not "
              "attempted; the deletions above are already committed")
    if res.sessions_skipped_revalidation:
        print(f"    skipped {res.sessions_skipped_revalidation} session(s) no longer "
              "eligible under the write lock")
    if res.sessions_kept_live_descendant:
        print(f"    kept {res.sessions_kept_live_descendant} old session(s) with a "
              "recently-updated descendant")
    if res.sessions_kept_unknown_age:
        print(f"    kept {res.sessions_kept_unknown_age} session(s) with a NULL "
              "time_updated (unknown age, not expired)")
    if res.sessions_kept_parent_cycle:
        print(f"    kept {res.sessions_kept_parent_cycle} session(s) in a parent_id "
              "cycle (no safe deletion order)")
    if res.pages_released:
        print(f"    released {res.pages_released:,} logical pages "
              f"({res.bytes_reclaimed:,} bytes actually off the disk)")
    for note in res.notes:
        print(f"    {note}")
    for err in res.errors:
        print(f"    ERROR: {err}", file=sys.stderr)
    return _status(res)


def _status(res: Result) -> int:
    if res.errors:
        return 1
    return 3 if res.incomplete else 0


if __name__ == "__main__":
    sys.exit(main())
