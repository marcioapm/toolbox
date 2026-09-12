#!/usr/bin/env python3
"""Garbage-collect old opencode sessions and hand the freed pages back to the OS.

opencode's SQLite store is append-forever: it never prunes finished sessions.
Measured on the vibes host 2026-09-09: 76.6 GB for 3,164 sessions --
2.30M `event` rows, 611k `part`, 145k `message` -- growing ~6 GB/day and the
single largest consumer of the root filesystem.

WHAT DELETING A SESSION DOES NOT DO
-----------------------------------
The obvious approach (`opencode session delete`, or `DELETE FROM session`) is
not enough, and quietly so. The FK graph, re-enumerated from the live schema
2026-09-12, is:

    message.session_id                -> session.id   ON DELETE CASCADE
    todo.session_id                   -> session.id   ON DELETE CASCADE
    session_message.session_id        -> session.id   ON DELETE CASCADE
    session_input.session_id          -> session.id   ON DELETE CASCADE
    session_share.session_id          -> session.id   ON DELETE CASCADE
    session_context_epoch.session_id  -> session.id   ON DELETE CASCADE
    part.message_id                   -> message.id   ON DELETE CASCADE
    event.aggregate_id  -> event_sequence.aggregate_id ON DELETE CASCADE
    event_sequence                    -> (nothing)

`event_sequence` has NO foreign key to `session`. Its `aggregate_id` merely
happens to equal a session id -- verified exhaustively on the live DB:
3,165/3,165 event_sequence rows and 2,301,539/2,301,539 event rows key on an
existing session id. So deleting a session leaves every one of its event rows
behind: the bulk of the file, orphaned and unreachable.

`PRAGMA foreign_keys` is also OFF by default in SQLite (confirmed: 0), so the
cascades above do not even fire unless enabled. This module therefore deletes
every table explicitly, in dependency order, instead of trusting cascade --
which is only correct while that list is complete, so `verify_usable` refuses
to run against a schema carrying a session child this module does not know.

THE WAL IS WHERE THE BYTES ACTUALLY ARE
---------------------------------------
In WAL mode a page released by `incremental_vacuum` does not leave the file
until a checkpoint folds the WAL back into it, and an uncheckpointed WAL is
itself on the disk: 15.28 GiB on vibes 2026-09-12, never once checkpointed.
Reclaiming without checkpointing therefore reports released pages while the
filesystem sees nothing, which is what `bytes_before`/`bytes_after` exists to
expose. So every mutating run checkpoints.

Which checkpoint matters. TRUNCATE and RESTART wait for readers and block
writers while they hold the WAL; PASSIVE never blocks. opencode instances are
writers, and a writer that exhausts its own `busy_timeout` against us dies with
"Failed to execute statement" -- observed. So the mode is chosen from whether
anything else holds the database: PASSIVE whenever it might, TRUNCATE only when
nothing does.

RECLAIM IN PLACE; REBUILD BY HAND, AGAINST A QUIESCENT DATABASE
---------------------------------------------------------------
`PRAGMA incremental_vacuum` relocates pages one at a time with pointer-map
updates, and measures at ~10-20 MB/min (2,141 pages in 60.7s on vibes; 1,891
pages in 62.1s on macmini). Draining a 55 GiB freelist at that rate takes
~90-108 hours: it is a trickle that keeps a pruned database from growing, not a
way to shrink one that already has.

The `rebuild` subcommand is. `VACUUM INTO` writes only the compacted copy, so
it needs live-size plus a margin rather than the 2x a plain `VACUUM` needs --
the distinction that makes a 32 GiB file with 6.2 GiB live reclaimable on 13
GiB of free disk.

It is a separate subcommand, and not a flag on the prune, because replacing a
database file safely WHILE ANOTHER PROCESS IS WRITING TO IT is a problem this
tool does not solve. Three designs were attempted and each produced Criticals:
a writer's commit landing in the replacement's `-wal` after the lock was
dropped; stragglers that opened the replacement after the first rename; a
pre-existing hard link making "the old inode has exactly one pathname" false;
and process death between the rename and the cleanup leaving two main files
sharing one `db-wal` namespace. Every one of them needs a concurrent writer.

So the precondition replaces the proof: `rebuild` is run by hand, by an
operator who has stopped opencode, and it requires an explicit
`--yes-nothing-is-using-it`. Nothing on a timer can reach it -- `prune` has no
flag that rebuilds. The checks that remain (holders, hard links, WAL foldable,
`data_version`) are PRE-FLIGHT CHECKS THAT CATCH A MISTAKEN OPERATOR. They are
not, and are not claimed to be, proof that nothing is attached. See
`rebuild_database`.

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
Committed batches cannot be undone. A deadline, a Ctrl-C, a lock error or an
I/O failure part-way through therefore still produces a full result --
committed counts, `incomplete: true`, and the number of eligible sessions left
-- instead of a traceback that tells automation nothing about what was
destroyed. Ctrl-C is the expected way to stop a multi-hour prune, so the
guarded region covers every statement of a batch and everything after the last
one: an interrupt landing between two guarded blocks would exit 130, a status
no report ever produces. Exit status is 0 for a complete run, 3 for one
stopped early, 1 for one that errored.
"""
from __future__ import annotations

import argparse
import fcntl
import heapq
import json
import math
import os
import shutil
import sqlite3
import stat
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

DEFAULT_DB = Path.home() / ".local/share/opencode/opencode.db"

# Children before parents, so every intermediate state of the transaction is
# referentially valid and the sequence works under immediate FK enforcement
# whatever the referential action -- the live schema's cascades happen to make
# either order work, and depending on that is depending on the schema not
# changing. Atomicity across an interrupt comes from the transaction, not from
# this order; the cross-batch guarantee comes from _order_descendant_first.
#
# Every table here was enumerated from the live schema, not assumed: each one
# whose foreign key names `session(id)`, plus the two event tables that key on
# a session id with no foreign key at all. With `PRAGMA foreign_keys` off --
# SQLite's default, and what this module runs under -- a table missing from
# this list is a table whose rows are orphaned by every session deleted.
#   (table, column holding the session id)
CHILD_TABLES = [
    ("part", "session_id"),
    ("message", "session_id"),
    ("todo", "session_id"),
    ("session_message", "session_id"),
    ("session_input", "session_id"),
    ("session_share", "session_id"),
    ("session_context_epoch", "session_id"),
    ("event", "aggregate_id"),
    ("event_sequence", "aggregate_id"),
]

# opencode has shipped several of the tables above only recently, and this tool
# has to keep working against a store written by an older build. Only these
# must exist; the rest are deleted from when present and ignored when not.
REQUIRED_CHILD_TABLES = frozenset({"part", "message", "event", "event_sequence"})

AUTO_VACUUM_NAMES = {0: "NONE", 1: "FULL", 2: "INCREMENTAL"}

# A retention floor under a day is almost certainly a typo, and this deletes
# irreplaceable history. Refuse rather than honour it.
MIN_RETENTION_DAYS = 1.0

# https://sqlite.org/lang_vacuum.html: VACUUM copies into a temporary database
# and then overwrites the original under a rollback journal, so it "can require
# up to twice as much temporary disk space as the original file".
VACUUM_COPY_FACTOR = 2
# `VACUUM INTO` is the other shape: it writes only the compacted copy and never
# overwrites the original, so it needs the live size plus a margin -- not 2x.
# That is why a 32 GiB file with 6.2 GiB live can be rebuilt on 13 GiB free,
# where a plain VACUUM of the same file needs ~64 GiB and is rightly refused.
REBUILD_COPY_HEADROOM = 1.05
# Never drive the filesystem to zero; a VACUUM that fits exactly still competes
# with every other writer on the host.
VACUUM_RESERVE_FRACTION = 0.05
MIN_VACUUM_RESERVE_BYTES = 64 * 1024 ** 2

# A `VACUUM INTO` is one uninterruptible SQLite call: no Python-level deadline
# applies to it. On 2026-09-12 one ran 23 minutes in uninterruptible disk
# sleep, wrote a 39.6 GB temp copy and drove the disk from 88% to 93% before it
# was killed by hand. These bounds, enforced from inside the statement by
# `set_progress_handler`, are what stop that.
DEFAULT_REBUILD_MAX_SECONDS = 900.0
DEFAULT_REBUILD_MIN_FREE_GIB = 25.0
# statvfs on every callback costs more than the vacuum it guards.
REBUILD_DISK_POLL_SECONDS = 2.0
# Frequent enough to notice a deadline promptly, rare enough not to measurably
# slow the rebuild.
REBUILD_PROGRESS_INSTRUCTIONS = 20_000
REBUILD_SUFFIX = ".rebuild-tmp"
# The old database is hard-linked here across the swap, so the live pathname is
# never without a complete database for an instant and the original stays
# recoverable until the replacement is durably in place. See `_swap_in`.
REBUILD_BACKUP_SUFFIX = ".rebuild-old"
# Two rebuilds share one REBUILD_SUFFIX path and would destroy each other's
# copy mid-write. flock is advisory and process-scoped, which is exactly the
# scope of the collision: two terminals, or an impatient second invocation.
REBUILD_LOCK_SUFFIX = ".rebuild-lock"

# Per-connection, so opencode's own connections keep their own (unlimited)
# setting; this only stops OUR transactions from leaving a huge WAL behind.
JOURNAL_SIZE_LIMIT_BYTES = 256 * 1024 ** 2

SIDECAR_SUFFIXES = ("-wal", "-shm")

# lsof is the only portable way to ask who has the file open, and it is not on
# the bare PATH launchd and systemd hand a job (macOS keeps it in /usr/sbin).
HOLDER_LOOKUP_TIMEOUT_S = 30.0

# Deleting in smaller batches with a pause between them hands the write lock
# back to opencode: WAL lets readers run concurrently, but opencode instances
# are writers and writers serialise, so one that exhausts its own busy_timeout
# against our transaction dies with "Failed to execute statement".
DEFAULT_BATCH = 25
DEFAULT_BATCH_SLEEP_MS = 1000


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
    # PASSIVE checkpoints run between batches, and the WAL pages they folded.
    checkpoints: int = 0
    pages_checkpointed: int = 0
    # The failure that stopped the run, if any. Whatever committed before it
    # is already durable, so `rows` is still authoritative for those batches.
    failure: str | None = None

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
    # incremental_vacuum could not move a page. The remainder stays in the
    # file and re-running will not change that, unlike a deadline or a budget.
    vacuum_stalled: bool = False
    # WAL pages folded back into the main database, and under which mode. In
    # WAL mode a released page is not an off-disk byte until this runs.
    wal_checkpoint_mode: str = ""
    wal_pages_checkpointed: int = 0
    # A checkpoint that could not fold the whole WAL because a reader or
    # writer was still using part of it. Not an error: PASSIVE does what it
    # can and yields, which is the entire reason it is safe to run.
    wal_checkpoint_busy: bool = False
    wal_bytes_before: int = 0
    wal_bytes_after: int = 0
    # Processes holding the database, ourselves excluded. None means it could
    # not be determined at all, which is not the same as nobody and never
    # treated as such.
    holders_before: list | None = None
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
        if not read_only:
            # Bound the WAL our own transactions leave behind. This is a
            # per-connection setting: it does not constrain opencode's
            # connections, only the one deleting millions of rows.
            conn.execute(f"PRAGMA journal_size_limit={JOURNAL_SIZE_LIMIT_BYTES}")
    except Exception:
        conn.close()
        raise
    return conn


def db_holders(db: Path) -> list[str] | None:
    """PIDs holding `db` or its sidecars open, ourselves excluded.

    Returns None when holders CANNOT BE DETERMINED -- lsof absent, or too slow
    to answer. That is deliberately distinct from `[]`, "determined: nobody":
    the rebuild destroys the file it swaps, so it must fail closed. launchd and
    systemd start jobs with a bare environment, and macOS keeps lsof in
    /usr/sbin, so "not on PATH" is the expected failure, not an exotic one. A
    missing lsof reading as idle would swap the file out from under live
    writers.

    An empty list is only a snapshot: a process can attach a millisecond later,
    and nothing here prevents that. It is a PRE-FLIGHT CHECK against an
    operator who believes they closed everything and did not -- it never
    establishes that the database is quiescent. The rebuild's correctness rests
    on the operator's precondition; this catches the common mistake cheaply.

    Every per-path lookup must be understood or the whole answer is `None`.
    Measured on macmini (lsof 4.91): a path that does not exist exits 1 with
    `lsof: status error on <path>: No such file or directory` on stderr and
    nothing on stdout. Reading stdout alone scores that as "no holders", and a
    missing `-shm` is routine, so one absent sidecar silently downgraded the
    whole database to "idle" -- the fail-open this fails closed on. Note the
    `--` separator is accepted by that lsof; the defect was never the argv.

    A missing sidecar is skipped, being an ordinary state of a healthy
    database. A missing MAIN file is not: the caller asked about a database
    that is not there, and "nobody holds it" is not a safe thing to tell a
    caller about to destroy something.
    """
    me = str(os.getpid())
    pids: set[str] = set()
    try:
        if not db.exists():
            return None
    except OSError:
        return None
    for path in (db, *(db.with_name(db.name + s) for s in SIDECAR_SUFFIXES)):
        try:
            if path != db and not path.exists():
                continue
        except OSError:
            return None
        try:
            proc = subprocess.run(
                ["lsof", "-t", "--", str(path)],
                capture_output=True, text=True, timeout=HOLDER_LOOKUP_TIMEOUT_S,
                check=False,   # a nonzero status is data here, not an error
            )
        except (OSError, subprocess.SubprocessError):
            # FileNotFoundError (no lsof), TimeoutExpired, and anything else
            # that stops it answering all mean the same thing: unknown.
            return None
        if proc.stderr.strip():
            # lsof reports permission problems and unreadable paths here while
            # still exiting 1 with empty stdout, which is indistinguishable
            # from "no matches" unless stderr is read.
            return None
        if proc.returncode not in (0, 1):
            return None
        found = proc.stdout.split()
        if proc.returncode == 1 and found:
            # Exit 1 means "no matching files"; PIDs alongside it mean the
            # output is not what this parser assumes it is.
            return None
        if not all(p.isdigit() for p in found):
            return None
        pids.update(p for p in found if p != me)
    return sorted(pids)


@dataclass
class CheckpointOutcome:
    """One WAL checkpoint. `busy` means part of the WAL could not be folded."""

    mode: str
    pages_checkpointed: int = 0
    wal_pages: int = 0
    busy: bool = False


def checkpoint_wal(conn: sqlite3.Connection, mode: str) -> CheckpointOutcome:
    """Fold the WAL back into the main database, without ever waiting.

    PASSIVE does what it can and yields. TRUNCATE and RESTART instead wait for
    readers and block writers while they hold the WAL -- and they wait for the
    connection's whole `busy_timeout`, which this tool sets to 30 seconds.
    Measured: with one reader holding an older snapshot, `wal_checkpoint`
    (TRUNCATE) returned busy after 31.85s; the same call with `busy_timeout=0`
    returned the identical busy result in 0.0s.

    That matters because `checkpoint_mode_for` can only ever act on a snapshot
    of who holds the database. A process attaching between that snapshot and
    this call would otherwise turn a checkpoint chosen as safe into a 30-second
    stall for every opencode writer queued behind it. So the timeout is
    suspended for the duration and restored afterwards: a blocking mode that
    cannot get the WAL gives up immediately and reports `busy` instead, and the
    frames fold on a later run.

    `PRAGMA wal_checkpoint` returns (busy, wal_pages, moved_pages). A database
    not in WAL mode answers (0, -1, -1) and is reported as zero pages moved,
    which is the truth: there is no WAL to fold.

    `busy` alone understates incompleteness. A PASSIVE checkpoint blocked by a
    reader's snapshot returns a ZERO busy flag while folding only part of the
    WAL -- measured on a fixture: `(0, 204, 104)`, half the frames left, and
    reported as a clean sweep. `moved < wal_pages` is the other way a
    checkpoint is incomplete, and an operator watching a WAL that never shrinks
    needs to be told about both.
    """
    previous = int(conn.execute("PRAGMA busy_timeout").fetchone()[0])
    conn.execute("PRAGMA busy_timeout=0")
    try:
        busy, wal_pages, moved = conn.execute(
            f"PRAGMA wal_checkpoint({mode})"
        ).fetchone()
    finally:
        conn.execute(f"PRAGMA busy_timeout={previous}")
    return CheckpointOutcome(
        mode=mode,
        pages_checkpointed=max(0, moved),
        wal_pages=max(0, wal_pages),
        busy=bool(busy) or (wal_pages > 0 and 0 <= moved < wal_pages),
    )


def checkpoint_mode_for(holders: list[str] | None) -> str:
    """PASSIVE unless the database is known to be idle.

    The blocking modes are permitted only on `[]` -- determined, and nobody.
    `None` is undeterminable and must not be optimistically read as idle, for
    the same reason the rebuild refuses on it.
    """
    return "TRUNCATE" if holders == [] else "PASSIVE"


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


def wal_bytes(db: Path) -> int:
    try:
        return db.with_name(db.name + "-wal").stat().st_size
    except OSError:
        return 0


def _order_descendant_first(
    deletable: set[str], parent_of: dict[str, str | None]
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
    rows: list[tuple], cutoff_ms: int, *, restrict_to: set[str] | None = None
) -> Selection:
    """Sessions whose entire subtree is older than the cutoff, deepest first.

    `rows` is (id, parent_id, time_updated) for the whole `session` table.

    `session.parent_id` has no foreign key, so deleting a parent whose child is
    still active would leave the child pointing at a row that no longer exists.
    A session is therefore only expired when it and every descendant are.

    A NULL `time_updated` is an unknown age, not an infinite one: such a
    session is kept and counted rather than treated as older than the cutoff.
    So is a non-numeric one. `time_updated` has INTEGER affinity, which SQLite
    applies as a preference and not a constraint: a value it cannot convert is
    stored verbatim as `text` or `blob` and read back as `str` or `bytes`.
    Comparing that against the cutoff raises TypeError, and it would raise
    under the write lock, after earlier batches had already committed
    irreversibly. An age that cannot be read is an unknown age, which this
    tool never deletes.

    `restrict_to` narrows the candidates to that set. Every other session in
    `rows` is then retained whatever its age, and still protects its ancestors
    -- which is what makes a mid-run recomputation safe: a session that
    appeared after the candidates were chosen was never authorised for
    deletion, so it counts as a live descendant even when it is old.
    """
    updated = {r[0]: r[2] for r in rows}
    parent_of: dict[str, str | None] = {r[0]: r[1] for r in rows}

    # int and float both carry a readable age; str, bytes and None do not.
    unknown = {sid for sid, t in updated.items() if not isinstance(t, (int, float))}
    old = {sid for sid, t in updated.items() if sid not in unknown and t < cutoff_ms}
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
    conn: sqlite3.Connection, cutoff_ms: int, *, restrict_to: set[str] | None = None
) -> Selection:
    rows = conn.execute("SELECT id, parent_id, time_updated FROM session").fetchall()
    return select_expired(rows, cutoff_ms, restrict_to=restrict_to)


def delete_sessions(
    conn: sqlite3.Connection,
    session_ids: list[str],
    *,
    cutoff_ms: int,
    batch: int,
    deadline: float | None,
    clock=time.monotonic,
    sleep_ms: int = 0,
    sleep=time.sleep,
) -> DeleteOutcome:
    """Delete sessions and all their rows, children first, in batches.

    Batched so a single statement never holds the write lock across millions of
    rows while opencode itself is running.

    Between batches the lock is handed back deliberately: a PASSIVE checkpoint
    folds what the batch just wrote back into the main database, and
    `sleep_ms` then pauses before the next `BEGIN IMMEDIATE`. WAL keeps readers
    out of the way, but opencode instances are writers and writers serialise --
    a concurrent writer that exhausts its own `busy_timeout` queued behind our
    transaction fails outright, so the pause is what gives it a window to win
    the lock. The checkpoint is PASSIVE precisely because it must never be the
    thing that blocks that writer. `sleep_ms=0` disables the pause.

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
    destroyed instead of losing the counts to a traceback. That covers the
    whole of each iteration, the deadline check and the bookkeeping included:
    Ctrl-C is how an operator stops this tool, and an interrupt landing between
    the guarded blocks would destroy rows and report nothing about them. Row
    counts are accumulated per transaction and merged into `rows` only once its
    COMMIT has returned: a rolled-back DELETE undid its rows, and reporting
    them as destroyed would misdirect the very recovery the counts exist to
    inform.

    An interrupt raised by the COMMIT call itself, from a transaction that is
    already durable, is recorded and then stops the run -- see `_commit_batch`.
    The few bytecodes between that merge and the next guarded statement are the
    one window left: an interrupt there is still caught, but lands after
    `rows` is correct and before `pending` shrinks, so the run overstates what
    is left rather than understating what was destroyed.

    Every iteration must shrink `pending`, by deleting candidates or by
    dropping the ones the lock found ineligible. A batch that does neither is
    reported as a failure rather than retried: against a live database the
    retry is an endless loop taking and releasing the write lock, which
    starves opencode and never terminates to report anything at all.

    The schema guard is this function's own precondition, enforced here rather
    than trusted to the caller, and re-enforced inside every batch's
    transaction. `verify_usable` before the loop describes the schema at one
    unlocked instant; the lock is deliberately released between batches, so an
    opencode migration adding a session child can land after it and be orphaned
    by the batches that follow. Revalidating under the write lock is what makes
    the refusal true of the schema each transaction actually deletes from.
    """
    verify_usable(conn)
    tables = [(t, c) for t, c in CHILD_TABLES if _table_exists(conn, t)]
    outcome = DeleteOutcome(rows={t: 0 for t, _ in tables} | {"session": 0})
    order_index = {sid: i for i, sid in enumerate(session_ids)}
    pending = set(session_ids)

    while pending:
        # Everything a batch does is inside the guarded region, the deadline
        # check and the bookkeeping included. Ctrl-C is how an operator stops
        # this tool, and by the second batch there are already committed
        # deletions that only this outcome can account for.
        began = False
        committed = {t: 0 for t, _ in tables} | {"session": 0}
        try:
            if deadline is not None and clock() > deadline:
                outcome.deadline_reached = True
                break
            # BEGIN IMMEDIATE is guarded too: it is the statement most likely
            # to fail, with SQLITE_BUSY, when opencode holds the write lock.
            conn.execute("BEGIN IMMEDIATE")
            began = True
            # Under the lock, and before this batch deletes anything: the
            # schema a migration may have changed since the last COMMIT is the
            # schema this transaction is about to delete from.
            try:
                verify_usable(conn)
            except RuntimeError as exc:
                _rollback_quietly(conn)
                began = False
                outcome.failure = (
                    f"the schema changed mid-run and this batch was rolled "
                    f"back: {exc}"
                )
                break
            # Re-read under the lock too: a known child table created by a
            # migration mid-run is one the pre-loop list does not have, and
            # every later batch would leave its rows behind.
            tables = [(t, c) for t, c in CHILD_TABLES if _table_exists(conn, t)]
            committed = {t: 0 for t, _ in tables} | {"session": 0}
            eligible = expired_session_ids(conn, cutoff_ms, restrict_to=pending).deletable
            doomed = eligible[:batch]
            # Anything left over is ineligible in the graph under this lock:
            # revived, newly sheltering a live descendant, or cyclic.
            ineligible = pending - set(eligible)
            if doomed:
                marks = ",".join("?" * len(doomed))
                for table, column in tables:
                    cur = conn.execute(
                        f"DELETE FROM {table} WHERE {column} IN ({marks})", doomed
                    )
                    committed[table] += max(0, cur.rowcount)
                cur = conn.execute(f"DELETE FROM session WHERE id IN ({marks})", doomed)
                committed["session"] += max(0, cur.rowcount)
            interrupted = _commit_batch(conn)
            # Past COMMIT: the deletes are durable, so their counts are now
            # facts. One dict.update rather than a `+=` per table, so an
            # interrupt cannot leave `rows` describing half a batch, and the
            # counts land before `pending` shrinks: an interrupt in between
            # overstates what is left rather than understating what was
            # destroyed.
            began = False
            outcome.rows.update({
                t: outcome.rows.get(t, 0) + n for t, n in committed.items()
            })
            outcome.skipped.extend(sorted(ineligible, key=order_index.__getitem__))
            remaining_before = len(pending)
            pending = pending - set(doomed) - ineligible
            if len(pending) == remaining_before:
                # Every batch either deletes candidates or drops the ones the
                # lock found ineligible, so this cannot happen -- and if it
                # ever does, the alternative is an endless loop taking and
                # releasing the write lock on a live database. Stop and say so.
                outcome.failure = (
                    f"RuntimeError: batch made no progress on {remaining_before} "
                    "remaining candidate(s); stopping rather than looping"
                )
                break
            if interrupted is not None:
                # The COMMIT itself returned and was then interrupted. Its
                # rows are gone and are now recorded, so the interrupt stops
                # the run rather than discarding the batch that caused it.
                outcome.failure = f"{type(interrupted).__name__}: {interrupted}"
                break
            if pending:
                # Outside the transaction, so neither of these holds the write
                # lock: fold this batch's WAL frames back, then stand aside
                # long enough for a queued opencode writer to take the lock.
                ckpt = checkpoint_wal(conn, "PASSIVE")
                outcome.pages_checkpointed += ckpt.pages_checkpointed
                outcome.checkpoints += 1
                if sleep_ms:
                    sleep(sleep_ms / 1000.0)
        except (sqlite3.Error, KeyboardInterrupt, MemoryError, OSError) as exc:
            if began:
                _rollback_quietly(conn)
            outcome.failure = f"{type(exc).__name__}: {exc}"
            break
        except BaseException:
            if began:
                _rollback_quietly(conn)
            raise

    outcome.remaining = len(pending)
    return outcome


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return bool(
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()
    )


def _commit_batch(conn: sqlite3.Connection) -> BaseException | None:
    """COMMIT, handing back an interrupt that arrived once it had returned.

    A batch spends nearly all of its time inside SQLite's C code, so that is
    where an interrupt lands: CPython runs the signal handler when the call
    returns, and `conn.execute("COMMIT")` raises KeyboardInterrupt from a
    transaction that is already durable. Letting that propagate would discard
    the counts for rows this process has just destroyed. Only the caller holds
    those counts, so the exception is returned for it to record and then stop
    on, rather than raised here.

    `in_transaction` separates the two cases: still in a transaction means the
    COMMIT did not happen and the interrupt is a plain failure.
    """
    try:
        conn.execute("COMMIT")
    except (KeyboardInterrupt, MemoryError) as exc:
        if conn.in_transaction:
            raise
        return exc
    return None


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
    tables = [(t, c) for t, c in CHILD_TABLES if _table_exists(conn, t)]
    counts: dict[str, int] = {t: 0 for t, _ in tables}
    counts["session"] = len(session_ids)
    for start in range(0, len(session_ids), batch):
        chunk = session_ids[start:start + batch]
        marks = ",".join("?" * len(chunk))
        for table, column in tables:
            counts[table] += conn.execute(
                f"SELECT count(*) FROM {table} WHERE {column} IN ({marks})", chunk
            ).fetchone()[0]
    return counts


@dataclass
class VacuumOutcome:
    """What one reclamation pass actually managed to hand back.

    `target` is what the freelist held when the pass started -- everything
    genuinely reclaimable -- and never the ceiling a `--vacuum-pages` budget
    put on this run. So a non-zero `remaining` always means what the flag
    documentation says it means: pages that were eligible are still in the
    file. Conflating the two let a capped run report a full freelist as
    reclaimed.
    """

    released: int = 0
    target: int = 0
    deadline_reached: bool = False
    # incremental_vacuum returned without moving a page. The rest of the
    # freelist is still in the file and this mechanism cannot shift it, so
    # re-running continues nothing -- unlike a deadline or a page budget.
    stalled: bool = False

    @property
    def remaining(self) -> int:
        return max(0, self.target - self.released)


def run_incremental_vacuum(
    conn: sqlite3.Connection, *, pages: int | None, deadline: float | None,
    clock=time.monotonic,
) -> VacuumOutcome:
    """Release freed pages back to the filesystem.

    Bounded per call so a huge freelist cannot block the database for minutes.
    In WAL mode the released pages only leave the file once a checkpoint runs,
    which a long-lived reader can defer, so the count is of logical pages --
    not a promise of bytes off the disk.

    `pages` budgets this run; it does not redefine what is reclaimable. The
    two are tracked separately because `--vacuum-pages` exists to bound a
    cautious first pass on a nearly-full disk, and an operator reaching for it
    is the one who most needs to be told the file is still oversized.

    Stopping early is reported rather than swallowed, whether the cause is the
    deadline, the page budget or a stall: exiting 0 with a still-oversized
    file tells an operator the reclamation finished when it did not. Other
    failures propagate: no row is at stake here, and the caller already routes
    them into the report alongside the deletion counts that are.
    """
    before = int(conn.execute("PRAGMA freelist_count").fetchone()[0])
    if before == 0:
        return VacuumOutcome()
    step = 2000
    budget = before if pages is None else min(pages, before)
    outcome = VacuumOutcome(target=before)
    while outcome.released < budget:
        if deadline is not None and clock() > deadline:
            outcome.deadline_reached = True
            break
        conn.execute(
            f"PRAGMA incremental_vacuum({min(step, budget - outcome.released)})"
        )
        now = int(conn.execute("PRAGMA freelist_count").fetchone()[0])
        released = max(0, before - now)
        progressed = released - outcome.released
        # A concurrent writer can grow the freelist under us; that is not
        # negative progress to report, it is no progress.
        outcome.released = released
        if progressed <= 0:
            outcome.stalled = True
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


def live_bytes(stats: DbStats) -> int:
    """What a rebuilt copy would hold: allocated pages minus the freelist.

    This, not the file size, is what `VACUUM INTO` has to write. On macmini
    2026-09-12 the difference was 32.1 GiB of file against 6.2 GiB live, which
    is the whole reason the rebuild fits on a disk a plain VACUUM cannot.
    """
    return stats.page_size * max(0, stats.page_count - stats.freelist_count)


@dataclass
class RebuildOutcome:
    """One `VACUUM INTO` attempt.

    `skipped` and `failure` are deliberately different outcomes. A guard that
    refused or aborted left the database untouched and the next run may well
    succeed, so it is a skip and exits 0; only something that went wrong is a
    failure. A failure raised after the replacement was installed is still a
    failure, and names both files -- see `SwapOutcome`.
    """

    attempted: bool = False
    completed: bool = False
    seconds: float = 0.0
    holders_before: list | None = None
    bytes_written: int = 0
    skipped: str | None = None
    failure: str | None = None


def _rebuild_guard(
    db: Path, deadline: float, min_free_bytes: int, state: dict, *, clock=time.monotonic,
) -> callable:
    """A progress handler that aborts a long or space-hungry `VACUUM INTO`.

    SQLite calls this every N virtual-machine instructions and a non-zero
    return aborts the running statement. It is the only way to bound
    `VACUUM INTO`, which is otherwise one uninterruptible call that no
    Python-level deadline can reach -- the reason the 23-minute, 39.6 GB
    runaway could not be stopped by `--max-seconds`.

    The reason is recorded in `state` and 1 returned; nothing is raised.
    Verified on CPython 3.14: an exception raised inside a progress handler is
    discarded and the statement resurfaces as a bare
    `sqlite3.OperationalError: interrupted`, so an `except` for a custom abort
    type is dead code that can never match, and a deliberate abort would be
    misreported as a failure.

    Disk is polled at most every REBUILD_DISK_POLL_SECONDS: statvfs on every
    callback costs more than the vacuum it guards.
    """
    started = clock()
    state["next_disk_check"] = 0.0

    def guard() -> int:
        now = clock()
        if now > deadline:
            state["reason"] = (
                f"exceeded the {round(deadline - started)}s wall-clock cap"
            )
            return 1
        if now >= state["next_disk_check"]:
            state["next_disk_check"] = now + REBUILD_DISK_POLL_SECONDS
            try:
                free = shutil.disk_usage(db.parent).free
            except OSError as exc:
                state["reason"] = f"free space could not be measured: {exc}"
                return 1
            if free < min_free_bytes:
                state["reason"] = (
                    f"free space fell to {free:,} bytes, below the "
                    f"{min_free_bytes:,} byte floor"
                )
                return 1
        return 0

    return guard


def _fsync_path(path: Path) -> None:
    """Force `path` to durable storage, directories included.

    On darwin `fsync(2)` only hands the write to the drive, which may hold it
    in a volatile cache; `F_FULLFSYNC` is the documented barrier that does not.
    It is not available on every filesystem (SMB, some network mounts answer
    ENOTTY/ENOTSUP), so a refusal falls back to fsync rather than failing the
    swap -- the alternative is refusing to rebuild on an ordinary share.
    """
    fd = os.open(str(path), os.O_RDONLY)
    try:
        full_fsync = getattr(fcntl, "F_FULLFSYNC", None)
        if full_fsync is not None:
            try:
                fcntl.fcntl(fd, full_fsync)
                return
            except OSError:
                pass
        os.fsync(fd)
    finally:
        os.close(fd)


def _prepare_replacement(target: Path, source_stat: os.stat_result, journal_mode: str
                         ) -> str | None:
    """Give the rebuilt copy the source's mode, owner and journal mode.

    Returns a reason not to swap it in, or None.

    `VACUUM INTO` writes a brand-new file: it takes the process umask, not the
    source's permissions, and it comes out in the default DELETE journal mode
    whatever the source used. Measured on macmini: a 0600 WAL source produced a
    0644 DELETE copy. Swapping that in publishes session history to every local
    user and silently changes opencode's concurrency model -- in rollback-journal
    mode a long reader blocks writers instead of coexisting with them.

    Both are established here and then read back from a fresh connection,
    because the value that matters is the one that persisted in the file rather
    than the one the setting statement returned. Failure to establish either is
    a refusal, not a warning: the swap is destructive and there is no way to
    put the old file back once opencode has written to the new one.
    """
    try:
        os.chmod(target, stat.S_IMODE(source_stat.st_mode))
    except OSError as exc:
        return f"the rebuilt copy could not be given the source's permissions: {exc}"
    if (source_stat.st_uid, source_stat.st_gid) != (os.geteuid(), os.getegid()):
        try:
            os.chown(target, source_stat.st_uid, source_stat.st_gid)
        except OSError as exc:
            return (
                f"the rebuilt copy could not be given the source's ownership "
                f"(uid {source_stat.st_uid}, gid {source_stat.st_gid}): {exc}"
            )

    if journal_mode.casefold() == "wal":
        # Set WAL and close: the mode is persistent in the file header, and a
        # clean close leaves no sidecar behind for the swap to worry about.
        conn = sqlite3.connect(str(target), isolation_level=None)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
        finally:
            conn.close()

    problem = _verify_rebuilt(target, journal_mode)
    if problem:
        return problem
    mode = stat.S_IMODE(target.stat().st_mode)
    if mode != stat.S_IMODE(source_stat.st_mode):
        return (
            f"the rebuilt copy is mode {mode:04o}, not the source's "
            f"{stat.S_IMODE(source_stat.st_mode):04o}"
        )
    return None


@dataclass
class SwapOutcome:
    """The result of installing the rebuilt copy.

    `refusal` and `failure` are different outcomes and the difference is where
    the database ended up. A refusal happened BEFORE the rename: the original
    is still at the live pathname, untouched, and a later run may well succeed,
    so it exits 0. A failure means something went wrong, and when
    `installed` is true it went wrong with the replacement ALREADY at the live
    pathname -- which is the one state an operator has to be told about by
    name, because only they can decide what to do with the preserved original.
    """

    installed: bool = False
    refusal: str | None = None
    failure: str | None = None


def _swap_in(db: Path, target: Path) -> SwapOutcome:
    """Install `target` at `db`, assuming nothing else has the database open.

    Pure filesystem work: the caller has already folded the WAL, checked
    `data_version` and CLOSED its connection, so by the time this runs the
    database at `db` is a self-contained file that no connection in this
    process is holding.

    WHAT MAKES THIS SAFE IS THE PRECONDITION, NOT THIS FUNCTION
    -----------------------------------------------------------
    This is a file replacement, and it is correct exactly when nothing else has
    the database open. It does not defend against a concurrent writer and does
    not try to: three earlier designs did, and each one closed a window while
    opening another (a commit landing in the replacement's `-wal` after the
    lock was released; stragglers that opened the replacement after a first
    rename; a pre-existing hard link falsifying "the old inode has exactly one
    pathname"; a crash between the rename and the sidecar cleanup leaving two
    main files sharing one `db-wal` name). The operator guarantees quiescence
    instead, and the guards in `rebuild_database` merely catch the case where
    they were mistaken.

    The ordering is what is left of that, and it is chosen so that the rename
    is the LAST mutating step:

    1. The sidecars are removed first, while the WAL is known to be empty --
       the caller folded it with a TRUNCATE checkpoint and refused if any frame
       could not be moved, and then closed the connection, which folds again.
       A `-wal` that carries no committed frame is not data, and removing it
       before the rename is what stops a crash mid-swap from leaving the NEW
       main file paired with the OLD file's `-wal`. SQLite would replay that
       WAL over the replacement.
    2. The original is hard-linked aside, so it survives the rename and stays
       recoverable until the replacement is durable.
    3. `os.replace` is atomic: the live pathname holds either the whole old
       database or the whole new one, never neither.

    Every failure up to and including step 3 therefore leaves a complete,
    openable database at `db` with every committed row. A failure AFTER it
    leaves the new database installed and the old one at the backup name, and
    says so with both paths in the message.
    """
    backup = db.with_name(db.name + REBUILD_BACKUP_SUFFIX)
    if backup.exists():
        return SwapOutcome(refusal=(
            f"{backup} already exists; a previous rebuild may have been "
            "interrupted mid-swap. Check it against the live database and "
            "remove it by hand before rebuilding again"
        ))

    try:
        _fsync_path(target)
    except OSError as exc:
        return SwapOutcome(failure=f"the rebuilt copy could not be flushed to disk: {exc}")

    # Before the rename, and only because the WAL is empty by now: a sidecar
    # surviving the swap describes the inode being retired, and SQLite replays
    # it over the replacement.
    for suffix in SIDECAR_SUFFIXES:
        _unlink_quietly(db.with_name(db.name + suffix))

    try:
        os.link(db, backup)
    except OSError as exc:
        return SwapOutcome(
            failure=f"the old database could not be preserved before the swap: {exc}"
        )

    try:
        os.replace(target, db)
    except OSError as exc:
        _unlink_quietly(backup)
        return SwapOutcome(failure=f"the rebuilt copy could not be swapped in: {exc}")

    # Past here the replacement IS the database. Nothing below can be reported
    # as a skip: the swap has happened and only its durability is in question.
    try:
        _fsync_path(db.parent)
    except OSError as exc:
        return SwapOutcome(installed=True, failure=(
            f"the rebuilt database is installed at {db} but the directory "
            f"entry could not be made durable: {exc}. The previous database is "
            f"preserved at {backup}; if the host loses power before the "
            f"filesystem flushes, the rename may not survive. Verify {db} and "
            f"remove {backup} by hand"
        ))

    try:
        backup.unlink()
    except OSError as exc:
        return SwapOutcome(installed=True, failure=(
            f"the rebuilt database is installed at {db}, but the preserved "
            f"original at {backup} could not be removed: {exc}. Remove it by "
            "hand; until it is gone every later rebuild refuses"
        ))
    return SwapOutcome(installed=True)


class _RebuildLock:
    """Single-instance guard around one database's rebuild.

    `flock(LOCK_EX|LOCK_NB)` on a lock file beside the database. Advisory and
    per-open-file-description, so the kernel releases it if the process dies
    however abruptly -- a stale lock file cannot wedge the next run, which is
    the failure mode a PID file would have.

    The lock file is deliberately never unlinked: removing it races another
    process that has already opened it and is about to flock the now-orphaned
    inode, which would let both proceed. It is an empty file.
    """

    def __init__(self, db: Path) -> None:
        self._path = db.with_name(db.name + REBUILD_LOCK_SUFFIX)
        self._fd: int | None = None

    def acquire(self) -> bool:
        fd = os.open(str(self._path), os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            return False
        self._fd = fd
        return True

    def release(self) -> None:
        if self._fd is None:
            return
        try:
            os.close(self._fd)   # closing drops the flock
        except OSError:
            pass
        self._fd = None


def rebuild_database(
    db: Path,
    stats: DbStats,
    *,
    max_seconds: float,
    min_free_bytes: int,
    clock=time.monotonic,
) -> RebuildOutcome:
    """Compact the database with `VACUUM INTO` and install the copy.

    THE PRECONDITION
    ----------------
    Nothing else may have the database open. This is the operator's
    responsibility -- they stopped opencode before running it -- and it is the
    reason the function is reachable only from an explicit `rebuild`
    subcommand behind an explicit confirmation flag, never from a prune and
    never from a timer.

    The guards below are pre-flight checks against a mistaken operator, not a
    proof of exclusion. Each is nearly free and catches the realistic error
    ("I thought I had closed everything"); none of them makes the swap safe
    against a writer that is genuinely running. In order:

    1. The single-instance `flock`. Two rebuilds share one `.rebuild-tmp` path
       and would each destroy the other's copy mid-write.
    2. `lsof`: if anything holds the database, refuse and NAME the pids, so the
       operator can go and close them. Undeterminable (`None`) is also a
       refusal -- a missing lsof must not read as idle.
    3. `st_nlink != 1`: a second hard link is a second public pathname to this
       inode, so "nothing is using it" cannot be established by looking at the
       one name we were given, and the rename leaves the other name pointing at
       the stale original. Refuse.
    4. Free space: the copy must fit AND still leave the floor free.
    5. `PRAGMA data_version` before and after the `VACUUM INTO`. This is the
       one check that detects a COMMIT rather than an attachment: it changes
       whenever another connection commits, and is stable across our own work.
       A rebuild takes about a minute on the real store, which is long enough
       for a forgotten agent to write, and an attachment check cannot see a
       writer that came and went inside it.
    6. The call itself is bounded from INSIDE SQLite, by wall clock and by free
       space, and the partial copy is unlinked on every abort path. This is
       unrelated to concurrency: it is the 23-minute, 39.6 GB runaway of
       2026-09-12, which no Python-level deadline could reach.
    7. The copy is verified before it is trusted: `quick_check` ok, auto_vacuum
       still INCREMENTAL, and the source's journal mode, permissions and
       ownership established on it and read back from a fresh connection.
    8. The source WAL is folded and this connection is CLOSED before the swap,
       so the file being replaced is self-contained and we are not ourselves a
       holder of it.

    Everything up to the rename fails closed, leaving the original in place.
    """
    outcome = RebuildOutcome()

    lock = _RebuildLock(db)
    try:
        if not lock.acquire():
            outcome.skipped = (
                "another opencode-gc rebuild is running against this database; "
                "they share one temporary copy and must not overlap"
            )
            return outcome
    except OSError as exc:
        outcome.skipped = f"the rebuild lock could not be taken: {exc}"
        return outcome

    try:
        return _rebuild_locked(
            db, stats, max_seconds=max_seconds, min_free_bytes=min_free_bytes,
            clock=clock, outcome=outcome,
        )
    finally:
        lock.release()


def rebuild_preflight(
    db: Path, stats: DbStats, *, min_free_bytes: int
) -> str | None:
    """Reasons not to start a rebuild, or None. Touches nothing.

    Shared by the real run and by `--dry-run`, so a preview reports the same
    refusals the run would hit rather than a second implementation of them.
    """
    holders = db_holders(db)
    if holders is None:
        return (
            "cannot determine who holds the database (lsof is missing from "
            "PATH, or too slow to answer); refusing rather than assuming it is "
            "idle"
        )
    if holders:
        return (
            f"{len(holders)} process(es) still hold the database "
            f"[{','.join(holders[:8])}]. Stop opencode and any running agents, "
            "then try again"
        )

    try:
        link_count = db.stat().st_nlink
    except OSError as exc:
        return f"the database could not be inspected: {exc}"
    if link_count != 1:
        # Another name for this inode is another way in, and one this tool was
        # never told about: `lsof` on the path it was given cannot enumerate
        # who may come through the other. The rename also only moves OUR name
        # -- the other one keeps pointing at the un-rebuilt original, which
        # then silently diverges from the live database.
        return (
            f"{db} has {link_count} hard links, so it is reachable under "
            "another name this tool cannot see. A process can open it through "
            "that name at any time, and the rebuild only replaces this one: "
            "the other name would keep the old database. Remove the extra "
            "link, or rebuild the file it points at"
        )

    needed = int(live_bytes(stats) * REBUILD_COPY_HEADROOM)
    try:
        free = shutil.disk_usage(db.parent).free
    except OSError as exc:
        return f"free space on {db.parent} could not be measured: {exc}"
    # `free - needed >= floor`, not `free >= needed`: a rebuild that fits
    # exactly still drives the filesystem to the edge while it runs.
    if free - needed < min_free_bytes:
        return (
            f"a rebuild needs ~{needed:,} bytes and must leave {min_free_bytes:,} "
            f"free; only {free:,} available on {db.parent}"
        )
    return None


def _rebuild_locked(
    db: Path,
    stats: DbStats,
    *,
    max_seconds: float,
    min_free_bytes: int,
    clock,
    outcome: RebuildOutcome,
) -> RebuildOutcome:
    outcome.holders_before = db_holders(db)
    refusal = rebuild_preflight(db, stats, min_free_bytes=min_free_bytes)
    if refusal:
        outcome.skipped = refusal
        return outcome

    try:
        source_stat = db.stat()
    except OSError as exc:
        outcome.failure = f"the database could not be inspected: {exc}"
        return outcome

    target = db.with_name(db.name + REBUILD_SUFFIX)
    state: dict = {}
    outcome.attempted = True
    try:
        target.unlink(missing_ok=True)
    except OSError as exc:
        outcome.attempted = False
        outcome.failure = f"could not clear {target}: {exc}"
        return outcome

    try:
        conn = connect(db, read_only=False)
    except (sqlite3.Error, OSError, ValueError) as exc:
        outcome.failure = f"{type(exc).__name__}: {exc}"
        return outcome

    # Everything from here owns `target` and the open connection. One
    # try/finally covers the whole lifetime of both: an interrupt during
    # verification or the swap would otherwise leave tens of gigabytes behind.
    installed = False
    closed = False
    try:
        try:
            journal_mode = str(conn.execute("PRAGMA journal_mode").fetchone()[0])
            version = int(conn.execute("PRAGMA data_version").fetchone()[0])
        except sqlite3.Error as exc:
            outcome.failure = f"the source database could not be read: {exc}"
            return outcome

        try:
            guard = _rebuild_guard(
                db, clock() + max_seconds, min_free_bytes, state, clock=clock
            )
            conn.set_progress_handler(guard, REBUILD_PROGRESS_INSTRUCTIONS)
            started = clock()
            try:
                conn.execute("VACUUM INTO ?", (str(target),))
            finally:
                conn.set_progress_handler(None, 0)
                outcome.seconds = round(clock() - started, 3)
        except (sqlite3.Error, OSError) as exc:
            if state.get("reason"):
                # A guard abort arrives here as sqlite3.OperationalError
                # ("interrupted") because CPython discards whatever a progress
                # handler raises. The recorded reason is what distinguishes a
                # deliberate skip from a real failure.
                outcome.skipped = f"rebuild aborted: {state['reason']}"
            else:
                outcome.failure = f"{type(exc).__name__}: {exc}"
            return outcome

        try:
            problem = _prepare_replacement(target, source_stat, journal_mode)
        except (sqlite3.Error, OSError) as exc:
            outcome.failure = f"the rebuilt copy could not be checked: {exc}"
            return outcome
        if problem:
            outcome.failure = problem
            return outcome

        try:
            outcome.bytes_written = target.stat().st_size
        except OSError as exc:
            outcome.failure = f"the rebuilt copy could not be measured: {exc}"
            return outcome

        # The quiescence check that is not an attachment check. A rebuild takes
        # about a minute on the real store; an opencode that started, committed
        # and exited inside it leaves no trace in any holder snapshot, and its
        # rows are not in the copy about to be installed.
        try:
            current = int(conn.execute("PRAGMA data_version").fetchone()[0])
        except sqlite3.Error as exc:
            outcome.failure = f"the source database could not be re-read: {exc}"
            return outcome
        if current != version:
            outcome.skipped = (
                "another connection committed while the rebuild ran "
                f"(data_version {version} -> {current}); the copy predates that "
                "write and installing it would discard it. Something is still "
                "using the database"
            )
            return outcome

        # Fold the WAL so the file about to be replaced is self-contained, and
        # so the sidecars the swap removes carry no committed frame.
        try:
            folded = checkpoint_wal(conn, "TRUNCATE")
        except sqlite3.Error as exc:
            outcome.failure = (
                f"the source WAL could not be checkpointed before the swap: {exc}"
            )
            return outcome
        if folded.busy:
            outcome.skipped = (
                "the source WAL could not be fully folded before the swap "
                f"({folded.pages_checkpointed} of {folded.wal_pages} page(s) "
                "moved); another connection is still using part of it"
            )
            return outcome

        # Closed before the swap: our own open handle would otherwise be a
        # holder of the inode being retired, and closing in WAL mode writes.
        try:
            conn.close()
            closed = True
        except sqlite3.Error as exc:
            outcome.failure = f"the source database could not be closed: {exc}"
            return outcome

        try:
            swap = _swap_in(db, target)
        except OSError as exc:
            outcome.failure = f"the swap failed: {type(exc).__name__}: {exc}"
            return outcome
        installed = swap.installed
        if swap.refusal:
            # Nothing was renamed: the original is still at the live pathname.
            outcome.skipped = swap.refusal
            return outcome
        if swap.failure:
            outcome.failure = swap.failure
            return outcome
        outcome.completed = True
        return outcome
    finally:
        if not installed:
            _unlink_quietly(target)
        if not closed:
            try:
                conn.close()
            except sqlite3.Error:
                pass


def _verify_rebuilt(target: Path, journal_mode: str = "") -> str | None:
    """Reasons not to trust the rebuilt copy, or None if it is sound.

    Opened fresh so every answer is what persisted in the file rather than what
    a setting statement returned on the connection that made it.
    """
    check = sqlite3.connect(str(target), isolation_level=None)
    try:
        integrity = check.execute("PRAGMA quick_check").fetchone()[0]
        auto_vacuum = int(check.execute("PRAGMA auto_vacuum").fetchone()[0])
        actual_journal = str(check.execute("PRAGMA journal_mode").fetchone()[0])
    finally:
        check.close()
    if integrity != "ok":
        return f"the rebuilt copy failed quick_check: {integrity}"
    if auto_vacuum != 2:
        # Swapping this in would cost the store its ability to reclaim
        # anything in place, and only another full rebuild could restore it.
        return (
            "the rebuilt copy came out auto_vacuum="
            f"{AUTO_VACUUM_NAMES.get(auto_vacuum, auto_vacuum)}, not INCREMENTAL"
        )
    if journal_mode and actual_journal.casefold() != journal_mode.casefold():
        return (
            f"the rebuilt copy came out journal_mode={actual_journal}, not the "
            f"source's {journal_mode}"
        )
    return None


def _unlink_quietly(path: Path) -> None:
    """Remove the partial copy. SQLite leaves it behind on an aborted
    `VACUUM INTO`, and it is as large as the rebuild got."""
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


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


def deleted_tables() -> set[str]:
    """Every table this module deletes rows from, normalised for comparison."""
    return {"session"} | {t.casefold() for t, _ in CHILD_TABLES}


def orphaned_by_deletion(conn: sqlite3.Connection) -> set[str]:
    """Tables whose rows this module's deletions would strand, from the schema.

    Asked of the database rather than assumed, because `CHILD_TABLES` being
    complete is what makes deleting with `PRAGMA foreign_keys` off equivalent
    to deleting with the cascades on. A table opencode adds later would
    otherwise be orphaned by every session this tool deletes, silently, which
    is exactly what `todo`, `session_message`, `session_input` and
    `session_share` were before they were added to the list.

    The closure is rooted at every table in the deletion plan and walked until
    it stops growing, rather than checking one hop. That is for the MESSAGE,
    not for the decision: a table two hops out cannot exist without an
    intermediate table one hop out, and that intermediate is itself uncovered,
    so a one-hop check refuses the same schemas. What the closure adds is the
    whole chain in the refusal, so an operator is told every table they have to
    add to `CHILD_TABLES` instead of discovering them one run at a time.

    The casefolding IS load-bearing. SQLite identifiers are case-insensitive
    but `PRAGMA foreign_key_list` echoes the spelling used in the declaration,
    so `REFERENCES SeSsIoN(id)` is a real reference that a `== "session"` test
    does not recognise -- and the table hanging off it would be orphaned with
    no refusal at all. Everything here is compared casefolded.
    """
    # child table -> the parent tables it references.
    parents: dict[str, set[str]] = {}
    for (name,) in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall():
        refs = {
            str(row[2]).casefold()
            for row in conn.execute(f'PRAGMA foreign_key_list("{name}")').fetchall()
        }
        if refs:
            parents[name.casefold()] = refs

    deleted = deleted_tables()
    # A table is orphaned if it references anything deleted, and once orphaned
    # it is itself a vanishing parent for whatever references it.
    vanishing = set(deleted)
    orphaned: set[str] = set()
    changed = True
    while changed:
        changed = False
        for child, refs in parents.items():
            if child in vanishing or not (refs & vanishing):
                continue
            orphaned.add(child)
            vanishing.add(child)
            changed = True
    return orphaned - deleted


def verify_usable(conn: sqlite3.Connection) -> None:
    """Confirm this is readable SQLite carrying the schema we delete from.

    Raises RuntimeError otherwise. Reaching the deletion loop and discovering
    there mid-run that a table is missing would leave a half-pruned database
    and a traceback; every table this tool touches is therefore probed first,
    while nothing has been written.

    The reverse direction matters just as much and is checked here too: a
    table the schema hangs off anything this tool deletes, and that
    `CHILD_TABLES` does not cover, would have its rows orphaned by every
    deletion, and nothing else would ever say so. Refusing is the only safe
    answer, since the alternative is destroying referential integrity quietly.
    """
    required = ["session"] + [t for t, _ in CHILD_TABLES if t in REQUIRED_CHILD_TABLES]
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

    try:
        uncovered = sorted(orphaned_by_deletion(conn))
    except sqlite3.DatabaseError as exc:
        raise RuntimeError(f"cannot read the schema: {exc}") from exc
    if uncovered:
        raise RuntimeError(
            f"table(s) {', '.join(uncovered)} reference a table this tool "
            "deletes from but are not in CHILD_TABLES, so deleting a session "
            "would orphan their rows. Add them to CHILD_TABLES, children "
            "first, before pruning this database"
        )

    # A table can exist without the columns this tool keys on.
    probes = [("session", "id"), ("session", "parent_id"), ("session", "time_updated")]
    probes += [(t, c) for t, c in CHILD_TABLES if t in present]
    for table, column in probes:
        try:
            cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        except sqlite3.DatabaseError as exc:
            raise RuntimeError(f"cannot read the schema of {table}: {exc}") from exc
        if column not in cols:
            raise RuntimeError(
                f"not an opencode database: {table}.{column} is missing"
            )


# The prune is what runs unattended, so it is what a bare invocation means and
# what every existing timer keeps invoking. `rebuild` has to be spelled out.
PRUNE_COMMAND = "prune"
REBUILD_COMMAND = "rebuild"
CONFIRM_FLAG = "--yes-nothing-is-using-it"

REBUILD_EPILOG = """\
REQUIRES A QUIESCENT DATABASE. Stop opencode and close every running agent
first. This replaces the database file: run it while opencode is live and
committed sessions can be lost. It refuses if it can see a holder, but that
check is a convenience for the common mistake, not a guarantee -- the
guarantee is yours.

  opencode-gc rebuild --dry-run                    # what it would reclaim
  opencode-gc rebuild --yes-nothing-is-using-it    # do it

There is deliberately no way to reach this from `prune`, and nothing on a
timer should ever invoke it.
"""


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="opencode-gc",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "commands:\n"
            f"  {PRUNE_COMMAND} (default)   delete expired sessions and reclaim "
            "pages. Safe to run\n"
            "                    while opencode is using the database, and the "
            "only\n                    thing a timer should ever invoke.\n"
            f"  {REBUILD_COMMAND}           compact the file with VACUUM INTO. "
            "MANUAL ONLY: requires\n"
            "                    that nothing else has the database open, and "
            f"an\n                    explicit {CONFIRM_FLAG}. Never\n"
            "                    run this from a timer.\n"
        ),
    )
    sub = ap.add_subparsers(dest="command")

    prune = sub.add_parser(
        PRUNE_COMMAND,
        help="delete expired sessions and reclaim pages (the default)",
        description="Delete expired sessions and reclaim freed pages. Safe to "
                    "run while opencode is using the database.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    prune.add_argument("--db", type=Path, default=DEFAULT_DB,
                       help=f"opencode database (default {DEFAULT_DB})")
    prune.add_argument("--retention-days", type=float, default=4.0,
                       help="keep sessions updated within this many days (default 4)")
    prune.add_argument("--apply", action="store_true",
                       help="actually delete; without this the run is a dry run")
    prune.add_argument("--batch", type=int, default=DEFAULT_BATCH,
                       help=f"sessions per transaction (default {DEFAULT_BATCH}; "
                            "clamped to SQLite's bound-variable limit)")
    prune.add_argument("--batch-sleep-ms", type=int, default=DEFAULT_BATCH_SLEEP_MS,
                       help="pause between batches, handing the write lock back to "
                            f"opencode (default {DEFAULT_BATCH_SLEEP_MS}; 0 disables)")
    prune.add_argument("--max-seconds", type=float, default=600.0,
                       help="stop starting new batches after this long (default "
                            "600); 0 means no limit. Not a bound on total runtime: "
                            "a batch or a VACUUM already in flight runs to completion")
    prune.add_argument("--vacuum-pages", type=int, default=None,
                       help="cap pages released per run, >= 1 "
                            "(default: the whole freelist)")
    prune.add_argument("--no-vacuum", action="store_true",
                       help="delete rows but do not release pages")
    prune.add_argument("--enable-incremental-vacuum", action="store_true",
                       help="switch auto_vacuum to INCREMENTAL. From FULL this is a "
                            "header change; from NONE it needs one full VACUUM and "
                            f"~{VACUUM_COPY_FACTOR}x the database size free")
    prune.add_argument("--json", action="store_true")

    rebuild = sub.add_parser(
        REBUILD_COMMAND,
        help="compact the file with VACUUM INTO. MANUAL ONLY: requires that "
             "nothing else has the database open",
        description="Compact the database with VACUUM INTO and install the "
                    "compacted copy.",
        epilog=REBUILD_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    rebuild.add_argument("--db", type=Path, default=DEFAULT_DB,
                         help=f"opencode database (default {DEFAULT_DB})")
    rebuild.add_argument(
        CONFIRM_FLAG, dest="confirmed", action="store_true",
        help="confirm that opencode is stopped and nothing else has this "
             "database open. Required to change anything; there is no --apply "
             "and no default-yes",
    )
    rebuild.add_argument(
        "--dry-run", action="store_true",
        help="report what a rebuild would reclaim -- current size, live bytes, "
             "expected saving -- and touch nothing",
    )
    rebuild.add_argument("--max-seconds", type=float,
                         default=DEFAULT_REBUILD_MAX_SECONDS,
                         help="abort the rebuild at the first SQLite progress "
                              "callback after this long (default "
                              f"{DEFAULT_REBUILD_MAX_SECONDS:g}); a statement "
                              "blocked in filesystem I/O runs no callbacks and "
                              "can overshoot this")
    rebuild.add_argument("--min-free-gib", type=float,
                         default=DEFAULT_REBUILD_MIN_FREE_GIB,
                         help="refuse to start a rebuild that would not leave "
                              "this much free, and abort one that drops below it "
                              f"(default {DEFAULT_REBUILD_MIN_FREE_GIB:g})")
    rebuild.add_argument("--json", action="store_true")
    return ap


def _argv_with_default_command(argv: list[str]) -> list[str]:
    """Insert the implicit `prune` so a bare `opencode-gc --apply` still prunes.

    The default runs in the direction that is safe: an operator who omits the
    command gets the prune, never the rebuild. `rebuild` is only ever reached
    by naming it.
    """
    for i, token in enumerate(argv):
        if token in (PRUNE_COMMAND, REBUILD_COMMAND):
            # A subcommand anywhere before the first option is the command; a
            # bare word after one is an argument to it and not ours to move.
            if all(a.startswith("-") for a in argv[:i]):
                return argv
            break
        if token in ("-h", "--help"):
            return argv
    return [PRUNE_COMMAND, *argv]


def main() -> int:
    ap = _build_parser()
    args = ap.parse_args(_argv_with_default_command(sys.argv[1:]))
    if args.command == REBUILD_COMMAND:
        return _rebuild_command(ap, args)
    return _prune_command(ap, args)


def _prune_command(ap: argparse.ArgumentParser, args) -> int:
    if not math.isfinite(args.retention_days) or args.retention_days < MIN_RETENTION_DAYS:
        ap.error(f"--retention-days must be finite and >= {MIN_RETENTION_DAYS}")
    if args.batch < 1:
        ap.error("--batch must be >= 1")
    if args.batch_sleep_ms < 0:
        ap.error("--batch-sleep-ms must be >= 0 (0 disables the pause)")
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
        conn = connect(db_path, read_only=not args.apply)
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
        res.wal_bytes_before = res.wal_bytes_after = wal_bytes(db_path)
        res.auto_vacuum_before = res.auto_vacuum_after = stats.auto_vacuum_name
        # Only asked when something will act on the answer. A dry run holds a
        # read-only handle and neither checkpoints nor rebuilds, so spawning
        # lsof for it would be pure cost. None means undeterminable, which
        # every consumer treats as "assume busy".
        holders = db_holders(db_path) if args.apply else None
        res.holders_before = holders
        if args.apply and holders is None:
            res.notes.append(
                "who holds the database could not be determined (lsof missing "
                "from PATH, or too slow); treating it as busy"
            )

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
                    sleep_ms=args.batch_sleep_ms,
                )
                res.rows_deleted = outcome.rows
                res.sessions_deleted = outcome.rows.get("session", 0)
                res.sessions_skipped_revalidation = len(outcome.skipped)
                res.sessions_remaining = outcome.remaining
                res.incomplete = outcome.incomplete
                res.deadline_reached = outcome.deadline_reached
                res.wal_pages_checkpointed += outcome.pages_checkpointed
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
                    res.vacuum_stalled = vac.stalled
                    if vac.remaining:
                        # Whatever stopped the pass, pages that were eligible
                        # are still in the file, so it is larger than a run
                        # reporting itself finished would imply.
                        res.incomplete = True
                        if vac.deadline_reached:
                            res.notes.append(
                                f"--max-seconds reached during page reclamation; "
                                f"{vac.remaining:,} of {vac.target:,} page(s) were not "
                                "released. Re-run to continue."
                            )
                        elif vac.stalled:
                            res.notes.append(
                                f"page reclamation stopped making progress with "
                                f"{vac.remaining:,} of {vac.target:,} page(s) still "
                                "on the freelist; they stay in the file. Re-running "
                                "will not release them."
                            )
                        else:
                            res.notes.append(
                                f"--vacuum-pages capped this run; {vac.remaining:,} of "
                                f"{vac.target:,} page(s) are still reclaimable. "
                                "Re-run to continue."
                            )
                elif current.freelist_count:
                    res.notes.append(
                        f"{current.freelist_count:,} pages are free but auto_vacuum is "
                        f"{current.auto_vacuum_name}; they stay in the file. Re-run with "
                        "--enable-incremental-vacuum to reclaim future deletes."
                    )

            if args.apply:
                # Without this the released pages stay in the file and the WAL
                # keeps whatever the deletes wrote into it -- 15.28 GiB of it
                # on vibes, never checkpointed. TRUNCATE only when the database
                # is known to be idle: it waits for readers and blocks writers
                # while it holds the WAL, and opencode instances are writers.
                ckpt = checkpoint_wal(conn, checkpoint_mode_for(holders))
                res.wal_checkpoint_mode = ckpt.mode
                res.wal_pages_checkpointed += ckpt.pages_checkpointed
                res.wal_checkpoint_busy = ckpt.busy
                if ckpt.busy:
                    res.notes.append(
                        f"the {ckpt.mode} checkpoint could not fold the whole WAL; "
                        "another connection was still using part of it. The rest "
                        "folds on a later run."
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
        except (sqlite3.Error, KeyboardInterrupt, OSError) as exc:
            # In WAL mode close() checkpoints, which on a large database is
            # real IO an operator can interrupt or a filesystem can fail.
            res.incomplete = True
            res.errors.append(f"closing the database failed: "
                              f"{type(exc).__name__}: {exc}")

    # No rebuild here, by design. A prune is the unattended path -- it is what
    # a timer invokes and what runs while opencode is live -- and the rebuild
    # replaces the database file, which is only correct when nothing else has
    # it open. It is a separate subcommand behind an explicit confirmation so
    # that no combination of flags on this path can reach it.
    try:
        res.bytes_after = on_disk_bytes(db_path)
        res.wal_bytes_after = wal_bytes(db_path)
    except (KeyboardInterrupt, OSError) as exc:
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


@dataclass
class RebuildReport:
    """What one `rebuild` invocation did, or would have done."""

    db: str
    dry_run: bool
    confirmed: bool = False
    bytes_before: int = 0
    bytes_after: int = 0
    live_bytes: int = 0
    expected_reclaim: int = 0
    auto_vacuum: str = ""
    holders: list | None = None
    attempted: bool = False
    completed: bool = False
    seconds: float = 0.0
    bytes_written: int = 0
    skipped: str | None = None
    errors: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    @property
    def bytes_reclaimed(self) -> int:
        return max(0, self.bytes_before - self.bytes_after)


def _rebuild_command(ap: argparse.ArgumentParser, args) -> int:
    """The manual rebuild. Nothing automated reaches this.

    Exit status: 0 for a completed rebuild, a preview, or a guard that refused
    (the database is untouched and a later run may succeed); 1 for a failure;
    2 for an unusable input or a missing confirmation.
    """
    if not math.isfinite(args.max_seconds) or args.max_seconds <= 0:
        # No "0 means unlimited" here: an unbounded VACUUM INTO is the exact
        # failure this cap exists to prevent.
        ap.error("--max-seconds must be finite and > 0")
    if not math.isfinite(args.min_free_gib) or args.min_free_gib < 0:
        ap.error("--min-free-gib must be finite and >= 0")
    if args.confirmed and args.dry_run:
        ap.error(f"--dry-run and {CONFIRM_FLAG} are mutually exclusive")
    if not args.confirmed and not args.dry_run:
        # Never a default-yes. An operator who has not said the words gets a
        # refusal naming them, and a --dry-run that costs nothing.
        print(
            "opencode-gc rebuild: this replaces the database file and requires "
            "that nothing else has it open.\n"
            "Stop opencode and close every running agent, then re-run with "
            f"{CONFIRM_FLAG}.\n"
            "To see what it would reclaim without touching anything: "
            "opencode-gc rebuild --dry-run",
            file=sys.stderr,
        )
        return 2
    if not args.db.is_file():
        print(f"opencode-gc: no database at {args.db}", file=sys.stderr)
        return 2
    db_path = args.db.resolve()

    report = RebuildReport(
        db=str(args.db), dry_run=args.dry_run, confirmed=args.confirmed
    )
    try:
        probe = connect(db_path, read_only=True)
    except (sqlite3.Error, RuntimeError, ValueError) as exc:
        print(f"opencode-gc: cannot open {args.db}: {exc}", file=sys.stderr)
        return 2
    try:
        verify_usable(probe)
        stats = read_stats(probe)
    except (RuntimeError, sqlite3.Error) as exc:
        probe.close()
        print(f"opencode-gc: cannot use {args.db}: {exc}", file=sys.stderr)
        return 2
    finally:
        try:
            probe.close()
        except sqlite3.Error:
            pass

    report.bytes_before = report.bytes_after = on_disk_bytes(db_path)
    report.live_bytes = live_bytes(stats)
    report.auto_vacuum = stats.auto_vacuum_name
    # What the file would lose: everything on the freelist. The copy is the
    # live pages plus the WAL it folds, so the saving is measured from the
    # real on-disk footprint rather than from page arithmetic alone.
    report.expected_reclaim = max(0, report.bytes_before - report.live_bytes)
    min_free_bytes = int(args.min_free_gib * 1024 ** 3)

    if stats.auto_vacuum != 2:
        report.skipped = (
            f"auto_vacuum is {stats.auto_vacuum_name}; a rebuilt copy would "
            "inherit that and could not reclaim in place. Run "
            "`opencode-gc prune --apply --enable-incremental-vacuum` first"
        )
        return _report_rebuild(args, report)

    if args.dry_run:
        # The same preflight the real run uses, so a preview reports the
        # refusals the run would hit rather than a second implementation.
        report.holders = db_holders(db_path)
        refusal = rebuild_preflight(db_path, stats, min_free_bytes=min_free_bytes)
        if refusal:
            report.skipped = refusal
        else:
            report.notes.append(
                "preflight passed: a rebuild would run now. Nothing has been "
                "touched."
            )
        return _report_rebuild(args, report)

    try:
        outcome = rebuild_database(
            db_path, stats,
            max_seconds=args.max_seconds, min_free_bytes=min_free_bytes,
        )
    except (KeyboardInterrupt, OSError, sqlite3.Error, ValueError) as exc:
        report.errors.append(f"the rebuild stopped: {type(exc).__name__}: {exc}")
        report.bytes_after = on_disk_bytes(db_path)
        return _report_rebuild(args, report)

    report.holders = outcome.holders_before
    report.attempted = outcome.attempted
    report.completed = outcome.completed
    report.seconds = outcome.seconds
    report.bytes_written = outcome.bytes_written
    report.skipped = outcome.skipped
    if outcome.failure:
        report.errors.append(outcome.failure)
    report.bytes_after = on_disk_bytes(db_path)
    return _report_rebuild(args, report)


def _human_bytes(n: int) -> str:
    """GiB once it is worth reading in GiB, MiB below that.

    A 32 GiB store reports in GiB; a small database -- or a test fixture --
    would otherwise print `0.00GiB` for every field and tell an operator
    checking the expected reclaim nothing at all.
    """
    if n >= 1024 ** 3:
        return f"{n / 1024 ** 3:.2f}GiB"
    return f"{n / 1024 ** 2:.1f}MiB"


def _report_rebuild(args, report: RebuildReport) -> int:
    if args.json:
        print(json.dumps(
            asdict(report) | {"bytes_reclaimed": report.bytes_reclaimed}, indent=2
        ))
        return 1 if report.errors else 0

    mode = "DRY RUN" if report.dry_run else "REBUILD"
    print(f"[opencode-gc] {mode}: {report.db}")
    print(f"    file {_human_bytes(report.bytes_before)}, live "
          f"{_human_bytes(report.live_bytes)}, auto_vacuum={report.auto_vacuum}")
    if report.dry_run:
        print(f"    a rebuild would reclaim about "
              f"{_human_bytes(report.expected_reclaim)}")
    if report.completed:
        print(f"    rebuilt in {report.seconds:g}s; "
              f"{report.bytes_written:,} bytes written, "
              f"{_human_bytes(report.bytes_reclaimed)} reclaimed")
    if report.skipped:
        print(f"    did not rebuild: {report.skipped}")
    for note in report.notes:
        print(f"    {note}")
    for err in report.errors:
        print(f"    ERROR: {err}", file=sys.stderr)
    return 1 if report.errors else 0


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
        print(f"    kept {res.sessions_kept_unknown_age} session(s) whose "
              "time_updated is NULL or not numeric (unknown age, not expired)")
    if res.sessions_kept_parent_cycle:
        print(f"    kept {res.sessions_kept_parent_cycle} session(s) in a parent_id "
              "cycle (no safe deletion order)")
    if res.pages_released:
        print(f"    released {res.pages_released:,} logical pages "
              f"({res.bytes_reclaimed:,} bytes actually off the disk)")
    if res.wal_checkpoint_mode:
        print(f"    checkpointed {res.wal_pages_checkpointed:,} WAL page(s) "
              f"({res.wal_checkpoint_mode}); wal {res.wal_bytes_before:,} -> "
              f"{res.wal_bytes_after:,} bytes")
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
