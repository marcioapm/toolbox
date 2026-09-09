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

WHY INCREMENTAL VACUUM AND NOT VACUUM
-------------------------------------
A plain `VACUUM` rewrites the whole database, needing free space roughly equal
to the file. On vibes that is 76 GB of temp against 41 GB free -- it would fail,
after hours of IO. `PRAGMA auto_vacuum=2` (INCREMENTAL) instead lets
`PRAGMA incremental_vacuum(N)` return freed pages to the filesystem in bounded
chunks, with no rewrite and no large temp file.

vibes already runs auto_vacuum=2, so deletes there are immediately reclaimable.
A host at auto_vacuum=0 or 1 needs one full VACUUM to switch modes, which is
what --enable-incremental-vacuum does, and only after a free-space check.
"""
from __future__ import annotations

import argparse
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
class Result:
    db: str
    dry_run: bool
    retention_days: float
    sessions_expired: int = 0
    sessions_deleted: int = 0
    sessions_kept_live_descendant: int = 0
    rows_deleted: dict = field(default_factory=dict)
    pages_released: int = 0
    bytes_before: int = 0
    bytes_after: int = 0
    auto_vacuum_before: str = ""
    auto_vacuum_after: str = ""
    errors: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    @property
    def bytes_freed(self) -> int:
        return max(0, self.bytes_before - self.bytes_after)


def connect(db: Path, *, read_only: bool, timeout_s: float = 30.0) -> sqlite3.Connection:
    uri = f"file:{db}?mode=ro" if read_only else f"file:{db}?mode=rw"
    conn = sqlite3.connect(uri, uri=True, timeout=timeout_s, isolation_level=None)
    conn.execute(f"PRAGMA busy_timeout={int(timeout_s * 1000)}")
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


def expired_session_ids(conn: sqlite3.Connection, cutoff_ms: int) -> tuple[list[str], int]:
    """Sessions whose entire subtree is older than the cutoff.

    `session.parent_id` has no foreign key, so deleting a parent whose child is
    still active would leave the child pointing at a row that no longer exists.
    A session is therefore only expired when it and every descendant are.

    Returns (deletable_ids, kept_because_a_descendant_is_live).
    """
    rows = conn.execute("SELECT id, parent_id, time_updated FROM session").fetchall()
    updated = {r[0]: (r[2] or 0) for r in rows}
    children: dict[Optional[str], list[str]] = {}
    for sid, parent, _t in rows:
        children.setdefault(parent, []).append(sid)

    old = {sid for sid, t in updated.items() if t < cutoff_ms}

    # Walk up from every live session, protecting its whole ancestor chain.
    parent_of = {r[0]: r[1] for r in rows}
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

    deletable = sorted(old - protected)
    return deletable, len(old & protected)


def delete_sessions(
    conn: sqlite3.Connection,
    session_ids: list[str],
    *,
    batch: int,
    deadline: Optional[float],
) -> dict[str, int]:
    """Delete sessions and all their rows, children first, in batches.

    Batched so a single statement never holds the write lock across millions of
    rows while opencode itself is running.
    """
    counts: dict[str, int] = {t: 0 for t, _ in CHILD_TABLES}
    counts["session"] = 0

    for start in range(0, len(session_ids), batch):
        if deadline is not None and time.monotonic() > deadline:
            break
        chunk = session_ids[start:start + batch]
        marks = ",".join("?" * len(chunk))
        conn.execute("BEGIN IMMEDIATE")
        try:
            for table, column in CHILD_TABLES:
                cur = conn.execute(
                    f"DELETE FROM {table} WHERE {column} IN ({marks})", chunk
                )
                counts[table] += cur.rowcount if cur.rowcount > 0 else 0
            cur = conn.execute(f"DELETE FROM session WHERE id IN ({marks})", chunk)
            counts["session"] += cur.rowcount if cur.rowcount > 0 else 0
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
    return counts


def count_rows_for(conn: sqlite3.Connection, session_ids: list[str], batch: int) -> dict[str, int]:
    """Rows a real run would delete. Used by the dry run so its preview is the
    same universe the apply path acts on."""
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


def run_incremental_vacuum(
    conn: sqlite3.Connection, *, pages: Optional[int], deadline: Optional[float]
) -> int:
    """Release freed pages back to the filesystem. Returns pages released.

    Bounded per call so a huge freelist cannot block the database for minutes.
    """
    before = int(conn.execute("PRAGMA freelist_count").fetchone()[0])
    if before == 0:
        return 0
    step = 2000
    released = 0
    target = before if pages is None else min(pages, before)
    while released < target:
        if deadline is not None and time.monotonic() > deadline:
            break
        conn.execute(f"PRAGMA incremental_vacuum({min(step, target - released)})")
        now = int(conn.execute("PRAGMA freelist_count").fetchone()[0])
        progressed = before - now - released
        released = before - now
        if progressed <= 0:
            break
    return released


def enable_incremental_vacuum(conn: sqlite3.Connection, db: Path, stats: DbStats) -> list[str]:
    """Switch the DB to auto_vacuum=INCREMENTAL. Requires one full VACUUM.

    Returns notes; raises RuntimeError when there is not enough free space,
    because a VACUUM that runs out mid-rewrite is far worse than not starting.
    """
    if stats.auto_vacuum == 2:
        return ["auto_vacuum already INCREMENTAL; nothing to change"]

    free = shutil.disk_usage(db.parent).free
    needed = int(stats.total_bytes * 1.1)
    if free < needed:
        raise RuntimeError(
            f"VACUUM needs ~{needed:,} bytes free to rewrite a "
            f"{stats.total_bytes:,} byte database; only {free:,} available. "
            "Delete rows first, or move the database to a larger filesystem."
        )
    conn.execute("PRAGMA auto_vacuum=2")
    conn.execute("VACUUM")
    return [f"auto_vacuum {stats.auto_vacuum_name} -> INCREMENTAL (full VACUUM run)"]


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
                    help="sessions per transaction (default 200)")
    ap.add_argument("--max-seconds", type=float, default=600.0,
                    help="stop starting new work after this long (default 600)")
    ap.add_argument("--vacuum-pages", type=int, default=None,
                    help="cap pages released per run (default: the whole freelist)")
    ap.add_argument("--no-vacuum", action="store_true",
                    help="delete rows but do not release pages")
    ap.add_argument("--enable-incremental-vacuum", action="store_true",
                    help="switch auto_vacuum to INCREMENTAL; needs one full "
                         "VACUUM and free space >= 1.1x the database")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if not math.isfinite(args.retention_days) or args.retention_days < MIN_RETENTION_DAYS:
        ap.error(f"--retention-days must be finite and >= {MIN_RETENTION_DAYS}")
    if args.batch < 1:
        ap.error("--batch must be >= 1")
    if not args.db.is_file():
        print(f"opencode-gc: no database at {args.db}", file=sys.stderr)
        return 2

    deadline = time.monotonic() + args.max_seconds if args.max_seconds > 0 else None
    res = Result(db=str(args.db), dry_run=not args.apply,
                 retention_days=args.retention_days)

    conn = connect(args.db, read_only=not args.apply)
    try:
        stats = read_stats(conn)
        res.bytes_before = stats.total_bytes
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
                    res.notes.extend(enable_incremental_vacuum(conn, args.db, stats))
                    res.auto_vacuum_after = read_stats(conn).auto_vacuum_name
                except RuntimeError as exc:
                    res.errors.append(str(exc))

        cutoff_ms = int((time.time() - args.retention_days * 86400) * 1000)
        expired, kept = expired_session_ids(conn, cutoff_ms)
        res.sessions_expired = len(expired)
        res.sessions_kept_live_descendant = kept

        if expired:
            if args.apply:
                res.rows_deleted = delete_sessions(
                    conn, expired, batch=args.batch, deadline=deadline
                )
                res.sessions_deleted = res.rows_deleted.get("session", 0)
            else:
                res.rows_deleted = count_rows_for(conn, expired, args.batch)

        if args.apply and not args.no_vacuum:
            current = read_stats(conn)
            if current.auto_vacuum == 2:
                res.pages_released = run_incremental_vacuum(
                    conn, pages=args.vacuum_pages, deadline=deadline
                )
            elif current.freelist_count:
                res.notes.append(
                    f"{current.freelist_count:,} pages are free but auto_vacuum is "
                    f"{current.auto_vacuum_name}; they stay in the file. Re-run with "
                    "--enable-incremental-vacuum to reclaim future deletes."
                )
        res.bytes_after = read_stats(conn).total_bytes
    finally:
        conn.close()

    if args.json:
        print(json.dumps(asdict(res) | {"bytes_freed": res.bytes_freed}, indent=2))
        return 1 if res.errors else 0

    mode = "APPLIED" if args.apply else "DRY RUN"
    gib = res.bytes_freed / 1024 ** 3
    print(f"[opencode-gc] {mode}: retention={res.retention_days}d "
          f"sessions_expired={res.sessions_expired} "
          f"sessions_deleted={res.sessions_deleted} "
          f"freed={gib:.2f}GiB auto_vacuum={res.auto_vacuum_after}")
    if res.rows_deleted:
        verb = "deleted" if args.apply else "would delete"
        rows = " ".join(f"{k}={v:,}" for k, v in sorted(res.rows_deleted.items()))
        print(f"    {verb}: {rows}")
    if res.sessions_kept_live_descendant:
        print(f"    kept {res.sessions_kept_live_descendant} old session(s) with a "
              "recently-updated descendant")
    if res.pages_released:
        print(f"    released {res.pages_released:,} pages to the filesystem")
    for note in res.notes:
        print(f"    {note}")
    for err in res.errors:
        print(f"    ERROR: {err}", file=sys.stderr)
    return 1 if res.errors else 0


if __name__ == "__main__":
    sys.exit(main())
