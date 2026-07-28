#!/usr/bin/env python3
"""threadctl — SQLite-backed state, Discord write ownership, and GitHub
polling for long-lived coding threads (one per Discord thread). Supersedes
`thread-state`.

Storage is SQLite (`~/.config/threadctl/state.db`, WAL mode) rather than a
JSON registry, because many agents hit the DB concurrently while a 1-minute
system cron/systemd timer (`threadctl cycle`) runs. WAL + a short
busy_timeout means concurrent readers/writers don't block each other or
corrupt state; see `connect_db`.

The core design rule: **the cycle is the only writer to Discord.** Agent-
facing commands (`bind`/`touch`/`set`/`add-pr`/`rm-pr`/`primary-pr`) only
ever touch SQLite — they never call the Discord API. `threadctl cycle`
(ops-facing, cron-only) is what recomputes each thread's icon/title,
reconciles the pinned status message, and renames the Discord thread. This
is what makes many small agent updates coalesce into a single rename
instead of each one burning a rename slot — Discord hard-limits thread
renames to 2 per 10 minutes.

Usage::

    threadctl bind <thread_id> --title "…" --repo owner/name
    threadctl touch <thread_id> [--note "…"]
    threadctl set <thread_id> [--title "…"] [--status active|paused|closed|merged]
                              [--question] [--answered]
    threadctl add-pr <thread_id> <ref> [--primary]
    threadctl rm-pr <thread_id> <ref>
    threadctl primary-pr <thread_id> <ref>
    threadctl show <thread_id> [--json]
    threadctl list [--json]
    threadctl cycle [--dry-run]
    threadctl migrate --from <registry.json>

A PR reference (REF) accepts `4694`, `#4694`, `owner/name#4694`, or a full
`https://github.com/owner/name/pull/4694` URL. A bare `N` resolves against
the thread's bound repo. Any PR whose owner doesn't match the thread's owner
is rejected — a thread only ever tracks PRs under the org/user it was bound
to.

`bind` on an already-bound thread is intentionally destructive: it wipes the
thread's PRs, clears the question flag, and resets status to `active` — a
fresh start for a thread being repurposed for new work. It warns loudly on
stderr and records what it discarded in `history` (see `cmd_bind`). It does
NOT touch `pin_message_id` directly — since bind is DB-only (per the
single-Discord-writer rule above), the old pinned message is actually
unpinned by the next `cycle` run, which notices the thread now has zero PRs
but a stale `pin_message_id` and unpins+forgets it as an ordinary case of
pin reconciliation (see `reconcile_pin` in the cycle module section).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

STATUSES = ("active", "paused", "closed", "merged")

THREAD_TITLE_MAX = 48
DISCORD_NAME_MAX = 100

DEFAULT_ACCOUNTS = {"absmartly": "marcio-absmartly", "marcioapm": "marcioapm"}

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS threads (
  thread_id      TEXT PRIMARY KEY,
  title          TEXT NOT NULL,
  owner          TEXT NOT NULL,
  repo           TEXT NOT NULL,
  status         TEXT NOT NULL,
  question       INTEGER NOT NULL DEFAULT 0,
  bound_at       TEXT NOT NULL,
  last_touch_at  TEXT NOT NULL,
  desired_name   TEXT,
  applied_name   TEXT,
  last_rename_at TEXT,
  rename_backoff_until TEXT,
  pin_message_id TEXT,
  created_at     TEXT NOT NULL,
  updated_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS prs (
  id             INTEGER PRIMARY KEY,
  thread_id      TEXT NOT NULL REFERENCES threads(thread_id) ON DELETE CASCADE,
  repo           TEXT NOT NULL,
  number         INTEGER NOT NULL,
  is_primary     INTEGER NOT NULL DEFAULT 0,
  state          TEXT,
  ci             TEXT,
  review         TEXT,
  mergeable      TEXT,
  is_draft       INTEGER DEFAULT 0,
  pr_title       TEXT,
  url            TEXT,
  checked_at     TEXT,
  created_at     TEXT NOT NULL,
  UNIQUE(thread_id, repo, number)
);

CREATE TABLE IF NOT EXISTS history (
  id         INTEGER PRIMARY KEY,
  thread_id  TEXT NOT NULL,
  ts         TEXT NOT NULL,
  kind       TEXT NOT NULL,
  text       TEXT
);

CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""

# A PR reference accepts "4694", "#4694", "owner/name#4694", or a full PR URL.
PR_REF_RE = re.compile(
    r"^(?:https://github\.com/(?P<url_owner>[\w.-]+)/(?P<url_name>[\w.-]+)/pull/(?P<url_num>\d+)"
    r"|(?:(?P<repo>[\w.-]+/[\w.-]+)#)?#?(?P<num>\d+))$"
)

REPO_RE = re.compile(r"^[\w.-]+/[\w.-]+$")


class _Abort(Exception):
    """Internal control-flow signal: abort the in-flight command with `code`
    after printing `message` to stderr. Mirrors thread-state's `_Abort`."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _utcnow() -> datetime:
    """Wraps `datetime.now(timezone.utc)` behind a seam tests can
    monkeypatch for deterministic staleness/rename-backoff assertions."""
    return datetime.now(timezone.utc)


def _parse_iso(ts: str) -> datetime:
    return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def _err(message: str) -> None:
    print(message, file=sys.stderr)


def parse_pr_ref(ref: str, default_repo: Optional[str] = None) -> Tuple[Optional[str], int]:
    """Parse an `add-pr`/`rm-pr`/`primary-pr` REF argument into
    (repo_or_None, number). `repo` is None when the ref didn't name one and
    no `default_repo` was given."""
    ref = ref.strip()
    m = PR_REF_RE.match(ref)
    if not m:
        raise ValueError(f"invalid PR reference '{ref}' (expected N, #N, owner/name#N, or a PR URL)")
    if m.group("url_num"):
        return f"{m.group('url_owner')}/{m.group('url_name')}", int(m.group("url_num"))
    repo = m.group("repo") or default_repo
    return repo, int(m.group("num"))


def _enforce_title(title: str) -> Tuple[str, bool]:
    """Truncate `title` to THREAD_TITLE_MAX chars, enforced at write time
    (not render time, per spec). Returns (title, was_truncated)."""
    title = title.strip()
    if len(title) <= THREAD_TITLE_MAX:
        return title, False
    return title[: THREAD_TITLE_MAX - 1].rstrip() + "…", True


# ---------------------------------------------------------------------------
# icon derivation (worst-wins), staleness, title rendering
# ---------------------------------------------------------------------------

# Worst-wins priority, most-severe first. Note ✅ ranks BELOW 🟢 — a PR that's
# approved-and-mergeable is *less* urgent than a thread actively being worked
# (🟢), since ✅ is waiting on nothing but a merge click. This is intentional,
# not a bug.
ICON_PRIORITY = ["❓", "🔴", "❗", "🟡", "🔵", "🟢", "✅", "🟣", "⚪"]

# Icons that mean "waiting on a human", not "waiting on an agent" — these
# never go stale regardless of last_touch_at age.
STALE_EXEMPT_ICONS = frozenset({"❓", "🔵", "✅", "❗", "🟣", "⚪"})

STATUS_ICON = {
    "active": "🟢",
    "paused": "⚪",
    "closed": "⚪",
    "merged": "🟣",
}

DEFAULT_STALE_AFTER_MIN = 30


def worst_icon(icons: List[str]) -> str:
    """Worst-wins across `icons`, per the fixed priority ladder
    ❓ > 🔴 > ❗ > 🟡 > 🔵 > 🟢 > ✅ > 🟣 > ⚪."""
    return min(icons, key=ICON_PRIORITY.index)


def pr_icon(pr: Dict[str, Any]) -> str:
    """Per-PR icon, per the fixed ladder in spec section 5. `pr` is a dict
    with state/ci/review/mergeable keys. MERGED/CLOSED are checked first
    since they're terminal; every other branch assumes state == OPEN."""
    state = pr.get("state")
    if state == "MERGED":
        return "🟣"
    if state == "CLOSED":
        return "⚪"
    ci = pr.get("ci")
    review = pr.get("review")
    mergeable = pr.get("mergeable")
    if ci == "FAILURE":
        return "🔴"
    if ci == "PENDING":
        return "🟡"
    if review == "APPROVED" and mergeable == "CONFLICTING":
        return "❗"
    if review == "APPROVED" and mergeable == "MERGEABLE" and ci == "SUCCESS":
        return "✅"
    if ci == "SUCCESS" and review != "APPROVED":
        return "🔵"
    return "🟢"


def effective_thread_icon(thread: Dict[str, Any], prs: List[Dict[str, Any]]) -> str:
    """Thread icon = worst of {thread-level state, all OPEN PR states}. The
    question flag contributes ❓, which always wins (it outranks everything
    in ICON_PRIORITY) — it doesn't need to short-circuit the other icons."""
    icons = [STATUS_ICON.get(thread.get("status"), "🟢")]
    if thread.get("question"):
        icons.append("❓")
    icons.extend(pr_icon(p) for p in prs if p.get("state") == "OPEN")
    return worst_icon(icons)


def is_stale(
    icon: str,
    last_touch_at: str,
    now: Optional[datetime] = None,
    stale_after_min: int = DEFAULT_STALE_AFTER_MIN,
) -> bool:
    """Stale = now - last_touch_at > stale_after_min, except for icons that
    are inherently "waiting on a human" (see STALE_EXEMPT_ICONS) — those
    never go stale no matter how old last_touch_at is."""
    if icon in STALE_EXEMPT_ICONS:
        return False
    now = now or _utcnow()
    age_seconds = (now - _parse_iso(last_touch_at)).total_seconds()
    return age_seconds > stale_after_min * 60


def _pr_label(pr: Dict[str, Any], thread_repo: Optional[str]) -> str:
    """`#4615` when the PR's repo matches the thread's bound repo, else
    `abs#4615` — the repo's short name (no owner, since owner is always
    shared between a thread and its PRs)."""
    repo = pr.get("repo") or thread_repo
    if repo == thread_repo:
        return f"#{pr['number']}"
    short = (repo or "").split("/", 1)[-1]
    return f"{short}#{pr['number']}"


def _sorted_open_prs(prs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """OPEN PRs only, primary first then ascending number."""
    open_prs = [p for p in prs if p.get("state") == "OPEN"]
    return sorted(open_prs, key=lambda p: (not p.get("primary"), p["number"]))


def render_title(thread: Dict[str, Any], prs: List[Dict[str, Any]], stale: bool) -> str:
    """The full desired Discord thread name:
    `{threadIcon} {title} · {prIcon}#{num} {prIcon}#{num} …{💤}`
    Only OPEN PRs appear (merged/closed live on in the pinned message only).
    Discord's 100-char cap is enforced by truncating the PR list from the
    right and appending `+N` — the icon and title are never truncated here
    (title is already capped to 48 chars at write time)."""
    icon = effective_thread_icon(thread, prs)
    title = thread.get("title", "")
    base = f"{icon} {title}"
    suffix = " 💤" if stale else ""

    open_prs = _sorted_open_prs(prs)
    if not open_prs:
        return base + suffix

    parts = [f"{pr_icon(p)}{_pr_label(p, thread.get('repo'))}" for p in open_prs]
    full = base + " · " + " ".join(parts) + suffix
    if len(full) <= DISCORD_NAME_MAX:
        return full

    candidate = base + " · " + f"+{len(parts)}" + suffix
    for keep in range(len(parts) - 1, -1, -1):
        dropped = len(parts) - keep
        shown = " ".join(parts[:keep])
        tail = f"{shown} +{dropped}" if shown else f"+{dropped}"
        candidate = base + " · " + tail + suffix
        if len(candidate) <= DISCORD_NAME_MAX:
            return candidate
    return candidate  # pathological: even a bare "+N" barely fits; best effort


# ---------------------------------------------------------------------------
# database: connection, schema, history
# ---------------------------------------------------------------------------

def connect_db(path: Path) -> sqlite3.Connection:
    """Open (creating if needed) the threadctl SQLite DB with WAL mode and a
    short busy_timeout, so many agents hitting the DB concurrently while the
    cycle runs don't block each other or corrupt state."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    return conn


def _resolve_db_path(args: argparse.Namespace) -> Path:
    """--db flag wins, then $THREADCTL_DB, then the default under
    ~/.config/threadctl/."""
    explicit = getattr(args, "db", None)
    if explicit:
        return Path(explicit).expanduser()
    env = os.environ.get("THREADCTL_DB")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".config" / "threadctl" / "state.db"


def _record_history(conn: sqlite3.Connection, thread_id: str, kind: str, text: Optional[str]) -> None:
    conn.execute(
        "INSERT INTO history (thread_id, ts, kind, text) VALUES (?, ?, ?, ?)",
        (thread_id, _now_iso(), kind, text),
    )


def get_thread(conn: sqlite3.Connection, thread_id: str) -> Optional[sqlite3.Row]:
    return conn.execute("SELECT * FROM threads WHERE thread_id=?", (thread_id,)).fetchone()


def get_prs(conn: sqlite3.Connection, thread_id: str) -> List[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM prs WHERE thread_id=? ORDER BY is_primary DESC, number ASC", (thread_id,)
    ).fetchall()


def get_all_threads(conn: sqlite3.Connection) -> List[sqlite3.Row]:
    return conn.execute("SELECT * FROM threads ORDER BY title ASC, thread_id ASC").fetchall()


# ---------------------------------------------------------------------------
# commands: bind / touch / set / add-pr / rm-pr / primary-pr / show / list
# ---------------------------------------------------------------------------

def cmd_bind(args: argparse.Namespace) -> int:
    path = _resolve_db_path(args)
    conn = connect_db(path)
    try:
        repo = args.repo
        if not REPO_RE.match(repo):
            _err(f"threadctl: invalid --repo '{repo}' (expected owner/name)")
            return 2
        owner = repo.split("/", 1)[0]
        title, truncated = _enforce_title(args.title)
        now = _now_iso()

        with conn:
            existing = get_thread(conn, args.thread_id)
            if existing is not None:
                pr_count = conn.execute(
                    "SELECT COUNT(*) FROM prs WHERE thread_id=?", (args.thread_id,)
                ).fetchone()[0]
                discard_summary = (
                    f"{pr_count} pr(s), status={existing['status']}, "
                    f"question={bool(existing['question'])}, pin_message_id={existing['pin_message_id'] or '-'}"
                )
                _err(
                    f"threadctl: WARNING — '{args.thread_id}' is already bound; bind is a fresh start "
                    f"and WIPES its PRs. Discarded: {discard_summary}"
                )
                conn.execute("DELETE FROM prs WHERE thread_id=?", (args.thread_id,))
                conn.execute(
                    "UPDATE threads SET title=?, owner=?, repo=?, status='active', question=0, "
                    "last_touch_at=?, updated_at=? WHERE thread_id=?",
                    (title, owner, repo, now, now, args.thread_id),
                )
                _record_history(conn, args.thread_id, "bind", f"rebound to {repo} (fresh start): discarded {discard_summary}")
            else:
                conn.execute(
                    "INSERT INTO threads (thread_id, title, owner, repo, status, question, bound_at, "
                    "last_touch_at, created_at, updated_at) VALUES (?, ?, ?, ?, 'active', 0, ?, ?, ?, ?)",
                    (args.thread_id, title, owner, repo, now, now, now, now),
                )
                _record_history(conn, args.thread_id, "bind", f"bound to {repo}")

        if truncated:
            _err(f"threadctl: title truncated to {THREAD_TITLE_MAX} chars: '{title}'")
        print(f"threadctl: bound '{args.thread_id}' to {repo}")
        return 0
    finally:
        conn.close()


def cmd_touch(args: argparse.Namespace) -> int:
    path = _resolve_db_path(args)
    conn = connect_db(path)
    try:
        now = _now_iso()
        with conn:
            if get_thread(conn, args.thread_id) is None:
                _err(f"threadctl: no such thread '{args.thread_id}' (bind it first)")
                return 1
            conn.execute(
                "UPDATE threads SET last_touch_at=?, updated_at=? WHERE thread_id=?",
                (now, now, args.thread_id),
            )
            _record_history(conn, args.thread_id, "touch", args.note)
        print(f"threadctl: touched '{args.thread_id}'")
        return 0
    finally:
        conn.close()


def cmd_set(args: argparse.Namespace) -> int:
    path = _resolve_db_path(args)
    conn = connect_db(path)
    try:
        if args.question and args.answered:
            _err("threadctl: --question and --answered are mutually exclusive")
            return 2

        with conn:
            thread = get_thread(conn, args.thread_id)
            if thread is None:
                _err(f"threadctl: no such thread '{args.thread_id}' (bind it first)")
                return 1

            now = _now_iso()
            fields: Dict[str, Any] = {"last_touch_at": now, "updated_at": now}

            if args.title is not None:
                title, truncated = _enforce_title(args.title)
                fields["title"] = title
                if truncated:
                    _err(f"threadctl: title truncated to {THREAD_TITLE_MAX} chars: '{title}'")

            if args.status is not None:
                if args.status not in STATUSES:
                    _err(f"threadctl: invalid --status '{args.status}' (must be one of: {', '.join(STATUSES)})")
                    return 2
                fields["status"] = args.status

            if args.question:
                fields["question"] = 1
            if args.answered:
                fields["question"] = 0

            set_clause = ", ".join(f"{k}=?" for k in fields)
            conn.execute(
                f"UPDATE threads SET {set_clause} WHERE thread_id=?",
                (*fields.values(), args.thread_id),
            )
            if args.status is not None:
                _record_history(conn, args.thread_id, "status", f"status -> {args.status}")
            if args.question:
                _record_history(conn, args.thread_id, "question", "question raised")
            if args.answered:
                _record_history(conn, args.thread_id, "question", "question answered")

        print(f"threadctl: updated '{args.thread_id}'")
        return 0
    finally:
        conn.close()


def _check_pr_owner(thread: sqlite3.Row, repo: Optional[str]) -> Optional[str]:
    """Returns an error message if `repo`'s owner doesn't match the thread's
    owner, else None. A None `repo` (ref had none and thread has no repo
    fallback) is also an error, surfaced by the caller."""
    if repo is None:
        return "PR reference has no repo and the thread has none bound"
    pr_owner = repo.split("/", 1)[0]
    if pr_owner != thread["owner"]:
        return f"PR owner '{pr_owner}' does not match thread owner '{thread['owner']}'"
    return None


def cmd_add_pr(args: argparse.Namespace) -> int:
    path = _resolve_db_path(args)
    conn = connect_db(path)
    try:
        thread = get_thread(conn, args.thread_id)
        if thread is None:
            _err(f"threadctl: no such thread '{args.thread_id}' (bind it first)")
            return 1
        try:
            repo, number = parse_pr_ref(args.ref, thread["repo"])
        except ValueError as exc:
            _err(f"threadctl: {exc}")
            return 2
        err = _check_pr_owner(thread, repo)
        if err:
            _err(f"threadctl: {err}")
            return 2

        now = _now_iso()
        with conn:
            existing = conn.execute(
                "SELECT id FROM prs WHERE thread_id=? AND repo=? AND number=?",
                (args.thread_id, repo, number),
            ).fetchone()
            pr_count_before = conn.execute(
                "SELECT COUNT(*) FROM prs WHERE thread_id=?", (args.thread_id,)
            ).fetchone()[0]
            make_primary = bool(args.primary) or (existing is None and pr_count_before == 0)

            if existing is None:
                conn.execute(
                    "INSERT INTO prs (thread_id, repo, number, is_primary, created_at) "
                    "VALUES (?, ?, ?, 0, ?)",
                    (args.thread_id, repo, number, now),
                )
            if make_primary:
                conn.execute("UPDATE prs SET is_primary=0 WHERE thread_id=?", (args.thread_id,))
                conn.execute(
                    "UPDATE prs SET is_primary=1 WHERE thread_id=? AND repo=? AND number=?",
                    (args.thread_id, repo, number),
                )
            conn.execute(
                "UPDATE threads SET last_touch_at=?, updated_at=? WHERE thread_id=?",
                (now, now, args.thread_id),
            )
            verb = "already tracked" if existing is not None else "added"
            suffix = " (primary)" if make_primary else ""
            _record_history(conn, args.thread_id, "pr", f"{verb} {repo}#{number}{suffix}")

        print(f"threadctl: {repo}#{number} tracked on '{args.thread_id}'" + (" (primary)" if make_primary else ""))
        return 0
    finally:
        conn.close()


def cmd_rm_pr(args: argparse.Namespace) -> int:
    path = _resolve_db_path(args)
    conn = connect_db(path)
    try:
        thread = get_thread(conn, args.thread_id)
        if thread is None:
            _err(f"threadctl: no such thread '{args.thread_id}' (bind it first)")
            return 1
        try:
            repo, number = parse_pr_ref(args.ref, thread["repo"])
        except ValueError as exc:
            _err(f"threadctl: {exc}")
            return 2

        with conn:
            row = conn.execute(
                "SELECT id, is_primary FROM prs WHERE thread_id=? AND repo=? AND number=?",
                (args.thread_id, repo, number),
            ).fetchone()
            if row is None:
                _err(f"threadctl: no such PR {repo}#{number} tracked on '{args.thread_id}'")
                return 1
            was_primary = bool(row["is_primary"])
            conn.execute("DELETE FROM prs WHERE id=?", (row["id"],))
            if was_primary:
                nxt = conn.execute(
                    "SELECT id FROM prs WHERE thread_id=? ORDER BY number ASC LIMIT 1", (args.thread_id,)
                ).fetchone()
                if nxt is not None:
                    conn.execute("UPDATE prs SET is_primary=1 WHERE id=?", (nxt["id"],))
            now = _now_iso()
            conn.execute(
                "UPDATE threads SET last_touch_at=?, updated_at=? WHERE thread_id=?",
                (now, now, args.thread_id),
            )
            _record_history(conn, args.thread_id, "pr", f"removed {repo}#{number}")

        print(f"threadctl: removed {repo}#{number} from '{args.thread_id}'")
        return 0
    finally:
        conn.close()


def cmd_primary_pr(args: argparse.Namespace) -> int:
    path = _resolve_db_path(args)
    conn = connect_db(path)
    try:
        thread = get_thread(conn, args.thread_id)
        if thread is None:
            _err(f"threadctl: no such thread '{args.thread_id}' (bind it first)")
            return 1
        try:
            repo, number = parse_pr_ref(args.ref, thread["repo"])
        except ValueError as exc:
            _err(f"threadctl: {exc}")
            return 2

        with conn:
            row = conn.execute(
                "SELECT id FROM prs WHERE thread_id=? AND repo=? AND number=?",
                (args.thread_id, repo, number),
            ).fetchone()
            if row is None:
                _err(f"threadctl: no such PR {repo}#{number} tracked on '{args.thread_id}'")
                return 1
            conn.execute("UPDATE prs SET is_primary=0 WHERE thread_id=?", (args.thread_id,))
            conn.execute("UPDATE prs SET is_primary=1 WHERE id=?", (row["id"],))
            now = _now_iso()
            conn.execute(
                "UPDATE threads SET last_touch_at=?, updated_at=? WHERE thread_id=?",
                (now, now, args.thread_id),
            )
            _record_history(conn, args.thread_id, "pr", f"primary -> {repo}#{number}")

        print(f"threadctl: {repo}#{number} is now primary on '{args.thread_id}'")
        return 0
    finally:
        conn.close()


def _pr_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    return {
        "repo": row["repo"],
        "number": row["number"],
        "primary": bool(row["is_primary"]),
        "state": row["state"],
        "ci": row["ci"],
        "review": row["review"],
        "mergeable": row["mergeable"],
        "draft": bool(row["is_draft"]),
        "title": row["pr_title"],
        "url": row["url"],
        "checkedAt": row["checked_at"],
    }


def _thread_to_dict(thread: sqlite3.Row, prs: List[sqlite3.Row]) -> Dict[str, Any]:
    return {
        "threadId": thread["thread_id"],
        "title": thread["title"],
        "owner": thread["owner"],
        "repo": thread["repo"],
        "status": thread["status"],
        "question": bool(thread["question"]),
        "boundAt": thread["bound_at"],
        "lastTouchAt": thread["last_touch_at"],
        "desiredName": thread["desired_name"],
        "appliedName": thread["applied_name"],
        "lastRenameAt": thread["last_rename_at"],
        "renameBackoffUntil": thread["rename_backoff_until"],
        "pinMessageId": thread["pin_message_id"],
        "createdAt": thread["created_at"],
        "updatedAt": thread["updated_at"],
        "prs": [_pr_to_dict(p) for p in prs],
    }


def _format_thread_human(thread: sqlite3.Row, prs: List[sqlite3.Row]) -> str:
    lines = [
        f"threadId: {thread['thread_id']}",
        f"title: {thread['title']}",
        f"owner/repo: {thread['owner']} / {thread['repo']}",
        f"status: {thread['status']}",
        f"question: {bool(thread['question'])}",
        f"boundAt: {thread['bound_at']}",
        f"lastTouchAt: {thread['last_touch_at']}",
        f"desiredName: {thread['desired_name'] or '-'}",
        f"appliedName: {thread['applied_name'] or '-'}",
        f"lastRenameAt: {thread['last_rename_at'] or '-'}",
        f"renameBackoffUntil: {thread['rename_backoff_until'] or '-'}",
        f"pinMessageId: {thread['pin_message_id'] or '-'}",
    ]
    for p in prs:
        marker = "*" if p["is_primary"] else " "
        lines.append(
            f"pr[{marker}]: {p['repo']}#{p['number']} state={p['state']} ci={p['ci']} "
            f"review={p['review']} mergeable={p['mergeable']} draft={bool(p['is_draft'])}"
        )
    return "\n".join(lines)


def cmd_show(args: argparse.Namespace) -> int:
    path = _resolve_db_path(args)
    conn = connect_db(path)
    try:
        thread = get_thread(conn, args.thread_id)
        if thread is None:
            _err(f"threadctl: no such thread '{args.thread_id}'")
            return 1
        prs = get_prs(conn, args.thread_id)
        if args.json:
            print(json.dumps(_thread_to_dict(thread, prs), indent=2, sort_keys=True))
        else:
            print(_format_thread_human(thread, prs))
        return 0
    finally:
        conn.close()


def cmd_list(args: argparse.Namespace) -> int:
    path = _resolve_db_path(args)
    conn = connect_db(path)
    try:
        threads = get_all_threads(conn)
        if args.json:
            out = [_thread_to_dict(t, get_prs(conn, t["thread_id"])) for t in threads]
            print(json.dumps(out, indent=2, sort_keys=True))
        elif not threads:
            print("threadctl: no threads")
        else:
            for t in threads:
                prs = get_prs(conn, t["thread_id"])
                pr_summary = ", ".join(f"{p['repo']}#{p['number']}" for p in prs) or "-"
                q = " ❓" if t["question"] else ""
                print(f"{t['thread_id']}  {t['title']}  [{t['status']}{q}]  {pr_summary}")
        return 0
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# CLI dispatch
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument(
        "--db",
        default=None,
        help="path to state.db (default: ~/.config/threadctl/state.db or $THREADCTL_DB)",
    )

    p = argparse.ArgumentParser(
        prog="threadctl",
        description="SQLite-backed state, Discord write ownership, and GitHub polling for coding threads.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    sp_bind = sub.add_parser("bind", parents=[parent], help="bind (or rebind, fresh start) a thread")
    sp_bind.add_argument("thread_id")
    sp_bind.add_argument("--title", required=True)
    sp_bind.add_argument("--repo", required=True, help="owner/name — always explicit, never inferred")
    sp_bind.set_defaults(func=cmd_bind)

    sp_touch = sub.add_parser("touch", parents=[parent], help="reset the staleness clock (never clears question)")
    sp_touch.add_argument("thread_id")
    sp_touch.add_argument("--note")
    sp_touch.set_defaults(func=cmd_touch)

    sp_set = sub.add_parser("set", parents=[parent], help="partially update a thread")
    sp_set.add_argument("thread_id")
    sp_set.add_argument("--title")
    sp_set.add_argument("--status", choices=STATUSES)
    sp_set.add_argument("--question", action="store_true", help="raise the ❓ flag")
    sp_set.add_argument("--answered", action="store_true", help="clear the ❓ flag")
    sp_set.set_defaults(func=cmd_set)

    sp_add_pr = sub.add_parser("add-pr", parents=[parent], help="track a PR on a thread")
    sp_add_pr.add_argument("thread_id")
    sp_add_pr.add_argument("ref", metavar="REF")
    sp_add_pr.add_argument("--primary", action="store_true")
    sp_add_pr.set_defaults(func=cmd_add_pr)

    sp_rm_pr = sub.add_parser("rm-pr", parents=[parent], help="stop tracking a PR on a thread")
    sp_rm_pr.add_argument("thread_id")
    sp_rm_pr.add_argument("ref", metavar="REF")
    sp_rm_pr.set_defaults(func=cmd_rm_pr)

    sp_primary_pr = sub.add_parser("primary-pr", parents=[parent], help="mark which PR is primary")
    sp_primary_pr.add_argument("thread_id")
    sp_primary_pr.add_argument("ref", metavar="REF")
    sp_primary_pr.set_defaults(func=cmd_primary_pr)

    sp_show = sub.add_parser("show", parents=[parent], help="show one thread")
    sp_show.add_argument("thread_id")
    sp_show.add_argument("--json", action="store_true")
    sp_show.set_defaults(func=cmd_show)

    sp_list = sub.add_parser("list", parents=[parent], help="list all threads")
    sp_list.add_argument("--json", action="store_true")
    sp_list.set_defaults(func=cmd_list)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
