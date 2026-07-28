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
import contextlib
import functools
import http.client
import json
import os
import re
import socket
import sqlite3
import ssl
import subprocess
import sys
import time
import tomllib
import urllib.error
import urllib.request
from pathlib import Path
from datetime import datetime, timedelta, timezone
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
  owner          TEXT,
  repo           TEXT,
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

DB_BUSY_TIMEOUT_MS = 30000


def connect_db(path: Path) -> sqlite3.Connection:
    """Open (creating if needed) the threadctl SQLite DB with WAL mode and a
    busy_timeout generous enough to ride out the cycle's network I/O
    (CRITICAL 3: previously 5000ms, which an agent command could exceed
    while the cycle held the write lock across `gh`/Discord calls — now
    that both `_poll_due_prs` and `reconcile_discord` commit incrementally
    instead of holding the lock for the whole cycle, 30s is pure headroom
    against an individual slow write, not a substitute for short
    transactions), so many agents hitting the DB concurrently while the
    cycle runs don't block each other or corrupt state."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=DB_BUSY_TIMEOUT_MS / 1000)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(f"PRAGMA busy_timeout={DB_BUSY_TIMEOUT_MS}")
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


def _get_meta(conn: sqlite3.Connection, key: str) -> Optional[str]:
    row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return row["value"] if row else None


def _set_meta(conn: sqlite3.Connection, key: str, value: Optional[str]) -> None:
    if value is None:
        conn.execute("DELETE FROM meta WHERE key=?", (key,))
    else:
        conn.execute(
            "INSERT INTO meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )


# ---------------------------------------------------------------------------
# cycle: GitHub polling and desired-name recomputation
# ---------------------------------------------------------------------------

DEFAULT_PR_RECHECK_MIN_S = 150
DEFAULT_MAX_PR_CHECKS_PER_CYCLE = 40
DEFAULT_RENAME_MIN_INTERVAL_S = 330
GH_TIMEOUT_S = 20
CI_FAILURE_STATES = frozenset({"FAILURE", "ERROR"})
CI_PENDING_STATES = frozenset({"PENDING", "IN_PROGRESS", "QUEUED"})

# PR states that are terminal (never transition back to OPEN). Once a PR's
# state is one of these, the poller checks it exactly once more to confirm
# the transition, then freezes it (`checked_at` stays set forever after).
TERMINAL_STATES = frozenset({"MERGED", "CLOSED"})


def _just_became_terminal(old_state: Optional[str], new_state: str) -> bool:
    """True the one time a PR's state transitions from non-terminal (or
    unknown) into MERGED/CLOSED. Spec §8: "MERGED/CLOSED PRs are terminal —
    check once more after transition, then freeze." That confirming poll is
    what clears `checked_at` back to `None` so `_due_prs` immediately picks
    it up one final time; after that confirming poll, `old_state` is
    already terminal and this returns False, freezing the PR for good."""
    return old_state not in TERMINAL_STATES and new_state in TERMINAL_STATES


def _run_command(cmd: List[str], timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _load_cycle_config(config_path: Optional[Path] = None) -> Dict[str, int]:
    """Read optional cycle tunables without making configuration mandatory."""
    config = {
        "stale_after_min": DEFAULT_STALE_AFTER_MIN,
        "pr_recheck_min_s": DEFAULT_PR_RECHECK_MIN_S,
        "max_pr_checks_per_cycle": DEFAULT_MAX_PR_CHECKS_PER_CYCLE,
        "rename_min_interval_s": DEFAULT_RENAME_MIN_INTERVAL_S,
    }
    path = config_path or Path.home() / ".config" / "threadctl" / "config.toml"
    if not path.exists():
        return config
    with path.open("rb") as f:
        data = tomllib.load(f)
    cycle = data.get("cycle", {})
    discord = data.get("discord", {})
    for key in ("stale_after_min", "pr_recheck_min_s", "max_pr_checks_per_cycle"):
        value = cycle.get(key)
        if isinstance(value, int) and value > 0:
            config[key] = value
    value = discord.get("rename_min_interval_s")
    if isinstance(value, int) and value > 0:
        config["rename_min_interval_s"] = value
    return config


def _gh_account_for(repo: str, accounts: Dict[str, str]) -> Optional[str]:
    return accounts.get(repo.split("/", 1)[0])


def _fetch_gh_token(account: str, timeout: int, runner) -> Optional[str]:
    try:
        proc = runner(["gh", "auth", "token", "--user", account], timeout)
    except (subprocess.TimeoutExpired, OSError):
        return None
    if proc.returncode != 0:
        return None
    token = (proc.stdout or "").strip()
    return token or None


# GraphQL's `StatusCheckRollup.state` values, mapped into our own
# SUCCESS|FAILURE|PENDING|NONE vocabulary (see schema comment on `prs.ci`).
CI_ROLLUP_STATE_MAP = {
    "SUCCESS": "SUCCESS",
    "FAILURE": "FAILURE",
    "ERROR": "FAILURE",
    "PENDING": "PENDING",
    "EXPECTED": "PENDING",
    "ACTION_REQUIRED": "PENDING",
}


def _context_conclusion(ctx: Dict[str, Any]) -> str:
    """Normalize one heterogeneous `statusCheckRollup.contexts` node into an
    uppercase conclusion-ish token. The union has two shapes distinguished
    by `__typename`: `CheckRun` (GitHub Actions etc.) carries
    `conclusion`/`status`; `StatusContext` (legacy commit statuses, e.g.
    CodeRabbit) carries `state`. Falls back to trying all three fields for
    a node with no/unknown `__typename`."""
    typename = ctx.get("__typename")
    if typename == "CheckRun":
        return str(ctx.get("conclusion") or ctx.get("status") or "").upper()
    if typename == "StatusContext":
        return str(ctx.get("state") or "").upper()
    return str(ctx.get("conclusion") or ctx.get("state") or ctx.get("status") or "").upper()


def _derive_ci(rollup: Optional[Dict[str, Any]]) -> str:
    """Derive our SUCCESS|FAILURE|PENDING|NONE vocabulary from a real
    `statusCheckRollup` object (`{state, contexts: {nodes: [...]}}}`,
    verified against `.taskdocs/fixtures/*.json` — see ADDENDUM.md).
    `rollup["state"]` is the simplest correct source and is used first.
    `rollup.contexts.nodes` — the heterogeneous CheckRun/StatusContext union
    — is only consulted as a defensive fallback for a rollup state GitHub
    might add that we don't recognise yet. `rollup` is `None` when the repo
    has no checks configured on that commit at all (no CI wired up)."""
    if not rollup:
        return "NONE"
    mapped = CI_ROLLUP_STATE_MAP.get(str(rollup.get("state") or "").upper())
    if mapped:
        return mapped

    contexts = ((rollup.get("contexts") or {}).get("nodes")) or []
    if not contexts:
        return "NONE"
    conclusions = {_context_conclusion(c) for c in contexts}
    if conclusions & CI_FAILURE_STATES:
        return "FAILURE"
    if conclusions & CI_PENDING_STATES:
        return "PENDING"
    return "SUCCESS"


# The verified-correct query shape (ADDENDUM.md): `statusCheckRollup` has no
# `nodes` field of its own (that was CRITICAL 1 — the old query never
# validated against GitHub's schema at all). CI detail lives under
# `commits(last: 1) -> nodes -> commit -> statusCheckRollup`, and
# `contexts(last: 100)` is a `StatusCheckRollupContext` union requiring
# inline fragments for its two concrete shapes.
def _graphql_query(repo: str, numbers: List[int]) -> str:
    owner, name = repo.split("/", 1)
    fields = " ".join(
        f"pr_{number}: pullRequest(number: {number}) {{ state mergeable reviewDecision isDraft title url "
        "commits(last: 1) { nodes { commit { statusCheckRollup { state contexts(last: 100) { nodes { "
        "__typename ... on CheckRun { conclusion status name } ... on StatusContext { state context } "
        "} } } } } } }"
        for number in numbers
    )
    return f'query {{ repository(owner: "{owner}", name: "{name}") {{ {fields} }} }}'


def _parse_graphql_pr(node: Optional[Dict[str, Any]], checked_at: str) -> Optional[Dict[str, Any]]:
    if node is None:
        return None
    commit_nodes = (node.get("commits") or {}).get("nodes") or []
    rollup = None
    if commit_nodes:
        rollup = (commit_nodes[0].get("commit") or {}).get("statusCheckRollup")
    return {
        "state": node.get("state") or "OPEN",
        "ci": _derive_ci(rollup),
        "review": node.get("reviewDecision") or "NONE",
        # MERGED PRs report mergeable=UNKNOWN (verified in fixtures) — never
        # treat UNKNOWN as CONFLICTING; passed through as-is here, and
        # `pr_icon` only branches on mergeable=='CONFLICTING' explicitly.
        "mergeable": node.get("mergeable") or "UNKNOWN",
        "is_draft": int(bool(node.get("isDraft"))),
        "pr_title": node.get("title") or "",
        "url": node.get("url") or "",
        "checked_at": checked_at,
    }


def _parse_gh_graphql_stdout(stdout: str) -> Dict[str, Any]:
    """Parse `gh api graphql` stdout, tolerating trailing non-JSON text.

    `gh` appends a human-readable error line after the JSON body when a
    query references a nonexistent resource (e.g. `gh: Could not resolve to
    a PullRequest with the number of 9999999.`) — verified in
    `.taskdocs/fixtures/graphql_terminal_and_missing.raw_gh_stdout.txt`. A
    naive `json.loads` raises `JSONDecodeError: Extra data` on that
    trailing text, so we decode just the leading JSON object and ignore
    whatever (if anything) follows it."""
    return json.JSONDecoder().raw_decode(stdout.strip())[0]


def fetch_repo_pr_states(
    repo: str,
    numbers: List[int],
    accounts: Dict[str, str],
    timeout: int,
    runner,
    token_cache: Optional[Dict[str, str]] = None,
) -> Tuple[Optional[Dict[int, Optional[Dict[str, Any]]]], Optional[str], List[str]]:
    """Fetch selected PRs from one repository with one GraphQL request.

    Account tokens are resolved locally and passed only to this subprocess;
    this avoids the shared mutable active-account state of `gh auth switch`.

    Returns `(states, fatal_error, graphql_errors)`. `fatal_error` is set
    only when the whole request is unusable (gh failed to run, non-JSON
    output, no `data.repository` at all). `graphql_errors` carries any
    top-level GraphQL `errors` entries (e.g. a `NOT_FOUND` for one alias)
    even when `states` is otherwise usable — a partial failure must not
    wipe good PR state for sibling aliases in the same response (verified
    in `.taskdocs/fixtures/graphql_terminal_and_missing.json`, where
    `pr_9999999` comes back `null` alongside two good PRs)."""
    if token_cache is None:
        token_cache = {}
    if not numbers:
        return {}, None, []

    cmd = ["gh", "api", "graphql", "-f", f"query={_graphql_query(repo, numbers)}"]
    account = _gh_account_for(repo, accounts)
    if account:
        token = token_cache.get(account)
        if token is None:
            token = _fetch_gh_token(account, timeout, runner)
            if token:
                token_cache[account] = token
        if token:
            cmd = ["env", f"GH_TOKEN={token}", *cmd]

    try:
        proc = runner(cmd, timeout)
    except subprocess.TimeoutExpired:
        return None, f"gh timed out after {timeout}s", []
    except OSError as exc:
        return None, f"gh failed to run: {exc}", []

    # R2-4 (round-2 critical): `gh api graphql` exits 1 on ANY top-level
    # GraphQL error (e.g. one alias referencing a nonexistent PR) while
    # still writing fully usable `data` for the other aliases to stdout —
    # verified live (ADDENDUM.md, .taskdocs/fixtures/graphql_terminal_and_
    # missing.raw_gh_stdout.txt). The old code checked `proc.returncode !=
    # 0` and returned early BEFORE ever looking at stdout, so a single bad
    # PR number discarded the whole batch's good data every cycle forever.
    # Parse stdout FIRST, unconditionally. `returncode` only matters as a
    # fallback explanation when stdout turns out to be genuinely
    # unusable — it must never gate whether we even attempt to parse.
    try:
        payload = _parse_gh_graphql_stdout(proc.stdout)
        repository = payload["data"]["repository"]
        if not isinstance(repository, dict):
            raise TypeError("missing repository object")
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        if proc.returncode != 0:
            return None, (proc.stderr or "").strip() or f"gh exited {proc.returncode}", []
        return None, f"gh returned invalid GraphQL JSON: {exc}", []

    graphql_errors = [
        str(e.get("message") or e) for e in (payload.get("errors") or [])
    ]

    checked_at = _now_iso()
    states = {
        number: _parse_graphql_pr(repository.get(f"pr_{number}"), checked_at)
        for number in numbers
    }
    return states, None, graphql_errors


def _due_prs(conn: sqlite3.Connection, now: datetime, recheck_min_s: int, limit: int) -> List[sqlite3.Row]:
    """PRs due for a GitHub re-check: non-terminal (state IS NULL/OPEN), or
    terminal-but-never-confirmed (checked_at IS NULL — the one-more-poll
    rule), whose `checked_at` (if any) is older than `recheck_min_s`.
    Pushed into SQL rather than loaded-then-filtered in Python: our
    timestamps are fixed-width `%Y-%m-%dT%H:%M:%SZ`, so lexicographic
    comparison is chronological comparison and SQLite can do the cutoff and
    the LIMIT directly. NULLs sort first under ASC by default, so the old
    `CASE WHEN checked_at IS NULL THEN 0 ELSE 1 END` ORDER BY prefix was a
    no-op; dropped here.

    Note: this no longer parses `checked_at` in Python, so a malformed
    value (e.g. from an unvalidated pre-migration registry.json) is simply
    treated as "due" rather than raising and crashing the whole cycle —
    a deliberate, strictly-better behaviour change (see
    REVIEW-SIMPLIFIER.md #4)."""
    cutoff = datetime.fromtimestamp(now.timestamp() - recheck_min_s, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    return conn.execute(
        "SELECT * FROM prs WHERE (state IS NULL OR state='OPEN' OR "
        "(state IN ('MERGED', 'CLOSED') AND checked_at IS NULL)) "
        "AND (checked_at IS NULL OR checked_at < ?) "
        "ORDER BY checked_at ASC, id ASC LIMIT ?",
        (cutoff, limit),
    ).fetchall()


def _poll_due_prs(
    conn: sqlite3.Connection,
    now: datetime,
    config: Dict[str, int],
    runner,
    accounts: Dict[str, str],
) -> Tuple[int, int]:
    due = _due_prs(conn, now, config["pr_recheck_min_s"], config["max_pr_checks_per_cycle"])
    by_repo: Dict[str, List[sqlite3.Row]] = {}
    for pr in due:
        by_repo.setdefault(pr["repo"], []).append(pr)

    refreshed = errors = 0
    token_cache: Dict[str, str] = {}
    for repo, prs in by_repo.items():
        # HIGH 4: a thread/PR migrated with an invalid/empty repo (e.g. the
        # 11 live threads with repo=null) must not blow up `_graphql_query`
        # (`''.split('/', 1)` -> ValueError) and, via that, the whole
        # cycle. Skip just these PRs and record why, per thread.
        if not repo or not REPO_RE.match(repo):
            errors += len(prs)
            for pr in prs:
                _record_history(
                    conn, pr["thread_id"], "error",
                    f"GitHub poll skipped: invalid repo '{repo}' on PR #{pr['number']} "
                    "(run `threadctl bind --repo owner/name` to fix)",
                )
            conn.commit()
            continue

        states, error, graphql_errors = fetch_repo_pr_states(
            repo, [p["number"] for p in prs], accounts, GH_TIMEOUT_S, runner, token_cache
        )
        if error is not None:
            errors += len(prs)
            for pr in prs:
                _record_history(conn, pr["thread_id"], "error", f"GitHub poll {repo}#{pr['number']}: {error}")
            conn.commit()
            continue
        assert states is not None
        # Top-level GraphQL `errors` (e.g. NOT_FOUND for one alias) are
        # recorded, not silently ignored — but they don't abort processing
        # of the other, still-usable aliases in the same response.
        if graphql_errors:
            for message in graphql_errors:
                _record_history(conn, prs[0]["thread_id"], "error", f"GraphQL {repo}: {message}")
        for pr in prs:
            state = states.get(pr["number"])
            if state is None:
                errors += 1
                _record_history(conn, pr["thread_id"], "error", f"GitHub poll {repo}#{pr['number']}: PR not found")
                continue
            # Spec §8 / REVIEW-SIMPLIFIER.md #3: MERGED/CLOSED is terminal —
            # confirm the transition with one more poll (checked_at ->
            # None, so `_due_prs` picks it straight back up), then freeze
            # (checked_at -> the real timestamp) forever after.
            if _just_became_terminal(pr["state"], state["state"]):
                checked_at = None
            else:
                checked_at = state["checked_at"]
            conn.execute(
                "UPDATE prs SET state=?, ci=?, review=?, mergeable=?, is_draft=?, pr_title=?, url=?, checked_at=? "
                "WHERE id=?",
                (
                    state["state"], state["ci"], state["review"], state["mergeable"], state["is_draft"],
                    state["pr_title"], state["url"], checked_at, pr["id"],
                ),
            )
            refreshed += 1
        # CRITICAL 3: commit per repo-batch rather than holding the write
        # lock across every remaining `gh` subprocess in the cycle.
        conn.commit()
    return refreshed, errors


def recompute_desired_names(conn: sqlite3.Connection, now: datetime, stale_after_min: int) -> int:
    count = 0
    for thread in get_all_threads(conn):
        prs = [_pr_to_dict(pr) for pr in get_prs(conn, thread["thread_id"])]
        thread_data = dict(thread)
        icon = effective_thread_icon(thread_data, prs)
        stale = is_stale(icon, thread["last_touch_at"], now, stale_after_min)
        desired_name = render_title(thread_data, prs, stale)
        count += 1
        # CRITICAL 3: skip the write entirely when nothing changed, instead
        # of rewriting all 25 thread rows every minute regardless.
        if desired_name == thread["desired_name"]:
            continue
        conn.execute(
            "UPDATE threads SET desired_name=?, updated_at=? WHERE thread_id=?",
            (desired_name, _now_iso(), thread["thread_id"]),
        )
    return count


# ---------------------------------------------------------------------------
# cycle: Discord writes (token resolution, pin reconciliation, rename)
# ---------------------------------------------------------------------------

DISCORD_API = "https://discord.com/api/v10"
DISCORD_TIMEOUT_S = 15


class DiscordResponse:
    """A normalized Discord API response. The injectable `transport` seam
    returns these instead of raw urllib objects, so tests fake one callable
    instead of HTTP internals."""

    def __init__(self, status: int, data: Optional[Dict[str, Any]] = None):
        self.status = status
        self.data = data


def _discord_transport(
    method: str, url: str, token: str, payload: Optional[Dict[str, Any]], timeout: int
) -> DiscordResponse:
    """R2-3 (regression of C2, ROUND2-FINDINGS.md): the old exception
    handling only covered failures `urlopen()` itself could raise.
    `resp.read()` and `json.loads(body)` sat OUTSIDE any try/except, so a
    non-JSON 2xx body (a proxy/CDN interstitial), `http.client.
    IncompleteRead` (the connection dropped mid-body), or
    `ConnectionResetError` (an `OSError` subclass, not a `URLError`) mid-
    read would escape uncaught. These are exactly the failure modes that
    occur AFTER Discord has already applied the write server-side — an
    uncaught exception here used to propagate into `reconcile_discord` and
    trigger a rollback of DB state for a write that had, in fact, already
    succeeded (the R2-1 failure mode). The whole `with urlopen(...)` body,
    including the read+parse, is now inside the try, and the except tuple
    covers OSError (catches ConnectionResetError, IncompleteRead is also
    an OSError subclass via http.client.HTTPException... but explicitly
    listed for clarity) and json.JSONDecodeError. Same treatment for
    `HTTPError.read()`/parse below."""
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Authorization": f"Bot {token}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
            return DiscordResponse(resp.status, json.loads(body) if body else None)
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read()
            parsed = json.loads(body) if body else None
        except (OSError, TimeoutError, ssl.SSLError, http.client.HTTPException, json.JSONDecodeError):
            parsed = None
        return DiscordResponse(exc.code, parsed)
    except (
        urllib.error.URLError,
        OSError,
        TimeoutError,
        socket.timeout,
        ssl.SSLError,
        http.client.HTTPException,
        json.JSONDecodeError,
    ) as exc:
        # CRITICAL 2 / R2-3: a transient network blip (DNS failure,
        # connection reset, read timeout, TLS error, truncated body, or a
        # non-JSON 2xx body) — none of which are HTTPError — must not
        # raise out of the transport. An uncaught exception here used to
        # escape reconcile_discord mid-cycle and roll back DB state for
        # writes that had *already* succeeded on Discord — burning a real
        # rename-slot / creating a duplicate pin on the inevitable retry.
        # status=0 is not a valid Discord HTTP status, so callers treat it
        # like any other non-2xx failure and simply report+skip this
        # thread. Note: OSError is a superset that also covers
        # ConnectionResetError and (via http.client) IncompleteRead;
        # URLError is included explicitly since it is NOT an OSError
        # subclass.
        return DiscordResponse(0, {"error": str(exc)})


def _resolve_discord_token(
    config_path: Optional[Path] = None,
    openclaw_path: Optional[Path] = None,
) -> Optional[str]:
    """Resolution order (spec section 4): $THREADCTL_DISCORD_TOKEN, then
    ~/.config/threadctl/config.toml -> discord.token, then a fallback read of
    .channels.discord.token from ~/.openclaw/openclaw.json."""
    env = os.environ.get("THREADCTL_DISCORD_TOKEN")
    if env:
        return env

    cfg_path = config_path or Path.home() / ".config" / "threadctl" / "config.toml"
    if cfg_path.exists():
        with cfg_path.open("rb") as f:
            data = tomllib.load(f)
        token = data.get("discord", {}).get("token")
        if token:
            return token

    oc_path = openclaw_path or Path.home() / ".openclaw" / "openclaw.json"
    if oc_path.exists():
        try:
            data = json.loads(oc_path.read_text())
        except (json.JSONDecodeError, OSError):
            return None
        token = ((data.get("channels") or {}).get("discord") or {}).get("token")
        if token:
            return token

    return None


def _pr_repo_short(repo: str) -> str:
    return repo.split("/", 1)[-1]


# Discord message content hard cap (spec §9 references Discord's 2000-char
# limit). Leave a little headroom for the truncation marker itself.
DISCORD_MESSAGE_MAX = 2000

# Zero-width space (U+200B) spliced into mention triggers neutralises them
# without changing what a human reading the pinned message sees. A PR
# titled literally "@everyone" must not ping the whole server (HIGH 5).
_MENTION_RE = re.compile(r"@everyone|@here|<@[!&]?\d+>")


def _neutralise_mentions(text: str) -> str:
    """Defuse @everyone/@here/<@user>/<@&role> mention syntax in
    user-controlled text (PR titles) before it ever reaches a Discord
    message body."""
    return _MENTION_RE.sub(lambda m: m.group(0)[0] + "\u200b" + m.group(0)[1:], text)


def render_pin_body(thread: Dict[str, Any], prs: List[Dict[str, Any]]) -> str:
    """The pinned status message body: every PR ever tracked (including
    merged/closed — unlike the title, which only shows OPEN PRs), each
    repo-qualified (never bare, unlike title rendering) so the list reads
    unambiguously as PRs accumulate over a thread's life (spec section 9).

    HIGH 5: capped to Discord's 2000-char message limit (a thread with
    enough PRs — e.g. 14 — can otherwise exceed it), truncating whole PR
    lines from the bottom and noting how many were dropped. PR titles are
    untrusted (an agent-opened PR titled `@everyone` would otherwise ping
    the server) and are neutralised before being interpolated.
    """
    lines = [f"**{_neutralise_mentions(thread.get('title', ''))}**"]
    for p in sorted(prs, key=lambda p: (not p.get("primary"), p["number"])):
        parts = [pr_icon(p), f"{_pr_repo_short(p['repo'])}#{p['number']}"]
        if p.get("title"):
            parts.append(_neutralise_mentions(p["title"]))
        if p.get("url"):
            parts.append(p["url"])
        lines.append(" ".join(parts))

    body = "\n".join(lines)
    if len(body) <= DISCORD_MESSAGE_MAX:
        return body

    # Drop PR lines from the bottom (keeping the header) until it fits,
    # then note how many were omitted.
    header = lines[0]
    pr_lines = lines[1:]
    for keep in range(len(pr_lines) - 1, -1, -1):
        dropped = len(pr_lines) - keep
        kept_block = ("\n" + "\n".join(pr_lines[:keep])) if keep else ""
        candidate = header + kept_block + f"\n_…and {dropped} more PR(s) omitted (length limit)_"
        if len(candidate) <= DISCORD_MESSAGE_MAX:
            return candidate
    # Pathological: even the header + omitted-note doesn't fit. Best effort.
    return header[:DISCORD_MESSAGE_MAX]


def _unpin_stale(
    conn: sqlite3.Connection, thread_id: str, pin_message_id: str, token: str, transport
) -> Tuple[bool, Optional[str]]:
    """reconcile_pin mode 1: the thread has zero PRs left but still has a
    pinned message from before (e.g. the last PR was rm-pr'd, or a `bind`
    wiped everything) — unpin it and forget it."""
    resp = transport(
        "DELETE", f"{DISCORD_API}/channels/{thread_id}/pins/{pin_message_id}", token, None, DISCORD_TIMEOUT_S
    )
    if resp.status not in (200, 204):
        return False, f"failed to unpin stale message: HTTP {resp.status}"
    conn.execute("UPDATE threads SET pin_message_id=NULL WHERE thread_id=?", (thread_id,))
    _set_meta(conn, f"pin_body:{thread_id}", None)
    _record_history(conn, thread_id, "pin", "unpinned (no PRs tracked)")
    return True, None


def _create_and_pin(
    conn: sqlite3.Connection, thread_id: str, body: str, token: str, transport
) -> Tuple[bool, Optional[str]]:
    """reconcile_pin mode 2: no pinned message exists yet — post one and
    pin it (spec §9: created on the first add-pr for a thread)."""
    resp = transport("POST", f"{DISCORD_API}/channels/{thread_id}/messages", token, {"content": body}, DISCORD_TIMEOUT_S)
    if resp.status not in (200, 201):
        return False, f"failed to create pin message: HTTP {resp.status}"
    message_id = str((resp.data or {}).get("id") or "")
    if not message_id:
        return False, "pin message created but response had no id"
    pin_resp = transport(
        "PUT", f"{DISCORD_API}/channels/{thread_id}/pins/{message_id}", token, None, DISCORD_TIMEOUT_S
    )
    if pin_resp.status not in (200, 204):
        return False, f"created pin message {message_id} but failed to pin it: HTTP {pin_resp.status}"
    conn.execute("UPDATE threads SET pin_message_id=? WHERE thread_id=?", (message_id, thread_id))
    _set_meta(conn, f"pin_body:{thread_id}", body)
    _record_history(conn, thread_id, "pin", f"created and pinned message {message_id}")
    return True, None


def _edit_pin_if_changed(
    conn: sqlite3.Connection, thread_id: str, pin_message_id: str, body: str, token: str, transport
) -> Tuple[bool, Optional[str]]:
    """reconcile_pin mode 3: a pinned message already exists — edit it in
    place, but only if the rendered body actually changed (compared
    against the cached body in `meta`), to avoid a pointless PATCH every
    cycle."""
    meta_key = f"pin_body:{thread_id}"
    if _get_meta(conn, meta_key) == body:
        return False, None
    resp = transport(
        "PATCH", f"{DISCORD_API}/channels/{thread_id}/messages/{pin_message_id}", token, {"content": body}, DISCORD_TIMEOUT_S
    )
    if resp.status != 200:
        return False, f"failed to edit pin message: HTTP {resp.status}"
    _set_meta(conn, meta_key, body)
    _record_history(conn, thread_id, "pin", "edited pin message")
    return True, None


def reconcile_pin(
    conn: sqlite3.Connection,
    thread: sqlite3.Row,
    prs: List[Dict[str, Any]],
    token: Optional[str],
    transport,
    dry_run: bool,
) -> Tuple[bool, Optional[str]]:
    """Cycle step 3: create/edit/unpin the pinned status message for one
    thread. Returns (applied, error). DB fields (`pin_message_id`, the
    cached pin body in `meta`) only ever change in lockstep with a confirmed
    Discord write.

    The dry-run/no-token guard is hoisted to the very top (simplifier #1):
    every mode below either performs a Discord write or is a pure no-op
    returning (False, None), so this is exactly behaviour-preserving, and
    it makes "the cycle writes nothing in dry-run" obvious at a glance —
    the most important safety property of this function. The three modes
    themselves (unpin / create+pin / edit-if-changed) are disjoint and
    split into their own helpers (simplifier #2)."""
    if dry_run or not token:
        return False, None

    thread_id = thread["thread_id"]
    pin_message_id = thread["pin_message_id"]

    if not prs:
        if pin_message_id is None:
            return False, None
        return _unpin_stale(conn, thread_id, pin_message_id, token, transport)

    body = render_pin_body(dict(thread), prs)
    if pin_message_id is None:
        return _create_and_pin(conn, thread_id, body, token, transport)
    return _edit_pin_if_changed(conn, thread_id, pin_message_id, body, token, transport)


def reconcile_rename(
    conn: sqlite3.Connection,
    thread: sqlite3.Row,
    now: datetime,
    config: Dict[str, int],
    token: Optional[str],
    transport,
    dry_run: bool,
) -> Tuple[bool, Optional[str]]:
    """Cycle step 4, the last step of the cycle: rename the Discord thread to
    `desired_name`. Never burns a rename slot on a no-op (desired ==
    applied — critical, see module docstring), honours the
    rename_min_interval_s coalescing window, and backs off on 429 using the
    server-provided retry_after (spec section 8).

    R2-2 (critical, ROUND2-FINDINGS.md): coalescing must be attempt-based,
    not success-based. `last_rename_at` used to be written ONLY on the
    success path (status 200); a 400/403/404/500 response — or `status=0`
    from a network blip that hit AFTER Discord had already applied the
    rename (see `_discord_transport`) — left `last_rename_at` untouched,
    so the very next cycle (60s later) re-issued the identical PATCH
    instead of waiting out the full 330s window. Discord counts 4xx
    responses against the same rate-limit bucket as successful renames, so
    a couple of those in a row burns the whole budget and locks the
    thread out of renames entirely — precisely the failure this 330s
    design exists to prevent. Every attempt that actually reaches the
    network (i.e. every branch below, from the transport call onward) now
    stamps `last_rename_at`, regardless of outcome."""
    thread_id = thread["thread_id"]
    desired = thread["desired_name"]
    if desired is None or desired == thread["applied_name"]:
        return False, None

    last_rename_at = thread["last_rename_at"]
    if last_rename_at is not None:
        elapsed = (now - _parse_iso(last_rename_at)).total_seconds()
        if elapsed < config["rename_min_interval_s"]:
            return False, None

    backoff_until = thread["rename_backoff_until"]
    if backoff_until is not None and _parse_iso(backoff_until) > now:
        return False, None

    if dry_run or not token:
        return False, None

    resp = transport("PATCH", f"{DISCORD_API}/channels/{thread_id}", token, {"name": desired}, DISCORD_TIMEOUT_S)
    # Attempt-based coalescing (R2-2): stamp the attempt time once, then
    # every branch below persists it alongside whatever else it needs to
    # write, so no outcome — success, 429, 4xx/5xx, or a synthetic
    # status=0 network blip — can leave last_rename_at untouched.
    attempt_at = _now_iso()
    if resp.status == 429:
        retry_after = (resp.data or {}).get("retry_after")
        retry_after = float(retry_after) if retry_after is not None else 60.0
        backoff_at = (now + timedelta(seconds=retry_after)).strftime("%Y-%m-%dT%H:%M:%SZ")
        conn.execute(
            "UPDATE threads SET last_rename_at=?, rename_backoff_until=? WHERE thread_id=?",
            (attempt_at, backoff_at, thread_id),
        )
        _record_history(conn, thread_id, "rename", f"rate-limited, backing off {retry_after}s")
        return False, None
    if resp.status != 200:
        conn.execute("UPDATE threads SET last_rename_at=? WHERE thread_id=?", (attempt_at, thread_id))
        _record_history(conn, thread_id, "error", f"rename failed: HTTP {resp.status}")
        return False, f"rename failed: HTTP {resp.status}"

    conn.execute(
        "UPDATE threads SET applied_name=?, last_rename_at=? WHERE thread_id=?",
        (desired, attempt_at, thread_id),
    )
    _record_history(conn, thread_id, "rename", f"renamed to '{desired}'")
    return True, None


def reconcile_discord(
    conn: sqlite3.Connection,
    now: datetime,
    config: Dict[str, int],
    token: Optional[str],
    transport,
    dry_run: bool,
) -> Tuple[int, int, int]:
    """Cycle steps 3-4 for every thread: pin reconciliation, then rename
    reconciliation (rename is always last, per spec). Returns
    (pins_changed, renames_applied, errors).

    CRITICAL 2: commits per thread, not once for the whole cycle. The old
    single `with conn:` around the entire cycle meant one thread's
    transport exception (`_discord_transport` used to only catch
    `HTTPError`, not `URLError`/timeout/`SSLError`) would roll back *every*
    thread's DB state in the same transaction — including renames/pins
    that had ALREADY succeeded on Discord for threads processed earlier in
    the loop. The next cycle would then see `applied_name`/`pin_message_id`
    reverted to their old values and re-issue the POST/PUT/PATCH, creating
    a second pinned message and burning a second rename slot against the
    2-per-10-min-per-thread limit — exactly the failure this whole design
    exists to prevent.

    R2-1 (regression of C2): committing once per *thread* was still not
    enough. `reconcile_pin` and `reconcile_rename` are two SEPARATE Discord
    interactions for the same thread; the old code ran both under one
    try/except and committed only at the very end, so `reconcile_pin`'s
    confirmed, already-applied write (`pin_message_id`) sat uncommitted
    while `reconcile_rename` went and did its own network I/O. If the
    rename call raised, the shared `except` rolled back — discarding the
    pin write that had ALREADY succeeded on Discord, even though pinning
    and renaming are independent Discord resources. The next cycle then
    saw `pin_message_id` reverted to its old value and posted a SECOND
    pinned message.

    Fixed by giving each Discord interaction its own try/commit/rollback:
    `reconcile_pin`'s result is committed (or, on an unexpected exception,
    rolled back — there is nothing pin-related to lose since pin's own
    write never happened) BEFORE `reconcile_rename` ever starts its
    network I/O. A crash in the rename step can then only roll back
    rename's own not-yet-committed write; the pin commit that already
    landed is untouchable. Never roll back confirmed work — only ever roll
    back the transaction for the interaction that actually crashed."""
    pins_changed = renames_applied = errors = 0
    for thread in get_all_threads(conn):
        thread_id = thread["thread_id"]
        prs = [_pr_to_dict(pr) for pr in get_prs(conn, thread_id)]

        try:
            applied, error = reconcile_pin(conn, thread, prs, token, transport, dry_run)
            if error:
                errors += 1
                _record_history(conn, thread_id, "error", f"pin reconcile: {error}")
            elif applied:
                pins_changed += 1
            conn.commit()
        except Exception as exc:  # noqa: BLE001 - isolate one thread's failure from the rest of the cycle
            conn.rollback()
            errors += 1
            _record_history(conn, thread_id, "error", f"pin reconcile crashed: {exc}")
            conn.commit()
            # The pin step is unconfirmed/unknown at this point; skip the
            # rename step for this thread this cycle rather than risk
            # renaming on top of a half-finished pin reconciliation. The
            # next cycle will retry both from a clean, committed state.
            continue

        # Re-fetch: reconcile_pin may have just cleared pin_message_id, and
        # its commit above is now durable regardless of what happens below.
        thread = get_thread(conn, thread_id)
        try:
            applied, error = reconcile_rename(conn, thread, now, config, token, transport, dry_run)
            if error:
                errors += 1
            elif applied:
                renames_applied += 1
            conn.commit()
        except Exception as exc:  # noqa: BLE001 - isolate one thread's failure from the rest of the cycle
            conn.rollback()
            errors += 1
            _record_history(conn, thread_id, "error", f"rename reconcile crashed: {exc}")
            conn.commit()
    return pins_changed, renames_applied, errors


def run_cycle(
    db_path: Path,
    *,
    runner=_run_command,
    config: Optional[Dict[str, int]] = None,
    accounts: Optional[Dict[str, str]] = None,
    token: Optional[str] = None,
    transport=None,
    dry_run: bool = False,
) -> Dict[str, int]:
    """Run the full cycle: poll due PRs, recompute every thread's desired
    name, then reconcile the pinned message and thread rename against
    Discord (the cycle is the only Discord writer in threadctl — see module
    docstring). `--dry-run` still polls GitHub and recomputes DB state, but
    `reconcile_pin`/`reconcile_rename` skip every Discord write.

    CRITICAL 2/3: unlike the old implementation, this does NOT wrap the
    whole cycle in one `with conn:` transaction. Holding the write lock
    across every `gh` subprocess (up to 20s each) and Discord request (up
    to 15s each, up to 3/thread) starved concurrent agent commands
    (`touch`, `add-pr`, ...) of the write lock for the whole cycle
    duration — with only a 5s busy_timeout, those commands would raise
    `sqlite3.OperationalError: database is locked` with a raw traceback.
    `_poll_due_prs` and `reconcile_discord` each commit incrementally
    (per repo-batch, per thread) instead, so the lock is only ever held for
    the brief duration of an actual DB write."""
    config = {**_load_cycle_config(), **(config or {})}
    accounts = accounts or DEFAULT_ACCOUNTS
    if token is None:
        token = _resolve_discord_token()
    transport = transport or _discord_transport
    conn = connect_db(db_path)
    try:
        now = _utcnow()
        refreshed, poll_errors = _poll_due_prs(conn, now, config, runner, accounts)
        threads = recompute_desired_names(conn, now, config["stale_after_min"])
        conn.commit()
        pins_changed, renames_applied, discord_errors = reconcile_discord(
            conn, now, config, token, transport, dry_run
        )
        return {
            "refreshed": refreshed,
            "errors": poll_errors + discord_errors,
            "threads": threads,
            "pins_changed": pins_changed,
            "renames_applied": renames_applied,
        }
    finally:
        conn.close()


def cmd_cycle(args: argparse.Namespace) -> int:
    try:
        result = run_cycle(_resolve_db_path(args), dry_run=args.dry_run)
    except (OSError, tomllib.TOMLDecodeError, sqlite3.Error) as exc:
        _err(f"threadctl: cycle failed: {exc}")
        return 1
    mode = " (dry run)" if args.dry_run else ""
    print(
        f"threadctl: cycle{mode}: refreshed {result['refreshed']} PR(s), "
        f"recomputed {result['threads']} thread(s), "
        f"{result['pins_changed']} pin(s) reconciled, {result['renames_applied']} rename(s) applied, "
        f"{result['errors']} error(s)"
    )
    return 0


# ---------------------------------------------------------------------------
# commands: bind / touch / set / add-pr / rm-pr / primary-pr / show / list
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _open_db(path: Path):
    """Common connect/try/finally-close frame, previously copy-pasted
    across all 8 `cmd_*` handlers (REVIEW-SIMPLIFIER.md #5)."""
    conn = connect_db(path)
    try:
        yield conn
    finally:
        conn.close()


def _require_thread(conn: sqlite3.Connection, thread_id: str) -> Optional[sqlite3.Row]:
    """The "no such thread, bind it first" guard, previously copy-pasted
    across 5 write-path `cmd_*` handlers. Prints the standard error and
    returns None on a miss; callers do `if thread is None: return 1`."""
    thread = get_thread(conn, thread_id)
    if thread is None:
        _err(f"threadctl: no such thread '{thread_id}' (bind it first)")
    return thread


def _bump_touch_clock(conn: sqlite3.Connection, thread_id: str, now: str) -> None:
    """The last_touch_at/updated_at UPDATE, previously copy-pasted across
    5 write-path `cmd_*` handlers — bind/touch/set/add-pr/rm-pr/primary-pr
    all reset the staleness clock as a side effect."""
    conn.execute(
        "UPDATE threads SET last_touch_at=?, updated_at=? WHERE thread_id=?",
        (now, now, thread_id),
    )


def _load_pr_target(thread: sqlite3.Row, ref: str) -> Optional[Tuple[Optional[str], int]]:
    """The parse-ref-or-print-and-fail block, previously copy-pasted across
    add-pr/rm-pr/primary-pr. Returns (repo, number) or None (having already
    printed the error) on failure; callers do `if target is None: return 2`."""
    try:
        return parse_pr_ref(ref, thread["repo"])
    except ValueError as exc:
        _err(f"threadctl: {exc}")
        return None


def _is_db_locked_error(exc: sqlite3.OperationalError) -> bool:
    return "locked" in str(exc).lower() or "busy" in str(exc).lower()


def _retry_on_locked(retries: int = 3, backoff_s: float = 0.5):
    """CRITICAL 3: agent-facing write commands must never surface a raw
    `sqlite3.OperationalError: database is locked` traceback just because
    the cycle happened to be mid-write at the same instant — that's a
    failed `touch` = a thread silently going 💤 while an agent is actively
    working, the exact false signal threadctl exists to eliminate.
    `busy_timeout` (30s, see connect_db) already absorbs the vast majority
    of contention by blocking inside SQLite before raising; this decorator
    is the last line of defense — a couple of short retries, then a clean
    one-line message instead of a stack trace."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(args: argparse.Namespace) -> int:
            last_exc: Optional[sqlite3.OperationalError] = None
            for attempt in range(retries):
                try:
                    return func(args)
                except sqlite3.OperationalError as exc:
                    if not _is_db_locked_error(exc):
                        raise
                    last_exc = exc
                    if attempt < retries - 1:
                        time.sleep(backoff_s * (attempt + 1))
            _err(f"threadctl: database busy, try again in a moment ({last_exc})")
            return 1
        return wrapper
    return decorator


@_retry_on_locked()
def cmd_bind(args: argparse.Namespace) -> int:
    repo = args.repo
    if not REPO_RE.match(repo):
        _err(f"threadctl: invalid --repo '{repo}' (expected owner/name)")
        return 2
    owner = repo.split("/", 1)[0]
    title, truncated = _enforce_title(args.title)
    now = _now_iso()

    with _open_db(_resolve_db_path(args)) as conn:
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


@_retry_on_locked()
def cmd_touch(args: argparse.Namespace) -> int:
    now = _now_iso()
    with _open_db(_resolve_db_path(args)) as conn:
        with conn:
            if _require_thread(conn, args.thread_id) is None:
                return 1
            _bump_touch_clock(conn, args.thread_id, now)
            _record_history(conn, args.thread_id, "touch", args.note)
        print(f"threadctl: touched '{args.thread_id}'")
        return 0


@_retry_on_locked()
def cmd_set(args: argparse.Namespace) -> int:
    if args.question and args.answered:
        _err("threadctl: --question and --answered are mutually exclusive")
        return 2

    with _open_db(_resolve_db_path(args)) as conn:
        with conn:
            thread = _require_thread(conn, args.thread_id)
            if thread is None:
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


@_retry_on_locked()
def cmd_add_pr(args: argparse.Namespace) -> int:
    with _open_db(_resolve_db_path(args)) as conn:
        thread = _require_thread(conn, args.thread_id)
        if thread is None:
            return 1
        target = _load_pr_target(thread, args.ref)
        if target is None:
            return 2
        repo, number = target
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
            _bump_touch_clock(conn, args.thread_id, now)
            verb = "already tracked" if existing is not None else "added"
            suffix = " (primary)" if make_primary else ""
            _record_history(conn, args.thread_id, "pr", f"{verb} {repo}#{number}{suffix}")

        print(f"threadctl: {repo}#{number} tracked on '{args.thread_id}'" + (" (primary)" if make_primary else ""))
        return 0


@_retry_on_locked()
def cmd_rm_pr(args: argparse.Namespace) -> int:
    with _open_db(_resolve_db_path(args)) as conn:
        thread = _require_thread(conn, args.thread_id)
        if thread is None:
            return 1
        target = _load_pr_target(thread, args.ref)
        if target is None:
            return 2
        repo, number = target

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
            _bump_touch_clock(conn, args.thread_id, now)
            _record_history(conn, args.thread_id, "pr", f"removed {repo}#{number}")

        print(f"threadctl: removed {repo}#{number} from '{args.thread_id}'")
        return 0


@_retry_on_locked()
def cmd_primary_pr(args: argparse.Namespace) -> int:
    with _open_db(_resolve_db_path(args)) as conn:
        thread = _require_thread(conn, args.thread_id)
        if thread is None:
            return 1
        target = _load_pr_target(thread, args.ref)
        if target is None:
            return 2
        repo, number = target

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
            _bump_touch_clock(conn, args.thread_id, now)
            _record_history(conn, args.thread_id, "pr", f"primary -> {repo}#{number}")

        print(f"threadctl: {repo}#{number} is now primary on '{args.thread_id}'")
        return 0


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
    with _open_db(_resolve_db_path(args)) as conn:
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


def cmd_list(args: argparse.Namespace) -> int:
    with _open_db(_resolve_db_path(args)) as conn:
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


# ---------------------------------------------------------------------------
# migrate: one-shot import from the old thread-state registry.json
# ---------------------------------------------------------------------------

# Old thread-state statuses -> new threadctl statuses (spec section 11).
# blocked/review were auto-derived from PR state in the old tool; the new
# cycle re-derives the equivalent icon from PR state itself, so both simply
# become "active" here.
MIGRATE_STATUS_MAP = {
    "active": "active",
    "blocked": "active",
    "review": "active",
    "merged": "merged",
    "closed": "closed",
    "done": "closed",
    "paused": "paused",
}


def _migrate_thread(conn: sqlite3.Connection, thread_id: str, entry: Dict[str, Any]) -> bool:
    """Import one thread from the old registry. Returns True if the
    thread's repo was invalid/missing and had to be imported as NULL
    (HIGH 4) — the caller aggregates these into a loud summary so a human
    can `bind --repo` them by hand.

    HIGH 4 root cause: the old code did `repo = entry.get("repo") or ""`,
    silently storing owner='' repo=''. `_check_pr_owner` then compared
    ''=='' and passed, so `add-pr` could insert a PR with repo='', and
    `_graphql_query`'s `''.split('/', 1)` raised `ValueError: not enough
    values to unpack` — uncaught, and (via the old single-transaction
    CRITICAL 2 bug) that one bad thread rolled back polling+renaming for
    all 25. Fixed at the boundary: validate against REPO_RE here and store
    NULL (not '') for anything invalid; the poller (`_poll_due_prs`) and
    `_graphql_query`'s callers already skip+record a NULL/invalid repo
    instead of raising."""
    raw_repo = entry.get("repo")
    repo_invalid = not raw_repo or not REPO_RE.match(raw_repo)
    repo = None if repo_invalid else raw_repo
    owner = None if repo_invalid else raw_repo.split("/", 1)[0]
    title, _truncated = _enforce_title(entry.get("title") or thread_id)
    old_status = entry.get("status") or "active"
    status = MIGRATE_STATUS_MAP.get(old_status, "active")
    now = _now_iso()
    last_touch_at = entry.get("updatedAt") or now
    created_at = entry.get("createdAt") or now

    conn.execute(
        "INSERT INTO threads (thread_id, title, owner, repo, status, question, bound_at, "
        "last_touch_at, applied_name, pin_message_id, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?, ?)",
        (
            thread_id, title, owner, repo, status,
            created_at, last_touch_at, entry.get("discordName"), entry.get("cardMessageId"),
            created_at, now,
        ),
    )
    for pr in entry.get("prs") or []:
        pr_repo = pr.get("repo") or repo
        number = pr.get("number")
        if number is None or not pr_repo or not REPO_RE.match(pr_repo):
            # A PR with no usable repo (inherits the thread's invalid repo,
            # or has none of its own) would otherwise crash the poller the
            # same way as HIGH 4 — skip it here instead of importing junk.
            continue
        pr_state = pr.get("prState") or {}
        conn.execute(
            "INSERT OR IGNORE INTO prs (thread_id, repo, number, is_primary, state, ci, "
            "pr_title, url, checked_at, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                thread_id, pr_repo, number, int(bool(pr.get("primary"))),
                pr_state.get("state"), pr_state.get("ci"),
                pr_state.get("title"), pr_state.get("url"), pr_state.get("checkedAt"),
                now,
            ),
        )
    note = f"migrated from thread-state registry.json (status={old_status})"
    if repo_invalid:
        note += f"; invalid/missing repo '{raw_repo}' imported as NULL — run `threadctl bind --repo owner/name` to fix"
    _record_history(conn, thread_id, "bind", note)
    return repo_invalid


def migrate_registry(conn: sqlite3.Connection, registry: Dict[str, Any]) -> Tuple[int, List[str], List[str]]:
    """Import every thread from an old thread-state registry dict. Returns
    (imported_count, skipped_thread_ids, invalid_repo_thread_ids) — a
    thread already present in the DB is left untouched and reported as
    skipped, so migrate is safe to re-run without clobbering live state.
    `invalid_repo_thread_ids` (HIGH 4) lists every thread imported with a
    NULL repo because the source data had none/an unparseable one — e.g.
    the 11 of 25 live threads with `"repo": null` — so a human can follow
    up with `bind --repo`."""
    imported = 0
    skipped = []
    invalid_repo = []
    threads = registry.get("threads") or {}
    for thread_id, entry in threads.items():
        if get_thread(conn, thread_id) is not None:
            skipped.append(thread_id)
            continue
        if _migrate_thread(conn, thread_id, entry):
            invalid_repo.append(thread_id)
        imported += 1
    return imported, skipped, invalid_repo


def cmd_migrate(args: argparse.Namespace) -> int:
    src = Path(args.from_path).expanduser()
    try:
        data = json.loads(src.read_text())
    except OSError as exc:
        _err(f"threadctl: could not read '{src}': {exc}")
        return 1
    except json.JSONDecodeError as exc:
        _err(f"threadctl: '{src}' is not valid JSON: {exc}")
        return 1

    with _open_db(_resolve_db_path(args)) as conn:
        try:
            with conn:
                imported, skipped, invalid_repo = migrate_registry(conn, data)
        except sqlite3.Error as exc:
            _err(f"threadctl: migration failed: {exc}")
            return 1

    print(f"threadctl: migrated {imported} thread(s) from '{src}'")
    if skipped:
        _err(f"threadctl: skipped {len(skipped)} already-bound thread(s): {', '.join(sorted(skipped))}")
    if invalid_repo:
        _err(
            f"threadctl: WARNING — {len(invalid_repo)} thread(s) imported with an invalid/missing repo, "
            f"as NULL (polling+renaming paused for them until fixed): {', '.join(sorted(invalid_repo))}"
        )
        _err("threadctl: fix each with: threadctl bind <thread_id> --title \"...\" --repo owner/name")
    print(f"threadctl: '{src}' left untouched as a backup")
    return 0


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

    sp_cycle = sub.add_parser("cycle", parents=[parent], help="poll GitHub and recompute desired Discord names")
    sp_cycle.add_argument("--dry-run", action="store_true", help="report the cycle without Discord writes")
    sp_cycle.set_defaults(func=cmd_cycle)

    sp_list = sub.add_parser("list", parents=[parent], help="list all threads")
    sp_list.add_argument("--json", action="store_true")
    sp_list.set_defaults(func=cmd_list)

    sp_migrate = sub.add_parser("migrate", parents=[parent], help="one-shot import from the old thread-state registry.json")
    sp_migrate.add_argument("--from", dest="from_path", required=True, metavar="PATH")
    sp_migrate.set_defaults(func=cmd_migrate)

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
