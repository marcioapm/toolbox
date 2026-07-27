#!/usr/bin/env python3
"""thread-state — registry + GitHub polling + rendering for long-lived
coding threads (one per Discord thread).

Owns state + GitHub polling + rendering only. It does NOT talk to Discord —
that happens elsewhere via an already-authenticated bot. This CLI is the
scriptable source of truth an assistant can query to render a pinned status
card and a thread title without re-deriving state from chat history.

Registry storage::

    ~/.config/thread-state/registry.json      default path
    $THREAD_STATE_REGISTRY / --registry PATH  overrides

    {"schemaVersion": 5, "accounts": {...}, "threads": {"<threadId>": {...}}}

A thread entry carries a `prs` list (one or more PRs, each with its own
`prState`), a `branches` list (one or more branches, one marked primary), and
a `runs` list (agent-run tasks tracked on remote hosts, each with its own
polled `state`). It also carries `lastActivityAt`/`lastActivitySource` — the
last time something was visibly posted in the thread, either set directly or
derived from a Discord message-id snowflake. It also carries an `updates`
list — a rolling, capped work-summary log (`{"ts", "text", "kind"}`, newest
last), appended explicitly via `set --progress` (`kind: "manual"`) or
automatically by `refresh`/`probe` on a genuine state transition (`kind:
"transition"`); see `append_update` and `_pr_transition_texts` /
`_run_transition_texts`. It also carries `cardMessageId`/`parentChannelId`/
`discordName` — the identity of the pinned Discord status-card message, so an
automated monitor can edit it in place instead of rediscovering it or posting
a duplicate. A v1 registry (scalar `pr` + top-level `prState`), a v2 registry
(scalar `branch`), a v3 registry (no `updates`), and a v4 registry (no card
identity fields) are migrated automatically on first load, chaining through
to the current version; see `_migrate_registry_data`.

Writes are atomic (temp file + os.replace) and guarded by a per-registry
in-process lock plus an advisory lock on a stable sidecar for the whole
read-modify-write, so concurrent invocations can't interleave and corrupt the
file. A registry that fails to parse is never overwritten — it is copied to
`registry.json.bad-<timestamp>` and the command exits non-zero.

`refresh` derives a thread's `status` from its polled PR states (see
`derive_thread_status`), but only ever sets `active` or `blocked`
automatically: an OPEN PR with failing CI sets `blocked`, and that clears
back to `active` once no PR is both open and failing. `merged`/`closed`/
`done`/`paused` are manual-only — `refresh` never sets them, since "every
currently-known PR is merged" is not proof a thread's workstream is
finished (long-running threads keep opening new PRs against the same
registry entry). Use `set --status` to move a thread into or out of one of
those explicitly.

Usage::

    thread-state add <threadId> --title T [--repo R] [--pr N] [--add-pr REF] ...
    thread-state set <threadId> [--title T] [--status S] [--add-pr REF] [--rm-pr REF] [--primary-pr REF] ...
    thread-state rm <threadId>
    thread-state list [--status S] [--repo R] [--tag T] [--silent-for DURATION] [--stalled]
    thread-state silent [--for DURATION]
    thread-state show <threadId>
    thread-state refresh [<threadId>... | --all]
    thread-state probe [<threadId>... | --all]
    thread-state card <threadId> [--format markdown|embed-json]
    thread-state title <threadId>
    thread-state scan <threadId> --file PATH [--apply]
    thread-state log <threadId> [--limit N]

A PR reference (REF) accepts `4694`, `#4694`, `owner/name#4694`, or a full
`https://github.com/owner/name/pull/4694` URL. A branch reference accepts a
bare branch name or `repo:branch`. A run reference is `host:name`.

Every command accepts --json for machine-readable stdout output.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

STATUSES = ("active", "review", "blocked", "merged", "closed", "paused", "done")

# Hosts `--add-run`/`probe` know how to reach. Adding a run against any other
# host requires --force (typo guard — these tasks run unattended, a silent
# typo in a host name would mean the run is polled forever and never flagged).
KNOWN_HOSTS = ("vibes", "macmini", "marcio-xps")

# Statuses considered "expected to be progressing" by the silent-thread
# detector; overridable with --include-status. Never includes a resolved
# status (merged/closed/done/paused) regardless of override.
DEFAULT_SILENT_STATUSES = ("active", "review")
NEVER_SILENT_STATUSES = ("merged", "closed", "done", "paused")

# Card rendering's own (fixed) threshold for flagging a `running` agent-run
# as producing no output — independent of `list --silent-for`'s caller-given
# duration, since the card has no DURATION argument to take one from.
CARD_STALL_THRESHOLD_SECONDS = 15 * 60

STATUS_EMOJI = {
    "active": "🟢",
    "review": "🔵",
    "blocked": "🔴",
    "merged": "✅",
    "closed": "⚫",
    "paused": "⏸",
    "done": "✅",
}

# Discord embed colors (decimal RGB ints). "paused" has no explicit spec
# color — reuses closed's neutral grey rather than inventing a new one.
STATUS_COLOR = {
    "active": 0xF1C40F,
    "review": 0x3498DB,
    "blocked": 0xE74C3C,
    "merged": 0x2ECC71,
    "closed": 0x95A5A6,
    "paused": 0x95A5A6,
    "done": 0x2ECC71,
}

# Statuses where CI failure never overrides the icon/color — the PR is
# already resolved one way or the other.
_TERMINAL_STATUSES = ("merged", "closed", "done")

DEFAULT_ACCOUNTS = {"absmartly": "marcio-absmartly", "marcioapm": "marcioapm"}

GH_TIMEOUT_DEFAULT = 20
REFRESH_MAX_WORKERS = 6

CI_FAILURE_STATES = {"FAILURE", "ERROR"}
CI_PENDING_STATES = {"PENDING", "IN_PROGRESS", "QUEUED"}

PR_URL_RE = re.compile(r"https://github\.com/([\w.-]+)/([\w.-]+)/pull/(\d+)")
BARE_PR_RE = re.compile(r"(?<![\w/])#(\d+)\b")

# Accepted forms for a `--add-pr`/`--rm-pr`/`--primary-pr` REF argument:
# "4694", "#4694", "owner/name#4694", or a full PR URL.
PR_REF_RE = re.compile(
    r"^(?:https://github\.com/(?P<url_owner>[\w.-]+)/(?P<url_name>[\w.-]+)/pull/(?P<url_num>\d+)"
    r"|(?:(?P<repo>[\w.-]+/[\w.-]+)#)?#?(?P<num>\d+))$"
)

THREAD_TITLE_MAX = 100

SCHEMA_VERSION = 5

# `updates` list caps: max entries retained (oldest evicted first) and max
# chars per entry (hard-truncated with an ellipsis, since these render as
# single dashboard lines, not a log blob).
UPDATES_MAX_ENTRIES = 10
UPDATE_TEXT_MAX_CHARS = 120

# How many of the most recent `updates` the card shows, and the degradation
# ladder (3 -> 2 -> 1) applied when the full card would exceed Discord's
# practical message-length safety margin.
CARD_RECENT_UPDATES = 3
CARD_MAX_CHARS = 1800

EMPTY_PR_STATE = {
    "state": None,
    "ci": "NONE",
    "title": "",
    "url": "",
    "mergedAt": None,
    "checkedAt": None,
    "stale": False,
    "error": None,
}

EMPTY_RUN_STATE = {
    "status": "missing",
    "exitCode": None,
    "logBytes": None,
    "logMtime": None,
    "checkedAt": None,
    "stale": False,
    "error": None,
}

# Discord snowflake -> Unix epoch: snowflakes encode a millisecond timestamp
# relative to the "Discord Epoch" (2015-01-01T00:00:00Z) in their high 42
# bits. See https://discord.com/developers/docs/reference#snowflakes.
DISCORD_EPOCH_MS = 1420070400000

# A Discord snowflake is a decimal-digit string (fits in an unsigned 64-bit
# int, but validation only cares that it's all digits — `--card-message-id`/
# `--parent-channel-id` reject anything else with a clear error).
SNOWFLAKE_RE = re.compile(r"^\d+$")

RUN_HOST_NAME_RE = re.compile(r"^(?P<host>[\w.-]+):(?P<name>[\w.-]+)$")

DURATION_RE = re.compile(r"^(?P<value>\d+)(?P<unit>s|m|h|d)$")
DURATION_UNIT_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400}


class RegistryError(Exception):
    """Raised when the registry file exists but cannot be trusted (corrupt
    JSON or an unexpected shape). Callers must never write on this path."""


class _Abort(Exception):
    """Internal control-flow signal: abort the in-flight registry
    read-modify-write without persisting any change, and exit with `code`."""

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
    monkeypatch for deterministic silent/stalled-detector assertions."""
    return datetime.now(timezone.utc)


def _err(message: str) -> None:
    print(message, file=sys.stderr)


def _validate_status(status: str) -> str:
    if status not in STATUSES:
        raise ValueError(f"invalid status '{status}', must be one of: {', '.join(STATUSES)}")
    return status


def _split_tags(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    return [t.strip() for t in raw.split(",") if t.strip()]


def snowflake_to_iso(snowflake: str) -> str:
    """Decode a Discord message-id snowflake into an ISO-8601 UTC timestamp,
    per Discord's documented scheme: the top 42 bits are milliseconds since
    the Discord Epoch (2015-01-01T00:00:00Z). Doing this decode ourselves
    (rather than asking the caller to pass a timestamp) avoids clock-skew
    mistakes at the call site."""
    epoch_ms = (int(snowflake) >> 22) + DISCORD_EPOCH_MS
    return datetime.fromtimestamp(epoch_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_duration(raw: str) -> int:
    """Parse a duration like `35m`, `2h`, `90s`, `1d` into seconds. Raises
    ValueError on anything else (no unit, unknown unit, negative, etc.)."""
    m = DURATION_RE.match(raw.strip())
    if not m:
        raise ValueError(f"invalid duration '{raw}' (expected e.g. 35m, 2h, 90s, 1d)")
    return int(m.group("value")) * DURATION_UNIT_SECONDS[m.group("unit")]


def validate_snowflake(raw: str) -> str:
    """Validate a Discord snowflake CLI argument (decimal digits only).
    Raises ValueError on anything else."""
    if not SNOWFLAKE_RE.match(raw):
        raise ValueError(f"invalid snowflake '{raw}' (expected a string of decimal digits)")
    return raw


def _parse_iso(ts: str) -> datetime:
    return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def _humanize_age_seconds(seconds: float) -> str:
    """A short age string like `41m`, `11h`, `2d` — the coarsest unit that
    doesn't round to 0."""
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m"
    hours = minutes // 60
    if hours < 24:
        return f"{hours}h"
    days = hours // 24
    return f"{days}d"


# ---------------------------------------------------------------------------
# schema migration: v1 (scalar `pr` + top-level `prState`) -> v2 (`prs` list)
# ---------------------------------------------------------------------------

def _migrate_entry_v1_to_v2(entry: Dict[str, Any]) -> None:
    """Mutate a single thread entry in place. A no-op if it already has a
    `prs` list (idempotent)."""
    if "prs" in entry:
        entry.pop("pr", None)
        entry.pop("prState", None)
        return
    pr = entry.pop("pr", None)
    old_pr_state = entry.pop("prState", None)
    if pr:
        new_state = dict(EMPTY_PR_STATE)
        if old_pr_state:
            new_state.update(old_pr_state)
        entry["prs"] = [{
            "repo": entry.get("repo"),
            "number": pr,
            "primary": True,
            "prState": new_state,
        }]
    else:
        entry["prs"] = []


def _migrate_entry_v2_to_v3(entry: Dict[str, Any]) -> None:
    """Mutate a single thread entry in place: scalar `branch` -> `branches`
    list, plus the new `runs`/`lastActivityAt`/`lastActivitySource` fields.
    A no-op if it already has a `branches` list (idempotent)."""
    if "branches" in entry:
        entry.pop("branch", None)
    else:
        branch = entry.pop("branch", None)
        if branch:
            entry["branches"] = [{"repo": entry.get("repo"), "name": branch, "primary": True}]
        else:
            entry["branches"] = []
    entry.setdefault("runs", [])
    entry.setdefault("lastActivityAt", None)
    entry.setdefault("lastActivitySource", None)


def _migrate_entry_v3_to_v4(entry: Dict[str, Any]) -> None:
    """Mutate a single thread entry in place: add the new `updates` list.
    A no-op if it already has one (idempotent)."""
    entry.setdefault("updates", [])


def _migrate_entry_v4_to_v5(entry: Dict[str, Any]) -> None:
    """Mutate a single thread entry in place: add the new `cardMessageId`/
    `parentChannelId`/`discordName` fields, defaulting to null. A no-op if
    they're already present (idempotent)."""
    entry.setdefault("cardMessageId", None)
    entry.setdefault("parentChannelId", None)
    entry.setdefault("discordName", None)


def _migrate_registry_data(data: Dict[str, Any]) -> bool:
    """Migrate an in-memory registry dict up to the current SCHEMA_VERSION,
    applying v1->v2, v2->v3, v3->v4, and v4->v5 in order as needed. Idempotent
    — a no-op once `schemaVersion` is already current. Returns True if any
    conversion actually happened, so callers that persist to disk know
    whether to back up the pre-migration file first."""
    version = data.get("schemaVersion", 1)
    if version >= SCHEMA_VERSION:
        return False
    if version < 2:
        for entry in data.get("threads", {}).values():
            _migrate_entry_v1_to_v2(entry)
        data["schemaVersion"] = 2
    if data["schemaVersion"] < 3:
        for entry in data.get("threads", {}).values():
            _migrate_entry_v2_to_v3(entry)
        data["schemaVersion"] = 3
    if data["schemaVersion"] < 4:
        for entry in data.get("threads", {}).values():
            _migrate_entry_v3_to_v4(entry)
        data["schemaVersion"] = 4
    if data["schemaVersion"] < 5:
        for entry in data.get("threads", {}).values():
            _migrate_entry_v4_to_v5(entry)
        data["schemaVersion"] = 5
    return True


# ---------------------------------------------------------------------------
# registry: path resolution, atomic read-modify-write, corruption handling
# ---------------------------------------------------------------------------

def _resolve_registry_path(args: argparse.Namespace) -> Path:
    """--registry flag wins, then $THREAD_STATE_REGISTRY, then the default
    under ~/.config/thread-state/."""
    explicit = getattr(args, "registry", None)
    if explicit:
        return Path(explicit).expanduser()
    env = os.environ.get("THREAD_STATE_REGISTRY")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".config" / "thread-state" / "registry.json"


def _empty_registry() -> Dict[str, Any]:
    return {"schemaVersion": SCHEMA_VERSION, "accounts": {}, "threads": {}}


def _timestamp_slug() -> str:
    return _now_iso().replace(":", "").replace("-", "")


def _parse_registry(path: Path) -> Dict[str, Any]:
    """Read and parse the registry, or return an empty one if it doesn't
    exist yet. Raises RegistryError (without touching the file) if it exists
    but is corrupt, after copying it aside for forensics. Does NOT migrate —
    callers that need v2 shape should go through `_load_and_migrate`."""
    if not path.exists():
        return _empty_registry()
    raw = path.read_text()
    if not raw.strip():
        return _empty_registry()
    try:
        data = json.loads(raw)
        if not isinstance(data, dict) or not isinstance(data.get("threads", {}), dict):
            raise ValueError("registry JSON must be an object with a 'threads' object")
    except (json.JSONDecodeError, ValueError) as exc:
        backup = path.with_name(path.name + f".bad-{_timestamp_slug()}")
        try:
            shutil.copy2(path, backup)
        except OSError:
            pass
        raise RegistryError(
            f"thread-state: registry at {path} is corrupt ({exc}); "
            f"backed up to {backup}, original left untouched"
        ) from exc
    data.setdefault("schemaVersion", 1)
    data.setdefault("accounts", {})
    data.setdefault("threads", {})
    return data


def _load_and_migrate(path: Path) -> Tuple[Dict[str, Any], bool]:
    """`_parse_registry` plus an automatic v1 -> v2 migration applied
    in-memory. Returns (data, migrated) — `migrated` tells callers that
    intend to persist whether they must back up the pre-migration file
    first (see `_locked_registry`)."""
    data = _parse_registry(path)
    migrated = _migrate_registry_data(data)
    return data, migrated


def _write_registry(path: Path, data: Dict[str, Any]) -> None:
    """Atomic write: temp file in the same dir, fsync, then os.replace()."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".registry.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


_registry_thread_locks: Dict[Path, Any] = {}
_registry_thread_locks_guard = threading.Lock()


def _registry_lock_path(path: Path) -> Path:
    return path.with_name(path.name + ".lock")


def _registry_thread_lock(path: Path) -> Any:
    resolved_path = path.expanduser().resolve()
    with _registry_thread_locks_guard:
        return _registry_thread_locks.setdefault(resolved_path, threading.RLock())


@contextlib.contextmanager
def _locked_registry(path: Path):
    """Exclusive-lock the registry for a read-modify-write. The yielded dict
    is only persisted if the block completes without raising — raise
    `_Abort` (or anything else) to discard in-progress changes.

    The lock lives on a stable sidecar rather than the registry itself because
    `_write_registry` replaces the registry inode. A per-path RLock supplies
    the thread-level exclusion that flock cannot guarantee within one process.

    If the on-disk file is behind SCHEMA_VERSION, it is migrated in memory
    before being handed to the caller; the very first time that migration is
    about to be persisted, the pre-migration bytes are backed up to
    `registry.json.v<N>-backup-<ts>` (N = the on-disk schemaVersion before
    migration) so no data can be lost."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with _registry_thread_lock(path):
        fd = os.open(_registry_lock_path(path), os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            pre_migration_raw = path.read_text() if path.exists() and path.stat().st_size > 0 else None
            pre_migration_version = 1
            if pre_migration_raw:
                try:
                    pre_migration_version = json.loads(pre_migration_raw).get("schemaVersion", 1)
                except json.JSONDecodeError:
                    pass
            data, migrated = _load_and_migrate(path)
            yield data
            if migrated and pre_migration_raw is not None:
                backup = path.with_name(path.name + f".v{pre_migration_version}-backup-{_timestamp_slug()}")
                try:
                    backup.write_text(pre_migration_raw)
                except OSError:
                    pass
            _write_registry(path, data)
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)


@contextlib.contextmanager
def _read_only_registry(path: Path):
    """Shared-lock read of the registry. Never writes — a v1 file is
    migrated to v2 in memory for the caller's benefit, but the on-disk file
    is left exactly as it was."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with _registry_thread_lock(path):
        fd = os.open(_registry_lock_path(path), os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_SH)
            data, _migrated = _load_and_migrate(path)
            yield data
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)


# ---------------------------------------------------------------------------
# multi-PR helpers: ref parsing, get/add/remove/primary, derived status
# ---------------------------------------------------------------------------

def parse_pr_ref(ref: str, default_repo: Optional[str] = None) -> Tuple[Optional[str], int]:
    """Parse a `--add-pr`/`--rm-pr`/`--primary-pr` argument into
    (repo_or_None, number). Accepts `4694`, `#4694`, `owner/name#4694`, or a
    full `https://github.com/owner/name/pull/4694` URL. `repo` is None when
    the ref didn't name one (caller should fall back to the entry's repo)."""
    ref = ref.strip()
    m = PR_REF_RE.match(ref)
    if not m:
        raise ValueError(f"invalid PR reference '{ref}' (expected N, #N, owner/name#N, or a PR URL)")
    if m.group("url_num"):
        return f"{m.group('url_owner')}/{m.group('url_name')}", int(m.group("url_num"))
    repo = m.group("repo") or default_repo
    return repo, int(m.group("num"))


def _pr_repo(pr_entry: Dict[str, Any], thread_repo: Optional[str]) -> Optional[str]:
    """A PR entry may override the thread's `repo`; if absent, inherit it."""
    return pr_entry.get("repo") or thread_repo


def get_prs(entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    return entry.get("prs") or []


def primary_pr(entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The PR marked `primary: true`, or the first PR if none is marked."""
    prs = get_prs(entry)
    if not prs:
        return None
    for p in prs:
        if p.get("primary"):
            return p
    return prs[0]


def set_prs_single(entry: Dict[str, Any], repo: Optional[str], number: int) -> None:
    """`--pr N` semantics: replace the PR list with exactly this one PR,
    marked primary."""
    entry["prs"] = [{
        "repo": repo,
        "number": number,
        "primary": True,
        "prState": dict(EMPTY_PR_STATE),
    }]


def add_pr(entry: Dict[str, Any], repo: Optional[str], number: int) -> None:
    """Append a PR, deduping by (effective repo, number). Re-adding an
    existing PR is a no-op. No `primary` flag is set here — `primary_pr`
    already falls back to the first PR in the list when none is marked."""
    thread_repo = entry.get("repo")
    target_repo = repo or thread_repo
    prs = entry.setdefault("prs", [])
    for p in prs:
        if _pr_repo(p, thread_repo) == target_repo and p.get("number") == number:
            return
    prs.append({
        "repo": repo,
        "number": number,
        "primary": False,
        "prState": dict(EMPTY_PR_STATE),
    })


def remove_pr(entry: Dict[str, Any], repo: Optional[str], number: int) -> bool:
    """Remove a PR by (effective repo, number). Returns True if a PR was
    removed. If the removed PR was primary and PRs remain, the new first PR
    becomes primary."""
    thread_repo = entry.get("repo")
    target_repo = repo or thread_repo
    prs = get_prs(entry)
    for i, p in enumerate(prs):
        if _pr_repo(p, thread_repo) == target_repo and p.get("number") == number:
            was_primary = bool(p.get("primary"))
            del prs[i]
            if was_primary and prs:
                prs[0]["primary"] = True
            return True
    return False


def set_primary_pr(entry: Dict[str, Any], repo: Optional[str], number: int) -> bool:
    """Mark exactly one PR (by effective repo, number) as primary, clearing
    the flag on all others. Returns True if the target PR was found."""
    thread_repo = entry.get("repo")
    target_repo = repo or thread_repo
    prs = get_prs(entry)
    found = False
    for p in prs:
        is_target = _pr_repo(p, thread_repo) == target_repo and p.get("number") == number
        p["primary"] = is_target
        found = found or is_target
    return found


def derive_thread_status(entry: Dict[str, Any]) -> Optional[str]:
    """Derive an overall thread status from its PR set, per the rules in
    the module docstring. Returns None when no status change should be
    applied (e.g. no PRs at all, or the existing status should be kept).

    `merged`/`closed`/`done`/`paused` are manual-only terminal-ish statuses —
    this function never *sets* them. A thread's known PR list is not
    necessarily complete: long-running workstream threads keep opening new
    PRs against the same thread, so "all known PRs are merged" is not
    sufficient evidence the workstream itself is finished. When that
    happens while the thread is in an in-flight status (active/review/
    blocked), the merge is already visible in the PR list and the updates
    log, so the safe move is to leave `status` alone (return None) rather
    than flip it to `merged`/`closed` and silently discard whatever the
    human had set."""
    prs = get_prs(entry)
    if not prs:
        return None

    current = entry.get("status")
    states = [(p.get("prState") or {}).get("state") for p in prs]
    cis = [(p.get("prState") or {}).get("ci") for p in prs]

    if any(s == "OPEN" and ci == "FAILURE" for s, ci in zip(states, cis)):
        return "blocked"

    if any(s == "OPEN" for s in states):
        if current in ("review", "blocked", "paused"):
            return current
        return "active"

    # No PR is OPEN and none is failing CI -- all are MERGED/CLOSED. This
    # clears an auto-set `blocked` (no more failing open PR to justify it)
    # but never promotes the thread to a manual-only status like
    # merged/closed on its own.
    if current == "blocked":
        return "active"
    return None


# ---------------------------------------------------------------------------
# branch helpers: get/add/remove/primary (mirrors the multi-PR helpers above)
# ---------------------------------------------------------------------------

def _branch_repo(branch_entry: Dict[str, Any], thread_repo: Optional[str]) -> Optional[str]:
    return branch_entry.get("repo") or thread_repo


def get_branches(entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    return entry.get("branches") or []


def primary_branch(entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The branch marked `primary: true`, or the first branch if none is
    marked."""
    branches = get_branches(entry)
    if not branches:
        return None
    for b in branches:
        if b.get("primary"):
            return b
    return branches[0]


def set_branches_single(entry: Dict[str, Any], repo: Optional[str], name: str) -> None:
    """`--branch B` semantics: replace the branch list with exactly this
    one, marked primary."""
    entry["branches"] = [{"repo": repo, "name": name, "primary": True}]


def add_branch(entry: Dict[str, Any], repo: Optional[str], name: str) -> None:
    """Append a branch, deduping by (effective repo, name). Re-adding an
    existing branch is a no-op."""
    thread_repo = entry.get("repo")
    target_repo = repo or thread_repo
    branches = entry.setdefault("branches", [])
    for b in branches:
        if _branch_repo(b, thread_repo) == target_repo and b.get("name") == name:
            return
    branches.append({"repo": repo, "name": name, "primary": not branches})


def remove_branch(entry: Dict[str, Any], repo: Optional[str], name: str) -> bool:
    """Remove a branch by (effective repo, name). Returns True if removed.
    If the removed branch was primary and branches remain, the new first
    branch becomes primary."""
    thread_repo = entry.get("repo")
    target_repo = repo or thread_repo
    branches = get_branches(entry)
    for i, b in enumerate(branches):
        if _branch_repo(b, thread_repo) == target_repo and b.get("name") == name:
            was_primary = bool(b.get("primary"))
            del branches[i]
            if was_primary and branches:
                branches[0]["primary"] = True
            return True
    return False


def set_primary_branch(entry: Dict[str, Any], repo: Optional[str], name: str) -> bool:
    """Mark exactly one branch (by effective repo, name) as primary,
    clearing the flag on all others. Returns True if the target was found."""
    thread_repo = entry.get("repo")
    target_repo = repo or thread_repo
    branches = get_branches(entry)
    found = False
    for b in branches:
        is_target = _branch_repo(b, thread_repo) == target_repo and b.get("name") == name
        b["primary"] = is_target
        found = found or is_target
    return found


def parse_branch_ref(ref: str, default_repo: Optional[str] = None) -> Tuple[Optional[str], str]:
    """Parse a `--add-branch`/`--rm-branch`/`--primary-branch` argument into
    (repo_or_None, name). Accepts a bare branch name or `repo:branch`."""
    ref = ref.strip()
    if ":" in ref:
        repo, name = ref.split(":", 1)
        if not repo or not name:
            raise ValueError(f"invalid branch reference '{ref}' (expected BRANCH or repo:BRANCH)")
        return repo, name
    if not ref:
        raise ValueError("invalid branch reference '' (expected BRANCH or repo:BRANCH)")
    return default_repo, ref


# ---------------------------------------------------------------------------
# run helpers: host:name parsing, get/add/remove
# ---------------------------------------------------------------------------

def parse_run_ref(ref: str) -> Tuple[str, str]:
    """Parse a `--add-run`/`--rm-run` argument of the form `host:name`."""
    m = RUN_HOST_NAME_RE.match(ref.strip())
    if not m:
        raise ValueError(f"invalid run reference '{ref}' (expected host:name)")
    return m.group("host"), m.group("name")


def get_runs(entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    return entry.get("runs") or []


def add_run(entry: Dict[str, Any], host: str, name: str) -> None:
    """Append a run, deduping by (host, name). Re-adding an existing run is
    a no-op."""
    runs = entry.setdefault("runs", [])
    for r in runs:
        if r.get("host") == host and r.get("name") == name:
            return
    runs.append({"host": host, "name": name, "state": dict(EMPTY_RUN_STATE)})


def remove_run(entry: Dict[str, Any], host: str, name: str) -> bool:
    """Remove a run by (host, name). Returns True if a run was removed."""
    runs = get_runs(entry)
    for i, r in enumerate(runs):
        if r.get("host") == host and r.get("name") == name:
            del runs[i]
            return True
    return False


# ---------------------------------------------------------------------------
# updates: rolling work-summary log (append/format/render)
# ---------------------------------------------------------------------------

def get_updates(entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    return entry.get("updates") or []


def _normalize_update_text(text: str) -> str:
    """Single dashboard line: strip newlines (replace with spaces), collapse
    surrounding whitespace, then hard-truncate to UPDATE_TEXT_MAX_CHARS with
    a trailing ellipsis if cut. Raises ValueError on empty/whitespace-only
    text."""
    normalized = " ".join(text.split())
    if not normalized:
        raise ValueError("update text must not be empty")
    if len(normalized) > UPDATE_TEXT_MAX_CHARS:
        normalized = normalized[: UPDATE_TEXT_MAX_CHARS - 1].rstrip() + "…"
    return normalized


def append_update(entry: Dict[str, Any], text: str, kind: str, ts: Optional[str] = None) -> None:
    """Append one update entry, deduping against the current most-recent
    entry (an identical `text` back-to-back is dropped rather than
    duplicated), then evicting the oldest entries past UPDATES_MAX_ENTRIES.
    Raises ValueError on empty/whitespace-only text."""
    normalized = _normalize_update_text(text)
    updates = entry.setdefault("updates", [])
    if updates and updates[-1].get("text") == normalized:
        return
    updates.append({"ts": ts or _now_iso(), "text": normalized, "kind": kind})
    if len(updates) > UPDATES_MAX_ENTRIES:
        del updates[: len(updates) - UPDATES_MAX_ENTRIES]


def clear_updates(entry: Dict[str, Any]) -> None:
    entry["updates"] = []


def _pr_transition_texts(before: Optional[Dict[str, Any]], after: Dict[str, Any], number: int) -> List[str]:
    """Detect real state transitions between a PR's previous and freshly
    fetched `prState`, in a fixed order (state change first, then CI). A
    None `before` (no prior prState at all) never synthesizes a transition —
    there is nothing to compare against. A PR's very first successful fetch
    (old state None -> OPEN) DOES count as a transition ("opened"), since
    that is genuinely new information for the card. CI transitions are only
    reported while the PR is currently OPEN — once merged/closed, its
    statusCheckRollup reflects stale/unrelated post-merge job runs, not
    something worth surfacing."""
    if before is None:
        return []
    texts = []
    old_state, new_state = before.get("state"), after.get("state")
    if new_state != old_state:
        if new_state == "MERGED":
            texts.append(f"PR #{number} merged")
        elif new_state == "CLOSED":
            texts.append(f"PR #{number} closed")
        elif new_state == "OPEN" and old_state is None:
            texts.append(f"PR #{number} opened")
        elif new_state == "OPEN":
            texts.append(f"PR #{number} reopened")

    old_ci, new_ci = before.get("ci"), after.get("ci")
    if new_state == "OPEN" and new_ci != old_ci:
        if new_ci == "FAILURE":
            texts.append(f"CI failed on PR #{number}")
        elif old_ci == "FAILURE" and new_ci == "SUCCESS":
            texts.append(f"CI green on PR #{number}")
    return texts


def _run_transition_texts(
    before: Optional[Dict[str, Any]],
    after: Dict[str, Any],
    host: str,
    name: str,
    stall_threshold_seconds: float,
    now: Optional[datetime] = None,
) -> List[str]:
    """Detect real state transitions between a run's previous and freshly
    probed `state`: a status change, or a running run going stalled (log
    hasn't grown in stall_threshold_seconds). A None `before` (no prior
    state at all) never synthesizes a transition."""
    if before is None:
        return []
    texts = []
    old_status, new_status = before.get("status"), after.get("status")
    if new_status != old_status and new_status is not None:
        label = f"run {host}/{name}"
        if new_status == "done":
            exit_code = after.get("exitCode")
            texts.append(f"{label} done" + (f" (exit {exit_code})" if exit_code is not None else ""))
        elif new_status == "failed":
            exit_code = after.get("exitCode")
            texts.append(f"{label} failed" + (f" (exit {exit_code})" if exit_code is not None else ""))
        else:
            texts.append(f"{label} {new_status}")

    was_stalled = run_is_stalled({"state": before}, stall_threshold_seconds, now=now)
    is_stalled = run_is_stalled({"state": after}, stall_threshold_seconds, now=now)
    if is_stalled and not was_stalled:
        mtime = after.get("logMtime")
        now = now or _utcnow()
        age_str = "?"
        if mtime:
            try:
                age_str = _humanize_age_seconds((now - _parse_iso(mtime)).total_seconds())
            except ValueError:
                age_str = "?"
        texts.append(f"run {host}/{name} stalled — no output for {age_str}")
    return texts


def _status_transition_text(old_status: Optional[str], new_status: Optional[str]) -> Optional[str]:
    if new_status is None or new_status == old_status:
        return None
    return f"status: {old_status} -> {new_status}"


# ---------------------------------------------------------------------------
# rendering: icon/color, title, card
# ---------------------------------------------------------------------------

def _any_open_ci_failure(entry: Dict[str, Any]) -> bool:
    return any(
        (p.get("prState") or {}).get("state") == "OPEN" and (p.get("prState") or {}).get("ci") == "FAILURE"
        for p in get_prs(entry)
    )


def _effective_icon(entry: Dict[str, Any]) -> str:
    """CI failure on an open PR overrides the status icon to red, except for
    statuses where the PR is already resolved (merged/closed/done)."""
    status = entry.get("status", "active")
    icon = STATUS_EMOJI.get(status, STATUS_EMOJI["active"])
    if _any_open_ci_failure(entry) and status not in _TERMINAL_STATUSES:
        return STATUS_EMOJI["blocked"]
    return icon


def _effective_color(entry: Dict[str, Any]) -> int:
    status = entry.get("status", "active")
    if _any_open_ci_failure(entry) and status not in _TERMINAL_STATUSES:
        return STATUS_COLOR["blocked"]
    return STATUS_COLOR.get(status, STATUS_COLOR["active"])


def render_title(entry: Dict[str, Any]) -> str:
    """Suggested Discord thread name, e.g. `🔵 mSPRT · PR#4694 · review`, or
    `🔵 mSPRT · PR#4694 +2 · review` when there are extra PRs beyond the
    primary. Discord thread names cap at 100 chars — the title component is
    truncated (with an ellipsis) to fit; the icon/PR/status suffix never is."""
    icon = _effective_icon(entry)
    title = entry.get("title") or ""
    status = entry.get("status", "active")
    prs = get_prs(entry)
    primary = primary_pr(entry)

    suffix_parts = []
    if primary:
        extra = len(prs) - 1
        pr_part = f"PR#{primary['number']}"
        if extra > 0:
            pr_part += f" +{extra}"
        suffix_parts.append(pr_part)
    suffix_parts.append(status)
    suffix = " · ".join(suffix_parts)

    prefix = f"{icon} "
    full = f"{prefix}{title} · {suffix}"
    if len(full) <= THREAD_TITLE_MAX:
        return full

    fixed_len = len(prefix) + len(" · ") + len(suffix)
    budget = max(0, THREAD_TITLE_MAX - fixed_len - 1)  # -1 for the ellipsis char
    truncated_title = title[:budget].rstrip() + "…"
    return f"{prefix}{truncated_title} · {suffix}"


def _pr_line(entry: Dict[str, Any], p: Dict[str, Any]) -> str:
    """One `PR: [#N](url) STATE · CI state` line for the card, with the
    primary PR bolded and a per-PR stale warning. CI is only meaningful while
    a PR is OPEN — a MERGED/CLOSED PR's statusCheckRollup reflects whatever
    unrelated job last ran against that branch post-merge, so the CI segment
    (and its ⚠️) is suppressed once the PR is resolved."""
    number = p["number"]
    repo = _pr_repo(p, entry.get("repo")) or ""
    pr_state = p.get("prState") or {}
    url = pr_state.get("url") or (f"https://github.com/{repo}/pull/{number}" if repo else "")
    state = pr_state.get("state") or "?"
    ci = pr_state.get("ci") or "NONE"
    label = f"[#{number}]({url})" if url else f"#{number}"
    if p.get("primary"):
        label = f"**{label}**"
    line = f"PR: {label} {state}"
    if state == "OPEN":
        line += f" · CI {ci}"
        if ci == "FAILURE":
            line += " ⚠️"
    if pr_state.get("stale"):
        line += f" — ⚠️ stale (last checked: {pr_state.get('checkedAt', '?')})"
    return line


def _branch_line(entry: Dict[str, Any]) -> str:
    """`branch: feat/x (primary), feat/y` for the card, when there are no
    PRs but there ARE branches."""
    branches = get_branches(entry)
    parts = []
    for b in branches:
        label = f"`{b['name']}`"
        if b.get("primary"):
            label += " (primary)"
        parts.append(label)
    return "branch: " + ", ".join(parts)


def run_is_stalled(run: Dict[str, Any], threshold_seconds: float, now: Optional[datetime] = None) -> bool:
    """A run is "stalled" when it claims `status == running` but its log
    hasn't grown in at least `threshold_seconds` — the high-confidence
    "agent claims to be working but is producing nothing" signal."""
    state = run.get("state") or {}
    if state.get("status") != "running":
        return False
    mtime = state.get("logMtime")
    if not mtime:
        return False
    now = now or _utcnow()
    try:
        age = (now - _parse_iso(mtime)).total_seconds()
    except ValueError:
        return False
    return age >= threshold_seconds


def _run_line(run: Dict[str, Any], now: Optional[datetime] = None) -> str:
    """`run: host/name status (log 4m ago)` for the card, with a loud
    `⚠️ no output for Nm` marker when stalled."""
    state = run.get("state") or {}
    status = state.get("status", "missing")
    line = f"run: {run.get('host')}/{run.get('name')} {status}"
    mtime = state.get("logMtime")
    now = now or _utcnow()
    age_str = None
    if mtime:
        try:
            age_str = _humanize_age_seconds((now - _parse_iso(mtime)).total_seconds())
        except ValueError:
            age_str = None
    if age_str is not None:
        line += f" (log {age_str} ago)"
    if run_is_stalled(run, CARD_STALL_THRESHOLD_SECONDS, now=now) and age_str is not None:
        line += f" ⚠️ no output for {age_str}"
    return line


RECENT_PREFIX = "recent: "


def _recent_updates_lines(entry: Dict[str, Any], count: int, now: Optional[datetime] = None) -> List[str]:
    """The `count` most recent `updates`, newest first, each on its own
    line prefixed with a compact relative age — `recent: 4m ago — text` for
    the first line, indented to match for the rest. Empty when there are no
    updates."""
    updates = list(reversed(get_updates(entry)))[:count]
    if not updates:
        return []
    now = now or _utcnow()
    lines = []
    for i, u in enumerate(updates):
        try:
            age = _humanize_age_seconds((now - _parse_iso(u["ts"])).total_seconds())
        except (ValueError, TypeError, KeyError):
            age = "?"
        prefix = RECENT_PREFIX if i == 0 else " " * len(RECENT_PREFIX)
        lines.append(f"{prefix}{age} ago — {u.get('text', '')}")
    return lines


def render_card_markdown(entry: Dict[str, Any]) -> str:
    """The pinned status card message body."""
    icon = _effective_icon(entry)
    prs = get_prs(entry)
    branches = get_branches(entry)
    runs = get_runs(entry)

    lines = [f"{icon} **{entry.get('title', '')}** — {entry.get('status', '')}"]
    if entry.get("description"):
        lines.append(entry["description"])

    meta = []
    if entry.get("repo"):
        meta.append(f"repo: {entry['repo']}")
    if meta:
        lines.append(" | ".join(meta))

    if not prs and branches:
        lines.append(_branch_line(entry))

    for p in prs:
        lines.append(_pr_line(entry, p))

    for r in runs:
        lines.append(_run_line(r))

    if entry.get("tags"):
        lines.append("tags: " + ", ".join(entry["tags"]))
    if entry.get("links"):
        lines.append("links: " + " ".join(entry["links"]))
    if entry.get("notes"):
        lines.append(f"notes: {entry['notes']}")

    base = "\n".join(lines)

    # `recent:` block, newest-first, with a length-cap degradation ladder
    # (3 -> 2 -> 1 entries) so the card never blows past Discord's practical
    # message-length safety margin. Omitted entirely when there are no
    # updates.
    for count in (CARD_RECENT_UPDATES, 2, 1):
        recent_lines = _recent_updates_lines(entry, count)
        if not recent_lines:
            return base
        candidate = base + "\n" + "\n".join(recent_lines)
        if len(candidate) <= CARD_MAX_CHARS or count == 1:
            return candidate
    return base


def render_card_embed(entry: Dict[str, Any]) -> Dict[str, Any]:
    """A Discord embed object, ready to hand to a Discord client."""
    prs = get_prs(entry)
    branches = get_branches(entry)
    primary = primary_pr(entry)
    primary_state = (primary or {}).get("prState") or {}

    fields = []
    if entry.get("repo"):
        fields.append({"name": "Repo", "value": entry["repo"], "inline": True})
    if prs:
        pr_value = f"#{primary['number']}"
        if len(prs) > 1:
            pr_value += f" (+{len(prs) - 1})"
        fields.append({"name": "PR", "value": pr_value, "inline": True})
    if not prs and branches:
        primary_b = primary_branch(entry)
        branch_value = primary_b["name"]
        if len(branches) > 1:
            branch_value += f" (+{len(branches) - 1})"
        fields.append({"name": "Branch", "value": branch_value, "inline": True})
    if primary_state.get("state") == "OPEN" and primary_state.get("ci"):
        fields.append({"name": "CI", "value": primary_state["ci"], "inline": True})
    fields.append({"name": "Status", "value": entry.get("status", ""), "inline": True})
    if entry.get("tags"):
        fields.append({"name": "Tags", "value": ", ".join(entry["tags"]), "inline": False})

    return {
        "title": f"{_effective_icon(entry)} {entry.get('title', '')}",
        "description": entry.get("description", ""),
        "color": _effective_color(entry),
        "fields": fields,
        "url": primary_state.get("url") or None,
        "footer": {"text": f"threadId={entry.get('threadId', '')}"},
    }


TABLE_MAX_WIDTH = 120
TABLE_DESCRIPTION_MAX_WITH_LAST = 30


def _format_table(
    entries: List[Dict[str, Any]],
    silent_map: Optional[Dict[Optional[str], Optional[float]]] = None,
    with_last: bool = False,
) -> str:
    headers = ["", "THREAD", "TITLE", "STATUS", "PR"]
    if silent_map is not None:
        headers.append("SILENT")
    headers += ["CI", "DESCRIPTION"]
    if with_last:
        headers.append("LAST")

    rows = []
    for e in entries:
        prs = get_prs(e)
        primary = primary_pr(e)
        if primary:
            pr_col = f"#{primary['number']}"
            if len(prs) > 1:
                pr_col += f"+{len(prs) - 1}"
        else:
            pr_col = "-"
        ci = ((primary or {}).get("prState") or {}).get("ci") or "-"
        if not primary or (primary.get("prState") or {}).get("state") != "OPEN":
            ci = "-"
        row = [
            _effective_icon(e),
            str(e.get("threadId", "")),
            str(e.get("title", "")),
            str(e.get("status", "")),
            pr_col,
        ]
        if silent_map is not None:
            sfs = silent_map.get(e.get("threadId"))
            row.append(_humanize_age_seconds(sfs) if sfs is not None else "?")
        description = str(e.get("description", ""))
        if with_last and len(description) > TABLE_DESCRIPTION_MAX_WITH_LAST:
            description = description[: TABLE_DESCRIPTION_MAX_WITH_LAST - 1].rstrip() + "…"
        row.append(ci)
        row.append(description)
        if with_last:
            updates = get_updates(e)
            row.append(updates[-1]["text"] if updates else "-")
        rows.append(row)

    # Truncate the last column as needed to keep the whole line under
    # TABLE_MAX_WIDTH — every other column is short/fixed-shape, so the
    # last (free-text) column is the only one that can grow unbounded.
    fixed_widths = [
        max([len(headers[i])] + [len(r[i]) for r in rows]) for i in range(len(headers) - 1)
    ]
    sep_width = 2 * (len(headers) - 1)
    budget = max(10, TABLE_MAX_WIDTH - sum(fixed_widths) - sep_width)
    for r in rows:
        if len(r[-1]) > budget:
            r[-1] = r[-1][: max(0, budget - 1)].rstrip() + "…"

    widths = fixed_widths + [max([len(headers[-1])] + [len(r[-1]) for r in rows])]

    def fmt_row(cols: List[str]) -> str:
        # Don't pad the last column — descriptions are free text.
        return "  ".join(
            c.ljust(w) if i < len(cols) - 1 else c for i, (c, w) in enumerate(zip(cols, widths))
        )

    lines = [fmt_row(headers)]
    lines.extend(fmt_row(r) for r in rows)
    return "\n".join(lines)


def _format_entry_human(entry: Dict[str, Any]) -> str:
    prs = get_prs(entry)
    branches = get_branches(entry)
    runs = get_runs(entry)
    lines = [
        f"threadId: {entry.get('threadId')}",
        f"title: {entry.get('title')}",
        f"status: {_effective_icon(entry)} {entry.get('status')}",
        f"repo: {entry.get('repo') or '-'}",
        f"description: {entry.get('description') or '-'}",
        f"tags: {', '.join(entry.get('tags') or []) or '-'}",
        f"links: {', '.join(entry.get('links') or []) or '-'}",
        f"notes: {entry.get('notes') or '-'}",
        f"lastActivityAt: {entry.get('lastActivityAt') or '-'} ({entry.get('lastActivitySource') or '-'})",
        f"discord: cardMessageId={entry.get('cardMessageId') or '-'} "
        f"parentChannelId={entry.get('parentChannelId') or '-'} "
        f"name={entry.get('discordName') or '-'}",
        f"createdAt: {entry.get('createdAt')}",
        f"updatedAt: {entry.get('updatedAt')}",
    ]
    for p in prs:
        pr_state = p.get("prState") or {}
        marker = "*" if p.get("primary") else " "
        repo = _pr_repo(p, entry.get("repo")) or "-"
        lines.append(
            f"pr[{marker}]: {repo}#{p.get('number')} state={pr_state.get('state')} "
            f"ci={pr_state.get('ci')} stale={pr_state.get('stale')} checkedAt={pr_state.get('checkedAt')}"
        )
    for b in branches:
        marker = "*" if b.get("primary") else " "
        repo = _branch_repo(b, entry.get("repo")) or "-"
        lines.append(f"branch[{marker}]: {repo}:{b.get('name')}")
    for r in runs:
        state = r.get("state") or {}
        lines.append(
            f"run: {r.get('host')}/{r.get('name')} status={state.get('status')} "
            f"exitCode={state.get('exitCode')} logBytes={state.get('logBytes')} "
            f"logMtime={state.get('logMtime')} stale={state.get('stale')}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# GitHub integration (subprocess-injected so tests never touch the network)
# ---------------------------------------------------------------------------

def _run_command(cmd: List[str], timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _derive_ci(rollup: Optional[List[Dict[str, Any]]]) -> str:
    """FAILURE if any check failed/errored, else PENDING if any is still
    running/queued, else SUCCESS if there are checks at all, else NONE."""
    if not rollup:
        return "NONE"
    states = {
        str(item.get("conclusion") or item.get("state") or item.get("status") or "").upper()
        for item in rollup
    }
    if states & CI_FAILURE_STATES:
        return "FAILURE"
    if states & CI_PENDING_STATES:
        return "PENDING"
    return "SUCCESS"


def _gh_account_for(repo: str, accounts: Dict[str, str]) -> Optional[str]:
    owner = repo.split("/", 1)[0]
    return accounts.get(owner)


def _fetch_gh_token(account: str, timeout: int, runner) -> Optional[str]:
    """Look up a token for `account` via `gh auth token`, which — unlike `gh
    auth switch` — only reads local credential storage and never mutates any
    shared active-account state, so it's safe to call from multiple threads
    concurrently without one call clobbering another's account selection."""
    try:
        proc = runner(["gh", "auth", "token", "--user", account], timeout)
    except (subprocess.TimeoutExpired, OSError):
        return None
    if proc.returncode != 0:
        return None
    token = (proc.stdout or "").strip()
    return token or None


def fetch_pr_state(
    repo: str,
    pr: int,
    accounts: Dict[str, str],
    timeout: int,
    runner,
    token_cache: Optional[Dict[str, str]] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Fetch fresh PR state via `gh`. If the repo's owner maps to a known
    account, its token is resolved via `gh auth token` (cached in
    `token_cache` so concurrent PRs for the same account don't repeat the
    lookup) and passed as `GH_TOKEN` on the `gh pr view` invocation itself —
    never via the global, stateful `gh auth switch`, which raced across
    concurrent refreshes for different owners' PRs (each worker thread would
    clobber the others' active account mid-poll). Returns (prState, None) on
    success or (None, error) on any failure — never raises, so callers can
    always fall back to the previously cached state."""
    if token_cache is None:
        token_cache = {}

    cmd = [
        "gh", "pr", "view", str(pr), "--repo", repo,
        "--json", "state,title,url,mergedAt,statusCheckRollup",
    ]

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
        return None, f"gh timed out after {timeout}s"
    except OSError as exc:
        return None, f"gh failed to run: {exc}"

    if proc.returncode != 0:
        err = (proc.stderr or "").strip() or f"gh exited {proc.returncode}"
        return None, err

    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        return None, f"gh returned invalid JSON: {exc}"

    pr_state = {
        "state": payload.get("state", "OPEN"),
        "ci": _derive_ci(payload.get("statusCheckRollup")),
        "title": payload.get("title", ""),
        "url": payload.get("url", ""),
        "mergedAt": payload.get("mergedAt"),
        "checkedAt": _now_iso(),
        "stale": False,
        "error": None,
    }
    return pr_state, None


# ---------------------------------------------------------------------------
# agent-run probing over SSH (subprocess-injected so tests never touch the
# network or a real ssh binary)
# ---------------------------------------------------------------------------

PROBE_LINE_RE = re.compile(r"^(STATUS|EXIT|BYTES|MTIME):(.*)$")
PROBE_MISSING_MARKER = "__TSTATE_MISSING__"


def _build_probe_command(name: str) -> str:
    """The remote shell one-liner for a single agent-run `name`, per the
    split-storage convention (ephemeral /tmp, persistent /var/tmp). Prefixing
    each value with a tag makes parsing order-independent, and trying both
    `stat` dialects (macOS `-f %m`, Linux `-c %Y`) in one `||` chain means we
    never need to know which OS the host runs — the wrong one just fails
    silently and falls through."""
    status_path = f"/tmp/agent-runs/{name}/status"
    exit_path = f"/tmp/agent-runs/{name}/exit_code"
    log_path = f"/var/tmp/agent-runs/{name}/log"
    return (
        f'echo "STATUS:$(cat {status_path} 2>/dev/null || echo {PROBE_MISSING_MARKER})"; '
        f'echo "EXIT:$(cat {exit_path} 2>/dev/null)"; '
        f'echo "BYTES:$(wc -c < {log_path} 2>/dev/null)"; '
        f'echo "MTIME:$(stat -f %m {log_path} 2>/dev/null || stat -c %Y {log_path} 2>/dev/null)"'
    )


def _parse_probe_output(stdout: str) -> Dict[str, Optional[str]]:
    fields: Dict[str, Optional[str]] = {"STATUS": None, "EXIT": None, "BYTES": None, "MTIME": None}
    for line in stdout.splitlines():
        m = PROBE_LINE_RE.match(line.strip())
        if m:
            fields[m.group(1)] = m.group(2).strip() or None
    return fields


def probe_run(host: str, name: str, timeout: int, runner) -> Dict[str, Any]:
    """Poll one agent-run task over SSH and return a fresh `state` dict.
    Never raises — any failure (SSH error, timeout, unparseable output)
    returns `stale: True` + `error` so the caller can fall back to the
    previously cached state instead of losing it."""
    cmd = [
        "ssh", "-o", "BatchMode=yes", "-o", f"ConnectTimeout={timeout}",
        host, _build_probe_command(name),
    ]
    try:
        proc = runner(cmd, timeout + 5)
    except subprocess.TimeoutExpired:
        return {"stale": True, "error": f"ssh timed out after {timeout}s"}
    except OSError as exc:
        return {"stale": True, "error": f"ssh failed to run: {exc}"}

    if proc.returncode != 0:
        err = (proc.stderr or "").strip() or f"ssh exited {proc.returncode}"
        return {"stale": True, "error": err}

    fields = _parse_probe_output(proc.stdout or "")
    checked_at = _now_iso()

    if fields["STATUS"] is None or fields["STATUS"] == PROBE_MISSING_MARKER:
        return {
            "status": "missing", "exitCode": None, "logBytes": None, "logMtime": None,
            "checkedAt": checked_at, "stale": False, "error": None,
        }

    exit_code = None
    if fields["EXIT"] is not None:
        try:
            exit_code = int(fields["EXIT"])
        except ValueError:
            exit_code = None

    log_bytes = None
    if fields["BYTES"] is not None:
        try:
            log_bytes = int(fields["BYTES"])
        except ValueError:
            log_bytes = None

    log_mtime = None
    if fields["MTIME"] is not None:
        try:
            epoch = int(fields["MTIME"])
            log_mtime = datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        except (ValueError, OverflowError, OSError):
            log_mtime = None

    return {
        "status": fields["STATUS"], "exitCode": exit_code, "logBytes": log_bytes, "logMtime": log_mtime,
        "checkedAt": checked_at, "stale": False, "error": None,
    }


# ---------------------------------------------------------------------------
# PR-reference scanning
# ---------------------------------------------------------------------------

def scan_text(text: str, known_repo: Optional[str]) -> List[Dict[str, Any]]:
    """Find PR references: explicit GitHub URLs (any repo) and bare `#N`
    (only matched when `known_repo` is set, since bare refs are ambiguous
    without a repo to anchor them)."""
    matches: List[Dict[str, Any]] = []
    for m in PR_URL_RE.finditer(text):
        owner, name, num = m.group(1), m.group(2), int(m.group(3))
        matches.append({"repo": f"{owner}/{name}", "pr": num, "kind": "url", "pos": m.start()})
    if known_repo:
        for m in BARE_PR_RE.finditer(text):
            matches.append({"repo": known_repo, "pr": int(m.group(1)), "kind": "bare", "pos": m.start()})
    matches.sort(key=lambda x: x["pos"])
    return matches


def summarize_matches(matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Dedupe by (repo, pr), counting occurrences, ordered so the
    most-recently-seen (highest source position) match is last."""
    groups: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for m in matches:
        key = (m["repo"], m["pr"])
        g = groups.setdefault(key, {"repo": m["repo"], "pr": m["pr"], "count": 0, "kinds": set(), "_last_pos": -1})
        g["count"] += 1
        g["kinds"].add(m["kind"])
        g["_last_pos"] = max(g["_last_pos"], m["pos"])
    ordered = sorted(groups.values(), key=lambda g: g["_last_pos"])
    for g in ordered:
        g["kinds"] = sorted(g["kinds"])
        del g["_last_pos"]
    return ordered


def best_match(matches: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Highest-confidence match: prefer an explicit URL over a bare `#N`;
    among ties, the last occurrence wins."""
    if not matches:
        return None
    url_matches = [m for m in matches if m["kind"] == "url"]
    pool = url_matches if url_matches else matches
    return max(pool, key=lambda m: m["pos"])


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------

def _entry_with_computed_pr(entry: Dict[str, Any]) -> Dict[str, Any]:
    """A shallow copy of `entry` for JSON output, with read-only `pr` and
    `branch` fields (the primary PR's number / primary branch's name, or
    null) so old consumers that only knew the v1/v2 scalar fields don't
    hard-break."""
    out = dict(entry)
    primary = primary_pr(entry)
    out["pr"] = primary.get("number") if primary else None
    primary_b = primary_branch(entry)
    out["branch"] = primary_b.get("name") if primary_b else None
    return out


def _apply_pr_mutations(entry: Dict[str, Any], args: argparse.Namespace) -> None:
    """Apply --pr/--add-pr/--rm-pr/--primary-pr to `entry`, in that order.
    Raises _Abort(2, ...) on a bad PR reference or an unmatched --primary-pr."""
    try:
        if getattr(args, "pr", None) is not None:
            set_prs_single(entry, entry.get("repo"), args.pr)
        for ref in getattr(args, "add_pr", None) or []:
            repo, num = parse_pr_ref(ref, entry.get("repo"))
            add_pr(entry, repo, num)
        for ref in getattr(args, "rm_pr", None) or []:
            repo, num = parse_pr_ref(ref, entry.get("repo"))
            remove_pr(entry, repo, num)
        primary_ref = getattr(args, "primary_pr", None)
        if primary_ref:
            repo, num = parse_pr_ref(primary_ref, entry.get("repo"))
            if not set_primary_pr(entry, repo, num):
                raise _Abort(2, f"thread-state: --primary-pr {primary_ref} does not match any PR on this thread")
    except ValueError as exc:
        raise _Abort(2, f"thread-state: {exc}") from exc


def _apply_branch_mutations(entry: Dict[str, Any], args: argparse.Namespace) -> None:
    """Apply --branch/--add-branch/--rm-branch/--primary-branch to `entry`,
    in that order. Raises _Abort(2, ...) on a bad reference or an unmatched
    --primary-branch."""
    try:
        if getattr(args, "branch", None) is not None:
            set_branches_single(entry, entry.get("repo"), args.branch)
        for ref in getattr(args, "add_branch", None) or []:
            repo, name = parse_branch_ref(ref, entry.get("repo"))
            add_branch(entry, repo, name)
        for ref in getattr(args, "rm_branch", None) or []:
            repo, name = parse_branch_ref(ref, entry.get("repo"))
            remove_branch(entry, repo, name)
        primary_ref = getattr(args, "primary_branch", None)
        if primary_ref:
            repo, name = parse_branch_ref(primary_ref, entry.get("repo"))
            if not set_primary_branch(entry, repo, name):
                raise _Abort(2, f"thread-state: --primary-branch {primary_ref} does not match any branch on this thread")
    except ValueError as exc:
        raise _Abort(2, f"thread-state: {exc}") from exc


def _apply_run_mutations(entry: Dict[str, Any], args: argparse.Namespace) -> None:
    """Apply --add-run/--rm-run to `entry`. Rejects an unknown host unless
    --force is set. Raises _Abort(2, ...) on a bad reference or unknown
    host."""
    force = bool(getattr(args, "force", False))
    try:
        for ref in getattr(args, "add_run", None) or []:
            host, name = parse_run_ref(ref)
            if host not in KNOWN_HOSTS and not force:
                raise _Abort(
                    2,
                    f"thread-state: unknown host '{host}' (known hosts: {', '.join(KNOWN_HOSTS)}; use --force to add anyway)",
                )
            add_run(entry, host, name)
        for ref in getattr(args, "rm_run", None) or []:
            host, name = parse_run_ref(ref)
            remove_run(entry, host, name)
    except ValueError as exc:
        raise _Abort(2, f"thread-state: {exc}") from exc


def _apply_last_activity_mutations(entry: Dict[str, Any], args: argparse.Namespace) -> None:
    """Apply --last-activity/--last-message-id to `entry`. The two are
    mutually exclusive at the CLI level (argparse doesn't enforce it here,
    so a caller passing both gets --last-message-id's decoded value win,
    applied second)."""
    last_activity = getattr(args, "last_activity", None)
    if last_activity is not None:
        entry["lastActivityAt"] = last_activity
        entry["lastActivitySource"] = "manual"
    last_message_id = getattr(args, "last_message_id", None)
    if last_message_id is not None:
        try:
            entry["lastActivityAt"] = snowflake_to_iso(last_message_id)
        except (ValueError, OverflowError) as exc:
            raise _Abort(2, f"thread-state: invalid --last-message-id '{last_message_id}': {exc}") from exc
        entry["lastActivitySource"] = "discord"


def _apply_update_mutations(entry: Dict[str, Any], args: argparse.Namespace) -> None:
    """Apply --clear-updates (first, so a combined --clear-updates
    --progress in one invocation ends with the new entries kept) then
    --progress to `entry`. Raises _Abort(2, ...) on empty/whitespace text."""
    if getattr(args, "clear_updates", False):
        clear_updates(entry)
    for text in getattr(args, "progress", None) or []:
        try:
            append_update(entry, text, "manual")
        except ValueError as exc:
            raise _Abort(2, f"thread-state: {exc}") from exc


def _apply_discord_meta_mutations(entry: Dict[str, Any], args: argparse.Namespace) -> None:
    """Apply --card-message-id/--parent-channel-id/--discord-name (and, on
    `set`, --clear-card-message-id) to `entry`. --clear-card-message-id is
    applied first, so a combined --clear-card-message-id --card-message-id
    in one invocation ends with the new id kept. Raises _Abort(2, ...) on an
    invalid snowflake."""
    if getattr(args, "clear_card_message_id", False):
        entry["cardMessageId"] = None
    try:
        card_message_id = getattr(args, "card_message_id", None)
        if card_message_id is not None:
            entry["cardMessageId"] = validate_snowflake(card_message_id)
        parent_channel_id = getattr(args, "parent_channel_id", None)
        if parent_channel_id is not None:
            entry["parentChannelId"] = validate_snowflake(parent_channel_id)
    except ValueError as exc:
        raise _Abort(2, f"thread-state: {exc}") from exc
    discord_name = getattr(args, "discord_name", None)
    if discord_name is not None:
        entry["discordName"] = discord_name


def _effective_silent_statuses(include_status_arg: Optional[str]) -> List[str]:
    """The statuses considered "expected to be progressing" for the silent
    detector: --include-status if given, else DEFAULT_SILENT_STATUSES — minus
    anything in NEVER_SILENT_STATUSES, which can never be overridden back
    in."""
    statuses = _split_tags(include_status_arg) if include_status_arg else list(DEFAULT_SILENT_STATUSES)
    return [s for s in statuses if s not in NEVER_SILENT_STATUSES]


def _silent_info(
    entry: Dict[str, Any],
    threshold_seconds: float,
    include_statuses: List[str],
    now: Optional[datetime] = None,
) -> Tuple[bool, Optional[float], bool]:
    """Returns (silent, silentForSeconds, stalled) for one entry against a
    DURATION threshold. `silentForSeconds` is None when `lastActivityAt` is
    null — silent-but-distinct: an unknown age, not a zero age."""
    now = now or _utcnow()
    if entry.get("status") not in include_statuses:
        return False, None, False

    last = entry.get("lastActivityAt")
    if last is None:
        silent_for: Optional[float] = None
        silent = True
    else:
        try:
            silent_for = (now - _parse_iso(last)).total_seconds()
        except ValueError:
            silent_for = None
        silent = silent_for is None or silent_for >= threshold_seconds

    stalled = any(run_is_stalled(r, threshold_seconds, now=now) for r in get_runs(entry))
    return silent, silent_for, stalled


def cmd_add(args: argparse.Namespace) -> int:
    path = _resolve_registry_path(args)
    try:
        status = _validate_status(args.status or "active")
    except ValueError as exc:
        _err(f"thread-state: {exc}")
        return 2

    entry: Dict[str, Any] = {}
    try:
        with _locked_registry(path) as data:
            threads = data["threads"]
            if args.thread_id in threads and not args.force:
                raise _Abort(1, f"thread-state: thread '{args.thread_id}' already exists (use --force to overwrite)")
            now = _now_iso()
            entry = {
                "threadId": args.thread_id,
                "title": args.title,
                "description": args.desc or "",
                "repo": args.repo,
                "status": status,
                "tags": _split_tags(args.tags),
                "links": list(args.link or []),
                "notes": "",
                "prs": [],
                "branches": [],
                "runs": [],
                "lastActivityAt": None,
                "lastActivitySource": None,
                "updates": [],
                "cardMessageId": None,
                "parentChannelId": None,
                "discordName": None,
                "createdAt": now,
                "updatedAt": now,
            }
            _apply_pr_mutations(entry, args)
            _apply_branch_mutations(entry, args)
            _apply_run_mutations(entry, args)
            _apply_last_activity_mutations(entry, args)
            _apply_discord_meta_mutations(entry, args)
            threads[args.thread_id] = entry
    except _Abort as exc:
        _err(exc.message)
        return exc.code
    except RegistryError as exc:
        _err(str(exc))
        return 3

    if args.json:
        print(json.dumps(_entry_with_computed_pr(entry), indent=2, sort_keys=True))
    else:
        print(f"thread-state: added '{args.thread_id}' ({entry['title']})")
    return 0


def cmd_set(args: argparse.Namespace) -> int:
    path = _resolve_registry_path(args)
    result: Dict[str, Any] = {}
    try:
        with _locked_registry(path) as data:
            entry = data["threads"].get(args.thread_id)
            if entry is None:
                raise _Abort(1, f"thread-state: no such thread '{args.thread_id}'")
            if args.title is not None:
                entry["title"] = args.title
            if args.desc is not None:
                entry["description"] = args.desc
            if args.repo is not None:
                entry["repo"] = args.repo
            if args.status is not None:
                try:
                    entry["status"] = _validate_status(args.status)
                except ValueError as exc:
                    raise _Abort(2, f"thread-state: {exc}")
            if args.tags is not None:
                entry["tags"] = _split_tags(args.tags)
            if args.link:
                entry["links"] = list(args.link)
            if args.note is not None:
                entry["notes"] = args.note
            _apply_pr_mutations(entry, args)
            _apply_branch_mutations(entry, args)
            _apply_run_mutations(entry, args)
            _apply_last_activity_mutations(entry, args)
            _apply_update_mutations(entry, args)
            _apply_discord_meta_mutations(entry, args)
            entry["updatedAt"] = _now_iso()
            result = _entry_with_computed_pr(entry)
    except _Abort as exc:
        _err(exc.message)
        return exc.code
    except RegistryError as exc:
        _err(str(exc))
        return 3

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"thread-state: updated '{args.thread_id}'")
    return 0


def cmd_rm(args: argparse.Namespace) -> int:
    path = _resolve_registry_path(args)
    try:
        with _locked_registry(path) as data:
            if args.thread_id not in data["threads"]:
                raise _Abort(1, f"thread-state: no such thread '{args.thread_id}'")
            del data["threads"][args.thread_id]
    except _Abort as exc:
        _err(exc.message)
        return exc.code
    except RegistryError as exc:
        _err(str(exc))
        return 3

    if args.json:
        print(json.dumps({"removed": args.thread_id}))
    else:
        print(f"thread-state: removed '{args.thread_id}'")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    path = _resolve_registry_path(args)
    try:
        with _read_only_registry(path) as data:
            entries = list(data["threads"].values())
    except RegistryError as exc:
        _err(str(exc))
        return 3

    if args.status:
        entries = [e for e in entries if e.get("status") == args.status]
    if args.repo:
        entries = [e for e in entries if e.get("repo") == args.repo]
    if args.tag:
        entries = [e for e in entries if args.tag in (e.get("tags") or [])]

    silent_for = getattr(args, "silent_for", None)
    stalled_only = bool(getattr(args, "stalled", False))
    silent_seconds: Optional[int] = None
    if silent_for is not None:
        try:
            silent_seconds = parse_duration(silent_for)
        except ValueError as exc:
            _err(f"thread-state: {exc}")
            return 2

    entries.sort(key=lambda e: (e.get("title") or "", e.get("threadId") or ""))

    if silent_seconds is not None:
        include_statuses = _effective_silent_statuses(getattr(args, "include_status", None))
        now = _utcnow()
        annotated = []
        for e in entries:
            silent, silent_for_seconds, stalled = _silent_info(e, silent_seconds, include_statuses, now=now)
            if not silent:
                continue
            if stalled_only and not stalled:
                continue
            annotated.append((e, silent_for_seconds, stalled))
    else:
        annotated = [(e, None, False) for e in entries]

    if args.json:
        out = []
        for e, silent_for_seconds, stalled in annotated:
            item = _entry_with_computed_pr(e)
            if silent_seconds is not None:
                item["silentForSeconds"] = silent_for_seconds
                item["stalled"] = stalled
            out.append(item)
        print(json.dumps(out, indent=2, sort_keys=True))
    elif not annotated:
        print("thread-state: no threads")
    else:
        print(_format_table(
            [e for e, _, _ in annotated],
            silent_map={e.get("threadId"): sfs for e, sfs, _stalled in annotated} if silent_seconds is not None else None,
            with_last=bool(getattr(args, "with_last", False)),
        ))
    return 0


def cmd_silent(args: argparse.Namespace) -> int:
    """Thin alias for `list --silent-for` — a stable cron entrypoint."""
    list_args = argparse.Namespace(**vars(args))
    list_args.silent_for = args.for_duration
    return cmd_list(list_args)


def cmd_show(args: argparse.Namespace) -> int:
    path = _resolve_registry_path(args)
    try:
        with _read_only_registry(path) as data:
            entry = data["threads"].get(args.thread_id)
    except RegistryError as exc:
        _err(str(exc))
        return 3

    if entry is None:
        _err(f"thread-state: no such thread '{args.thread_id}'")
        return 1

    if args.json:
        print(json.dumps(_entry_with_computed_pr(entry), indent=2, sort_keys=True))
    else:
        print(_format_entry_human(entry))
    return 0


def cmd_refresh(args: argparse.Namespace) -> int:
    path = _resolve_registry_path(args)
    results: List[Dict[str, Any]] = []
    try:
        with _locked_registry(path) as data:
            accounts = {**DEFAULT_ACCOUNTS, **data.get("accounts", {})}
            threads = data["threads"]

            if args.all:
                ids = list(threads.keys())
            else:
                ids = args.thread_ids
                missing = [tid for tid in ids if tid not in threads]
                if missing:
                    raise _Abort(2, f"thread-state: unknown thread id(s): {', '.join(missing)}")

            # Flatten every PR of every targeted thread into one global job
            # list so the 6-worker cap applies across ALL PRs, not per thread.
            jobs: List[Tuple[str, int]] = []
            for tid in ids:
                entry = threads[tid]
                for idx, p in enumerate(get_prs(entry)):
                    repo = _pr_repo(p, entry.get("repo"))
                    if repo and p.get("number"):
                        jobs.append((tid, idx))
            targets_tids = {tid for tid, _ in jobs}
            skipped = [tid for tid in ids if tid not in targets_tids]

            # Shared across worker threads so concurrent PRs for the same
            # account resolve its token once; a lookup race just means an
            # occasional duplicate `gh auth token` call, never a correctness
            # issue (see fetch_pr_state).
            token_cache: Dict[str, str] = {}

            def _work(tid: str, idx: int):
                entry = threads[tid]
                p = get_prs(entry)[idx]
                repo = _pr_repo(p, entry.get("repo"))
                pr_state, error = fetch_pr_state(
                    repo, p["number"], accounts, args.timeout, _run_command, token_cache
                )
                return tid, idx, pr_state, error

            per_thread_total: Dict[str, int] = {}
            for tid, _idx in jobs:
                per_thread_total[tid] = per_thread_total.get(tid, 0) + 1
            per_thread_done: Dict[str, int] = {tid: 0 for tid in targets_tids}
            per_thread_errors: Dict[str, List[str]] = {}

            worker_count = min(REFRESH_MAX_WORKERS, len(jobs)) or 1
            with ThreadPoolExecutor(max_workers=worker_count) as ex:
                futures = {ex.submit(_work, tid, idx): (tid, idx) for tid, idx in jobs}
                for fut in as_completed(futures):
                    tid, idx, pr_state, error = fut.result()
                    entry = threads[tid]
                    p = get_prs(entry)[idx]
                    before_state = dict(p.get("prState") or {})
                    if error is not None:
                        prev = dict(p.get("prState") or {})
                        prev["stale"] = True
                        prev["error"] = error
                        prev["checkedAt"] = _now_iso()
                        p["prState"] = prev
                        per_thread_errors.setdefault(tid, []).append(error)
                    else:
                        p["prState"] = pr_state
                        for text in _pr_transition_texts(before_state, pr_state, p["number"]):
                            append_update(entry, text, "transition")

                    per_thread_done[tid] += 1
                    if per_thread_done[tid] == per_thread_total[tid]:
                        old_status = entry.get("status")
                        new_status = derive_thread_status(entry)
                        if new_status is not None:
                            entry["status"] = new_status
                        status_text = _status_transition_text(old_status, new_status)
                        if status_text:
                            append_update(entry, status_text, "transition")
                        entry["updatedAt"] = _now_iso()
                        errs = per_thread_errors.get(tid)
                        if errs:
                            results.append({"threadId": tid, "ok": False, "error": "; ".join(errs)})
                        else:
                            results.append({"threadId": tid, "ok": True, "prs": get_prs(entry)})

            for tid in skipped:
                results.append({"threadId": tid, "ok": False, "error": "no repo/pr set, skipped"})
    except _Abort as exc:
        _err(exc.message)
        return exc.code
    except RegistryError as exc:
        _err(str(exc))
        return 3

    if args.json:
        print(json.dumps(results, indent=2, sort_keys=True))
    else:
        for r in sorted(results, key=lambda r: r["threadId"]):
            if r["ok"]:
                print(f"{r['threadId']}: refreshed ({len(r['prs'])} pr(s))")
            else:
                print(f"{r['threadId']}: ERROR {r['error']}")

    if args.all:
        return 0
    return 1 if any(not r["ok"] for r in results) else 0


def cmd_probe(args: argparse.Namespace) -> int:
    path = _resolve_registry_path(args)
    results: List[Dict[str, Any]] = []
    try:
        with _locked_registry(path) as data:
            threads = data["threads"]

            if args.all:
                ids = list(threads.keys())
            else:
                ids = args.thread_ids
                missing = [tid for tid in ids if tid not in threads]
                if missing:
                    raise _Abort(2, f"thread-state: unknown thread id(s): {', '.join(missing)}")

            # Flatten every run of every targeted thread into one global job
            # list so the 6-worker cap applies across ALL runs, not per thread.
            jobs: List[Tuple[str, int]] = []
            for tid in ids:
                entry = threads[tid]
                for idx, _r in enumerate(get_runs(entry)):
                    jobs.append((tid, idx))
            targets_tids = {tid for tid, _ in jobs}
            skipped = [tid for tid in ids if tid not in targets_tids]

            def _work(tid: str, idx: int):
                entry = threads[tid]
                r = get_runs(entry)[idx]
                state = probe_run(r["host"], r["name"], args.timeout, _run_command)
                return tid, idx, state

            per_thread_total: Dict[str, int] = {}
            for tid, _idx in jobs:
                per_thread_total[tid] = per_thread_total.get(tid, 0) + 1
            per_thread_done: Dict[str, int] = {tid: 0 for tid in targets_tids}
            per_thread_errors: Dict[str, List[str]] = {}

            worker_count = min(REFRESH_MAX_WORKERS, len(jobs)) or 1
            with ThreadPoolExecutor(max_workers=worker_count) as ex:
                futures = {ex.submit(_work, tid, idx): (tid, idx) for tid, idx in jobs}
                for fut in as_completed(futures):
                    tid, idx, state = fut.result()
                    entry = threads[tid]
                    r = get_runs(entry)[idx]
                    before_state = dict(r.get("state") or {})
                    if state.get("error") is not None:
                        prev = dict(r.get("state") or EMPTY_RUN_STATE)
                        prev["stale"] = True
                        prev["error"] = state["error"]
                        prev["checkedAt"] = _now_iso()
                        r["state"] = prev
                        per_thread_errors.setdefault(tid, []).append(state["error"])
                    else:
                        r["state"] = state
                        for text in _run_transition_texts(
                            before_state, state, r["host"], r["name"], CARD_STALL_THRESHOLD_SECONDS
                        ):
                            append_update(entry, text, "transition")

                    per_thread_done[tid] += 1
                    if per_thread_done[tid] == per_thread_total[tid]:
                        entry["updatedAt"] = _now_iso()
                        errs = per_thread_errors.get(tid)
                        if errs:
                            results.append({"threadId": tid, "ok": False, "error": "; ".join(errs)})
                        else:
                            results.append({"threadId": tid, "ok": True, "runs": get_runs(entry)})

            for tid in skipped:
                results.append({"threadId": tid, "ok": False, "error": "no runs tracked, skipped"})
    except _Abort as exc:
        _err(exc.message)
        return exc.code
    except RegistryError as exc:
        _err(str(exc))
        return 3

    if args.json:
        print(json.dumps(results, indent=2, sort_keys=True))
    else:
        for r in sorted(results, key=lambda r: r["threadId"]):
            if r["ok"]:
                print(f"{r['threadId']}: probed ({len(r['runs'])} run(s))")
            else:
                print(f"{r['threadId']}: ERROR {r['error']}")

    if args.all:
        return 0
    return 1 if any(not r["ok"] for r in results) else 0


def cmd_card(args: argparse.Namespace) -> int:
    path = _resolve_registry_path(args)
    try:
        with _read_only_registry(path) as data:
            entry = data["threads"].get(args.thread_id)
    except RegistryError as exc:
        _err(str(exc))
        return 3

    if entry is None:
        _err(f"thread-state: no such thread '{args.thread_id}'")
        return 1

    fmt = "embed-json" if getattr(args, "json", False) else args.format
    if fmt == "embed-json":
        print(json.dumps(render_card_embed(entry), indent=2, sort_keys=True))
    else:
        print(render_card_markdown(entry))
    return 0


def cmd_title(args: argparse.Namespace) -> int:
    path = _resolve_registry_path(args)
    try:
        with _read_only_registry(path) as data:
            entry = data["threads"].get(args.thread_id)
    except RegistryError as exc:
        _err(str(exc))
        return 3

    if entry is None:
        _err(f"thread-state: no such thread '{args.thread_id}'")
        return 1

    title = render_title(entry)
    if getattr(args, "json", False):
        print(json.dumps({"title": title}))
    else:
        print(title)
    return 0


def cmd_log(args: argparse.Namespace) -> int:
    path = _resolve_registry_path(args)
    try:
        with _read_only_registry(path) as data:
            entry = data["threads"].get(args.thread_id)
    except RegistryError as exc:
        _err(str(exc))
        return 3

    if entry is None:
        _err(f"thread-state: no such thread '{args.thread_id}'")
        return 1

    updates = list(reversed(get_updates(entry)))
    if args.limit is not None:
        updates = updates[: args.limit]

    if args.json:
        print(json.dumps(updates, indent=2, sort_keys=True))
    elif not updates:
        print("thread-state: no updates")
    else:
        for u in updates:
            print(f"[{u.get('ts')}] ({u.get('kind')}) {u.get('text')}")
    return 0


def cmd_scan(args: argparse.Namespace) -> int:
    path = _resolve_registry_path(args)

    if args.file == "-":
        text = sys.stdin.read()
    else:
        try:
            text = Path(args.file).read_text()
        except OSError as exc:
            _err(f"thread-state: cannot read {args.file}: {exc}")
            return 1

    summary: List[Dict[str, Any]] = []
    applied: Optional[Dict[str, Any]] = None
    try:
        registry_ctx = _locked_registry(path) if args.apply else _read_only_registry(path)
        with registry_ctx as data:
            entry = data["threads"].get(args.thread_id)
            if entry is None:
                raise _Abort(1, f"thread-state: no such thread '{args.thread_id}'")
            matches = scan_text(text, entry.get("repo"))
            summary = summarize_matches(matches)
            if args.apply:
                added = []
                for g in summary:
                    before = len(get_prs(entry))
                    add_pr(entry, g["repo"], g["pr"])
                    if len(get_prs(entry)) > before:
                        added.append({"repo": g["repo"], "pr": g["pr"]})
                if not any(p.get("primary") for p in get_prs(entry)):
                    bm = best_match(matches)
                    if bm:
                        set_primary_pr(entry, bm["repo"], bm["pr"])
                if entry.get("repo") is None:
                    pr = primary_pr(entry)
                    if pr:
                        entry["repo"] = _pr_repo(pr, None)
                entry["updatedAt"] = _now_iso()
                applied = {"added": added, "prs": get_prs(entry)}
    except _Abort as exc:
        _err(exc.message)
        return exc.code
    except RegistryError as exc:
        _err(str(exc))
        return 3

    if args.json:
        print(json.dumps({"matches": summary, "applied": applied}, indent=2, sort_keys=True))
    else:
        if not summary:
            print("thread-state: no PR references found")
        for g in summary:
            print(f"  {g['repo']}#{g['pr']}  x{g['count']} ({'/'.join(g['kinds'])})")
        if applied:
            for a in applied["added"]:
                print(f"applied: {a['repo']}#{a['pr']}")
    return 0


# ---------------------------------------------------------------------------
# CLI dispatch
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument(
        "--registry",
        default=None,
        help="path to registry.json (default: ~/.config/thread-state/registry.json or $THREAD_STATE_REGISTRY)",
    )

    def _add_branch_run_activity_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--branch", help="set the branch list to exactly this one, primary")
        sp.add_argument("--add-branch", action="append", metavar="[REPO:]BRANCH", help="append a branch; repeatable")
        sp.add_argument("--rm-branch", action="append", metavar="[REPO:]BRANCH", help="remove a branch by reference; repeatable")
        sp.add_argument("--primary-branch", metavar="[REPO:]BRANCH", help="mark which branch is primary")
        sp.add_argument("--add-run", action="append", metavar="HOST:NAME", help="track an agent-run task; repeatable")
        sp.add_argument("--rm-run", action="append", metavar="HOST:NAME", help="stop tracking an agent-run task; repeatable")
        sp.add_argument("--last-activity", metavar="ISO8601", help="set lastActivityAt directly (source=manual)")
        sp.add_argument("--last-message-id", metavar="SNOWFLAKE", help="derive lastActivityAt from a Discord message id (source=discord)")

    def _add_discord_meta_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--card-message-id", metavar="SNOWFLAKE", help="the Discord message id of the pinned status card")
        sp.add_argument("--parent-channel-id", metavar="SNOWFLAKE", help="the Discord parent-channel id this thread lives under")
        sp.add_argument("--discord-name", metavar="NAME", help="the last known raw Discord thread name")

    p = argparse.ArgumentParser(
        prog="thread-state",
        description="Registry + GitHub polling + rendering for long-lived coding threads.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    sp_add = sub.add_parser("add", parents=[parent], help="create a new thread entry")
    sp_add.add_argument("thread_id")
    sp_add.add_argument("--title", required=True)
    sp_add.add_argument("--repo")
    sp_add.add_argument("--pr", type=int, help="set the PR list to exactly this one PR, primary")
    sp_add.add_argument("--add-pr", action="append", metavar="REF", help="append a PR (N, #N, owner/name#N, or a PR URL); repeatable")
    sp_add.add_argument("--rm-pr", action="append", metavar="REF", help="remove a PR by reference; repeatable")
    sp_add.add_argument("--primary-pr", metavar="REF", help="mark which PR is primary")
    sp_add.add_argument("--desc")
    sp_add.add_argument("--tags")
    _add_branch_run_activity_args(sp_add)
    _add_discord_meta_args(sp_add)
    sp_add.add_argument("--status", default="active")
    sp_add.add_argument("--link", action="append")
    sp_add.add_argument("--force", action="store_true")
    sp_add.add_argument("--json", action="store_true")
    sp_add.set_defaults(func=cmd_add)

    sp_set = sub.add_parser("set", parents=[parent], help="partially update a thread entry")
    sp_set.add_argument("thread_id")
    sp_set.add_argument("--title")
    sp_set.add_argument("--repo")
    sp_set.add_argument("--pr", type=int, help="set the PR list to exactly this one PR, primary")
    sp_set.add_argument("--add-pr", action="append", metavar="REF", help="append a PR (N, #N, owner/name#N, or a PR URL); repeatable")
    sp_set.add_argument("--rm-pr", action="append", metavar="REF", help="remove a PR by reference; repeatable")
    sp_set.add_argument("--primary-pr", metavar="REF", help="mark which PR is primary")
    sp_set.add_argument("--desc")
    sp_set.add_argument("--tags")
    _add_branch_run_activity_args(sp_set)
    _add_discord_meta_args(sp_set)
    sp_set.add_argument("--clear-card-message-id", action="store_true", help="clear cardMessageId (set it to null)")
    sp_set.add_argument("--status")
    sp_set.add_argument("--link", action="append")
    sp_set.add_argument("--note")
    sp_set.add_argument("--force", action="store_true", help="allow --add-run against an unknown host")
    sp_set.add_argument("--progress", action="append", metavar="TEXT", help="append a manual work-summary entry to the updates log; repeatable")
    sp_set.add_argument("--clear-updates", action="store_true", help="wipe the updates log")
    sp_set.add_argument("--json", action="store_true")
    sp_set.set_defaults(func=cmd_set)

    sp_rm = sub.add_parser("rm", parents=[parent], help="remove a thread entry")
    sp_rm.add_argument("thread_id")
    sp_rm.add_argument("--json", action="store_true")
    sp_rm.set_defaults(func=cmd_rm)

    sp_list = sub.add_parser("list", parents=[parent], help="list thread entries")
    sp_list.add_argument("--status")
    sp_list.add_argument("--repo")
    sp_list.add_argument("--tag")
    sp_list.add_argument("--silent-for", metavar="DURATION", help="only show threads silent for longer than DURATION (e.g. 35m, 2h, 90s, 1d)")
    sp_list.add_argument(
        "--include-status", metavar="A,B",
        help="statuses considered 'expected to be progressing' for --silent-for (default: active,review)",
    )
    sp_list.add_argument(
        "--stalled", action="store_true",
        help="with --silent-for, only show entries that ALSO have a running run whose log hasn't grown in DURATION",
    )
    sp_list.add_argument("--with-last", action="store_true", help="append a trailing column with the most recent update")
    sp_list.add_argument("--json", action="store_true")
    sp_list.set_defaults(func=cmd_list)

    sp_silent = sub.add_parser("silent", parents=[parent], help="alias for `list --silent-for` (cron entrypoint)")
    sp_silent.add_argument("--for", dest="for_duration", metavar="DURATION", default="30m")
    sp_silent.add_argument("--status")
    sp_silent.add_argument("--repo")
    sp_silent.add_argument("--tag")
    sp_silent.add_argument("--include-status", metavar="A,B")
    sp_silent.add_argument("--stalled", action="store_true")
    sp_silent.add_argument("--json", action="store_true")
    sp_silent.set_defaults(func=cmd_silent)

    sp_show = sub.add_parser("show", parents=[parent], help="show one thread entry")
    sp_show.add_argument("thread_id")
    sp_show.add_argument("--json", action="store_true")
    sp_show.set_defaults(func=cmd_show)

    sp_refresh = sub.add_parser("refresh", parents=[parent], help="refresh cached PR state from GitHub")
    sp_refresh.add_argument("thread_ids", nargs="*")
    sp_refresh.add_argument("--all", action="store_true")
    sp_refresh.add_argument("--timeout", type=int, default=GH_TIMEOUT_DEFAULT)
    sp_refresh.add_argument("--json", action="store_true")
    sp_refresh.set_defaults(func=cmd_refresh)

    sp_probe = sub.add_parser("probe", parents=[parent], help="poll agent-run tasks over SSH and refresh cached run state")
    sp_probe.add_argument("thread_ids", nargs="*")
    sp_probe.add_argument("--all", action="store_true")
    sp_probe.add_argument("--timeout", type=int, default=8)
    sp_probe.add_argument("--json", action="store_true")
    sp_probe.set_defaults(func=cmd_probe)

    sp_card = sub.add_parser("card", parents=[parent], help="render the pinned status card")
    sp_card.add_argument("thread_id")
    sp_card.add_argument("--format", choices=["markdown", "embed-json"], default="markdown")
    sp_card.add_argument("--json", action="store_true")
    sp_card.set_defaults(func=cmd_card)

    sp_title = sub.add_parser("title", parents=[parent], help="render the suggested Discord thread title")
    sp_title.add_argument("thread_id")
    sp_title.add_argument("--json", action="store_true")
    sp_title.set_defaults(func=cmd_title)

    sp_scan = sub.add_parser("scan", parents=[parent], help="extract PR references from pasted thread text")
    sp_scan.add_argument("thread_id")
    sp_scan.add_argument("--file", required=True, help="path to text file, or '-' for stdin")
    sp_scan.add_argument("--apply", action="store_true")
    sp_scan.add_argument("--json", action="store_true")
    sp_scan.set_defaults(func=cmd_scan)

    sp_log = sub.add_parser("log", parents=[parent], help="print the updates (work-summary) history, newest first")
    sp_log.add_argument("thread_id")
    sp_log.add_argument("--limit", type=int, default=10)
    sp_log.add_argument("--json", action="store_true")
    sp_log.set_defaults(func=cmd_log)

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
