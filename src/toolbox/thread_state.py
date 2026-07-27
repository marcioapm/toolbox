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

    {"schemaVersion": 2, "accounts": {...}, "threads": {"<threadId>": {...}}}

A thread entry carries a `prs` list — one or more PRs, each with its own
`prState` — rather than a single PR. A v1 registry (scalar `pr` +
top-level `prState`) is migrated automatically on first load; see
`_migrate_registry`.

Writes are atomic (temp file + os.replace) and guarded by an flock held for
the whole read-modify-write, so concurrent invocations can't interleave and
corrupt the file. A registry that fails to parse is never overwritten — it
is copied to `registry.json.bad-<timestamp>` and the command exits non-zero.

Usage::

    thread-state add <threadId> --title T [--repo R] [--pr N] [--add-pr REF] ...
    thread-state set <threadId> [--title T] [--status S] [--add-pr REF] [--rm-pr REF] [--primary-pr REF] ...
    thread-state rm <threadId>
    thread-state list [--status S] [--repo R] [--tag T]
    thread-state show <threadId>
    thread-state refresh [<threadId>... | --all]
    thread-state card <threadId> [--format markdown|embed-json]
    thread-state title <threadId>
    thread-state scan <threadId> --file PATH [--apply]

A PR reference (REF) accepts `4694`, `#4694`, `owner/name#4694`, or a full
`https://github.com/owner/name/pull/4694` URL.

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
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

STATUSES = ("active", "review", "blocked", "merged", "closed", "paused", "done")

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

SCHEMA_VERSION = 2

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


def _migrate_registry_data(data: Dict[str, Any]) -> bool:
    """Migrate an in-memory registry dict from v1 to v2 (`prs` list per
    entry). Idempotent — a no-op once `schemaVersion` is already >= 2.
    Returns True if a v1 -> v2 conversion actually happened, so callers that
    persist to disk know whether to back up the pre-migration file first."""
    if data.get("schemaVersion", 1) >= SCHEMA_VERSION:
        return False
    for entry in data.get("threads", {}).values():
        _migrate_entry_v1_to_v2(entry)
    data["schemaVersion"] = SCHEMA_VERSION
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


@contextlib.contextmanager
def _locked_registry(path: Path):
    """Exclusive-lock the registry for a read-modify-write. The yielded dict
    is only persisted if the block completes without raising — raise
    `_Abort` (or anything else) to discard in-progress changes.

    If the on-disk file is still v1, it is migrated to v2 in memory before
    being handed to the caller; the very first time that migration is about
    to be persisted, the pre-migration bytes are backed up to
    `registry.json.v1-backup-<ts>` so no data can be lost."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        pre_migration_raw = path.read_text() if path.exists() and path.stat().st_size > 0 else None
        data, migrated = _load_and_migrate(path)
        yield data
        if migrated and pre_migration_raw is not None:
            backup = path.with_name(path.name + f".v1-backup-{_timestamp_slug()}")
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
    if not path.exists():
        yield _empty_registry()
        return
    fd = os.open(path, os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_SH)
        try:
            data, _migrated = _load_and_migrate(path)
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
        yield data
    finally:
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
    applied (e.g. no PRs at all, or the existing status should be kept)."""
    prs = get_prs(entry)
    if not prs:
        return None

    states = [(p.get("prState") or {}).get("state") for p in prs]
    cis = [(p.get("prState") or {}).get("ci") for p in prs]

    if any(s == "OPEN" and ci == "FAILURE" for s, ci in zip(states, cis)):
        return "blocked"

    if all(s == "MERGED" for s in states):
        return "merged"

    if (
        any(s == "CLOSED" for s in states)
        and all(s in ("CLOSED", "MERGED") for s in states)
        and not any(s == "OPEN" for s in states)
    ):
        return "closed"

    if any(s == "OPEN" for s in states):
        current = entry.get("status")
        if current in ("review", "blocked", "paused"):
            return current
        return "active"

    return None


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
    primary PR bolded and a per-PR stale warning."""
    number = p["number"]
    repo = _pr_repo(p, entry.get("repo")) or ""
    pr_state = p.get("prState") or {}
    url = pr_state.get("url") or (f"https://github.com/{repo}/pull/{number}" if repo else "")
    state = pr_state.get("state") or "?"
    ci = pr_state.get("ci") or "NONE"
    label = f"[#{number}]({url})" if url else f"#{number}"
    if p.get("primary"):
        label = f"**{label}**"
    line = f"PR: {label} {state} · CI {ci}"
    if ci == "FAILURE":
        line += " ⚠️"
    if pr_state.get("stale"):
        line += f" — ⚠️ stale (last checked: {pr_state.get('checkedAt', '?')})"
    return line


def render_card_markdown(entry: Dict[str, Any]) -> str:
    """The pinned status card message body."""
    icon = _effective_icon(entry)
    prs = get_prs(entry)

    lines = [f"{icon} **{entry.get('title', '')}** — {entry.get('status', '')}"]
    if entry.get("description"):
        lines.append(entry["description"])

    meta = []
    if entry.get("repo"):
        meta.append(f"repo: {entry['repo']}")
    if not prs and entry.get("branch"):
        meta.append(f"branch: `{entry['branch']}`")
    if meta:
        lines.append(" | ".join(meta))

    for p in prs:
        lines.append(_pr_line(entry, p))

    if entry.get("tags"):
        lines.append("tags: " + ", ".join(entry["tags"]))
    if entry.get("links"):
        lines.append("links: " + " ".join(entry["links"]))
    if entry.get("notes"):
        lines.append(f"notes: {entry['notes']}")

    return "\n".join(lines)


def render_card_embed(entry: Dict[str, Any]) -> Dict[str, Any]:
    """A Discord embed object, ready to hand to a Discord client."""
    prs = get_prs(entry)
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
    if not prs and entry.get("branch"):
        fields.append({"name": "Branch", "value": entry["branch"], "inline": True})
    if primary_state.get("ci"):
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


def _format_table(entries: List[Dict[str, Any]]) -> str:
    headers = ["", "THREAD", "TITLE", "STATUS", "PR", "CI", "DESCRIPTION"]
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
        rows.append([
            _effective_icon(e),
            str(e.get("threadId", "")),
            str(e.get("title", "")),
            str(e.get("status", "")),
            pr_col,
            ci,
            str(e.get("description", "")),
        ])
    widths = [
        max([len(headers[i])] + [len(r[i]) for r in rows]) for i in range(len(headers))
    ]

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
    lines = [
        f"threadId: {entry.get('threadId')}",
        f"title: {entry.get('title')}",
        f"status: {_effective_icon(entry)} {entry.get('status')}",
        f"repo: {entry.get('repo') or '-'}",
        f"branch: {entry.get('branch') or '-'}",
        f"description: {entry.get('description') or '-'}",
        f"tags: {', '.join(entry.get('tags') or []) or '-'}",
        f"links: {', '.join(entry.get('links') or []) or '-'}",
        f"notes: {entry.get('notes') or '-'}",
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


def fetch_pr_state(
    repo: str,
    pr: int,
    accounts: Dict[str, str],
    timeout: int,
    runner,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Fetch fresh PR state via `gh`, switching accounts first if the repo's
    owner maps to one. Returns (prState, None) on success or (None, error)
    on any failure — never raises, so callers can always fall back to the
    previously cached state."""
    account = _gh_account_for(repo, accounts)
    if account:
        try:
            runner(["gh", "auth", "switch", "--user", account], timeout)
        except (subprocess.TimeoutExpired, OSError):
            pass  # best-effort — gh may already be on the right account

    cmd = [
        "gh", "pr", "view", str(pr), "--repo", repo,
        "--json", "state,title,url,mergedAt,statusCheckRollup",
    ]
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
    """A shallow copy of `entry` for JSON output, with a read-only `pr` field
    (the primary PR's number, or null) so old consumers that only knew the
    v1 scalar `pr` field don't hard-break."""
    out = dict(entry)
    primary = primary_pr(entry)
    out["pr"] = primary.get("number") if primary else None
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
                "branch": args.branch,
                "status": status,
                "tags": _split_tags(args.tags),
                "links": list(args.link or []),
                "notes": "",
                "prs": [],
                "createdAt": now,
                "updatedAt": now,
            }
            _apply_pr_mutations(entry, args)
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
            if args.branch is not None:
                entry["branch"] = args.branch
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
    entries.sort(key=lambda e: (e.get("title") or "", e.get("threadId") or ""))

    if args.json:
        print(json.dumps([_entry_with_computed_pr(e) for e in entries], indent=2, sort_keys=True))
    elif not entries:
        print("thread-state: no threads")
    else:
        print(_format_table(entries))
    return 0


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

            def _work(tid: str, idx: int):
                entry = threads[tid]
                p = get_prs(entry)[idx]
                repo = _pr_repo(p, entry.get("repo"))
                pr_state, error = fetch_pr_state(repo, p["number"], accounts, args.timeout, _run_command)
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
                    if error is not None:
                        prev = dict(p.get("prState") or {})
                        prev["stale"] = True
                        prev["error"] = error
                        prev["checkedAt"] = _now_iso()
                        p["prState"] = prev
                        per_thread_errors.setdefault(tid, []).append(error)
                    else:
                        p["prState"] = pr_state

                    per_thread_done[tid] += 1
                    if per_thread_done[tid] == per_thread_total[tid]:
                        new_status = derive_thread_status(entry)
                        if new_status is not None:
                            entry["status"] = new_status
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
    sp_add.add_argument("--branch")
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
    sp_set.add_argument("--branch")
    sp_set.add_argument("--status")
    sp_set.add_argument("--link", action="append")
    sp_set.add_argument("--note")
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
    sp_list.add_argument("--json", action="store_true")
    sp_list.set_defaults(func=cmd_list)

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
