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

    {"schemaVersion": 1, "accounts": {...}, "threads": {"<threadId>": {...}}}

Writes are atomic (temp file + os.replace) and guarded by an flock held for
the whole read-modify-write, so concurrent invocations can't interleave and
corrupt the file. A registry that fails to parse is never overwritten — it
is copied to `registry.json.bad-<timestamp>` and the command exits non-zero.

Usage::

    thread-state add <threadId> --title T [--repo R] [--pr N] ...
    thread-state set <threadId> [--title T] [--status S] ...
    thread-state rm <threadId>
    thread-state list [--status S] [--repo R] [--tag T]
    thread-state show <threadId>
    thread-state refresh [<threadId>... | --all]
    thread-state card <threadId> [--format markdown|embed-json]
    thread-state title <threadId>
    thread-state scan <threadId> --file PATH [--apply]

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

THREAD_TITLE_MAX = 100


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
    return {"schemaVersion": 1, "accounts": {}, "threads": {}}


def _parse_registry(path: Path) -> Dict[str, Any]:
    """Read and parse the registry, or return an empty one if it doesn't
    exist yet. Raises RegistryError (without touching the file) if it exists
    but is corrupt, after copying it aside for forensics."""
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
        backup = path.with_name(path.name + f".bad-{_now_iso().replace(':', '').replace('-', '')}")
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
    `_Abort` (or anything else) to discard in-progress changes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        data = _parse_registry(path)
        yield data
        _write_registry(path, data)
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


@contextlib.contextmanager
def _read_only_registry(path: Path):
    """Shared-lock read of the registry. Never writes."""
    if not path.exists():
        yield _empty_registry()
        return
    fd = os.open(path, os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_SH)
        try:
            data = _parse_registry(path)
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
        yield data
    finally:
        os.close(fd)


# ---------------------------------------------------------------------------
# rendering: icon/color, title, card
# ---------------------------------------------------------------------------

def _effective_icon(entry: Dict[str, Any]) -> str:
    """CI failure overrides the status icon to red, except for statuses
    where the PR is already resolved (merged/closed/done)."""
    status = entry.get("status", "active")
    icon = STATUS_EMOJI.get(status, STATUS_EMOJI["active"])
    ci = (entry.get("prState") or {}).get("ci")
    if ci == "FAILURE" and status not in _TERMINAL_STATUSES:
        return STATUS_EMOJI["blocked"]
    return icon


def _effective_color(entry: Dict[str, Any]) -> int:
    status = entry.get("status", "active")
    ci = (entry.get("prState") or {}).get("ci")
    if ci == "FAILURE" and status not in _TERMINAL_STATUSES:
        return STATUS_COLOR["blocked"]
    return STATUS_COLOR.get(status, STATUS_COLOR["active"])


def render_title(entry: Dict[str, Any]) -> str:
    """Suggested Discord thread name, e.g. `🔵 mSPRT · PR#4694 · review`.
    Discord thread names cap at 100 chars — the title component is
    truncated (with an ellipsis) to fit; the icon/PR/status suffix never is."""
    icon = _effective_icon(entry)
    title = entry.get("title") or ""
    status = entry.get("status", "active")
    pr = entry.get("pr")

    suffix_parts = []
    if pr:
        suffix_parts.append(f"PR#{pr}")
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


def render_card_markdown(entry: Dict[str, Any]) -> str:
    """The pinned status card message body."""
    icon = _effective_icon(entry)
    pr_state = entry.get("prState") or {}

    lines = [f"{icon} **{entry.get('title', '')}** — {entry.get('status', '')}"]
    if entry.get("description"):
        lines.append(entry["description"])

    meta = []
    if entry.get("repo"):
        meta.append(f"repo: {entry['repo']}")
    if entry.get("pr"):
        url = pr_state.get("url") or f"https://github.com/{entry.get('repo', '')}/pull/{entry['pr']}"
        meta.append(f"PR: [#{entry['pr']}]({url})")
    if entry.get("branch"):
        meta.append(f"branch: `{entry['branch']}`")
    if pr_state.get("ci"):
        meta.append(f"CI: {pr_state['ci']}")
    if meta:
        lines.append(" | ".join(meta))

    if pr_state.get("stale"):
        lines.append(f"⚠️ stale prState (last checked: {pr_state.get('checkedAt', '?')})")
    if entry.get("tags"):
        lines.append("tags: " + ", ".join(entry["tags"]))
    if entry.get("links"):
        lines.append("links: " + " ".join(entry["links"]))
    if entry.get("notes"):
        lines.append(f"notes: {entry['notes']}")

    return "\n".join(lines)


def render_card_embed(entry: Dict[str, Any]) -> Dict[str, Any]:
    """A Discord embed object, ready to hand to a Discord client."""
    pr_state = entry.get("prState") or {}

    fields = []
    if entry.get("repo"):
        fields.append({"name": "Repo", "value": entry["repo"], "inline": True})
    if entry.get("pr"):
        fields.append({"name": "PR", "value": f"#{entry['pr']}", "inline": True})
    if entry.get("branch"):
        fields.append({"name": "Branch", "value": entry["branch"], "inline": True})
    if pr_state.get("ci"):
        fields.append({"name": "CI", "value": pr_state["ci"], "inline": True})
    fields.append({"name": "Status", "value": entry.get("status", ""), "inline": True})
    if entry.get("tags"):
        fields.append({"name": "Tags", "value": ", ".join(entry["tags"]), "inline": False})

    return {
        "title": f"{_effective_icon(entry)} {entry.get('title', '')}",
        "description": entry.get("description", ""),
        "color": _effective_color(entry),
        "fields": fields,
        "url": pr_state.get("url") or None,
        "footer": {"text": f"threadId={entry.get('threadId', '')}"},
    }


def _format_table(entries: List[Dict[str, Any]]) -> str:
    headers = ["", "THREAD", "TITLE", "STATUS", "PR", "CI", "DESCRIPTION"]
    rows = []
    for e in entries:
        pr_state = e.get("prState") or {}
        rows.append([
            _effective_icon(e),
            str(e.get("threadId", "")),
            str(e.get("title", "")),
            str(e.get("status", "")),
            f"#{e['pr']}" if e.get("pr") else "-",
            pr_state.get("ci") or "-",
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
    lines = [
        f"threadId: {entry.get('threadId')}",
        f"title: {entry.get('title')}",
        f"status: {_effective_icon(entry)} {entry.get('status')}",
        f"repo: {entry.get('repo') or '-'}",
        f"pr: {entry.get('pr') or '-'}",
        f"branch: {entry.get('branch') or '-'}",
        f"description: {entry.get('description') or '-'}",
        f"tags: {', '.join(entry.get('tags') or []) or '-'}",
        f"links: {', '.join(entry.get('links') or []) or '-'}",
        f"notes: {entry.get('notes') or '-'}",
        f"createdAt: {entry.get('createdAt')}",
        f"updatedAt: {entry.get('updatedAt')}",
    ]
    pr_state = entry.get("prState")
    if pr_state:
        lines.append(
            f"prState: state={pr_state.get('state')} ci={pr_state.get('ci')} "
            f"stale={pr_state.get('stale')} checkedAt={pr_state.get('checkedAt')}"
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
    }
    return pr_state, None


def _apply_auto_status(entry: Dict[str, Any], pr_state: Dict[str, Any]) -> None:
    """MERGED -> status merged, CLOSED (not merged) -> status closed. Any
    other PR state leaves the manually-set status untouched — this is what
    keeps a manual `blocked`/`paused` from being auto-downgraded."""
    state = pr_state.get("state")
    if state == "MERGED":
        entry["status"] = "merged"
    elif state == "CLOSED":
        entry["status"] = "closed"


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
                "pr": args.pr,
                "branch": args.branch,
                "status": status,
                "tags": _split_tags(args.tags),
                "links": list(args.link or []),
                "notes": "",
                "prState": None,
                "createdAt": now,
                "updatedAt": now,
            }
            threads[args.thread_id] = entry
    except _Abort as exc:
        _err(exc.message)
        return exc.code
    except RegistryError as exc:
        _err(str(exc))
        return 3

    if args.json:
        print(json.dumps(entry, indent=2, sort_keys=True))
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
            if args.pr is not None:
                entry["pr"] = args.pr
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
            entry["updatedAt"] = _now_iso()
            result = dict(entry)
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
        print(json.dumps(entries, indent=2, sort_keys=True))
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
        print(json.dumps(entry, indent=2, sort_keys=True))
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

            targets = [tid for tid in ids if threads[tid].get("repo") and threads[tid].get("pr")]
            skipped = [tid for tid in ids if tid not in targets]

            def _work(tid: str):
                entry = threads[tid]
                pr_state, error = fetch_pr_state(entry["repo"], entry["pr"], accounts, args.timeout, _run_command)
                return tid, pr_state, error

            worker_count = min(REFRESH_MAX_WORKERS, len(targets)) or 1
            with ThreadPoolExecutor(max_workers=worker_count) as ex:
                futures = {ex.submit(_work, tid): tid for tid in targets}
                for fut in as_completed(futures):
                    tid, pr_state, error = fut.result()
                    entry = threads[tid]
                    if error is not None:
                        prev = dict(entry.get("prState") or {})
                        prev["stale"] = True
                        prev["error"] = error
                        prev["checkedAt"] = _now_iso()
                        entry["prState"] = prev
                        results.append({"threadId": tid, "ok": False, "error": error})
                    else:
                        entry["prState"] = pr_state
                        _apply_auto_status(entry, pr_state)
                        entry["updatedAt"] = _now_iso()
                        results.append({"threadId": tid, "ok": True, "prState": pr_state})

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
                ps = r["prState"]
                print(f"{r['threadId']}: refreshed (state={ps['state']} ci={ps['ci']})")
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
                bm = best_match(matches)
                if bm:
                    entry["repo"] = bm["repo"]
                    entry["pr"] = bm["pr"]
                    entry["updatedAt"] = _now_iso()
                    applied = {"repo": bm["repo"], "pr": bm["pr"]}
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
            print(f"applied: {applied['repo']}#{applied['pr']}")
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
    sp_add.add_argument("--pr", type=int)
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
    sp_set.add_argument("--pr", type=int)
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
