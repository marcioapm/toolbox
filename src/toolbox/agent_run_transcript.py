#!/usr/bin/env python3
"""Read harness conversation stores for ``agent-run transcript``.

Each reader returns normalized entries plus the number of malformed records
it skipped. Missing, unreadable, or incompatible stores raise
``TranscriptSourceError``. Readers never modify their stores.
"""
from __future__ import annotations

import fnmatch
import json
import os
import sqlite3
import stat as _stat_module
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional


# Overridable the same way agent_run.py's AGENT_RUN_STATE_DIR/AGENT_RUN_LOG_DIR
# are: an env var read once at import, so tests can point every reader at a
# fixture store without ever touching the real, per-machine ones.
OPENCODE_DB_PATH = Path(
    os.environ.get(
        "AGENT_RUN_OPENCODE_DB", str(Path.home() / ".local" / "share" / "opencode" / "opencode.db")
    )
)
CLAUDE_PROJECTS_DIR = Path(
    os.environ.get("AGENT_RUN_CLAUDE_PROJECTS_DIR", str(Path.home() / ".claude" / "projects"))
)
CODEX_SESSIONS_DIR = Path(
    os.environ.get("AGENT_RUN_CODEX_SESSIONS_DIR", str(Path.home() / ".codex" / "sessions"))
)


class TranscriptSourceError(Exception):
    """A session store could not be located or read."""

    def __init__(self, message: str, code: str = "store_unreadable") -> None:
        super().__init__(message)
        self.code = code


# These helpers report plain absence but leave other OSErrors observable.
def _stat_kind(path: Path) -> Optional[int]:
    """Return `st_mode`, or ``None`` when *path* is absent."""
    try:
        return os.stat(path).st_mode
    except FileNotFoundError:
        return None


def _path_is_file(path: Path) -> bool:
    mode = _stat_kind(path)
    return mode is not None and _stat_module.S_ISREG(mode)


def _path_is_dir(path: Path) -> bool:
    mode = _stat_kind(path)
    return mode is not None and _stat_module.S_ISDIR(mode)


def _scandir_names(directory: Path) -> list[str]:
    with os.scandir(directory) as it:
        return [entry.name for entry in it]


def _glob_one_level(directory: Path, pattern: str) -> list[str]:
    return sorted(name for name in _scandir_names(directory) if fnmatch.fnmatch(name, pattern))


def _glob_project_session_files(root: Path, filename: str) -> list[Path]:
    """Return existing ``<root>/*/<filename>`` paths."""
    try:
        child_names = _scandir_names(root)
    except FileNotFoundError:
        return []
    matches = []
    for name in child_names:
        candidate = root / name / filename
        if _path_is_file(candidate):
            matches.append(candidate)
    return sorted(matches)


def _glob_codex_rollout_files(root: Path, session_id: str) -> list[Path]:
    """Return rollout files in the fixed ``YYYY/MM/DD`` layout."""
    suffix = f"-{session_id}.jsonl"
    try:
        years = _scandir_names(root)
    except FileNotFoundError:
        return []
    matches: list[Path] = []
    for year in years:
        year_dir = root / year
        try:
            months = _scandir_names(year_dir)
        except FileNotFoundError:
            continue
        for month in months:
            month_dir = year_dir / month
            try:
                days = _scandir_names(month_dir)
            except FileNotFoundError:
                continue
            for day in days:
                day_dir = month_dir / day
                try:
                    names = _scandir_names(day_dir)
                except FileNotFoundError:
                    continue
                for name in names:
                    if name.startswith("rollout-") and name.endswith(suffix):
                        matches.append(day_dir / name)
    return sorted(matches)


# Reject values unsafe as literal path components or future glob inputs.
_SESSION_ID_FORBIDDEN_SUBSTRINGS = ("/", "\\", "..", "\x00")
_SESSION_ID_GLOB_METACHARS = frozenset("*?[]")


def _validate_session_id(session_id: str) -> None:
    """Require ``session_id`` to be usable as a single filename component.
    Raises ``TranscriptSourceError`` (``store_unreadable``) otherwise."""
    if not isinstance(session_id, str) or not session_id:
        raise TranscriptSourceError("empty or non-string session_id", code="store_unreadable")
    if any(bad in session_id for bad in _SESSION_ID_FORBIDDEN_SUBSTRINGS):
        raise TranscriptSourceError(
            f"unsafe session_id {session_id!r}: contains a path separator, NUL, or '..'",
            code="store_unreadable",
        )
    if _SESSION_ID_GLOB_METACHARS & set(session_id):
        raise TranscriptSourceError(
            f"unsafe session_id {session_id!r}: contains a glob metacharacter", code="store_unreadable"
        )


@dataclass
class TranscriptEntry:
    """One normalised conversation event, common to every harness reader.

    ``type`` is one of "user", "assistant", "reasoning", "tool". Text
    entries (user/assistant/reasoning) carry ``text``; tool entries carry
    ``tool_name`` and any of ``tool_summary``/``tool_input``/``tool_output``/
    ``tool_status`` the source store provided -- any of these may be
    ``None`` when the store omitted them. ``time`` is an ISO-8601 UTC
    string when the source recorded one, else ``None``. ``subagent`` is a
    label (claude only) when the entry came from a subagent transcript
    rather than the main session.
    """

    type: str
    harness: str
    time: Optional[str] = None
    text: Optional[str] = None
    tool_name: Optional[str] = None
    tool_summary: Optional[str] = None
    tool_input: Optional[dict] = None
    tool_output: Optional[str] = None
    tool_status: Optional[str] = None
    subagent: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "harness": self.harness,
            "time": self.time,
            "subagent": self.subagent,
            "text": self.text,
            "tool_name": self.tool_name,
            "tool_summary": self.tool_summary,
            "tool_input": self.tool_input,
            "tool_output": self.tool_output,
            "tool_status": self.tool_status,
        }


# A single tool result is bounded independently of any --head/--tail entry
# slicing: a `read` of a 3000-line file must not paste 3000 lines into every
# rendering of the transcript, sliced or not.
TOOL_OUTPUT_MAX_LINES = 40
TOOL_OUTPUT_MAX_BYTES = 4 * 1024


def _bound_tool_output(text: str) -> str:
    """Cap a tool result to TOOL_OUTPUT_MAX_LINES lines and
    TOOL_OUTPUT_MAX_BYTES bytes, whichever is reached first, appending a
    marker naming how many lines were dropped."""
    lines = text.splitlines()
    kept = lines[:TOOL_OUTPUT_MAX_LINES]
    dropped_by_lines = len(lines) - len(kept)
    truncated_by_bytes = False
    encoded = "\n".join(kept).encode("utf-8")
    if len(encoded) > TOOL_OUTPUT_MAX_BYTES:
        encoded = encoded[:TOOL_OUTPUT_MAX_BYTES]
        kept_text = encoded.decode("utf-8", errors="ignore")
        kept = kept_text.split("\n")
        truncated_by_bytes = True
    if dropped_by_lines <= 0 and not truncated_by_bytes:
        return text
    if truncated_by_bytes:
        marker = f"[agent-run: tool output truncated, {len(lines)} lines / {len(text.encode('utf-8'))} bytes total]"
    else:
        marker = f"[agent-run: tool output truncated, {dropped_by_lines} more line(s) dropped]"
    return "\n".join(kept) + "\n" + marker


def read_transcript(
    harness: str, session_id: str, cwd: Optional[str]
) -> tuple[list[TranscriptEntry], int]:
    """Dispatch to the reader for ``harness``. Raises ``TranscriptSourceError``
    for an unrecognised harness or a store that cannot be located/opened at
    all. Returns (entries, skipped_record_count), with every entry's
    ``tool_output`` already bounded by ``_bound_tool_output``."""
    if harness == "opencode":
        entries, skipped = _read_opencode(session_id)
    elif harness == "claude":
        entries, skipped = _read_claude(session_id, cwd)
    elif harness == "codex":
        entries, skipped = _read_codex(session_id)
    else:
        raise TranscriptSourceError(f"unrecognised harness {harness!r} in session.json", code="unknown_harness")
    for entry in entries:
        if entry.tool_output:
            entry.tool_output = _bound_tool_output(entry.tool_output)
    return entries, skipped


def count_transcript(
    harness: str, session_id: str, cwd: Optional[str]
) -> tuple[int, Optional[str]]:
    """Return ``(entries, newest_timestamp)`` without building a transcript.

    Claude and Codex count renderable entries exactly. OpenCode counts raw
    indexed `part` rows, so a nonzero count need not render. A zero is returned
    only after a trustworthy store search and read; failures raise
    ``TranscriptSourceError``.
    """
    if harness == "opencode":
        return _count_opencode(session_id)
    if harness == "claude":
        return _count_claude(session_id, cwd)
    if harness == "codex":
        return _count_codex(session_id)
    raise TranscriptSourceError(f"unrecognised harness {harness!r} in session.json", code="unknown_harness")


# ---------------------------------------------------------------------------
# The submission-verification predicate
# ---------------------------------------------------------------------------
#
# THE PREDICATE. A submission counts as delivered only when the session's
# conversation store gained a **user-role turn whose own text carries the
# submitted prompt**. Three conditions, all required:
#
#   1. user role -- an assistant reply, a reasoning trace or a tool record in
#      the current turn must never satisfy it, since none of them implies new
#      input reached the harness;
#   2. our content -- a bare user-role envelope with no text part yet, or a
#      user-role message the harness synthesised for itself, is not our
#      prompt, however much it moves a record count;
#   3. observed, not assumed -- a store that cannot be fully read yields no
#      number at all (``TranscriptSourceError``), never a zero, because a zero
#      is read as proof the prompt did not land and licenses a resend.
#
# Every witness source -- opencode's HTTP endpoint, its SQLite store, and the
# claude/codex JSONL rollouts -- implements exactly this, so "verified" means
# one thing across every transport.

def normalize_prompt_text(text: str) -> str:
    """Collapse every whitespace run to a single space and strip the ends.

    A TUI re-wraps a long prompt before its harness records the turn, so a
    byte-exact comparison would miss a prompt that plainly did land. Nothing
    but whitespace is normalised: the words themselves must match.
    """
    return " ".join(text.split())


def turn_carries_prompt(turn_text: Optional[str], normalized_prompt: str) -> bool:
    """True when one turn's text carries *normalized_prompt* (already passed
    through ``normalize_prompt_text``).

    An empty prompt matches nothing: it would otherwise degrade the predicate
    back to a bare user-record count, which is what content attribution
    exists to replace.
    """
    if not normalized_prompt or not isinstance(turn_text, str):
        return False
    return normalized_prompt in normalize_prompt_text(turn_text)


def count_prompt_turns(harness: str, session_id: str, cwd: Optional[str], prompt: str) -> int:
    """Return how many user-role turns in one session carry *prompt*.

    Implements THE PREDICATE above for the harness's own store. Failures
    raise ``TranscriptSourceError`` rather than returning 0.
    """
    normalized = normalize_prompt_text(prompt)
    if harness == "opencode":
        return _count_opencode_prompt_turns(session_id, normalized)
    if harness == "claude":
        return _count_claude_prompt_turns(session_id, cwd, normalized)
    if harness == "codex":
        return _count_codex_prompt_turns(session_id, normalized)
    raise TranscriptSourceError(f"unrecognised harness {harness!r} in session.json", code="unknown_harness")


def _iso_from_epoch_ms(value: Any) -> Optional[str]:
    if not isinstance(value, (int, float)):
        return None
    try:
        return datetime.fromtimestamp(value / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except (OverflowError, OSError, ValueError):
        return None


def _iso_utc_z(when: datetime) -> str:
    """Render *when* in UTC before appending the `Z` suffix."""
    return when.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# opencode: SQLite at ~/.local/share/opencode/opencode.db
# ---------------------------------------------------------------------------

def _opencode_connect() -> sqlite3.Connection:
    """Open the opencode store read-only with a non-blocking busy handler.

    `file:...?mode=ro`, `uri=True`: a live opencode run holds this database
    open under WAL, and no reader may write to it or trigger a checkpoint.

    `timeout=0` plus an explicit `PRAGMA busy_timeout=0` (the latter for
    SQLite builds where the connect-time timeout does not bind the busy
    handler) make a held write lock fail immediately instead of blocking
    for the default 5s: `mode=ro` stops this reader from writing, but it
    does not make lock acquisition non-blocking.

    Raises TranscriptSourceError when the file does not exist, a path
    component cannot be stat'd (e.g. permission denied -- distinct from
    genuine absence), or the connection itself cannot be opened.
    """
    try:
        db_present = _stat_kind(OPENCODE_DB_PATH) is not None
    except OSError as exc:
        raise TranscriptSourceError(
            f"cannot check opencode session store at {OPENCODE_DB_PATH}: {exc}", code="store_unreadable"
        ) from exc
    if not db_present:
        raise TranscriptSourceError(f"opencode session store not found at {OPENCODE_DB_PATH}", code="store_missing")
    try:
        conn = sqlite3.connect(f"file:{OPENCODE_DB_PATH}?mode=ro", uri=True, timeout=0)
        conn.execute("pragma busy_timeout=0")
    except sqlite3.OperationalError as exc:
        raise TranscriptSourceError(
            f"cannot open opencode session store at {OPENCODE_DB_PATH}: {exc}", code="store_unreadable"
        ) from exc
    return conn


def _opencode_query_error(exc: sqlite3.Error) -> TranscriptSourceError:
    """Classify a query-time sqlite3 error as a locked store (retryable by a
    caller) versus an unreadable one (not).

    sqlite3 in this Python version exposes no reliable error code on
    OperationalError (sqlite_errorcode is CPython 3.11+ only and not
    guaranteed populated), so "locked"/"busy" substring matching on the
    lowercased message is the only portable way to tell a held write lock
    apart from a broken store (e.g. a missing table) -- both raise
    OperationalError. DatabaseError covers a file that opens but is not a
    valid/complete sqlite database (mid-write copy, truncated backup): the
    store as a whole is unusable, not one bad row.
    """
    if isinstance(exc, sqlite3.OperationalError):
        message = str(exc).lower()
        if "locked" in message or "busy" in message:
            return TranscriptSourceError(
                f"opencode session store at {OPENCODE_DB_PATH} is locked by another writer: {exc}",
                code="store_locked",
            )
    return TranscriptSourceError(f"opencode session store unreadable: {exc}", code="store_unreadable")


def _read_opencode(session_id: str) -> tuple[list[TranscriptEntry], int]:
    """Read the `part`/`message` tables for one session.

    `part.data` carries the event; the joined `message.data` supplies the
    role (user/assistant) and fallback timestamp a `text`/`reasoning` part
    does not itself carry.
    """
    conn = _opencode_connect()
    entries: list[TranscriptEntry] = []
    skipped = 0
    try:
        cursor = conn.execute(
            "select p.data, m.data from part p "
            "join message m on p.message_id = m.id "
            "where p.session_id = ? order by p.time_created",
            (session_id,),
        )
        for part_raw, message_raw in cursor:
            try:
                part = json.loads(part_raw)
                message = json.loads(message_raw)
            except (json.JSONDecodeError, TypeError):
                skipped += 1
                continue
            if not isinstance(part, dict) or not isinstance(message, dict):
                skipped += 1
                continue
            entry = _opencode_entry(part, message)
            if entry is not None:
                entries.append(entry)
    except (sqlite3.OperationalError, sqlite3.DatabaseError) as exc:
        raise _opencode_query_error(exc) from exc
    finally:
        conn.close()
    return entries, skipped


def _count_opencode(session_id: str) -> tuple[int, Optional[str]]:
    """`part` row count and newest `time_created` for one session, via the
    `part_session_idx` index, without loading or parsing any row's `data`.
    Raises the same errors as `_opencode_connect`/query execution
    (missing/locked/unreadable store)."""
    conn = _opencode_connect()
    try:
        cursor = conn.execute(
            "select count(*), max(time_created) from part where session_id = ?",
            (session_id,),
        )
        count, newest_raw = cursor.fetchone()
    except (sqlite3.OperationalError, sqlite3.DatabaseError) as exc:
        raise _opencode_query_error(exc) from exc
    finally:
        conn.close()
    return count, _iso_from_epoch_ms(newest_raw)


def _count_opencode_prompt_turns(session_id: str, normalized_prompt: str) -> int:
    """User-role `message` rows for one session whose text parts carry the
    prompt.

    Role comes from `message.data`, but the prompt itself lives in the turn's
    `part` rows, and opencode persists the envelope and its parts as separate
    events -- so a row that exists proves nothing about what it will
    eventually hold. Both tables are read in one pass and joined here rather
    than per message, which would issue a query per turn. Raises the same
    errors as `_opencode_connect`/query execution (missing/locked/unreadable
    store).
    """
    conn = _opencode_connect()
    try:
        user_message_ids = set()
        for message_id, raw in conn.execute(
            "select id, data from message where session_id = ?", (session_id,)
        ):
            try:
                data = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(data, dict) and data.get("role") == "user":
                user_message_ids.add(message_id)

        texts_by_message: dict[str, list[str]] = {}
        for message_id, raw in conn.execute(
            "select message_id, data from part where session_id = ? order by time_created",
            (session_id,),
        ):
            if message_id not in user_message_ids:
                continue
            try:
                part = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
                texts_by_message.setdefault(message_id, []).append(part["text"])

        return sum(
            1 for message_id in user_message_ids
            if turn_carries_prompt("\n".join(texts_by_message.get(message_id, ())), normalized_prompt)
        )
    except (sqlite3.OperationalError, sqlite3.DatabaseError) as exc:
        raise _opencode_query_error(exc) from exc
    finally:
        conn.close()


def _opencode_entry(part: dict, message: dict) -> Optional[TranscriptEntry]:
    ptype = part.get("type")
    role = message.get("role")
    message_time = (message.get("time") or {}).get("created")

    if ptype == "text":
        text = part.get("text")
        if not isinstance(text, str) or not text.strip():
            return None
        entry_type = "assistant" if role == "assistant" else "user"
        return TranscriptEntry(
            type=entry_type, harness="opencode", time=_iso_from_epoch_ms(message_time), text=text
        )
    if ptype == "reasoning":
        text = part.get("text")
        if not isinstance(text, str) or not text.strip():
            return None
        return TranscriptEntry(
            type="reasoning", harness="opencode", time=_iso_from_epoch_ms(message_time), text=text
        )
    if ptype == "tool":
        state = part.get("state") if isinstance(part.get("state"), dict) else {}
        state_time = state.get("time") if isinstance(state.get("time"), dict) else {}
        title = state.get("title")
        tool_input = state.get("input")
        tool_output = state.get("output")
        status = state.get("status")
        when = state_time.get("start") or state_time.get("end") or message_time
        return TranscriptEntry(
            type="tool",
            harness="opencode",
            time=_iso_from_epoch_ms(when),
            tool_name=part.get("tool") if isinstance(part.get("tool"), str) else None,
            tool_summary=title if isinstance(title, str) else None,
            tool_input=tool_input if isinstance(tool_input, dict) else None,
            tool_output=tool_output if isinstance(tool_output, str) else None,
            tool_status=status if isinstance(status, str) else None,
        )
    # patch/step-start/step-finish/compaction: bookkeeping, not conversation.
    return None


# ---------------------------------------------------------------------------
# claude: JSONL at ~/.claude/projects/<mangled-cwd>/<session_id>.jsonl
# ---------------------------------------------------------------------------

def _mangle_cwd(cwd: str) -> str:
    return cwd.replace("/", "-")


def _claude_classify_file_for_session(path: Path, session_id: str) -> tuple[bool, bool]:
    """Return whether conversation records match or conflict with *session_id*."""
    skip_counter = _JsonlSkipCounter()
    matches = False
    conflicts = False
    for record in _iter_jsonl_objects(path, skip_counter):
        if record.get("type") not in ("user", "assistant"):
            continue
        record_session_id = record.get("sessionId")
        if record_session_id == session_id:
            matches = True
        elif record_session_id is not None:
            conflicts = True
    return matches, conflicts


def _claude_session_path(session_id: str, cwd: Optional[str]) -> Path:
    """Locate one session's main JSONL file: the cwd-derived path first,
    falling back to a project-tree-wide search by filename when `cwd` is
    absent, unknown, or the derived path doesn't identify this session.
    Raises ``TranscriptSourceError`` (``store_missing`` if no candidate
    exists anywhere, ``store_unreadable`` for a read failure, a content
    conflict, or an unresolved ambiguity between multiple candidates)."""
    _validate_session_id(session_id)
    if cwd:
        candidate = CLAUDE_PROJECTS_DIR / _mangle_cwd(cwd) / f"{session_id}.jsonl"
        try:
            candidate_is_file = _path_is_file(candidate)
        except OSError as exc:
            raise TranscriptSourceError(
                f"cannot check claude session candidate {candidate}: {exc}", code="store_unreadable"
            ) from exc
        if candidate_is_file:
            _matches, conflicts = _claude_classify_file_for_session(candidate, session_id)
            if conflicts and not _matches:
                raise TranscriptSourceError(
                    f"claude session candidate {candidate} holds records for a different "
                    f"session, not {session_id!r}",
                    code="store_unreadable",
                )
            return candidate
    try:
        matches = _glob_project_session_files(CLAUDE_PROJECTS_DIR, f"{session_id}.jsonl")
    except OSError as exc:
        raise TranscriptSourceError(
            f"cannot search for claude session transcript under {CLAUDE_PROJECTS_DIR}: {exc}",
            code="store_unreadable",
        ) from exc
    if not matches:
        raise TranscriptSourceError(
            f"no claude session transcript found for session {session_id!r} under {CLAUDE_PROJECTS_DIR}",
            code="store_missing",
        )
    verified: list[Path] = []
    unreadable: list[Path] = []
    for path in matches:
        try:
            matches_session, _ = _claude_classify_file_for_session(path, session_id)
            if matches_session:
                verified.append(path)
        except TranscriptSourceError:
            unreadable.append(path)
    if not verified:
        if unreadable:
            joined = ", ".join(str(path) for path in unreadable)
            raise TranscriptSourceError(
                f"cannot verify claude session transcript candidate(s) for session {session_id!r}: {joined}",
                code="store_unreadable",
            )
        raise TranscriptSourceError(
            f"no claude session transcript found for session {session_id!r} under {CLAUDE_PROJECTS_DIR}",
            code="store_missing",
        )
    if len(verified) > 1:
        joined = ", ".join(str(path) for path in verified)
        raise TranscriptSourceError(
            f"ambiguous claude session transcript for session {session_id!r}: "
            f"{len(verified)} candidates carry matching records: {joined}",
            code="store_unreadable",
        )
    return verified[0]


def _claude_subagent_files(main_path: Path) -> list[Path]:
    """Return subagent JSONL files; an unreadable directory is an error."""
    subagents_dir = main_path.parent / main_path.stem / "subagents"
    try:
        subagents_present = _path_is_dir(subagents_dir)
    except OSError as exc:
        raise TranscriptSourceError(
            f"cannot check claude subagent directory {subagents_dir}: {exc}", code="store_unreadable"
        ) from exc
    if not subagents_present:
        return []
    try:
        return [subagents_dir / name for name in _glob_one_level(subagents_dir, "*.jsonl")]
    except OSError as exc:
        raise TranscriptSourceError(
            f"cannot list claude subagent directory {subagents_dir}: {exc}", code="store_unreadable"
        ) from exc


def _read_claude(session_id: str, cwd: Optional[str]) -> "tuple[list[TranscriptEntry], int]":
    path = _claude_session_path(session_id, cwd)
    main_entries, skipped = _read_claude_jsonl(path, session_id, subagent=None)
    sources = [main_entries]

    for sub_path in _claude_subagent_files(path):
        sub_entries, sub_skipped = _read_claude_jsonl(sub_path, session_id, subagent=sub_path.stem)
        sources.append(sub_entries)
        skipped += sub_skipped
    return _merge_claude_entries(sources), skipped


@dataclass
class _CountStats:
    entries: int = 0
    skipped: int = 0
    mismatched: int = 0
    newest: Optional[datetime] = None

    def add(self, timestamp: Optional[str]) -> None:
        self.entries += 1
        parsed = _parse_iso_timestamp(timestamp)
        if parsed is not None and (self.newest is None or parsed > self.newest):
            self.newest = parsed

    def merge(self, other: "_CountStats") -> None:
        self.entries += other.entries
        self.skipped += other.skipped
        self.mismatched += other.mismatched
        if other.newest is not None and (self.newest is None or other.newest > self.newest):
            self.newest = other.newest


def _count_tool_result(
    call_id: Any,
    output_text: Optional[str],
    pending_ids: set[str],
    stats: _CountStats,
    timestamp: Optional[str],
) -> None:
    if isinstance(call_id, str) and call_id in pending_ids:
        pending_ids.remove(call_id)
    elif output_text:
        stats.add(timestamp)


def _count_claude_jsonl(path: Path, session_id: str) -> _CountStats:
    """Count one Claude file, retaining session mismatches separately."""
    skip_counter = _JsonlSkipCounter()
    stats = _CountStats()
    pending_tool_use_ids: set[str] = set()
    for record in _iter_jsonl_objects(path, skip_counter):
        rtype = record.get("type")
        if rtype not in ("user", "assistant"):
            continue
        record_session_id = record.get("sessionId")
        if record_session_id is not None and record_session_id != session_id:
            stats.mismatched += 1
            continue
        message = record.get("message")
        if not isinstance(message, dict):
            skip_counter.skipped += 1
            continue
        role = message.get("role")
        timestamp = record.get("timestamp") if isinstance(record.get("timestamp"), str) else None
        content = message.get("content")

        if isinstance(content, str):
            if content and role in ("user", "assistant"):
                stats.add(timestamp)
            continue
        if not isinstance(content, list):
            continue

        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                text = block.get("text")
                if isinstance(text, str) and text and role in ("user", "assistant"):
                    stats.add(timestamp)
            elif btype == "tool_use":
                stats.add(timestamp)
                tool_use_id = block.get("id")
                if isinstance(tool_use_id, str):
                    pending_tool_use_ids.add(tool_use_id)
            elif btype == "tool_result":
                tool_use_id = block.get("tool_use_id")
                output_text = _claude_tool_result_text(block.get("content"))
                _count_tool_result(tool_use_id, output_text, pending_tool_use_ids, stats, timestamp)
    stats.skipped = skip_counter.skipped
    return stats


def _count_claude(session_id: str, cwd: Optional[str]) -> tuple[int, Optional[str]]:
    """Count Claude files; skipped or mismatched-only input is unreadable."""
    path = _claude_session_path(session_id, cwd)
    total = _CountStats()
    total.merge(_count_claude_jsonl(path, session_id))
    for source in _claude_subagent_files(path):
        total.merge(_count_claude_jsonl(source, session_id))

    if total.entries == 0 and (total.skipped > 0 or total.mismatched > 0):
        raise TranscriptSourceError(
            f"claude session transcript for {session_id!r} at {path} is unreadable: "
            f"{total.skipped} record(s) unparseable, {total.mismatched} record(s) named a "
            f"different session, 0 counted for this session",
            code="store_unreadable",
        )
    return total.entries, _iso_utc_z(total.newest) if total.newest is not None else None


def _claude_user_input_text(record: dict, session_id: str) -> Optional[str]:
    """The text of one claude JSONL record when it is a user turn, else None.

    A `tool_result` block also arrives as a ``type: "user"`` record with
    ``role: "user"``; only text content counts as input the user supplied.
    """
    if record.get("type") != "user":
        return None
    record_session_id = record.get("sessionId")
    if record_session_id is not None and record_session_id != session_id:
        return None
    message = record.get("message")
    if not isinstance(message, dict) or message.get("role") != "user":
        return None
    content = message.get("content")
    if isinstance(content, str):
        return content or None
    if not isinstance(content, list):
        return None
    texts = [
        block["text"] for block in content
        if isinstance(block, dict) and block.get("type") == "text"
        and isinstance(block.get("text"), str) and block["text"]
    ]
    return "\n".join(texts) if texts else None


def _count_claude_prompt_turns(session_id: str, cwd: Optional[str], normalized_prompt: str) -> int:
    """User turns in a claude session carrying the prompt, main file only.

    Subagent files are skipped: they carry the parent agent's prompts to a
    subagent, not input delivered to this session. Streaming the JSONL and
    testing each record beats a full `_read_claude` parse, which also builds
    and merges every assistant/tool entry.

    Any skipped record makes the answer unknown rather than low: a file
    caught mid-append has an unterminated trailing line, and that line may be
    the very record carrying this prompt.
    """
    path = _claude_session_path(session_id, cwd)
    skip_counter = _JsonlSkipCounter()
    count = 0
    for record in _iter_jsonl_objects(path, skip_counter):
        if turn_carries_prompt(_claude_user_input_text(record, session_id), normalized_prompt):
            count += 1
    if skip_counter.skipped > 0:
        raise TranscriptSourceError(
            f"claude session transcript for {session_id!r} at {path} is unreadable: "
            f"{skip_counter.skipped} record(s) unparseable, {count} prompt turn(s) counted "
            f"among the readable ones",
            code="store_unreadable",
        )
    return count


def _parse_iso_timestamp(value: Optional[str]) -> Optional[datetime]:
    """Parse a record timestamp (claude or codex; both store ISO-8601
    strings) to an aware UTC datetime.

    A store quirk can leave one record's timestamp missing its offset
    while a neighbour's carries `Z`; `datetime.fromisoformat` happily
    returns a naive datetime for the former, and comparing that against an
    aware one raises TypeError during a merge sort. A naive result is
    assumed UTC (every claude/codex timestamp seen in practice is UTC, `Z`
    suffix or not) and given that tzinfo explicitly rather than discarded,
    since it still carries real ordering information.
    """
    if not isinstance(value, str) or not value:
        return None
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _merge_claude_entries(sources: list[list[TranscriptEntry]]) -> list[TranscriptEntry]:
    """Interleave the main stream and every subagent stream by parsed
    timestamp instead of concatenating them: appending each subagent file
    after the whole main stream makes every entry in the default
    `--tail 50` window subagent output whenever a subagent ran long.

    An entry with no parseable timestamp of its own inherits its nearest
    timestamped neighbour's sort key within the *same* file, measured in
    entry positions to the nearest preceding and nearest following
    timestamped entry; a tie is broken toward the preceding entry. This
    keeps the entry next to whichever neighbour it actually followed there
    rather than drifting to another file's part of the merge, while never
    sorting before an entry it inherited backward from or after one it
    inherited forward from. A file with no timestamps anywhere keeps its
    entries' relative order, sorted after every timestamped entry.
    """
    keyed: list[tuple[Optional[datetime], int, int, TranscriptEntry]] = []
    for file_index, entries in enumerate(sources):
        own = [_parse_iso_timestamp(entry.time) for entry in entries]
        n = len(own)

        preceding_time: list[Optional[datetime]] = [None] * n
        preceding_distance: list[Optional[int]] = [None] * n
        last_time: Optional[datetime] = None
        last_distance: Optional[int] = None
        for i in range(n):
            if own[i] is not None:
                last_time, last_distance = own[i], 0
            elif last_time is not None:
                last_distance += 1
            preceding_time[i], preceding_distance[i] = last_time, last_distance

        following_time: list[Optional[datetime]] = [None] * n
        following_distance: list[Optional[int]] = [None] * n
        next_time: Optional[datetime] = None
        next_distance: Optional[int] = None
        for i in range(n - 1, -1, -1):
            if own[i] is not None:
                next_time, next_distance = own[i], 0
            elif next_time is not None:
                next_distance += 1
            following_time[i], following_distance[i] = next_time, next_distance

        sort_time: list[Optional[datetime]] = []
        for i in range(n):
            if own[i] is not None:
                sort_time.append(own[i])
                continue
            pt, pd = preceding_time[i], preceding_distance[i]
            nt, nd = following_time[i], following_distance[i]
            if pt is not None and nt is not None:
                sort_time.append(pt if pd <= nd else nt)
            elif pt is not None:
                sort_time.append(pt)
            else:
                sort_time.append(nt)

        for position, (entry, when) in enumerate(zip(entries, sort_time)):
            keyed.append((when, file_index, position, entry))
    keyed.sort(key=lambda item: (item[0] is None, item[0] or datetime.min, item[1], item[2]))
    return [item[3] for item in keyed]


def _claude_tool_result_text(content: Any) -> Optional[str]:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            block.get("text")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str)
        ]
        return "\n".join(parts) if parts else None
    return None


class _JsonlSkipCounter:
    __slots__ = ("skipped",)

    def __init__(self) -> None:
        self.skipped = 0


def _iter_jsonl_objects(path: Path, counter: _JsonlSkipCounter) -> Iterator[dict]:
    """Stream JSON objects, counting malformed lines and wrapping read errors."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    counter.skipped += 1
                    continue
                if isinstance(record, dict):
                    yield record
                else:
                    counter.skipped += 1
    except OSError as exc:
        raise TranscriptSourceError(f"cannot read {path}: {exc}", code="store_unreadable") from exc


def _read_jsonl_objects(path: Path) -> tuple[list[dict], int]:
    counter = _JsonlSkipCounter()
    records = list(_iter_jsonl_objects(path, counter))
    return records, counter.skipped


def _read_claude_jsonl(
    path: Path, session_id: str, subagent: Optional[str]
) -> tuple[list[TranscriptEntry], int]:
    """Parse one claude project JSONL file.

    ``pending`` matches a `tool_use` block to the `tool_result` block that
    arrives in a later `user` record (claude reports tool output as a
    synthetic follow-up user turn), so the two collapse into one
    TranscriptEntry the same way opencode's single `tool` part does.
    """
    records, skipped = _read_jsonl_objects(path)
    entries: list[TranscriptEntry] = []
    pending: dict[str, TranscriptEntry] = {}
    for record in records:
        rtype = record.get("type")
        if rtype not in ("user", "assistant"):
            continue  # attachment/queue-operation/mode/permission-mode/etc: not conversation
        record_session_id = record.get("sessionId")
        if record_session_id is not None and record_session_id != session_id:
            continue  # a record naming a different session is not part of this transcript
        message = record.get("message")
        if not isinstance(message, dict):
            skipped += 1
            continue
        role = message.get("role")
        timestamp = record.get("timestamp") if isinstance(record.get("timestamp"), str) else None
        content = message.get("content")

        if isinstance(content, str):
            if content and role in ("user", "assistant"):
                entries.append(
                    TranscriptEntry(type=role, harness="claude", time=timestamp, text=content, subagent=subagent)
                )
            continue
        if not isinstance(content, list):
            continue

        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                text = block.get("text")
                if isinstance(text, str) and text and role in ("user", "assistant"):
                    entries.append(
                        TranscriptEntry(
                            type=role, harness="claude", time=timestamp, text=text, subagent=subagent
                        )
                    )
            elif btype == "tool_use":
                name = block.get("name")
                tool_input = block.get("input") if isinstance(block.get("input"), dict) else None
                summary = None
                if name == "Bash" and isinstance(tool_input, dict):
                    summary = tool_input.get("description") or tool_input.get("command")
                entry = TranscriptEntry(
                    type="tool",
                    harness="claude",
                    time=timestamp,
                    tool_name=name if isinstance(name, str) else None,
                    tool_summary=summary if isinstance(summary, str) else None,
                    tool_input=tool_input,
                    subagent=subagent,
                )
                entries.append(entry)
                tool_use_id = block.get("id")
                if isinstance(tool_use_id, str):
                    pending[tool_use_id] = entry
            elif btype == "tool_result":
                tool_use_id = block.get("tool_use_id")
                output_text = _claude_tool_result_text(block.get("content"))
                target = pending.pop(tool_use_id, None) if isinstance(tool_use_id, str) else None
                if target is not None:
                    target.tool_output = output_text
                    target.tool_status = "completed"
                elif output_text:
                    # A result with no matching call in this file (e.g. the
                    # call was in an earlier, unparsed record): still worth
                    # showing rather than silently dropping the output.
                    entries.append(
                        TranscriptEntry(
                            type="tool",
                            harness="claude",
                            time=timestamp,
                            tool_name=None,
                            tool_output=output_text,
                            subagent=subagent,
                        )
                    )
    return entries, skipped


# ---------------------------------------------------------------------------
# codex: JSONL at ~/.codex/sessions/YYYY/MM/DD/rollout-<timestamp>-<uuid>.jsonl
# ---------------------------------------------------------------------------

def _codex_message_text(content: Any) -> Optional[str]:
    if not isinstance(content, list):
        return None
    parts = [
        block.get("text")
        for block in content
        if isinstance(block, dict)
        and block.get("type") in ("input_text", "output_text")
        and isinstance(block.get("text"), str)
    ]
    return "\n".join(parts) if parts else None


def _codex_reasoning_text(summary: Any) -> Optional[str]:
    if not isinstance(summary, list):
        return None
    parts = []
    for item in summary:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict) and isinstance(item.get("text"), str):
            parts.append(item["text"])
    text = "\n".join(p for p in parts if p)
    return text or None


def _codex_session_path(session_id: str) -> Path:
    """Locate one session's rollout file under the fixed
    `YYYY/MM/DD/rollout-*-<session_id>.jsonl` layout: three directory
    levels, not a recursive `**`, but every year/month/day branch present
    on disk is still scanned -- discovery cost is bounded by tree depth,
    not by how much session history exists. A search failure (permission-
    denied component, I/O error) is a read failure, not evidence the file
    is absent."""
    _validate_session_id(session_id)
    try:
        matches = _glob_codex_rollout_files(CODEX_SESSIONS_DIR, session_id)
    except OSError as exc:
        raise TranscriptSourceError(
            f"cannot search for codex session transcript under {CODEX_SESSIONS_DIR}: {exc}",
            code="store_unreadable",
        ) from exc
    if not matches:
        raise TranscriptSourceError(
            f"no codex session transcript found for session {session_id!r} under {CODEX_SESSIONS_DIR}",
            code="store_missing",
        )
    return matches[0]


def _read_codex(session_id: str) -> tuple[list[TranscriptEntry], int]:
    """Read one codex rollout file.

    The reasoning `payload.summary` is near-always empty in practice --
    codex's actual reasoning trace is carried in `encrypted_content`, which
    this reader has no key to decode -- so reasoning entries are rare;
    `event_msg` records are skipped entirely because they duplicate the
    `response_item` message/reasoning content this reader already reads.
    """
    path = _codex_session_path(session_id)
    records, skipped = _read_jsonl_objects(path)
    entries: list[TranscriptEntry] = []
    pending: dict[str, TranscriptEntry] = {}
    for record in records:
        if record.get("type") != "response_item":
            continue  # session_meta/event_msg/world_state/turn_context: not conversation
        payload = record.get("payload")
        if not isinstance(payload, dict):
            skipped += 1
            continue
        timestamp = record.get("timestamp") if isinstance(record.get("timestamp"), str) else None
        ptype = payload.get("type")

        if ptype == "message":
            role = payload.get("role")
            if role not in ("user", "assistant"):
                continue  # "developer" role: injected environment/permissions text, not conversation
            text = _codex_message_text(payload.get("content"))
            if text and not _is_codex_injected_context(text):
                entries.append(TranscriptEntry(type=role, harness="codex", time=timestamp, text=text))
        elif ptype == "reasoning":
            text = _codex_reasoning_text(payload.get("summary"))
            if text:
                entries.append(TranscriptEntry(type="reasoning", harness="codex", time=timestamp, text=text))
        elif ptype == "function_call":
            name = payload.get("name")
            args_raw = payload.get("arguments")
            tool_input = None
            if isinstance(args_raw, str):
                try:
                    parsed_args = json.loads(args_raw)
                except json.JSONDecodeError:
                    parsed_args = None
                tool_input = parsed_args if isinstance(parsed_args, dict) else None
            summary = None
            if isinstance(tool_input, dict):
                summary = tool_input.get("cmd") or tool_input.get("command")
            entry = TranscriptEntry(
                type="tool",
                harness="codex",
                time=timestamp,
                tool_name=name if isinstance(name, str) else None,
                tool_summary=summary if isinstance(summary, str) else None,
                tool_input=tool_input,
            )
            entries.append(entry)
            call_id = payload.get("call_id")
            if isinstance(call_id, str):
                pending[call_id] = entry
        elif ptype == "function_call_output":
            call_id = payload.get("call_id")
            output = payload.get("output")
            output_text = output if isinstance(output, str) else None
            target = pending.pop(call_id, None) if isinstance(call_id, str) else None
            if target is not None:
                target.tool_output = output_text
                target.tool_status = "completed"
            elif output_text:
                entries.append(
                    TranscriptEntry(type="tool", harness="codex", time=timestamp, tool_name=None, tool_output=output_text)
                )
    return entries, skipped


def _count_codex(session_id: str) -> tuple[int, Optional[str]]:
    """Count a Codex rollout; skipped-only input is unreadable."""
    path = _codex_session_path(session_id)
    skip_counter = _JsonlSkipCounter()
    stats = _CountStats()
    pending_call_ids: set[str] = set()
    for record in _iter_jsonl_objects(path, skip_counter):
        if record.get("type") != "response_item":
            continue
        payload = record.get("payload")
        if not isinstance(payload, dict):
            skip_counter.skipped += 1
            continue
        timestamp = record.get("timestamp") if isinstance(record.get("timestamp"), str) else None
        ptype = payload.get("type")

        if ptype == "message":
            role = payload.get("role")
            if role not in ("user", "assistant"):
                continue
            text = _codex_message_text(payload.get("content"))
            if text and not _is_codex_injected_context(text):
                stats.add(timestamp)
        elif ptype == "reasoning":
            text = _codex_reasoning_text(payload.get("summary"))
            if text:
                stats.add(timestamp)
        elif ptype == "function_call":
            stats.add(timestamp)
            call_id = payload.get("call_id")
            if isinstance(call_id, str):
                pending_call_ids.add(call_id)
        elif ptype == "function_call_output":
            call_id = payload.get("call_id")
            output = payload.get("output")
            output_text = output if isinstance(output, str) else None
            _count_tool_result(call_id, output_text, pending_call_ids, stats, timestamp)

    stats.skipped = skip_counter.skipped
    if stats.skipped > 0 and stats.entries == 0:
        raise TranscriptSourceError(
            f"codex session transcript for {session_id!r} at {path} is unreadable: "
            f"{stats.skipped} record(s) unparseable, 0 counted",
            code="store_unreadable",
        )
    return stats.entries, _iso_utc_z(stats.newest) if stats.newest is not None else None


def _count_codex_prompt_turns(session_id: str, normalized_prompt: str) -> int:
    """User turns in a codex rollout carrying the prompt.

    Uses the same `_is_codex_injected_context` filter as `_count_codex`, so
    the harness's own `<environment_context>`/`<user_instructions>` preamble
    -- written at thread start, before any prompt is submitted -- is not
    mistaken for a delivered user turn.

    Any skipped record makes the answer unknown rather than low: a rollout
    caught mid-append has an unterminated trailing line, and that line may be
    the very record carrying this prompt.
    """
    path = _codex_session_path(session_id)
    skip_counter = _JsonlSkipCounter()
    count = 0
    for record in _iter_jsonl_objects(path, skip_counter):
        if record.get("type") != "response_item":
            continue
        payload = record.get("payload")
        if not isinstance(payload, dict) or payload.get("type") != "message":
            continue
        if payload.get("role") != "user":
            continue
        text = _codex_message_text(payload.get("content"))
        if text and not _is_codex_injected_context(text) and turn_carries_prompt(text, normalized_prompt):
            count += 1
    if skip_counter.skipped > 0:
        raise TranscriptSourceError(
            f"codex session transcript for {session_id!r} at {path} is unreadable: "
            f"{skip_counter.skipped} record(s) unparseable, {count} prompt turn(s) counted "
            f"among the readable ones",
            code="store_unreadable",
        )
    return count


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

# Wrapper elements codex prepends to a session's first user turn, carrying cwd,
# shell and permission profile rather than anything the user typed. Paired
# as (opening tag, closing tag); a session's first turn can carry more than
# one of these back to back in a single message.
_CODEX_INJECTED_CONTEXT_TAGS = (
    ("<environment_context>", "</environment_context>"),
    ("<user_instructions>", "</user_instructions>"),
)


def _is_codex_injected_context(text: str) -> bool:
    """True when a codex user message is wholly one or more harness-injected
    wrappers, not input the user typed. A real first turn can carry
    `<environment_context>` immediately followed by `<user_instructions>`
    in the same message, so this repeatedly peels a leading complete
    wrapper -- remaining text starts with a recognised opening tag and
    contains that tag's closing counterpart -- off the front, discarding
    the block and any whitespace after it, until nothing recognised
    remains at the front. Suppressed only when that process consumes the
    entire message; a wrapper followed by genuine text, or text that
    merely starts with, quotes, or mentions a tag without closing it,
    leaves something behind and survives."""
    stripped = text.strip()
    cursor = 0
    end = len(stripped)
    consumed_any = False
    progressed = True
    # An integer cursor instead of slicing: re-slicing the suffix per wrapper
    # makes a long run of wrappers quadratic in total bytes.
    while progressed and cursor < end:
        progressed = False
        for open_tag, close_tag in _CODEX_INJECTED_CONTEXT_TAGS:
            if stripped.startswith(open_tag, cursor):
                close_index = stripped.find(close_tag, cursor)
                if close_index != -1:
                    cursor = close_index + len(close_tag)
                    while cursor < end and stripped[cursor].isspace():
                        cursor += 1
                    consumed_any = True
                    progressed = True
                    break
    return consumed_any and cursor >= end


def _fallback_tool_summary(tool_input: Optional[dict]) -> Optional[str]:
    if not tool_input:
        return None
    for key in ("command", "cmd", "filePath", "file_path", "path", "pattern"):
        value = tool_input.get(key)
        if isinstance(value, str):
            return value
    rendered = json.dumps(tool_input, sort_keys=True)
    return rendered if len(rendered) <= 200 else rendered[:200] + "..."


def render_text(entries: list[TranscriptEntry]) -> str:
    """Render entries in reader order as human- and agent-readable plain text:
    who spoke, what they said, which tools ran with their arguments, and
    tool results."""
    lines: list[str] = []
    for entry in entries:
        prefix = f"[{entry.time}] " if entry.time else ""
        if entry.subagent:
            prefix += f"(subagent {entry.subagent}) "

        if entry.type in ("user", "assistant"):
            lines.append(f"{prefix}{entry.type}: {entry.text or ''}")
        elif entry.type == "reasoning":
            lines.append(f"{prefix}assistant (reasoning): {entry.text or ''}")
        elif entry.type == "tool":
            name = entry.tool_name or "tool"
            summary = entry.tool_summary or _fallback_tool_summary(entry.tool_input)
            header = f"{prefix}tool {name}: {summary}" if summary else f"{prefix}tool {name}"
            if entry.tool_status and entry.tool_status not in ("completed", "success"):
                header += f" [{entry.tool_status}]"
            lines.append(header)
            if entry.tool_output:
                for output_line in entry.tool_output.splitlines():
                    lines.append(f"    {output_line}")
        else:
            lines.append(f"{prefix}{entry.type}: {entry.text or ''}")
    return "".join(line + "\n" for line in lines)
