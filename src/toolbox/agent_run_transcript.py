#!/usr/bin/env python3
"""Read harness conversation stores for ``agent-run transcript``.

Each reader returns normalized entries plus the number of malformed records
it skipped. Missing, unreadable, or incompatible stores raise
``TranscriptSourceError``. Readers never modify their stores.
"""
from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


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
    """The harness store for a session could not be located or opened at
    all -- as opposed to an individual unparseable record, which readers
    skip and count instead of raising."""


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
        raise TranscriptSourceError(f"unrecognised harness {harness!r} in session.json")
    for entry in entries:
        if entry.tool_output:
            entry.tool_output = _bound_tool_output(entry.tool_output)
    return entries, skipped


def _iso_from_epoch_ms(value: Any) -> Optional[str]:
    if not isinstance(value, (int, float)):
        return None
    try:
        return datetime.fromtimestamp(value / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except (OverflowError, OSError, ValueError):
        return None


# ---------------------------------------------------------------------------
# opencode: SQLite at ~/.local/share/opencode/opencode.db
# ---------------------------------------------------------------------------

def _read_opencode(session_id: str) -> tuple[list[TranscriptEntry], int]:
    """Read the `part`/`message` tables for one session.

    Opened read-only (`file:...?mode=ro`, `uri=True`): a live opencode run
    holds this database open under WAL, and this reader must never write to
    it or trigger a checkpoint. `part.data` carries the event; the joined
    `message.data` supplies the role (user/assistant) and fallback
    timestamp a `text`/`reasoning` part does not itself carry.

    `timeout=0` plus an explicit `PRAGMA busy_timeout=0` (the latter for
    SQLite builds where the connect-time timeout does not bind the busy
    handler) make a held write lock fail immediately instead of blocking
    for the default 5s: `mode=ro` stops this reader from writing, but it
    does not make lock acquisition non-blocking.
    """
    if not OPENCODE_DB_PATH.exists():
        raise TranscriptSourceError(f"opencode session store not found at {OPENCODE_DB_PATH}")
    try:
        conn = sqlite3.connect(f"file:{OPENCODE_DB_PATH}?mode=ro", uri=True, timeout=0)
        conn.execute("pragma busy_timeout=0")
    except sqlite3.OperationalError as exc:
        raise TranscriptSourceError(f"cannot open opencode session store at {OPENCODE_DB_PATH}: {exc}") from exc

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
    except sqlite3.OperationalError as exc:
        raise TranscriptSourceError(
            f"opencode session store at {OPENCODE_DB_PATH} is locked by another writer: {exc}"
        ) from exc
    except sqlite3.DatabaseError as exc:
        # A file that opens but isn't a valid/complete sqlite database (mid-
        # write copy, truncated backup): the store as a whole is unusable,
        # not one bad row -- surface it as a source error rather than
        # silently returning an empty transcript.
        raise TranscriptSourceError(f"opencode session store unreadable: {exc}") from exc
    finally:
        conn.close()
    return entries, skipped


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


def _claude_file_matches_session(path: Path, session_id: str) -> bool:
    """True if `path` holds at least one user/assistant record whose own
    `sessionId` equals `session_id` -- the only basis for trusting a
    filename match, since two project directories can each hold a
    `<session_id>.jsonl` left by unrelated runs."""
    try:
        records, _ = _read_jsonl_objects(path)
    except TranscriptSourceError:
        return False
    return any(
        record.get("type") in ("user", "assistant") and record.get("sessionId") == session_id
        for record in records
    )


def _claude_session_path(session_id: str, cwd: Optional[str]) -> Path:
    if cwd:
        candidate = CLAUDE_PROJECTS_DIR / _mangle_cwd(cwd) / f"{session_id}.jsonl"
        if candidate.is_file():
            return candidate
    try:
        matches = sorted(CLAUDE_PROJECTS_DIR.glob(f"*/{session_id}.jsonl"))
    except OSError:
        matches = []
    if not matches:
        raise TranscriptSourceError(
            f"no claude session transcript found for session {session_id!r} under {CLAUDE_PROJECTS_DIR}"
        )
    # The cwd-derived path missed (wrong or unknown cwd), so the filename
    # alone is the only lead -- and a filename collision across project
    # directories is possible. Content, not pathname order, decides.
    verified = [path for path in matches if _claude_file_matches_session(path, session_id)]
    if not verified:
        raise TranscriptSourceError(
            f"no claude session transcript found for session {session_id!r} under {CLAUDE_PROJECTS_DIR}"
        )
    if len(verified) > 1:
        joined = ", ".join(str(path) for path in verified)
        raise TranscriptSourceError(
            f"ambiguous claude session transcript for session {session_id!r}: "
            f"{len(verified)} candidates carry matching records: {joined}"
        )
    return verified[0]


def _read_claude(session_id: str, cwd: Optional[str]) -> "tuple[list[TranscriptEntry], int]":
    path = _claude_session_path(session_id, cwd)
    main_entries, skipped = _read_claude_jsonl(path, session_id, subagent=None)
    sources = [main_entries]

    # Subagent transcripts live in <session>/subagents/*.jsonl next to the
    # main .jsonl file. Included only when each record's own sessionId
    # matches the session we were asked for -- not inferred from directory
    # placement alone.
    subagents_dir = path.parent / path.stem / "subagents"
    try:
        sub_files = sorted(subagents_dir.glob("*.jsonl")) if subagents_dir.is_dir() else []
    except OSError:
        sub_files = []
    for sub_path in sub_files:
        sub_entries, sub_skipped = _read_claude_jsonl(sub_path, session_id, subagent=sub_path.stem)
        sources.append(sub_entries)
        skipped += sub_skipped
    return _merge_claude_entries(sources), skipped


def _parse_claude_timestamp(value: Optional[str]) -> Optional[datetime]:
    """Parse a claude record timestamp to an aware UTC datetime.

    A store quirk can leave one record's timestamp missing its offset
    while a neighbour's carries `Z`; `datetime.fromisoformat` happily
    returns a naive datetime for the former, and comparing that against an
    aware one raises TypeError during the merge sort. A naive result is
    assumed UTC (every claude timestamp this reader has seen is UTC, `Z`
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
        own = [_parse_claude_timestamp(entry.time) for entry in entries]
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


def _read_jsonl_objects(path: Path) -> tuple[list[dict], int]:
    """Parse one JSONL file into (records, skipped_count).

    Iterates the file line by line rather than reading the whole text and
    splitting it into a list at once: the two would otherwise coexist in
    memory, roughly doubling peak usage for a large transcript.
    """
    records = []
    skipped = 0
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1
                    continue
                if isinstance(record, dict):
                    records.append(record)
                else:
                    skipped += 1
    except OSError as exc:
        raise TranscriptSourceError(f"cannot read {path}: {exc}") from exc
    return records, skipped


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


def _read_codex(session_id: str) -> tuple[list[TranscriptEntry], int]:
    """Read one codex rollout file.

    The reasoning `payload.summary` is near-always empty in practice --
    codex's actual reasoning trace is carried in `encrypted_content`, which
    this reader has no key to decode -- so reasoning entries are rare;
    `event_msg` records are skipped entirely because they duplicate the
    `response_item` message/reasoning content this reader already reads.
    """
    try:
        matches = sorted(CODEX_SESSIONS_DIR.glob(f"**/rollout-*-{session_id}.jsonl"))
    except OSError:
        matches = []
    if not matches:
        raise TranscriptSourceError(
            f"no codex session transcript found for session {session_id!r} under {CODEX_SESSIONS_DIR}"
        )
    path = matches[0]
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
    remaining = text.strip()
    consumed_any = False
    progressed = True
    while progressed:
        progressed = False
        for open_tag, close_tag in _CODEX_INJECTED_CONTEXT_TAGS:
            if remaining.startswith(open_tag):
                close_index = remaining.find(close_tag)
                if close_index != -1:
                    remaining = remaining[close_index + len(close_tag):].lstrip()
                    consumed_any = True
                    progressed = True
                    break
    return consumed_any and not remaining


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
