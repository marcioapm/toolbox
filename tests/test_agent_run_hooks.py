"""Tests for `agent-run hook` — harness hook ingestion.

Covers run resolution (--name / AGENT_RUN_NAME / process ancestry), the
always-exit-0 contract, payload parsing for all three harness shapes, bounded
stdin reads, atomic appends under concurrency, the line and event caps, the
`watch` hooks aggregate, and the runner's env export.

The hostile-input cases are load-bearing rather than paranoid: hooks.jsonl sits
in a directory the agent under observation can write, and `watch` must keep
reporting a live run as live no matter what is planted there.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import pytest

from toolbox import agent_run


SRC_DIR = str(Path(__file__).parents[1] / "src")
FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _fixture(name: str) -> dict:
    """Load a captured real harness hook payload from tests/fixtures/."""
    return json.loads((FIXTURES_DIR / name).read_text())


def _make_run_dirs(
    state_root: Path,
    log_root: Path,
    name: str,
    *,
    write_state: bool = True,
) -> tuple[Path, Path]:
    """Create minimal state + log dirs for a run."""
    sd = state_root / name
    ld = log_root / name
    ld.mkdir(parents=True, exist_ok=True)
    sd.mkdir(parents=True, exist_ok=True)
    if write_state:
        (sd / "status").write_text("running\n")
        (sd / "pid").write_text(f"{os.getpid()}\n")
        (sd / "pgid").write_text(f"{os.getpgid(0)}\n")
    return sd, ld


def _hook_args(
    event: str = "stop",
    *,
    name: Optional[str] = None,
    json_payload: Optional[str] = None,
    extra: Optional[list] = None,
) -> argparse.Namespace:
    return argparse.Namespace(
        event=event,
        name=name,
        json_payload=json_payload,
        extra=extra or [],
    )


def _read_hooks(log_dir: Path) -> list[dict]:
    """Parsed hooks.jsonl records, skipping unparseable lines."""
    path = log_dir / "hooks.jsonl"
    if not path.exists():
        return []
    records = []
    for line in path.read_bytes().split(b"\n"):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return records


def _hook_lines(log_dir: Path) -> list[bytes]:
    """Raw non-empty lines of hooks.jsonl, newline stripped."""
    raw = (log_dir / "hooks.jsonl").read_bytes()
    return [ln for ln in raw.split(b"\n") if ln.strip()]


def _watch_args(name: str, *, as_json: bool = True) -> argparse.Namespace:
    return argparse.Namespace(name=name, json=as_json, repo=None)


def _hook_with_stdin(payload: bytes, *, close_writer: bool) -> tuple[int, float]:
    """Run cmd_hook with *payload* on a pipe as stdin; return (rc, elapsed).

    When close_writer is False the write end stays open for the duration, which
    is the case a blocking read would hang on forever.
    """
    r_fd, w_fd = os.pipe()
    if payload:
        os.write(w_fd, payload)
    if close_writer:
        os.close(w_fd)
    old_stdin = sys.stdin
    sys.stdin = open(r_fd, "rb", closefd=True)
    try:
        start = time.monotonic()
        rc = agent_run.cmd_hook(_hook_args("stop"))
        elapsed = time.monotonic() - start
    finally:
        sys.stdin.close()
        if not close_writer:
            os.close(w_fd)
        sys.stdin = old_stdin
    return rc, elapsed


def _wait_for_terminal(status_file: Path, timeout: float = 10.0) -> str:
    """Poll status_file until it holds a terminal status; return it."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if status_file.exists():
            status = status_file.read_text().strip()
            if status in {"done", "failed", "launch_failed"}:
                return status
        time.sleep(0.1)
    return status_file.read_text().strip() if status_file.exists() else "unknown"


class TestHookResolution:
    def test_env_var_hit_writes_record(
        self, isolated_runs_root, isolated_log_root, monkeypatch,
    ):
        """AGENT_RUN_NAME naming a real run resolves and writes one record."""
        _, ld = _make_run_dirs(isolated_runs_root, isolated_log_root, "myrun")
        monkeypatch.setenv("AGENT_RUN_NAME", "myrun")

        rc = agent_run.cmd_hook(_hook_args("stop", json_payload='{"a":1}'))

        assert rc == 0
        records = _read_hooks(ld)
        assert len(records) == 1
        assert records[0]["event"] == "stop"
        assert records[0]["resolved_by"] == "env"

    def test_env_var_nonexistent_run_writes_nothing(
        self, isolated_runs_root, isolated_log_root, monkeypatch,
    ):
        """A stale AGENT_RUN_NAME resolves to nothing rather than to a
        neighbouring run via the ancestry fallback."""
        _, ld = _make_run_dirs(isolated_runs_root, isolated_log_root, "realrun")
        monkeypatch.setenv("AGENT_RUN_NAME", "ghost")

        rc = agent_run.cmd_hook(_hook_args("stop"))

        assert rc == 0
        assert not (isolated_log_root / "ghost" / "hooks.jsonl").exists()
        assert not _read_hooks(ld), "stale env name fell through to another run"

    def test_name_override_takes_priority(
        self, isolated_runs_root, isolated_log_root, monkeypatch,
    ):
        """--name overrides AGENT_RUN_NAME."""
        _make_run_dirs(isolated_runs_root, isolated_log_root, "target")
        _make_run_dirs(isolated_runs_root, isolated_log_root, "other")
        monkeypatch.setenv("AGENT_RUN_NAME", "other")

        rc = agent_run.cmd_hook(
            _hook_args("stop", name="target", json_payload='{"x":1}')
        )

        assert rc == 0
        assert _read_hooks(isolated_log_root / "target")
        assert not _read_hooks(isolated_log_root / "other")

    def test_nothing_resolvable_writes_nothing(
        self, isolated_runs_root, isolated_log_root, monkeypatch,
    ):
        """No env var and no ancestry match: exit 0, no hooks.jsonl anywhere."""
        monkeypatch.delenv("AGENT_RUN_NAME", raising=False)

        rc = agent_run.cmd_hook(_hook_args("stop", json_payload='{"x":1}'))

        assert rc == 0
        for d in isolated_log_root.iterdir():
            assert not (d / "hooks.jsonl").exists()

    def test_ancestry_fallback_via_pgid(
        self, isolated_runs_root, isolated_log_root, monkeypatch,
    ):
        """With no env var, a run whose pgid equals our sid is attributed."""
        sd, ld = _make_run_dirs(isolated_runs_root, isolated_log_root, "ancestorrun")
        (sd / "pgid").write_text(f"{os.getpgid(0)}\n")
        monkeypatch.delenv("AGENT_RUN_NAME", raising=False)

        rc = agent_run.cmd_hook(_hook_args("stop", json_payload='{"x":1}'))

        assert rc == 0
        records = _read_hooks(ld)
        assert len(records) == 1
        assert records[0]["resolved_by"] == "ancestry"

    def test_ancestry_direct_ppid_match_wins(
        self, isolated_runs_root, isolated_log_root, monkeypatch,
    ):
        """A run recording our direct ppid is attributed by 1-hop ancestry."""
        sd, ld = _make_run_dirs(isolated_runs_root, isolated_log_root, "ppidrun")
        (sd / "pid").write_text(f"{os.getppid()}\n")
        monkeypatch.delenv("AGENT_RUN_NAME", raising=False)

        rc = agent_run.cmd_hook(_hook_args("stop", json_payload='{"x":1}'))

        assert rc == 0
        records = _read_hooks(ld)
        assert len(records) == 1
        assert records[0]["resolved_by"] == "ancestry"

    def test_ancestry_ambiguous_sid_attributes_to_neither(
        self, isolated_runs_root, isolated_log_root, monkeypatch,
    ):
        """Two runs sharing a session id and neither an ancestor: no match.

        Guessing between them would file another agent's events under this run.
        """
        fake_sid = 888777666  # cannot collide with a live pid
        for i in range(2):
            sd, _ = _make_run_dirs(
                isolated_runs_root, isolated_log_root, f"ambig{i}",
                write_state=False,
            )
            (sd / "pgid").write_text(f"{fake_sid}\n")
            (sd / "pid").write_text("999999888\n")  # not an ancestor
        monkeypatch.delenv("AGENT_RUN_NAME", raising=False)
        monkeypatch.setattr(os, "getsid", lambda _: fake_sid)

        result = agent_run._hook_resolve_run(None)

        assert result is None or result[0] not in ("ambig0", "ambig1")

    def test_invalid_name_override_writes_nothing(
        self, isolated_runs_root, isolated_log_root, monkeypatch,
    ):
        """A --name with path separators is rejected, not used to build a path."""
        monkeypatch.delenv("AGENT_RUN_NAME", raising=False)

        rc = agent_run.cmd_hook(_hook_args("stop", name="../../etc/passwd"))

        assert rc == 0
        assert not (isolated_log_root.parent / "passwd").exists()


class TestExitCodeAlways0:
    """A claude Stop hook exiting non-zero reads as a decision to block, so
    every failure path must still return 0."""

    def test_unwritable_log_dir(
        self, isolated_runs_root, isolated_log_root, monkeypatch,
    ):
        _, ld = _make_run_dirs(isolated_runs_root, isolated_log_root, "nowrite")
        monkeypatch.setenv("AGENT_RUN_NAME", "nowrite")
        ld.chmod(0o555)
        try:
            rc = agent_run.cmd_hook(_hook_args("stop", json_payload='{"x":1}'))
        finally:
            ld.chmod(0o755)
        assert rc == 0

    def test_unresolvable_run(
        self, isolated_runs_root, isolated_log_root, monkeypatch,
    ):
        monkeypatch.delenv("AGENT_RUN_NAME", raising=False)
        assert agent_run.cmd_hook(_hook_args("stop")) == 0

    def test_unexpected_exception_is_swallowed(
        self, isolated_runs_root, isolated_log_root, monkeypatch,
    ):
        """An exception escaping the body still leaves the exit code at 0."""
        monkeypatch.setenv("AGENT_RUN_NAME", "boom")
        monkeypatch.setattr(
            agent_run, "_cmd_hook_inner",
            lambda _a: (_ for _ in ()).throw(RuntimeError("kaboom")),
        )
        assert agent_run.cmd_hook(_hook_args("stop")) == 0

    @pytest.mark.parametrize("argv", [
        pytest.param(["hook"], id="missing-event"),
        pytest.param(["hook", "stop", "--nope"], id="unknown-flag"),
        pytest.param(["hook", "stop", "--name"], id="flag-without-value"),
    ])
    def test_argv_error_exits_0(self, argv):
        """argparse exits 2 on bad argv; the hook path converts that to 0."""
        assert agent_run.main(argv) == 0

    def test_argv_error_keeps_stdout_empty(self, capsys):
        """argparse usage text must not reach stdout: codex warns on any
        output from a notify program."""
        agent_run.main(["hook"])
        assert capsys.readouterr().out == ""


class TestPayloadParsing:
    @pytest.mark.parametrize("fixture_name,expected_harness", [
        pytest.param("hook_claude_stop.json", "claude", id="claude"),
        pytest.param("hook_codex_turn_complete.json", "codex", id="codex"),
        pytest.param("hook_opencode_session_idle.json", "opencode", id="opencode"),
        pytest.param(None, None, id="unknown"),
    ])
    def test_harness_detected_from_payload_shape(
        self, isolated_runs_root, isolated_log_root, monkeypatch,
        fixture_name, expected_harness,
    ):
        """Driven through captured real per-harness payloads, not hand-built
        toy shapes — a fixture missing a real field would hide a detection
        bug the way the phantom opencode "session" key once did."""
        _, ld = _make_run_dirs(isolated_runs_root, isolated_log_root, "harness")
        monkeypatch.setenv("AGENT_RUN_NAME", "harness")
        payload = _fixture(fixture_name) if fixture_name else {"unrecognised": 1}

        agent_run.cmd_hook(_hook_args("stop", json_payload=json.dumps(payload)))

        assert _read_hooks(ld)[0]["harness"] == expected_harness

    def test_json_flag_source(
        self, isolated_runs_root, isolated_log_root, monkeypatch,
    ):
        """--json is the opencode plugin path; source='flag'."""
        _, ld = _make_run_dirs(isolated_runs_root, isolated_log_root, "flagrun")
        monkeypatch.setenv("AGENT_RUN_NAME", "flagrun")

        agent_run.cmd_hook(_hook_args("stop", json_payload='{"session_id":"abc"}'))

        assert _read_hooks(ld)[0]["source"] == "flag"

    def test_argv_json_source(
        self, isolated_runs_root, isolated_log_root, monkeypatch,
    ):
        """codex notify passes its JSON as argv[1]; source='argv'."""
        _, ld = _make_run_dirs(isolated_runs_root, isolated_log_root, "argvrun")
        monkeypatch.setenv("AGENT_RUN_NAME", "argvrun")
        codex_payload = json.dumps(_fixture("hook_codex_turn_complete.json"))

        rc = agent_run.cmd_hook(_hook_args("turn-complete", extra=[codex_payload]))

        assert rc == 0
        records = _read_hooks(ld)
        assert records[0]["source"] == "argv"
        assert records[0]["harness"] == "codex"

    def test_non_json_argv_is_not_treated_as_payload(
        self, isolated_runs_root, isolated_log_root, monkeypatch,
    ):
        """A positional that does not open a JSON object or array is ignored."""
        _, ld = _make_run_dirs(isolated_runs_root, isolated_log_root, "plainargv")
        monkeypatch.setenv("AGENT_RUN_NAME", "plainargv")

        agent_run.cmd_hook(_hook_args("stop", extra=["just a sentence"]))

        assert _read_hooks(ld)[0]["source"] != "argv"

    def test_stdin_json_source(
        self, isolated_runs_root, isolated_log_root, monkeypatch,
    ):
        """claude delivers its Stop payload on stdin; source='stdin'."""
        _, ld = _make_run_dirs(isolated_runs_root, isolated_log_root, "stdinrun")
        monkeypatch.setenv("AGENT_RUN_NAME", "stdinrun")
        claude_payload = json.dumps(_fixture("hook_claude_stop.json")).encode()

        rc, _ = _hook_with_stdin(claude_payload, close_writer=True)

        assert rc == 0
        records = _read_hooks(ld)
        assert records[0]["source"] == "stdin"
        assert records[0]["harness"] == "claude"

    def test_malformed_json_kept_as_raw_excerpt(
        self, isolated_runs_root, isolated_log_root, monkeypatch,
    ):
        """Unparseable input is recorded as a raw excerpt, not discarded."""
        _, ld = _make_run_dirs(isolated_runs_root, isolated_log_root, "badjson")
        monkeypatch.setenv("AGENT_RUN_NAME", "badjson")

        rc = agent_run.cmd_hook(_hook_args("stop", json_payload="not json {{{"))

        assert rc == 0
        records = _read_hooks(ld)
        assert len(records) == 1
        assert "raw" in records[0]["payload"]

    @pytest.mark.parametrize("given,recorded", [
        pytest.param("STOP", "stop", id="uppercase"),
        pytest.param("my-custom-event-XYZ", "my-custom-event-xyz", id="arbitrary"),
    ])
    def test_event_name_lowercased_and_unrestricted(
        self, isolated_runs_root, isolated_log_root, monkeypatch, given, recorded,
    ):
        """Event names are lowercased; there is no allowed-value list."""
        _, ld = _make_run_dirs(isolated_runs_root, isolated_log_root, "eventname")
        monkeypatch.setenv("AGENT_RUN_NAME", "eventname")

        agent_run.cmd_hook(_hook_args(given))

        assert _read_hooks(ld)[0]["event"] == recorded


class TestKindDerivation:
    """kind normalises event+payload across harnesses and user-renamed hook
    argv. Driven through cmd_hook with a misleading event string on every
    case, so payload evidence winning over the event argv is load-bearing,
    not incidental."""

    @pytest.mark.parametrize("fixture_name,event,expected_kind", [
        pytest.param("hook_claude_stop.json", "not-the-real-event-name",
                     "turn_complete", id="claude-turn-complete"),
        pytest.param("hook_claude_permission_request.json", "not-the-real-event-name",
                     "permission_required", id="claude-permission-required"),
        pytest.param("hook_claude_session_start.json", "not-the-real-event-name",
                     "session_start", id="claude-session-start"),
        pytest.param("hook_codex_turn_complete.json", "not-the-real-event-name",
                     "turn_complete", id="codex-turn-complete"),
        pytest.param("hook_codex_permission_request.json", "not-the-real-event-name",
                     "permission_required", id="codex-permission-required"),
        pytest.param("hook_codex_session_start.json", "not-the-real-event-name",
                     "session_start", id="codex-session-start"),
        pytest.param("hook_opencode_session_idle.json", "not-the-real-event-name",
                     "turn_complete", id="opencode-turn-complete"),
        pytest.param("hook_opencode_permission_asked.json", "not-the-real-event-name",
                     "permission_required", id="opencode-permission-asked"),
        pytest.param("hook_opencode_permission_replied.json", "not-the-real-event-name",
                     "other", id="opencode-permission-replied-is-not-pending"),
        pytest.param("hook_opencode_session_created.json", "not-the-real-event-name",
                     "session_start", id="opencode-session-start"),
    ])
    def test_kind_from_payload_evidence_beats_misleading_event(
        self, isolated_runs_root, isolated_log_root, monkeypatch,
        fixture_name, event, expected_kind,
    ):
        _, ld = _make_run_dirs(isolated_runs_root, isolated_log_root, "kindrun")
        monkeypatch.setenv("AGENT_RUN_NAME", "kindrun")
        payload = _fixture(fixture_name)

        agent_run.cmd_hook(_hook_args(event, json_payload=json.dumps(payload)))

        assert _read_hooks(ld)[0]["kind"] == expected_kind

    @pytest.mark.parametrize("event,expected_kind", [
        pytest.param("stop", "turn_complete", id="fallback-stop"),
        pytest.param("turn-complete", "turn_complete", id="fallback-turn-complete"),
        pytest.param("session-idle", "turn_complete", id="fallback-session-idle"),
        pytest.param("permission-asked", "permission_required", id="fallback-permission"),
        pytest.param("permission-request", "permission_required", id="fallback-permission-request"),
        pytest.param("session-start-permission-audit", "other",
                     id="fallback-permission-substring-is-not-a-request"),
        pytest.param("session-start", "session_start", id="fallback-session-start"),
        pytest.param("something-else-entirely", "other", id="fallback-other"),
    ])
    def test_kind_falls_back_to_event_string_for_unrecognised_payload(
        self, isolated_runs_root, isolated_log_root, monkeypatch, event, expected_kind,
    ):
        """No payload evidence (unknown harness shape): the normalised event
        argv string decides."""
        _, ld = _make_run_dirs(isolated_runs_root, isolated_log_root, "kindfallback")
        monkeypatch.setenv("AGENT_RUN_NAME", "kindfallback")

        agent_run.cmd_hook(_hook_args(event, json_payload='{"unrecognised": 1}'))

        assert _read_hooks(ld)[0]["kind"] == expected_kind

    def test_kind_raw_event_field_kept_unchanged_for_debugging(
        self, isolated_runs_root, isolated_log_root, monkeypatch,
    ):
        """Deriving kind must not mutate the original event field."""
        _, ld = _make_run_dirs(isolated_runs_root, isolated_log_root, "kindraw")
        monkeypatch.setenv("AGENT_RUN_NAME", "kindraw")

        agent_run.cmd_hook(
            _hook_args("my-custom-stop-alias",
                       json_payload=json.dumps(_fixture("hook_claude_stop.json")))
        )

        record = _read_hooks(ld)[0]
        assert record["event"] == "my-custom-stop-alias"
        assert record["kind"] == "turn_complete"


class TestMessageExtraction:
    """The assistant's last turn is promoted into the envelope so a wake is
    actionable without re-parsing the harness-specific payload shape."""

    def test_claude_message_survives_end_to_end_in_watch_aggregate(
        self, isolated_runs_root, isolated_log_root, monkeypatch, capsys,
    ):
        """A realistic 4096-char claude last_assistant_message survives
        intact through cmd_hook and appears in watch.hooks.last.message."""
        _, ld = _make_run_dirs(isolated_runs_root, isolated_log_root, "msgrun")
        monkeypatch.setenv("AGENT_RUN_NAME", "msgrun")
        payload = dict(_fixture("hook_claude_stop.json"))
        long_message = "The answer is 42. " * 215  # 4085 chars, under the cap
        assert len(long_message) <= agent_run._HOOK_MESSAGE_MAX_CHARS
        payload["last_assistant_message"] = long_message

        agent_run.cmd_hook(_hook_args("stop", json_payload=json.dumps(payload)))

        record = _read_hooks(ld)[0]
        assert record["message"] == long_message
        assert "message_truncated" not in record

        agent_run.cmd_watch(_watch_args("msgrun"))
        watch_payload = json.loads(capsys.readouterr().out)
        assert watch_payload["hooks"]["last"]["message"] == long_message

    def test_codex_message_key(
        self, isolated_runs_root, isolated_log_root, monkeypatch,
    ):
        _, ld = _make_run_dirs(isolated_runs_root, isolated_log_root, "msgcodex")
        monkeypatch.setenv("AGENT_RUN_NAME", "msgcodex")
        payload = json.dumps(_fixture("hook_codex_turn_complete.json"))

        agent_run.cmd_hook(_hook_args("turn-complete", extra=[payload]))

        assert _read_hooks(ld)[0]["message"] == "CXPONG"

    def test_opencode_properties_message_key(
        self, isolated_runs_root, isolated_log_root, monkeypatch,
    ):
        _, ld = _make_run_dirs(isolated_runs_root, isolated_log_root, "msgoc")
        monkeypatch.setenv("AGENT_RUN_NAME", "msgoc")
        payload = {"type": "session.idle",
                   "properties": {"sessionID": "s1", "message": "done thinking"}}

        agent_run.cmd_hook(_hook_args("session-idle", json_payload=json.dumps(payload)))

        assert _read_hooks(ld)[0]["message"] == "done thinking"

    def test_opencode_properties_text_key(
        self, isolated_runs_root, isolated_log_root, monkeypatch,
    ):
        _, ld = _make_run_dirs(isolated_runs_root, isolated_log_root, "msgoctext")
        monkeypatch.setenv("AGENT_RUN_NAME", "msgoctext")
        payload = {"type": "session.idle",
                   "properties": {"sessionID": "s1", "text": "the text form"}}

        agent_run.cmd_hook(_hook_args("session-idle", json_payload=json.dumps(payload)))

        assert _read_hooks(ld)[0]["message"] == "the text form"

    def test_no_message_field_is_null_not_missing(
        self, isolated_runs_root, isolated_log_root, monkeypatch,
    ):
        _, ld = _make_run_dirs(isolated_runs_root, isolated_log_root, "nomsg")
        monkeypatch.setenv("AGENT_RUN_NAME", "nomsg")

        agent_run.cmd_hook(_hook_args("stop", json_payload='{"session_id":"s1"}'))

        record = _read_hooks(ld)[0]
        assert "message" in record
        assert record["message"] is None

    @pytest.mark.parametrize("bad_value", [
        pytest.param(123, id="int"),
        pytest.param(["a", "b"], id="list"),
        pytest.param({"nested": True}, id="dict"),
        pytest.param(None, id="none"),
        pytest.param(True, id="bool"),
    ])
    def test_non_str_message_value_tolerated_not_raised(
        self, isolated_runs_root, isolated_log_root, monkeypatch, bad_value,
    ):
        """A non-str last_assistant_message must not raise; the field is
        simply absent from the record."""
        _, ld = _make_run_dirs(isolated_runs_root, isolated_log_root, "badmsg")
        monkeypatch.setenv("AGENT_RUN_NAME", "badmsg")
        payload = {"session_id": "s1", "hook_event_name": "Stop",
                   "last_assistant_message": bad_value}

        rc = agent_run.cmd_hook(_hook_args("stop", json_payload=json.dumps(payload)))

        assert rc == 0
        assert _read_hooks(ld)[0]["message"] is None
    """stdin must never outlast _HOOK_STDIN_TIMEOUT_SECONDS: the harness turn
    does not resume until the hook returns."""

    def test_tty_stdin_is_never_read(
        self, isolated_runs_root, isolated_log_root, monkeypatch,
    ):
        """An interactive stdin has no payload waiting and would block forever."""
        _, ld = _make_run_dirs(isolated_runs_root, isolated_log_root, "ttyrun")
        monkeypatch.setenv("AGENT_RUN_NAME", "ttyrun")

        class _FakeTTY:
            def isatty(self):
                return True

            def fileno(self):
                raise AssertionError("tty stdin must not be read")

        old_stdin = sys.stdin
        sys.stdin = _FakeTTY()
        try:
            rc = agent_run.cmd_hook(_hook_args("stop"))
        finally:
            sys.stdin = old_stdin

        assert rc == 0
        assert _read_hooks(ld)[0]["source"] == "none"

    @pytest.mark.parametrize("data,close_writer,slack", [
        pytest.param(b"", True, 1.0, id="eof-immediately"),
        pytest.param(b"", False, 1.0, id="empty-writer-held-open"),
        pytest.param(b'{"partial": "data"', False, 1.5, id="partial-writer-held-open"),
    ])
    def test_returns_within_deadline(
        self, isolated_runs_root, isolated_log_root, monkeypatch,
        data, close_writer, slack,
    ):
        _, ld = _make_run_dirs(isolated_runs_root, isolated_log_root, "deadline")
        monkeypatch.setenv("AGENT_RUN_NAME", "deadline")

        rc, elapsed = _hook_with_stdin(data, close_writer=close_writer)

        assert rc == 0
        assert elapsed < agent_run._HOOK_STDIN_TIMEOUT_SECONDS + slack
        assert _read_hooks(ld), "a record is written even with no usable payload"


def _fork_concurrent_hooks(n_writers, run_name, payload_for):
    """Drive n_writers concurrent cmd_hook appends from forked children.

    Children block on a pipe until the parent releases them, so every writer
    reaches os.write at once instead of being staggered by process startup.
    """
    read_fd, write_fd = os.pipe()
    pids = []
    for i in range(n_writers):
        pid = os.fork()
        if pid == 0:
            # Child: never return into pytest, never flush its buffers.
            try:
                os.close(write_fd)
                while not os.read(read_fd, 1):
                    pass
                ns = argparse.Namespace(
                    event="stop", name=run_name,
                    json_payload=json.dumps(payload_for(i)), extra=[],
                )
                agent_run.cmd_hook(ns)
            except BaseException:
                os._exit(1)
            os._exit(0)
        pids.append(pid)

    os.close(read_fd)
    os.write(write_fd, b"x" * n_writers)
    os.close(write_fd)

    for pid in pids:
        _, status = os.waitpid(pid, 0)
        assert os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0, (
            "writer %d exited abnormally: %s" % (pid, status)
        )


class TestAtomicAppend:
    def test_concurrent_writers_produce_complete_lines(
        self, isolated_runs_root, isolated_log_root,
    ):
        """20 concurrent hooks yield 20 parseable lines, none spliced."""
        name = "concurrent"
        _, ld = _make_run_dirs(isolated_runs_root, isolated_log_root, name)
        n_writers = 20

        _fork_concurrent_hooks(
            n_writers, name, lambda i: {"writer": i},
        )

        lines = _hook_lines(ld)
        assert len(lines) == n_writers, f"expected {n_writers} lines, got {len(lines)}"
        for line in lines:
            assert json.loads(line)["event"] == "stop"

    def test_40_writers_near_max_line_bytes_produce_no_torn_lines(
        self, isolated_runs_root, isolated_log_root,
    ):
        """40 concurrent processes each append a record sized close to
        _HOOK_MAX_LINE_BYTES (8192): the new size where atomicity must still
        hold on a local regular file's O_APPEND fd, per PIPE_BUF not applying
        here. Every writer's index is embedded in its payload so a spliced
        line is detectable by content mismatch, not just by unparseability.
        """
        name = "concurrent8k"
        _, ld = _make_run_dirs(isolated_runs_root, isolated_log_root, name)
        n_writers = 40

        _fork_concurrent_hooks(
            n_writers,
            name,
            lambda i: {
                "writer": i,
                "last_assistant_message": "m" * 6000,
                "session_id": "writer-%d" % i,
            },
        )

        lines = _hook_lines(ld)
        assert len(lines) == n_writers, f"expected {n_writers} lines, got {len(lines)}"
        seen_writers = set()
        for line in lines:
            assert len(line) <= agent_run._HOOK_MAX_LINE_BYTES
            record = json.loads(line)
            assert record["event"] == "stop"
            writer = record["payload"].get("writer")
            assert writer not in seen_writers, "duplicate writer index: torn/duplicated line"
            seen_writers.add(writer)
        assert seen_writers == set(range(n_writers)), "a writer's line was lost or spliced"


class TestLineCap:
    """Every line must fit _HOOK_MAX_LINE_BYTES, the atomicity bound for a
    single os.write to an O_APPEND fd on a local regular file."""

    @pytest.mark.parametrize("args,label", [
        pytest.param(
            {"json_payload": json.dumps({f"field_{i:02d}": "x" * 300 for i in range(5)})},
            "oversized-payload", id="oversized-payload",
        ),
        pytest.param({"event": "e" * 100000}, "huge-ascii-event", id="huge-ascii-event"),
        pytest.param({"event": "\u65e5" * 100000}, "huge-bmp-event", id="huge-bmp-event"),
        pytest.param(
            {"event": "\U0001F600" * 100000}, "huge-astral-event", id="huge-astral-event",
        ),
        pytest.param(
            {"event": "\U0001F600" * 5000,
             "json_payload": json.dumps({"type": "session.idle",
                                         "properties": {"sessionID": "s" * 400},
                                         "detail": "d" * 400})},
            "huge-event-and-payload", id="huge-event-and-payload",
        ),
        pytest.param(
            {"json_payload": json.dumps({
                "last_assistant_message": "m" * 20000,
                "transcript_path": "/tmp/" + "p" * 20000,
            })},
            "huge-message-and-payload", id="huge-message-and-payload",
        ),
    ])
    def test_line_never_exceeds_max_line_bytes(
        self, isolated_runs_root, isolated_log_root, monkeypatch, args, label,
    ):
        """Sizes are measured on the ensure_ascii encoding, where one astral
        character costs 12 bytes — a character-count cap would not bound this.
        """
        _, ld = _make_run_dirs(isolated_runs_root, isolated_log_root, "linecap")
        monkeypatch.setenv("AGENT_RUN_NAME", "linecap")
        event = args.get("event", "stop")

        rc = agent_run.cmd_hook(
            _hook_args(event, json_payload=args.get("json_payload"))
        )

        assert rc == 0
        lines = _hook_lines(ld)
        assert len(lines) == 1
        assert len(lines[0]) <= agent_run._HOOK_MAX_LINE_BYTES, (
            f"{label} line exceeds _HOOK_MAX_LINE_BYTES={agent_run._HOOK_MAX_LINE_BYTES}"
        )

    def test_oversized_payload_is_flagged_truncated(
        self, isolated_runs_root, isolated_log_root, monkeypatch,
    ):
        """A shrunk payload is marked, so a reader can tell a small payload
        from an elided one."""
        _, ld = _make_run_dirs(isolated_runs_root, isolated_log_root, "truncrun")
        monkeypatch.setenv("AGENT_RUN_NAME", "truncrun")
        big = {f"field_{i:02d}": "x" * 300 for i in range(5)}

        agent_run.cmd_hook(_hook_args("stop", json_payload=json.dumps(big)))

        assert json.loads(_hook_lines(ld)[0])["payload_truncated"] is True

    def test_payload_fits_within_budget_is_not_flagged(
        self, isolated_runs_root, isolated_log_root, monkeypatch,
    ):
        _, ld = _make_run_dirs(isolated_runs_root, isolated_log_root, "smallrun")
        monkeypatch.setenv("AGENT_RUN_NAME", "smallrun")

        agent_run.cmd_hook(_hook_args("stop", json_payload='{"k":"v"}'))

        record = _read_hooks(ld)[0]
        assert "payload_truncated" not in record
        assert record["payload"] == {"k": "v"}

    def test_repeated_writes_all_within_line_cap(
        self, isolated_runs_root, isolated_log_root, monkeypatch,
    ):
        _, ld = _make_run_dirs(isolated_runs_root, isolated_log_root, "pipebufrun")
        monkeypatch.setenv("AGENT_RUN_NAME", "pipebufrun")

        for _ in range(10):
            agent_run.cmd_hook(
                _hook_args("stop", json_payload='{"k":"' + "v" * 200 + '"}')
            )

        for line in _hook_lines(ld):
            assert len(line) <= agent_run._HOOK_MAX_LINE_BYTES

    def test_event_cap_stops_appending(
        self, isolated_runs_root, isolated_log_root, monkeypatch,
    ):
        """Past AGENT_RUN_HOOK_MAX_EVENTS records, further events are dropped."""
        _, ld = _make_run_dirs(isolated_runs_root, isolated_log_root, "caprun")
        monkeypatch.setenv("AGENT_RUN_NAME", "caprun")
        monkeypatch.setattr(agent_run, "AGENT_RUN_HOOK_MAX_EVENTS", 3)

        for _ in range(5):
            agent_run.cmd_hook(_hook_args("stop", json_payload='{"x":1}'))

        assert len(_read_hooks(ld)) == 3


class TestPriorityTruncation:
    """message is clipped last, after payload has already been shrunk to {},
    so a huge transcript_path cannot starve the field that makes a wake
    actionable."""

    def test_huge_payload_field_and_long_message_keeps_message_clips_payload(
        self, isolated_runs_root, isolated_log_root, monkeypatch,
    ):
        _, ld = _make_run_dirs(isolated_runs_root, isolated_log_root, "prioritytrunc")
        monkeypatch.setenv("AGENT_RUN_NAME", "prioritytrunc")
        message = "The assistant's important final answer. " * 40  # ~1680 chars
        payload = {
            "transcript_path": "/very/long/synthetic/path/" + "p" * 10000,
            "session_id": "s1",
            "hook_event_name": "Stop",
            "last_assistant_message": message,
        }

        rc = agent_run.cmd_hook(_hook_args("stop", json_payload=json.dumps(payload)))

        assert rc == 0
        record = _read_hooks(ld)[0]
        assert record["message"] == message
        assert "message_truncated" not in record
        assert record["payload_truncated"] is True
        assert len(record["payload"].get("transcript_path", "")) < 10000

    def test_over_long_message_sets_message_truncated(self):
        """_hook_extract_message already clips to _HOOK_MESSAGE_MAX_CHARS
        encoded bytes, so an over-budget message reaching _hook_encode_line
        can only come from a record built some other way — exercise the
        ladder rung directly rather than through cmd_hook, which cannot
        produce this input by construction."""
        huge_message = "m" * 50000
        record = {
            "event": "stop", "at": "2026-08-19T00:00:00Z", "harness": "claude",
            "kind": "turn_complete", "message": huge_message,
            "source": "stdin", "pid": 123, "resolved_by": "env",
            "payload": {},
        }

        line = agent_run._hook_encode_line(record)

        assert len(line) <= agent_run._HOOK_MAX_LINE_BYTES
        record_out = json.loads(line)
        assert record_out["message_truncated"] is True
        assert huge_message.startswith(record_out["message"])
        assert 0 < len(record_out["message"]) < len(huge_message)

    def test_message_only_present_when_payload_carries_one(
        self, isolated_runs_root, isolated_log_root, monkeypatch,
    ):
        """A normal-sized record with no message field is never flagged
        message_truncated."""
        _, ld = _make_run_dirs(isolated_runs_root, isolated_log_root, "nomsgtrunc")
        monkeypatch.setenv("AGENT_RUN_NAME", "nomsgtrunc")

        agent_run.cmd_hook(_hook_args("stop", json_payload='{"session_id":"s1"}'))

        record = _read_hooks(ld)[0]
        assert "message_truncated" not in record
        assert record["message"] is None


class TestSymlinkRefused:
    def test_symlink_at_hooks_jsonl_is_not_followed(
        self, isolated_runs_root, isolated_log_root, monkeypatch, tmp_path,
    ):
        """O_NOFOLLOW: a symlink planted at hooks.jsonl must not redirect the
        write to its target."""
        _, ld = _make_run_dirs(isolated_runs_root, isolated_log_root, "symlinkrun")
        monkeypatch.setenv("AGENT_RUN_NAME", "symlinkrun")
        target = tmp_path / "precious.txt"
        target.write_text("original content\n")
        (ld / "hooks.jsonl").symlink_to(target)

        rc = agent_run.cmd_hook(_hook_args("stop", json_payload='{"a":1}'))

        assert rc == 0
        assert target.read_text() == "original content\n"

    def test_symlink_to_absent_path_is_not_created(
        self, isolated_runs_root, isolated_log_root, monkeypatch, tmp_path,
    ):
        """A dangling symlink must not be followed into creating its target."""
        _, ld = _make_run_dirs(isolated_runs_root, isolated_log_root, "danglerun")
        monkeypatch.setenv("AGENT_RUN_NAME", "danglerun")
        target = tmp_path / "should-not-appear.txt"
        (ld / "hooks.jsonl").symlink_to(target)

        rc = agent_run.cmd_hook(_hook_args("stop", json_payload='{"a":1}'))

        assert rc == 0
        assert not target.exists()

    def test_fifo_at_hooks_jsonl_does_not_block_write(
        self, isolated_runs_root, isolated_log_root, monkeypatch,
    ):
        """A FIFO with no reader would block an ordinary open forever."""
        _, ld = _make_run_dirs(isolated_runs_root, isolated_log_root, "fifowrite")
        monkeypatch.setenv("AGENT_RUN_NAME", "fifowrite")
        os.mkfifo(str(ld / "hooks.jsonl"))

        start = time.monotonic()
        rc = agent_run.cmd_hook(_hook_args("stop", json_payload='{"a":1}'))
        elapsed = time.monotonic() - start

        assert rc == 0
        assert elapsed < 5.0


class TestWatchHooks:
    def test_hooks_null_when_no_file(
        self, isolated_runs_root, isolated_log_root, capsys,
    ):
        _make_run_dirs(isolated_runs_root, isolated_log_root, "nofile")

        agent_run.cmd_watch(_watch_args("nofile"))

        payload = json.loads(capsys.readouterr().out)
        assert "hooks" in payload, "the watch contract always carries the key"
        assert payload["hooks"] is None

    def test_hooks_aggregate(
        self, isolated_runs_root, isolated_log_root, monkeypatch, capsys,
    ):
        _make_run_dirs(isolated_runs_root, isolated_log_root, "watchagg")
        monkeypatch.setenv("AGENT_RUN_NAME", "watchagg")
        agent_run.cmd_hook(_hook_args("stop", json_payload='{"x":1}'))
        agent_run.cmd_hook(_hook_args("stop", json_payload='{"x":2}'))
        agent_run.cmd_hook(_hook_args("notification", json_payload='{"x":3}'))

        agent_run.cmd_watch(_watch_args("watchagg"))

        hooks = json.loads(capsys.readouterr().out)["hooks"]
        assert hooks["count"] == 3
        assert hooks["events"] == {"stop": 2, "notification": 1}
        assert hooks["last"]["event"] == "notification"
        assert hooks["last_event_age_s"] >= 0.0
        assert hooks["at_cap"] is False

    def test_hooks_aggregate_kinds_and_last_kind_message(
        self, isolated_runs_root, isolated_log_root, monkeypatch, capsys,
    ):
        """watch.hooks.last gains kind+message, and a top-level kinds counter
        aggregates alongside the existing events counter — purely additive."""
        _make_run_dirs(isolated_runs_root, isolated_log_root, "watchkinds")
        monkeypatch.setenv("AGENT_RUN_NAME", "watchkinds")
        agent_run.cmd_hook(_hook_args(
            "stop", json_payload=json.dumps(_fixture("hook_claude_stop.json"))))
        agent_run.cmd_hook(_hook_args(
            "turn-complete", extra=[json.dumps(_fixture("hook_codex_turn_complete.json"))]))
        agent_run.cmd_hook(_hook_args(
            "permission-asked",
            json_payload=json.dumps(_fixture("hook_opencode_permission_asked.json"))))

        agent_run.cmd_watch(_watch_args("watchkinds"))

        hooks = json.loads(capsys.readouterr().out)["hooks"]
        assert hooks["kinds"] == {"turn_complete": 2, "permission_required": 1}
        assert hooks["last"]["kind"] == "permission_required"
        assert hooks["last"]["message"] is None  # permission.asked carries no message
        # count/last/last_event_age_s/events/at_cap/truncated stay as they were.
        assert hooks["count"] == 3
        assert set(hooks["events"]) == {"stop", "turn-complete", "permission-asked"}

    def test_hooks_aggregate_last_message_survives(
        self, isolated_runs_root, isolated_log_root, monkeypatch, capsys,
    ):
        _make_run_dirs(isolated_runs_root, isolated_log_root, "watchmsg")
        monkeypatch.setenv("AGENT_RUN_NAME", "watchmsg")
        agent_run.cmd_hook(_hook_args(
            "stop", json_payload=json.dumps(_fixture("hook_claude_stop.json"))))

        agent_run.cmd_watch(_watch_args("watchmsg"))

        hooks = json.loads(capsys.readouterr().out)["hooks"]
        assert hooks["last"]["message"] == "PONGCAPTURE"
        assert hooks["last"]["kind"] == "turn_complete"

    def test_at_cap_reported_at_cap(
        self, isolated_runs_root, isolated_log_root, monkeypatch, capsys,
    ):
        _make_run_dirs(isolated_runs_root, isolated_log_root, "watchcap")
        monkeypatch.setattr(agent_run, "AGENT_RUN_HOOK_MAX_EVENTS", 2)
        monkeypatch.setenv("AGENT_RUN_NAME", "watchcap")
        for _ in range(2):
            agent_run.cmd_hook(_hook_args("stop", json_payload='{"x":1}'))

        agent_run.cmd_watch(_watch_args("watchcap"))

        assert json.loads(capsys.readouterr().out)["hooks"]["at_cap"] is True

    @pytest.mark.parametrize("plant,expect_hooks_null", [
        pytest.param(
            lambda p: p.write_bytes(b"not json\n{also bad\nmore garbage\n"),
            False, id="corrupt-lines",
        ),
        pytest.param(lambda p: os.mkfifo(str(p)), True, id="fifo"),
        pytest.param(lambda p: p.symlink_to("/dev/zero"), True, id="symlink-dev-zero"),
        pytest.param(lambda p: p.mkdir(), True, id="directory"),
        pytest.param(
            lambda p: p.write_bytes(b"x" * (10 * 1024 * 1024)), False, id="10mb-one-line",
        ),
        pytest.param(
            lambda p: p.write_bytes(b"[" * 200000 + b"]" * 200000 + b"\n"),
            False, id="deeply-nested-json",
        ),
        pytest.param(lambda p: p.write_bytes(b"\xff\xfe\n\x00\x01\n"), False, id="binary"),
    ])
    def test_hostile_file_never_reports_live_run_as_terminal(
        self, isolated_runs_root, isolated_log_root, capsys, plant, expect_hooks_null,
    ):
        """watch must return, and must not degrade to status 'unknown' — which
        _watch_is_terminal treats as terminal, i.e. a live run reported dead.

        The 10 MB line exceeds no read cap by accident: it is capped at
        _HOOKS_READ_MAX_BYTES. The nested case exhausts the recursion limit
        inside json.loads.
        """
        name = "hostile"
        sd, ld = _make_run_dirs(isolated_runs_root, isolated_log_root, name)
        (sd / "pid").write_text(f"{os.getpid()}\n")
        hooks_path = ld / "hooks.jsonl"
        try:
            plant(hooks_path)
            start = time.monotonic()
            rc = agent_run.cmd_watch(_watch_args(name))
            elapsed = time.monotonic() - start
        finally:
            if hooks_path.is_dir():
                hooks_path.rmdir()
            else:
                hooks_path.unlink(missing_ok=True)

        assert rc == 0
        assert elapsed < 10.0, "watch did not return promptly"
        payload = json.loads(capsys.readouterr().out)
        assert payload["terminal"] is False
        assert payload["status"] != "unknown"
        assert payload["observation_error"] is None
        if expect_hooks_null:
            assert payload["hooks"] is None


class TestReadHooksJsonl:
    def test_absent_file_returns_none(self, tmp_path):
        assert agent_run._read_hooks_jsonl(tmp_path) is None

    def test_empty_file_returns_zero_count(self, tmp_path):
        (tmp_path / "hooks.jsonl").write_bytes(b"")
        assert agent_run._read_hooks_jsonl(tmp_path)["count"] == 0

    def test_valid_records_aggregated(self, tmp_path):
        lines = [
            json.dumps({"event": "stop", "at": "2026-08-18T22:31:00Z",
                        "harness": "claude", "payload": {}}),
            json.dumps({"event": "stop", "at": "2026-08-18T22:32:00Z",
                        "harness": "claude", "payload": {}}),
            json.dumps({"event": "notification", "at": "2026-08-18T22:33:00Z",
                        "harness": None, "payload": {}}),
        ]
        (tmp_path / "hooks.jsonl").write_text("\n".join(lines) + "\n")

        result = agent_run._read_hooks_jsonl(tmp_path)

        assert result["count"] == 3
        assert result["events"] == {"stop": 2, "notification": 1}
        assert result["last"]["event"] == "notification"

    def test_corrupt_lines_skipped(self, tmp_path):
        (tmp_path / "hooks.jsonl").write_text(
            'not json\n{"event":"stop","at":"2026-08-18T22:31:00Z"}\n'
            "also garbage\n"
        )

        result = agent_run._read_hooks_jsonl(tmp_path)

        assert result["count"] == 1
        assert result["events"] == {"stop": 1}

    def test_kinds_aggregated_alongside_events(self, tmp_path):
        lines = [
            json.dumps({"event": "stop", "at": "2026-08-18T22:31:00Z",
                        "harness": "claude", "kind": "turn_complete", "payload": {}}),
            json.dumps({"event": "turn-complete", "at": "2026-08-18T22:32:00Z",
                        "harness": "codex", "kind": "turn_complete", "payload": {}}),
            json.dumps({"event": "notification", "at": "2026-08-18T22:33:00Z",
                        "harness": None, "kind": "permission_required", "payload": {}}),
        ]
        (tmp_path / "hooks.jsonl").write_text("\n".join(lines) + "\n")

        result = agent_run._read_hooks_jsonl(tmp_path)

        assert result["kinds"] == {"turn_complete": 2, "permission_required": 1}
        assert result["last"]["kind"] == "permission_required"

    @pytest.mark.parametrize("bad_kind,bad_message", [
        pytest.param("12345", "null", id="int-kind-null-message"),
        pytest.param("null", "67890", id="null-kind-int-message"),
        pytest.param("[1,2]", '{"a":1}', id="list-kind-dict-message"),
        pytest.param("true", "false", id="bool-kind-bool-message"),
    ])
    def test_non_string_kind_and_message_do_not_raise(
        self, tmp_path, bad_kind, bad_message,
    ):
        """A non-string kind/message in the file is data from a process that
        can write anything to hooks.jsonl — summarise as null, never raise."""
        (tmp_path / "hooks.jsonl").write_text(
            '{"event":"stop","at":"2026-08-18T22:31:00Z","kind":' + bad_kind
            + ',"message":' + bad_message + "}\n"
        )

        result = agent_run._read_hooks_jsonl(tmp_path)

        assert result is not None
        assert result["last"]["kind"] is None
        assert result["last"]["message"] is None
        # A non-string kind contributes nothing countable to the aggregate.
        assert result["kinds"] == {}

    @pytest.mark.parametrize("at_value", [
        pytest.param('"2026-01-01T00:00:00Z"', id="valid"),
        pytest.param('"not-a-date"', id="unparseable"),
        pytest.param("12345", id="not-a-string"),
        pytest.param("null", id="null"),
    ])
    def test_last_event_age_s_never_negative(self, tmp_path, at_value):
        """A future or malformed timestamp yields null or >= 0, never a
        negative age a poller might read as a fresh event."""
        (tmp_path / "hooks.jsonl").write_text(
            '{"event":"stop","at":' + at_value + "}\n"
        )

        age = agent_run._read_hooks_jsonl(tmp_path)["last_event_age_s"]

        assert age is None or age >= 0.0

    def test_summary_failure_returns_none_rather_than_raising(
        self, tmp_path, monkeypatch, capsys,
    ):
        """The boundary is total even if aggregation itself breaks: a raise
        here would reach cmd_watch and degrade status to 'unknown'."""
        (tmp_path / "hooks.jsonl").write_text('{"event":"stop"}\n')
        monkeypatch.setattr(
            agent_run, "_hooks_summary",
            lambda _d: (_ for _ in ()).throw(RuntimeError("boom")),
        )

        assert agent_run._read_hooks_jsonl(tmp_path) is None
        assert capsys.readouterr().out == ""


class TestJsonClip:
    @pytest.mark.parametrize("text", [
        pytest.param("plain ascii text " * 50, id="ascii"),
        pytest.param("\u65e5\u672c\u8a9e" * 100, id="bmp"),
        pytest.param("\U0001F600" * 100, id="astral"),
        pytest.param('quotes " and \\ backslashes ' * 30, id="escapes"),
        pytest.param("\n\t\x00" * 50, id="control-chars"),
    ])
    def test_clipped_encoding_fits_budget(self, text):
        """The budget is in encoded bytes: a clipped prefix must still encode
        within it, and must never split an escape sequence."""
        budget = 64

        clipped = agent_run._json_clip(text, budget)

        assert len(json.dumps(clipped, ensure_ascii=True)) - 2 <= budget
        assert text.startswith(clipped)

    def test_short_text_returned_unchanged(self):
        assert agent_run._json_clip("short", 64) == "short"


class TestMaxEventsEnv:
    @pytest.mark.parametrize("value", [
        pytest.param("abc", id="not-a-number"),
        pytest.param("", id="empty"),
        pytest.param("0", id="zero"),
        pytest.param("-5", id="negative"),
        pytest.param("1e6", id="float-notation"),
    ])
    def test_malformed_value_falls_back_to_default(self, value):
        """AGENT_RUN_HOOK_MAX_EVENTS is read at import time, so a bad value
        must not stop the module loading. Checked in a subprocess because the
        constant is resolved during import."""
        result = subprocess.run(
            [sys.executable, "-c",
             "from toolbox import agent_run; print(agent_run.AGENT_RUN_HOOK_MAX_EVENTS)"],
            capture_output=True, text=True,
            env={**os.environ, "PYTHONPATH": SRC_DIR,
                 "AGENT_RUN_HOOK_MAX_EVENTS": value},
            timeout=30,
        )

        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "1000"

    def test_valid_value_is_honoured(self):
        result = subprocess.run(
            [sys.executable, "-c",
             "from toolbox import agent_run; print(agent_run.AGENT_RUN_HOOK_MAX_EVENTS)"],
            capture_output=True, text=True,
            env={**os.environ, "PYTHONPATH": SRC_DIR, "AGENT_RUN_HOOK_MAX_EVENTS": "7"},
            timeout=30,
        )

        assert result.stdout.strip() == "7"

    def test_hook_subprocess_exits_0_with_malformed_value(
        self, isolated_runs_root, isolated_log_root,
    ):
        name = "badmax"
        _make_run_dirs(isolated_runs_root, isolated_log_root, name)

        result = subprocess.run(
            [sys.executable, "-m", "toolbox.agent_run", "hook", "stop", "--name", name],
            capture_output=True,
            env={**os.environ, "PYTHONPATH": SRC_DIR,
                 "AGENT_RUN_HOOK_MAX_EVENTS": "abc",
                 "AGENT_RUN_STATE_DIR": str(isolated_runs_root),
                 "AGENT_RUN_LOG_DIR": str(isolated_log_root)},
            timeout=30,
        )

        assert result.returncode == 0, result.stderr
        assert result.stdout == b"", "hook stdout must stay empty"


class TestEnvExport:
    def test_runner_exports_run_identity(
        self, isolated_runs_root, isolated_log_root,
    ):
        """The agent's environment carries AGENT_RUN_NAME plus both roots, so a
        hook resolves the run it was launched under, not a default root."""
        name = "envexport"
        check = (
            "import os, sys\n"
            f"assert os.environ['AGENT_RUN_NAME'] == {name!r}\n"
            f"assert os.environ['AGENT_RUN_STATE_DIR'] == {str(isolated_runs_root)!r}\n"
            f"assert os.environ['AGENT_RUN_LOG_DIR'] == {str(isolated_log_root)!r}\n"
        )

        result = subprocess.run(
            [sys.executable, "-m", "toolbox.agent_run", name, "--",
             sys.executable, "-c", check],
            capture_output=True, text=True,
            env={**os.environ,
                 "AGENT_RUN_STATE_DIR": str(isolated_runs_root),
                 "AGENT_RUN_LOG_DIR": str(isolated_log_root)},
            timeout=30,
        )

        assert result.returncode == 0, f"launch failed: {result.stderr!r}"
        status = _wait_for_terminal(isolated_runs_root / name / "status")
        assert status == "done", f"run ended with status={status!r}"
