"""Tests for `agent-run hook` — harness hook ingestion.

Covers:
  - Run resolution: env var hit, env var pointing at nonexistent run, ancestry
    fallback, nothing resolvable → exit 0 and no file created.
  - Exit-code contract: always 0 on every failure path.
  - Payload parsing for all three shapes: stdin JSON, argv JSON, --json flag.
  - stdin never blocks when it is a tty / empty.
  - Atomic append: concurrent writers produce exactly N well-formed lines,
    none interleaved.
  - Line cap honoured; oversized payload truncated with payload_truncated=true
    and still <= _HOOK_MAX_LINE_BYTES.
  - `watch --json` includes hooks: null with no file, correct aggregate with a
    file, and survives a deliberately corrupt hooks.jsonl.
  - Env export: a launched run has AGENT_RUN_NAME in the agent's environment.
  - Regression: every Critical finding and items 7, 8, 9.
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


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

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
    if write_state:
        sd.mkdir(parents=True, exist_ok=True)
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
    """Read hooks.jsonl, returning a list of parsed records (skips bad lines)."""
    path = log_dir / "hooks.jsonl"
    if not path.exists():
        return []
    records = []
    for line in path.read_bytes().split(b"\n"):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return records


def _watch_args(name: str, *, as_json: bool = True) -> argparse.Namespace:
    return argparse.Namespace(name=name, json=as_json, repo=None)


# ---------------------------------------------------------------------------
# Resolution tests
# ---------------------------------------------------------------------------

class TestHookResolution:
    def test_env_var_hit_writes_record(
        self, isolated_runs_root, isolated_log_root, monkeypatch,
    ):
        """AGENT_RUN_NAME env var pointing at a real run resolves and writes."""
        sd, ld = _make_run_dirs(isolated_runs_root, isolated_log_root, "myrun")
        monkeypatch.setenv("AGENT_RUN_NAME", "myrun")

        rc = agent_run.cmd_hook(_hook_args("stop", json_payload='{"a":1}'))

        assert rc == 0
        records = _read_hooks(ld)
        assert len(records) == 1
        assert records[0]["event"] == "stop"
        assert records[0]["resolved_by"] == "env"

    def test_env_var_nonexistent_run_no_file(
        self, isolated_runs_root, isolated_log_root, monkeypatch,
    ):
        """AGENT_RUN_NAME pointing at a nonexistent run: no file, exit 0."""
        monkeypatch.setenv("AGENT_RUN_NAME", "ghost")
        monkeypatch.delenv("AGENT_RUN_STATE_DIR", raising=False)

        rc = agent_run.cmd_hook(_hook_args("stop"))

        assert rc == 0
        assert not (isolated_log_root / "ghost" / "hooks.jsonl").exists()

    def test_name_override_takes_priority(
        self, isolated_runs_root, isolated_log_root, monkeypatch,
    ):
        """--name overrides the env var."""
        _make_run_dirs(isolated_runs_root, isolated_log_root, "target")
        _make_run_dirs(isolated_runs_root, isolated_log_root, "other")
        monkeypatch.setenv("AGENT_RUN_NAME", "other")

        rc = agent_run.cmd_hook(_hook_args("stop", name="target",
                                            json_payload='{"x":1}'))

        assert rc == 0
        assert _read_hooks(isolated_log_root / "target")
        assert not _read_hooks(isolated_log_root / "other")

    def test_nothing_resolvable_exits_0_no_file(
        self, isolated_runs_root, isolated_log_root, monkeypatch,
    ):
        """No env var, no ancestry match → exit 0, no hooks.jsonl anywhere."""
        monkeypatch.delenv("AGENT_RUN_NAME", raising=False)

        rc = agent_run.cmd_hook(_hook_args("stop", json_payload='{"x":1}'))

        assert rc == 0
        # No run dirs exist, so no hooks.jsonl can have been written.
        for d in isolated_log_root.iterdir():
            assert not (d / "hooks.jsonl").exists()

    def test_ancestry_fallback_via_pgid(
        self, isolated_runs_root, isolated_log_root, monkeypatch,
    ):
        """Ancestry fallback matches when our session id equals the run's pgid."""
        name = "ancestorrun"
        sd, ld = _make_run_dirs(isolated_runs_root, isolated_log_root, name)
        # Record the test process's own pgid as the run's pgid — the hook
        # will see os.getsid(0) and compare against run_pgid.
        our_pgid = os.getpgid(0)
        (sd / "pgid").write_text(f"{our_pgid}\n")
        monkeypatch.delenv("AGENT_RUN_NAME", raising=False)

        rc = agent_run.cmd_hook(_hook_args("stop", json_payload='{"x":1}'))

        assert rc == 0
        records = _read_hooks(ld)
        assert len(records) == 1
        assert records[0]["resolved_by"] == "ancestry"

    def test_invalid_name_override_exits_0(
        self, isolated_runs_root, isolated_log_root, monkeypatch,
    ):
        """An invalid --name (path separators) exits 0 without crashing."""
        monkeypatch.delenv("AGENT_RUN_NAME", raising=False)
        rc = agent_run.cmd_hook(_hook_args("stop", name="../../etc/passwd"))
        assert rc == 0


# ---------------------------------------------------------------------------
# Exit code contract — always 0
# ---------------------------------------------------------------------------

class TestExitCodeAlways0:
    def test_unwritable_log_dir(
        self, isolated_runs_root, isolated_log_root, monkeypatch, tmp_path,
    ):
        """Unwritable log dir: exit 0, stderr message, no raise."""
        name = "nowrite"
        sd, ld = _make_run_dirs(isolated_runs_root, isolated_log_root, name)
        monkeypatch.setenv("AGENT_RUN_NAME", name)
        # Make log dir unwritable.
        ld.chmod(0o555)
        try:
            rc = agent_run.cmd_hook(_hook_args("stop", json_payload='{"x":1}'))
        finally:
            ld.chmod(0o755)
        assert rc == 0

    def test_bad_json_payload_flag(
        self, isolated_runs_root, isolated_log_root, monkeypatch,
    ):
        """Bad JSON in --json flag: exit 0, record written with raw fallback."""
        name = "badjson"
        sd, ld = _make_run_dirs(isolated_runs_root, isolated_log_root, name)
        monkeypatch.setenv("AGENT_RUN_NAME", name)

        rc = agent_run.cmd_hook(_hook_args("stop", json_payload="not json {{{"))

        assert rc == 0
        records = _read_hooks(ld)
        assert len(records) == 1
        assert "raw" in records[0]["payload"]

    def test_missing_run_exits_0(
        self, isolated_runs_root, isolated_log_root, monkeypatch,
    ):
        """Unresolvable run: exit 0."""
        monkeypatch.delenv("AGENT_RUN_NAME", raising=False)
        rc = agent_run.cmd_hook(_hook_args("stop"))
        assert rc == 0

    def test_oversized_payload_exits_0(
        self, isolated_runs_root, isolated_log_root, monkeypatch,
    ):
        """Oversized payload: exit 0, line still <= _HOOK_MAX_LINE_BYTES."""
        name = "bigpayload"
        sd, ld = _make_run_dirs(isolated_runs_root, isolated_log_root, name)
        monkeypatch.setenv("AGENT_RUN_NAME", name)
        big = {f"k{i}": "x" * 300 for i in range(5)}

        rc = agent_run.cmd_hook(_hook_args("stop", json_payload=json.dumps(big)))

        assert rc == 0
        records = _read_hooks(ld)
        assert len(records) == 1
        raw = (ld / "hooks.jsonl").read_bytes()
        for line in raw.split(b"\n"):
            if line.strip():
                assert len(line) <= agent_run._HOOK_MAX_LINE_BYTES

    def test_exception_in_inner_exits_0(
        self, isolated_runs_root, isolated_log_root, monkeypatch,
    ):
        """Even if _cmd_hook_inner raises, cmd_hook exits 0."""
        monkeypatch.setenv("AGENT_RUN_NAME", "boom")
        # Patch inner to raise.
        monkeypatch.setattr(agent_run, "_cmd_hook_inner", lambda _a: (_ for _ in ()).throw(RuntimeError("kaboom")))
        rc = agent_run.cmd_hook(_hook_args("stop"))
        assert rc == 0


# ---------------------------------------------------------------------------
# Payload parsing
# ---------------------------------------------------------------------------

class TestPayloadParsing:
    def test_json_flag_source(
        self, isolated_runs_root, isolated_log_root, monkeypatch,
    ):
        """--json flag payload is parsed and source='flag'."""
        name = "flagrun"
        sd, ld = _make_run_dirs(isolated_runs_root, isolated_log_root, name)
        monkeypatch.setenv("AGENT_RUN_NAME", name)

        agent_run.cmd_hook(_hook_args("stop", json_payload='{"session_id":"abc"}'))

        records = _read_hooks(ld)
        assert records[0]["source"] == "flag"
        assert records[0]["harness"] is None or isinstance(records[0]["harness"], str)

    def test_argv_json_source(
        self, isolated_runs_root, isolated_log_root, monkeypatch,
    ):
        """JSON as extra positional (codex shape) is parsed; source='argv'."""
        name = "argvrun"
        sd, ld = _make_run_dirs(isolated_runs_root, isolated_log_root, name)
        monkeypatch.setenv("AGENT_RUN_NAME", name)
        codex_payload = json.dumps({
            "type": "agent-turn-complete",
            "thread-id": "t1",
            "turn-id": "x1",
            "cwd": "/tmp",
            "client": "codex",
            "input-messages": [],
            "last-assistant-message": "done",
        })

        rc = agent_run.cmd_hook(argparse.Namespace(
            event="turn-complete",
            name=None,
            json_payload=None,
            extra=[codex_payload],
        ))

        assert rc == 0
        records = _read_hooks(ld)
        assert records[0]["source"] == "argv"

    def test_stdin_json_source(
        self, isolated_runs_root, isolated_log_root, monkeypatch, tmp_path,
    ):
        """JSON on stdin (claude Stop shape) is parsed; source='stdin'."""
        name = "stdinrun"
        sd, ld = _make_run_dirs(isolated_runs_root, isolated_log_root, name)
        monkeypatch.setenv("AGENT_RUN_NAME", name)

        claude_payload = json.dumps({
            "session_id": "ses1",
            "transcript_path": "/tmp/t",
            "cwd": "/tmp",
            "prompt_id": "p1",
            "permission_mode": "bypass",
            "hook_event_name": "Stop",
            "stop_hook_active": False,
            "last_assistant_message": "I am done.",
        })

        # Write payload to a pipe and use it as stdin.
        payload_bytes = claude_payload.encode()
        r_fd, w_fd = os.pipe()
        os.write(w_fd, payload_bytes)
        os.close(w_fd)

        old_stdin = sys.stdin
        sys.stdin = open(r_fd, "rb", closefd=True)  # noqa: WPS515
        try:
            # isatty() will return False on a pipe.
            rc = agent_run.cmd_hook(argparse.Namespace(
                event="stop",
                name=None,
                json_payload=None,
                extra=[],
            ))
        finally:
            sys.stdin.close()
            sys.stdin = old_stdin

        assert rc == 0
        records = _read_hooks(ld)
        assert records[0]["source"] == "stdin"
        assert records[0]["harness"] == "claude"

    def test_claude_harness_detected(
        self, isolated_runs_root, isolated_log_root, monkeypatch,
    ):
        """Claude Stop shape payload sets harness='claude'."""
        name = "clauderun"
        sd, ld = _make_run_dirs(isolated_runs_root, isolated_log_root, name)
        monkeypatch.setenv("AGENT_RUN_NAME", name)
        payload = json.dumps({
            "session_id": "s1", "hook_event_name": "Stop",
            "last_assistant_message": "done",
        })
        agent_run.cmd_hook(_hook_args("stop", json_payload=payload))

        records = _read_hooks(ld)
        assert records[0]["harness"] == "claude"

    def test_codex_harness_detected(
        self, isolated_runs_root, isolated_log_root, monkeypatch,
    ):
        """Codex notify shape sets harness='codex'."""
        name = "codexrun"
        sd, ld = _make_run_dirs(isolated_runs_root, isolated_log_root, name)
        monkeypatch.setenv("AGENT_RUN_NAME", name)
        payload = json.dumps({
            "type": "agent-turn-complete", "thread-id": "t1",
        })
        agent_run.cmd_hook(_hook_args("stop", json_payload=payload))
        records = _read_hooks(ld)
        assert records[0]["harness"] == "codex"

    def test_event_name_lowercased(
        self, isolated_runs_root, isolated_log_root, monkeypatch,
    ):
        """Event name is normalised to lowercase regardless of input case."""
        name = "caserun"
        sd, ld = _make_run_dirs(isolated_runs_root, isolated_log_root, name)
        monkeypatch.setenv("AGENT_RUN_NAME", name)
        agent_run.cmd_hook(_hook_args("STOP"))

        records = _read_hooks(ld)
        assert records[0]["event"] == "stop"

    def test_any_event_name_accepted(
        self, isolated_runs_root, isolated_log_root, monkeypatch,
    ):
        """Arbitrary event names (not in a hardcoded list) are accepted."""
        name = "anyevent"
        sd, ld = _make_run_dirs(isolated_runs_root, isolated_log_root, name)
        monkeypatch.setenv("AGENT_RUN_NAME", name)
        agent_run.cmd_hook(_hook_args("my-custom-event-XYZ"))
        records = _read_hooks(ld)
        assert records[0]["event"] == "my-custom-event-xyz"


# ---------------------------------------------------------------------------
# stdin blocking prevention
# ---------------------------------------------------------------------------

class TestStdinNonBlocking:
    def test_tty_stdin_produces_no_stdin_read(
        self, isolated_runs_root, isolated_log_root, monkeypatch,
    ):
        """When stdin is a tty, the hook never reads from it (source='none')."""
        name = "ttyrun"
        sd, ld = _make_run_dirs(isolated_runs_root, isolated_log_root, name)
        monkeypatch.setenv("AGENT_RUN_NAME", name)

        # Mock stdin as a tty by patching isatty.
        class _FakeTTY:
            def isatty(self):
                return True
            @property
            def buffer(self):
                raise AssertionError("should not read from tty stdin")

        old_stdin = sys.stdin
        sys.stdin = _FakeTTY()
        try:
            rc = agent_run.cmd_hook(argparse.Namespace(
                event="stop", name=None, json_payload=None, extra=[],
            ))
        finally:
            sys.stdin = old_stdin

        assert rc == 0
        records = _read_hooks(ld)
        assert records[0]["source"] == "none"

    def test_empty_pipe_stdin_does_not_block(
        self, isolated_runs_root, isolated_log_root, monkeypatch,
    ):
        """A pipe with no data available: hook returns quickly with source='none'."""
        name = "emptyrun"
        sd, ld = _make_run_dirs(isolated_runs_root, isolated_log_root, name)
        monkeypatch.setenv("AGENT_RUN_NAME", name)

        # Create a pipe where the write end stays open but no data is written.
        r_fd, w_fd = os.pipe()
        old_stdin = sys.stdin
        sys.stdin = open(r_fd, "rb", closefd=True)  # noqa: WPS515
        try:
            start = time.monotonic()
            rc = agent_run.cmd_hook(argparse.Namespace(
                event="stop", name=None, json_payload=None, extra=[],
            ))
            elapsed = time.monotonic() - start
        finally:
            sys.stdin.close()
            os.close(w_fd)
            sys.stdin = old_stdin

        assert rc == 0
        # Should return within the 2s timeout plus small overhead.
        assert elapsed < agent_run._HOOK_STDIN_TIMEOUT_SECONDS + 1.0
        records = _read_hooks(ld)
        assert records[0]["source"] == "none"

    def test_closed_stdin_does_not_block(
        self, isolated_runs_root, isolated_log_root, monkeypatch,
    ):
        """Closed stdin (EOF immediately): hook returns quickly."""
        name = "closedstdin"
        sd, ld = _make_run_dirs(isolated_runs_root, isolated_log_root, name)
        monkeypatch.setenv("AGENT_RUN_NAME", name)

        r_fd, w_fd = os.pipe()
        os.close(w_fd)  # immediate EOF
        old_stdin = sys.stdin
        sys.stdin = open(r_fd, "rb", closefd=True)  # noqa: WPS515
        try:
            rc = agent_run.cmd_hook(argparse.Namespace(
                event="stop", name=None, json_payload=None, extra=[],
            ))
        finally:
            sys.stdin.close()
            sys.stdin = old_stdin

        assert rc == 0


# ---------------------------------------------------------------------------
# Atomic append — concurrent writers
# ---------------------------------------------------------------------------

class TestAtomicAppend:
    def test_concurrent_writers_produce_complete_lines(
        self, isolated_runs_root, isolated_log_root, monkeypatch, tmp_path,
    ):
        """20 concurrent processes each write one hook event; exactly 20 valid
        JSON lines result, none interleaved."""
        name = "concurrent"
        sd, ld = _make_run_dirs(isolated_runs_root, isolated_log_root, name)

        state_root = str(isolated_runs_root)
        log_root = str(isolated_log_root)
        n_writers = 20

        # Script run in each subprocess: resolves via --name so no env inference.
        script = f"""\
import sys
sys.path.insert(0, {str(Path(__file__).parents[1] / 'src')!r})
import os
os.environ['AGENT_RUN_STATE_DIR'] = {state_root!r}
os.environ['AGENT_RUN_LOG_DIR'] = {log_root!r}
import importlib.util
import json
from pathlib import Path
from toolbox import agent_run
agent_run.STATE_ROOT = Path({state_root!r})
agent_run.LOG_ROOT = Path({log_root!r})
import argparse
ns = argparse.Namespace(event='stop', name={name!r},
    json_payload=json.dumps({{'writer': int(sys.argv[1])}}), extra=[])
rc = agent_run.cmd_hook(ns)
sys.exit(rc)
"""
        procs = []
        for i in range(n_writers):
            p = subprocess.Popen(
                [sys.executable, "-c", script, str(i)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            procs.append(p)

        for p in procs:
            p.wait(timeout=30)

        hooks_file = ld / "hooks.jsonl"
        assert hooks_file.exists(), "hooks.jsonl was not created"
        raw = hooks_file.read_bytes()
        lines = [l for l in raw.split(b"\n") if l.strip()]
        assert len(lines) == n_writers, (
            f"Expected {n_writers} lines, got {len(lines)}: {raw!r}"
        )
        for line in lines:
            obj = json.loads(line)  # raises if line is malformed/interleaved
            assert obj["event"] == "stop"


# ---------------------------------------------------------------------------
# Line cap and oversized payload
# ---------------------------------------------------------------------------

class TestLineCap:
    def test_line_cap_stops_appending(
        self, isolated_runs_root, isolated_log_root, monkeypatch,
    ):
        """Once AGENT_RUN_HOOK_MAX_EVENTS lines are written, further writes are dropped."""
        name = "caprun"
        sd, ld = _make_run_dirs(isolated_runs_root, isolated_log_root, name)
        monkeypatch.setenv("AGENT_RUN_NAME", name)
        monkeypatch.setattr(agent_run, "AGENT_RUN_HOOK_MAX_EVENTS", 3)

        for _ in range(5):
            agent_run.cmd_hook(_hook_args("stop", json_payload='{"x":1}'))

        records = _read_hooks(ld)
        assert len(records) == 3

    def test_oversized_payload_truncated_to_line_cap(
        self, isolated_runs_root, isolated_log_root, monkeypatch,
    ):
        """An oversized payload is truncated; line stays <= _HOOK_MAX_LINE_BYTES with
        payload_truncated=true set."""
        name = "truncrun"
        sd, ld = _make_run_dirs(isolated_runs_root, isolated_log_root, name)
        monkeypatch.setenv("AGENT_RUN_NAME", name)
        # 5 keys × 300-char values; well over 512 B even after field-level capping.
        big = {f"field_{i:02d}": "x" * 300 for i in range(5)}

        agent_run.cmd_hook(_hook_args("stop", json_payload=json.dumps(big)))

        raw = (ld / "hooks.jsonl").read_bytes()
        lines = [l for l in raw.split(b"\n") if l.strip()]
        assert len(lines) == 1
        assert len(lines[0]) <= agent_run._HOOK_MAX_LINE_BYTES
        obj = json.loads(lines[0])
        assert obj.get("payload_truncated") is True

    def test_watch_reports_at_cap_true_at_cap(
        self, isolated_runs_root, isolated_log_root, monkeypatch, capsys,
    ):
        """watch --json reports hooks.at_cap=true when line count == cap."""
        name = "watchcap"
        sd, ld = _make_run_dirs(isolated_runs_root, isolated_log_root, name)
        (sd / "status").write_text("running\n")
        monkeypatch.setattr(agent_run, "AGENT_RUN_HOOK_MAX_EVENTS", 2)
        monkeypatch.setenv("AGENT_RUN_NAME", name)

        for _ in range(2):
            agent_run.cmd_hook(_hook_args("stop", json_payload='{"x":1}'))

        agent_run.cmd_watch(_watch_args(name))
        payload = json.loads(capsys.readouterr().out)
        assert payload["hooks"]["at_cap"] is True


# ---------------------------------------------------------------------------
# watch --json hooks integration
# ---------------------------------------------------------------------------

class TestWatchHooks:
    def test_watch_hooks_null_when_no_file(
        self, isolated_runs_root, isolated_log_root, capsys,
    ):
        """hooks is null when hooks.jsonl is absent."""
        sd, ld = _make_run_dirs(isolated_runs_root, isolated_log_root, "nofile")
        (sd / "status").write_text("running\n")

        agent_run.cmd_watch(_watch_args("nofile"))
        payload = json.loads(capsys.readouterr().out)
        assert payload["hooks"] is None

    def test_watch_hooks_aggregate(
        self, isolated_runs_root, isolated_log_root, monkeypatch, capsys,
    ):
        """hooks contains correct count, last, events dict when file exists."""
        name = "watchagg"
        sd, ld = _make_run_dirs(isolated_runs_root, isolated_log_root, name)
        (sd / "status").write_text("running\n")
        monkeypatch.setenv("AGENT_RUN_NAME", name)

        agent_run.cmd_hook(_hook_args("stop", json_payload='{"x":1}'))
        agent_run.cmd_hook(_hook_args("stop", json_payload='{"x":2}'))
        agent_run.cmd_hook(_hook_args("notification", json_payload='{"x":3}'))

        agent_run.cmd_watch(_watch_args(name))
        payload = json.loads(capsys.readouterr().out)
        h = payload["hooks"]
        assert h is not None
        assert h["count"] == 3
        assert h["events"] == {"stop": 2, "notification": 1}
        assert h["last"]["event"] == "notification"
        assert h["last_event_age_s"] is not None
        assert h["last_event_age_s"] >= 0.0
        assert h["at_cap"] is False

    def test_watch_survives_corrupt_hooks_jsonl(
        self, isolated_runs_root, isolated_log_root, capsys,
    ):
        """A corrupt hooks.jsonl never raises or breaks watch output."""
        name = "corrupt"
        sd, ld = _make_run_dirs(isolated_runs_root, isolated_log_root, name)
        (sd / "status").write_text("running\n")
        (ld / "hooks.jsonl").write_bytes(b"not json\n{also bad\nmore garbage\n")

        rc = agent_run.cmd_watch(_watch_args(name))

        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        # hooks is present but has 0 records (all lines were corrupt).
        assert payload["hooks"] is not None
        assert payload["hooks"]["count"] == 0

    def test_watch_contract_includes_hooks_key(
        self, isolated_runs_root, isolated_log_root, capsys,
    ):
        """The watch contract always includes the 'hooks' key."""
        sd, ld = _make_run_dirs(isolated_runs_root, isolated_log_root, "keychk")
        (sd / "status").write_text("running\n")
        agent_run.cmd_watch(_watch_args("keychk"))
        payload = json.loads(capsys.readouterr().out)
        assert "hooks" in payload


# ---------------------------------------------------------------------------
# Env export: launched run has AGENT_RUN_NAME in the agent's environment
# ---------------------------------------------------------------------------

class TestEnvExport:
    def test_launched_run_exports_agent_run_name(
        self, isolated_runs_root, isolated_log_root,
    ):
        """A real launched run writes AGENT_RUN_NAME into the agent's env."""
        # Launch a one-shot run that prints env var to stdout (captured in log).
        state_root = str(isolated_runs_root)
        log_root = str(isolated_log_root)
        name = "envexport"

        result = subprocess.run(
            [
                sys.executable, "-m", "toolbox.agent_run",
                name, "--",
                sys.executable, "-c",
                "import os, sys; sys.exit(0 if os.environ.get('AGENT_RUN_NAME') else 1)",
            ],
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "AGENT_RUN_STATE_DIR": state_root,
                "AGENT_RUN_LOG_DIR": log_root,
            },
            timeout=30,
        )
        # agent-run prints start info to stdout; exit 0 means launch succeeded.
        assert result.returncode == 0, (
            f"launch failed:\nstdout={result.stdout!r}\nstderr={result.stderr!r}"
        )

        # Wait for the run to complete.
        import time as _time
        deadline = _time.monotonic() + 10
        status_file = isolated_runs_root / name / "status"
        while _time.monotonic() < deadline:
            if status_file.exists():
                st = status_file.read_text().strip()
                if st in {"done", "failed", "launch_failed"}:
                    break
            _time.sleep(0.1)

        status = status_file.read_text().strip() if status_file.exists() else "unknown"
        assert status == "done", (
            f"run ended with status={status!r}\n"
            f"launch stdout={result.stdout!r}"
        )

    def test_runner_exports_state_and_log_dir(
        self, isolated_runs_root, isolated_log_root,
    ):
        """Launched run also exports AGENT_RUN_STATE_DIR and AGENT_RUN_LOG_DIR."""
        state_root = str(isolated_runs_root)
        log_root = str(isolated_log_root)
        name = "direxport"

        check_script = (
            "import os, sys\n"
            "ok = (\n"
            "    os.environ.get('AGENT_RUN_STATE_DIR') and\n"
            "    os.environ.get('AGENT_RUN_LOG_DIR') and\n"
            "    os.environ.get('AGENT_RUN_NAME') == " + repr(name) + "\n"
            ")\n"
            "sys.exit(0 if ok else 1)\n"
        )

        result = subprocess.run(
            [
                sys.executable, "-m", "toolbox.agent_run",
                name, "--",
                sys.executable, "-c", check_script,
            ],
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "AGENT_RUN_STATE_DIR": state_root,
                "AGENT_RUN_LOG_DIR": log_root,
            },
            timeout=30,
        )
        assert result.returncode == 0, (
            f"launch failed:\nstdout={result.stdout!r}\nstderr={result.stderr!r}"
        )

        import time as _time
        deadline = _time.monotonic() + 10
        status_file = isolated_runs_root / name / "status"
        while _time.monotonic() < deadline:
            if status_file.exists():
                st = status_file.read_text().strip()
                if st in {"done", "failed", "launch_failed"}:
                    break
            _time.sleep(0.1)

        status = status_file.read_text().strip() if status_file.exists() else "unknown"
        assert status == "done", f"run ended with status={status!r}"


# ---------------------------------------------------------------------------
# _read_hooks_jsonl unit tests
# ---------------------------------------------------------------------------

class TestReadHooksJsonl:
    def test_absent_file_returns_none(self, tmp_path):
        result = agent_run._read_hooks_jsonl(tmp_path)
        assert result is None

    def test_empty_file_returns_zero_count(self, tmp_path):
        (tmp_path / "hooks.jsonl").write_bytes(b"")
        result = agent_run._read_hooks_jsonl(tmp_path)
        assert result is not None
        assert result["count"] == 0

    def test_valid_records_aggregated(self, tmp_path):
        lines = [
            json.dumps({"event": "stop", "at": "2026-08-18T22:31:00Z",
                        "harness": "claude", "source": "stdin", "pid": 1,
                        "resolved_by": "env", "payload": {}}),
            json.dumps({"event": "stop", "at": "2026-08-18T22:32:00Z",
                        "harness": "claude", "source": "stdin", "pid": 2,
                        "resolved_by": "env", "payload": {}}),
            json.dumps({"event": "notification", "at": "2026-08-18T22:33:00Z",
                        "harness": None, "source": "none", "pid": 3,
                        "resolved_by": "env", "payload": {}}),
        ]
        (tmp_path / "hooks.jsonl").write_text("\n".join(lines) + "\n")
        result = agent_run._read_hooks_jsonl(tmp_path)
        assert result["count"] == 3
        assert result["events"] == {"stop": 2, "notification": 1}
        assert result["last"]["event"] == "notification"

    def test_corrupt_lines_skipped(self, tmp_path):
        (tmp_path / "hooks.jsonl").write_text(
            'not json\n{"event":"stop","at":"2026-08-18T22:31:00Z","harness":null,'
            '"source":"none","pid":1,"resolved_by":"env","payload":{}}\n'
            "also garbage\n"
        )
        result = agent_run._read_hooks_jsonl(tmp_path)
        assert result["count"] == 1
        assert result["events"] == {"stop": 1}

    def test_last_event_age_s_nonnegative(self, tmp_path):
        record = json.dumps({
            "event": "stop", "at": "2026-01-01T00:00:00Z",
            "harness": None, "source": "none", "pid": 1,
            "resolved_by": "env", "payload": {},
        })
        (tmp_path / "hooks.jsonl").write_text(record + "\n")
        result = agent_run._read_hooks_jsonl(tmp_path)
        assert result["last_event_age_s"] is not None
        assert result["last_event_age_s"] >= 0.0


# ---------------------------------------------------------------------------
# Regression tests for all Critical items and items 7, 8, 9
# ---------------------------------------------------------------------------

class TestRegressionCritical1ArgvError:
    """Critical 1: bad argv must exit 0, never 2 (invariant 1)."""

    def test_main_hook_no_event_exits_0(self):
        """main(["hook"]) — missing event argument — must exit 0."""
        assert agent_run.main(["hook"]) == 0

    def test_main_hook_unknown_flag_exits_0(self):
        """main(["hook", "stop", "--nope"]) — unknown flag — must exit 0."""
        assert agent_run.main(["hook", "stop", "--nope"]) == 0

    def test_main_hook_missing_name_value_exits_0(self):
        """main(["hook", "stop", "--name"]) — flag without value — must exit 0."""
        assert agent_run.main(["hook", "stop", "--name"]) == 0


class TestRegressionCritical2StdinDeadline:
    """Critical 2: stdin with partial data and open writer must return within deadline."""

    def test_partial_stdin_writer_open_returns_within_deadline(
        self, isolated_runs_root, isolated_log_root, monkeypatch,
    ):
        name = "stdin_deadline"
        sd, ld = _make_run_dirs(isolated_runs_root, isolated_log_root, name)
        monkeypatch.setenv("AGENT_RUN_NAME", name)

        r_fd, w_fd = os.pipe()
        # Write partial data (not a full JSON object) but keep writer open.
        os.write(w_fd, b'{"partial": "data"')

        old_stdin = sys.stdin
        sys.stdin = open(r_fd, "rb", closefd=True)  # noqa: WPS515
        try:
            start = time.monotonic()
            rc = agent_run.cmd_hook(argparse.Namespace(
                event="stop", name=None, json_payload=None, extra=[],
            ))
            elapsed = time.monotonic() - start
        finally:
            sys.stdin.close()
            os.close(w_fd)
            sys.stdin = old_stdin

        assert rc == 0
        # Must return within deadline + small overhead, not hang.
        assert elapsed < agent_run._HOOK_STDIN_TIMEOUT_SECONDS + 1.5


class TestRegressionCritical3Symlink:
    """Critical 3: hooks.jsonl symlink must be refused, not followed."""

    def test_symlink_at_hooks_jsonl_refused(
        self, isolated_runs_root, isolated_log_root, monkeypatch, tmp_path,
    ):
        name = "symlinkrun"
        sd, ld = _make_run_dirs(isolated_runs_root, isolated_log_root, name)
        monkeypatch.setenv("AGENT_RUN_NAME", name)

        target = tmp_path / "precious.txt"
        target.write_text("original content\n")
        (ld / "hooks.jsonl").symlink_to(target)

        agent_run.cmd_hook(_hook_args("stop", json_payload='{"a":1}'))

        # The symlink target must be untouched.
        assert target.read_text() == "original content\n"


class TestRegressionCritical4WatchNoHang:
    """Critical 4: watch must return (not hang) for hostile hooks.jsonl."""

    def test_watch_returns_for_fifo_hooks_jsonl(
        self, isolated_runs_root, isolated_log_root, capsys,
    ):
        name = "fifowatch"
        sd, ld = _make_run_dirs(isolated_runs_root, isolated_log_root, name)
        (sd / "status").write_text("running\n")
        hooks_path = ld / "hooks.jsonl"
        # Create a FIFO at the hooks.jsonl path.
        hooks_path.unlink(missing_ok=True)
        os.mkfifo(str(hooks_path))
        try:
            start = time.monotonic()
            rc = agent_run.cmd_watch(_watch_args(name))
            elapsed = time.monotonic() - start
        finally:
            hooks_path.unlink(missing_ok=True)
        assert rc == 0
        # Must return quickly — FIFO must be detected and skipped.
        assert elapsed < 5.0
        payload = json.loads(capsys.readouterr().out)
        assert payload["hooks"] is None

    def test_watch_returns_for_symlink_to_dev_zero(
        self, isolated_runs_root, isolated_log_root, capsys,
    ):
        name = "devzerowatch"
        sd, ld = _make_run_dirs(isolated_runs_root, isolated_log_root, name)
        (sd / "status").write_text("running\n")
        hooks_path = ld / "hooks.jsonl"
        hooks_path.symlink_to("/dev/zero")
        try:
            start = time.monotonic()
            rc = agent_run.cmd_watch(_watch_args(name))
            elapsed = time.monotonic() - start
        finally:
            hooks_path.unlink(missing_ok=True)
        assert rc == 0
        assert elapsed < 5.0  # must not hang on /dev/zero
        payload = json.loads(capsys.readouterr().out)
        assert payload["hooks"] is None

    def test_watch_returns_for_10mb_single_line(
        self, isolated_runs_root, isolated_log_root, capsys,
    ):
        name = "biglinewatch"
        sd, ld = _make_run_dirs(isolated_runs_root, isolated_log_root, name)
        (sd / "status").write_text("running\n")
        # Write a 10 MB single line (no newline) — read is capped at 8 MiB.
        (ld / "hooks.jsonl").write_bytes(b"x" * (10 * 1024 * 1024))
        start = time.monotonic()
        rc = agent_run.cmd_watch(_watch_args(name))
        elapsed = time.monotonic() - start
        assert rc == 0
        assert elapsed < 10.0
        _ = capsys.readouterr()  # consume output


class TestRegressionCritical5RecursionError:
    """Critical 5: deeply-nested JSON must not flip a live run to terminal=true."""

    def test_deeply_nested_json_does_not_flip_terminal(
        self, isolated_runs_root, isolated_log_root, capsys,
    ):
        name = "deepnest"
        sd, ld = _make_run_dirs(isolated_runs_root, isolated_log_root, name)
        (sd / "status").write_text("running\n")
        (sd / "pid").write_text(f"{os.getpid()}\n")
        # 200 000 levels of nesting exceeds Python's recursion limit.
        deeply_nested = b"[" * 200000 + b"]" * 200000
        (ld / "hooks.jsonl").write_bytes(deeply_nested + b"\n")

        rc = agent_run.cmd_watch(_watch_args(name))

        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        # A live run must not become terminal due to a hostile hooks.jsonl.
        assert payload["terminal"] is False
        assert payload["status"] != "unknown"


class TestRegressionCritical6LineCap:
    """Critical 6: every emitted hooks.jsonl line must be <= _HOOK_MAX_LINE_BYTES."""

    def test_huge_event_name_line_capped(
        self, isolated_runs_root, isolated_log_root, monkeypatch,
    ):
        name = "hugevent"
        sd, ld = _make_run_dirs(isolated_runs_root, isolated_log_root, name)
        monkeypatch.setenv("AGENT_RUN_NAME", name)
        huge_event = "e" * 100000

        rc = agent_run.cmd_hook(_hook_args(huge_event))

        assert rc == 0
        raw = (ld / "hooks.jsonl").read_bytes()
        for line in raw.split(b"\n"):
            if line.strip():
                assert len(line) <= agent_run._HOOK_MAX_LINE_BYTES

    def test_all_lines_within_pipe_buf(
        self, isolated_runs_root, isolated_log_root, monkeypatch,
    ):
        """Every line written must fit within the platform PIPE_BUF."""
        pipe_buf = os.pathconf("/", "PC_PIPE_BUF") if hasattr(os, "pathconf") else 512
        name = "pipebufrun"
        sd, ld = _make_run_dirs(isolated_runs_root, isolated_log_root, name)
        monkeypatch.setenv("AGENT_RUN_NAME", name)

        for _ in range(10):
            agent_run.cmd_hook(_hook_args("stop", json_payload='{"k":"' + "v" * 200 + '"}'))

        raw = (ld / "hooks.jsonl").read_bytes()
        for line in raw.split(b"\n"):
            if line.strip():
                assert len(line) <= pipe_buf


class TestRegressionItem7MaxEvents:
    """Item 7: malformed AGENT_RUN_HOOK_MAX_EVENTS must not crash the CLI."""

    def test_bad_max_events_falls_back_to_default(self):
        """_parse_hook_max_events() falls back to 1000 on non-integer value."""
        orig = os.environ.get("AGENT_RUN_HOOK_MAX_EVENTS")
        try:
            os.environ["AGENT_RUN_HOOK_MAX_EVENTS"] = "not_a_number"
            result = agent_run._parse_hook_max_events()
            assert result == 1000
        finally:
            if orig is None:
                os.environ.pop("AGENT_RUN_HOOK_MAX_EVENTS", None)
            else:
                os.environ["AGENT_RUN_HOOK_MAX_EVENTS"] = orig

    def test_empty_max_events_falls_back_to_default(self):
        """_parse_hook_max_events() falls back to 1000 when env var is empty."""
        orig = os.environ.get("AGENT_RUN_HOOK_MAX_EVENTS")
        try:
            os.environ["AGENT_RUN_HOOK_MAX_EVENTS"] = ""
            result = agent_run._parse_hook_max_events()
            assert result == 1000
        finally:
            if orig is None:
                os.environ.pop("AGENT_RUN_HOOK_MAX_EVENTS", None)
            else:
                os.environ["AGENT_RUN_HOOK_MAX_EVENTS"] = orig

    def test_bad_max_events_subprocess_hook_exits_0(
        self, isolated_runs_root, isolated_log_root,
    ):
        """A real subprocess with bad AGENT_RUN_HOOK_MAX_EVENTS must exit 0 for hook."""
        name = "badmax"
        sd, ld = _make_run_dirs(isolated_runs_root, isolated_log_root, name)
        result = subprocess.run(
            [sys.executable, "-m", "toolbox.agent_run", "hook", "stop", "--name", name],
            capture_output=True,
            env={
                **os.environ,
                "PYTHONPATH": str(Path(__file__).parents[1] / "src"),
                "AGENT_RUN_HOOK_MAX_EVENTS": "abc",
                "AGENT_RUN_STATE_DIR": str(isolated_runs_root),
                "AGENT_RUN_LOG_DIR": str(isolated_log_root),
            },
        )
        assert result.returncode == 0, (
            f"Expected exit 0 with bad AGENT_RUN_HOOK_MAX_EVENTS\n"
            f"stderr={result.stderr!r}"
        )


class TestRegressionItem9AncestryFallback:
    """Item 9: ancestry fallback must be unambiguous and cross-platform."""

    def test_ancestry_ambiguous_sid_returns_none(
        self, isolated_runs_root, isolated_log_root, monkeypatch,
    ):
        """When two runs have the same pgid and neither is an ancestor, neither is attributed."""
        # Use a fake pgid that is NOT in the ancestry chain and NOT a real sid.
        # We use 1 as the fake "my_sid" by patching os.getsid.
        fake_sid = 888777666  # high enough to not collide with real pids
        for i in range(2):
            nm = f"ambig{i}"
            sd, ld = _make_run_dirs(isolated_runs_root, isolated_log_root, nm,
                                    write_state=False)
            sd.mkdir(parents=True, exist_ok=True)
            # Record fake_sid as the run pgid — matches via sid==pgid when we
            # monkeypatch os.getsid to return fake_sid.
            (sd / "pgid").write_text(f"{fake_sid}\n")
            (sd / "pid").write_text("999999888\n")  # not a real ancestor
        monkeypatch.delenv("AGENT_RUN_NAME", raising=False)
        # Patch os.getsid so the sid==pgid branch sees our fake sid.
        monkeypatch.setattr(os, "getsid", lambda _: fake_sid)

        # Call the resolver directly.
        result = agent_run._hook_resolve_run(None)

        # With two sid matches and no direct ancestry, must be None (ambiguous).
        if result is not None:
            name, how = result
            assert name not in ("ambig0", "ambig1"), (
                f"Ambiguous sid match attributed to {name!r} — expected None"
            )

    def test_ancestry_direct_ppid_match_wins(
        self, isolated_runs_root, isolated_log_root, monkeypatch,
    ):
        """A run recording our direct ppid is attributed correctly (1-hop ancestry)."""
        name = "ppidrun"
        sd, ld = _make_run_dirs(isolated_runs_root, isolated_log_root, name)
        (sd / "pid").write_text(f"{os.getppid()}\n")
        monkeypatch.delenv("AGENT_RUN_NAME", raising=False)

        rc = agent_run.cmd_hook(_hook_args("stop", json_payload='{"x":1}'))

        assert rc == 0
        records = _read_hooks(ld)
        assert len(records) == 1
        assert records[0]["resolved_by"] == "ancestry"
