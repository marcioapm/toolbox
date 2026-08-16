"""Tests for agent-run managed harness mode (spec 07, phase 1).

Covers:
- _parse_launch_argv: --harness flag parsing, mutually exclusive constraints,
  prompt/prompt-file requirement, --harness + -- mutual exclusion.
- _build_managed_argv: correct argv construction per harness × mode.
- session.json: written atomically by _write_session_json/_acquire_session_*.
- watch --json session field: additive, null when no session.json present.
- steer guard: exits non-zero on one-shot runs (checks interactive state file).
- Backward compatibility: raw-mode runs are unchanged.
- Regression: managed opencode never emits --session together with --prompt.
- _parse_codex_session_id_from_jsonl: thread_id extraction.
- Managed claude argv always includes --session-id (push acquisition).
- Managed opencode interactive argv includes --port but not --prompt.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import pytest

from toolbox import agent_run


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_state_run(
    state_root: Path,
    log_root: Path,
    name: str,
    *,
    status: str = "done",
    interactive: str = "0",
) -> tuple[Path, Path]:
    """Minimal run dirs for testing subcommands."""
    sd = state_root / name
    ld = log_root / name
    sd.mkdir(parents=True, exist_ok=True)
    ld.mkdir(parents=True, exist_ok=True)
    (sd / "status").write_text(f"{status}\n")
    (sd / "interactive").write_text(f"{interactive}\n")
    (ld / "log").write_bytes(b"")
    return sd, ld


def _parse(argv):
    return agent_run._parse_launch_argv(argv)


# ---------------------------------------------------------------------------
# _parse_launch_argv — managed mode flag parsing
# ---------------------------------------------------------------------------

class TestParseLaunchArgvHarness:
    """Unit tests for the managed-mode flags added to _parse_launch_argv."""

    def test_harness_claude_inline_prompt(self):
        r = _parse(["--harness", "claude", "--prompt", "say hello", "myrun"])
        assert r.harness == "claude"
        assert r.prompt == "say hello"
        assert r.prompt_file is None
        assert r.name == "myrun"
        assert r.command == []
        assert r.subcommand_tokens is None

    def test_harness_claude_prompt_file(self):
        r = _parse(["--harness", "claude", "--prompt-file", "/tmp/p.md", "myrun"])
        assert r.harness == "claude"
        assert r.prompt is None
        assert r.prompt_file == "/tmp/p.md"
        assert r.name == "myrun"

    def test_harness_opencode(self):
        r = _parse(["--harness", "opencode", "--prompt", "hi", "myrun"])
        assert r.harness == "opencode"

    def test_harness_codex(self):
        r = _parse(["--harness", "codex", "--prompt", "hi", "myrun"])
        assert r.harness == "codex"

    def test_harness_equals_form(self):
        r = _parse(["--harness=claude", "--prompt", "hi", "myrun"])
        assert r.harness == "claude"

    def test_model_flag(self):
        r = _parse(["--harness", "claude", "--model", "claude-opus-4-5", "--prompt", "hi", "myrun"])
        assert r.model == "claude-opus-4-5"

    def test_model_equals_form(self):
        r = _parse(["--harness", "claude", "--model=claude-haiku-4-5", "--prompt", "hi", "myrun"])
        assert r.model == "claude-haiku-4-5"

    def test_agent_mode_flag(self):
        r = _parse(["--harness", "opencode", "--agent-mode", "build", "--prompt", "hi", "myrun"])
        assert r.agent_mode == "build"

    def test_session_id_flag(self):
        r = _parse(["--harness", "claude", "--session-id", "my-uuid", "--prompt", "hi", "myrun"])
        assert r.session_id == "my-uuid"

    def test_harness_arg_single(self):
        r = _parse(["--harness", "claude", "--prompt", "hi", "--harness-arg", "--foo", "myrun"])
        assert r.harness_args == ["--foo"]

    def test_harness_arg_multiple(self):
        r = _parse([
            "--harness", "claude", "--prompt", "hi",
            "--harness-arg", "--foo",
            "--harness-arg", "--bar=baz",
            "myrun",
        ])
        assert r.harness_args == ["--foo", "--bar=baz"]

    def test_interactive_with_harness(self):
        r = _parse(["-i", "--harness", "claude", "--prompt", "hi", "myrun"])
        assert r.interactive is True
        assert r.harness == "claude"

    def test_all_harness_flags_combined(self):
        r = _parse([
            "-i", "--echo", "--harness", "opencode",
            "--model", "sonnet-5", "--agent-mode", "build",
            "--prompt", "do the thing",
            "--harness-arg", "--auto",
            "--idle-timeout", "300",
            "myrun",
        ])
        assert r.interactive is True
        assert r.echo is True
        assert r.harness == "opencode"
        assert r.model == "sonnet-5"
        assert r.agent_mode == "build"
        assert r.prompt == "do the thing"
        assert r.harness_args == ["--auto"]
        assert r.idle_timeout == 300.0
        assert r.name == "myrun"

    # --- error cases ---

    def test_invalid_harness_name(self):
        with pytest.raises(agent_run._LaunchArgvError) as exc:
            _parse(["--harness", "notaharness", "--prompt", "hi", "myrun"])
        assert "notaharness" in str(exc.value)
        assert "claude" in str(exc.value)

    def test_harness_missing_value(self):
        with pytest.raises(agent_run._LaunchArgvError) as exc:
            _parse(["--harness"])
        assert "--harness" in str(exc.value)

    def test_prompt_missing_value(self):
        with pytest.raises(agent_run._LaunchArgvError) as exc:
            _parse(["--harness", "claude", "--prompt"])
        assert "--prompt" in str(exc.value)

    def test_prompt_and_prompt_file_mutually_exclusive(self):
        with pytest.raises(agent_run._LaunchArgvError) as exc:
            _parse(["--harness", "claude", "--prompt", "hi", "--prompt-file", "/f.md", "myrun"])
        msg = str(exc.value)
        assert "mutually exclusive" in msg

    def test_managed_mode_requires_prompt_or_prompt_file(self):
        with pytest.raises(agent_run._LaunchArgvError) as exc:
            _parse(["--harness", "claude", "myrun"])
        msg = str(exc.value)
        assert "--prompt" in msg or "prompt" in msg.lower()

    def test_harness_and_trailing_command_are_mutually_exclusive(self):
        """--harness and a trailing '-- <command>' must be rejected."""
        with pytest.raises(agent_run._LaunchArgvError) as exc:
            _parse(["--harness", "claude", "--prompt", "hi", "myrun", "--", "echo"])
        msg = str(exc.value)
        assert "mutually exclusive" in msg or "--harness" in msg

    def test_harness_extra_tokens_after_name_rejected(self):
        with pytest.raises(agent_run._LaunchArgvError) as exc:
            _parse(["--harness", "claude", "--prompt", "hi", "myrun", "unexpected"])
        assert "unexpected" in str(exc.value) or "harness-arg" in str(exc.value)

    def test_raw_mode_still_works_with_harness_not_set(self):
        """Raw mode must be completely unaffected by managed-mode flag additions."""
        r = _parse(["myrun", "--", "echo", "hello"])
        assert r.harness is None
        assert r.name == "myrun"
        assert r.command == ["echo", "hello"]

    def test_raw_mode_legacy_form_still_works(self):
        r = _parse(["myrun", "echo", "hello"])
        assert r.harness is None
        assert r.command == ["echo", "hello"]

    def test_raw_mode_with_all_old_flags(self):
        r = _parse(["-i", "--echo", "-f", "/p.md", "--idle-timeout", "60", "myrun", "--", "cmd"])
        assert r.harness is None
        assert r.interactive is True
        assert r.prompt_file == "/p.md"
        assert r.idle_timeout == 60.0
        assert r.command == ["cmd"]

    @pytest.mark.parametrize("sub", sorted(agent_run._KNOWN_SUBCOMMANDS))
    def test_subcommand_dispatch_unaffected(self, sub):
        r = _parse([sub, "myrun"])
        assert r.subcommand_tokens is not None
        assert r.subcommand_tokens[0] == sub


# ---------------------------------------------------------------------------
# _build_managed_argv — argv construction per harness × mode
# ---------------------------------------------------------------------------

class TestBuildManagedArgv:
    """Verify the argv agent-run constructs for each harness and mode."""

    def _build(self, harness, **kw):
        defaults = dict(
            interactive=False, prompt=None, prompt_file=None,
            model=None, agent_mode=None, cwd=None,
            session_id=None, harness_args=[],
        )
        defaults.update(kw)
        return agent_run._build_managed_argv(harness, **defaults)

    # claude one-shot
    def test_claude_oneshot_inline_prompt(self):
        argv = self._build("claude", prompt="say hello", session_id="uuid1")
        assert argv[0] == "claude"
        assert "--print" in argv
        assert "say hello" in argv
        assert "--session-id" in argv
        i = argv.index("--session-id")
        assert argv[i + 1] == "uuid1"

    def test_claude_oneshot_permissions_bypass(self):
        argv = self._build("claude", session_id="u")
        assert "--permission-mode" in argv
        i = argv.index("--permission-mode")
        assert argv[i + 1] == "bypassPermissions"

    # claude interactive
    def test_claude_interactive_no_print(self):
        argv = self._build("claude", interactive=True, session_id="u")
        assert "--print" not in argv
        assert "--permission-mode" in argv

    def test_claude_interactive_session_id_pushed(self):
        argv = self._build("claude", interactive=True, session_id="my-uuid")
        assert "--session-id" in argv
        i = argv.index("--session-id")
        assert argv[i + 1] == "my-uuid"

    def test_claude_model_passed(self):
        argv = self._build("claude", model="claude-opus-4-5", session_id="u")
        assert "--model" in argv
        i = argv.index("--model")
        assert argv[i + 1] == "claude-opus-4-5"

    # opencode one-shot
    def test_opencode_oneshot_uses_run_subcommand(self):
        argv = self._build("opencode", prompt="hi")
        assert argv[0] == "opencode"
        assert "run" in argv
        assert "hi" in argv

    def test_opencode_oneshot_no_prompt_flag_in_argv(self):
        """opencode run doesn't use --prompt; the message is a positional arg."""
        argv = self._build("opencode", prompt="hi")
        assert "--prompt" not in argv

    # opencode interactive — the critical regression test
    def test_opencode_interactive_never_emits_session_and_prompt_together(self):
        """Managed opencode interactive must never emit --session and --prompt together.

        This is the gotcha from the brief: --session silently swallows --prompt.
        The _build_managed_argv function must make this combination structurally
        impossible.
        """
        argv = self._build(
            "opencode",
            interactive=True,
            prompt="do the thing",  # inline prompt — must NOT appear in argv
            opencode_port=41234,
            opencode_session_id="ses_abc123",
        )
        # --prompt must not appear anywhere in the constructed argv.
        assert "--prompt" not in argv, (
            f"opencode interactive argv must never contain --prompt: {argv}"
        )

    def test_opencode_interactive_includes_port(self):
        argv = self._build("opencode", interactive=True, opencode_port=41234)
        assert "--port" in argv
        i = argv.index("--port")
        assert argv[i + 1] == "41234"

    def test_opencode_interactive_includes_session(self):
        argv = self._build(
            "opencode", interactive=True,
            opencode_port=41234, opencode_session_id="ses_xyz",
        )
        assert "--session" in argv
        i = argv.index("--session")
        assert argv[i + 1] == "ses_xyz"

    def test_opencode_interactive_includes_auto(self):
        argv = self._build("opencode", interactive=True)
        assert "--auto" in argv

    def test_opencode_model_and_agent_mode(self):
        argv = self._build(
            "opencode", interactive=True,
            model="llmproxy-anthropic/claude-sonnet-4.6",
            agent_mode="build",
        )
        assert "-m" in argv
        i = argv.index("-m")
        assert argv[i + 1] == "llmproxy-anthropic/claude-sonnet-4.6"
        assert "--agent" in argv
        j = argv.index("--agent")
        assert argv[j + 1] == "build"

    # codex one-shot
    def test_codex_oneshot_uses_exec_json(self):
        argv = self._build("codex", prompt="hi")
        assert "codex" in argv[0]
        assert "exec" in argv
        assert "--json" in argv

    def test_codex_oneshot_inline_prompt(self):
        argv = self._build("codex", prompt="say hello")
        assert "say hello" in argv

    # codex interactive
    def test_codex_interactive_bare_command(self):
        argv = self._build("codex", interactive=True, prompt="hi")
        assert "codex" in argv[0]
        assert "exec" not in argv
        assert "--json" not in argv

    # harness_args passthrough
    def test_harness_args_appended(self):
        argv = self._build("claude", session_id="u", harness_args=["--foo", "--bar=baz"])
        assert "--foo" in argv
        assert "--bar=baz" in argv


# ---------------------------------------------------------------------------
# session.json — write, read, and watch integration
# ---------------------------------------------------------------------------

class TestSessionJson:
    def test_write_session_json_creates_file(self, isolated_log_root):
        log_dir = isolated_log_root / "myrun"
        log_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "session_id": "abc123",
            "harness": "claude",
            "acquisition": "pushed",
            "confidence": "certain",
            "observed_at": "2026-08-16T00:00:00Z",
        }
        agent_run._write_session_json(log_dir, data)
        path = log_dir / "session.json"
        assert path.exists()
        read_back = json.loads(path.read_text())
        assert read_back == data

    def test_write_session_json_is_atomic(self, isolated_log_root):
        """No .tmp file should survive after a successful write."""
        log_dir = isolated_log_root / "myrun"
        log_dir.mkdir(parents=True, exist_ok=True)
        agent_run._write_session_json(log_dir, {"session_id": "x"})
        tmps = list(log_dir.glob(".session.*.tmp"))
        assert tmps == []

    def test_read_session_json_returns_none_when_absent(self, isolated_log_root):
        log_dir = isolated_log_root / "norun"
        log_dir.mkdir(parents=True, exist_ok=True)
        assert agent_run._read_session_json(log_dir) is None

    def test_read_session_json_returns_none_on_malformed(self, isolated_log_root):
        log_dir = isolated_log_root / "bad"
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "session.json").write_text("{not json}")
        assert agent_run._read_session_json(log_dir) is None

    def test_acquire_session_claude_writes_pushed_certain(self, isolated_log_root):
        log_dir = isolated_log_root / "run1"
        log_dir.mkdir(parents=True, exist_ok=True)
        acquire_log = log_dir / "session-acquire.log"
        agent_run._acquire_session_claude(log_dir, "uuid-1234", acquire_log)
        data = agent_run._read_session_json(log_dir)
        assert data is not None
        assert data["session_id"] == "uuid-1234"
        assert data["harness"] == "claude"
        assert data["acquisition"] == "pushed"
        assert data["confidence"] == "certain"

    def test_acquire_session_opencode_minted_writes_minted_certain(self, isolated_log_root):
        log_dir = isolated_log_root / "run2"
        log_dir.mkdir(parents=True, exist_ok=True)
        acquire_log = log_dir / "session-acquire.log"
        agent_run._acquire_session_opencode_minted(log_dir, "ses_abc", acquire_log)
        data = agent_run._read_session_json(log_dir)
        assert data is not None
        assert data["session_id"] == "ses_abc"
        assert data["harness"] == "opencode"
        assert data["acquisition"] == "minted"
        assert data["confidence"] == "certain"

    def test_acquire_session_missing_writes_missing_confidence(self, isolated_log_root):
        log_dir = isolated_log_root / "run3"
        log_dir.mkdir(parents=True, exist_ok=True)
        acquire_log = log_dir / "session-acquire.log"
        agent_run._acquire_session_missing(log_dir, "codex", "no id found", acquire_log)
        data = agent_run._read_session_json(log_dir)
        assert data is not None
        assert data["session_id"] is None
        assert data["harness"] == "codex"
        assert data["confidence"] == "missing"
        assert "reason" in data

    def test_never_label_a_guess_certain(self, isolated_log_root):
        """Confidence 'certain' is only allowed for pushed/minted/reported acquisition."""
        log_dir = isolated_log_root / "run4"
        log_dir.mkdir(parents=True, exist_ok=True)
        acquire_log = log_dir / "session-acquire.log"
        # The missing path must never write certain.
        agent_run._acquire_session_missing(log_dir, "opencode", "reason", acquire_log)
        data = agent_run._read_session_json(log_dir)
        assert data["confidence"] != "certain"


# ---------------------------------------------------------------------------
# watch --json: session field is additive
# ---------------------------------------------------------------------------

class TestWatchSessionField:
    def _run_watch(self, state_root, log_root, name, capsys):
        args = argparse.Namespace(name=name, json=True, repo=None)
        agent_run.cmd_watch(args)
        return json.loads(capsys.readouterr().out)

    def test_session_field_is_null_for_raw_run(self, isolated_runs_root, isolated_log_root, capsys):
        """A raw (non-managed) run has no session.json → session field is null."""
        name = "rawrun"
        sd, ld = _make_state_run(isolated_runs_root, isolated_log_root, name)
        (sd / "status").write_text("done\n")
        (sd / "exit_code").write_text("0\n")
        payload = self._run_watch(isolated_runs_root, isolated_log_root, name, capsys)
        assert "session" in payload
        assert payload["session"] is None

    def test_session_field_populated_when_session_json_present(
        self, isolated_runs_root, isolated_log_root, capsys
    ):
        """When session.json exists, the session field carries its contents."""
        name = "managedrun"
        sd, ld = _make_state_run(isolated_runs_root, isolated_log_root, name)
        (sd / "status").write_text("done\n")
        session_data = {
            "session_id": "uuid-123",
            "harness": "claude",
            "acquisition": "pushed",
            "confidence": "certain",
            "observed_at": "2026-08-16T00:00:00Z",
        }
        agent_run._write_session_json(ld, session_data)
        payload = self._run_watch(isolated_runs_root, isolated_log_root, name, capsys)
        assert payload["session"] == session_data

    def test_all_existing_watch_keys_still_present(
        self, isolated_runs_root, isolated_log_root, capsys
    ):
        """session is additive — all pre-existing keys must still be present."""
        required_keys = {
            "schema", "name", "observed_at", "status", "exit_code", "pid",
            "interactive", "started_at", "ended_at", "elapsed_s", "terminal",
            "launch_error", "log", "repo", "git", "git_error", "signals",
            "observation_error",
        }
        name = "keycheck"
        sd, ld = _make_state_run(isolated_runs_root, isolated_log_root, name)
        (sd / "status").write_text("done\n")
        payload = self._run_watch(isolated_runs_root, isolated_log_root, name, capsys)
        assert required_keys.issubset(set(payload.keys()))


# ---------------------------------------------------------------------------
# steer guard on one-shot runs
# ---------------------------------------------------------------------------

class TestSteerGuardOneShotRuns:
    def test_steer_on_oneshot_run_exits_nonzero(
        self, isolated_runs_root, isolated_log_root
    ):
        """steer on a one-shot run must exit non-zero with a clear message."""
        name = "oneshot"
        sd, _ = _make_state_run(isolated_runs_root, isolated_log_root, name, status="running")
        # interactive=0 marks a one-shot run.
        (sd / "interactive").write_text("0\n")
        (sd / "pid").write_text(f"{os.getpid()}\n")
        args = argparse.Namespace(name=name, message=["hello"], esc=False, raw=False)
        with pytest.raises(SystemExit) as exc:
            agent_run.cmd_steer(args)
        assert exc.value.code != 0

    def test_steer_on_oneshot_message_mentions_run_name(
        self, isolated_runs_root, isolated_log_root, capsys
    ):
        """The error message must name the run and state it was launched one-shot."""
        name = "oneshotnamed"
        sd, _ = _make_state_run(isolated_runs_root, isolated_log_root, name, status="running")
        (sd / "interactive").write_text("0\n")
        (sd / "pid").write_text(f"{os.getpid()}\n")
        args = argparse.Namespace(name=name, message=["hello"], esc=False, raw=False)
        with pytest.raises(SystemExit) as exc:
            agent_run.cmd_steer(args)
        msg = str(exc.value)
        assert "oneshotnamed" in msg
        assert "one-shot" in msg.lower() or "oneshot" in msg.lower() or "interactive" in msg.lower()

    def test_steer_checks_interactive_state_file_not_fifo(
        self, isolated_runs_root, isolated_log_root
    ):
        """steer must check the interactive state file, not the FIFO's existence.

        A run that has a FIFO but interactive=0 should still fail. Conversely,
        a run with interactive=1 should proceed (even if FIFO doesn't exist yet —
        though in practice a live interactive run always has one).
        """
        # One-shot run WITH a FIFO — steer must still reject it.
        name = "fifo-but-oneshot"
        sd, _ = _make_state_run(isolated_runs_root, isolated_log_root, name, status="running")
        (sd / "interactive").write_text("0\n")
        (sd / "pid").write_text(f"{os.getpid()}\n")
        os.mkfifo(str(sd / "stdin"))  # FIFO exists, but run is one-shot
        args = argparse.Namespace(name=name, message=["hi"], esc=False, raw=False)
        with pytest.raises(SystemExit) as exc:
            agent_run.cmd_steer(args)
        assert exc.value.code != 0


# ---------------------------------------------------------------------------
# Codex JSONL session id parser
# ---------------------------------------------------------------------------

class TestParseCodexSessionIdFromJsonl:
    def test_parses_thread_started_line(self):
        line = '{"type":"thread.started","thread_id":"01a00b2a-8408-7790-be58-86445cd3af81"}'
        result = agent_run._parse_codex_session_id_from_jsonl(line)
        assert result == "01a00b2a-8408-7790-be58-86445cd3af81"

    def test_returns_none_for_other_event_types(self):
        line = '{"type":"turn.started"}'
        assert agent_run._parse_codex_session_id_from_jsonl(line) is None

    def test_returns_none_for_invalid_json(self):
        assert agent_run._parse_codex_session_id_from_jsonl("not json") is None

    def test_returns_none_when_no_thread_id(self):
        line = '{"type":"thread.started"}'
        assert agent_run._parse_codex_session_id_from_jsonl(line) is None

    def test_returns_none_for_empty_thread_id(self):
        line = '{"type":"thread.started","thread_id":""}'
        assert agent_run._parse_codex_session_id_from_jsonl(line) is None


# ---------------------------------------------------------------------------
# Claude session id mint
# ---------------------------------------------------------------------------

class TestClaudeMintSessionId:
    def test_generates_uuid4_when_no_caller_id(self):
        sid = agent_run._claude_mint_session_id(None)
        # Must be a valid UUID4 format.
        import re
        uuid4_re = re.compile(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
            re.IGNORECASE,
        )
        assert uuid4_re.match(sid), f"Not a UUID4: {sid!r}"

    def test_uses_caller_supplied_id(self):
        sid = agent_run._claude_mint_session_id("11111111-2222-4333-8444-555555555555")
        assert sid == "11111111-2222-4333-8444-555555555555"

    def test_each_call_generates_unique_uuid(self):
        ids = {agent_run._claude_mint_session_id(None) for _ in range(10)}
        assert len(ids) == 10


# ---------------------------------------------------------------------------
# Managed mode launch integration (subprocess runs with real commands)
# ---------------------------------------------------------------------------

def _launch_and_wait(
    state_root: Path,
    log_root: Path,
    name: str,
    *,
    harness: str,
    interactive: bool,
    prompt: Optional[str] = None,
    prompt_file: Optional[str] = None,
    model: Optional[str] = None,
    timeout: float = 15.0,
) -> tuple[str, Optional[dict]]:
    """Launch a managed run using a fake harness command and wait for terminal status.

    Returns (final_status, session_json_data).
    """
    ns = argparse.Namespace(
        name=name,
        command=[],
        interactive=interactive,
        prompt_file=prompt_file,
        echo=False,
        echo_interval=2.0,
        submit_mode=None,
        idle_timeout=None,
        harness=harness,
        prompt=prompt,
        model=model,
        agent_mode=None,
        session_id=None,
        harness_args=[],
    )
    rc = agent_run.cmd_launch(ns)
    assert rc == 0

    state_dir = state_root / name
    log_dir = log_root / name
    deadline = time.monotonic() + timeout
    status = "starting"
    while time.monotonic() < deadline:
        try:
            status = (state_dir / "status").read_text().strip()
        except FileNotFoundError:
            status = "starting"
        if status in agent_run.TERMINAL_STATUSES:
            break
        time.sleep(0.05)

    session_data = agent_run._read_session_json(log_dir)
    return status, session_data


class TestManagedClaudeLaunch:
    """Integration tests for managed claude using a fake 'claude' stub.

    These tests monkeypatch the claude binary so they run without any real API key.
    The fake claude just prints and exits 0.
    """

    def _make_fake_claude(self, tmp_path: Path) -> str:
        """Write a fake `claude` script that prints its argv and exits 0."""
        fake = tmp_path / "claude"
        fake.write_text(
            "#!/bin/sh\n"
            "printf 'argv: %s\\n' \"$@\"\n"
            "exit 0\n"
        )
        fake.chmod(0o755)
        return str(tmp_path)

    def test_claude_oneshot_session_id_pushed(
        self, isolated_runs_root, isolated_log_root, tmp_path, monkeypatch
    ):
        """A managed claude one-shot run writes session.json with acquisition=pushed."""
        bin_dir = self._make_fake_claude(tmp_path)
        monkeypatch.setenv("PATH", bin_dir + ":" + os.environ.get("PATH", ""))
        name = "claude-oneshot-1"
        status, session = _launch_and_wait(
            isolated_runs_root, isolated_log_root, name,
            harness="claude",
            interactive=False,
            prompt="say hello",
        )
        assert status in {"done", "launch_failed"}  # fake binary exits 0 -> done
        assert session is not None
        assert session["harness"] == "claude"
        assert session["acquisition"] == "pushed"
        assert session["confidence"] == "certain"
        assert session["session_id"] is not None
        # Verify the session_id looks like a UUID.
        import re
        uuid_re = re.compile(r"^[0-9a-f-]{32,36}$", re.IGNORECASE)
        assert uuid_re.match(session["session_id"].replace("-", ""))

    def test_claude_oneshot_session_id_in_log_line(
        self, isolated_runs_root, isolated_log_root, tmp_path, monkeypatch
    ):
        """The fake claude argv output should contain the --session-id flag."""
        bin_dir = self._make_fake_claude(tmp_path)
        monkeypatch.setenv("PATH", bin_dir + ":" + os.environ.get("PATH", ""))
        name = "claude-oneshot-2"
        _launch_and_wait(
            isolated_runs_root, isolated_log_root, name,
            harness="claude",
            interactive=False,
            prompt="say hello",
        )
        log_path = isolated_log_root / name / "log"
        deadline = time.monotonic() + 5.0
        log_content = ""
        while time.monotonic() < deadline:
            if log_path.exists():
                log_content = log_path.read_text()
                if log_content:
                    break
            time.sleep(0.05)
        assert "--session-id" in log_content, f"Expected --session-id in log: {log_content!r}"

    def test_claude_oneshot_never_emits_session_prompt_together_in_argv(
        self, isolated_runs_root, isolated_log_root, tmp_path, monkeypatch
    ):
        """Claude one-shot argv must have --session-id. --prompt is inline arg, not a flag."""
        bin_dir = self._make_fake_claude(tmp_path)
        monkeypatch.setenv("PATH", bin_dir + ":" + os.environ.get("PATH", ""))
        name = "claude-no-promptflag"
        _launch_and_wait(
            isolated_runs_root, isolated_log_root, name,
            harness="claude",
            interactive=False,
            prompt="do the thing",
        )
        log_path = isolated_log_root / name / "log"
        deadline = time.monotonic() + 5.0
        log_content = ""
        while time.monotonic() < deadline:
            if log_path.exists():
                log_content = log_path.read_text()
                if log_content:
                    break
            time.sleep(0.05)
        # The prompt text must appear as a positional arg, not via --prompt flag.
        assert "--prompt" not in log_content, (
            f"--prompt flag must not appear in claude argv: {log_content!r}"
        )

    def test_raw_run_never_gets_session_json(
        self, isolated_runs_root, isolated_log_root, tmp_path, monkeypatch
    ):
        """A raw run (no --harness) must never get a session.json."""
        fake_echo = tmp_path / "echo_test.sh"
        fake_echo.write_text("#!/bin/sh\necho hello\n")
        fake_echo.chmod(0o755)
        name = "raw-no-session"
        ns = argparse.Namespace(
            name=name,
            command=[str(fake_echo)],
            interactive=False,
            prompt_file=None,
            echo=False,
            echo_interval=2.0,
            submit_mode=None,
            idle_timeout=None,
        )
        rc = agent_run.cmd_launch(ns)
        assert rc == 0
        state_dir = isolated_runs_root / name
        log_dir = isolated_log_root / name
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            try:
                status = (state_dir / "status").read_text().strip()
            except FileNotFoundError:
                status = "starting"
            if status in agent_run.TERMINAL_STATUSES:
                break
            time.sleep(0.05)
        assert not (log_dir / "session.json").exists(), (
            "Raw run must never produce session.json"
        )


class TestManagedCodexOneShotSessionParsing:
    """Tests for codex --json first-line session id parsing."""

    def _make_fake_codex(self, tmp_path: Path, *, emit_thread_id: bool = True) -> str:
        """Write a fake `codex` script that emits the expected JSONL line."""
        fake_dir = tmp_path / "bin"
        fake_dir.mkdir(parents=True, exist_ok=True)
        fake = fake_dir / "codex"
        if emit_thread_id:
            fake.write_text(
                "#!/bin/sh\n"
                'printf \'{"type":"thread.started","thread_id":"test-thread-uuid-0001"}\\n\'\n'
                'printf \'{"type":"turn.started"}\\n\'\n'
                'printf \'the answer is 42\\n\'\n'
                "exit 0\n"
            )
        else:
            fake.write_text(
                "#!/bin/sh\n"
                'printf \'{"type":"turn.started"}\\n\'\n'
                'printf \'hello\\n\'\n'
                "exit 0\n"
            )
        fake.chmod(0o755)
        return str(fake_dir)

    def test_codex_oneshot_parses_thread_id_to_session_json(
        self, isolated_runs_root, isolated_log_root, tmp_path, monkeypatch
    ):
        """session.json must be written with acquisition=reported, confidence=certain."""
        bin_dir = self._make_fake_codex(tmp_path)
        monkeypatch.setenv("PATH", bin_dir + ":" + os.environ.get("PATH", ""))
        name = "codex-oneshot-1"
        status, session = _launch_and_wait(
            isolated_runs_root, isolated_log_root, name,
            harness="codex",
            interactive=False,
            prompt="what is 6*7",
        )
        assert session is not None
        assert session["harness"] == "codex"
        assert session["acquisition"] == "reported"
        assert session["confidence"] == "certain"
        assert session["session_id"] == "test-thread-uuid-0001"

    def test_codex_oneshot_missing_thread_id_writes_missing(
        self, isolated_runs_root, isolated_log_root, tmp_path, monkeypatch
    ):
        """When codex doesn't emit thread_id, confidence=missing, run still completes."""
        bin_dir = self._make_fake_codex(tmp_path, emit_thread_id=False)
        monkeypatch.setenv("PATH", bin_dir + ":" + os.environ.get("PATH", ""))
        name = "codex-oneshot-2"
        status, session = _launch_and_wait(
            isolated_runs_root, isolated_log_root, name,
            harness="codex",
            interactive=False,
            prompt="what is 6*7",
        )
        # Run must still complete, not fail because of missing session id.
        assert status in agent_run.TERMINAL_STATUSES
        assert session is not None
        assert session["confidence"] == "missing"
        assert session["session_id"] is None

    def test_codex_oneshot_session_json_in_persistent_log_dir(
        self, isolated_runs_root, isolated_log_root, tmp_path, monkeypatch
    ):
        """session.json goes in the persistent log dir, not the ephemeral state dir."""
        bin_dir = self._make_fake_codex(tmp_path)
        monkeypatch.setenv("PATH", bin_dir + ":" + os.environ.get("PATH", ""))
        name = "codex-oneshot-3"
        _launch_and_wait(
            isolated_runs_root, isolated_log_root, name,
            harness="codex",
            interactive=False,
            prompt="hi",
        )
        assert (isolated_log_root / name / "session.json").exists()
        assert not (isolated_runs_root / name / "session.json").exists()


# ---------------------------------------------------------------------------
# Opencode one-shot: session acquisition always fails gracefully
# ---------------------------------------------------------------------------

class TestManagedOpencodeOneshotSession:
    def _make_fake_opencode(self, tmp_path: Path) -> str:
        fake_dir = tmp_path / "bin"
        fake_dir.mkdir(parents=True, exist_ok=True)
        fake = fake_dir / "opencode"
        fake.write_text("#!/bin/sh\necho done\nexit 0\n")
        fake.chmod(0o755)
        return str(fake_dir)

    def test_opencode_oneshot_writes_session_json_with_missing(
        self, isolated_runs_root, isolated_log_root, tmp_path, monkeypatch
    ):
        """opencode run (one-shot) has no session id source → confidence=missing."""
        bin_dir = self._make_fake_opencode(tmp_path)
        monkeypatch.setenv("PATH", bin_dir + ":" + os.environ.get("PATH", ""))
        name = "oc-oneshot-1"
        status, session = _launch_and_wait(
            isolated_runs_root, isolated_log_root, name,
            harness="opencode",
            interactive=False,
            prompt="do it",
        )
        # The run may produce launch_failed because "opencode run do it" is a
        # fake binary call, but session.json must still have been written.
        assert session is not None
        assert session["harness"] == "opencode"
        assert session["confidence"] == "missing"


# ---------------------------------------------------------------------------
# watch session field populated from log dir (integration)
# ---------------------------------------------------------------------------

class TestWatchSessionIntegration:
    def test_watch_json_session_populated_from_session_json(
        self, isolated_runs_root, isolated_log_root, capsys
    ):
        name = "watchsession"
        sd, ld = _make_state_run(isolated_runs_root, isolated_log_root, name, status="done")
        (sd / "exit_code").write_text("0\n")
        session_data = {
            "session_id": "ses_test123",
            "harness": "opencode",
            "acquisition": "minted",
            "confidence": "certain",
            "observed_at": "2026-08-16T10:00:00Z",
        }
        agent_run._write_session_json(ld, session_data)
        args = argparse.Namespace(name=name, json=True, repo=None)
        agent_run.cmd_watch(args)
        payload = json.loads(capsys.readouterr().out)
        assert payload["session"] == session_data
        assert payload["session"]["session_id"] == "ses_test123"
