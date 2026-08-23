"""Tests for agent-run managed harness mode (spec 07, phase 1).

Covers:
- _parse_launch_argv: --harness flag parsing, mutually exclusive constraints,
  prompt/prompt-file requirement, --harness + -- mutual exclusion.
- _build_parser: managed-mode options (--harness, --permissions, etc.) appear
  in --help output and do not break subcommand dispatch.
- _build_managed_argv: correct argv construction per harness × mode.
- session.json: written atomically by _write_session_json/_record_session.
- watch --json session field: additive, null when no session.json present.
- steer guard: exits non-zero on one-shot runs (checks interactive state file).
- Backward compatibility: raw-mode runs are unchanged.
- Regression: managed opencode never emits --session together with --prompt.
- _parse_codex_session_id_from_jsonl: thread_id extraction.
- Managed claude argv always includes --session-id (push acquisition).
- Managed opencode interactive argv includes --port, --session, but not --prompt.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid as _uuid
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

    def test_model_flag_accepted_for_codex(self):
        r = _parse(["--harness", "codex", "--model", "o4-mini", "--prompt", "hi", "myrun"])
        assert r.harness == "codex"
        assert r.model == "o4-mini"

    def test_agent_mode_flag(self):
        r = _parse(["--harness", "opencode", "--agent-mode", "build", "--prompt", "hi", "myrun"])
        assert r.agent_mode == "build"



    def test_harness_arg_single(self):
        r = _parse(["--harness", "claude", "--prompt", "hi", "--harness-arg", "--foo", "myrun"])
        assert list(r.harness_args) == ["--foo"]

    def test_harness_arg_multiple(self):
        r = _parse([
            "--harness", "claude", "--prompt", "hi",
            "--harness-arg", "--foo",
            "--harness-arg", "--bar=baz",
            "myrun",
        ])
        assert list(r.harness_args) == ["--foo", "--bar=baz"]

    @pytest.mark.parametrize("flag", ["--session", "-s", "--session-id", "--port", "--prompt"])
    def test_harness_arg_rejects_flags_agent_run_owns(self, flag):
        # --session/-s/--session-id/--port carry the identity agent-run mints and
        # must not be forgeable; --prompt has an agent-run flag of its own. The
        # "flag=value" spelling splits to the same name, so it is guarded too.
        with pytest.raises(agent_run._LaunchArgvError) as exc:
            _parse([
                "--harness", "claude", "--prompt", "hi",
                "--harness-arg", f"{flag}=x", "myrun",
            ])
        assert flag in str(exc.value)


    def test_interactive_with_harness(self):
        r = _parse(["-i", "--harness", "claude", "--prompt", "hi", "myrun"])
        assert r.interactive is True
        assert r.harness == "claude"

    def test_capability_flags_require_harness(self):
        with pytest.raises(agent_run._LaunchArgvError):
            _parse(["--enable-planning", "myrun", "--", "echo", "hi"])

    def test_capability_flags_parse_for_managed_harnesses(self):
        r = _parse([
            "--harness", "opencode", "--enable-planning", "--enable-questions",
            "--prompt", "hi", "myrun",
        ])
        assert r.enable_planning is True
        assert r.enable_questions is True

    def test_codex_enable_planning_fails_fast(self):
        with pytest.raises(agent_run._LaunchArgvError, match="unsupported.*codex"):
            _parse(["--harness", "codex", "--enable-planning", "--prompt", "hi", "myrun"])

    def test_claude_permission_mode_plan_rejected_when_planning_disabled(self):
        """Two-token --permission-mode plan would start Claude in plan mode,
        bypassing the EnterPlanMode/ExitPlanMode denies."""
        with pytest.raises(agent_run._LaunchArgvError, match="plan mode"):
            _parse([
                "--harness", "claude", "--prompt", "hi",
                "--harness-arg", "--permission-mode",
                "--harness-arg", "plan",
                "myrun",
            ])

    def test_claude_permission_mode_equals_plan_rejected_when_planning_disabled(self):
        with pytest.raises(agent_run._LaunchArgvError, match="plan mode"):
            _parse([
                "--harness", "claude", "--prompt", "hi",
                "--harness-arg", "--permission-mode=plan",
                "myrun",
            ])

    def test_claude_permission_mode_plan_allowed_with_enable_planning(self):
        r = _parse([
            "--harness", "claude", "--enable-planning",
            "--prompt", "hi",
            "--harness-arg", "--permission-mode",
            "--harness-arg", "plan",
            "myrun",
        ])
        assert r.enable_planning is True
        assert list(r.harness_args) == ["--permission-mode", "plan"]

    def test_claude_permission_mode_bypass_not_rejected(self):
        """Only the 'plan' value is rejected; other --permission-mode values pass."""
        r = _parse([
            "--harness", "claude", "--prompt", "hi",
            "--harness-arg", "--permission-mode",
            "--harness-arg", "bypassPermissions",
            "myrun",
        ])
        assert "--permission-mode" in list(r.harness_args)


    def test_all_harness_flags_combined(self):
        r = _parse([
            "-i", "--harness", "opencode",
            "--model", "sonnet-5", "--agent-mode", "build",
            "--prompt", "do the thing",
            "--harness-arg", "--auto",
            "--idle-timeout", "300",
            "myrun",
        ])
        assert r.interactive is True
        assert r.harness == "opencode"
        assert r.model == "sonnet-5"
        assert r.agent_mode == "build"
        assert r.prompt == "do the thing"
        assert list(r.harness_args) == ["--auto"]
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
        r = _parse(["-i", "-f", "/p.md", "--idle-timeout", "60", "myrun", "--", "cmd"])
        assert r.harness is None
        assert r.interactive is True
        assert r.prompt_file == "/p.md"
        assert r.idle_timeout == 60.0
        assert r.command == ["cmd"]

    def test_raw_mode_rejects_managed_only_flags(self):
        """Managed-only flags in a raw launch are rejected, not silently swallowed."""
        with pytest.raises(agent_run._LaunchArgvError):
            _parse(["--prompt", "hi", "myrun", "--", "echo"])

    def test_raw_mode_rejects_model_flag(self):
        with pytest.raises(agent_run._LaunchArgvError):
            _parse(["--model", "foo", "myrun", "--", "echo"])

    def test_cwd_flag_parsed_in_managed_mode(self):
        r = _parse(["--harness", "claude", "--prompt", "hi", "--cwd", "/tmp", "myrun"])
        assert r.cwd == "/tmp"
        assert r.name == "myrun"


    @pytest.mark.parametrize("sub", sorted(agent_run._KNOWN_SUBCOMMANDS))
    def test_subcommand_dispatch_unaffected(self, sub):
        r = _parse([sub, "myrun"])
        assert r.subcommand_tokens is not None
        assert r.subcommand_tokens[0] == sub


# ---------------------------------------------------------------------------
# _build_parser help text — managed-mode options must be discoverable
# ---------------------------------------------------------------------------

class TestParserHelpShowsManagedMode:
    """_build_parser() must register managed-mode options so they appear in
    --help output.  Absence silently hides a user-facing feature; this class
    prevents that regression.
    """

    def _help_text(self) -> str:
        return agent_run._build_parser().format_help()

    def test_harness_flag_in_help(self):
        assert "--harness" in self._help_text()

    def test_permissions_flag_in_help(self):
        assert "--permissions" in self._help_text()

    def test_prompt_flag_in_help(self):
        assert "--prompt" in self._help_text()

    def test_prompt_file_flag_in_help(self):
        assert "--prompt-file" in self._help_text()

    def test_model_flag_in_help(self):
        assert "--model" in self._help_text()

    def test_agent_mode_flag_in_help(self):
        assert "--agent-mode" in self._help_text()

    def test_harness_arg_flag_in_help(self):
        assert "--harness-arg" in self._help_text()

    @pytest.mark.parametrize("flag", ["--enable-planning", "--enable-questions"])
    def test_capability_flags_in_help(self, flag):
        assert flag in self._help_text()


    def test_raw_usage_line_present(self):
        # The raw-mode usage line must remain visible so the existing contract
        # is not obscured by the managed-mode additions.
        assert "NAME -- <cmd" in self._help_text()

    def test_bypass_is_documented_as_default(self):
        # --permissions bypass must be described as the default so users know
        # they get bypassPermissions without specifying anything.
        help_text = self._help_text()
        assert "bypass" in help_text
        assert "bypassPermissions" in help_text

    def test_subcommand_dispatch_unaffected_by_new_args(self):
        # Registering managed-mode flags in the top-level parser must not break
        # subcommand dispatch: parse_args(['status', 'myrun']) must still work.
        ns = agent_run._build_parser().parse_args(["status", "myrun"])
        assert ns.sub == "status"
        assert ns.name == "myrun"


# ---------------------------------------------------------------------------
# _build_managed_argv — argv construction per harness × mode
# ---------------------------------------------------------------------------

class TestBuildManagedArgv:
    """Verify the argv agent-run constructs for each harness and mode."""

    def _build(self, harness, **kw):
        defaults = dict(
            interactive=False, prompt=None,
            model=None, agent_mode=None,
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

    def test_claude_default_denies_planning_and_questions_after_prompt(self):
        argv = self._build("claude", prompt="deliver this prompt", session_id="u")
        assert argv.index("deliver this prompt") < argv.index("--disallowedTools")
        denied = [argv[index + 1] for index, value in enumerate(argv[:-1]) if value == "--disallowedTools"]
        assert denied == ["EnterPlanMode", "ExitPlanMode", "AskUserQuestion"]

    def test_claude_escape_hatches_omit_corresponding_denies(self):
        argv = self._build(
            "claude", prompt="hi", enable_planning=True, enable_questions=True,
        )
        assert "--disallowedTools" not in argv


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

        --session silently swallows --prompt (verified live). The built argv for
        an interactive opencode run must never contain --prompt; the prompt is
        delivered post-attach via the FIFO steer path.
        """
        argv = self._build(
            "opencode",
            interactive=True,
            prompt="do the thing",  # inline prompt — must NOT appear in argv
            opencode_port=41234,
            session_id="ses_abc123",
        )
        # --prompt must not appear anywhere in the constructed argv.
        assert "--prompt" not in argv, (
            f"opencode interactive argv must never contain --prompt: {argv}"
        )
        # --session MUST appear so the TUI attaches to the pre-minted session.
        assert "--session" in argv, (
            f"opencode interactive argv must contain --session: {argv}"
        )

    def test_opencode_interactive_includes_port(self):
        argv = self._build("opencode", interactive=True, opencode_port=41234)
        assert "--port" in argv
        i = argv.index("--port")
        assert argv[i + 1] == "41234"

    def test_opencode_interactive_includes_session(self):
        argv = self._build(
            "opencode", interactive=True,
            opencode_port=41234, session_id="ses_xyz",
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

    # codex — _build_managed_argv is not called for codex (app-server path);
    # these tests verify that the function returns an empty list for codex
    # and does not accidentally exec a codex process.
    def test_codex_build_argv_returns_empty(self):
        """_build_managed_argv returns [] for codex (codex uses app-server, not exec)."""
        argv = self._build("codex", prompt="hi")
        assert argv == []

    def test_codex_build_argv_with_session_id_returns_empty(self):
        argv = self._build("codex", prompt="hi", session_id="some-thread-id")
        assert argv == []

    def test_codex_build_argv_interactive_returns_empty(self):
        argv = self._build("codex", interactive=True, prompt="hi")
        assert argv == []

    # harness_args passthrough
    def test_harness_args_appended(self):
        argv = self._build("claude", session_id="u", harness_args=["--foo", "--bar=baz"])
        assert "--foo" in argv
        assert "--bar=baz" in argv

    def test_claude_bare_harness_arg_not_swallowed_by_disallowed_tools(self):
        """A bare (non-flag) --harness-arg token must survive as its own argv element.

        --disallowedTools is variadic: the claude CLI consumes every bare token
        following it as a tool name.  harness_args must appear before the deny
        block so a bare user arg cannot be silently absorbed into the deny list.
        """
        bare_token = "SomeBareArg"
        argv = self._build(
            "claude",
            prompt="hi",
            session_id="u",
            harness_args=[bare_token],
            # denies active so --disallowedTools is present
            enable_planning=False,
            enable_questions=False,
        )
        assert bare_token in argv, f"bare harness_arg {bare_token!r} missing from argv: {argv}"
        # The token must not appear as the argument to any --disallowedTools entry.
        deny_targets = [
            argv[i + 1] for i, tok in enumerate(argv[:-1]) if tok == "--disallowedTools"
        ]
        assert bare_token not in deny_targets, (
            f"bare harness_arg {bare_token!r} was absorbed into --disallowedTools targets: "
            f"{deny_targets}; full argv: {argv}"
        )
        # harness_args must precede the first --disallowedTools flag.
        if "--disallowedTools" in argv:
            assert argv.index(bare_token) < argv.index("--disallowedTools"), (
                f"bare harness_arg {bare_token!r} appears after --disallowedTools: {argv}"
            )


class TestOpenCodePolicyConfig:
    def test_preserves_unrelated_content_and_denies_by_default(self):
        config = json.loads(agent_run._opencode_policy_config(
            json.dumps({"model": "kept", "permission": {"bash": "allow"}}),
            enable_planning=False,
            enable_questions=False,
        ))
        assert config["model"] == "kept"
        assert config["permission"] == {
            "bash": "allow",
            "question": "deny",
            "plan_enter": "deny",
            "plan_exit": "deny",
        }

    def test_escape_hatches_allow_only_requested_capabilities(self):
        config = json.loads(agent_run._opencode_policy_config(
            None, enable_planning=True, enable_questions=True,
        ))
        assert config["permission"] == {
            "question": "allow",
            "plan_enter": "allow",
            "plan_exit": "allow",
        }

    def test_per_agent_allow_block_is_overridden_by_policy(self):
        # A project opencode.json with a per-agent allow block must come out
        # with the deny applied at both global scope and the agent scope.
        existing = json.dumps({
            "permission": {"bash": "allow"},
            "agent": {
                "build": {
                    "permission": {
                        "question": "allow",
                        "plan_enter": "allow",
                        "plan_exit": "allow",
                    },
                    "model": "kept",
                },
            },
        })
        config = json.loads(agent_run._opencode_policy_config(
            existing, enable_planning=False, enable_questions=False,
        ))
        # Global scope: deny applied, unrelated key preserved.
        assert config["permission"]["bash"] == "allow"
        assert config["permission"]["question"] == "deny"
        assert config["permission"]["plan_enter"] == "deny"
        assert config["permission"]["plan_exit"] == "deny"
        # Per-agent scope: deny applied, unrelated field preserved.
        build = config["agent"]["build"]
        assert build["model"] == "kept"
        assert build["permission"]["question"] == "deny"
        assert build["permission"]["plan_enter"] == "deny"
        assert build["permission"]["plan_exit"] == "deny"

    def test_per_agent_blocks_without_policy_keys_receive_policy_injected(self):
        # "build" is always a target agent, so it gets a permission block even
        # when it had none. "test" is not a target but already has a permission
        # block, so the policy is merged into it. Unrelated fields survive.
        existing = json.dumps({
            "agent": {
                "build": {"model": "claude-3"},
                "test": {"permission": {"bash": "allow"}},

            },
        })
        config = json.loads(agent_run._opencode_policy_config(
            existing, enable_planning=False, enable_questions=False,
        ))
        assert config["agent"]["build"]["model"] == "claude-3"
        assert config["agent"]["build"]["permission"]["question"] == "deny"
        assert config["agent"]["build"]["permission"]["plan_enter"] == "deny"
        assert config["agent"]["build"]["permission"]["plan_exit"] == "deny"
        assert config["agent"]["test"]["permission"]["bash"] == "allow"
        assert config["agent"]["test"]["permission"]["question"] == "deny"


class TestOpenCodeAgentNameParsing:
    @pytest.mark.parametrize("harness_args,expected", [
        (["--agent", "myagent"], {"myagent"}),
        (["--agent=myagent"], {"myagent"}),
        (["--agent", "a", "--agent", "b"], {"a", "b"}),
        (["--auto", "--model", "sonnet"], set()),
        ([], set()),
    ])
    def test_agent_names_extracted(self, harness_args, expected):
        assert agent_run._opencode_agent_names_from_harness_args(harness_args) == expected

    def test_agent_name_drives_policy_target(self):
        """The name parsed out of harness_args is the exact key
        _opencode_policy_config writes a deny block under. If the two ever
        disagree on the string, a per-agent allow escapes the policy.
        """
        agent_name = "myspecialagent"
        extra = agent_run._opencode_agent_names_from_harness_args(["--agent", agent_name])
        policy = json.loads(agent_run._opencode_policy_config(
            None,
            enable_planning=False,
            enable_questions=False,
            extra_agent_names=extra,
        ))
        agent_block = policy.get("agent", {}).get(agent_name)
        assert agent_block is not None, (
            f"_opencode_policy_config did not create a block for agent {agent_name!r}; "
            f"the parsed name from harness_args did not reach target_agents"
        )
        assert agent_block.get("permission", {}).get("question") == "deny", (
            f"agent block for {agent_name!r} does not carry the deny: {agent_block!r}"
        )


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
        agent_run._record_session(log_dir, acquire_log, "claude", "uuid-1234", "pushed", "certain")
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
        agent_run._record_session(log_dir, acquire_log, "opencode", "ses_abc", "minted", "certain")
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
        agent_run._record_session(log_dir, acquire_log, "codex", None, "missing", "missing", "no id found")
        data = agent_run._read_session_json(log_dir)
        assert data is not None
        assert data["session_id"] is None
        assert data["harness"] == "codex"
        assert data["confidence"] == "missing"
        assert data["acquisition"] == "missing"  # "missing" is the spec-valid value
        assert "reason" in data

    def test_never_label_a_guess_certain(self, isolated_log_root):
        """Confidence 'certain' is only allowed for pushed/minted/reported acquisition."""
        log_dir = isolated_log_root / "run4"
        log_dir.mkdir(parents=True, exist_ok=True)
        acquire_log = log_dir / "session-acquire.log"
        # The missing path must never write certain.
        agent_run._record_session(log_dir, acquire_log, "opencode", None, "missing", "missing", "reason")
        data = agent_run._read_session_json(log_dir)
        assert data["confidence"] == "missing"
        assert data["confidence"] != "certain"

    def test_write_session_json_oserror_does_not_propagate(self, isolated_log_root, monkeypatch):
        """_record_session must never raise; spec §6."""
        log_dir = isolated_log_root / "run5"
        log_dir.mkdir(parents=True, exist_ok=True)
        acquire_log = log_dir / "session-acquire.log"
        # Simulate ENOSPC on the write.
        original = agent_run._write_session_json
        def _raise(*args, **kwargs):
            raise OSError(28, "No space left on device")
        monkeypatch.setattr(agent_run, "_write_session_json", _raise)
        # Must not raise.
        agent_run._record_session(log_dir, acquire_log, "claude", "some-uuid", "pushed", "certain")
        agent_run._record_session(log_dir, acquire_log, "codex", None, "missing", "missing", "test")
        # The acquire log should mention the failure.
        log_text = acquire_log.read_text() if acquire_log.exists() else ""
        assert "could not write session.json" in log_text

    def test_read_session_json_unicode_error_returns_none(self, isolated_log_root):
        """Non-UTF-8 bytes in session.json degrade to None, not an exception."""
        log_dir = isolated_log_root / "badenc"
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "session.json").write_bytes(b"\xff\xfe torn json")
        # Must not raise.
        result = agent_run._read_session_json(log_dir)
        # read_text(errors="replace") allows partial reads; json.loads on garbage returns None.
        assert result is None or isinstance(result, dict)

    def test_read_session_json_non_dict_returns_none(self, isolated_log_root):
        """session.json containing a JSON array or string returns None."""
        log_dir = isolated_log_root / "nondictjson"
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "session.json").write_text("[1, 2, 3]")
        result = agent_run._read_session_json(log_dir)
        assert result is None


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
# Claude session id generation
# ---------------------------------------------------------------------------

UUID4_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)

class TestClaudeSessionIdGeneration:
    def test_generates_uuid4_when_no_caller_id(self):
        """_record_session writes a UUID4 session_id for claude."""
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            log_dir = Path(td)
            acquire_log = log_dir / "session-acquire.log"
            # Use a fresh UUID4 directly (the function that generates it is inlined).
            import uuid
            sid = str(uuid.uuid4())
            agent_run._record_session(log_dir, acquire_log, "claude", sid, "pushed", "certain")
            data = agent_run._read_session_json(log_dir)
            assert data is not None
            assert UUID4_RE.match(data["session_id"]), f"Not a UUID4: {data['session_id']!r}"

    def test_each_call_generates_unique_uuid(self):
        import uuid
        ids = {str(uuid.uuid4()) for _ in range(10)}
        assert len(ids) == 10


# ---------------------------------------------------------------------------
# Managed mode launch integration (subprocess runs with real commands)
# ---------------------------------------------------------------------------

def _kill_run_pid(state_dir: Path) -> None:
    """Send SIGTERM then SIGKILL to the runner pid recorded in state_dir.

    Silently does nothing if the pid file is absent or the process is already
    gone — so callers can invoke this unconditionally on both pass and fail.
    """
    try:
        pid = int((state_dir / "pid").read_text().strip())
    except (FileNotFoundError, ValueError, OSError):
        return
    if pid <= 0:
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.kill(pid, sig)
        except (ProcessLookupError, PermissionError):
            return
        if sig == signal.SIGTERM:
            time.sleep(0.3)



def _wait_terminal(state_dir: Path, timeout: float = 15.0) -> str:
    """Poll <state_dir>/status until it is terminal; return the last status seen.

    On timeout the runner is killed so a slow fake harness cannot outlive the
    test, and the non-terminal status is returned for the caller to assert on.
    """
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
    if status not in agent_run.TERMINAL_STATUSES:
        _kill_run_pid(state_dir)
    return status



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

    Returns (final_status, session_json_data). Kills the runner on timeout so
    a slow fake harness cannot leave an orphaned process after the test ends.
    """
    ns = argparse.Namespace(
        name=name,
        command=[],
        interactive=interactive,
        prompt_file=prompt_file,
        submit_mode=None,
        idle_timeout=None,
        harness=harness,
        prompt=prompt,
        model=model,
        agent_mode=None,


        harness_args=[],
        permissions="bypass",
    )
    rc = agent_run.cmd_launch(ns)
    assert rc == 0

    state_dir = state_root / name
    log_dir = log_root / name
    status = _wait_terminal(state_dir, timeout=timeout)


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


    def test_claude_oneshot_delivers_prompt_via_stdin_with_denies_in_argv(
        self, isolated_runs_root, isolated_log_root, tmp_path, monkeypatch
    ):
        fake = tmp_path / "claude"
        fake.write_text(
            "#!/bin/sh\n"
            "for arg in \"$@\"; do printf '<%s>\\n' \"$arg\"; done\n"
            "printf '<stdin:%s>\\n' \"$(cat)\"\n"
        )
        fake.chmod(0o755)
        monkeypatch.setenv("PATH", str(tmp_path) + ":" + os.environ.get("PATH", ""))
        name = "claude-prompt-after-denies"
        _launch_and_wait(
            isolated_runs_root, isolated_log_root, name,
            harness="claude", interactive=False, prompt="unique managed prompt",
        )
        delivered = (isolated_log_root / name / "log").read_text()
        # Prompt arrives on stdin; the managed path materialises it to a file
        # and _build_managed_argv suppresses the positional arg.
        assert "<stdin:unique managed prompt>" in delivered
        # Each deny must appear as a flag/value pair, not as a bare positional.
        lines = delivered.splitlines()
        for tool in ("EnterPlanMode", "ExitPlanMode", "AskUserQuestion"):
            try:
                idx = lines.index(f"<{tool}>")
            except ValueError:
                raise AssertionError(f"<{tool}> not found in delivered argv: {delivered!r}")
            assert lines[idx - 1] == "<--disallowedTools>", (
                f"<{tool}> must be immediately preceded by <--disallowedTools>: {delivered!r}"
            )


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


class TestManagedCodexOneShotAppServer:
    """Tests for codex managed one-shot: app-server JSON-RPC mint+run."""

    def _make_fake_codex_suite(
        self,
        tmp_path: Path,
        *,
        thread_id: str = "test-thread-uuid-0001",
        appserver_works: bool = True,
        params_out: Optional[Path] = None,
    ) -> str:
        """Write a fake `codex` Python script that handles app-server JSON-RPC.

        When called as 'codex app-server', speaks the full JSON-RPC protocol:
        initialize → initialized notification → thread/start → turn/start →
        item/agentMessage/delta events → turn/completed.

        When app-server fails (appserver_works=False), exits immediately.
        When params_out is set, the received thread/start params are written
        to that path as JSON before the fake responds.
        """
        fake_dir = tmp_path / "bin"
        fake_dir.mkdir(parents=True, exist_ok=True)
        fake = fake_dir / "codex"

        if appserver_works:
            params_dump = (
                f"open({str(params_out)!r}, 'w').write(json.dumps(msg['params']))\n"
                if params_out is not None else ""
            )
            # Full JSON-RPC app-server implemented in Python for reliability.
            fake.write_text(
                "#!/usr/bin/env python3\n"
                "import sys, json, time\n"
                "\n"
                "def send(obj):\n"
                "    sys.stdout.write(json.dumps(obj) + '\\n')\n"
                "    sys.stdout.flush()\n"
                "\n"
                "def recv():\n"
                "    line = sys.stdin.readline()\n"
                "    if not line:\n"
                "        return None\n"
                "    try:\n"
                "        return json.loads(line.strip())\n"
                "    except Exception:\n"
                "        return None\n"
                "\n"
                "if len(sys.argv) < 2 or sys.argv[1] != 'app-server':\n"
                "    sys.exit(1)\n"
                "\n"
                f"THREAD_ID = '{thread_id}'\n"
                "\n"
                "# Read initialize\n"
                "msg = recv()\n"
                "if not msg or msg.get('method') != 'initialize':\n"
                "    sys.exit(1)\n"
                "send({'id': msg['id'], 'result': {'userAgent': 'codex_exec/0.144.1'}})\n"
                "\n"
                "# Read initialized notification (no response)\n"
                "msg = recv()\n"
                "\n"
                "# Read thread/start\n"
                "msg = recv()\n"
                "if not msg or msg.get('method') != 'thread/start':\n"
                "    sys.exit(1)\n"
                f"{params_dump}"
                "send({'id': msg['id'], 'result': {'thread': {\n"
                "    'id': THREAD_ID, 'sessionId': THREAD_ID,\n"
                "    'path': f'~/.codex/sessions/rollout-{THREAD_ID}.jsonl',\n"
                "    'parentThreadId': None, 'forkedFromId': None,\n"
                "}}})\n"
                "\n"
                "# Read turn/start\n"
                "msg = recv()\n"
                "if not msg or msg.get('method') != 'turn/start':\n"
                "    sys.exit(1)\n"
                "turn_id = 'turn-001'\n"
                "send({'id': msg['id'], 'result': {'turn': {'id': turn_id, 'status': 'inProgress'}}})\n"
                "\n"
                "# Emit agent text via delta events (the answer to any prompt).\n"
                "send({'method': 'item/agentMessage/delta', 'params': {\n"
                "    'threadId': THREAD_ID, 'turnId': turn_id,\n"
                "    'itemId': 'item-001', 'delta': 'the answer is 42',\n"
                "}})\n"
                "\n"
                "# turn/completed\n"
                "send({'method': 'turn/completed', 'params': {\n"
                "    'threadId': THREAD_ID,\n"
                "    'turn': {'id': turn_id, 'status': 'completed'},\n"
                "}})\n"
                "\n"
                "# Keep stdin open so the manager can read all output before killing.\n"
                "time.sleep(10)\n"
            )
        else:
            # app-server fails immediately.
            fake.write_text(
                "#!/usr/bin/env python3\n"
                "import sys\n"
                "if len(sys.argv) >= 2 and sys.argv[1] == 'app-server':\n"
                "    sys.exit(1)\n"
                "sys.exit(1)\n"
            )
        fake.chmod(0o755)
        return str(fake_dir)

    def test_codex_oneshot_minted_session_json(
        self, isolated_runs_root, isolated_log_root, tmp_path, monkeypatch
    ):
        """When app-server succeeds, session.json has acquisition=minted, confidence=certain."""
        bin_dir = self._make_fake_codex_suite(tmp_path, thread_id="minted-thread-0001")
        monkeypatch.setenv("PATH", bin_dir + ":" + os.environ.get("PATH", ""))
        name = "codex-appserver-1"
        status, session = _launch_and_wait(
            isolated_runs_root, isolated_log_root, name,
            harness="codex",
            interactive=False,
            prompt="what is 6*7",
        )
        assert session is not None
        assert session["harness"] == "codex"
        assert session["acquisition"] == "minted"
        assert session["confidence"] == "certain"
        assert session["session_id"] == "minted-thread-0001"

    def test_codex_oneshot_fallback_to_missing_when_appserver_fails(
        self, isolated_runs_root, isolated_log_root, tmp_path, monkeypatch
    ):
        """When app-server fails, run still completes with confidence=missing."""
        bin_dir = self._make_fake_codex_suite(tmp_path, appserver_works=False)
        monkeypatch.setenv("PATH", bin_dir + ":" + os.environ.get("PATH", ""))
        name = "codex-appserver-2"
        status, session = _launch_and_wait(
            isolated_runs_root, isolated_log_root, name,
            harness="codex",
            interactive=False,
            prompt="what is 6*7",
        )
        assert status in agent_run.TERMINAL_STATUSES
        assert session is not None
        assert session["confidence"] == "missing"
        assert session["session_id"] is None

    def test_codex_oneshot_session_json_in_persistent_log_dir(
        self, isolated_runs_root, isolated_log_root, tmp_path, monkeypatch
    ):
        """session.json goes in the persistent log dir, not the ephemeral state dir."""
        bin_dir = self._make_fake_codex_suite(tmp_path)
        monkeypatch.setenv("PATH", bin_dir + ":" + os.environ.get("PATH", ""))
        name = "codex-appserver-3"
        _launch_and_wait(
            isolated_runs_root, isolated_log_root, name,
            harness="codex",
            interactive=False,
            prompt="hi",
        )
        assert (isolated_log_root / name / "session.json").exists()
        assert not (isolated_runs_root / name / "session.json").exists()

    def test_codex_oneshot_model_reaches_thread_start(
        self, isolated_runs_root, isolated_log_root, tmp_path, monkeypatch
    ):
        """--model on a one-shot codex launch lands in thread/start's params."""
        params_file = tmp_path / "thread_start_params.json"
        bin_dir = self._make_fake_codex_suite(
            tmp_path, thread_id="oneshot-tid", params_out=params_file
        )
        monkeypatch.setenv("PATH", bin_dir + ":" + os.environ.get("PATH", ""))
        name = "codex-oneshot-model"
        _launch_and_wait(
            isolated_runs_root, isolated_log_root, name,
            harness="codex",
            interactive=False,
            prompt="hi",
            model="gpt-5.6-luna",
        )
        params = json.loads(params_file.read_text())
        assert params["model"] == "gpt-5.6-luna"

    def test_codex_log_contains_readable_output_not_jsonl(
        self, isolated_runs_root, isolated_log_root, tmp_path, monkeypatch
    ):
        """After Part A: the codex log must contain readable prose, not --json events."""
        bin_dir = self._make_fake_codex_suite(tmp_path, thread_id="readable-test")
        monkeypatch.setenv("PATH", bin_dir + ":" + os.environ.get("PATH", ""))
        name = "codex-readable-log"
        _launch_and_wait(
            isolated_runs_root, isolated_log_root, name,
            harness="codex",
            interactive=False,
            prompt="what is 6*7",
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
        # Must contain the human-readable answer, not a JSONL event stream.
        assert "the answer is 42" in log_content, (
            f"Expected readable output in codex log, got: {log_content!r}"
        )
        assert '"type"' not in log_content, (
            f"JSONL event stream must not appear in codex log: {log_content!r}"
        )


class TestMintThreadModelParam:
    """Unit tests for _CodexAppServer.mint_thread's model parameter."""

    def _make_fake_codex_capture(self, tmp_path: Path) -> tuple[str, Path]:
        """Fake app-server that dumps thread/start's params to a file, then mints."""
        fake_dir = tmp_path / "bin"
        fake_dir.mkdir(parents=True, exist_ok=True)
        fake = fake_dir / "codex"
        params_file = tmp_path / "thread_start_params.json"
        fake.write_text(
            "#!/usr/bin/env python3\n"
            "import sys, json\n"
            "def send(obj):\n"
            "    sys.stdout.write(json.dumps(obj) + '\\n')\n"
            "    sys.stdout.flush()\n"
            "def recv():\n"
            "    line = sys.stdin.readline()\n"
            "    return json.loads(line.strip()) if line.strip() else None\n"
            "if len(sys.argv) < 2 or sys.argv[1] != 'app-server':\n"
            "    sys.exit(1)\n"
            "msg = recv(); send({'id': msg['id'], 'result': {'userAgent': 'test'}})\n"
            "recv()  # initialized\n"
            "msg = recv()\n"
            f"open({str(params_file)!r}, 'w').write(json.dumps(msg['params']))\n"
            "send({'id': msg['id'], 'result': {'thread': {"
            "'id': 'tid', 'sessionId': 'tid', 'path': 'r.jsonl'}}})\n"
        )
        fake.chmod(0o755)
        return str(fake_dir), params_file

    def _mint(self, tmp_path: Path, monkeypatch, *, model: Optional[str]) -> dict:
        bin_dir, params_file = self._make_fake_codex_capture(tmp_path)
        monkeypatch.setenv("PATH", bin_dir + ":" + os.environ.get("PATH", ""))
        state_dir = tmp_path / "state"
        log_dir = tmp_path / "log"
        state_dir.mkdir()
        log_dir.mkdir()
        server = agent_run._CodexAppServer(state_dir, log_dir, log_dir / "acquire.log", "test")
        assert server.start([], str(tmp_path), lambda: None)
        try:
            thread_id = server.mint_thread(str(tmp_path), model=model)
        finally:
            server.close()
        assert thread_id == "tid"
        return json.loads(params_file.read_text())

    def test_mint_thread_includes_model_when_given(self, tmp_path, monkeypatch):
        params = self._mint(tmp_path, monkeypatch, model="gpt-5.6-luna")
        assert params["model"] == "gpt-5.6-luna"

    def test_mint_thread_omits_model_key_when_not_given(self, tmp_path, monkeypatch):
        """Regression guard: the default path must not send model:null."""
        params = self._mint(tmp_path, monkeypatch, model=None)
        assert "model" not in params

    def test_mint_thread_error_fails_fast_with_diagnostic(self, tmp_path, monkeypatch):
        """A thread/start error must return promptly, not wait out the full timeout,
        and session.json's reason must carry the concrete JSON-RPC error text."""
        fake_dir = tmp_path / "bin"
        fake_dir.mkdir(parents=True, exist_ok=True)
        fake = fake_dir / "codex"
        fake.write_text(
            "#!/usr/bin/env python3\n"
            "import sys, json, time\n"
            "def send(obj):\n"
            "    sys.stdout.write(json.dumps(obj) + '\\n')\n"
            "    sys.stdout.flush()\n"
            "def recv():\n"
            "    line = sys.stdin.readline()\n"
            "    return json.loads(line.strip()) if line.strip() else None\n"
            "if len(sys.argv) < 2 or sys.argv[1] != 'app-server':\n"
            "    sys.exit(1)\n"
            "msg = recv(); send({'id': msg['id'], 'result': {'userAgent': 'test'}})\n"
            "recv()  # initialized\n"
            "msg = recv()\n"
            "send({'id': msg['id'], 'error': {'code': -32602, "
            "'message': 'unknown model: gpt-nonexistent'}})\n"
            "# Stay alive: the app-server does not exit merely because a request\n"
            "# it rejected, so mint_thread must not rely on EOF or process exit.\n"
            "time.sleep(30)\n"
        )
        fake.chmod(0o755)
        monkeypatch.setenv("PATH", str(fake_dir) + ":" + os.environ.get("PATH", ""))
        state_dir = tmp_path / "state"
        log_dir = tmp_path / "log"
        state_dir.mkdir()
        log_dir.mkdir()
        server = agent_run._CodexAppServer(state_dir, log_dir, log_dir / "acquire.log", "test")
        assert server.start([], str(tmp_path), lambda: None)
        start = time.monotonic()
        try:
            thread_id = server.mint_thread(str(tmp_path), model="gpt-nonexistent")
        finally:
            server.close()
        elapsed = time.monotonic() - start
        assert thread_id is None
        assert elapsed < 5.0, f"mint_thread took {elapsed:.1f}s, expected well under 5s"
        session = agent_run._read_session_json(log_dir)
        assert session is not None
        assert "unknown model: gpt-nonexistent" in session["reason"]

    def test_mint_thread_result_wins_over_same_batch_error(self, tmp_path, monkeypatch):
        """A minted thread is not undone by a same-id error frame in the same read
        batch: the mint succeeds and no failure reason is recorded."""
        fake_dir = tmp_path / "bin"
        fake_dir.mkdir(parents=True, exist_ok=True)
        fake = fake_dir / "codex"
        fake.write_text(
            "#!/usr/bin/env python3\n"
            "import sys, json, time\n"
            "def send(obj):\n"
            "    sys.stdout.write(json.dumps(obj) + '\\n')\n"
            "def recv():\n"
            "    line = sys.stdin.readline()\n"
            "    return json.loads(line.strip()) if line.strip() else None\n"
            "if len(sys.argv) < 2 or sys.argv[1] != 'app-server':\n"
            "    sys.exit(1)\n"
            "msg = recv(); send({'id': msg['id'], 'result': {'userAgent': 'test'}})\n"
            "sys.stdout.flush()\n"
            "recv()\n"
            "msg = recv()\n"
            "send({'id': msg['id'], 'result': {'thread': {'id': 'batch-tid', "
            "'sessionId': 'batch-tid', 'path': 'r.jsonl'}}})\n"
            "send({'id': msg['id'], 'error': {'code': -32602, 'message': 'late error'}})\n"
            "sys.stdout.flush()\n"
            "time.sleep(30)\n"
        )
        fake.chmod(0o755)
        monkeypatch.setenv("PATH", str(fake_dir) + ":" + os.environ.get("PATH", ""))
        state_dir = tmp_path / "state"
        log_dir = tmp_path / "log"
        state_dir.mkdir()
        log_dir.mkdir()
        server = agent_run._CodexAppServer(state_dir, log_dir, log_dir / "acquire.log", "test")
        assert server.start([], str(tmp_path), lambda: None)
        try:
            thread_id = server.mint_thread(str(tmp_path))
        finally:
            server.close()
        assert thread_id == "batch-tid"
        session = agent_run._read_session_json(log_dir)
        assert session is not None
        assert session["acquisition"] == "minted"
        assert session.get("reason") is None

    def test_mint_thread_error_reason_stays_within_bound(self, tmp_path, monkeypatch):
        """An oversized multi-line error payload is stored single-line and within
        _THREAD_START_ERROR_REASON_MAX, truncation marker included."""
        fake_dir = tmp_path / "bin"
        fake_dir.mkdir(parents=True, exist_ok=True)
        fake = fake_dir / "codex"
        fake.write_text(
            "#!/usr/bin/env python3\n"
            "import sys, json, time\n"
            "def send(obj):\n"
            "    sys.stdout.write(json.dumps(obj) + '\\n')\n"
            "    sys.stdout.flush()\n"
            "def recv():\n"
            "    line = sys.stdin.readline()\n"
            "    return json.loads(line.strip()) if line.strip() else None\n"
            "if len(sys.argv) < 2 or sys.argv[1] != 'app-server':\n"
            "    sys.exit(1)\n"
            "msg = recv(); send({'id': msg['id'], 'result': {'userAgent': 'test'}})\n"
            "recv()\n"
            "msg = recv()\n"
            "send({'id': msg['id'], 'error': {'code': -32602, "
            "'message': 'boom\\n' + 'x' * 4000}})\n"
            "time.sleep(30)\n"
        )
        fake.chmod(0o755)
        monkeypatch.setenv("PATH", str(fake_dir) + ":" + os.environ.get("PATH", ""))
        state_dir = tmp_path / "state"
        log_dir = tmp_path / "log"
        state_dir.mkdir()
        log_dir.mkdir()
        server = agent_run._CodexAppServer(state_dir, log_dir, log_dir / "acquire.log", "test")
        assert server.start([], str(tmp_path), lambda: None)
        try:
            assert server.mint_thread(str(tmp_path), model="huge") is None
        finally:
            server.close()
        session = agent_run._read_session_json(log_dir)
        reason = session["reason"]
        assert len(reason) <= agent_run._THREAD_START_ERROR_REASON_MAX
        assert "\n" not in reason
        assert reason.endswith("...")


# ---------------------------------------------------------------------------
# _CodexAppServer.turn_start_error: early rejection detection for turn/start
# ---------------------------------------------------------------------------

class TestTurnStartError:
    def _server_pair(self, tmp_path: Path, monkeypatch, script: str) -> "agent_run._CodexAppServer":
        """Spawn a fake codex app-server that performs the initialize
        handshake, then runs *script* (already-indented Python source)."""
        fake_dir = tmp_path / "bin"
        fake_dir.mkdir(parents=True, exist_ok=True)
        fake = fake_dir / "codex"
        fake.write_text(
            "#!/usr/bin/env python3\n"
            "import sys, json, time\n"
            "def send(obj):\n"
            "    sys.stdout.write(json.dumps(obj) + '\\n')\n"
            "    sys.stdout.flush()\n"
            "def recv():\n"
            "    line = sys.stdin.readline()\n"
            "    return json.loads(line.strip()) if line.strip() else None\n"
            "if len(sys.argv) < 2 or sys.argv[1] != 'app-server':\n"
            "    sys.exit(1)\n"
            "msg = recv(); send({'id': msg['id'], 'result': {'userAgent': 'test'}})\n"
            "recv()  # initialized\n"
            + script
        )
        fake.chmod(0o755)
        monkeypatch.setenv("PATH", str(fake_dir) + ":" + os.environ.get("PATH", ""))
        state_dir = tmp_path / "state"
        log_dir = tmp_path / "log"
        state_dir.mkdir()
        log_dir.mkdir()
        server = agent_run._CodexAppServer(state_dir, log_dir, log_dir / "acquire.log", "test")
        assert server.start([], str(tmp_path), lambda: None)
        # Perform the same initialize handshake mint_thread does, so the fake
        # server's first real request is turn/start, matching the protocol
        # order turn_start_error is used under in practice.
        server.call("initialize", {"clientInfo": {"name": "test", "title": "test", "version": "0"}})
        server.send({"jsonrpc": "2.0", "method": "initialized", "params": {}})
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            frames, saw_lines, _eof = server.read_frames(0.2)
            if any(f.get("id") == 1 for f in frames):
                break
        return server

    def test_turn_start_error_reports_rejection(self, tmp_path, monkeypatch):
        """A turn/start response carrying an error is surfaced promptly."""
        server = self._server_pair(
            tmp_path, monkeypatch,
            "msg = recv()  # turn/start\n"
            "send({'id': msg['id'], 'error': {'code': -32602, 'message': 'bad turn params'}})\n"
            "time.sleep(30)\n",
        )
        try:
            rpc_id = server.start_turn("hello")
            start = time.monotonic()
            error = server.turn_start_error(rpc_id, timeout_s=5.0)
            elapsed = time.monotonic() - start
        finally:
            server.close()
        assert error is not None and "bad turn params" in error
        assert elapsed < 5.0, f"turn_start_error took {elapsed:.1f}s, expected well under 5s"

    def test_turn_start_error_returns_none_on_success(self, tmp_path, monkeypatch):
        """A normal turn/start result is not mistaken for a rejection, and the
        result frame remains available to the caller's own frame loop."""
        server = self._server_pair(
            tmp_path, monkeypatch,
            "msg = recv()  # turn/start\n"
            "send({'id': msg['id'], 'result': {'turn': {'id': 'turn-ok', 'status': 'inProgress'}}})\n"
            "time.sleep(30)\n",
        )
        try:
            rpc_id = server.start_turn("hello")
            error = server.turn_start_error(rpc_id, timeout_s=5.0)
            # The success frame must be requeued, not dropped.
            frames, _, _ = server.read_frames(0.5)
        finally:
            server.close()
        assert error is None
        assert any(f.get("id") == rpc_id and "result" in f for f in frames)

    def test_turn_start_error_none_when_still_pending(self, tmp_path, monkeypatch):
        """A slow server that never responds within the poll window reports
        None (unknown), not a false rejection."""
        server = self._server_pair(
            tmp_path, monkeypatch,
            "recv()  # turn/start received but never answered\n"
            "time.sleep(30)\n",
        )
        try:
            rpc_id = server.start_turn("hello")
            error = server.turn_start_error(rpc_id, timeout_s=0.3)
        finally:
            server.close()
        assert error is None

    def test_turn_start_error_requeues_unrelated_frames(self, tmp_path, monkeypatch):
        """A delta frame arriving before the matching response is preserved
        for the caller's frame loop rather than discarded."""
        server = self._server_pair(
            tmp_path, monkeypatch,
            "msg = recv()  # turn/start\n"
            "send({'method': 'item/agentMessage/delta', 'params': {'delta': 'stray'}})\n"
            "send({'id': msg['id'], 'result': {'turn': {'id': 't', 'status': 'inProgress'}}})\n"
            "time.sleep(30)\n",
        )
        try:
            rpc_id = server.start_turn("hello")
            error = server.turn_start_error(rpc_id, timeout_s=5.0)
            frames, _, _ = server.read_frames(0.5)
        finally:
            server.close()
        assert error is None
        deltas = [agent_run._turn_delta_text(f) for f in frames]
        assert "stray" in deltas


# ---------------------------------------------------------------------------
# Codex interactive: app-server JSON-RPC mint + turn/steer
# ---------------------------------------------------------------------------

class TestManagedCodexInteractiveAppServer:
    """Tests for codex managed interactive: app-server stays alive, FIFO steer → turn/steer."""

    def _make_fake_codex_interactive(
        self,
        tmp_path: Path,
        *,
        thread_id: str = "interactive-thread-0001",
        appserver_works: bool = True,
        steer_wait_seconds: float = 3.0,
        params_out: Optional[Path] = None,
    ) -> str:
        """Write a fake codex app-server that handles interactive protocol.

        Sequence: initialize → thread/start → turn/start (initial prompt) →
        item/agentMessage/delta → turn/completed → wait up to steer_wait_seconds
        for a steer (turn/steer or turn/start) → if received, emit steered response
        → exit.  If the steer wait times out or stdin closes, exit immediately.
        When params_out is set, the received thread/start params are written to
        that path as JSON before the fake responds.
        """
        fake_dir = tmp_path / "bin"
        fake_dir.mkdir(parents=True, exist_ok=True)
        fake = fake_dir / "codex"

        if appserver_works:
            params_dump = (
                f"open({str(params_out)!r}, 'w').write(json.dumps(msg['params']))\n"
                if params_out is not None else ""
            )
            fake.write_text(
                "#!/usr/bin/env python3\n"
                "import sys, json, time, select\n"
                "\n"
                "def send(obj):\n"
                "    sys.stdout.write(json.dumps(obj) + '\\n')\n"
                "    sys.stdout.flush()\n"
                "\n"
                "def recv(timeout=None):\n"
                "    if timeout is not None:\n"
                "        ready, _, _ = select.select([sys.stdin], [], [], timeout)\n"
                "        if not ready:\n"
                "            return None\n"
                "    line = sys.stdin.readline()\n"
                "    if not line:\n"
                "        return None\n"
                "    try:\n"
                "        return json.loads(line.strip())\n"
                "    except Exception:\n"
                "        return None\n"
                "\n"
                "if len(sys.argv) < 2 or sys.argv[1] != 'app-server':\n"
                "    sys.exit(1)\n"
                "\n"
                f"THREAD_ID = '{thread_id}'\n"
                f"STEER_WAIT = {steer_wait_seconds}\n"
                "\n"
                "# Handshake: initialize\n"
                "msg = recv()\n"
                "if not msg or msg.get('method') != 'initialize':\n"
                "    sys.exit(1)\n"
                "send({'id': msg['id'], 'result': {'userAgent': 'codex_exec/0.144.1'}})\n"
                "\n"
                "# initialized notification (no response)\n"
                "msg = recv()\n"
                "\n"
                "# thread/start\n"
                "msg = recv()\n"
                "if not msg or msg.get('method') != 'thread/start':\n"
                "    sys.exit(1)\n"
                f"{params_dump}"
                "send({'id': msg['id'], 'result': {'thread': {\n"
                "    'id': THREAD_ID, 'sessionId': THREAD_ID,\n"
                "    'path': f'~/.codex/sessions/rollout-{THREAD_ID}.jsonl',\n"
                "    'parentThreadId': None, 'forkedFromId': None,\n"
                "}}})\n"
                "\n"
                "# Initial turn/start (initial prompt)\n"
                "msg = recv()\n"
                "if not msg or msg.get('method') != 'turn/start':\n"
                "    sys.exit(1)\n"
                "turn_id = 'turn-001'\n"
                "send({'id': msg['id'], 'result': {'turn': {'id': turn_id, 'status': 'inProgress'}}})\n"
                "send({'method': 'item/agentMessage/delta', 'params': {\n"
                "    'threadId': THREAD_ID, 'turnId': turn_id, 'itemId': 'i1',\n"
                "    'delta': 'initial response',\n"
                "}})\n"
                "send({'method': 'turn/completed', 'params': {\n"
                "    'threadId': THREAD_ID,\n"
                "    'turn': {'id': turn_id, 'status': 'completed'},\n"
                "}})\n"
                "\n"
                "# Wait up to STEER_WAIT seconds for a steer message.\n"
                "msg = recv(timeout=STEER_WAIT)\n"
                "if msg is None:\n"
                "    sys.exit(0)\n"
                "steer_method = msg.get('method', '')\n"
                "if steer_method in ('turn/steer', 'turn/start'):\n"
                "    turn_id2 = 'turn-002'\n"
                "    send({'id': msg['id'], 'result': {'turn': {'id': turn_id2, 'status': 'inProgress'}}})\n"
                "    send({'method': 'item/agentMessage/delta', 'params': {\n"
                "        'threadId': THREAD_ID, 'turnId': turn_id2, 'itemId': 'i2',\n"
                "        'delta': 'steered response',\n"
                "    }})\n"
                "    send({'method': 'turn/completed', 'params': {\n"
                "        'threadId': THREAD_ID,\n"
                "        'turn': {'id': turn_id2, 'status': 'completed'},\n"
                "    }})\n"
                "\n"
                "# Exit so the runner loop sees EOF and terminates.\n"
                "sys.exit(0)\n"
            )
        else:
            fake.write_text(
                "#!/usr/bin/env python3\n"
                "import sys\n"
                "if len(sys.argv) >= 2 and sys.argv[1] == 'app-server':\n"
                "    sys.exit(1)\n"
                "sys.exit(1)\n"
            )
        fake.chmod(0o755)
        return str(fake_dir)

    def _wait_for_running(self, state_dir: Path, timeout: float = 10.0) -> bool:
        """Wait until the run's status file shows 'running'. Returns True if reached.
        Kills the runner on timeout so a wedged fake binary cannot outlive the test.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                status = (state_dir / "status").read_text().strip()
                if status == "running":
                    return True
                if status in agent_run.TERMINAL_STATUSES:
                    return False
            except FileNotFoundError:
                pass
            time.sleep(0.05)
        _kill_run_pid(state_dir)
        return False

    def _wait_for_terminal(self, state_dir: Path, timeout: float = 15.0) -> str:
        """Wait for terminal status. Returns final status string.
        Kills the runner on timeout so a wedged fake binary cannot outlive the test.
        """
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
        if status not in agent_run.TERMINAL_STATUSES:
            _kill_run_pid(state_dir)
        return status

    def test_interactive_codex_model_reaches_thread_start(
        self, isolated_runs_root, isolated_log_root, tmp_path, monkeypatch
    ):
        """--model on an interactive codex launch lands in thread/start's params."""
        params_file = tmp_path / "thread_start_params.json"
        # steer_wait_seconds=0.5: fake exits quickly after initial turn completes.
        bin_dir = self._make_fake_codex_interactive(
            tmp_path, thread_id="iact-model-tid", steer_wait_seconds=0.5,
            params_out=params_file,
        )
        monkeypatch.setenv("PATH", bin_dir + ":" + os.environ.get("PATH", ""))
        name = "codex-iact-model"
        ns = argparse.Namespace(
            name=name, command=[], interactive=True, prompt_file=None,
            submit_mode=None, idle_timeout=None,
            harness="codex", prompt="hi", model="gpt-5.6-luna", agent_mode=None,
            harness_args=[],
        )
        rc = agent_run.cmd_launch(ns)
        assert rc == 0
        state_dir = isolated_runs_root / name
        self._wait_for_running(state_dir, timeout=10.0)
        self._wait_for_terminal(state_dir, timeout=10.0)
        params = json.loads(params_file.read_text())
        assert params["model"] == "gpt-5.6-luna"

    def test_interactive_codex_minted_session_json(
        self, isolated_runs_root, isolated_log_root, tmp_path, monkeypatch
    ):
        """Interactive codex via app-server writes session.json with minted/certain."""
        # steer_wait_seconds=0.5: fake exits quickly after initial turn completes.
        bin_dir = self._make_fake_codex_interactive(
            tmp_path, thread_id="iact-thread-0001", steer_wait_seconds=0.5
        )
        monkeypatch.setenv("PATH", bin_dir + ":" + os.environ.get("PATH", ""))
        name = "codex-iact-session"
        ns = argparse.Namespace(
            name=name,
            command=[],
            interactive=True,
            prompt_file=None,
            submit_mode=None,
            idle_timeout=None,
            harness="codex",
            prompt="say hi",
            model=None,
            agent_mode=None,
            harness_args=[],
        )
        rc = agent_run.cmd_launch(ns)
        assert rc == 0
        state_dir = isolated_runs_root / name
        log_dir = isolated_log_root / name

        # Wait for the run to reach running (session.json is written then).
        reached_running = self._wait_for_running(state_dir, timeout=10.0)

        assert reached_running, "Interactive codex run must reach 'running'"

        # session.json must show minted/certain at this point.
        session = agent_run._read_session_json(log_dir)
        assert session is not None, "session.json must be written for interactive codex"
        assert session["harness"] == "codex"
        assert session["acquisition"] == "minted"
        assert session["confidence"] == "certain"
        assert session["session_id"] == "iact-thread-0001"

        # Fake exits after the steer wait times out; run will reach terminal status.
        self._wait_for_terminal(state_dir, timeout=10.0)

    def test_interactive_codex_log_contains_initial_response(
        self, isolated_runs_root, isolated_log_root, tmp_path, monkeypatch
    ):
        """Log must show readable text from the initial turn, not JSON-RPC frames."""
        bin_dir = self._make_fake_codex_interactive(
            tmp_path, thread_id="iact-thread-0002", steer_wait_seconds=0.5
        )
        monkeypatch.setenv("PATH", bin_dir + ":" + os.environ.get("PATH", ""))
        name = "codex-iact-log"
        ns = argparse.Namespace(
            name=name, command=[], interactive=True, prompt_file=None,
            submit_mode=None, idle_timeout=None,
            harness="codex", prompt="say hi", model=None, agent_mode=None,
            harness_args=[],
        )
        agent_run.cmd_launch(ns)
        state_dir = isolated_runs_root / name
        log_dir = isolated_log_root / name

        self._wait_for_terminal(state_dir, timeout=12.0)


        log_path = log_dir / "log"
        log_content = log_path.read_text() if log_path.exists() else ""
        assert "initial response" in log_content, (
            f"Log must contain readable agent text, got: {log_content!r}"
        )
        assert '"method"' not in log_content, (
            f"JSON-RPC frames must not appear in the run log: {log_content!r}"
        )

    def test_interactive_codex_steer_lands_and_answered(
        self, isolated_runs_root, isolated_log_root, tmp_path, monkeypatch
    ):
        """agent-run steer sends turn/steer or turn/start to the app-server; response lands in log."""
        # steer_wait_seconds=5.0: fake waits long enough for the test to send a steer.
        bin_dir = self._make_fake_codex_interactive(
            tmp_path, thread_id="iact-thread-0003", steer_wait_seconds=5.0
        )
        monkeypatch.setenv("PATH", bin_dir + ":" + os.environ.get("PATH", ""))
        name = "codex-iact-steer"
        ns = argparse.Namespace(
            name=name, command=[], interactive=True, prompt_file=None,
            submit_mode=None, idle_timeout=None,
            harness="codex", prompt="initial prompt", model=None, agent_mode=None,
            harness_args=[],
        )

        agent_run.cmd_launch(ns)
        state_dir = isolated_runs_root / name
        log_dir = isolated_log_root / name

        # Wait for running, then steer after a brief settle.
        reached = self._wait_for_running(state_dir, timeout=10.0)
        assert reached, "Run must reach running before steer"
        time.sleep(0.8)  # let initial turn complete so the agent is idle

        # Send steer via cmd_steer. The fake app-server has no real codex
        # rollout file to witness against, so simulate one rising right after
        # the FIFO write -- the transport behavior under test (turn/steer
        # dispatch and the response landing in the log) is independent of
        # which store backs the witness.
        witness_calls = {"n": 0}

        def fake_witness(_state_dir, _log_dir):
            witness_calls["n"] += 1
            return 0 if witness_calls["n"] == 1 else 1

        monkeypatch.setattr(agent_run, "_submission_witness_count", fake_witness)
        steer_ns = argparse.Namespace(name=name, message=["STEER_MSG"], raw=False, esc=False)
        rc = agent_run.cmd_steer(steer_ns)
        assert rc == 0, "steer must exit 0 on an interactive run"

        # Wait for the run to finish (fake exits after handling one steer).
        self._wait_for_terminal(state_dir, timeout=15.0)

        log_path = log_dir / "log"
        log_content = log_path.read_text() if log_path.exists() else ""
        assert "steered response" in log_content, (
            f"Log must contain the steered response, got: {log_content!r}"
        )

    def test_interactive_codex_fallback_to_missing_when_appserver_fails(
        self, isolated_runs_root, isolated_log_root, tmp_path, monkeypatch
    ):
        """When app-server fails, session.json has confidence=missing but run still starts."""
        bin_dir = self._make_fake_codex_interactive(tmp_path, appserver_works=False)
        monkeypatch.setenv("PATH", bin_dir + ":" + os.environ.get("PATH", ""))
        name = "codex-iact-missing"
        ns = argparse.Namespace(
            name=name, command=[], interactive=True, prompt_file=None,
            submit_mode=None, idle_timeout=None,
            harness="codex", prompt="hi", model=None, agent_mode=None,
            harness_args=[],
        )
        agent_run.cmd_launch(ns)
        state_dir = isolated_runs_root / name
        log_dir = isolated_log_root / name

        self._wait_for_terminal(state_dir, timeout=15.0)

        session = agent_run._read_session_json(log_dir)
        assert session is not None, "session.json must always be written"
        assert session["confidence"] == "missing"
        assert session["session_id"] is None


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

    def test_opencode_oneshot_writes_session_json(
        self, isolated_runs_root, isolated_log_root, tmp_path, monkeypatch
    ):
        """opencode run (one-shot) attempts mint-then-attach; falls back to missing
        if the fake binary does not speak HTTP, but session.json is always written."""
        bin_dir = self._make_fake_opencode(tmp_path)
        monkeypatch.setenv("PATH", bin_dir + ":" + os.environ.get("PATH", ""))
        name = "oc-oneshot-1"
        status, session = _launch_and_wait(
            isolated_runs_root, isolated_log_root, name,
            harness="opencode",
            interactive=False,
            prompt="do it",
            timeout=40.0,  # health poll timeout is 30s
        )
        # The run may produce launch_failed because the fake binary doesn't speak HTTP,
        # but session.json must always be written.
        assert session is not None
        assert session["harness"] == "opencode"
        # With a fake binary that doesn't serve HTTP, the mint fails → missing.
        # A real opencode would produce certain.
        assert session["confidence"] == "missing"


# ---------------------------------------------------------------------------
# Opencode session-directory identity check — guards against port-race attack
# ---------------------------------------------------------------------------

class TestOpencodeSessionIdentityCheck:
    """The session-directory identity check in _opencode_mint_session is the
    fix for a critical vulnerability: a foreign opencode server that wins the
    port race would otherwise get its session id recorded as certain.

    These tests verify that a responder returning a session whose directory
    does not match the launch cwd produces None (confidence=missing), never
    a session id. Deleting the identity check causes this class to fail.
    """

    def _start_session_server(self, tmp_path: Path, session_directory: str) -> tuple[int, object]:
        """Spin up a minimal HTTP server that responds to POST /session.

        Returns (port, server) where server.shutdown() stops it. The server
        runs in a background daemon thread so the test can call
        _opencode_mint_session without blocking.
        """
        import http.server
        import threading

        response_body = json.dumps({
            "id": "ses_foreign_0001",
            "directory": session_directory,
        }).encode()

        class _Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                self.rfile.read(length)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response_body)))
                self.end_headers()
                self.wfile.write(response_body)

            def log_message(self, *args):
                pass  # suppress access log noise

        server = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
        port = server.server_address[1]
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return port, server

    def test_foreign_directory_yields_none(self, tmp_path):
        """A responder whose session directory does not match the launch cwd must
        yield None, never a session id (confidence must be missing, not certain).

        This guards against a foreign opencode process that wins the port race:
        without this check it would be recorded as certain. Deleting the identity
        check (the ``if os.path.realpath(session_dir) != os.path.realpath(expected_cwd)``
        branch in _opencode_mint_session) causes this test to fail.
        """
        expected_cwd = str(tmp_path / "my-launch-dir")
        foreign_dir = str(tmp_path / "foreign-server-dir")
        assert os.path.realpath(expected_cwd) != os.path.realpath(foreign_dir)

        acquire_log = tmp_path / "acquire.log"
        port, server = self._start_session_server(tmp_path, foreign_dir)
        try:
            result = agent_run._opencode_mint_session(
                port, "test-run", expected_cwd, acquire_log
            )
        finally:
            server.shutdown()

        assert result is None, (
            f"_opencode_mint_session returned {result!r} for a server whose "
            "directory does not match the launch cwd — the identity check is "
            "missing or broken. confidence must be missing, never certain."
        )

    def test_matching_directory_yields_session_id(self, tmp_path):
        """When the server's session directory equals the expected cwd,
        _opencode_mint_session must return the session id (not None).
        Verifies the identity check does not over-reject legitimate responses.
        """
        expected_cwd = str(tmp_path)
        acquire_log = tmp_path / "acquire.log"
        port, server = self._start_session_server(tmp_path, expected_cwd)
        try:
            result = agent_run._opencode_mint_session(
                port, "test-run", expected_cwd, acquire_log
            )
        finally:
            server.shutdown()

        assert result == "ses_foreign_0001", (
            f"Expected session id from a matching-directory server, got {result!r}"
        )


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


# ---------------------------------------------------------------------------
# run.json — reboot-durable launch/exit manifest (C2)
# ---------------------------------------------------------------------------

class TestRunJson:
    """Tests for the run.json persistent manifest written to log_dir."""

    def test_write_run_json_creates_file(self, isolated_log_root):
        log_dir = isolated_log_root / "myrun"
        log_dir.mkdir(parents=True, exist_ok=True)
        agent_run._write_run_json(log_dir, {"name": "myrun", "cwd": "/tmp"})
        path = log_dir / "run.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["name"] == "myrun"
        assert data["cwd"] == "/tmp"

    def test_write_run_json_merges_exit_fields(self, isolated_log_root):
        """Exit-time update preserves launch fields and adds exit fields."""
        log_dir = isolated_log_root / "myrun2"
        log_dir.mkdir(parents=True, exist_ok=True)
        agent_run._write_run_json(log_dir, {"name": "myrun2", "started_at": "2026-08-16T10:00:00Z"})
        agent_run._write_run_json(log_dir, {"exit_code": 0, "status": "done", "ended_at": "2026-08-16T10:01:00Z"})
        data = json.loads((log_dir / "run.json").read_text())
        assert data["name"] == "myrun2"
        assert data["started_at"] == "2026-08-16T10:00:00Z"
        assert data["exit_code"] == 0
        assert data["status"] == "done"

    def test_write_run_json_is_atomic(self, isolated_log_root):
        log_dir = isolated_log_root / "myrun3"
        log_dir.mkdir(parents=True, exist_ok=True)
        agent_run._write_run_json(log_dir, {"x": 1})
        assert list(log_dir.glob(".run.*.tmp")) == []

    def test_run_json_written_at_launch(
        self, isolated_runs_root, isolated_log_root, tmp_path, monkeypatch
    ):
        """A managed run writes run.json with launch fields before the fork."""
        fake = tmp_path / "claude"
        fake.write_text("#!/bin/sh\necho hi\nexit 0\n")
        fake.chmod(0o755)
        monkeypatch.setenv("PATH", str(tmp_path) + ":" + os.environ.get("PATH", ""))
        name = "runjson-launch"
        _launch_and_wait(
            isolated_runs_root, isolated_log_root, name,
            harness="claude",
            interactive=False,
            prompt="hello",
        )
        run_json_path = isolated_log_root / name / "run.json"
        assert run_json_path.exists(), "run.json must be written at launch"
        data = json.loads(run_json_path.read_text())
        assert data["name"] == name
        assert data["harness"] == "claude"
        assert isinstance(data["argv"], list)
        assert data["interactive"] is False

    def test_run_json_written_at_exit(
        self, isolated_runs_root, isolated_log_root, tmp_path, monkeypatch
    ):
        """run.json gains exit_code, ended_at, and status after the run completes."""
        fake = tmp_path / "claude"
        fake.write_text("#!/bin/sh\necho hi\nexit 0\n")
        fake.chmod(0o755)
        monkeypatch.setenv("PATH", str(tmp_path) + ":" + os.environ.get("PATH", ""))
        name = "runjson-exit"
        status, _ = _launch_and_wait(
            isolated_runs_root, isolated_log_root, name,
            harness="claude",
            interactive=False,
            prompt="hello",
        )
        run_json_path = isolated_log_root / name / "run.json"
        assert run_json_path.exists()
        data = json.loads(run_json_path.read_text())
        assert "exit_code" in data
        assert "ended_at" in data
        assert "status" in data

    def test_run_json_survives_ephemeral_state_deletion(
        self, isolated_runs_root, isolated_log_root, tmp_path, monkeypatch
    ):
        """Simulates a reboot: after /tmp dir is deleted, run.json in /var/tmp still has facts."""
        fake = tmp_path / "claude"
        fake.write_text("#!/bin/sh\necho hi\nexit 0\n")
        fake.chmod(0o755)
        monkeypatch.setenv("PATH", str(tmp_path) + ":" + os.environ.get("PATH", ""))
        name = "runjson-reboot"
        _launch_and_wait(
            isolated_runs_root, isolated_log_root, name,
            harness="claude",
            interactive=False,
            prompt="hello",
        )
        # Simulate reboot: delete the ephemeral /tmp state dir.
        state_dir = isolated_runs_root / name
        import shutil as _shutil
        if state_dir.exists():
            _shutil.rmtree(state_dir)
        assert not state_dir.exists(), "State dir should be gone (simulating reboot)"

        # After reboot, the persistent log dir still has all attribution facts.
        log_dir = isolated_log_root / name
        run_json_path = log_dir / "run.json"
        assert run_json_path.exists(), "run.json must survive in /var/tmp"
        data = json.loads(run_json_path.read_text())
        assert data.get("name") == name
        assert data.get("cwd") is not None
        assert data.get("argv") is not None
        assert data.get("started_at") is not None
        assert data.get("exit_code") is not None

    def test_raw_run_also_gets_run_json(
        self, isolated_runs_root, isolated_log_root, tmp_path, monkeypatch
    ):
        """Raw runs get run.json too (it is launch metadata, not session acquisition)."""
        fake_echo = tmp_path / "myecho.sh"
        fake_echo.write_text("#!/bin/sh\necho hello\n")
        fake_echo.chmod(0o755)
        name = "runjson-raw"
        ns = argparse.Namespace(
            name=name,
            command=[str(fake_echo)],
            interactive=False,
            prompt_file=None,
            submit_mode=None,
            idle_timeout=None,
        )
        agent_run.cmd_launch(ns)
        state_dir = isolated_runs_root / name
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            try:
                s = (state_dir / "status").read_text().strip()
            except FileNotFoundError:
                s = "starting"
            if s in agent_run.TERMINAL_STATUSES:
                break
            time.sleep(0.05)
        run_json_path = isolated_log_root / name / "run.json"
        assert run_json_path.exists(), "Raw run must also get run.json"
        data = json.loads(run_json_path.read_text())
        assert data["harness"] is None  # raw run has no harness
        # Raw run still must not get session.json.
        assert not (isolated_log_root / name / "session.json").exists()


# ---------------------------------------------------------------------------
# End-to-end through main() — argv parser → cmd_launch round trip
# ---------------------------------------------------------------------------

class TestEndToEndThroughMain:
    """Tests that drive managed mode through agent_run.main() so the
    _parse_launch_argv → main() → cmd_launch path is exercised end-to-end.
    This is the layer that `_launch_and_wait` (which starts from cmd_launch)
    skips entirely.
    """

    def _make_fake_claude(self, tmp_path: Path) -> str:
        """Fake claude that prints its argv and exits 0."""
        fake = tmp_path / "claude"
        fake.write_text("#!/bin/sh\nprintf 'argv: %s\\n' \"$@\"\nexit 0\n")
        fake.chmod(0o755)
        return str(tmp_path)

    def _make_fake_codex_oneshot(self, tmp_path: Path, thread_id: str = "main-thread-001") -> str:
        """Fake codex app-server for one-shot runs."""
        fake_dir = tmp_path / "bin"
        fake_dir.mkdir(parents=True, exist_ok=True)
        fake = fake_dir / "codex"
        fake.write_text(
            "#!/usr/bin/env python3\n"
            "import sys, json, time\n"
            "def send(obj):\n"
            "    sys.stdout.write(json.dumps(obj) + '\\n')\n"
            "    sys.stdout.flush()\n"
            "def recv():\n"
            "    line = sys.stdin.readline()\n"
            "    return json.loads(line.strip()) if line.strip() else None\n"
            "if len(sys.argv) < 2 or sys.argv[1] != 'app-server':\n"
            "    sys.exit(1)\n"
            f"TID = '{thread_id}'\n"
            "msg = recv()\n"
            "send({'id': msg['id'], 'result': {'userAgent': 'codex_exec/test'}})\n"
            "recv()  # initialized\n"
            "msg = recv()\n"
            "send({'id': msg['id'], 'result': {'thread': {'id': TID, 'sessionId': TID, 'path': '~/.codex/sessions/r.jsonl'}}})\n"
            "msg = recv()\n"
            "send({'id': msg['id'], 'result': {'turn': {'id': 'turn-1', 'status': 'inProgress'}}})\n"
            "send({'method': 'item/agentMessage/delta', 'params': {'delta': 'main answer'}})\n"
            "send({'method': 'turn/completed', 'params': {'turn': {'id': 'turn-1', 'status': 'completed'}}})\n"
            "time.sleep(5)\n"
        )
        fake.chmod(0o755)
        return str(fake_dir)

    def _wait_terminal(self, state_dir: Path, timeout: float = 15.0) -> str:
        """Wait for terminal status. Kills the runner on timeout."""
        deadline = time.monotonic() + timeout
        status = "starting"
        while time.monotonic() < deadline:
            try:
                status = (state_dir / "status").read_text().strip()
            except FileNotFoundError:
                pass
            if status in agent_run.TERMINAL_STATUSES:
                break
            time.sleep(0.05)
        if status not in agent_run.TERMINAL_STATUSES:
            _kill_run_pid(state_dir)
        return status

    def test_main_claude_oneshot_produces_session_json(
        self, isolated_runs_root, isolated_log_root, tmp_path, monkeypatch
    ):
        """main() with --harness claude produces a run with session.json via the push path."""
        bin_dir = self._make_fake_claude(tmp_path)
        monkeypatch.setenv("PATH", bin_dir + ":" + os.environ.get("PATH", ""))
        name = "main-claude-oneshot"
        rc = agent_run.main(["--harness", "claude", "--prompt", "say hi", name])
        assert rc == 0, f"main() returned non-zero: {rc}"
        state_dir = isolated_runs_root / name
        self._wait_terminal(state_dir)
        session = agent_run._read_session_json(isolated_log_root / name)
        assert session is not None
        assert session["harness"] == "claude"
        assert session["acquisition"] == "pushed"
        assert session["confidence"] == "certain"


    def test_main_codex_oneshot_produces_session_json(
        self, isolated_runs_root, isolated_log_root, tmp_path, monkeypatch
    ):
        """main() with --harness codex drives the app-server path end-to-end."""
        bin_dir = self._make_fake_codex_oneshot(tmp_path, thread_id="main-e2e-tid")
        monkeypatch.setenv("PATH", bin_dir + ":" + os.environ.get("PATH", ""))
        name = "main-codex-oneshot"
        rc = agent_run.main(["--harness", "codex", "--prompt", "say hello", name])
        assert rc == 0
        state_dir = isolated_runs_root / name
        self._wait_terminal(state_dir)
        session = agent_run._read_session_json(isolated_log_root / name)
        assert session is not None
        assert session["session_id"] == "main-e2e-tid"
        assert session["confidence"] == "certain"

    def test_main_codex_questions_policy_reaches_appserver(
        self, isolated_runs_root, isolated_log_root, tmp_path, monkeypatch
    ):
        fake_dir = tmp_path / "codex-bin"
        fake_dir.mkdir()
        fake = fake_dir / "codex"
        fake.write_text(
            "#!/usr/bin/env python3\n"
            "import json, sys\n"
            "with open(sys.argv[0] + '.args', 'w') as f: json.dump(sys.argv[1:], f)\n"
            "def recv(): return json.loads(sys.stdin.readline())\n"
            "def send(obj): print(json.dumps(obj), flush=True)\n"
            "msg = recv(); send({'id': msg['id'], 'result': {}}); recv()\n"
            "msg = recv(); send({'id': msg['id'], 'result': {'thread': {'id': 'policy-thread'}}})\n"
            "msg = recv(); send({'id': msg['id'], 'result': {'turn': {'id': 'turn-1'}}})\n"
            "send({'method': 'turn/completed', 'params': {'turn': {'status': 'completed'}}})\n"
        )
        fake.chmod(0o755)
        monkeypatch.setenv("PATH", str(fake_dir) + ":" + os.environ.get("PATH", ""))
        name = "main-codex-question-policy"
        assert agent_run.main(["--harness", "codex", "--prompt", "hi", name]) == 0
        self._wait_terminal(isolated_runs_root / name)
        args = json.loads((fake.with_name("codex.args")).read_text())
        assert "tools.experimental_request_user_input={enabled=false}" in args

    def test_main_codex_questions_enabled_arm_reaches_appserver(
        self, isolated_runs_root, isolated_log_root, tmp_path, monkeypatch
    ):
        """--enable-questions flips the codex policy arg to {enabled=true}."""
        fake_dir = tmp_path / "codex-bin-enabled"
        fake_dir.mkdir()
        fake = fake_dir / "codex"
        fake.write_text(
            "#!/usr/bin/env python3\n"
            "import json, sys\n"
            "with open(sys.argv[0] + '.args', 'w') as f: json.dump(sys.argv[1:], f)\n"
            "def recv(): return json.loads(sys.stdin.readline())\n"
            "def send(obj): print(json.dumps(obj), flush=True)\n"
            "msg = recv(); send({'id': msg['id'], 'result': {}}); recv()\n"
            "msg = recv(); send({'id': msg['id'], 'result': {'thread': {'id': 'enabled-thread'}}})\n"
            "msg = recv(); send({'id': msg['id'], 'result': {'turn': {'id': 'turn-1'}}})\n"
            "send({'method': 'turn/completed', 'params': {'turn': {'status': 'completed'}}})\n"
        )
        fake.chmod(0o755)
        monkeypatch.setenv("PATH", str(fake_dir) + ":" + os.environ.get("PATH", ""))
        name = "main-codex-question-enabled"
        assert agent_run.main([
            "--harness", "codex", "--enable-questions", "--prompt", "hi", name,
        ]) == 0
        self._wait_terminal(isolated_runs_root / name)
        args = json.loads((fake.with_name("codex.args")).read_text())
        assert "tools.experimental_request_user_input={enabled=true}" in args
        assert "tools.experimental_request_user_input={enabled=false}" not in args

    def test_main_opencode_managed_launch_delivers_policy_env(
        self, isolated_runs_root, isolated_log_root, tmp_path, monkeypatch
    ):
        """OPENCODE_CONFIG_CONTENT with the three policy keys reaches the child process."""
        fake = tmp_path / "opencode"
        fake.write_text(
            "#!/bin/sh\n"
            "printf 'ENVPROBE:%s\\n' \"$OPENCODE_CONFIG_CONTENT\"\n"
        )
        fake.chmod(0o755)
        monkeypatch.setenv("PATH", str(tmp_path) + ":" + os.environ.get("PATH", ""))
        name = "oc-policy-env"
        assert agent_run.main(["--harness", "opencode", "--prompt", "hi", name]) == 0
        self._wait_terminal(isolated_runs_root / name)
        log = (isolated_log_root / name / "log").read_text()
        probe_lines = [l for l in log.splitlines() if l.startswith("ENVPROBE:")]
        assert probe_lines, f"ENVPROBE line not found in log: {log!r}"
        cfg = json.loads(probe_lines[0].split("ENVPROBE:", 1)[1].strip())
        assert cfg["permission"]["question"] == "deny"
        assert cfg["permission"]["plan_enter"] == "deny"
        assert cfg["permission"]["plan_exit"] == "deny"

    def test_main_claude_escape_hatches_reach_argv_via_cli(
        self, isolated_runs_root, isolated_log_root, tmp_path, monkeypatch
    ):
        """--enable-planning and --enable-questions remove the deny flags through main()."""
        fake = tmp_path / "claude"
        fake.write_text(
            "#!/bin/sh\n"
            "for arg in \"$@\"; do printf '<%s>\\n' \"$arg\"; done\n"
            "printf '<stdin:%s>\\n' \"$(cat)\"\n"
        )
        fake.chmod(0o755)
        monkeypatch.setenv("PATH", str(tmp_path) + ":" + os.environ.get("PATH", ""))
        name = "main-claude-escape-hatches"
        assert agent_run.main([
            "--harness", "claude",
            "--enable-questions", "--enable-planning",
            "--prompt", "hi",
            name,
        ]) == 0
        self._wait_terminal(isolated_runs_root / name)
        log = (isolated_log_root / name / "log").read_text()
        assert "<--disallowedTools>" not in log, (
            f"--disallowedTools must be absent when both escape hatches are set: {log!r}"
        )


    def test_main_model_accepted_for_codex(
        self, isolated_runs_root, isolated_log_root, tmp_path, monkeypatch
    ):
        """main() accepts --model with --harness codex and the launch completes."""
        bin_dir = self._make_fake_codex_oneshot(tmp_path, thread_id="model-accept-tid")
        monkeypatch.setenv("PATH", bin_dir + ":" + os.environ.get("PATH", ""))
        name = "main-codex-model-accept"
        rc = agent_run.main([
            "--harness", "codex", "--model", "gpt-5", "--prompt", "hi", name,
        ])
        assert rc == 0
        self._wait_terminal(isolated_runs_root / name)
        session = agent_run._read_session_json(isolated_log_root / name)
        assert session is not None
        assert session["session_id"] == "model-accept-tid"

    def test_main_session_id_flag_no_longer_accepted(
        self, isolated_runs_root, isolated_log_root, monkeypatch
    ):
        """--session-id was removed: main() rejects it before creating state."""
        import uuid as _uuid
        name = "main-sid-removed"
        with pytest.raises(SystemExit) as exc_info:
            agent_run.main([
                "--harness", "claude",
                "--session-id", str(_uuid.uuid4()),
                "--prompt", "hi",
                name,
            ])
        assert exc_info.value.code != 0
        assert not (isolated_runs_root / name).exists()


    def test_main_bad_harness_arg_rejected_before_starting(
        self, isolated_runs_root, isolated_log_root, monkeypatch
    ):
        """main() with forbidden --harness-arg exits non-zero; no phantom starting run."""
        name = "main-bad-harg"
        with pytest.raises(SystemExit) as exc_info:
            agent_run.main([
                "--harness", "claude",
                "--prompt", "hi",
                "--harness-arg", "--session",
                name,
            ])
        assert exc_info.value.code != 0
        assert not (isolated_runs_root / name).exists()

    def test_main_claude_minted_id_matches_argv(
        self, isolated_runs_root, isolated_log_root, tmp_path, monkeypatch
    ):
        """The internally minted UUID4 reaches claude's argv and session.json unchanged."""
        bin_dir = self._make_fake_claude(tmp_path)
        monkeypatch.setenv("PATH", bin_dir + ":" + os.environ.get("PATH", ""))
        fixed_id = "11111111-2222-4333-8444-555555555555"
        monkeypatch.setattr(agent_run.uuid, "uuid4", lambda: _uuid.UUID(fixed_id))
        name = "main-claude-minted-id"
        rc = agent_run.main(["--harness", "claude", "--prompt", "say hi", name])
        assert rc == 0
        state_dir = isolated_runs_root / name
        _wait_terminal(state_dir)
        argv = json.loads((state_dir / "argv").read_text())
        assert argv[argv.index("--session-id") + 1] == fixed_id
        session = agent_run._read_session_json(isolated_log_root / name)
        assert session["session_id"] == fixed_id


# ---------------------------------------------------------------------------
# Escape-hatch wiring: --enable-planning × --enable-questions, per harness.
#
# Each case captures the argv/env actually delivered to a fake child process,
# so wiring the two booleans to each other or dropping a deny target fails here.
# ---------------------------------------------------------------------------

def _policy_flags(enable_planning: bool, enable_questions: bool) -> list[str]:
    return (
        (["--enable-planning"] if enable_planning else [])
        + (["--enable-questions"] if enable_questions else [])
    )


def _assert_opencode_policy_values(
    cfg: dict, *, enable_planning: bool, enable_questions: bool, where: str
) -> None:
    """Assert the three policy keys carry the expected values at global scope
    and in the "build" agent block, which is the default agent selection.
    """
    expected_question = "allow" if enable_questions else "deny"
    expected_plan = "allow" if enable_planning else "deny"
    perm = cfg.get("permission", {})
    build_perm = cfg.get("agent", {}).get("build", {}).get("permission", {})
    for scope, block in (("permission", perm), ("agent.build.permission", build_perm)):
        for key, expected in (
            ("question", expected_question),
            ("plan_enter", expected_plan),
            ("plan_exit", expected_plan),
        ):
            assert block.get(key) == expected, (
                f"{where}: {scope}.{key}={block.get(key)!r} != {expected!r}; "
                f"full config: {cfg!r}"
            )


class TestFlagCombinations:
    def _wait_terminal(self, state_dir: Path, timeout: float = 12.0) -> str:
        deadline = time.monotonic() + timeout
        status = "starting"
        while time.monotonic() < deadline:
            try:
                status = (state_dir / "status").read_text().strip()
            except FileNotFoundError:
                pass
            if status in agent_run.TERMINAL_STATUSES:
                break
            time.sleep(0.05)
        if status not in agent_run.TERMINAL_STATUSES:
            _kill_run_pid(state_dir)
        return status

    def _fake_codex_on_path(self, bin_dir: Path, monkeypatch) -> Path:
        """Install a codex app-server stub that records its argv and completes one turn.

        Returns the path the stub writes its argv JSON to.
        """
        bin_dir.mkdir()
        argv_file = bin_dir / "codex.argv"
        fake = bin_dir / "codex"
        fake.write_text(
            "#!/usr/bin/env python3\n"
            "import json, sys\n"
            f"open({str(argv_file)!r}, 'w').write(json.dumps(sys.argv[1:]))\n"
            "def recv(): return json.loads(sys.stdin.readline())\n"
            "def send(obj): print(json.dumps(obj), flush=True)\n"
            "msg = recv(); send({'id': msg['id'], 'result': {'userAgent': 'test'}}); recv()\n"
            "msg = recv(); send({'id': msg['id'], 'result': {'thread': {'id': 'tid', 'sessionId': 'tid', 'path': '~/.codex/r.jsonl'}}})\n"
            "msg = recv(); send({'id': msg['id'], 'result': {'turn': {'id': 't1'}}})\n"
            "send({'method': 'turn/completed', 'params': {'turn': {'status': 'completed'}}})\n"
        )
        fake.chmod(0o755)
        monkeypatch.setenv("PATH", str(bin_dir) + ":" + os.environ.get("PATH", ""))
        return argv_file

    @pytest.mark.parametrize("enable_planning,enable_questions,expected_denied", [
        (False, False, ["EnterPlanMode", "ExitPlanMode", "AskUserQuestion"]),
        (True,  False, ["AskUserQuestion"]),
        (False, True,  ["EnterPlanMode", "ExitPlanMode"]),
        (True,  True,  []),
    ])
    def test_claude_flag_combinations_exact_deny_targets(
        self, isolated_runs_root, isolated_log_root, tmp_path, monkeypatch,
        enable_planning, enable_questions, expected_denied,
    ):
        argv_file = tmp_path / "claude.argv"
        script = tmp_path / "claude"
        script.write_text(
            "#!/usr/bin/env python3\n"
            "import json, sys\n"
            f"open({str(argv_file)!r}, 'w').write(json.dumps(sys.argv[1:]))\n"
            "sys.exit(0)\n"
        )
        script.chmod(0o755)
        monkeypatch.setenv("PATH", str(tmp_path) + ":" + os.environ.get("PATH", ""))

        name = f"policy-claude-ep{int(enable_planning)}-eq{int(enable_questions)}"
        assert agent_run.main([
            "--harness", "claude",
            *_policy_flags(enable_planning, enable_questions),
            "--prompt", "hi", name,
        ]) == 0
        self._wait_terminal(isolated_runs_root / name)

        assert argv_file.exists(), (
            f"fake claude did not write argv file; "
            f"log: {(isolated_log_root / name / 'log').read_text()!r}"
        )
        argv = json.loads(argv_file.read_text())
        denied = [argv[i + 1] for i, tok in enumerate(argv[:-1]) if tok == "--disallowedTools"]
        assert sorted(denied) == sorted(expected_denied), (
            f"claude enable_planning={enable_planning} enable_questions={enable_questions}: "
            f"denied tools {denied!r} != expected {expected_denied!r}; full argv: {argv}"
        )

    @pytest.mark.parametrize("enable_planning,enable_questions", [
        (False, False),
        (True,  True),
    ])
    def test_opencode_flag_combinations_exact_env(
        self, isolated_runs_root, isolated_log_root, tmp_path, monkeypatch,
        enable_planning, enable_questions,
    ):
        """Full main() path proves OPENCODE_CONFIG_CONTENT reaches the runner
        grandchild, not only the prefork mint.

        Only the fully-deny and fully-allow cases run here because each costs the
        prefork-mint health poll; the crossed combinations are covered without
        that cost in TestPreforkMintEnvDelivery.
        """
        env_file = tmp_path / "opencode.env"
        script = tmp_path / "opencode"
        script.write_text(
            "#!/usr/bin/env python3\n"
            "import os, sys\n"
            f"open({str(env_file)!r}, 'w').write(os.environ.get('OPENCODE_CONFIG_CONTENT', ''))\n"
            "sys.exit(0)\n"
        )
        script.chmod(0o755)
        monkeypatch.setenv("PATH", str(tmp_path) + ":" + os.environ.get("PATH", ""))

        name = f"policy-oc-ep{int(enable_planning)}-eq{int(enable_questions)}"
        assert agent_run.main([
            "--harness", "opencode",
            *_policy_flags(enable_planning, enable_questions),
            "--prompt", "hi", name,
        ]) == 0
        # The prefork-mint health poll runs up to 30 s before falling back to
        # launching the runner; 40 s covers the poll plus the runner exec.
        self._wait_terminal(isolated_runs_root / name, timeout=40.0)

        assert env_file.exists(), (
            f"fake opencode did not write env file; log: "
            f"{(isolated_log_root / name / 'log').read_text()!r}"
        )
        cfg = json.loads(env_file.read_text())
        _assert_opencode_policy_values(
            cfg, enable_planning=enable_planning, enable_questions=enable_questions,
            where=f"opencode ep={enable_planning} eq={enable_questions}",
        )

    @pytest.mark.parametrize("enable_questions,expected_arg", [
        (False, "tools.experimental_request_user_input={enabled=false}"),
        (True,  "tools.experimental_request_user_input={enabled=true}"),
    ])
    def test_codex_flag_combinations_exact_argv(
        self, isolated_runs_root, isolated_log_root, tmp_path, monkeypatch,
        enable_questions, expected_arg,
    ):
        """Codex has no planning equivalent, so only enable_questions varies;
        --enable-planning with codex is rejected by the parser.
        """
        argv_file = self._fake_codex_on_path(tmp_path / "codex-bin", monkeypatch)

        name = f"policy-codex-eq{int(enable_questions)}"
        assert agent_run.main([
            "--harness", "codex",
            *_policy_flags(False, enable_questions),
            "--prompt", "hi", name,
        ]) == 0
        self._wait_terminal(isolated_runs_root / name)

        assert argv_file.exists(), (
            f"fake codex did not write argv file; log: "
            f"{(isolated_log_root / name / 'log').read_text()!r}"
        )
        argv = json.loads(argv_file.read_text())
        assert expected_arg in argv, (
            f"codex eq={enable_questions}: {expected_arg!r} not in argv {argv!r}"
        )

    def test_codex_caller_dash_c_lands_last(
        self, isolated_runs_root, isolated_log_root, tmp_path, monkeypatch
    ):
        """codex resolves duplicate -c keys last-occurrence-wins, so a caller's
        -c must be positioned after the managed policy -c to take effect.
        """
        argv_file = self._fake_codex_on_path(tmp_path / "codex-last-bin", monkeypatch)

        name = "policy-codex-caller-last"
        caller_c = "tools.experimental_request_user_input={enabled=true}"
        assert agent_run.main([
            "--harness", "codex", "--prompt", "hi",
            "--harness-arg", "-c",
            "--harness-arg", caller_c,
            name,
        ]) == 0
        self._wait_terminal(isolated_runs_root / name)

        assert argv_file.exists(), (
            f"fake codex did not write argv file; log: "
            f"{(isolated_log_root / name / 'log').read_text()!r}"
        )
        argv = json.loads(argv_file.read_text())

        policy_arg = "tools.experimental_request_user_input={enabled=false}"
        assert policy_arg in argv, f"policy arg missing from argv: {argv!r}"
        assert caller_c in argv, f"caller arg missing from argv: {argv!r}"
        assert argv.index(caller_c) > argv.index(policy_arg), (
            f"caller -c arg {caller_c!r} appears before policy arg {policy_arg!r}: {argv!r}"
        )


class TestPreforkMintEnvDelivery:
    """_opencode_prefork_mint must pass the policy to the temporary opencode
    process it starts, or the minted session is created under an allow config.
    A fake Popen captures the env without starting a real process.
    """

    @pytest.mark.parametrize("enable_planning,enable_questions", [
        (False, False),
        (True,  False),
        (False, True),
        (True,  True),
    ])
    def test_prefork_mint_delivers_policy_env(
        self, tmp_path, monkeypatch, enable_planning, enable_questions
    ):
        captured_envs: list[dict] = []

        class _FakePopen:
            def __init__(self, cmd, **kwargs):
                captured_envs.append(dict(kwargs.get("env", {})))
                self.pid = os.getpid()
                self.returncode = 0

            def terminate(self):
                pass

            def wait(self, timeout=None):
                return 0

            def kill(self):
                pass

        monkeypatch.setattr(subprocess, "Popen", _FakePopen)

        # Returns None (no real opencode to poll), but the env is captured.
        agent_run._opencode_prefork_mint(
            12345, "test-run", str(tmp_path), tmp_path / "acquire.log",
            enable_planning=enable_planning,
            enable_questions=enable_questions,
            opencode_agent_mode="build",
        )
        assert captured_envs, "Popen was not called"
        env_content = captured_envs[0].get("OPENCODE_CONFIG_CONTENT", "")
        assert env_content, "OPENCODE_CONFIG_CONTENT not set in prefork-mint env"
        try:
            cfg = json.loads(env_content)
        except json.JSONDecodeError:
            raise AssertionError(
                f"OPENCODE_CONFIG_CONTENT is not valid JSON: {env_content!r}"
            )
        _assert_opencode_policy_values(
            cfg, enable_planning=enable_planning, enable_questions=enable_questions,
            where=f"prefork mint ep={enable_planning} eq={enable_questions}",
        )


# ---------------------------------------------------------------------------
# Teardown tests: no app-server process must survive kill or signal
# ---------------------------------------------------------------------------

class TestAppServerTeardown:
    """Verify that codex app-server processes are killed when the runner is
    signalled or `agent-run kill` is used.  These tests MUST FAIL against
    the code before appserver_pid was added to _AUX_PID_FIELDS.
    """

    def _make_persistent_codex(self, tmp_path: Path, thread_id: str = "teardown-tid") -> str:
        """Fake codex app-server that stays alive until killed; does not exit on EOF."""
        fake_dir = tmp_path / "bin"
        fake_dir.mkdir(parents=True, exist_ok=True)
        fake = fake_dir / "codex"
        # The stub completes the handshake and the initial turn, then blocks
        # indefinitely.  It ignores stdin EOF so it stays alive even after the
        # runner's stdin pipe closes — mimicking a real app-server talking to an API.
        fake.write_text(
            "#!/usr/bin/env python3\n"
            "import sys, json, time, signal\n"
            "signal.signal(signal.SIGTERM, lambda *a: sys.exit(0))\n"
            "def send(obj):\n"
            "    sys.stdout.write(json.dumps(obj) + '\\n')\n"
            "    sys.stdout.flush()\n"
            "def recv():\n"
            "    line = sys.stdin.readline()\n"
            "    return json.loads(line.strip()) if line.strip() else None\n"
            "if len(sys.argv) < 2 or sys.argv[1] != 'app-server':\n"
            "    sys.exit(1)\n"
            f"TID = '{thread_id}'\n"
            "msg = recv()\n"
            "send({'id': msg['id'], 'result': {'userAgent': 'codex_exec/test'}})\n"
            "recv()\n"  # initialized
            "msg = recv()\n"
            "send({'id': msg['id'], 'result': {'thread': {'id': TID, 'sessionId': TID, 'path': '~/.codex/r.jsonl'}}})\n"
            "msg = recv()\n"
            "send({'id': msg['id'], 'result': {'turn': {'id': 'turn-1', 'status': 'inProgress'}}})\n"
            "send({'method': 'item/agentMessage/delta', 'params': {'delta': 'working...'}})\n"
            "# Block indefinitely — a real app-server stays alive after answering.\n"
            "while True:\n"
            "    time.sleep(3600)\n"
        )
        fake.chmod(0o755)
        return str(fake_dir)

    def _get_codex_pids(self, tmp_path: Path) -> list[int]:
        """Return pids of any codex app-server processes in our fake bin dir."""
        import subprocess as _sp
        fake_bin = str(tmp_path / "bin" / "codex")
        try:
            out = _sp.check_output(
                ["pgrep", "-f", fake_bin],
                stderr=_sp.DEVNULL,
                text=True,
            )
            return [int(p) for p in out.strip().splitlines() if p.strip()]
        except _sp.CalledProcessError:
            return []

    def _wait_for_status(self, state_dir: Path, target: str, timeout: float = 10.0) -> bool:
        """Wait for a specific status, returning True if reached within timeout.
        Kills the runner on timeout so a wedged process cannot outlive the test.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                s = (state_dir / "status").read_text().strip()
                if s == target:
                    return True
                if s in agent_run.TERMINAL_STATUSES and s != target:
                    return False
            except FileNotFoundError:
                pass
            time.sleep(0.05)
        _kill_run_pid(state_dir)
        return False

    def test_appserver_killed_on_sigterm_to_runner(
        self, isolated_runs_root, isolated_log_root, tmp_path, monkeypatch
    ):
        """SIGTERM to the runner must kill the codex app-server (appserver_pid in _AUX_PID_FIELDS)."""
        bin_dir = self._make_persistent_codex(tmp_path, thread_id="teardown-term")
        monkeypatch.setenv("PATH", bin_dir + ":" + os.environ.get("PATH", ""))
        name = "teardown-sigterm"

        ns = argparse.Namespace(
            name=name, command=[], interactive=False, prompt_file=None,
            submit_mode=None, idle_timeout=None,
            harness="codex", prompt="work hard", model=None, agent_mode=None,
            harness_args=[],
        )
        rc = agent_run.cmd_launch(ns)
        assert rc == 0

        state_dir = isolated_runs_root / name
        # Wait for runner to start and app-server to be running.
        reached = self._wait_for_status(state_dir, "running", timeout=10.0)

        assert reached, f"Run never reached running; status={agent_run._read(state_dir / 'status')!r}"

        # Confirm app-server is alive.
        pids_before = self._get_codex_pids(tmp_path)
        assert len(pids_before) >= 1, "App-server must be alive before kill"

        # SIGTERM the runner via agent-run kill.
        kill_ns = argparse.Namespace(name=name, signal="TERM", force=False)
        agent_run.cmd_kill(kill_ns)

        # Wait for terminal status.
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline:
            try:
                s = (state_dir / "status").read_text().strip()
                if s in agent_run.TERMINAL_STATUSES:
                    break
            except FileNotFoundError:
                pass
            time.sleep(0.1)

        # App-server must be gone.
        time.sleep(0.3)  # let teardown complete
        pids_after = self._get_codex_pids(tmp_path)
        assert pids_after == [], (
            f"App-server process(es) survived SIGTERM to runner: {pids_after}. "
            "appserver_pid must be in _AUX_PID_FIELDS for teardown to reach it."
        )

    def test_appserver_killed_on_agent_run_kill_KILL(
        self, isolated_runs_root, isolated_log_root, tmp_path, monkeypatch
    ):
        """agent-run kill KILL must also kill the app-server (via _force_kill / appserver_pid)."""
        bin_dir = self._make_persistent_codex(tmp_path, thread_id="teardown-kill")
        monkeypatch.setenv("PATH", bin_dir + ":" + os.environ.get("PATH", ""))
        name = "teardown-sigkill"

        ns = argparse.Namespace(
            name=name, command=[], interactive=False, prompt_file=None,
            submit_mode=None, idle_timeout=None,
            harness="codex", prompt="work hard", model=None, agent_mode=None,
            harness_args=[],
        )
        rc = agent_run.cmd_launch(ns)
        assert rc == 0

        state_dir = isolated_runs_root / name
        reached = self._wait_for_status(state_dir, "running", timeout=10.0)

        assert reached, "Run never reached running"

        pids_before = self._get_codex_pids(tmp_path)
        assert len(pids_before) >= 1, "App-server must be alive before KILL"

        # agent-run kill KILL (the documented user-facing path the brief reproduced).
        kill_ns = argparse.Namespace(name=name, signal="KILL", force=False)
        agent_run.cmd_kill(kill_ns)

        time.sleep(0.5)
        pids_after = self._get_codex_pids(tmp_path)
        assert pids_after == [], (
            f"App-server survived agent-run kill KILL: {pids_after}"
        )


# ---------------------------------------------------------------------------
# Interactive codex: result:null frame and steer error response
# ---------------------------------------------------------------------------

class TestCodexRpcEdgeCases:
    """Tests for edge cases in the interactive codex JSON-RPC protocol."""

    def _make_null_result_codex(self, tmp_path: Path, thread_id: str = "null-result-tid") -> str:
        """Fake app-server that emits a result:null frame after the first turn."""
        fake_dir = tmp_path / "bin"
        fake_dir.mkdir(parents=True, exist_ok=True)
        fake = fake_dir / "codex"
        fake.write_text(
            "#!/usr/bin/env python3\n"
            "import sys, json\n"
            "def send(obj):\n"
            "    sys.stdout.write(json.dumps(obj) + '\\n')\n"
            "    sys.stdout.flush()\n"
            "def recv():\n"
            "    line = sys.stdin.readline()\n"
            "    return json.loads(line.strip()) if line.strip() else None\n"
            "if len(sys.argv) < 2 or sys.argv[1] != 'app-server':\n"
            "    sys.exit(1)\n"
            f"TID = '{thread_id}'\n"
            "msg = recv(); send({'id': msg['id'], 'result': {'userAgent': 'codex_exec/test'}})\n"
            "recv()\n"
            "msg = recv(); send({'id': msg['id'], 'result': {'thread': {'id': TID, 'sessionId': TID, 'path': '~/.codex/r.jsonl'}}})\n"
            "msg = recv(); send({'id': msg['id'], 'result': {'turn': {'id': 'turn-1', 'status': 'inProgress'}}})\n"
            "send({'method': 'item/agentMessage/delta', 'params': {'delta': 'answer text'}})\n"
            "send({'method': 'turn/completed', 'params': {'turn': {'id': 'turn-1', 'status': 'completed'}}})\n"
            "# Emit result:null — a legal JSON-RPC 2.0 acknowledgement with no return value.\n"
            "send({'id': 99, 'result': None})\n"
            "sys.exit(0)\n"
        )
        fake.chmod(0o755)
        return str(fake_dir)

    def _make_steer_error_codex(self, tmp_path: Path, thread_id: str = "steer-err-tid") -> str:
        """Fake app-server that rejects a turn/steer with a JSON-RPC error."""
        fake_dir = tmp_path / "bin"
        fake_dir.mkdir(parents=True, exist_ok=True)
        fake = fake_dir / "codex"
        fake.write_text(
            "#!/usr/bin/env python3\n"
            "import sys, json, select as _sel\n"
            "def send(obj):\n"
            "    sys.stdout.write(json.dumps(obj) + '\\n')\n"
            "    sys.stdout.flush()\n"
            "def recv(timeout=None):\n"
            "    if timeout is not None:\n"
            "        r, _, _ = _sel.select([sys.stdin], [], [], timeout)\n"
            "        if not r: return None\n"
            "    line = sys.stdin.readline()\n"
            "    return json.loads(line.strip()) if line.strip() else None\n"
            "if len(sys.argv) < 2 or sys.argv[1] != 'app-server':\n"
            "    sys.exit(1)\n"
            f"TID = '{thread_id}'\n"
            "msg = recv(); send({'id': msg['id'], 'result': {'userAgent': 'codex_exec/test'}})\n"
            "recv()\n"
            "msg = recv(); send({'id': msg['id'], 'result': {'thread': {'id': TID, 'sessionId': TID, 'path': '~/.codex/r.jsonl'}}})\n"
            "msg = recv(); send({'id': msg['id'], 'result': {'turn': {'id': 'turn-1', 'status': 'inProgress'}}})\n"
            "send({'method': 'item/agentMessage/delta', 'params': {'delta': 'first answer'}})\n"
            "# Complete the turn WITHOUT an id to exercise the 'no id' clear path.\n"
            "send({'method': 'turn/completed', 'params': {'turn': {'status': 'completed'}}})\n"
            "# Wait for a steer; respond with a JSON-RPC error (stale expectedTurnId).\n"
            "msg = recv(timeout=5.0)\n"
            "if msg and msg.get('method') == 'turn/steer':\n"
            "    send({'id': msg['id'], 'error': {'code': -32001, 'message': 'expectedTurnId does not match'}})\n"
            "sys.exit(0)\n"
        )
        fake.chmod(0o755)
        return str(fake_dir)

    def _make_active_turn_steer_rejecting_codex(self, tmp_path: Path, thread_id: str = "active-err-tid") -> str:
        """Fake app-server that keeps a turn active and rejects turn/steer with a JSON-RPC error.

        Unlike _make_steer_error_codex, this server sends a turn.id in the turn/start
        result so active_turn_id is populated. A subsequent steer will therefore be sent
        as turn/steer (not turn/start), and the server rejects it with an rpc error.
        """
        fake_dir = tmp_path / "bin-active"
        fake_dir.mkdir(parents=True, exist_ok=True)
        fake = fake_dir / "codex"
        fake.write_text(
            "#!/usr/bin/env python3\n"
            "import sys, json, select as _sel\n"
            "def send(obj):\n"
            "    sys.stdout.write(json.dumps(obj) + '\\n')\n"
            "    sys.stdout.flush()\n"
            "def recv(timeout=None):\n"
            "    if timeout is not None:\n"
            "        r, _, _ = _sel.select([sys.stdin], [], [], timeout)\n"
            "        if not r: return None\n"
            "    line = sys.stdin.readline()\n"
            "    return json.loads(line.strip()) if line.strip() else None\n"
            "if len(sys.argv) < 2 or sys.argv[1] != 'app-server':\n"
            "    sys.exit(1)\n"
            f"TID = '{thread_id}'\n"
            "msg = recv(); send({'id': msg['id'], 'result': {'userAgent': 'codex_exec/test'}})\n"
            "recv()\n"
            "msg = recv(); send({'id': msg['id'], 'result': {'thread': {'id': TID, 'sessionId': TID, 'path': '~/.codex/r.jsonl'}}})\n"
            "# Respond with a turn id so active_turn_id is set in the runner.\n"
            "msg = recv(); send({'id': msg['id'], 'result': {'turn': {'id': 'active-turn-1', 'status': 'inProgress'}}})\n"
            "send({'method': 'item/agentMessage/delta', 'params': {'delta': 'working...'}})\n"
            "# The turn is still running. Wait for a steer (which arrives as turn/steer).\n"
            "msg = recv(timeout=5.0)\n"
            "if msg and msg.get('method') == 'turn/steer':\n"
            "    send({'id': msg['id'], 'error': {'code': -32001, 'message': 'steer rejected by server'}})\n"
            "send({'method': 'turn/completed', 'params': {'turn': {'id': 'active-turn-1', 'status': 'completed'}}})\n"
            "sys.exit(0)\n"
        )
        fake.chmod(0o755)
        return str(fake_dir)

    def _wait_running(self, state_dir: Path, timeout: float = 10.0) -> bool:
        """Wait for running status. Kills the runner on timeout."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                s = (state_dir / "status").read_text().strip()
                if s == "running":
                    return True
                if s in agent_run.TERMINAL_STATUSES:
                    return False
            except FileNotFoundError:
                pass
            time.sleep(0.05)
        _kill_run_pid(state_dir)
        return False

    def _wait_terminal(self, state_dir: Path, timeout: float = 15.0) -> str:
        """Wait for terminal status. Kills the runner on timeout."""
        deadline = time.monotonic() + timeout
        status = "starting"
        while time.monotonic() < deadline:
            try:
                status = (state_dir / "status").read_text().strip()
            except FileNotFoundError:
                pass
            if status in agent_run.TERMINAL_STATUSES:
                break
            time.sleep(0.05)
        if status not in agent_run.TERMINAL_STATUSES:
            _kill_run_pid(state_dir)
        return status

    def test_result_null_frame_does_not_crash_interactive_runner(
        self, isolated_runs_root, isolated_log_root, tmp_path, monkeypatch
    ):
        """A legal result:null JSON-RPC frame must not crash the interactive runner.

        Before the fix, msg['result'].get(...) on None raised AttributeError which
        crashed the runner and wrote a traceback into the run log.
        """
        bin_dir = self._make_null_result_codex(tmp_path)
        monkeypatch.setenv("PATH", bin_dir + ":" + os.environ.get("PATH", ""))
        name = "null-result-test"
        ns = argparse.Namespace(
            name=name, command=[], interactive=True, prompt_file=None,
            submit_mode=None, idle_timeout=None,
            harness="codex", prompt="test prompt", model=None, agent_mode=None,
            harness_args=[],
        )
        agent_run.cmd_launch(ns)
        state_dir = isolated_runs_root / name
        log_dir = isolated_log_root / name

        self._wait_terminal(state_dir, timeout=15.0)


        # Run must not crash (status must be a normal terminal, not a traceback-induced failure)
        status = agent_run._read(state_dir / "status", "").strip()
        assert status in agent_run.TERMINAL_STATUSES, f"Unexpected status: {status!r}"

        # Log must contain the agent text, not a Python traceback.
        log_path = log_dir / "log"
        log_content = log_path.read_text() if log_path.exists() else ""
        assert "Traceback" not in log_content, (
            f"Runner crashed with traceback in log: {log_content[:500]!r}"
        )
        assert "answer text" in log_content, (
            f"Agent answer must be in log: {log_content!r}"
        )

    def test_steer_error_response_does_not_report_success(
        self, isolated_runs_root, isolated_log_root, tmp_path, monkeypatch
    ):
        """A turn/steer rejected by the app-server must be logged in the acquire log.

        The fix adds error handling for turn/steer rejection so the acquire log
        records the failure. Uses a server that keeps the turn active (sends a turn id)
        so the steer dispatches as turn/steer and can be rejected at the protocol level.
        """
        bin_dir = self._make_active_turn_steer_rejecting_codex(tmp_path)
        monkeypatch.setenv("PATH", bin_dir + ":" + os.environ.get("PATH", ""))
        name = "steer-error-test"
        ns = argparse.Namespace(
            name=name, command=[], interactive=True, prompt_file=None,
            submit_mode=None, idle_timeout=None,
            harness="codex", prompt="initial", model=None, agent_mode=None,
            harness_args=[],
        )

        agent_run.cmd_launch(ns)
        state_dir = isolated_runs_root / name
        log_dir = isolated_log_root / name

        reached = self._wait_running(state_dir, timeout=10.0)
        assert reached, "Run must reach running"
        time.sleep(0.5)  # let the server send the delta and wait for a steer

        # The turn is still active (server sent active-turn-1 id). The steer
        # dispatches as turn/steer. The server rejects it with an rpc error.
        # No real codex rollout file exists for the fake server to witness
        # against; force an immediate witness rise so cmd_steer doesn't burn
        # its full verify/retry budget on an assertion that only cares about
        # the acquire log content.
        witness_calls = {"n": 0}

        def fake_witness(_state_dir, _log_dir):
            witness_calls["n"] += 1
            return 0 if witness_calls["n"] == 1 else 1

        monkeypatch.setattr(agent_run, "_submission_witness_count", fake_witness)
        steer_ns = argparse.Namespace(name=name, message=["DO SOMETHING"], raw=False, esc=False)
        agent_run.cmd_steer(steer_ns)

        self._wait_terminal(state_dir, timeout=12.0)

        acquire_log = log_dir / "session-acquire.log"
        log_text = acquire_log.read_text() if acquire_log.exists() else ""
        # The steer must have gone as turn/steer (active_turn_id was set).
        assert "turn/steer" in log_text, (

            f"Steer must dispatch as turn/steer when active_turn_id is set: {log_text!r}"
        )
        # The rpc error from the server must be recorded.
        assert "rpc error id=" in log_text, (
            f"Acquire log must record the rpc error from the server: {log_text!r}"
        )

    def test_turn_completed_without_id_clears_active_turn_id(
        self, isolated_runs_root, isolated_log_root, tmp_path, monkeypatch
    ):
        """turn/completed without a turn.id must clear active_turn_id (prevents stuck steer)."""
        # The steer_error_codex server sends turn/completed with no id, then waits for a
        # steer.  If active_turn_id was NOT cleared, the steer would be sent as turn/steer
        # with a stale expectedTurnId that the server just finished.  If it IS cleared,
        # the adapter sends turn/start (the idle path) and the server can respond normally.
        bin_dir = self._make_steer_error_codex(tmp_path, thread_id="clear-active-tid")
        monkeypatch.setenv("PATH", bin_dir + ":" + os.environ.get("PATH", ""))
        name = "clear-active-tid-test"
        ns = argparse.Namespace(
            name=name, command=[], interactive=True, prompt_file=None,
            submit_mode=None, idle_timeout=None,
            harness="codex", prompt="first", model=None, agent_mode=None,
            harness_args=[],
        )

        agent_run.cmd_launch(ns)
        state_dir = isolated_runs_root / name
        log_dir = isolated_log_root / name

        reached = self._wait_running(state_dir, timeout=10.0)
        assert reached, "Run must reach running"
        time.sleep(0.8)

        # Send steer: if active_turn_id was cleared correctly, this becomes turn/start.
        # No real codex rollout file exists for the fake server; force an
        # immediate witness rise so cmd_steer doesn't burn its verify/retry
        # budget on an assertion that only cares about the dispatch method.
        witness_calls = {"n": 0}

        def fake_witness(_state_dir, _log_dir):
            witness_calls["n"] += 1
            return 0 if witness_calls["n"] == 1 else 1

        monkeypatch.setattr(agent_run, "_submission_witness_count", fake_witness)
        steer_ns = argparse.Namespace(name=name, message=["after-turn"], raw=False, esc=False)
        agent_run.cmd_steer(steer_ns)
        self._wait_terminal(state_dir, timeout=12.0)


        acquire_log = log_dir / "session-acquire.log"
        log_text = acquire_log.read_text() if acquire_log.exists() else ""
        # With no id in turn/completed, active_turn_id should be None, so the adapter
        # sends turn/start (idle path), not turn/steer with a stale expectedTurnId.
        assert "turn/start (steer idle)" in log_text, (
            f"Acquire log must show idle-path turn/start (active_turn_id was cleared): {log_text!r}"
        )




# ---------------------------------------------------------------------------
# --cwd: working directory for the launched command
# ---------------------------------------------------------------------------

class TestCwdParsing:
    """_parse_launch_argv accepts --cwd in both raw and managed mode."""

    def test_cwd_space_form_raw(self):
        r = _parse(["--cwd", "/tmp", "myrun", "--", "pwd"])
        assert r.cwd == "/tmp"
        assert r.command == ["pwd"]

    def test_cwd_equals_form_raw(self):
        r = _parse(["--cwd=/tmp", "myrun", "--", "pwd"])
        assert r.cwd == "/tmp"

    def test_cwd_equals_form_managed(self):
        r = _parse(["--harness", "claude", "--prompt", "hi", "--cwd=/tmp", "myrun"])
        assert r.cwd == "/tmp"
        assert r.harness == "claude"

    def test_cwd_relative_value_passed_through_unresolved(self):
        # The parser is pure; resolution happens in cmd_launch.
        r = _parse(["--cwd", "sub/dir", "myrun", "--", "pwd"])
        assert r.cwd == "sub/dir"

    def test_cwd_missing_value(self):
        with pytest.raises(agent_run._LaunchArgvError) as exc:
            _parse(["--cwd"])
        assert "--cwd" in str(exc.value)

    @pytest.mark.parametrize(
        "argv",
        [
            ["--cwd", "", "myrun", "--", "pwd"],
            ["--cwd=", "myrun", "--", "pwd"],
        ],
        ids=["space-form", "equals-form"],
    )
    def test_cwd_empty_value_rejected(self, argv):
        # An empty value would otherwise leave the run in the invocation
        # directory and record that directory in <state_dir>/cwd.
        with pytest.raises(agent_run._LaunchArgvError) as exc:
            _parse(argv)
        assert "--cwd" in str(exc.value)

    def test_cwd_before_a_subcommand_name_makes_it_a_run_name(self):
        # A launch flag suppresses subcommand dispatch, so the token is the
        # name of a run rather than the subcommand it is spelled like.
        r = _parse(["--cwd", "/tmp", "status", "--", "pwd"])
        assert r.subcommand_tokens is None
        assert r.name == "status"
        assert r.command == ["pwd"]

    def test_cwd_in_help(self):
        assert "--cwd" in agent_run._build_parser().format_help()


class TestCwdResolution:
    """_apply_launch_cwd resolves the requested DIR and enters it."""

    @pytest.fixture(autouse=True)
    def _restore_cwd(self):
        # _apply_launch_cwd chdirs the calling process; the CLI exits right
        # after, but the test process must be put back or later tests inherit
        # the move.
        origin = os.getcwd()
        yield
        os.chdir(origin)

    def _enter(self, requested: Optional[str]) -> None:
        agent_run._apply_launch_cwd(argparse.Namespace(cwd=requested))

    def test_absent_cwd_leaves_the_process_where_it_was(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        here = os.getcwd()
        self._enter(None)
        assert os.getcwd() == here

    def test_relative_resolves_against_the_invocation_directory(self, tmp_path, monkeypatch):
        (tmp_path / "workdir").mkdir()
        monkeypatch.chdir(tmp_path)
        self._enter("workdir")
        assert os.getcwd() == str((tmp_path / "workdir").resolve())

    def test_tilde_is_expanded(self, tmp_path, monkeypatch):
        # argparse does not expand '~'; agent-run must.
        home = tmp_path / "home"
        (home / "sub").mkdir(parents=True)
        monkeypatch.setenv("HOME", str(home))
        self._enter("~/sub")
        assert os.getcwd() == str((home / "sub").resolve())

    def test_symlinked_component_is_resolved(self, tmp_path):
        real = tmp_path / "real"
        real.mkdir()
        (tmp_path / "link").symlink_to(real)
        self._enter(str(tmp_path / "link" / "."))
        assert os.getcwd() == str(real.resolve())

    def test_pwd_is_repointed_and_oldpwd_dropped(self, tmp_path, monkeypatch):
        # os.chdir moves only the kernel cwd; a stale PWD would make the
        # launched command disagree with its own working directory.
        target = tmp_path / "workdir"
        target.mkdir()
        monkeypatch.setenv("PWD", "/nowhere")
        monkeypatch.setenv("OLDPWD", "/elsewhere")
        self._enter(str(target))
        assert os.environ["PWD"] == str(target.resolve())
        assert "OLDPWD" not in os.environ


class TestCwdLaunch:
    """--cwd is validated and entered before any run state is published."""

    @pytest.fixture(autouse=True)
    def _restore_cwd(self):
        # cmd_launch chdirs the calling process; the CLI exits right after, but
        # the test process must be put back or later tests inherit the move.
        origin = os.getcwd()
        yield
        os.chdir(origin)

    def test_command_runs_in_cwd_and_the_directory_is_recorded(
        self, isolated_runs_root, isolated_log_root, tmp_path
    ):
        target = tmp_path / "workdir"
        target.mkdir()
        name = "cwd-basic"
        assert agent_run.main(["--cwd", str(target), name, "--", "/bin/pwd"]) == 0
        state_dir = isolated_runs_root / name
        _wait_terminal(state_dir)
        assert str(target.resolve()) in (isolated_log_root / name / "log").read_text()
        assert (state_dir / "cwd").read_text().strip() == str(target.resolve())
        run_json = json.loads((isolated_log_root / name / "run.json").read_text())
        assert run_json["cwd"] == str(target.resolve())

    def test_launch_head_is_read_from_the_cwd_repo(
        self, isolated_runs_root, isolated_log_root, tmp_path
    ):
        # The launch directory is not a repo, so a launch_head at all proves it
        # came from --cwd rather than from the invoking directory.
        repo = tmp_path / "repo"
        repo.mkdir()
        env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t.com",
               "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t.com"}
        for args in (["init", "-q"], ["add", "-A"], ["commit", "-q", "-m", "initial", "--allow-empty"]):
            subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, env=env)
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, check=True,
            capture_output=True, text=True, env=env,
        ).stdout.strip()
        name = "cwd-launch-head"
        assert agent_run.main(["--cwd", str(repo), name, "--", "/bin/pwd"]) == 0
        state_dir = isolated_runs_root / name
        _wait_terminal(state_dir)
        assert (state_dir / "launch_head").read_text().strip() == head
        assert (state_dir / "cwd").read_text().strip() == str(repo.resolve())

    def test_interactive_run_uses_cwd(self, isolated_runs_root, isolated_log_root, tmp_path):
        # The interactive path launches through a pty rather than a plain fork.
        target = tmp_path / "workdir"
        target.mkdir()
        name = "cwd-interactive"
        assert agent_run.main(["-i", "--cwd", str(target), name, "--", "/bin/pwd"]) == 0
        state_dir = isolated_runs_root / name
        _wait_terminal(state_dir)
        assert (state_dir / "interactive").read_text().strip() == "1"
        assert (state_dir / "cwd").read_text().strip() == str(target.resolve())
        assert str(target.resolve()) in (isolated_log_root / name / "log").read_text()

    def test_managed_run_uses_cwd(
        self, isolated_runs_root, isolated_log_root, tmp_path, monkeypatch
    ):
        # Managed mode builds the argv itself, so it reaches exec by another route.
        fake = tmp_path / "claude"
        fake.write_text("#!/bin/sh\npwd\nexit 0\n")
        fake.chmod(0o755)
        monkeypatch.setenv("PATH", str(tmp_path) + ":" + os.environ.get("PATH", ""))
        target = tmp_path / "workdir"
        target.mkdir()
        name = "cwd-managed"
        assert agent_run.main([
            "--harness", "claude", "--prompt", "hi", "--cwd", str(target), name,
        ]) == 0
        state_dir = isolated_runs_root / name
        _wait_terminal(state_dir)
        assert (state_dir / "cwd").read_text().strip() == str(target.resolve())
        assert str(target.resolve()) in (isolated_log_root / name / "log").read_text()

    def test_prompt_file_relative_to_invocation_dir(
        self, isolated_runs_root, isolated_log_root, tmp_path, monkeypatch
    ):
        # A relative --prompt-file names a file the caller typed, so it is
        # resolved against the invocation directory, not against --cwd.
        fake = tmp_path / "claude"
        fake.write_text("#!/bin/sh\npwd\nexit 0\n")
        fake.chmod(0o755)
        monkeypatch.setenv("PATH", str(tmp_path) + ":" + os.environ.get("PATH", ""))
        launch_dir = tmp_path / "launch"
        launch_dir.mkdir()
        (launch_dir / "brief.md").write_text("do the thing\n")
        target = tmp_path / "workdir"
        target.mkdir()
        monkeypatch.chdir(launch_dir)
        name = "cwd-prompt-file"
        assert agent_run.main([
            "--harness", "claude", "--prompt-file", "brief.md", "--cwd", str(target), name,
        ]) == 0
        state_dir = isolated_runs_root / name
        _wait_terminal(state_dir)
        assert (isolated_log_root / name / "prompt").read_text() == "do the thing\n"

    @pytest.mark.parametrize(
        "cwd_argv", [["--cwd", ""], ["--cwd="]], ids=["space-form", "equals-form"]
    )
    def test_empty_cwd_exits_and_creates_no_run_dir(
        self, cwd_argv, isolated_runs_root, isolated_log_root, tmp_path, monkeypatch
    ):
        # An unset shell variable expands to an empty argument; running in the
        # invocation directory instead would record the wrong <state_dir>/cwd.
        monkeypatch.chdir(tmp_path)
        name = "cwd-empty"
        with pytest.raises(SystemExit) as exc_info:
            agent_run.main([*cwd_argv, name, "--", "/bin/pwd"])
        assert exc_info.value.code != 0
        assert "--cwd" in str(exc_info.value.code)
        assert not (isolated_runs_root / name).exists()
        assert not (isolated_log_root / name).exists()

    def test_nonexistent_cwd_exits_and_creates_no_run_dir(
        self, isolated_runs_root, isolated_log_root, tmp_path
    ):
        name = "cwd-missing"
        with pytest.raises(SystemExit) as exc_info:
            agent_run.main(["--cwd", str(tmp_path / "nope"), name, "--", "/bin/pwd"])
        assert exc_info.value.code != 0
        assert not (isolated_runs_root / name).exists()
        assert not (isolated_log_root / name).exists()

    def test_cwd_that_is_a_file_exits_and_creates_no_run_dir(
        self, isolated_runs_root, isolated_log_root, tmp_path
    ):
        target = tmp_path / "afile"
        target.write_text("x\n")
        name = "cwd-is-file"
        with pytest.raises(SystemExit) as exc_info:
            agent_run.main(["--cwd", str(target), name, "--", "/bin/pwd"])
        assert exc_info.value.code != 0
        assert "not a directory" in str(exc_info.value.code).lower()
        assert not (isolated_runs_root / name).exists()

    def test_unenterable_cwd_exits_and_creates_no_run_dir(
        self, isolated_runs_root, isolated_log_root, tmp_path
    ):
        if os.getuid() == 0:
            pytest.skip("root bypasses directory permission checks")
        target = tmp_path / "locked"
        target.mkdir(mode=0o000)
        name = "cwd-unenterable"
        try:
            with pytest.raises(SystemExit) as exc_info:
                agent_run.main(["--cwd", str(target), name, "--", "/bin/pwd"])
            assert exc_info.value.code != 0
            assert not (isolated_runs_root / name).exists()
        finally:
            target.chmod(0o700)


# ---------------------------------------------------------------------------
# OpenCode real-resolver policy tests
#
# These run the real `opencode debug agent` resolver (no model calls, no API
# cost) to check that the managed deny beats an allow from each external config
# source. Skipped when opencode is not on PATH.
# ---------------------------------------------------------------------------

_ALLOW_ALL_POLICY = {"question": "allow", "plan_enter": "allow", "plan_exit": "allow"}


def _allow_agent_config(agent_name: str) -> str:
    """Serialised opencode.json granting an agent all three policy keys."""
    return json.dumps({"agent": {agent_name: {"permission": dict(_ALLOW_ALL_POLICY)}}})


def _oc_policy_env(agent_mode: Optional[str] = None) -> dict:
    """Return an env dict with OPENCODE_CONFIG_CONTENT set to the full deny policy."""
    policy = agent_run._opencode_policy_config(
        None,
        enable_planning=False,
        enable_questions=False,
        opencode_agent_mode=agent_mode,
    )
    return {**os.environ, "OPENCODE_CONFIG_CONTENT": policy}


def _oc_debug_agent(agent_name: str, cwd: str, env: dict, timeout: float = 10.0) -> dict:
    """Run `opencode debug agent <agent_name>` and return the parsed JSON result.

    Raises AssertionError if the command exits non-zero or the output is not
    valid JSON — callers must require successful exit before asserting policy.
    """
    result = subprocess.run(
        ["opencode", "debug", "agent", agent_name],
        capture_output=True, text=True, env=env, cwd=cwd, timeout=timeout,
    )
    stderr = result.stderr
    assert result.returncode == 0, (
        f"opencode debug agent {agent_name!r} exited {result.returncode}; "
        f"stderr={stderr!r}"
    )
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise AssertionError(
            f"opencode debug agent {agent_name!r} output is not valid JSON: {exc}; "
            f"stdout={result.stdout[:300]!r}"
        ) from exc


@pytest.mark.skipif(shutil.which("opencode") is None, reason="requires real opencode binary on PATH")
class TestOpenCodeRealResolverPolicy:
    """The managed deny must beat an allow from the project, user/XDG,
    OPENCODE_CONFIG, and named-agent config sources.

    TestOpenCodePolicyConfig only covers the JSON transformation; these tests
    run the resolver that decides at run time, so they also validate the config
    load order the transformation depends on.
    """

    def _assert_policy_denied(self, data: dict, agent_name: str) -> None:
        """Assert the three policy keys resolve to deny for ``agent_name``.

        question resolves to a tool (``tools.question`` must be False);
        plan_enter and plan_exit resolve to permission rules only, where the
        last matching rule decides.
        """
        tools = data.get("tools", {})
        assert tools.get("question") is False, (
            f"opencode resolved agent {agent_name!r} has tools.question={tools.get('question')!r} "
            f"(expected False); the managed deny did not win. Full tools: {tools}"
        )
        perms = data.get("permission", [])
        for perm_key in ("plan_enter", "plan_exit"):
            matching = [p for p in perms if p.get("permission") == perm_key]
            assert matching, (
                f"opencode resolved agent {agent_name!r} has no permission rules for {perm_key!r}; "
                f"deny was not applied. Full permission list: {perms}"
            )
            last_action = matching[-1].get("action")
            assert last_action == "deny", (
                f"opencode resolved agent {agent_name!r}: last {perm_key!r} rule is "
                f"{last_action!r} (expected 'deny'); the managed deny did not win. "
                f"Matching rules: {matching}"
            )

    @pytest.mark.parametrize("agent_name", ["build", "myagent"])
    def test_project_per_agent_allow_defeated(self, tmp_path, agent_name):
        """A per-agent allow in the project opencode.json is defeated, for both
        the default agent and a custom named one."""
        proj = tmp_path / f"proj-{agent_name}"
        proj.mkdir()
        (proj / "opencode.json").write_text(_allow_agent_config(agent_name))
        data = _oc_debug_agent(agent_name, str(proj), _oc_policy_env(agent_mode=agent_name))
        self._assert_policy_denied(data, agent_name)

    def test_opencode_config_env_per_agent_allow_defeated(self, tmp_path):
        """A per-agent allow in the file named by OPENCODE_CONFIG is defeated."""
        allow_cfg = tmp_path / "allow.json"
        allow_cfg.write_text(_allow_agent_config("build"))
        proj = tmp_path / "proj-envcfg"
        proj.mkdir()
        env = {
            **_oc_policy_env(agent_mode="build"),
            "OPENCODE_CONFIG": str(allow_cfg),
        }
        data = _oc_debug_agent("build", str(proj), env)
        self._assert_policy_denied(data, "build")

    def test_user_xdg_per_agent_allow_defeated(self, tmp_path):
        """A per-agent allow in the user/XDG config is defeated."""
        oc_config_dir = tmp_path / "config" / "opencode"
        oc_config_dir.mkdir(parents=True)
        (oc_config_dir / "opencode.json").write_text(_allow_agent_config("build"))
        proj = tmp_path / "proj-xdg"
        proj.mkdir()
        (tmp_path / "fakehome").mkdir()
        env = {
            **_oc_policy_env(agent_mode="build"),
            # Both are redirected so the real user config is neither read nor written.
            "XDG_CONFIG_HOME": str(tmp_path / "config"),
            "HOME": str(tmp_path / "fakehome"),
        }
        data = _oc_debug_agent("build", str(proj), env)
        self._assert_policy_denied(data, "build")

    def test_global_deny_beats_project_allow_no_agent_block(self, tmp_path):
        """A project allow in the global permission section, with no agent block
        anywhere, is defeated by the global scope of the managed policy."""
        proj = tmp_path / "proj-global"
        proj.mkdir()
        (proj / "opencode.json").write_text(json.dumps({"permission": dict(_ALLOW_ALL_POLICY)}))
        data = _oc_debug_agent("build", str(proj), _oc_policy_env(agent_mode="build"))
        self._assert_policy_denied(data, "build")

    def test_harness_arg_agent_allow_defeated(self, tmp_path):
        """An agent selected via --harness-arg --agent rather than --agent-mode
        still gets a deny block, sourced from the harness-args scan.
        """
        proj = tmp_path / "proj-harg"
        proj.mkdir()
        (proj / "opencode.json").write_text(_allow_agent_config("myagent"))
        policy = agent_run._opencode_policy_config(
            None,
            enable_planning=False,
            enable_questions=False,
            opencode_agent_mode=None,
            extra_agent_names=agent_run._opencode_agent_names_from_harness_args(
                ["--agent", "myagent"]
            ),
        )
        env = {**os.environ, "OPENCODE_CONFIG_CONTENT": policy}
        data = _oc_debug_agent("myagent", str(proj), env)
        self._assert_policy_denied(data, "myagent")


# ---------------------------------------------------------------------------
# Codex config key tripwire — fails loudly if codex renames the policy key
# ---------------------------------------------------------------------------

def _codex_strict_config(config_key: str) -> tuple[int, str, str]:
    """Run `codex app-server --strict-config -c <config_key>`.

    Returns (returncode, stdout, stderr). stdin is /dev/null, so app-server
    exits on EOF and only a config rejection produces a non-zero exit.
    """
    result = subprocess.run(
        ["codex", "app-server", "--strict-config", "-c", config_key],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=10.0,
    )
    return (
        result.returncode,
        result.stdout.decode("utf-8", errors="replace"),
        result.stderr.decode("utf-8", errors="replace"),
    )


@pytest.mark.skipif(shutil.which("codex") is None, reason="requires real codex binary on PATH")
class TestCodexPolicyKeyTripwire:
    """Verifies that codex still recognises tools.experimental_request_user_input.

    codex app-server silently ignores unknown -c keys by default, so a rename
    would silently disable the policy on every managed run. --strict-config
    converts that silent no-op into a launch failure, which this test catches.
    """

    def test_codex_strict_config_accepts_policy_key(self):
        returncode, stdout, stderr = _codex_strict_config(
            "tools.experimental_request_user_input={enabled=false}"
        )
        assert returncode == 0, (
            f"tools.experimental_request_user_input is no longer recognised by codex "
            f"(or codex exited non-zero for another reason). "
            f"If the key was renamed, update _codex_policy_args. "
            f"returncode={returncode} stderr={stderr!r} stdout={stdout!r}"
        )
        assert "unknown configuration field" not in stderr, (
            f"tools.experimental_request_user_input is no longer recognised by codex. "
            f"The managed policy key has been renamed — update _codex_policy_args. "
            f"stderr={stderr!r}"
        )

    def test_codex_strict_config_rejects_unknown_key(self):
        """Negative control: without it, the test above cannot tell an accepted
        key from strict validation not running at all.
        """
        returncode, stdout, stderr = _codex_strict_config(
            "tools.agent_run_nonexistent_sentinel_key_xyz={enabled=false}"
        )
        assert returncode != 0, (
            f"codex accepted a nonexistent key under --strict-config; "
            f"strict validation may be disabled or the flag name changed. "
            f"returncode={returncode} stderr={stderr!r} stdout={stdout!r}"
        )
        assert "unknown configuration field" in stderr or "unknown" in stderr.lower(), (
            f"codex exited non-zero but did not mention an unknown field; "
            f"the strict-config error message may have changed. "
            f"returncode={returncode} stderr={stderr!r}"
        )

