"""Live acceptance tests: A1–A5 for managed-mode planning/questions policy.

Gate: AGENT_RUN_LIVE_TESTS=1.  All tests skip unless that variable is set.

A1 — Codex questions differential through the real managed app-server path.
     Run 5 times; fail unless every repetition passes.

A2 — OpenCode deny beats project allow (bash).
     Subprocess tests prove the mechanism end-to-end.  Managed-launch is not
     included because the OpenCode prefork-mint + TUI startup reliably exceeds
     a practical test timeout (>180 s observed).  The codex managed-launch path
     is exercised in A1 and A5.

A3 — Claude deny beats a genuinely honoured allow.

A5 — Enabled arms observably enable per harness.

A4/A6/A7/A8 are hermetic and live in test_agent_run_harness.py:
  A4 (positional prompt survives variadic deny):
      TestBuildManagedArgv.test_claude_default_denies_planning_and_questions_after_prompt
  A6 (--enable-planning codex fails fast):
      TestParseLaunchArgvHarness.test_codex_enable_planning_fails_fast
  A7 (flags require --harness, appear in help, raw unaffected):
      TestParseLaunchArgvHarness, TestParserHelpShowsManagedMode.test_capability_flags_in_help
  A8 (policy reaches child through full managed launch path):
      TestManagedClaudeLaunch.test_claude_oneshot_delivers_prompt_via_stdin_with_denies_in_argv
      TestEndToEndThroughMain.test_main_codex_questions_policy_reaches_appserver
      TestEndToEndThroughMain.test_main_opencode_managed_launch_delivers_policy_env
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
import argparse
from pathlib import Path
from typing import Optional

import pytest

from toolbox import agent_run

# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------

LIVE = os.environ.get("AGENT_RUN_LIVE_TESTS") == "1"
live_only = pytest.mark.skipif(not LIVE, reason="set AGENT_RUN_LIVE_TESTS=1 to run live acceptance tests")

_CLAUDE_MODEL = os.environ.get("AGENT_RUN_LIVE_CLAUDE_MODEL", "")
_OPENCODE_MODEL = os.environ.get("AGENT_RUN_LIVE_OPENCODE_MODEL", "llmproxy-anthropic/claude-sonnet-4.6")

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _wait_terminal(state_dir: Path, timeout: float = 120.0) -> str:
    deadline = time.monotonic() + timeout
    status = "starting"
    while time.monotonic() < deadline:
        try:
            status = (state_dir / "status").read_text().strip()
        except FileNotFoundError:
            status = "starting"
        if status in agent_run.TERMINAL_STATUSES:
            return status
        time.sleep(0.1)
    try:
        pid = int((state_dir / "pid").read_text().strip())
        if pid > 0:
            import signal as _sig
            for sig in (_sig.SIGTERM, _sig.SIGKILL):
                try:
                    os.kill(pid, sig)
                except (ProcessLookupError, PermissionError):
                    break
                time.sleep(0.5)
    except (FileNotFoundError, ValueError, OSError):
        pass
    return status


def _read_log(log_dir: Path) -> str:
    try:
        return (log_dir / "log").read_text(errors="replace")
    except OSError:
        return ""


def _extract_tools_from_log(log: str) -> list[str]:
    """Parse the model's JSON tool-list response from a codex run log."""
    for m in re.finditer(r'\{[^}]*"tools"[^}]*\}', log, re.DOTALL):
        try:
            data = json.loads(m.group())
            if isinstance(data.get("tools"), list):
                return data["tools"]
        except json.JSONDecodeError:
            pass
    return []


# ---------------------------------------------------------------------------
# A1: Codex questions differential — real managed app-server path
# ---------------------------------------------------------------------------

_A1_PROMPT = (
    "List every function/tool you have available. "
    "Output ONLY a JSON object with a single key 'tools' whose value is an array of "
    "the exact internal function/tool names, like: "
    '{\"tools\": [\"functions.exec\", \"functions.request_user_input\", ...]}. '
    "No markdown. No prose. Only the JSON object."
)
_A1_SENTINEL = "functions.exec"          # positive control: must be present in both arms
_A1_DIFFERENTIAL = "functions.request_user_input"  # absent when disabled, present when enabled
_A1_REPEATS = 5


def _a1_run_arm(
    enabled: bool,
    run_name: str,
    state_root: Path,
    log_root: Path,
) -> dict:
    ns = argparse.Namespace(
        name=run_name,
        command=[],
        interactive=False,
        prompt_file=None,
        echo=False,
        echo_interval=2.0,
        submit_mode=None,
        idle_timeout=None,
        harness="codex",
        prompt=_A1_PROMPT,
        model=None,
        agent_mode=None,
        session_id=None,
        harness_args=[],
        permissions="bypass",
        enable_planning=False,
        enable_questions=enabled,
    )
    rc = agent_run.cmd_launch(ns)
    assert rc == 0, f"cmd_launch returned {rc}"

    state_dir = state_root / run_name
    log_dir = log_root / run_name
    status = _wait_terminal(state_dir, timeout=120.0)
    log = _read_log(log_dir)
    tools = _extract_tools_from_log(log)

    return {
        "status": status,
        "log_snippet": log[:400],
        "tools": tools,
        "has_sentinel": any(_A1_SENTINEL in t for t in tools),
        "has_differential": any(_A1_DIFFERENTIAL in t for t in tools),
    }


@live_only
class TestA1CodexQuestionsDifferential:
    """A1: functions.request_user_input present when enabled, absent when disabled.

    Both arms must contain the positive control tool (functions.exec); its
    absence means the run did not produce a valid tool list and the attempt
    does not count as a pass or vacuous non-failure.
    """

    def test_a1_codex_questions_differential(self, isolated_runs_root, isolated_log_root):
        enabled_passes = 0
        disabled_passes = 0
        failures: list[str] = []

        for i in range(_A1_REPEATS):
            e = _a1_run_arm(True,  f"a1-enabled-{i}",  isolated_runs_root, isolated_log_root)
            d = _a1_run_arm(False, f"a1-disabled-{i}", isolated_runs_root, isolated_log_root)

            if not e["has_sentinel"]:
                failures.append(
                    f"run {i} enabled: {_A1_SENTINEL!r} absent — no valid tool list "
                    f"(status={e['status']} log={e['log_snippet']!r})"
                )
                continue
            if not d["has_sentinel"]:
                failures.append(
                    f"run {i} disabled: {_A1_SENTINEL!r} absent — no valid tool list "
                    f"(status={d['status']} log={d['log_snippet']!r})"
                )
                continue

            if not e["has_differential"]:
                failures.append(
                    f"run {i} enabled: {_A1_DIFFERENTIAL!r} absent despite enable_questions=True; "
                    f"tools={e['tools']}"
                )
            else:
                enabled_passes += 1

            if d["has_differential"]:
                failures.append(
                    f"run {i} disabled: {_A1_DIFFERENTIAL!r} present despite enable_questions=False; "
                    f"tools={d['tools']}"
                )
            else:
                disabled_passes += 1

        summary = (
            f"A1 pass count: enabled {enabled_passes}/{_A1_REPEATS}, "
            f"disabled {disabled_passes}/{_A1_REPEATS}"
        )
        print(f"\n{summary}")
        assert not failures, f"{summary}\n" + "\n".join(failures)
        assert enabled_passes == _A1_REPEATS, summary
        assert disabled_passes == _A1_REPEATS, summary


# ---------------------------------------------------------------------------
# A2: OpenCode deny beats project allow
# ---------------------------------------------------------------------------

def _opencode_subprocess(prompt: str, project_dir: Path, extra_env: Optional[dict] = None,
                          timeout: int = 90) -> tuple[list[str], str, subprocess.CompletedProcess]:
    """Run opencode run --format json. Returns (tools_used, full_output_text, completed_process).

    Callers must check the completed process return code before asserting policy;
    a non-zero exit may indicate a configuration failure rather than a policy result.
    """
    env = dict(os.environ)
    if extra_env:
        env.update(extra_env)
    model_args = ["--model", _OPENCODE_MODEL] if _OPENCODE_MODEL else []
    result = subprocess.run(
        ["opencode", "run", *model_args, "--auto", "--format", "json", prompt],
        capture_output=True, text=True, env=env, cwd=str(project_dir), timeout=timeout,
    )
    tools_used: list[str] = []
    text_parts: list[str] = []
    for line in result.stdout.splitlines():
        try:
            ev = json.loads(line)
            if ev.get("type") == "tool_use":
                tools_used.append(ev.get("part", {}).get("tool", ""))
            if ev.get("type") == "text":
                text_parts.append(ev.get("part", {}).get("text", ""))
        except json.JSONDecodeError:
            pass
    return tools_used, " ".join(text_parts), result


@live_only
class TestA2OpenCodeDenyBeatsProjectAllow:
    """A2: OPENCODE_CONFIG_CONTENT deny overrides project opencode.json allow.

    Positive sentinel ("bash" in tools_used) proves the control arm actually
    invoked bash.  The deny arm must have "bash" absent from tools_used, and
    must produce non-empty text output proving OpenCode ran.
    """

    _PROMPT = "Run: echo SENTINEL_OUTPUT_XYZ"

    def test_a2_control_bash_executes(self, tmp_path):
        """Control: project opencode.json allows bash → bash tool invoked."""
        project = tmp_path / "project"
        project.mkdir()
        (project / "opencode.json").write_text(json.dumps({"permission": {"bash": "allow"}}))

        tools_used, text, proc = _opencode_subprocess(self._PROMPT, project)
        assert proc.returncode == 0, (
            f"Control arm: opencode exited {proc.returncode}; this is a process failure. "
            f"stderr={proc.stderr!r} tools={tools_used!r}"
        )
        assert "bash" in tools_used, (
            f"Control arm: bash not invoked despite project allow. "
            f"tools={tools_used!r} text={text[:200]!r}"
        )

    def test_a2_deny_blocks_bash(self, tmp_path):
        """Deny arm: process-local OPENCODE_CONFIG_CONTENT deny beats project allow."""
        project = tmp_path / "project"
        project.mkdir()
        (project / "opencode.json").write_text(json.dumps({"permission": {"bash": "allow"}}))

        deny_cfg = json.dumps({"permission": {"bash": "deny"}})
        tools_used, text, proc = _opencode_subprocess(
            self._PROMPT, project,
            extra_env={"OPENCODE_CONFIG_CONTENT": deny_cfg},
        )
        assert proc.returncode == 0, (
            f"Deny arm: opencode exited {proc.returncode}; this is a process failure, "
            f"not policy evidence. stderr={proc.stderr!r}"
        )
        # Positive control: OpenCode ran and produced non-empty text.
        assert text.strip(), (
            "Deny arm positive control failed: OpenCode produced no text output. "
            f"tools={tools_used!r} stderr={proc.stderr!r}"
        )
        # Actual assertion.
        assert "bash" not in tools_used, (
            f"Deny arm: bash was invoked despite deny. tools={tools_used!r}"
        )


# ---------------------------------------------------------------------------
# A3: Claude deny beats a genuinely honoured allow
# ---------------------------------------------------------------------------

def _claude_run(output_format: str, prompt_stdin: str, extra_argv: list[str], timeout: int) -> subprocess.CompletedProcess:
    """Run claude --print in the given output format and return the completed process.

    Callers must check the return code before treating stdout as policy evidence;
    a non-zero exit may indicate a launch failure rather than a policy result.
    """
    model_args = ["--model", _CLAUDE_MODEL] if _CLAUDE_MODEL else []
    verbose = ["--verbose"] if output_format == "stream-json" else []
    return subprocess.run(
        [
            "claude",
            *model_args,
            "--print",
            "--output-format", output_format,
            *verbose,
            "--permission-mode", "bypassPermissions",
            *extra_argv,
        ],
        input=prompt_stdin,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _claude_stream_tool_names(
    prompt_stdin: str,
    extra_argv: list[str] = (),
    timeout: int = 60,
) -> tuple[list[str], str]:
    """Return (names of every tool_use block in assistant messages, result text).

    result text is the final result event's `result` field, or "" if the run
    produced no result event — callers use it as the positive control that
    claude ran at all. An empty tool list with empty result text is a failed
    run, not evidence that a tool was denied.
    """
    proc = _claude_run("stream-json", prompt_stdin, list(extra_argv), timeout)
    assert proc.returncode == 0, (
        f"claude exited {proc.returncode}; this is a process failure, not policy evidence. "
        f"stderr={proc.stderr!r} stdout={proc.stdout[:300]!r}"
    )
    tool_names: list[str] = []
    result_text = ""
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("type") == "assistant":
            for block in ev.get("message", {}).get("content", []):
                if block.get("type") == "tool_use":
                    name = block.get("name", "")
                    if name:
                        tool_names.append(name)
        elif ev.get("type") == "result":
            result_text = ev.get("result", "") or ""
    return tool_names, result_text


def _claude_json(prompt_stdin: str, extra_argv: list[str] = (), timeout: int = 60) -> dict:
    """Run claude --print --output-format json and return the parsed result object."""
    proc = _claude_run("json", prompt_stdin, list(extra_argv), timeout)
    assert proc.returncode == 0, (
        f"claude exited {proc.returncode}; this is a process failure, not policy evidence. "
        f"stderr={proc.stderr!r} stdout={proc.stdout[:300]!r}"
    )
    return json.loads(proc.stdout)


@live_only
class TestA3ClaudeDenyBeatsHonouredAllow:
    """A3: --disallowedTools deny beats a genuinely honoured allow.

    Assertions are on tool_use records, never on side effects: denying one tool
    removes the tool, not the capability, so the model can reproduce the same
    effect through another tool.

    Bash stands in for the policy's own tools. EnterPlanMode and AskUserQuestion
    are not exposed by Claude Code 2.1.207 for this workspace, so deny-vs-allow
    cannot be demonstrated for them here; the third test tripwires that.
    """

    _PROMPT = "Use Bash to run: echo hello"

    def test_a3_control_bash_invoked(self):
        """Control: no --disallowedTools → 'Bash' appears in invoked tool names."""
        tool_names, result_text = _claude_stream_tool_names(self._PROMPT)
        assert result_text.strip(), (
            f"Control arm: Claude produced no output. tool_names={tool_names!r}"
        )
        assert "Bash" in tool_names, (
            f"Control arm: 'Bash' not in tool_names — Bash was not invoked. "
            f"tool_names={tool_names!r} result={result_text[:200]!r}"
        )

    def test_a3_deny_blocks_bash(self):
        """Deny arm: --disallowedTools Bash → 'Bash' absent from invoked tool names.

        Non-empty result proves Claude ran rather than crashed.
        """
        tool_names, result_text = _claude_stream_tool_names(
            self._PROMPT, extra_argv=["--disallowedTools", "Bash"]
        )
        assert result_text.strip(), (
            f"Deny arm positive control failed: Claude produced no output. "
            f"tool_names={tool_names!r}"
        )
        assert "Bash" not in tool_names, (
            f"Deny arm: 'Bash' found in tool_names despite --disallowedTools Bash. "
            f"tool_names={tool_names!r}"
        )

    def test_a3_enterplanmode_askuserquestion_skip_reason(self):
        """Confirms skip reason: EnterPlanMode/AskUserQuestion unavailable in this workspace.

        If either appears in the model's tool list this test fails, prompting
        implementation of the full A3 proof for those tools.

        Requires Bash to appear in the tool list before trusting the absence
        conclusion — a parse failure or empty response would otherwise silently
        pass as "tools absent".
        """
        data = _claude_json(
            'List all tools you have. Reply ONLY with JSON: {"tools": [...]}. No prose.'
        )
        try:
            m = re.search(r'\{.*\}', data.get("result", ""), re.DOTALL)
            tools = json.loads(m.group()).get("tools", []) if m else []
        except (json.JSONDecodeError, AttributeError):
            tools = []

        # Serialise to catch dict-wrapped names such as [{"name": "EnterPlanMode"}].
        flat = json.dumps(tools)

        # Positive control: Bash must appear so we know the tool list was parsed.
        # Without this, a refused/empty/unparseable response would read as "all clear".
        if "Bash" not in flat:
            pytest.fail(
                f"Tripwire could not parse a valid tool list (positive control "
                f"'Bash' absent) — absence of EnterPlanMode is not evidence. "
                f"raw={data.get('result', '')[:300]!r}"
            )

        if "EnterPlanMode" in flat or "AskUserQuestion" in flat:
            pytest.fail(
                "EnterPlanMode or AskUserQuestion is NOW available. "
                "Implement the full A3 proof: trust the workspace, confirm the control "
                "arm honours the allow (no 'Ignoring ... has not been trusted' warning), "
                "then confirm deny wins. Do not mutate ~/.claude.json. "
                f"tools={tools}"
            )
        pytest.skip(
            "EnterPlanMode and AskUserQuestion are not exposed by Claude Code 2.1.207 "
            "for this workspace configuration. Neither appears in the model's tool list "
            "and neither can be triggered with --allowedTools. "
            "Demonstrating deny-vs-honoured-allow for those specific tools requires "
            "either mutating ~/.claude.json (forbidden) or a workspace where they are "
            "genuinely available. Bash deny is proven in test_a3_deny_blocks_bash."
        )


# ---------------------------------------------------------------------------
# A5: Enabled arms observably enable
# ---------------------------------------------------------------------------

@live_only
class TestA5EnabledArmsObservablyEnable:
    """A5: the capability is present in the enabled arm, not merely 'not denied'.

    Codex: functions.request_user_input present in tool list when --enable-questions.
    OpenCode: policy merge with question=allow does not corrupt bash allow.
    Claude: Bash callable when no --disallowedTools (enabled arm).
    """

    def test_a5_codex_enabled_differential_tool_present(
        self, isolated_runs_root, isolated_log_root
    ):
        """functions.request_user_input present when enable_questions=True."""
        obs = _a1_run_arm(True, "a5-codex-enabled", isolated_runs_root, isolated_log_root)
        assert obs["has_sentinel"], (
            f"A5 codex enabled: positive sentinel {_A1_SENTINEL!r} absent — "
            f"no valid tool list. status={obs['status']} log={obs['log_snippet']!r}"
        )
        assert obs["has_differential"], (
            f"A5 codex enabled: {_A1_DIFFERENTIAL!r} not present despite enable_questions=True. "
            f"tools={obs['tools']}"
        )

    def test_a5_opencode_policy_merge_preserves_bash(self, tmp_path):
        """OpenCode: OPENCODE_CONFIG_CONTENT with question=allow does not block bash."""
        merged = agent_run._opencode_policy_config(
            json.dumps({"permission": {"bash": "allow"}, "extra": "kept"}),
            enable_planning=False,
            enable_questions=True,
        )
        cfg = json.loads(merged)
        assert cfg.get("extra") == "kept", f"merge dropped 'extra': {merged!r}"
        assert cfg["permission"]["bash"] == "allow", f"merge corrupted bash: {merged!r}"
        assert cfg["permission"]["question"] == "allow"
        assert cfg["permission"]["plan_enter"] == "deny"

        project = tmp_path / "project"
        project.mkdir()
        (project / "opencode.json").write_text(json.dumps({"permission": {"bash": "allow"}}))

        enable_q_cfg = agent_run._opencode_policy_config(
            None, enable_planning=False, enable_questions=True,
        )
        tools_used, _, proc = _opencode_subprocess(
            "Run: echo SENTINEL_OUTPUT_XYZ",
            project,
            extra_env={"OPENCODE_CONFIG_CONTENT": enable_q_cfg},
        )
        assert proc.returncode == 0, (
            f"A5 opencode: opencode exited {proc.returncode}; this is a process failure. "
            f"stderr={proc.stderr!r}"
        )
        assert "bash" in tools_used, (
            f"A5 opencode: bash not invoked when question=allow. tools={tools_used!r}"
        )

    def test_a5_claude_enabled_arm_bash_callable(self):
        """Claude enabled arm (no --disallowedTools): 'Bash' appears in invoked tool names."""
        argv = agent_run._build_managed_argv(
            "claude",
            interactive=False,
            prompt="placeholder",
            model=None,
            agent_mode=None,
            session_id="00000000-0000-4000-8000-000000000001",
            harness_args=[],
            enable_planning=True,
            enable_questions=True,
        )
        assert "--disallowedTools" not in argv, (
            f"A5 claude: --disallowedTools present in enabled-arm argv: {argv}"
        )

        tool_names, result_text = _claude_stream_tool_names(
            "Use Bash to run: echo hello"
        )
        assert result_text.strip(), (
            f"A5 claude enabled: Claude produced no output. tool_names={tool_names!r}"
        )
        assert "Bash" in tool_names, (
            f"A5 claude enabled: 'Bash' not in tool_names. tool_names={tool_names!r}"
        )
