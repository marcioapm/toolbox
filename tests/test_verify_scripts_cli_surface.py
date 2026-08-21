"""Regression guard: the standalone `scripts/verify-*` diagnostics shell out to
`agent-run` as a subprocess and are not imported or exercised by the pytest
suite, so a CLI rename here silently breaks them. Assert their argv strings
track the current subcommand surface directly against the source text."""
from __future__ import annotations

from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"


def test_verify_hook_delivery_uses_logs_clean_not_removed_clean_subcommand():
    source = (SCRIPTS_DIR / "verify-hook-delivery").read_text()
    assert '"logs", run_name, "--clean"' in source
    assert '"clean", run_name' not in source


def test_verify_session_attribution_uses_logs_clean_not_removed_clean_subcommand():
    source = (SCRIPTS_DIR / "verify-session-attribution").read_text()
    assert '"logs", run_name, "--clean"' in source
    assert '"clean", run_name' not in source


def test_verify_session_attribution_routes_every_call_through_the_resolved_binary():
    """Every `agent-run` invocation must go through `_agent_run_bin`, resolved
    the same way `verify-hook-delivery` resolves it -- not the bare `agent-run`
    string, which hits whatever build happens to be on PATH."""
    source = (SCRIPTS_DIR / "verify-session-attribution").read_text()
    assert "_resolve_agent_run_bin" in source
    assert "_preflight_branch_binary" in source
    assert '"agent-run"' not in source
    assert "'agent-run'" not in source


def test_verify_session_attribution_fails_loudly_when_clean_is_unsupported():
    """A resolved binary that lacks `logs --clean` must abort with a clear
    message, not degrade `_count_in_assistant_region` failures into a silent
    empty transcript that gets reported as a missing interactive reply."""
    source = (SCRIPTS_DIR / "verify-session-attribution").read_text()
    assert "preflight_err" in source
    assert "ABORT" in source
