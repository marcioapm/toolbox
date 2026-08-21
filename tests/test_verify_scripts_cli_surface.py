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
