"""Regression guard: the standalone `scripts/verify-*` diagnostics shell out to
`agent-run` as a subprocess and are not imported or exercised by the pytest
suite, so a CLI rename here silently breaks them. Assert their argv strings
track the current subcommand surface directly against the source text.

The verify-submission.sh checks are also guarded against becoming vacuous:
a check that cannot fail proves nothing about the harness it targets, so the
shapes each one must reject are asserted here against the extracted check.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

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


def test_verify_session_attribution_emits_json_when_an_abort_precedes_cells():
    """--json callers decode stdout unconditionally, so an abort that returns
    before the summary must still write a failure object rather than nothing."""
    source = (SCRIPTS_DIR / "verify-session-attribution").read_text()
    assert "def _emit_abort_json(" in source
    # Both pre-cell aborts route through it: the binary preflight and the
    # database check.
    assert source.count("_emit_abort_json(args.json,") == 2
    assert '"aborted": reason' in source


def _h5_counter_source() -> str:
    """The inline python3 program `distinct_user_messages_containing` pipes a
    /session/<id>/message response through. Extracted from the script rather
    than duplicated, so a change to the check is a change to what is tested."""
    source = (SCRIPTS_DIR / "verify-submission.sh").read_text()
    body = source[source.index("distinct_user_messages_containing() {"):]
    body = body[: body.index("\n}\n")]
    return body[body.index("python3 -c '") + len("python3 -c '") : body.rindex("'")]


def _count_distinct(payload: str, text_a: str = "TEXT-A", text_b: str = "TEXT-B") -> int:
    result = subprocess.run(
        [sys.executable, "-c", _h5_counter_source()],
        input=payload,
        capture_output=True,
        text=True,
        env={**os.environ, "TEXT_A": text_a, "TEXT_B": text_b},
        check=True,
    )
    return int(result.stdout.strip())


def _user_message(message_id: str, *texts: str) -> dict:
    return {
        "info": {"id": message_id, "role": "user"},
        "parts": [{"type": "text", "text": text} for text in texts],
    }


def test_h5_counts_two_records_carrying_the_duplicated_message_id():
    """H5 exists to prove opencode's messageID is not idempotent: two POSTs
    with the same id must produce two separate user-message records."""
    payload = json.dumps([
        _user_message("msg_a", "TEXT-A"),
        _user_message("msg_b", "TEXT-B"),
    ])
    assert _count_distinct(payload) >= 2


def test_h5_fails_when_opencode_merges_both_texts_into_one_record():
    """The append/merge behaviour, which is precisely what H5 must reject:
    counting per-needle matches would total 2 here and pass vacuously."""
    payload = json.dumps([_user_message("msg_merged", "TEXT-A", "TEXT-B")])
    assert _count_distinct(payload) < 2


def test_h5_fails_when_only_one_of_the_two_texts_is_present():
    """A response missing one POST's text is not evidence of duplication,
    however many records carry the other one."""
    one_needle_twice = json.dumps([
        _user_message("msg_a", "TEXT-A"),
        _user_message("msg_b", "TEXT-A"),
    ])
    assert _count_distinct(one_needle_twice) < 2
    assert _count_distinct(json.dumps([_user_message("msg_a", "TEXT-A")])) < 2
    assert _count_distinct(json.dumps([])) < 2
    assert _count_distinct("not json at all") < 2


def test_h5_ignores_assistant_records():
    """An assistant reply echoing both texts is not a user message."""
    payload = json.dumps([
        {
            "info": {"id": "msg_reply", "role": "assistant"},
            "parts": [{"type": "text", "text": "TEXT-A"}, {"type": "text", "text": "TEXT-B"}],
        },
    ])
    assert _count_distinct(payload) < 2


def _shell_function_source(name: str) -> str:
    """The named bash function, extracted from verify-submission.sh so a
    change to the check is a change to what is tested."""
    source = (SCRIPTS_DIR / "verify-submission.sh").read_text()
    body = source[source.index(f"{name}() {{"):]
    return body[: body.index("\n}\n") + 2]


def _steer_reported_verified(steer_output: str) -> bool:
    result = subprocess.run(
        ["bash", "-c", f'{_shell_function_source("steer_reported_verified")}\n'
                       f'steer_reported_verified "$1"', "_", steer_output],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def test_c2_rejects_the_failure_message_that_contains_the_word_verified():
    """C2's whole purpose is catching an unverified steer. "could not be
    verified as delivered" contains "verified", so a substring match passes
    on the very output reporting failure -- the check would be vacuous even
    if unverified steer regressed to exit 0."""
    assert not _steer_reported_verified(
        "agent-run: steer to 'x' could not be verified as delivered "
        "(timeout, 2 attempt(s) via keystroke)"
    )


def test_c2_accepts_only_the_positive_success_form():
    assert _steer_reported_verified("agent-run: steered 'run-1' (37 bytes, verified)")


@pytest.mark.parametrize(
    "steer_output",
    [
        pytest.param("agent-run: steered 'x' (5 bytes, raw, unverified)", id="raw"),
        pytest.param("agent-run: steered 'x' (5 bytes, unwitnessed — raw run)", id="unwitnessed"),
        pytest.param("", id="no_output"),
        pytest.param("verified", id="bare_word"),
    ],
)
def test_c2_rejects_every_non_success_steer_output(steer_output):
    assert not _steer_reported_verified(steer_output)


def _h4_verifier_source() -> str:
    """The inline python3 program `h4_record_inserted` pipes a
    /session/<id>/message response through."""
    body = _shell_function_source("h4_record_inserted")
    return body[body.index("python3 -c '") + len("python3 -c '") : body.rindex("'")]


def _h4_inserted(payload: str, sentinel: str = "H4-SENTINEL", before: int = 1) -> bool:
    result = subprocess.run(
        [sys.executable, "-c", _h4_verifier_source()],
        input=payload,
        capture_output=True,
        text=True,
        env={**os.environ, "H4_SENTINEL": sentinel, "H4_BEFORE_COUNT": str(before)},
    )
    return result.returncode == 0


def _h4_message(role: str, *texts: str) -> dict:
    return {
        "info": {"id": f"msg_{role}", "role": role},
        "parts": [{"type": "text", "text": text} for text in texts],
    }


def test_h4_accepts_exactly_one_new_user_record_carrying_the_sentinel():
    """The behaviour H4 exists to confirm: noReply inserts the message and
    suppresses generation."""
    payload = json.dumps([_h4_message("user", "earlier"), _h4_message("user", "H4-SENTINEL")])
    assert _h4_inserted(payload)


def test_h4_rejects_an_unrelated_user_record():
    """A user record that arrived from something other than the H4 POST is
    not evidence the POST inserted anything: without a content check H4
    passes on a rejected POST plus unrelated traffic."""
    payload = json.dumps([_h4_message("user", "earlier"), _h4_message("user", "something else")])
    assert not _h4_inserted(payload)


def test_h4_rejects_an_assistant_reply_after_the_sentinel():
    """noReply must suppress generation; a reply following the record means
    it did not."""
    payload = json.dumps([
        _h4_message("user", "earlier"),
        _h4_message("user", "H4-SENTINEL"),
        _h4_message("assistant", "replying anyway"),
    ])
    assert not _h4_inserted(payload)


def test_h4_rejects_a_sentinel_echoed_only_by_an_assistant():
    payload = json.dumps([_h4_message("user", "earlier"), _h4_message("assistant", "H4-SENTINEL")])
    assert not _h4_inserted(payload)


@pytest.mark.parametrize(
    ("payload", "label"),
    [
        pytest.param(json.dumps([]), "empty response", id="empty"),
        pytest.param("not json at all", "unparseable response", id="not_json"),
        pytest.param(json.dumps({"error": "nope"}), "error object", id="error_object"),
    ],
)
def test_h4_rejects_a_response_that_is_not_a_message_list(payload, label):
    assert not _h4_inserted(payload), label


def test_h4_rejects_a_count_that_did_not_rise_by_exactly_one():
    """Two new records means something other than the H4 POST also wrote."""
    payload = json.dumps([
        _h4_message("user", "earlier"),
        _h4_message("user", "unrelated"),
        _h4_message("user", "H4-SENTINEL"),
    ])
    assert not _h4_inserted(payload, before=1)
