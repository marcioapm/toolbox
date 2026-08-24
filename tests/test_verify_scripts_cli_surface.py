"""CLI-surface and anti-vacuity tests for the verify-* diagnostics.

The verify-session-attribution and verify-hook-delivery tests guard against
subcommand renames that would break those scripts without showing up in the
pytest suite (they are not imported).

The verify-submission tests are behavioural: each imported check function is
called directly with fixtures that prove it can fail, and separate
anti-vacuity tests confirm that a broken subject causes a FAIL result.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"


# ---------------------------------------------------------------------------
# verify-hook-delivery and verify-session-attribution — source-text guards
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# verify-submission — imported behavioural tests (H5)
# ---------------------------------------------------------------------------

from toolbox.verify_submission import (
    duplicated_prompt_texts_present,
    h3_client_aborted,
    h4_record_inserted,
    opencode_message_route_present,
    steer_reported_verified,
    transcript_contains,
)


def _user_message(message_id: str, *texts: str) -> dict:
    return {
        "info": {"id": message_id, "role": "user"},
        "parts": [{"type": "text", "text": text} for text in texts],
    }


def _count_distinct(
    messages: list, text_a: str = "TEXT-A", text_b: str = "TEXT-B"
) -> int:
    return duplicated_prompt_texts_present(messages, text_a, text_b)


def test_h5_counts_both_texts_when_they_land_in_separate_records():
    """Two POSTs under one messageID stored as two records: both texts
    survived, so the id did not deduplicate the content."""
    messages = [
        _user_message("msg_a", "TEXT-A"),
        _user_message("msg_b", "TEXT-B"),
    ]
    assert _count_distinct(messages) == 2


def test_h5_counts_both_texts_when_opencode_merges_them_into_one_record():
    """Measured opencode 1.18.21 behaviour: a repeated messageID appends a part
    to the existing record. Both texts are stored, so the duplication H5 tests
    for is real even though only one record exists."""
    messages = [_user_message("msg_merged", "TEXT-A", "TEXT-B")]
    assert _count_distinct(messages) == 2


def test_h5_fails_when_the_duplicate_post_is_dropped():
    """The regression H5 must catch: were a repeated messageID to become
    idempotent, retries would silently deduplicate and only one text remains."""
    messages = [_user_message("msg_only", "TEXT-A")]
    assert _count_distinct(messages) < 2


def test_h5_fails_when_only_one_of_the_two_texts_is_present():
    """A response missing one POST's text is not evidence of duplication,
    however many records carry the other one."""
    one_needle_twice = [
        _user_message("msg_a", "TEXT-A"),
        _user_message("msg_b", "TEXT-A"),
    ]
    assert _count_distinct(one_needle_twice) < 2
    assert _count_distinct([_user_message("msg_a", "TEXT-A")]) < 2
    assert _count_distinct([]) < 2
    assert duplicated_prompt_texts_present("not a list", "TEXT-A", "TEXT-B") < 2  # type: ignore[arg-type]


def test_h5_ignores_assistant_records():
    """An assistant reply echoing both texts is not a user message."""
    messages = [
        {
            "info": {"id": "msg_reply", "role": "assistant"},
            "parts": [
                {"type": "text", "text": "TEXT-A"},
                {"type": "text", "text": "TEXT-B"},
            ],
        },
    ]
    assert _count_distinct(messages) < 2


# Anti-vacuity: the check can fail when both texts are absent.
def test_h5_anti_vacuity_fails_when_no_texts_are_stored():
    """If neither text survives, H5 must report a count below 2 (failure path)."""
    messages = [_user_message("msg_empty", "something else entirely")]
    assert duplicated_prompt_texts_present(messages, "TEXT-A", "TEXT-B") < 2


# ---------------------------------------------------------------------------
# verify-submission — imported behavioural tests (C2 / steer_reported_verified)
# ---------------------------------------------------------------------------


def test_c2_rejects_the_failure_message_that_contains_the_word_verified():
    """C2's whole purpose is catching an unverified steer. "could not be
    verified as delivered" contains "verified", so a substring match passes
    on the very output reporting failure -- the check would be vacuous even
    if unverified steer regressed to exit 0."""
    assert not steer_reported_verified(
        "agent-run: steer to 'x' could not be verified as delivered "
        "(timeout, 2 attempt(s) via keystroke)"
    )


def test_c2_accepts_only_the_positive_success_form():
    assert steer_reported_verified("agent-run: steered 'run-1' (37 bytes, verified)")


@pytest.mark.parametrize(
    "steer_output",
    [
        pytest.param("agent-run: steered 'x' (5 bytes, raw, unverified)", id="raw"),
        pytest.param(
            "agent-run: steered 'x' (5 bytes, unwitnessed — raw run)", id="unwitnessed"
        ),
        pytest.param("", id="no_output"),
        pytest.param("verified", id="bare_word"),
    ],
)
def test_c2_rejects_every_non_success_steer_output(steer_output):
    assert not steer_reported_verified(steer_output)


# Anti-vacuity: the check fails when the steer output is empty (no reply at all).
def test_c2_anti_vacuity_fails_on_absent_output():
    """An absent steer output must not be treated as success."""
    assert not steer_reported_verified("")
    assert not steer_reported_verified("some unrelated output")


# ---------------------------------------------------------------------------
# verify-submission — imported behavioural tests (H4 / h4_record_inserted)
# ---------------------------------------------------------------------------


def _h4_message(role: str, *texts: str) -> dict:
    return {
        "info": {"id": f"msg_{role}", "role": role},
        "parts": [{"type": "text", "text": text} for text in texts],
    }


def _h4_inserted(
    messages: list, sentinel: str = "H4-SENTINEL", before: int = 1
) -> bool:
    return h4_record_inserted(messages, sentinel, before)


def test_h4_accepts_exactly_one_new_user_record_carrying_the_sentinel():
    """The behaviour H4 exists to confirm: noReply inserts the message and
    suppresses generation."""
    messages = [_h4_message("user", "earlier"), _h4_message("user", "H4-SENTINEL")]
    assert _h4_inserted(messages)


def test_h4_rejects_an_unrelated_user_record():
    """A user record that arrived from something other than the H4 POST is
    not evidence the POST inserted anything: without a content check H4
    passes on a rejected POST plus unrelated traffic."""
    messages = [_h4_message("user", "earlier"), _h4_message("user", "something else")]
    assert not _h4_inserted(messages)


def test_h4_rejects_an_assistant_reply_after_the_sentinel():
    """noReply must suppress generation; a reply following the record means
    it did not."""
    messages = [
        _h4_message("user", "earlier"),
        _h4_message("user", "H4-SENTINEL"),
        _h4_message("assistant", "replying anyway"),
    ]
    assert not _h4_inserted(messages)


def test_h4_rejects_a_sentinel_echoed_only_by_an_assistant():
    messages = [_h4_message("user", "earlier"), _h4_message("assistant", "H4-SENTINEL")]
    assert not _h4_inserted(messages)


@pytest.mark.parametrize(
    ("messages_json", "label"),
    [
        pytest.param(json.dumps([]), "empty response", id="empty"),
        pytest.param("not json at all", "unparseable response", id="not_json"),
        pytest.param(json.dumps({"error": "nope"}), "error object", id="error_object"),
    ],
)
def test_h4_rejects_a_response_that_is_not_a_message_list(messages_json, label):
    try:
        messages = json.loads(messages_json)
    except json.JSONDecodeError:
        messages = messages_json  # pass invalid input directly
    assert not _h4_inserted(messages), label  # type: ignore[arg-type]


def test_h4_rejects_a_count_that_did_not_rise_by_exactly_one():
    """Two new records means something other than the H4 POST also wrote."""
    messages = [
        _h4_message("user", "earlier"),
        _h4_message("user", "unrelated"),
        _h4_message("user", "H4-SENTINEL"),
    ]
    assert not _h4_inserted(messages, before=1)


# Anti-vacuity: an empty message list must fail.
def test_h4_anti_vacuity_fails_on_empty_list():
    """h4_record_inserted must return False when no records are present."""
    assert not h4_record_inserted([], "H4-SENTINEL", 0)


# Anti-vacuity: a list where count did not rise must fail.
def test_h4_anti_vacuity_fails_when_count_unchanged():
    messages = [_h4_message("user", "H4-SENTINEL")]
    # before=1 means we expected 2 records, got 1 — must fail.
    assert not h4_record_inserted(messages, "H4-SENTINEL", 1)


# ---------------------------------------------------------------------------
# verify-submission — imported behavioural tests (H3 / h3_client_aborted)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "http_code",
    [
        pytest.param("200", id="headers_arrived"),
        pytest.param("", id="no_status_line"),
        pytest.param("000", id="zero_status"),
    ],
)
def test_h3_accepts_a_client_timeout_however_far_the_response_got(http_code):
    """opencode answers 200 headers as soon as it accepts the message and
    streams the turn afterwards, so curl reports a status *and* times out
    waiting for the body. That is the client-side disconnect H3 exercises;
    requiring an empty %{http_code} fails H3 on correct behaviour."""
    assert h3_client_aborted(28, http_code)


@pytest.mark.parametrize(
    ("curl_rc", "http_code"),
    [
        pytest.param(0, "200", id="completed_normally"),
        pytest.param(7, "", id="connection_refused"),
        pytest.param(52, "", id="empty_reply"),
        pytest.param(56, "200", id="recv_failure"),
    ],
)
def test_h3_rejects_every_exit_that_is_not_a_client_timeout(curl_rc, http_code):
    """The check must stay able to fail: a POST that completed, or failed for
    an unrelated reason, never exercised a mid-turn disconnect."""
    assert not h3_client_aborted(curl_rc, http_code)


@pytest.mark.parametrize(
    "http_code",
    [
        pytest.param("400", id="bad_request"),
        pytest.param("404", id="route_gone"),
        pytest.param("500", id="server_error"),
    ],
)
def test_h3_rejects_a_timeout_after_a_rejected_request(http_code):
    """A request the server refused never started a turn, so a timeout after
    it is not the mid-turn disconnect H3 is asserting survives."""
    assert not h3_client_aborted(28, http_code)


# Anti-vacuity: a non-28 curl rc must never be accepted.
def test_h3_anti_vacuity_fails_when_curl_completed_normally():
    """curl rc=0 means the request completed; the client never disconnected."""
    assert not h3_client_aborted(0, "200")
    assert not h3_client_aborted(0, "")


# ---------------------------------------------------------------------------
# verify-submission — C1 anti-vacuity
# ---------------------------------------------------------------------------


def test_c1_anti_vacuity_submission_verified_requires_marker_file(tmp_path):
    """prompt_submitted must exist and prompt_unverified must be absent for
    a verified result; either condition violated must fail."""
    from toolbox.verify_submission import _predicate_submission_verified

    state_dir = str(tmp_path)
    # Nothing present: not verified.
    assert _predicate_submission_verified(state_dir) != 0

    # Only prompt_unverified: not verified.
    (tmp_path / "prompt_unverified").write_text("timeout")
    assert _predicate_submission_verified(state_dir) != 0

    # Both files: not verified (unverified takes precedence).
    (tmp_path / "prompt_submitted").write_text("")
    assert _predicate_submission_verified(state_dir) != 0

    # Only prompt_submitted: verified.
    (tmp_path / "prompt_unverified").unlink()
    assert _predicate_submission_verified(state_dir) == 0


# ---------------------------------------------------------------------------
# verify-submission — H1 anti-vacuity (route presence check)
# ---------------------------------------------------------------------------


def test_h1_anti_vacuity_route_check_requires_post_method():
    """The H1 route check must fail when the /doc JSON has no POST method."""
    doc = {"paths": {"/session/{sessionID}/message": {"get": {}}}}
    assert not opencode_message_route_present(doc), "GET-only route must not satisfy H1"


def test_h1_anti_vacuity_route_check_fails_on_empty_doc():
    """An empty /doc response has no session message route."""
    assert not opencode_message_route_present({})


# ---------------------------------------------------------------------------
# verify-submission — transcript_contains assistant-only guard
# ---------------------------------------------------------------------------


def _make_transcript_result(*records: dict) -> MagicMock:
    """Fake _agent_run return value with JSONL stdout."""
    mock = MagicMock()
    mock.stdout = "\n".join(json.dumps(r) for r in records)
    return mock


def test_transcript_contains_finds_needle_in_assistant_record():
    """A needle present in an assistant-role record returns True."""
    records = [{"type": "assistant", "text": "the answer is SENTINEL"}]
    with patch("toolbox.verify_submission._agent_run", return_value=_make_transcript_result(*records)):
        assert transcript_contains("run-1", "SENTINEL")


def test_transcript_contains_rejects_needle_in_user_record():
    """A needle that appears only in a user-role record must return False.

    Without the assistant-only guard, a prompt echo in a user record would
    pass the check before the agent replies, inverting H2's meaning.
    """
    records = [{"type": "user", "text": "please reply with SENTINEL"}]
    with patch("toolbox.verify_submission._agent_run", return_value=_make_transcript_result(*records)):
        assert not transcript_contains("run-1", "SENTINEL")


def test_transcript_contains_finds_needle_when_user_and_assistant_both_present():
    """Needle in an assistant record returns True even when a user record also
    carries the needle — the assistant-role filter must not discard the match."""
    records = [
        {"type": "user", "text": "please reply with SENTINEL"},
        {"type": "assistant", "text": "SENTINEL"},
    ]
    with patch("toolbox.verify_submission._agent_run", return_value=_make_transcript_result(*records)):
        assert transcript_contains("run-1", "SENTINEL")


# Anti-vacuity: needle only in a user record must not satisfy the check.
def test_h2_anti_vacuity_transcript_contains_requires_assistant_record():
    """If the needle appears only in a user record, transcript_contains must
    return False — a prompt echo must not count as a harness reply."""
    records = [{"type": "user", "text": "SENTINEL in user prompt"}]
    with patch("toolbox.verify_submission._agent_run", return_value=_make_transcript_result(*records)):
        assert not transcript_contains("run-1", "SENTINEL")
