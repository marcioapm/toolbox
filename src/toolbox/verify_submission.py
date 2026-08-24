"""verify_submission — live harness-compatibility check for agent-run's verified
prompt submission (launch + steer) across opencode, claude, codex, plus the
opencode HTTP message contract the launch/steer witness relies on.

Each check asserts:
  C1  prompt is submitted and the verification marker is stamped
  C2  steer is submitted and self-reports as verified
  C3  anti-vacuity negative control: marker is NOT stamped when delivery is broken
  C4  --raw steer is exempt from verification (or rejected for codex)
  H1  POST /session/{id}/message route is present in /doc
  H2  POST drives the agent and the reply reaches the transcript
  H3  a dropped client connection does not abort the turn
  H4  noReply inserts a message without generating a reply
  H5  a duplicated messageID is not idempotent (both texts are stored)

Unlike the unit test suite (fakes only), this drives real installed harness
binaries so a harness upgrade that silently breaks submission is caught here.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths and environment
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_BRANCH_PYTHON = _REPO_ROOT / ".venv" / "bin" / "python"
_AGENT_RUN_SCRIPT = _REPO_ROOT / "src" / "toolbox" / "agent_run.py"

# Polling cap per predicate invocation (seconds).
_PREDICATE_LIMIT = 15.0
# Wall-clock bound for `<harness> --version` (seconds).
_VERSION_PROBE_LIMIT = 15.0


# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------


@dataclass
class _HarnessRow:
    harness: str
    version: str
    c1: str = "FAIL"
    c2: str = "FAIL"
    c3: str = "FAIL"
    c4: str = "FAIL"


@dataclass
class _State:
    rows: list[_HarnessRow] = field(default_factory=list)
    h_results: list[str] = field(default_factory=list)
    checks: int = 0
    failures: int = 0
    skips: int = 0

    def count(self, verdict: str) -> None:
        if verdict == "PASS":
            self.checks += 1
        elif verdict.startswith("FAIL"):
            self.checks += 1
            self.failures += 1
        elif verdict.startswith("SKIP"):
            self.skips += 1


# ---------------------------------------------------------------------------
# Global teardown registry
# ---------------------------------------------------------------------------

_launched_runs: list[str] = []
_temp_repos: list[str] = []
_claude_trusted_dirs: list[str] = []
_work_dir: str = ""
_keep: bool = False
_json_mode: bool = False


def _log(msg: str) -> None:
    print(f"verify-submission: {msg}", file=sys.stderr)


def _agent_run_bin() -> list[str]:
    if _BRANCH_PYTHON.exists():
        return [str(_BRANCH_PYTHON), str(_AGENT_RUN_SCRIPT)]
    return [sys.executable, str(_AGENT_RUN_SCRIPT)]


def _agent_run(*args: str, capture: bool = True) -> subprocess.CompletedProcess:
    cmd = [*_agent_run_bin(), *args]
    return subprocess.run(cmd, capture_output=capture, text=True, check=False)


# ---------------------------------------------------------------------------
# Bounded execution
# ---------------------------------------------------------------------------


def _run_bounded(limit: float, fn, *args) -> int:
    """Call fn(*args) in a thread; return its exit code or 124 on timeout."""
    result_box: list[int] = [-1]

    def target():
        try:
            rc = fn(*args)
            result_box[0] = rc if isinstance(rc, int) else 0
        except Exception:  # noqa: BLE001
            result_box[0] = 1

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout=limit)
    if t.is_alive():
        return 124
    return result_box[0]


# ---------------------------------------------------------------------------
# Predicates (callable with 0-exit = satisfied)
# ---------------------------------------------------------------------------


def _predicate_file_nonempty(path: str) -> int:
    return 0 if (os.path.isfile(path) and os.path.getsize(path) > 0) else 1


def _predicate_submission_verdict_exists(state_dir: str) -> int:
    if os.path.isfile(os.path.join(state_dir, "prompt_submitted")):
        return 0
    if os.path.isfile(os.path.join(state_dir, "prompt_unverified")):
        return 0
    return 1


def _predicate_submission_verified(state_dir: str) -> int:
    submitted = os.path.isfile(os.path.join(state_dir, "prompt_submitted"))
    unverified = os.path.isfile(os.path.join(state_dir, "prompt_unverified"))
    return 0 if (submitted and not unverified) else 1


def _predicate_run_is_running(state_dir: str) -> int:
    status_file = os.path.join(state_dir, "status")
    try:
        return 0 if Path(status_file).read_text().strip() == "running" else 1
    except OSError:
        return 1


def _predicate_url_answers(url: str) -> int:
    rc = subprocess.run(
        ["curl", "-fsS", "--max-time", "2", "-o", "/dev/null", url],
        capture_output=True,
        check=False,
    ).returncode
    return rc


def _predicate_message_count_equals(url: str, expected: int) -> int:
    n = _message_count(url)
    return 0 if n == expected else 1


def _predicate_transcript_contains(run: str, needle: str) -> int:
    return 0 if transcript_contains(run, needle) else 1


# ---------------------------------------------------------------------------
# wait_for: poll until a predicate (callable returning 0) succeeds or timeout
# ---------------------------------------------------------------------------


def _wait_for(timeout: float, interval: float, desc: str, pred_fn, *args) -> bool:
    """Poll pred_fn(*args) until it returns 0 or timeout elapses.

    Each predicate invocation is itself bounded by _PREDICATE_LIMIT so a hung
    check (a blocked socket read, a stalled curl) cannot outlive the timeout.
    """
    start = time.monotonic()
    while True:

        def _call():
            return pred_fn(*args)

        rc = _run_bounded(_PREDICATE_LIMIT, _call)
        if rc == 0:
            elapsed = time.monotonic() - start
            _log(f"  ok ({elapsed:.2f}s): {desc}")
            return True
        if rc == 124:
            _log(f"  (predicate exceeded {_PREDICATE_LIMIT}s and was killed: {desc})")
        elapsed = time.monotonic() - start
        if elapsed > timeout:
            _log(f"  TIMEOUT after {timeout}s: {desc}")
            return False
        time.sleep(interval)


# ---------------------------------------------------------------------------
# transcript_contains: assistant-only search (prevents prompt-echo vacuity)
# ---------------------------------------------------------------------------


def transcript_contains(run: str, needle: str) -> bool:
    """Return True if any ASSISTANT-role transcript record contains needle.

    Restricted to assistant records: the sentinel also appears verbatim in the
    user's prompt, so searching the full transcript would pass the instant the
    prompt is echoed back, before the agent replies.
    """
    cp = _agent_run("transcript", run, "--json")
    for line in cp.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("type") == "assistant" and needle in (rec.get("text") or ""):
            return True
    return False


# ---------------------------------------------------------------------------
# HTTP helpers — curl used deliberately (see module docstring re H3 semantics)
# ---------------------------------------------------------------------------


def _json_message_payload(
    text: str, noreply: bool = False, message_id: str = ""
) -> dict:
    payload: dict = {"parts": [{"type": "text", "text": text}]}
    if noreply:
        payload["noReply"] = True
    if message_id:
        payload["messageID"] = message_id
    return payload


def _post_json(timeout: float, url: str, payload: dict) -> tuple[int, str]:
    """POST JSON payload to url; return (curl_rc, http_code).

    Uses curl to preserve H3's client-abort semantics. The %{http_code} write-out
    is always appended so callers consistently receive the status string (empty
    string when no status line arrived).
    """
    body = json.dumps(payload)
    cmd = [
        "curl",
        "-sS",
        "--max-time",
        str(timeout),
        "-o",
        "/dev/null",
        "-X",
        "POST",
        "-H",
        "Content-Type: application/json",
        "-d",
        body,
        "-w",
        "%{http_code}",
        url,
    ]
    cp = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return cp.returncode, cp.stdout.strip()


def _message_count(url: str) -> int:
    """Number of message records at a /session/{id}/message URL.

    Returns -1 when the response is not a JSON array (unreachable server,
    error body).
    """
    cp = subprocess.run(
        ["curl", "-sS", "--max-time", "5", url],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        data = json.loads(cp.stdout)
        return len(data) if isinstance(data, list) else -1
    except json.JSONDecodeError:
        return -1


# ---------------------------------------------------------------------------
# h3_client_aborted: classify a curl exit as a mid-turn disconnect
# ---------------------------------------------------------------------------


def h3_client_aborted(curl_rc: int, http_code: str) -> bool:
    """Return True when curl gave up before the server finished.

    Exit 28 is the abort signal whether or not a status line already arrived.
    opencode answers 200 headers as soon as it accepts a message and streams
    the turn afterwards, so curl reports a status and *then* times out; requiring
    an empty http_code would reject that abort and fail H3 on correct behaviour.

    A status that did arrive is still consulted: a request the server rejected
    never started a turn, so a timeout after a 4xx/5xx is not a mid-turn
    disconnect and must not be accepted as one.
    """
    if curl_rc != 28:
        return False
    if not http_code or http_code in ("000",):
        return True
    return bool(re.match(r"^2\d\d$", http_code))


# ---------------------------------------------------------------------------
# h4_record_inserted: verify noReply inserted exactly one content-matching record
# ---------------------------------------------------------------------------


def _message_texts(message: dict):
    """Yield the text of each text-part in a message record."""
    for part in message.get("parts") or []:
        if isinstance(part, dict) and isinstance(part.get("text"), str):
            yield part["text"]


def h4_record_inserted(messages: list, sentinel: str, before_count: int) -> bool:
    """Return True when exactly one new user-role record carrying sentinel was
    appended and nothing generated a reply afterwards.

    The content check is what stops this passing vacuously: count-plus-one with
    a user role at the end is also satisfied by a rejected POST alongside an
    unrelated user record.
    """
    if not isinstance(messages, list):
        return False
    if before_count < 0 or len(messages) != before_count + 1:
        return False

    carrier = None
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            return False
        if (message.get("info") or {}).get("role") != "user":
            continue
        if sentinel in "\n".join(_message_texts(message)):
            carrier = index

    if carrier is None:
        return False
    # noReply means nothing generated afterwards: the sentinel record must be last.
    return carrier == len(messages) - 1


def _fetch_messages(url: str):
    """Fetch the message list at url; return parsed JSON or [] on failure."""
    cp = subprocess.run(
        ["curl", "-sS", "--max-time", "5", url],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        return json.loads(cp.stdout)
    except json.JSONDecodeError:
        return []


def _h4_record_inserted_from_url(url: str, sentinel: str, before_count: int) -> bool:
    return h4_record_inserted(_fetch_messages(url), sentinel, before_count)


# ---------------------------------------------------------------------------
# duplicated_prompt_texts_present: count how many of two texts survived
# ---------------------------------------------------------------------------


def duplicated_prompt_texts_present(messages: list, text_a: str, text_b: str) -> int:
    """How many of the two posted texts are stored in user turns.

    opencode 1.18.21 merges a repeated messageID into one record with two parts;
    counting records understates the duplication. What matters for retry design
    is whether the second POST's content survived at all: 2 = duplicate kept,
    1 = dropped.
    """
    if not isinstance(messages, list):
        return 0
    wanted = (text_a, text_b)
    found: set[str] = set()
    for message in messages:
        if not isinstance(message, dict):
            continue
        if (message.get("info") or {}).get("role") != "user":
            continue
        blob = "\n".join(_message_texts(message))
        for needle in wanted:
            if needle in blob:
                found.add(needle)
    return len(found)


def _duplicated_prompt_texts_from_url(url: str, text_a: str, text_b: str) -> int:
    return duplicated_prompt_texts_present(_fetch_messages(url), text_a, text_b)


# ---------------------------------------------------------------------------
# steer_reported_verified: exact positive-form match
# ---------------------------------------------------------------------------


def steer_reported_verified(steer_output: str) -> bool:
    """Return True only if steer's output matches the positive verification form.

    Checked against the exact success string; "could not be verified as
    delivered" is rejected explicitly because it contains the substring
    "verified", which a simple grep would pass on the very output reporting
    failure.
    """
    if "could not be verified" in steer_output:
        return False
    return bool(
        re.search(
            r"^agent-run: steered '[^']+' \(\d+ bytes, verified\)$",
            steer_output,
            re.MULTILINE,
        )
    )


# ---------------------------------------------------------------------------
# reason_matches: fault-reason comparison tolerating "kind: detail" suffixes
# ---------------------------------------------------------------------------


def reason_matches(reason: str, *expected_kinds: str) -> bool:
    """Return True if reason is one of the expected kinds.

    A reason that carries detail after a colon ("transport_error: [Errno 61]…")
    matches on the kind prefix, keeping the check specific to the fault class
    without pinning the errno string.
    """
    for kind in expected_kinds:
        if reason == kind:
            return True
        if reason.startswith(kind + ": "):
            return True
    return False


# ---------------------------------------------------------------------------
# Repository and run management
# ---------------------------------------------------------------------------


def _init_test_repo(repo_dir: str) -> None:
    for cmd in [
        ["git", "-C", repo_dir, "init", "-q"],
        [
            "git",
            "-C",
            repo_dir,
            "config",
            "user.email",
            "verify-submission@example.invalid",
        ],
        ["git", "-C", repo_dir, "config", "user.name", "verify-submission"],
    ]:
        subprocess.run(cmd, capture_output=True, check=True)
    Path(repo_dir, "README.md").write_text("placeholder\n")
    subprocess.run(
        ["git", "-C", repo_dir, "add", "README.md"], capture_output=True, check=True
    )
    subprocess.run(
        ["git", "-C", repo_dir, "commit", "-q", "-m", "init"],
        capture_output=True,
        check=True,
    )


def _kill_run(run_name: str) -> None:
    _agent_run("kill", run_name)


def _kill_run_and_wait(run_name: str) -> None:
    _kill_run(run_name)
    time.sleep(0.3)


def _launch_interactive_run(
    repo_dir: str, harness: str, run_name: str, prompt: str, model_args: list[str]
) -> bool:
    cmd = [
        *_agent_run_bin(),
        "--cwd",
        repo_dir,
        "--harness",
        harness,
        "-i",
        "--prompt",
        prompt,
        *model_args,
        run_name,
    ]
    cp = subprocess.run(cmd, capture_output=True, check=False)
    return cp.returncode == 0


def _claude_trust_dir(directory: str) -> bool:
    """Pre-accept Claude's workspace-trust dialog for a temp repository.

    Seeds both the supplied path and its realpath. Returns False when
    ~/.claude.json does not yet exist (first-run case; C1 may hit the dialog).
    """
    resolved = os.path.realpath(directory)
    claude_json = Path.home() / ".claude.json"
    if not claude_json.exists():
        _log(
            "claude trust: ~/.claude.json does not exist yet — skipping pre-trust, C1 may hit the trust dialog"
        )
        return True
    try:
        with open(claude_json) as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        _log(f"~/.claude.json is not valid JSON: {exc}")
        return False
    entry = {"hasTrustDialogAccepted": True}
    projects = data.setdefault("projects", {})
    projects[directory] = entry
    projects[resolved] = entry
    with open(claude_json, "w") as f:
        json.dump(data, f)
    with open(claude_json) as f:
        written = json.load(f)
    if resolved not in written.get("projects", {}):
        _log(f"trust entry for {resolved} did not persist after write")
        return False
    _claude_trusted_dirs.extend([directory, resolved])
    return True


def _revoke_claude_trust(directory: str) -> None:
    claude_json = Path.home() / ".claude.json"
    try:
        with open(claude_json) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return
    data.get("projects", {}).pop(directory, None)
    try:
        with open(claude_json, "w") as f:
            json.dump(data, f)
    except OSError:
        pass


def _cleanup() -> None:
    if _keep:
        print(
            f"verify-submission: --keep set, leaving {_work_dir} and "
            f"{len(_launched_runs)} run(s) in place",
            file=sys.stderr,
        )
        return
    for run in _launched_runs:
        if run:
            _kill_run(run)
    time.sleep(0.3)
    for d in _claude_trusted_dirs:
        if d:
            _revoke_claude_trust(d)
    for repo in _temp_repos:
        if repo:
            shutil.rmtree(repo, ignore_errors=True)
    if _work_dir:
        shutil.rmtree(_work_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Per-harness cells: C1-C4
# ---------------------------------------------------------------------------


def _uniform_row(
    harness: str, version: str, verdict: str, state: _State
) -> _HarnessRow:
    """Return a row where all four cells share the same verdict and count it."""
    row = _HarnessRow(
        harness=harness, version=version, c1=verdict, c2=verdict, c3=verdict, c4=verdict
    )
    for v in (row.c1, row.c2, row.c3, row.c4):
        state.count(v)
    return row


def _run_harness_version(harness: str, version_file: str) -> int:
    """Write harness --version output to version_file; return exit code or 124."""
    with open(version_file, "w") as fh:
        cp = subprocess.run(
            [harness, "--version"],
            stdout=fh,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return cp.returncode


def run_harness_cells(
    harness: str,
    state_dir: str,
    log_dir: str,
    work_dir: str,
    model_args: list[str],
    state: _State,
) -> _HarnessRow:
    """Run C1-C4 for a single harness and return the result row."""
    run_name = f"vsub-{harness}-{os.getpid()}"

    if not shutil.which(harness):
        _log(f"harness '{harness}' not installed — SKIP")
        return _uniform_row(harness, "-", "SKIP(not installed)", state)

    version_file = os.path.join(work_dir, f"{harness}-version.txt")
    version_rc = _run_bounded(
        _VERSION_PROBE_LIMIT,
        _run_harness_version,
        harness,
        version_file,
    )
    version_line = ""
    try:
        version_line = Path(version_file).read_text().splitlines()[0].strip()
    except (OSError, IndexError):
        pass

    if version_rc == 124:
        _log(
            f"harness '{harness}' --version did not answer within {_VERSION_PROBE_LIMIT}s — FAIL"
        )
        return _uniform_row(harness, "unresponsive", "FAIL(version hung)", state)
    if version_rc != 0:
        _log(
            f"harness '{harness}' --version exited {version_rc} — FAIL: {version_line}"
        )
        return _uniform_row(harness, "broken", f"FAIL(version rc={version_rc})", state)

    _log(f"=== harness={harness} version={version_line} run={run_name} ===")

    repo_dir = tempfile.mkdtemp()
    _temp_repos.append(repo_dir)
    _init_test_repo(repo_dir)
    if harness == "claude" and not _claude_trust_dir(os.path.realpath(repo_dir)):
        _log("C1 FAIL: could not pre-trust the temp repo for claude")
        return _uniform_row(harness, version_line, "FAIL(claude trust failed)", state)

    sentinel1 = f"SENTINEL1_{harness}_{os.getpid()}"
    sentinel2 = f"SENTINEL2_{harness}_{os.getpid()}"
    c1 = "FAIL"
    c2 = "FAIL"
    c3 = "FAIL"
    c4 = "FAIL"

    # --- C1: prompt lands and is verified ------------------------------------
    _log(f"C1: launching interactive {harness} run with sentinel prompt")
    env = {**os.environ, "AGENT_RUN_STATE_DIR": state_dir, "AGENT_RUN_LOG_DIR": log_dir}
    launch_env = {**env}  # inherited by subprocess below

    def _do_launch():
        cmd = [
            *_agent_run_bin(),
            "--cwd",
            repo_dir,
            "--harness",
            harness,
            "-i",
            "--prompt",
            f"Reply with exactly this word and nothing else: {sentinel1}",
            *model_args,
            run_name,
        ]
        cp = subprocess.run(cmd, capture_output=True, env=launch_env, check=False)
        return cp.returncode

    launched_ok = _do_launch() == 0
    if launched_ok:
        _launched_runs.append(run_name)
        run_state = os.path.join(state_dir, run_name)
        if _wait_for(
            60,
            0.5,
            "prompt_submitted appears, prompt_unverified absent",
            _predicate_submission_verified,
            run_state,
        ):
            if _wait_for(
                30,
                0.5,
                "sentinel1 reaches the transcript",
                _predicate_transcript_contains,
                run_name,
                sentinel1,
            ):
                c1 = "PASS"
            else:
                _log("C1 FAIL: sentinel never appeared in transcript")
        else:
            unverified_file = os.path.join(run_state, "prompt_unverified")
            try:
                detail = Path(unverified_file).read_text().strip()
            except OSError:
                detail = "(prompt_unverified not written)"
            trust_note = ""
            if harness == "claude":
                log_path = os.path.join(log_dir, run_name, "log")
                try:
                    log_text = (
                        Path(log_path).read_bytes().decode("utf-8", errors="replace")
                    )
                    trust_note = f" trust_dialog_in_log={'yes' if 'trust' in log_text.lower() else 'no'}"
                except OSError:
                    trust_note = ""
            _log(f"C1 FAIL: prompt_unverified detail=[{detail}]{trust_note}")
    else:
        _log("C1 FAIL: launch itself failed")

    # --- C2: steer lands and is verified -------------------------------------
    if c1 == "PASS":
        _log("C2: steering with sentinel2")
        cp = _agent_run(
            "steer",
            run_name,
            f"Reply with exactly this word and nothing else: {sentinel2}",
        )
        steer_out = (cp.stdout + cp.stderr).strip()
        if cp.returncode == 0:
            if steer_reported_verified(steer_out) and _wait_for(
                30,
                0.5,
                "sentinel2 reaches the transcript",
                _predicate_transcript_contains,
                run_name,
                sentinel2,
            ):
                c2 = "PASS"
            else:
                _log(
                    f"C2 FAIL: steer succeeded but did not report verified, or the transcript record is missing: {steer_out}"
                )
        else:
            _log(f"C2 FAIL: steer exited non-zero: {steer_out}")
    else:
        c2 = "SKIP(C1 failed)"

    if c1 == "PASS":
        _kill_run_and_wait(run_name)
    state.count(c1)
    state.count(c2)

    # --- C3: anti-vacuity negative control -----------------------------------
    c3_run = f"{run_name}-neg"
    c3_state = os.path.join(state_dir, c3_run)
    c3_fault_landed = False
    c3_expected_reasons: tuple[str, ...] = ()
    c3_launched = False

    if harness == "codex":
        _log(
            "C3: negative control via an unreadable codex rollout store (launch marker)"
        )
        c3_sessions_dir = os.path.join(work_dir, "empty-codex-sessions")
        c3_expected_reasons = ("witness_unreadable",)
        c3_env = {**env, "AGENT_RUN_CODEX_SESSIONS_DIR": c3_sessions_dir}
        cmd = [
            *_agent_run_bin(),
            "--cwd",
            repo_dir,
            "--harness",
            harness,
            "-i",
            "--prompt",
            f"Reply with exactly this word and nothing else: {sentinel1}_neg",
            *model_args,
            c3_run,
        ]
        cp = subprocess.run(cmd, capture_output=True, env=c3_env, check=False)
        if cp.returncode == 0:
            _launched_runs.append(c3_run)
            c3_launched = True
            rollouts = (
                list(Path(c3_sessions_dir).rglob("rollout-*.jsonl"))
                if Path(c3_sessions_dir).is_dir()
                else []
            )
            if not rollouts:
                c3_fault_landed = True
            else:
                _log(
                    "C3: rollout override directory unexpectedly holds a rollout — fault did not land"
                )
            _wait_for(
                20,
                0.5,
                "prompt_submitted or prompt_unverified appears",
                _predicate_submission_verdict_exists,
                c3_state,
            )
        else:
            _log("C3 FAIL: launch itself failed")

    elif harness == "opencode":
        _log(
            "C3: negative control via an unreachable opencode_port (HTTP witness lands too fast for a timing control)"
        )
        c3_expected_reasons = ("transport_error", "witness_unreadable")
        cmd = [
            *_agent_run_bin(),
            "--cwd",
            repo_dir,
            "--harness",
            harness,
            "-i",
            "--prompt",
            f"Reply with exactly this word and nothing else: {sentinel1}_neg",
            *model_args,
            c3_run,
        ]
        cp = subprocess.run(cmd, capture_output=True, env=env, check=False)
        if cp.returncode == 0:
            _launched_runs.append(c3_run)
            c3_launched = True
            port_file = os.path.join(c3_state, "opencode_port")
            if _wait_for(
                10,
                0.1,
                "opencode_port file appears",
                _predicate_file_nonempty,
                port_file,
            ):
                Path(port_file).write_text("1")
                _wait_for(
                    60,
                    0.5,
                    "prompt_submitted or prompt_unverified appears",
                    _predicate_submission_verdict_exists,
                    c3_state,
                )
                c3_port = (
                    Path(port_file).read_text().strip()
                    if Path(port_file).exists()
                    else ""
                )
                nothing_at_1 = (
                    subprocess.run(
                        [
                            "curl",
                            "-fsS",
                            "--max-time",
                            "2",
                            "-o",
                            "/dev/null",
                            "http://127.0.0.1:1/doc",
                        ],
                        capture_output=True,
                        check=False,
                    ).returncode
                    != 0
                )
                if c3_port == "1" and nothing_at_1:
                    c3_fault_landed = True
                else:
                    _log(
                        f"C3: opencode_port is [{c3_port}] and/or that port answers — fault did not land"
                    )
            else:
                _log("C3: opencode_port never appeared — could not inject the fault")
        else:
            _log("C3 FAIL: launch itself failed")

    else:
        _log("C3: negative control with AGENT_RUN_SUBMIT_VERIFY_TIMEOUT=0.01")
        c3_expected_reasons = ("timeout", "witness_unreadable")
        c3_env = {
            **env,
            "AGENT_RUN_SUBMIT_VERIFY_TIMEOUT": "0.01",
            "AGENT_RUN_SUBMIT_ATTEMPTS": "1",
        }
        cmd = [
            *_agent_run_bin(),
            "--cwd",
            repo_dir,
            "--harness",
            harness,
            "-i",
            "--prompt",
            f"Reply with exactly this word and nothing else: {sentinel1}_neg",
            *model_args,
            c3_run,
        ]
        c3_start = time.monotonic()
        cp = subprocess.run(cmd, capture_output=True, env=c3_env, check=False)
        if cp.returncode == 0:
            _launched_runs.append(c3_run)
            c3_launched = True
            _wait_for(
                20,
                0.2,
                "prompt_submitted or prompt_unverified appears",
                _predicate_submission_verdict_exists,
                c3_state,
            )
            c3_elapsed = time.monotonic() - c3_start
            if c3_elapsed < 10:
                c3_fault_landed = True
            else:
                _log(
                    f"C3: verdict took {c3_elapsed:.2f}s, no faster than the default window — timeout override did not land"
                )
        else:
            _log("C3 FAIL: launch itself failed")

    c3_reason = ""
    c3_unverified = os.path.join(c3_state, "prompt_unverified")
    if os.path.isfile(c3_unverified):
        try:
            c3_reason = Path(c3_unverified).read_text().strip()
        except OSError:
            pass

    c3_session_json = os.path.join(log_dir, c3_run, "session.json")

    if not c3_launched:
        _log("C3 FAIL: the run never launched, so nothing was controlled for")
    elif not (os.path.isfile(c3_session_json) and os.path.getsize(c3_session_json) > 0):
        _log(
            "C3 FAIL: no session.json — the run failed before submission, not because of the injected fault"
        )
    elif not c3_fault_landed:
        _log(
            "C3 FAIL: the injected fault could not be confirmed, so an unverified prompt proves nothing"
        )
    elif os.path.isfile(os.path.join(c3_state, "prompt_submitted")):
        _log(
            "C3 FAIL (VACUOUS): prompt_submitted was stamped despite the negative control -- verification is not gating the marker"
        )
    elif not os.path.isfile(c3_unverified):
        _log("C3 FAIL: neither prompt_submitted nor prompt_unverified is present")
    elif not c3_reason:
        _log("C3 FAIL: prompt_unverified exists but carries no reason")
    elif not reason_matches(c3_reason, *c3_expected_reasons):
        _log(
            f"C3 FAIL: prompt_unverified reason is [{c3_reason}], expected one of [{', '.join(c3_expected_reasons)}] for this fault"
        )
    else:
        c3 = "PASS"

    _kill_run_and_wait(c3_run)
    state.count(c3)

    # --- C4: --raw steer is exempt from verification -------------------------
    c4_run = f"{run_name}-raw"
    _log("C4: --raw steer exemption")
    c4_launched = _do_launch_for_c4(
        repo_dir, harness, c4_run, sentinel1, model_args, env
    )
    if c4_launched:
        _launched_runs.append(c4_run)
        c4_state = os.path.join(state_dir, c4_run)
        _wait_for(30, 0.5, "run reaches running", _predicate_run_is_running, c4_state)
        raw_cp = _agent_run_with_env("steer", "--raw", c4_run, "\x1b[Z", env=env)
        raw_out = (raw_cp.stdout + raw_cp.stderr).strip()
        raw_rc = raw_cp.returncode
        if harness == "codex":
            if raw_rc != 0 and "raw" in raw_out.lower():
                c4 = "PASS"
            else:
                _log(
                    f"C4 FAIL: codex --raw steer must be rejected by design, got rc={raw_rc}: {raw_out}"
                )
        else:
            if (
                raw_rc == 0
                and "raw, unverified)" in raw_out
                and ", verified)" not in raw_out
            ):
                c4 = "PASS"
            else:
                _log(f"C4 FAIL: rc={raw_rc} out={raw_out}")
        _kill_run_and_wait(c4_run)
    else:
        _log("C4 FAIL: launch itself failed")
    state.count(c4)

    return _HarnessRow(
        harness=harness, version=version_line, c1=c1, c2=c2, c3=c3, c4=c4
    )


def _do_launch_for_c4(
    repo_dir: str,
    harness: str,
    run_name: str,
    sentinel1: str,
    model_args: list[str],
    env: dict,
) -> bool:
    cmd = [
        *_agent_run_bin(),
        "--cwd",
        repo_dir,
        "--harness",
        harness,
        "-i",
        "--prompt",
        f"Reply with exactly this word and nothing else: {sentinel1}_raw",
        *model_args,
        run_name,
    ]
    cp = subprocess.run(cmd, capture_output=True, env=env, check=False)
    return cp.returncode == 0


def _agent_run_with_env(*args: str, env: dict) -> subprocess.CompletedProcess:
    cmd = [*_agent_run_bin(), *args]
    return subprocess.run(cmd, capture_output=True, text=True, env=env, check=False)


# ---------------------------------------------------------------------------
# opencode HTTP contract: H1-H5
# ---------------------------------------------------------------------------


def run_opencode_http_contract(
    state_dir: str,
    log_dir: str,
    work_dir: str,
    model_args: list[str],
    state: _State,
) -> list[str]:
    """Run H1-H5 and return the five verdict strings."""
    if not shutil.which("opencode"):
        _log("opencode not installed — skipping HTTP contract checks (H1-H5)")
        results = ["SKIP"] * 5
        for r in results:
            state.count(r)
        return results

    run_name = f"vsub-http-{os.getpid()}"
    sentinel = f"SENTINELHTTP_{os.getpid()}"

    repo_dir = tempfile.mkdtemp()
    _temp_repos.append(repo_dir)
    _init_test_repo(repo_dir)

    _log(f"=== opencode HTTP contract: launching {run_name} ===")
    h1 = h2 = h3 = h4 = h5 = "FAIL"

    env = {**os.environ, "AGENT_RUN_STATE_DIR": state_dir, "AGENT_RUN_LOG_DIR": log_dir}

    def _fail_all(msg: str, kill_name: str = "") -> list[str]:
        _log(f"{msg}; all of H1-H5 FAIL")
        if kill_name:
            _kill_run(kill_name)
        results = ["FAIL"] * 5
        for r in results:
            state.count(r)
        return results

    cmd = [
        *_agent_run_bin(),
        "--cwd",
        repo_dir,
        "--harness",
        "opencode",
        "-i",
        "--prompt",
        f"Reply with exactly this word and nothing else: {sentinel}",
        *model_args,
        run_name,
    ]
    cp = subprocess.run(cmd, capture_output=True, env=env, check=False)
    if cp.returncode != 0:
        return _fail_all("HTTP contract: launch failed")
    _launched_runs.append(run_name)

    run_state = os.path.join(state_dir, run_name)
    port_file = os.path.join(run_state, "opencode_port")

    if not _wait_for(
        30, 0.5, "opencode_port file appears", _predicate_file_nonempty, port_file
    ):
        return _fail_all("HTTP contract: opencode_port never appeared", run_name)

    port = Path(port_file).read_text().strip()
    session_json_path = os.path.join(log_dir, run_name, "session.json")
    session_id = ""
    try:
        session_id = json.loads(Path(session_json_path).read_text())["session_id"]
    except (OSError, json.JSONDecodeError, KeyError):
        pass
    if not session_id:
        return _fail_all(
            "HTTP contract: could not resolve session_id from session.json", run_name
        )

    base = f"http://127.0.0.1:{port}"
    message_url = f"{base}/session/{session_id}/message"

    if not _wait_for(
        30,
        0.5,
        "opencode HTTP API answers GET /doc",
        _predicate_url_answers,
        f"{base}/doc",
    ):
        return _fail_all("HTTP contract: opencode HTTP API never came up", run_name)

    # Wait for the launch turn to fully settle (2 message records: user + assistant)
    # before H2-H5 run; racing it corrupts H4/H5's before/after counts.
    if not _wait_for(
        30,
        0.5,
        "launch turn settles (2 message records)",
        _predicate_message_count_equals,
        message_url,
        2,
    ):
        return _fail_all("HTTP contract: launch turn never settled", run_name)

    # H1: route still exists.
    _log("H1: GET /doc, checking for POST /session/{sessionID}/message")
    cp = subprocess.run(
        ["curl", "-sS", "--max-time", "5", f"{base}/doc"],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        doc = json.loads(cp.stdout)
        paths = doc.get("paths", {})
        key = next(
            (
                k
                for k in paths
                if k.startswith("/session/{") and k.endswith("}/message")
            ),
            None,
        )
        if key and "post" in paths[key]:
            h1 = "PASS"
        else:
            _log("H1 FAIL: /doc missing a POST /session/{sessionID}/message route")
    except (json.JSONDecodeError, TypeError):
        _log("H1 FAIL: /doc response is not valid JSON")

    # H2: POST drives the agent.
    _log("H2: POST a sentinel message, expect 200 and a transcript reply")
    h2_sentinel = f"H2_{sentinel}"
    h2_payload = _json_message_payload(
        f"Reply with exactly this word and nothing else: {h2_sentinel}"
    )
    _h2_rc, h2_code = _post_json(10, message_url, h2_payload)
    if h2_code == "200" and _wait_for(
        30,
        0.5,
        "H2 sentinel reaches the transcript",
        _predicate_transcript_contains,
        run_name,
        h2_sentinel,
    ):
        h2 = "PASS"
    else:
        _log(f"H2 FAIL: http_code={h2_code}")

    # H3: fire-and-forget is safe (dropped connection does not abort the turn).
    _log("H3: POST with a short client timeout, expect the turn to still complete")
    # Each attempt carries its own sentinel: reusing one across the ladder makes
    # the transcript check unable to say which request produced the record.
    # The 2/0.5/0.2 ladder forces the abort; do not lengthen or collapse it.
    h3_aborted = False
    h3_sentinel = ""
    h3_rc = 0
    h3_code = ""
    for h3_attempt, h3_limit in enumerate((2, 0.5, 0.2), start=1):
        h3_sentinel = f"H3_{h3_attempt}_{sentinel}"
        h3_payload = _json_message_payload(
            f"Count slowly from one to five, one number per line, "
            f"then say exactly this word and nothing else: {h3_sentinel}"
        )
        h3_rc, h3_code = _post_json(h3_limit, message_url, h3_payload)
        if h3_client_aborted(h3_rc, h3_code):
            h3_aborted = True
            _log(
                f"  client connection aborted at --max-time {h3_limit}s "
                f"(curl rc=28, http_code=[{h3_code}]), sentinel {h3_sentinel}"
            )
            break
        _log(
            f"  POST completed within {h3_limit}s (rc={h3_rc} http_code=[{h3_code}]); "
            f"retrying with a shorter client timeout"
        )

    if not h3_aborted:
        _log(
            f"H3 FAIL: the client connection was never aborted "
            f"(last curl rc={h3_rc} http_code=[{h3_code}]); "
            f"fire-and-forget was never exercised"
        )
    elif _wait_for(
        60,
        1,
        "the dropped request's own sentinel reaches the transcript",
        _predicate_transcript_contains,
        run_name,
        h3_sentinel,
    ):
        h3 = "PASS"
    else:
        _log(
            f"H3 FAIL: the dropped request's turn ({h3_sentinel}) did not complete after the client disconnected"
        )

    # H4: noReply suppresses generation.
    _log("H4: POST with noReply:true, expect the message inserted with no reply")
    h4_sentinel = f"H4_{sentinel}"
    before_count = _message_count(message_url)
    h4_payload = _json_message_payload(h4_sentinel, noreply=True)
    _h4_rc, h4_code = _post_json(5, message_url, h4_payload)
    time.sleep(3)
    if h4_code != "200":
        _log(
            f"H4 FAIL: POST returned {h4_code}; a rejected request cannot demonstrate insertion"
        )
    elif _h4_record_inserted_from_url(message_url, h4_sentinel, before_count):
        h4 = "PASS"
    else:
        _log(
            f"H4 FAIL: before={before_count} — expected exactly one new record, "
            f"user-role, carrying {h4_sentinel}, with no assistant reply after it"
        )

    # H5: messageID is not idempotent.
    _log("H5: POST the same messageID twice, expect both texts to be stored")
    # Both statuses must be 200; a rejected duplicate is not evidence of non-idempotence.
    # opencode validates messageID as a string with a "msg" prefix.
    mid = f"msgverifysubmissionh5{os.getpid()}"
    h5_text_a = f"h5-first-{sentinel}"
    h5_text_b = f"h5-second-{sentinel}"
    _h5_rc_a, h5_code_a = _post_json(
        5, message_url, _json_message_payload(h5_text_a, noreply=True, message_id=mid)
    )
    _h5_rc_b, h5_code_b = _post_json(
        5, message_url, _json_message_payload(h5_text_b, noreply=True, message_id=mid)
    )
    time.sleep(1)
    h5_distinct = _duplicated_prompt_texts_from_url(message_url, h5_text_a, h5_text_b)

    if h5_code_a != "200" or h5_code_b != "200":
        _log(
            f"H5 FAIL: expected both POSTs to return 200, got {h5_code_a} and {h5_code_b} "
            f"— a rejected duplicate is not evidence of non-idempotence"
        )
    elif h5_distinct >= 2:
        h5 = "PASS"
    else:
        _log(
            f"H5 FAIL: expected both texts of the duplicated messageID to be stored, "
            f"got {h5_distinct} of 2 — a dropped duplicate would make retries silently idempotent"
        )

    _kill_run(run_name)

    results = [h1, h2, h3, h4, h5]
    for r in results:
        state.count(r)
    return results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _print_table(rows: list[_HarnessRow]) -> None:
    out = sys.stderr if _json_mode else sys.stdout
    print(
        f"{'harness':<10} {'version':<14} {'C1':<6} {'C2':<6} {'C3':<6} {'C4':<6}",
        file=out,
    )
    for row in rows:
        print(
            f"{row.harness:<10} {row.version:<14} {row.c1:<6} {row.c2:<6} {row.c3:<6} {row.c4:<6}",
            file=out,
        )
    print(file=out)


def _human_print(msg: str) -> None:
    """Print a human-readable line; redirect to stderr when --json is active."""
    print(msg, file=sys.stderr if _json_mode else sys.stdout)


def _summary_line(state: _State) -> str:
    if state.failures > 0:
        return (
            f"RESULT: FAIL ({state.checks} checks, {state.failures} failures, "
            f"{state.skips} skips)"
        )
    return f"RESULT: PASS ({state.checks} checks, 0 failures, {state.skips} skips)"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    global _keep, _work_dir, _json_mode

    parser = argparse.ArgumentParser(
        prog="verify-submission",
        description=(
            "Live harness-compatibility check for agent-run verified prompt submission.\n"
            "Drives real installed harness binaries so a harness upgrade that silently\n"
            "breaks submission is caught before it reaches production."
        ),
    )
    parser.add_argument(
        "--harness",
        choices=["opencode", "claude", "codex"],
        metavar="opencode|claude|codex",
        help="Run checks for this harness only (default: all three)",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Preserve temp dirs and launched runs (default: clean up on exit)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON summary to stdout",
    )
    args = parser.parse_args(argv)

    _keep = args.keep
    _json_mode = args.json

    atexit.register(_cleanup)

    def _sig_handler(signum, frame):
        _cleanup()
        sys.exit(1)

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    _work_dir = tempfile.mkdtemp()
    state_dir = os.path.join(_work_dir, "state")
    log_dir = os.path.join(_work_dir, "log")
    os.makedirs(state_dir)
    os.makedirs(log_dir)
    os.environ["AGENT_RUN_STATE_DIR"] = state_dir
    os.environ["AGENT_RUN_LOG_DIR"] = log_dir

    model_args: list[str] = []
    model_env = os.environ.get("VERIFY_SUBMISSION_MODEL", "")
    if model_env:
        model_args = ["--model", model_env]

    harnesses = [args.harness] if args.harness else ["opencode", "claude", "codex"]

    _human_print("verify-submission")
    if args.harness:
        _human_print(f"harness: {args.harness}")
    else:
        _human_print("harnesses: opencode claude codex (installed only)")
    _human_print(f"workdir: {_work_dir}")
    _human_print("")

    state = _State()

    rows: list[_HarnessRow] = []
    for h in harnesses:
        row = run_harness_cells(h, state_dir, log_dir, _work_dir, model_args, state)
        rows.append(row)
        _human_print("")

    h_results: list[str] = []
    if not args.harness:
        h_results = run_opencode_http_contract(
            state_dir, log_dir, _work_dir, model_args, state
        )
        _human_print("")

    _print_table(rows)

    if not args.harness and h_results:
        _human_print(
            f"opencode HTTP contract: H1 {h_results[0]}  H2 {h_results[1]}  "
            f"H3 {h_results[2]}  H4 {h_results[3]}  H5 {h_results[4]}"
        )
        _human_print("")

    summary = _summary_line(state)
    _human_print(summary)

    if args.json:

        def _row_dict(r: _HarnessRow) -> dict:
            return {
                "harness": r.harness,
                "version": r.version,
                "C1": r.c1,
                "C2": r.c2,
                "C3": r.c3,
                "C4": r.c4,
            }

        out = {
            "checks": state.checks,
            "failures": state.failures,
            "skips": state.skips,
            "pass": state.failures == 0,
            "harness_rows": [_row_dict(r) for r in rows],
        }
        if h_results:
            out["http_contract"] = {
                f"H{i + 1}": h_results[i] for i in range(len(h_results))
            }
        print(json.dumps(out, indent=2))

    return 1 if state.failures > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
