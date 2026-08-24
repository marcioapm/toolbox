#!/usr/bin/env bash
# verify-submission.sh — live harness-compatibility check for agent-run's
# verified prompt submission (launch + steer) across opencode, claude, codex,
# plus the opencode HTTP message contract the launch/steer witness relies on.
#
# Unlike the unit test suite (fakes only), this drives real installed harness
# binaries so a harness upgrade that silently breaks submission is caught
# here, not discovered in production. See README "Verified submission" for
# when to run this and how to read a failure.
#
# Usage: scripts/verify-submission.sh [--harness opencode|claude|codex] [--keep]
set -euo pipefail

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd -P)"

ONLY_HARNESS=""
KEEP=0
while [ $# -gt 0 ]; do
  case "$1" in
    --harness)
      ONLY_HARNESS="$2"
      shift 2
      ;;
    --harness=*)
      ONLY_HARNESS="${1#*=}"
      shift
      ;;
    --keep)
      KEEP=1
      shift
      ;;
    -h|--help)
      echo "Usage: $0 [--harness opencode|claude|codex] [--keep]"
      exit 0
      ;;
    *)
      echo "verify-submission: unrecognized argument: $1" >&2
      exit 2
      ;;
  esac
done

case "$ONLY_HARNESS" in
  ""|opencode|claude|codex) ;;
  *)
    echo "verify-submission: --harness must be opencode, claude, or codex, got: $ONLY_HARNESS" >&2
    exit 2
    ;;
esac

AGENT_RUN_PYTHON="${REPO_ROOT}/.venv/bin/python"
AGENT_RUN_SCRIPT="${REPO_ROOT}/src/toolbox/agent_run.py"
if [ ! -x "$AGENT_RUN_PYTHON" ]; then
  AGENT_RUN_PYTHON="python3"
fi

agent_run() {
  "$AGENT_RUN_PYTHON" "$AGENT_RUN_SCRIPT" "$@"
}

WORK_DIR="$(mktemp -d)"
STATE_DIR="${WORK_DIR}/state"
LOG_DIR="${WORK_DIR}/log"
mkdir -p "$STATE_DIR" "$LOG_DIR"
export AGENT_RUN_STATE_DIR="$STATE_DIR"
export AGENT_RUN_LOG_DIR="$LOG_DIR"

RUN_MODEL="${VERIFY_SUBMISSION_MODEL:-}"

LAUNCHED_RUNS=()
TEMP_REPOS=()
CLAUDE_TRUSTED_DIRS=()

# Pre-accept Claude's workspace-trust dialog for a temporary repository.
# Seed both the supplied path and its realpath, and require the latter to persist.
# The read-modify-write is not synchronized with other Claude processes.
claude_trust_dir() {
  local dir="$1"
  local resolved status
  resolved="$(python3 -c "import os, sys; print(os.path.realpath(sys.argv[1]))" "$dir")"
  status=0
  DIR_RAW="$dir" DIR_RESOLVED="$resolved" python3 -c "
import json, os, sys

path = os.path.expanduser('~/.claude.json')
try:
    with open(path) as f:
        data = json.load(f)
except FileNotFoundError:
    # Claude must create its initial configuration itself.
    sys.exit(2)
except json.JSONDecodeError as exc:
    print(f'~/.claude.json is not valid JSON: {exc}', file=sys.stderr)
    sys.exit(1)

raw, resolved = os.environ['DIR_RAW'], os.environ['DIR_RESOLVED']
entry = {'hasTrustDialogAccepted': True}
projects = data.setdefault('projects', {})
projects[raw] = entry
projects[resolved] = entry
with open(path, 'w') as f:
    json.dump(data, f)

with open(path) as f:
    written = json.load(f)
if resolved not in written.get('projects', {}):
    print(f'trust entry for {resolved} did not persist after write', file=sys.stderr)
    sys.exit(1)
" || status=$?
  if [ "$status" -eq 2 ]; then
    log "claude trust: ~/.claude.json does not exist yet — skipping pre-trust, C1 may hit the trust dialog"
    return 0
  fi
  if [ "$status" -ne 0 ]; then
    echo "verify-submission: could not confirm claude trust entry for ${resolved} in ~/.claude.json" >&2
    exit 1
  fi
  CLAUDE_TRUSTED_DIRS+=("$dir" "$resolved")
}

# shellcheck disable=SC2329  # invoked indirectly via `trap cleanup EXIT`
cleanup() {
  local status=$?
  if [ "$KEEP" -eq 1 ]; then
    echo "verify-submission: --keep set, leaving ${WORK_DIR} and ${#LAUNCHED_RUNS[@]} run(s) in place" >&2
    return "$status"
  fi
  local run
  for run in "${LAUNCHED_RUNS[@]:-}"; do
    [ -z "$run" ] && continue
    agent_run kill "$run" >/dev/null 2>&1 || true
  done
  # Give TERM a moment before the temp dirs (and their FIFOs/sockets) vanish
  # under a still-tearing-down runner.
  sleep 0.3
  local dir
  for dir in "${CLAUDE_TRUSTED_DIRS[@]:-}"; do
    [ -z "$dir" ] && continue
    python3 -c "
import json
path = '${HOME}/.claude.json'
try:
    with open(path) as f:
        data = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    raise SystemExit
data.get('projects', {}).pop('${dir}', None)
with open(path, 'w') as f:
    json.dump(data, f)
" 2>/dev/null || true
  done
  local repo
  for repo in "${TEMP_REPOS[@]:-}"; do
    [ -z "$repo" ] && continue
    rm -rf -- "$repo"
  done
  rm -rf -- "$WORK_DIR"
  return "$status"
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

RESULTS_HARNESS=()   # harness name, one entry per row
RESULTS_VERSION=()
RESULTS_C1=()
RESULTS_C2=()
RESULTS_C3=()
RESULTS_C4=()
OVERALL_FAIL=0
OVERALL_CHECKS=0
OVERALL_SKIPS=0

# Longest any single wait_for predicate may run before it is killed and the
# round counts as unsatisfied. Comfortably above a healthy store read or
# --max-time-bounded curl, far below any wait_for's own timeout.
WAIT_FOR_PREDICATE_LIMIT_SECONDS=15
# Longest `<harness> --version` may take. A shim that sleeps forever must be
# reported as a broken harness, not left to block the whole run.
HARNESS_VERSION_PROBE_SECONDS=15

log() { echo "verify-submission: $*" >&2; }

# run_bounded <limit_s> <cmd...>: run cmd under a wall-clock bound.
# Exits with the command's own status, or 124 if the bound elapsed first.
# timeout(1) is not present on a stock macOS, so the bound is enforced with
# a background child and a polled wait. The kill path silences bash's own
# "Terminated" job notice; the command's stdout and stderr pass through
# untouched either way.
#
# Redirect this to a file rather than capturing it with $(...): killing the
# child does not reap a grandchild it spawned, and a surviving grandchild
# holding the write end of a command-substitution pipe blocks the read
# forever, defeating the bound. A file has no such reader.
run_bounded() {
  local limit="$1"
  shift
  "$@" &
  local child=$! waited=0 status=0
  while kill -0 "$child" 2>/dev/null; do
    if awk -v w="$waited" -v l="$limit" 'BEGIN{exit !(w>=l)}'; then
      { kill -TERM "$child" 2>/dev/null || true; } 2>/dev/null
      sleep 0.2
      { kill -KILL "$child" 2>/dev/null || true; } 2>/dev/null
      { wait "$child" || true; } 2>/dev/null
      return 124
    fi
    sleep 0.05
    waited="$(awk -v w="$waited" 'BEGIN{printf "%.2f", w+0.05}')"
  done
  wait "$child" || status=$?
  return "$status"
}

init_test_repo() {
  local repo_dir="$1"
  git -C "$repo_dir" init -q
  git -C "$repo_dir" config user.email "verify-submission@example.invalid"
  git -C "$repo_dir" config user.name "verify-submission"
  printf '%s\n' "placeholder" > "$repo_dir/README.md"
  git -C "$repo_dir" add README.md
  git -C "$repo_dir" commit -q -m "init"
}

kill_run_and_wait() {
  agent_run kill "$1" >/dev/null 2>&1 || true
  sleep 0.3
}

# wait_for <timeout_s> <poll_interval_s> <description> <cmd...>
# Polls <cmd...> (a predicate; exit 0 = satisfied) until it succeeds or the
# timeout elapses. Prints elapsed time on success so a merely-slower harness
# is visible rather than silently swallowed by a generous timeout.
#
# Each predicate invocation is itself bounded, not just the loop: a predicate
# that blocks (a hung store read, an unanswered socket) would otherwise run
# past the advertised timeout with nothing to interrupt it. A predicate that
# exceeds its slice counts as unsatisfied for that round.
wait_for() {
  local timeout="$1" interval="$2" desc="$3"
  shift 3
  local start elapsed rc
  start=$(date +%s.%N)
  while true; do
    rc=0
    run_bounded "$WAIT_FOR_PREDICATE_LIMIT_SECONDS" "$@" || rc=$?
    if [ "$rc" -eq 0 ]; then
      elapsed=$(awk -v s="$start" -v n="$(date +%s.%N)" 'BEGIN{printf "%.2f", n-s}')
      log "  ok (${elapsed}s): ${desc}"
      return 0
    fi
    if [ "$rc" -eq 124 ]; then
      log "  (predicate exceeded ${WAIT_FOR_PREDICATE_LIMIT_SECONDS}s and was killed: ${desc})"
    fi
    elapsed=$(awk -v s="$start" -v n="$(date +%s.%N)" 'BEGIN{print (n-s)}')
    if awk -v e="$elapsed" -v t="$timeout" 'BEGIN{exit !(e>t)}'; then
      log "  TIMEOUT after ${timeout}s: ${desc}"
      return 1
    fi
    sleep "$interval"
  done
}

# transcript_contains <run_name> <needle>: an ASSISTANT-role transcript
# record's text contains needle. Restricted to assistant records
# deliberately -- the sentinel also appears verbatim in the user's own
# prompt text, so a plain grep over the whole transcript would pass the
# instant the prompt is echoed back, before the agent ever replies,
# making every check that calls this vacuous.
# shellcheck disable=SC2329  # invoked indirectly via `wait_for ... transcript_contains ...`
transcript_contains() {
  local run="$1" needle="$2"
  agent_run transcript "$run" --json 2>/dev/null | NEEDLE="$needle" python3 -c '
import json, os, sys
needle = os.environ["NEEDLE"]
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        rec = json.loads(line)
    except json.JSONDecodeError:
        continue
    if rec.get("type") == "assistant" and needle in (rec.get("text") or ""):
        sys.exit(0)
sys.exit(1)
' 2>/dev/null
}

# Write a message request body, passing shell values through the environment
# so Python handles all JSON escaping.
json_message_payload() {
  local outfile="$1" text="$2" noreply="${3:-}" message_id="${4:-}"
  MSG_TEXT="$text" MSG_NOREPLY="$noreply" MSG_ID="$message_id" \
    python3 <<'PYEOF' > "$outfile"
import json
import os

payload = {"parts": [{"type": "text", "text": os.environ["MSG_TEXT"]}]}
if os.environ.get("MSG_NOREPLY"):
    payload["noReply"] = True
if os.environ.get("MSG_ID"):
    payload["messageID"] = os.environ["MSG_ID"]
print(json.dumps(payload))
PYEOF
}

post_json() {
  local timeout="$1" url="$2" payload="$3"
  shift 3
  curl -sS --max-time "$timeout" -o /dev/null \
    -X POST -H 'Content-Type: application/json' -d @"$payload" \
    "$@" "$url" 2>/dev/null
}

# message_count <url>: number of message records at a /session/{id}/message URL,
# or -1 when the response is not a JSON array (unreachable server, error body).
message_count() {
  curl -sS --max-time 5 "$1" 2>/dev/null | python3 -c '
import json, sys
try:
    data = json.load(sys.stdin)
except json.JSONDecodeError:
    print(-1)
else:
    print(len(data) if isinstance(data, list) else -1)
' 2>/dev/null || echo -1
}

# distinct_user_messages_containing <url> <text_a> <text_b>: how many of the
# two texts appear in a user-role message, counting each message id at most
# once. 2 means the duplicated messageID produced two separate records rather
# than one deduplicated or overwritten one.
distinct_user_messages_containing() {
  curl -sS --max-time 5 "$1" 2>/dev/null | TEXT_A="$2" TEXT_B="$3" python3 -c '
import json, os, sys

def texts(message):
    for part in message.get("parts") or []:
        if isinstance(part, dict) and isinstance(part.get("text"), str):
            yield part["text"]

try:
    data = json.load(sys.stdin)
except json.JSONDecodeError:
    print(0)
    raise SystemExit

wanted = (os.environ["TEXT_A"], os.environ["TEXT_B"])
found = {}
for message in data if isinstance(data, list) else []:
    if not isinstance(message, dict):
        continue
    if (message.get("info") or {}).get("role") != "user":
        continue
    message_id = (message.get("info") or {}).get("id")
    blob = "\n".join(texts(message))
    for needle in wanted:
        if needle in blob:
            found.setdefault(needle, set()).add(message_id)
print(sum(len(ids) for ids in found.values()))
' 2>/dev/null || echo 0
}

record_row() {
  RESULTS_HARNESS+=("$1")
  RESULTS_VERSION+=("$2")
  RESULTS_C1+=("$3")
  RESULTS_C2+=("$4")
  RESULTS_C3+=("$5")
  RESULTS_C4+=("$6")
}

count_check() {
  # $1 = PASS/FAIL/SKIP(...)
  case "$1" in
    PASS) OVERALL_CHECKS=$((OVERALL_CHECKS + 1)) ;;
    FAIL) OVERALL_CHECKS=$((OVERALL_CHECKS + 1)); OVERALL_FAIL=$((OVERALL_FAIL + 1)) ;;
    SKIP*) OVERALL_SKIPS=$((OVERALL_SKIPS + 1)) ;;
  esac
}

set_all_http_results() {
  local result="$1" _
  H_RESULTS=("$result" "$result" "$result" "$result" "$result")
  for _ in 1 2 3 4 5; do count_check "$result"; done
}

# ---------------------------------------------------------------------------
# Per-harness cells: C1 (launch verified), C2 (steer verified),
# C3 (anti-vacuity negative control), C4 (--raw exemption)
# ---------------------------------------------------------------------------

run_harness_cells() {
  local harness="$1"
  local run_name="vsub-${harness}-$$"
  local sentinel1="SENTINEL1_${harness}_$$"
  local sentinel2="SENTINEL2_${harness}_$$"
  local version
  local c1="FAIL" c2="FAIL" c3="FAIL" c4="FAIL"

  if ! command -v "$harness" >/dev/null 2>&1; then
    log "harness '$harness' not installed — SKIP"
    record_row "$harness" "-" "SKIP(not installed)" "SKIP(not installed)" \
      "SKIP(not installed)" "SKIP(not installed)"
    count_check "SKIP"
    count_check "SKIP"
    count_check "SKIP"
    count_check "SKIP"
    return 0
  fi

  # An installed-but-broken harness must be a reported FAIL for its own row,
  # never a hang or a `set -e` abort of the whole run: `--version` is
  # bounded, and both a timeout and a non-zero exit fall through to the
  # FAIL row below rather than propagating.
  local version_rc=0
  local version_file="${WORK_DIR}/${harness}-version.txt"
  run_bounded "$HARNESS_VERSION_PROBE_SECONDS" "$harness" --version \
    > "$version_file" 2>&1 || version_rc=$?
  version="$(head -1 "$version_file" 2>/dev/null | tr -d '\r')"
  if [ "$version_rc" -eq 124 ]; then
    log "harness '$harness' --version did not answer within ${HARNESS_VERSION_PROBE_SECONDS}s — FAIL"
    record_row "$harness" "unresponsive" "FAIL(version hung)" "FAIL(version hung)" \
      "FAIL(version hung)" "FAIL(version hung)"
    count_check "FAIL"; count_check "FAIL"; count_check "FAIL"; count_check "FAIL"
    return 0
  fi
  if [ "$version_rc" -ne 0 ]; then
    log "harness '$harness' --version exited ${version_rc} — FAIL: ${version}"
    record_row "$harness" "broken" "FAIL(version rc=${version_rc})" "FAIL(version rc=${version_rc})" \
      "FAIL(version rc=${version_rc})" "FAIL(version rc=${version_rc})"
    count_check "FAIL"; count_check "FAIL"; count_check "FAIL"; count_check "FAIL"
    return 0
  fi
  log "=== harness=${harness} version=${version} run=${run_name} ==="

  local repo_dir
  repo_dir="$(mktemp -d)"
  TEMP_REPOS+=("$repo_dir")
  init_test_repo "$repo_dir"
  if [ "$harness" = "claude" ]; then
    claude_trust_dir "$(cd -- "$repo_dir" && pwd -P)"
  fi

  local -a model_args=()
  if [ -n "$RUN_MODEL" ]; then
    model_args=(--model "$RUN_MODEL")
  fi
  local -a launch_args=(--cwd "$repo_dir" --harness "$harness" -i)

  # --- C1: prompt lands and is verified ------------------------------------
  log "C1: launching interactive ${harness} run with sentinel prompt"
  if agent_run "${launch_args[@]}" \
      --prompt "Reply with exactly this word and nothing else: ${sentinel1}" \
      ${model_args[@]+"${model_args[@]}"} "$run_name" >/dev/null 2>&1; then
    LAUNCHED_RUNS+=("$run_name")
    local state_dir="${STATE_DIR}/${run_name}"
    # Allow for harness startup and raw-mode readiness under load.
    if wait_for 60 0.5 "prompt_submitted appears, prompt_unverified absent" \
        bash -c "[ -f '${state_dir}/prompt_submitted' ] && [ ! -f '${state_dir}/prompt_unverified' ]"; then
      if wait_for 30 0.5 "sentinel1 reaches the transcript" \
          transcript_contains "$run_name" "$sentinel1"; then
        c1="PASS"
      else
        log "C1 FAIL: sentinel never appeared in transcript"
      fi
    else
      local unverified_detail trust_dialog_seen log_path
      unverified_detail="$(cat "${state_dir}/prompt_unverified" 2>/dev/null || echo "(prompt_unverified not written)")"
      log_path="${LOG_DIR}/${run_name}/log"
      trust_dialog_seen="no"
      if [ "$harness" = "claude" ] && grep -qi "trust" "$log_path" 2>/dev/null; then
        trust_dialog_seen="yes"
      fi
      log "C1 FAIL: prompt_unverified detail=[${unverified_detail}] trust_dialog_in_log=${trust_dialog_seen}"
    fi
  else
    log "C1 FAIL: launch itself failed"
  fi

  # --- C2: steer lands and is verified -------------------------------------
  if [ "$c1" = "PASS" ]; then
    log "C2: steering with sentinel2"
    local steer_out
    if steer_out="$(agent_run steer "$run_name" "Reply with exactly this word and nothing else: ${sentinel2}" 2>&1)"; then
      if echo "$steer_out" | grep -q "verified" \
          && wait_for 30 0.5 "sentinel2 reaches the transcript" \
             transcript_contains "$run_name" "$sentinel2"; then
        c2="PASS"
      else
        log "C2 FAIL: steer succeeded but 'verified' or the transcript record is missing: ${steer_out}"
      fi
    else
      log "C2 FAIL: steer exited non-zero: ${steer_out}"
    fi
  else
    c2="SKIP(C1 failed)"
  fi
  count_check "$c2"

  # Tear down the C1/C2 run before starting C3's fresh one.
  if [ "$c1" = "PASS" ]; then
    kill_run_and_wait "$run_name"
  fi
  count_check "$c1"

  # --- C3: anti-vacuity negative control ------------------------------------
  # Break each transport's witness independently to prove prompt_submitted is
  # delivery-gated: unreadable rollout, unreachable HTTP port, or tiny timeout.
  #
  # A negative control that merely observes "something went wrong" proves
  # nothing -- a failed launch or an unrelated error would pass it just as
  # well. Every branch below therefore asserts three things: the run actually
  # started (so the fault is what stopped verification), the injected fault
  # demonstrably took effect, and prompt_unverified names the reason that
  # fault produces rather than any reason at all.
  local c3_run="${run_name}-neg"
  local c3_state="${STATE_DIR}/${c3_run}"
  local c3_fault_landed=0        # the injected fault was confirmed in place
  local -a c3_expected_reasons=()  # the reason must be one of these
  local c3_launched=0
  if [ "$harness" = "codex" ]; then
    log "C3: negative control via an unreadable codex rollout store (launch marker)"
    local c3_sessions_dir="${WORK_DIR}/empty-codex-sessions"
    c3_expected_reasons=(witness_unreadable)
    if AGENT_RUN_CODEX_SESSIONS_DIR="$c3_sessions_dir" \
        agent_run "${launch_args[@]}" \
        --prompt "Reply with exactly this word and nothing else: ${sentinel1}_neg" \
        ${model_args[@]+"${model_args[@]}"} "$c3_run" >/dev/null 2>&1; then
      LAUNCHED_RUNS+=("$c3_run")
      c3_launched=1
      # The fault is the override pointing at a directory holding no rollout
      # for this session, so every witness read must fail. Confirm that is
      # still true rather than assuming the env var reached the runner.
      if [ -z "$(find "$c3_sessions_dir" -name 'rollout-*.jsonl' 2>/dev/null | head -1)" ]; then
        c3_fault_landed=1
      else
        log "C3: rollout override directory unexpectedly holds a rollout — fault did not land"
      fi
      # The unreadable store uses the full default verification window.
      wait_for 20 0.5 "prompt_submitted or prompt_unverified appears" \
        bash -c "[ -f '${c3_state}/prompt_submitted' ] || [ -f '${c3_state}/prompt_unverified' ]" || true
    else
      log "C3 FAIL: launch itself failed"
    fi
  elif [ "$harness" = "opencode" ]; then
    log "C3: negative control via an unreachable opencode_port (HTTP witness lands too fast for a timing control)"
    c3_expected_reasons=(witness_unreadable)
    if agent_run "${launch_args[@]}" \
        --prompt "Reply with exactly this word and nothing else: ${sentinel1}_neg" \
        ${model_args[@]+"${model_args[@]}"} "$c3_run" >/dev/null 2>&1; then
      LAUNCHED_RUNS+=("$c3_run")
      c3_launched=1
      if wait_for 10 0.1 "opencode_port file appears" \
          bash -c "[ -s '${c3_state}/opencode_port' ]"; then
        # Corrupt the port before the delayed prompt helper submits.
        echo 1 > "${c3_state}/opencode_port"
        sleep 6
        # The fault is the witness pointing at a port nothing serves. Confirm
        # the corrupted value survived to submission time and that nothing
        # answers there, so an unrelated failure cannot pass this control.
        local c3_port
        c3_port="$(cat "${c3_state}/opencode_port" 2>/dev/null || echo "")"
        if [ "$c3_port" = "1" ] \
            && ! curl -fsS --max-time 2 -o /dev/null "http://127.0.0.1:1/doc" 2>/dev/null; then
          c3_fault_landed=1
        else
          log "C3: opencode_port is [${c3_port}] and/or that port answers — fault did not land"
        fi
      else
        log "C3: opencode_port never appeared — could not inject the fault"
        sleep 6
      fi
    else
      log "C3 FAIL: launch itself failed"
    fi
  else
    log "C3: negative control with AGENT_RUN_SUBMIT_VERIFY_TIMEOUT=0.01"
    # A 10ms window ends before any store read can confirm delivery. Whether
    # the store was readable-but-flat (timeout) or not yet created
    # (witness_unreadable) depends on how fast the harness materialises its
    # transcript file; both are this fault's signature, and neither is a
    # transport error, a rejection or an unwitnessed run.
    c3_expected_reasons=(timeout witness_unreadable)
    local c3_start c3_elapsed
    c3_start=$(date +%s.%N)
    if AGENT_RUN_SUBMIT_VERIFY_TIMEOUT=0.01 AGENT_RUN_SUBMIT_ATTEMPTS=1 \
        agent_run "${launch_args[@]}" \
        --prompt "Reply with exactly this word and nothing else: ${sentinel1}_neg" \
        ${model_args[@]+"${model_args[@]}"} "$c3_run" >/dev/null 2>&1; then
      LAUNCHED_RUNS+=("$c3_run")
      c3_launched=1
      wait_for 20 0.2 "prompt_submitted or prompt_unverified appears" \
        bash -c "[ -f '${c3_state}/prompt_submitted' ] || [ -f '${c3_state}/prompt_unverified' ]" || true
      # The fault is the shortened deadline. Its observable effect is that
      # the verdict lands far sooner than the default window could produce
      # one: the fixed pre-submit delay plus a 10ms verify, not 10s.
      c3_elapsed=$(awk -v s="$c3_start" -v n="$(date +%s.%N)" 'BEGIN{printf "%.2f", n-s}')
      if awk -v e="$c3_elapsed" 'BEGIN{exit !(e < 10)}'; then
        c3_fault_landed=1
      else
        log "C3: verdict took ${c3_elapsed}s, no faster than the default window — timeout override did not land"
      fi
    else
      log "C3 FAIL: launch itself failed"
    fi
  fi

  local c3_reason=""
  if [ -f "${c3_state}/prompt_unverified" ]; then
    c3_reason="$(tr -d '\r\n' < "${c3_state}/prompt_unverified" 2>/dev/null || echo "")"
  fi
  if [ "$c3_launched" -ne 1 ]; then
    log "C3 FAIL: the run never launched, so nothing was controlled for"
  elif [ ! -s "${LOG_DIR}/${c3_run}/session.json" ]; then
    log "C3 FAIL: no session.json — the run failed before submission, not because of the injected fault"
  elif [ "$c3_fault_landed" -ne 1 ]; then
    log "C3 FAIL: the injected fault could not be confirmed, so an unverified prompt proves nothing"
  elif [ -f "${c3_state}/prompt_submitted" ]; then
    log "C3 FAIL (VACUOUS): prompt_submitted was stamped despite the negative control -- verification is not gating the marker"
  elif [ ! -f "${c3_state}/prompt_unverified" ]; then
    log "C3 FAIL: neither prompt_submitted nor prompt_unverified is present"
  elif [ -z "$c3_reason" ]; then
    log "C3 FAIL: prompt_unverified exists but carries no reason"
  elif ! printf '%s\n' "${c3_expected_reasons[@]}" | grep -qxF "$c3_reason"; then
    log "C3 FAIL: prompt_unverified reason is [${c3_reason}], expected one of [${c3_expected_reasons[*]}] for this fault"
  else
    c3="PASS"
  fi
  kill_run_and_wait "$c3_run"
  count_check "$c3"

  # --- C4: --raw steer is exempt from verification --------------------------
  local c4_run="${run_name}-raw"
  log "C4: --raw steer exemption"
  if agent_run "${launch_args[@]}" \
      --prompt "Reply with exactly this word and nothing else: ${sentinel1}_raw" \
      ${model_args[@]+"${model_args[@]}"} "$c4_run" >/dev/null 2>&1; then
    LAUNCHED_RUNS+=("$c4_run")
    local c4_state="${STATE_DIR}/${c4_run}"
    wait_for 30 0.5 "run reaches running" \
      bash -c "grep -qx running '${c4_state}/status' 2>/dev/null" || true
    local raw_out raw_rc=0
    raw_out="$(agent_run steer --raw "$c4_run" $'\033[Z' 2>&1)" || raw_rc=$?
    if [ "$harness" = "codex" ]; then
      # _reject_raw_steer_for_codex rejects --raw by design for managed codex.
      if [ "$raw_rc" -ne 0 ] && echo "$raw_out" | grep -qi "raw"; then
        c4="PASS"
      else
        log "C4 FAIL: codex --raw steer must be rejected by design, got rc=${raw_rc}: ${raw_out}"
      fi
    else
      if [ "$raw_rc" -eq 0 ] && echo "$raw_out" | grep -q "raw, unverified)" \
          && ! echo "$raw_out" | grep -q ", verified)"; then
        c4="PASS"
      else
        log "C4 FAIL: rc=${raw_rc} out=${raw_out}"
      fi
    fi
    kill_run_and_wait "$c4_run"
  else
    log "C4 FAIL: launch itself failed"
  fi
  count_check "$c4"

  record_row "$harness" "$version" "$c1" "$c2" "$c3" "$c4"
}

# ---------------------------------------------------------------------------
# opencode HTTP contract (H1-H5)
# ---------------------------------------------------------------------------

H_RESULTS=()

run_opencode_http_contract() {
  if ! command -v opencode >/dev/null 2>&1; then
    log "opencode not installed — skipping HTTP contract checks (H1-H5)"
    set_all_http_results "SKIP"
    return 0
  fi

  local run_name="vsub-http-$$"
  local sentinel="SENTINELHTTP_$$"
  local repo_dir
  repo_dir="$(mktemp -d)"
  TEMP_REPOS+=("$repo_dir")
  init_test_repo "$repo_dir"

  local -a model_args=()
  if [ -n "$RUN_MODEL" ]; then
    model_args=(--model "$RUN_MODEL")
  fi

  log "=== opencode HTTP contract: launching ${run_name} ==="
  local h1="FAIL" h2="FAIL" h3="FAIL" h4="FAIL" h5="FAIL"
  if ! agent_run --cwd "$repo_dir" --harness opencode -i \
      --prompt "Reply with exactly this word and nothing else: ${sentinel}" \
      ${model_args[@]+"${model_args[@]}"} "$run_name" >/dev/null 2>&1; then
    log "HTTP contract: launch failed; all of H1-H5 FAIL"
    set_all_http_results "FAIL"
    return 0
  fi
  LAUNCHED_RUNS+=("$run_name")
  local state_dir="${STATE_DIR}/${run_name}"

  if ! wait_for 30 0.5 "opencode_port file appears" \
      bash -c "[ -s '${state_dir}/opencode_port' ]"; then
    log "HTTP contract: opencode_port never appeared; all of H1-H5 FAIL"
    set_all_http_results "FAIL"
    agent_run kill "$run_name" >/dev/null 2>&1 || true
    return 0
  fi
  local port session_id
  port="$(cat "${state_dir}/opencode_port")"
  session_id="$(python3 -c "
import json
print(json.load(open('${LOG_DIR}/${run_name}/session.json'))['session_id'])
" 2>/dev/null || true)"
  if [ -z "$session_id" ]; then
    log "HTTP contract: could not resolve session_id from session.json; all of H1-H5 FAIL"
    set_all_http_results "FAIL"
    agent_run kill "$run_name" >/dev/null 2>&1 || true
    return 0
  fi
  local base="http://127.0.0.1:${port}"
  local message_url="${base}/session/${session_id}/message"

  # opencode_port is written as soon as the port is selected, before the TUI
  # process has actually bound it -- wait for /doc to answer before running
  # any check against it, so H1-H5 measure the harness contract, not a launch
  # race.
  if ! wait_for 30 0.5 "opencode HTTP API answers GET /doc" \
      bash -c "curl -fsS --max-time 2 -o /dev/null '${base}/doc' 2>/dev/null"; then
    log "HTTP contract: opencode HTTP API never came up; all of H1-H5 FAIL"
    set_all_http_results "FAIL"
    agent_run kill "$run_name" >/dev/null 2>&1 || true
    return 0
  fi

  # The launch prompt's own turn must fully settle (its assistant reply
  # recorded) before H2-H5 run: racing it corrupts H4/H5's before/after
  # message counts and their role-sequence assertions against a turn that
  # is still generating.
  if ! wait_for 30 0.5 "launch turn settles (2 message records)" \
      bash -c "[ \"\$(curl -fsS --max-time 2 '${message_url}' 2>/dev/null | python3 -c 'import json,sys; d=json.load(sys.stdin); print(len(d))' 2>/dev/null)\" = 2 ]"; then
    log "HTTP contract: launch turn never settled; all of H1-H5 FAIL"
    set_all_http_results "FAIL"
    agent_run kill "$run_name" >/dev/null 2>&1 || true
    return 0
  fi

  # H1: route still exists.
  log "H1: GET /doc, checking for POST /session/{sessionID}/message"
  local doc_json
  if doc_json="$(curl -sS --max-time 5 "${base}/doc" 2>/dev/null)" \
      && echo "$doc_json" | python3 -c "
import json, sys
d = json.load(sys.stdin)
paths = d.get('paths', {})
key = next((k for k in paths if k.startswith('/session/{') and k.endswith('}/message')), None)
sys.exit(0 if key and 'post' in paths[key] else 1)
" 2>/dev/null; then
    h1="PASS"
  else
    log "H1 FAIL: /doc missing a POST /session/{sessionID}/message route"
  fi

  # H2: POST drives the agent.
  log "H2: POST a sentinel message, expect 200 and a transcript reply"
  local h2_sentinel="H2_${sentinel}"
  local h2_payload="${WORK_DIR}/h2-payload.json"
  json_message_payload "$h2_payload" "Reply with exactly this word and nothing else: ${h2_sentinel}"
  local http_code
  http_code="$(post_json 10 "$message_url" "$h2_payload" -w '%{http_code}')" || http_code="000"
  if [ "$http_code" = "200" ] && wait_for 30 0.5 "H2 sentinel reaches the transcript" \
      transcript_contains "$run_name" "$h2_sentinel"; then
    h2="PASS"
  else
    log "H2 FAIL: http_code=${http_code}"
  fi

  # H3: fire-and-forget is safe (dropped connection does not abort the turn).
  log "H3: POST with a short client timeout, expect the turn to still complete"
  local h3_sentinel="H3_${sentinel}"
  local h3_payload="${WORK_DIR}/h3-payload.json"
  json_message_payload "$h3_payload" \
    "Count slowly from one to five, one number per line, then say exactly this word and nothing else: ${h3_sentinel}"
  # The whole point of H3 is that the *client* goes away mid-turn. A POST that
  # completed normally inside the timeout never exercised that, so discarding
  # curl's result would let H3 pass without testing anything: assert the
  # connection was actually aborted (curl exit 28, no http_code) *and* that
  # the turn still ran to completion afterwards.
  local h3_code="" h3_rc=0
  h3_code="$(post_json 2 "$message_url" "$h3_payload" -w '%{http_code}')" || h3_rc=$?
  local h3_aborted=0
  if [ "$h3_rc" -eq 28 ] && [ -z "${h3_code//0/}" ]; then
    h3_aborted=1
  fi
  if [ "$h3_aborted" -ne 1 ]; then
    log "H3 FAIL: the client connection was not aborted (curl rc=${h3_rc} http_code=[${h3_code}]); fire-and-forget was never exercised"
  elif wait_for 60 1 "H3 sentinel reaches the transcript after a dropped connection" \
      transcript_contains "$run_name" "$h3_sentinel"; then
    h3="PASS"
  else
    log "H3 FAIL: turn did not complete after the client dropped the connection"
  fi

  # H4: noReply suppresses generation.
  log "H4: POST with noReply:true, expect the message inserted with no reply"
  local h4_sentinel="H4_${sentinel}"
  local h4_payload="${WORK_DIR}/h4-payload.json"
  local before_count after_count
  before_count="$(message_count "$message_url")"
  json_message_payload "$h4_payload" "$h4_sentinel" noreply
  post_json 5 "$message_url" "$h4_payload" || true
  sleep 3
  after_count="$(message_count "$message_url")"
  local h4_last_role
  h4_last_role="$(curl -sS --max-time 5 "$message_url" 2>/dev/null | python3 -c '
import json, sys
try:
    data = json.load(sys.stdin)
    print(data[-1]["info"]["role"] if data else "")
except (json.JSONDecodeError, IndexError, KeyError):
    print("")
' 2>/dev/null || echo "")"
  if [ "$before_count" -ge 0 ] 2>/dev/null && [ "$after_count" -eq $((before_count + 1)) ] 2>/dev/null \
      && [ "$h4_last_role" = "user" ]; then
    h4="PASS"
  else
    log "H4 FAIL: before=${before_count} after=${after_count} last_role=${h4_last_role} (expected exactly one new user record with no trailing assistant reply)"
  fi

  # H5: messageID is not idempotent.
  log "H5: POST the same messageID twice, expect two distinct results"
  # This is the property the whole retry design rests on: a resubmission
  # duplicates rather than deduplicates. Ignoring both POST statuses would let
  # H5 pass on a server that rejected the second request outright, which is
  # the *opposite* finding. Assert both POSTs returned 200 and that two
  # distinct user messages exist.
  # opencode validates messageID as a string with a "msg" prefix; anything
  # else is a 400 before the duplication behavior under test even runs.
  local mid="msgverifysubmissionh5$$"
  local h5_payload_a="${WORK_DIR}/h5-payload-a.json"
  local h5_payload_b="${WORK_DIR}/h5-payload-b.json"
  local h5_text_a="h5-first-${sentinel}"
  local h5_text_b="h5-second-${sentinel}"
  json_message_payload "$h5_payload_a" "$h5_text_a" noreply "$mid"
  json_message_payload "$h5_payload_b" "$h5_text_b" noreply "$mid"
  local h5_code_a="000" h5_code_b="000"
  h5_code_a="$(post_json 5 "$message_url" "$h5_payload_a" -w '%{http_code}')" || h5_code_a="000"
  h5_code_b="$(post_json 5 "$message_url" "$h5_payload_b" -w '%{http_code}')" || h5_code_b="000"
  sleep 1
  local h5_distinct
  h5_distinct="$(distinct_user_messages_containing "$message_url" "$h5_text_a" "$h5_text_b")"
  if [ "$h5_code_a" != "200" ] || [ "$h5_code_b" != "200" ]; then
    log "H5 FAIL: expected both POSTs to return 200, got ${h5_code_a} and ${h5_code_b} — a rejected duplicate is not evidence of non-idempotence"
  elif [ "$h5_distinct" -ge 2 ] 2>/dev/null; then
    h5="PASS"
  else
    log "H5 FAIL: expected 2 distinct user messages for the duplicated messageID, got ${h5_distinct}"
  fi

  agent_run kill "$run_name" >/dev/null 2>&1 || true

  H_RESULTS=("$h1" "$h2" "$h3" "$h4" "$h5")
  for r in "${H_RESULTS[@]}"; do count_check "$r"; done
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

echo "verify-submission"
if [ -n "$ONLY_HARNESS" ]; then
  echo "harness: ${ONLY_HARNESS}"
else
  echo "harnesses: opencode claude codex (installed only)"
fi
echo "workdir: ${WORK_DIR}"
echo

HARNESSES=(opencode claude codex)
if [ -n "$ONLY_HARNESS" ]; then
  HARNESSES=("$ONLY_HARNESS")
fi

for h in "${HARNESSES[@]}"; do
  run_harness_cells "$h"
  echo
done

if [ -z "$ONLY_HARNESS" ]; then
  run_opencode_http_contract
  echo
fi

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

printf '%-10s %-14s %-6s %-6s %-6s %-6s\n' harness version C1 C2 C3 C4
i=0
while [ "$i" -lt "${#RESULTS_HARNESS[@]}" ]; do
  printf '%-10s %-14s %-6s %-6s %-6s %-6s\n' \
    "${RESULTS_HARNESS[$i]}" "${RESULTS_VERSION[$i]}" \
    "${RESULTS_C1[$i]}" "${RESULTS_C2[$i]}" "${RESULTS_C3[$i]}" "${RESULTS_C4[$i]}"
  i=$((i + 1))
done
echo

if [ -z "$ONLY_HARNESS" ]; then
  printf 'opencode HTTP contract: H1 %s  H2 %s  H3 %s  H4 %s  H5 %s\n' \
    "${H_RESULTS[0]}" "${H_RESULTS[1]}" "${H_RESULTS[2]}" "${H_RESULTS[3]}" "${H_RESULTS[4]}"
  echo
fi

if [ "$OVERALL_FAIL" -gt 0 ]; then
  echo "RESULT: FAIL (${OVERALL_CHECKS} checks, ${OVERALL_FAIL} failures, ${OVERALL_SKIPS} skips)"
  exit 1
fi
echo "RESULT: PASS (${OVERALL_CHECKS} checks, 0 failures, ${OVERALL_SKIPS} skips)"
exit 0
