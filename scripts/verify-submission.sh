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

# claude_trust_dir <realpath>: pre-accept claude's workspace-trust dialog for
# a throwaway repo so an interactive launch doesn't block on it forever.
# Claude keys ~/.claude.json's per-project trust entry by realpath, not by
# whatever path mktemp -d returned (macOS's /var/... is a symlink to
# /private/var/...); passing the raw mktemp path here silently no-ops.
# Read-modify-write races with a concurrently running `claude` are accepted:
# this script never runs more than one harness launch against ~/.claude.json
# at a time.
claude_trust_dir() {
  local dir="$1"
  python3 -c "
import json, sys
path = '${HOME}/.claude.json'
try:
    with open(path) as f:
        data = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    sys.exit(0)  # nothing to seed into; claude creates its own defaults
data.setdefault('projects', {})['${dir}'] = {'hasTrustDialogAccepted': True}
with open(path, 'w') as f:
    json.dump(data, f)
" 2>/dev/null || true
  CLAUDE_TRUSTED_DIRS+=("$dir")
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

log() { echo "verify-submission: $*" >&2; }

# wait_for <timeout_s> <poll_interval_s> <description> <cmd...>
# Polls <cmd...> (a predicate; exit 0 = satisfied) until it succeeds or the
# timeout elapses. Prints elapsed time on success so a merely-slower harness
# is visible rather than silently swallowed by a generous timeout.
wait_for() {
  local timeout="$1" interval="$2" desc="$3"
  shift 3
  local start elapsed
  start=$(date +%s.%N)
  while true; do
    if "$@"; then
      elapsed=$(awk -v s="$start" -v n="$(date +%s.%N)" 'BEGIN{printf "%.2f", n-s}')
      log "  ok (${elapsed}s): ${desc}"
      return 0
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

# json_message_payload <outfile> <text> [noreply] [message_id]: write a
# POST /session/{id}/message JSON body to outfile. Values cross into python
# via environment variables, never interpolated into python source text --
# a JSON literal embedded in a python -c string, itself embedded in a shell
# command substitution inside curl -d "...", breaks under nested quote
# escaping (single quotes inside the JSON collide with the quoting layer
# python -c and $() each add). Environment passing has no such layer to
# collide across.
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

# message_parts_for_id <url> <message_id>: number of parts recorded under the
# message whose info.id equals message_id, or 0 when no such message exists.
message_parts_for_id() {
  curl -sS --max-time 5 "$1" 2>/dev/null | MSG_ID="$2" python3 -c '
import json, os, sys
try:
    data = json.load(sys.stdin)
except json.JSONDecodeError:
    print(0)
else:
    mid = os.environ["MSG_ID"]
    msg = next((m for m in data if m.get("info", {}).get("id") == mid), None)
    print(len(msg["parts"]) if msg else 0)
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

# ---------------------------------------------------------------------------
# Per-harness cells: C1 (launch verified), C2 (steer verified),
# C3 (anti-vacuity negative control), C4 (--raw exemption)
# ---------------------------------------------------------------------------

run_harness_cells() {
  local harness="$1"
  local run_name="vsub-${harness}-$$"
  local sentinel1="SENTINEL1_${harness}_$$"
  local sentinel2="SENTINEL2_${harness}_$$"
  local version="UNKNOWN"
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
  version="$("$harness" --version 2>&1 | head -1 | tr -d '\r')"
  log "=== harness=${harness} version=${version} run=${run_name} ==="

  local repo_dir
  repo_dir="$(mktemp -d)"
  TEMP_REPOS+=("$repo_dir")
  git -C "$repo_dir" init -q
  git -C "$repo_dir" config user.email "verify-submission@example.invalid"
  git -C "$repo_dir" config user.name "verify-submission"
  echo "placeholder" > "$repo_dir/README.md"
  git -C "$repo_dir" add README.md
  git -C "$repo_dir" commit -q -m "init"
  if [ "$harness" = "claude" ]; then
    claude_trust_dir "$(cd -- "$repo_dir" && pwd -P)"
  fi

  local -a model_args=()
  if [ -n "$RUN_MODEL" ]; then
    model_args=(--model "$RUN_MODEL")
  fi

  # --- C1: prompt lands and is verified ------------------------------------
  log "C1: launching interactive ${harness} run with sentinel prompt"
  if agent_run --cwd "$repo_dir" --harness "$harness" -i \
      --prompt "Reply with exactly this word and nothing else: ${sentinel1}" \
      ${model_args[@]+"${model_args[@]}"} "$run_name" >/dev/null 2>&1; then
    LAUNCHED_RUNS+=("$run_name")
    local state_dir="${STATE_DIR}/${run_name}"
    # 60s, not 30: PROMPT_SUBMISSION_DELAY_SECONDS (the TUI raw-mode-readiness
    # wait) is a fixed 4.0s regardless of system load, so under real
    # contention (other harness processes competing for CPU) claude's own
    # startup can push well past a tight bound before it ever reaches raw
    # mode -- a slow environment, not a verification failure.
    if wait_for 60 0.5 "prompt_submitted appears, prompt_unverified absent" \
        bash -c "[ -f '${state_dir}/prompt_submitted' ] && [ ! -f '${state_dir}/prompt_unverified' ]"; then
      if wait_for 30 0.5 "sentinel1 reaches the transcript" \
          transcript_contains "$run_name" "$sentinel1"; then
        c1="PASS"
      else
        c1="FAIL"
        log "C1 FAIL: sentinel never appeared in transcript"
      fi
    else
      c1="FAIL"
      log "C1 FAIL: prompt_submitted/prompt_unverified state is wrong"
    fi
  else
    c1="FAIL"
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
        c2="FAIL"
        log "C2 FAIL: steer succeeded but 'verified' or the transcript record is missing: ${steer_out}"
      fi
    else
      c2="FAIL"
      log "C2 FAIL: steer exited non-zero: ${steer_out}"
    fi
  else
    c2="SKIP(C1 failed)"
  fi
  count_check "$c2"

  # Tear down the C1/C2 run before starting C3's fresh one.
  if [ "$c1" = "PASS" ] || [ "$c2" != "SKIP(C1 failed)" ]; then
    agent_run kill "$run_name" >/dev/null 2>&1 || true
    sleep 0.3
  fi
  count_check "$c1"

  # --- C3: anti-vacuity negative control ------------------------------------
  # Proves the verification marker is gated by real delivery, not by "bytes
  # handed to a transport". Which marker C3 tests, and how the control is
  # engineered, differs per harness because a uniform approach is
  # architecturally impossible across all three transports:
  #
  # opencode: tests the launch marker (prompt_submitted). Its HTTP
  #   message-count witness (GET /session/<id>/message) lands in single-digit
  #   milliseconds (measured live: ~8ms), so no AGENT_RUN_SUBMIT_VERIFY_TIMEOUT
  #   short enough to be "impossible" exists for that transport --
  #   _submit_and_verify's own post-timeout re-check would still observe the
  #   already-landed witness and report verified. Instead corrupt
  #   state_dir/opencode_port to an unreachable port immediately after
  #   launch, before the prompt-submission helper's fixed
  #   PROMPT_SUBMISSION_DELAY_SECONDS pre-write delay elapses, so every HTTP
  #   submission attempt hits a real connection failure.
  #
  # claude: tests the launch marker (prompt_submitted), gated by
  #   _submit_and_verify exactly like opencode's. Transcript-store landing
  #   latency is seconds (measured live: ~4.3s), so
  #   AGENT_RUN_SUBMIT_VERIFY_TIMEOUT=0.01 genuinely cannot be satisfied by
  #   either the poll loop or the single post-timeout re-check.
  #
  # codex: tests the STEER marker instead of the launch marker. codex's
  #   launch-time prompt_submitted is gated only by turn_start_error (an
  #   RPC-level ack that the turn started), never by the transcript witness
  #   -- _submit_and_verify is wired into the interactive codex runner's
  #   steer path only, not its initial turn/start. AGENT_RUN_SUBMIT_VERIFY_TIMEOUT
  #   therefore has no effect on codex's launch marker; testing it there
  #   would silently no-op rather than prove anything. Redirecting
  #   AGENT_RUN_CODEX_SESSIONS_DIR to an empty directory makes the rollout
  #   witness unreadable for every submission attempt, so a steer's
  #   verification cannot succeed regardless of any timeout.
  #
  # Whichever marker/technique applies, if this control ever reports
  # PASS-verification, that marker is not actually gated by the witness and
  # the whole feature is vacuous for it -- treat that as a hard failure, not
  # a mere anomaly.
  local c3_run="${run_name}-neg"
  local c3_state="${STATE_DIR}/${c3_run}"
  if [ "$harness" = "codex" ]; then
    log "C3: negative control via an unreadable codex rollout store (steer marker; codex's launch marker is not witness-gated)"
    if agent_run --cwd "$repo_dir" --harness "$harness" -i \
        --prompt "Reply with exactly this word and nothing else: ${sentinel1}_neg" \
        ${model_args[@]+"${model_args[@]}"} "$c3_run" >/dev/null 2>&1; then
      LAUNCHED_RUNS+=("$c3_run")
      wait_for 10 0.5 "run reaches running" \
        bash -c "grep -qx running '${c3_state}/status' 2>/dev/null" || true
      local c3_steer_out c3_steer_rc=0
      c3_steer_out="$(AGENT_RUN_CODEX_SESSIONS_DIR="${WORK_DIR}/empty-codex-sessions" \
        agent_run steer "$c3_run" "Reply with exactly this word and nothing else: ${sentinel1}_neg_steer" 2>&1)" \
        || c3_steer_rc=$?
      if [ "$c3_steer_rc" -eq 1 ] && echo "$c3_steer_out" | grep -q "could not be verified"; then
        c3="PASS"
      else
        c3="FAIL"
        if [ "$c3_steer_rc" -eq 0 ]; then
          log "C3 FAIL (VACUOUS): steer reported verified despite an unreadable rollout store -- verification is not gating the steer marker"
        else
          log "C3 FAIL: rc=${c3_steer_rc} out=${c3_steer_out}"
        fi
      fi
    else
      c3="FAIL"
      log "C3 FAIL: launch itself failed"
    fi
    agent_run kill "$c3_run" >/dev/null 2>&1 || true
    sleep 0.3
    count_check "$c3"
  elif [ "$harness" = "opencode" ]; then
    log "C3: negative control via an unreachable opencode_port (HTTP witness lands too fast for a timing control)"
    if agent_run --cwd "$repo_dir" --harness "$harness" -i \
        --prompt "Reply with exactly this word and nothing else: ${sentinel1}_neg" \
        ${model_args[@]+"${model_args[@]}"} "$c3_run" >/dev/null 2>&1; then
      LAUNCHED_RUNS+=("$c3_run")
      if wait_for 10 0.1 "opencode_port file appears" \
          bash -c "[ -s '${c3_state}/opencode_port' ]"; then
        # The prompt-submission helper still has ~PROMPT_SUBMISSION_DELAY_SECONDS
        # of its own pre-write delay left; corrupt the port before it fires so
        # every HTTP attempt it makes hits a real connection refusal.
        echo 1 > "${c3_state}/opencode_port"
      fi
      sleep 6
    else
      log "C3 FAIL: launch itself failed"
    fi
  else
    log "C3: negative control with AGENT_RUN_SUBMIT_VERIFY_TIMEOUT=0.01"
    if AGENT_RUN_SUBMIT_VERIFY_TIMEOUT=0.01 AGENT_RUN_SUBMIT_ATTEMPTS=1 \
        agent_run --cwd "$repo_dir" --harness "$harness" -i \
        --prompt "Reply with exactly this word and nothing else: ${sentinel1}_neg" \
        ${model_args[@]+"${model_args[@]}"} "$c3_run" >/dev/null 2>&1; then
      LAUNCHED_RUNS+=("$c3_run")
      # The helper's own fixed pre-write delay is unaffected by the timeout
      # override, so give it the same grace before checking the markers.
      sleep 6
    else
      log "C3 FAIL: launch itself failed"
    fi
  fi
  if [ "$harness" != "codex" ]; then
    if [ -f "${c3_state}/prompt_submitted" ]; then
      c3="FAIL"
      log "C3 FAIL (VACUOUS): prompt_submitted was stamped despite the negative control -- verification is not gating the marker"
    elif [ ! -f "${c3_state}/prompt_unverified" ]; then
      c3="FAIL"
      log "C3 FAIL: neither prompt_submitted nor prompt_unverified is present"
    elif [ ! -s "${c3_state}/prompt_unverified" ]; then
      c3="FAIL"
      log "C3 FAIL: prompt_unverified exists but carries no reason"
    else
      c3="PASS"
    fi
    agent_run kill "$c3_run" >/dev/null 2>&1 || true
    sleep 0.3
    count_check "$c3"
  fi

  # --- C4: --raw steer is exempt from verification --------------------------
  local c4_run="${run_name}-raw"
  log "C4: --raw steer exemption"
  if agent_run --cwd "$repo_dir" --harness "$harness" -i \
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
        c4="FAIL"
        log "C4 FAIL: codex --raw steer must be rejected by design, got rc=${raw_rc}: ${raw_out}"
      fi
    else
      if [ "$raw_rc" -eq 0 ] && echo "$raw_out" | grep -q "raw, unverified)" \
          && ! echo "$raw_out" | grep -q ", verified)"; then
        c4="PASS"
      else
        c4="FAIL"
        log "C4 FAIL: rc=${raw_rc} out=${raw_out}"
      fi
    fi
    agent_run kill "$c4_run" >/dev/null 2>&1 || true
    sleep 0.3
  else
    c4="FAIL"
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
    H_RESULTS=(SKIP SKIP SKIP SKIP SKIP)
    for _ in 1 2 3 4 5; do count_check "SKIP"; done
    return 0
  fi

  local run_name="vsub-http-$$"
  local sentinel="SENTINELHTTP_$$"
  local repo_dir
  repo_dir="$(mktemp -d)"
  TEMP_REPOS+=("$repo_dir")
  git -C "$repo_dir" init -q
  git -C "$repo_dir" config user.email "verify-submission@example.invalid"
  git -C "$repo_dir" config user.name "verify-submission"
  echo "placeholder" > "$repo_dir/README.md"
  git -C "$repo_dir" add README.md
  git -C "$repo_dir" commit -q -m "init"

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
    H_RESULTS=(FAIL FAIL FAIL FAIL FAIL)
    for _ in 1 2 3 4 5; do count_check "FAIL"; done
    return 0
  fi
  LAUNCHED_RUNS+=("$run_name")
  local state_dir="${STATE_DIR}/${run_name}"

  if ! wait_for 30 0.5 "opencode_port file appears" \
      bash -c "[ -s '${state_dir}/opencode_port' ]"; then
    log "HTTP contract: opencode_port never appeared; all of H1-H5 FAIL"
    H_RESULTS=(FAIL FAIL FAIL FAIL FAIL)
    for _ in 1 2 3 4 5; do count_check "FAIL"; done
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
    H_RESULTS=(FAIL FAIL FAIL FAIL FAIL)
    for _ in 1 2 3 4 5; do count_check "FAIL"; done
    agent_run kill "$run_name" >/dev/null 2>&1 || true
    return 0
  fi
  local base="http://127.0.0.1:${port}"

  # opencode_port is written as soon as the port is selected, before the TUI
  # process has actually bound it -- wait for /doc to answer before running
  # any check against it, so H1-H5 measure the harness contract, not a launch
  # race.
  if ! wait_for 30 0.5 "opencode HTTP API answers GET /doc" \
      bash -c "curl -fsS --max-time 2 -o /dev/null '${base}/doc' 2>/dev/null"; then
    log "HTTP contract: opencode HTTP API never came up; all of H1-H5 FAIL"
    H_RESULTS=(FAIL FAIL FAIL FAIL FAIL)
    for _ in 1 2 3 4 5; do count_check "FAIL"; done
    agent_run kill "$run_name" >/dev/null 2>&1 || true
    return 0
  fi

  # The launch prompt's own turn must fully settle (its assistant reply
  # recorded) before H2-H5 run: racing it corrupts H4/H5's before/after
  # message counts and their role-sequence assertions against a turn that
  # is still generating.
  if ! wait_for 30 0.5 "launch turn settles (2 message records)" \
      bash -c "[ \"\$(curl -fsS --max-time 2 '${base}/session/${session_id}/message' 2>/dev/null | python3 -c 'import json,sys; d=json.load(sys.stdin); print(len(d))' 2>/dev/null)\" = 2 ]"; then
    log "HTTP contract: launch turn never settled; all of H1-H5 FAIL"
    H_RESULTS=(FAIL FAIL FAIL FAIL FAIL)
    for _ in 1 2 3 4 5; do count_check "FAIL"; done
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
    h1="FAIL"
    log "H1 FAIL: /doc missing a POST /session/{sessionID}/message route"
  fi

  # H2: POST drives the agent.
  log "H2: POST a sentinel message, expect 200 and a transcript reply"
  local h2_sentinel="H2_${sentinel}"
  local h2_payload="${WORK_DIR}/h2-payload.json"
  json_message_payload "$h2_payload" "Reply with exactly this word and nothing else: ${h2_sentinel}"
  local http_code
  http_code="$(curl -sS --max-time 10 -o /dev/null -w '%{http_code}' \
    -X POST "${base}/session/${session_id}/message" \
    -H 'Content-Type: application/json' \
    -d @"$h2_payload" \
    2>/dev/null)" || http_code="000"
  if [ "$http_code" = "200" ] && wait_for 30 0.5 "H2 sentinel reaches the transcript" \
      transcript_contains "$run_name" "$h2_sentinel"; then
    h2="PASS"
  else
    h2="FAIL"
    log "H2 FAIL: http_code=${http_code}"
  fi

  # H3: fire-and-forget is safe (dropped connection does not abort the turn).
  log "H3: POST with a short client timeout, expect the turn to still complete"
  local h3_sentinel="H3_${sentinel}"
  local h3_payload="${WORK_DIR}/h3-payload.json"
  json_message_payload "$h3_payload" \
    "Count slowly from one to five, one number per line, then say exactly this word and nothing else: ${h3_sentinel}"
  curl -sS --max-time 2 -o /dev/null \
    -X POST "${base}/session/${session_id}/message" \
    -H 'Content-Type: application/json' \
    -d @"$h3_payload" \
    2>/dev/null || true
  if wait_for 60 1 "H3 sentinel reaches the transcript after a dropped connection" \
      transcript_contains "$run_name" "$h3_sentinel"; then
    h3="PASS"
  else
    h3="FAIL"
    log "H3 FAIL: turn did not complete after the client dropped the connection"
  fi

  # H4: noReply suppresses generation.
  log "H4: POST with noReply:true, expect the message inserted with no reply"
  local h4_sentinel="H4_${sentinel}"
  local h4_payload="${WORK_DIR}/h4-payload.json"
  local message_url="${base}/session/${session_id}/message"
  local before_count after_count
  before_count="$(message_count "$message_url")"
  json_message_payload "$h4_payload" "$h4_sentinel" noreply
  curl -sS --max-time 5 -o /dev/null \
    -X POST "$message_url" \
    -H 'Content-Type: application/json' \
    -d @"$h4_payload" \
    2>/dev/null || true
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
    h4="FAIL"
    log "H4 FAIL: before=${before_count} after=${after_count} last_role=${h4_last_role} (expected exactly one new user record with no trailing assistant reply)"
  fi

  # H5: messageID is not idempotent.
  log "H5: POST the same messageID twice, expect two distinct results"
  # opencode validates messageID as a string with a "msg" prefix; anything
  # else is a 400 before the duplication behavior under test even runs.
  local mid="msgverifysubmissionh5$$"
  local h5_payload_a="${WORK_DIR}/h5-payload-a.json"
  local h5_payload_b="${WORK_DIR}/h5-payload-b.json"
  json_message_payload "$h5_payload_a" "h5-first" noreply "$mid"
  json_message_payload "$h5_payload_b" "h5-second" noreply "$mid"
  curl -sS --max-time 5 -o /dev/null \
    -X POST "$message_url" \
    -H 'Content-Type: application/json' \
    -d @"$h5_payload_a" \
    2>/dev/null || true
  curl -sS --max-time 5 -o /dev/null \
    -X POST "$message_url" \
    -H 'Content-Type: application/json' \
    -d @"$h5_payload_b" \
    2>/dev/null || true
  sleep 1
  local after_len
  after_len="$(message_parts_for_id "$message_url" "$mid")"
  if [ "$after_len" -ge 2 ] 2>/dev/null; then
    h5="PASS"
  else
    h5="FAIL"
    log "H5 FAIL: expected >=2 parts recorded for the duplicated messageID, got ${after_len}"
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
  if [ "${#H_RESULTS[@]}" -eq 5 ]; then
    printf 'opencode HTTP contract: H1 %s  H2 %s  H3 %s  H4 %s  H5 %s\n' \
      "${H_RESULTS[0]}" "${H_RESULTS[1]}" "${H_RESULTS[2]}" "${H_RESULTS[3]}" "${H_RESULTS[4]}"
  fi
  echo
fi

if [ "$OVERALL_FAIL" -gt 0 ]; then
  echo "RESULT: FAIL (${OVERALL_CHECKS} checks, ${OVERALL_FAIL} failures, ${OVERALL_SKIPS} skips)"
  exit 1
fi
echo "RESULT: PASS (${OVERALL_CHECKS} checks, 0 failures, ${OVERALL_SKIPS} skips)"
exit 0
