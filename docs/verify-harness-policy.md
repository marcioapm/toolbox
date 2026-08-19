# verify-harness-policy

End-to-end verifier for the managed-mode capability policy
(`--enable-planning` / `--enable-questions`, default-off).

## What it proves

The managed-mode policy is built on version-sensitive harness internals: a renamed codex
config key, a renamed claude tool, or a change in OpenCode's config merge order all leave
the test suite green while the policy stops being enforced.

This script detects two distinct failure modes:

- **Harness drift** — a harness upgrade renames a key, tool, or changes merge semantics.
  Cells C1–C5 detect this by calling the harness binary directly.
- **Our regression** — a change to `_opencode_policy_config()`, `_codex_policy_args()`,
  or `_build_managed_argv()` in `agent_run.py` silently breaks the policy builder.
  Cell C6 detects this by calling the real production function and verifying its output.

These are different failure modes; a reader must be able to tell which cells would catch
which kind of break. See the table in [What each cell would catch](#what-each-cell-would-catch).

## Two tiers

### Tier 1 — contract cells (`--free-only`)

No model calls. Completes in seconds. **Cost: $0.**

| Cell | What it asserts |
|------|----------------|
| `C1-codex-key-exists` | `tools.experimental_request_user_input` accepted by `codex app-server --strict-config`; a generated nonsense key is rejected. Both arms required — a tripwire with no negative control fails open. |
| `C2-codex-value-shape` | A bare boolean is rejected with the struct-type error, proving the struct literal our policy args use is still required. |
| `C3-codex-last-wins` | Duplicate `-c` overrides use last-occurrence-wins: invalid-first/valid-last exits 0; valid-first/invalid-last exits non-zero. The policy args are prepended so caller-supplied `-c` values override them. |
| `C4-claude-tool-names` | `--disallowedTools EnterPlanMode ExitPlanMode AskUserQuestion` produces no "matches no known tool" warning. A bogus name MUST produce it. No model call: claude validates tool names at startup in ~0.3 s. |
| `C5-claude-variadic` | `--disallowedTools` is variadic (claude splits a space-separated value into individual tool names) and additive (multiple `--disallowedTools` occurrences are cumulative, not last-wins). |
| `C6-opencode-resolver` | Calls the real `ar._opencode_policy_config()` and confirms `opencode debug agent` sees `question`/`plan_enter`/`plan_exit` as **denied** in the last matching rule, in six scenarios spanning default agent, `--agent-mode`, `--harness-arg --agent`, project allow, `OPENCODE_CONFIG` env override, and the enabled arm. |

### Tier 2 — behavioural cells (`--paid-only`)

Real API calls against the installed models. Approximate cost: **~$0.20–0.40 per full run**
(haiku/mini models, trivial prompts).

| Cell | What it asserts |
|------|----------------|
| `A1-codex-questions` | `functions.request_user_input` present when `enable_questions=True`, absent when `False`. Uses `functions.exec` as a positive sentinel — a run that fails to produce the sentinel doesn't count. Repeated five times. |
| `A2-opencode-deny-allow` | `OPENCODE_CONFIG_CONTENT` deny overrides a project `opencode.json` allow. Positive control proves bash was invoked in the allow arm. |
| `A3-claude-deny-allow` | `--disallowedTools Bash` removes `Bash` from `tool_use` records. Assertions are on tool records, never side effects — denying a tool removes the tool, not the capability. |
| `A5-enabled-arms` | Positive proof that each harness observably enables its capability when the flags are set (codex: differential tool present; opencode: bash allowed; claude: no `--disallowedTools` in argv). |

Tier-2 cells reuse helpers from `tests/test_agent_run_live_acceptance.py` (imported at
runtime). The pytest file keeps working exactly as it does today — no shared logic was
duplicated or moved.

## What each cell would catch

Two failure modes matter:
- **Our regression**: `agent_run.py` policy builder produces wrong output (e.g. missing per-agent deny block).
- **Harness drift**: a harness upgrade renames a key, tool, or changes merge/argv semantics.

| Cell | Detects our regression | Detects harness drift | How |
|------|----------------------|----------------------|-----|
| C1 | No | **Yes** | Calls `codex app-server --strict-config` directly — no production code involved. |
| C2 | No | **Yes** | Calls `codex app-server --strict-config` directly. |
| C3 | No | **Yes** | Calls `codex app-server --strict-config` with duplicate `-c`. |
| C4 | No | **Yes** | Calls `claude --disallowedTools` directly — no production code involved. |
| C5 | No | **Yes** | Calls `claude --disallowedTools` directly. |
| C6 | **Yes** | **Yes** | Calls `ar._opencode_policy_config()` (our code) then `opencode debug agent` (harness). A gutted policy builder (`target_agents = set()`) is detected at scenario 1: the project per-agent allow wins instead of the policy deny. |
| A1 | **Yes** | **Yes** | Calls `agent_run.cmd_launch` through the full managed path; result observed in the model's tool list. |
| A2 | **Yes** | **Yes** | Calls `opencode run` via `_opencode_subprocess`; result observed in `tool_use` records. |
| A3 | No | **Yes** | Calls `claude --disallowedTools` directly; result observed in `tool_use` records. |
| A5 | **Yes** | **Yes** | Exercises `_opencode_policy_config()` and `_build_managed_argv()` output. |

**Key point**: C1–C5 are harness-drift detectors only. A bug introduced into
`_opencode_policy_config()` or `_build_managed_argv()` would not be caught by any of them.
C6 (and the Tier-2 cells) are the ones that catch our own regressions.

## How to run

The script resolves `toolbox.agent_run` from its own repository unconditionally — no
`PYTHONPATH` or `uv run` required. It can be invoked from any working directory:

```sh
# Tier 1 only — free, takes ~5 s
python3 scripts/verify-harness-policy --free-only

# Tier 2 only — requires API credentials
python3 scripts/verify-harness-policy --paid-only

# Full suite
python3 scripts/verify-harness-policy

# Single cell
python3 scripts/verify-harness-policy --only C4-claude-tool-names

# Cells for one harness
python3 scripts/verify-harness-policy --harness codex

# Machine-readable JSON (stdout is pure JSON; human log goes to stderr)
python3 scripts/verify-harness-policy --free-only --json

# Keep temp dirs for debugging
python3 scripts/verify-harness-policy --keep --only C6-opencode-resolver

# From any working directory by absolute path
python3 /path/to/toolbox-harness/scripts/verify-harness-policy --free-only
```

Exit code is **0** when all cells are PASS or SKIP, **1** on any FAIL, **2** on import
assertion failure (wrong `agent_run.py` resolved despite the `sys.path` setup).

The header line `agent_run: /path/to/toolbox-harness/src/toolbox/agent_run.py` names
the exact source file validated. The `--json` output includes it as `"agent_run_file"`.
When this script reports PASS, the record shows which source it ran against.

Sample header:

```
verify-harness-policy
cells: C1-codex-key-exists, ...
keep=False
agent_run: /Users/marcio/git/toolbox-harness/src/toolbox/agent_run.py
```

## Claude tool-name check — why it's free

The `C4-claude-tool-names` cell exploits a property of claude's `--disallowedTools`
flag: when a value is passed as a single space-separated string, claude splits it on
spaces and validates each word as a known tool name **before making any model call**.
The process exits in ~0.3 s with `rc=1` ("Input must be provided") regardless of
whether tool names are valid.

Evidence:
- Real policy tools (`EnterPlanMode ExitPlanMode AskUserQuestion`): no warning, 0.3 s.
- Bogus names (`BOGUSVERIFY SENTINELA SENTINELB`): "matches no known tool" in stderr, 0.3 s.

**Names containing underscores do not trigger the warning.** Claude appears to treat
underscore-qualified names as namespace references and skips validation for them. Bogus
names in C4 and C5 therefore use plain uppercase without underscores.

This approach was determined empirically against claude 2.1.207. If a future claude
version changes the validation timing (e.g. moves it post-model-call), C4's sub-second
run time will spike and the FAIL message will say the negative control stopped working.

## Hermetic operation

The script never mutates:
- `~/.claude.json`
- `~/.config/opencode/` (XDG_CONFIG_HOME is redirected in C6 scenarios)
- `~/.codex/`

All temp dirs land under `/var/tmp/vhp-XXXXXXXX/` and are removed after a PASS unless
`--keep` is given. A FAIL preserves the cell's temp dir automatically.

## Prerequisites

All binaries must be in `PATH`:
- `codex` — for C1, C2, C3, A1
- `claude` — for C4, C5, A3, A5
- `opencode` — for C6, A2, A5

Missing binaries are reported as **SKIP**, never as PASS or FAIL.

### Model credentials for Tier 2

- **codex**: `~/.codex/auth.json` with `OPENAI_API_KEY`; `~/.codex/config.toml` with provider config.
- **claude**: `~/.claude/settings.json` with `ANTHROPIC_BASE_URL` (or direct API key).
- **opencode**: `~/.config/opencode/opencode.json` with the `llmproxy-anthropic` provider.

## Environment variables

| Variable | Default | Meaning |
|----------|---------|---------|
| `CLAUDE_MODEL` | `claude-haiku-4-5` | Model for claude Tier-2 cells |
| `OPENCODE_MODEL` | `llmproxy-anthropic/claude-haiku-4.5` | Model for opencode Tier-2 cells |
| `CODEX_MODEL` | `gpt-5.4` | Model for codex Tier-2 cells |
| `CODEX_PROVIDER` | `llmproxy` | Provider for codex Tier-2 cells |
| `VERIFY_POLICY_LIVE_TIMEOUT` | `120` | Per-subprocess timeout for Tier-2 cells (seconds) |

## JSON output

`--json` writes a machine-readable summary to stdout:

```json
{
  "pass": 6,
  "fail": 0,
  "skip": 0,
  "cells": [
    {
      "name": "C1-codex-key-exists",
      "tier": "contract",
      "verdict": "PASS",
      "failure_detail": null,
      "notes": ["real key accepted; bogus key rejected…"],
      "version_info": {"codex": "codex-cli 0.144.1"}
    }
  ]
}
```

`version_info` is the key reason to use `--json` in CI: it records which harness versions
the policy was verified against.

## What each failure means

| Cell | Failure | What broke |
|------|---------|-----------|
| C1 | "key may have been renamed" | `tools.experimental_request_user_input` no longer exists in codex config |
| C1 | "bogus key: expected non-zero" | `--strict-config` no longer enforces unknown keys |
| C2 | "bare boolean was accepted" | The struct-literal requirement was relaxed — `_codex_policy_args()` needs updating |
| C3 | "invalid-first/valid-last exited non-zero" | last-occurrence-wins semantics changed |
| C4 | "produced 'matches no known tool'" | One of the three policy tools was renamed — the deny is silently broken |
| C4 | "bogus names did not produce warning" | Tool-name validation moved or changed — negative control is broken |
| C5 | "token not found in stderr" | `--disallowedTools` became last-wins instead of additive |
| C6 | "policy mismatch: question: last rule is 'allow'" | OpenCode's merge order changed — deny no longer wins |
| A1 | "disabled: differential present" | Questions policy not reaching the app-server |
| A2 | "bash was invoked despite deny" | `OPENCODE_CONFIG_CONTENT` deny no longer overrides project allow |
| A3 | "Bash found in tool_names despite deny" | `--disallowedTools` no longer blocking the named tool |
| A5 | "bash not invoked when question=allow" | Policy merge corrupted the bash allow |

## Limitations

Known gaps that are NOT fixed here (see commit message):

- **Agent discovery narrower than OpenCode's**: no `opencode.jsonc`, no `.opencode/agent/*.md`,
  no parent-directory config. An agent defined only that way gets no deny block unless explicitly
  named via `--agent-mode` or `--harness-arg --agent`.

- **`--harness-arg --permission-mode plan`** starts claude directly in plan mode; the policy
  denies the transition tools (`EnterPlanMode`/`ExitPlanMode`), not the mode itself.

- **A3 live tripwire**: the A3 test uses `Bash` as a proxy for the policy tools
  (`EnterPlanMode`/`AskUserQuestion`) because those tools are not available in this workspace
  configuration (claude 2.1.207). If they become available, the full deny-vs-honoured-allow
  proof must be added. The existing tripwire
  (`test_a3_enterplanmode_askuserquestion_skip_reason`) detects this automatically and will fail.

- **C4/C5 underscore behaviour**: tool names with underscores do not trigger claude's
  "matches no known tool" warning. This is why bogus-name checks in C4 and C5 use
  plain uppercase without underscores. If a future policy tool has an underscore in its name,
  the C4 positive check still works (no warning = known), but a rename that adds an underscore
  would silently pass C4's positive arm even if the tool is unknown. This is a structural
  limitation of the zero-cost approach.
