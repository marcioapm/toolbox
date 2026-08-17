# VERIFY-RESULTS

## Codex sub-agent production verification

Run against the production llmproxy before the separate proxy fix that ingests
`x-codex-parent-thread-id`, macmini, 2026-08-17.

## Run command

```sh
python3 scripts/verify-session-attribution --only codex-subagent
```

## Result

The command exited `1`, as required while production stores Codex child rows as roots.
It was a real execution and failed at `link5-parent-linkage`, not SKIP.

```text
verify-session-attribution
cells: codex-subagent
DB: ssh vibes → ssh llmproxy → psql -U llmproxy -d llmproxy
keep=False skip-subagents=False

Checking DB path (preflight SELECT 1)…
DB preflight OK

============================================================
CELL: codex-subagent  (run=vsa-codex_subagent-24e8479f)
============================================================
  Launching codex sub-agent → run='vsa-codex_subagent-24e8479f'
  [link1] PASS status=done
  [link2] PASS session_id='01a0106d-6107-7e83-abad-4147e5859f63' confidence=certain acquisition=minted
  [link3] PASS disk: /Users/marcio/.codex/sessions/2026/08/17/rollout-2026-08-17T16-53-18-01a0106d-6107-7e83-abad-4147e5859f63.jsonl
  [link4] PASS DB: client='codex' created_at='2026-08-17 15:53:22.421067+00'
  [link3-child] PASS rollout: /Users/marcio/.codex/sessions/2026/08/17/rollout-2026-08-17T16-53-26-01a0106d-7ddd-76c2-b714-9b0f1a7b06ad.jsonl: parent_thread_id='01a0106d-6107-7e83-abad-4147e5859f63'
  [link5+] PASS sub-agent reply 'SUBPONG' confirmed in log
  Polling DB for child row (parent='01a0106d-6107-7e83-abad-4147e5859f63', timeout=15.0s)…

  → FAIL: codex-subagent
     FAILED at [link5-parent-linkage]: no child client_sessions row with parent_client_session_id='01a0106d-6107-7e83-abad-4147e5859f63' after 15.0s — sub-agent did not fire, or parent link broken

============================================================
SUMMARY
============================================================
  ✗ FAIL  codex-subagent  [link5-parent-linkage]

  PASS=0  FAIL=1  SKIP=0
```

## Interpretation

The managed root completed, recorded a certain minted session, appeared in its own Codex
rollout, and was attributed as `client='codex'` in production. Codex also created a child
rollout whose `parent_thread_id` resolves to that root. The missing production DB child
link therefore isolates the current defect to llm-proxy's Codex parent-header ingestion.

The verifier cleaned the managed run; no `vsa-*` run, Codex app-server, or capture proxy was
left running. Re-run this cell after the proxy deployment and require PASS at link 5.
