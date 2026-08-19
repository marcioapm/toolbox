# Acceptance evidence — agent-run session id ↔ proxy csid

Captured live on macmini 2026-08-16 with a local HTTP header-capture listener standing in for
llmproxy (`/tmp/hdrcap.py`, records every inbound request's headers, replies 500).

Purpose: prove the id `agent-run` records in `session.json` is **the same string** the harness
sends to the proxy as its client session id (csid), for all three harnesses.

Proxy side reads (`crates/llm-proxy-core/src/proxy/session.rs`):
- opencode/generic: csid = `x-session-id`, parent = `x-parent-session-id`
- claude-code: csid = `x-claude-code-agent-id` ?? `x-claude-code-session-id`,
  parent = `x-claude-code-parent-agent-id` ?? `x-claude-code-session-id`
- codex: csid = `x-client-request-id` (only when UA/originator looks like codex),
  parent = validated `x-codex-parent-thread-id` when present

---

## 1. claude-code — MATCH (root)

Launch: `claude --settings <cap> --session-id 0a016103-af23-41e2-8f99-24310dab90a6 --print "hi"`
Redirect: `ANTHROPIC_BASE_URL=http://127.0.0.1:47412` via `--settings` file (the `env` block;
a bare `ANTHROPIC_BASE_URL=` env var is overridden by `~/.claude/settings.json`).

Captured on `/v1/messages?beta=true`, 8 requests:

    x-claude-code-session-id = 0a016103-af23-41e2-8f99-24310dab90a6
    user-agent               = claude-cli/2.1.207 (external, sdk-cli)

**Pushed id == header id.** Root csid confirmed deterministic end to end.

⚠️ `--session-id` is **rejected if the id already exists**: `Error: Session ID <uuid> is already
in use.` agent-run mints a fresh UUID4 per run so this is fine, but a retry/resume path must not
reuse one.

## 2. opencode — MATCH (root)

Minted via HTTP, exactly the settled design: `POST /session` → `ses_ff43b3152ffeFcIW4MkxPkwF38`,
then drove that session with `POST /session/<id>/message`.

Captured on `/v1/messages`, 6 requests:

    x-session-id       = ses_ff43b3152ffeFcIW4MkxPkwF38
    x-session-affinity = ses_ff43b3152ffeFcIW4MkxPkwF38
    user-agent         = opencode/1.18.18 ai-sdk/provider-utils/4.0.27 runtime/bun/1.3.14

**Minted id == header id.** Root csid confirmed deterministic end to end.

Setup note: the custom provider must be in `opencode.json` **in the server's cwd** (or global
config). A provider defined only in a file elsewhere yields `UnknownError ... err_b079af9f` on
`POST /session/<id>/message`.

## 3. codex — MATCH (root)

Launch: `codex exec --json -c model_providers.llmproxy.base_url="http://127.0.0.1:47411/v1"`.
`OPENAI_API_KEY` must be set in the environment or codex fails before any request.

stdout first line: `{"type":"thread.started","thread_id":"01a00bc1-3d42-7131-a784-950cee0044c4"}`

Captured on `/v1/responses`, 15 requests:

    x-client-request-id = 01a00bc1-3d42-7131-a784-950cee0044c4
    session-id          = 01a00bc1-3d42-7131-a784-950cee0044c4
    thread-id           = 01a00bc1-3d42-7131-a784-950cee0044c4
    originator          = codex_exec
    user-agent          = codex_exec/0.144.1 (Mac OS 26.5.1; arm64) ...

**`thread.started.thread_id` == `x-client-request-id`** — the exact header the proxy reads for
codex. Root csid confirmed deterministic end to end.

Note: root codex requests send the same value under three header names and no parent header.

---

## 4. Sub-agent parent linkage — VERIFIED for opencode and claude-code

Acceptance for sub-agents is *reachability*, not string equality (Márcio, 2026-08-16): agent-run
records the root only, and each sub-agent must send a parent id that resolves back to it.

The 500-replying capture listener could not test this — no model call ever completed, so no
sub-agent could spawn. Replaced with a **recording forward proxy** (`fwdcap.py`: logs headers,
forwards to the real llmproxy, streams the response back, redacts credential headers), so real
sub-agents actually run.

### opencode — parent header correct, no changes needed

Root minted via `POST /session` = `ses_ff42d1902ffe5qhWQUEpyVXker`. Prompt instructed the build
agent to spawn an `explore` sub-agent. Five requests captured on the forward proxy:

    /v1/messages  sid=ses_ff42d1902ffe5qhWQUEpyVXker  parent=None                                (root)
    /v1/messages  sid=ses_ff42cf1a7ffegu7k6HQL8UaFPY  parent=ses_ff42d1902ffe5qhWQUEpyVXker      (sub-agent)
    /v1/messages  sid=ses_ff42cf1a7ffegu7k6HQL8UaFPY  parent=ses_ff42d1902ffe5qhWQUEpyVXker      (sub-agent)
    /v1/messages  sid=ses_ff42d1902ffe5qhWQUEpyVXker  parent=None                                (root resumes)

`x-parent-session-id` on every sub-agent request is exactly the minted root. Confirmed against
the db (read-only):

    ses_ff42d1902ffe5qhWQUEpyVXker  parent_id=            agent=build    title=subagent-parent-test
    ses_ff42cf1a7ffegu7k6HQL8UaFPY  parent_id=ses_ff42d…  agent=explore  title=Read sample.txt contents (@explore subagent)

### claude-code — reachable, but via the `??` fallback, not a parent header

Root pushed with `--session-id 765ada12-72d9-4151-acf2-06fdf120d2ea`. Task sub-agent spawned and
completed. Five requests captured:

    /v1/messages  sess=765ada12-…  agent=None               parent=None   (root)
    /v1/messages  sess=765ada12-…  agent=a2e5caae509437e4b  parent=None   (sub-agent)
    /v1/messages  sess=765ada12-…  agent=a2e5caae509437e4b  parent=None   (sub-agent)
    /v1/messages  sess=765ada12-…  agent=None               parent=None   (root resumes)

🔴 **`x-claude-code-parent-agent-id` is NOT sent** for a depth-1 sub-agent — only `agent-id` and
`session-id`. Per `01-header-contract.md` the sub-agent's csid is the **agent-id**
(`a2e5caae509437e4b`), so its parent comes from the documented fallback in
`claude_code_session_info` (`session.rs:487-493`):

    parent = x-claude-code-parent-agent-id ?? x-claude-code-session-id

which yields `765ada12-…` — the id agent-run recorded. **The `??` fallback is load-bearing and
now has live evidence**, not just design intent. Remove it and every depth-1 claude sub-agent
detaches from its root.

### codex — prompt-triggered sub-agent and parent header captured

A managed Codex root was prompted: `Spawn a subagent and have it reply with exactly: PONG`.
It created these rollouts:

```
root   = 01a01061-5cd3-70e3-ba36-8fbc18140c1a
child  = 01a01061-676c-71a0-99ad-1071b1abe10c
```

The child's rollout `session_meta.payload` records:

```
parent_thread_id = 01a01061-5cd3-70e3-ba36-8fbc18140c1a
agent_nickname   = Ramanujan
```

The recording forward proxy captured the child's `/v1/responses` request:

```
originator                = codex_exec
x-client-request-id       = 01a01061-676c-71a0-99ad-1071b1abe10c
x-codex-parent-thread-id  = 01a01061-5cd3-70e3-ba36-8fbc18140c1a
x-openai-subagent         = collab_spawn
thread-id                 = 01a01061-676c-71a0-99ad-1071b1abe10c
session-id                = 01a01061-5cd3-70e3-ba36-8fbc18140c1a
```

This proves both a reliable prompt-triggered child and the wire parent link. Before the
llm-proxy fix, `codex_session_info()` hardcodes the parent to `None`, so production stores
both rows as roots. The proxy must ingest `x-codex-parent-thread-id` through its existing
validated `header_id()` path; until that deployment, the real `codex-subagent` verification
correctly fails at `link5-parent-linkage`, rather than SKIP.

### Conclusion

OpenCode and Claude parent attribution are verified as above. Codex also emits the parent
link; the remaining production defect is proxy ingestion of the already-present
`x-codex-parent-thread-id`, not client delegation or `agent-run` session acquisition.

---

## 5. codex — `app-server` mints ids up front (supersedes `--json` stdout parsing)

Investigated 2026-08-16 after the `--json` stdout-pollution concern. **A mint-then-attach
mechanism does exist for codex**, structurally identical to opencode's.

### Why `--json` was the wrong mechanism

Measured on the same prompt (`Reply with exactly: PONG`):

- **without `--json`** — stdout = `PONG` (5 bytes, 1 line); banner/workdir/model/token-count all
  on **stderr** (18 lines), including `session id: 01a00bd6-f973-75e2-a882-159af9a70210`
- **with `--json`** — stdout = 4 JSONL events; stderr = one line

So `--json` is not pre-existing noise: it *replaces* the human-readable run log with an event
stream, degrading `clean`/`tail`/`logs` for every managed codex run.

### `codex app-server` — pushed/minted id, verified

`codex app-server` speaks JSON-RPC over stdio. Method names are **lowercase** on the wire
(`thread/start`; the generated schema's `Thread/startRequest` names are Rust variants, and
`Thread/start` is rejected with `unknown variant`). Schema dump: `codex app-server
generate-json-schema --out <dir>` (39 files, 516 v2 defs).

    -> {"jsonrpc":"2.0","id":1,"method":"initialize","params":{"clientInfo":{...}}}
    -> {"jsonrpc":"2.0","method":"initialized","params":{}}
    -> {"jsonrpc":"2.0","id":2,"method":"thread/start","params":{"cwd":"/tmp/cxprobe"}}
    <- {"thread":{"id":"01a00bf2-e26b-74c3-8e46-0c6dd9637ce0",
                  "sessionId":"01a00bf2-e26b-74c3-8e46-0c6dd9637ce0",
                  "path":"~/.codex/sessions/2026/08/16/rollout-…-01a00bf2-e26b-….jsonl",
                  "parentThreadId":null,"forkedFromId":null,…}}

The id is returned **before any prompt is sent**, and the rollout path is handed back with it.
A subsequent `turn/start` produced `PONG`, and the forward-capture proxy recorded:

    /v1/responses   x-client-request-id = 01a00bf2-e26b-74c3-8e46-0c6dd9637ce0
                    session-id          = 01a00bf2-e26b-74c3-8e46-0c6dd9637ce0
                    thread-id           = 01a00bf2-e26b-74c3-8e46-0c6dd9637ce0

**Minted id == the header the proxy reads for codex.** `acquisition: "minted"`,
`confidence: "certain"`, stdout untouched, no `--json`, no stderr scraping.

🔴 **`clientInfo.name` sets `originator` AND the User-Agent.** Probing with
`clientInfo.name = "agent-run-probe"` produced `originator: agent-run-probe` and
`user-agent: agent-run-probe/0.144.1 …`. The proxy only reads `x-client-request-id` when the
request *looks like codex* (`identify_client`: UA contains `codex`/`codex_exec`, or
`originator: codex_exec` — `session.rs:526,552`). So a naively-named app-server client is
**silently unattributed**. Re-probed with `clientInfo.name = "codex_exec"`: `originator` and UA
both restored. Any app-server integration must set this deliberately.

Other app-server surface worth knowing: `thread/resume` (by `threadId`), `thread/fork`,
`turn/start`, `turn/steer` (has `expectedTurnId` precondition — a real steer mechanism),
`turn/interrupt`. `ThreadStartParams` accepts `cwd`, `model`, `modelProvider`, `sandbox`,
`approvalPolicy`, `baseInstructions`, `developerInstructions`, `ephemeral`.

## 6. claude-code `x-claude-code-parent-agent-id` is REAL, not hallucinated

Checked directly in the shipped binary
(`@anthropic-ai/claude-code-darwin-arm64/claude`). The header is constructed conditionally:

    ...c?.agentId && {"x-claude-code-agent-id": bhi(c.agentId)},
    ...c?.parentAgentId && {"x-claude-code-parent-agent-id": bhi(c.parentAgentId)}

and the context object only carries `parentAgentId` when one exists:

    let a = {agentId: e.agentId, parentSessionId: e.parentSessionId, agentType: e.agentType};
    if (e.parentAgentId) a.parentAgentId = e.parentAgentId;

So the header exists in the client and is emitted **only when the sub-agent itself has a parent
agent** — i.e. depth ≥ 2. The depth-1 capture in §4 (header absent) is consistent with this, not
evidence of absence from the product.

⚠️ Consequence: deleting the `x-claude-code-parent-agent-id` branch from
`claude_code_session_info` would make every depth-2+ sub-agent report its **root** as parent
instead of its immediate parent, silently flattening the tree. The existing
`parent-agent-id ?? session-id` order already yields the correct answer at both depths and costs
one header lookup. Recommend keeping it.
