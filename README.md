# toolbox

A collection of lightweight CLI tools for AI content generation and chat operations. Zero dependencies beyond Python 3.10+ stdlib.

## Tools

| CLI | Description | Auth |
|-----|-------------|------|
| `gemini-image` | Generate images via Imagen 4.0 / Gemini native models | `GEMINI_API_KEY` |
| `gemini-tts` | Text-to-speech via Gemini native audio | `GEMINI_API_KEY` |
| `gemini-transcribe` | Transcribe audio files via Gemini | `GEMINI_API_KEY` |
| `gemini-video` | Generate video via Google Veo 2/3/3.1 | `GEMINI_API_KEY` |
| `gemini-vision` | Analyze images/videos via Gemini (supports YouTube, Instagram, TikTok) | `GEMINI_API_KEY` |
| `slackcli` | Lightweight Slack client (channels, messages, search, reactions) | `SLACK_USER_TOKEN` |
| `llm-usage` | Monitor LLM token usage, costs, and quotas across providers | `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY` |
| `agent-run` | Background wrapper for coding agents (Claude Code, OpenCode, Codex) with PTY steering, interactive attach, live log streaming, and managed mode (`--harness`) for deterministic session-id capture | — |

## Install

```bash
# One-liner (pip)
pip install git+https://github.com/marcioapm/toolbox.git

# Or clone and install in editable mode
git clone https://github.com/marcioapm/toolbox.git
cd toolbox
pip install -e .

# Or use the install script
curl -sSL https://raw.githubusercontent.com/marcioapm/toolbox/main/install.sh | bash
```

## Setup

Set your API keys as environment variables:

```bash
# Gemini API key (get one at https://aistudio.google.com/apikey)
export GEMINI_API_KEY="your-key-here"

# Slack user token (get one at https://api.slack.com/apps → OAuth & Permissions)
export SLACK_USER_TOKEN="xoxp-your-token-here"
```

Add them to your shell profile (`~/.bashrc`, `~/.zshrc`, etc.) for persistence.

---

## gemini-image

Generate images using Google's Imagen 4.0 or Gemini native image models.

### Models

| Model | Speed | Quality | Notes |
|-------|-------|---------|-------|
| `imagen-4.0-generate-001` | Medium | Best | Default, production-ready |
| `imagen-4.0-ultra-generate-001` | Slow | Highest | Maximum quality |
| `imagen-4.0-fast-generate-001` | Fast | Good | Quick iterations |
| `nano-banana-pro-preview` | Medium | Good | Gemini native |
| `gemini-3-pro-image-preview` | Medium | Good | Gemini 3 Pro |
| `gemini-3.1-flash-image-preview` | Fast | OK | Fastest native |

### Usage

```bash
# Basic generation
gemini-image "a cat riding a skateboard"

# Custom output and model
gemini-image "corporate logo, minimal" -o logo.png -m imagen-4.0-fast-generate-001

# Multiple images
gemini-image "abstract art" -n 4 -o art.png
# Saves: art.png, art-2.png, art-3.png, art-4.png

# Custom aspect ratio
gemini-image "landscape photo" --aspect 16:9 -o wide.png

# Using Gemini native model
gemini-image "watercolor painting of a forest" -m gemini-3-pro-image-preview
```

### Options

```
positional:
  prompt              Image generation prompt

options:
  -o, --output FILE   Output file (default: output.png)
  -m, --model MODEL   Model to use (default: imagen-4.0-generate-001)
  -n, --count N       Number of images, 1-4 (default: 1)
  --aspect RATIO      Aspect ratio (default: 1:1)
```

---

## gemini-tts

Text-to-speech using Gemini's native audio generation.

### Models & Voices

**Models:**
| Model | Speed | Quality |
|-------|-------|---------|
| `gemini-2.5-flash-preview-tts` | Fast | Good (default) |
| `gemini-2.5-pro-preview-tts` | Slower | More expressive |

**Voices:**
| Voice | Character |
|-------|-----------|
| Kore | Default, neutral |
| Aoede | Deep, expressive |
| Charon | Deep, authoritative |
| Fenrir | Strong, bold |
| Puck | Light, playful |
| Orbit | Calm, measured |
| Vale | Warm, gentle |

### Usage

```bash
# Basic TTS
gemini-tts "Hello, world!" -o hello.wav

# Choose voice and model
gemini-tts "Breaking news from the tech world" -v Charon -m gemini-2.5-pro-preview-tts

# Expressive voice for storytelling
gemini-tts "Once upon a time in a land far away..." -v Aoede -m gemini-2.5-pro-preview-tts -o story.wav

# Quick announcement
gemini-tts "Your build has completed successfully" -v Puck
```

### Options

```
positional:
  text                Text to speak

options:
  -o, --output FILE   Output WAV file (default: output.wav)
  -m, --model MODEL   TTS model (default: gemini-2.5-flash-preview-tts)
  -v, --voice VOICE   Voice name (default: Kore)
```

---

## gemini-transcribe

Transcribe audio files via the Gemini API.

### Models

| Model | Speed | Notes |
|-------|-------|-------|
| `gemini-2.5-flash` | Fast | Default, cheap |
| `gemini-2.5-pro` | Slower | More accurate, pricier |

Model names are accepted as free-form strings, so any new Gemini model can be passed via `-m`.

### Usage

```bash
# Basic transcription (prints transcript to stdout)
gemini-transcribe meeting.mp3

# Save transcript to a file
gemini-transcribe call.ogg -o transcript.txt

# Use the more accurate model with a language hint
gemini-transcribe lecture.wav -m gemini-2.5-pro --language Portuguese

# Custom prompt (e.g. add speaker labels)
gemini-transcribe interview.m4a --prompt "Transcribe with speaker labels (Speaker A, Speaker B)."

# Get full JSON response instead of just text
gemini-transcribe note.opus --json
```

### Options

```
positional:
  audio_path           Path to audio file (.ogg/.opus, .mp3, .wav, .m4a, .flac, .aac, .webm)

options:
  -o, --output FILE    Write transcript to file (default: stdout)
  -m, --model MODEL    Gemini model (default: gemini-2.5-flash)
  --prompt TEXT        Custom transcription prompt
  --language TEXT      Optional language hint, e.g. "Portuguese"
  --json               Output the full JSON response
  --api-key TEXT       Gemini API key [env: GEMINI_API_KEY]
```

**Note:** Files larger than ~19 MB are rejected (Gemini's `inline_data` limit is 20 MB). Use the Files API for larger audio.

---

## gemini-video

Generate videos using Google's Veo models. Submits an async job and polls until completion.

### Models

| Model | Speed | Quality | Notes |
|-------|-------|---------|-------|
| `veo-3.0-fast-generate-001` | Fast | Good | Default |
| `veo-3.0-generate-001` | Slow | High | Best Veo 3 |
| `veo-3.1-fast-generate-preview` | Fast | Good | Latest fast |
| `veo-3.1-generate-preview` | Slow | Highest | Latest quality |
| `veo-2.0-generate-001` | Medium | OK | Older model |

### Usage

```bash
# Basic video generation
gemini-video "a drone flying over mountains at sunset"

# High quality with specific model
gemini-video "time-lapse of a flower blooming" -m veo-3.0-generate-001 -o flower.mp4

# Vertical video (e.g., for mobile/social)
gemini-video "person walking through a neon-lit city" --aspect 9:16 -o vertical.mp4

# Quick draft
gemini-video "ocean waves crashing on rocks" -m veo-3.0-fast-generate-001
```

### Options

```
positional:
  prompt              Video generation prompt

options:
  -o, --output FILE   Output file (default: output.mp4)
  -m, --model MODEL   Model (default: veo-3.0-fast-generate-001)
  --aspect RATIO      Aspect ratio (default: 16:9)
```

**Note:** Video generation is async. The CLI submits the job and polls every 5 seconds. Typical generation takes 1-5 minutes depending on the model.

---

## slackcli

Lightweight Slack CLI that uses a user token to act as you (not a bot).

### Usage

```bash
# List channels
slackcli channels
slackcli ch -n 50

# Read history
slackcli history C02DLS4PFH7
slackcli h C02DLS4PFH7 -n 30

# Send a message
slackcli send C02DLS4PFH7 "Hello from the CLI!"
slackcli s C02DLS4PFH7 "Quick update: deploy complete"

# Reply in a thread
slackcli reply C02DLS4PFH7 1710430020.123456 "Thread reply here"

# Search messages
slackcli search "deployment failed" -n 5
slackcli search "from:@alice bug report"

# List users
slackcli users
slackcli userinfo U01234ABCDE

# Get DM channel ID
slackcli dm U01234ABCDE

# Check unread messages
slackcli unread

# React to a message
slackcli react C02DLS4PFH7 1710430020.123456 thumbsup
slackcli unreact C02DLS4PFH7 1710430020.123456 thumbsup
```

### Commands

| Command | Alias | Description |
|---------|-------|-------------|
| `channels` | `ch` | List channels (public, private, DMs) |
| `history` | `h` | Read channel message history |
| `send` | `s` | Send a message to a channel |
| `reply` | `r` | Reply in a thread |
| `search` | — | Search messages across workspace |
| `users` | `u` | List workspace members |
| `userinfo` | `ui` | Show user details (name, email, timezone) |
| `dm` | — | Get or create a DM channel ID |
| `unread` | — | Show channels with unread messages |
| `react` | — | Add an emoji reaction |
| `unreact` | — | Remove an emoji reaction |

### Token Scopes

Your `SLACK_USER_TOKEN` needs these scopes:
- `channels:read`, `channels:history` — Read public channels
- `groups:read`, `groups:history` — Read private channels
- `im:read`, `im:history` — Read DMs
- `chat:write` — Send messages
- `search:read` — Search messages
- `users:read` — List/view users
- `reactions:write` — Add/remove reactions

---

---

## gemini-vision

Analyze images and videos using Gemini's multimodal capabilities. Auto-downloads videos from YouTube, Instagram, TikTok, X/Twitter, Vimeo, and more via yt-dlp.

### Usage

```bash
# Analyze a local image
gemini-vision photo.jpg

# Describe with custom prompt
gemini-vision screenshot.png -p "What's the error in this screenshot?"

# Transcribe speech from a video
gemini-vision video.mp4 -p "Transcribe all speech in this video"

# Analyze YouTube video
gemini-vision "https://youtube.com/watch?v=dQw4w9WgXcQ" -p "Summarize this video"

# Instagram reel
gemini-vision "https://instagram.com/reel/ABC123/" -p "Describe what happens"

# TikTok / X post
gemini-vision "https://tiktok.com/@user/video/123" -p "What's in this video?"
gemini-vision "https://x.com/user/status/123" -p "Describe the video"

# Use a different model
gemini-vision photo.jpg -m gemini-2.5-pro -p "Detailed art analysis"

# Keep the downloaded video file
gemini-vision "https://youtube.com/watch?v=..." --keep
```

### Supported platforms

YouTube, Instagram, TikTok, X/Twitter, Vimeo, Facebook, Reddit — anything yt-dlp supports.

### Options

```
positional:
  file                  Image/video path, URL, or social media link

options:
  -p, --prompt TEXT     Analysis prompt  [default: Describe what you see in detail.]
  -m, --model [...]     Gemini model  [default: gemini-2.5-flash]
  --keep                Keep downloaded video (don't delete temp file)
  --api-key TEXT        Gemini API key [env: GEMINI_API_KEY]
```

### Requirements

- `yt-dlp` for social media downloads: `brew install yt-dlp`

---

## llm-usage

Monitor LLM token usage and quotas across Anthropic, OpenAI, and Google Gemini.

### What it checks

| Provider | Rate limits | Token usage | Cost |
|----------|------------|-------------|------|
| Anthropic | ✅ via response headers | ✅ via OpenClaw logs | ✅ |
| OpenAI | ✅ via response headers | ✅ org API (needs admin key) + OpenClaw logs | ✅ |
| Gemini | — (no API) | ✅ via OpenClaw logs | ✅ |

### Usage

```bash
# Check all providers
llm-usage

# Check specific provider
llm-usage -p anthropic
llm-usage -p openai

# JSON output (for scripts/agents)
llm-usage --json

# Skip OpenClaw local stats
llm-usage --no-openclaw
```

### Example output

```
🔍 LLM Usage Monitor
   2026-03-15 17:39

✅ ANTHROPIC
  Status: ok
  Rate limits:
    requests-limit: 4000
    requests-remaining: 3999
    tokens-limit: 400000
    tokens-remaining: 399990

✅ OPENAI
  Status: ok

✅ GEMINI
  Status: ok
  Available models: 45

📊 OPENCLAW LOCAL USAGE
  Today:
    anthropic: 121.2M tokens (925 in / 192.9K out) · cache: 112.9M read / 8.2M write · $112.39 · 728 reqs
    openai-codex: 7.8M tokens (3.4M in / 5.7K out) · cache: 4.3M read / 0 write · $6.80 · 42 reqs
  Week:
    anthropic: 1298.7M tokens (12.7K in / 2.4M out) · cache: 1190.8M read / 105.5M write · $1314.18 · 11751 reqs
    google: 33.6M tokens (33.5M in / 63.0K out) · $73.59 · 451 reqs
    openai-codex: 83.2M tokens (17.2M in / 146.4K out) · $43.62 · 794 reqs
```

### Options

```
options:
  -p, --provider [anthropic|openai|gemini|all]  Provider(s) to check  [default: all]
  --openclaw / --no-openclaw    Include OpenClaw local session usage  [default: openclaw]
  --json                        Output as JSON
  --anthropic-api-key TEXT      Anthropic API key [env: ANTHROPIC_API_KEY]
  --openai-api-key TEXT         OpenAI API key [env: OPENAI_API_KEY]
  --openai-admin-key TEXT       OpenAI admin key for usage API [env: OPENAI_ADMIN_KEY]
  --gemini-api-key TEXT         Gemini API key [env: GEMINI_API_KEY]
```

---

## threadctl

`threadctl` is **not shipped by this package**. It now lives in its own
repo (`marcioapm/threadctl`) — the toolbox copy was a stale fork that was
missing live subcommands and got installed over the real binary on a
production host, so it was deleted here rather than re-synced.

---

## For LLMs / AI Agents

All tools follow the same patterns:

1. **Auth via environment variables** — set `GEMINI_API_KEY` and/or `SLACK_USER_TOKEN`
2. **Positional argument for main input** — prompt text, search query, etc.
3. **Flags for options** — `-o` output, `-m` model, `-n` count, `-v` voice
4. **Exit codes** — 0 = success, 1 = error (with stderr message)
5. **Human-readable stdout** — file paths, message timestamps, channel IDs
6. **No interactive prompts** — everything is flags/args, suitable for scripting

### Quick reference for agents

```bash
# Image: generate → save to file
gemini-image "prompt" -o /tmp/out.png -m imagen-4.0-fast-generate-001

# TTS: text → WAV file
gemini-tts "text to speak" -o /tmp/speech.wav -v Aoede

# Transcribe: audio file → text
gemini-transcribe meeting.mp3 -o /tmp/transcript.txt

# Video: prompt → MP4 (takes minutes, async polling)
gemini-video "prompt" -o /tmp/video.mp4

# Vision: analyze images/videos (YouTube, Instagram, etc.)
gemini-vision photo.jpg -p "What's in this image?"
gemini-vision "https://youtube.com/watch?v=..." -p "Summarize this video"
gemini-vision video.mp4 -p "Transcribe the speech"

# Slack: read unread → send reply
slackcli unread
slackcli history CHANNEL_ID -n 10
slackcli send CHANNEL_ID "message"
slackcli react CHANNEL_ID TIMESTAMP emoji_name

# Usage: check token spending across providers
llm-usage
llm-usage --json
llm-usage -p anthropic

# Agent-run: background coding agents with steering + live logs
agent-run build claude --permission-mode bypassPermissions --print 'Refactor X'
agent-run tail build                      # follow logs in real time
agent-run status build                    # running | done | failed | died | killed
agent-run -i chat claude --permission-mode bypassPermissions
agent-run steer chat 'Also add tests for edge cases.'
agent-run kill chat                       # TERM the run; runner does its own cleanup
agent-run reap --dry-run                  # preview idle-kills + terminal-state cleanup

# Managed mode: agent-run builds the command, records the session id
agent-run --harness claude --prompt 'Refactor X' build
agent-run --harness opencode --model llmproxy-anthropic/claude-sonnet-4.6 --prompt 'Refactor X' build
agent-run --harness codex --model o4-mini --prompt 'Refactor X' build
agent-run -i --harness claude --prompt 'Start task' chat   # interactive; steer after launch
```

---

## agent-run

Background wrapper for long-running coding agents (Claude Code, Codex, Pi, OpenCode).
Creates a run directory with structured state files you can poll safely — no
brittle process-poll loops — and adds a stdin FIFO when you need to steer an
interactive agent mid-flight.

Storage is split across two roots so a hard crash or reboot never loses a
log even though the ephemeral process state is gone:

- `/tmp/agent-runs/<name>/` — ephemeral process state (pid, status,
  exit_code, FIFO). tmpfs on Linux, wiped on reboot — a missing entry here
  unambiguously means "not running". Override with `AGENT_RUN_STATE_DIR`.
- `/var/tmp/agent-runs/<name>/` — persistent log, a
  copy of the prompt file, and a per-run scratch dir (`tmp/`, mode 0700)
  exported as `TMPDIR` and `BUN_TMPDIR` into the launched command's
  environment. Survives reboot/crash; the log fd is opened here from the
  start, so there's no copy-on-exit step a crash could lose. Override with
  `AGENT_RUN_LOG_DIR`. `log`/`prompt` are pruned automatically
  after 21 days; the `tmp/` scratch dir is instead cleaned up by
  `agent-run reap` (see below).

### Scratch dir (`TMPDIR`, `BUN_TMPDIR`)

Every launch gets its own disk-backed scratch dir at
`$AGENT_RUN_LOG_DIR/<name>/tmp/`, exported as both `TMPDIR` and
`BUN_TMPDIR` — same path, same value — into the launched command's
environment (and therefore every descendant it forks/execs). `BUN_TMPDIR`
is set because Bun (which OpenCode runs on) does not consult `TMPDIR` for
its own scratch space; without it, a Bun-based agent would keep spilling
into the shared system temp despite `TMPDIR` being redirected. The agent's
argv is never modified — only the environment carries this. This exists to
contain tools that dump large, un-cleaned scratch data into the ambient
temp dir: OpenCode's bundled JDTLS, for example, creates
`mkdtemp()`-based Eclipse workspaces (routinely hundreds of MB for a large
Java repo) and never removes them. Left to land in the system `/tmp` —
which is tmpfs (RAM) on most Linux hosts — enough leaked runs can consume
tens of GB of RAM. Routing each run's scratch space into its own directory
means reaping the run also reaps whatever it leaked.

The scratch dir is **not** deleted when the run ends — postmortem
artifacts matter for debugging a crashed or misbehaving agent. `agent-run
reap` removes it when terminal state is old enough (and independently removes
aged orphaned scratch after a reboot loses state). Relaunching the same run
name intentionally replaces the prior log directory, including its scratch.

### Launch

```bash
# Recommended: separate agent-run's own flags/name from the launch command
# with `--`. Everything after it is taken verbatim, dashes and all.
agent-run build -- claude --permission-mode bypassPermissions --print 'Build the thing'

# Interactive (steerable via stdin FIFO):
agent-run -i chat -- claude --permission-mode bypassPermissions

# Omitting `--` still works for a plain command with no leading-dash args:
agent-run build claude --permission-mode bypassPermissions --print 'Build the thing'
```

The run name must precede `--`; a bare `agent-run --` (no name yet) is
rejected with an error showing the correct shape, rather than guessing.
Everything after `--` — including a literal `--` — is passed through
unmodified and never dispatched as a subcommand: `agent-run mytask -- list
foo` launches the command `list foo`, it does not run `agent-run list`. A
flag typed after the name without `--` (e.g. `agent-run build --foo`) is
still rejected, since it would otherwise silently become `argv[0]`; the
error names the offending token and points at `--`.

#### Working directory (`--cwd DIR`)

```bash
agent-run --cwd ~/git/myrepo build -- claude --print 'Build the thing'
agent-run --cwd=../myrepo --harness claude --prompt 'Refactor X' build
```

`--cwd DIR` runs the launched command with `DIR` as its working directory,
in raw and managed mode and on both the one-shot and interactive (`-i`)
paths. `DIR` may be relative (resolved against the invocation directory) and
may start with `~` (expanded by agent-run; argparse does not do this). The
directory is entered before any run state is published, so `<state_dir>/cwd`
and `<state_dir>/launch_head` — which `watch` reads to report git facts —
record the directory the command actually runs in.

Symlinked components are collapsed: the recorded `cwd` is the real path. A
`DIR` that does not exist, is not a directory, or cannot be entered exits
non-zero and creates no run dir.

Relative paths on the command line resolve against two different directories,
deliberately. A relative *command* path (`agent-run --cwd ~/git/myrepo build --
./scripts/run.sh`) is resolved by the launched command after the chdir, so it is
relative to `DIR` — that is what makes `--cwd` useful for repo-local scripts. A
relative `-f/--prompt-file` names a file the caller typed at their shell, so it
is resolved against the invocation directory before the chdir; the recorded
`<state_dir>/prompt_file` is the resulting absolute path.

### Managed mode

`--harness claude|opencode|codex` switches to managed mode: agent-run builds
the launch command itself instead of taking a verbatim trailing command. This
lets it acquire the agent's session id deterministically — before the first
prompt goes over the wire — because it controls the invocation.

**Raw mode is unchanged and not deprecated.** `--harness` and a trailing
`-- <command>` are mutually exclusive; supplying both is an argument error.
All existing raw-mode launch forms keep working.

#### Options

```
--harness claude|opencode|codex   required; selects the harness
--prompt TEXT                     inline prompt (mutually exclusive with --prompt-file)
-f, --prompt-file PATH            read prompt from a file
-i                                interactive/steerable (stays running; accepts steer)
--model MODEL                     model string forwarded verbatim to the harness; the
                                  harness/provider must be configured to recognise it
--agent-mode NAME                 harness agent/mode name (e.g. opencode --agent build)
--permissions bypass|prompt       bypass (default): appends --permission-mode bypassPermissions
                                  or --auto; prompt: omits those flags so the harness's own
                                  permission UI is used
--harness-arg FLAG                pass FLAG verbatim after the harness's own args; repeatable.
                                  For claude and codex, values are appended after
                                  agent-run's own managed-mode args so a caller can
                                  override injected flags (codex uses last-occurrence-wins
                                  for -c). For opencode, the three managed policy keys
                                  (question, plan_enter, plan_exit) are not overridable
                                  this way; use --enable-planning / --enable-questions
                                  to relax them.
--enable-planning                 allow planning; disabled by default. Unsupported by codex,
                                  whose managed app-server path does not expose plan mode
--enable-questions                allow interactive questions; disabled by default
--cwd DIR                         working directory for the launched command (also raw mode)
```

`--enable-planning` and `--enable-questions` are managed-mode escape hatches.
Without them, agent-run disables planning and interactive questions for every
managed child process: Claude receives per-process `--disallowedTools` (the
deny arm; the enabled arm simply omits those flags and falls back to the user's
own Claude project config), OpenCode receives a merged process-local
`OPENCODE_CONFIG_CONTENT` permission policy with explicit `allow`/`deny` for
both states, and Codex app-server receives
`tools.experimental_request_user_input={enabled=false}` or `enabled=true`.
Codex's managed app-server API does not expose plan mode, so
`--enable-planning --harness codex` fails before creating run state. Raw mode
is unaffected.

**`--harness-arg` precedence:** for claude and codex, values are appended after
agent-run's injected managed-mode args so a caller can override them. Codex uses
last-occurrence-wins for `-c` keys, so a duplicate key in `--harness-arg` takes
effect. Claude's `--disallowedTools` is additive, so a caller cannot re-enable a
denied tool via `--harness-arg` regardless of ordering. For opencode, policy is
delivered via `OPENCODE_CONFIG_CONTENT` (env), not argv; `--harness-arg` passes
flags to the opencode process but cannot override the three managed policy keys
(`question`, `plan_enter`, `plan_exit`) — use `--enable-planning` or
`--enable-questions` instead.

**`--harness-arg --permission-mode` for claude:** passing `--permission-mode plan`
via `--harness-arg` is rejected when planning is disabled, because it would start
Claude directly in plan mode, bypassing the `EnterPlanMode`/`ExitPlanMode` tool
denies. Use `--enable-planning` to enable plan mode through the supported policy
path.

#### One-shot examples

```bash
# claude — sends --print, records the session id agent-run pushed as a UUID4
agent-run --harness claude --prompt 'Refactor X' build

# opencode — mints a session via POST /session before launch
agent-run --harness opencode --model llmproxy-anthropic/claude-sonnet-4.6 \
          --prompt 'Refactor X' build

# codex — mints a thread via app-server thread/start
agent-run --harness codex --model o4-mini \
          --prompt 'Refactor X' build

# prompt from a file (works with any harness)
agent-run --harness claude --prompt-file brief.md build
```

`--model` is forwarded verbatim to every harness, including codex's
`thread/start`; it is not validated against a known-model list. The harness
must be configured to recognise the string (e.g. codex's `model_provider` in
`~/.codex/config.toml`) — provider selection and auth are the operator's
responsibility. `--harness-arg -c key=value` remains the escape hatch for
other codex config keys.

#### Interactive examples

```bash
# claude interactive — stays in TUI, steer after launch
agent-run -i --harness claude --prompt 'Start the task' chat
agent-run steer chat 'Also add tests for edge cases.'

# opencode interactive
agent-run -i --harness opencode \
          --model llmproxy-anthropic/claude-sonnet-4.6 \
          --prompt 'Start the task' chat

# codex interactive (uses app-server turn/steer)
agent-run -i --harness codex --model o4-mini \
          --prompt 'Start the task' chat
```

`steer` on a one-shot run exits non-zero with a message naming the run;
it does not silently no-op.

#### Session acquisition per harness

Each harness uses a different mechanism. All are `certain` — never a guess.

| Harness | Mechanism | `acquisition` |
|---------|-----------|---------------|
| `claude` | agent-run always generates a UUID4 and passes it to the claude binary as `--session-id <uuid>`; the id is not caller-selectable | `pushed` |
| `opencode` | launches `opencode --port N --auto`, polls `/global/health`, `POST /session`, then attaches with `--session <id>`; the returned `directory` is verified against the launch cwd so a foreign server that won the port race is rejected | `minted` |
| `codex` | keeps a `codex app-server` process alive, sends `thread/start` over JSON-RPC to mint the thread id, then sends `turn/start` | `minted` |

Acquisition failure never affects run status, exit code, or log content.
When acquisition fails, `session.json` is written with `confidence: "missing"` and the run continues normally.

#### Persistent files (managed mode)

These files are written to the persistent log dir (`/var/tmp/agent-runs/<name>/`)
and survive reboots.

`session.json` — session attribution:

```json
{
  "session_id": "ses_ff511aa00ffe5t83A70X8YqR7F",
  "harness": "opencode",
  "acquisition": "minted",
  "confidence": "certain",
  "observed_at": "2026-08-16T15:20:11Z"
}
```

Fields: `session_id` (the harness's own session/thread identifier),
`harness` (`claude`/`opencode`/`codex`), `acquisition` (`pushed`/`minted`/`reported`/`missing`),
`confidence` (`certain` or `missing` — never a guess), `observed_at` (ISO-8601 UTC).
Absent for raw-mode runs.

`run.json` — launch and terminal facts, reboot-durable:

```json
{
  "name": "build",
  "argv": ["agent-run", "--harness", "claude", "--prompt", "Refactor X", "build"],
  "command": "agent-run --harness claude ...",
  "cwd": "/Users/you/project",
  "started_at": "2026-08-16T15:20:10Z",
  "harness": "claude",
  "agent_run_version": "0.1.0",
  "interactive": false,
  "model": null,
  "agent_mode": null,
  "ended_at": "2026-08-16T15:22:04Z",
  "exit_code": 0,
  "status": "done"
}
```

Written atomically at launch with the fields above, then updated at exit with
`ended_at`, `exit_code`, and `status`. `agent_run_version` records the harness
version that created the run. Present for both raw and managed runs.
Liveness state (`pid`, `status`, `stdin` FIFO, etc.) remains in the ephemeral
`/tmp/agent-runs/<name>/` directory only — a missing entry there unambiguously
means "not running", and that invariant is intact.

#### `watch --json` — additive `session` object

`agent-run watch <name> --json` gained an additive `session` field. Every
pre-existing key (`status`, `terminal`, `elapsed_s`, `log.*`, `git.*`,
`signals.*`) is unchanged in name and meaning. When `session.json` exists in
the run's persistent log dir, `session` is the parsed object; otherwise it is
`null`.

#### `watch --json` — additive `transcript` object

`agent-run watch <name> --json` also gained an additive `transcript` field,
alongside `scratch` and `hooks`:

```json
"transcript": {
    "available": true,
    "entries": 12,
    "newest_age_s": 4.2,
    "submitted_age_s": 5.9,
    "error": null
}
```

`available` is `true` only when the harness's own conversation store was
located, opened and queried successfully. Every decision field is `null`
rather than `0` on failure — modeled on `scratch`'s `_scratch_unknown` rule —
so a poller can never read a failed count as an observed empty transcript;
`entries: 0` is trustworthy exactly because the failure paths never produce
it. `error` is one of `no_session_json` (raw run — routine, not a warning),
`no_session_id`, `unknown_harness`, `store_missing`, `store_locked`,
`store_unreadable`, or `not_observed` (the payload default before observation
runs). `entries` counts raw store records, not rendered transcript entries —
for `opencode` this is `part` table rows before `_opencode_entry`'s filtering,
so a count of `0` guarantees nothing renders, but a nonzero count does not
guarantee anything does either; for `claude`/`codex` a file where every
record was unparseable or named a different session (nothing counted, at
least one unparseable or mismatched) reports `store_unreadable` rather than
`entries: 0`, since a wholly malformed or misidentified file is not evidence
of an empty session. This block never raises
past `watch`'s never-raise boundary; any read failure, including an
unexpected exception type from the count call, degrades to the unavailable
form.

`submitted_age_s` is seconds since `state_dir/prompt_submitted` was written,
when initial-prompt submission was confirmed, not since process launch
(`elapsed_s`), which also counts harness boot, session mint and TUI readiness.
Populated only for interactive runs (`interactive: true`)
that have written the marker; `null` otherwise, including when the run is
still starting up, is not interactive, or the marker is unreadable. Computed
independently of the store read, so it survives a transcript-store failure
that nulls out `entries`.

#### `transcript` — the harness's own conversation record

```bash
agent-run transcript <name> [--tail N | --head N]  # last/first N entries (default --tail 50)
agent-run transcript <name> --json                 # one JSON object per entry
```

Rendering a PTY log through pyte (`logs --clean`) reconstructs what the
terminal *drew*: progress bars and status chrome, paragraphs re-wrapped at
different widths, tool output collapsed to `Click to expand`. Every managed
harness already stores the real conversation — user turns, assistant turns,
tool calls with arguments and results — keyed by the session id in
`session.json`. `transcript` reads that store directly instead:

| Harness | Store |
|---------|-------|
| `opencode` | SQLite at `~/.local/share/opencode/opencode.db` (read-only, `mode=ro`), `part`/`message` rows for the session |
| `claude` | `~/.claude/projects/<mangled-cwd>/<session_id>.jsonl`, plus any matching `subagents/*.jsonl` |
| `codex` | `~/.codex/sessions/<YYYY>/<MM>/<DD>/rollout-*-<session_id>.jsonl` |

Every store is opened strictly read-only — a live run may hold it open, and
`transcript` never writes, migrates, or checkpoints it. Requires a
**managed-mode** run (`--harness claude|opencode|codex`): a raw run
(`agent-run NAME -- <cmd>`) has no `session.json` and therefore no
transcript; `transcript` exits non-zero naming the two alternatives (relaunch
under `--harness`, or use `logs --clean` on the captured PTY log). Unparseable
individual records are skipped and counted on stderr. Exit status
discriminates *availability*, not emptiness: a store that was located,
opened and queried successfully exits **0** whether or not it holds any
entries for this session — an empty result prints nothing on stdout and an
explanatory note on stderr, so a caller can branch on "rc=0, blank stdout"
to mean "available and empty" specifically, distinct from "no transcript to
read at all" (raw run, unacquired session id, missing/unreadable store),
which still exits non-zero. A single tool result is
truncated at 40 lines / 4 KiB with a marker naming how much was dropped,
independent of `--head`/`--tail`, which slice whole entries (not bytes) using
the same defaults as `logs`. For `claude`, entries from the main session and
any subagent transcripts are merged by timestamp, not concatenated per file.


```json
{
  "schema": "agent-run.watch.v2",
  "agent_run_version": "0.1.0",
  "name": "build",
  "status": "done",
  "terminal": true,
  "session": {
    "session_id": "ses_ff511aa00ffe5t83A70X8YqR7F",
    "harness": "opencode",
    "acquisition": "minted",
    "confidence": "certain",
    "observed_at": "2026-08-16T15:20:11Z"
  },
  "..."
}
```

### Inspect / control

```bash
agent-run list                            # non-terminal runs only (default)
AGENT_RUN_LIST_DEFAULT=all agent-run list # restore the pre-filter default for a caller
agent-run list --all                      # every recognized run, including done/failed/died/killed
agent-run list --status died,killed       # only runs whose status is in this set
agent-run list --include-logs             # also show preserved-log-only runs
agent-run status <name>                   # one-line status
agent-run logs <name> [--tail N | --head N]    # last/first N lines (default --tail 50)
agent-run logs <name> --plain                  # ANSI-stripped, for grepping/piping
agent-run logs <name> --clean                  # pyte-rendered transcript; slow on a large log
agent-run transcript <name> [--tail N | --head N] [--json]  # harness's own conversation record
agent-run tail <name>                     # follow log (exits when agent dies)
agent-run steer <name> '<message>'        # write to agent stdin (needs -i)
agent-run attach <name>                   # attach interactively (Ctrl-C detaches)
agent-run kill <name> [SIGNAL]            # default TERM; KILL force-terminates
agent-run reap [--dry-run] [--idle-hours N] [--min-age-hours N] [--force-unknown] [--name NAME]
                [--all]
                [--include-logs] [--log-min-age-hours N]
                [--include-worktrees] [--worktree-min-age-hours N] [--force-dirty]
                [--orphan-processes] [--orphan-min-age-hours N]
                [--max-seconds N]
agent-run du [--by-run] [--top N] [--bytes|--json]  # disk usage; read-only
```

Unlike `tail`, which only streams output, `attach` gives live keyboard and
terminal-resize passthrough to an interactive run — it adopts the attached
terminal's size immediately and after every subsequent resize. Ctrl-C
detaches the attach client locally, exiting 0, without touching the wrapped
agent, which keeps running. Text pasted with bracketed paste is forwarded
verbatim: a `0x03` byte or a literal Ctrl-C escape sequence inside a paste
is content, not a detach keystroke. Multiple simultaneous attaches are
allowed and all of them see the output, but only one should type at a time
— each write is atomic up to `PIPE_BUF`, but a longer burst is split across
several writes that another client can interleave between, so an escape
sequence can still tear mid-sequence. The PTY size tracks whichever client
resized most recently.

Attaching from inside another attach session shares one terminal between
two raw-mode owners: the inner session restores the outer session's raw
mode when it exits, so exit them in reverse order or the terminal is left
raw (`reset` fixes it).

`kill` sends TERM/INT/HUP straight to the identity-verified runner, which
catches it and runs its own teardown (kill/reap the workload, publish
terminal state). `agent-run kill <name> KILL` does not send a raw SIGKILL
to the runner — an uncatchable signal would skip that teardown and orphan
the running agent while state still said "running". Instead it TERMs the
runner first and waits a bounded window for normal teardown; only if the
runner is still alive after that does it re-verify identity and
parentage, KILL the runner and its recorded children directly, and
publish terminal state itself. Only TERM, INT, HUP, and KILL are accepted;
other signals are rejected rather than forwarded.

`status` reports `not running (log preserved)` when the process state is
gone (e.g. after a reboot) but the log survived in `/var/tmp`. `list`
defaults to showing only non-terminal runs (`starting`/`running`/`stalled`)
from `/tmp` — pass `--all` or `--status <list>` to include conclusively
terminal ones (`done`/`failed`/`died`/`killed`), or set
`AGENT_RUN_LIST_DEFAULT=all` to restore the prior default without changing
call sites. Unrecognized/legacy/corrupt statuses are always shown under an
explicit `Unrecognized / needs attention` heading rather than as live runs;
use `reap --force-unknown` only after operator review. Preserved-log-only
runs (state dir already gone) are hidden by default — pass `--include-logs`
(or set `AGENT_RUN_LIST_INCLUDE_LOGS=1`) to show them, orthogonal to
`--all`/`--status`, which govern only the state-backed sections above it.
When hidden preserved logs exist, a one-line hint is printed to stderr
(never stdout, so `agent-run list | grep ...` stays honest). `logs`/`tail`/
`attach` always read the persistent log, falling back to the old
single-directory layout for runs launched before the state/log split.

### Verified submission

An interactive launch prompt and every `steer` are self-verifying: delivery
is confirmed against the harness's own transcript record count (opencode's
HTTP `/session/<id>/message` endpoint when a port is known, else the
transcript store), not merely by a successful FIFO/HTTP write. `steer`
prints `verified` and exits 0 only once the record count has risen above
its pre-submit baseline; a submission that cannot be verified within the
default 10 seconds is retried once (two attempts total) when the witness was
readable but stayed flat -- proof the prompt did not land. A witness that never became
readable through the whole attempt is left at a single submission: it
carries no evidence the prompt failed to land, so a retry could only
duplicate it (messageID is not idempotent). Either way an unverified
submission has `steer` exit 1 with the reason on stderr and a launch prompt
write `state_dir/prompt_unverified` instead of stamping `prompt_submitted`.
Override the timeout/attempt count with `AGENT_RUN_SUBMIT_VERIFY_TIMEOUT`
and `AGENT_RUN_SUBMIT_ATTEMPTS`; `--raw` steer skips verification entirely
(arbitrary bytes leave nothing in the transcript to witness). `opencode_port`
records the port a managed interactive opencode run's HTTP API is
listening on, so the witness can query it directly.

`scripts/verify-submission.sh [--harness opencode|claude|codex] [--keep]`
runs the end-to-end compatibility checks against installed harnesses and
prints their PASS/FAIL/SKIP results.


`agent-run reap` reconciles stale state and cleans up old runs in one pass:

1. **Stale-running reconciliation** (unchanged from before): a `running`
    run whose pid is missing, malformed, or gone is marked `died`; a
    `running` run whose pid is alive but whose log has been idle longer than
    `--idle-hours` (or `AGENT_RUN_IDLE_KILL_HOURS`, default 24h) is
    idle-killed through the same identity-verified escalation
    `agent-run kill <name> KILL` uses, and marked `killed`.
2. **Linked-worktree garbage collection** (opt-in, `--include-worktrees`):
    a terminal run's recorded launch `cwd` is removed **only when that
    directory is a linked git worktree** and the *youngest* terminal run
    sharing it is older than `--worktree-min-age-hours` (or
    `AGENT_RUN_WORKTREE_MIN_AGE_HOURS`, default 168h/7 days — its own
    independent threshold, because this destroys a working tree rather than
    bookkeeping). **A non-git directory is never removed**: a run's `cwd` is
    frequently a real project checkout or `$HOME`, so main/root worktrees,
    bare repositories, non-git directories, and anything git cannot classify
    are all refused, as is any path with a symlinked component. A git failure
    of any kind (git missing, timeout, unreadable repo) is a refusal, never a
    green light. Removal goes through `git worktree remove` against the
    owning repository, so the parent's `.git/worktrees/<name>` administrative
    entry is unregistered as well — an `rmtree` would strand it. Also
    refused, with the reason printed: a worktree with uncommitted changes,
    untracked files, or commits present on no remote (override with
    `--force-dirty`, off by default, which destroys unpushed work
    irreversibly), and one whose directory any live (non-terminal) run is
    still using. A worktree shared by several terminal runs is handled once,
    not once per run. Runs before step 3, because the `cwd` it reads lives in
    the state dir step 3 deletes. The summary line always reports
    `worktrees_removed=N worktrees_skipped=N`.
3. **Terminal-state and orphan-scratch garbage collection**: runs whose
    status is conclusively terminal (`done`, `failed`, `died`, `killed`) and
    whose `ended_at` is older than `--min-age-hours` (or preferred
    `AGENT_RUN_MIN_AGE_HOURS`; compatible alias `AGENT_RUN_REAP_MIN_AGE_HOURS`,
    default 168h/7 days) have their ephemeral state dir *and* scratch dir
    (`tmp/`, the `TMPDIR`/`BUN_TMPDIR` target) removed. A state-less `tmp/`
    left after a reboot is independently collected once its contents have
    aged past the same threshold. A run reconciled to `died`/`killed` in
    this invocation is never collected in the same invocation. Unknown/
    legacy/corrupt statuses are left intact by default and require
    `--force-unknown` to collect after review.
4. **Preserved-log garbage collection** (opt-in, `--include-logs`): whole
    preserved-log-only run directories (state dir already gone) whose
    newest recursive mtime is older than `--log-min-age-hours` (or
    `AGENT_RUN_LOG_MIN_AGE_HOURS`, default 21 days — matching the existing
    unconditional whole-log-dir prune) are removed entirely, including
    `log`/`prompt`/`tmp/`. **Deliberately independent of
    `--min-age-hours`**: a preserved log is the artifact an operator wanted
    to keep, not disposable state bookkeeping, so it defaults to a much
    longer retention window and is never influenced by the state-dir
    threshold. Off by default — without `--include-logs`, reap never
    touches a preserved log, matching the persistent `log`/
    `prompt` behaviour of step 3. A run with a live state dir is never
    touched by this step regardless of age. Runs after step 3 (so a state
    dir step 3 just removed can become log-only and eligible in the same
    invocation) and before the orphan-scratch sweep (so a log dir removed
    whole here is never also probed for a leftover `tmp/`).
5. **Orphan-process termination** (opt-in, `--orphan-processes`): find and
    terminate live agent-run runner processes that have **no state directory**
    — invisible to passes 1-4 because they hold no entry in
    `$AGENT_RUN_STATE_DIR`. This kills processes agent-run has **no state
    record for**, selected by argv parsing, and is opt-in for that reason.
    Candidates are identified by strict argv matching (basename check, not a
    substring — `bash -lc "cat /var/tmp/agent-runs/foo/log"` is explicitly
    not a runner), then filtered by age (`--orphan-min-age-hours` or
    `AGENT_RUN_ORPHAN_MIN_AGE_HOURS`, default 24h — independent of all GC
    thresholds), self/ancestor/process-group/pid-1/uid safety rules, and a
    state-dir check. Identity captured at discovery is re-verified
    immediately before every signal; any ambiguity aborts the candidate
    rather than sending a signal (PID reuse is the central hazard with no
    state dir to cross-check against). SIGTERM with a bounded grace window,
    then SIGKILL for anything still alive. The summary line always includes
    `orphan_procs_killed=N orphan_procs_skipped=N`; skipped discovery
    candidates are printed with their skip reason so declined processes are
    visible rather than silently ignored. **Never run without `--dry-run`
    until you have reviewed its output.** A missed orphan costs a process
    slot; a wrong kill destroys someone's running work.

Every step only ever acts after re-verifying a live pid belongs to the
recorded runner (`_pid_alive`/`_process_identity`) or an inode hasn't been
swapped out from under the scan (`_safe_rmtree`'s root-contained,
inode-reverified deletion, plus the same per-name launch lock used to
serialize relaunches). State/log GC first atomically renames a directory to
a reserved sentinel so an interrupted deletion is resumed by the next reap;
`--dry-run` runs the same read-only eligibility checks and prints only actions
a real reap would take, without mutating or deleting anything.

`--all` enables every opt-in pass at once — `--include-logs`,
`--orphan-processes` and `--include-worktrees` — and nothing else. It changes
no age threshold: `--min-age-hours`, `--log-min-age-hours`,
`--orphan-min-age-hours` and `--worktree-min-age-hours` all keep their
values and defaults. **It does not imply `--force-dirty` or
`--force-unknown`**: those override *refusals* (unpushed or uncommitted work,
unclassifiable status) rather than enable a pass, so "do all the passes"
never silently means "ignore the safety refusals" — request either
explicitly, after review. Passing `--all` together with an individual pass
flag is redundant, not an error. `--dry-run --all` previews every pass with
the same accuracy each pass has on its own.

`--max-seconds N` sets a soft candidate-admission budget. The monotonic budget
is checked between candidates in every pass, not during an operation already in
progress. Scans, lock waits, recursive filesystem reads, and TERM/KILL grace
periods can therefore overrun it; an uninterruptible kernel read cannot be
bounded by this process. When the budget expires, later candidates are skipped,
the summary reports `deferred=N`, and the process exits 0 so a later scheduled
tick can resume. Leave enough scheduling margin for one in-progress operation.

### Scheduling

Running `agent-run reap` once by hand is useful for immediate cleanup; the
feature is designed to run periodically and unattended so orphaned processes and
accumulated state are reclaimed automatically without operator intervention.
The units below are ready to copy and adapt for the two common platforms.

**Before enabling any timer, run the exact scheduled command with `--dry-run`
by hand and read the output.** This is especially important the first time
`--orphan-processes` is used on a host: the candidate list and skip reasons are
printed, making it straightforward to confirm that nothing unexpected would be
killed before committing to a live run.

The flag set used throughout this section:

```
--idle-hours 96 --orphan-processes --orphan-min-age-hours 72 \
--include-logs --log-min-age-hours 336 --max-seconds 600
```

#### Choosing thresholds

The retention knobs are deliberately independent so each can be tuned without
affecting the others:

- **`--idle-hours N`** — a *running* agent whose log has been idle for this
  long is idle-killed (TERM then escalating to KILL). Controls live-run
  timeout. Default 24 h.
- **`--min-age-hours N`** — terminal-state run directories (`done`, `failed`,
  `died`, `killed`) are removed once `ended_at` is older than this. Controls
  how long bookkeeping state is retained after a run finishes. Default 168 h
  (7 days).
- **`--log-min-age-hours N`** — preserved-log-only directories (state dir
  already gone) are removed once their newest mtime is older than this.
  Deliberately longer than `--min-age-hours` because preserved logs are an
  artifact an operator chose to keep, not disposable state bookkeeping.
  Default 504 h (21 days).
- **`--orphan-min-age-hours N`** — live processes with no state directory
  must have been running for at least this long before they are eligible for
  orphan termination. Independent of all GC thresholds. Default 24 h.
- **`--worktree-min-age-hours N`** — a linked git worktree used as a
  terminal run's launch `cwd` is removed (only under `--include-worktrees`)
  once the youngest run sharing it is older than this. Independent of every
  threshold above because it destroys a working tree, not bookkeeping.
  Default 168 h (7 days).

Linked worktrees are typically where the bytes actually are: on one measured
host, 94 runs resolved to 23 launch directories, 20 of them linked worktrees
totalling 18 GB, against a few hundred MB in `STATE_ROOT` plus `LOG_ROOT`.
Run `agent-run du --by-run --top 20` to see that, and
`agent-run reap --include-worktrees --dry-run` before ever enabling the
removal on a timer.

On disk, `--log-min-age-hours` mainly affects the long tail of old logs: on a
busy host, most bytes are in *recent* logs, and PTY-captured `-i` runs are
by far the largest individual directories (potentially hundreds of MB each
from the captured transcript). A long retention window may free very little on
a host whose volume is dominated by the last few days. Run
`agent-run du --by-run --top 20` to see where the bytes actually are before
tuning retention thresholds.

#### Interval vs. budget

Keep `--max-seconds` comfortably under the scheduling interval and allow for an
in-progress operation to overrun it. Overlapping reap invocations contend on
the same per-name locks, which serializes work and defeats the point of running
on a short interval. A 30-minute interval with `--max-seconds 600` usually
leaves ample margin; use an external service timeout if a hard bound is needed.

#### systemd (Linux, user units)

`~/.config/systemd/user/agent-run-reap.service`:

```ini
[Unit]
Description=agent-run reap (reconcile stale status, idle-kill lingering runs, GC terminal run dirs, reap orphan processes and old logs)

[Service]
Type=oneshot
ExecStart=%h/.local/bin/agent-run reap --idle-hours 96 --orphan-processes --orphan-min-age-hours 72 --include-logs --log-min-age-hours 336 --max-seconds 600
```

`~/.config/systemd/user/agent-run-reap.timer`:

```ini
[Unit]
Description=Run agent-run reap every 30 minutes

[Timer]
OnBootSec=10min
OnUnitActiveSec=30min
Persistent=true

[Install]
WantedBy=timers.target
```

Enable:

```bash
systemctl --user daemon-reload
systemctl --user enable --now agent-run-reap.timer
```

Inspect:

```bash
systemctl --user list-timers agent-run-reap.timer
journalctl --user -u agent-run-reap.service -n 20
```

User units only run while the user has an active login session. On a headless
box where the user account has no persistent session, set
`loginctl enable-linger <user>` so the user's systemd instance starts at boot
and the timer fires regardless of whether anyone is logged in.

#### launchd (macOS)

`~/Library/LaunchAgents/com.example.agent-run-reap.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.example.agent-run-reap</string>

  <key>ProgramArguments</key>
  <array>
    <string>/Users/you/.local/bin/agent-run</string>
    <string>reap</string>
    <string>--idle-hours</string>
    <string>96</string>
    <string>--orphan-processes</string>
    <string>--orphan-min-age-hours</string>
    <string>72</string>
    <string>--include-logs</string>
    <string>--log-min-age-hours</string>
    <string>336</string>
    <string>--max-seconds</string>
    <string>600</string>
  </array>

  <key>StartInterval</key>
  <integer>1800</integer>

  <key>RunAtLoad</key>
  <false/>

  <key>StandardOutPath</key>
  <string>/Users/you/Library/Logs/agent-run-reap.log</string>

  <key>StandardErrorPath</key>
  <string>/Users/you/Library/Logs/agent-run-reap.log</string>
</dict>
</plist>
```

Replace `com.example.agent-run-reap` with your own reverse-DNS label and
`/Users/you` with the absolute path to your home directory. launchd does not
inherit a login shell's `PATH`, so the path to `agent-run` must be absolute;
launchd also does not parse a shell command line, so each flag and its value
must be a separate `<string>` element in `ProgramArguments`.

Load and manage:

```bash
# Validate the plist before loading
plutil -lint ~/Library/LaunchAgents/com.example.agent-run-reap.plist

# Load the agent
launchctl load ~/Library/LaunchAgents/com.example.agent-run-reap.plist

# Confirm it is registered
launchctl list | grep agent-run-reap

# Trigger a manual run immediately (without waiting for the interval)
launchctl start com.example.agent-run-reap
```

### Disk usage (`du`)

`agent-run du` reports apparent disk usage (`st_size`), by default one row
per effective status (plus a `preserved-log-only` row) — pass `--by-run`
for one row per run instead. Each row breaks down state-dir, log-dir
(excluding `tmp/`), scratch (`tmp/`), and linked-worktree bytes, plus their
total; rows sort by total descending, then name, with a `TOTAL` row last
that always covers every run, even under `--top N`. `--bytes` prints exact
integers instead of human-readable sizes (`1.2G`, `340M`, `12K`); `--json`
emits a machine-readable object and always uses exact integers, so combining
it with `--bytes` is rejected. `du` never mutates anything — no locks, no
`_opportunistic_heal`, no `_prune_old_logs` — and tolerates races
(`FileNotFoundError`/`PermissionError`) by skipping the affected entry.
When a permission error is encountered during a walk, the affected row is
marked incomplete: the table prints a WARNING line and `--json` adds
`"complete": false` to the row.  Totals for incomplete rows are lower
bounds, not exact byte counts.

#### Linked worktrees

Agents are frequently launched inside a linked git worktree created for the
task, and those trees usually hold far more bytes than `STATE_ROOT` and
`LOG_ROOT` combined. `du` therefore adds a `WORKTREE` column (and
`worktree_bytes` under `--json`) sizing each run's recorded launch `cwd`
**when that directory is a linked worktree**. Main/root worktrees, bare
repositories, non-git directories, missing directories, and anything git
cannot classify all contribute 0 — a git failure is never read as a
countable worktree.

Attribution: launch directories are deduplicated by realpath, and each
worktree is charged in full to the **first run** (by run name) that recorded
it; every other run sharing it shows 0. Without that, one 900 MB worktree
shared by four runs would be reported as 3.6 GB. `TOTAL` therefore counts
every readable byte exactly once, and `TOTAL` == `STATE` + `LOG` + `SCRATCH` +
`WORKTREE` in both `--by-run` and rollup mode, subject to any incomplete
walks noted above.

Walking those trees dominates the command's runtime: worktrees are large,
and sizing them costs several times a `STATE_ROOT`-plus-`LOG_ROOT`-only
walk. Detection is read-only `git rev-parse` plumbing — it never runs
`git gc` or `git worktree prune` and never writes to the inspected
repository.

```bash
agent-run du                    # per-status rollup, human-readable
agent-run du --by-run --top 10  # 10 largest runs by total size
agent-run du --json             # machine-readable, exact byte integers
```

### Files

Ephemeral, under `$AGENT_RUN_STATE_DIR/<name>/` (default `/tmp/agent-runs`):

| File | Contents |
|------|----------|
| `status` | `starting` / `running` / `done` / `failed` / `died` / `killed` |
| `exit_code` | numeric exit code (after completion) |
| `pid`, `pgid` | agent session/group leader pid (== pgid under setsid); `pgid` is informational only, not a kill target |
| `process_identity` | platform-specific runner birth token, verified before `kill` signals the runner |
| `command` | pretty-printed launch command |
| `argv` | JSON-encoded argv (authoritative form for replay) |
| `cwd` | absolute working directory the command runs in (`--cwd` if given, else the invocation directory) |
| `launch_head` | full git commit hash `HEAD` pointed at when the run started, if `cwd` is a repo; `watch` counts commits from it |
| `started_at`, `ended_at` | ISO-8601 UTC timestamps |
| `stdin` | FIFO for `steer` (only when launched with `-i`) |
| `resize` | FIFO for terminal-resize records used by `attach` (only when launched with `-i`) |
| `pty_pid` | PID of the PTY child (interactive only) |
| `keeper_pid` | PID of the FIFO keeper (interactive only) |
| `interactive` | `1` if launched with `-i`, else `0` |
| `reap_reason` | set by `agent-run reap` when it changes status (died/killed) |
| `tmp_dir` | absolute path to this run's scratch dir |

Persistent, under `$AGENT_RUN_LOG_DIR/<name>/` (default `/var/tmp/agent-runs`):

| File | Contents |
|------|----------|
| `log` | combined stdout+stderr (PTY-captured in interactive mode); `agent-run logs --clean` renders it into a readable transcript on demand, and `agent-run transcript` reads the harness's own conversation record instead (managed mode only; see [`transcript`](#transcript--the-harnesss-own-conversation-record)) |
| `prompt` | copy of the `-f`/`--prompt-file` input, if one was given |
| `session.json` | session attribution (managed mode only): `session_id`, `harness`, `acquisition`, `confidence`, `observed_at`; absent for raw runs |
| `run.json` | launch and terminal facts (all modes): `name`, `argv`, `command`, `cwd`, `started_at`, `harness`, `interactive`, `model`, `agent_mode`; augmented with `ended_at`, `exit_code`, `status` at exit |
| `hooks.jsonl` | append-only record of harness hook notifications, one JSON object per line: `event`, `kind`, `at`, `harness`, `message`, `payload`. Absent until a hook fires; agent-run never writes it itself, and never edits a harness config to make it happen. See [Harness hooks](#harness-hooks) |
| `tmp/` | per-run scratch dir exported as `TMPDIR` and `BUN_TMPDIR`; removed only by `agent-run reap`, never on normal run exit |

### Harness hooks

A harness can tell `agent-run` when its agent finished a turn or stopped to ask
for permission. `agent-run hook <event>` appends one record to the run's
`hooks.jsonl`, and `agent-run watch --json` reports an aggregate, so a
supervisor learns within about a second instead of inferring it from a silent
log.

**Hooks are opt-in and you configure them.** `agent-run` never writes, edits or
injects harness configuration; it only exports `AGENT_RUN_NAME`,
`AGENT_RUN_STATE_DIR` and `AGENT_RUN_LOG_DIR` into the agent's environment so
the hook can identify the run without being told.

`agent-run hook` always exits 0 and writes nothing to stdout, on every failure
path including an unknown run, an unwritable log directory and malformed
arguments. A hook that failed loudly would be worse than one that did nothing:
claude treats a non-zero exit from a `Stop` hook as a decision to block, and
codex warns on any output from a notify program.

#### claude — `~/.claude/settings.json`

```json
{"hooks": {
  "Stop": [{"matcher": "", "hooks": [{"type": "command",
    "command": "agent-run hook stop"}]}],
  "PermissionRequest": [{"matcher": "", "hooks": [{"type": "command",
    "command": "agent-run hook permission-request"}]}]}}
```

The payload arrives on stdin and carries `last_assistant_message`.

#### codex — `~/.codex/config.toml`

```toml
notify = ["agent-run", "hook", "turn-complete"]
```

The payload arrives as `argv[1]` and carries `last-assistant-message`. codex
reports only turn completion this way. Its richer hook engine does have a
`PermissionRequest` event, but enabling it needs
`--dangerously-bypass-hook-trust`, which `codex app-server` — the binary that
managed mode drives — does not accept, so `permission_required` is not
available for managed codex runs.

#### opencode — `~/.config/opencode/plugin/agent-run-hook.js`

```js
import { execFileSync } from "child_process"

export const AgentRunHook = async () => ({
  event: async ({ event }) => {
    const kind = event.type === "session.idle" ? "session-idle"
      : event.type.startsWith("permission.") ? "permission-request"
      : null
    if (!kind) return
    execFileSync("agent-run", ["hook", kind, "--json", JSON.stringify(event)])
  },
})
```

The export must be a named factory returning an object of handlers. A default
export such as `export default {onEvent}` loads without complaint and is never
invoked, so the hook silently never fires.

#### What a consumer reads

The event name is yours to choose; `kind` is what a supervisor should branch on.
It normalises each harness's vocabulary into `turn_complete`,
`permission_required`, `session_start` or `other`, preferring evidence in the
payload over the event name, so renaming your hook cannot break a consumer.

```console
$ agent-run watch my-run --json | jq .hooks
{
  "count": 3,
  "last": {
    "event": "stop",
    "kind": "turn_complete",
    "at": "2026-08-19T15:06:29Z",
    "harness": "claude",
    "message": "Refactor done; the suite is green at 1199 passed."
  },
  "last_event_age_s": 0.94,
  "events": {"stop": 2, "permission-request": 1},
  "kinds": {"turn_complete": 2, "permission_required": 1},
  "at_cap": false,
  "truncated": false
}
```

`message` is the last assistant turn when the payload carries one, kept to
4096 characters. Records are capped at `AGENT_RUN_HOOK_MAX_EVENTS` (default
1000) per run; past that the hook drops the record rather than growing the file
without bound.

`scripts/verify-hook-delivery` exercises the whole chain — hook invoked,
`AGENT_RUN_NAME` inherited, `hooks.jsonl` written, `watch` reporting it — for
each harness in one-shot and interactive mode, and records the harness versions
it tested against.

### Design notes

- Written in Python (`src/toolbox/agent_run.py`), installed as a
  `[project.scripts]` entry point alongside the rest of the toolbox.
- Double-forks and `os.setsid()` on launch so the run becomes its own
  session + process-group leader. `agent-run kill` signals the
  identity-verified runner directly (never the whole process group); on
  Linux the signal is bound to a pidfd opened before the final identity
  check, closing the ordinary PID-recycling window. Darwin has no pidfd
  equivalent, so a narrow TOCTOU gap remains between the last identity
  re-check and the actual signal call — accepted as a residual risk, not
  fully closed.
- Each run name is serialized by a permanent per-name lock file under
  `$AGENT_RUN_STATE_DIR/.locks/<name>.lock`; these are never pruned, so
  concurrent launches/prunes of the same name always contend on the same
  lock inode. The detached runner holds its own inherited copy of that
  lock fd until it has published its identity and resolved readiness, so
  a launcher dying mid-setup cannot release the lock early.
- On interactive runs, a dedicated "keeper" child process holds the FIFO
  open for writing (`O_RDWR`) so readers never see EOF between steers.
- The PTY is allocated via `pty.fork()`; the parent runs a `select()` loop
  that shuttles FIFO → PTY master (keystrokes) and PTY master → log file
  (agent output). Works identically on Linux and macOS without depending
  on the (different) `script(1)` flavors.
- SIGTERM/INT/HUP handlers always finalize `status` + `exit_code` +
  `ended_at`, even when the launcher is killed mid-run. `status` starts at
  `starting` (published before the detached runner exists) and only moves
  to `running` once the runner is actually controllable — child
  spawned/exec'd for one-shot, or PTY/FIFO/keeper ready for interactive.
  Any setup failure before that point still resolves synchronously to
  `failed`, never leaving `starting` stranded with no process behind it.
- `TMPDIR` and `BUN_TMPDIR` (same value: the run's scratch dir) are
  exported into `os.environ` inside the detached runner before any child is
  forked/exec'd, so the launched agent — and anything it in turn forks or
  execs — inherits both automatically; the launch argv itself is never
  touched. `BUN_TMPDIR` is set because Bun does not consult `TMPDIR` for
  its own scratch space. This is deliberately environment-only rather than
  argv-injected `env TMPDIR=... BUN_TMPDIR=...`, since some agents' own
  argument parsers would otherwise need to understand and pass through an
  unrecognized wrapper prefix.
- `agent-run list`'s default view and `agent-run reap`'s garbage-collection
  eligibility share one definition of "terminal" (`done`/`failed`/`died`/
  `killed`) so they never disagree about which runs are "done with". A
  script that previously scraped every line under `list`'s "Live runs"
  heading and relied on terminal runs appearing there needs `--all` now —
  this is the one deliberate behavior break in this feature; the heading
  text also changes to describe the filter actually in effect.

## License

MIT
