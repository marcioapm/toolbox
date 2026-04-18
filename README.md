# toolbox

A collection of lightweight CLI tools for AI content generation and chat operations. Zero dependencies beyond Python 3.10+ stdlib.

## Tools

| CLI | Description | Auth |
|-----|-------------|------|
| `gemini-image` | Generate images via Imagen 4.0 / Gemini native models | `GEMINI_API_KEY` |
| `gemini-tts` | Text-to-speech via Gemini native audio | `GEMINI_API_KEY` |
| `gemini-video` | Generate video via Google Veo 2/3/3.1 | `GEMINI_API_KEY` |
| `gemini-vision` | Analyze images/videos via Gemini (supports YouTube, Instagram, TikTok) | `GEMINI_API_KEY` |
| `slackcli` | Lightweight Slack client (channels, messages, search, reactions) | `SLACK_USER_TOKEN` |
| `llm-usage` | Monitor LLM token usage, costs, and quotas across providers | `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY` |
| `agent-run` | Background wrapper for coding agents (Claude Code, Codex…) with steering + live log streaming | — |

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
agent-run status build                    # running | done | failed
agent-run -i chat claude --permission-mode bypassPermissions
agent-run steer chat 'Also add tests for edge cases.'
agent-run kill chat                       # clean kill of the whole group
```

---

## agent-run

Background wrapper for long-running coding agents (Claude Code, Codex, Pi, OpenCode).
Creates a run directory under `/tmp/agent-runs/<name>/` with structured state
files you can poll safely — no brittle process-poll loops — and adds a stdin
FIFO when you need to steer an interactive agent mid-flight.

### Launch

```bash
# Non-interactive (one-shot, e.g. claude --print, codex exec):
agent-run build claude --permission-mode bypassPermissions --print 'Build the thing'

# Interactive (steerable via stdin FIFO):
agent-run -i chat claude --permission-mode bypassPermissions
```

### Inspect / control

```bash
agent-run list                            # show all runs
agent-run status <name>                   # one-line status
agent-run logs <name> [N]                 # last N lines (default 50)
agent-run tail <name>                     # follow log (exits when agent dies)
agent-run steer <name> '<message>'        # write to agent stdin (needs -i)
agent-run kill <name> [SIGNAL]            # default TERM; use 9 if stuck
```

### Files under `/tmp/agent-runs/<name>/`

| File | Contents |
|------|----------|
| `status` | `running` / `done` / `failed` |
| `exit_code` | numeric exit code (after completion) |
| `pid`, `pgid` | agent session/group leader pid (== pgid under setsid) |
| `log` | combined stdout+stderr, tee'd live |
| `command` | pretty-printed launch command |
| `argv` | NUL-delimited argv (used for faithful replay) |
| `started_at`, `ended_at` | ISO-8601 UTC timestamps |
| `stdin` | FIFO for `steer` (only when launched with `-i`) |
| `interactive` | `1` if launched with `-i`, else `0` |

### Why `setsid` + FIFO?

- `setsid -f` makes each run its own session/process-group leader so
  `kill <name>` reliably reaps the agent plus `tee` plus any children.
- On interactive runs, the launcher holds **both** a reader and a writer fd
  on the FIFO. Without the writer-fd, the first `steer` closes the only
  writer and the agent sees EOF on stdin and exits after a single message.
- Argv is persisted NUL-delimited and replayed via `exec` instead of `eval`,
  so commands with quoted arguments or shell syntax (e.g.
  `bash -c 'for i in ...; do ...; done'`) work correctly.
- SIGTERM/INT/HUP traps always finalize `status` + `exit_code` + `ended_at`,
  even when the launcher is killed mid-pipeline.

## License

MIT
