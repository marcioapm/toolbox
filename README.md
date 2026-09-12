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
| `opencode-gc` | Prune old opencode sessions and release freed SQLite pages back to the filesystem | — |

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
```

## License

MIT

## opencode-gc

opencode's SQLite store never prunes finished sessions. Measured on one host:
**76.6 GB across 3,164 sessions** (2.30M `event` rows), growing ~6 GB/day.

```bash
opencode-gc                              # dry run, 4-day retention
opencode-gc --apply                      # delete + reclaim + checkpoint
opencode-gc --retention-days 14 --apply
opencode-gc --apply --rebuild            # also compact the file, if idle
opencode-gc --apply --enable-incremental-vacuum   # first run on a new host
opencode-gc --json                       # machine-readable
```

### Why a session delete is not enough

Every foreign key that references `session(id)`, plus the two event tables that
key on a session id with no foreign key at all:

```
message.session_id               -> session.id   ON DELETE CASCADE
todo.session_id                  -> session.id   ON DELETE CASCADE
session_message.session_id       -> session.id   ON DELETE CASCADE
session_input.session_id         -> session.id   ON DELETE CASCADE
session_share.session_id         -> session.id   ON DELETE CASCADE
session_context_epoch.session_id -> session.id   ON DELETE CASCADE
part.message_id                  -> message.id   ON DELETE CASCADE
event.aggregate_id -> event_sequence.aggregate_id ON DELETE CASCADE
event_sequence                   -> (nothing)
```

`event_sequence` has **no** foreign key to `session` — its `aggregate_id` merely
happens to equal a session id. So `opencode session delete` (or a plain
`DELETE FROM session`) strands every event row, which is the bulk of the file.
`PRAGMA foreign_keys` is also off by default, so the cascades above do not fire
unless enabled. This tool deletes each table explicitly, children first.

That list being complete is what makes deleting with foreign keys off equivalent
to deleting with the cascades on, so the tool checks it against the schema and
**refuses to run** if the database has a session child it does not know about.
A future opencode migration adding one would otherwise orphan its rows silently.

### Checkpointing the WAL

In WAL mode a page released by `incremental_vacuum` does not leave the file until
a checkpoint folds the WAL back into it — and an uncheckpointed WAL is itself on
the disk. One host had accumulated **15.28 GiB** of WAL that had never been
checkpointed; a single `wal_checkpoint(TRUNCATE)` folded it in 2.5 seconds.

Every `--apply` run therefore checkpoints, and the mode depends on who else has
the database open. `TRUNCATE` and `RESTART` wait for readers and block writers
while they hold the WAL; `PASSIVE` never blocks. opencode instances are writers,
and a writer that exhausts its own `busy_timeout` behind this tool dies with
`Error: Failed to execute statement`. So `TRUNCATE` runs only when `lsof` reports
that **nothing** holds the database, and `PASSIVE` runs in every other case —
including when holders cannot be determined at all.

### Reclaiming space: incremental, and `--rebuild`

A plain `VACUUM` copies the database to a temporary file and then overwrites the
original under a journal, so SQLite documents it as needing up to **twice** the
file size in free space — impossible at 76 GB on a full disk. `PRAGMA
auto_vacuum=2` (INCREMENTAL) lets `PRAGMA incremental_vacuum(N)` hand pages back
in bounded chunks with no rewrite and no large temp file.

Incremental reclamation is a trickle, not a reclaim path: it relocates pages one
at a time with pointer-map updates, measured at **~10–20 MB/min** (2,141 pages in
60.7s on one host, 1,891 in 62.1s on another). Draining a 55 GiB freelist at that
rate would take 90–108 hours. It keeps a pruned database from growing; it will
not shrink one that already has.

`--rebuild` is what shrinks it. `VACUUM INTO` writes only the compacted copy, so
it needs the **live** size plus ~5% rather than 2x the file — which is why a
32 GiB file holding 6.2 GiB of live data can be rebuilt on 13 GiB of free disk
where a plain `VACUUM` of the same file needs ~64 GiB and is rightly refused.

It is heavily guarded, because the rebuild is safe but the **swap** is what
loses data. The cutover does **not** rest on holder snapshots: `lsof` answers
"who was attached at instant T", and that is not the question. A writer that
opens after a check, commits after `VACUUM INTO` took its snapshot, and exits
before the next check is invisible to *both* while its transaction is absent
from the copy — so the swap installed a database missing a committed row.
Reproduced. No number of extra checks closes it, because every one of them
samples process liveness and liveness is not the property at issue.

Three mechanisms replace the sampling, each answering a question a snapshot
cannot:

- **`PRAGMA data_version` — did anyone commit?** It changes whenever *another*
  connection commits, and is stable across our own writes and our own
  `VACUUM INTO`. Read on a connection held open for the whole rebuild and again
  under the lock below; any change discards the copy. Process liveness is
  irrelevant to it.
- **`BEGIN EXCLUSIVE`, held across the rename — can anyone commit *now*?** In
  WAL mode it blocks other connections' writes while leaving readers alone, and
  it survives `os.replace`: a writer queued against it stays queued until this
  connection releases. That closes the window between the last look and the
  rename, which is the window no check can cover. `VACUUM INTO` cannot run
  inside a transaction, so the lock is taken *after* the rebuild, for the
  cutover only.
- **`lsof` on the backup name — is anyone still holding the *old* inode?** The
  one snapshot that is not a race. The old database is hard-linked aside before
  the rename, so afterwards the old inode is reachable through exactly one
  pathname that nothing else on the host knows. The set of processes holding it
  is therefore **closed** — it cannot grow, so observing it empty is a proof
  rather than a sample. A straggler found there sends the rename back, restoring
  the original at the live path with its writes intact.

`lsof` is still consulted first, but only as a cheap **pre-filter** so a busy
host does not pay for a 20-minute rebuild the cutover will refuse. It no longer
authorises anything. A missing `lsof` still reads as *unknown*, never *idle*:
launchd and systemd start jobs with a bare environment and macOS keeps it in
`/usr/sbin`. Any diagnostic on stderr, unexpected exit status or unparseable
output is likewise *unknown* — a missing `-shm` is routine, and asking `lsof`
about it earned a diagnostic that read as "nobody holds this".

The rest of the protocol:

- **Nothing is destroyed before the replacement is in place.** The source WAL is
  folded first so the preserved old file is self-contained, the old database is
  hard-linked aside, the new file is installed, the file and its directory are
  synced, and only then is the old set removed. Every partial-failure path — a
  failed `link`, `replace` or `fsync` — leaves a database that still opens at
  the live pathname with all committed rows. The previous order unlinked the
  `-wal` *before* `os.replace`, so a handled `OSError` from the rename destroyed
  every WAL-only-committed transaction.
- **The copy is verified before it is trusted**: `quick_check` ok, `auto_vacuum`
  still INCREMENTAL, and the source's **journal mode** and **permissions**
  established on it and read back from a fresh connection. `VACUUM INTO` writes
  its output in the default `DELETE` mode and at the process umask whatever the
  source used — measured, a `0600` WAL source produced a `0644` `DELETE` copy.
  Either would be swapped in silently: one changes opencode's concurrency model,
  the other publishes session history to every local user. Both now fail closed.
- **One rebuild at a time**, via `flock` on a lock file beside the database. Two
  invocations share one `.rebuild-tmp` and would destroy each other's copy
  mid-write; a manual run landing on a timed one is the ordinary way that
  happens. The lock is advisory and per-open-file-description, so the kernel
  drops it however abruptly the holder dies.
- The call is bounded from *inside* SQLite by a wall-clock cap and a free-space
  floor, via `set_progress_handler`. `--max-seconds` cannot bound `VACUUM INTO`,
  which is one uninterruptible call: an unbounded one ran **23 minutes**, wrote a
  **39.6 GB** temp copy and drove a disk from 88% to 93% before it was killed by
  hand. The partial copy SQLite leaves behind is unlinked on every abort path.
  Note the cap is enforced at the next progress callback: a statement blocked in
  filesystem I/O runs no callbacks and can overshoot it.
- A guard that refuses or aborts is a **skip, not an error**: the database is
  untouched and a later run may succeed, so the exit status stays 0. That
  includes a refused cutover — a safe no-op beats a hopeful swap.

Files it may leave beside the database, and what to do about them:

| File | Meaning |
| --- | --- |
| `opencode.db.rebuild-lock` | empty; always present after one rebuild. Ignore. |
| `opencode.db.rebuild-tmp` | a partial copy from an interrupted rebuild. Safe to delete. |
| `opencode.db.rebuild-old` | **the previous database.** A rebuild was interrupted mid-swap. Compare it against the live file and remove it by hand; until then every rebuild refuses. |

`--enable-incremental-vacuum` switches a database to INCREMENTAL. From
`auto_vacuum=FULL` this is a header change and costs nothing. From
`auto_vacuum=NONE` it costs one full `VACUUM`, so it refuses unless the
filesystem holding the database has 2x the database size (including its WAL)
plus a reserve free, and unless SQLite's temp filesystem, when it is a different
one, has room for a copy. Run it once per host, ideally before the file gets
large.

### Safety

- Dry run by default; `--apply` is required to delete anything. Dry-run row
  counts are a point-in-time estimate, reported with the cutoff they used.
- A session is only expired when it **and every descendant** are older than the
  retention window — `session.parent_id` has no foreign key, so deleting a
  parent out from under a live child would leave a dangling reference.
- Eligibility is decided **again inside each write transaction**. The database is
  live, so a session opencode touched (or gave a live child) after the selection
  pass is skipped rather than deleted.
- The **schema guard is re-checked inside each batch's transaction**, not once
  before the run. Deleting with `PRAGMA foreign_keys` off is only equivalent to
  deleting with the cascades on while the table list is complete, and the write
  lock is released between batches — so an opencode migration adding a session
  child can land after an unlocked check and be orphaned by every batch after it.
  A schema that changes mid-run rolls that batch back and stops, reporting what
  was already committed. The check covers the **transitive** closure: `message`,
  `part`, `event` and `event_sequence` are deleted explicitly too, so a table
  hanging off any of them is orphaned exactly as surely as a direct child of
  `session`. Identifiers are compared case-insensitively, since
  `REFERENCES SeSsIoN(id)` is valid SQLite. `delete_sessions` enforces this
  itself rather than trusting its caller to have done so.
- A `NULL time_updated` is an unknown age, not an infinite one: such sessions are
  kept and reported.
- `--retention-days` below 1 is refused; this deletes irreplaceable history.
- Deletes run in batches inside transactions, ordered deepest-descendant-first,
  so opencode can keep running and an interrupted run never leaves a session
  pointing at a deleted parent. Sessions in a `parent_id` cycle have no safe
  order and are retained.
- Between batches the write lock is handed back deliberately: a PASSIVE
  checkpoint folds what the batch wrote, then `--batch-sleep-ms` pauses before
  the next transaction. Writers serialise, so this is what gives a queued
  opencode instance a window to win the lock instead of timing out.
- `PRAGMA journal_size_limit` is set on this tool's own connection so its
  transactions cannot leave a huge WAL behind. It is per-connection and does not
  affect opencode's own connections.
- If `--enable-incremental-vacuum` is requested and the conversion fails, nothing
  is deleted.
- Committed batches cannot be undone, so a deadline or a lock/IO failure part-way
  through still prints a full result — committed counts, `incomplete: true`, and
  how many eligible sessions were left — instead of a traceback.
- `--max-seconds` stops *starting* new batches; it is not a bound on total
  runtime, since a batch or a `VACUUM` already in flight runs to completion.
  `--vacuum-pages` bounds page reclamation, and `--rebuild-max-seconds` is the
  only thing that can bound a `VACUUM INTO`. All reject NaN and out-of-range
  values rather than silently disabling themselves.
- Released pages and reclaimed bytes are reported separately: in WAL mode a
  long-lived reader can defer the checkpoint that actually shrinks the file, so
  bytes are measured from the real database and `-wal` file sizes.
- `--db` opens exactly the named file. A path containing `?`, `#` or `%` is not
  reparsed as URI syntax, which would otherwise point the tool at a neighbouring
  database.

### Exit status

| Code | Meaning |
|------|---------|
| `0` | completed (a skipped `--rebuild` is still a completed run) |
| `1` | an error occurred (nothing deleted, or a partial delete that is reported) |
| `2` | bad arguments, no database at `--db`, or a schema this tool cannot safely prune |
| `3` | no error, but a deadline left eligible sessions unprocessed; re-run to continue |

| Flag | Default | Meaning |
|------|---------|---------|
| `--db` | `~/.local/share/opencode/opencode.db` | database path |
| `--retention-days` | `4` | keep sessions updated within this window |
| `--apply` | off | actually delete |
| `--batch` | `25` | sessions per transaction (clamped to SQLite's variable limit) |
| `--batch-sleep-ms` | `1000` | pause between batches, yielding the write lock (0 disables) |
| `--max-seconds` | `600` | stop starting new batches after this long (0 = no limit) |
| `--vacuum-pages` | all | cap pages released per run (>= 1) |
| `--no-vacuum` | off | delete rows but do not release pages |
| `--enable-incremental-vacuum` | off | switch `auto_vacuum` to INCREMENTAL |
| `--rebuild` | off | compact with `VACUUM INTO` when nothing holds the database |
| `--rebuild-max-seconds` | `900` | abort the rebuild at the first progress callback after this long |
| `--rebuild-min-free-gib` | `25` | refuse/abort a rebuild that would leave less free |
