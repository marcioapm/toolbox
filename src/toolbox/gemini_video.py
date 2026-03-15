#!/usr/bin/env python3
"""Generate video via Google Veo API (async long-running operation).

Environment:
    GEMINI_API_KEY  — Required. Google AI API key from https://aistudio.google.com/apikey

Models:
    veo-2.0-generate-001         — Veo 2.0
    veo-3.0-generate-001         — Veo 3.0 (high quality)
    veo-3.0-fast-generate-001    — Veo 3.0 Fast (default, good balance)
    veo-3.1-generate-preview     — Veo 3.1 Preview
    veo-3.1-fast-generate-preview — Veo 3.1 Fast Preview

Usage:
    gemini-video "a drone flying over mountains at sunset" -o mountains.mp4
    gemini-video "cat playing piano" -m veo-3.0-generate-001
    gemini-video "timelapse of flowers blooming" --aspect 9:16 -o vertical.mp4
"""

import base64
import json
import sys
import time
import urllib.request

import click

MODELS = [
    "veo-2.0-generate-001",
    "veo-3.0-generate-001",
    "veo-3.0-fast-generate-001",
    "veo-3.1-generate-preview",
    "veo-3.1-fast-generate-preview",
]

BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
POLL_INTERVAL = 5
MAX_POLLS = 120  # 10 minutes


@click.command()
@click.argument("prompt")
@click.option("-o", "--output", default="output.mp4", show_default=True, help="Output file path.")
@click.option("-m", "--model", default="veo-3.0-fast-generate-001", show_default=True,
              type=click.Choice(MODELS, case_sensitive=False), help="Veo model.")
@click.option("--aspect", default="16:9", show_default=True, help="Aspect ratio (e.g. 16:9, 9:16, 1:1).")
@click.option("--api-key", envvar="GEMINI_API_KEY", required=True, help="Gemini API key [env: GEMINI_API_KEY].")
def main(prompt, output, model, aspect, api_key):
    """Generate video via Google Veo API."""
    # Submit async generation
    url = f"{BASE_URL}/{model}:predictLongRunning?key={api_key}"
    payload = {
        "instances": [{"prompt": prompt}],
        "parameters": {"aspectRatio": aspect, "sampleCount": 1},
    }

    click.echo(f"Submitting to {model}...")
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    try:
        resp = urllib.request.urlopen(req, timeout=30)
        data = json.load(resp)
    except urllib.error.HTTPError as e:
        raise click.ClickException(f"API error: {e.read().decode()}")

    op_name = data.get("name")
    if not op_name:
        raise click.ClickException(f"No operation name returned.\n{json.dumps(data, indent=2)[:500]}")

    click.echo(f"Operation: {op_name}")
    click.echo("Polling for completion...")

    # Poll until done
    poll_url = f"https://generativelanguage.googleapis.com/v1beta/{op_name}?key={api_key}"
    for i in range(MAX_POLLS):
        time.sleep(POLL_INTERVAL)
        try:
            poll_resp = urllib.request.urlopen(urllib.request.Request(poll_url), timeout=30)
            poll_data = json.load(poll_resp)
        except Exception as e:
            click.echo(f"  Poll error: {e}", err=True)
            continue

        if poll_data.get("done"):
            click.echo("Generation complete!")
            result = poll_data.get("response", poll_data.get("result", poll_data))
            videos = (
                result.get("generateVideoResponse", {}).get("generatedSamples")
                or result.get("generatedVideos")
                or result.get("videos")
                or []
            )

            for vid in videos:
                video_obj = vid.get("video", vid)
                uri = video_obj.get("uri", "")

                if uri:
                    click.echo("Downloading...")
                    dl_url = uri + ("&key=" + api_key if "?" in uri else "?key=" + api_key)
                    dl_resp = urllib.request.urlopen(urllib.request.Request(dl_url), timeout=120)
                    vid_bytes = dl_resp.read()
                elif "bytesBase64Encoded" in video_obj:
                    vid_bytes = base64.b64decode(video_obj["bytesBase64Encoded"])
                else:
                    continue

                with open(output, "wb") as f:
                    f.write(vid_bytes)
                click.echo(f"Saved {output} ({len(vid_bytes)} bytes)")
                return

            raise click.ClickException(f"No video data in response.\n{json.dumps(result, indent=2)[:800]}")

        elapsed = (i + 1) * POLL_INTERVAL
        click.echo(f"  Waiting... ({elapsed}s)")

    raise click.ClickException("Timeout after 10 minutes")


if __name__ == "__main__":
    main()
