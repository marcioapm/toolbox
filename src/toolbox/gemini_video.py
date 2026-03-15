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

import argparse
import base64
import json
import os
import sys
import time
import urllib.request

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


def get_api_key():
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        print("Error: GEMINI_API_KEY not set. Get one at https://aistudio.google.com/apikey", file=sys.stderr)
        sys.exit(1)
    return key


def submit_generation(api_key, model, prompt, aspect):
    """Submit async video generation request, return operation name."""
    url = f"{BASE_URL}/{model}:predictLongRunning?key={api_key}"
    payload = {
        "instances": [{"prompt": prompt}],
        "parameters": {"aspectRatio": aspect, "sampleCount": 1},
    }

    print(f"Submitting to {model}...")
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    try:
        resp = urllib.request.urlopen(req, timeout=30)
        data = json.load(resp)
    except urllib.error.HTTPError as e:
        print(f"API error: {e.read().decode()}", file=sys.stderr)
        sys.exit(1)

    op_name = data.get("name")
    if not op_name:
        print("No operation name returned", file=sys.stderr)
        print(json.dumps(data, indent=2)[:500], file=sys.stderr)
        sys.exit(1)

    print(f"Operation: {op_name}")
    return op_name


def poll_for_completion(api_key, op_name):
    """Poll operation until done, return result data."""
    print("Polling for completion...")
    poll_url = f"https://generativelanguage.googleapis.com/v1beta/{op_name}?key={api_key}"

    for i in range(MAX_POLLS):
        time.sleep(POLL_INTERVAL)
        try:
            req = urllib.request.Request(poll_url)
            resp = urllib.request.urlopen(req, timeout=30)
            data = json.load(resp)
        except Exception as e:
            print(f"  Poll error: {e}", file=sys.stderr)
            continue

        if data.get("done"):
            print("Generation complete!")
            return data.get("response", data.get("result", data))

        elapsed = (i + 1) * POLL_INTERVAL
        print(f"  Waiting... ({elapsed}s)")

    print("Timeout after 10 minutes", file=sys.stderr)
    sys.exit(1)


def download_video(api_key, result, output):
    """Extract and save video from result."""
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
            print("Downloading...")
            dl_url = uri + ("&key=" + api_key if "?" in uri else "?key=" + api_key)
            dl_resp = urllib.request.urlopen(urllib.request.Request(dl_url), timeout=120)
            vid_bytes = dl_resp.read()
        elif "bytesBase64Encoded" in video_obj:
            vid_bytes = base64.b64decode(video_obj["bytesBase64Encoded"])
        else:
            print(f"Unknown video format: {json.dumps(vid, indent=2)[:300]}", file=sys.stderr)
            continue

        with open(output, "wb") as f:
            f.write(vid_bytes)
        print(f"Saved {output} ({len(vid_bytes)} bytes)")
        return

    print("No video data in response", file=sys.stderr)
    print(json.dumps(result, indent=2)[:800], file=sys.stderr)
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        prog="gemini-video",
        description="Generate video via Google Veo API",
    )
    parser.add_argument("prompt", help="Video generation prompt")
    parser.add_argument("-o", "--output", default="output.mp4", help="Output file (default: output.mp4)")
    parser.add_argument(
        "-m", "--model", default="veo-3.0-fast-generate-001", choices=MODELS,
        help="Model (default: veo-3.0-fast-generate-001)",
    )
    parser.add_argument("--aspect", default="16:9", help="Aspect ratio (default: 16:9)")
    args = parser.parse_args()

    api_key = get_api_key()
    op_name = submit_generation(api_key, args.model, args.prompt, args.aspect)
    result = poll_for_completion(api_key, op_name)
    download_video(api_key, result, args.output)


if __name__ == "__main__":
    main()
