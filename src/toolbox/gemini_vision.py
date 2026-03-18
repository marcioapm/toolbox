#!/usr/bin/env python3
"""Analyze images and videos via Gemini Vision.

Supports local files, URLs, and social media links (YouTube, Instagram, TikTok,
X/Twitter, Vimeo, Facebook, Reddit). Videos from social platforms are auto-downloaded
via yt-dlp, uploaded to Gemini Files API, and analyzed.

Environment:
    GEMINI_API_KEY  — Required. Google AI API key from https://aistudio.google.com/apikey

Requirements:
    yt-dlp          — Required for social media video downloads (brew install yt-dlp)

Models:
    gemini-2.5-flash       — Fast, good quality (default)
    gemini-2.5-pro         — Best quality, slower
    gemini-3-pro-preview   — Gemini 3 Pro
    gemini-3-flash-preview — Gemini 3 Flash
    gemini-3.1-pro-preview — Gemini 3.1 Pro (latest)

Usage:
    gemini-vision photo.jpg
    gemini-vision screenshot.png -p "What's the error in this screenshot?"
    gemini-vision video.mp4 -p "Transcribe the speech in this video"
    gemini-vision "https://youtube.com/watch?v=..." -p "Summarize this video"
    gemini-vision "https://instagram.com/reel/..." -p "Describe what happens"
    gemini-vision "https://x.com/user/status/123" -p "What's in this post's video?"
    gemini-vision photo.jpg -m gemini-2.5-pro -p "Analyze the composition"
"""

import base64
import json
import mimetypes
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request

import click

MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-3-pro-preview",
    "gemini-3-flash-preview",
    "gemini-3.1-pro-preview",
]

VIDEO_EXTS = {".mp4", ".webm", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".m4v"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff"}

SOCIAL_PATTERNS = [
    r"youtube\.com/watch", r"youtu\.be/", r"youtube\.com/shorts",
    r"instagram\.com/(p|reel|tv)/", r"tiktok\.com/",
    r"(twitter|x)\.com/.+/status/", r"vimeo\.com/",
    r"facebook\.com/.+/videos/", r"reddit\.com/.+/comments/",
]

BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


def _is_social_url(url):
    return any(re.search(p, url) for p in SOCIAL_PATTERNS)


def _download_video(url):
    """Download video using yt-dlp, return local path."""
    tmp = tempfile.mktemp(suffix=".mp4", dir="/tmp")
    click.echo("Downloading video...", err=True)
    cmd = [
        "yt-dlp", "-f", "best[ext=mp4][filesize<50M]/best[ext=mp4]/best[filesize<50M]/best",
        "--max-filesize", "50M", "-o", tmp, "--no-playlist", "--quiet", url,
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=False)
    except FileNotFoundError:
        raise click.ClickException("yt-dlp not found. Install it: brew install yt-dlp")
    except subprocess.TimeoutExpired:
        raise click.ClickException("Download timed out (120s)")

    for candidate in [tmp, tmp + ".mp4"]:
        if os.path.exists(candidate):
            return candidate
    raise click.ClickException("Download failed: output file not found")


def _upload_video(filepath, api_key):
    """Upload video via Gemini Files API, return (file_uri, mime_type)."""
    mime = mimetypes.guess_type(filepath)[0] or "video/mp4"
    file_size = os.path.getsize(filepath)
    click.echo(f"Uploading video ({file_size / 1024 / 1024:.1f}MB)...", err=True)

    upload_url = f"{BASE_URL.replace('/v1beta', '/upload/v1beta')}/files?key={api_key}"
    meta = json.dumps({"file": {"display_name": os.path.basename(filepath)}}).encode()

    # Initiate resumable upload
    init_req = urllib.request.Request(upload_url, method="POST", headers={
        "X-Goog-Upload-Protocol": "resumable",
        "X-Goog-Upload-Command": "start",
        "X-Goog-Upload-Header-Content-Length": str(file_size),
        "X-Goog-Upload-Header-Content-Type": mime,
        "Content-Type": "application/json",
    }, data=meta)
    init_resp = urllib.request.urlopen(init_req, timeout=30)
    resume_url = init_resp.headers.get("X-Goog-Upload-URL")

    # Upload file data
    with open(filepath, "rb") as f:
        file_data = f.read()

    upload_req = urllib.request.Request(resume_url, method="PUT", headers={
        "X-Goog-Upload-Command": "upload, finalize",
        "X-Goog-Upload-Offset": "0",
        "Content-Length": str(len(file_data)),
        "Content-Type": mime,
    }, data=file_data)
    upload_resp = urllib.request.urlopen(upload_req, timeout=120)
    result = json.load(upload_resp)

    file_uri = result["file"]["uri"]
    file_name = result["file"]["name"]
    state = result["file"].get("state", "ACTIVE")

    # Wait for processing
    while state == "PROCESSING":
        click.echo("  Processing video...", err=True)
        time.sleep(3)
        check_url = f"{BASE_URL}/{file_name}?key={api_key}"
        check_resp = urllib.request.urlopen(urllib.request.Request(check_url), timeout=30)
        state = json.load(check_resp).get("state", "ACTIVE")

    click.echo("Video ready.", err=True)
    return file_uri, mime


def _resolve_input(file_input, api_key, keep):
    """Resolve input to Gemini content parts. Returns (parts, temp_file_or_none)."""
    is_url = file_input.startswith("http")
    ext = os.path.splitext(file_input.split("?")[0])[1].lower()
    temp_file = None

    if is_url and _is_social_url(file_input):
        temp_file = _download_video(file_input)
        file_uri, mime = _upload_video(temp_file, api_key)
        return [{"fileData": {"mimeType": mime, "fileUri": file_uri}}], temp_file

    if not is_url and ext in VIDEO_EXTS:
        if not os.path.exists(file_input):
            raise click.ClickException(f"File not found: {file_input}")
        file_uri, mime = _upload_video(file_input, api_key)
        return [{"fileData": {"mimeType": mime, "fileUri": file_uri}}], None

    if is_url and ext in VIDEO_EXTS:
        temp_file = tempfile.mktemp(suffix=ext, dir="/tmp")
        click.echo("Downloading video...", err=True)
        urllib.request.urlretrieve(file_input, temp_file)
        file_uri, mime = _upload_video(temp_file, api_key)
        return [{"fileData": {"mimeType": mime, "fileUri": file_uri}}], temp_file

    if is_url:
        mime = mimetypes.guess_type(file_input.split("?")[0])[0] or "image/jpeg"
        return [{"fileData": {"mimeType": mime, "fileUri": file_input}}], None

    # Local image
    if not os.path.exists(file_input):
        raise click.ClickException(f"File not found: {file_input}")
    mime = mimetypes.guess_type(file_input)[0] or "image/jpeg"
    with open(file_input, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()
    return [{"inlineData": {"mimeType": mime, "data": img_b64}}], None


@click.command()
@click.argument("file")
@click.option("-p", "--prompt", default="Describe what you see in detail.", show_default=True,
              help="Analysis prompt.")
@click.option("-m", "--model", default="gemini-2.5-flash", show_default=True,
              type=click.Choice(MODELS, case_sensitive=False), help="Gemini model.")
@click.option("--keep", is_flag=True, help="Keep downloaded video (don't delete temp file).")
@click.option("--api-key", envvar="GEMINI_API_KEY", required=True,
              help="Gemini API key [env: GEMINI_API_KEY].")
def main(file, prompt, model, keep, api_key):
    """Analyze images and videos via Gemini Vision.

    Accepts local files, URLs, or social media links (YouTube, Instagram,
    TikTok, X/Twitter). Videos are auto-downloaded and uploaded for analysis.
    """
    media_parts, temp_file = _resolve_input(file, api_key, keep)

    try:
        url = f"{BASE_URL}/models/{model}:generateContent?key={api_key}"
        parts = [{"text": prompt}] + media_parts
        payload = {"contents": [{"parts": parts}]}

        req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            resp = urllib.request.urlopen(req, timeout=120)
            data = json.load(resp)
        except urllib.error.HTTPError as e:
            raise click.ClickException(f"API error: {e.read().decode()}")
        except Exception as e:
            raise click.ClickException(str(e))

        for candidate in data.get("candidates", []):
            for part in candidate.get("content", {}).get("parts", []):
                if "text" in part:
                    click.echo(part["text"])
    finally:
        if temp_file and not keep and os.path.exists(temp_file):
            os.unlink(temp_file)


if __name__ == "__main__":
    main()
