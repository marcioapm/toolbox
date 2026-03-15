#!/usr/bin/env python3
"""Generate images via Google Gemini API (Imagen 4.0 / Gemini native image models).

Environment:
    GEMINI_API_KEY  — Required. Google AI API key from https://aistudio.google.com/apikey

Models:
    imagen-4.0-generate-001       — Imagen 4.0 (default, best quality)
    imagen-4.0-ultra-generate-001 — Imagen 4.0 Ultra (highest quality, slower)
    imagen-4.0-fast-generate-001  — Imagen 4.0 Fast (quickest)
    nano-banana-pro-preview       — Nano Banana Pro (Gemini native)
    gemini-3-pro-image-preview    — Gemini 3 Pro Image
    gemini-3.1-flash-image-preview — Gemini 3.1 Flash Image (fastest native)

Usage:
    gemini-image "a cat riding a skateboard" -o cat.png
    gemini-image "logo for a tech startup" -m imagen-4.0-fast-generate-001 -n 4
    gemini-image "watercolor landscape" -m gemini-3-pro-image-preview --aspect 16:9
"""

import base64
import json
import os
import sys
import urllib.request

import click

MODELS = [
    "imagen-4.0-generate-001",
    "imagen-4.0-ultra-generate-001",
    "imagen-4.0-fast-generate-001",
    "nano-banana-pro-preview",
    "gemini-3-pro-image-preview",
    "gemini-3.1-flash-image-preview",
]

GEMINI_NATIVE_MODELS = {
    "nano-banana-pro-preview",
    "gemini-3-pro-image-preview",
    "gemini-3.1-flash-image-preview",
}

BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"


def _api_call(url, payload):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    try:
        resp = urllib.request.urlopen(req, timeout=120)
        return json.load(resp)
    except urllib.error.HTTPError as e:
        raise click.ClickException(f"API error: {e.read().decode()}")
    except Exception as e:
        raise click.ClickException(str(e))


@click.command()
@click.argument("prompt")
@click.option("-o", "--output", default="output.png", show_default=True, help="Output file path.")
@click.option("-m", "--model", default="imagen-4.0-generate-001", show_default=True,
              type=click.Choice(MODELS, case_sensitive=False), help="Generation model.")
@click.option("-n", "--count", default=1, show_default=True, type=click.IntRange(1, 4), help="Number of images.")
@click.option("--aspect", default="1:1", show_default=True, help="Aspect ratio (e.g. 16:9, 9:16, 1:1).")
@click.option("--api-key", envvar="GEMINI_API_KEY", required=True, help="Gemini API key [env: GEMINI_API_KEY].")
def main(prompt, output, model, count, aspect, api_key):
    """Generate images via Google Gemini API."""
    is_native = model in GEMINI_NATIVE_MODELS

    if is_native:
        url = f"{BASE_URL}/{model}:generateContent?key={api_key}"
        payload = {
            "contents": [{"parts": [{"text": f"Generate an image: {prompt}"}]}],
            "generationConfig": {"responseModalities": ["image", "text"]},
        }
    else:
        url = f"{BASE_URL}/{model}:predict?key={api_key}"
        payload = {
            "instances": [{"prompt": prompt}],
            "parameters": {"sampleCount": count, "aspectRatio": aspect},
        }

    data = _api_call(url, payload)

    saved = 0
    base, ext = os.path.splitext(output)

    if is_native:
        for candidate in data.get("candidates", []):
            for part in candidate.get("content", {}).get("parts", []):
                if "inlineData" in part:
                    img_bytes = base64.b64decode(part["inlineData"]["data"])
                    mime = part["inlineData"].get("mimeType", "image/png")
                    file_ext = ".jpg" if "jpeg" in mime else ".png"
                    saved += 1
                    out = output if saved == 1 else f"{base}-{saved}{file_ext}"
                    with open(out, "wb") as f:
                        f.write(img_bytes)
                    click.echo(f"Saved {out} ({len(img_bytes)} bytes)")
    else:
        for pred in data.get("predictions", []):
            img_bytes = base64.b64decode(pred["bytesBase64Encoded"])
            saved += 1
            out = output if saved == 1 else f"{base}-{saved}{ext or '.png'}"
            with open(out, "wb") as f:
                f.write(img_bytes)
            click.echo(f"Saved {out} ({len(img_bytes)} bytes)")

    if saved == 0:
        raise click.ClickException(f"No images generated.\n{json.dumps(data, indent=2)[:500]}")


if __name__ == "__main__":
    main()
