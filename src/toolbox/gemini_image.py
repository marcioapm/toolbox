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

import argparse
import base64
import json
import os
import sys
import urllib.request

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


def get_api_key():
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        print("Error: GEMINI_API_KEY not set. Get one at https://aistudio.google.com/apikey", file=sys.stderr)
        sys.exit(1)
    return key


def generate_imagen(api_key, model, prompt, count, aspect):
    """Generate via Imagen predict endpoint."""
    url = f"{BASE_URL}/{model}:predict?key={api_key}"
    payload = {
        "instances": [{"prompt": prompt}],
        "parameters": {"sampleCount": min(count, 4), "aspectRatio": aspect},
    }
    return _api_call(url, payload)


def generate_gemini_native(api_key, model, prompt):
    """Generate via Gemini generateContent endpoint (native image models)."""
    url = f"{BASE_URL}/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": f"Generate an image: {prompt}"}]}],
        "generationConfig": {"responseModalities": ["image", "text"]},
    }
    return _api_call(url, payload)


def _api_call(url, payload):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    try:
        resp = urllib.request.urlopen(req, timeout=120)
        return json.load(resp)
    except urllib.error.HTTPError as e:
        print(f"API error: {e.read().decode()}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def save_images(data, output, is_gemini_native):
    """Extract and save images from API response."""
    count = 0
    base, ext = os.path.splitext(output)

    if is_gemini_native:
        for candidate in data.get("candidates", []):
            for part in candidate.get("content", {}).get("parts", []):
                if "inlineData" in part:
                    img_bytes = base64.b64decode(part["inlineData"]["data"])
                    mime = part["inlineData"].get("mimeType", "image/png")
                    file_ext = ".jpg" if "jpeg" in mime else ".png"
                    count += 1
                    out = output if count == 1 else f"{base}-{count}{file_ext}"
                    with open(out, "wb") as f:
                        f.write(img_bytes)
                    print(f"Saved {out} ({len(img_bytes)} bytes)")
    else:
        for pred in data.get("predictions", []):
            img_bytes = base64.b64decode(pred["bytesBase64Encoded"])
            count += 1
            out = output if count == 1 else f"{base}-{count}{ext or '.png'}"
            with open(out, "wb") as f:
                f.write(img_bytes)
            print(f"Saved {out} ({len(img_bytes)} bytes)")

    if count == 0:
        print("No images generated", file=sys.stderr)
        print(json.dumps(data, indent=2)[:500], file=sys.stderr)
        sys.exit(1)

    return count


def main():
    parser = argparse.ArgumentParser(
        prog="gemini-image",
        description="Generate images via Google Gemini API",
    )
    parser.add_argument("prompt", help="Image generation prompt")
    parser.add_argument("-o", "--output", default="output.png", help="Output file (default: output.png)")
    parser.add_argument(
        "-m", "--model", default="imagen-4.0-generate-001", choices=MODELS,
        help="Model (default: imagen-4.0-generate-001)",
    )
    parser.add_argument("-n", "--count", type=int, default=1, help="Number of images, 1-4 (default: 1)")
    parser.add_argument("--aspect", default="1:1", help="Aspect ratio (default: 1:1)")
    args = parser.parse_args()

    api_key = get_api_key()
    is_native = args.model in GEMINI_NATIVE_MODELS

    if is_native:
        data = generate_gemini_native(api_key, args.model, args.prompt)
    else:
        data = generate_imagen(api_key, args.model, args.prompt, args.count, args.aspect)

    save_images(data, args.output, is_native)


if __name__ == "__main__":
    main()
