#!/usr/bin/env python3
"""Text-to-speech via Gemini native audio models.

Environment:
    GEMINI_API_KEY  — Required. Google AI API key from https://aistudio.google.com/apikey

Models:
    gemini-2.5-flash-preview-tts — Flash TTS (default, fast)
    gemini-2.5-pro-preview-tts   — Pro TTS (more expressive, slower)

Voices:
    Aoede, Charon, Fenrir, Kore (default), Puck, Orbit, Vale

Usage:
    gemini-tts "Hello, world!" -o hello.wav
    gemini-tts "Tell me a story" -v Aoede -m gemini-2.5-pro-preview-tts
    gemini-tts "Breaking news" -v Charon -o news.wav
"""

import argparse
import base64
import json
import os
import sys
import urllib.request
import wave

MODELS = ["gemini-2.5-flash-preview-tts", "gemini-2.5-pro-preview-tts"]
VOICES = ["Aoede", "Charon", "Fenrir", "Kore", "Puck", "Orbit", "Vale"]
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"


def get_api_key():
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        print("Error: GEMINI_API_KEY not set. Get one at https://aistudio.google.com/apikey", file=sys.stderr)
        sys.exit(1)
    return key


def generate_speech(api_key, model, text, voice):
    """Call Gemini TTS API and return audio bytes."""
    url = f"{BASE_URL}/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": text}]}],
        "generationConfig": {
            "responseModalities": ["audio"],
            "speechConfig": {
                "voiceConfig": {
                    "prebuiltVoiceConfig": {"voiceName": voice}
                }
            },
        },
    }

    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    try:
        resp = urllib.request.urlopen(req, timeout=60)
        data = json.load(resp)
    except urllib.error.HTTPError as e:
        print(f"API error: {e.read().decode()}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    for candidate in data.get("candidates", []):
        for part in candidate.get("content", {}).get("parts", []):
            if "inlineData" in part:
                return base64.b64decode(part["inlineData"]["data"])

    print("No audio in response", file=sys.stderr)
    sys.exit(1)


def save_audio(audio_bytes, output):
    """Save audio bytes as WAV, wrapping raw PCM if needed."""
    if audio_bytes[:4] == b"RIFF":
        # Already a valid WAV
        with open(output, "wb") as f:
            f.write(audio_bytes)
    else:
        # Raw PCM — wrap in WAV container (16-bit LE, 24kHz mono)
        with wave.open(output, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            wf.writeframes(audio_bytes)

    print(f"Saved {output} ({os.path.getsize(output)} bytes)")


def main():
    parser = argparse.ArgumentParser(
        prog="gemini-tts",
        description="Text-to-speech via Gemini native audio",
    )
    parser.add_argument("text", help="Text to speak")
    parser.add_argument("-o", "--output", default="output.wav", help="Output file (default: output.wav)")
    parser.add_argument(
        "-m", "--model", default="gemini-2.5-flash-preview-tts", choices=MODELS,
        help="TTS model (default: gemini-2.5-flash-preview-tts)",
    )
    parser.add_argument(
        "-v", "--voice", default="Kore", choices=VOICES,
        help="Voice name (default: Kore)",
    )
    args = parser.parse_args()

    api_key = get_api_key()
    audio_bytes = generate_speech(api_key, args.model, args.text, args.voice)
    save_audio(audio_bytes, args.output)


if __name__ == "__main__":
    main()
