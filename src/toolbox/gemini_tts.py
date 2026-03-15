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

import base64
import json
import os
import sys
import urllib.request
import wave

import click

MODELS = ["gemini-2.5-flash-preview-tts", "gemini-2.5-pro-preview-tts"]
VOICES = ["Aoede", "Charon", "Fenrir", "Kore", "Puck", "Orbit", "Vale"]
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"


@click.command()
@click.argument("text")
@click.option("-o", "--output", default="output.wav", show_default=True, help="Output WAV file path.")
@click.option("-m", "--model", default="gemini-2.5-flash-preview-tts", show_default=True,
              type=click.Choice(MODELS, case_sensitive=False), help="TTS model.")
@click.option("-v", "--voice", default="Kore", show_default=True,
              type=click.Choice(VOICES, case_sensitive=False), help="Voice name.")
@click.option("--api-key", envvar="GEMINI_API_KEY", required=True, help="Gemini API key [env: GEMINI_API_KEY].")
def main(text, output, model, voice, api_key):
    """Text-to-speech via Gemini native audio."""
    url = f"{BASE_URL}/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": text}]}],
        "generationConfig": {
            "responseModalities": ["audio"],
            "speechConfig": {
                "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}
            },
        },
    }

    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    try:
        resp = urllib.request.urlopen(req, timeout=60)
        data = json.load(resp)
    except urllib.error.HTTPError as e:
        raise click.ClickException(f"API error: {e.read().decode()}")
    except Exception as e:
        raise click.ClickException(str(e))

    for candidate in data.get("candidates", []):
        for part in candidate.get("content", {}).get("parts", []):
            if "inlineData" in part:
                audio_bytes = base64.b64decode(part["inlineData"]["data"])

                if audio_bytes[:4] == b"RIFF":
                    with open(output, "wb") as f:
                        f.write(audio_bytes)
                else:
                    with wave.open(output, "wb") as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(24000)
                        wf.writeframes(audio_bytes)

                click.echo(f"Saved {output} ({os.path.getsize(output)} bytes)")
                return

    raise click.ClickException("No audio in response")


if __name__ == "__main__":
    main()
