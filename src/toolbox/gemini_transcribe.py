#!/usr/bin/env python3
"""Audio transcription via the Gemini API.

Environment:
    GEMINI_API_KEY  — Required. Google AI API key from https://aistudio.google.com/apikey

Models:
    gemini-2.5-flash — Fast, cheap (default)
    gemini-2.5-pro   — More accurate, slower / pricier
    Any other model name is also accepted (free-form).

Usage:
    gemini-transcribe meeting.mp3
    gemini-transcribe call.ogg -o transcript.txt
    gemini-transcribe lecture.wav -m gemini-2.5-pro --language Portuguese
    gemini-transcribe note.m4a --json
"""

import base64
import json
import os
import sys
import urllib.error
import urllib.request

import click

BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
INLINE_LIMIT_BYTES = 19 * 1024 * 1024  # 19 MB; Gemini inline_data limit is 20 MB

EXT_TO_MIME = {
    ".ogg": "audio/ogg",
    ".opus": "audio/ogg",
    ".mp3": "audio/mp3",
    ".wav": "audio/wav",
    ".m4a": "audio/mp4",
    ".flac": "audio/flac",
    ".aac": "audio/aac",
    ".webm": "audio/webm",
}

DEFAULT_PROMPT = "Transcribe this audio verbatim. Output only the transcript text, no preamble."


def _mime_for(path):
    ext = os.path.splitext(path)[1].lower()
    mime = EXT_TO_MIME.get(ext)
    if not mime:
        raise click.ClickException(
            f"Unsupported audio extension '{ext}'. Supported: {', '.join(sorted(EXT_TO_MIME))}"
        )
    return mime


@click.command()
@click.argument("audio_path", type=click.Path(exists=False, dir_okay=False))
@click.option("-o", "--output", type=click.Path(dir_okay=False), default=None,
              help="Write transcript to file (default: stdout).")
@click.option("-m", "--model", default="gemini-2.5-flash", show_default=True,
              help="Gemini model (free-form; e.g. gemini-2.5-flash, gemini-2.5-pro).")
@click.option("--prompt", default=DEFAULT_PROMPT, show_default=False,
              help="Custom transcription prompt.")
@click.option("--language", default=None,
              help='Optional language hint, e.g. "Portuguese". Appended to prompt.')
@click.option("--json", "as_json", is_flag=True,
              help="Output the full JSON response instead of just the transcript text.")
@click.option("--api-key", envvar="GEMINI_API_KEY", required=True,
              help="Gemini API key [env: GEMINI_API_KEY].")
def main(audio_path, output, model, prompt, language, as_json, api_key):
    """Transcribe an audio file via the Gemini API."""
    if not os.path.exists(audio_path):
        click.echo(f"Error: file not found: {audio_path}", err=True)
        sys.exit(1)

    try:
        size = os.path.getsize(audio_path)
        if size > INLINE_LIMIT_BYTES:
            click.echo(
                f"Error: {audio_path} is {size / 1024 / 1024:.1f} MB, exceeds inline limit "
                f"of {INLINE_LIMIT_BYTES / 1024 / 1024:.0f} MB. Use the Files API for larger files.",
                err=True,
            )
            sys.exit(1)

        mime = _mime_for(audio_path)

        with open(audio_path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode()

        full_prompt = prompt
        if language:
            full_prompt = f"{prompt}\n\nLanguage: {language}."

        url = f"{BASE_URL}/{model}:generateContent?key={api_key}"
        payload = {
            "contents": [{"parts": [
                {"text": full_prompt},
                {"inline_data": {"mime_type": mime, "data": audio_b64}},
            ]}]
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            resp = urllib.request.urlopen(req, timeout=300)
            data = json.load(resp)
        except urllib.error.HTTPError as e:
            click.echo(f"API error: {e.read().decode()}", err=True)
            sys.exit(1)
        except urllib.error.URLError as e:
            click.echo(f"Network error: {e}", err=True)
            sys.exit(1)

        if as_json:
            payload_out = json.dumps(data, indent=2)
            if output:
                with open(output, "w") as f:
                    f.write(payload_out)
            else:
                click.echo(payload_out)
            return

        try:
            text = data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError):
            click.echo(
                f"Error: no transcript in response. Full response:\n{json.dumps(data, indent=2)}",
                err=True,
            )
            sys.exit(1)

        if output:
            with open(output, "w") as f:
                f.write(text)
        else:
            click.echo(text)

    except click.ClickException:
        raise
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
