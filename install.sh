#!/usr/bin/env bash
# Install mmartins-toolbox CLIs
# Usage: curl -sSL https://raw.githubusercontent.com/marcioapm/toolbox/main/install.sh | bash

set -euo pipefail

REPO="https://github.com/marcioapm/toolbox.git"

echo "🛠️  Installing mmartins-toolbox..."

# Check Python version
if ! command -v python3 &>/dev/null; then
    echo "❌ Python 3 is required. Install it first."
    exit 1
fi

PY_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)

if [ "$PY_MAJOR" -lt 3 ] || ([ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]); then
    echo "❌ Python 3.10+ required (found $PY_VERSION)"
    exit 1
fi

echo "✅ Python $PY_VERSION"

# Install via pip
if command -v pipx &>/dev/null; then
    echo "📦 Installing with pipx..."
    pipx install "git+${REPO}" --force
elif command -v uv &>/dev/null; then
    echo "📦 Installing with uv..."
    uv tool install "git+${REPO}" --force
else
    echo "📦 Installing with pip..."
    python3 -m pip install --user "git+${REPO}"
fi

echo ""
echo "✅ Installed! Available commands:"
echo "   gemini-image  — Generate images via Gemini API"
echo "   gemini-tts    — Text-to-speech via Gemini"
echo "   gemini-video  — Generate video via Veo"
echo "   slackcli      — Lightweight Slack client"
echo ""
echo "⚙️  Set your API keys:"
echo "   export GEMINI_API_KEY='your-key'"
echo "   export SLACK_USER_TOKEN='xoxp-your-token'"
