#!/usr/bin/env python3
"""Monitor LLM token usage and quotas across providers.

Checks Anthropic, OpenAI/Codex, and Google Gemini for:
- Rate limits and remaining quota (from API headers)
- Token usage for current billing period
- Session/weekly limits where applicable

Environment:
    ANTHROPIC_API_KEY    — Anthropic API key (sk-ant-api*)
    OPENAI_API_KEY       — OpenAI API key (for usage API, needs org admin)
    OPENAI_ADMIN_KEY     — OpenAI admin key (alternative, for usage endpoint)
    GEMINI_API_KEY       — Google Gemini API key
    OPENCLAW_STATE_DIR   — OpenClaw state directory (default: ~/.openclaw)

The tool also reads OpenClaw's auth-profiles.json and session logs to aggregate
local usage tracking even when provider APIs don't expose billing data.

Usage:
    llm-usage                    # Check all providers
    llm-usage --provider anthropic  # Check only Anthropic
    llm-usage --provider openai     # Check only OpenAI
    llm-usage --provider gemini     # Check only Gemini
    llm-usage --openclaw            # Include OpenClaw session usage stats
    llm-usage --json                # Machine-readable JSON output
"""

import glob
import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import click

PROVIDERS = ["anthropic", "openai", "gemini"]

OPENCLAW_STATE = Path(os.environ.get("OPENCLAW_STATE_DIR", Path.home() / ".openclaw"))
AUTH_PROFILES = OPENCLAW_STATE / "agents" / "main" / "agent" / "auth-profiles.json"
SESSIONS_DIR = OPENCLAW_STATE / "agents" / "main" / "sessions"


# ─── Anthropic ────────────────────────────────────────────────────────────────

def check_anthropic(api_key):
    """Check Anthropic rate limits via a minimal API call."""
    result = {"provider": "anthropic", "status": "unknown", "rate_limits": {}, "error": None}

    if not api_key:
        result["status"] = "no_key"
        result["error"] = "No ANTHROPIC_API_KEY set"
        return result

    url = "https://api.anthropic.com/v1/messages"
    payload = json.dumps({
        "model": "claude-sonnet-4-6",
        "max_tokens": 1,
        "messages": [{"role": "user", "content": "1"}],
    }).encode()

    req = urllib.request.Request(url, data=payload, headers={
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    })

    try:
        resp = urllib.request.urlopen(req, timeout=15)
        headers = dict(resp.headers)
        body = json.loads(resp.read())

        result["status"] = "ok"
        result["model_used"] = body.get("model", "?")

        # Extract rate limit headers
        for key in ["anthropic-ratelimit-requests-limit", "anthropic-ratelimit-requests-remaining",
                     "anthropic-ratelimit-requests-reset",
                     "anthropic-ratelimit-tokens-limit", "anthropic-ratelimit-tokens-remaining",
                     "anthropic-ratelimit-tokens-reset",
                     "anthropic-ratelimit-input-tokens-limit", "anthropic-ratelimit-input-tokens-remaining",
                     "anthropic-ratelimit-output-tokens-limit", "anthropic-ratelimit-output-tokens-remaining",
                     "x-ratelimit-limit-requests", "x-ratelimit-remaining-requests",
                     "x-ratelimit-limit-tokens", "x-ratelimit-remaining-tokens",
                     "retry-after"]:
            val = headers.get(key) or headers.get(key.lower())
            if val:
                short = key.replace("anthropic-ratelimit-", "").replace("x-ratelimit-", "")
                result["rate_limits"][short] = val

        # Usage from the response
        usage = body.get("usage", {})
        if usage:
            result["call_usage"] = usage

    except urllib.error.HTTPError as e:
        headers = dict(e.headers) if hasattr(e, "headers") else {}
        body_text = e.read().decode() if hasattr(e, "read") else ""

        if e.code == 429:
            result["status"] = "rate_limited"
            result["error"] = "Rate limited"
            for key, val in headers.items():
                if "ratelimit" in key.lower() or "retry" in key.lower():
                    short = key.replace("anthropic-ratelimit-", "").replace("x-ratelimit-", "")
                    result["rate_limits"][short] = val
        elif e.code == 401:
            result["status"] = "auth_error"
            result["error"] = "Invalid API key"
        elif e.code == 529:
            result["status"] = "overloaded"
            result["error"] = "API overloaded"
        else:
            result["status"] = "error"
            result["error"] = f"HTTP {e.code}: {body_text[:200]}"
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)

    return result


# ─── OpenAI ───────────────────────────────────────────────────────────────────

def check_openai(api_key, admin_key=None):
    """Check OpenAI usage via organization API and rate-limit probe."""
    result = {"provider": "openai", "status": "unknown", "rate_limits": {}, "usage": {}, "error": None}

    effective_key = admin_key or api_key
    if not effective_key:
        result["status"] = "no_key"
        result["error"] = "No OPENAI_API_KEY or OPENAI_ADMIN_KEY set"
        return result

    # 1. Try organization usage API (needs admin/org-level key)
    now = int(time.time())
    day_start = now - (now % 86400)  # Start of today UTC
    week_start = day_start - (6 * 86400)  # 7 days ago

    for period_name, start_time in [("today", day_start), ("week", week_start)]:
        usage_url = f"https://api.openai.com/v1/organization/usage/completions?start_time={start_time}"
        req = urllib.request.Request(usage_url, headers={"Authorization": f"Bearer {effective_key}"})
        try:
            resp = urllib.request.urlopen(req, timeout=15)
            data = json.load(resp)
            buckets = data.get("data", [])
            total_input = sum(sum(r.get("input_tokens", 0) for r in b.get("results", [])) for b in buckets)
            total_output = sum(sum(r.get("output_tokens", 0) for r in b.get("results", [])) for b in buckets)
            total_requests = sum(sum(r.get("num_model_requests", 0) for r in b.get("results", [])) for b in buckets)
            result["usage"][period_name] = {
                "input_tokens": total_input,
                "output_tokens": total_output,
                "total_tokens": total_input + total_output,
                "requests": total_requests,
            }
            result["status"] = "ok"
        except urllib.error.HTTPError as e:
            if e.code == 403:
                result["usage"][period_name] = {"error": "insufficient_permissions (need api.usage.read scope)"}
            else:
                result["usage"][period_name] = {"error": f"HTTP {e.code}"}
        except Exception as e:
            result["usage"][period_name] = {"error": str(e)}

    # 2. Probe rate limits via models endpoint (lightweight, no tokens consumed)
    probe_url = "https://api.openai.com/v1/models"
    req = urllib.request.Request(probe_url, headers={"Authorization": f"Bearer {api_key or effective_key}"})
    try:
        resp = urllib.request.urlopen(req, timeout=10)
        headers = dict(resp.headers)
        for key, val in headers.items():
            if "ratelimit" in key.lower() or "remaining" in key.lower():
                short = key.replace("x-ratelimit-", "")
                result["rate_limits"][short] = val
        if result["status"] == "unknown":
            result["status"] = "ok"
    except urllib.error.HTTPError as e:
        if e.code == 401:
            result["status"] = "auth_error"
            result["error"] = "Invalid API key"
    except Exception as e:
        if result["status"] == "unknown":
            result["status"] = "error"
            result["error"] = str(e)

    return result


# ─── Gemini ───────────────────────────────────────────────────────────────────

def check_gemini(api_key):
    """Check Gemini API availability (no usage API exists)."""
    result = {"provider": "gemini", "status": "unknown", "models": [], "error": None}

    if not api_key:
        result["status"] = "no_key"
        result["error"] = "No GEMINI_API_KEY set"
        return result

    # Gemini has no usage/billing API. Just verify the key works and list available models.
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}&pageSize=100"
    req = urllib.request.Request(url)
    try:
        resp = urllib.request.urlopen(req, timeout=10)
        data = json.load(resp)
        models = data.get("models", [])
        result["status"] = "ok"
        result["model_count"] = len(models)
        # Highlight key models
        key_models = [m["name"].split("/")[-1] for m in models
                      if any(k in m["name"] for k in ["gemini-3", "gemini-2.5", "imagen", "veo"])]
        result["key_models"] = sorted(set(key_models))
    except urllib.error.HTTPError as e:
        if e.code == 400:
            result["status"] = "auth_error"
            result["error"] = "Invalid API key"
        else:
            result["status"] = "error"
            result["error"] = f"HTTP {e.code}"
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)

    return result


# ─── OpenClaw local usage ─────────────────────────────────────────────────────

def get_openclaw_usage():
    """Aggregate token usage from OpenClaw session logs."""
    result = {"source": "openclaw_sessions", "today": {}, "week": {}}

    if not SESSIONS_DIR.exists():
        result["error"] = f"Session dir not found: {SESSIONS_DIR}"
        return result

    now = datetime.now(timezone.utc)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    week_start = today_start - timedelta(days=7)

    provider_usage = {}  # {provider: {input_tokens, output_tokens, requests, cache_read, cache_write}}

    for session_file in SESSIONS_DIR.glob("*.jsonl"):
        try:
            mtime = datetime.fromtimestamp(session_file.stat().st_mtime, tz=timezone.utc)
            if mtime < week_start:
                continue  # Skip old sessions

            with open(session_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or '"usage"' not in line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    msg = entry.get("message", entry)
                    usage = msg.get("usage") or entry.get("usage")
                    if not usage:
                        continue

                    provider = msg.get("provider", entry.get("provider", "unknown"))
                    ts_str = entry.get("timestamp", "")
                    try:
                        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    except (ValueError, AttributeError):
                        ts = mtime

                    is_today = ts >= today_start
                    periods = ["week"]
                    if is_today:
                        periods.append("today")

                    for period in periods:
                        key = f"{provider}:{period}"
                        if key not in provider_usage:
                            provider_usage[key] = {
                                "input_tokens": 0, "output_tokens": 0,
                                "cache_read_tokens": 0, "cache_write_tokens": 0,
                                "total_tokens": 0, "cost": 0.0,
                                "requests": 0,
                            }
                        pu = provider_usage[key]
                        # OpenClaw format: input/output/cacheRead/cacheWrite/totalTokens
                        pu["input_tokens"] += usage.get("input", usage.get("input_tokens", 0))
                        pu["output_tokens"] += usage.get("output", usage.get("output_tokens", 0))
                        pu["cache_read_tokens"] += usage.get("cacheRead", usage.get("cache_read_input_tokens", 0))
                        pu["cache_write_tokens"] += usage.get("cacheWrite", usage.get("cache_creation_input_tokens", 0))
                        pu["total_tokens"] += usage.get("totalTokens", 0)
                        cost = usage.get("cost", {})
                        if isinstance(cost, dict):
                            pu["cost"] += cost.get("total", 0)
                        elif isinstance(cost, (int, float)):
                            pu["cost"] += cost
                        pu["requests"] += 1

        except Exception:
            continue

    # Organize by period
    for key, usage in provider_usage.items():
        provider, period = key.rsplit(":", 1)
        if usage["total_tokens"] == 0:
            usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
        if provider not in result[period]:
            result[period][provider] = usage
        else:
            for k, v in usage.items():
                result[period][provider][k] = result[period][provider].get(k, 0) + v

    return result


# ─── Display ──────────────────────────────────────────────────────────────────

def _fmt_tokens(n):
    """Format token count with K/M suffix."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    elif n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def display_result(result):
    """Pretty-print a provider check result."""
    provider = result["provider"]
    status = result["status"]

    status_icon = {"ok": "✅", "rate_limited": "⚠️", "auth_error": "❌",
                   "no_key": "⬚", "overloaded": "🔥", "error": "❌"}.get(status, "?")

    click.echo(f"\n{status_icon} {provider.upper()}")
    click.echo(f"  Status: {status}")

    if result.get("error"):
        click.echo(f"  Error: {result['error']}")

    if result.get("model_used"):
        click.echo(f"  Model: {result['model_used']}")

    # Rate limits
    rl = result.get("rate_limits", {})
    if rl:
        click.echo("  Rate limits:")
        for key, val in sorted(rl.items()):
            click.echo(f"    {key}: {val}")

    # Usage (OpenAI)
    usage = result.get("usage", {})
    for period, data in usage.items():
        if isinstance(data, dict) and "error" not in data:
            click.echo(f"  Usage ({period}):")
            click.echo(f"    Input:    {_fmt_tokens(data['input_tokens'])}")
            click.echo(f"    Output:   {_fmt_tokens(data['output_tokens'])}")
            click.echo(f"    Total:    {_fmt_tokens(data['total_tokens'])}")
            click.echo(f"    Requests: {data['requests']}")
        elif isinstance(data, dict) and "error" in data:
            click.echo(f"  Usage ({period}): {data['error']}")

    # Gemini models
    if result.get("model_count"):
        click.echo(f"  Available models: {result['model_count']}")
    if result.get("key_models"):
        click.echo(f"  Key models: {', '.join(result['key_models'][:10])}")


def display_openclaw_usage(data):
    """Pretty-print OpenClaw session usage."""
    click.echo(f"\n📊 OPENCLAW LOCAL USAGE")

    for period in ["today", "week"]:
        providers = data.get(period, {})
        if not providers:
            click.echo(f"  {period.title()}: no usage recorded")
            continue

        click.echo(f"  {period.title()}:")
        for provider, usage in sorted(providers.items()):
            total = usage.get("total_tokens", 0)
            inp = usage.get("input_tokens", 0)
            out = usage.get("output_tokens", 0)
            cache_read = usage.get("cache_read_tokens", 0)
            cache_write = usage.get("cache_write_tokens", 0)
            cost = usage.get("cost", 0)
            reqs = usage.get("requests", 0)
            parts = [f"{_fmt_tokens(total)} tokens ({_fmt_tokens(inp)} in / {_fmt_tokens(out)} out)"]
            if cache_read or cache_write:
                parts.append(f"cache: {_fmt_tokens(cache_read)} read / {_fmt_tokens(cache_write)} write")
            if cost > 0:
                parts.append(f"${cost:.2f}")
            parts.append(f"{reqs} reqs")
            click.echo(f"    {provider}: {' · '.join(parts)}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

@click.command()
@click.option("-p", "--provider", "providers", multiple=True,
              type=click.Choice(PROVIDERS + ["all"], case_sensitive=False),
              default=["all"], show_default=True, help="Provider(s) to check.")
@click.option("--openclaw/--no-openclaw", default=True, show_default=True,
              help="Include OpenClaw local session usage.")
@click.option("--json-output", "--json", "json_output", is_flag=True, help="Output as JSON.")
@click.option("--anthropic-api-key", envvar="ANTHROPIC_API_KEY", default=None,
              help="Anthropic API key [env: ANTHROPIC_API_KEY].")
@click.option("--openai-api-key", envvar="OPENAI_API_KEY", default=None,
              help="OpenAI API key [env: OPENAI_API_KEY].")
@click.option("--openai-admin-key", envvar="OPENAI_ADMIN_KEY", default=None,
              help="OpenAI admin key for usage API [env: OPENAI_ADMIN_KEY].")
@click.option("--gemini-api-key", envvar="GEMINI_API_KEY", default=None,
              help="Gemini API key [env: GEMINI_API_KEY].")
def main(providers, openclaw, json_output, anthropic_api_key, openai_api_key, openai_admin_key, gemini_api_key):
    """Monitor LLM token usage and quotas across providers."""
    check_all = "all" in providers
    results = []

    if check_all or "anthropic" in providers:
        results.append(check_anthropic(anthropic_api_key))

    if check_all or "openai" in providers:
        results.append(check_openai(openai_api_key, openai_admin_key))

    if check_all or "gemini" in providers:
        results.append(check_gemini(gemini_api_key))

    openclaw_data = None
    if openclaw:
        openclaw_data = get_openclaw_usage()

    if json_output:
        output = {"providers": results, "timestamp": datetime.now(timezone.utc).isoformat()}
        if openclaw_data:
            output["openclaw_usage"] = openclaw_data
        click.echo(json.dumps(output, indent=2))
    else:
        click.echo("🔍 LLM Usage Monitor")
        click.echo(f"   {datetime.now().strftime('%Y-%m-%d %H:%M %Z')}")

        for r in results:
            display_result(r)

        if openclaw_data:
            display_openclaw_usage(openclaw_data)

        click.echo("")


if __name__ == "__main__":
    main()
