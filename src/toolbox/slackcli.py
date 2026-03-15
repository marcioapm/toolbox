#!/usr/bin/env python3
"""Lightweight Slack CLI using a user token.

Environment:
    SLACK_USER_TOKEN  — Required. Slack user token (xoxp-...) with appropriate scopes.

Commands:
    slackcli channels         List channels (public, private, DMs)
    slackcli history <ch>     Read channel message history
    slackcli send <ch> <msg>  Send a message to a channel
    slackcli reply <ch> <ts> <msg>  Reply in a thread
    slackcli search <query>   Search messages
    slackcli users            List workspace members
    slackcli userinfo <uid>   Show user details
    slackcli dm <uid>         Get/create DM channel ID
    slackcli unread           Show channels with unread messages
    slackcli react <ch> <ts> <emoji>    Add a reaction
    slackcli unreact <ch> <ts> <emoji>  Remove a reaction

Usage:
    slackcli channels
    slackcli history C02DLS4PFH7 -n 30
    slackcli send C02DLS4PFH7 "Hello from the CLI!"
    slackcli unread
    slackcli react C02DLS4PFH7 1710430020.123456 thumbsup
"""

import json
import os
import sys
import urllib.parse
import urllib.request
from datetime import datetime

import click

API = "https://slack.com/api"


def _get_token(ctx):
    token = ctx.obj.get("token", "")
    if not token:
        raise click.ClickException("SLACK_USER_TOKEN not set. Pass --token or set the env var.")
    return token


def _api_get(token, method, params=None):
    url = f"{API}/{method}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def _api_post(token, method, data):
    req = urllib.request.Request(
        f"{API}/{method}",
        data=json.dumps(data).encode(),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def _check(data):
    if not data.get("ok"):
        raise click.ClickException(f"Slack API error: {data.get('error', 'unknown')}")
    return data


@click.group()
@click.option("--token", envvar="SLACK_USER_TOKEN", required=True, help="Slack user token [env: SLACK_USER_TOKEN].")
@click.pass_context
def main(ctx, token):
    """Lightweight Slack CLI."""
    ctx.ensure_object(dict)
    ctx.obj["token"] = token


@main.command()
@click.option("-t", "--type", "channel_type", default=None, help="Channel types (e.g. public_channel,im).")
@click.option("-n", "--limit", default=20, show_default=True, help="Max channels to list.")
@click.pass_context
def channels(ctx, channel_type, limit):
    """List channels (public, private, DMs)."""
    token = _get_token(ctx)
    types = channel_type or "public_channel,private_channel,im,mpim"
    data = _check(_api_get(token, "conversations.list", {"types": types, "limit": limit, "exclude_archived": "true"}))
    for ch in data["channels"]:
        cid = ch["id"]
        if ch.get("is_im"):
            name = f"DM:{ch.get('user', '?')}"
        elif ch.get("is_mpim"):
            name = ch.get("name", "group-dm")
        else:
            name = f"#{ch.get('name', '?')}"
        members = ch.get("num_members", "-")
        click.echo(f"{cid}  {name}  (members: {members})")


@main.command()
@click.argument("channel")
@click.option("-n", "--limit", default=20, show_default=True, help="Max messages to show.")
@click.pass_context
def history(ctx, channel, limit):
    """Read channel message history."""
    token = _get_token(ctx)
    data = _check(_api_get(token, "conversations.history", {"channel": channel, "limit": limit}))
    msgs = data.get("messages", [])
    msgs.reverse()
    for m in msgs:
        ts = float(m.get("ts", 0))
        dt = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
        user = m.get("user", m.get("username", "bot"))
        text = m.get("text", "")
        thread = f" [{m.get('reply_count', 0)} replies]" if m.get("reply_count") else ""
        click.echo(f"[{dt}] <{user}>{thread} {text}")


@main.command()
@click.argument("channel")
@click.argument("message", nargs=-1, required=True)
@click.pass_context
def send(ctx, channel, message):
    """Send a message to a channel."""
    token = _get_token(ctx)
    text = " ".join(message)
    data = _check(_api_post(token, "chat.postMessage", {"channel": channel, "text": text}))
    click.echo(f"Sent (ts: {data.get('ts')})")


@main.command()
@click.argument("channel")
@click.argument("thread_ts")
@click.argument("message", nargs=-1, required=True)
@click.pass_context
def reply(ctx, channel, thread_ts, message):
    """Reply in a thread."""
    token = _get_token(ctx)
    text = " ".join(message)
    data = _check(_api_post(token, "chat.postMessage", {
        "channel": channel, "thread_ts": thread_ts, "text": text,
    }))
    click.echo(f"Replied (ts: {data.get('ts')})")


@main.command()
@click.argument("query", nargs=-1, required=True)
@click.option("-n", "--limit", default=10, show_default=True, help="Max results.")
@click.pass_context
def search(ctx, query, limit):
    """Search messages across workspace."""
    token = _get_token(ctx)
    q = " ".join(query)
    data = _check(_api_get(token, "search.messages", {"query": q, "count": limit}))
    for m in data.get("messages", {}).get("matches", []):
        ts = float(m.get("ts", 0))
        dt = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
        user = m.get("username", m.get("user", "?"))
        ch = m.get("channel", {}).get("name", "?")
        text = m.get("text", "")[:150]
        click.echo(f"[{dt}] #{ch} <{user}> {text}")


@main.command()
@click.option("-n", "--limit", default=200, show_default=True, help="Max users to list.")
@click.pass_context
def users(ctx, limit):
    """List workspace members."""
    token = _get_token(ctx)
    data = _check(_api_get(token, "users.list", {"limit": limit}))
    for u in data.get("members", []):
        if u.get("deleted") or u.get("is_bot"):
            continue
        uid = u["id"]
        name = u.get("real_name", u.get("name", "?"))
        display = u.get("profile", {}).get("display_name", "")
        click.echo(f"{uid}  {name}  @{display}")


@main.command()
@click.argument("user")
@click.pass_context
def userinfo(ctx, user):
    """Show user details."""
    token = _get_token(ctx)
    data = _check(_api_get(token, "users.info", {"user": user}))
    u = data["user"]
    p = u.get("profile", {})
    click.echo(f"ID:      {u['id']}")
    click.echo(f"Name:    {u.get('real_name', '?')}")
    click.echo(f"Display: {p.get('display_name', '')}")
    click.echo(f"Email:   {p.get('email', '')}")
    click.echo(f"Title:   {p.get('title', '')}")
    click.echo(f"TZ:      {u.get('tz', '')}")


@main.command()
@click.argument("user")
@click.pass_context
def dm(ctx, user):
    """Get or create a DM channel ID."""
    token = _get_token(ctx)
    data = _check(_api_post(token, "conversations.open", {"users": user}))
    click.echo(data["channel"]["id"])


@main.command()
@click.pass_context
def unread(ctx):
    """Show channels with unread messages."""
    token = _get_token(ctx)
    data = _check(_api_get(token, "conversations.list", {
        "types": "public_channel,private_channel,im,mpim",
        "limit": 200, "exclude_archived": "true",
    }))
    found = False
    for ch in data.get("channels", []):
        count = ch.get("unread_count_display", 0)
        if count > 0:
            found = True
            cid = ch["id"]
            if ch.get("is_im"):
                name = f"DM:{ch.get('user', '?')}"
            elif ch.get("is_mpim"):
                name = ch.get("name", "group-dm")
            else:
                name = f"#{ch.get('name', '?')}"
            click.echo(f"{cid}  {name}  ({count} unread)")
    if not found:
        click.echo("No unread messages")


@main.command()
@click.argument("channel")
@click.argument("ts")
@click.argument("emoji")
@click.pass_context
def react(ctx, channel, ts, emoji):
    """Add an emoji reaction to a message."""
    token = _get_token(ctx)
    data = _api_post(token, "reactions.add", {"channel": channel, "timestamp": ts, "name": emoji})
    click.echo("Reacted ✅" if data.get("ok") else f"Error: {data.get('error')}")


@main.command()
@click.argument("channel")
@click.argument("ts")
@click.argument("emoji")
@click.pass_context
def unreact(ctx, channel, ts, emoji):
    """Remove an emoji reaction from a message."""
    token = _get_token(ctx)
    data = _api_post(token, "reactions.remove", {"channel": channel, "timestamp": ts, "name": emoji})
    click.echo("Removed ✅" if data.get("ok") else f"Error: {data.get('error')}")


if __name__ == "__main__":
    main()
