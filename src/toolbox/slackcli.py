#!/usr/bin/env python3
"""Lightweight Slack CLI using a user token.

Environment:
    SLACK_USER_TOKEN  — Required. Slack user token (xoxp-...) with appropriate scopes.
                        Get one at https://api.slack.com/apps → OAuth & Permissions.

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
    slackcli reply C02DLS4PFH7 1710430020.123456 "Thread reply"
    slackcli search "deployment failed" -n 5
    slackcli unread
    slackcli react C02DLS4PFH7 1710430020.123456 thumbsup
"""

import argparse
import json
import os
import sys
import urllib.parse
import urllib.request
from datetime import datetime

API = "https://slack.com/api"


def get_token():
    token = os.environ.get("SLACK_USER_TOKEN", "")
    if not token:
        print("Error: SLACK_USER_TOKEN not set", file=sys.stderr)
        sys.exit(1)
    return token


TOKEN = None  # Set in main()


def api_get(method, params=None):
    url = f"{API}/{method}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {TOKEN}"})
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def api_post(method, data):
    req = urllib.request.Request(
        f"{API}/{method}",
        data=json.dumps(data).encode(),
        headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def check(data):
    if not data.get("ok"):
        print(f"Error: {data.get('error', 'unknown')}", file=sys.stderr)
        sys.exit(1)
    return data


def cmd_channels(args):
    types = args.type or "public_channel,private_channel,im,mpim"
    data = check(api_get("conversations.list", {"types": types, "limit": args.limit, "exclude_archived": "true"}))
    for ch in data["channels"]:
        cid = ch["id"]
        if ch.get("is_im"):
            name = f"DM:{ch.get('user', '?')}"
        elif ch.get("is_mpim"):
            name = ch.get("name", "group-dm")
        else:
            name = f"#{ch.get('name', '?')}"
        members = ch.get("num_members", "-")
        print(f"{cid}  {name}  (members: {members})")


def cmd_history(args):
    params = {"channel": args.channel, "limit": args.limit}
    data = check(api_get("conversations.history", params))
    msgs = data.get("messages", [])
    msgs.reverse()
    for m in msgs:
        ts = float(m.get("ts", 0))
        dt = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
        user = m.get("user", m.get("username", "bot"))
        text = m.get("text", "")
        thread = f" [{m.get('reply_count', 0)} replies]" if m.get("reply_count") else ""
        print(f"[{dt}] <{user}>{thread} {text}")


def cmd_send(args):
    message = " ".join(args.message)
    data = check(api_post("chat.postMessage", {"channel": args.channel, "text": message}))
    print(f"Sent (ts: {data.get('ts')})")


def cmd_reply(args):
    message = " ".join(args.message)
    data = check(api_post("chat.postMessage", {
        "channel": args.channel, "thread_ts": args.thread_ts, "text": message,
    }))
    print(f"Replied (ts: {data.get('ts')})")


def cmd_search(args):
    query = " ".join(args.query)
    data = check(api_get("search.messages", {"query": query, "count": args.limit}))
    for m in data.get("messages", {}).get("matches", []):
        ts = float(m.get("ts", 0))
        dt = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
        user = m.get("username", m.get("user", "?"))
        ch = m.get("channel", {}).get("name", "?")
        text = m.get("text", "")[:150]
        print(f"[{dt}] #{ch} <{user}> {text}")


def cmd_users(args):
    data = check(api_get("users.list", {"limit": args.limit}))
    for u in data.get("members", []):
        if u.get("deleted") or u.get("is_bot"):
            continue
        uid = u["id"]
        name = u.get("real_name", u.get("name", "?"))
        display = u.get("profile", {}).get("display_name", "")
        print(f"{uid}  {name}  @{display}")


def cmd_userinfo(args):
    data = check(api_get("users.info", {"user": args.user}))
    u = data["user"]
    p = u.get("profile", {})
    print(f"ID:      {u['id']}")
    print(f"Name:    {u.get('real_name', '?')}")
    print(f"Display: {p.get('display_name', '')}")
    print(f"Email:   {p.get('email', '')}")
    print(f"Title:   {p.get('title', '')}")
    print(f"TZ:      {u.get('tz', '')}")


def cmd_dm(args):
    data = check(api_post("conversations.open", {"users": args.user}))
    print(data["channel"]["id"])


def cmd_unread(args):
    data = check(api_get("conversations.list", {
        "types": "public_channel,private_channel,im,mpim",
        "limit": 200, "exclude_archived": "true",
    }))
    found = False
    for ch in data.get("channels", []):
        unread = ch.get("unread_count_display", 0)
        if unread > 0:
            found = True
            cid = ch["id"]
            if ch.get("is_im"):
                name = f"DM:{ch.get('user', '?')}"
            elif ch.get("is_mpim"):
                name = ch.get("name", "group-dm")
            else:
                name = f"#{ch.get('name', '?')}"
            print(f"{cid}  {name}  ({unread} unread)")
    if not found:
        print("No unread messages")


def cmd_react(args):
    data = api_post("reactions.add", {
        "channel": args.channel, "timestamp": args.ts, "name": args.emoji,
    })
    print("Reacted ✅" if data.get("ok") else f"Error: {data.get('error')}")


def cmd_unreact(args):
    data = api_post("reactions.remove", {
        "channel": args.channel, "timestamp": args.ts, "name": args.emoji,
    })
    print("Removed ✅" if data.get("ok") else f"Error: {data.get('error')}")


def main():
    global TOKEN
    TOKEN = get_token()

    parser = argparse.ArgumentParser(prog="slackcli", description="Lightweight Slack CLI")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("channels", aliases=["ch"], help="List channels")
    p.add_argument("--type", "-t", default=None)
    p.add_argument("--limit", "-n", type=int, default=20)
    p.set_defaults(func=cmd_channels)

    p = sub.add_parser("history", aliases=["h"], help="Read channel messages")
    p.add_argument("channel")
    p.add_argument("--limit", "-n", type=int, default=20)
    p.set_defaults(func=cmd_history)

    p = sub.add_parser("send", aliases=["s"], help="Send a message")
    p.add_argument("channel")
    p.add_argument("message", nargs="+")
    p.set_defaults(func=cmd_send)

    p = sub.add_parser("reply", aliases=["r"], help="Reply in thread")
    p.add_argument("channel")
    p.add_argument("thread_ts")
    p.add_argument("message", nargs="+")
    p.set_defaults(func=cmd_reply)

    p = sub.add_parser("search", help="Search messages")
    p.add_argument("query", nargs="+")
    p.add_argument("--limit", "-n", type=int, default=10)
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("users", aliases=["u"], help="List workspace members")
    p.add_argument("--limit", "-n", type=int, default=200)
    p.set_defaults(func=cmd_users)

    p = sub.add_parser("userinfo", aliases=["ui"], help="User details")
    p.add_argument("user")
    p.set_defaults(func=cmd_userinfo)

    p = sub.add_parser("dm", help="Get/create DM channel")
    p.add_argument("user")
    p.set_defaults(func=cmd_dm)

    p = sub.add_parser("unread", help="Channels with unread messages")
    p.set_defaults(func=cmd_unread)

    p = sub.add_parser("react", help="Add reaction")
    p.add_argument("channel")
    p.add_argument("ts")
    p.add_argument("emoji")
    p.set_defaults(func=cmd_react)

    p = sub.add_parser("unreact", help="Remove reaction")
    p.add_argument("channel")
    p.add_argument("ts")
    p.add_argument("emoji")
    p.set_defaults(func=cmd_unreact)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)
    args.func(args)


if __name__ == "__main__":
    main()
