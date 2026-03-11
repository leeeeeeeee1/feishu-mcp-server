"""Feishu Bot with Claude Code session forwarding.

Receives messages from Feishu, forwards to a persistent Claude Code session,
and sends Claude's response back to Feishu.

Usage:
    FEISHU_APP_ID=xxx FEISHU_APP_SECRET=xxx python3 -m feishu_mcp.bot
"""

import os
import json
import subprocess
import threading
import time
import lark_oapi as lark
from lark_oapi.api.im.v1 import (
    P2ImMessageReceiveV1,
    ReplyMessageRequest,
    ReplyMessageRequestBody,
)

# Global state
_client: lark.Client = None
_session_id: str = None
_lock = threading.Lock()

# Message dedup: message_id -> timestamp (prevent duplicate event delivery)
_seen_messages: dict[str, float] = {}
_DEDUP_TTL = 60  # seconds


def _dedup_check(message_id: str) -> bool:
    """Return True if this message was already seen (duplicate)."""
    now = time.time()

    # Cleanup old entries
    expired = [k for k, t in _seen_messages.items() if now - t > _DEDUP_TTL]
    for k in expired:
        del _seen_messages[k]

    if message_id in _seen_messages:
        return True

    _seen_messages[message_id] = now
    return False


def _call_claude(text: str) -> str:
    """Call Claude Code CLI with session persistence."""
    global _session_id

    env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
    cmd = ["claude", "-p", text, "--output-format", "text"]

    if _session_id:
        cmd.extend(["-r", _session_id])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            env=env,
        )

        if result.returncode != 0:
            stderr = result.stderr.strip()
            if _session_id and ("session" in stderr.lower() or "not found" in stderr.lower()):
                _session_id = None
                return _call_claude(text)
            return f"Claude Code error: {stderr or 'unknown error'}"

        response = result.stdout.strip()

        # Try to capture session ID for resumption
        if not _session_id:
            for line in result.stderr.splitlines():
                if "session:" in line.lower() or "id:" in line.lower():
                    parts = line.strip().split()
                    if parts:
                        _session_id = parts[-1]
                        break

        return response or "(empty response)"

    except subprocess.TimeoutExpired:
        return "Claude Code response timed out (5 min limit)"
    except FileNotFoundError:
        return "Error: 'claude' command not found. Is Claude Code installed?"


def _reply(message_id: str, text: str):
    """Reply to a Feishu message, splitting long text into chunks."""
    chunks = [text[i:i+3500] for i in range(0, len(text), 3500)]

    for i, chunk in enumerate(chunks):
        content = json.dumps({"text": chunk})
        req = (
            ReplyMessageRequest.builder()
            .message_id(message_id)
            .request_body(
                ReplyMessageRequestBody.builder()
                .msg_type("text")
                .content(content)
                .build()
            )
            .build()
        )
        resp = _client.im.v1.message.reply(req)
        if resp.success():
            print(f"[REPLY] OK chunk {i+1}/{len(chunks)}")
        else:
            print(f"[REPLY] FAIL code={resp.code} msg={resp.msg}")
            break


def on_message(data: P2ImMessageReceiveV1):
    """Handle incoming messages: forward to Claude, reply with response."""
    event = data.event
    msg = event.message
    sender = event.sender

    # Skip messages from bots (including ourselves) to prevent loops
    if sender and sender.sender_type == "app":
        return

    # Dedup: skip if we already processed this message
    if _dedup_check(msg.message_id):
        print(f"[DEDUP] Skipping duplicate: {msg.message_id}")
        return

    sender_id = sender.sender_id.open_id if sender and sender.sender_id else "unknown"
    raw_content = msg.content if msg else ""
    print(f"[MSG] from={sender_id} type={msg.message_type} content={raw_content}")

    # Extract text
    try:
        content_obj = json.loads(raw_content)
        text = content_obj.get("text", raw_content)
    except (json.JSONDecodeError, TypeError):
        text = raw_content

    if not text or not text.strip():
        return

    # Strip @bot mentions (Feishu adds @_user_1 for bot mentions)
    text = text.replace("@_user_1", "").strip()
    if not text:
        return

    # Forward to Claude (serialized to avoid session conflicts)
    with _lock:
        print(f"[CLAUDE] Forwarding: {text[:100]}...")
        response = _call_claude(text)
        print(f"[CLAUDE] Response: {response[:100]}...")

    # Send single reply
    _reply(msg.message_id, response)


def main():
    global _client

    app_id = os.environ.get("FEISHU_APP_ID", "")
    app_secret = os.environ.get("FEISHU_APP_SECRET", "")

    if not app_id or not app_secret:
        raise ValueError("FEISHU_APP_ID and FEISHU_APP_SECRET are required")

    _client = (
        lark.Client.builder()
        .app_id(app_id)
        .app_secret(app_secret)
        .domain(lark.FEISHU_DOMAIN)
        .build()
    )

    handler = (
        lark.EventDispatcherHandler.builder("", "")
        .register_p2_im_message_receive_v1(on_message)
        .build()
    )

    cli = lark.ws.Client(
        app_id, app_secret, event_handler=handler, log_level=lark.LogLevel.INFO
    )

    print("Starting Feishu bot with Claude Code forwarding...")
    print("Bot is online. Messages will be forwarded to Claude Code.")
    print("Press Ctrl+C to stop.")
    cli.start()


if __name__ == "__main__":
    main()
