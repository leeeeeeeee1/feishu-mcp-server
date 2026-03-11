"""Feishu IM (Messaging) tools."""

from __future__ import annotations

import json

import lark_oapi as lark
from lark_oapi.api.im.v1 import (
    CreateMessageRequest,
    CreateMessageRequestBody,
    ListChatRequest,
    ListMessageRequest,
    ReplyMessageRequest,
    ReplyMessageRequestBody,
)

from feishu_mcp.utils import extract_response

TOOLS = [
    {
        "name": "feishu_send_message",
        "description": "Send a message to a Feishu chat or user. Supports text, rich text, and interactive cards.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "receive_id": {
                    "type": "string",
                    "description": "Receiver ID (chat_id, open_id, union_id, or email)",
                },
                "receive_id_type": {
                    "type": "string",
                    "enum": ["chat_id", "open_id", "union_id", "email"],
                    "description": "Type of receive_id",
                    "default": "chat_id",
                },
                "msg_type": {
                    "type": "string",
                    "enum": ["text", "post", "interactive"],
                    "description": "Message type",
                    "default": "text",
                },
                "content": {
                    "type": "string",
                    "description": 'Message content as JSON string. For text: {"text":"hello"}. For post: rich text JSON. For interactive: card JSON.',
                },
            },
            "required": ["receive_id", "content"],
        },
    },
    {
        "name": "feishu_reply_message",
        "description": "Reply to an existing Feishu message.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "message_id": {
                    "type": "string",
                    "description": "ID of the message to reply to",
                },
                "msg_type": {
                    "type": "string",
                    "enum": ["text", "post", "interactive"],
                    "default": "text",
                },
                "content": {
                    "type": "string",
                    "description": "Reply content as JSON string",
                },
            },
            "required": ["message_id", "content"],
        },
    },
    {
        "name": "feishu_list_messages",
        "description": "List messages in a Feishu chat. Returns recent messages with sender, content, and timestamps.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "container_id": {
                    "type": "string",
                    "description": "Chat ID to list messages from",
                },
                "page_size": {
                    "type": "integer",
                    "description": "Number of messages to return (max 50)",
                    "default": 20,
                },
                "page_token": {
                    "type": "string",
                    "description": "Pagination token for next page",
                },
            },
            "required": ["container_id"],
        },
    },
    {
        "name": "feishu_list_chats",
        "description": "List all chats/groups the bot has joined.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "page_size": {
                    "type": "integer",
                    "description": "Number of chats to return (max 100)",
                    "default": 20,
                },
                "page_token": {
                    "type": "string",
                    "description": "Pagination token",
                },
            },
        },
    },
]


def handle(client: lark.Client, name: str, args: dict) -> dict | str:
    if name == "feishu_send_message":
        return _send_message(client, args)
    elif name == "feishu_reply_message":
        return _reply_message(client, args)
    elif name == "feishu_list_messages":
        return _list_messages(client, args)
    elif name == "feishu_list_chats":
        return _list_chats(client, args)
    raise ValueError(f"Unknown IM tool: {name}")


def _send_message(client: lark.Client, args: dict) -> dict | str:
    receive_id_type = args.get("receive_id_type", "chat_id")
    msg_type = args.get("msg_type", "text")
    content = args["content"]

    # Auto-wrap plain text, validate JSON for other types
    if msg_type == "text" and not content.startswith("{"):
        content = json.dumps({"text": content})
    elif content.startswith("{"):
        try:
            json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON content: {e}")

    req = (
        CreateMessageRequest.builder()
        .receive_id_type(receive_id_type)
        .request_body(
            CreateMessageRequestBody.builder()
            .receive_id(args["receive_id"])
            .msg_type(msg_type)
            .content(content)
            .build()
        )
        .build()
    )
    resp = client.im.v1.message.create(req)
    return extract_response(resp)


def _reply_message(client: lark.Client, args: dict) -> dict | str:
    msg_type = args.get("msg_type", "text")
    content = args["content"]

    if msg_type == "text" and not content.startswith("{"):
        content = json.dumps({"text": content})
    elif content.startswith("{"):
        try:
            json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON content: {e}")

    req = (
        ReplyMessageRequest.builder()
        .message_id(args["message_id"])
        .request_body(
            ReplyMessageRequestBody.builder()
            .msg_type(msg_type)
            .content(content)
            .build()
        )
        .build()
    )
    resp = client.im.v1.message.reply(req)
    return extract_response(resp)


def _list_messages(client: lark.Client, args: dict) -> dict | str:
    builder = (
        ListMessageRequest.builder()
        .container_id_type("chat")
        .container_id(args["container_id"])
        .page_size(min(args.get("page_size", 20), 50))
    )
    if args.get("page_token"):
        builder = builder.page_token(args["page_token"])

    resp = client.im.v1.message.list(builder.build())
    return extract_response(resp)


def _list_chats(client: lark.Client, args: dict) -> dict | str:
    builder = ListChatRequest.builder().page_size(
        min(args.get("page_size", 20), 100)
    )
    if args.get("page_token"):
        builder = builder.page_token(args["page_token"])

    resp = client.im.v1.chat.list(builder.build())
    return extract_response(resp)
