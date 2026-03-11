"""Feishu bidirectional gateway — receive messages + proactive push.

Supports text, image, file sending, and streaming message updates via PATCH.
"""

import json
import logging
import os
import time
import threading
from pathlib import Path
from typing import Callable, Optional

import lark_oapi as lark
from lark_oapi.api.im.v1 import (
    CreateMessageRequest,
    CreateMessageRequestBody,
    PatchMessageRequest,
    PatchMessageRequestBody,
    ReplyMessageRequest,
    ReplyMessageRequestBody,
    CreateImageRequest,
    CreateImageRequestBody,
    CreateFileRequest,
    CreateFileRequestBody,
    P2ImMessageReceiveV1,
    P2ImMessageMessageReadV1,
)

logger = logging.getLogger(__name__)

# Message dedup
_seen_messages: dict[str, float] = {}
_DEDUP_TTL = 60


def _dedup_check(message_id: str) -> bool:
    """Return True if this message was already seen (duplicate)."""
    now = time.time()
    expired = [k for k, t in _seen_messages.items() if now - t > _DEDUP_TTL]
    for k in expired:
        del _seen_messages[k]
    if message_id in _seen_messages:
        return True
    _seen_messages[message_id] = now
    return False


class FeishuGateway:
    """Bidirectional Feishu gateway with streaming and multimedia support."""

    def __init__(
        self,
        app_id: Optional[str] = None,
        app_secret: Optional[str] = None,
        push_chat_id: Optional[str] = None,
    ):
        self.app_id = app_id or os.environ.get("FEISHU_APP_ID", "")
        self.app_secret = app_secret or os.environ.get("FEISHU_APP_SECRET", "")
        self.push_chat_id = push_chat_id or os.environ.get("FEISHU_PUSH_CHAT_ID", "")

        if not self.app_id or not self.app_secret:
            raise ValueError("FEISHU_APP_ID and FEISHU_APP_SECRET are required")

        self.client = (
            lark.Client.builder()
            .app_id(self.app_id)
            .app_secret(self.app_secret)
            .domain(lark.FEISHU_DOMAIN)
            .build()
        )

        self._on_message: Optional[Callable] = None
        self._on_message_read: Optional[Callable] = None
        self._ws_client = None

    def set_message_handler(self, handler: Callable):
        """Set the callback for incoming messages.

        handler(sender_id: str, message_id: str, chat_id: str, msg_type: str, content: str, raw_event)
        """
        self._on_message = handler

    def set_message_read_handler(self, handler: Callable):
        """Set the callback for message-read events.

        handler(reader_id: str, message_id_list: list[str], read_time: str)
        """
        self._on_message_read = handler

    def _handle_message(self, data: P2ImMessageReceiveV1):
        """Internal message handler with dedup and bot filtering."""
        event = data.event
        msg = event.message
        sender = event.sender

        # Skip bot messages (prevent loops)
        if sender and sender.sender_type == "app":
            return

        # Dedup
        if _dedup_check(msg.message_id):
            logger.debug("Dedup: skipping %s", msg.message_id)
            return

        sender_id = sender.sender_id.open_id if sender and sender.sender_id else "unknown"
        chat_id = msg.chat_id if msg.chat_id else ""
        msg_type = msg.message_type or "text"
        raw_content = msg.content or ""

        logger.info("[MSG] from=%s type=%s chat=%s", sender_id, msg_type, chat_id)

        # Extract text content
        text = raw_content
        if msg_type == "text":
            try:
                content_obj = json.loads(raw_content)
                text = content_obj.get("text", raw_content)
            except (json.JSONDecodeError, TypeError):
                pass
            # Strip @bot mentions
            text = text.replace("@_user_1", "").strip()

        if self._on_message:
            self._on_message(
                sender_id=sender_id,
                message_id=msg.message_id,
                chat_id=chat_id,
                msg_type=msg_type,
                content=text,
                raw_event=data,
            )

    def _handle_message_read(self, data: P2ImMessageMessageReadV1):
        """Internal handler for message-read events."""
        event = data.event
        if not event:
            return

        reader_id = ""
        if event.reader and event.reader.reader_id:
            reader_id = event.reader.reader_id.open_id or ""

        message_ids = event.message_id_list or []
        read_time = event.reader.read_time if event.reader else ""

        logger.debug("[READ] reader=%s messages=%s", reader_id, message_ids)

        if self._on_message_read:
            self._on_message_read(
                reader_id=reader_id,
                message_id_list=message_ids,
                read_time=read_time,
            )

    def start_receiving(self):
        """Start the WebSocket long connection to receive messages.

        This blocks the calling thread.
        """
        handler = (
            lark.EventDispatcherHandler.builder("", "")
            .register_p2_im_message_receive_v1(self._handle_message)
            .register_p2_im_message_message_read_v1(self._handle_message_read)
            .register_p2_im_chat_access_event_bot_p2p_chat_entered_v1(lambda _: None)
            .build()
        )

        self._ws_client = lark.ws.Client(
            self.app_id,
            self.app_secret,
            event_handler=handler,
            log_level=lark.LogLevel.WARNING,
        )

        logger.info("Starting Feishu WebSocket receiver...")
        self._ws_client.start()

    # ── Sending: Text ──

    def send_message(self, chat_id: str, text: str, msg_type: str = "text") -> Optional[str]:
        """Send a message to a chat. Returns the message_id for later updates."""
        if msg_type == "text":
            content = json.dumps({"text": text})
        elif msg_type == "post":
            content = text  # Already formatted as post JSON
        else:
            content = json.dumps({"text": text})

        req = (
            CreateMessageRequest.builder()
            .receive_id_type("chat_id")
            .request_body(
                CreateMessageRequestBody.builder()
                .receive_id(chat_id)
                .msg_type(msg_type)
                .content(content)
                .build()
            )
            .build()
        )
        resp = self.client.im.v1.message.create(req)
        if resp.success():
            mid = resp.data.message_id if resp.data else None
            logger.info("[SEND] OK to %s, message_id=%s", chat_id, mid)
            return mid
        else:
            logger.error("[SEND] FAIL code=%s msg=%s", resp.code, resp.msg)
            return None

    def reply_message(self, message_id: str, text: str, msg_type: str = "text") -> Optional[str]:
        """Reply to a specific message. Returns the reply message_id."""
        if msg_type == "text":
            content = json.dumps({"text": text})
        else:
            content = text

        req = (
            ReplyMessageRequest.builder()
            .message_id(message_id)
            .request_body(
                ReplyMessageRequestBody.builder()
                .msg_type(msg_type)
                .content(content)
                .build()
            )
            .build()
        )
        resp = self.client.im.v1.message.reply(req)
        if resp.success():
            mid = resp.data.message_id if resp.data else None
            logger.info("[REPLY] OK message_id=%s", mid)
            return mid
        else:
            logger.error("[REPLY] FAIL code=%s msg=%s", resp.code, resp.msg)
            return None

    def update_message(self, message_id: str, text: str, msg_type: str = "text"):
        """Update (PATCH) an existing message — used for streaming output."""
        if msg_type == "text":
            content = json.dumps({"text": text})
        else:
            content = text

        req = (
            PatchMessageRequest.builder()
            .message_id(message_id)
            .request_body(
                PatchMessageRequestBody.builder()
                .content(content)
                .build()
            )
            .build()
        )
        resp = self.client.im.v1.message.patch(req)
        if not resp.success():
            logger.warning("[PATCH] FAIL code=%s msg=%s", resp.code, resp.msg)

    # ── Sending: Media ──

    def upload_image(self, image_path: str) -> Optional[str]:
        """Upload an image and return its image_key."""
        path = Path(image_path)
        if not path.exists():
            logger.error("Image not found: %s", image_path)
            return None

        req = (
            CreateImageRequest.builder()
            .request_body(
                CreateImageRequestBody.builder()
                .image_type("message")
                .image(open(image_path, "rb"))
                .build()
            )
            .build()
        )
        resp = self.client.im.v1.image.create(req)
        if resp.success():
            key = resp.data.image_key if resp.data else None
            logger.info("[UPLOAD] Image OK: %s", key)
            return key
        else:
            logger.error("[UPLOAD] Image FAIL: %s", resp.msg)
            return None

    def upload_file(self, file_path: str, file_type: str = "stream") -> Optional[str]:
        """Upload a file and return its file_key."""
        path = Path(file_path)
        if not path.exists():
            logger.error("File not found: %s", file_path)
            return None

        req = (
            CreateFileRequest.builder()
            .request_body(
                CreateFileRequestBody.builder()
                .file_type(file_type)
                .file_name(path.name)
                .file(open(file_path, "rb"))
                .build()
            )
            .build()
        )
        resp = self.client.im.v1.file.create(req)
        if resp.success():
            key = resp.data.file_key if resp.data else None
            logger.info("[UPLOAD] File OK: %s", key)
            return key
        else:
            logger.error("[UPLOAD] File FAIL: %s", resp.msg)
            return None

    def send_image(self, chat_id: str, image_path: str) -> Optional[str]:
        """Upload and send an image to a chat."""
        image_key = self.upload_image(image_path)
        if not image_key:
            return None
        content = json.dumps({"image_key": image_key})
        return self.send_message(chat_id, content, msg_type="image")

    def send_file(self, chat_id: str, file_path: str) -> Optional[str]:
        """Upload and send a file to a chat."""
        file_key = self.upload_file(file_path)
        if not file_key:
            return None
        content = json.dumps({"file_key": file_key})
        return self.send_message(chat_id, content, msg_type="file")

    # ── Push helpers ──

    def push_message(self, text: str, chat_id: Optional[str] = None) -> Optional[str]:
        """Proactively push a text message to the default or specified chat."""
        target = chat_id or self.push_chat_id
        if not target:
            logger.warning("No push_chat_id configured, cannot push message")
            return None
        return self.send_message(target, text)

    def push_image(self, image_path: str, chat_id: Optional[str] = None) -> Optional[str]:
        """Proactively push an image."""
        target = chat_id or self.push_chat_id
        if not target:
            logger.warning("No push_chat_id configured")
            return None
        return self.send_image(target, image_path)

    def push_file(self, file_path: str, chat_id: Optional[str] = None) -> Optional[str]:
        """Proactively push a file."""
        target = chat_id or self.push_chat_id
        if not target:
            logger.warning("No push_chat_id configured")
            return None
        return self.send_file(target, file_path)
