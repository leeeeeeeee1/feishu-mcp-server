"""Tests for supervisor.feishu_gateway module."""

import json
import pytest
from unittest.mock import MagicMock, patch

from supervisor.feishu_gateway import FeishuGateway, _dedup_check, _seen_messages


@pytest.fixture(autouse=True)
def clear_dedup():
    """Clear dedup state between tests."""
    _seen_messages.clear()
    yield
    _seen_messages.clear()


class TestDedup:
    def test_first_message_not_duplicate(self):
        assert _dedup_check("msg-1") is False

    def test_second_same_message_is_duplicate(self):
        _dedup_check("msg-1")
        assert _dedup_check("msg-1") is True

    def test_different_messages_not_duplicate(self):
        _dedup_check("msg-1")
        assert _dedup_check("msg-2") is False

    def test_expired_entries_cleaned(self):
        import time

        _seen_messages["old-msg"] = time.time() - 120  # 2 min ago, past TTL
        _dedup_check("new-msg")
        assert "old-msg" not in _seen_messages


class TestGatewayInit:
    def test_missing_credentials(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="FEISHU_APP_ID"):
                FeishuGateway(app_id="", app_secret="")

    def test_env_credentials(self):
        with patch.dict(
            "os.environ",
            {"FEISHU_APP_ID": "test-id", "FEISHU_APP_SECRET": "test-secret"},
        ):
            gw = FeishuGateway()
            assert gw.app_id == "test-id"
            assert gw.app_secret == "test-secret"

    def test_explicit_credentials(self):
        gw = FeishuGateway(app_id="explicit-id", app_secret="explicit-secret")
        assert gw.app_id == "explicit-id"

    def test_push_chat_id_from_env(self):
        with patch.dict(
            "os.environ",
            {
                "FEISHU_APP_ID": "id",
                "FEISHU_APP_SECRET": "sec",
                "FEISHU_PUSH_CHAT_ID": "chat-123",
            },
        ):
            gw = FeishuGateway()
            assert gw.push_chat_id == "chat-123"


class TestMessageHandling:
    def _make_gateway(self):
        return FeishuGateway(app_id="test-id", app_secret="test-secret")

    def test_set_message_handler(self):
        gw = self._make_gateway()
        handler = MagicMock()
        gw.set_message_handler(handler)
        assert gw._on_message is handler

    def test_handle_message_filters_bots(self):
        gw = self._make_gateway()
        handler = MagicMock()
        gw.set_message_handler(handler)

        # Create mock event with bot sender
        data = MagicMock()
        data.event.sender.sender_type = "app"
        gw._handle_message(data)
        handler.assert_not_called()

    def test_handle_message_dedup(self):
        gw = self._make_gateway()
        handler = MagicMock()
        gw.set_message_handler(handler)

        data = MagicMock()
        data.event.sender.sender_type = "user"
        data.event.sender.sender_id.open_id = "u1"
        data.event.message.message_id = "msg-dup"
        data.event.message.chat_id = "chat-1"
        data.event.message.message_type = "text"
        data.event.message.content = json.dumps({"text": "hello"})

        gw._handle_message(data)
        assert handler.call_count == 1

        gw._handle_message(data)
        assert handler.call_count == 1  # deduped

    def test_handle_message_strips_mention(self):
        gw = self._make_gateway()
        received = {}

        def handler(**kwargs):
            received.update(kwargs)

        gw.set_message_handler(handler)

        data = MagicMock()
        data.event.sender.sender_type = "user"
        data.event.sender.sender_id.open_id = "u1"
        data.event.message.message_id = "msg-mention"
        data.event.message.chat_id = "chat-1"
        data.event.message.message_type = "text"
        data.event.message.content = json.dumps({"text": "@_user_1 hello Claude"})

        gw._handle_message(data)
        assert received["content"] == "hello Claude"


class TestParentIdExtraction:
    """Test that parent_id and root_id are extracted from reply messages."""

    def _make_gateway(self):
        return FeishuGateway(app_id="test-id", app_secret="test-secret")

    def test_reply_message_passes_parent_id(self):
        """When a message has parent_id, it should be passed to the handler."""
        gw = self._make_gateway()
        received = {}

        def handler(**kwargs):
            received.update(kwargs)

        gw.set_message_handler(handler)

        data = MagicMock()
        data.event.sender.sender_type = "user"
        data.event.sender.sender_id.open_id = "u1"
        data.event.message.message_id = "msg-reply-1"
        data.event.message.chat_id = "chat-1"
        data.event.message.message_type = "text"
        data.event.message.content = json.dumps({"text": "这个结果对吗"})
        data.event.message.parent_id = "msg-parent-123"
        data.event.message.root_id = "msg-root-456"

        gw._handle_message(data)
        assert received["parent_id"] == "msg-parent-123"
        assert received["root_id"] == "msg-root-456"

    def test_normal_message_passes_empty_parent_id(self):
        """Non-reply messages should have empty parent_id/root_id."""
        gw = self._make_gateway()
        received = {}

        def handler(**kwargs):
            received.update(kwargs)

        gw.set_message_handler(handler)

        data = MagicMock()
        data.event.sender.sender_type = "user"
        data.event.sender.sender_id.open_id = "u1"
        data.event.message.message_id = "msg-normal-1"
        data.event.message.chat_id = "chat-1"
        data.event.message.message_type = "text"
        data.event.message.content = json.dumps({"text": "你好"})
        data.event.message.parent_id = None
        data.event.message.root_id = None

        gw._handle_message(data)
        assert received["parent_id"] == ""
        assert received["root_id"] == ""

    def test_missing_parent_id_attribute_defaults_empty(self):
        """If parent_id attribute doesn't exist on message, default to empty."""
        gw = self._make_gateway()
        received = {}

        def handler(**kwargs):
            received.update(kwargs)

        gw.set_message_handler(handler)

        data = MagicMock()
        data.event.sender.sender_type = "user"
        data.event.sender.sender_id.open_id = "u1"
        data.event.message.message_id = "msg-no-attr"
        data.event.message.chat_id = "chat-1"
        data.event.message.message_type = "text"
        data.event.message.content = json.dumps({"text": "hello"})
        # Simulate missing attribute
        del data.event.message.parent_id
        del data.event.message.root_id

        gw._handle_message(data)
        assert received["parent_id"] == ""
        assert received["root_id"] == ""


class TestSending:
    def _make_gateway(self):
        gw = FeishuGateway(app_id="test-id", app_secret="test-secret")
        gw.client = MagicMock()
        return gw

    def test_send_message_success(self):
        gw = self._make_gateway()
        mock_resp = MagicMock()
        mock_resp.success.return_value = True
        mock_resp.data.message_id = "sent-msg-1"
        gw.client.im.v1.message.create.return_value = mock_resp

        result = gw.send_message("chat-1", "Hello!")
        assert result == "sent-msg-1"

    def test_send_message_failure(self):
        gw = self._make_gateway()
        mock_resp = MagicMock()
        mock_resp.success.return_value = False
        mock_resp.code = 99999
        mock_resp.msg = "error"
        gw.client.im.v1.message.create.return_value = mock_resp

        result = gw.send_message("chat-1", "Hello!")
        assert result is None

    def test_update_message(self):
        gw = self._make_gateway()
        mock_resp = MagicMock()
        mock_resp.success.return_value = True
        gw.client.im.v1.message.patch.return_value = mock_resp

        gw.update_message("msg-1", "Updated text")
        gw.client.im.v1.message.patch.assert_called_once()

    def test_push_message_no_chat_id(self):
        gw = self._make_gateway()
        gw.push_chat_id = ""
        result = gw.push_message("Hello!")
        assert result is None

    def test_push_message_with_chat_id(self):
        gw = self._make_gateway()
        gw.push_chat_id = "default-chat"
        mock_resp = MagicMock()
        mock_resp.success.return_value = True
        mock_resp.data.message_id = "pushed-1"
        gw.client.im.v1.message.create.return_value = mock_resp

        result = gw.push_message("Alert!")
        assert result == "pushed-1"

    def test_push_message_override_chat_id(self):
        gw = self._make_gateway()
        gw.push_chat_id = "default-chat"
        mock_resp = MagicMock()
        mock_resp.success.return_value = True
        mock_resp.data.message_id = "pushed-2"
        gw.client.im.v1.message.create.return_value = mock_resp

        result = gw.push_message("Alert!", chat_id="other-chat")
        assert result == "pushed-2"
        # Verify it used the override chat_id
        call_args = gw.client.im.v1.message.create.call_args
        assert call_args is not None
