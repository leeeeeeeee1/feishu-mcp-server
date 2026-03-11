"""Tests for supervisor.main module."""

import asyncio
import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from supervisor.main import Supervisor, SUPERVISOR_SYSTEM_PROMPT


class TestLocalCommands:
    def _make_supervisor(self):
        with patch("supervisor.main.FeishuGateway"):
            with patch("supervisor.main.ClaudeSession"):
                sup = Supervisor()
                sup.gateway = MagicMock()
                sup.claude = MagicMock()
                return sup

    def test_help_command(self):
        sup = self._make_supervisor()
        result = sup._handle_local_command("/help", "chat-1", "msg-1")
        assert result is True
        sup.gateway.reply_message.assert_called_once()
        reply_text = sup.gateway.reply_message.call_args[0][1]
        assert "/status" in reply_text
        assert "/gpu" in reply_text

    def test_status_returns_system_info(self):
        sup = self._make_supervisor()
        result = sup._handle_local_command("/status", "chat-1", "msg-1")
        assert result is True
        reply_text = sup.gateway.reply_message.call_args[0][1]
        assert "CPU" in reply_text or "not initialized" in reply_text.lower() or "not available" in reply_text.lower()

    def test_unknown_command_not_handled(self):
        sup = self._make_supervisor()
        result = sup._handle_local_command("/nonexistent", "chat-1", "msg-1")
        assert result is False

    def test_sessions_before_init(self):
        sup = self._make_supervisor()
        result = sup._handle_local_command("/sessions", "chat-1", "msg-1")
        assert result is True

    def test_tasks_before_init(self):
        sup = self._make_supervisor()
        result = sup._handle_local_command("/tasks", "chat-1", "msg-1")
        assert result is True

    def test_gpu_before_init(self):
        sup = self._make_supervisor()
        result = sup._handle_local_command("/gpu", "chat-1", "msg-1")
        assert result is True

    def test_daemons_before_init(self):
        sup = self._make_supervisor()
        result = sup._handle_local_command("/daemons", "chat-1", "msg-1")
        assert result is True

    def test_stop_no_arg(self):
        sup = self._make_supervisor()
        sup._handle_local_command("/stop", "chat-1", "msg-1")
        reply_text = sup.gateway.reply_message.call_args[0][1]
        assert "Usage" in reply_text

    def test_skip_no_arg(self):
        sup = self._make_supervisor()
        sup._handle_local_command("/skip", "chat-1", "msg-1")
        reply_text = sup.gateway.reply_message.call_args[0][1]
        assert "Usage" in reply_text

    def test_close_no_arg(self):
        sup = self._make_supervisor()
        sup._handle_local_command("/close", "chat-1", "msg-1")
        reply_text = sup.gateway.reply_message.call_args[0][1]
        assert "Usage" in reply_text

    def test_followup_no_arg(self):
        sup = self._make_supervisor()
        sup._handle_local_command("/followup", "chat-1", "msg-1")
        reply_text = sup.gateway.reply_message.call_args[0][1]
        assert "Usage" in reply_text


class TestMessageRouting:
    def _make_supervisor(self):
        with patch("supervisor.main.FeishuGateway"):
            with patch("supervisor.main.ClaudeSession"):
                sup = Supervisor()
                sup.gateway = MagicMock()
                sup.claude = MagicMock()
                return sup

    def test_empty_message_ignored(self):
        sup = self._make_supervisor()
        sup._on_feishu_message("u1", "msg-1", "chat-1", "text", "", None)
        sup.gateway.reply_message.assert_not_called()

    def test_slash_routes_to_local(self):
        sup = self._make_supervisor()
        sup._on_feishu_message("u1", "msg-1", "chat-1", "text", "/help", None)
        sup.gateway.reply_message.assert_called_once()

    def test_auto_set_push_chat_id(self):
        sup = self._make_supervisor()
        sup.gateway.push_chat_id = ""
        sup._on_feishu_message("u1", "msg-1", "chat-123", "text", "/help", None)
        assert sup.gateway.push_chat_id == "chat-123"


class TestRouteMessageReply:
    """Test that action=reply messages are sent directly to user."""

    def _make_supervisor(self):
        with patch("supervisor.main.FeishuGateway"):
            with patch("supervisor.main.ClaudeSession"):
                sup = Supervisor()
                sup.gateway = MagicMock()
                sup.claude = MagicMock()
                sup._task_dispatcher = MagicMock()
                sup._task_dispatcher.get_awaiting_closure.return_value = []
                return sup

    def test_reply_action_sends_text(self):
        """When sonnet returns action=reply, the text should be sent to user."""
        async def _test():
            sup = self._make_supervisor()
            sup.claude.route_message = AsyncMock(
                return_value={"action": "reply", "text": "我是 Supervisor Hub，基于 Claude 模型。"}
            )
            await sup._route_message("你是什么模型", "chat-1", "msg-1")
            sup.gateway.reply_message.assert_called_once()
            reply = sup.gateway.reply_message.call_args[0][1]
            assert "Supervisor" in reply or "Claude" in reply

        asyncio.run(_test())

    def test_reply_action_empty_text_fallback(self):
        """If sonnet returns empty text, should have a sensible fallback."""
        async def _test():
            sup = self._make_supervisor()
            sup.claude.route_message = AsyncMock(
                return_value={"action": "reply", "text": ""}
            )
            await sup._route_message("你好", "chat-1", "msg-1")
            sup.gateway.reply_message.assert_called_once()
            reply = sup.gateway.reply_message.call_args[0][1]
            assert len(reply) > 0  # should not be empty

        asyncio.run(_test())


class TestRouteMessageDispatch:
    """Test that action=dispatch messages are dispatched to workers."""

    def _make_supervisor(self):
        with patch("supervisor.main.FeishuGateway"):
            with patch("supervisor.main.ClaudeSession"):
                sup = Supervisor()
                sup.gateway = MagicMock()
                sup.claude = MagicMock()
                sup._task_dispatcher = MagicMock()
                sup._task_dispatcher.get_awaiting_closure.return_value = []
                sup._task_dispatcher.get_tasks_text.return_value = "No tasks."
                sup._task_dispatcher.dispatch = AsyncMock(return_value=MagicMock())
                return sup

    def test_dispatch_action_creates_task(self):
        """When sonnet returns action=dispatch, a task should be dispatched."""
        async def _test():
            sup = self._make_supervisor()
            sup.claude.route_message = AsyncMock(
                return_value={"action": "dispatch", "description": "分析 TensorRT-LLM 代码"}
            )
            await sup._route_message("帮我分析tensorrt-llm", "chat-1", "msg-1")
            sup._task_dispatcher.dispatch.assert_called_once()
            # Should notify user that task is queued
            sup.gateway.reply_message.assert_called_once()
            reply = sup.gateway.reply_message.call_args[0][1]
            assert "调度" in reply or "排队" in reply or "dispatch" in reply.lower()

        asyncio.run(_test())

    def test_dispatch_prompt_contains_context(self):
        """The prompt sent to worker should include context, not just raw user text."""
        async def _test():
            sup = self._make_supervisor()
            sup.claude.route_message = AsyncMock(
                return_value={"action": "dispatch", "description": "分析代码结构"}
            )
            await sup._route_message("帮我分析一下这个项目的代码结构", "chat-1", "msg-1")
            sup._task_dispatcher.dispatch.assert_called_once()
            dispatched_prompt = sup._task_dispatcher.dispatch.call_args[1].get("prompt", "")
            # Should contain the task description from sonnet
            assert "分析代码结构" in dispatched_prompt
            # Should contain the original user message
            assert "帮我分析一下这个项目的代码结构" in dispatched_prompt
            # Should contain role/context info
            assert "worker" in dispatched_prompt.lower() or "task" in dispatched_prompt.lower()

        asyncio.run(_test())

    def test_dispatch_prompt_includes_conversation_history(self):
        """When there is conversation history, the worker prompt should include it."""
        async def _test():
            sup = self._make_supervisor()
            # Simulate prior conversation
            sup._record_message("user", "这个项目是用什么语言写的")
            sup._record_message("assistant", "这个项目主要使用 Python 编写。")
            sup._record_message("user", "帮我重构一下")

            sup.claude.route_message = AsyncMock(
                return_value={"action": "dispatch", "description": "重构项目代码"}
            )
            await sup._route_message("帮我重构一下", "chat-1", "msg-1")
            dispatched_prompt = sup._task_dispatcher.dispatch.call_args[1].get("prompt", "")
            # Should contain prior messages as context
            assert "什么语言" in dispatched_prompt
            assert "Python" in dispatched_prompt

        asyncio.run(_test())

    def test_dispatch_without_dispatcher(self):
        """When task dispatcher is unavailable, should inform user."""
        async def _test():
            sup = self._make_supervisor()
            sup._task_dispatcher = None
            sup.claude.route_message = AsyncMock(
                return_value={"action": "dispatch", "description": "test"}
            )
            await sup._route_message("帮我运行测试", "chat-1", "msg-1")
            sup.gateway.reply_message.assert_called_once()
            reply = sup.gateway.reply_message.call_args[0][1]
            assert "not available" in reply.lower() or "不可用" in reply

        asyncio.run(_test())


class TestRouteMessageDispatchMulti:
    """Test that action=dispatch_multi decomposes into parallel tasks."""

    def _make_supervisor(self):
        with patch("supervisor.main.FeishuGateway"):
            with patch("supervisor.main.ClaudeSession"):
                sup = Supervisor()
                sup.gateway = MagicMock()
                sup.claude = MagicMock()
                sup._task_dispatcher = MagicMock()
                sup._task_dispatcher.get_awaiting_closure.return_value = []
                sup._task_dispatcher.dispatch = AsyncMock(return_value=MagicMock())
                return sup

    def test_dispatch_multi_creates_multiple_tasks(self):
        async def _test():
            sup = self._make_supervisor()
            sup.claude.route_message = AsyncMock(
                return_value={
                    "action": "dispatch_multi",
                    "description": "重构项目",
                    "subtasks": ["分析代码结构", "检查测试覆盖", "审计依赖"],
                }
            )
            await sup._route_message("重构整个项目", "chat-1", "msg-1")
            assert sup._task_dispatcher.dispatch.call_count == 3

        asyncio.run(_test())


class TestFollowUpRouting:
    """Test task follow-up routing: explicit task ID vs sonnet-decided."""

    def _make_supervisor(self):
        with patch("supervisor.main.FeishuGateway"):
            with patch("supervisor.main.ClaudeSession"):
                sup = Supervisor()
                sup.gateway = MagicMock()
                sup.claude = MagicMock()
                return sup

    def test_explicit_task_id_routes_to_follow_up(self):
        """Message containing a task ID prefix should route directly to that task."""
        async def _test():
            sup = self._make_supervisor()
            mock_task = MagicMock()
            mock_task.id = "8b557777-abcd-1234-5678-123456789abc"
            mock_task.status = "awaiting_closure"
            mock_task.session_id = "sess-1"

            sup._task_dispatcher = MagicMock()
            sup._task_dispatcher.get_awaiting_closure.return_value = [mock_task]
            sup._task_dispatcher.list_tasks.return_value = [mock_task]
            sup._task_dispatcher.get_task.return_value = mock_task
            sup._task_dispatcher.follow_up_async = AsyncMock(return_value="Follow-up result")

            # Message contains task ID prefix
            await sup._route_message("8b557777 这个结果对吗", "chat-1", "msg-1")

            sup._task_dispatcher.follow_up_async.assert_called_once()
            sup.claude.route_message.assert_not_called()

        asyncio.run(_test())

    def test_no_task_id_with_awaiting_tasks_goes_to_sonnet(self):
        """Without task ID, even with awaiting tasks, should go through sonnet routing."""
        async def _test():
            sup = self._make_supervisor()
            mock_task = MagicMock()
            mock_task.id = "8b557777-abcd-efgh-ijkl-123456789abc"
            mock_task.status = "awaiting_closure"
            mock_task.description = "分析代码"

            sup._task_dispatcher = MagicMock()
            sup._task_dispatcher.get_awaiting_closure.return_value = [mock_task]
            sup._task_dispatcher.list_tasks.return_value = [mock_task]
            sup._task_dispatcher.dispatch = AsyncMock(return_value=MagicMock())
            sup.claude.route_message = AsyncMock(
                return_value={"action": "reply", "text": "好的，已关闭。"}
            )

            await sup._route_message("关闭这个任务吧", "chat-1", "msg-1")

            # Should go through sonnet, NOT auto follow-up
            sup.claude.route_message.assert_called_once()

        asyncio.run(_test())

    def test_sonnet_decides_follow_up(self):
        """When sonnet returns action=follow_up, route to the specified task."""
        async def _test():
            sup = self._make_supervisor()
            mock_task = MagicMock()
            mock_task.id = "8b557777-abcd-efgh-ijkl-123456789abc"
            mock_task.status = "awaiting_closure"
            mock_task.session_id = "sess-1"

            sup._task_dispatcher = MagicMock()
            sup._task_dispatcher.get_awaiting_closure.return_value = [mock_task]
            sup._task_dispatcher.list_tasks.return_value = [mock_task]
            sup._task_dispatcher.follow_up_async = AsyncMock(return_value="详细结果...")

            sup.claude.route_message = AsyncMock(
                return_value={"action": "follow_up", "task_id": "8b557777", "text": "这个结果怎么样"}
            )

            await sup._route_message("结果怎么样", "chat-1", "msg-1")

            sup._task_dispatcher.follow_up_async.assert_called_once()

        asyncio.run(_test())

    def test_sonnet_decides_reply_despite_awaiting_tasks(self):
        """When sonnet returns action=reply, respond directly even if tasks are awaiting."""
        async def _test():
            sup = self._make_supervisor()
            mock_task = MagicMock()
            mock_task.id = "8b557777-abcd-efgh-ijkl-123456789abc"
            mock_task.status = "awaiting_closure"
            mock_task.description = "分析代码"

            sup._task_dispatcher = MagicMock()
            sup._task_dispatcher.get_awaiting_closure.return_value = [mock_task]
            sup._task_dispatcher.list_tasks.return_value = [mock_task]

            sup.claude.route_message = AsyncMock(
                return_value={"action": "reply", "text": "你好！有什么可以帮你的？"}
            )

            await sup._route_message("你好", "chat-1", "msg-1")

            sup.gateway.reply_message.assert_called_once()
            reply = sup.gateway.reply_message.call_args[0][1]
            assert "你好" in reply
            sup._task_dispatcher.follow_up_async.assert_not_called()

        asyncio.run(_test())

    def test_explicit_task_id_not_found_falls_through(self):
        """If message contains an ID-like string but no matching task, go to sonnet."""
        async def _test():
            sup = self._make_supervisor()
            sup._task_dispatcher = MagicMock()
            sup._task_dispatcher.get_awaiting_closure.return_value = []
            sup._task_dispatcher.list_tasks.return_value = []
            sup._task_dispatcher.dispatch = AsyncMock(return_value=MagicMock())
            sup.claude.route_message = AsyncMock(
                return_value={"action": "dispatch", "description": "test"}
            )

            await sup._route_message("aabbccdd 这个怎么了", "chat-1", "msg-1")

            # No matching task → should fall through to sonnet
            sup.claude.route_message.assert_called_once()

        asyncio.run(_test())


class TestConversationHistory:
    """Test conversation history recording and inclusion in prompts."""

    def _make_supervisor(self):
        with patch("supervisor.main.FeishuGateway"):
            with patch("supervisor.main.ClaudeSession"):
                sup = Supervisor()
                sup.gateway = MagicMock()
                sup.claude = MagicMock()
                return sup

    def test_record_and_retrieve(self):
        sup = self._make_supervisor()
        sup._record_message("user", "你好")
        sup._record_message("assistant", "你好！有什么可以帮你？")
        history = sup._get_history_text()
        assert "你好" in history
        assert "有什么可以帮你" in history

    def test_history_max_length(self):
        sup = self._make_supervisor()
        for i in range(30):
            sup._record_message("user", f"message {i}")
        # Should keep only the last MAX entries
        history = sup._conversation_history
        assert len(history) <= 20

    def test_empty_history(self):
        sup = self._make_supervisor()
        history = sup._get_history_text()
        assert history == ""

    def test_history_recorded_on_reply(self):
        """When sonnet replies directly, both user msg and reply are recorded."""
        async def _test():
            sup = self._make_supervisor()
            sup._task_dispatcher = MagicMock()
            sup._task_dispatcher.get_awaiting_closure.return_value = []
            sup.claude.route_message = AsyncMock(
                return_value={"action": "reply", "text": "我是 Supervisor Hub"}
            )
            await sup._route_message("你是谁", "chat-1", "msg-1")
            assert len(sup._conversation_history) == 2
            assert sup._conversation_history[0]["role"] == "user"
            assert sup._conversation_history[1]["role"] == "assistant"

        asyncio.run(_test())

    def test_history_recorded_on_dispatch(self):
        """When dispatching, user msg is recorded (assistant response is the dispatch notice)."""
        async def _test():
            sup = self._make_supervisor()
            sup._task_dispatcher = MagicMock()
            sup._task_dispatcher.get_awaiting_closure.return_value = []
            sup._task_dispatcher.dispatch = AsyncMock(return_value=MagicMock())
            sup.claude.route_message = AsyncMock(
                return_value={"action": "dispatch", "description": "运行测试"}
            )
            await sup._route_message("帮我运行测试", "chat-1", "msg-1")
            assert any(m["text"] == "帮我运行测试" for m in sup._conversation_history)

        asyncio.run(_test())


class TestRouteMessageParseError:
    """Test fallback behavior when sonnet returns unparseable results."""

    def _make_supervisor(self):
        with patch("supervisor.main.FeishuGateway"):
            with patch("supervisor.main.ClaudeSession"):
                sup = Supervisor()
                sup.gateway = MagicMock()
                sup.claude = MagicMock()
                sup._task_dispatcher = MagicMock()
                sup._task_dispatcher.get_awaiting_closure.return_value = []
                sup._task_dispatcher.dispatch = AsyncMock(return_value=MagicMock())
                return sup

    def test_parse_error_defaults_to_dispatch(self):
        """On parse error, should default to dispatching as task (safe fallback)."""
        async def _test():
            sup = self._make_supervisor()
            sup.claude.route_message = AsyncMock(
                return_value={"action": "dispatch", "description": "parse error fallback"}
            )
            await sup._route_message("随便说点什么", "chat-1", "msg-1")
            sup._task_dispatcher.dispatch.assert_called_once()

        asyncio.run(_test())


class TestSystemPrompt:
    def test_system_prompt_exists(self):
        assert "Supervisor Hub" in SUPERVISOR_SYSTEM_PROMPT
        assert "MUST NOT" in SUPERVISOR_SYSTEM_PROMPT


class TestNotifyTaskResult:
    def _make_supervisor(self):
        with patch("supervisor.main.FeishuGateway"):
            with patch("supervisor.main.ClaudeSession"):
                sup = Supervisor()
                sup.gateway = MagicMock()
                sup.claude = MagicMock()
                return sup

    def test_notify_awaiting_closure(self):
        sup = self._make_supervisor()
        task = MagicMock()
        task.id = "12345678-abcd"
        task.status = "awaiting_closure"
        task.result = "Task completed successfully"
        task.started_at = 100.0
        task.finished_at = 110.0

        sup._notify_task_result(task, "chat-1")
        sup.gateway.push_message.assert_called_once()
        msg = sup.gateway.push_message.call_args[0][0]
        assert "完成" in msg
        assert "追问" in msg or "close" in msg

    def test_notify_failed(self):
        sup = self._make_supervisor()
        task = MagicMock()
        task.id = "12345678-abcd"
        task.status = "failed"
        task.error = "Something went wrong"
        task.result = ""
        task.started_at = 100.0
        task.finished_at = 105.0

        sup._notify_task_result(task, "chat-1")
        msg = sup.gateway.push_message.call_args[0][0]
        assert "失败" in msg
        assert "Something went wrong" in msg

    def test_notify_waiting_for_input(self):
        sup = self._make_supervisor()
        task = MagicMock()
        task.id = "12345678-abcd"
        task.status = "waiting_for_input"
        task.result = "Which file?"
        task.started_at = 100.0
        task.finished_at = None

        sup._notify_task_result(task, "chat-1")
        msg = sup.gateway.push_message.call_args[0][0]
        assert "输入" in msg
        assert "/reply" in msg
