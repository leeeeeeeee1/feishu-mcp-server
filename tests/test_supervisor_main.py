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

    def test_close_multiple_ids(self):
        """'/close id1 id2' closes multiple tasks."""
        sup = self._make_supervisor()
        mock_dispatcher = MagicMock()
        mock_t1 = MagicMock()
        mock_t1.id = "aaaa1111-0000-0000-0000-000000000000"
        mock_t2 = MagicMock()
        mock_t2.id = "bbbb2222-0000-0000-0000-000000000000"
        mock_dispatcher.list_tasks.return_value = [mock_t1, mock_t2]
        mock_dispatcher.close_tasks.return_value = [
            "Task aaaa1111 closed.",
            "Task bbbb2222 closed.",
        ]
        sup._task_dispatcher = mock_dispatcher
        sup._handle_local_command("/close aaaa1111 bbbb2222", "chat-1", "msg-1")
        reply_text = sup.gateway.reply_message.call_args[0][1]
        assert "aaaa1111" in reply_text
        assert "bbbb2222" in reply_text
        mock_dispatcher.close_tasks.assert_called_once_with(
            [mock_t1.id, mock_t2.id]
        )

    def test_close_mixed_valid_and_invalid_prefix(self):
        """'/close bad good' reports error for bad and closes the valid one."""
        sup = self._make_supervisor()
        mock_dispatcher = MagicMock()
        mock_t1 = MagicMock()
        mock_t1.id = "aaaa1111-0000-0000-0000-000000000000"
        mock_dispatcher.list_tasks.return_value = [mock_t1]
        mock_dispatcher.close_tasks.return_value = ["Task aaaa1111 closed."]
        sup._task_dispatcher = mock_dispatcher
        sup._handle_local_command("/close unknownprefix aaaa1111", "chat-1", "msg-1")
        reply_text = sup.gateway.reply_message.call_args[0][1]
        assert "aaaa1111" in reply_text
        assert "No task found" in reply_text

    def test_close_all_case_insensitive(self):
        """'/close ALL' works the same as '/close all'."""
        sup = self._make_supervisor()
        mock_dispatcher = MagicMock()
        mock_t1 = MagicMock()
        mock_t1.id = "aaaa1111-0000-0000-0000-000000000000"
        mock_dispatcher.get_awaiting_closure.return_value = [mock_t1]
        mock_dispatcher.close_tasks.return_value = ["Task aaaa1111 closed."]
        sup._task_dispatcher = mock_dispatcher
        sup._handle_local_command("/close ALL", "chat-1", "msg-1")
        reply_text = sup.gateway.reply_message.call_args[0][1]
        assert "aaaa1111" in reply_text
        mock_dispatcher.close_tasks.assert_called_once_with([mock_t1.id])

    def test_close_all(self):
        """'/close all' closes all awaiting_closure tasks."""
        sup = self._make_supervisor()
        mock_dispatcher = MagicMock()
        mock_t1 = MagicMock()
        mock_t1.id = "aaaa1111-0000-0000-0000-000000000000"
        mock_t2 = MagicMock()
        mock_t2.id = "bbbb2222-0000-0000-0000-000000000000"
        mock_dispatcher.get_awaiting_closure.return_value = [mock_t1, mock_t2]
        mock_dispatcher.close_tasks.return_value = [
            "Task aaaa1111 closed.",
            "Task bbbb2222 closed.",
        ]
        sup._task_dispatcher = mock_dispatcher
        sup._handle_local_command("/close all", "chat-1", "msg-1")
        reply_text = sup.gateway.reply_message.call_args[0][1]
        assert "aaaa1111" in reply_text
        assert "bbbb2222" in reply_text
        mock_dispatcher.close_tasks.assert_called_once_with(
            [mock_t1.id, mock_t2.id]
        )

    def test_close_all_nothing_to_close(self):
        """'/close all' when no tasks awaiting closure."""
        sup = self._make_supervisor()
        mock_dispatcher = MagicMock()
        mock_dispatcher.get_awaiting_closure.return_value = []
        sup._task_dispatcher = mock_dispatcher
        sup._handle_local_command("/close all", "chat-1", "msg-1")
        reply_text = sup.gateway.reply_message.call_args[0][1]
        assert "No tasks" in reply_text or "没有" in reply_text

    def test_close_help_text_shows_batch(self):
        """Help text mentions batch close and 'all' keyword."""
        sup = self._make_supervisor()
        sup._handle_local_command("/help", "chat-1", "msg-1")
        reply_text = sup.gateway.reply_message.call_args[0][1]
        assert "all" in reply_text.lower()

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


class TestGetTasksContext:
    """Test _get_tasks_context: all non-terminal statuses must be visible."""

    ALL_STATUSES = [
        "pending", "running", "waiting_for_input", "done",
        "awaiting_closure", "follow_up", "review", "learning",
        "completed", "failed", "cancelled",
    ]
    TERMINAL_STATUSES = {"completed", "cancelled"}

    def _make_supervisor(self):
        with patch("supervisor.main.FeishuGateway"):
            with patch("supervisor.main.ClaudeSession"):
                sup = Supervisor()
                sup.gateway = MagicMock()
                sup.claude = MagicMock()
                return sup

    def _make_task(self, status, idx=0):
        t = MagicMock()
        t.id = f"aa{idx:06d}-1234-5678-9abc-123456789abc"
        t.status = status
        t.description = f"{status} task"
        t.prompt = f"{status} prompt"
        t.error = "some error" if status == "failed" else ""
        t.current_step = "step1"
        t.steps_completed = []
        t.started_at = 100.0
        return t

    def test_all_non_terminal_statuses_visible(self):
        """Every non-terminal status must appear in either active or awaiting."""
        sup = self._make_supervisor()
        tasks = [self._make_task(s, i) for i, s in enumerate(self.ALL_STATUSES)]
        sup._task_dispatcher = MagicMock()
        sup._task_dispatcher.list_tasks.return_value = tasks
        ctx = sup._get_tasks_context()

        active_ids = {item["id"] for item in ctx["active"]}
        awaiting_ids = {item["id"] for item in ctx["awaiting"]}
        visible_ids = active_ids | awaiting_ids

        for i, status in enumerate(self.ALL_STATUSES):
            task_id_prefix = f"aa{i:06d}"[:8]
            if status in self.TERMINAL_STATUSES:
                assert task_id_prefix not in visible_ids, (
                    f"Terminal status '{status}' should NOT be visible"
                )
            else:
                assert task_id_prefix in visible_ids, (
                    f"Non-terminal status '{status}' MUST be visible"
                )

    def test_terminal_statuses_excluded(self):
        sup = self._make_supervisor()
        tasks = [self._make_task(s, i) for i, s in enumerate(self.TERMINAL_STATUSES)]
        sup._task_dispatcher = MagicMock()
        sup._task_dispatcher.list_tasks.return_value = tasks
        ctx = sup._get_tasks_context()
        assert len(ctx["active"]) == 0
        assert len(ctx["awaiting"]) == 0

    def test_awaiting_closure_in_awaiting_not_active(self):
        sup = self._make_supervisor()
        tasks = [self._make_task("awaiting_closure")]
        sup._task_dispatcher = MagicMock()
        sup._task_dispatcher.list_tasks.return_value = tasks
        ctx = sup._get_tasks_context()
        assert len(ctx["awaiting"]) == 1
        assert len(ctx["active"]) == 0

    def test_includes_failed_tasks_with_error(self):
        sup = self._make_supervisor()
        t = self._make_task("failed")
        sup._task_dispatcher = MagicMock()
        sup._task_dispatcher.list_tasks.return_value = [t]
        ctx = sup._get_tasks_context()
        assert len(ctx["active"]) == 1
        assert ctx["active"][0]["status"] == "failed"
        assert "error" in ctx["active"][0]

    def test_status_notes_for_non_obvious_statuses(self):
        expected_notes = {
            "waiting_for_input": "Needs user input",
            "follow_up": "正在执行追问",
            "review": "等待人工审核",
            "learning": "正在提取经验",
            "done": "执行完成，等待关闭",
        }
        for status, expected_note in expected_notes.items():
            sup = self._make_supervisor()
            t = self._make_task(status)
            sup._task_dispatcher = MagicMock()
            sup._task_dispatcher.list_tasks.return_value = [t]
            ctx = sup._get_tasks_context()
            assert len(ctx["active"]) == 1, f"Status '{status}' not in active"
            assert ctx["active"][0].get("note") == expected_note, (
                f"Status '{status}' should have note '{expected_note}'"
            )

    def test_still_includes_running_and_pending(self):
        sup = self._make_supervisor()
        tasks = [self._make_task(s, i) for i, s in enumerate(("running", "pending"))]
        sup._task_dispatcher = MagicMock()
        sup._task_dispatcher.list_tasks.return_value = tasks
        ctx = sup._get_tasks_context()
        assert len(ctx["active"]) == 2


class TestExtractTaskIdExpanded:
    """Test _extract_task_id_from_text matches non-awaiting statuses."""

    def _make_supervisor(self):
        with patch("supervisor.main.FeishuGateway"):
            with patch("supervisor.main.ClaudeSession"):
                sup = Supervisor()
                sup.gateway = MagicMock()
                sup.claude = MagicMock()
                return sup

    def test_matches_running_task(self):
        sup = self._make_supervisor()
        t = MagicMock()
        t.id = "aabbccdd-1234-5678-9abc-123456789abc"
        t.status = "running"
        sup._task_dispatcher = MagicMock()
        sup._task_dispatcher.list_tasks.return_value = [t]
        assert sup._extract_task_id_from_text("aabbccdd 怎么样了") == t.id

    def test_matches_failed_task(self):
        sup = self._make_supervisor()
        t = MagicMock()
        t.id = "aabbccdd-1234-5678-9abc-123456789abc"
        t.status = "failed"
        sup._task_dispatcher = MagicMock()
        sup._task_dispatcher.list_tasks.return_value = [t]
        assert sup._extract_task_id_from_text("aabbccdd 失败了吗") == t.id

    def test_does_not_match_completed_task(self):
        sup = self._make_supervisor()
        t = MagicMock()
        t.id = "aabbccdd-1234-5678-9abc-123456789abc"
        t.status = "completed"
        sup._task_dispatcher = MagicMock()
        sup._task_dispatcher.list_tasks.return_value = [t]
        assert sup._extract_task_id_from_text("aabbccdd 结果呢") is None

    def test_running_task_not_auto_follow_up(self):
        """Running task ID match should pass to sonnet, not auto follow-up."""
        async def _test():
            sup = self._make_supervisor()
            t = MagicMock()
            t.id = "aabbccdd-1234-5678-9abc-123456789abc"
            t.status = "running"
            sup._task_dispatcher = MagicMock()
            sup._task_dispatcher.list_tasks.return_value = [t]
            sup._task_dispatcher.get_task.return_value = t
            sup._task_dispatcher.get_awaiting_closure.return_value = []
            sup._task_dispatcher.dispatch = AsyncMock(return_value=MagicMock())
            sup.claude.route_message = AsyncMock(
                return_value={"action": "reply", "text": "Task is running."}
            )
            await sup._route_message("aabbccdd 进度如何", "chat-1", "msg-1")
            sup.claude.route_message.assert_called_once()

        asyncio.run(_test())


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

    def test_notify_records_in_conversation_history(self):
        sup = self._make_supervisor()
        task = MagicMock()
        task.id = "12345678-abcd"
        task.status = "awaiting_closure"
        task.result = "Task completed successfully"
        task.started_at = 100.0
        task.finished_at = 110.0

        sup._notify_task_result(task, "chat-1")
        assert len(sup._conversation_history) == 1
        assert sup._conversation_history[0]["role"] == "assistant"
        assert "完成" in sup._conversation_history[0]["text"]

    def test_notify_truncates_long_result_in_history(self):
        sup = self._make_supervisor()
        task = MagicMock()
        task.id = "12345678-abcd"
        task.status = "awaiting_closure"
        task.result = "x" * 5000
        task.started_at = 100.0
        task.finished_at = 110.0

        sup._notify_task_result(task, "chat-1")
        assert len(sup._conversation_history[0]["text"]) <= 500


class TestLocalCommandHistory:
    """Test that /commands record user input and response in conversation history."""

    def _make_supervisor(self):
        with patch("supervisor.main.FeishuGateway"):
            with patch("supervisor.main.ClaudeSession"):
                sup = Supervisor()
                sup.gateway = MagicMock()
                sup.claude = MagicMock()
                return sup

    def test_command_records_history(self):
        sup = self._make_supervisor()
        sup._handle_local_command("/help", "chat-1", "msg-1")
        assert len(sup._conversation_history) == 2
        assert sup._conversation_history[0]["role"] == "user"
        assert sup._conversation_history[0]["text"] == "/help"
        assert sup._conversation_history[1]["role"] == "assistant"

    def test_command_response_truncated_in_history(self):
        sup = self._make_supervisor()
        sup._cmd_status = lambda _arg: "x" * 1000
        sup._handle_local_command("/status", "chat-1", "msg-1")
        assert len(sup._conversation_history[1]["text"]) <= 500

    def test_unknown_command_no_history(self):
        sup = self._make_supervisor()
        result = sup._handle_local_command("/nonexistent", "chat-1", "msg-1")
        assert len(sup._conversation_history) == 0


class TestSonnetCloseAction:
    """Test that action=close from sonnet routing closes the task."""

    def _make_supervisor(self):
        with patch("supervisor.main.FeishuGateway"):
            with patch("supervisor.main.ClaudeSession"):
                sup = Supervisor()
                sup.gateway = MagicMock()
                sup.claude = MagicMock()
                return sup

    def test_close_action_closes_task(self):
        """When sonnet returns action=close with task_id, task should be closed."""
        async def _test():
            sup = self._make_supervisor()
            mock_task = MagicMock()
            mock_task.id = "aabb1122-abcd-efgh-ijkl-123456789abc"
            mock_task.status = "awaiting_closure"

            sup._task_dispatcher = MagicMock()
            sup._task_dispatcher.list_tasks.return_value = [mock_task]
            sup._task_dispatcher.close_tasks.return_value = ["Task aabb1122 closed."]

            sup.claude.route_message = AsyncMock(
                return_value={"action": "close", "task_id": "aabb1122"}
            )

            await sup._route_message("关闭这个任务吧", "chat-1", "msg-1")

            sup._task_dispatcher.close_tasks.assert_called_once_with([mock_task.id])
            sup.gateway.reply_message.assert_called()
            reply = sup.gateway.reply_message.call_args[0][1]
            assert "closed" in reply.lower()

        asyncio.run(_test())

    def test_close_action_no_task_id_fallback(self):
        """When sonnet returns close without task_id, should reply with error."""
        async def _test():
            sup = self._make_supervisor()
            sup._task_dispatcher = MagicMock()
            sup._task_dispatcher.list_tasks.return_value = []
            sup._task_dispatcher.dispatch = AsyncMock(return_value=MagicMock())

            sup.claude.route_message = AsyncMock(
                return_value={"action": "close", "task_id": ""}
            )

            await sup._route_message("关了吧", "chat-1", "msg-1")

            sup.gateway.reply_message.assert_called()

        asyncio.run(_test())

    def test_close_action_task_not_found_fallback(self):
        """When close task_id doesn't match any task, reply with helpful message."""
        async def _test():
            sup = self._make_supervisor()
            sup._task_dispatcher = MagicMock()
            sup._task_dispatcher.list_tasks.return_value = []

            sup.claude.route_message = AsyncMock(
                return_value={"action": "close", "task_id": "deadbeef"}
            )

            await sup._route_message("关闭 deadbeef", "chat-1", "msg-1")

            sup.gateway.reply_message.assert_called()

        asyncio.run(_test())

    def test_close_records_history(self):
        """Close action should record user message and result in conversation history."""
        async def _test():
            sup = self._make_supervisor()
            mock_task = MagicMock()
            mock_task.id = "aabb1122-abcd-efgh-ijkl-123456789abc"
            mock_task.status = "awaiting_closure"

            sup._task_dispatcher = MagicMock()
            sup._task_dispatcher.list_tasks.return_value = [mock_task]
            sup._task_dispatcher.close_tasks.return_value = ["Task aabb1122 closed."]

            sup.claude.route_message = AsyncMock(
                return_value={"action": "close", "task_id": "aabb1122"}
            )

            await sup._route_message("不用了", "chat-1", "msg-1")

            assert any(m["role"] == "user" for m in sup._conversation_history)
            assert any(m["role"] == "assistant" for m in sup._conversation_history)

        asyncio.run(_test())


class TestSonnetBatchClose:
    """Test that action=close_all from sonnet routing closes all awaiting tasks."""

    def _make_supervisor(self):
        with patch("supervisor.main.FeishuGateway"):
            with patch("supervisor.main.ClaudeSession"):
                sup = Supervisor()
                sup.gateway = MagicMock()
                sup.claude = MagicMock()
                return sup

    def test_close_all_action_closes_all_awaiting(self):
        """When sonnet returns action=close_all, all awaiting_closure tasks should be closed."""
        async def _test():
            sup = self._make_supervisor()
            mock_t1 = MagicMock()
            mock_t1.id = "aabb1122-0000-0000-0000-000000000000"
            mock_t1.status = "awaiting_closure"
            mock_t2 = MagicMock()
            mock_t2.id = "ccdd3344-0000-0000-0000-000000000000"
            mock_t2.status = "awaiting_closure"

            sup._task_dispatcher = MagicMock()
            sup._task_dispatcher.get_awaiting_closure.return_value = [mock_t1, mock_t2]
            sup._task_dispatcher.close_tasks.return_value = [
                "Task aabb1122 closed.",
                "Task ccdd3344 closed.",
            ]

            sup.claude.route_message = AsyncMock(
                return_value={"action": "close_all"}
            )

            await sup._route_message("把这些任务都关了", "chat-1", "msg-1")

            sup._task_dispatcher.close_tasks.assert_called_once_with(
                [mock_t1.id, mock_t2.id]
            )
            sup.gateway.reply_message.assert_called()
            reply = sup.gateway.reply_message.call_args[0][1]
            assert "aabb1122" in reply
            assert "ccdd3344" in reply

        asyncio.run(_test())

    def test_close_all_no_awaiting_tasks(self):
        """When sonnet returns close_all but no tasks await closure, reply with message."""
        async def _test():
            sup = self._make_supervisor()
            sup._task_dispatcher = MagicMock()
            sup._task_dispatcher.get_awaiting_closure.return_value = []

            sup.claude.route_message = AsyncMock(
                return_value={"action": "close_all"}
            )

            await sup._route_message("全部关了吧", "chat-1", "msg-1")

            sup.gateway.reply_message.assert_called()
            reply = sup.gateway.reply_message.call_args[0][1]
            assert "没有" in reply or "No" in reply

        asyncio.run(_test())

    def test_close_all_records_history(self):
        """close_all action should record conversation history."""
        async def _test():
            sup = self._make_supervisor()
            mock_t1 = MagicMock()
            mock_t1.id = "aabb1122-0000-0000-0000-000000000000"
            mock_t1.status = "awaiting_closure"

            sup._task_dispatcher = MagicMock()
            sup._task_dispatcher.get_awaiting_closure.return_value = [mock_t1]
            sup._task_dispatcher.close_tasks.return_value = ["Task aabb1122 closed."]

            sup.claude.route_message = AsyncMock(
                return_value={"action": "close_all"}
            )

            await sup._route_message("全部关闭", "chat-1", "msg-1")

            assert any(m["role"] == "user" for m in sup._conversation_history)
            assert any(m["role"] == "assistant" for m in sup._conversation_history)

        asyncio.run(_test())

    def test_close_with_task_ids_array(self):
        """When sonnet returns close with task_ids array, close specified tasks."""
        async def _test():
            sup = self._make_supervisor()
            mock_t1 = MagicMock()
            mock_t1.id = "aabb1122-0000-0000-0000-000000000000"
            mock_t1.status = "awaiting_closure"
            mock_t2 = MagicMock()
            mock_t2.id = "ccdd3344-0000-0000-0000-000000000000"
            mock_t2.status = "awaiting_closure"
            mock_t3 = MagicMock()
            mock_t3.id = "eeff5566-0000-0000-0000-000000000000"
            mock_t3.status = "awaiting_closure"

            sup._task_dispatcher = MagicMock()
            sup._task_dispatcher.list_tasks.return_value = [mock_t1, mock_t2, mock_t3]
            sup._task_dispatcher.close_tasks.return_value = [
                "Task aabb1122 closed.",
                "Task ccdd3344 closed.",
            ]

            sup.claude.route_message = AsyncMock(
                return_value={"action": "close", "task_ids": ["aabb1122", "ccdd3344"]}
            )

            await sup._route_message("把前两个任务关了", "chat-1", "msg-1")

            sup._task_dispatcher.close_tasks.assert_called_once_with(
                [mock_t1.id, mock_t2.id]
            )

        asyncio.run(_test())


class TestSmartCloseFallback:
    """Test that follow_up with close intent in original text auto-closes instead."""

    def _make_supervisor(self):
        with patch("supervisor.main.FeishuGateway"):
            with patch("supervisor.main.ClaudeSession"):
                sup = Supervisor()
                sup.gateway = MagicMock()
                sup.claude = MagicMock()
                return sup

    def test_follow_up_with_close_intent_auto_closes(self):
        async def _test():
            sup = self._make_supervisor()
            mock_task = MagicMock()
            mock_task.id = "ccdd3344-abcd-efgh-ijkl-123456789abc"
            mock_task.status = "awaiting_closure"

            sup._task_dispatcher = MagicMock()
            sup._task_dispatcher.list_tasks.return_value = [mock_task]
            sup._task_dispatcher.close_tasks.return_value = ["Task ccdd3344 closed."]
            sup._task_dispatcher.follow_up_async = AsyncMock()

            sup.claude.route_message = AsyncMock(
                return_value={
                    "action": "follow_up",
                    "task_id": "ccdd3344",
                    "text": "用户想关闭这个任务",
                }
            )

            await sup._route_message("检查系统日志那个关闭了", "chat-1", "msg-1")

            sup._task_dispatcher.close_tasks.assert_called_once_with([mock_task.id])
            sup._task_dispatcher.follow_up_async.assert_not_called()

        asyncio.run(_test())

    def test_follow_up_without_close_intent_proceeds(self):
        async def _test():
            sup = self._make_supervisor()
            mock_task = MagicMock()
            mock_task.id = "ccdd3344-abcd-efgh-ijkl-123456789abc"
            mock_task.status = "awaiting_closure"
            mock_task.session_id = "sess-1"

            sup._task_dispatcher = MagicMock()
            sup._task_dispatcher.list_tasks.return_value = [mock_task]
            sup._task_dispatcher.follow_up_async = AsyncMock(return_value="result")

            sup.gateway.send_message.return_value = "msg-200"

            sup.claude.route_message = AsyncMock(
                return_value={
                    "action": "follow_up",
                    "task_id": "ccdd3344",
                    "text": "这个结果看起来不对",
                }
            )

            await sup._route_message("这个结果看起来不对", "chat-1", "msg-1")

            sup._task_dispatcher.follow_up_async.assert_called_once()
            sup._task_dispatcher.close_task.assert_not_called()

        asyncio.run(_test())

    def test_technical_close_not_intercepted(self):
        """'关闭连接' should NOT trigger smart close."""
        async def _test():
            sup = self._make_supervisor()
            mock_task = MagicMock()
            mock_task.id = "ccdd3344-abcd-efgh-ijkl-123456789abc"
            mock_task.status = "awaiting_closure"
            mock_task.session_id = "sess-1"

            sup._task_dispatcher = MagicMock()
            sup._task_dispatcher.list_tasks.return_value = [mock_task]
            sup._task_dispatcher.follow_up_async = AsyncMock(return_value="result")

            sup.gateway.send_message.return_value = "msg-200"

            sup.claude.route_message = AsyncMock(
                return_value={
                    "action": "follow_up",
                    "task_id": "ccdd3344",
                    "text": "帮我关闭连接后重试",
                }
            )

            await sup._route_message("帮我关闭连接后重试", "chat-1", "msg-1")

            sup._task_dispatcher.close_task.assert_not_called()
            sup._task_dispatcher.follow_up_async.assert_called_once()

        asyncio.run(_test())


class TestReplyBasedClose:
    """Test that short acknowledgement replies auto-close tasks."""

    def _make_supervisor(self):
        with patch("supervisor.main.FeishuGateway"):
            with patch("supervisor.main.ClaudeSession"):
                sup = Supervisor()
                sup.gateway = MagicMock()
                sup.claude = MagicMock()
                return sup

    def test_reply_ok_closes_task(self):
        """Replying '好的' to a task result should close, not follow-up."""
        async def _test():
            sup = self._make_supervisor()
            mock_task = MagicMock()
            mock_task.id = "aaaabbbb-1234-5678-9abc-123456789abc"
            mock_task.status = "awaiting_closure"

            sup._task_dispatcher = MagicMock()
            sup._task_dispatcher.get_task.return_value = mock_task
            sup._task_dispatcher.close_task.return_value = "Task aaaabbbb closed."
            sup._task_dispatcher.follow_up_async = AsyncMock()

            sup._message_task_map["feishu-msg-100"] = mock_task.id

            await sup._route_message(
                "好的", "chat-1", "msg-reply-1",
                parent_id="feishu-msg-100",
            )

            sup._task_dispatcher.close_task.assert_called_once_with(mock_task.id)
            sup._task_dispatcher.follow_up_async.assert_not_called()

        asyncio.run(_test())

    def test_reply_thanks_closes_task(self):
        """Replying '谢谢' should close."""
        async def _test():
            sup = self._make_supervisor()
            mock_task = MagicMock()
            mock_task.id = "aaaabbbb-1234-5678-9abc-123456789abc"
            mock_task.status = "awaiting_closure"

            sup._task_dispatcher = MagicMock()
            sup._task_dispatcher.get_task.return_value = mock_task
            sup._task_dispatcher.close_task.return_value = "closed"
            sup._task_dispatcher.follow_up_async = AsyncMock()

            sup._message_task_map["feishu-msg-100"] = mock_task.id

            await sup._route_message(
                "谢谢", "chat-1", "msg-reply-1",
                parent_id="feishu-msg-100",
            )

            sup._task_dispatcher.close_task.assert_called_once()
            sup._task_dispatcher.follow_up_async.assert_not_called()

        asyncio.run(_test())

    def test_reply_long_message_triggers_follow_up(self):
        """A long reply should still trigger follow-up, not close."""
        async def _test():
            sup = self._make_supervisor()
            mock_task = MagicMock()
            mock_task.id = "aaaabbbb-1234-5678-9abc-123456789abc"
            mock_task.status = "awaiting_closure"
            mock_task.session_id = "sess-1"

            sup._task_dispatcher = MagicMock()
            sup._task_dispatcher.get_task.return_value = mock_task
            sup._task_dispatcher.follow_up_async = AsyncMock(return_value="result")

            sup.gateway.send_message.return_value = "msg-200"

            sup._message_task_map["feishu-msg-100"] = mock_task.id

            await sup._route_message(
                "好的，但是这个接口还需要加个鉴权功能", "chat-1", "msg-reply-1",
                parent_id="feishu-msg-100",
            )

            sup._task_dispatcher.follow_up_async.assert_called_once()

        asyncio.run(_test())


class TestReplyBasedFollowUp:
    """Test that Feishu reply messages auto-route to follow_up."""

    def _make_supervisor(self):
        with patch("supervisor.main.FeishuGateway"):
            with patch("supervisor.main.ClaudeSession"):
                sup = Supervisor()
                sup.gateway = MagicMock()
                sup.claude = MagicMock()
                return sup

    def test_message_task_map_initialized(self):
        """Supervisor should have a _message_task_map dict."""
        sup = self._make_supervisor()
        assert hasattr(sup, "_message_task_map")
        assert isinstance(sup._message_task_map, dict)

    def test_notify_records_message_task_mapping(self):
        """When task result is pushed, the sent message_id should map to task_id."""
        sup = self._make_supervisor()
        sup.gateway.push_message.return_value = "feishu-msg-100"

        task = MagicMock()
        task.id = "aaaabbbb-1234-5678-9abc-123456789abc"
        task.status = "awaiting_closure"
        task.result = "Task done"
        task.started_at = 100.0
        task.finished_at = 110.0

        sup._notify_task_result(task, "chat-1")
        assert "feishu-msg-100" in sup._message_task_map
        assert sup._message_task_map["feishu-msg-100"] == task.id

    def test_notify_no_mapping_when_push_fails(self):
        """If push_message returns None, no mapping should be created."""
        sup = self._make_supervisor()
        sup.gateway.push_message.return_value = None

        task = MagicMock()
        task.id = "aaaabbbb-1234-5678-9abc-123456789abc"
        task.status = "awaiting_closure"
        task.result = "Task done"
        task.started_at = 100.0
        task.finished_at = 110.0

        sup._notify_task_result(task, "chat-1")
        assert len(sup._message_task_map) == 0

    def test_reply_to_task_message_routes_follow_up(self):
        """Reply to a task result message should auto-route as follow_up."""
        async def _test():
            sup = self._make_supervisor()
            mock_task = MagicMock()
            mock_task.id = "aaaabbbb-1234-5678-9abc-123456789abc"
            mock_task.status = "awaiting_closure"
            mock_task.session_id = "sess-1"

            sup._task_dispatcher = MagicMock()
            sup._task_dispatcher.list_tasks.return_value = [mock_task]
            sup._task_dispatcher.get_task.return_value = mock_task
            sup._task_dispatcher.follow_up_async = AsyncMock(return_value="Follow-up result")

            # Simulate: task result was sent as feishu-msg-100
            sup._message_task_map["feishu-msg-100"] = mock_task.id

            # User replies to that message (parent_id = feishu-msg-100)
            await sup._route_message(
                "这个结果对吗", "chat-1", "msg-reply-1",
                parent_id="feishu-msg-100",
            )

            # Should auto-route to follow_up, NOT go through sonnet
            sup._task_dispatcher.follow_up_async.assert_called_once()
            sup.claude.route_message.assert_not_called()

        asyncio.run(_test())

    def test_reply_to_non_task_message_goes_to_sonnet(self):
        """Reply to a non-task message should go through normal sonnet routing."""
        async def _test():
            sup = self._make_supervisor()
            sup._task_dispatcher = MagicMock()
            sup._task_dispatcher.list_tasks.return_value = []
            sup._task_dispatcher.dispatch = AsyncMock(return_value=MagicMock())
            sup.claude.route_message = AsyncMock(
                return_value={"action": "reply", "text": "你好"}
            )

            # Reply to a message that is NOT in _message_task_map
            await sup._route_message(
                "你好", "chat-1", "msg-1",
                parent_id="feishu-msg-unknown",
            )

            # Should fall through to sonnet
            sup.claude.route_message.assert_called_once()

        asyncio.run(_test())

    def test_reply_to_closed_task_goes_to_sonnet(self):
        """Reply to a task that's already closed should go through sonnet."""
        async def _test():
            sup = self._make_supervisor()
            mock_task = MagicMock()
            mock_task.id = "aaaabbbb-1234-5678-9abc-123456789abc"
            mock_task.status = "completed"  # already closed

            sup._task_dispatcher = MagicMock()
            sup._task_dispatcher.list_tasks.return_value = [mock_task]
            sup._task_dispatcher.get_task.return_value = mock_task
            sup._task_dispatcher.dispatch = AsyncMock(return_value=MagicMock())
            sup.claude.route_message = AsyncMock(
                return_value={"action": "dispatch", "description": "new task"}
            )

            sup._message_task_map["feishu-msg-100"] = mock_task.id

            await sup._route_message(
                "再帮我看看", "chat-1", "msg-1",
                parent_id="feishu-msg-100",
            )

            # Task is closed, should fall through to sonnet
            sup.claude.route_message.assert_called_once()

        asyncio.run(_test())

    def test_no_parent_id_normal_routing(self):
        """Messages without parent_id should use normal routing."""
        async def _test():
            sup = self._make_supervisor()
            sup._task_dispatcher = MagicMock()
            sup._task_dispatcher.list_tasks.return_value = []
            sup._task_dispatcher.dispatch = AsyncMock(return_value=MagicMock())
            sup.claude.route_message = AsyncMock(
                return_value={"action": "reply", "text": "你好"}
            )

            await sup._route_message("你好", "chat-1", "msg-1")

            sup.claude.route_message.assert_called_once()

        asyncio.run(_test())

    def test_on_feishu_message_passes_parent_id(self):
        """_on_feishu_message should extract parent_id from raw_event and pass to routing."""
        sup = self._make_supervisor()
        sup._loop = MagicMock()

        # Mock raw_event with parent_id
        raw_event = MagicMock()
        raw_event.event.message.parent_id = "msg-parent-123"
        raw_event.event.message.root_id = "msg-root-456"

        with patch.object(asyncio, "run_coroutine_threadsafe") as mock_run:
            sup._on_feishu_message(
                "u1", "msg-1", "chat-1", "text", "追问内容",
                raw_event=raw_event,
                parent_id="msg-parent-123",
                root_id="msg-root-456",
            )
            mock_run.assert_called_once()
            # The coroutine should have parent_id passed
            coro = mock_run.call_args[0][0]
            assert coro is not None

    def test_follow_up_reply_tracked_for_chain_replies(self):
        """Follow-up reply message should also be tracked for chain replies."""
        async def _test():
            sup = self._make_supervisor()
            mock_task = MagicMock()
            mock_task.id = "aaaabbbb-1234-5678-9abc-123456789abc"
            mock_task.status = "awaiting_closure"
            mock_task.session_id = "sess-1"

            sup._task_dispatcher = MagicMock()
            sup._task_dispatcher.list_tasks.return_value = [mock_task]
            sup._task_dispatcher.get_task.return_value = mock_task
            sup._task_dispatcher.follow_up_async = AsyncMock(return_value="First follow-up result")

            # Gateway returns message IDs for sent messages
            sup.gateway.reply_message.return_value = "msg-ack"
            sup.gateway.send_message.return_value = "feishu-followup-reply-msg"

            # Simulate initial reply-based follow-up
            sup._message_task_map["feishu-msg-100"] = mock_task.id
            await sup._route_message(
                "这个结果对吗", "chat-1", "msg-reply-1",
                parent_id="feishu-msg-100",
            )

            # The follow-up reply message should now be tracked
            assert "feishu-followup-reply-msg" in sup._message_task_map
            assert sup._message_task_map["feishu-followup-reply-msg"] == mock_task.id

        asyncio.run(_test())

    def test_message_task_map_bounded_via_notify(self):
        """_message_task_map should be capped when new entries are added via notify."""
        sup = self._make_supervisor()
        # Pre-fill with 505 entries to exceed cap
        for i in range(505):
            sup._message_task_map[f"msg-{i}"] = f"task-{i}"

        # Trigger notify which adds a new entry and caps the map
        sup.gateway.push_message.return_value = "msg-new"
        task = MagicMock()
        task.id = "aaaabbbb-1234-5678-9abc-123456789abc"
        task.status = "awaiting_closure"
        task.result = "done"
        task.started_at = 100.0
        task.finished_at = 110.0

        sup._notify_task_result(task, "chat-1")
        assert len(sup._message_task_map) <= 501  # 500 kept + 1 new
