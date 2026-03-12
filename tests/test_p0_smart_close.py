"""Tests for P0-3: Fix smart close silent failure.

When Sonnet returns action="reply" with text containing close intent,
the system should detect this and auto-close the task if unambiguous.
"""

import asyncio
import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from supervisor.main import Supervisor


def _make_supervisor():
    with patch("supervisor.main.FeishuGateway"):
        with patch("supervisor.main.ClaudeSession"):
            sup = Supervisor()
            sup.gateway = MagicMock()
            sup.gateway.reply_message = MagicMock()
            sup.gateway.send_message = MagicMock()
            sup.gateway.push_message = MagicMock()
            sup.claude = MagicMock()
            sup._loop = asyncio.new_event_loop()
            return sup


def _make_task(task_id, status="awaiting_closure", description="test task"):
    task = MagicMock()
    task.id = task_id
    task.status = status
    task.description = description
    task.prompt = description
    task.result = "some result"
    task.session_id = "sess-1234"
    task.started_at = 1000.0
    task.finished_at = 1001.0
    task.steps_completed = []
    task.current_step = ""
    task.error = ""
    return task


class TestReplyWithCloseIntent:
    """When Sonnet says reply but text has close intent, auto-close if unambiguous."""

    def test_reply_close_intent_single_task_auto_closes(self):
        def _test():
            sup = _make_supervisor()
            task = _make_task("aabb1122-0000-0000-0000-000000000000")

            sup._task_dispatcher = MagicMock()
            sup._task_dispatcher.get_awaiting_closure.return_value = [task]
            sup._task_dispatcher.close_task.return_value = "Task aabb1122 closed."
            sup._task_dispatcher.list_tasks.return_value = [task]

            sup.claude.route_message = AsyncMock(return_value={
                "action": "reply",
                "text": "好的，任务已完成，已关闭了。",
            })

            asyncio.get_event_loop().run_until_complete(
                sup._route_message("关闭吧", "chat-1", "msg-1")
            )

            sup._task_dispatcher.close_task.assert_called_once_with(task.id)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            _test()
        finally:
            loop.close()

    def test_reply_close_intent_no_tasks_no_action(self):
        def _test():
            sup = _make_supervisor()

            sup._task_dispatcher = MagicMock()
            sup._task_dispatcher.get_awaiting_closure.return_value = []
            sup._task_dispatcher.list_tasks.return_value = []

            sup.claude.route_message = AsyncMock(return_value={
                "action": "reply",
                "text": "好的，没问题。",
            })

            asyncio.get_event_loop().run_until_complete(
                sup._route_message("好的", "chat-1", "msg-1")
            )

            sup._task_dispatcher.close_task.assert_not_called()
            sup.gateway.reply_message.assert_called()

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            _test()
        finally:
            loop.close()

    def test_reply_close_intent_multiple_tasks_no_auto_close(self):
        def _test():
            sup = _make_supervisor()
            task1 = _make_task("aabb1122-0000-0000-0000-000000000000")
            task2 = _make_task("ccdd3344-0000-0000-0000-000000000000")

            sup._task_dispatcher = MagicMock()
            sup._task_dispatcher.get_awaiting_closure.return_value = [task1, task2]
            sup._task_dispatcher.list_tasks.return_value = [task1, task2]

            sup.claude.route_message = AsyncMock(return_value={
                "action": "reply",
                "text": "好的，已关闭了。",
            })

            asyncio.get_event_loop().run_until_complete(
                sup._route_message("关掉", "chat-1", "msg-1")
            )

            sup._task_dispatcher.close_task.assert_not_called()

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            _test()
        finally:
            loop.close()

    def test_reply_without_close_intent_no_auto_close(self):
        def _test():
            sup = _make_supervisor()
            task = _make_task("aabb1122-0000-0000-0000-000000000000")

            sup._task_dispatcher = MagicMock()
            sup._task_dispatcher.get_awaiting_closure.return_value = [task]
            sup._task_dispatcher.list_tasks.return_value = [task]

            sup.claude.route_message = AsyncMock(return_value={
                "action": "reply",
                "text": "Python 是一种编程语言，广泛用于数据科学和Web开发。",
            })

            asyncio.get_event_loop().run_until_complete(
                sup._route_message("什么是Python", "chat-1", "msg-1")
            )

            sup._task_dispatcher.close_task.assert_not_called()

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            _test()
        finally:
            loop.close()


class TestDirectCloseStillWorks:
    def test_sonnet_close_action_works(self):
        def _test():
            sup = _make_supervisor()
            task = _make_task("aabb1122-0000-0000-0000-000000000000")

            sup._task_dispatcher = MagicMock()
            sup._task_dispatcher.list_tasks.return_value = [task]
            sup._task_dispatcher.close_tasks.return_value = ["Task aabb1122 closed."]

            sup.claude.route_message = AsyncMock(return_value={
                "action": "close",
                "task_id": "aabb1122",
            })

            asyncio.get_event_loop().run_until_complete(
                sup._route_message("关闭这个任务", "chat-1", "msg-1")
            )

            sup._task_dispatcher.close_tasks.assert_called_once()

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            _test()
        finally:
            loop.close()


class TestCloseIntentDetection:
    def test_close_intent_chinese_phrases(self):
        from supervisor.task_dispatcher import _contains_close_intent
        assert _contains_close_intent("好的，已帮你关闭了")
        assert _contains_close_intent("任务已结束了")
        assert _contains_close_intent("已关掉")

    def test_no_close_intent_normal_text(self):
        from supervisor.task_dispatcher import _contains_close_intent
        assert not _contains_close_intent("Python 是一种编程语言")
        assert not _contains_close_intent("让我来帮你分析一下")
