"""Tests for orchestrator pattern — single task with subagent coordination.

Replaces the old dispatch_multi (N independent tasks) with a single
orchestrator task that uses Claude's Agent tool for subagent coordination.
"""

import asyncio
import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from supervisor.router_skill import (
    build_route_prompt,
    build_route_user_prompt,
    build_route_system_prompt,
    ROUTING_RULES,
    ROUTING_EXAMPLES,
)
from supervisor.main import Supervisor


# ══════════════════════════════════════════════════════════
# Router Skill: orchestrate action in rules/examples/format
# ══════════════════════════════════════════════════════════


class TestOrchestrateRoutingRules:
    """Routing rules must define the 'orchestrate' action."""

    def test_rules_mention_orchestrate(self):
        assert "orchestrate" in ROUTING_RULES

    def test_rules_explain_when_to_orchestrate(self):
        """Rules should explain orchestrate is for complex multi-faceted tasks."""
        assert "orchestrate" in ROUTING_RULES
        # Should mention subagent or Agent tool coordination
        lower = ROUTING_RULES.lower()
        assert "subagent" in lower or "agent" in lower or "coordinate" in lower

    def test_dispatch_multi_removed_from_rules(self):
        """dispatch_multi should no longer appear in routing rules."""
        assert "dispatch_multi" not in ROUTING_RULES


class TestOrchestrateRoutingExamples:
    """Few-shot examples must include orchestrate cases."""

    def test_examples_have_orchestrate_cases(self):
        assert '"orchestrate"' in ROUTING_EXAMPLES

    def test_examples_show_subtasks(self):
        """Orchestrate examples should show subtasks list."""
        assert "subtasks" in ROUTING_EXAMPLES

    def test_dispatch_multi_removed_from_examples(self):
        """dispatch_multi should no longer appear in examples."""
        assert "dispatch_multi" not in ROUTING_EXAMPLES


class TestOrchestratePromptFormat:
    """The prompt format section must document orchestrate JSON shape."""

    def test_orchestrate_format_in_system_prompt(self):
        system = build_route_system_prompt()
        assert '"orchestrate"' in system

    def test_orchestrate_format_has_subtasks(self):
        system = build_route_system_prompt()
        assert "subtasks" in system


# ══════════════════════════════════════════════════════════
# Main.py: _handle_orchestrate creates a SINGLE task
# ══════════════════════════════════════════════════════════


class TestHandleOrchestrate:
    """Orchestrate action creates ONE task (not N), with subagent instructions."""

    def _make_supervisor(self):
        with patch("supervisor.main.FeishuGateway"):
            with patch("supervisor.main.ClaudeSession"):
                sup = Supervisor()
                sup.gateway = MagicMock()
                sup.claude = MagicMock()
                sup._task_dispatcher = MagicMock()
                sup._task_dispatcher.get_awaiting_closure.return_value = []
                sup._task_dispatcher.list_tasks.return_value = []
                mock_task = MagicMock()
                mock_task.id = "aaaa1111-bbbb-cccc-dddd-eeeeffffgggg"
                sup._task_dispatcher.dispatch = AsyncMock(return_value=mock_task)
                return sup

    def test_orchestrate_creates_single_task(self):
        """Orchestrate should dispatch exactly ONE task, not N."""
        async def _test():
            sup = self._make_supervisor()
            sup.claude.route_message = AsyncMock(
                return_value={
                    "action": "orchestrate",
                    "description": "重构项目",
                    "subtasks": ["分析代码结构", "检查测试覆盖", "审计依赖"],
                }
            )
            await sup._route_message("重构整个项目", "chat-1", "msg-1")
            # Only ONE dispatch call, not 3
            assert sup._task_dispatcher.dispatch.call_count == 1

        asyncio.run(_test())

    def test_orchestrate_prompt_contains_subtasks(self):
        """The single task prompt must contain all subtask descriptions."""
        async def _test():
            sup = self._make_supervisor()
            sup.claude.route_message = AsyncMock(
                return_value={
                    "action": "orchestrate",
                    "description": "重构项目",
                    "subtasks": ["分析代码", "修复bug", "写测试"],
                }
            )
            await sup._route_message("重构", "chat-1", "msg-1")
            call_kwargs = sup._task_dispatcher.dispatch.call_args
            prompt = call_kwargs.kwargs.get("prompt") or call_kwargs[1].get("prompt", "")
            assert "分析代码" in prompt
            assert "修复bug" in prompt
            assert "写测试" in prompt

        asyncio.run(_test())

    def test_orchestrate_prompt_instructs_subagent_usage(self):
        """The orchestrator prompt must instruct the worker to use Agent tool."""
        async def _test():
            sup = self._make_supervisor()
            sup.claude.route_message = AsyncMock(
                return_value={
                    "action": "orchestrate",
                    "description": "复杂任务",
                    "subtasks": ["子任务A", "子任务B"],
                }
            )
            await sup._route_message("做一个复杂的事", "chat-1", "msg-1")
            call_kwargs = sup._task_dispatcher.dispatch.call_args
            prompt = call_kwargs.kwargs.get("prompt") or call_kwargs[1].get("prompt", "")
            lower = prompt.lower()
            assert "agent" in lower or "subagent" in lower

        asyncio.run(_test())

    def test_orchestrate_notifies_user_with_subtask_list(self):
        """User should see the list of subtasks in the notification."""
        async def _test():
            sup = self._make_supervisor()
            sup.claude.route_message = AsyncMock(
                return_value={
                    "action": "orchestrate",
                    "description": "重构项目",
                    "subtasks": ["分析代码", "修复bug"],
                }
            )
            await sup._route_message("重构", "chat-1", "msg-1")
            reply_text = sup.gateway.reply_message.call_args[0][1]
            assert "分析代码" in reply_text
            assert "修复bug" in reply_text

        asyncio.run(_test())

    def test_orchestrate_falls_back_to_dispatch_without_subtasks(self):
        """If orchestrate has no subtasks, fall back to single dispatch."""
        async def _test():
            sup = self._make_supervisor()
            sup.claude.route_message = AsyncMock(
                return_value={
                    "action": "orchestrate",
                    "description": "做个事",
                    "subtasks": [],
                }
            )
            await sup._route_message("做个事", "chat-1", "msg-1")
            # Should still dispatch (fallback)
            assert sup._task_dispatcher.dispatch.call_count == 1

        asyncio.run(_test())

    def test_dispatch_multi_backward_compat_routes_to_orchestrate(self):
        """If sonnet still returns dispatch_multi, treat it as orchestrate."""
        async def _test():
            sup = self._make_supervisor()
            sup.claude.route_message = AsyncMock(
                return_value={
                    "action": "dispatch_multi",
                    "description": "重构",
                    "subtasks": ["A", "B", "C"],
                }
            )
            await sup._route_message("重构", "chat-1", "msg-1")
            # Should create only ONE task (orchestrator mode)
            assert sup._task_dispatcher.dispatch.call_count == 1

        asyncio.run(_test())


# ══════════════════════════════════════════════════════════
# Orchestrator prompt builder
# ══════════════════════════════════════════════════════════


class TestBuildOrchestratorPrompt:
    """Test the orchestrator prompt builder function."""

    def _make_supervisor(self):
        with patch("supervisor.main.FeishuGateway"):
            with patch("supervisor.main.ClaudeSession"):
                sup = Supervisor()
                sup.gateway = MagicMock()
                sup.claude = MagicMock()
                return sup

    def test_prompt_contains_orchestrator_role(self):
        sup = self._make_supervisor()
        prompt = sup._build_orchestrator_prompt(
            "重构项目", "重构", ["分析代码", "写测试"]
        )
        assert "orchestrator" in prompt.lower() or "协调" in prompt

    def test_prompt_lists_subtasks(self):
        sup = self._make_supervisor()
        prompt = sup._build_orchestrator_prompt(
            "重构项目", "重构", ["分析代码", "写测试", "修复bug"]
        )
        assert "分析代码" in prompt
        assert "写测试" in prompt
        assert "修复bug" in prompt

    def test_prompt_mentions_agent_tool(self):
        sup = self._make_supervisor()
        prompt = sup._build_orchestrator_prompt(
            "重构项目", "重构", ["分析代码", "写测试"]
        )
        assert "Agent" in prompt

    def test_prompt_includes_conversation_history(self):
        sup = self._make_supervisor()
        sup._conversation_history.append({"role": "user", "text": "之前的对话"})
        prompt = sup._build_orchestrator_prompt(
            "重构项目", "重构", ["分析代码"]
        )
        assert "之前的对话" in prompt
