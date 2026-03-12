"""Tests for supervisor.conversation_monitor module.

TDD: These tests are written BEFORE the implementation.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

from supervisor.conversation_monitor import (
    analyze_conversation,
    build_analysis_prompt,
    format_issue_notification,
    format_fix_plan,
    parse_analysis_response,
    looks_like_confirm,
    looks_like_reject,
)


# ── Pure function tests ──


class TestBuildAnalysisPrompt:
    def test_returns_system_and_user_prompt(self):
        messages = [{"role": "user", "text": "部署出错了"}]
        system, user = build_analysis_prompt(
            messages=messages,
            active_tasks=[],
            active_sessions="",
        )
        assert isinstance(system, str)
        assert isinstance(user, str)
        assert len(system) > 0
        assert len(user) > 0

    def test_includes_messages_in_user_prompt(self):
        messages = [
            {"role": "user", "text": "为什么任务卡住了"},
            {"role": "assistant", "text": "我来看看"},
        ]
        _, user = build_analysis_prompt(
            messages=messages,
            active_tasks=[],
            active_sessions="",
        )
        assert "为什么任务卡住了" in user
        assert "我来看看" in user

    def test_includes_task_context(self):
        tasks = [{"id": "abc12345", "status": "running", "description": "修复bug"}]
        _, user = build_analysis_prompt(
            messages=[{"role": "user", "text": "hello"}],
            active_tasks=tasks,
            active_sessions="",
        )
        assert "abc12345" in user or "修复bug" in user

    def test_includes_session_context(self):
        _, user = build_analysis_prompt(
            messages=[{"role": "user", "text": "hello"}],
            active_tasks=[],
            active_sessions="2 active sessions",
        )
        assert "2 active sessions" in user

    def test_system_prompt_has_detection_dimensions(self):
        system, _ = build_analysis_prompt(
            messages=[{"role": "user", "text": "test"}],
            active_tasks=[],
            active_sessions="",
        )
        # Should mention error detection, stuck tasks, etc.
        assert "JSON" in system


class TestParseAnalysisResponse:
    def test_parses_valid_json(self):
        response = json.dumps({
            "has_issues": True,
            "issues": [{"severity": "HIGH", "description": "Task stuck", "suggested_fix": "Restart"}],
            "summary": "1 issue found",
        })
        result = parse_analysis_response(response)
        assert result["has_issues"] is True
        assert len(result["issues"]) == 1
        assert result["issues"][0]["severity"] == "HIGH"

    def test_parses_no_issues(self):
        response = json.dumps({
            "has_issues": False,
            "issues": [],
            "summary": "一切正常",
        })
        result = parse_analysis_response(response)
        assert result["has_issues"] is False
        assert result["issues"] == []

    def test_handles_markdown_wrapped_json(self):
        response = "```json\n" + json.dumps({
            "has_issues": False,
            "issues": [],
            "summary": "OK",
        }) + "\n```"
        result = parse_analysis_response(response)
        assert result["has_issues"] is False

    def test_handles_invalid_json(self):
        result = parse_analysis_response("not valid json at all")
        assert result["has_issues"] is False
        assert result["issues"] == []

    def test_handles_empty_string(self):
        result = parse_analysis_response("")
        assert result["has_issues"] is False

    def test_handles_missing_fields(self):
        response = json.dumps({"has_issues": True})
        result = parse_analysis_response(response)
        assert result["has_issues"] is True
        assert result["issues"] == []
        assert "summary" in result


class TestFormatIssueNotification:
    def test_formats_single_issue(self):
        issues = [{"severity": "HIGH", "description": "任务卡住30分钟", "suggested_fix": "重启任务"}]
        text = format_issue_notification(issues)
        assert "1" in text
        assert "任务卡住30分钟" in text
        assert "重启任务" in text
        assert "修复" in text  # Should include action hint

    def test_formats_multiple_issues(self):
        issues = [
            {"severity": "HIGH", "description": "Issue 1", "suggested_fix": "Fix 1"},
            {"severity": "MEDIUM", "description": "Issue 2", "suggested_fix": "Fix 2"},
        ]
        text = format_issue_notification(issues)
        assert "2" in text
        assert "Issue 1" in text
        assert "Issue 2" in text

    def test_empty_issues(self):
        text = format_issue_notification([])
        assert text == ""


class TestFormatFixPlan:
    def test_formats_plan(self):
        issues = [
            {"severity": "HIGH", "description": "任务卡住", "suggested_fix": "重启任务"},
        ]
        plan = format_fix_plan(issues)
        assert "重启任务" in plan
        assert len(plan) > 0

    def test_multiple_steps(self):
        issues = [
            {"severity": "HIGH", "description": "A", "suggested_fix": "Fix A"},
            {"severity": "MEDIUM", "description": "B", "suggested_fix": "Fix B"},
        ]
        plan = format_fix_plan(issues)
        assert "Fix A" in plan
        assert "Fix B" in plan


class TestLooksLikeConfirm:
    def test_chinese_confirm_words(self):
        assert looks_like_confirm("修复") is True
        assert looks_like_confirm("好的") is True
        assert looks_like_confirm("确认") is True
        assert looks_like_confirm("开始") is True
        assert looks_like_confirm("好") is True

    def test_english_confirm_words(self):
        assert looks_like_confirm("fix") is True
        assert looks_like_confirm("yes") is True
        assert looks_like_confirm("ok") is True
        assert looks_like_confirm("confirm") is True

    def test_non_confirm(self):
        assert looks_like_confirm("帮我写个函数") is False
        assert looks_like_confirm("今天天气如何") is False

    def test_case_insensitive(self):
        assert looks_like_confirm("OK") is True
        assert looks_like_confirm("Yes") is True
        assert looks_like_confirm("FIX") is True


class TestLooksLikeReject:
    def test_chinese_reject_words(self):
        assert looks_like_reject("不用了") is True
        assert looks_like_reject("不需要") is True
        assert looks_like_reject("取消") is True
        assert looks_like_reject("算了") is True

    def test_english_reject_words(self):
        assert looks_like_reject("no") is True
        assert looks_like_reject("cancel") is True
        assert looks_like_reject("skip") is True

    def test_non_reject(self):
        assert looks_like_reject("修复") is False
        assert looks_like_reject("好的") is False


# ── Async analyze_conversation tests ──


class TestAnalyzeConversation:
    def test_empty_messages_returns_no_issues(self):
        async def _test():
            result = await analyze_conversation(
                messages=[],
                active_tasks=[],
                active_sessions="",
                api_key="test-key",
            )
            assert result["has_issues"] is False
            assert result["issues"] == []

        asyncio.run(_test())

    def test_calls_anthropic_api(self):
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps({
            "has_issues": True,
            "issues": [{"severity": "HIGH", "description": "error", "suggested_fix": "fix it"}],
            "summary": "Found issue",
        }))]
        mock_response.usage = MagicMock(input_tokens=100, output_tokens=50)

        async def _test():
            with patch("supervisor.conversation_monitor._anthropic_mod") as mock_anthropic:
                mock_client = AsyncMock()
                mock_client.messages.create = AsyncMock(return_value=mock_response)
                mock_anthropic.AsyncAnthropic.return_value = mock_client

                result = await analyze_conversation(
                    messages=[{"role": "user", "text": "出错了"}],
                    active_tasks=[],
                    active_sessions="",
                    api_key="test-key",
                )
                assert result["has_issues"] is True
                assert len(result["issues"]) == 1

                # Verify API was called with correct key
                mock_anthropic.AsyncAnthropic.assert_called_once_with(api_key="test-key")

        asyncio.run(_test())

    def test_api_failure_returns_no_issues(self):
        async def _test():
            with patch("supervisor.conversation_monitor._anthropic_mod") as mock_anthropic:
                mock_client = AsyncMock()
                mock_client.messages.create = AsyncMock(side_effect=Exception("API down"))
                mock_anthropic.AsyncAnthropic.return_value = mock_client

                result = await analyze_conversation(
                    messages=[{"role": "user", "text": "test"}],
                    active_tasks=[],
                    active_sessions="",
                    api_key="test-key",
                )
                assert result["has_issues"] is False

        asyncio.run(_test())

    def test_uses_sonnet_model(self):
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"has_issues": false, "issues": [], "summary": "ok"}')]
        mock_response.usage = MagicMock(input_tokens=50, output_tokens=20)

        async def _test():
            with patch("supervisor.conversation_monitor._anthropic_mod") as mock_anthropic:
                mock_client = AsyncMock()
                mock_client.messages.create = AsyncMock(return_value=mock_response)
                mock_anthropic.AsyncAnthropic.return_value = mock_client

                await analyze_conversation(
                    messages=[{"role": "user", "text": "hello"}],
                    active_tasks=[],
                    active_sessions="",
                    api_key="key",
                )

                call_kwargs = mock_client.messages.create.call_args.kwargs
                assert "sonnet" in call_kwargs["model"]

        asyncio.run(_test())
