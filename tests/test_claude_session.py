"""Tests for supervisor.claude_session module."""

import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from supervisor.claude_session import (
    ClaudeSession,
    StreamEvent,
    _build_cmd,
    _build_env,
    _parse_stream_line,
    DEFAULT_MODEL,
    DEFAULT_EFFORT,
    DEFAULT_PERMISSION_MODE,
)


# ── _build_cmd tests ──


class TestBuildCmd:
    def test_basic_streaming(self):
        cmd = _build_cmd("hello", streaming=True)
        assert cmd[0] == "claude"
        assert "-p" in cmd
        assert "hello" in cmd
        assert "--output-format" in cmd
        assert "stream-json" in cmd
        assert "--verbose" in cmd
        assert "--model" in cmd
        assert "--effort" in cmd
        assert "--permission-mode" in cmd

    def test_basic_json(self):
        cmd = _build_cmd("hello", streaming=False)
        assert "json" in cmd
        assert "--verbose" not in cmd
        assert "stream-json" not in cmd

    def test_with_session_id(self):
        cmd = _build_cmd("test", session_id="abc-123")
        assert "--resume" in cmd
        idx = cmd.index("--resume")
        assert cmd[idx + 1] == "abc-123"

    def test_without_session_id(self):
        cmd = _build_cmd("test")
        assert "--resume" not in cmd

    def test_with_system_prompt(self):
        cmd = _build_cmd("test", system_prompt="Be helpful")
        assert "--append-system-prompt" in cmd
        idx = cmd.index("--append-system-prompt")
        assert cmd[idx + 1] == "Be helpful"

    def test_custom_model(self):
        cmd = _build_cmd("test", model="sonnet")
        idx = cmd.index("--model")
        assert cmd[idx + 1] == "sonnet"

    def test_default_model(self):
        cmd = _build_cmd("test")
        idx = cmd.index("--model")
        assert cmd[idx + 1] == DEFAULT_MODEL

    def test_effort_max(self):
        cmd = _build_cmd("test")
        idx = cmd.index("--effort")
        assert cmd[idx + 1] == "max"

    def test_bypass_permissions(self):
        cmd = _build_cmd("test")
        idx = cmd.index("--permission-mode")
        assert cmd[idx + 1] == "bypassPermissions"


# ── _build_env tests ──


class TestBuildEnv:
    def test_removes_claudecode(self):
        with patch.dict("os.environ", {"CLAUDECODE": "1", "PATH": "/usr/bin"}):
            env = _build_env()
            assert "CLAUDECODE" not in env
            assert "PATH" in env

    def test_preserves_other_vars(self):
        with patch.dict("os.environ", {"FOO": "bar", "BAZ": "qux"}, clear=True):
            env = _build_env()
            assert env["FOO"] == "bar"
            assert env["BAZ"] == "qux"


# ── _parse_stream_line tests ──


class TestParseStreamLine:
    def test_empty_line(self):
        assert _parse_stream_line("") is None
        assert _parse_stream_line("   ") is None

    def test_invalid_json(self):
        assert _parse_stream_line("not json") is None

    def test_result_event(self):
        data = {
            "type": "result",
            "subtype": "success",
            "result": "Hello world",
            "session_id": "sess-1",
        }
        event = _parse_stream_line(json.dumps(data))
        assert event.type == "result"
        assert event.subtype == "success"
        assert event.text == "Hello world"
        assert event.session_id == "sess-1"
        assert event.is_final is True

    def test_assistant_event(self):
        data = {
            "type": "assistant",
            "message": {
                "content": [{"type": "text", "text": "Hi there"}]
            },
            "session_id": "sess-2",
        }
        event = _parse_stream_line(json.dumps(data))
        assert event.type == "assistant"
        assert event.text == "Hi there"
        assert event.session_id == "sess-2"
        assert event.is_final is False

    def test_assistant_multi_content(self):
        data = {
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "text", "text": "Hello "},
                    {"type": "text", "text": "world"},
                ]
            },
            "session_id": "s",
        }
        event = _parse_stream_line(json.dumps(data))
        assert event.text == "Hello world"

    def test_system_event(self):
        data = {"type": "system", "subtype": "init", "session_id": "s"}
        event = _parse_stream_line(json.dumps(data))
        assert event.type == "system"
        assert event.subtype == "init"

    def test_result_error(self):
        data = {
            "type": "result",
            "subtype": "error",
            "result": "something failed",
            "session_id": "s",
        }
        event = _parse_stream_line(json.dumps(data))
        assert event.is_final is True
        assert event.subtype == "error"

    def test_unknown_type(self):
        data = {"type": "unknown_event", "session_id": "s"}
        event = _parse_stream_line(json.dumps(data))
        assert event.type == "unknown_event"


# ── ClaudeSession tests ──


class TestClaudeSession:
    def test_init_no_session(self):
        with patch.object(ClaudeSession, "_load_session_id", return_value=None):
            session = ClaudeSession()
            assert session.session_id is None

    def test_init_with_session_id(self):
        session = ClaudeSession(session_id="explicit-id")
        assert session.session_id == "explicit-id"

    def test_init_with_model(self):
        session = ClaudeSession(model="sonnet")
        assert session.model == "sonnet"

    def test_call_timeout(self):
        import asyncio

        async def _test():
            session = ClaudeSession(session_id=None)
            with patch("supervisor.claude_session.asyncio.create_subprocess_exec") as mock_exec:
                mock_proc = AsyncMock()
                mock_exec.return_value = mock_proc
                with patch("supervisor.claude_session.asyncio.wait_for", side_effect=asyncio.TimeoutError):
                    mock_proc.kill = AsyncMock()
                    result = await session.call("test")
                    assert "timed out" in result.lower()

        asyncio.run(_test())

    def test_call_not_found(self):
        import asyncio

        async def _test():
            session = ClaudeSession(session_id=None)
            with patch("supervisor.claude_session.asyncio.create_subprocess_exec", side_effect=FileNotFoundError):
                result = await session.call("test")
                assert "not found" in result.lower()

        asyncio.run(_test())

    def test_call_success(self):
        import asyncio

        async def _test():
            session = ClaudeSession(session_id=None)
            response_data = {
                "type": "result",
                "subtype": "success",
                "result": "Hello!",
                "session_id": "new-session-id",
            }
            stdout = json.dumps(response_data).encode()

            mock_proc = AsyncMock()
            mock_proc.communicate = AsyncMock(return_value=(stdout, b""))
            mock_proc.returncode = 0

            with patch("supervisor.claude_session.asyncio.create_subprocess_exec", return_value=mock_proc):
                with patch("supervisor.claude_session.asyncio.wait_for", return_value=(stdout, b"")):
                    with patch.object(session, "_save_session_id"):
                        result = await session.call("test")
                        assert result == "Hello!"
                        assert session.session_id == "new-session-id"

        asyncio.run(_test())

    def test_route_message_json_reply(self):
        """When sonnet returns valid JSON with action=reply, return it as-is."""
        import asyncio

        async def _test():
            session = ClaudeSession(session_id=None)
            inner_json = '{"action": "reply", "text": "我是 Supervisor Hub"}'
            response_data = {"result": inner_json}
            stdout = json.dumps(response_data).encode()

            mock_proc = AsyncMock()
            mock_proc.communicate = AsyncMock(return_value=(stdout, b""))
            mock_proc.returncode = 0

            with patch("supervisor.claude_session.asyncio.create_subprocess_exec", return_value=mock_proc):
                with patch("supervisor.claude_session.asyncio.wait_for", return_value=(stdout, b"")):
                    result = await session.route_message("你好", "system", "user")
                    assert result["action"] == "reply"
                    assert "Supervisor" in result["text"]

        asyncio.run(_test())

    def test_route_message_plain_text_treated_as_reply(self):
        """When sonnet returns plain text (not JSON), treat it as a reply."""
        import asyncio

        async def _test():
            session = ClaudeSession(session_id=None)
            response_data = {"result": "你好！我是 Supervisor Hub，有什么可以帮你的？"}
            stdout = json.dumps(response_data).encode()

            mock_proc = AsyncMock()
            mock_proc.communicate = AsyncMock(return_value=(stdout, b""))
            mock_proc.returncode = 0

            with patch("supervisor.claude_session.asyncio.create_subprocess_exec", return_value=mock_proc):
                with patch("supervisor.claude_session.asyncio.wait_for", return_value=(stdout, b"")):
                    result = await session.route_message("你好", "system", "user")
                    assert result["action"] == "reply"
                    assert "Supervisor" in result["text"]

        asyncio.run(_test())

    def test_route_message_dispatch(self):
        """When sonnet returns action=dispatch JSON, return it."""
        import asyncio

        async def _test():
            session = ClaudeSession(session_id=None)
            inner_json = '{"action": "dispatch", "description": "分析代码"}'
            response_data = {"result": inner_json}
            stdout = json.dumps(response_data).encode()

            mock_proc = AsyncMock()
            mock_proc.communicate = AsyncMock(return_value=(stdout, b""))
            mock_proc.returncode = 0

            with patch("supervisor.claude_session.asyncio.create_subprocess_exec", return_value=mock_proc):
                with patch("supervisor.claude_session.asyncio.wait_for", return_value=(stdout, b"")):
                    result = await session.route_message("帮我分析代码", "system", "user")
                    assert result["action"] == "dispatch"
                    assert "分析" in result["description"]

        asyncio.run(_test())

    def test_route_message_markdown_wrapped_json(self):
        """When sonnet wraps JSON in markdown code blocks, strip and parse."""
        import asyncio

        async def _test():
            session = ClaudeSession(session_id=None)
            inner = '```json\n{"action": "reply", "text": "hello"}\n```'
            response_data = {"result": inner}
            stdout = json.dumps(response_data).encode()

            mock_proc = AsyncMock()
            mock_proc.communicate = AsyncMock(return_value=(stdout, b""))
            mock_proc.returncode = 0

            with patch("supervisor.claude_session.asyncio.create_subprocess_exec", return_value=mock_proc):
                with patch("supervisor.claude_session.asyncio.wait_for", return_value=(stdout, b"")):
                    result = await session.route_message("hi", "system", "user")
                    assert result["action"] == "reply"

        asyncio.run(_test())

    def test_route_message_unescaped_chinese_quotes(self):
        """When sonnet returns JSON with unescaped Chinese quotes, extract action via regex."""
        import asyncio

        async def _test():
            session = ClaudeSession(session_id=None)
            # Sonnet generates: "看到"排队"的现象" — the inner " breaks json.loads
            broken_json = '{"action": "reply", "text": "能说说你在哪里看到"排队"的现象吗？"}'
            response_data = {"result": broken_json}
            stdout = json.dumps(response_data).encode()

            mock_proc = AsyncMock()
            mock_proc.communicate = AsyncMock(return_value=(stdout, b""))
            mock_proc.returncode = 0

            with patch("supervisor.claude_session.asyncio.create_subprocess_exec", return_value=mock_proc):
                with patch("supervisor.claude_session.asyncio.wait_for", return_value=(stdout, b"")):
                    result = await session.route_message("为什么排队", "system", "user")
                    assert result["action"] == "reply"
                    # Should extract the text, not return raw JSON string
                    assert "排队" in result["text"]
                    assert '"action"' not in result["text"]

        asyncio.run(_test())

    def test_route_message_unescaped_quotes_dispatch(self):
        """Broken JSON with action=dispatch should still extract correctly."""
        import asyncio

        async def _test():
            session = ClaudeSession(session_id=None)
            broken_json = '{"action": "dispatch", "description": "分析"核心"模块的代码"}'
            response_data = {"result": broken_json}
            stdout = json.dumps(response_data).encode()

            mock_proc = AsyncMock()
            mock_proc.communicate = AsyncMock(return_value=(stdout, b""))
            mock_proc.returncode = 0

            with patch("supervisor.claude_session.asyncio.create_subprocess_exec", return_value=mock_proc):
                with patch("supervisor.claude_session.asyncio.wait_for", return_value=(stdout, b"")):
                    result = await session.route_message("分析核心模块", "system", "user")
                    assert result["action"] == "dispatch"

        asyncio.run(_test())

    def test_route_message_follow_up_action(self):
        """follow_up action should be recognized as valid."""
        import asyncio

        async def _test():
            session = ClaudeSession(session_id=None)
            inner_json = '{"action": "follow_up", "task_id": "8b557777", "text": "结果对吗"}'
            response_data = {"result": inner_json}
            stdout = json.dumps(response_data).encode()

            mock_proc = AsyncMock()
            mock_proc.communicate = AsyncMock(return_value=(stdout, b""))
            mock_proc.returncode = 0

            with patch("supervisor.claude_session.asyncio.create_subprocess_exec", return_value=mock_proc):
                with patch("supervisor.claude_session.asyncio.wait_for", return_value=(stdout, b"")):
                    result = await session.route_message("结果对吗", "system", "user")
                    assert result["action"] == "follow_up"
                    assert result["task_id"] == "8b557777"

        asyncio.run(_test())

    def test_route_message_newlines_in_text(self):
        """JSON with escaped newlines in text should parse correctly."""
        import asyncio

        async def _test():
            session = ClaudeSession(session_id=None)
            inner_json = '{"action": "reply", "text": "第一行\\n第二行\\n第三行"}'
            response_data = {"result": inner_json}
            stdout = json.dumps(response_data).encode()

            mock_proc = AsyncMock()
            mock_proc.communicate = AsyncMock(return_value=(stdout, b""))
            mock_proc.returncode = 0

            with patch("supervisor.claude_session.asyncio.create_subprocess_exec", return_value=mock_proc):
                with patch("supervisor.claude_session.asyncio.wait_for", return_value=(stdout, b"")):
                    result = await session.route_message("test", "system", "user")
                    assert result["action"] == "reply"
                    assert "第一行" in result["text"]

        asyncio.run(_test())

    def test_route_message_timeout_logging(self):
        """Timeout should be logged clearly and fallback to dispatch."""
        import asyncio

        async def _test():
            session = ClaudeSession(session_id=None)

            with patch("supervisor.claude_session.asyncio.create_subprocess_exec") as mock_exec:
                mock_proc = AsyncMock()
                mock_exec.return_value = mock_proc
                with patch("supervisor.claude_session.asyncio.wait_for", side_effect=asyncio.TimeoutError):
                    mock_proc.kill = AsyncMock()
                    result = await session.route_message("test", "system", "user")
                    assert result["action"] == "dispatch"

        asyncio.run(_test())

    def test_route_message_multiple_unescaped_quotes(self):
        """Multiple pairs of unescaped Chinese quotes."""
        import asyncio

        async def _test():
            session = ClaudeSession(session_id=None)
            broken = '{"action": "reply", "text": "你说的"A"和"B"都对"}'
            response_data = {"result": broken}
            stdout = json.dumps(response_data).encode()

            mock_proc = AsyncMock()
            mock_proc.communicate = AsyncMock(return_value=(stdout, b""))
            mock_proc.returncode = 0

            with patch("supervisor.claude_session.asyncio.create_subprocess_exec", return_value=mock_proc):
                with patch("supervisor.claude_session.asyncio.wait_for", return_value=(stdout, b"")):
                    result = await session.route_message("test", "system", "user")
                    assert result["action"] == "reply"
                    assert "A" in result["text"]
                    assert "B" in result["text"]
                    assert '"action"' not in result["text"]

        asyncio.run(_test())

    def test_route_message_json_with_colon_in_text(self):
        """JSON where text contains colons (common in Chinese responses)."""
        import asyncio

        async def _test():
            session = ClaudeSession(session_id=None)
            inner = '{"action": "reply", "text": "原因如下：1. 网络延迟 2. 负载过高"}'
            response_data = {"result": inner}
            stdout = json.dumps(response_data).encode()

            mock_proc = AsyncMock()
            mock_proc.communicate = AsyncMock(return_value=(stdout, b""))
            mock_proc.returncode = 0

            with patch("supervisor.claude_session.asyncio.create_subprocess_exec", return_value=mock_proc):
                with patch("supervisor.claude_session.asyncio.wait_for", return_value=(stdout, b"")):
                    result = await session.route_message("test", "system", "user")
                    assert result["action"] == "reply"
                    assert "原因如下" in result["text"]

        asyncio.run(_test())

    def test_route_message_broken_json_starts_with_brace_no_dispatch_as_text(self):
        """If result starts with { but can't be parsed, should NOT send raw JSON as reply text."""
        import asyncio

        async def _test():
            session = ClaudeSession(session_id=None)
            # Completely mangled JSON that regex can't extract either
            mangled = '{"action: broken'
            response_data = {"result": mangled}
            stdout = json.dumps(response_data).encode()

            mock_proc = AsyncMock()
            mock_proc.communicate = AsyncMock(return_value=(stdout, b""))
            mock_proc.returncode = 0

            with patch("supervisor.claude_session.asyncio.create_subprocess_exec", return_value=mock_proc):
                with patch("supervisor.claude_session.asyncio.wait_for", return_value=(stdout, b"")):
                    result = await session.route_message("test", "system", "user")
                    # Should fall back to dispatch, NOT send raw JSON as reply
                    assert result["action"] == "dispatch"

        asyncio.run(_test())

    def test_route_message_plain_text_action_verb_dispatches(self):
        """Short plain text with action verbs → dispatch."""
        import asyncio
        async def _test():
            session = ClaudeSession(session_id=None)
            data = {"result": "帮我分析一下这段代码的性能问题"}
            stdout = json.dumps(data).encode()
            mock_proc = AsyncMock()
            mock_proc.communicate = AsyncMock(return_value=(stdout, b""))
            mock_proc.returncode = 0
            with patch("supervisor.claude_session.asyncio.create_subprocess_exec", return_value=mock_proc):
                with patch("supervisor.claude_session.asyncio.wait_for", return_value=(stdout, b"")):
                    result = await session.route_message("分析代码", "system", "user")
                    assert result["action"] == "dispatch"
                    assert "分析" in result["description"]
        asyncio.run(_test())

    def test_route_message_plain_text_long_replies(self):
        """Long plain text (>=200 chars) → reply even with action verbs."""
        import asyncio
        async def _test():
            session = ClaudeSession(session_id=None)
            long_text = "这是一段很长的分析结果，" * 20
            data = {"result": long_text}
            stdout = json.dumps(data).encode()
            mock_proc = AsyncMock()
            mock_proc.communicate = AsyncMock(return_value=(stdout, b""))
            mock_proc.returncode = 0
            with patch("supervisor.claude_session.asyncio.create_subprocess_exec", return_value=mock_proc):
                with patch("supervisor.claude_session.asyncio.wait_for", return_value=(stdout, b"")):
                    result = await session.route_message("test", "system", "user")
                    assert result["action"] == "reply"
        asyncio.run(_test())

    def test_route_message_plain_text_no_verb_replies(self):
        """Short plain text without action verbs → reply."""
        import asyncio
        async def _test():
            session = ClaudeSession(session_id=None)
            data = {"result": "你好！有什么可以帮你的？"}
            stdout = json.dumps(data).encode()
            mock_proc = AsyncMock()
            mock_proc.communicate = AsyncMock(return_value=(stdout, b""))
            mock_proc.returncode = 0
            with patch("supervisor.claude_session.asyncio.create_subprocess_exec", return_value=mock_proc):
                with patch("supervisor.claude_session.asyncio.wait_for", return_value=(stdout, b"")):
                    result = await session.route_message("你好", "system", "user")
                    assert result["action"] == "reply"
        asyncio.run(_test())

    def test_route_message_orchestrate_regex_subtasks(self):
        """Broken orchestrate JSON → extract subtasks via regex."""
        import asyncio
        async def _test():
            session = ClaudeSession(session_id=None)
            broken = '{"action": "orchestrate", "description": "执行"多步"任务", "subtasks": ["分析代码", "运行测试", "生成报告"]}'
            data = {"result": broken}
            stdout = json.dumps(data).encode()
            mock_proc = AsyncMock()
            mock_proc.communicate = AsyncMock(return_value=(stdout, b""))
            mock_proc.returncode = 0
            with patch("supervisor.claude_session.asyncio.create_subprocess_exec", return_value=mock_proc):
                with patch("supervisor.claude_session.asyncio.wait_for", return_value=(stdout, b"")):
                    result = await session.route_message("多任务", "system", "user")
                    assert result["action"] == "orchestrate"
                    assert "分析代码" in result["subtasks"]
                    assert "运行测试" in result["subtasks"]
        asyncio.run(_test())

    def test_route_message_dispatch_multi_compat_becomes_orchestrate(self):
        """Legacy dispatch_multi from sonnet → converted to orchestrate."""
        import asyncio
        async def _test():
            session = ClaudeSession(session_id=None)
            legacy = '{"action": "dispatch_multi", "description": "重构", "subtasks": ["A", "B"]}'
            data = {"result": legacy}
            stdout = json.dumps(data).encode()
            mock_proc = AsyncMock()
            mock_proc.communicate = AsyncMock(return_value=(stdout, b""))
            mock_proc.returncode = 0
            with patch("supervisor.claude_session.asyncio.create_subprocess_exec", return_value=mock_proc):
                with patch("supervisor.claude_session.asyncio.wait_for", return_value=(stdout, b"")):
                    result = await session.route_message("重构", "system", "user")
                    assert result["action"] == "orchestrate"
        asyncio.run(_test())

    def test_route_message_empty_result(self):
        """Empty result should dispatch, not crash."""
        import asyncio

        async def _test():
            session = ClaudeSession(session_id=None)
            response_data = {"result": ""}
            stdout = json.dumps(response_data).encode()

            mock_proc = AsyncMock()
            mock_proc.communicate = AsyncMock(return_value=(stdout, b""))
            mock_proc.returncode = 0

            with patch("supervisor.claude_session.asyncio.create_subprocess_exec", return_value=mock_proc):
                with patch("supervisor.claude_session.asyncio.wait_for", return_value=(stdout, b"")):
                    result = await session.route_message("test", "system", "user")
                    assert result["action"] == "dispatch"

        asyncio.run(_test())


class TestExtractFieldValue:
    """Unit tests for _extract_field_value regex extraction."""

    def test_simple_value(self):
        text = '{"action": "reply", "text": "hello world"}'
        assert ClaudeSession._extract_field_value(text, "text") == "hello world"

    def test_value_with_unescaped_quotes(self):
        text = '{"action": "reply", "text": "看到"排队"的现象"}'
        result = ClaudeSession._extract_field_value(text, "text")
        assert "排队" in result
        assert "看到" in result

    def test_value_with_multiple_unescaped_quotes(self):
        text = '{"action": "reply", "text": "你说的"A"和"B"都对"}'
        result = ClaudeSession._extract_field_value(text, "text")
        assert "A" in result
        assert "B" in result

    def test_description_field(self):
        text = '{"action": "dispatch", "description": "分析代码结构"}'
        assert ClaudeSession._extract_field_value(text, "description") == "分析代码结构"

    def test_task_id_field(self):
        text = '{"action": "follow_up", "task_id": "8b557777", "text": "结果对吗"}'
        assert ClaudeSession._extract_field_value(text, "task_id") == "8b557777"

    def test_missing_field(self):
        text = '{"action": "reply", "text": "hello"}'
        assert ClaudeSession._extract_field_value(text, "nonexistent") == ""

    def test_value_with_newlines(self):
        text = r'{"action": "reply", "text": "第一行\n第二行"}'
        result = ClaudeSession._extract_field_value(text, "text")
        assert "第一行" in result

    def test_value_with_colons(self):
        text = '{"action": "reply", "text": "原因：1. 网络 2. 负载"}'
        result = ClaudeSession._extract_field_value(text, "text")
        assert "原因" in result


# ── Bug fix: close_all and close action parsing ──


class TestCloseActionParsing:
    """Bug: close_all not in _VALID_ACTIONS → silently falls back to dispatch."""

    def test_close_all_in_valid_actions(self):
        """close_all must be a valid action for route parsing."""
        assert "close_all" in ClaudeSession._VALID_ACTIONS

    def test_close_in_valid_actions(self):
        """close must be a valid action for route parsing."""
        assert "close" in ClaudeSession._VALID_ACTIONS

    def test_route_message_close_all_json(self):
        """Sonnet returning close_all JSON should be parsed correctly."""
        import asyncio

        async def _test():
            session = ClaudeSession(session_id=None)
            response_json = '{"action": "close_all"}'
            data = {"result": response_json}
            stdout = json.dumps(data).encode()
            mock_proc = AsyncMock()
            mock_proc.communicate = AsyncMock(return_value=(stdout, b""))
            mock_proc.returncode = 0
            with patch("supervisor.claude_session.asyncio.create_subprocess_exec", return_value=mock_proc):
                with patch("supervisor.claude_session.asyncio.wait_for", return_value=(stdout, b"")):
                    result = await session.route_message("全部关闭", "system", "user")
                    assert result["action"] == "close_all"

        asyncio.run(_test())

    def test_route_message_close_with_task_id(self):
        """Sonnet returning close JSON with task_id should be parsed correctly."""
        import asyncio

        async def _test():
            session = ClaudeSession(session_id=None)
            response_json = '{"action": "close", "task_id": "8b557777"}'
            data = {"result": response_json}
            stdout = json.dumps(data).encode()
            mock_proc = AsyncMock()
            mock_proc.communicate = AsyncMock(return_value=(stdout, b""))
            mock_proc.returncode = 0
            with patch("supervisor.claude_session.asyncio.create_subprocess_exec", return_value=mock_proc):
                with patch("supervisor.claude_session.asyncio.wait_for", return_value=(stdout, b"")):
                    result = await session.route_message("关闭8b557777", "system", "user")
                    assert result["action"] == "close"
                    assert result["task_id"] == "8b557777"

        asyncio.run(_test())

    def test_route_message_close_with_task_ids(self):
        """Sonnet returning close JSON with task_ids array should be parsed."""
        import asyncio

        async def _test():
            session = ClaudeSession(session_id=None)
            response_json = '{"action": "close", "task_ids": ["aaa111", "bbb222"]}'
            data = {"result": response_json}
            stdout = json.dumps(data).encode()
            mock_proc = AsyncMock()
            mock_proc.communicate = AsyncMock(return_value=(stdout, b""))
            mock_proc.returncode = 0
            with patch("supervisor.claude_session.asyncio.create_subprocess_exec", return_value=mock_proc):
                with patch("supervisor.claude_session.asyncio.wait_for", return_value=(stdout, b"")):
                    result = await session.route_message("关闭前两个", "system", "user")
                    assert result["action"] == "close"
                    assert result["task_ids"] == ["aaa111", "bbb222"]

        asyncio.run(_test())

    def test_regex_extract_close_action(self):
        """Regex fallback should handle close action with task_id."""
        session = ClaudeSession(session_id=None)
        malformed = '{"action": "close", "task_id": "8b557777"}'
        result = session._try_regex_extract(malformed)
        assert result is not None
        assert result["action"] == "close"
        assert result["task_id"] == "8b557777"

    def test_regex_extract_close_no_id_returns_none(self):
        """Regex fallback should return None for close without any identifier."""
        session = ClaudeSession(session_id=None)
        malformed = '{"action": "close"'  # truncated, no task_id
        result = session._try_regex_extract(malformed)
        assert result is None

    def test_regex_extract_close_all_action(self):
        """Regex fallback should handle close_all action."""
        session = ClaudeSession(session_id=None)
        malformed = '{"action": "close_all"}'
        result = session._try_regex_extract(malformed)
        assert result is not None
        assert result["action"] == "close_all"
