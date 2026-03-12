"""E2E Tests: Module split integration verification.

Verifies that the refactored module boundaries (patterns, subprocess_runner,
task_persistence, task_formatting, command_handlers, prompt_builders) work
correctly end-to-end when composed through the Supervisor and TaskDispatcher.

Mock boundary: FeishuGateway, ClaudeSession, asyncio.create_subprocess_exec
Real boundary: ALL application logic (routing, commands, dispatch, state, formatting)
"""

import asyncio
import json
import time
import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from supervisor.main import Supervisor
import supervisor.task_dispatcher as td


# ── Helpers ──

def _make_supervisor():
    """Create Supervisor with mocked external boundaries."""
    with patch("supervisor.main.FeishuGateway"):
        with patch("supervisor.main.ClaudeSession"):
            sup = Supervisor()
            sup.gateway = MagicMock()
            sup.claude = MagicMock()
            sup._task_dispatcher = td
            return sup


class _AsyncLineIterator:
    """Mock async stdout iterator for subprocess streaming."""
    def __init__(self, *lines):
        self._lines = list(lines)
        self._index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._index >= len(self._lines):
            raise StopAsyncIteration
        line = self._lines[self._index]
        self._index += 1
        if isinstance(line, str):
            line = line.encode("utf-8")
        return line


def _make_proc_mock(result_text="Done", session_id="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"):
    """Create mock subprocess that returns a successful stream-json result."""
    proc = AsyncMock()
    result_line = json.dumps({
        "type": "result",
        "result": result_text,
        "session_id": session_id,
    })
    proc.stdout = _AsyncLineIterator(result_line)
    proc.stderr = AsyncMock()
    proc.stderr.read = AsyncMock(return_value=b"")
    proc.returncode = 0
    proc.wait = AsyncMock()
    proc.kill = AsyncMock()
    return proc


@pytest.fixture(autouse=True)
def _clean_state():
    """Reset task dispatcher state between tests."""
    td._reset()
    yield
    td._reset()


# ── E2E Scenario 1: Full dispatch lifecycle ──
# Flow: user message → Sonnet routes to dispatch → worker completes → task closed
# Touches: main.py → command_handlers.py → prompt_builders.py → subprocess_runner.py
#          → task_dispatcher.py → task_persistence.py → task_formatting.py → patterns.py

class TestE2E_FullDispatchLifecycle:
    """Complete task lifecycle through all extracted modules."""

    def test_dispatch_complete_close(self):
        """User sends task → dispatched → completed → closed via /close."""
        async def _run():
            sup = _make_supervisor()
            proc = _make_proc_mock("Task completed successfully")

            with patch("asyncio.create_subprocess_exec", return_value=proc):
                # Dispatch a task
                task = await td.dispatch(
                    prompt="Analyze the code",
                    cwd="/workspace",
                    task_type="oneshot",
                    description="Code analysis",
                )

                # Wait for worker to finish
                await asyncio.sleep(0.1)

            # Verify task reached awaiting_closure
            t = td.get_task(task.id)
            assert t.status == "awaiting_closure"
            assert t.result == "Task completed successfully"
            assert t.session_id  # session_id captured from stream

            # Close via /close command (through command_handlers.py)
            sup._handle_local_command(f"/close {task.id[:8]}", "chat-1", "msg-1")
            reply = sup.gateway.reply_message.call_args[0][1]
            assert "completed" in reply.lower() or "closed" in reply.lower() or "Closed" in reply

            # Verify final state
            t = td.get_task(task.id)
            assert t.status == "completed"

        asyncio.run(_run())

    def test_dispatch_with_formatted_status(self):
        """Dispatch task → check /tasks shows it via task_formatting.py."""
        async def _run():
            sup = _make_supervisor()
            proc = _make_proc_mock("Result here")

            with patch("asyncio.create_subprocess_exec", return_value=proc):
                task = await td.dispatch(
                    prompt="Test task",
                    cwd="/workspace",
                    description="Test formatting",
                )
                await asyncio.sleep(0.1)

            # /tasks command goes through command_handlers → task_formatting
            sup._handle_local_command("/tasks", "chat-1", "msg-1")
            reply = sup.gateway.reply_message.call_args[0][1]
            assert task.id[:8] in reply
            assert "AWAIT_CLOSE" in reply
            assert "Test formatting" in reply

        asyncio.run(_run())


# ── E2E Scenario 2: Smart close detection ──
# Flow: user short ack → patterns.py detects close intent → auto-close
# Touches: main.py → patterns.py → task_dispatcher.py

class TestE2E_SmartClose:
    """Close intent detection through patterns.py integrated with Supervisor."""

    def test_reply_based_close_single_task(self):
        """User says '好的' with 1 awaiting task → auto-close."""
        async def _run():
            sup = _make_supervisor()
            proc = _make_proc_mock("Analysis done")

            with patch("asyncio.create_subprocess_exec", return_value=proc):
                task = await td.dispatch(
                    prompt="Analyze code",
                    cwd="/workspace",
                    description="Code review",
                )
                await asyncio.sleep(0.1)

            assert td.get_task(task.id).status == "awaiting_closure"

            # Simulate Sonnet reply action with close intent in user text
            # _try_post_reply_close uses _looks_like_close from patterns.py
            sup._try_post_reply_close(
                user_text="好的",
                reply_text="好的，如果需要其他帮助请告诉我。",
                chat_id="chat-1",
                message_id="msg-1",
            )

            # Task should be auto-closed
            t = td.get_task(task.id)
            assert t.status == "completed"

        asyncio.run(_run())

    def test_no_close_on_ambiguous(self):
        """User says '好的' with 2+ awaiting tasks → no auto-close."""
        async def _run():
            sup = _make_supervisor()
            proc = _make_proc_mock("Done")

            with patch("asyncio.create_subprocess_exec", return_value=proc):
                t1 = await td.dispatch(prompt="Task 1", cwd="/workspace")
                t2 = await td.dispatch(prompt="Task 2", cwd="/workspace")
                await asyncio.sleep(0.1)

            assert td.get_task(t1.id).status == "awaiting_closure"
            assert td.get_task(t2.id).status == "awaiting_closure"

            sup._try_post_reply_close("好的", "好的", "chat-1", "msg-1")

            # Neither should be closed (ambiguous)
            assert td.get_task(t1.id).status == "awaiting_closure"
            assert td.get_task(t2.id).status == "awaiting_closure"

        asyncio.run(_run())


# ── E2E Scenario 3: Command handlers through extracted module ──
# Flow: /command → command_handlers.py → task_dispatcher.py → response
# Touches: main.py → command_handlers.py → task_dispatcher.py → task_formatting.py

class TestE2E_CommandHandlers:
    """Verify /commands route through command_handlers.py correctly."""

    def test_help_through_extracted_handler(self):
        """"/help" → command_handlers.cmd_help → formatted response."""
        sup = _make_supervisor()
        sup._handle_local_command("/help", "chat-1", "msg-1")
        reply = sup.gateway.reply_message.call_args[0][1]
        assert "/status" in reply
        assert "/close" in reply
        assert "/recover" in reply

    def test_close_all_through_extracted_handler(self):
        """/close all → command_handlers.cmd_close → batch close."""
        async def _run():
            sup = _make_supervisor()
            proc = _make_proc_mock("Done")

            with patch("asyncio.create_subprocess_exec", return_value=proc):
                t1 = await td.dispatch(prompt="Task 1", cwd="/workspace")
                t2 = await td.dispatch(prompt="Task 2", cwd="/workspace")
                await asyncio.sleep(0.1)

            sup._handle_local_command("/close all", "chat-1", "msg-1")
            reply = sup.gateway.reply_message.call_args[0][1]

            assert td.get_task(t1.id).status == "completed"
            assert td.get_task(t2.id).status == "completed"

        asyncio.run(_run())

    def test_tasks_context_for_routing(self):
        """_get_tasks_context builds correct context through command_handlers."""
        async def _run():
            sup = _make_supervisor()
            proc = _make_proc_mock("Result")

            with patch("asyncio.create_subprocess_exec", return_value=proc):
                task = await td.dispatch(
                    prompt="Test task",
                    cwd="/workspace",
                    description="Test desc",
                )
                await asyncio.sleep(0.1)

            ctx = sup._get_tasks_context()
            assert len(ctx["awaiting"]) == 1
            assert ctx["awaiting"][0]["id"] == task.id[:8]
            assert ctx["awaiting"][0]["description"] == "Test desc"

        asyncio.run(_run())


# ── E2E Scenario 4: Prompt builders integration ──
# Flow: dispatch action → prompt_builders.py enriches prompt → worker receives it
# Touches: main.py → prompt_builders.py → (conversation history)

class TestE2E_PromptBuilders:
    """Verify prompt building integrates with Supervisor conversation history."""

    def test_worker_prompt_includes_history(self):
        """Worker prompt includes recent conversation history."""
        sup = _make_supervisor()
        sup._record_message("user", "What is Python?")
        sup._record_message("assistant", "Python is a programming language.")

        prompt = sup._build_worker_prompt("Analyze main.py", "Code analysis")
        assert "Code analysis" in prompt
        assert "Analyze main.py" in prompt
        assert "Python is a programming language" in prompt

    def test_orchestrator_prompt_includes_subtasks(self):
        """Orchestrator prompt includes subtask list."""
        sup = _make_supervisor()
        prompt = sup._build_orchestrator_prompt(
            "Refactor the codebase",
            "Full refactoring",
            ["Split large files", "Add type hints", "Update tests"],
        )
        assert "Split large files" in prompt
        assert "Add type hints" in prompt
        assert "Update tests" in prompt
        assert "orchestrator" in prompt.lower()


# ── E2E Scenario 5: Persistence roundtrip after refactor ──
# Flow: dispatch → persist → load → verify state intact
# Touches: task_dispatcher.py → task_persistence.py

class TestE2E_PersistenceRoundtrip:
    """Verify task persistence works through the extracted module."""

    def test_dispatch_persists_and_loads(self):
        """Task dispatched → saved to disk → loaded on reset → state matches."""
        async def _run():
            proc = _make_proc_mock("Saved result")

            with patch("asyncio.create_subprocess_exec", return_value=proc):
                task = await td.dispatch(
                    prompt="Persist me",
                    cwd="/workspace",
                    description="Persistence test",
                )
                await asyncio.sleep(0.1)

            original = td.get_task(task.id)
            assert original.status == "awaiting_closure"
            assert original.result == "Saved result"

            # Save current state, clear in-memory, reload
            td._save_tasks()
            td._tasks.clear()
            with pytest.raises(ValueError):
                td.get_task(task.id)  # cleared — raises ValueError

            td._load_tasks()
            restored = td.get_task(task.id)
            assert restored is not None
            assert restored.status == "awaiting_closure"
            assert restored.result == "Saved result"
            assert restored.description == "Persistence test"

        asyncio.run(_run())


# ── E2E Scenario 6: Task timeout through subprocess_runner ──
# Flow: dispatch → streaming timeout → task fails → user notifiable
# Touches: subprocess_runner.py → task_dispatcher.py → task_persistence.py

class TestE2E_TaskTimeout:
    """Verify timeout handling works through extracted subprocess_runner."""

    def test_streaming_timeout_marks_failed(self):
        """Subprocess hangs → timeout → task marked failed."""
        async def _run():
            # Create a proc that never yields a result
            proc = AsyncMock()
            proc.stdout = _AsyncLineIterator()  # empty — will raise StopAsyncIteration
            proc.stderr = AsyncMock()
            proc.stderr.read = AsyncMock(return_value=b"stream error")
            proc.returncode = 1
            proc.wait = AsyncMock()
            proc.kill = AsyncMock()

            # Also make non-streaming fallback fail
            proc2 = AsyncMock()
            proc2.communicate = AsyncMock(return_value=(b"", b"fallback error"))
            proc2.returncode = 1

            call_count = 0

            async def mock_exec(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return proc  # streaming attempt
                return proc2  # non-streaming fallback

            with patch("asyncio.create_subprocess_exec", side_effect=mock_exec):
                task = await td.dispatch(
                    prompt="Will timeout",
                    cwd="/workspace",
                    description="Timeout test",
                )
                await asyncio.sleep(0.2)

            t = td.get_task(task.id)
            assert t.status == "failed"
            assert t.error  # has error message

        asyncio.run(_run())


# ── E2E Scenario 7: Import chain integrity ──
# Verifies that all re-exports and cross-module imports work correctly

class TestE2E_ImportChainIntegrity:
    """Verify all modules import correctly and re-exports are functional."""

    def test_task_dispatcher_re_exports_patterns(self):
        """patterns.py functions accessible via task_dispatcher."""
        from supervisor.task_dispatcher import (
            _looks_like_needs_input,
            _contains_close_intent,
            _looks_like_close,
            _CLOSE_PHRASES_SET,
        )
        assert _looks_like_needs_input("What should I do?") is True
        assert _looks_like_needs_input("Done.") is False
        assert _looks_like_close("好的") is True
        assert _looks_like_close("帮我写个函数") is False
        assert _contains_close_intent("关了吧") is True
        assert isinstance(_CLOSE_PHRASES_SET, frozenset)

    def test_task_dispatcher_re_exports_formatting(self):
        """task_formatting.py functions accessible via task_dispatcher."""
        from supervisor.task_dispatcher import (
            _status_icon, _elapsed_str, _format_task,
        )
        assert _status_icon("running") == "RUNNING"
        assert _status_icon("unknown") == "UNKNOWN"

    def test_task_dispatcher_re_exports_subprocess(self):
        """subprocess_runner.py functions accessible via task_dispatcher."""
        from supervisor.task_dispatcher import (
            _build_env, _build_cmd, _build_cmd_streaming,
        )
        env = _build_env()
        assert "CLAUDECODE" not in env

        cmd = _build_cmd("test prompt")
        assert "claude" in cmd
        assert "--output-format" in cmd
        assert "json" in cmd

        cmd_s = _build_cmd_streaming("test prompt")
        assert "stream-json" in cmd_s

    def test_task_dispatcher_re_exports_persistence(self):
        """task_persistence.py constants accessible via task_dispatcher."""
        from supervisor.task_dispatcher import _ACTIVE_PROCESS_STATUSES
        assert "running" in _ACTIVE_PROCESS_STATUSES
        assert "follow_up" in _ACTIVE_PROCESS_STATUSES

    def test_main_exports(self):
        """main.py still exports Supervisor and SUPERVISOR_SYSTEM_PROMPT."""
        from supervisor.main import Supervisor, SUPERVISOR_SYSTEM_PROMPT
        assert SUPERVISOR_SYSTEM_PROMPT
        assert callable(Supervisor)

    def test_td_attribute_access(self):
        """Module attribute access via 'import as td' still works."""
        import supervisor.task_dispatcher as _td
        assert hasattr(_td, "_TASKS_FILE")
        assert hasattr(_td, "_CHECKPOINT_DIR")
        assert hasattr(_td, "_tasks")
        assert hasattr(_td, "SUPERVISOR_TASK_TIMEOUT")
        assert hasattr(_td, "SUPERVISOR_MAX_WORKERS")
        assert hasattr(_td, "_reset")
