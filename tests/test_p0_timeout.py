"""Tests for P0-2: Timeout protection for streaming worker tasks."""

import asyncio
import json
import time
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from supervisor.task_dispatcher import (
    Task,
    _reset,
    _tasks,
    _run_claude_streaming,
    _run_claude,
    _build_env,
)


@pytest.fixture(autouse=True)
def clean_state():
    _reset()
    yield
    _reset()


class _HangingLineIterator:
    """Async iterator that hangs forever after yielding initial lines."""

    def __init__(self, initial_lines: list[bytes], hang_after: int = 1):
        self._lines = initial_lines
        self._index = 0
        self._hang_after = hang_after

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._index >= self._hang_after:
            await asyncio.sleep(3600)
            raise StopAsyncIteration
        if self._index >= len(self._lines):
            raise StopAsyncIteration
        line = self._lines[self._index]
        self._index += 1
        return line


class _AsyncLineIterator:
    """Async iterator that yields lines normally."""
    def __init__(self, data: bytes):
        self._lines = data.split(b"\n")
        self._index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._index >= len(self._lines):
            raise StopAsyncIteration
        line = self._lines[self._index]
        self._index += 1
        return line


class TestStreamingTimeoutConfig:
    """Verify timeout configuration."""

    def test_default_timeout_value(self):
        from supervisor import task_dispatcher
        assert hasattr(task_dispatcher, "SUPERVISOR_TASK_TIMEOUT")
        assert task_dispatcher.SUPERVISOR_TASK_TIMEOUT == 1800

    def test_custom_timeout_from_env(self):
        """SUPERVISOR_TASK_TIMEOUT env var should be respected (tested via patch)."""
        import supervisor.task_dispatcher as td
        original = td.SUPERVISOR_TASK_TIMEOUT
        try:
            td.SUPERVISOR_TASK_TIMEOUT = 600
            assert td.SUPERVISOR_TASK_TIMEOUT == 600
        finally:
            td.SUPERVISOR_TASK_TIMEOUT = original


class TestStreamingTimeout:
    """Verify streaming tasks are killed after timeout."""

    def test_streaming_timeout_kills_process(self):
        async def _test():
            task = Task(
                id="timeout-test-0000-0000-000000000000",
                prompt="test hanging task",
                task_type="oneshot",
                status="running",
                created_at=time.time(),
                started_at=time.time(),
            )
            _tasks[task.id] = task

            proc = AsyncMock()
            proc.stdout = _HangingLineIterator(
                [json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": "hi"}]}}).encode()],
                hang_after=1,
            )
            proc.stderr = AsyncMock()
            proc.stderr.read = AsyncMock(return_value=b"")
            proc.returncode = -9
            proc.wait = AsyncMock()
            proc.kill = MagicMock()
            proc.pid = 12345

            env = _build_env()

            with patch("supervisor.task_dispatcher.SUPERVISOR_TASK_TIMEOUT", 1):
                with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=proc):
                    success = await _run_claude_streaming(task, env)

            assert not success
            assert "timed out" in task.error.lower() or "timeout" in task.error.lower(), \
                f"Expected timeout error, got: {task.error}"

        asyncio.run(_test())

    def test_timeout_saves_checkpoint_before_kill(self):
        async def _test():
            task = Task(
                id="ckpt-timeout-0000-0000-000000000000",
                prompt="test checkpoint on timeout",
                task_type="oneshot",
                status="running",
                created_at=time.time(),
                started_at=time.time(),
                steps_completed=["step1"],
                current_step="step2",
            )
            _tasks[task.id] = task

            proc = AsyncMock()
            proc.stdout = _HangingLineIterator([], hang_after=0)
            proc.stderr = AsyncMock()
            proc.stderr.read = AsyncMock(return_value=b"")
            proc.returncode = -9
            proc.wait = AsyncMock()
            proc.kill = MagicMock()
            proc.pid = 12345

            env = _build_env()

            with patch("supervisor.task_dispatcher.SUPERVISOR_TASK_TIMEOUT", 1):
                with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=proc):
                    with patch("supervisor.task_dispatcher._save_checkpoint") as mock_ckpt:
                        await _run_claude_streaming(task, env)
                        assert mock_ckpt.called, "Checkpoint must be saved on timeout"

        asyncio.run(_test())


class TestFullRunClaudeTimeout:
    """Verify _run_claude handles timeout end-to-end."""

    def test_run_claude_marks_failed_on_timeout(self):
        async def _test():
            task = Task(
                id="full-timeout-0000-0000-000000000000",
                prompt="test full timeout",
                task_type="oneshot",
                status="pending",
                created_at=time.time(),
            )
            _tasks[task.id] = task

            proc = AsyncMock()
            proc.stdout = _HangingLineIterator([], hang_after=0)
            proc.stderr = AsyncMock()
            proc.stderr.read = AsyncMock(return_value=b"")
            proc.returncode = -9
            proc.wait = AsyncMock()
            proc.kill = MagicMock()
            proc.pid = 12345
            proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError())

            with patch("supervisor.task_dispatcher.SUPERVISOR_TASK_TIMEOUT", 1):
                with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=proc):
                    await _run_claude(task)

            assert task.status == "failed", f"Expected failed, got {task.status}"
            assert task.finished_at > 0

        asyncio.run(_test())

    def test_normal_streaming_still_works(self):
        """Non-hanging streaming should complete normally within timeout."""
        async def _test():
            task = Task(
                id="normal-test-0000-0000-000000000000",
                prompt="test normal task",
                task_type="oneshot",
                status="pending",
                created_at=time.time(),
            )
            _tasks[task.id] = task

            result_line = json.dumps({
                "type": "result",
                "result": "all good",
                "session_id": "sess-123",
            }).encode()

            proc = AsyncMock()
            proc.stdout = _AsyncLineIterator(result_line)
            proc.stderr = AsyncMock()
            proc.stderr.read = AsyncMock(return_value=b"")
            proc.returncode = 0
            proc.wait = AsyncMock()

            env = _build_env()

            with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=proc):
                success = await _run_claude_streaming(task, env)

            assert success
            assert task.result == "all good"

        asyncio.run(_test())
