"""Tests for /recover command and restart recovery enhancements.

TDD Red Phase: These tests define the expected behavior for:
1. /recover command registration and handler in Supervisor
2. Recovery of tasks in active-process states (follow_up, learning) on restart
3. recover_task() integration via the command handler
"""

import asyncio
import json
import time
from dataclasses import asdict
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from supervisor.task_dispatcher import (
    Task,
    _load_tasks,
    _save_tasks,
    _reset,
    _tasks,
    list_interrupted,
    recover_task,
)
import supervisor.task_dispatcher as td


# ── Helpers ──


def _make_task(**overrides) -> Task:
    """Create a Task with sensible defaults."""
    defaults = dict(
        id="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        prompt="test prompt",
        task_type="oneshot",
        status="running",
        description="test task",
        current_step="doing stuff",
        steps_completed=["step1", "step2"],
        session_id="11111111-2222-3333-4444-555555555555",
        cwd="/workspace",
        result="",
        error="",
        created_at=1000.0,
        started_at=1001.0,
        finished_at=0.0,
        retries=0,
        max_retries=3,
    )
    defaults.update(overrides)
    return Task(**defaults)


def _write_tasks_json(tasks_file: Path, tasks: dict[str, Task]) -> None:
    """Write tasks dict to a JSON file."""
    data = {tid: asdict(t) for tid, t in tasks.items()}
    tasks_file.write_text(json.dumps(data, ensure_ascii=False, default=str))


# ── Fixtures ──


@pytest.fixture(autouse=True)
def _clean_state():
    """Reset dispatcher state before and after each test."""
    _reset()
    yield
    _reset()


@pytest.fixture
def tasks_file(tmp_path):
    """Provide a temp tasks file and restore original after test."""
    f = tmp_path / "tasks.json"
    original = td._TASKS_FILE
    td._TASKS_FILE = f
    _tasks.clear()
    yield f
    td._TASKS_FILE = original
    _tasks.clear()


# ══════════════════════════════════════════════════════════
# 1. /recover command registration and handler
# ══════════════════════════════════════════════════════════


class TestRecoverCommandRegistration:
    """/recover must be registered and routable via _handle_local_command."""

    def _make_supervisor(self):
        with patch("supervisor.main.FeishuGateway"):
            with patch("supervisor.main.ClaudeSession"):
                from supervisor.main import Supervisor
                sup = Supervisor()
                sup.gateway = MagicMock()
                sup.claude = MagicMock()
                return sup

    def test_recover_command_is_registered(self):
        """/recover should be handled (return True), not fall through."""
        sup = self._make_supervisor()
        result = sup._handle_local_command("/recover", "chat-1", "msg-1")
        assert result is True

    def test_recover_no_arg_no_interrupted_tasks(self):
        """/recover with no args and no interrupted tasks should say so."""
        sup = self._make_supervisor()
        sup._handle_local_command("/recover", "chat-1", "msg-1")
        reply_text = sup.gateway.reply_message.call_args[0][1]
        assert "No interrupted" in reply_text or "no interrupted" in reply_text.lower()

    def test_recover_shows_interrupted_list_when_no_id(self):
        """/recover without task_id should list interrupted tasks."""
        sup = self._make_supervisor()
        mock_dispatcher = MagicMock()
        mock_t1 = MagicMock()
        mock_t1.id = "aaaa1111-0000-0000-0000-000000000000"
        mock_t1.status = "interrupted"
        mock_t1.description = "test task"
        mock_t1.session_id = "sess-1"
        mock_t1.steps_completed = ["s1"]
        mock_dispatcher.list_interrupted.return_value = [mock_t1]
        sup._task_dispatcher = mock_dispatcher
        sup._handle_local_command("/recover", "chat-1", "msg-1")
        reply_text = sup.gateway.reply_message.call_args[0][1]
        assert "aaaa1111" in reply_text

    def _run_with_loop(self, sup, command):
        """Run a command with a real running event loop (for async handlers)."""
        import threading

        loop = asyncio.new_event_loop()
        sup._loop = loop
        t = threading.Thread(target=loop.run_forever, daemon=True)
        t.start()
        try:
            sup._handle_local_command(command, "chat-1", "msg-1")
        finally:
            loop.call_soon_threadsafe(loop.stop)
            t.join(timeout=5)
            loop.close()

    def test_recover_with_task_id_resume(self):
        """/recover <id> resume should call recover_task with mode='resume'."""
        sup = self._make_supervisor()
        mock_dispatcher = MagicMock()
        mock_task = MagicMock()
        mock_task.id = "aaaa1111-0000-0000-0000-000000000000"
        mock_task.status = "interrupted"
        mock_dispatcher.list_tasks.return_value = [mock_task]

        new_task = MagicMock()
        new_task.id = "bbbb2222-0000-0000-0000-000000000000"
        mock_dispatcher.recover_task = AsyncMock(return_value=new_task)
        sup._task_dispatcher = mock_dispatcher

        self._run_with_loop(sup, "/recover aaaa1111 resume")

        reply_text = sup.gateway.reply_message.call_args[0][1]
        assert "bbbb2222" in reply_text or "recover" in reply_text.lower()

    def test_recover_with_task_id_default_mode_is_resume(self):
        """/recover <id> without mode should default to 'resume'."""
        sup = self._make_supervisor()
        mock_dispatcher = MagicMock()
        mock_task = MagicMock()
        mock_task.id = "aaaa1111-0000-0000-0000-000000000000"
        mock_task.status = "interrupted"
        mock_dispatcher.list_tasks.return_value = [mock_task]

        new_task = MagicMock()
        new_task.id = "bbbb2222-0000-0000-0000-000000000000"
        mock_dispatcher.recover_task = AsyncMock(return_value=new_task)
        sup._task_dispatcher = mock_dispatcher

        self._run_with_loop(sup, "/recover aaaa1111")

        mock_dispatcher.recover_task.assert_called_once()
        call_args = mock_dispatcher.recover_task.call_args
        assert call_args[1].get("mode", call_args[0][1] if len(call_args[0]) > 1 else "resume") == "resume"

    def test_recover_dismiss_mode(self):
        """/recover <id> dismiss should dismiss the task."""
        sup = self._make_supervisor()
        mock_dispatcher = MagicMock()
        mock_task = MagicMock()
        mock_task.id = "aaaa1111-0000-0000-0000-000000000000"
        mock_task.status = "interrupted"
        mock_dispatcher.list_tasks.return_value = [mock_task]
        mock_dispatcher.recover_task = AsyncMock(return_value=None)
        sup._task_dispatcher = mock_dispatcher

        self._run_with_loop(sup, "/recover aaaa1111 dismiss")

        reply_text = sup.gateway.reply_message.call_args[0][1]
        assert "dismiss" in reply_text.lower() or "failed" in reply_text.lower()

    def test_recover_retry_mode(self):
        """/recover <id> retry should call recover_task with mode='retry'."""
        sup = self._make_supervisor()
        mock_dispatcher = MagicMock()
        mock_task = MagicMock()
        mock_task.id = "aaaa1111-0000-0000-0000-000000000000"
        mock_task.status = "interrupted"
        mock_task.session_id = ""
        mock_dispatcher.list_tasks.return_value = [mock_task]

        new_task = MagicMock()
        new_task.id = "cccc3333-0000-0000-0000-000000000000"
        mock_dispatcher.recover_task = AsyncMock(return_value=new_task)
        sup._task_dispatcher = mock_dispatcher

        self._run_with_loop(sup, "/recover aaaa1111 retry")

        reply_text = sup.gateway.reply_message.call_args[0][1]
        assert "cccc3333" in reply_text
        assert "retry" in reply_text.lower()

    def test_recover_resume_fallback_to_retry_message(self):
        """/recover <id> resume with no session should show fallback message."""
        sup = self._make_supervisor()
        mock_dispatcher = MagicMock()
        mock_task = MagicMock()
        mock_task.id = "aaaa1111-0000-0000-0000-000000000000"
        mock_task.status = "interrupted"
        mock_task.session_id = ""  # No session — resume will fall back to retry
        mock_dispatcher.list_tasks.return_value = [mock_task]

        new_task = MagicMock()
        new_task.id = "dddd4444-0000-0000-0000-000000000000"
        mock_dispatcher.recover_task = AsyncMock(return_value=new_task)
        sup._task_dispatcher = mock_dispatcher

        self._run_with_loop(sup, "/recover aaaa1111 resume")

        reply_text = sup.gateway.reply_message.call_args[0][1]
        assert "retry" in reply_text.lower() or "no session" in reply_text.lower()

    def test_recover_invalid_mode_rejected(self):
        """/recover <id> badmode should return an error."""
        sup = self._make_supervisor()
        mock_dispatcher = MagicMock()
        mock_task = MagicMock()
        mock_task.id = "aaaa1111-0000-0000-0000-000000000000"
        mock_task.status = "interrupted"
        mock_dispatcher.list_tasks.return_value = [mock_task]
        sup._task_dispatcher = mock_dispatcher

        sup._handle_local_command("/recover aaaa1111 badmode", "chat-1", "msg-1")
        reply_text = sup.gateway.reply_message.call_args[0][1]
        assert "resume" in reply_text.lower() or "retry" in reply_text.lower() or "invalid" in reply_text.lower()

    def test_recover_nonexistent_task(self):
        """/recover <unknown_id> should return 'not found' error."""
        sup = self._make_supervisor()
        mock_dispatcher = MagicMock()
        mock_dispatcher.list_tasks.return_value = []
        sup._task_dispatcher = mock_dispatcher

        sup._handle_local_command("/recover unknown123", "chat-1", "msg-1")
        reply_text = sup.gateway.reply_message.call_args[0][1]
        assert "No task found" in reply_text or "not found" in reply_text.lower()

    def test_recover_in_help_text(self):
        """/recover should appear in /help output."""
        sup = self._make_supervisor()
        sup._handle_local_command("/help", "chat-1", "msg-1")
        reply_text = sup.gateway.reply_message.call_args[0][1]
        assert "/recover" in reply_text

    def test_recover_no_dispatcher(self):
        """/recover when dispatcher is not available."""
        sup = self._make_supervisor()
        sup._task_dispatcher = None
        sup._handle_local_command("/recover aaaa1111", "chat-1", "msg-1")
        reply_text = sup.gateway.reply_message.call_args[0][1]
        assert "not available" in reply_text.lower() or "dispatcher" in reply_text.lower()


# ══════════════════════════════════════════════════════════
# 2. Active-process states recovery on restart
# ══════════════════════════════════════════════════════════


class TestActiveProcessStateRecovery:
    """States with active processes (follow_up, learning) need special recovery on restart.

    When supervisor crashes while follow_up or learning is running,
    the subprocess is lost. These tasks should be recovered to a safe state.
    """

    def test_follow_up_marked_interrupted_on_restart(self, tasks_file):
        """follow_up status means a follow-up process was active — should be interrupted."""
        _write_tasks_json(tasks_file, {
            "t1": _make_task(
                id="t1",
                status="follow_up",
                result="Original task result",
                session_id="sess-123",
            ),
        })
        _load_tasks()

        assert _tasks["t1"].status == "interrupted"
        assert _tasks["t1"].result == "Original task result"
        assert _tasks["t1"].session_id == "sess-123"

    def test_learning_marked_interrupted_on_restart(self, tasks_file):
        """learning status means a learning process was active — should be interrupted."""
        _write_tasks_json(tasks_file, {
            "t1": _make_task(
                id="t1",
                status="learning",
                result="Task result before learning",
                session_id="sess-456",
            ),
        })
        _load_tasks()

        assert _tasks["t1"].status == "interrupted"

    def test_follow_up_preserves_original_result(self, tasks_file):
        """When follow_up is interrupted, the original task result must survive."""
        _write_tasks_json(tasks_file, {
            "t1": _make_task(
                id="t1",
                status="follow_up",
                result="Build completed: 3 files changed",
            ),
        })
        _load_tasks()

        assert _tasks["t1"].result == "Build completed: 3 files changed"

    def test_follow_up_session_id_preserved(self, tasks_file):
        """Session ID from follow_up task must be preserved for recovery."""
        sid = "aaaa-bbbb-cccc-dddd"
        _write_tasks_json(tasks_file, {
            "t1": _make_task(id="t1", status="follow_up", session_id=sid),
        })
        _load_tasks()

        assert _tasks["t1"].session_id == sid

    def test_mixed_active_states_on_restart(self, tasks_file):
        """Multiple active-process states all get correct recovery."""
        _write_tasks_json(tasks_file, {
            "running": _make_task(id="running", status="running"),
            "follow_up": _make_task(id="follow_up", status="follow_up"),
            "learning": _make_task(id="learning", status="learning"),
            "pending": _make_task(id="pending", status="pending"),
            "awaiting": _make_task(id="awaiting", status="awaiting_closure"),
        })
        _load_tasks()

        assert _tasks["running"].status == "interrupted"
        assert _tasks["follow_up"].status == "interrupted"
        assert _tasks["learning"].status == "interrupted"
        assert _tasks["pending"].status == "pending"
        assert _tasks["awaiting"].status == "awaiting_closure"


# ══════════════════════════════════════════════════════════
# 3. Checkpoint merge for follow_up/learning states
# ══════════════════════════════════════════════════════════


class TestCheckpointMergeActiveStates:
    """Checkpoint data should also be merged for follow_up/learning states."""

    def test_follow_up_checkpoint_merged(self, tasks_file, tmp_path):
        """follow_up task checkpoint data should be merged on restart."""
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()

        _write_tasks_json(tasks_file, {
            "t1": _make_task(
                id="t1",
                status="follow_up",
                steps_completed=["s1"],
                session_id="orig-session",
            ),
        })
        ckpt_data = {
            "task_id": "t1",
            "timestamp": time.time(),
            "steps_completed": ["s1", "s2", "s3"],
            "current_step": "s4",
            "partial_result": "follow-up partial result",
            "session_id": "orig-session",
        }
        (ckpt_dir / "t1.json").write_text(json.dumps(ckpt_data))

        original_ckpt = td._CHECKPOINT_DIR
        td._CHECKPOINT_DIR = ckpt_dir
        try:
            _load_tasks()
        finally:
            td._CHECKPOINT_DIR = original_ckpt

        t = _tasks["t1"]
        assert t.status == "interrupted"
        assert t.steps_completed == ["s1", "s2", "s3"]
        assert t.current_step == "s4"

    def test_learning_checkpoint_merged(self, tasks_file, tmp_path):
        """learning task checkpoint data should be merged on restart."""
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()

        _write_tasks_json(tasks_file, {
            "t1": _make_task(
                id="t1",
                status="learning",
                steps_completed=[],
                session_id="learn-sess",
            ),
        })
        ckpt_data = {
            "task_id": "t1",
            "timestamp": time.time(),
            "steps_completed": ["extract_patterns"],
            "current_step": "saving",
            "partial_result": "",
            "session_id": "learn-sess",
        }
        (ckpt_dir / "t1.json").write_text(json.dumps(ckpt_data))

        original_ckpt = td._CHECKPOINT_DIR
        td._CHECKPOINT_DIR = ckpt_dir
        try:
            _load_tasks()
        finally:
            td._CHECKPOINT_DIR = original_ckpt

        t = _tasks["t1"]
        assert t.status == "interrupted"
        assert t.steps_completed == ["extract_patterns"]


# ══════════════════════════════════════════════════════════
# 4. recover_task edge cases
# ══════════════════════════════════════════════════════════


class TestRecoverTaskEdgeCases:
    """Edge cases for recover_task() function."""

    def test_recover_dismiss_sets_failed(self, tasks_file):
        """Dismissing an interrupted task should mark it as failed."""
        task = _make_task(id="t1", status="interrupted", error="Supervisor restarted")
        _tasks["t1"] = task

        result = asyncio.run(
            recover_task("t1", mode="dismiss")
        )

        assert result is None
        assert _tasks["t1"].status == "failed"
        assert "Dismissed" in _tasks["t1"].error

    def test_recover_non_interrupted_task_raises(self, tasks_file):
        """Attempting to recover a non-interrupted task should raise ValueError."""
        task = _make_task(id="t1", status="awaiting_closure")
        _tasks["t1"] = task

        with pytest.raises(ValueError, match="not interrupted"):
            asyncio.run(
                recover_task("t1", mode="resume")
            )

    def test_recover_unknown_task_raises(self, tasks_file):
        """Attempting to recover a nonexistent task should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown task"):
            asyncio.run(
                recover_task("nonexistent-id", mode="resume")
            )

    def test_recover_dismiss_clears_checkpoint(self, tasks_file, tmp_path):
        """Dismissing should also clean up the checkpoint file."""
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        ckpt_file = ckpt_dir / "t1.json"
        ckpt_file.write_text('{"task_id": "t1"}')

        task = _make_task(id="t1", status="interrupted")
        _tasks["t1"] = task

        original_ckpt = td._CHECKPOINT_DIR
        td._CHECKPOINT_DIR = ckpt_dir
        try:
            asyncio.run(
                recover_task("t1", mode="dismiss")
            )
        finally:
            td._CHECKPOINT_DIR = original_ckpt

        assert not ckpt_file.exists()
