"""Tests for task state recovery after supervisor restart.

Verifies that _load_tasks / _save_tasks correctly persists and restores
task state across supervisor restarts, including:
- Round-trip fidelity of all Task fields
- Running tasks marked as interrupted (not failed) on restart
- Pending tasks preserved as pending on restart
- Completed/cancelled tasks preserved or pruned by age
- Session IDs preserved for follow-up resumption
- Daemon retry counters preserved
- Atomic write safety (.tmp file handling)
- Corrupted/missing file handling
"""

import json
import time
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import pytest

from supervisor.task_dispatcher import (
    Task,
    _load_tasks,
    _save_tasks,
    _reset,
    _tasks,
    list_interrupted,
)
import supervisor.task_dispatcher as td


# ── Helpers ──


def _make_task(**overrides) -> Task:
    """Create a Task with sensible defaults, overriding specific fields."""
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
    """Write tasks dict to a JSON file (simulating _save_tasks output)."""
    data = {tid: asdict(t) for tid, t in tasks.items()}
    tasks_file.write_text(json.dumps(data, ensure_ascii=False, default=str))


def _swap_tasks_file(tmp_path: Path):
    """Context manager-like setup: swap _TASKS_FILE to tmp_path, clear _tasks."""
    tasks_file = tmp_path / "tasks.json"
    original = td._TASKS_FILE
    td._TASKS_FILE = tasks_file
    _tasks.clear()
    return tasks_file, original


def _restore_tasks_file(original: Path):
    """Restore original _TASKS_FILE and clear _tasks."""
    td._TASKS_FILE = original
    _tasks.clear()


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
    f, original = _swap_tasks_file(tmp_path)
    yield f
    _restore_tasks_file(original)


# ── Round-trip fidelity ──


class TestSaveLoadRoundTrip:
    """_save_tasks then _load_tasks should preserve all Task fields exactly."""

    def test_all_fields_preserved(self, tasks_file):
        task = _make_task(
            status="awaiting_closure",
            result="some result text",
            steps_completed=["init", "build", "test"],
            retries=2,
            finished_at=2000.0,
        )
        _tasks[task.id] = task
        _save_tasks()
        _tasks.clear()

        _load_tasks()
        restored = _tasks[task.id]

        assert restored.id == task.id
        assert restored.prompt == task.prompt
        assert restored.task_type == task.task_type
        assert restored.status == task.status
        assert restored.description == task.description
        assert restored.current_step == task.current_step
        assert restored.steps_completed == task.steps_completed
        assert restored.session_id == task.session_id
        assert restored.cwd == task.cwd
        assert restored.result == task.result
        assert restored.error == task.error
        assert restored.created_at == task.created_at
        assert restored.started_at == task.started_at
        assert restored.finished_at == task.finished_at
        assert restored.retries == task.retries
        assert restored.max_retries == task.max_retries

    def test_multiple_tasks_round_trip(self, tasks_file):
        t1 = _make_task(id="task-001", status="completed", finished_at=time.time())
        t2 = _make_task(id="task-002", status="awaiting_closure", result="done")
        t3 = _make_task(id="task-003", status="failed", error="boom")
        _tasks["task-001"] = t1
        _tasks["task-002"] = t2
        _tasks["task-003"] = t3
        _save_tasks()
        _tasks.clear()

        _load_tasks()
        assert len(_tasks) == 3
        assert _tasks["task-001"].status == "completed"
        assert _tasks["task-002"].status == "awaiting_closure"
        assert _tasks["task-003"].status == "failed"

    def test_unicode_content_preserved(self, tasks_file):
        task = _make_task(
            status="awaiting_closure",
            prompt="分析代码结构并优化性能",
            description="分析代码",
            result="已完成代码分析，发现3个优化点",
        )
        _tasks[task.id] = task
        _save_tasks()
        _tasks.clear()

        _load_tasks()
        restored = _tasks[task.id]
        assert restored.prompt == "分析代码结构并优化性能"
        assert restored.result == "已完成代码分析，发现3个优化点"

    def test_empty_fields_preserved(self, tasks_file):
        task = _make_task(
            status="awaiting_closure",
            session_id="",
            result="",
            error="",
            cwd="",
            current_step="",
            steps_completed=[],
        )
        _tasks[task.id] = task
        _save_tasks()
        _tasks.clear()

        _load_tasks()
        restored = _tasks[task.id]
        assert restored.session_id == ""
        assert restored.result == ""
        assert restored.steps_completed == []


# ── Restart state transitions ──


class TestRestartStateTransitions:
    """Tasks in various states should be handled correctly on restart."""

    def test_running_task_marked_interrupted(self, tasks_file):
        _write_tasks_json(tasks_file, {
            "t1": _make_task(id="t1", status="running"),
        })
        _load_tasks()

        assert _tasks["t1"].status == "interrupted"
        assert "restarted" in _tasks["t1"].error.lower()
        assert _tasks["t1"].finished_at > 0
        # Session ID preserved for recovery
        assert _tasks["t1"].session_id != ""

    def test_pending_task_stays_pending(self, tasks_file):
        """Pending tasks (never started) should stay pending, not be marked failed."""
        _write_tasks_json(tasks_file, {
            "t1": _make_task(id="t1", status="pending", started_at=0.0),
        })
        _load_tasks()

        assert _tasks["t1"].status == "pending"
        assert _tasks["t1"].error == ""

    def test_awaiting_closure_preserved(self, tasks_file):
        """Tasks in awaiting_closure should NOT be marked failed — they completed."""
        _write_tasks_json(tasks_file, {
            "t1": _make_task(
                id="t1",
                status="awaiting_closure",
                result="Task completed successfully",
                finished_at=time.time(),
            ),
        })
        _load_tasks()

        assert _tasks["t1"].status == "awaiting_closure"
        assert _tasks["t1"].result == "Task completed successfully"

    def test_waiting_for_input_preserved(self, tasks_file):
        """Tasks waiting for input should NOT be marked failed."""
        _write_tasks_json(tasks_file, {
            "t1": _make_task(
                id="t1",
                status="waiting_for_input",
                result="Which file should I modify?",
            ),
        })
        _load_tasks()

        assert _tasks["t1"].status == "waiting_for_input"
        assert _tasks["t1"].result == "Which file should I modify?"

    def test_done_status_preserved(self, tasks_file):
        _write_tasks_json(tasks_file, {
            "t1": _make_task(id="t1", status="done", finished_at=time.time()),
        })
        _load_tasks()
        assert _tasks["t1"].status == "done"

    def test_review_status_preserved(self, tasks_file):
        _write_tasks_json(tasks_file, {
            "t1": _make_task(id="t1", status="review"),
        })
        _load_tasks()
        assert _tasks["t1"].status == "review"

    def test_failed_status_preserved_without_overwrite(self, tasks_file):
        """Already-failed tasks should keep their original error message."""
        _write_tasks_json(tasks_file, {
            "t1": _make_task(
                id="t1",
                status="failed",
                error="Original error: OOM killed",
                finished_at=1500.0,
            ),
        })
        _load_tasks()

        assert _tasks["t1"].status == "failed"
        assert _tasks["t1"].error == "Original error: OOM killed"
        assert _tasks["t1"].finished_at == 1500.0

    def test_cancelled_status_preserved(self, tasks_file):
        _write_tasks_json(tasks_file, {
            "t1": _make_task(
                id="t1",
                status="cancelled",
                finished_at=time.time(),
            ),
        })
        _load_tasks()
        assert _tasks["t1"].status == "cancelled"

    def test_mixed_states_on_restart(self, tasks_file):
        """Comprehensive: multiple tasks in different states at crash time."""
        now = time.time()
        _write_tasks_json(tasks_file, {
            "running": _make_task(id="running", status="running"),
            "pending": _make_task(id="pending", status="pending"),
            "awaiting": _make_task(id="awaiting", status="awaiting_closure", finished_at=now),
            "input": _make_task(id="input", status="waiting_for_input"),
            "done": _make_task(id="done", status="done", finished_at=now),
            "failed": _make_task(id="failed", status="failed", error="prev error"),
            "completed": _make_task(id="completed", status="completed", finished_at=now),
        })
        _load_tasks()

        assert _tasks["running"].status == "interrupted"  # not failed
        assert _tasks["pending"].status == "pending"  # not failed
        assert _tasks["awaiting"].status == "awaiting_closure"
        assert _tasks["input"].status == "waiting_for_input"
        assert _tasks["done"].status == "done"
        assert _tasks["failed"].status == "failed"
        assert _tasks["failed"].error == "prev error"  # not overwritten
        assert _tasks["completed"].status == "completed"


# ── Session ID preservation for follow-up ──


class TestSessionIdPreservation:
    """Session IDs must survive restart so follow-up can resume the session."""

    def test_session_id_preserved_on_restart(self, tasks_file):
        sid = "550e8400-e29b-41d4-a716-446655440000"
        _write_tasks_json(tasks_file, {
            "t1": _make_task(
                id="t1",
                status="awaiting_closure",
                session_id=sid,
                result="Done building",
                finished_at=time.time(),
            ),
        })
        _load_tasks()

        assert _tasks["t1"].session_id == sid

    def test_session_id_preserved_on_interrupted_running_task(self, tasks_file):
        """Interrupted tasks should keep their session_id for recovery."""
        sid = "660e8400-e29b-41d4-a716-446655440000"
        _write_tasks_json(tasks_file, {
            "t1": _make_task(id="t1", status="running", session_id=sid),
        })
        _load_tasks()

        assert _tasks["t1"].status == "interrupted"
        assert _tasks["t1"].session_id == sid

    def test_empty_session_id_no_crash(self, tasks_file):
        _write_tasks_json(tasks_file, {
            "t1": _make_task(id="t1", status="awaiting_closure", session_id=""),
        })
        _load_tasks()
        assert _tasks["t1"].session_id == ""


# ── Task age pruning ──


class TestTaskAgePruning:
    """Old completed/cancelled tasks should be pruned on load."""

    def test_old_completed_task_pruned(self, tasks_file):
        """Completed tasks older than 24h should be discarded."""
        old_time = time.time() - 90000  # 25 hours ago
        _write_tasks_json(tasks_file, {
            "old": _make_task(id="old", status="completed", finished_at=old_time),
        })
        _load_tasks()
        assert "old" not in _tasks

    def test_recent_completed_task_kept(self, tasks_file):
        """Completed tasks within 24h should be kept."""
        recent_time = time.time() - 3600  # 1 hour ago
        _write_tasks_json(tasks_file, {
            "recent": _make_task(id="recent", status="completed", finished_at=recent_time),
        })
        _load_tasks()
        assert "recent" in _tasks

    def test_old_cancelled_task_pruned(self, tasks_file):
        old_time = time.time() - 90000
        _write_tasks_json(tasks_file, {
            "old": _make_task(id="old", status="cancelled", finished_at=old_time),
        })
        _load_tasks()
        assert "old" not in _tasks

    def test_recent_cancelled_task_kept(self, tasks_file):
        recent_time = time.time() - 3600
        _write_tasks_json(tasks_file, {
            "recent": _make_task(id="recent", status="cancelled", finished_at=recent_time),
        })
        _load_tasks()
        assert "recent" in _tasks

    def test_old_running_task_not_pruned(self, tasks_file):
        """Non-completed/cancelled tasks should NOT be pruned regardless of age."""
        _write_tasks_json(tasks_file, {
            "old_running": _make_task(id="old_running", status="running", created_at=100.0),
        })
        _load_tasks()
        # Should exist (marked interrupted, but not pruned)
        assert "old_running" in _tasks
        assert _tasks["old_running"].status == "interrupted"

    def test_old_awaiting_closure_not_pruned(self, tasks_file):
        """awaiting_closure tasks should never be pruned — user hasn't seen result."""
        old_time = time.time() - 90000
        _write_tasks_json(tasks_file, {
            "old_await": _make_task(
                id="old_await",
                status="awaiting_closure",
                finished_at=old_time,
            ),
        })
        _load_tasks()
        assert "old_await" in _tasks
        assert _tasks["old_await"].status == "awaiting_closure"

    def test_mixed_age_pruning(self, tasks_file):
        now = time.time()
        _write_tasks_json(tasks_file, {
            "old_done": _make_task(id="old_done", status="completed", finished_at=now - 90000),
            "new_done": _make_task(id="new_done", status="completed", finished_at=now - 3600),
            "await": _make_task(id="await", status="awaiting_closure", finished_at=now - 90000),
            "old_cancelled": _make_task(id="old_cancelled", status="cancelled", finished_at=now - 90000),
        })
        _load_tasks()

        assert "old_done" not in _tasks
        assert "new_done" in _tasks
        assert "await" in _tasks
        assert "old_cancelled" not in _tasks


# ── Daemon retry counter ──


class TestDaemonRetryPreservation:
    """Daemon retry state must persist across restarts."""

    def test_retry_count_preserved(self, tasks_file):
        _write_tasks_json(tasks_file, {
            "d1": _make_task(
                id="d1",
                task_type="daemon",
                status="awaiting_closure",
                retries=2,
                max_retries=5,
            ),
        })
        _load_tasks()

        assert _tasks["d1"].retries == 2
        assert _tasks["d1"].max_retries == 5

    def test_running_daemon_marked_interrupted_retries_preserved(self, tasks_file):
        """A running daemon at crash time keeps its retry count."""
        _write_tasks_json(tasks_file, {
            "d1": _make_task(
                id="d1",
                task_type="daemon",
                status="running",
                retries=1,
                max_retries=3,
            ),
        })
        _load_tasks()

        assert _tasks["d1"].status == "interrupted"
        assert _tasks["d1"].retries == 1  # Not reset


# ── Atomic write safety ──


class TestAtomicWriteSafety:
    """Test .tmp file handling for crash-safe persistence."""

    def test_orphaned_tmp_recovered_when_main_missing(self, tasks_file):
        """If only .tmp exists (crash during rename), recover from it."""
        tmp_file = tasks_file.with_suffix(".tmp")
        _write_tasks_json(tmp_file, {
            "t1": _make_task(id="t1", status="awaiting_closure"),
        })
        # Main file does not exist
        assert not tasks_file.exists()

        _load_tasks()

        assert tasks_file.exists()
        assert "t1" in _tasks

    def test_orphaned_tmp_deleted_when_main_exists(self, tasks_file):
        """If both .tmp and main exist, main wins and .tmp is deleted."""
        tmp_file = tasks_file.with_suffix(".tmp")
        # Main file has the real data
        _write_tasks_json(tasks_file, {
            "real": _make_task(id="real", status="awaiting_closure"),
        })
        # .tmp is stale/incomplete
        _write_tasks_json(tmp_file, {
            "stale": _make_task(id="stale", status="running"),
        })

        _load_tasks()

        assert "real" in _tasks
        assert "stale" not in _tasks
        assert not tmp_file.exists()

    def test_save_creates_atomic_file(self, tasks_file):
        """_save_tasks should write atomically (no partial writes)."""
        task = _make_task(id="t1", status="done")
        _tasks["t1"] = task
        _save_tasks()

        # File should be valid JSON
        data = json.loads(tasks_file.read_text())
        assert "t1" in data
        assert data["t1"]["status"] == "done"

        # No leftover .tmp
        assert not tasks_file.with_suffix(".tmp").exists()


# ── Edge cases ──


class TestEdgeCases:
    """Handle corrupted, empty, and missing files gracefully."""

    def test_missing_file_no_crash(self, tasks_file):
        """No tasks file → empty state, no error."""
        assert not tasks_file.exists()
        _load_tasks()
        assert len(_tasks) == 0

    def test_empty_json_object(self, tasks_file):
        tasks_file.write_text("{}")
        _load_tasks()
        assert len(_tasks) == 0

    def test_corrupted_json_no_crash(self, tasks_file):
        tasks_file.write_text("{{not valid json!!")
        _load_tasks()  # Should not raise
        assert len(_tasks) == 0

    def test_null_float_fields_coerced(self, tasks_file):
        """Float fields that are None in JSON should be coerced to 0.0."""
        data = {
            "t1": {
                "id": "t1",
                "prompt": "test",
                "task_type": "oneshot",
                "status": "awaiting_closure",
                "created_at": None,
                "started_at": None,
                "finished_at": None,
            }
        }
        tasks_file.write_text(json.dumps(data))
        _load_tasks()

        assert _tasks["t1"].created_at == 0.0
        assert _tasks["t1"].started_at == 0.0
        assert _tasks["t1"].finished_at == 0.0

    def test_extra_fields_in_json_ignored(self, tasks_file):
        """Unknown fields in JSON should not crash _load_tasks."""
        data = {
            "t1": {
                "id": "t1",
                "prompt": "test",
                "task_type": "oneshot",
                "status": "done",
                "created_at": 1000.0,
                "started_at": 1001.0,
                "finished_at": 1002.0,
                "unknown_field": "should be ignored",
                "another_extra": 42,
            }
        }
        tasks_file.write_text(json.dumps(data))
        _load_tasks()

        assert "t1" in _tasks
        assert _tasks["t1"].status == "done"

    def test_missing_optional_fields_use_defaults(self, tasks_file):
        """Minimal JSON (only required fields) should load with defaults."""
        data = {
            "t1": {
                "id": "t1",
                "prompt": "test",
                "task_type": "oneshot",
                "status": "done",
            }
        }
        tasks_file.write_text(json.dumps(data))
        _load_tasks()

        t = _tasks["t1"]
        assert t.description == ""
        assert t.session_id == ""
        assert t.retries == 0
        assert t.max_retries == 3
        assert t.steps_completed == []

    def test_completed_task_with_zero_finished_at_is_pruned(self, tasks_file):
        """Completed task with finished_at=0 (missing timestamp) gets pruned
        because time.time() - 0 > 86400 always."""
        data = {
            "t1": {
                "id": "t1",
                "prompt": "test",
                "task_type": "oneshot",
                "status": "completed",
                "finished_at": 0,
            }
        }
        tasks_file.write_text(json.dumps(data))
        _load_tasks()

        # finished_at=0 means age is ~50+ years, so it gets pruned
        assert "t1" not in _tasks


# ── Persistence triggered by status change ──


class TestPersistenceOnStatusChange:
    """_set_status should trigger _save_tasks automatically."""

    def test_set_status_persists(self, tasks_file):
        from supervisor.task_dispatcher import _set_status

        task = _make_task(id="t1", status="running")
        _tasks["t1"] = task

        _set_status(task, "done")

        # Read the file directly
        data = json.loads(tasks_file.read_text())
        assert data["t1"]["status"] == "done"

    def test_multiple_status_changes_all_persisted(self, tasks_file):
        from supervisor.task_dispatcher import _set_status

        task = _make_task(id="t1", status="pending")
        _tasks["t1"] = task

        _set_status(task, "running")
        _set_status(task, "done")
        _set_status(task, "awaiting_closure")

        data = json.loads(tasks_file.read_text())
        assert data["t1"]["status"] == "awaiting_closure"


# ── Checkpoint merge on restart ──


class TestCheckpointMerge:
    """_load_tasks merges checkpoint data for running tasks."""

    def test_checkpoint_steps_merged(self, tasks_file, tmp_path):
        """Checkpoint with more steps_completed should win over task's stale data."""
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()

        # Task saved with 1 step
        _write_tasks_json(tasks_file, {
            "t1": _make_task(
                id="t1",
                status="running",
                steps_completed=["step1"],
                current_step="step2",
                session_id="",
            ),
        })
        # Checkpoint has 3 steps (more recent progress)
        ckpt_data = {
            "task_id": "t1",
            "timestamp": time.time(),
            "steps_completed": ["step1", "step2", "step3"],
            "current_step": "step4",
            "partial_result": "partial output here",
            "session_id": "aaaa-bbbb-cccc-dddd",
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
        assert t.steps_completed == ["step1", "step2", "step3"]
        assert t.current_step == "step4"
        assert t.result == "partial output here"
        assert t.session_id == "aaaa-bbbb-cccc-dddd"

    def test_checkpoint_session_id_does_not_overwrite_existing(self, tasks_file, tmp_path):
        """If task already has session_id, checkpoint should not overwrite it."""
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()

        existing_sid = "1111-2222-3333-4444"
        _write_tasks_json(tasks_file, {
            "t1": _make_task(
                id="t1",
                status="running",
                session_id=existing_sid,
                steps_completed=[],
            ),
        })
        ckpt_data = {
            "task_id": "t1",
            "timestamp": time.time(),
            "steps_completed": [],
            "current_step": "",
            "partial_result": "",
            "session_id": "5555-6666-7777-8888",
        }
        (ckpt_dir / "t1.json").write_text(json.dumps(ckpt_data))

        original_ckpt = td._CHECKPOINT_DIR
        td._CHECKPOINT_DIR = ckpt_dir
        try:
            _load_tasks()
        finally:
            td._CHECKPOINT_DIR = original_ckpt

        assert _tasks["t1"].session_id == existing_sid

    def test_no_checkpoint_still_marks_interrupted(self, tasks_file, tmp_path):
        """Running task without checkpoint should still be interrupted."""
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()  # Empty dir — no checkpoint files

        _write_tasks_json(tasks_file, {
            "t1": _make_task(id="t1", status="running"),
        })

        original_ckpt = td._CHECKPOINT_DIR
        td._CHECKPOINT_DIR = ckpt_dir
        try:
            _load_tasks()
        finally:
            td._CHECKPOINT_DIR = original_ckpt

        assert _tasks["t1"].status == "interrupted"
        assert "restarted" in _tasks["t1"].error.lower()

    def test_checkpoint_fewer_steps_not_merged(self, tasks_file, tmp_path):
        """Checkpoint with fewer steps should not overwrite task's data."""
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()

        _write_tasks_json(tasks_file, {
            "t1": _make_task(
                id="t1",
                status="running",
                steps_completed=["s1", "s2", "s3"],
            ),
        })
        ckpt_data = {
            "task_id": "t1",
            "timestamp": time.time(),
            "steps_completed": ["s1"],
            "current_step": "",
            "partial_result": "",
            "session_id": "",
        }
        (ckpt_dir / "t1.json").write_text(json.dumps(ckpt_data))

        original_ckpt = td._CHECKPOINT_DIR
        td._CHECKPOINT_DIR = ckpt_dir
        try:
            _load_tasks()
        finally:
            td._CHECKPOINT_DIR = original_ckpt

        assert _tasks["t1"].steps_completed == ["s1", "s2", "s3"]


# ── list_interrupted ──


class TestListInterrupted:
    """list_interrupted() should return only tasks with 'interrupted' status."""

    def test_returns_interrupted_tasks(self, tasks_file):
        _write_tasks_json(tasks_file, {
            "t1": _make_task(id="t1", status="running"),
            "t2": _make_task(id="t2", status="awaiting_closure"),
            "t3": _make_task(id="t3", status="pending"),
        })
        _load_tasks()

        interrupted = list_interrupted()
        ids = {t.id for t in interrupted}
        assert ids == {"t1"}

    def test_empty_when_no_interrupted(self, tasks_file):
        _write_tasks_json(tasks_file, {
            "t1": _make_task(id="t1", status="awaiting_closure"),
        })
        _load_tasks()

        assert list_interrupted() == []

    def test_multiple_interrupted(self, tasks_file):
        _write_tasks_json(tasks_file, {
            "t1": _make_task(id="t1", status="running"),
            "t2": _make_task(id="t2", status="running"),
        })
        _load_tasks()

        interrupted = list_interrupted()
        assert len(interrupted) == 2
