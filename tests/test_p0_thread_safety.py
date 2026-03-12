"""Tests for P0-1: Thread-safe locking on _tasks dict.

These tests verify that concurrent access to the global _tasks dict
from multiple threads does not cause data corruption or races.
"""

import asyncio
import threading
import time
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from supervisor.task_dispatcher import (
    Task,
    _reset,
    dispatch,
    list_tasks,
    get_task,
    close_task,
    cancel_task,
    _tasks,
    _save_tasks,
)


@pytest.fixture(autouse=True)
def clean_state():
    """Reset dispatcher state before each test."""
    _reset()
    yield
    _reset()


class TestTasksLockExists:
    """Verify that a threading lock protects _tasks access."""

    def test_tasks_lock_is_exposed(self):
        """The module should expose a _tasks_lock for thread safety."""
        from supervisor import task_dispatcher
        assert hasattr(task_dispatcher, "_tasks_lock"), (
            "_tasks_lock must exist in task_dispatcher module"
        )

    def test_tasks_lock_is_threading_lock(self):
        """The lock should be a threading.Lock (not asyncio.Lock)."""
        from supervisor import task_dispatcher
        lock = getattr(task_dispatcher, "_tasks_lock", None)
        assert isinstance(lock, type(threading.Lock())), (
            "_tasks_lock must be a threading.Lock instance"
        )


class TestConcurrentReadWrite:
    """Verify concurrent reads and writes don't corrupt _tasks."""

    def test_concurrent_write_and_list(self):
        """Writing tasks from one thread while listing from another should not crash."""
        errors = []

        def write_thread():
            try:
                for i in range(50):
                    task = Task(
                        id=f"write-{i}-0000-0000-0000-000000000000",
                        prompt=f"task {i}",
                        task_type="oneshot",
                        status="pending",
                        created_at=time.time(),
                    )
                    _tasks[task.id] = task
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def list_thread():
            try:
                for _ in range(50):
                    _ = list_tasks()
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=write_thread)
        t2 = threading.Thread(target=list_thread)
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        assert not errors, f"Concurrent access errors: {errors}"

    def test_concurrent_close_and_iteration(self):
        """Closing tasks while iterating should not raise RuntimeError."""
        # Pre-populate tasks
        for i in range(10):
            task = Task(
                id=f"test-{i}-0000-0000-0000-000000000000",
                prompt=f"task {i}",
                task_type="oneshot",
                status="awaiting_closure",
                created_at=time.time(),
            )
            _tasks[task.id] = task

        errors = []

        def close_thread():
            try:
                for i in range(10):
                    tid = f"test-{i}-0000-0000-0000-000000000000"
                    try:
                        close_task(tid)
                    except ValueError:
                        pass
            except Exception as e:
                errors.append(e)

        def iterate_thread():
            try:
                for _ in range(20):
                    for t in list_tasks():
                        _ = t.status
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=close_thread)
        t2 = threading.Thread(target=iterate_thread)
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        assert not errors, f"Concurrent close+iterate errors: {errors}"


class TestSaveTasksUnderLock:
    """Verify _save_tasks doesn't deadlock when called under lock."""

    def test_save_tasks_does_not_deadlock(self):
        """_save_tasks called from within locked context should not deadlock."""
        task = Task(
            id="deadlock-test-0000-0000-000000000000",
            prompt="test",
            task_type="oneshot",
            status="pending",
            created_at=time.time(),
        )
        _tasks[task.id] = task

        # _save_tasks is called internally by _set_status which should acquire lock
        # This should complete within 5 seconds (no deadlock)
        done = threading.Event()

        def save_thread():
            _save_tasks()
            done.set()

        t = threading.Thread(target=save_thread)
        t.start()
        assert done.wait(timeout=5), "_save_tasks deadlocked"


class TestListTasksReturnsSnapshot:
    """list_tasks should return a snapshot, not a live reference."""

    def test_list_tasks_returns_copy(self):
        """Modifying the returned list should not affect internal state."""
        task = Task(
            id="snapshot-test-0000-0000-000000000000",
            prompt="test",
            task_type="oneshot",
            status="pending",
            created_at=time.time(),
        )
        _tasks[task.id] = task

        result = list_tasks()
        result.clear()

        # Internal state should be unchanged
        assert len(_tasks) == 1, "list_tasks must return a copy, not the internal dict values"
