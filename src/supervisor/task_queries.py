"""Read-only query helpers for task state.

List functions return list snapshots (safe to iterate without the lock).
Individual Task objects are live references — callers must not hold
the lock while mutating them.
Imports singleton state from task_state to avoid circular dependencies.
"""

from __future__ import annotations

from .task_state import Task, _tasks, _tasks_lock
from .task_formatting import _format_task


# ── Single-task lookup ──


def get_task(task_id: str) -> Task:
    """Get a single task by ID."""
    with _tasks_lock:
        task = _tasks.get(task_id)
    if task is None:
        raise ValueError(f"Unknown task: {task_id}")
    return task


# ── List queries ──


def list_tasks() -> list[Task]:
    """Return a snapshot of all tasks (safe to iterate without lock)."""
    with _tasks_lock:
        return list(_tasks.values())


def list_daemons() -> list[Task]:
    """Return only daemon tasks."""
    with _tasks_lock:
        return [t for t in _tasks.values() if t.task_type == "daemon"]


def get_review_pending() -> list[Task]:
    """Return tasks in review or waiting_for_input status."""
    with _tasks_lock:
        return [t for t in _tasks.values() if t.status in ("review", "waiting_for_input")]


def list_interrupted() -> list[Task]:
    """Return tasks in interrupted status (recoverable after restart)."""
    with _tasks_lock:
        return [t for t in _tasks.values() if t.status == "interrupted"]


def get_awaiting_closure() -> list[Task]:
    """Return tasks in awaiting_closure status."""
    with _tasks_lock:
        return [t for t in _tasks.values() if t.status == "awaiting_closure"]


# ── Formatted text output ──


def get_tasks_text() -> str:
    """Formatted summary of all tasks for Feishu."""
    with _tasks_lock:
        if not _tasks:
            return "No tasks."
        snapshot = list(_tasks.values())

    running = [t for t in snapshot if t.status == "running"]
    pending = [t for t in snapshot if t.status == "pending"]
    finished = [t for t in snapshot if t.status not in ("running", "pending")]

    sections: list[str] = []
    sections.append(f"Tasks: {len(running)} running, {len(pending)} pending, {len(finished)} finished")
    sections.append("-" * 40)

    for task in running + pending + finished:
        sections.append(_format_task(task))

    return "\n".join(sections)


def get_daemons_text() -> str:
    """Formatted summary of daemon tasks for Feishu."""
    daemons = list_daemons()
    if not daemons:
        return "No daemon tasks."
    lines = [_format_task(t) for t in daemons]
    return "\n".join(lines)
