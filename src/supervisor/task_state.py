"""Task state: dataclass, singleton containers, semaphores, and helpers.

Central module for task state that other modules (task_queries, task_dispatcher,
subprocess_runner) import from. Keeps all mutable state in one place.
"""

import asyncio
import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Configuration from environment ──

SUPERVISOR_MAX_WORKERS = int(os.environ.get("SUPERVISOR_MAX_WORKERS", "3"))
SUPERVISOR_MAX_DAEMONS = int(os.environ.get("SUPERVISOR_MAX_DAEMONS", "2"))
SUPERVISOR_TASK_TIMEOUT = int(os.environ.get("SUPERVISOR_TASK_TIMEOUT", "1800"))

# Checkpoint directory for crash recovery
_CHECKPOINT_DIR = Path(os.environ.get("SUPERVISOR_CHECKPOINT_DIR", "/tmp/supervisor-checkpoints"))

# asyncio StreamReader default limit is 64KB — too small for claude stream-json
# which can emit single lines >64KB (e.g. large code blocks). 10MB is safe.
_STREAM_LIMIT = 10 * 1024 * 1024  # 10 MB

# Task persistence file
_TASKS_FILE = Path(os.environ.get("SUPERVISOR_TASKS_FILE", "/tmp/supervisor-tasks.json"))


# ── Task dataclass ──


@dataclass
class Task:
    """Represents a dispatched task."""
    id: str
    prompt: str
    task_type: str  # "oneshot" or "daemon"
    status: str  # pending, running, waiting_for_input, done, awaiting_closure, follow_up, review, learning, completed, failed, interrupted, cancelled
    description: str = ""  # short human-readable summary (auto-generated from prompt)
    current_step: str = ""  # what the task is doing right now
    steps_completed: list = field(default_factory=list)  # list of completed step descriptions
    session_id: str = ""
    cwd: str = ""
    result: str = ""
    error: str = ""
    created_at: float = 0.0
    started_at: float = 0.0
    finished_at: float = 0.0
    retries: int = 0
    max_retries: int = 3


def _generate_description(prompt: str) -> str:
    """Generate a short description from the task prompt."""
    # Take the first line, truncated
    first_line = prompt.strip().split("\n")[0]
    if len(first_line) > 80:
        return first_line[:77] + "..."
    return first_line


# ── Dispatcher singleton state ──

_tasks: dict[str, Task] = {}
_tasks_lock = threading.Lock()  # Protects all reads/writes to _tasks
_worker_semaphore: asyncio.Semaphore | None = None
_daemon_semaphore: asyncio.Semaphore | None = None
_background_handles: dict[str, asyncio.Task] = {}


# ── Semaphore getters ──


def _get_worker_semaphore() -> asyncio.Semaphore:
    global _worker_semaphore
    if _worker_semaphore is None:
        _worker_semaphore = asyncio.Semaphore(SUPERVISOR_MAX_WORKERS)
    return _worker_semaphore


def _get_daemon_semaphore() -> asyncio.Semaphore:
    global _daemon_semaphore
    if _daemon_semaphore is None:
        _daemon_semaphore = asyncio.Semaphore(SUPERVISOR_MAX_DAEMONS)
    return _daemon_semaphore


# ── Status transition helper ──


def _set_status(task: Task, new_status: str) -> None:
    from .task_persistence import save_tasks_unlocked
    with _tasks_lock:
        old = task.status
        task.status = new_status
        logger.info("Task %s: %s -> %s", task.id[:8], old, new_status)
        save_tasks_unlocked(_tasks, _TASKS_FILE)


# ── Cleanup (for testing) ──


def _reset() -> None:
    """Reset all internal state. Used by tests only."""
    global _worker_semaphore, _daemon_semaphore, _TASKS_FILE, _CHECKPOINT_DIR
    with _tasks_lock:
        _tasks.clear()
    for h in _background_handles.values():
        if not h.done():
            h.cancel()
    _background_handles.clear()
    _worker_semaphore = None
    _daemon_semaphore = None
    # Redirect persistence to temp paths to avoid polluting production data
    _TASKS_FILE = Path("/tmp/supervisor-tasks-test.json")
    _TASKS_FILE.unlink(missing_ok=True)
    _CHECKPOINT_DIR = Path("/tmp/supervisor-checkpoints-test")
    _CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
