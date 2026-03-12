"""Task dispatching module for the Supervisor Hub.

Manages oneshot (run-to-completion) and daemon (persistent) tasks,
dispatching them via `claude -p` with concurrency control, automatic
daemon restart, and an experience-review loop.
"""

import asyncio
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional
from uuid import uuid4

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

# Re-export pattern matching functions for backward compatibility
from .patterns import (  # noqa: F401
    _INPUT_PHRASES,
    _looks_like_needs_input,
    _CLOSE_PHRASES,
    _CLOSE_PHRASES_SET,
    _CLOSE_FALSE_POSITIVES,
    _CLOSE_INTENT_PATTERNS,
    _contains_close_intent,
    _looks_like_close,
)


# Re-export subprocess functions for backward compatibility
from .subprocess_runner import (  # noqa: F401
    _build_env,
    _build_cmd,
    _build_cmd_streaming,
    _run_claude_streaming,
    _run_claude_non_streaming,
    _run_claude,
    _follow_up_streaming,
    _follow_up_non_streaming,
)


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

# ── Task persistence ──

_TASKS_FILE = Path(os.environ.get("SUPERVISOR_TASKS_FILE", "/tmp/supervisor-tasks.json"))

from .task_persistence import (  # noqa: E402, F401
    _ACTIVE_PROCESS_STATUSES,
    save_tasks as _save_tasks_impl,
    save_tasks_unlocked as _save_tasks_unlocked_impl,
    checkpoint_path as _checkpoint_path_impl,
    save_checkpoint as _save_checkpoint_impl,
    load_checkpoint as _load_checkpoint_impl,
    clear_checkpoint as _clear_checkpoint_impl,
    load_tasks as _load_tasks_impl,
)


def _save_tasks() -> None:
    """Persist all tasks to disk."""
    _save_tasks_impl(_tasks, _tasks_lock, _TASKS_FILE)


def _save_tasks_unlocked() -> None:
    """Persist all tasks to disk. Caller MUST hold _tasks_lock."""
    _save_tasks_unlocked_impl(_tasks, _TASKS_FILE)


def _checkpoint_path(task_id: str) -> Path:
    """Build a safe checkpoint file path."""
    return _checkpoint_path_impl(task_id, _CHECKPOINT_DIR)


def _save_checkpoint(task: "Task") -> None:
    """Save a checkpoint for a running task."""
    _save_checkpoint_impl(task, _CHECKPOINT_DIR)


def _load_checkpoint(task_id: str) -> dict | None:
    """Load checkpoint data for a task."""
    return _load_checkpoint_impl(task_id, _CHECKPOINT_DIR)


def _clear_checkpoint(task_id: str) -> None:
    """Remove checkpoint file after task completes normally."""
    _clear_checkpoint_impl(task_id, _CHECKPOINT_DIR)


def _load_tasks() -> None:
    """Load tasks from disk on startup."""
    _load_tasks_impl(_tasks, _tasks_lock, _TASKS_FILE, _CHECKPOINT_DIR, Task)


# Load on import
_load_tasks()


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


def _set_status(task: Task, new_status: str) -> None:
    with _tasks_lock:
        old = task.status
        task.status = new_status
        logger.info("Task %s: %s -> %s", task.id[:8], old, new_status)
        _save_tasks_unlocked()




# ── Worker wrappers (semaphore-gated) ──


async def _oneshot_worker(task: Task) -> None:
    """Run a oneshot task, gated by the worker semaphore."""
    sem = _get_worker_semaphore()
    async with sem:
        await _run_claude(task)


async def _daemon_worker(task: Task) -> None:
    """Run a daemon task with auto-restart on failure."""
    sem = _get_daemon_semaphore()
    async with sem:
        while True:
            await _run_claude(task)

            # If cancelled mid-run, stop
            if task.status == "cancelled":
                return

            # If completed normally, stop looping
            if task.status in ("done", "awaiting_closure", "waiting_for_input"):
                return

            # On failure, maybe restart
            if task.status == "failed" and task.retries < task.max_retries:
                task.retries += 1
                logger.info(
                    "Daemon %s failed, restarting (retry %d/%d)",
                    task.id[:8], task.retries, task.max_retries,
                )
                _set_status(task, "pending")
                continue

            # Exhausted retries or unknown status
            return


# ── Worker wrappers with completion callbacks ──


async def _oneshot_worker_with_callback(
    task: Task, on_complete: Optional[Callable] = None
) -> None:
    """Run a oneshot task, then invoke on_complete callback."""
    await _oneshot_worker(task)
    if on_complete:
        try:
            on_complete(task)
        except Exception as e:
            logger.error("on_complete callback failed for task %s: %s", task.id[:8], e)


async def _daemon_worker_with_callback(
    task: Task, on_complete: Optional[Callable] = None
) -> None:
    """Run a daemon task, then invoke on_complete callback."""
    await _daemon_worker(task)
    if on_complete:
        try:
            on_complete(task)
        except Exception as e:
            logger.error("on_complete callback failed for task %s: %s", task.id[:8], e)


# ── Public API ──


async def dispatch(
    prompt: str,
    cwd: Optional[str] = None,
    task_type: str = "oneshot",
    chat_id: Optional[str] = None,
    on_complete: Optional[Callable] = None,
    description: str = "",
    session_id: str = "",
) -> Task:
    """Create and launch a new task.

    Args:
        prompt: The user's request text.
        cwd: Working directory for the subprocess.
        task_type: "oneshot" or "daemon".
        chat_id: Optional Feishu chat ID for notifications.
        on_complete: Optional callback(task) called when task finishes,
                     fails, or needs user input.
        description: Optional short description. Auto-generated from prompt if empty.
        session_id: Optional session ID to resume an existing Claude session.

    Returns:
        The newly created Task (already scheduled in the background).
    """
    task = Task(
        id=str(uuid4()),
        prompt=prompt,
        task_type=task_type,
        status="pending",
        description=description or _generate_description(prompt),
        current_step="Waiting in queue",
        session_id=session_id,
        cwd=cwd or "",
        created_at=time.time(),
    )
    with _tasks_lock:
        _tasks[task.id] = task
        _save_tasks_unlocked()
    logger.info("Dispatched %s task %s: %s", task_type, task.id[:8], prompt[:80])

    if task_type == "daemon":
        handle = asyncio.create_task(_daemon_worker_with_callback(task, on_complete))
    else:
        handle = asyncio.create_task(_oneshot_worker_with_callback(task, on_complete))

    _background_handles[task.id] = handle
    return task


async def resume_task(task_id: str, user_input: str) -> str:
    """Resume a task that is waiting for user input.

    Args:
        task_id: The task ID to resume.
        user_input: The user's reply to the question.

    Returns:
        The result text from Claude.

    Raises:
        ValueError: If task is not in waiting_for_input status.
    """
    with _tasks_lock:
        task = _tasks.get(task_id)
        if task is None:
            raise ValueError(f"Unknown task: {task_id}")
        if task.status != "waiting_for_input":
            raise ValueError(f"Task {task_id} is not waiting for input (status={task.status})")

    cmd = _build_cmd(user_input, session_id=task.session_id)
    env = _build_env()

    _set_status(task, "running")

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=_STREAM_LIMIT,
            env=env,
            cwd=task.cwd or None,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=SUPERVISOR_TASK_TIMEOUT,
        )
    except asyncio.TimeoutError:
        task.error = f"Timed out after {SUPERVISOR_TASK_TIMEOUT}s"
        _set_status(task, "failed")
        return f"Error: {task.error}"
    except Exception as exc:  # noqa: BLE001
        task.error = str(exc)
        _set_status(task, "failed")
        return f"Error: {exc}"

    if proc.returncode != 0:
        err = stderr.decode("utf-8", errors="replace").strip()
        task.error = err
        _set_status(task, "failed")
        return f"Error: {err}"

    raw = stdout.decode("utf-8", errors="replace").strip()
    try:
        data = json.loads(raw)
        task.result = data.get("result", "") or raw
        task.session_id = data.get("session_id", task.session_id) or task.session_id
    except json.JSONDecodeError:
        task.result = raw

    if _looks_like_needs_input(task.result):
        _set_status(task, "waiting_for_input")
    else:
        _set_status(task, "done")

    return task.result


# ── Experience Review Loop ──


def submit_review(task_id: str, feedback: str) -> str:
    """Submit human review feedback for a completed task.

    Sets status to learning, stores feedback, then advances to completed.

    Returns:
        The feedback text (caller feeds this to Claude for skill extraction).
    """
    with _tasks_lock:
        task = _tasks.get(task_id)
        if task is None:
            raise ValueError(f"Unknown task: {task_id}")
        if task.status != "review":
            raise ValueError(f"Task {task_id} is not in review (status={task.status})")

    _set_status(task, "learning")
    _set_status(task, "completed")
    return feedback


def _validate_follow_up(task_id: str) -> "Task":
    """Validate that a task can receive a follow-up.

    Args:
        task_id: The task ID.

    Returns:
        The validated Task object.

    Raises:
        ValueError: If task not found, not awaiting closure, or has no session.
    """
    with _tasks_lock:
        task = _tasks.get(task_id)
    if task is None:
        raise ValueError(f"Unknown task: {task_id}")
    if task.status != "awaiting_closure":
        raise ValueError(
            f"Task {task_id[:8]} is not awaiting closure (status={task.status})"
        )
    if not task.session_id:
        raise ValueError(f"Task {task_id[:8]} has no session to resume")
    # claude --resume requires a valid UUID
    if task.session_id.count("-") != 4 or len(task.session_id) != 36:
        raise ValueError(
            f"Task {task_id[:8]} has invalid session_id: {task.session_id!r}"
        )

    return task


async def follow_up_async(task_id: str, user_input: str) -> str:
    """Execute follow-up on a task's existing session.

    Tries streaming first for progress tracking, falls back to non-streaming.
    Returns the response text.
    """
    task = _validate_follow_up(task_id)

    _set_status(task, "follow_up")
    task.current_step = f"Follow-up: {user_input[:60]}"
    env = _build_env()

    result = await _follow_up_streaming(task, user_input, env)
    if result is None:
        logger.info("Task %s follow-up: streaming failed, retrying non-streaming", task.id[:8])
        result = await _follow_up_non_streaming(task, user_input, env)

    task.result = result or "(empty response)"
    _set_status(task, "awaiting_closure")
    task.current_step = "Done — awaiting user confirmation to close"
    return task.result




def close_task(task_id: str) -> str:
    """Close a task that is in awaiting_closure state.

    Returns confirmation message.
    """
    with _tasks_lock:
        task = _tasks.get(task_id)
        if task is None:
            raise ValueError(f"Unknown task: {task_id}")
        if task.status not in ("awaiting_closure", "done", "review"):
            raise ValueError(
                f"Task {task_id[:8]} cannot be closed (status={task.status})"
            )
        task.finished_at = time.time()
        old = task.status
        task.status = "completed"
        logger.info("Task %s: %s -> completed", task.id[:8], old)
        _save_tasks_unlocked()
    return f"Task {task_id[:8]} closed."


def close_tasks(task_ids: list[str]) -> list[str]:
    """Close multiple tasks at once.

    Returns a list of result messages, one per task_id.
    Errors are reported per-task without stopping the batch.
    """
    results: list[str] = []
    for task_id in task_ids:
        try:
            results.append(close_task(task_id))
        except ValueError as e:
            results.append(f"Error: {e}")
    return results


def get_awaiting_closure() -> list[Task]:
    """Return tasks in awaiting_closure status."""
    with _tasks_lock:
        return [t for t in _tasks.values() if t.status == "awaiting_closure"]


def skip_review(task_id: str) -> str:
    """Skip the review step for a task, moving straight to completed.

    Returns:
        Confirmation message.
    """
    with _tasks_lock:
        task = _tasks.get(task_id)
        if task is None:
            raise ValueError(f"Unknown task: {task_id}")
        if task.status not in ("done", "review"):
            raise ValueError(f"Task {task_id} cannot skip review (status={task.status})")

    _set_status(task, "completed")
    return f"Task {task_id[:8]} marked as completed (review skipped)."


# ── Query helpers ──


def get_task(task_id: str) -> Task:
    """Get a single task by ID."""
    with _tasks_lock:
        task = _tasks.get(task_id)
    if task is None:
        raise ValueError(f"Unknown task: {task_id}")
    return task


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


async def recover_task(
    task_id: str,
    mode: str = "resume",
    on_complete: Optional[Callable] = None,
) -> Optional[Task]:
    """Recover an interrupted task.

    Args:
        task_id: The interrupted task ID.
        mode: Recovery strategy:
            - "resume": Re-dispatch using existing session_id (continues where it left off).
            - "retry": Re-dispatch from scratch (fresh session).
            - "dismiss": Mark as failed and do not retry.
        on_complete: Optional callback for the new task.

    Returns:
        The newly dispatched Task (for resume/retry), or None (for dismiss).

    Raises:
        ValueError: If task not found or not in interrupted status.
    """
    with _tasks_lock:
        task = _tasks.get(task_id)
        if task is None:
            raise ValueError(f"Unknown task: {task_id}")
        if task.status != "interrupted":
            raise ValueError(
                f"Task {task_id[:8]} is not interrupted (status={task.status})"
            )

    if mode == "dismiss":
        with _tasks_lock:
            task.status = "failed"
            task.error = "Dismissed by user after restart interruption"
            task.finished_at = time.time()
            _save_tasks_unlocked()
        _clear_checkpoint(task_id)
        logger.info("Task %s: interrupted → dismissed (failed)", task_id[:8])
        return None

    # Mark original as completed (superseded by recovery task)
    with _tasks_lock:
        task.status = "completed"
        task.finished_at = time.time()
        _save_tasks_unlocked()
    _clear_checkpoint(task_id)

    # Build the recovery dispatch
    if mode == "resume" and task.session_id:
        # Resume: use existing session, ask to continue
        new_task = await dispatch(
            prompt=f"Continue the previous task. Context: {task.prompt[:200]}",
            cwd=task.cwd,
            task_type=task.task_type,
            on_complete=on_complete,
            description=f"[resumed] {task.description}",
            session_id=task.session_id,
        )
    elif mode == "resume" and not task.session_id:
        # Resume requested but no session — fall back to retry
        logger.warning(
            "Task %s: resume requested but no session_id — falling back to retry",
            task_id[:8],
        )
        new_task = await dispatch(
            prompt=task.prompt,
            cwd=task.cwd,
            task_type=task.task_type,
            on_complete=on_complete,
            description=f"[retried] {task.description}",
        )
    else:
        # Retry: fresh start
        new_task = await dispatch(
            prompt=task.prompt,
            cwd=task.cwd,
            task_type=task.task_type,
            on_complete=on_complete,
            description=f"[retried] {task.description}",
        )

    logger.info(
        "Task %s: recovered via %s → new task %s",
        task_id[:8], mode, new_task.id[:8],
    )
    return new_task


def stop_daemon(task_id: str) -> str:
    """Stop a running daemon task.

    Returns:
        Confirmation message.
    """
    with _tasks_lock:
        task = _tasks.get(task_id)
        if task is None:
            raise ValueError(f"Unknown task: {task_id}")
        if task.task_type != "daemon":
            raise ValueError(f"Task {task_id} is not a daemon")

    _set_status(task, "cancelled")
    task.finished_at = time.time()

    handle = _background_handles.get(task_id)
    if handle and not handle.done():
        handle.cancel()

    return f"Daemon {task_id[:8]} stopped."


def cancel_task(task_id: str) -> str:
    """Cancel any task (oneshot or daemon).

    Returns:
        Confirmation message.
    """
    with _tasks_lock:
        task = _tasks.get(task_id)
        if task is None:
            raise ValueError(f"Unknown task: {task_id}")

    _set_status(task, "cancelled")
    task.finished_at = time.time()

    handle = _background_handles.get(task_id)
    if handle and not handle.done():
        handle.cancel()

    return f"Task {task_id[:8]} cancelled."


# Re-export formatting functions for backward compatibility
from .task_formatting import _status_icon, _elapsed_str, _format_task  # noqa: F401


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
