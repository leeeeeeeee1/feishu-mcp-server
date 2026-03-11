"""Task dispatching module for the Supervisor Hub.

Manages oneshot (run-to-completion) and daemon (persistent) tasks,
dispatching them via `claude -p` with concurrency control, automatic
daemon restart, and an experience-review loop.
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)

# ── Configuration from environment ──

SUPERVISOR_MAX_WORKERS = int(os.environ.get("SUPERVISOR_MAX_WORKERS", "3"))
SUPERVISOR_MAX_DAEMONS = int(os.environ.get("SUPERVISOR_MAX_DAEMONS", "2"))

# Phrases that hint Claude is asking for user input
_INPUT_PHRASES = ("please", "which", "should i", "confirm")


def _looks_like_needs_input(text: str) -> bool:
    """Heuristic: does the output look like it needs user input?"""
    if "?" not in text:
        return False
    lower = text.lower()
    return any(phrase in lower for phrase in _INPUT_PHRASES)


def _build_env() -> dict[str, str]:
    """Build environment for subprocess, removing CLAUDECODE to avoid nesting."""
    return {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}


def _build_cmd(
    prompt: str,
    session_id: Optional[str] = None,
) -> list[str]:
    """Build the claude CLI command for task execution (non-streaming)."""
    cmd = [
        "claude", "-p", prompt,
        "--model", "opus",
        "--effort", "max",
        "--permission-mode", "bypassPermissions",
        "--output-format", "json",
    ]
    if session_id:
        cmd.extend(["--resume", session_id])
    return cmd


def _build_cmd_streaming(
    prompt: str,
    session_id: Optional[str] = None,
) -> list[str]:
    """Build the claude CLI command with stream-json output for progress tracking."""
    cmd = [
        "claude", "-p", prompt,
        "--model", "opus",
        "--effort", "max",
        "--permission-mode", "bypassPermissions",
        "--output-format", "stream-json",
        "--verbose",
    ]
    if session_id:
        cmd.extend(["--resume", session_id])
    return cmd


# ── Task dataclass ──


@dataclass
class Task:
    """Represents a dispatched task."""
    id: str
    prompt: str
    task_type: str  # "oneshot" or "daemon"
    status: str  # pending, running, waiting_for_input, done, awaiting_closure, follow_up, review, learning, completed, failed, cancelled
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
_worker_semaphore: asyncio.Semaphore | None = None
_daemon_semaphore: asyncio.Semaphore | None = None
_background_handles: dict[str, asyncio.Task] = {}

# ── Task persistence ──

_TASKS_FILE = Path(os.environ.get("SUPERVISOR_TASKS_FILE", "/tmp/supervisor-tasks.json"))


def _save_tasks() -> None:
    """Persist all tasks to disk."""
    try:
        data = {tid: asdict(t) for tid, t in _tasks.items()}
        tmp = _TASKS_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, default=str))
        tmp.rename(_TASKS_FILE)
    except Exception as e:
        logger.error("Failed to save tasks: %s", e)


def _load_tasks() -> None:
    """Load tasks from disk on startup."""
    if not _TASKS_FILE.exists():
        return
    try:
        data = json.loads(_TASKS_FILE.read_text())
        for tid, d in data.items():
            # Skip completed/cancelled tasks older than 24h
            if d.get("status") in ("completed", "cancelled"):
                if time.time() - d.get("finished_at", 0) > 86400:
                    continue
            task = Task(**{k: v for k, v in d.items() if k in Task.__dataclass_fields__})
            # Tasks that were running when we crashed → mark as failed
            if task.status in ("running", "pending"):
                task.status = "failed"
                task.error = "Supervisor restarted while task was in progress"
                task.finished_at = time.time()
            _tasks[tid] = task
        logger.info("Loaded %d tasks from disk", len(_tasks))
    except Exception as e:
        logger.error("Failed to load tasks: %s", e)


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
    old = task.status
    task.status = new_status
    logger.info("Task %s: %s -> %s", task.id[:8], old, new_status)
    _save_tasks()


# ── Core subprocess runner ──


async def _run_claude(task: Task) -> None:
    """Execute `claude -p` for *task* using streaming to track progress."""
    cmd = _build_cmd_streaming(task.prompt, session_id=task.session_id or None)
    env = _build_env()

    _set_status(task, "running")
    task.current_step = "Starting Claude..."
    task.started_at = time.time()

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            cwd=task.cwd or None,
        )
    except FileNotFoundError:
        task.error = "'claude' command not found"
        task.finished_at = time.time()
        _set_status(task, "failed")
        return
    except Exception as exc:  # noqa: BLE001
        task.error = str(exc)
        task.finished_at = time.time()
        _set_status(task, "failed")
        return

    # Stream stdout to track progress in real-time
    accumulated_text = ""
    try:
        async for raw_line in proc.stdout:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            event_type = data.get("type", "")

            # Capture session ID
            sid = data.get("session_id", "")
            if sid and not task.session_id:
                task.session_id = sid

            if event_type == "assistant":
                msg = data.get("message", {})
                if not isinstance(msg, dict):
                    continue
                for block in msg.get("content", []):
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "tool_use":
                        tool_name = block.get("name", "unknown")
                        tool_input = str(block.get("input", {}))[:80]
                        step_desc = f"{tool_name}: {tool_input}"
                        task.current_step = step_desc
                        task.steps_completed.append(step_desc)
                    elif block.get("type") == "text":
                        text = block.get("text", "")
                        if text:
                            accumulated_text = text

            elif event_type == "result":
                task.result = data.get("result", "") or accumulated_text
                sid = data.get("session_id", "")
                if sid:
                    task.session_id = sid
                break

    except Exception as exc:  # noqa: BLE001
        task.error = str(exc)
        task.finished_at = time.time()
        _set_status(task, "failed")
        return

    await proc.wait()

    if proc.returncode != 0:
        stderr_data = await proc.stderr.read()
        task.error = stderr_data.decode("utf-8", errors="replace").strip() or "unknown error"
        task.finished_at = time.time()
        _set_status(task, "failed")
        return

    if not task.result:
        task.result = accumulated_text or "(empty response)"

    task.finished_at = time.time()
    task.current_step = "Finished"

    # Check if Claude is asking for input
    if _looks_like_needs_input(task.result):
        _set_status(task, "waiting_for_input")
        task.current_step = "Waiting for user input"
    else:
        _set_status(task, "awaiting_closure")
        task.current_step = "Done — awaiting user confirmation to close"


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
) -> Task:
    """Create and launch a new task.

    Args:
        prompt: The user's request text.
        cwd: Working directory for the subprocess.
        task_type: "oneshot" or "daemon".
        chat_id: Optional Feishu chat ID for notifications.
        on_complete: Optional callback(task) called when task finishes,
                     fails, or needs user input.

    Returns:
        The newly created Task (already scheduled in the background).
    """
    task = Task(
        id=str(uuid4()),
        prompt=prompt,
        task_type=task_type,
        status="pending",
        description=_generate_description(prompt),
        current_step="Waiting in queue",
        cwd=cwd or "",
        created_at=time.time(),
    )
    _tasks[task.id] = task
    _save_tasks()
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
            env=env,
            cwd=task.cwd or None,
        )
        stdout, stderr = await proc.communicate()
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
    task = _tasks.get(task_id)
    if task is None:
        raise ValueError(f"Unknown task: {task_id}")
    if task.status != "awaiting_closure":
        raise ValueError(
            f"Task {task_id[:8]} is not awaiting closure (status={task.status})"
        )
    if not task.session_id:
        raise ValueError(f"Task {task_id[:8]} has no session to resume")

    return task


async def follow_up_async(task_id: str, user_input: str) -> str:
    """Execute follow-up on a task's existing session.

    Returns the response text.
    """
    task = _validate_follow_up(task_id)

    _set_status(task, "follow_up")
    task.current_step = f"Follow-up: {user_input[:60]}"

    cmd = _build_cmd_streaming(user_input, session_id=task.session_id)
    env = _build_env()

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            cwd=task.cwd or None,
        )
    except Exception as exc:
        _set_status(task, "awaiting_closure")
        return f"Error: {exc}"

    accumulated_text = ""
    try:
        async for raw_line in proc.stdout:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            if data.get("type") == "assistant":
                msg = data.get("message", {})
                if isinstance(msg, dict):
                    for block in msg.get("content", []):
                        if isinstance(block, dict) and block.get("type") == "text":
                            text = block.get("text", "")
                            if text:
                                accumulated_text = text

            elif data.get("type") == "result":
                accumulated_text = data.get("result", "") or accumulated_text
                break
    except Exception as exc:
        _set_status(task, "awaiting_closure")
        return f"Error during follow-up: {exc}"

    await proc.wait()

    task.result = accumulated_text or "(empty response)"
    _set_status(task, "awaiting_closure")
    task.current_step = "Done — awaiting user confirmation to close"
    return task.result


def close_task(task_id: str) -> str:
    """Close a task that is in awaiting_closure state.

    Returns confirmation message.
    """
    task = _tasks.get(task_id)
    if task is None:
        raise ValueError(f"Unknown task: {task_id}")
    if task.status not in ("awaiting_closure", "done", "review"):
        raise ValueError(
            f"Task {task_id[:8]} cannot be closed (status={task.status})"
        )

    _set_status(task, "completed")
    task.finished_at = time.time()
    return f"Task {task_id[:8]} closed."


def get_awaiting_closure() -> list[Task]:
    """Return tasks in awaiting_closure status."""
    return [t for t in _tasks.values() if t.status == "awaiting_closure"]


def skip_review(task_id: str) -> str:
    """Skip the review step for a task, moving straight to completed.

    Returns:
        Confirmation message.
    """
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
    task = _tasks.get(task_id)
    if task is None:
        raise ValueError(f"Unknown task: {task_id}")
    return task


def list_tasks() -> list[Task]:
    """Return all tasks."""
    return list(_tasks.values())


def list_daemons() -> list[Task]:
    """Return only daemon tasks."""
    return [t for t in _tasks.values() if t.task_type == "daemon"]


def get_review_pending() -> list[Task]:
    """Return tasks in review or waiting_for_input status."""
    return [t for t in _tasks.values() if t.status in ("review", "waiting_for_input")]


def stop_daemon(task_id: str) -> str:
    """Stop a running daemon task.

    Returns:
        Confirmation message.
    """
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
    task = _tasks.get(task_id)
    if task is None:
        raise ValueError(f"Unknown task: {task_id}")

    _set_status(task, "cancelled")
    task.finished_at = time.time()

    handle = _background_handles.get(task_id)
    if handle and not handle.done():
        handle.cancel()

    return f"Task {task_id[:8]} cancelled."


# ── Text formatting for Feishu ──


def _status_icon(status: str) -> str:
    """Map task status to a short label."""
    icons = {
        "pending": "PENDING",
        "running": "RUNNING",
        "waiting_for_input": "WAITING",
        "done": "DONE",
        "awaiting_closure": "AWAIT_CLOSE",
        "follow_up": "FOLLOW_UP",
        "review": "REVIEW",
        "learning": "LEARNING",
        "completed": "COMPLETED",
        "failed": "FAILED",
        "cancelled": "CANCELLED",
    }
    return icons.get(status, status.upper())


def _elapsed_str(task: Task) -> str:
    """Human-readable elapsed time."""
    if not task.started_at:
        return ""
    end = task.finished_at or time.time()
    secs = int(end - task.started_at)
    if secs < 60:
        return f"{secs}s"
    mins = secs // 60
    remaining = secs % 60
    return f"{mins}m{remaining}s"


def _format_task(task: Task) -> str:
    """Format a single task with description, progress, and current step."""
    lines: list[str] = []

    # Header: status + id + type + elapsed
    elapsed = _elapsed_str(task)
    elapsed_part = f" | {elapsed}" if elapsed else ""
    lines.append(
        f"[{_status_icon(task.status)}] {task.id[:8]} | {task.task_type}{elapsed_part}"
    )

    # Description
    desc = task.description or task.prompt[:60]
    lines.append(f"  Desc: {desc}")

    # Current step (for running tasks)
    if task.status == "running" and task.current_step:
        lines.append(f"  Step: {task.current_step}")

    # Progress: how many steps completed
    if task.steps_completed:
        step_count = len(task.steps_completed)
        last_step = task.steps_completed[-1]
        if len(last_step) > 60:
            last_step = last_step[:57] + "..."
        lines.append(f"  Progress: {step_count} steps | last: {last_step}")

    # Result or error snippet for finished tasks
    if task.status in ("done", "completed", "failed"):
        snippet = (task.result or task.error or "")[:100]
        if len(task.result or task.error or "") > 100:
            snippet += "..."
        if snippet:
            label = "Error" if task.status == "failed" else "Result"
            lines.append(f"  {label}: {snippet}")

    return "\n".join(lines)


def get_tasks_text() -> str:
    """Formatted summary of all tasks for Feishu."""
    if not _tasks:
        return "No tasks."

    running = [t for t in _tasks.values() if t.status == "running"]
    pending = [t for t in _tasks.values() if t.status == "pending"]
    finished = [t for t in _tasks.values() if t.status not in ("running", "pending")]

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
    global _worker_semaphore, _daemon_semaphore
    _tasks.clear()
    for h in _background_handles.values():
        if not h.done():
            h.cancel()
    _background_handles.clear()
    _worker_semaphore = None
    _daemon_semaphore = None
