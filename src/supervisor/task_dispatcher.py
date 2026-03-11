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

# asyncio StreamReader default limit is 64KB — too small for claude stream-json
# which can emit single lines >64KB (e.g. large code blocks). 10MB is safe.
_STREAM_LIMIT = 10 * 1024 * 1024  # 10 MB

# Phrases that hint Claude is asking for user input
_INPUT_PHRASES = ("please", "which", "should i", "confirm")


def _looks_like_needs_input(text: str) -> bool:
    """Heuristic: does the output look like it needs user input?"""
    if "?" not in text:
        return False
    lower = text.lower()
    return any(phrase in lower for phrase in _INPUT_PHRASES)


# Phrases that indicate user acknowledges / wants to close
_CLOSE_PHRASES = (
    "好的", "收到", "ok", "谢谢", "thanks", "可以了", "没问题",
    "完成", "done", "lgtm", "不用了", "就这样", "👍", "thank you",
)


_CLOSE_PHRASES_SET = frozenset(p.lower() for p in _CLOSE_PHRASES)


import re as _re

# Technical nouns — if these appear near 关闭/关掉/结束, it's NOT task closure
_TECHNICAL_NOUNS = r"连接|端口|服务|进程|窗口|文件|通道|线程|循环|socket|server|session|db|数据库|nginx|redis"

_CLOSE_FALSE_POSITIVES = _re.compile(
    rf"关闭({_TECHNICAL_NOUNS})"
    rf"|关掉({_TECHNICAL_NOUNS})"
    rf"|结束({_TECHNICAL_NOUNS})"
    rf"|({_TECHNICAL_NOUNS})关掉"
    rf"|({_TECHNICAL_NOUNS})关闭"
    rf"|({_TECHNICAL_NOUNS})结束",
    _re.IGNORECASE,
)

_CLOSE_INTENT_PATTERNS = [
    _re.compile(r"关闭(了|吧|这个|那个)"),  # 关闭了/关闭吧/关闭这个/关闭那个
    _re.compile(r"关了"),
    _re.compile(r"关掉"),
    _re.compile(r"结束(吧|了|掉|这个|那个|任务)"),  # anchored: require suffix
    _re.compile(r"不用了"),
    _re.compile(r"完事了"),
    _re.compile(r"可以关了"),
    _re.compile(r"\bclose\b", _re.IGNORECASE),
    _re.compile(r"\bdone with it\b", _re.IGNORECASE),
]


def _contains_close_intent(text: str) -> bool:
    """Detect close intent in longer text (not just short phrases).

    Unlike _looks_like_close (≤10 char exact match), this works on any
    length text. Conservative: excludes technical phrases like "关闭连接".
    """
    if not text or not text.strip():
        return False
    if _CLOSE_FALSE_POSITIVES.search(text):
        return False
    return any(pat.search(text) for pat in _CLOSE_INTENT_PATTERNS)


def _looks_like_close(text: str) -> bool:
    """Heuristic: short acknowledgement → close intent, not follow-up.

    Uses exact match after normalization to avoid false positives like
    "not ok" or "I am done".
    """
    stripped = text.strip()
    if not stripped:
        return False
    # Questions are never close intent
    if "?" in stripped or "？" in stripped or "吗" in stripped:
        return False
    # Strip trailing punctuation for matching
    normalized = stripped.lower().rstrip("。.!！~，,").strip()
    # Short messages only (≤10 unicode chars)
    if len(normalized) > 10:
        return False
    # Exact match against known close phrases
    return normalized in _CLOSE_PHRASES_SET


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
    # Clean up orphaned .tmp file from interrupted save
    tmp = _TASKS_FILE.with_suffix(".tmp")
    if tmp.exists() and not _TASKS_FILE.exists():
        try:
            tmp.rename(_TASKS_FILE)
        except OSError:
            pass
    elif tmp.exists():
        try:
            tmp.unlink()
        except OSError:
            pass

    if not _TASKS_FILE.exists():
        return
    try:
        data = json.loads(_TASKS_FILE.read_text())
        for tid, d in data.items():
            # Skip completed/cancelled tasks older than 24h
            if d.get("status") in ("completed", "cancelled"):
                finished = d.get("finished_at") or 0
                if time.time() - finished > 86400:
                    continue
            # Coerce float fields that may be None from JSON
            filtered = {k: v for k, v in d.items() if k in Task.__dataclass_fields__}
            for float_field in ("created_at", "started_at", "finished_at"):
                if float_field in filtered and filtered[float_field] is None:
                    filtered[float_field] = 0.0
            task = Task(**filtered)
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


async def _run_claude_streaming(task: Task, env: dict) -> bool:
    """Try to execute task via streaming. Returns True on success, False if fallback needed."""
    cmd = _build_cmd_streaming(task.prompt, session_id=task.session_id or None)

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=_STREAM_LIMIT,
            env=env,
            cwd=task.cwd or None,
        )
    except FileNotFoundError:
        task.error = "'claude' command not found"
        return False
    except Exception as exc:  # noqa: BLE001
        task.error = str(exc)
        return False

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
        return False

    await proc.wait()

    if proc.returncode != 0:
        stderr_data = await proc.stderr.read()
        task.error = stderr_data.decode("utf-8", errors="replace").strip() or "unknown error"
        logger.warning(
            "Task %s streaming failed (code %d): %s",
            task.id[:8], proc.returncode, task.error[:300],
        )
        return False

    if not task.result:
        task.result = accumulated_text or ""

    return True


async def _run_claude_non_streaming(task: Task, env: dict) -> bool:
    """Execute task via non-streaming JSON mode. Returns True on success."""
    cmd = _build_cmd(task.prompt, session_id=task.session_id or None)
    task.current_step = "Running (non-streaming fallback)..."

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=_STREAM_LIMIT,
            env=env,
            cwd=task.cwd or None,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=600)
    except asyncio.TimeoutError:
        task.error = "Timed out (10 min limit)"
        return False
    except FileNotFoundError:
        task.error = "'claude' command not found"
        return False
    except Exception as exc:  # noqa: BLE001
        task.error = str(exc)
        return False

    if proc.returncode != 0:
        task.error = stderr.decode("utf-8", errors="replace").strip() or "unknown error"
        logger.error(
            "Task %s non-streaming failed (code %d): %s",
            task.id[:8], proc.returncode, task.error[:500],
        )
        return False

    try:
        data = json.loads(stdout.decode("utf-8"))
        task.result = data.get("result", "") or "(empty response)"
        sid = data.get("session_id", "")
        if sid:
            task.session_id = sid
    except (json.JSONDecodeError, TypeError):
        task.result = stdout.decode("utf-8", errors="replace").strip() or "(empty response)"

    return True


async def _run_claude(task: Task) -> None:
    """Execute `claude -p` for a task.

    Strategy: try streaming first (for progress tracking), fall back to
    non-streaming if streaming crashes (e.g. HTTP chunk size errors).
    """
    env = _build_env()

    _set_status(task, "running")
    task.current_step = "Starting Claude..."
    task.started_at = time.time()

    # Try streaming first
    success = await _run_claude_streaming(task, env)

    if not success:
        # Streaming failed — retry with non-streaming
        logger.info(
            "Task %s: streaming failed, retrying non-streaming. error=%s",
            task.id[:8], (task.error or "")[:200],
        )
        task.error = ""  # clear streaming error
        success = await _run_claude_non_streaming(task, env)

    if not success:
        task.finished_at = time.time()
        _set_status(task, "failed")
        return

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
    description: str = "",
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
            limit=_STREAM_LIMIT,
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


async def _follow_up_streaming(task: Task, user_input: str, env: dict) -> Optional[str]:
    """Try follow-up via streaming. Returns result text or None on failure."""
    cmd = _build_cmd_streaming(user_input, session_id=task.session_id)

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=_STREAM_LIMIT,
            env=env,
            cwd=task.cwd or None,
        )
    except Exception as exc:
        logger.warning("Follow-up streaming spawn failed: %s", exc)
        return None

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
        logger.warning("Follow-up streaming error: %s", exc)
        return None

    await proc.wait()
    if proc.returncode != 0:
        stderr_data = await proc.stderr.read()
        logger.warning(
            "Follow-up streaming failed (code %d): %s",
            proc.returncode, stderr_data.decode("utf-8", errors="replace")[:300],
        )
        return None

    return accumulated_text or "(empty response)"


async def _follow_up_non_streaming(task: Task, user_input: str, env: dict) -> Optional[str]:
    """Follow-up via non-streaming JSON mode (fallback)."""
    cmd = _build_cmd(user_input, session_id=task.session_id)

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=_STREAM_LIMIT,
            env=env,
            cwd=task.cwd or None,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=600)
    except Exception as exc:
        logger.error("Follow-up non-streaming failed: %s", exc)
        return f"Error: {exc}"

    if proc.returncode != 0:
        err = stderr.decode("utf-8", errors="replace").strip()
        logger.error("Follow-up non-streaming error (code %d): %s", proc.returncode, err[:300])
        return f"Error: {err}"

    try:
        data = json.loads(stdout.decode("utf-8"))
        return data.get("result", "") or "(empty response)"
    except (json.JSONDecodeError, TypeError):
        return stdout.decode("utf-8", errors="replace").strip() or "(empty response)"


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
    global _worker_semaphore, _daemon_semaphore, _TASKS_FILE
    _tasks.clear()
    for h in _background_handles.values():
        if not h.done():
            h.cancel()
    _background_handles.clear()
    _worker_semaphore = None
    _daemon_semaphore = None
    # Redirect persistence to a temp file to avoid polluting production data
    _TASKS_FILE = Path("/tmp/supervisor-tasks-test.json")
    _TASKS_FILE.unlink(missing_ok=True)
