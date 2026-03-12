"""Local command handlers for Supervisor Hub.

Each function accepts a supervisor instance (sup) and an argument string,
returning the response text. Supervisor delegates /commands here.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Optional

logger = logging.getLogger(__name__)


def cmd_status(sup, _arg: str) -> str:
    parts: list[str] = []
    if sup._container_monitor:
        parts.append(sup._container_monitor.get_status_text())
    if sup._session_monitor:
        parts.append(sup._session_monitor.get_sessions_text())
    if sup._task_dispatcher:
        tasks_text = sup._task_dispatcher.get_tasks_text()
        if tasks_text != "No tasks.":
            parts.append(tasks_text)
    return "\n\n".join(parts) if parts else "Monitors not initialized yet."


def cmd_sessions(sup, _arg: str) -> str:
    if sup._session_monitor:
        return sup._session_monitor.get_sessions_text()
    return "Session monitor not available."


def cmd_tasks(sup, _arg: str) -> str:
    if sup._task_dispatcher:
        return sup._task_dispatcher.get_tasks_text()
    return "Task dispatcher not available."


def cmd_gpu(sup, _arg: str) -> str:
    if sup._container_monitor:
        return sup._container_monitor.get_gpu_text()
    return "Container monitor not available."


def cmd_daemons(sup, _arg: str) -> str:
    if sup._task_dispatcher:
        return sup._task_dispatcher.get_daemons_text()
    return "Task dispatcher not available."


def cmd_stop(sup, arg: str) -> str:
    if not arg:
        return "Usage: /stop <task_id>"
    if sup._task_dispatcher:
        return sup._task_dispatcher.stop_daemon(arg)
    return "Task dispatcher not available."


def cmd_skip(sup, arg: str) -> str:
    if not arg:
        return "Usage: /skip <task_id>"
    if sup._task_dispatcher:
        return sup._task_dispatcher.skip_review(arg)
    return "Task dispatcher not available."


def cmd_close(sup, arg: str) -> str:
    """Close one or more tasks. Supports: /close <id>, /close <id1> <id2>, /close all."""
    if not arg:
        return "Usage: /close <id1> [id2 ...] or /close all"
    if not sup._task_dispatcher:
        return "Task dispatcher not available."

    if arg.strip().lower() == "all":
        awaiting = sup._task_dispatcher.get_awaiting_closure()
        if not awaiting:
            return "No tasks awaiting closure."
        task_ids = [t.id for t in awaiting]
        results = sup._task_dispatcher.close_tasks(task_ids)
        return "\n".join(results)

    prefixes = arg.split()
    if len(prefixes) == 1:
        task = find_task_by_prefix(sup, prefixes[0])
        if isinstance(task, str):
            return task
        return sup._task_dispatcher.close_task(task.id)

    # Resolve all prefixes, preserving input order
    resolved: list[tuple[str | None, str | None]] = []
    for prefix in prefixes:
        task = find_task_by_prefix(sup, prefix)
        if isinstance(task, str):
            resolved.append((None, task))
        else:
            resolved.append((task.id, None))

    valid_ids = [tid for tid, _ in resolved if tid is not None]
    close_results = iter(
        sup._task_dispatcher.close_tasks(valid_ids) if valid_ids else []
    )
    output: list[str] = []
    for task_id, error in resolved:
        if error is not None:
            output.append(error)
        else:
            output.append(next(close_results))
    return "\n".join(output)


def cmd_followup(sup, arg: str) -> str:
    """Send a follow-up to a task awaiting closure."""
    parts = arg.split(maxsplit=1)
    if len(parts) < 2:
        return "Usage: /followup <task_id> <your question>"
    task_id_prefix, user_input = parts[0], parts[1]
    if not sup._task_dispatcher:
        return "Task dispatcher not available."
    task = find_task_by_prefix(sup, task_id_prefix)
    if isinstance(task, str):
        return task

    if task.status != "awaiting_closure":
        return f"Task {task.id[:8]} is not awaiting closure (status={task.status})"

    if sup._loop:
        future = asyncio.run_coroutine_threadsafe(
            sup._task_dispatcher.follow_up_async(task.id, user_input),
            sup._loop,
        )
        try:
            result = future.result(timeout=600)
            truncated = result[:3000] + ("..." if len(result) > 3000 else "")
            return f"📎 追问回复 [{task.id[:8]}]\n\n{truncated}\n\n回复 /close {task.id[:8]} 关闭，或继续 /followup {task.id[:8]} <问题>"
        except Exception as e:
            return f"Error: {e}"
    return "Event loop not available."


def cmd_reply(sup, arg: str) -> str:
    """Resume a task that is waiting for user input."""
    parts = arg.split(maxsplit=1)
    if len(parts) < 2:
        return "Usage: /reply <task_id> <your reply>"
    task_id_prefix, user_input = parts[0], parts[1]
    if not sup._task_dispatcher:
        return "Task dispatcher not available."
    task = find_task_by_prefix(sup, task_id_prefix)
    if isinstance(task, str):
        return task

    if task.status != "waiting_for_input":
        return f"Task {task.id[:8]} is not waiting for input (status={task.status})"

    if sup._loop:
        future = asyncio.run_coroutine_threadsafe(
            sup._task_dispatcher.resume_task(task.id, user_input),
            sup._loop,
        )
        try:
            result = future.result(timeout=600)
            return f"Task {task.id[:8]} resumed.\n\n{result[:3000]}"
        except Exception as e:
            return f"Error: {e}"
    return "Event loop not available."


def cmd_recover(sup, arg: str) -> str:
    """Recover interrupted tasks after supervisor restart."""
    if not sup._task_dispatcher:
        return "Task dispatcher not available."

    _VALID_MODES = ("resume", "retry", "dismiss")

    # No argument: list interrupted tasks
    if not arg.strip():
        interrupted = sup._task_dispatcher.list_interrupted()
        if not interrupted:
            return "No interrupted tasks to recover."
        lines = ["Interrupted tasks (recoverable after restart):\n"]
        for t in interrupted:
            resumable = "resumable" if t.session_id else "retryable"
            lines.append(
                f"  [{t.id[:8]}] {t.description or t.prompt[:60]} "
                f"({resumable}, {len(t.steps_completed)} steps)"
            )
        lines.append(
            "\nUsage: /recover <task_id> [resume|retry|dismiss]"
        )
        return "\n".join(lines)

    # Parse task_id and optional mode
    parts = arg.strip().split(maxsplit=1)
    task_id_prefix = parts[0]
    mode = parts[1].strip().lower() if len(parts) > 1 else "resume"

    if mode not in _VALID_MODES:
        return (
            f"Invalid mode '{mode}'. "
            f"Valid modes: resume, retry, dismiss"
        )

    task = find_task_by_prefix(sup, task_id_prefix)
    if isinstance(task, str):
        return task  # error message

    if task.status != "interrupted":
        return (
            f"Task {task.id[:8]} is not interrupted "
            f"(status={task.status}). Only interrupted tasks can be recovered."
        )

    if not sup._loop:
        return "Event loop not available."

    # Wire notification callback so user gets result when recovered task completes
    chat_id = getattr(sup, "_current_chat_id", None)

    def on_complete(finished_task, _cid=chat_id):
        if _cid:
            sup._notify_task_result(finished_task, _cid)

    try:
        future = asyncio.run_coroutine_threadsafe(
            sup._task_dispatcher.recover_task(
                task.id, mode=mode, on_complete=on_complete,
            ),
            sup._loop,
        )
        new_task = future.result(timeout=60)
    except Exception as e:
        return f"Recovery failed: {e}"

    if mode == "dismiss":
        return f"Task {task.id[:8]} dismissed (marked as failed)."

    if new_task is None:
        return f"Task {task.id[:8]} recovered via {mode} but no new task was created."

    # Surface resume→retry fallback when no session was available
    actual_mode = mode
    if mode == "resume" and not task.session_id:
        actual_mode = "retry (no session available for resume)"

    return (
        f"Task {task.id[:8]} recovered via {actual_mode} "
        f"-> new task {new_task.id[:8]} dispatched."
    )


def cmd_help(sup, _arg: str) -> str:
    return (
        "Supervisor Hub Commands:\n"
        "/status   — System resources + sessions + tasks\n"
        "/gpu      — GPU status\n"
        "/sessions — List all Claude Code sessions\n"
        "/tasks    — List all dispatched tasks\n"
        "/daemons  — List daemon (persistent) tasks\n"
        "/stop <id>       — Stop a daemon task\n"
        "/close <id> [id2 ...] — Close one or more tasks\n"
        "/close all       — Close all awaiting_closure tasks\n"
        "/followup <id> <text> — Ask follow-up on a completed task\n"
        "/reply <id> <text>    — Reply to a task waiting for input\n"
        "/recover [id] [mode]  — Recover interrupted tasks after restart\n"
        "/skip <id>       — Skip review for a task\n"
        "/help            — This message\n\n"
        "Message routing (sonnet auto-classifies):\n"
        "• Greetings, knowledge, conversation → direct reply\n"
        "• Tasks requiring execution → dispatched to worker\n"
        "• Complex tasks → decomposed into parallel workers\n\n"
        "When a task completes, you can ask follow-up questions.\n"
        "Use /close <id> when done, or /close all to close all at once."
    )


def find_task_by_prefix(sup, prefix: str):
    """Find a task by ID prefix. Returns Task or error string."""
    matching = [
        t for t in sup._task_dispatcher.list_tasks()
        if t.id.startswith(prefix)
    ]
    if not matching:
        return f"No task found matching '{prefix}'"
    if len(matching) > 1:
        return f"Ambiguous prefix '{prefix}', matches {len(matching)} tasks"
    return matching[0]


def extract_task_id_from_text(sup, text: str) -> Optional[str]:
    """Try to find a task ID prefix (8-char hex) in the message text."""
    if not sup._task_dispatcher:
        return None

    candidates = re.findall(r'\b([0-9a-f]{8,})\b', text.lower())
    if not candidates:
        return None

    matchable = {
        t.id: t for t in sup._task_dispatcher.list_tasks()
        if t.status not in sup._TERMINAL_STATUSES
    }
    for candidate in candidates:
        for task_id in matchable:
            if task_id.startswith(candidate) or task_id.replace("-", "").startswith(candidate):
                return task_id
    return None


def get_tasks_context(sup) -> dict:
    """Build task context for the sonnet route prompt.

    Uses EXCLUSION approach: all statuses visible EXCEPT terminal ones.
    """
    if not sup._task_dispatcher:
        return {"awaiting": [], "active": []}

    tasks = sup._task_dispatcher.list_tasks()
    awaiting = []
    for t in tasks:
        if t.status != "awaiting_closure":
            continue
        info: dict = {"id": t.id[:8], "description": t.description or t.prompt[:60]}
        if t.finished_at:
            elapsed = int(time.time() - t.finished_at)
            if elapsed < 60:
                info["completed_at"] = f"{elapsed}s ago"
            else:
                info["completed_at"] = f"{elapsed // 60}m ago"
        if t.result:
            info["result_summary"] = t.result[:120].replace("\n", " ")
        awaiting.append(info)
    active = []
    for t in tasks:
        if t.status in sup._TERMINAL_STATUSES or t.status == "awaiting_closure":
            continue
        info = {
            "id": t.id[:8],
            "status": t.status,
            "description": t.description or t.prompt[:60],
        }
        if t.current_step:
            info["current_step"] = t.current_step
        if t.steps_completed:
            info["steps_done"] = len(t.steps_completed)
        if t.started_at:
            elapsed = int(time.time() - t.started_at)
            info["elapsed"] = f"{elapsed}s" if elapsed < 60 else f"{elapsed // 60}m{elapsed % 60}s"
        if t.status == "failed" and t.error:
            info["error"] = t.error[:120]
        note = sup._STATUS_NOTES.get(t.status)
        if note:
            info["note"] = note
        active.append(info)
    return {"awaiting": awaiting, "active": active}
