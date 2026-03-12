"""Feishu message formatting for tasks.

Converts Task objects into human-readable text for Feishu chat display.
Uses duck typing — no import of Task needed.
"""

from __future__ import annotations

import time


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
        "interrupted": "INTERRUPTED",
        "cancelled": "CANCELLED",
    }
    return icons.get(status, status.upper())


def _elapsed_str(task) -> str:
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


def _format_task(task) -> str:
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

    # Interrupted task: show recovery info
    if task.status == "interrupted":
        resumable = "resumable" if task.session_id else "retryable"
        lines.append(f"  Recovery: {resumable} (session={'yes' if task.session_id else 'no'}, steps={len(task.steps_completed)})")
        if task.error:
            lines.append(f"  Reason: {task.error[:100]}")

    # Result or error snippet for finished tasks
    if task.status in ("done", "completed", "failed"):
        snippet = (task.result or task.error or "")[:100]
        if len(task.result or task.error or "") > 100:
            snippet += "..."
        if snippet:
            label = "Error" if task.status == "failed" else "Result"
            lines.append(f"  {label}: {snippet}")

    return "\n".join(lines)
