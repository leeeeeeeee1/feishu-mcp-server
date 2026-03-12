"""Notification helpers for Supervisor Hub.

Each function accepts a supervisor instance as first argument,
following the same delegation pattern as command_handlers.py.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from .task_dispatcher import _looks_like_close, _contains_close_intent

if TYPE_CHECKING:
    from .main import Supervisor

logger = logging.getLogger(__name__)


def notify_task_result(supervisor: Supervisor, task, chat_id: str) -> None:
    """Push task result/status to Feishu when a task finishes."""
    tid = task.id[:8]
    elapsed = ""
    if task.started_at:
        end = task.finished_at or time.time()
        secs = int(end - task.started_at)
        if secs < 60:
            elapsed = f" ({secs}s)"
        else:
            elapsed = f" ({secs // 60}m{secs % 60}s)"

    if task.status == "awaiting_closure":
        result_text = task.result or "(empty)"
        if len(result_text) > 3000:
            result_text = result_text[:3000] + "\n...(truncated)"
        msg = (
            f"✅ 任务完成 [{tid}]{elapsed}\n\n"
            f"{result_text}\n\n"
            f"可以直接追问，或 /close {tid} 关闭任务"
        )
    elif task.status == "waiting_for_input":
        msg = (
            f"⏸️ 任务需要你的输入 [{tid}]{elapsed}\n\n"
            f"{task.result or '(waiting for input)'}\n\n"
            f"/reply {tid} <your input>"
        )
    elif task.status == "failed":
        error_text = task.error or task.result or "unknown error"
        if len(error_text) > 1000:
            error_text = error_text[:1000] + "..."
        msg = f"❌ 任务失败 [{tid}]{elapsed}\n\n{error_text}"
    else:
        msg = f"ℹ️ 任务状态变更 [{tid}]: {task.status}{elapsed}"

    supervisor._record_message("assistant", msg[:500])

    try:
        sent_msg_id = supervisor.gateway.push_message(msg, chat_id=chat_id)
        # Track message→task mapping for reply-based follow-up
        if sent_msg_id and task.status == "awaiting_closure":
            supervisor._message_task_map[sent_msg_id] = task.id
            # Cap the map to prevent unbounded growth
            if len(supervisor._message_task_map) > 500:
                oldest_keys = list(supervisor._message_task_map.keys())[:-500]
                for k in oldest_keys:
                    del supervisor._message_task_map[k]
    except Exception as e:
        logger.error("Failed to notify task result: %s", e)


def try_post_reply_close(
    supervisor: Supervisor, user_text: str, reply_text: str,
    chat_id: str, message_id: str,
) -> None:
    """Post-reply close detection: fix for Sonnet misclassifying close as reply.

    If EITHER the user's message OR Sonnet's reply contains close intent,
    and exactly 1 task is awaiting closure, auto-close it.
    Skip if ambiguous (0 or 2+ awaiting tasks).
    """
    if not supervisor._task_dispatcher:
        return

    has_close_intent = (
        _contains_close_intent(reply_text)
        or _looks_like_close(user_text)
        or _contains_close_intent(user_text)
    )
    if not has_close_intent:
        return

    awaiting = supervisor._task_dispatcher.get_awaiting_closure()
    if len(awaiting) != 1:
        return

    task = awaiting[0]
    try:
        supervisor._task_dispatcher.close_task(task.id)
        logger.info(
            "Post-reply auto-close: task %s (user=%s, reply=%s)",
            task.id[:8], user_text[:40], reply_text[:40],
        )
        # Notify user so auto-close is not silent
        supervisor.gateway.reply_message(
            message_id, f"(任务 [{task.id[:8]}] 已自动关闭)"
        )
    except ValueError as e:
        logger.warning("Post-reply auto-close failed: %s", e)
