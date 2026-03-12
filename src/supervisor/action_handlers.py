"""Action handlers for Supervisor Hub message routing.

Each function accepts a supervisor instance and relevant parameters,
implementing the action decided by Sonnet routing. Same delegation
pattern as command_handlers.py.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .main import Supervisor

logger = logging.getLogger(__name__)


async def handle_dispatch(
    supervisor: Supervisor, text: str, chat_id: str, message_id: str, result: dict
) -> None:
    """Dispatch a single task to a worker."""
    description = result.get("description", "") or text[:80]
    if not supervisor._task_dispatcher:
        supervisor.gateway.reply_message(message_id, "Task dispatcher not available.")
        return

    supervisor.gateway.reply_message(
        message_id,
        f"📋 任务已调度\n描述: {description}\n状态: 排队中...",
    )

    enriched_prompt = supervisor._build_worker_prompt(text, description)

    def on_complete(task):
        from .notification import notify_task_result
        notify_task_result(supervisor, task, chat_id)

    task = await supervisor._task_dispatcher.dispatch(
        prompt=enriched_prompt,
        cwd="/workspace",
        task_type="oneshot",
        chat_id=chat_id,
        on_complete=on_complete,
        description=description,
    )
    logger.info("Dispatched: %s -> %s", task.id[:8], description)


async def handle_orchestrate(
    supervisor: Supervisor, text: str, subtasks: list[str], chat_id: str,
    message_id: str, description: str,
) -> None:
    """Dispatch a single orchestrator task that coordinates subagents."""
    if not supervisor._task_dispatcher:
        supervisor.gateway.reply_message(message_id, "Task dispatcher not available.")
        return

    subtask_list = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(subtasks))
    supervisor.gateway.reply_message(
        message_id,
        f"📋 编排任务已调度\n总述: {description}\n\n子任务:\n{subtask_list}\n\n"
        f"单一 orchestrator 将协调 {len(subtasks)} 个 subagent 并行执行",
    )

    enriched_prompt = supervisor._build_orchestrator_prompt(text, description, subtasks)

    def on_complete(task):
        from .notification import notify_task_result
        notify_task_result(supervisor, task, chat_id)

    task = await supervisor._task_dispatcher.dispatch(
        prompt=enriched_prompt,
        cwd="/workspace",
        task_type="oneshot",
        chat_id=chat_id,
        on_complete=on_complete,
        description=f"[orchestrator] {description}",
    )
    logger.info(
        "Orchestrator dispatched: %s -> %s (%d subtasks)",
        task.id[:8], description, len(subtasks),
    )


async def handle_follow_up(
    supervisor: Supervisor, task, text: str, chat_id: str, message_id: str
) -> None:
    """Handle follow-up message to an awaiting_closure task."""
    supervisor.gateway.reply_message(
        message_id,
        f"📎 追问转发给 worker [{task.id[:8]}]...",
    )

    try:
        result = await supervisor._task_dispatcher.follow_up_async(task.id, text)
    except Exception as e:
        supervisor.gateway.send_message(chat_id, f"追问失败: {e}")
        return

    truncated = result[:3000] + ("..." if len(result) > 3000 else "")
    sent_msg_id = supervisor.gateway.send_message(
        chat_id,
        f"📎 追问回复 [{task.id[:8]}]\n\n{truncated}\n\n"
        f"继续追问或 /close {task.id[:8]} 关闭任务",
    )
    # Track follow-up reply for chain replies
    if sent_msg_id:
        supervisor._message_task_map[sent_msg_id] = task.id


async def handle_sonnet_follow_up(
    supervisor: Supervisor, result: dict, text: str, chat_id: str, message_id: str
) -> None:
    """Handle follow_up action decided by sonnet."""
    task_id_prefix = result.get("task_id", "")
    if not task_id_prefix or not supervisor._task_dispatcher:
        # Fallback: treat as dispatch
        logger.warning("follow_up action but no task_id, falling back to dispatch")
        await handle_dispatch(supervisor, text, chat_id, message_id, result)
        return

    # Find the task by prefix
    matching = [
        t for t in supervisor._task_dispatcher.list_tasks()
        if t.id.startswith(task_id_prefix) and t.status == "awaiting_closure"
    ]
    if not matching:
        logger.warning("follow_up task_id %s not found, falling back to dispatch", task_id_prefix)
        await handle_dispatch(supervisor, text, chat_id, message_id, result)
        return

    task = matching[0]

    # Trust Sonnet's classification — no hardcoded fallback overrides.
    follow_up_text = result.get("text", text)
    await handle_follow_up(supervisor, task, follow_up_text, chat_id, message_id)


async def handle_sonnet_close(
    supervisor: Supervisor, result: dict, _chat_id: str, message_id: str
) -> None:
    """Handle close action decided by sonnet. Supports single task_id or task_ids array."""
    if not supervisor._task_dispatcher:
        reply_text = "没有找到可关闭的任务，请用 /close <id> 指定。"
        supervisor.gateway.reply_message(message_id, reply_text)
        supervisor._record_message("assistant", reply_text)
        return

    # Batch close: task_ids array takes priority over single task_id
    task_id_prefixes = result.get("task_ids", [])
    if not task_id_prefixes:
        single_id = result.get("task_id", "")
        task_id_prefixes = [single_id] if single_id else []

    if not task_id_prefixes:
        reply_text = "没有找到可关闭的任务，请用 /close <id> 指定。"
        supervisor.gateway.reply_message(message_id, reply_text)
        supervisor._record_message("assistant", reply_text)
        return

    # Resolve prefixes, preserving input order for interleaved output
    all_tasks = supervisor._task_dispatcher.list_tasks()
    resolved: list[tuple[str | None, str | None]] = []  # (task_id, error)
    for prefix in task_id_prefixes:
        matching = [
            t for t in all_tasks
            if t.id.startswith(prefix) and t.status == "awaiting_closure"
        ]
        if matching:
            resolved.append((matching[0].id, None))
        else:
            resolved.append((None, f"未找到匹配的待关闭任务 (id={prefix})"))

    valid_ids = [tid for tid, _ in resolved if tid is not None]
    if not valid_ids:
        reply_text = "\n".join(err for _, err in resolved if err) + "\n请用 /tasks 查看。"
        supervisor.gateway.reply_message(message_id, reply_text)
        supervisor._record_message("assistant", reply_text)
        return

    close_results = iter(supervisor._task_dispatcher.close_tasks(valid_ids))
    output: list[str] = []
    for task_id, error in resolved:
        if error is not None:
            output.append(error)
        else:
            output.append(next(close_results))
    reply_text = "\n".join(output)
    supervisor.gateway.reply_message(message_id, reply_text)
    supervisor._record_message("assistant", reply_text)


async def handle_sonnet_close_all(
    supervisor: Supervisor, _chat_id: str, message_id: str
) -> None:
    """Handle close_all action — close all tasks in awaiting_closure state."""
    if not supervisor._task_dispatcher:
        supervisor.gateway.reply_message(message_id, "Task dispatcher not available.")
        return

    awaiting = supervisor._task_dispatcher.get_awaiting_closure()
    if not awaiting:
        reply_text = "没有待关闭的任务。"
        supervisor.gateway.reply_message(message_id, reply_text)
        supervisor._record_message("assistant", reply_text)
        return

    task_ids = [t.id for t in awaiting]
    results = supervisor._task_dispatcher.close_tasks(task_ids)
    reply_text = "\n".join(results)
    supervisor.gateway.reply_message(message_id, reply_text)
    supervisor._record_message("assistant", reply_text)
