"""Supervisor Hub — Main daemon entry point.

Connects Feishu Gateway ↔ Claude Session with unified sonnet routing
and worker task dispatch. Sonnet classifies and responds in one call.

Architecture:
  Supervisor = Control Plane (conversation, orchestration, monitoring)
  Workers    = Execution Plane (actual tasks via claude -p)

Usage:
    FEISHU_APP_ID=xxx FEISHU_APP_SECRET=xxx python3 -m supervisor.main
"""

import asyncio
import json
import logging
import os
import re
import signal
import sys
import threading
import time
from typing import Optional

from .claude_session import ClaudeSession, StreamEvent
from .feishu_gateway import FeishuGateway
from .router_skill import build_route_prompt, build_route_system_prompt, build_route_user_prompt
from .task_dispatcher import _looks_like_close, _contains_close_intent

logger = logging.getLogger(__name__)

# System prompt for Tier 1 answers — supervisor answers knowledge questions only
SUPERVISOR_SYSTEM_PROMPT = """You are the Supervisor Hub — the central control session for this development container.
Your role is STRICTLY limited to: answering knowledge questions, providing information, and conversation.

RULES:
- You MUST NOT execute commands, modify files, or change system state.
- You MUST NOT use Bash, Edit, Write, or any execution tool.
- You CAN answer general knowledge questions from your training data.
- Be concise. Answer in the user's language (Chinese if they write in Chinese).
- Keep responses under 2000 characters."""


class Supervisor:
    """Main supervisor daemon orchestrating all components."""

    # Statuses that are fully terminal — invisible to sonnet.
    _TERMINAL_STATUSES = frozenset({"completed", "cancelled"})

    # Human-readable notes for non-obvious statuses.
    _STATUS_NOTES = {
        "waiting_for_input": "Needs user input",
        "follow_up": "正在执行追问",
        "review": "等待人工审核",
        "learning": "正在提取经验",
        "done": "执行完成，等待关闭",
    }

    MAX_HISTORY = 20  # keep last N messages

    def __init__(self):
        self.gateway = FeishuGateway()
        self.claude = ClaudeSession(system_prompt=SUPERVISOR_SYSTEM_PROMPT)
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._shutdown = asyncio.Event()
        self._conversation_history: list[dict[str, str]] = []
        self._read_messages: dict[str, float] = {}  # message_id -> timestamp, ordered by insertion
        self._message_task_map: dict[str, str] = {}  # feishu_message_id → task_id

        # Monitors
        try:
            from . import session_monitor as _sm
            self._session_monitor = _sm
        except ImportError:
            self._session_monitor = None

        try:
            from . import container_monitor as _cm
            self._container_monitor = _cm
        except ImportError:
            self._container_monitor = None

        # Task Dispatcher
        try:
            from . import task_dispatcher as _td
            self._task_dispatcher = _td
        except ImportError:
            self._task_dispatcher = None

        # Scheduler (wired up in _run_async)
        self._scheduler = None

    # ══════════════════════════════════════════════════════════
    # Layer 1: Local Commands (/status, /tasks, etc.)
    # ══════════════════════════════════════════════════════════

    def _handle_local_command(self, cmd: str, chat_id: str, message_id: str) -> bool:
        """Handle /commands locally. Returns True if handled."""
        parts = cmd.strip().split(maxsplit=1)
        command = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        handlers = {
            "/status": self._cmd_status,
            "/sessions": self._cmd_sessions,
            "/tasks": self._cmd_tasks,
            "/gpu": self._cmd_gpu,
            "/daemons": self._cmd_daemons,
            "/stop": self._cmd_stop,
            "/skip": self._cmd_skip,
            "/close": self._cmd_close,
            "/followup": self._cmd_followup,
            "/reply": self._cmd_reply,
            "/help": self._cmd_help,
        }

        handler = handlers.get(command)
        if handler:
            try:
                result = handler(arg)
            except Exception as e:
                result = f"Error: {e}"
            self.gateway.reply_message(message_id, result)
            self._record_message("user", cmd)
            self._record_message("assistant", result[:500])
            return True
        return False

    def _cmd_status(self, _arg: str) -> str:
        parts: list[str] = []
        if self._container_monitor:
            parts.append(self._container_monitor.get_status_text())
        if self._session_monitor:
            parts.append(self._session_monitor.get_sessions_text())
        if self._task_dispatcher:
            tasks_text = self._task_dispatcher.get_tasks_text()
            if tasks_text != "No tasks.":
                parts.append(tasks_text)
        return "\n\n".join(parts) if parts else "Monitors not initialized yet."

    def _cmd_sessions(self, _arg: str) -> str:
        if self._session_monitor:
            return self._session_monitor.get_sessions_text()
        return "Session monitor not available."

    def _cmd_tasks(self, _arg: str) -> str:
        if self._task_dispatcher:
            return self._task_dispatcher.get_tasks_text()
        return "Task dispatcher not available."

    def _cmd_gpu(self, _arg: str) -> str:
        if self._container_monitor:
            return self._container_monitor.get_gpu_text()
        return "Container monitor not available."

    def _cmd_daemons(self, _arg: str) -> str:
        if self._task_dispatcher:
            return self._task_dispatcher.get_daemons_text()
        return "Task dispatcher not available."

    def _cmd_stop(self, arg: str) -> str:
        if not arg:
            return "Usage: /stop <task_id>"
        if self._task_dispatcher:
            return self._task_dispatcher.stop_daemon(arg)
        return "Task dispatcher not available."

    def _cmd_skip(self, arg: str) -> str:
        if not arg:
            return "Usage: /skip <task_id>"
        if self._task_dispatcher:
            return self._task_dispatcher.skip_review(arg)
        return "Task dispatcher not available."

    def _cmd_close(self, arg: str) -> str:
        """Close a task that is awaiting closure."""
        if not arg:
            return "Usage: /close <task_id>"
        if not self._task_dispatcher:
            return "Task dispatcher not available."
        task = self._find_task_by_prefix(arg)
        if isinstance(task, str):
            return task  # error message
        return self._task_dispatcher.close_task(task.id)

    def _cmd_followup(self, arg: str) -> str:
        """Send a follow-up to a task awaiting closure."""
        parts = arg.split(maxsplit=1)
        if len(parts) < 2:
            return "Usage: /followup <task_id> <your question>"
        task_id_prefix, user_input = parts[0], parts[1]
        if not self._task_dispatcher:
            return "Task dispatcher not available."
        task = self._find_task_by_prefix(task_id_prefix)
        if isinstance(task, str):
            return task

        if task.status != "awaiting_closure":
            return f"Task {task.id[:8]} is not awaiting closure (status={task.status})"

        if self._loop:
            future = asyncio.run_coroutine_threadsafe(
                self._task_dispatcher.follow_up_async(task.id, user_input),
                self._loop,
            )
            try:
                result = future.result(timeout=600)
                truncated = result[:3000] + ("..." if len(result) > 3000 else "")
                return f"📎 追问回复 [{task.id[:8]}]\n\n{truncated}\n\n回复 /close {task.id[:8]} 关闭，或继续 /followup {task.id[:8]} <问题>"
            except Exception as e:
                return f"Error: {e}"
        return "Event loop not available."

    def _cmd_reply(self, arg: str) -> str:
        """Resume a task that is waiting for user input."""
        parts = arg.split(maxsplit=1)
        if len(parts) < 2:
            return "Usage: /reply <task_id> <your reply>"
        task_id_prefix, user_input = parts[0], parts[1]
        if not self._task_dispatcher:
            return "Task dispatcher not available."
        task = self._find_task_by_prefix(task_id_prefix)
        if isinstance(task, str):
            return task

        if task.status != "waiting_for_input":
            return f"Task {task.id[:8]} is not waiting for input (status={task.status})"

        if self._loop:
            future = asyncio.run_coroutine_threadsafe(
                self._task_dispatcher.resume_task(task.id, user_input),
                self._loop,
            )
            try:
                result = future.result(timeout=600)
                return f"Task {task.id[:8]} resumed.\n\n{result[:3000]}"
            except Exception as e:
                return f"Error: {e}"
        return "Event loop not available."

    def _cmd_help(self, _arg: str) -> str:
        return (
            "Supervisor Hub Commands:\n"
            "/status   — System resources + sessions + tasks\n"
            "/gpu      — GPU status\n"
            "/sessions — List all Claude Code sessions\n"
            "/tasks    — List all dispatched tasks\n"
            "/daemons  — List daemon (persistent) tasks\n"
            "/stop <id>       — Stop a daemon task\n"
            "/close <id>      — Close a completed task\n"
            "/followup <id> <text> — Ask follow-up on a completed task\n"
            "/reply <id> <text>    — Reply to a task waiting for input\n"
            "/skip <id>       — Skip review for a task\n"
            "/help            — This message\n\n"
            "Message routing (sonnet auto-classifies):\n"
            "• Greetings, knowledge, conversation → direct reply\n"
            "• Tasks requiring execution → dispatched to worker\n"
            "• Complex tasks → decomposed into parallel workers\n\n"
            "When a task completes, you can ask follow-up questions.\n"
            "Use /close <id> when you're done with a task."
        )

    def _find_task_by_prefix(self, prefix: str):
        """Find a task by ID prefix. Returns Task or error string."""
        matching = [
            t for t in self._task_dispatcher.list_tasks()
            if t.id.startswith(prefix)
        ]
        if not matching:
            return f"No task found matching '{prefix}'"
        if len(matching) > 1:
            return f"Ambiguous prefix '{prefix}', matches {len(matching)} tasks"
        return matching[0]

    # ══════════════════════════════════════════════════════════
    # Layer 2: Smart Message Routing (sonnet classify + respond)
    # ══════════════════════════════════════════════════════════

    def _extract_task_id_from_text(self, text: str) -> Optional[str]:
        """Try to find a task ID prefix (8-char hex) in the message text.

        Returns the matched task ID (full UUID) if found and in a matchable
        status (running, pending, failed, waiting_for_input, awaiting_closure),
        or None.
        """
        if not self._task_dispatcher:
            return None

        candidates = re.findall(r'\b([0-9a-f]{8,})\b', text.lower())
        if not candidates:
            return None

        matchable = {
            t.id: t for t in self._task_dispatcher.list_tasks()
            if t.status not in self._TERMINAL_STATUSES
        }
        for candidate in candidates:
            for task_id in matchable:
                if task_id.startswith(candidate) or task_id.replace("-", "").startswith(candidate):
                    return task_id
        return None

    def _get_tasks_context(self) -> dict:
        """Build task context for the sonnet route prompt.

        Uses EXCLUSION approach: all statuses visible EXCEPT terminal ones
        (completed, cancelled). New statuses are visible by default.

        Returns dict with:
            "awaiting": list of awaiting_closure tasks
            "active": list of all non-terminal, non-awaiting tasks
        """
        if not self._task_dispatcher:
            return {"awaiting": [], "active": []}

        tasks = self._task_dispatcher.list_tasks()
        awaiting = [
            {"id": t.id[:8], "description": t.description or t.prompt[:60]}
            for t in tasks if t.status == "awaiting_closure"
        ]
        active = []
        for t in tasks:
            if t.status in self._TERMINAL_STATUSES or t.status == "awaiting_closure":
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
            note = self._STATUS_NOTES.get(t.status)
            if note:
                info["note"] = note
            active.append(info)
        return {"awaiting": awaiting, "active": active}

    async def _route_message(
        self, text: str, chat_id: str, message_id: str,
        parent_id: str = "",
    ):
        """Route incoming message via sonnet.

        0. Reply to task result message (parent_id match) → direct follow-up
        1. Explicit task ID in message → direct follow-up
        2. Sonnet classifies (with awaiting task context) → reply/dispatch/follow_up
        """
        # ── Step 0: Reply-based follow-up or close (parent_id → task mapping)
        if parent_id and parent_id in self._message_task_map:
            task_id = self._message_task_map[parent_id]
            if self._task_dispatcher:
                task = self._task_dispatcher.get_task(task_id)
                if task and task.status == "awaiting_closure":
                    self._record_message("user", text)
                    # Short acknowledgement → close task
                    if _looks_like_close(text):
                        logger.info(
                            "Reply-based close: parent=%s → task %s",
                            parent_id, task.id[:8],
                        )
                        try:
                            self._task_dispatcher.close_task(task.id)
                        except ValueError as e:
                            self.gateway.reply_message(message_id, f"关闭失败: {e}")
                            return
                        reply_text = f"任务 [{task.id[:8]}] 已关闭"
                        self.gateway.reply_message(message_id, reply_text)
                        self._record_message("assistant", reply_text)
                        return
                    # Otherwise → follow-up
                    logger.info(
                        "Reply-based follow-up: parent=%s → task %s",
                        parent_id, task.id[:8],
                    )
                    await self._handle_follow_up(task, text, chat_id, message_id)
                    self._record_message("assistant", f"[follow-up sent to {task.id[:8]}]")
                    return

        # ── Step 1: Explicit task ID → direct follow-up (awaiting_closure only)
        matched_task_id = self._extract_task_id_from_text(text)
        if matched_task_id:
            task = self._task_dispatcher.get_task(matched_task_id)
            if task and task.status == "awaiting_closure":
                logger.info(
                    "Explicit task ID match, follow-up to %s: %s",
                    task.id[:8], text[:60],
                )
                self._record_message("user", text)
                await self._handle_follow_up(task, text, chat_id, message_id)
                self._record_message("assistant", f"[follow-up sent to {task.id[:8]}]")
                return
            elif task:
                logger.info(
                    "Explicit task ID match (%s) but status=%s, passing to sonnet",
                    task.id[:8], task.status,
                )

        # ── Step 2: Record user message
        self._record_message("user", text)

        # ── Step 3: Sonnet classify + respond (with task context)
        tasks_ctx = self._get_tasks_context()
        system_prompt = build_route_system_prompt()
        user_prompt = build_route_user_prompt(
            text,
            awaiting_tasks=tasks_ctx["awaiting"] or None,
            active_tasks=tasks_ctx["active"] or None,
            conversation_history=self._get_history_text(),
        )
        result = await self.claude.route_message(text, system_prompt, user_prompt)
        action = result.get("action", "dispatch")
        logger.info("Route result: action=%s result=%s", action, str(result)[:200])

        if action == "reply":
            reply_text = result.get("text", "").strip()
            if not reply_text:
                reply_text = "收到，有什么可以帮你的？"
            self.gateway.reply_message(message_id, reply_text)
            self._record_message("assistant", reply_text)

        elif action == "close":
            await self._handle_sonnet_close(result, chat_id, message_id)

        elif action == "follow_up":
            await self._handle_sonnet_follow_up(result, text, chat_id, message_id)

        elif action == "dispatch_multi":
            subtasks = result.get("subtasks", [])
            description = result.get("description", text[:80])
            if subtasks:
                await self._handle_dispatch_multi(
                    text, subtasks, chat_id, message_id, description,
                )
            else:
                await self._handle_dispatch(text, chat_id, message_id, result)

        else:
            # action == "dispatch" (or unknown → safe default)
            await self._handle_dispatch(text, chat_id, message_id, result)

    # ── Action Handlers ──

    # ── Conversation History ──

    def _record_message(self, role: str, text: str) -> None:
        """Append a message to conversation history, capped at MAX_HISTORY."""
        self._conversation_history.append({"role": role, "text": text})
        if len(self._conversation_history) > self.MAX_HISTORY:
            self._conversation_history = self._conversation_history[-self.MAX_HISTORY:]

    def _get_history_text(self) -> str:
        """Format recent conversation history as text for worker context."""
        if not self._conversation_history:
            return ""
        lines = []
        for msg in self._conversation_history:
            role_label = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{role_label}: {msg['text']}")
        return "\n".join(lines)

    def _build_worker_prompt(self, text: str, description: str) -> str:
        """Build an enriched prompt for the worker with task context.

        Args:
            text: The original user message.
            description: Sonnet-generated task description.

        Returns:
            Enriched prompt string with context for the worker.
        """
        parts = [
            f"You are a worker agent in a development container.",
            f"Task: {description}",
            f"User's original request: {text}",
            f"Working directory: /workspace",
        ]

        history = self._get_history_text()
        if history:
            parts.append(f"\n## Recent conversation history\n{history}")

        parts.append(
            "\nExecute this task thoroughly. Be concise in your response.\n"
            "Answer in the same language as the user's request."
        )

        return "\n".join(parts)

    async def _handle_dispatch(
        self, text: str, chat_id: str, message_id: str, result: dict
    ):
        """Dispatch a single task to a worker."""
        description = result.get("description", "") or text[:80]
        if not self._task_dispatcher:
            self.gateway.reply_message(message_id, "Task dispatcher not available.")
            return

        self.gateway.reply_message(
            message_id,
            f"📋 任务已调度\n描述: {description}\n状态: 排队中...",
        )

        enriched_prompt = self._build_worker_prompt(text, description)

        def on_complete(task):
            self._notify_task_result(task, chat_id)

        task = await self._task_dispatcher.dispatch(
            prompt=enriched_prompt,
            cwd="/workspace",
            task_type="oneshot",
            chat_id=chat_id,
            on_complete=on_complete,
            description=description,
        )
        logger.info("Dispatched: %s -> %s", task.id[:8], description)

    async def _handle_dispatch_multi(
        self, text: str, subtasks: list[str], chat_id: str,
        message_id: str, description: str,
    ):
        """Decompose into parallel sub-tasks."""
        if not self._task_dispatcher:
            self.gateway.reply_message(message_id, "Task dispatcher not available.")
            return

        subtask_list = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(subtasks))
        self.gateway.reply_message(
            message_id,
            f"📋 复杂任务已拆解\n总述: {description}\n\n子任务:\n{subtask_list}\n\n并行调度中...",
        )

        for sub_prompt in subtasks:
            enriched_prompt = (
                f"This is a sub-task of a larger request: \"{text[:200]}\"\n\n"
                f"Your specific sub-task: {sub_prompt}\n\n"
                f"Focus only on this sub-task. Be thorough and concise."
            )

            def on_complete(task, _cid=chat_id):
                self._notify_task_result(task, _cid)

            task = await self._task_dispatcher.dispatch(
                prompt=enriched_prompt,
                cwd="/workspace",
                task_type="oneshot",
                chat_id=chat_id,
                on_complete=on_complete,
                description=sub_prompt[:80],
            )
            logger.info("Sub-task dispatched: %s -> %s", task.id[:8], sub_prompt[:60])

    async def _handle_follow_up(
        self, task, text: str, chat_id: str, message_id: str
    ):
        """Handle follow-up message to an awaiting_closure task."""
        self.gateway.reply_message(
            message_id,
            f"📎 追问转发给 worker [{task.id[:8]}]...",
        )

        try:
            result = await self._task_dispatcher.follow_up_async(task.id, text)
        except Exception as e:
            self.gateway.send_message(chat_id, f"追问失败: {e}")
            return

        truncated = result[:3000] + ("..." if len(result) > 3000 else "")
        sent_msg_id = self.gateway.send_message(
            chat_id,
            f"📎 追问回复 [{task.id[:8]}]\n\n{truncated}\n\n"
            f"继续追问或 /close {task.id[:8]} 关闭任务",
        )
        # Track follow-up reply for chain replies
        if sent_msg_id:
            self._message_task_map[sent_msg_id] = task.id

    async def _handle_sonnet_follow_up(
        self, result: dict, text: str, chat_id: str, message_id: str
    ):
        """Handle follow_up action decided by sonnet."""
        task_id_prefix = result.get("task_id", "")
        if not task_id_prefix or not self._task_dispatcher:
            # Fallback: treat as dispatch
            logger.warning("follow_up action but no task_id, falling back to dispatch")
            await self._handle_dispatch(text, chat_id, message_id, result)
            return

        # Find the task by prefix
        matching = [
            t for t in self._task_dispatcher.list_tasks()
            if t.id.startswith(task_id_prefix) and t.status == "awaiting_closure"
        ]
        if not matching:
            logger.warning("follow_up task_id %s not found, falling back to dispatch", task_id_prefix)
            await self._handle_dispatch(text, chat_id, message_id, result)
            return

        task = matching[0]

        # Smart close fallback: if original user text contains close intent,
        # close the task instead of following up (sonnet misclassification)
        if _contains_close_intent(text):
            logger.info(
                "Smart close fallback: follow_up but text '%s' has close intent → closing %s",
                text[:60], task.id[:8],
            )
            await self._handle_sonnet_close(
                {"action": "close", "task_id": task_id_prefix}, chat_id, message_id,
            )
            return

        follow_up_text = result.get("text", text)
        await self._handle_follow_up(task, follow_up_text, chat_id, message_id)

    async def _handle_sonnet_close(
        self, result: dict, chat_id: str, message_id: str
    ):
        """Handle close action decided by sonnet."""
        task_id_prefix = result.get("task_id", "")
        if not task_id_prefix or not self._task_dispatcher:
            self.gateway.reply_message(
                message_id, "没有找到可关闭的任务，请用 /close <id> 指定。"
            )
            return

        matching = [
            t for t in self._task_dispatcher.list_tasks()
            if t.id.startswith(task_id_prefix) and t.status == "awaiting_closure"
        ]
        if not matching:
            self.gateway.reply_message(
                message_id,
                f"未找到匹配的待关闭任务 (id={task_id_prefix})，请用 /tasks 查看。",
            )
            return

        task = matching[0]
        try:
            self._task_dispatcher.close_task(task.id)
        except ValueError as e:
            self.gateway.reply_message(message_id, f"关闭失败: {e}")
            return
        reply_text = f"任务 [{task.id[:8]}] 已关闭"
        self.gateway.reply_message(message_id, reply_text)
        self._record_message("assistant", reply_text)

    # ── Task Result Notification ──

    def _notify_task_result(self, task, chat_id: str):
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

        self._record_message("assistant", msg[:500])

        try:
            sent_msg_id = self.gateway.push_message(msg, chat_id=chat_id)
            # Track message→task mapping for reply-based follow-up
            if sent_msg_id and task.status == "awaiting_closure":
                self._message_task_map[sent_msg_id] = task.id
                # Cap the map to prevent unbounded growth
                if len(self._message_task_map) > 500:
                    oldest_keys = list(self._message_task_map.keys())[:-500]
                    for k in oldest_keys:
                        del self._message_task_map[k]
        except Exception as e:
            logger.error("Failed to notify task result: %s", e)

    # ══════════════════════════════════════════════════════════
    # Incoming Message Router (entry point from Feishu)
    # ══════════════════════════════════════════════════════════

    def _on_message_read(self, reader_id: str, message_id_list: list[str], read_time: str):
        """Track which messages the user has read."""
        now = time.time()
        for mid in message_id_list:
            self._read_messages[mid] = now
        logger.info("[READ] reader=%s read %d messages", reader_id, len(message_id_list))
        # Cap the dict — evict oldest entries (insertion-ordered in Python 3.7+)
        if len(self._read_messages) > 500:
            keys = list(self._read_messages.keys())
            for k in keys[:len(keys) - 500]:
                del self._read_messages[k]

    def is_message_read(self, message_id: str) -> bool:
        """Check if a message has been read by the user."""
        return message_id in self._read_messages

    def _on_feishu_message(
        self,
        sender_id: str,
        message_id: str,
        chat_id: str,
        msg_type: str,
        content: str,
        raw_event=None,
        parent_id: str = "",
        root_id: str = "",
    ):
        """Route incoming Feishu messages."""
        if not content or not content.strip():
            return

        if not self.gateway.push_chat_id and chat_id:
            self.gateway.push_chat_id = chat_id
            logger.info("Auto-set push_chat_id=%s from first message", chat_id)

        text = content.strip()

        # Layer 1: /commands
        if text.startswith("/"):
            self._handle_local_command(text, chat_id, message_id)
            return

        # Layer 2: Smart routing (pass parent_id for reply-based follow-up)
        if self._loop:
            asyncio.run_coroutine_threadsafe(
                self._route_message(text, chat_id, message_id, parent_id=parent_id),
                self._loop,
            )

    # ══════════════════════════════════════════════════════════
    # Lifecycle
    # ══════════════════════════════════════════════════════════

    async def _run_async(self):
        """Main async loop."""
        self._loop = asyncio.get_event_loop()

        self.gateway.set_message_handler(self._on_feishu_message)
        self.gateway.set_message_read_handler(self._on_message_read)
        ws_thread = threading.Thread(
            target=self.gateway.start_receiving, daemon=True
        )
        ws_thread.start()
        logger.info("Feishu gateway started in background thread")

        try:
            from .scheduler import Scheduler
            self._scheduler = Scheduler(
                get_system_status=(
                    self._container_monitor.get_system_status
                    if self._container_monitor else None
                ),
                get_gpu_status=(
                    self._container_monitor.get_gpu_status
                    if self._container_monitor else None
                ),
                get_sessions_text=(
                    self._session_monitor.get_sessions_text
                    if self._session_monitor else None
                ),
                get_tasks_text=(
                    self._task_dispatcher.get_tasks_text
                    if self._task_dispatcher else None
                ),
                push_message=self.gateway.push_message,
            )
            await self._scheduler.start()
            logger.info("Scheduler started")
        except ImportError:
            logger.warning("Scheduler module not available")

        await self._shutdown.wait()
        logger.info("Shutdown signal received, cleaning up...")

        if self._scheduler:
            await self._scheduler.stop()

    def run(self):
        """Start the supervisor daemon."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        )

        logger.info("=" * 50)
        logger.info("Supervisor Hub starting...")
        logger.info("Router: unified sonnet classify+respond")
        logger.info("=" * 50)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, lambda: self._shutdown.set())

        try:
            loop.run_until_complete(self._run_async())
        except KeyboardInterrupt:
            pass
        finally:
            loop.close()
            logger.info("Supervisor Hub stopped.")


def main():
    """Entry point."""
    supervisor = Supervisor()
    supervisor.run()


if __name__ == "__main__":
    main()
