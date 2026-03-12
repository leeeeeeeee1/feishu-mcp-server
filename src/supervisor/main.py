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
import logging
import re
import signal
import threading
import time
from typing import Optional

from collections import deque

from .claude_session import ClaudeSession
from .feishu_gateway import FeishuGateway
from .router_skill import build_route_system_prompt, build_route_user_prompt
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
        self._conversation_history: deque[dict[str, str]] = deque(maxlen=self.MAX_HISTORY)
        self._read_messages: dict[str, float] = {}  # message_id -> timestamp, ordered by insertion
        self._message_task_map: dict[str, str] = {}  # feishu_message_id → task_id
        self._current_chat_id: Optional[str] = None

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
            "/recover": self._cmd_recover,
            "/help": self._cmd_help,
        }

        handler = handlers.get(command)
        if handler:
            try:
                self._current_chat_id = chat_id
                result = handler(arg)
            except Exception as e:
                result = f"Error: {e}"
            finally:
                self._current_chat_id = None
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
        """Close one or more tasks. Supports: /close <id>, /close <id1> <id2>, /close all."""
        if not arg:
            return "Usage: /close <id1> [id2 ...] or /close all"
        if not self._task_dispatcher:
            return "Task dispatcher not available."

        if arg.strip().lower() == "all":
            awaiting = self._task_dispatcher.get_awaiting_closure()
            if not awaiting:
                return "No tasks awaiting closure."
            # Snapshot IDs; close_tasks handles per-task errors if status changes.
            task_ids = [t.id for t in awaiting]
            results = self._task_dispatcher.close_tasks(task_ids)
            return "\n".join(results)

        prefixes = arg.split()
        if len(prefixes) == 1:
            task = self._find_task_by_prefix(prefixes[0])
            if isinstance(task, str):
                return task
            return self._task_dispatcher.close_task(task.id)

        # Resolve all prefixes, preserving input order
        resolved: list[tuple[str | None, str | None]] = []  # (task_id, error)
        for prefix in prefixes:
            task = self._find_task_by_prefix(prefix)
            if isinstance(task, str):
                resolved.append((None, task))
            else:
                resolved.append((task.id, None))

        valid_ids = [tid for tid, _ in resolved if tid is not None]
        close_results = iter(
            self._task_dispatcher.close_tasks(valid_ids) if valid_ids else []
        )
        output: list[str] = []
        for task_id, error in resolved:
            if error is not None:
                output.append(error)
            else:
                output.append(next(close_results))
        return "\n".join(output)

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

    def _cmd_recover(self, arg: str) -> str:
        """Recover interrupted tasks after supervisor restart."""
        if not self._task_dispatcher:
            return "Task dispatcher not available."

        _VALID_MODES = ("resume", "retry", "dismiss")

        # No argument: list interrupted tasks
        if not arg.strip():
            interrupted = self._task_dispatcher.list_interrupted()
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

        task = self._find_task_by_prefix(task_id_prefix)
        if isinstance(task, str):
            return task  # error message

        if task.status != "interrupted":
            return (
                f"Task {task.id[:8]} is not interrupted "
                f"(status={task.status}). Only interrupted tasks can be recovered."
            )

        if not self._loop:
            return "Event loop not available."

        # Wire notification callback so user gets result when recovered task completes
        chat_id = getattr(self, "_current_chat_id", None)

        def on_complete(finished_task, _cid=chat_id):
            if _cid:
                self._notify_task_result(finished_task, _cid)

        try:
            future = asyncio.run_coroutine_threadsafe(
                self._task_dispatcher.recover_task(
                    task.id, mode=mode, on_complete=on_complete,
                ),
                self._loop,
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

    def _cmd_help(self, _arg: str) -> str:
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

        All user messages are classified by Sonnet (no hardcoded pattern matching).
        Reply context (parent_id → task mapping) enriches the Sonnet prompt to
        help it distinguish close vs follow_up intent.
        """
        # ── Step 0: Reply-based quick close or context enrichment
        reply_to_task: dict | None = None
        if parent_id and parent_id in self._message_task_map:
            task_id = self._message_task_map[parent_id]
            if self._task_dispatcher:
                task = self._task_dispatcher.get_task(task_id)
                if task and task.status == "awaiting_closure":
                    # Fast local close for obvious acknowledgements/close intent
                    # (avoids Sonnet API round-trip for simple replies)
                    if _looks_like_close(text) or _contains_close_intent(text):
                        self._record_message("user", text)
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
                    # Non-close reply: enrich Sonnet context with task info
                    reply_to_task = {
                        "id": task.id[:8],
                        "description": task.description or task.prompt[:60],
                    }
                    logger.info(
                        "Reply context: parent=%s → task %s",
                        parent_id, task.id[:8],
                    )

        # ── Step 1: Record user message
        self._record_message("user", text)

        # ── Step 2: Sonnet classify + respond (with task context + reply context)
        tasks_ctx = self._get_tasks_context()
        system_prompt = build_route_system_prompt()
        user_prompt = build_route_user_prompt(
            text,
            awaiting_tasks=tasks_ctx["awaiting"] or None,
            active_tasks=tasks_ctx["active"] or None,
            conversation_history=self._get_history_text(),
            reply_to_task=reply_to_task,
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

        elif action == "close_all":
            await self._handle_sonnet_close_all(chat_id, message_id)

        elif action == "follow_up":
            await self._handle_sonnet_follow_up(result, text, chat_id, message_id)

        elif action in ("orchestrate", "dispatch_multi"):  # dispatch_multi: compat (normalised by route_message)
            subtasks = result.get("subtasks", [])
            description = result.get("description", text[:80])
            if subtasks:
                await self._handle_orchestrate(
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
        """Append a message to conversation history (auto-capped by deque maxlen)."""
        self._conversation_history.append({"role": role, "text": text})

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
            "You are a worker agent in a development container.",
            f"Task: {description}",
            f"User's original request: {text}",
            "Working directory: /workspace",
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

    def _build_orchestrator_prompt(
        self, text: str, description: str, subtasks: list[str],
    ) -> str:
        """Build a prompt for the orchestrator worker with subagent instructions.

        The orchestrator is a single worker that coordinates multiple subagents
        via Claude's Agent tool, sharing context and managing dependencies.

        Args:
            text: The original user message.
            description: Sonnet-generated task description.
            subtasks: List of subtask descriptions to coordinate.

        Returns:
            Enriched prompt string with orchestrator instructions.
        """
        subtask_list = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(subtasks))

        parts = [
            "You are an orchestrator worker in a development container.",
            f"Task: {description}",
            f"User's original request: {text}",
            "Working directory: /workspace",
            "",
            "## Orchestration Instructions",
            "",
            "You must coordinate the following subtasks using the Agent tool.",
            "Launch subagents IN PARALLEL for independent subtasks to maximize efficiency.",
            "Each subagent shares the same codebase but works independently.",
            "",
            f"## Subtasks to coordinate\n{subtask_list}",
            "",
            "## How to orchestrate",
            "1. Launch parallel Agent subagents for independent subtasks",
            "2. Wait for results and check for conflicts or dependencies",
            "3. If subtasks have dependencies, run them sequentially",
            "4. Synthesize all results into a final summary for the user",
            "5. Handle any merge conflicts or inconsistencies between subagent outputs",
            "",
            "## Important",
            "- Use the Agent tool to launch subagents, NOT direct tool calls for each subtask",
            "- Subagents can share context — pass relevant info between them",
            "- If a subagent fails, decide whether to retry or proceed with remaining subtasks",
            "- Provide a unified final summary when all subtasks complete",
        ]

        history = self._get_history_text()
        if history:
            parts.append(f"\n## Recent conversation history\n{history}")

        parts.append(
            "\nExecute this task thoroughly. Be concise in your response.\n"
            "Answer in the same language as the user's request."
        )

        return "\n".join(parts)

    async def _handle_orchestrate(
        self, text: str, subtasks: list[str], chat_id: str,
        message_id: str, description: str,
    ):
        """Dispatch a single orchestrator task that coordinates subagents.

        Unlike the old dispatch_multi (N independent tasks), this creates ONE
        task whose worker uses the Agent tool to launch and coordinate subagents.
        Benefits: shared context, inter-agent communication, conflict handling.
        """
        if not self._task_dispatcher:
            self.gateway.reply_message(message_id, "Task dispatcher not available.")
            return

        subtask_list = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(subtasks))
        self.gateway.reply_message(
            message_id,
            f"📋 编排任务已调度\n总述: {description}\n\n子任务:\n{subtask_list}\n\n"
            f"单一 orchestrator 将协调 {len(subtasks)} 个 subagent 并行执行",
        )

        enriched_prompt = self._build_orchestrator_prompt(text, description, subtasks)

        def on_complete(task):
            self._notify_task_result(task, chat_id)

        task = await self._task_dispatcher.dispatch(
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

        # Trust Sonnet's classification — no hardcoded fallback overrides.
        # The routing prompt includes clear disambiguation rules for close vs follow_up.
        follow_up_text = result.get("text", text)
        await self._handle_follow_up(task, follow_up_text, chat_id, message_id)

    async def _handle_sonnet_close(
        self, result: dict, _chat_id: str, message_id: str
    ):
        """Handle close action decided by sonnet. Supports single task_id or task_ids array."""
        if not self._task_dispatcher:
            reply_text = "没有找到可关闭的任务，请用 /close <id> 指定。"
            self.gateway.reply_message(message_id, reply_text)
            self._record_message("assistant", reply_text)
            return

        # Batch close: task_ids array takes priority over single task_id
        task_id_prefixes = result.get("task_ids", [])
        if not task_id_prefixes:
            single_id = result.get("task_id", "")
            task_id_prefixes = [single_id] if single_id else []

        if not task_id_prefixes:
            reply_text = "没有找到可关闭的任务，请用 /close <id> 指定。"
            self.gateway.reply_message(message_id, reply_text)
            self._record_message("assistant", reply_text)
            return

        # Resolve prefixes, preserving input order for interleaved output
        all_tasks = self._task_dispatcher.list_tasks()
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
            self.gateway.reply_message(message_id, reply_text)
            self._record_message("assistant", reply_text)
            return

        close_results = iter(self._task_dispatcher.close_tasks(valid_ids))
        output: list[str] = []
        for task_id, error in resolved:
            if error is not None:
                output.append(error)
            else:
                output.append(next(close_results))
        reply_text = "\n".join(output)
        self.gateway.reply_message(message_id, reply_text)
        self._record_message("assistant", reply_text)

    async def _handle_sonnet_close_all(self, _chat_id: str, message_id: str):
        """Handle close_all action — close all tasks in awaiting_closure state."""
        if not self._task_dispatcher:
            self.gateway.reply_message(message_id, "Task dispatcher not available.")
            return

        awaiting = self._task_dispatcher.get_awaiting_closure()
        if not awaiting:
            reply_text = "没有待关闭的任务。"
            self.gateway.reply_message(message_id, reply_text)
            self._record_message("assistant", reply_text)
            return

        task_ids = [t.id for t in awaiting]
        results = self._task_dispatcher.close_tasks(task_ids)
        reply_text = "\n".join(results)
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
