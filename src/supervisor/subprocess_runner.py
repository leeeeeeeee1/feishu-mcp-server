"""Claude CLI subprocess execution.

Handles building commands and executing `claude -p` via streaming and
non-streaming modes, with timeout handling and fallback logic.

Circular dependency note: task_dispatcher re-exports from this module,
and this module late-imports task_dispatcher inside functions to access
_set_status, _save_checkpoint, _clear_checkpoint, _STREAM_LIMIT, and
SUPERVISOR_TASK_TIMEOUT. This is intentional — do not convert to
top-level imports or the circular import will fail at load time.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Optional

from .patterns import _looks_like_needs_input

logger = logging.getLogger(__name__)


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


async def _run_claude_streaming(task, env: dict) -> bool:
    """Try to execute task via streaming. Returns True on success, False if fallback needed."""
    from . import task_dispatcher as _td

    cmd = _build_cmd_streaming(task.prompt, session_id=task.session_id or None)

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=_td._STREAM_LIMIT,
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

    async def _read_stream():
        nonlocal accumulated_text
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
                        _td._save_checkpoint(task)
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

    try:
        await asyncio.wait_for(_read_stream(), timeout=_td.SUPERVISOR_TASK_TIMEOUT)
    except asyncio.TimeoutError:
        _td._save_checkpoint(task)
        try:
            proc.kill()
            await proc.wait()
        except ProcessLookupError:
            pass
        task.error = f"Timed out after {_td.SUPERVISOR_TASK_TIMEOUT}s (streaming)"
        logger.warning("Task %s: streaming timed out after %ds", task.id[:8], _td.SUPERVISOR_TASK_TIMEOUT)
        return False
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


async def _run_claude_non_streaming(task, env: dict) -> bool:
    """Execute task via non-streaming JSON mode. Returns True on success."""
    from . import task_dispatcher as _td

    cmd = _build_cmd(task.prompt, session_id=task.session_id or None)
    task.current_step = "Running (non-streaming fallback)..."

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=_td._STREAM_LIMIT,
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


async def _run_claude(task) -> None:
    """Execute `claude -p` for a task.

    Strategy: try streaming first (for progress tracking), fall back to
    non-streaming if streaming crashes (e.g. HTTP chunk size errors).

    NOTE: Calls _run_claude_streaming/_run_claude_non_streaming via the
    task_dispatcher module to allow tests to mock them at that level.
    """
    from . import task_dispatcher as _td

    env = _build_env()

    _td._set_status(task, "running")
    task.current_step = "Starting Claude..."
    task.started_at = time.time()

    # Try streaming first (via task_dispatcher for mockability)
    success = await _td._run_claude_streaming(task, env)

    if not success:
        # Streaming failed — retry with non-streaming
        logger.info(
            "Task %s: streaming failed, retrying non-streaming. error=%s",
            task.id[:8], (task.error or "")[:200],
        )
        task.error = ""  # clear streaming error
        success = await _td._run_claude_non_streaming(task, env)

    if not success:
        task.finished_at = time.time()
        _td._set_status(task, "failed")
        return

    task.finished_at = time.time()
    task.current_step = "Finished"
    _td._clear_checkpoint(task.id)

    # Check if Claude is asking for input
    if _looks_like_needs_input(task.result):
        _td._set_status(task, "waiting_for_input")
        task.current_step = "Waiting for user input"
    else:
        _td._set_status(task, "awaiting_closure")
        task.current_step = "Done — awaiting user confirmation to close"


async def _follow_up_streaming(task, user_input: str, env: dict) -> Optional[str]:
    """Try follow-up via streaming. Returns result text or None on failure."""
    from . import task_dispatcher as _td

    cmd = _build_cmd_streaming(user_input, session_id=task.session_id)

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=_td._STREAM_LIMIT,
            env=env,
            cwd=task.cwd or None,
        )
    except Exception as exc:
        logger.warning("Follow-up streaming spawn failed: %s", exc)
        return None

    accumulated_text = ""

    async def _read_follow_up():
        nonlocal accumulated_text
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

    try:
        await asyncio.wait_for(_read_follow_up(), timeout=_td.SUPERVISOR_TASK_TIMEOUT)
    except asyncio.TimeoutError:
        try:
            proc.kill()
            await proc.wait()
        except ProcessLookupError:
            pass
        logger.warning("Follow-up streaming timed out after %ds", _td.SUPERVISOR_TASK_TIMEOUT)
        return None
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


async def _follow_up_non_streaming(task, user_input: str, env: dict) -> Optional[str]:
    """Follow-up via non-streaming JSON mode (fallback)."""
    from . import task_dispatcher as _td

    cmd = _build_cmd(user_input, session_id=task.session_id)

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=_td._STREAM_LIMIT,
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
