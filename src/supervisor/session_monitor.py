"""Monitor all Claude Code sessions in the container.

Scans ~/.claude/projects/ for .jsonl transcript files and
~/.claude/sessions/*.tmp for session summaries, providing
structured metadata — including task descriptions, current activity,
and progress steps — about each discovered session.
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Base directories for Claude session data
_CLAUDE_DIR = Path.home() / ".claude"
_PROJECTS_DIR = _CLAUDE_DIR / "projects"
_SESSIONS_DIR = _CLAUDE_DIR / "sessions"

# A session is considered "active" if its file was modified within this window
_ACTIVE_THRESHOLD_SECONDS = 120  # 2 minutes


def _safe_read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file, skipping corrupt lines. Returns list of parsed dicts."""
    entries: list[dict] = []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.debug("Corrupt JSONL line %d in %s", lineno, path)
    except PermissionError:
        logger.warning("Permission denied reading %s", path)
    except OSError as e:
        logger.warning("Error reading %s: %s", path, e)
    return entries


def _extract_human_texts(entries: list[dict]) -> list[str]:
    """Extract all human text messages (not tool_result) from entries."""
    texts: list[str] = []
    for entry in entries:
        if entry.get("type") not in ("user", "human"):
            continue
        msg = entry.get("message", {})
        if not isinstance(msg, dict):
            continue
        content = msg.get("content", [])
        if isinstance(content, str):
            text = content.strip()
            if text:
                texts.append(text)
            continue
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text", "").strip()
                    if text:
                        texts.append(text)
    return texts


def _extract_recent_tool_calls(entries: list[dict], tail: int = 30) -> list[dict]:
    """Extract recent tool_use calls from the last *tail* entries."""
    tools: list[dict] = []
    for entry in entries[-tail:]:
        msg = entry.get("message", {})
        if not isinstance(msg, dict):
            continue
        for block in msg.get("content", []):
            if isinstance(block, dict) and block.get("type") == "tool_use":
                tools.append({
                    "name": block.get("name", "unknown"),
                    "input_snippet": str(block.get("input", {}))[:120],
                })
    return tools


def _extract_last_assistant_text(entries: list[dict], tail: int = 10) -> str:
    """Extract the last assistant text snippet from recent entries."""
    for entry in reversed(entries[-tail:]):
        msg = entry.get("message", {})
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "assistant":
            continue
        for block in msg.get("content", []):
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "").strip()
                if text:
                    return text[:200]
    return ""


def _extract_session_metadata(path: Path, entries: list[dict]) -> dict:
    """Extract metadata from parsed JSONL entries for a single session file."""
    session_id = path.stem
    message_count = 0
    tool_calls: list[str] = []
    last_activity: Optional[float] = None
    first_activity: Optional[float] = None
    human_count = 0
    assistant_count = 0

    for entry in entries:
        message_count += 1
        entry_type = entry.get("type", "")

        # Track timestamps — use the file mtime as fallback
        ts = entry.get("timestamp")
        if ts is not None:
            try:
                ts_float = float(ts)
                if first_activity is None or ts_float < first_activity:
                    first_activity = ts_float
                if last_activity is None or ts_float > last_activity:
                    last_activity = ts_float
            except (ValueError, TypeError):
                pass

        if entry_type in ("user", "human"):
            human_count += 1
        elif entry_type == "assistant":
            assistant_count += 1
            # Extract tool calls from assistant messages
            msg = entry.get("message", {})
            if isinstance(msg, dict):
                for block in msg.get("content", []):
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        tool_name = block.get("name", "unknown")
                        if tool_name not in tool_calls:
                            tool_calls.append(tool_name)

    # Fall back to file modification time
    try:
        stat = path.stat()
        file_mtime = stat.st_mtime
    except OSError:
        file_mtime = None

    if last_activity is None:
        last_activity = file_mtime
    if first_activity is None:
        first_activity = file_mtime

    # Derive the project path from the parent directory name
    project_path = path.parent.name

    # --- Rich task extraction ---
    human_texts = _extract_human_texts(entries)
    recent_tools = _extract_recent_tool_calls(entries)
    last_assistant_text = _extract_last_assistant_text(entries)

    # Task description = first human message (the initial request)
    task_description = human_texts[0] if human_texts else ""
    # Current request = last human message (what it's working on now)
    current_request = human_texts[-1] if human_texts else ""

    # Progress: list all human requests as steps
    progress_steps = human_texts

    # Current activity: last tool being called
    last_tool = recent_tools[-1] if recent_tools else None

    # Is actively running: file modified within threshold
    is_active = False
    if file_mtime is not None:
        is_active = (time.time() - file_mtime) < _ACTIVE_THRESHOLD_SECONDS

    return {
        "session_id": session_id,
        "project_path": project_path,
        "file_path": str(path),
        "message_count": message_count,
        "human_messages": human_count,
        "assistant_messages": assistant_count,
        "tool_calls": tool_calls,
        "first_activity": first_activity,
        "last_activity": last_activity,
        # Rich task info
        "task_description": task_description,
        "current_request": current_request,
        "progress_steps": progress_steps,
        "progress_total": len(progress_steps),
        "last_tool": last_tool,
        "last_assistant_text": last_assistant_text,
        "is_active": is_active,
    }


def _scan_jsonl_files(projects_dir: Optional[Path] = None) -> list[Path]:
    """Recursively find all .jsonl files under the projects directory."""
    base = projects_dir or _PROJECTS_DIR
    if not base.is_dir():
        logger.debug("Projects directory does not exist: %s", base)
        return []
    try:
        return sorted(base.rglob("*.jsonl"))
    except PermissionError:
        logger.warning("Permission denied scanning %s", base)
        return []
    except OSError as e:
        logger.warning("Error scanning %s: %s", base, e)
        return []


def _scan_session_summaries(sessions_dir: Optional[Path] = None) -> dict[str, str]:
    """Read *.tmp files from the sessions directory.

    Returns a mapping of session_id (stem) -> summary text.
    """
    base = sessions_dir or _SESSIONS_DIR
    summaries: dict[str, str] = {}
    if not base.is_dir():
        logger.debug("Sessions directory does not exist: %s", base)
        return summaries
    try:
        for tmp_file in base.glob("*.tmp"):
            try:
                text = tmp_file.read_text(encoding="utf-8", errors="replace").strip()
                summaries[tmp_file.stem] = text
            except PermissionError:
                logger.warning("Permission denied reading %s", tmp_file)
            except OSError as e:
                logger.warning("Error reading %s: %s", tmp_file, e)
    except OSError as e:
        logger.warning("Error scanning %s: %s", base, e)
    return summaries


# ── Public API ──


def list_sessions(
    projects_dir: Optional[Path] = None,
    sessions_dir: Optional[Path] = None,
) -> list[dict]:
    """Return all discovered sessions with metadata.

    Each session dict contains:
        session_id, project_path, file_path, message_count,
        human_messages, assistant_messages, tool_calls,
        first_activity, last_activity, summary (if available)
    """
    jsonl_files = _scan_jsonl_files(projects_dir)
    summaries = _scan_session_summaries(sessions_dir)

    sessions: list[dict] = []
    for path in jsonl_files:
        entries = _safe_read_jsonl(path)
        meta = _extract_session_metadata(path, entries)
        # Attach summary if one exists
        meta["summary"] = summaries.get(meta["session_id"], "")
        sessions.append(meta)

    # Sort by last_activity descending (most recent first), None last
    sessions.sort(key=lambda s: s.get("last_activity") or 0, reverse=True)
    return sessions


def get_session_detail(
    session_id: str,
    projects_dir: Optional[Path] = None,
    sessions_dir: Optional[Path] = None,
) -> dict:
    """Return detailed info for a specific session.

    Returns an empty dict if the session is not found.
    """
    jsonl_files = _scan_jsonl_files(projects_dir)
    summaries = _scan_session_summaries(sessions_dir)

    for path in jsonl_files:
        if path.stem == session_id:
            entries = _safe_read_jsonl(path)
            meta = _extract_session_metadata(path, entries)
            meta["summary"] = summaries.get(session_id, "")
            # Include raw entry count for detail view
            meta["raw_entry_count"] = len(entries)
            return meta

    return {}


def get_active_sessions(
    threshold_minutes: int = 30,
    projects_dir: Optional[Path] = None,
    sessions_dir: Optional[Path] = None,
) -> list[dict]:
    """Return sessions whose last activity is within *threshold_minutes* of now."""
    cutoff = time.time() - (threshold_minutes * 60)
    all_sessions = list_sessions(projects_dir, sessions_dir)
    return [
        s for s in all_sessions
        if s.get("last_activity") is not None and s["last_activity"] >= cutoff
    ]


def _format_timestamp(ts: Optional[float]) -> str:
    """Format a Unix timestamp for display, or return 'N/A'."""
    if ts is None:
        return "N/A"
    try:
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    except (OSError, ValueError, OverflowError):
        return "N/A"


def _minutes_ago(ts: Optional[float]) -> str:
    """Human-readable 'X min ago' string."""
    if ts is None:
        return "unknown"
    diff = time.time() - ts
    if diff < 60:
        return "just now"
    minutes = int(diff / 60)
    if minutes < 60:
        return f"{minutes}m ago"
    hours = minutes // 60
    remaining = minutes % 60
    return f"{hours}h {remaining}m ago"


def _status_icon(session: dict) -> str:
    """Return a status icon for the session."""
    if session.get("is_active"):
        return "[ACTIVE]"
    return "[IDLE]"


def _truncate(text: str, max_len: int = 60) -> str:
    """Truncate text with ellipsis."""
    if len(text) <= max_len:
        return text
    return text[:max_len - 3] + "..."


def get_sessions_text(
    projects_dir: Optional[Path] = None,
    sessions_dir: Optional[Path] = None,
) -> str:
    """Formatted text output with task descriptions and progress."""
    sessions = list_sessions(projects_dir, sessions_dir)

    if not sessions:
        return "No Claude Code sessions found."

    # Separate active vs idle
    active = [s for s in sessions if s.get("is_active")]
    idle = [s for s in sessions if not s.get("is_active")]

    lines: list[str] = []
    lines.append(f"Claude Sessions: {len(active)} active, {len(idle)} idle")
    lines.append("=" * 44)

    for i, s in enumerate(active + idle, 1):
        icon = _status_icon(s)
        sid = s["session_id"][:8]
        lines.append(f"\n{icon} [{i}] {sid} | {s['project_path']}")

        # Task description
        task_desc = s.get("task_description", "")
        if task_desc:
            lines.append(f"  Task: {_truncate(task_desc, 70)}")

        # Current request (if different from task description)
        current = s.get("current_request", "")
        if current and current != task_desc:
            lines.append(f"  Now:  {_truncate(current, 70)}")

        # Progress: X steps completed
        total_steps = s.get("progress_total", 0)
        if total_steps > 1:
            lines.append(f"  Progress: {total_steps} requests processed")

        # Current activity (last tool call)
        last_tool = s.get("last_tool")
        if last_tool and s.get("is_active"):
            tool_name = last_tool["name"]
            snippet = _truncate(last_tool["input_snippet"], 60)
            lines.append(f"  Running: {tool_name}({snippet})")

        # Last assistant output snippet (for active sessions)
        if s.get("is_active"):
            last_text = s.get("last_assistant_text", "")
            if last_text:
                lines.append(f"  Output: {_truncate(last_text, 80)}")

        # Timing
        lines.append(
            f"  Messages: {s['message_count']} | "
            f"Last: {_minutes_ago(s['last_activity'])}"
        )

        if s.get("summary"):
            summary = _truncate(s["summary"], 120)
            lines.append(f"  Summary: {summary}")

    return "\n".join(lines)
