"""Route response parsing for Supervisor Hub.

Parses the raw text output from Sonnet routing calls into structured
action dicts. Handles valid JSON, malformed JSON with unescaped quotes,
markdown-wrapped responses, and plain text fallbacks.

Extracted from claude_session.py — pure functions, no I/O.
"""

import json
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# Valid actions that Sonnet can return
VALID_ACTIONS = frozenset({
    "reply", "dispatch", "orchestrate", "dispatch_multi",
    "follow_up", "close", "close_all",
})

# Action verbs used for plain-text heuristic (dispatch vs reply)
_ACTION_VERBS = re.compile(
    r"(执行|分析|运行|检查|创建|编写|部署|安装|配置|重构|优化|修复"
    r"|run|check|analyze|build|deploy|install|fix|create|write|refactor)",
    re.IGNORECASE,
)


def parse_route_response(result_text: str, fallback_text: str) -> dict:
    """Parse the raw text from a Sonnet routing call into an action dict.

    Args:
        result_text: The raw response text from Sonnet (already stripped of
            markdown wrappers by the caller).
        fallback_text: Original user message, used for fallback descriptions.

    Returns a dict with at minimum an "action" key.
    Falls back to action=dispatch on any parse failure (safe default).
    """
    if not result_text.strip():
        logger.warning("Route returned empty result")
        return {"action": "dispatch", "description": fallback_text[:80]}

    # ── Try 1: Standard JSON parse
    parsed = _try_json_parse(result_text)
    if parsed:
        return _normalize_action(parsed)

    # ── Try 2: Regex-based extraction (handles unescaped quotes in text)
    parsed = _try_regex_extract(result_text)
    if parsed:
        return _normalize_action(parsed)

    # ── Try 3: Plain text heuristic (sonnet ignored JSON instruction)
    if not result_text.strip().startswith("{"):
        plain = result_text.strip()
        if len(plain) < 200 and _ACTION_VERBS.search(plain):
            logger.info("Sonnet returned plain text with action verbs, treating as dispatch: %s", plain[:100])
            return {"action": "dispatch", "description": plain}
        logger.info("Sonnet returned plain text, treating as reply: %s", plain[:100])
        return {"action": "reply", "text": plain}

    # ── Fallback: dispatch (safe default)
    logger.warning("Could not parse route result, defaulting to dispatch. Raw: %s", result_text[:200])
    return {"action": "dispatch", "description": fallback_text[:80]}


def strip_markdown_wrapper(text: str) -> str:
    """Strip outer markdown code block fences (```json ... ```) from text."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove only the opening fence line
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        # Remove only the closing fence line
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _try_json_parse(text: str) -> Optional[dict]:
    """Try standard JSON parse. Returns parsed dict or None."""
    try:
        parsed = json.loads(text)
        action = parsed.get("action", "")
        if action in VALID_ACTIONS:
            return parsed
        logger.warning("Unknown action in route result: %s", action)
    except (json.JSONDecodeError, ValueError):
        pass
    return None


def _try_regex_extract(text: str) -> Optional[dict]:
    """Extract action and fields from malformed JSON using regex.

    Handles cases like unescaped Chinese quotes:
    {"action": "reply", "text": "看到"排队"的现象"}
    """
    # Extract action
    action_match = re.search(r'"action"\s*:\s*"(\w+)"', text)
    if not action_match:
        return None

    action = action_match.group(1)
    if action not in VALID_ACTIONS:
        return None

    if action == "reply":
        reply_text = _extract_field_value(text, "text")
        if reply_text:
            return {"action": "reply", "text": reply_text}

    elif action == "dispatch":
        desc = _extract_field_value(text, "description")
        return {"action": "dispatch", "description": desc or ""}

    elif action == "follow_up":
        task_id = _extract_field_value(text, "task_id")
        follow_text = _extract_field_value(text, "text")
        if task_id:
            return {"action": "follow_up", "task_id": task_id, "text": follow_text or ""}

    elif action in ("orchestrate", "dispatch_multi"):
        desc = _extract_field_value(text, "description")
        subtasks_match = re.search(r'"subtasks"\s*:\s*\[(.*?)\]', text, re.DOTALL)
        if subtasks_match:
            subtasks = re.findall(r'"([^"]+)"', subtasks_match.group(1))
            if subtasks:
                return {"action": "orchestrate", "description": desc or "", "subtasks": subtasks}
        return {"action": "dispatch", "description": desc or ""}

    elif action == "close":
        task_id = _extract_field_value(text, "task_id")
        result: dict = {"action": "close"}
        if task_id:
            result["task_id"] = task_id
        # Also try task_ids array (takes precedence in _handle_sonnet_close)
        ids_match = re.search(r'"task_ids"\s*:\s*\[(.*?)\]', text, re.DOTALL)
        if ids_match:
            task_ids = re.findall(r'"([^"]+)"', ids_match.group(1))
            if task_ids:
                result["task_ids"] = task_ids
        # Require at least one identifier — otherwise fall through to dispatch
        if len(result) == 1:
            return None
        return result

    elif action == "close_all":
        return {"action": "close_all"}

    logger.info("Regex extracted action=%s from malformed JSON", action)
    return None


def _extract_field_value(text: str, field: str) -> str:
    """Extract the value of a JSON field from potentially malformed JSON.

    Handles unescaped quotes in values by finding the correct closing quote:
    the LAST " that is followed by } or , (end of field boundary).
    """
    pattern = rf'"{field}"\s*:\s*"'
    match = re.search(pattern, text)
    if not match:
        return ""

    start = match.end()
    remaining = text[start:]

    # Collect candidates: positions of " followed by , or }
    comma_ends = []  # " followed by , (another field after this)
    brace_ends = []  # " followed by } (end of object)

    for m in re.finditer(r'"', remaining):
        pos = m.start()
        after = remaining[pos + 1:].lstrip()
        if after.startswith(","):
            comma_ends.append(pos)
        elif after.startswith("}") or after == "":
            brace_ends.append(pos)

    # Prefer the first comma boundary (shortest match — field value ends, next field starts)
    if comma_ends:
        return remaining[:comma_ends[0]]

    # Otherwise use the last brace boundary (longest match — last field in object)
    if brace_ends:
        return remaining[:brace_ends[-1]]

    # No clean end found — take everything up to last "
    last_quote = remaining.rfind('"')
    if last_quote > 0:
        return remaining[:last_quote]

    return remaining.rstrip('"}')


def _normalize_action(parsed: dict) -> dict:
    """Normalize legacy action names to current ones.

    dispatch_multi → orchestrate (backward compatibility).
    Returns a new dict to avoid mutating the input.
    """
    if parsed.get("action") == "dispatch_multi":
        return {**parsed, "action": "orchestrate"}
    return dict(parsed)
