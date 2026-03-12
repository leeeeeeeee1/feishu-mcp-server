"""Conversation monitor — periodic dialogue analysis for proactive issue detection.

Analyzes incremental conversation buffers using Sonnet API to detect problems
(errors, stuck tasks, unresolved questions) and notify the user via Feishu.
Supports a two-step confirmation flow before dispatching automatic fixes.

All public functions are pure (no module-level state) for testability.
"""

import asyncio
import json
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# Lazy-imported to avoid hard dependency at module level
from typing import Any
_anthropic_mod: Any = None  # runtime holder for anthropic module

_ANALYSIS_MODEL = "claude-sonnet-4-20250514"

# ── Confirm / Reject detection ──

_CONFIRM_WORDS = frozenset({
    # Chinese
    "修复", "好的", "确认", "开始", "好", "可以", "行", "是", "执行", "同意",
    # English
    "fix", "yes", "ok", "confirm", "start", "go", "sure", "do it",
})

_REJECT_WORDS = frozenset({
    # Chinese
    "不用了", "不需要", "取消", "算了", "不", "不要", "忽略", "跳过",
    # English
    "no", "cancel", "skip", "ignore", "nope", "nevermind", "never mind",
})


def looks_like_confirm(text: str) -> bool:
    """Check if user text looks like a confirmation."""
    normalized = text.strip().lower()
    return normalized in _CONFIRM_WORDS


def looks_like_reject(text: str) -> bool:
    """Check if user text looks like a rejection."""
    normalized = text.strip().lower()
    return normalized in _REJECT_WORDS


# ── Prompt construction ──

_SYSTEM_PROMPT = """你是 AI 大管家的对话质量检测器。分析以下对话记录和任务状态，检测是否存在需要修复的问题。

检测维度：
1. 用户提到的错误/异常是否已被处理
2. 任务是否长时间卡住 (>30min 无进展)
3. 用户是否在等待回复但没收到
4. 对话中是否有未解决的技术问题
5. 是否有重复失败的任务
6. 是否存在需要人工介入的紧急情况

严格返回 JSON 格式，不要输出其他内容：
{
  "has_issues": true/false,
  "issues": [
    {"severity": "HIGH/MEDIUM/LOW", "description": "问题描述", "suggested_fix": "建议修复方案"}
  ],
  "summary": "简短总结"
}

如果没有发现问题，返回：
{"has_issues": false, "issues": [], "summary": "一切正常"}

注意：
- 只报告真正需要关注的问题，不要过度报告
- 正常的对话交互不算问题
- severity: HIGH=需要立即处理, MEDIUM=建议处理, LOW=可选"""


def build_analysis_prompt(
    messages: list[dict],
    active_tasks: list[dict],
    active_sessions: str,
) -> tuple[str, str]:
    """Build system + user prompt for conversation analysis.

    Returns (system_prompt, user_prompt).
    """
    _MAX_MSG_LEN = 500
    parts = ["## 本采样周期对话记录\n"]
    for msg in messages:
        role_label = "用户" if msg.get("role") == "user" else "助手"
        text_safe = msg.get("text", "")[:_MAX_MSG_LEN]
        parts.append(f"[{role_label}] {text_safe}")

    if active_tasks:
        parts.append("\n## 当前活跃任务")
        for task in active_tasks:
            tid = task.get("id", "?")[:8]
            status = task.get("status", "unknown")
            desc = task.get("description", "")
            parts.append(f"- [{tid}] {status}: {desc}")

    if active_sessions:
        parts.append(f"\n## 活跃会话\n{active_sessions}")

    user_prompt = "\n".join(parts)
    return _SYSTEM_PROMPT, user_prompt


# ── Response parsing ──

def parse_analysis_response(response_text: str) -> dict:
    """Parse Sonnet's analysis response JSON. Tolerant of markdown wrappers."""
    if not response_text or not response_text.strip():
        return {"has_issues": False, "issues": [], "summary": ""}

    text = response_text.strip()

    # Strip markdown code block wrapper
    md_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if md_match:
        text = md_match.group(1).strip()

    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        logger.warning("Failed to parse analysis response as JSON: %.200s", text)
        return {"has_issues": False, "issues": [], "summary": ""}

    return {
        "has_issues": bool(data.get("has_issues", False)),
        "issues": data.get("issues", []),
        "summary": data.get("summary", ""),
    }


# ── Notification formatting ──

def format_issue_notification(issues: list[dict]) -> str:
    """Format issues as a user-facing Feishu notification message."""
    if not issues:
        return ""

    count = len(issues)
    lines = [f"🔍 对话监控发现 {count} 个问题：\n"]

    for i, issue in enumerate(issues, 1):
        severity = issue.get("severity", "MEDIUM")
        desc = issue.get("description", "")
        fix = issue.get("suggested_fix", "")
        lines.append(f"{i}. [{severity}] {desc}")
        if fix:
            lines.append(f"   建议: {fix}")

    lines.append("\n回复 '修复' 开始处理，或忽略此消息。")
    return "\n".join(lines)


def format_fix_plan(issues: list[dict]) -> str:
    """Format a fix plan for the second confirmation step."""
    if not issues:
        return ""

    lines = ["修复计划：\n"]
    for i, issue in enumerate(issues, 1):
        fix = issue.get("suggested_fix", "无具体方案")
        desc = issue.get("description", "")
        lines.append(f"步骤 {i}: {fix}")
        lines.append(f"  (针对问题: {desc})")

    return "\n".join(lines)


# ── Main analysis function ──

async def analyze_conversation(
    messages: list[dict],
    active_tasks: list[dict],
    active_sessions: str,
    api_key: str,
) -> dict:
    """Analyze conversation buffer for issues using Sonnet API.

    If messages is empty, returns no-issues immediately (no API call).

    Returns:
        {
            "has_issues": bool,
            "issues": [{"severity": str, "description": str, "suggested_fix": str}],
            "summary": str,
        }
    """
    if not messages:
        return {"has_issues": False, "issues": [], "summary": "无新对话"}

    # Lazy import anthropic
    global _anthropic_mod
    if _anthropic_mod is None:
        try:
            import anthropic as _anthropic
            _anthropic_mod = _anthropic
        except ImportError:
            logger.warning("anthropic package not installed, skipping analysis")
            return {"has_issues": False, "issues": [], "summary": "anthropic unavailable"}

    system_prompt, user_prompt = build_analysis_prompt(
        messages, active_tasks, active_sessions,
    )

    try:
        client = _anthropic_mod.AsyncAnthropic(api_key=api_key)
        response = await asyncio.wait_for(
            client.messages.create(
                model=_ANALYSIS_MODEL,
                max_tokens=1024,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            ),
            timeout=30,
        )

        text_blocks = [b.text for b in response.content if hasattr(b, "text")]
        result_text = text_blocks[0] if text_blocks else ""
        logger.info(
            "Conversation analysis: %d input tokens, %d output tokens",
            response.usage.input_tokens, response.usage.output_tokens,
        )
        return parse_analysis_response(result_text)

    except asyncio.TimeoutError:
        logger.warning("Conversation analysis timed out (30s)")
        return {"has_issues": False, "issues": [], "summary": "analysis timeout"}
    except Exception as e:
        error_type = type(e).__name__
        logger.error("Conversation analysis failed: %s", error_type)
        return {"has_issues": False, "issues": [], "summary": f"error: {error_type}"}
