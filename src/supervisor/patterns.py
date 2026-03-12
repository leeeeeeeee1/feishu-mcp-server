"""Pattern matching and heuristics for close/input intent detection.

Provides fast local detection of user intent (close task, provide input)
without requiring a Sonnet routing call. Used by task_dispatcher and main.py.
"""

import re as _re

# Phrases that hint Claude is asking for user input
_INPUT_PHRASES = ("please", "which", "should i", "confirm")


def _looks_like_needs_input(text: str) -> bool:
    """Heuristic: does the output look like it needs user input?"""
    if "?" not in text:
        return False
    lower = text.lower()
    return any(phrase in lower for phrase in _INPUT_PHRASES)


# Close intent detection — used by main.py for reply-based quick close (Step 0).
# These provide a fast local fallback for Feishu thread replies where Sonnet
# routing would add unnecessary latency for obvious acknowledgements.
_CLOSE_PHRASES = (
    "好的", "收到", "ok", "谢谢", "thanks", "可以了", "没问题",
    "完成", "done", "lgtm", "不用了", "就这样", "👍", "thank you",
)

_CLOSE_PHRASES_SET = frozenset(p.lower() for p in _CLOSE_PHRASES)

# Technical nouns — if these appear near 关闭/关掉/结束, it's NOT task closure
_TECHNICAL_NOUNS = r"连接|端口|服务|进程|窗口|文件|通道|线程|循环|socket|server|session|db|数据库|nginx|redis"

_CLOSE_FALSE_POSITIVES = _re.compile(
    rf"关闭({_TECHNICAL_NOUNS})"
    rf"|关掉({_TECHNICAL_NOUNS})"
    rf"|结束({_TECHNICAL_NOUNS})"
    rf"|({_TECHNICAL_NOUNS})关掉"
    rf"|({_TECHNICAL_NOUNS})关闭"
    rf"|({_TECHNICAL_NOUNS})结束",
    _re.IGNORECASE,
)

_CLOSE_INTENT_PATTERNS = [
    _re.compile(r"关闭(了|吧|这个|那个)"),
    _re.compile(r"关了"),
    _re.compile(r"关掉"),
    _re.compile(r"结束(吧|了|掉|这个|那个|任务)"),
    _re.compile(r"不用了"),
    _re.compile(r"完事了"),
    _re.compile(r"可以关了"),
    _re.compile(r"\bclose\b", _re.IGNORECASE),
    _re.compile(r"\bdone with it\b", _re.IGNORECASE),
]


def _contains_close_intent(text: str) -> bool:
    """Detect close intent in longer text (not just short phrases).

    Unlike _looks_like_close (≤10 char exact match), this works on any
    length text. Conservative: excludes technical phrases like "关闭连接".
    Note: the false-positive guard only covers post-verb position.
    """
    if not text or not text.strip():
        return False
    if _CLOSE_FALSE_POSITIVES.search(text):
        return False
    return any(pat.search(text) for pat in _CLOSE_INTENT_PATTERNS)


def _looks_like_close(text: str) -> bool:
    """Heuristic: short acknowledgement → close intent, not follow-up.

    Uses exact match after normalization to avoid false positives.
    """
    stripped = text.strip()
    if not stripped:
        return False
    if "?" in stripped or "？" in stripped or "吗" in stripped:
        return False
    normalized = stripped.lower().rstrip("。.!！~，,").strip()
    if len(normalized) > 10:
        return False
    return normalized in _CLOSE_PHRASES_SET
