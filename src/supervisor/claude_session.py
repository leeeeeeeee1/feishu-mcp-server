"""Claude Code CLI session manager with streaming support.

Wraps `claude -p` with session persistence, stream-json parsing,
and configurable model/effort/permissions.
"""

import asyncio
import json
import os
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, Callable, Optional

logger = logging.getLogger(__name__)

# Persistent file to store session ID across restarts
_SESSION_FILE = Path.home() / ".supervisor" / "main_session_id"

# Default CLI flags
DEFAULT_MODEL = os.environ.get("SUPERVISOR_CLAUDE_MODEL", "opus")
DEFAULT_EFFORT = "max"
DEFAULT_PERMISSION_MODE = "bypassPermissions"


@dataclass
class StreamEvent:
    """A parsed event from claude stream-json output."""
    type: str           # "system", "assistant", "result"
    subtype: str = ""   # "init", "success", "error", etc.
    text: str = ""      # accumulated text content
    session_id: str = ""
    is_final: bool = False
    raw: dict = field(default_factory=dict)


def _build_cmd(
    prompt: str,
    session_id: Optional[str] = None,
    model: Optional[str] = None,
    streaming: bool = True,
    system_prompt: Optional[str] = None,
    cwd: Optional[str] = None,
) -> list[str]:
    """Build the claude CLI command."""
    cmd = [
        "claude", "-p", prompt,
        "--model", model or DEFAULT_MODEL,
        "--effort", DEFAULT_EFFORT,
        "--permission-mode", DEFAULT_PERMISSION_MODE,
    ]

    if streaming:
        cmd.extend(["--output-format", "stream-json", "--verbose"])
    else:
        cmd.extend(["--output-format", "json"])

    if session_id:
        cmd.extend(["--resume", session_id])

    if system_prompt:
        cmd.extend(["--append-system-prompt", system_prompt])

    return cmd


def _build_env() -> dict[str, str]:
    """Build environment for subprocess, removing CLAUDECODE to avoid nesting check."""
    return {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}


def _parse_stream_line(line: str) -> Optional[StreamEvent]:
    """Parse a single line of stream-json output into a StreamEvent."""
    line = line.strip()
    if not line:
        return None

    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        logger.warning("Failed to parse stream line: %s", line[:200])
        return None

    event_type = data.get("type", "")
    session_id = data.get("session_id", "")

    if event_type == "assistant":
        # Extract text from message content
        msg = data.get("message", {})
        content_blocks = msg.get("content", [])
        text = ""
        for block in content_blocks:
            if block.get("type") == "text":
                text += block.get("text", "")
        return StreamEvent(
            type="assistant",
            text=text,
            session_id=session_id,
            raw=data,
        )

    elif event_type == "result":
        return StreamEvent(
            type="result",
            subtype=data.get("subtype", ""),
            text=data.get("result", ""),
            session_id=session_id,
            is_final=True,
            raw=data,
        )

    elif event_type == "system":
        return StreamEvent(
            type="system",
            subtype=data.get("subtype", ""),
            session_id=session_id,
            raw=data,
        )

    return StreamEvent(type=event_type, session_id=session_id, raw=data)


class ClaudeSession:
    """Manages a persistent Claude Code CLI session."""

    def __init__(
        self,
        session_id: Optional[str] = None,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ):
        self.session_id = session_id or self._load_session_id()
        self.model = model
        self.system_prompt = system_prompt
        self._process: Optional[asyncio.subprocess.Process] = None

    def _load_session_id(self) -> Optional[str]:
        """Load session ID from persistent file."""
        try:
            if _SESSION_FILE.exists():
                sid = _SESSION_FILE.read_text().strip()
                if sid:
                    logger.info("Loaded session ID: %s", sid)
                    return sid
        except OSError:
            pass
        return None

    def _save_session_id(self, session_id: str):
        """Persist session ID to file."""
        try:
            _SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)
            _SESSION_FILE.write_text(session_id)
            logger.info("Saved session ID: %s", session_id)
        except OSError as e:
            logger.warning("Failed to save session ID: %s", e)

    async def call_streaming(
        self,
        prompt: str,
        on_text: Optional[Callable[[str], None]] = None,
        cwd: Optional[str] = None,
    ) -> AsyncIterator[StreamEvent]:
        """Call Claude with streaming, yielding events as they arrive.

        Args:
            prompt: The user message to send.
            on_text: Optional callback for each text chunk.
            cwd: Optional working directory for the subprocess.

        Yields:
            StreamEvent objects as they arrive.
        """
        cmd = _build_cmd(
            prompt=prompt,
            session_id=self.session_id,
            model=self.model,
            streaming=True,
            system_prompt=self.system_prompt,
        )

        logger.info("Starting streaming claude: %s", " ".join(cmd[:6]) + "...")
        env = _build_env()

        try:
            self._process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=cwd,
            )

            async for line in self._process.stdout:
                decoded = line.decode("utf-8", errors="replace")
                event = _parse_stream_line(decoded)
                if event is None:
                    continue

                # Capture session ID from first event that has it
                if event.session_id and not self.session_id:
                    self.session_id = event.session_id
                    self._save_session_id(event.session_id)

                if event.type == "assistant" and event.text and on_text:
                    on_text(event.text)

                yield event

                if event.is_final:
                    break

            await self._process.wait()

        except Exception as e:
            logger.error("Stream error: %s", e)
            yield StreamEvent(
                type="result",
                subtype="error",
                text=f"Claude session error: {e}",
                is_final=True,
            )

    async def call(self, prompt: str, cwd: Optional[str] = None, _retry: bool = True) -> str:
        """Call Claude and return the full response text (non-streaming).

        Args:
            prompt: The user message to send.
            cwd: Optional working directory.

        Returns:
            The response text.
        """
        cmd = _build_cmd(
            prompt=prompt,
            session_id=self.session_id,
            model=self.model,
            streaming=False,
            system_prompt=self.system_prompt,
        )

        env = _build_env()
        logger.info("Calling claude (non-stream): %s", prompt[:100])

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=cwd,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=600)
        except asyncio.TimeoutError:
            proc.kill()
            return "Claude Code response timed out (10 min limit)"
        except FileNotFoundError:
            return "Error: 'claude' command not found. Is Claude Code installed?"

        if proc.returncode != 0:
            err = stderr.decode("utf-8", errors="replace").strip()
            # Session expired — retry without session (once only)
            if _retry and self.session_id and ("session" in err.lower() or "not found" in err.lower()):
                logger.warning("Session expired, retrying without session ID")
                self.session_id = None
                return await self.call(prompt, cwd=cwd, _retry=False)
            return f"Claude Code error: {err or 'unknown error'}"

        # Parse JSON result
        try:
            data = json.loads(stdout.decode("utf-8"))
            sid = data.get("session_id", "")
            if sid and sid != self.session_id:
                self.session_id = sid
                self._save_session_id(sid)
            return data.get("result", "") or "(empty response)"
        except json.JSONDecodeError:
            text = stdout.decode("utf-8").strip()
            return text or "(empty response)"

    # Valid actions that sonnet can return
    _VALID_ACTIONS = frozenset({"reply", "dispatch", "dispatch_multi", "follow_up"})

    async def route_message(self, text: str, route_prompt: str) -> dict:
        """Route a user message using sonnet — classify AND generate response in one call.

        Args:
            text: The user message.
            route_prompt: The full routing prompt (from build_route_prompt).

        Returns a dict with:
            - {"action": "reply", "text": "..."} for direct responses
            - {"action": "dispatch", "description": "..."} for single tasks
            - {"action": "dispatch_multi", "description": "...", "subtasks": [...]}
            - {"action": "follow_up", "task_id": "...", "text": "..."} for task follow-ups

        Falls back to action=dispatch on any error (safe default).
        """
        cmd = [
            "claude", "-p", route_prompt,
            "--model", "sonnet",
            "--output-format", "json",
        ]

        env = _build_env()
        logger.info("Routing message: %s", text[:80])

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=60)
        except asyncio.TimeoutError:
            logger.warning("Route call timed out (60s), defaulting to dispatch")
            return {"action": "dispatch", "description": text[:80]}
        except (FileNotFoundError, OSError) as e:
            logger.warning("Route call failed: %s, defaulting to dispatch", e)
            return {"action": "dispatch", "description": text[:80]}

        # ── Parse outer JSON from `claude --output-format json`
        try:
            data = json.loads(stdout.decode("utf-8"))
        except (json.JSONDecodeError, TypeError) as e:
            raw = stdout.decode("utf-8", errors="replace")[:200]
            logger.warning("Route outer parse error: %s, raw: %s", e, raw)
            return {"action": "dispatch", "description": text[:80]}

        result_text = data.get("result", "")
        result_text = self._strip_markdown_wrapper(result_text)

        if not result_text.strip():
            logger.warning("Route returned empty result")
            return {"action": "dispatch", "description": text[:80]}

        # ── Try 1: Standard JSON parse
        parsed = self._try_json_parse(result_text)
        if parsed:
            return parsed

        # ── Try 2: Regex-based extraction (handles unescaped quotes in text)
        parsed = self._try_regex_extract(result_text)
        if parsed:
            return parsed

        # ── Try 3: Plain text → treat as reply (sonnet ignored JSON instruction)
        if not result_text.strip().startswith("{"):
            logger.info("Sonnet returned plain text, treating as reply: %s", result_text[:100])
            return {"action": "reply", "text": result_text.strip()}

        # ── Fallback: dispatch (safe default)
        logger.warning("Could not parse route result, defaulting to dispatch. Raw: %s", result_text[:200])
        return {"action": "dispatch", "description": text[:80]}

    def _try_json_parse(self, text: str) -> Optional[dict]:
        """Try standard JSON parse. Returns parsed dict or None."""
        try:
            parsed = json.loads(text)
            action = parsed.get("action", "")
            if action in self._VALID_ACTIONS:
                return parsed
            logger.warning("Unknown action in route result: %s", action)
        except (json.JSONDecodeError, ValueError):
            pass
        return None

    def _try_regex_extract(self, text: str) -> Optional[dict]:
        """Extract action and fields from malformed JSON using regex.

        Handles cases like unescaped Chinese quotes:
        {"action": "reply", "text": "看到"排队"的现象"}
        """
        # Extract action
        action_match = re.search(r'"action"\s*:\s*"(\w+)"', text)
        if not action_match:
            return None

        action = action_match.group(1)
        if action not in self._VALID_ACTIONS:
            return None

        if action == "reply":
            reply_text = self._extract_field_value(text, "text")
            if reply_text:
                return {"action": "reply", "text": reply_text}

        elif action == "dispatch":
            desc = self._extract_field_value(text, "description")
            return {"action": "dispatch", "description": desc or ""}

        elif action == "follow_up":
            task_id = self._extract_field_value(text, "task_id")
            follow_text = self._extract_field_value(text, "text")
            if task_id:
                return {"action": "follow_up", "task_id": task_id, "text": follow_text or ""}

        elif action == "dispatch_multi":
            desc = self._extract_field_value(text, "description")
            # For dispatch_multi, subtasks are hard to extract with regex — fallback
            return {"action": "dispatch", "description": desc or ""}

        logger.info("Regex extracted action=%s from malformed JSON", action)
        return None

    @staticmethod
    def _extract_field_value(text: str, field: str) -> str:
        """Extract the value of a JSON field from potentially malformed JSON.

        Handles unescaped quotes in values by finding the correct closing quote:
        the LAST " that is followed by } or , (end of field boundary).

        For "text" field (typically the last/longest field), the last "} boundary works.
        For short fields like "task_id", the first ", boundary is correct.

        Strategy: collect all candidate closing positions, then pick the best one.
        - If field is followed by another field (", pattern), use the FIRST such boundary.
        - Otherwise use the LAST "} boundary.
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

    @staticmethod
    def _strip_markdown_wrapper(text: str) -> str:
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

    async def resume(self, prompt: str, session_id: str, cwd: Optional[str] = None) -> str:
        """Resume a specific session with a new prompt (for worker pause/resume)."""
        old_sid = self.session_id
        self.session_id = session_id
        try:
            return await self.call(prompt, cwd=cwd)
        finally:
            self.session_id = old_sid
