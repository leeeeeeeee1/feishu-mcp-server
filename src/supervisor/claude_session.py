"""Claude Code CLI session manager with streaming support.

Wraps `claude -p` with session persistence, stream-json parsing,
and configurable model/effort/permissions.
"""

import asyncio
import json
import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, Callable, Optional

from . import route_parser

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
                limit=10 * 1024 * 1024,  # 10 MB — match task_dispatcher
                env=env,
                cwd=cwd,
            )

            if self._process.stdout is None:
                raise RuntimeError("stdout pipe not opened")
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
                limit=10 * 1024 * 1024,  # 10 MB — match task_dispatcher
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

    # Valid actions that sonnet can return (delegated to route_parser, kept as alias for tests)
    _VALID_ACTIONS = route_parser.VALID_ACTIONS

    # Route model — sonnet via direct API (fast) or CLI fallback
    _ROUTE_MODEL = "claude-sonnet-4-6"

    async def route_message(self, text: str, system_prompt: str, user_prompt: str) -> dict:
        """Route a user message using sonnet — classify AND generate response in one call.

        Args:
            text: The user message (for fallback description).
            system_prompt: Stable routing rules (cacheable).
            user_prompt: Dynamic context + user message.

        Returns a dict with:
            - {"action": "reply", "text": "..."} for direct responses
            - {"action": "dispatch", "description": "..."} for single tasks
            - {"action": "orchestrate", "description": "...", "subtasks": [...]}
            - {"action": "follow_up", "task_id": "...", "text": "..."} for task follow-ups

        Falls back to action=dispatch on any error (safe default).
        """
        logger.info("Routing message: %s", text[:80])

        # Try direct API first (much faster, no CLI overhead)
        result_text = await self._route_via_api(system_prompt, user_prompt)

        # Fallback to CLI if API not available
        if result_text is None:
            combined = system_prompt + "\n" + user_prompt
            result_text = await self._route_via_cli(text, combined)

        if result_text is None:
            return {"action": "dispatch", "description": text[:80]}

        result_text = route_parser.strip_markdown_wrapper(result_text)
        return route_parser.parse_route_response(result_text, text)

    @staticmethod
    def _resolve_api_key() -> str:
        """Resolve Anthropic API key from environment or Claude Code config.

        Checks in order:
        1. ANTHROPIC_API_KEY environment variable
        2. primaryApiKey in ~/.claude.json (Claude Code's stored key)
        """
        key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if key:
            return key

        try:
            config_path = Path.home() / ".claude.json"
            if config_path.exists():
                data = json.loads(config_path.read_text())
                key = data.get("primaryApiKey", "").strip()
                if key:
                    return key
        except (OSError, json.JSONDecodeError, TypeError):
            pass

        return ""

    async def _route_via_api(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """Route via direct Anthropic API call (fast, no CLI overhead).

        Args:
            system_prompt: Stable routing rules (cacheable).
            user_prompt: Dynamic context + user message.

        Returns the response text, or None if API is not available.
        """
        try:
            import anthropic
        except ImportError:
            return None

        api_key = self._resolve_api_key()
        if not api_key:
            return None

        try:
            client = anthropic.AsyncAnthropic(api_key=api_key)
            response = await asyncio.wait_for(
                client.messages.create(
                    model=self._ROUTE_MODEL,
                    max_tokens=2048,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                ),
                timeout=30,
            )
            text_blocks = [b.text for b in response.content if hasattr(b, "text")]
            result = text_blocks[0] if text_blocks else ""
            logger.info("Route via API: %d input tokens, %d output tokens",
                        response.usage.input_tokens, response.usage.output_tokens)
            return result
        except asyncio.TimeoutError:
            logger.warning("Route API timed out (30s)")
            return None
        except Exception as e:
            logger.info("Route API unavailable (%s), falling back to CLI", type(e).__name__)
            return None

    async def _route_via_cli(self, text: str, combined_prompt: str) -> Optional[str]:
        """Route via claude CLI (fallback when API is not available).

        Returns the result text, or None on failure.
        """
        cmd = [
            "claude", "-p", combined_prompt,
            "--model", "sonnet",
            "--effort", "low",
            "--output-format", "json",
            "--allowedTools", "",
        ]

        env = _build_env()

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=60)
        except asyncio.TimeoutError:
            logger.warning("Route CLI timed out (60s)")
            return None
        except (FileNotFoundError, OSError) as e:
            logger.warning("Route CLI failed: %s", e)
            return None

        try:
            data = json.loads(stdout.decode("utf-8"))
        except (json.JSONDecodeError, TypeError) as e:
            raw = stdout.decode("utf-8", errors="replace")[:200]
            logger.warning("Route CLI parse error: %s, raw: %s", e, raw)
            return None

        return data.get("result", "")

    # Backward-compatible delegation: instance methods call module-level functions
    # so that existing tests using session._try_json_parse() etc. still work.
    @staticmethod
    def _try_json_parse(text: str) -> Optional[dict]:
        return route_parser._try_json_parse(text)

    @staticmethod
    def _try_regex_extract(text: str) -> Optional[dict]:
        return route_parser._try_regex_extract(text)

    @staticmethod
    def _extract_field_value(text: str, field: str) -> str:
        return route_parser._extract_field_value(text, field)

    @staticmethod
    def _normalize_action(parsed: dict) -> dict:
        return route_parser._normalize_action(parsed)

    @staticmethod
    def _strip_markdown_wrapper(text: str) -> str:
        return route_parser.strip_markdown_wrapper(text)

    async def resume(self, prompt: str, session_id: str, cwd: Optional[str] = None) -> str:
        """Resume a specific session with a new prompt (for worker pause/resume)."""
        old_sid = self.session_id
        self.session_id = session_id
        try:
            return await self.call(prompt, cwd=cwd)
        finally:
            self.session_id = old_sid
