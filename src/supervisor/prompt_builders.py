"""Prompt building for worker and orchestrator tasks.

Pure functions that construct enriched prompts from user text,
task descriptions, and conversation history.
"""

from __future__ import annotations

from collections import deque


def get_history_text(conversation_history: deque) -> str:
    """Format recent conversation history as text for worker context."""
    if not conversation_history:
        return ""
    lines = []
    for msg in conversation_history:
        role_label = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role_label}: {msg['text']}")
    return "\n".join(lines)


def build_worker_prompt(text: str, description: str, conversation_history: deque) -> str:
    """Build an enriched prompt for the worker with task context.

    Args:
        text: The original user message.
        description: Sonnet-generated task description.
        conversation_history: Recent conversation deque.

    Returns:
        Enriched prompt string with context for the worker.
    """
    parts = [
        "You are a worker agent in a development container.",
        f"Task: {description}",
        f"User's original request: {text}",
        "Working directory: /workspace",
    ]

    history = get_history_text(conversation_history)
    if history:
        parts.append(f"\n## Recent conversation history\n{history}")

    parts.append(
        "\nExecute this task thoroughly. Be concise in your response.\n"
        "Answer in the same language as the user's request."
    )

    return "\n".join(parts)


def build_orchestrator_prompt(
    text: str, description: str, subtasks: list[str],
    conversation_history: deque,
) -> str:
    """Build a prompt for the orchestrator worker with subagent instructions.

    The orchestrator is a single worker that coordinates multiple subagents
    via Claude's Agent tool, sharing context and managing dependencies.

    Args:
        text: The original user message.
        description: Sonnet-generated task description.
        subtasks: List of subtask descriptions to coordinate.
        conversation_history: Recent conversation deque.

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

    history = get_history_text(conversation_history)
    if history:
        parts.append(f"\n## Recent conversation history\n{history}")

    parts.append(
        "\nExecute this task thoroughly. Be concise in your response.\n"
        "Answer in the same language as the user's request."
    )

    return "\n".join(parts)
