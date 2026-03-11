"""Utility functions for Feishu MCP Server."""

from __future__ import annotations

import json
from typing import Any


def ok(data: Any) -> list:
    """Wrap successful response for MCP tool output."""
    from mcp.types import TextContent

    if isinstance(data, str):
        text = data
    else:
        text = json.dumps(data, ensure_ascii=False, indent=2)
    return [TextContent(type="text", text=text)]


def err(message: str) -> list:
    """Wrap error response for MCP tool output."""
    from mcp.types import TextContent

    return [TextContent(type="text", text=f"Error: {message}")]


def extract_response(resp: Any) -> dict | str:
    """Extract data from a Feishu API response, raising on error."""
    if not resp.success():
        raise RuntimeError(
            f"Feishu API error: code={resp.code}, msg={resp.msg}, "
            f"log_id={resp.get_log_id()}"
        )
    if resp.data is None:
        return "OK"
    # lark-oapi response data objects have a to_dict() or similar
    if hasattr(resp.data, "__dict__"):
        return _clean_dict(resp.data.__dict__)
    return str(resp.data)


def _clean_dict(d: dict) -> dict:
    """Recursively convert objects to dicts for JSON serialization."""
    result = {}
    for k, v in d.items():
        if k.startswith("_"):
            continue
        if hasattr(v, "__dict__") and not isinstance(v, type):
            result[k] = _clean_dict(v.__dict__)
        elif isinstance(v, list):
            result[k] = [
                _clean_dict(item.__dict__) if hasattr(item, "__dict__") else item
                for item in v
            ]
        else:
            result[k] = v
    return result
