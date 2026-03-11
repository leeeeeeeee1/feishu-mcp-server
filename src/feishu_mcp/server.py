"""Feishu MCP Server - Main entry point.

A Model Context Protocol server for Feishu/Lark, providing tools for
drive, documents, and bitable — features NOT covered by the official
@larksuiteoapi/lark-mcp server.

Official Lark MCP covers: im, calendar, contact (removed from this server).

Usage:
    FEISHU_APP_ID=xxx FEISHU_APP_SECRET=xxx feishu-mcp

    # For Lark international:
    FEISHU_DOMAIN=https://open.larksuite.com feishu-mcp

    # Register with Claude Code:
    claude mcp add feishu -- python -m feishu_mcp.server
"""

from __future__ import annotations

import asyncio
import logging

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from feishu_mcp.utils import ok, err
from feishu_mcp.tools import docs, drive, bitable

logger = logging.getLogger("feishu-mcp")

# Collect all tool definitions from modules
_MODULES = [docs, drive, bitable]

# Build tool name -> module mapping
_TOOL_REGISTRY: dict[str, object] = {}
_ALL_TOOLS: list[dict] = []

for _mod in _MODULES:
    for _tool_def in _mod.TOOLS:
        _TOOL_REGISTRY[_tool_def["name"]] = _mod
        _ALL_TOOLS.append(_tool_def)


def _create_server() -> Server:
    server = Server("feishu-mcp-server")

    # Lazy client init — deferred until first tool call so MCP handshake
    # succeeds even if credentials are missing (allows list_tools to work).
    _client_cache: dict = {}

    def _get_client():
        if "client" not in _client_cache:
            from feishu_mcp.auth import create_client
            _client_cache["client"] = create_client()
        return _client_cache["client"]

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name=t["name"],
                description=t["description"],
                inputSchema=t["inputSchema"],
            )
            for t in _ALL_TOOLS
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        mod = _TOOL_REGISTRY.get(name)
        if mod is None:
            return err(f"Unknown tool: {name}")

        try:
            client = _get_client()
            result = await asyncio.to_thread(mod.handle, client, name, arguments)
            return ok(result)
        except Exception as e:
            logger.exception("Tool %s failed", name)
            return err(str(e))

    return server


async def _run():
    server = _create_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream, write_stream, server.create_initialization_options()
        )


def main():
    logging.basicConfig(level=logging.INFO)
    asyncio.run(_run())


if __name__ == "__main__":
    main()
