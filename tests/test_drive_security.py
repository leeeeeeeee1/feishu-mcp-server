"""Tests for drive path validation security."""

import os
import pytest
from feishu_mcp.tools.drive import _validate_path


class TestPathValidation:
    def test_allowed_path(self):
        path = "/tmp/feishu-files/test.txt"
        result = _validate_path(path, "upload")
        assert result == os.path.realpath(path)

    def test_blocked_etc_passwd(self):
        with pytest.raises(PermissionError, match="outside allowed directories"):
            _validate_path("/etc/passwd", "upload")

    def test_blocked_home_ssh(self):
        with pytest.raises(PermissionError, match="outside allowed directories"):
            _validate_path("/home/user/.ssh/id_rsa", "download")

    def test_blocked_path_traversal(self):
        with pytest.raises(PermissionError, match="outside allowed directories"):
            _validate_path("/tmp/feishu-files/../../etc/shadow", "upload")

    def test_blocked_root(self):
        with pytest.raises(PermissionError, match="outside allowed directories"):
            _validate_path("/", "download")


class TestToolRegistration:
    def test_all_tools_registered(self):
        from feishu_mcp.server import _ALL_TOOLS, _TOOL_REGISTRY

        assert len(_ALL_TOOLS) == 11  # drive(3) + docs(5) + bitable(3), im/calendar/contact removed
        assert len(_TOOL_REGISTRY) == 11

    def test_all_tools_have_required_fields(self):
        from feishu_mcp.server import _ALL_TOOLS

        for tool in _ALL_TOOLS:
            assert "name" in tool, f"Tool missing 'name'"
            assert "description" in tool, f"{tool.get('name')} missing 'description'"
            assert "inputSchema" in tool, f"{tool['name']} missing 'inputSchema'"
            assert tool["inputSchema"]["type"] == "object"

    def test_no_duplicate_tool_names(self):
        from feishu_mcp.server import _ALL_TOOLS

        names = [t["name"] for t in _ALL_TOOLS]
        assert len(names) == len(set(names)), f"Duplicate tool names: {names}"
