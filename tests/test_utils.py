"""Tests for feishu_mcp.utils."""

import json
import pytest
from feishu_mcp.utils import ok, err, _clean_dict


class TestOk:
    def test_string_input(self):
        result = ok("hello")
        assert len(result) == 1
        assert result[0].text == "hello"
        assert result[0].type == "text"

    def test_dict_input(self):
        result = ok({"key": "value", "num": 42})
        assert len(result) == 1
        parsed = json.loads(result[0].text)
        assert parsed["key"] == "value"
        assert parsed["num"] == 42

    def test_list_input(self):
        result = ok([1, 2, 3])
        assert len(result) == 1
        parsed = json.loads(result[0].text)
        assert parsed == [1, 2, 3]

    def test_unicode(self):
        result = ok({"msg": "你好飞书"})
        assert "你好飞书" in result[0].text


class TestErr:
    def test_error_message(self):
        result = err("something failed")
        assert len(result) == 1
        assert "Error: something failed" == result[0].text


class TestCleanDict:
    def test_simple_dict(self):
        assert _clean_dict({"a": 1, "b": "two"}) == {"a": 1, "b": "two"}

    def test_skips_private_keys(self):
        result = _clean_dict({"_internal": True, "public": False})
        assert "_internal" not in result
        assert result["public"] is False

    def test_nested_object(self):
        class Inner:
            def __init__(self):
                self.x = 10
                self._private = "skip"

        result = _clean_dict({"inner": Inner()})
        assert result["inner"] == {"x": 10}

    def test_list_of_objects(self):
        class Item:
            def __init__(self, val):
                self.val = val

        result = _clean_dict({"items": [Item(1), Item(2)]})
        assert result["items"] == [{"val": 1}, {"val": 2}]

    def test_list_of_primitives(self):
        result = _clean_dict({"tags": ["a", "b", "c"]})
        assert result["tags"] == ["a", "b", "c"]

    def test_none_values(self):
        result = _clean_dict({"key": None})
        assert result["key"] is None
