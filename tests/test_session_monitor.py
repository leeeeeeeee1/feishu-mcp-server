"""Tests for supervisor.session_monitor module."""

import json
import time
from pathlib import Path

import pytest

from supervisor.session_monitor import (
    _extract_session_metadata,
    _format_timestamp,
    _minutes_ago,
    _safe_read_jsonl,
    _scan_jsonl_files,
    _scan_session_summaries,
    get_active_sessions,
    get_session_detail,
    get_sessions_text,
    list_sessions,
)


# ── Helpers ──


def _write_jsonl(path: Path, entries: list[dict]):
    """Write a list of dicts as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


def _make_project_tree(tmp_path: Path, sessions: dict[str, list[dict]]) -> Path:
    """Create a fake projects dir with JSONL files.

    sessions: mapping of "project_name/uuid" -> list of JSONL entries
    """
    projects_dir = tmp_path / "projects"
    for rel_path, entries in sessions.items():
        jsonl_path = projects_dir / (rel_path + ".jsonl")
        _write_jsonl(jsonl_path, entries)
    return projects_dir


# ── _safe_read_jsonl ──


class TestSafeReadJsonl:
    def test_valid_file(self, tmp_path):
        p = tmp_path / "test.jsonl"
        entries = [{"type": "human", "timestamp": 1000}, {"type": "assistant"}]
        _write_jsonl(p, entries)
        result = _safe_read_jsonl(p)
        assert len(result) == 2
        assert result[0]["type"] == "human"

    def test_corrupt_lines_skipped(self, tmp_path):
        p = tmp_path / "test.jsonl"
        p.write_text('{"ok": true}\nNOT JSON\n{"also": "ok"}\n')
        result = _safe_read_jsonl(p)
        assert len(result) == 2
        assert result[0]["ok"] is True
        assert result[1]["also"] == "ok"

    def test_empty_file(self, tmp_path):
        p = tmp_path / "empty.jsonl"
        p.write_text("")
        result = _safe_read_jsonl(p)
        assert result == []

    def test_missing_file(self, tmp_path):
        p = tmp_path / "nonexistent.jsonl"
        result = _safe_read_jsonl(p)
        assert result == []

    def test_blank_lines_ignored(self, tmp_path):
        p = tmp_path / "blanks.jsonl"
        p.write_text('\n\n{"a": 1}\n\n')
        result = _safe_read_jsonl(p)
        assert len(result) == 1


# ── _scan_jsonl_files ──


class TestScanJsonlFiles:
    def test_finds_nested_files(self, tmp_path):
        projects_dir = _make_project_tree(tmp_path, {
            "proj-a/sess-1": [{"type": "human"}],
            "proj-b/sess-2": [{"type": "human"}],
        })
        files = _scan_jsonl_files(projects_dir)
        assert len(files) == 2
        stems = {f.stem for f in files}
        assert stems == {"sess-1", "sess-2"}

    def test_missing_directory(self, tmp_path):
        files = _scan_jsonl_files(tmp_path / "does_not_exist")
        assert files == []


# ── _scan_session_summaries ──


class TestScanSessionSummaries:
    def test_reads_tmp_files(self, tmp_path):
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()
        (sessions_dir / "abc-123.tmp").write_text("summary text here")
        (sessions_dir / "def-456.tmp").write_text("another summary")
        result = _scan_session_summaries(sessions_dir)
        assert result["abc-123"] == "summary text here"
        assert result["def-456"] == "another summary"

    def test_missing_dir(self, tmp_path):
        result = _scan_session_summaries(tmp_path / "nope")
        assert result == {}


# ── _extract_session_metadata ──


class TestExtractMetadata:
    def test_counts_messages(self, tmp_path):
        entries = [
            {"type": "human", "timestamp": 1000.0},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "hi"}]}, "timestamp": 1001.0},
            {"type": "human", "timestamp": 1002.0},
        ]
        p = tmp_path / "proj" / "sess-abc.jsonl"
        _write_jsonl(p, entries)
        meta = _extract_session_metadata(p, entries)
        assert meta["session_id"] == "sess-abc"
        assert meta["message_count"] == 3
        assert meta["human_messages"] == 2
        assert meta["assistant_messages"] == 1

    def test_extracts_tool_calls(self, tmp_path):
        entries = [
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {"type": "tool_use", "name": "Read"},
                        {"type": "tool_use", "name": "Bash"},
                        {"type": "text", "text": "done"},
                    ]
                },
            },
        ]
        p = tmp_path / "proj" / "s1.jsonl"
        _write_jsonl(p, entries)
        meta = _extract_session_metadata(p, entries)
        assert "Read" in meta["tool_calls"]
        assert "Bash" in meta["tool_calls"]

    def test_deduplicates_tool_calls(self, tmp_path):
        entries = [
            {"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "Read"}]}},
            {"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "Read"}]}},
        ]
        p = tmp_path / "proj" / "s2.jsonl"
        _write_jsonl(p, entries)
        meta = _extract_session_metadata(p, entries)
        assert meta["tool_calls"] == ["Read"]

    def test_timestamps(self, tmp_path):
        entries = [
            {"type": "human", "timestamp": 1000.0},
            {"type": "human", "timestamp": 2000.0},
        ]
        p = tmp_path / "proj" / "s3.jsonl"
        _write_jsonl(p, entries)
        meta = _extract_session_metadata(p, entries)
        assert meta["first_activity"] == 1000.0
        assert meta["last_activity"] == 2000.0

    def test_falls_back_to_mtime(self, tmp_path):
        entries = [{"type": "human"}]  # no timestamp
        p = tmp_path / "proj" / "s4.jsonl"
        _write_jsonl(p, entries)
        meta = _extract_session_metadata(p, entries)
        # Should fall back to file mtime
        assert meta["last_activity"] is not None


# ── list_sessions ──


class TestListSessions:
    def test_returns_sorted(self, tmp_path):
        projects_dir = _make_project_tree(tmp_path, {
            "p/old": [{"type": "human", "timestamp": 1000.0}],
            "p/new": [{"type": "human", "timestamp": 9000.0}],
        })
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()
        result = list_sessions(projects_dir, sessions_dir)
        assert len(result) == 2
        assert result[0]["session_id"] == "new"
        assert result[1]["session_id"] == "old"

    def test_attaches_summary(self, tmp_path):
        projects_dir = _make_project_tree(tmp_path, {
            "p/abc-def": [{"type": "human", "timestamp": 1000.0}],
        })
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()
        (sessions_dir / "abc-def.tmp").write_text("my summary")
        result = list_sessions(projects_dir, sessions_dir)
        assert result[0]["summary"] == "my summary"

    def test_empty_dirs(self, tmp_path):
        result = list_sessions(tmp_path / "nope", tmp_path / "nada")
        assert result == []


# ── get_session_detail ──


class TestGetSessionDetail:
    def test_found(self, tmp_path):
        projects_dir = _make_project_tree(tmp_path, {
            "p/target-id": [
                {"type": "human", "timestamp": 500.0},
                {"type": "assistant", "message": {"content": []}, "timestamp": 501.0},
            ],
        })
        result = get_session_detail("target-id", projects_dir, tmp_path)
        assert result["session_id"] == "target-id"
        assert result["raw_entry_count"] == 2

    def test_not_found(self, tmp_path):
        projects_dir = _make_project_tree(tmp_path, {
            "p/other": [{"type": "human"}],
        })
        result = get_session_detail("missing-id", projects_dir, tmp_path)
        assert result == {}


# ── get_active_sessions ──


class TestGetActiveSessions:
    def test_filters_by_threshold(self, tmp_path):
        now = time.time()
        projects_dir = _make_project_tree(tmp_path, {
            "p/recent": [{"type": "human", "timestamp": now - 60}],      # 1 min ago
            "p/old": [{"type": "human", "timestamp": now - 7200}],       # 2 hours ago
        })
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()
        result = get_active_sessions(30, projects_dir, sessions_dir)
        assert len(result) == 1
        assert result[0]["session_id"] == "recent"

    def test_all_stale(self, tmp_path):
        old_ts = time.time() - 86400
        projects_dir = _make_project_tree(tmp_path, {
            "p/s1": [{"type": "human", "timestamp": old_ts}],
        })
        result = get_active_sessions(30, projects_dir, tmp_path)
        assert result == []


# ── get_sessions_text ──


class TestGetSessionsText:
    def test_no_sessions(self, tmp_path):
        text = get_sessions_text(tmp_path / "nope", tmp_path / "nada")
        assert text == "No Claude Code sessions found."

    def test_formatted_output(self, tmp_path):
        now = time.time()
        projects_dir = _make_project_tree(tmp_path, {
            "myproject/sess-xyz": [
                {"type": "human", "timestamp": now - 120},
                {"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "Bash"}]}, "timestamp": now - 60},
            ],
        })
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()
        (sessions_dir / "sess-xyz.tmp").write_text("Did some work")

        text = get_sessions_text(projects_dir, sessions_dir)
        assert "sess-xyz" in text
        assert "myproject" in text
        assert "Bash" in text
        assert "Did some work" in text
        assert "Messages:" in text


# ── Formatting helpers ──


class TestFormatHelpers:
    def test_format_timestamp_none(self):
        assert _format_timestamp(None) == "N/A"

    def test_format_timestamp_valid(self):
        # 2024-01-01 00:00:00 UTC = 1704067200
        result = _format_timestamp(1704067200.0)
        assert "2024-01-01" in result
        assert "UTC" in result

    def test_minutes_ago_none(self):
        assert _minutes_ago(None) == "unknown"

    def test_minutes_ago_recent(self):
        assert _minutes_ago(time.time() - 5) == "just now"

    def test_minutes_ago_minutes(self):
        result = _minutes_ago(time.time() - 300)
        assert "5m ago" in result

    def test_minutes_ago_hours(self):
        result = _minutes_ago(time.time() - 7200)
        assert "2h" in result
