"""Tests for supervisor.scheduler module."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from supervisor.scheduler import (
    Scheduler,
    HEALTH_INTERVAL,
    CPU_THRESHOLD,
    MEMORY_THRESHOLD,
    DISK_THRESHOLD,
)


class TestHealthCheck:
    def test_no_alert_when_below_threshold(self):
        push = MagicMock()
        scheduler = Scheduler(
            get_system_status=lambda: {
                "cpu_percent": 50,
                "memory": {"percent": 60},
                "disk": {"percent": 70},
            },
            push_message=push,
        )
        scheduler._run_health_check()
        push.assert_not_called()
        assert scheduler._health_check_count == 1
        assert scheduler._alert_count == 0

    def test_cpu_alert(self):
        push = MagicMock()
        scheduler = Scheduler(
            get_system_status=lambda: {
                "cpu_percent": 95,
                "memory": {"percent": 50},
                "disk": {"percent": 50},
            },
            push_message=push,
        )
        scheduler._run_health_check()
        push.assert_called_once()
        assert "CPU" in push.call_args[0][0]
        assert scheduler._alert_count == 1

    def test_memory_alert(self):
        push = MagicMock()
        scheduler = Scheduler(
            get_system_status=lambda: {
                "cpu_percent": 50,
                "memory": {"percent": 95},
                "disk": {"percent": 50},
            },
            push_message=push,
        )
        scheduler._run_health_check()
        push.assert_called_once()
        assert "Memory" in push.call_args[0][0]

    def test_disk_alert(self):
        push = MagicMock()
        scheduler = Scheduler(
            get_system_status=lambda: {
                "cpu_percent": 50,
                "memory": {"percent": 50},
                "disk": {"percent": 95},
            },
            push_message=push,
        )
        scheduler._run_health_check()
        push.assert_called_once()
        assert "Disk" in push.call_args[0][0]

    def test_multiple_alerts(self):
        push = MagicMock()
        scheduler = Scheduler(
            get_system_status=lambda: {
                "cpu_percent": 95,
                "memory": {"percent": 95},
                "disk": {"percent": 95},
            },
            push_message=push,
        )
        scheduler._run_health_check()
        push.assert_called_once()
        msg = push.call_args[0][0]
        assert "CPU" in msg
        assert "Memory" in msg
        assert "Disk" in msg
        assert scheduler._alert_count == 3

    def test_gpu_alert(self):
        push = MagicMock()
        scheduler = Scheduler(
            get_system_status=lambda: {
                "cpu_percent": 50,
                "memory_percent": 50,
                "disk_percent": 50,
            },
            get_gpu_status=lambda: [
                {"memory_used": 15500, "memory_total": 16000}
            ],
            push_message=push,
        )
        scheduler._run_health_check()
        push.assert_called_once()
        assert "GPU" in push.call_args[0][0]

    def test_no_status_handler(self):
        scheduler = Scheduler(get_system_status=None)
        scheduler._run_health_check()  # Should not raise
        assert scheduler._health_check_count == 1

    def test_push_error_handled(self):
        def bad_push(text):
            raise RuntimeError("push failed")

        scheduler = Scheduler(
            get_system_status=lambda: {
                "cpu_percent": 99,
                "memory": {"percent": 50},
                "disk": {"percent": 50},
            },
            push_message=bad_push,
        )
        scheduler._run_health_check()  # Should not raise


class TestSessionDigest:
    def test_digest_with_sessions_and_tasks(self):
        push = MagicMock()
        scheduler = Scheduler(
            get_sessions_text=lambda: "2 active sessions",
            get_tasks_text=lambda: "1 running task",
            push_message=push,
        )
        scheduler._run_session_digest()
        push.assert_called_once()
        msg = push.call_args[0][0]
        assert "Sessions" in msg
        assert "Tasks" in msg

    def test_digest_sessions_only(self):
        push = MagicMock()
        scheduler = Scheduler(
            get_sessions_text=lambda: "2 active sessions",
            push_message=push,
        )
        scheduler._run_session_digest()
        push.assert_called_once()

    def test_digest_nothing(self):
        push = MagicMock()
        scheduler = Scheduler(push_message=push)
        scheduler._run_session_digest()
        push.assert_not_called()


class TestDailyReport:
    def test_daily_report_format(self):
        push = MagicMock()
        scheduler = Scheduler(
            get_system_status=lambda: {
                "cpu_percent": 45,
                "memory": {"percent": 60},
                "disk": {"percent": 30},
            },
            get_sessions_text=lambda: "Session info",
            get_tasks_text=lambda: "Task info",
            push_message=push,
        )
        scheduler._health_check_count = 10
        scheduler._alert_count = 2

        scheduler._run_daily_report()
        push.assert_called_once()
        msg = push.call_args[0][0]
        assert "Daily" in msg
        assert "10" in msg  # health checks
        assert "2" in msg   # alerts
        assert "CPU" in msg

    def test_daily_report_resets_counters(self):
        scheduler = Scheduler(push_message=MagicMock())
        scheduler._health_check_count = 10
        scheduler._alert_count = 5
        scheduler._run_daily_report()
        assert scheduler._health_check_count == 0
        assert scheduler._alert_count == 0


class TestLifecycle:
    def test_start_stop(self):
        scheduler = Scheduler()

        async def _test():
            await scheduler.start()
            assert scheduler._running is True
            assert len(scheduler._tasks) == 4
            await scheduler.stop()
            assert scheduler._running is False
            assert len(scheduler._tasks) == 0

        asyncio.run(_test())

    def test_double_start(self):
        scheduler = Scheduler()

        async def _test():
            await scheduler.start()
            await scheduler.start()  # Should warn, not crash
            assert len(scheduler._tasks) == 4
            await scheduler.stop()

        asyncio.run(_test())


class TestConversationMonitor:
    def test_no_analyzer_skips(self):
        scheduler = Scheduler(analyze_conversation=None)

        async def _test():
            await scheduler._run_conversation_monitor()
            assert scheduler._monitor_check_count == 1

        asyncio.run(_test())

    def test_issues_found_calls_handler(self):
        on_issues = MagicMock()

        async def fake_analyze():
            return {
                "has_issues": True,
                "issues": [{"severity": "HIGH", "description": "stuck", "suggested_fix": "restart"}],
                "summary": "found issue",
            }

        scheduler = Scheduler(
            analyze_conversation=fake_analyze,
            on_issues_found=on_issues,
            push_message=MagicMock(),
        )

        async def _test():
            await scheduler._run_conversation_monitor()
            assert scheduler._monitor_check_count == 1
            on_issues.assert_called_once()
            issues = on_issues.call_args[0][0]
            assert len(issues) == 1
            assert issues[0]["severity"] == "HIGH"

        asyncio.run(_test())

    def test_no_issues_does_not_call_handler(self):
        on_issues = MagicMock()

        async def fake_analyze():
            return {"has_issues": False, "issues": [], "summary": "all good"}

        scheduler = Scheduler(
            analyze_conversation=fake_analyze,
            on_issues_found=on_issues,
        )

        async def _test():
            await scheduler._run_conversation_monitor()
            on_issues.assert_not_called()

        asyncio.run(_test())

    def test_fallback_push_when_no_handler(self):
        push = MagicMock()

        async def fake_analyze():
            return {
                "has_issues": True,
                "issues": [{"severity": "HIGH", "description": "error", "suggested_fix": "fix"}],
                "summary": "issue",
            }

        scheduler = Scheduler(
            analyze_conversation=fake_analyze,
            on_issues_found=None,
            push_message=push,
        )

        async def _test():
            await scheduler._run_conversation_monitor()
            push.assert_called_once()
            assert "error" in push.call_args[0][0]

        asyncio.run(_test())

    def test_analyzer_error_handled(self):
        async def broken_analyze():
            raise RuntimeError("API down")

        scheduler = Scheduler(
            analyze_conversation=broken_analyze,
            push_message=MagicMock(),
        )

        async def _test():
            # Should not raise
            await scheduler._run_conversation_monitor()

        asyncio.run(_test())
