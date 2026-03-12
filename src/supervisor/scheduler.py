"""Periodic scheduler for health checks, session digests, and daily reports.

Runs asyncio periodic tasks that trigger monitoring and push results to Feishu.
"""

import asyncio
import logging
import os
import time
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Configurable intervals (seconds)
HEALTH_INTERVAL = int(os.environ.get("SUPERVISOR_HEALTH_INTERVAL", 300))  # 5 min
SESSION_DIGEST_INTERVAL = int(os.environ.get("SUPERVISOR_SESSION_DIGEST_INTERVAL", 900))  # 15 min
DAILY_REPORT_INTERVAL = int(os.environ.get("SUPERVISOR_DAILY_REPORT_INTERVAL", 86400))  # 24h
CONVERSATION_MONITOR_INTERVAL = int(os.environ.get("SUPERVISOR_CONV_MONITOR_INTERVAL", 300))  # 5 min

# Alert thresholds
CPU_THRESHOLD = float(os.environ.get("SUPERVISOR_CPU_THRESHOLD", 90))
MEMORY_THRESHOLD = float(os.environ.get("SUPERVISOR_MEMORY_THRESHOLD", 90))
DISK_THRESHOLD = float(os.environ.get("SUPERVISOR_DISK_THRESHOLD", 90))
GPU_MEMORY_THRESHOLD = float(os.environ.get("SUPERVISOR_GPU_MEMORY_THRESHOLD", 95))


class Scheduler:
    """Periodic task scheduler using asyncio."""

    def __init__(
        self,
        get_system_status: Optional[Callable] = None,
        get_gpu_status: Optional[Callable] = None,
        get_sessions_text: Optional[Callable] = None,
        get_tasks_text: Optional[Callable] = None,
        push_message: Optional[Callable] = None,
        analyze_conversation: Optional[Callable] = None,
        on_issues_found: Optional[Callable] = None,
    ):
        """Initialize with callable hooks from other modules.

        Args:
            get_system_status: Returns dict with cpu_percent, memory_percent, disk_percent
            get_gpu_status: Returns list of GPU dicts with memory_used, memory_total
            get_sessions_text: Returns formatted session summary string
            get_tasks_text: Returns formatted task summary string
            push_message: Callable(text) to push message to Feishu
            analyze_conversation: Async callable() returning analysis result dict
            on_issues_found: Callable(issues) when monitor detects problems
        """
        self._get_system_status = get_system_status
        self._get_gpu_status = get_gpu_status
        self._get_sessions_text = get_sessions_text
        self._get_tasks_text = get_tasks_text
        self._push_message = push_message
        self._analyze_conversation = analyze_conversation
        self._on_issues_found = on_issues_found
        self._tasks: list[asyncio.Task] = []
        self._running = False
        self._start_time = 0.0
        # Counters for daily report
        self._health_check_count = 0
        self._alert_count = 0
        self._monitor_check_count = 0

    async def start(self):
        """Start all periodic tasks."""
        if self._running:
            logger.warning("Scheduler already running")
            return

        self._running = True
        self._start_time = time.time()
        logger.info(
            "Scheduler starting: health=%ds, digest=%ds, daily=%ds",
            HEALTH_INTERVAL, SESSION_DIGEST_INTERVAL, DAILY_REPORT_INTERVAL,
        )

        self._tasks = [
            asyncio.create_task(self._periodic_health_check()),
            asyncio.create_task(self._periodic_session_digest()),
            asyncio.create_task(self._periodic_daily_report()),
            asyncio.create_task(self._periodic_conversation_monitor()),
        ]

    async def stop(self):
        """Stop all periodic tasks."""
        self._running = False
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        logger.info("Scheduler stopped")

    def _push(self, text: str):
        """Push a message to Feishu if handler is set."""
        if self._push_message:
            try:
                self._push_message(text)
            except Exception as e:
                logger.error("Failed to push message: %s", e)

    # ── Health Check ──

    async def _periodic_health_check(self):
        """Run health checks every HEALTH_INTERVAL seconds."""
        while self._running:
            await asyncio.sleep(HEALTH_INTERVAL)
            try:
                self._run_health_check()
            except Exception as e:
                logger.error("Health check failed: %s", e)

    def _run_health_check(self):
        """Check system resources and alert on threshold violations."""
        self._health_check_count += 1

        if not self._get_system_status:
            return

        status = self._get_system_status()
        alerts = []

        cpu = status.get("cpu_percent", 0)
        # container_monitor nests memory/disk in sub-dicts
        mem_info = status.get("memory", {})
        disk_info = status.get("disk", {})
        mem = mem_info.get("percent", 0) if isinstance(mem_info, dict) else status.get("memory_percent", 0)
        disk = disk_info.get("percent", 0) if isinstance(disk_info, dict) else status.get("disk_percent", 0)

        if cpu > CPU_THRESHOLD:
            alerts.append(f"CPU: {cpu:.1f}% (threshold: {CPU_THRESHOLD}%)")
        if mem > MEMORY_THRESHOLD:
            alerts.append(f"Memory: {mem:.1f}% (threshold: {MEMORY_THRESHOLD}%)")
        if disk > DISK_THRESHOLD:
            alerts.append(f"Disk: {disk:.1f}% (threshold: {DISK_THRESHOLD}%)")

        # GPU check
        if self._get_gpu_status:
            gpus = self._get_gpu_status()
            for i, gpu in enumerate(gpus):
                used = gpu.get("memory_used_mb", gpu.get("memory_used", 0))
                total = gpu.get("memory_total_mb", gpu.get("memory_total", 1))
                if total > 0:
                    pct = (used / total) * 100
                    if pct > GPU_MEMORY_THRESHOLD:
                        alerts.append(
                            f"GPU {i}: {pct:.1f}% memory (threshold: {GPU_MEMORY_THRESHOLD}%)"
                        )

        if alerts:
            alert_key = tuple(sorted(alerts))
            if alert_key != getattr(self, "_last_alert_key", None):
                # New or changed alert — push notification
                self._alert_count += len(alerts)
                msg = "⚠️ System Alert\n" + "\n".join(f"• {a}" for a in alerts)
                logger.warning("Health check alerts: %s", alerts)
                self._push(msg)
                self._last_alert_key = alert_key
            else:
                # Same alert as last time — log at debug only
                logger.debug("Health check alerts unchanged: %s", alerts)
        else:
            self._last_alert_key = None
            logger.debug("Health check OK: CPU=%.1f%% MEM=%.1f%% DISK=%.1f%%", cpu, mem, disk)

    # ── Session Digest ──

    async def _periodic_session_digest(self):
        """Push session digest every SESSION_DIGEST_INTERVAL seconds."""
        while self._running:
            await asyncio.sleep(SESSION_DIGEST_INTERVAL)
            try:
                self._run_session_digest()
            except Exception as e:
                logger.error("Session digest failed: %s", e)

    def _run_session_digest(self):
        """Generate and push session digest."""
        parts = []

        if self._get_sessions_text:
            sessions = self._get_sessions_text()
            if sessions:
                parts.append(f"📋 Sessions\n{sessions}")

        if self._get_tasks_text:
            tasks = self._get_tasks_text()
            if tasks:
                parts.append(f"📌 Tasks\n{tasks}")

        if parts:
            self._push("\n\n".join(parts))

    # ── Daily Report ──

    async def _periodic_daily_report(self):
        """Push daily summary report."""
        while self._running:
            await asyncio.sleep(DAILY_REPORT_INTERVAL)
            try:
                self._run_daily_report()
            except Exception as e:
                logger.error("Daily report failed: %s", e)

    def _run_daily_report(self):
        """Generate and push daily report."""
        uptime = time.time() - self._start_time
        hours = int(uptime // 3600)
        minutes = int((uptime % 3600) // 60)

        lines = [
            "📊 Daily Supervisor Report",
            f"Uptime: {hours}h {minutes}m",
            f"Health checks: {self._health_check_count}",
            f"Alerts triggered: {self._alert_count}",
        ]

        if self._get_system_status:
            status = self._get_system_status()
            mem_info = status.get("memory", {})
            disk_info = status.get("disk", {})
            mem_pct = mem_info.get("percent", 0) if isinstance(mem_info, dict) else 0
            disk_pct = disk_info.get("percent", 0) if isinstance(disk_info, dict) else 0
            lines.append(
                f"Current: CPU {status.get('cpu_percent', 0):.1f}% | "
                f"MEM {mem_pct:.1f}% | "
                f"DISK {disk_pct:.1f}%"
            )

        if self._get_sessions_text:
            lines.append(f"\n{self._get_sessions_text()}")

        if self._get_tasks_text:
            lines.append(f"\n{self._get_tasks_text()}")

        self._push("\n".join(lines))

        # Reset counters
        self._health_check_count = 0
        self._alert_count = 0
        self._monitor_check_count = 0

    # ── Conversation Monitor ──

    async def _periodic_conversation_monitor(self):
        """Analyze conversation buffer every CONVERSATION_MONITOR_INTERVAL seconds."""
        while self._running:
            await asyncio.sleep(CONVERSATION_MONITOR_INTERVAL)
            try:
                await self._run_conversation_monitor()
            except Exception as e:
                logger.error("Conversation monitor failed: %s", e)

    async def _run_conversation_monitor(self):
        """Analyze recent conversations and notify if issues found."""
        self._monitor_check_count += 1

        if not self._analyze_conversation:
            return

        try:
            result = await self._analyze_conversation()
        except Exception as e:
            logger.error("Conversation analysis failed: %s", e)
            return

        if result.get("has_issues") and result.get("issues"):
            if self._on_issues_found:
                try:
                    self._on_issues_found(result["issues"])
                except Exception as e:
                    logger.error("Failed to handle found issues: %s", e)
            else:
                # Fallback: push raw notification
                from . import conversation_monitor as _cm
                msg = _cm.format_issue_notification(result["issues"])
                if msg:
                    self._push(msg)
        else:
            logger.debug("Conversation monitor: %s", result.get("summary", "no issues"))
