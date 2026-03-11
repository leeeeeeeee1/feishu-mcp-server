"""Tests for supervisor.container_monitor module."""

import subprocess
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from supervisor.container_monitor import (
    get_dev_processes,
    get_gpu_status,
    get_gpu_text,
    get_listening_ports,
    get_status_text,
    get_system_status,
)


# ── Helpers ──


def _make_virtual_memory(total=16 * 1024 ** 3, used=8 * 1024 ** 3, percent=50.0):
    return SimpleNamespace(total=total, used=used, percent=percent)


def _make_disk_usage(total=500 * 1024 ** 3, used=200 * 1024 ** 3, percent=40.0):
    return SimpleNamespace(total=total, used=used, percent=percent)


def _make_proc(pid, name, cpu_percent=1.0, memory_percent=0.5, cmdline=None):
    """Create a mock process that works with psutil.process_iter."""
    proc = MagicMock()
    info = {
        "pid": pid,
        "name": name,
        "cpu_percent": cpu_percent,
        "memory_percent": memory_percent,
        "cmdline": cmdline or [f"/usr/bin/{name}"],
    }
    proc.info = info
    return proc


def _make_connection(port, pid, status="LISTEN"):
    return SimpleNamespace(
        laddr=SimpleNamespace(ip="0.0.0.0", port=port),
        raddr=None,
        status=status,
        pid=pid,
    )


# ── System Status ──


class TestGetSystemStatus:
    @patch("supervisor.container_monitor.psutil")
    def test_returns_cpu_memory_disk(self, mock_psutil):
        mock_psutil.cpu_percent.return_value = 42.5
        mock_psutil.virtual_memory.return_value = _make_virtual_memory()
        mock_psutil.disk_usage.return_value = _make_disk_usage()

        status = get_system_status()

        assert status["cpu_percent"] == 42.5
        assert status["memory"]["percent"] == 50.0
        assert status["memory"]["total_mb"] == pytest.approx(16384.0, rel=1e-1)
        assert status["disk"]["percent"] == 40.0
        mock_psutil.disk_usage.assert_called_once_with("/")

    @patch("supervisor.container_monitor.psutil")
    def test_memory_used_mb(self, mock_psutil):
        mock_psutil.cpu_percent.return_value = 0
        mock_psutil.virtual_memory.return_value = _make_virtual_memory(
            used=4 * 1024 ** 3
        )
        mock_psutil.disk_usage.return_value = _make_disk_usage()

        status = get_system_status()
        assert status["memory"]["used_mb"] == pytest.approx(4096.0, rel=1e-1)


# ── GPU Status ──


class TestGetGpuStatus:
    @patch("supervisor.container_monitor.subprocess.run")
    def test_parses_nvidia_smi_output(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Tesla V100, 1024, 16384, 35, 55\n",
        )

        gpus = get_gpu_status()

        assert len(gpus) == 1
        assert gpus[0]["name"] == "Tesla V100"
        assert gpus[0]["memory_used_mb"] == 1024.0
        assert gpus[0]["memory_total_mb"] == 16384.0
        assert gpus[0]["utilization_percent"] == 35.0
        assert gpus[0]["temperature_c"] == 55.0

    @patch("supervisor.container_monitor.subprocess.run")
    def test_multiple_gpus(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="A100, 2048, 40960, 50, 60\nA100, 4096, 40960, 70, 65\n",
        )

        gpus = get_gpu_status()
        assert len(gpus) == 2
        assert gpus[1]["memory_used_mb"] == 4096.0

    @patch("supervisor.container_monitor.subprocess.run")
    def test_nvidia_smi_not_found(self, mock_run):
        mock_run.side_effect = FileNotFoundError

        gpus = get_gpu_status()
        assert gpus == []

    @patch("supervisor.container_monitor.subprocess.run")
    def test_nvidia_smi_nonzero_exit(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout="")

        gpus = get_gpu_status()
        assert gpus == []

    @patch("supervisor.container_monitor.subprocess.run")
    def test_nvidia_smi_timeout(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=5)

        gpus = get_gpu_status()
        assert gpus == []


# ── Dev Processes ──


class TestGetDevProcesses:
    @patch("supervisor.container_monitor.psutil")
    def test_filters_dev_processes(self, mock_psutil):
        mock_psutil.process_iter.return_value = [
            _make_proc(100, "python", cpu_percent=10.0, memory_percent=2.5,
                       cmdline=["python", "app.py"]),
            _make_proc(200, "bash"),  # not a dev process
            _make_proc(300, "cargo", cpu_percent=50.0, memory_percent=5.0),
        ]
        mock_psutil.NoSuchProcess = type("NoSuchProcess", (Exception,), {})
        mock_psutil.AccessDenied = type("AccessDenied", (Exception,), {})
        mock_psutil.ZombieProcess = type("ZombieProcess", (Exception,), {})

        procs = get_dev_processes()

        names = [p["name"] for p in procs]
        assert "python" in names
        assert "cargo" in names
        assert "bash" not in names
        assert len(procs) == 2

    @patch("supervisor.container_monitor.psutil")
    def test_process_fields(self, mock_psutil):
        mock_psutil.process_iter.return_value = [
            _make_proc(42, "gcc", cpu_percent=25.0, memory_percent=3.14,
                       cmdline=["gcc", "-o", "main", "main.c"]),
        ]
        mock_psutil.NoSuchProcess = type("NoSuchProcess", (Exception,), {})
        mock_psutil.AccessDenied = type("AccessDenied", (Exception,), {})
        mock_psutil.ZombieProcess = type("ZombieProcess", (Exception,), {})

        procs = get_dev_processes()
        assert procs[0]["pid"] == 42
        assert procs[0]["cpu_percent"] == 25.0
        assert procs[0]["command"] == "gcc -o main main.c"


# ── Listening Ports ──


class TestGetListeningPorts:
    @patch("supervisor.container_monitor.psutil")
    def test_returns_listening_ports(self, mock_psutil):
        mock_psutil.net_connections.return_value = [
            _make_connection(8080, 100, "LISTEN"),
            _make_connection(3000, 200, "LISTEN"),
            _make_connection(9999, 300, "ESTABLISHED"),  # not listening
        ]
        mock_psutil.AccessDenied = type("AccessDenied", (Exception,), {})
        mock_proc = MagicMock()
        mock_proc.name.return_value = "python"
        mock_psutil.Process.return_value = mock_proc
        mock_psutil.NoSuchProcess = type("NoSuchProcess", (Exception,), {})

        ports = get_listening_ports()

        port_numbers = [p["port"] for p in ports]
        assert 8080 in port_numbers
        assert 3000 in port_numbers
        assert 9999 not in port_numbers
        # sorted by port
        assert ports[0]["port"] < ports[1]["port"]

    @patch("supervisor.container_monitor.psutil")
    def test_access_denied(self, mock_psutil):
        exc = type("AccessDenied", (Exception,), {})
        mock_psutil.AccessDenied = exc
        mock_psutil.net_connections.side_effect = exc()

        ports = get_listening_ports()
        assert ports == []


# ── Text Formatting ──


class TestGetGpuText:
    @patch("supervisor.container_monitor.get_gpu_status")
    def test_no_gpu(self, mock_gpu):
        mock_gpu.return_value = []
        assert get_gpu_text() == "GPU: not available"

    @patch("supervisor.container_monitor.get_gpu_status")
    def test_with_gpu(self, mock_gpu):
        mock_gpu.return_value = [
            {
                "name": "A100",
                "memory_used_mb": 1024.0,
                "memory_total_mb": 40960.0,
                "utilization_percent": 50.0,
                "temperature_c": 60.0,
            }
        ]
        text = get_gpu_text()
        assert "A100" in text
        assert "1024" in text
        assert "50%" in text


class TestGetStatusText:
    @patch("supervisor.container_monitor.get_listening_ports")
    @patch("supervisor.container_monitor.get_dev_processes")
    @patch("supervisor.container_monitor.get_gpu_status")
    @patch("supervisor.container_monitor.get_system_status")
    def test_combined_output(self, mock_sys, mock_gpu, mock_procs, mock_ports):
        mock_sys.return_value = {
            "cpu_percent": 25.0,
            "memory": {"total_mb": 16384.0, "used_mb": 8192.0, "percent": 50.0},
            "disk": {"total_gb": 500.0, "used_gb": 200.0, "percent": 40.0},
        }
        mock_gpu.return_value = []
        mock_procs.return_value = [
            {"pid": 1, "name": "python", "cpu_percent": 5.0,
             "memory_percent": 1.0, "command": "python app.py"},
        ]
        mock_ports.return_value = [{"port": 8080, "pid": 1, "process": "python"}]

        text = get_status_text()

        assert "CPU: 25.0%" in text
        assert "Memory:" in text
        assert "Disk:" in text
        assert "GPU: not available" in text
        assert "python" in text
        assert ":8080" in text
