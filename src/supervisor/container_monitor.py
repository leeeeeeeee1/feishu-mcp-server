"""Monitor container/system resources — CPU, memory, disk, GPU, processes, ports."""

import subprocess
from typing import Any

import psutil

# Development process names to track.
_DEV_PROCESS_NAMES = frozenset(
    ["claude", "python", "cargo", "gcc", "g++", "rustc", "nvcc", "make", "cmake"]
)


# ── System (CPU / Memory / Disk) ──


def get_system_status() -> dict[str, Any]:
    """Return CPU, memory, and disk usage as a dict."""
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage("/")
    return {
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "memory": {
            "total_mb": round(mem.total / (1024 * 1024), 1),
            "used_mb": round(mem.used / (1024 * 1024), 1),
            "percent": mem.percent,
        },
        "disk": {
            "total_gb": round(disk.total / (1024 ** 3), 1),
            "used_gb": round(disk.used / (1024 ** 3), 1),
            "percent": disk.percent,
        },
    }


# ── GPU ──


def get_gpu_status() -> list[dict[str, Any]]:
    """Query nvidia-smi and return a list of GPU info dicts.

    Returns an empty list when nvidia-smi is unavailable or fails.
    """
    cmd = [
        "nvidia-smi",
        "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=5  # noqa: S603
        )
        if result.returncode != 0:
            return []
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return []

    gpus: list[dict[str, Any]] = []
    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            gpus.append(
                {
                    "name": parts[0],
                    "memory_used_mb": float(parts[1]),
                    "memory_total_mb": float(parts[2]),
                    "utilization_percent": float(parts[3]),
                    "temperature_c": float(parts[4]),
                }
            )
        except (ValueError, IndexError):
            continue
    return gpus


# ── Development Processes ──


def get_dev_processes() -> list[dict[str, Any]]:
    """Return running dev-related processes with PID, CPU%, mem%, command."""
    procs: list[dict[str, Any]] = []
    for proc in psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent", "cmdline"]):
        try:
            info = proc.info  # type: ignore[attr-defined]
            name = (info.get("name") or "").lower()
            if name not in _DEV_PROCESS_NAMES:
                continue
            cmdline = info.get("cmdline") or []
            procs.append(
                {
                    "pid": info["pid"],
                    "name": info["name"],
                    "cpu_percent": info.get("cpu_percent", 0.0),
                    "memory_percent": round(info.get("memory_percent", 0.0), 2),
                    "command": " ".join(cmdline) if cmdline else name,
                }
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return procs


# ── Listening Ports ──


def get_listening_ports() -> list[dict[str, Any]]:
    """Return list of listening TCP/UDP ports with pid and process name."""
    ports: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    try:
        connections = psutil.net_connections(kind="inet")
    except (psutil.AccessDenied, OSError):
        return []

    for conn in connections:
        if conn.status != "LISTEN":
            continue
        laddr = conn.laddr
        if not laddr:
            continue
        port = laddr.port
        pid = conn.pid
        key = (port, pid or 0)
        if key in seen:
            continue
        seen.add(key)

        proc_name = ""
        if pid:
            try:
                proc_name = psutil.Process(pid).name()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        ports.append({"port": port, "pid": pid, "process": proc_name})
    return sorted(ports, key=lambda p: p["port"])


# ── Formatted Text Helpers ──


def get_gpu_text() -> str:
    """Return a formatted text block describing GPU status."""
    gpus = get_gpu_status()
    if not gpus:
        return "GPU: not available"
    lines: list[str] = []
    for i, gpu in enumerate(gpus):
        lines.append(
            f"GPU {i}: {gpu['name']}  |  "
            f"Mem {gpu['memory_used_mb']:.0f}/{gpu['memory_total_mb']:.0f} MB  |  "
            f"Util {gpu['utilization_percent']:.0f}%  |  "
            f"Temp {gpu['temperature_c']:.0f}°C"
        )
    return "\n".join(lines)


def get_status_text() -> str:
    """Return a combined formatted status text suitable for Feishu messages."""
    sys = get_system_status()
    gpus = get_gpu_status()
    procs = get_dev_processes()
    ports = get_listening_ports()

    sections: list[str] = []

    # System
    sections.append(
        f"CPU: {sys['cpu_percent']}%\n"
        f"Memory: {sys['memory']['used_mb']:.0f}/{sys['memory']['total_mb']:.0f} MB "
        f"({sys['memory']['percent']}%)\n"
        f"Disk: {sys['disk']['used_gb']:.1f}/{sys['disk']['total_gb']:.1f} GB "
        f"({sys['disk']['percent']}%)"
    )

    # GPU
    if gpus:
        gpu_lines: list[str] = []
        for i, gpu in enumerate(gpus):
            gpu_lines.append(
                f"  GPU {i}: {gpu['name']}  "
                f"Mem {gpu['memory_used_mb']:.0f}/{gpu['memory_total_mb']:.0f} MB  "
                f"Util {gpu['utilization_percent']:.0f}%  "
                f"Temp {gpu['temperature_c']:.0f}°C"
            )
        sections.append("GPU:\n" + "\n".join(gpu_lines))
    else:
        sections.append("GPU: not available")

    # Processes
    if procs:
        proc_lines = [
            f"  [{p['pid']}] {p['name']}  CPU {p['cpu_percent']}%  "
            f"Mem {p['memory_percent']}%  {p['command']}"
            for p in procs
        ]
        sections.append("Dev Processes:\n" + "\n".join(proc_lines))
    else:
        sections.append("Dev Processes: none")

    # Ports
    if ports:
        port_lines = [
            f"  :{p['port']}  PID {p['pid']}  {p['process']}" for p in ports
        ]
        sections.append("Listening Ports:\n" + "\n".join(port_lines))
    else:
        sections.append("Listening Ports: none")

    return "\n\n".join(sections)
