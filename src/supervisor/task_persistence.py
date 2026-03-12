"""Task persistence and crash recovery.

Handles saving/loading tasks to disk, checkpoint management for crash
recovery, and startup recovery logic. Functions accept state parameters
(tasks dict, lock, paths) to avoid circular imports with task_dispatcher.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import asdict, replace as dc_replace
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import threading

logger = logging.getLogger(__name__)

_SAFE_ID_RE = re.compile(r'^[0-9a-zA-Z\-]{1,64}$')

_ACTIVE_PROCESS_STATUSES = frozenset(("running", "follow_up", "learning"))
"""States that imply an active subprocess was running at crash time.
These processes are lost on restart and need recovery.
Other states (pending, waiting_for_input, review, done, awaiting_closure)
have no live subprocess and are preserved unchanged on restart."""


def save_tasks_unlocked(tasks: dict, tasks_file: Path) -> None:
    """Persist all tasks to disk. Caller MUST hold _tasks_lock."""
    try:
        data = {tid: asdict(t) for tid, t in tasks.items()}
        tmp = tasks_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, default=str))
        tmp.rename(tasks_file)
    except Exception as e:
        logger.error("Failed to save tasks: %s", e)


def save_tasks(tasks: dict, lock: threading.Lock, tasks_file: Path) -> None:
    """Persist all tasks to disk.

    NOTE: Caller must NOT hold _tasks_lock when calling this.
    """
    with lock:
        save_tasks_unlocked(tasks, tasks_file)


def checkpoint_path(task_id: str, checkpoint_dir: Path) -> Path:
    """Build a safe checkpoint file path, rejecting path-traversal IDs."""
    if not _SAFE_ID_RE.match(task_id):
        raise ValueError(f"Invalid task_id for checkpoint: {task_id!r}")
    return checkpoint_dir / f"{task_id}.json"


def save_checkpoint(task, checkpoint_dir: Path) -> None:
    """Save a checkpoint for a running task (crash recovery)."""
    try:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "task_id": task.id,
            "timestamp": time.time(),
            "steps_completed": list(task.steps_completed),
            "current_step": task.current_step,
            "partial_result": task.result or "",
            "session_id": task.session_id or "",
        }
        ckpt_file = checkpoint_path(task.id, checkpoint_dir)
        tmp = ckpt_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False))
        tmp.rename(ckpt_file)
    except Exception as e:
        logger.warning("Failed to save checkpoint for %s: %s", task.id[:8], e)


def load_checkpoint(task_id: str, checkpoint_dir: Path) -> dict | None:
    """Load checkpoint data for a task. Returns None if not found."""
    try:
        ckpt_file = checkpoint_path(task_id, checkpoint_dir)
    except ValueError:
        logger.warning("Skipping checkpoint load for invalid task_id: %s", task_id[:8])
        return None
    if not ckpt_file.exists():
        return None
    try:
        return json.loads(ckpt_file.read_text())
    except Exception as e:
        logger.warning("Failed to load checkpoint for %s: %s", task_id[:8], e)
        return None


def clear_checkpoint(task_id: str, checkpoint_dir: Path) -> None:
    """Remove checkpoint file after task completes normally."""
    try:
        ckpt_file = checkpoint_path(task_id, checkpoint_dir)
        ckpt_file.unlink(missing_ok=True)
    except ValueError:
        pass  # invalid ID — no checkpoint to clear
    except Exception as e:
        logger.warning("Failed to clear checkpoint for %s: %s", task_id[:8], e)


def load_tasks(
    tasks: dict,
    lock,
    tasks_file: Path,
    checkpoint_dir: Path,
    task_cls,
) -> None:
    """Load tasks from disk on startup.

    Recovery strategy for tasks that were active when supervisor crashed:
    - pending: keep as pending — can be re-dispatched
    - running/follow_up/learning: mark as interrupted — resumable via /recover
    - waiting_for_input/review/done/awaiting_closure: preserved (no live process)
    - completed/cancelled: preserved if <24h old, pruned otherwise
    Checkpoint data (if available) is merged to preserve progress information.
    """
    # Clean up orphaned .tmp file from interrupted save
    tmp = tasks_file.with_suffix(".tmp")
    if tmp.exists() and not tasks_file.exists():
        try:
            tmp.rename(tasks_file)
        except OSError:
            pass
    elif tmp.exists():
        try:
            tmp.unlink()
        except OSError:
            pass

    if not tasks_file.exists():
        return
    try:
        data = json.loads(tasks_file.read_text())
        interrupted_count = 0
        for tid, d in data.items():
            # Skip completed/cancelled tasks older than 24h
            if d.get("status") in ("completed", "cancelled"):
                finished = d.get("finished_at") or 0
                if time.time() - finished > 86400:
                    continue
            # Coerce float fields that may be None from JSON
            filtered = {k: v for k, v in d.items() if k in task_cls.__dataclass_fields__}
            for float_field in ("created_at", "started_at", "finished_at"):
                if float_field in filtered and filtered[float_field] is None:
                    filtered[float_field] = 0.0
            task = task_cls(**filtered)

            # Smart recovery for tasks active during crash
            if task.status == "pending":
                # Never started — keep as pending (safe to re-queue)
                pass
            elif task.status in _ACTIVE_PROCESS_STATUSES:
                # Was executing — mark as interrupted, merge checkpoint data
                old_status = task.status
                merge_kwargs: dict = {
                    "status": "interrupted",
                    "error": f"Supervisor restarted while task was {old_status}",
                    "finished_at": time.time(),
                }

                ckpt = load_checkpoint(tid, checkpoint_dir)
                if ckpt:
                    # Prefer checkpoint data when it's more complete
                    ckpt_steps = ckpt.get("steps_completed", [])
                    if len(ckpt_steps) > len(task.steps_completed):
                        merge_kwargs["steps_completed"] = ckpt_steps
                    ckpt_step = ckpt.get("current_step", "")
                    if ckpt_step:
                        merge_kwargs["current_step"] = ckpt_step
                    partial = ckpt.get("partial_result", "")
                    if partial and not task.result:
                        merge_kwargs["result"] = partial
                    ckpt_sid = ckpt.get("session_id", "")
                    if ckpt_sid and not task.session_id:
                        merge_kwargs["session_id"] = ckpt_sid

                task = dc_replace(task, **merge_kwargs)
                interrupted_count += 1
                logger.warning(
                    "Task %s: %s → interrupted (session=%s, steps=%d)",
                    tid[:8], old_status, bool(task.session_id), len(task.steps_completed),
                )

            with lock:
                tasks[tid] = task

        if interrupted_count:
            logger.info(
                "Loaded %d tasks (%d interrupted, recoverable via /recover)",
                len(tasks), interrupted_count,
            )
        else:
            logger.info("Loaded %d tasks from disk", len(tasks))
    except Exception as e:
        logger.error("Failed to load tasks: %s", e)
