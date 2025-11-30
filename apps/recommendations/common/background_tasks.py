from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

logger = logging.getLogger(__name__)

@dataclass
class TaskStatus:
    task_id: str
    status: str
    result: Any = None
    error: str | None = None
    progress: int = 0
    current_step: str = ""
    total_steps: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    started_at: datetime | None = None
    completed_at: datetime | None = None

class BackgroundTaskManager:

    def __init__(self):
        self._tasks: dict[str, TaskStatus] = {}
        self._lock = threading.Lock()

    def submit(
        self,
        func: Callable,
        *args,
        task_id: str | None = None,
        **kwargs
    ) -> str:
        if task_id is None:
            task_id = str(uuid.uuid4())

        task_status = TaskStatus(
            task_id=task_id,
            status="pending"
        )

        with self._lock:
            self._tasks[task_id] = task_status

        def run_task():
            try:
                with self._lock:
                    task_status.status = "running"
                    task_status.started_at = datetime.now()

                result = func(*args, **kwargs)

                with self._lock:
                    task_status.status = "success"
                    task_status.result = result
                    task_status.progress = 100
                    task_status.completed_at = datetime.now()

            except Exception as e:
                logger.error(f"Task {task_id} failed: {e}", exc_info=True)
                with self._lock:
                    task_status.status = "failure"
                    task_status.error = str(e)
                    task_status.completed_at = datetime.now()

        thread = threading.Thread(target=run_task, daemon=True)
        thread.start()

        return task_id

    def get_status(self, task_id: str) -> TaskStatus | None:
        with self._lock:
            return self._tasks.get(task_id)

    def update_progress(
        self,
        task_id: str,
        progress: int,
        current_step: str = "",
        total_steps: str = ""
    ):
        with self._lock:
            if task_id in self._tasks:
                task = self._tasks[task_id]
                task.progress = progress
                task.current_step = current_step
                task.total_steps = total_steps

    def cleanup_old_tasks(self, max_age_hours: int = 24):
        cutoff = datetime.now().timestamp() - (max_age_hours * 3600)
        with self._lock:
            to_remove = [
                task_id for task_id, task in self._tasks.items()
                if task.completed_at and task.completed_at.timestamp() < cutoff
            ]
            for task_id in to_remove:
                del self._tasks[task_id]
            if to_remove:
                logger.info(f"Cleaned up {len(to_remove)} old tasks")

_task_manager = BackgroundTaskManager()

def get_task_manager() -> BackgroundTaskManager:
    return _task_manager

def submit_task(
    func: Callable,
    *args,
    task_id: str | None = None,
    **kwargs
) -> str:
    return _task_manager.submit(func, *args, task_id=task_id, **kwargs)

def get_task_status(task_id: str) -> TaskStatus | None:
    return _task_manager.get_status(task_id)

