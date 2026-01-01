"""Task Manager - MCP Tasks primitive for long-running operations."""

import asyncio
import logging
import uuid
from collections.abc import Coroutine
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from threading import Lock
from typing import Any, Optional

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """Task execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """Represents a long-running task."""

    task_id: str
    name: str
    status: TaskStatus = TaskStatus.PENDING
    progress: dict[str, Any] = field(default_factory=dict)
    result: Any | None = None
    error: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))
    started_at: datetime | None = None
    completed_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class TaskManager:
    """Manager for async tasks with progress tracking."""

    _instance: Optional["TaskManager"] = None
    _lock = Lock()

    def __init__(self) -> None:
        """Initialize task manager (use get_instance() instead)."""
        if TaskManager._instance is not None:
            msg = "Use TaskManager.get_instance() instead of direct instantiation"
            raise RuntimeError(msg)

        self._tasks: dict[str, Task] = {}
        self._task_futures: dict[str, asyncio.Task[Any]] = {}
        self._tasks_lock = Lock()

    @classmethod
    def get_instance(cls) -> "TaskManager":
        """Get singleton instance of TaskManager."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def create_task(
        self,
        name: str,
        coro: Coroutine[Any, Any, Any],
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Create and start a new async task."""
        task_id = str(uuid.uuid4())

        task = Task(
            task_id=task_id,
            name=name,
            metadata=metadata or {},
        )

        with self._tasks_lock:
            self._tasks[task_id] = task

        # Wrap coroutine to update task status
        async def wrapped_coro() -> Any:
            try:
                task.status = TaskStatus.RUNNING
                task.started_at = datetime.now(tz=UTC)
                logger.info("Task %s (%s) started", task_id, name)

                result = await coro

                task.status = TaskStatus.COMPLETED
                task.result = result
                task.completed_at = datetime.now(tz=UTC)
                logger.info("Task %s (%s) completed", task_id, name)

                return result  # noqa: TRY300

            except asyncio.CancelledError:
                task.status = TaskStatus.CANCELLED
                task.completed_at = datetime.now(tz=UTC)
                logger.info("Task %s (%s) cancelled", task_id, name)
                raise

            except Exception as e:
                task.status = TaskStatus.FAILED
                task.error = str(e)
                task.completed_at = datetime.now(tz=UTC)
                logger.exception("Task %s (%s) failed", task_id, name)
                raise

        # Create asyncio task
        future = asyncio.create_task(wrapped_coro())
        with self._tasks_lock:
            self._task_futures[task_id] = future

        return task_id

    def get_task(self, task_id: str) -> Task | None:
        """Get task by ID."""
        with self._tasks_lock:
            return self._tasks.get(task_id)

    def update_progress(
        self,
        task_id: str,
        progress: dict[str, Any],
    ) -> None:
        """Update task progress."""
        with self._tasks_lock:
            if task_id in self._tasks:
                self._tasks[task_id].progress = progress

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task."""
        with self._tasks_lock:
            if task_id not in self._task_futures:
                return False

            future = self._task_futures[task_id]
            future.cancel()
            return True

    def cleanup_completed_tasks(self, max_age_seconds: int = 3600) -> int:
        """Clean up old completed tasks."""
        now = datetime.now(tz=UTC)
        removed = 0

        with self._tasks_lock:
            task_ids = list(self._tasks.keys())
            for task_id in task_ids:
                task = self._tasks[task_id]
                if (
                    task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED)
                    and task.completed_at is not None
                    and (now - task.completed_at).total_seconds() > max_age_seconds
                ):
                    del self._tasks[task_id]
                    if task_id in self._task_futures:
                        del self._task_futures[task_id]
                    removed += 1

        if removed > 0:
            logger.info("Cleaned up %d old tasks", removed)

        return removed

    def list_tasks(self, status: TaskStatus | None = None) -> list[Task]:
        """List all tasks, optionally filtered by status."""
        with self._tasks_lock:
            tasks = list(self._tasks.values())

        if status is not None:
            tasks = [t for t in tasks if t.status == status]

        return sorted(tasks, key=lambda t: t.created_at, reverse=True)
