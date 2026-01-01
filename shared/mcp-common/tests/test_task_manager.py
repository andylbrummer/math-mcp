"""Tests for TaskManager."""

import asyncio

import pytest
from mcp_common.task_manager import TaskManager, TaskStatus


def test_task_manager_singleton() -> None:
    """Test TaskManager is a singleton."""
    tm1 = TaskManager.get_instance()
    tm2 = TaskManager.get_instance()
    assert tm1 is tm2


@pytest.mark.asyncio
async def test_create_task_success() -> None:
    """Test creating a successful task."""
    tm = TaskManager.get_instance()

    async def simple_task() -> str:
        await asyncio.sleep(0.1)
        return "success"

    task_id = tm.create_task("test_task", simple_task())

    # Wait a bit for task to complete
    await asyncio.sleep(0.2)

    task = tm.get_task(task_id)
    assert task is not None
    assert task.status == TaskStatus.COMPLETED
    assert task.result == "success"
    assert task.error is None


@pytest.mark.asyncio
async def test_create_task_failure() -> None:
    """Test creating a failing task."""
    tm = TaskManager.get_instance()

    async def failing_task() -> None:
        await asyncio.sleep(0.1)
        msg = "test error"
        raise ValueError(msg)

    task_id = tm.create_task("failing_task", failing_task())

    # Wait for task to fail
    await asyncio.sleep(0.2)

    task = tm.get_task(task_id)
    assert task is not None
    assert task.status == TaskStatus.FAILED
    assert task.error == "test error"


@pytest.mark.asyncio
async def test_update_progress() -> None:
    """Test updating task progress."""
    tm = TaskManager.get_instance()

    async def long_task() -> None:
        await asyncio.sleep(0.5)

    task_id = tm.create_task("long_task", long_task())

    # Update progress
    await asyncio.sleep(0.1)
    tm.update_progress(task_id, {"percentage": 50, "step": 5})

    task = tm.get_task(task_id)
    assert task is not None
    assert task.progress["percentage"] == 50
    assert task.progress["step"] == 5


@pytest.mark.asyncio
async def test_cancel_task() -> None:
    """Test cancelling a task."""
    tm = TaskManager.get_instance()

    async def long_task() -> None:
        await asyncio.sleep(10.0)

    task_id = tm.create_task("long_task", long_task())

    # Cancel the task
    await asyncio.sleep(0.1)
    success = tm.cancel_task(task_id)
    assert success

    # Wait a bit
    await asyncio.sleep(0.2)

    task = tm.get_task(task_id)
    assert task is not None
    assert task.status == TaskStatus.CANCELLED


def test_list_tasks() -> None:
    """Test listing tasks."""
    tm = TaskManager.get_instance()

    # List all tasks
    tasks = tm.list_tasks()
    assert isinstance(tasks, list)

    # List by status
    completed_tasks = tm.list_tasks(status=TaskStatus.COMPLETED)
    assert isinstance(completed_tasks, list)
