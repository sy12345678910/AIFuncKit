from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Any, Callable, Optional

from log_config import logger
from schemas.pipeline_context import PipelineContext
from orchestrator.pipelines.base import BasePipeline


class ScheduledTask:
    """A registered recurring pipeline task with its own asyncio background loop."""

    def __init__(
        self,
        task_id: str,
        prompt: str,
        interval_seconds: float,
        run_callback: Callable[..., Any],
    ) -> None:
        self.task_id = task_id
        self.prompt = prompt
        self.interval_seconds = interval_seconds
        self.run_callback = run_callback
        self.is_running: bool = False
        self._asyncio_task: Optional[asyncio.Task[None]] = None
        self.last_run: Optional[datetime] = None
        self.next_run: Optional[datetime] = None
        self.run_count: int = 0
        self.last_error: Optional[str] = None

    async def _loop(self) -> None:
        """Run the task repeatedly at the configured interval."""
        logger.info(
            "Scheduled task '%s' started (interval=%.1fs, prompt='%s')",
            self.task_id,
            self.interval_seconds,
            self.prompt,
        )
        while self.is_running:
            self.next_run = datetime.now() + timedelta(seconds=self.interval_seconds)
            try:
                self.last_run = datetime.now()
                self.run_count += 1
                await self.run_callback(self.prompt)
                self.last_error = None
                logger.info(
                    "Scheduled task '%s' run #%d completed",
                    self.task_id,
                    self.run_count,
                )
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self.last_error = str(exc)
                logger.error(
                    "Scheduled task '%s' run #%d failed: %s",
                    self.task_id,
                    self.run_count,
                    exc,
                    exc_info=True,
                )

            try:
                await asyncio.sleep(self.interval_seconds)
            except asyncio.CancelledError:
                break

    def start(self) -> None:
        """Start the background asyncio task."""
        if self.is_running:
            logger.warning("Scheduled task '%s' is already running", self.task_id)
            return
        self.is_running = True
        self._asyncio_task = asyncio.create_task(
            self._loop(), name=f"aie-schedule-{self.task_id}"
        )

    def stop(self) -> None:
        """Cancel the background asyncio task gracefully."""
        self.is_running = False
        if self._asyncio_task and not self._asyncio_task.done():
            self._asyncio_task.cancel()
        logger.info("Scheduled task '%s' stopped after %d run(s)", self.task_id, self.run_count)

    def info(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot of this task's state."""
        return {
            "task_id": self.task_id,
            "prompt": self.prompt,
            "interval_seconds": self.interval_seconds,
            "is_running": self.is_running,
            "run_count": self.run_count,
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "next_run": self.next_run.isoformat() if self.next_run else None,
            "last_error": self.last_error,
        }


class SchedulerService:
    """Manage a registry of recurring scheduled AIE tasks."""

    def __init__(self) -> None:
        self._tasks: dict[str, ScheduledTask] = {}

    def schedule(
        self,
        prompt: str,
        interval_seconds: float,
        run_callback: Callable[..., Any],
        task_id: Optional[str] = None,
    ) -> str:
        """Register and immediately start a new recurring task. Returns the task_id."""
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be a positive number")
        tid = task_id or str(uuid.uuid4())[:8]
        if tid in self._tasks:
            raise ValueError(
                f"Scheduled task '{tid}' already exists. Call stop(task_id) first."
            )
        task = ScheduledTask(
            task_id=tid,
            prompt=prompt,
            interval_seconds=interval_seconds,
            run_callback=run_callback,
        )
        task.start()
        self._tasks[tid] = task
        logger.info(
            "Registered scheduled task '%s': '%s' every %.1fs",
            tid, prompt, interval_seconds,
        )
        return tid

    def stop(self, task_id: str) -> None:
        """Stop and remove a scheduled task by its ID."""
        task = self._tasks.pop(task_id, None)
        if task is None:
            raise KeyError(f"No scheduled task with id '{task_id}'")
        task.stop()

    def stop_all(self) -> None:
        """Stop and clear all running scheduled tasks."""
        for task in list(self._tasks.values()):
            task.stop()
        self._tasks.clear()
        logger.info("All scheduled tasks stopped")

    def list_tasks(self) -> list[dict[str, Any]]:
        """Return info snapshots for all registered tasks."""
        return [t.info() for t in self._tasks.values()]

    def __len__(self) -> int:
        return len(self._tasks)
