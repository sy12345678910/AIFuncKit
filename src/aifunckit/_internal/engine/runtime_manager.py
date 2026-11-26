from __future__ import annotations

import inspect
from typing import Any, Dict, List, Optional

from aifunckit._internal.engine.plugin_register import PluginRegisterClass
from aifunckit._internal.log_config import logger
from aifunckit._internal.orchestrator.pipeline import MasterOrchestrationEngine
from aifunckit._internal.orchestrator.pipelines.schedule import SchedulerService


class RuntimeManager:
    """Manage engine lifecycle, execution, scheduling, and emergency braking."""

    def __init__(
        self,
        orchestrator: MasterOrchestrationEngine,
        scheduler: SchedulerService,
        plugin_register: PluginRegisterClass,
    ) -> None:
        self.orchestrator = orchestrator
        self.scheduler = scheduler
        self.plugin_register = plugin_register

    async def initialize(self) -> None:
        """Ensure the embedding model and runtime memory cache are initialized."""
        try:
            if hasattr(self.orchestrator, "ensure_embedding_model"):
                await self.orchestrator.ensure_embedding_model()
            if hasattr(self.orchestrator.runtime_memory, "init_and_load_cache"):
                await self.orchestrator.runtime_memory.init_and_load_cache()
            logger.info("AIEngine runtime initialized")
        except Exception as exc:
            logger.error("Failed to initialize AIEngine runtime: %s", exc, exc_info=True)
            raise

    async def run_engine(self, user_prompt: str) -> List[Dict[str, Any]]:
        """Execute the pipeline for a prompt and return the action results."""
        self.orchestrator.breaker_circus.check_status()
        logger.info("Engine run started for prompt: %s", user_prompt)
        current_context: dict[str, Any] = self.plugin_register.get_current_global_context()

        try:
            data_for_execute: list[dict[str, Any]] = await self.orchestrator.run_pipeline(
                user_prompt, current_context
            )
        except Exception as exc:
            logger.error("Orchestrator pipeline failed completely: %s", exc, exc_info=True)
            return []

        results: List[Dict[str, Any]] = []

        for data in data_for_execute:
            target_func: Any = data.get("method")
            args: dict[str, Any] = data.get("args", {})
            action_name: Optional[str] = data.get("action_name")

            if isinstance(target_func, str):
                target_func = self.plugin_register.actions.get(target_func)

            if not target_func or not callable(target_func):
                logger.error(
                    "Action execution aborted: '%s' is not a registered callable function in memory",
                    action_name,
                )
                continue

            try:
                if "result" in data:
                    result = data["result"]
                elif inspect.iscoroutinefunction(target_func):
                    result = await target_func(**args)
                else:
                    result = target_func(**args)

                results.append({"action": action_name, "result": result})
                logger.info("Action '%s' completed successfully", action_name)
            except Exception as exc:
                logger.error(
                    "User plugin action '%s' generated an unhandled exception: %s",
                    action_name,
                    exc,
                    exc_info=True,
                )
                continue

        logger.info("Engine run finished with %s result(s)", len(results))
        return results

    def schedule_task(
        self,
        prompt: str,
        interval_seconds: float,
        task_id: Optional[str] = None,
    ) -> str:
        """Register a recurring task that calls run_engine(prompt) every interval_seconds."""
        return self.scheduler.schedule(
            prompt=prompt,
            interval_seconds=interval_seconds,
            run_callback=self.run_engine,
            task_id=task_id,
        )

    def unschedule_task(self, task_id: str) -> None:
        """Stop and remove a scheduled task by its ID."""
        self.scheduler.stop(task_id)

    def list_scheduled_tasks(self) -> List[Dict[str, Any]]:
        """Return info snapshots for all currently running scheduled tasks."""
        return self.scheduler.list_tasks()

    async def delete_engine(self) -> None:
        """Release orchestrator resources and clear all engine state."""
        try:
            self.scheduler.stop_all()

            if hasattr(self.orchestrator.runtime_memory, "search_engine"):
                se = self.orchestrator.runtime_memory.search_engine
                for k in getattr(se, "_ids", {}):
                    se._ids[k] = []
                for k in getattr(se, "_matrices", {}):
                    se._matrices[k] = None
            elif hasattr(self.orchestrator.runtime_memory, "vector_list_cache"):
                for layer in self.orchestrator.runtime_memory.vector_list_cache:
                    self.orchestrator.runtime_memory.vector_list_cache[layer] = []

            if hasattr(self.orchestrator.runtime_memory, "close"):
                await self.orchestrator.runtime_memory.close()

            if hasattr(self.orchestrator.client, "close"):
                if inspect.iscoroutinefunction(self.orchestrator.client.close):
                    await self.orchestrator.client.close()
                else:
                    self.orchestrator.client.close()

            self.plugin_register.cached_metadata.clear()
            self.plugin_register.actions.clear()
            self.plugin_register.plugins.clear()
            logger.info("AIEngine resources released")
        except Exception as exc:
            logger.error("Failed while shutting down AIEngine: %s", exc, exc_info=True)
            raise

    def stop_engine(self, reason: str = "Stopped by user") -> None:
        """Emergency brake."""
        logger.warning("Emergancy brake %s", reason)
        self.orchestrator.breaker_circus.emergency_halt(reason)
        self.scheduler.stop_all()

    def resume_engine(self) -> None:
        """Resume emergency brake."""
        self.orchestrator.breaker_circus.reset()
        logger.info("Engine is repeared and now it's working")
