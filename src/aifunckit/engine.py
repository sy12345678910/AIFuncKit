from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from aifunckit._internal.engine.plugin_register import PluginRegisterClass
from aifunckit._internal.engine.runtime_manager import RuntimeManager
from aifunckit._internal.log_config import configure_logging, logger
from aifunckit._internal.orchestrator.pipeline import MasterOrchestrationEngine
from aifunckit._internal.orchestrator.pipelines.schedule import SchedulerService
from aifunckit.sdk import BasePlugin


class AIEngine:
    """High-level public facade coordinating plugin registration and runtime execution."""

    def __init__(
        self,
        ai_model: str = "gemma4:e4b",
        context_window: int = 4096,
        db_path: Path | str = Path("aifunckit_runtine"),
        provider: str = "ollama",
        emb_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
        routing_model: str = "vector",
        verbose: Optional[bool] = None,
        log_to_file: Optional[str] = "aifunckit.log",
        log_level: Optional[str | int] = None,
        failure_threshold: int = 3,
        prompt_injection_protection: Optional[List[str]] = None,
        max_prompt_length: int = 4096,
        guard_model_path: Optional[str] = None,
        guard_tokenizer_path: Optional[str] = None,
        guard_threshold: float = 0.85,
    ) -> None:
        """Initialize logging, shared metadata, orchestrator, and runtime managers."""
        configure_logging(verbose=verbose, log_to_file=log_to_file, log_level=log_level)
        self.verbose: Optional[bool] = verbose
        self.log_level: Optional[str | int] = log_level
        self.log_to_file: Optional[str] = log_to_file
        self.embidding_model: str = emb_model

        self.cached_metadata: list[dict[str, Any]] = []
        self.actions: dict[str, Callable[..., Any]] = {}
        self.plugins: dict[str, BasePlugin] = {}

        self.orchestrator: MasterOrchestrationEngine = MasterOrchestrationEngine(
            cached_metadata=self.cached_metadata,
            provider=provider,
            model_name=ai_model,
            context_window=context_window,
            db_path=str(db_path),
            emb_model=emb_model,
            routing_model=routing_model,
            failure_threshold=failure_threshold,
            prompt_injection_protection=prompt_injection_protection,
            max_prompt_length=max_prompt_length,
            guard_model_path=guard_model_path,
            guard_tokenizer_path=guard_tokenizer_path,
            guard_threshold=guard_threshold,
        )
        self.scheduler: SchedulerService = SchedulerService()
        self.plugin_register = PluginRegisterClass(
            cached_metadata=self.cached_metadata,
            actions=self.actions,
            plugins=self.plugins,
        )
        self.runtime_manager = RuntimeManager(
            orchestrator=self.orchestrator,
            scheduler=self.scheduler,
            plugin_register=self.plugin_register,
        )
        logger.info(
            "AIEngine created (provider=%s, model=%s, routing=%s)",
            provider,
            ai_model,
            routing_model,
        )

    async def initialize(self) -> None:
        """Ensure the embedding model and runtime memory cache are initialized."""
        await self.runtime_manager.initialize()

    def register_plugin(self, obj: BasePlugin) -> None:
        """Register a plugin instance and add its decorated methods to the action registry."""
        self.plugin_register.register_plugin(obj)

    def action(
        self,
        description: str,
        name: Optional[str] = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator that registers a function as an engine action with metadata."""
        return self.plugin_register.action(description=description, name=name)

    def get_current_global_context(self) -> dict[str, Any]:
        """Collect the live state from all registered plugins into one context map."""
        return self.plugin_register.get_current_global_context()

    async def run_engine(self, user_prompt: str) -> List[Dict[str, Any]]:
        """Execute the pipeline for a prompt and return the action results."""
        return await self.runtime_manager.run_engine(user_prompt)

    def schedule_task(
        self,
        prompt: str,
        interval_seconds: float,
        task_id: Optional[str] = None,
    ) -> str:
        """Register a recurring task that calls run_engine(prompt) every interval_seconds."""
        return self.runtime_manager.schedule_task(
            prompt=prompt,
            interval_seconds=interval_seconds,
            task_id=task_id,
        )

    def unschedule_task(self, task_id: str) -> None:
        """Stop and remove a scheduled task by its ID."""
        self.runtime_manager.unschedule_task(task_id)

    def list_scheduled_tasks(self) -> List[Dict[str, Any]]:
        """Return info snapshots for all currently running scheduled tasks."""
        return self.runtime_manager.list_scheduled_tasks()

    async def delete_engine(self) -> None:
        """Release orchestrator resources and clear all engine state."""
        await self.runtime_manager.delete_engine()

    def delete_plugin(self, name: str) -> None:
        """Remove a plugin and all of its registered actions from the engine."""
        self.plugin_register.delete_plugin(name)

    def stop_engine(self, reason: str = "Stopped by user") -> None:
        """Emergency brake."""
        self.runtime_manager.stop_engine(reason=reason)

    def resume_engine(self) -> None:
        """Resume emergency brake."""
        self.runtime_manager.resume_engine()
