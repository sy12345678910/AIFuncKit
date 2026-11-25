from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from aifunckit._internal.log_config import logger
from aifunckit._internal.schemas.pipeline_context import PipelineContext
from aifunckit._internal.orchestrator.general_methods import maybe_await
from aifunckit._internal.orchestrator.services.action_execution import ActionExecutionService
from aifunckit._internal.orchestrator.services.argument_extraction import ArgumentExtractionService
from aifunckit._internal.orchestrator.services.plugin_routing import PluginRoutingService
from aifunckit._internal.orchestrator.services.break_circus import BreakCircusService
from aifunckit._internal.schemas.memory_schemas import MemoryRecord, RecordData
from aifunckit.exceptions.pipeline_exceptions import (
    InvalidPipelineStateException,
    InvaildPipelineStateException,
)


class BasePipeline(ABC):
    """Abstract execution topology that operates on a shared pipeline context."""

    pipeline_type: str = "base"

    def __init__(
        self,
        routing: Optional[PluginRoutingService] = None,
        extraction: Optional[ArgumentExtractionService] = None,
        execution: Optional[ActionExecutionService] = None,
        circuit_breaker: Optional[BreakCircusService] = None,
        circus_breaker: Optional[BreakCircusService] = None,
        breaker_circus: Optional[BreakCircusService] = None,
    ) -> None:
        self.routing = routing or PluginRoutingService()
        self.extraction = extraction or ArgumentExtractionService()
        self.execution = execution or ActionExecutionService()
        self.strategy: Any = None
        self.circuit_breaker = (
            circuit_breaker or circus_breaker or breaker_circus or BreakCircusService()
        )

    @property
    def circus_breaker(self) -> BreakCircusService:
        return self.circuit_breaker

    @circus_breaker.setter
    def circus_breaker(self, value: BreakCircusService) -> None:
        self.circuit_breaker = value

    @property
    def breaker_circus(self) -> BreakCircusService:
        return self.circuit_breaker

    @breaker_circus.setter
    def breaker_circus(self, value: BreakCircusService) -> None:
        self.circuit_breaker = value

    @abstractmethod
    async def execute(self, context: PipelineContext) -> None:
        """Run this topology against the provided context."""
        
    async def _execute_task_node(
        self,
        task: Any,
        context: PipelineContext,
        plugin_meta: Optional[dict[str, Any]] = None,
    ) -> Optional[dict[str, Any]]:
        """execute one task"""
        if hasattr(self, "circuit_breaker"):
            self.circuit_breaker.check_status()

        result = await self.process_task(task, context, plugin_meta=plugin_meta)
        if result is None:
            if hasattr(self, "circuit_breaker"):
                self.circuit_breaker.record_failure()
            return None

        func_data, task_ctx, task_emb = result
        try:
            executed = await self.execution.execute(func_data)
            func_data["result"] = executed
            context.execution_results.append(func_data)

            self._update_context_after_execution(func_data, executed, context, task)
            await self.persist_extraction(context, task, task_ctx, func_data, task_emb)

            if hasattr(self, "circuit_breaker"):
                self.circuit_breaker.record_succes()

            return func_data
        except Exception as exc:
            logger.error("Pipeline task '%s' execution failed: %s", task, exc, exc_info=True)
            if hasattr(self, "circuit_breaker"):
                self.circuit_breaker.record_failure()
            return None

    def _update_context_after_execution(
        self,
        func_data: dict[str, Any],
        executed: Any,
        context: PipelineContext,
        task: Any,
    ) -> None:
        """Update plugin's live context and result of steps in enviroment"""
        plugin_name = func_data.get("plugin_name")
        target_method = func_data.get("method")
        
        if target_method and hasattr(target_method, "__self__"):
            plugin_inst = getattr(target_method, "__self__", None)
            if plugin_inst and hasattr(plugin_inst, "get_live_context"):
                live_ctx = plugin_inst.get_live_context()
                if plugin_name and isinstance(live_ctx, dict):
                    context.environment_state.setdefault(plugin_name, {}).update(live_ctx)

        if executed is not None:
            context.environment_state["_last_result"] = executed
            action_key = func_data.get("action_name") or str(task)
            context.environment_state.setdefault("_step_results", {})[action_key] = executed

    async def process_task(
        self,
        task: Any,
        context: PipelineContext,
        plugin_meta: Optional[dict[str, Any]] = None,
    ) -> Optional[tuple[dict[str, Any], dict[str, Any], NDArray[np.float32]]]:
        """Route, extract arguments, and prepare a single atomic task."""
        task_str: str = str(task)
        try:
            if plugin_meta is None:
                try:
                    plugin_meta = await self.routing.route(task_str, context)
                except Exception as exc:
                    logger.error(
                        "Routing failed for task '%s': %s",
                        task_str,
                        exc,
                        exc_info=True,
                    )
                    plugin_meta = {
                        "name": task_str,
                        "execution_type": "parallel",
                        "type": "general",
                        "file": "main",
                    }

            plugin_name: str = plugin_meta.get("plugin_name", "")
            plugin_context: dict[str, Any] = context.environment_state.get(plugin_name, {})
            expected_schema: dict[str, Any] = plugin_meta.get("args", {})

            task_emb: NDArray[np.float32] = await maybe_await(
                context.embedding._encode_text(task_str)
            )
            try:
                memory_example: Optional[MemoryRecord] = await context.memory.search_relevant_example(
                    "extraction", task_emb
                )
            except Exception as exc:
                logger.error("RAG lookup for extraction failed: %s", exc, exc_info=True)
                memory_example = None

            needable_args: dict[str, Any] = await self.extraction.extract(
                task_str,
                expected_schema,
                plugin_context,
                context.llm_config,
                memory_example,
            )
            func_data: dict[str, Any] = self.execution.build_payload(plugin_meta, needable_args)
            return func_data, plugin_context, task_emb
        except Exception as exc:
            logger.error(
                "Plugin task '%s' failed: %s",
                task_str,
                exc,
                exc_info=True,
            )
            
            return None
    
    

    async def persist_extraction(
        self,
        context: PipelineContext,
        task: Any,
        task_ctx: dict[str, Any],
        func_data: dict[str, Any],
        task_emb: NDArray[np.float32],
    ) -> None:
        """Store a successful extraction example in runtime memory."""
        try:
            clean_args: dict[str, Any] = func_data.get("args") or {}
            record_data = RecordData(
                layer="extraction",
                user_prompt=str(task),
                context_dict=task_ctx,
                args=clean_args,
                raw_vector=task_emb,
            )
            await context.memory.add_record(record_data=record_data)
        except Exception as exc:
            logger.error(
                "Failed to persist extraction memory for '%s': %s",
                task,
                exc,
                exc_info=True,
            )

    def _is_conditional_node(self, node: Any) -> bool:
        return isinstance(node, dict) and "condition_variable" in node

    def _is_loop_node(self, node: Any) -> bool:
        return isinstance(node, dict) and (
            "loop_variable" in node or "iterations" in node or "repetitions" in node
        )

    def _is_for_each_node(self, node: Any) -> bool:
        return isinstance(node, dict) and (
            "items" in node or "for_each" in node or "collection" in node
        )

    def _is_timeout_node(self, node: Any) -> bool:
        return isinstance(node, dict) and ("timeout" in node or "timeout_seconds" in node)

    def validate_state(self, context: PipelineContext) -> None:
        """Validate pipeline context and tasks before execution."""
        if context is None:
            raise InvalidPipelineStateException("Pipeline context is empty or None.")

        if not context.metadata or not context.metadata.cached_metadata:
            raise InvalidPipelineStateException(
                "No registered plugins or actions found in pipeline metadata."
            )

        if not context.decomposed_tasks:
            raise InvalidPipelineStateException("No tasks provided for pipeline execution.")

        for task in context.decomposed_tasks:
            if isinstance(task, dict):
                if self._is_conditional_node(task) and not task.get("condition_variable"):
                    raise InvalidPipelineStateException(
                        "Conditional task missing required 'condition_variable'."
                    )
                if self._is_loop_node(task):
                    reps = task.get("repetitions") or task.get("iterations")
                    if reps is not None and int(reps) < 0:
                        raise InvalidPipelineStateException(
                            f"Loop repetitions must be non-negative, got {reps}."
                        )
                if self._is_timeout_node(task):
                    timeout = task.get("timeout") or task.get("timeout_seconds")
                    if timeout is not None and float(timeout) < 0:
                        raise InvalidPipelineStateException(
                            f"Timeout value must be non-negative, got {timeout}."
                        )
                if self._is_for_each_node(task):
                    items = task.get("items") or task.get("for_each") or task.get("collection")
                    if items is None:
                        raise InvalidPipelineStateException(
                            "ForEach task missing required 'items' collection."
                        )

    def _pre_validation(
        self,
        tasks: Optional[Any] = None,
        context: Optional[PipelineContext] = None,
    ) -> None:
        """Compatibility wrapper for pipeline pre-validation."""
        if isinstance(tasks, PipelineContext):
            self.validate_state(tasks)
        elif context is not None:
            self.validate_state(context)
