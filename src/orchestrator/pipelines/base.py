from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from log_config import logger
from schemas.pipeline_context import PipelineContext
from orchestrator.general_methods import maybe_await
from orchestrator.services.action_execution import ActionExecutionService
from orchestrator.services.argument_extraction import ArgumentExtractionService
from orchestrator.services.plugin_routing import PluginRoutingService
from schemas.memory_schemas import MemoryRecord, RecordData


class BasePipeline(ABC):
    """Abstract execution topology that operates on a shared pipeline context."""

    def __init__(
        self,
        routing: Optional[PluginRoutingService] = None,
        extraction: Optional[ArgumentExtractionService] = None,
        execution: Optional[ActionExecutionService] = None,
    ) -> None:
        self.routing = routing or PluginRoutingService()
        self.extraction = extraction or ArgumentExtractionService()
        self.execution = execution or ActionExecutionService()

    @abstractmethod
    async def execute(self, context: PipelineContext) -> None:
        """Run this topology against the provided context."""

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
            record_data = RecordData(
                layer="extraction",
                user_prompt=str(task),
                context_dict=task_ctx,
                args=func_data,
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
