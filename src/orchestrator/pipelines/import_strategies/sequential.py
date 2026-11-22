from __future__ import annotations

import copy
from typing import Any, Iterable

from log_config import logger
from schemas.pipeline_context import PipelineContext
from orchestrator.pipelines.import_strategies.base import ExecutionStrategy


class SequentialExecutionStrategy(ExecutionStrategy):
    strategy_type = "sequential"

    async def execute(self, context: PipelineContext, nodes: Iterable[Any]) -> None:
        for node in list(nodes):
            if self.pipeline._is_conditional_node(node):
                from orchestrator.pipelines.conditional import ConditionalPipeline

                branch_context = copy.copy(context)
                branch_context.decomposed_tasks = [node]
                await ConditionalPipeline(
                    routing=self.pipeline.routing,
                    extraction=self.pipeline.extraction,
                    execution=self.pipeline.execution,
                ).execute(branch_context)
                context.execution_results = branch_context.execution_results
                continue

            if self.pipeline._is_loop_node(node):
                from orchestrator.pipelines.loop import LoopPipeline

                loop_pipeline = LoopPipeline(
                    routing=self.pipeline.routing,
                    extraction=self.pipeline.extraction,
                    execution=self.pipeline.execution,
                )
                await loop_pipeline.execute_for_node(node, context)
                continue

            if self.pipeline._is_timeout_node(node):
                from orchestrator.pipelines.timeout import TimeoutPipeline

                timeout_pipeline = TimeoutPipeline(
                    routing=self.pipeline.routing,
                    extraction=self.pipeline.extraction,
                    execution=self.pipeline.execution,
                )
                await timeout_pipeline.execute_for_node(node, context)
                continue

            try:
                await self.pipeline._execute_task_node(node, context)
            except Exception as exc:  # pragma: no cover - defensive fallback
                logger.error(
                    "Sequential plugin task '%s' failed: %s",
                    node,
                    exc,
                    exc_info=True,
                )
                continue
