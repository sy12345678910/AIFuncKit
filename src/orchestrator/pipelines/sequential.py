from __future__ import annotations

import copy
from typing import Any

from log_config import logger
from schemas.pipeline_context import PipelineContext
from orchestrator.pipelines.base import BasePipeline


class SequentialPipeline(BasePipeline):
    """Execute atomic tasks one after another on a shared pipeline context."""

    async def execute(self, context: PipelineContext) -> None:
        """Iterate over atomic tasks sequentially through routing, extraction, and execution."""
        for node in list(context.decomposed_tasks):
            if self._is_conditional_node(node):
                from orchestrator.pipelines.conditional import ConditionalPipeline

                branch_context = copy.copy(context)
                branch_context.decomposed_tasks = [node]
                
                await ConditionalPipeline(
                    routing=self.routing,
                    extraction=self.extraction,
                    execution=self.execution,
                ).execute(branch_context)
                
                context.execution_results = branch_context.execution_results
                continue

            try:
                result = await self.process_task(node, context)
                if result is None:
                    continue
                func_data, task_ctx, task_emb = result
                context.execution_results.append(func_data)
                await self.persist_extraction(context, node, task_ctx, func_data, task_emb)
            except Exception as exc:
                logger.error(
                    "Sequential plugin task '%s' failed: %s",
                    node,
                    exc,
                    exc_info=True,
                )
                continue
