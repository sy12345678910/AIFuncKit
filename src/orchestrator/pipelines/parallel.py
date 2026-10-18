from __future__ import annotations
import asyncio
from typing import Any
from log_config import logger
from schemas.pipeline_context import PipelineContext
from orchestrator.pipelines.base import BasePipeline


class ParallelPipeline(BasePipeline):
    """Execute independent tasks concurrently with asyncio.gather."""

    async def execute(self, context: PipelineContext) -> None:
        nodes: list[Any] = [
            node for node in context.decomposed_tasks if not self._is_conditional_node(node)
        ]
        await self.run_nodes(nodes, context)

    async def run_nodes(self, nodes: list[Any], context: PipelineContext) -> None:
        """Resolve and run a set of independent tasks concurrently."""
        task_workers = [self.process_task(task, context) for task in nodes]
        
        workers_results: list[Any] = await asyncio.gather(*task_workers, return_exceptions=True)

        for task, result in zip(nodes, workers_results):
            if isinstance(result, Exception):
                logger.error(
                    "Parallel pipeline task '%s' raised a critical exception: %s",
                    task,
                    result,
                    exc_info=(type(result), result, result.__traceback__),
                )
                continue            
            if result is None:
                continue
            func_data, task_ctx, task_emb = result
            context.execution_results.append(func_data)
            await self.persist_extraction(context, task, task_ctx, func_data, task_emb)
