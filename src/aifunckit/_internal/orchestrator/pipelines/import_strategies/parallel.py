from __future__ import annotations

import asyncio
from typing import Any, Iterable

from aifunckit._internal.log_config import logger
from aifunckit._internal.schemas.pipeline_context import PipelineContext
from aifunckit._internal.orchestrator.pipelines.import_strategies.base import ExecutionStrategy


class ParallelExecutionStrategy(ExecutionStrategy):
    strategy_type = "parallel"

    async def execute(self, context: PipelineContext, nodes: Iterable[Any]) -> None:
        task_pairs: list[tuple[Any, Any]] = []
        for node in list(nodes):
            if self.pipeline._is_conditional_node(node) or self.pipeline._is_loop_node(node) or self.pipeline._is_timeout_node(node):
                continue
            task_pairs.append((node, self.pipeline._execute_task_node(node, context)))

        if not task_pairs:
            return

        results = await asyncio.gather(*(future for _, future in task_pairs), return_exceptions=True)
        for (node, _), result in zip(task_pairs, results):
            if isinstance(result, Exception):
                logger.error(
                    "Parallel pipeline task '%s' raised a critical exception: %s",
                    node,
                    result,
                    exc_info=(type(result), result, result.__traceback__),
                )
                continue
            if result is None:
                continue
