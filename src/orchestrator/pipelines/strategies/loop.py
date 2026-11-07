from __future__ import annotations

import asyncio
from typing import Any, Iterable

from schemas.pipeline_context import PipelineContext
from orchestrator.pipelines.sequential import SequentialPipeline
from orchestrator.pipelines.strategies.base import ExecutionStrategy


class LoopExecutionStrategy(ExecutionStrategy):
    strategy_type = "loop"

    async def execute(self, context: PipelineContext, nodes: Iterable[Any]) -> None:
        for node in list(nodes):
            if not self.pipeline._is_loop_node(node):
                continue
            await self.pipeline.execute_for_node(node, context)

    async def run_nodes(self, nodes: Iterable[Any], context: PipelineContext) -> None:
        await self.execute(context, nodes)
