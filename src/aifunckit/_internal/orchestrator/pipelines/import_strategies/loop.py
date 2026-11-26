from __future__ import annotations

import asyncio
from typing import Any, Iterable

from aifunckit._internal.schemas.pipeline_context import PipelineContext
from aifunckit._internal.orchestrator.pipelines.sequential import SequentialPipeline
from aifunckit._internal.orchestrator.pipelines.import_strategies.base import ExecutionStrategy


class LoopExecutionStrategy(ExecutionStrategy):
    strategy_type = "loop"

    async def execute(self, context: PipelineContext, nodes: Iterable[Any]) -> None:
        for node in list(nodes):
            if not self.pipeline._is_loop_node(node):
                continue
            await self.pipeline.execute_for_node(node, context)

    async def run_nodes(self, nodes: Iterable[Any], context: PipelineContext) -> None:
        await self.execute(context, nodes)
