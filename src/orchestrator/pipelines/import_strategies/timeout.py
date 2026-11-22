from __future__ import annotations

import asyncio
from typing import Any, Iterable

from schemas.pipeline_context import PipelineContext
from orchestrator.pipelines.import_strategies.base import ExecutionStrategy


class TimeoutExecutionStrategy(ExecutionStrategy):
    strategy_type = "timeout"

    async def execute(self, context: PipelineContext, nodes: Iterable[Any]) -> None:
        for node in list(nodes):
            if not self.pipeline._is_timeout_node(node):
                continue
            await self.pipeline.execute_for_node(node, context)

    async def run_nodes(self, nodes: Iterable[Any], context: PipelineContext) -> None:
        await self.execute(context, nodes)
