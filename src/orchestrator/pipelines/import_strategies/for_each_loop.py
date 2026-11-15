from __future__ import annotations

from typing import Any, Iterable

from schemas.pipeline_context import PipelineContext
from orchestrator.pipelines.import_strategies.base import ExecutionStrategy


class ForEachLoopExecutionStrategy(ExecutionStrategy):
    strategy_type = "for_each_loop"

    async def execute(self, context: PipelineContext, nodes: Iterable[Any]) -> None:
        for node in list(nodes):
            is_for_each = getattr(self.pipeline, "_is_for_each_node", None)
            if is_for_each and not is_for_each(node):
                continue
            await self.pipeline.execute_for_node(node, context)

    async def run_nodes(self, nodes: Iterable[Any], context: PipelineContext) -> None:
        await self.execute(context, nodes)


# Backward-compatible spelling alias
ForEachLoopExexutionStrategy = ForEachLoopExecutionStrategy
