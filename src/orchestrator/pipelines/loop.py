from __future__ import annotations

from typing import Any

from schemas.pipeline_context import PipelineContext
from orchestrator.pipelines.base import BasePipeline
from orchestrator.pipelines.strategies.loop import LoopExecutionStrategy


class LoopPipeline(BasePipeline):
    """Repeat one or more child tasks for a configured number of iterations."""

    pipeline_type = "loop"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.strategy = LoopExecutionStrategy(self)

    async def execute(self, context: PipelineContext) -> None:
        await self.run_nodes(list(context.decomposed_tasks), context)

    async def run_nodes(self, nodes: list[Any], context: PipelineContext) -> None:
        await self.strategy.execute(context, nodes)

    async def execute_for_node(self, node: Any, context: PipelineContext) -> None:
        from orchestrator.pipelines.sequential import SequentialPipeline

        loop_node = node if isinstance(node, dict) else {"task": str(node), "iterations": 1}
        body = loop_node.get("body") or loop_node.get("tasks") or [loop_node.get("task")]
        iterations = int(loop_node.get("iterations") or loop_node.get("repetitions") or 1)

        original_tasks = context.decomposed_tasks
        for _ in range(max(0, iterations)):
            context.decomposed_tasks = list(body)
            await SequentialPipeline(
                routing=self.routing,
                extraction=self.extraction,
                execution=self.execution,
            ).execute(context)
        context.decomposed_tasks = original_tasks