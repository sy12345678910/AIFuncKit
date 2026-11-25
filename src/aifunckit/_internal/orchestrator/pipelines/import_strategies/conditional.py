from __future__ import annotations

from typing import Any, Iterable

from aifunckit._internal.schemas.pipeline_context import PipelineContext
from aifunckit._internal.orchestrator.pipelines.import_strategies.base import ExecutionStrategy
from aifunckit._internal.orchestrator.pipelines.sequential import SequentialPipeline


class ConditionalExecutionStrategy(ExecutionStrategy):
    strategy_type = "conditional"

    async def execute(self, context: PipelineContext, nodes: Iterable[Any]) -> None:
        remaining: list[Any] = []
        for node in list(nodes):
            if not self.pipeline._is_conditional_node(node):
                remaining.append(node)
                continue

            if remaining:
                await self._run_sequential_branch(remaining, context)
                remaining = []

            await self.pipeline._execute_conditional_node(node, context)

        if remaining:
            await self._run_sequential_branch(remaining, context)

    async def _run_sequential_branch(self, tasks: list[Any], context: PipelineContext) -> None:
        original_tasks = context.decomposed_tasks
        context.decomposed_tasks = list(tasks)
        await SequentialPipeline(
            routing=self.pipeline.routing,
            extraction=self.pipeline.extraction,
            execution=self.pipeline.execution,
            circuit_breaker=self.pipeline.circuit_breaker,
        ).execute(context)
        context.decomposed_tasks = original_tasks
