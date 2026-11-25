from __future__ import annotations

import asyncio
from typing import Any

from aifunckit._internal.schemas.pipeline_context import PipelineContext
from aifunckit._internal.orchestrator.pipelines.base import BasePipeline
from aifunckit._internal.orchestrator.pipelines.import_strategies.timeout import TimeoutExecutionStrategy


class TimeoutPipeline(BasePipeline):
    """Execute a task with a timeout guard and abort if it overruns."""

    pipeline_type = "timeout"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.strategy = TimeoutExecutionStrategy(self)

    async def execute(self, context: PipelineContext) -> None:
        await self.execute_with_timeout(context, list(context.decomposed_tasks))

    async def execute_with_timeout(self, context: PipelineContext, nodes: list[Any]) -> None:
        await self.strategy.execute(context, nodes)

    async def execute_for_node(self, node: Any, context: PipelineContext) -> None:
        timeout_seconds = float(node.get("timeout") if isinstance(node, dict) else 0.0)
        task = node.get("task") if isinstance(node, dict) else node
        if not timeout_seconds:
            await self._execute_task_node(task, context)
            return

        try:
            await asyncio.wait_for(self._execute_task_node(task, context), timeout=timeout_seconds)
        except asyncio.TimeoutError:
            context.execution_results.append({
                "action_name": str(task),
                "plugin_name": "timeout",
                "method": str(task),
                "args": {},
                "result": {"timeout": timeout_seconds, "status": "timed_out"},
            })
    