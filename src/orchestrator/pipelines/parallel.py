from __future__ import annotations

from typing import Any

from schemas.pipeline_context import PipelineContext
from orchestrator.pipelines.base import BasePipeline
from orchestrator.pipelines.import_strategies.parallel import ParallelExecutionStrategy


class ParallelPipeline(BasePipeline):
    """Execute independent tasks concurrently with asyncio.gather."""

    pipeline_type = "parallel"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.strategy = ParallelExecutionStrategy(self)

    async def execute(self, context: PipelineContext) -> None:
        await self.strategy.execute(context, list(context.decomposed_tasks))
