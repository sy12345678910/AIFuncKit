from __future__ import annotations

from typing import Any

from schemas.pipeline_context import PipelineContext
from orchestrator.pipelines.base import BasePipeline
from orchestrator.pipelines.strategies.sequential import SequentialExecutionStrategy


class SequentialPipeline(BasePipeline):
    """Execute atomic tasks one after another on a shared pipeline context."""

    pipeline_type = "sequential"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.strategy = SequentialExecutionStrategy(self)

    async def execute(self, context: PipelineContext) -> None:
        """Iterate over atomic tasks sequentially through routing, extraction, and execution."""
        await self.strategy.execute(context, list(context.decomposed_tasks))
