from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable

from schemas.pipeline_context import PipelineContext


class ExecutionStrategy(ABC):
    """Strategy that decides how nodes are executed within a pipeline."""

    strategy_type: str = "base"

    def __init__(self, pipeline: Any) -> None:
        self.pipeline = pipeline

    @abstractmethod
    async def execute(self, context: PipelineContext, nodes: Iterable[Any]) -> None:
        """Run the strategy against the current context and node list."""
