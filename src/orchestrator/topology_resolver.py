from __future__ import annotations

from typing import Any, Callable, ClassVar, Iterable

from schemas.pipeline_context import PipelineContext
from orchestrator.pipelines.base import BasePipeline
from orchestrator.pipelines.conditional import ConditionalPipeline
from orchestrator.pipelines.sequential import SequentialPipeline

Matcher = Callable[[Iterable[Any]], bool]


class TopologyResolver:
    """Choose a pipeline topology from decomposed task structure via a matcher registry."""

    _matchers: ClassVar[list[tuple[Matcher, type[BasePipeline]]]] = []
    _default: ClassVar[type[BasePipeline]] = SequentialPipeline

    @classmethod
    def register(cls, matcher: Matcher, pipeline_cls: type[BasePipeline]) -> None:
        """Register a topology matcher without modifying the resolver core."""
        cls._matchers.append((matcher, pipeline_cls))

    def resolve(self, context: PipelineContext) -> type[BasePipeline]:
        """Return the pipeline class that matches the current decomposed task structure."""
        for matcher, pipeline_cls in self._matchers:
            if matcher(context.decomposed_tasks):
                return pipeline_cls
        return self._default


def _has_conditional_keys(tasks: Iterable[Any]) -> bool:
    return any(isinstance(node, dict) and "condition_variable" in node for node in tasks or [])


if not TopologyResolver._matchers:
    TopologyResolver.register(_has_conditional_keys, ConditionalPipeline)
