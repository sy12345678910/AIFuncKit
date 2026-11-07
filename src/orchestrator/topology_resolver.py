from __future__ import annotations

from typing import Any, Callable, ClassVar, Iterable

from schemas.pipeline_context import PipelineContext
from orchestrator.pipelines.base import BasePipeline
from orchestrator.pipelines.conditional import ConditionalPipeline
from orchestrator.pipelines.loop import LoopPipeline
from orchestrator.pipelines.parallel import ParallelPipeline
from orchestrator.pipelines.sequential import SequentialPipeline
from orchestrator.pipelines.timeout import TimeoutPipeline

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


def _has_parallel_keys(tasks: Iterable[Any]) -> bool:
    return any(
        isinstance(node, dict) and node.get("execution_type") == "parallel"
        for node in tasks or []
    ) or any(
        isinstance(node, dict) and node.get("type") == "parallel" for node in tasks or []
    )


def _has_conditional_keys(tasks: Iterable[Any]) -> bool:
    return any(isinstance(node, dict) and "condition_variable" in node for node in tasks or [])


def _has_loop_keys(tasks: Iterable[Any]) -> bool:
    return any(
        isinstance(node, dict)
        and ("loop_variable" in node or "iterations" in node or "repetitions" in node)
        for node in tasks or []
    )


def _has_timeout_keys(tasks: Iterable[Any]) -> bool:
    return any(
        isinstance(node, dict) and ("timeout" in node or "timeout_seconds" in node)
        for node in tasks or []
    )


if not TopologyResolver._matchers:
    TopologyResolver.register(_has_parallel_keys, ParallelPipeline)
    TopologyResolver.register(_has_conditional_keys, ConditionalPipeline)
    TopologyResolver.register(_has_loop_keys, LoopPipeline)
    TopologyResolver.register(_has_timeout_keys, TimeoutPipeline)
