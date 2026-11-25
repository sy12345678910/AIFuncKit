from __future__ import annotations

from typing import Any, Callable, ClassVar, Iterable

from aifunckit._internal.schemas.pipeline_context import PipelineContext
from aifunckit._internal.orchestrator.pipelines.base import BasePipeline
from aifunckit._internal.orchestrator.pipelines.conditional import ConditionalPipeline
from aifunckit._internal.orchestrator.pipelines.loop import LoopPipeline
from aifunckit._internal.orchestrator.pipelines.parallel import ParallelPipeline
from aifunckit._internal.orchestrator.pipelines.sequential import SequentialPipeline
from aifunckit._internal.orchestrator.pipelines.timeout import TimeoutPipeline
from aifunckit._internal.orchestrator.pipelines.for_each_loop import ForEachLoopPipeline

Matcher = Callable[[Iterable[Any]], bool]


class TopologyResolver:
    """Choose a pipeline topology from decomposed task structure via a matcher registry."""

    _matchers: ClassVar[list[tuple[Matcher, type[BasePipeline]]]] = []
    _default: ClassVar[type[BasePipeline]] = SequentialPipeline

    @classmethod
    def register(cls, matcher: Matcher, pipeline_cls: type[BasePipeline]) -> None:
        """Register a topology matcher without modifying the resolver core."""
        cls._matchers.append((matcher, pipeline_cls))

    @classmethod
    def resolve(cls, context: Any) -> type[BasePipeline]:
        """Return the pipeline class that matches the current decomposed task structure."""
        tasks = getattr(context, "decomposed_tasks", context)
        for matcher, pipeline_cls in cls._matchers:
            if matcher(tasks):
                return pipeline_cls
        return cls._default


def _has_parallel_keys(tasks: Iterable[Any]) -> bool:
    for node in tasks or []:
        if isinstance(node, dict) and (node.get("execution_type") == "parallel" or node.get("type") == "parallel"):
            return True
        if hasattr(node, "execution_type") and getattr(node, "execution_type") == "parallel":
            return True
    return False


def _has_conditional_keys(tasks: Iterable[Any]) -> bool:
    for node in tasks or []:
        if isinstance(node, dict) and "condition_variable" in node:
            return True
        if hasattr(node, "condition_variable"):
            return True
    return False


def _has_loop_keys(tasks: Iterable[Any]) -> bool:
    for node in tasks or []:
        if isinstance(node, dict) and ("loop_variable" in node or "iterations" in node or "repetitions" in node):
            return True
        if hasattr(node, "loop_variable") or hasattr(node, "repetitions") or hasattr(node, "iterations"):
            return True
    return False


def _has_timeout_keys(tasks: Iterable[Any]) -> bool:
    for node in tasks or []:
        if isinstance(node, dict) and ("timeout" in node or "timeout_seconds" in node):
            return True
        if hasattr(node, "timeout") or hasattr(node, "timeout_seconds"):
            return True
    return False


def _has_for_each_keys(tasks: Iterable[Any]) -> bool:
    for node in tasks or []:
        if isinstance(node, dict):
            if "items" in node or "for_each" in node or "collection" in node:
                return True
        else:
            if hasattr(node, "items") or hasattr(node, "for_each") or hasattr(node, "collection"):
                return True
    return False


if not TopologyResolver._matchers:
    TopologyResolver.register(_has_parallel_keys, ParallelPipeline)
    TopologyResolver.register(_has_conditional_keys, ConditionalPipeline)
    TopologyResolver.register(_has_for_each_keys, ForEachLoopPipeline)
    TopologyResolver.register(_has_loop_keys, LoopPipeline)
    TopologyResolver.register(_has_timeout_keys, TimeoutPipeline)
