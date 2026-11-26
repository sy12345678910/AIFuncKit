from __future__ import annotations

from typing import Any, Callable, ClassVar, Iterable

from aifunckit._internal.log_config import logger
from aifunckit._internal.orchestrator.pipelines.base import BasePipeline
from aifunckit._internal.orchestrator.pipelines.conditional import ConditionalPipeline
from aifunckit._internal.orchestrator.pipelines.for_each_loop import ForEachLoopPipeline
from aifunckit._internal.orchestrator.pipelines.loop import LoopPipeline
from aifunckit._internal.orchestrator.pipelines.node_solver import NodeSolver
from aifunckit._internal.orchestrator.pipelines.parallel import ParallelPipeline
from aifunckit._internal.orchestrator.pipelines.sequential import SequentialPipeline
from aifunckit._internal.orchestrator.pipelines.timeout import TimeoutPipeline

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
            try:
                if matcher(tasks):
                    logger.info("TopologyResolver selected '%s'", pipeline_cls.__name__)
                    return pipeline_cls
            except Exception as exc:
                logger.error(
                    "Topology matcher '%s' failed: %s",
                    getattr(matcher, "__name__", matcher),
                    exc,
                    exc_info=True,
                )
        logger.info("TopologyResolver defaulted to '%s'", cls._default.__name__)
        return cls._default


if not TopologyResolver._matchers:
    TopologyResolver.register(NodeSolver.has_parallel_nodes, ParallelPipeline)
    TopologyResolver.register(NodeSolver.has_conditional_nodes, ConditionalPipeline)
    TopologyResolver.register(NodeSolver.has_for_each_nodes, ForEachLoopPipeline)
    TopologyResolver.register(NodeSolver.has_loop_nodes, LoopPipeline)
    TopologyResolver.register(NodeSolver.has_timeout_nodes, TimeoutPipeline)
