from __future__ import annotations

from typing import Any, Optional

from aifunckit._internal.log_config import logger
from aifunckit._internal.orchestrator.pipelines.node_solver import NodeSolver
from aifunckit._internal.schemas.pipeline_context import PipelineContext
from aifunckit.exceptions.pipeline_exceptions import InvalidPipelineStateException


class PipelineValidationManager:
    """Validate pipeline context, metadata, and task nodes before execution."""

    def __init__(self, node_solver: Optional[NodeSolver] = None) -> None:
        self.node_solver = node_solver or NodeSolver()

    def validate_state(self, context: PipelineContext) -> None:
        """Validate pipeline context and tasks before execution."""
        if context is None:
            logger.error("Pipeline validation failed: context is None")
            raise InvalidPipelineStateException("Pipeline context is empty or None.")

        if not context.metadata or not context.metadata.cached_metadata:
            logger.error("Pipeline validation failed: no registered plugins or actions in metadata")
            raise InvalidPipelineStateException(
                "No registered plugins or actions found in pipeline metadata."
            )

        if not context.decomposed_tasks:
            logger.error("Pipeline validation failed: decomposed_tasks is empty")
            raise InvalidPipelineStateException("No tasks provided for pipeline execution.")

        for task in context.decomposed_tasks:
            if isinstance(task, dict):
                if self.node_solver.is_conditional_node(task) and not task.get("condition_variable"):
                    logger.error("Conditional task missing 'condition_variable': %s", task)
                    raise InvalidPipelineStateException(
                        "Conditional task missing required 'condition_variable'."
                    )
                if self.node_solver.is_loop_node(task):
                    reps = task.get("repetitions") or task.get("iterations")
                    if reps is not None and int(reps) < 0:
                        logger.error("Invalid loop repetitions: %s", reps)
                        raise InvalidPipelineStateException(
                            f"Loop repetitions must be non-negative, got {reps}."
                        )
                if self.node_solver.is_timeout_node(task):
                    timeout = task.get("timeout") or task.get("timeout_seconds")
                    if timeout is not None and float(timeout) < 0:
                        logger.error("Invalid timeout value: %s", timeout)
                        raise InvalidPipelineStateException(
                            f"Timeout value must be non-negative, got {timeout}."
                        )
                if self.node_solver.is_for_each_node(task):
                    items = task.get("items") or task.get("for_each") or task.get("collection")
                    if items is None:
                        logger.error("ForEach task missing 'items': %s", task)
                        raise InvalidPipelineStateException(
                            "ForEach task missing required 'items' collection."
                        )

    def _pre_validation(
        self,
        tasks: Optional[Any] = None,
        context: Optional[PipelineContext] = None,
    ) -> None:
        """Compatibility wrapper for pipeline pre-validation."""
        if isinstance(tasks, PipelineContext):
            self.validate_state(tasks)
        elif context is not None:
            self.validate_state(context)
