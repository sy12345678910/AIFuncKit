from __future__ import annotations

from typing import Any

from aifunckit._internal.log_config import logger
from aifunckit._internal.schemas.pipeline_context import PipelineContext
from aifunckit._internal.orchestrator.pipelines.base import BasePipeline
from aifunckit._internal.orchestrator.pipelines.import_strategies.conditional import ConditionalExecutionStrategy
from aifunckit.exceptions.pipeline_exceptions import InvaildPipelineStateException

class ConditionalPipeline(BasePipeline):
    """Route execution into if_true / if_false sequential branches from live state."""

    pipeline_type = "conditional"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.strategy = ConditionalExecutionStrategy(self)

    async def execute(self, context: PipelineContext) -> None:
        await self.strategy.execute(context, list(context.decomposed_tasks))

    async def _execute_conditional_node(
        self,
        node: dict[str, Any],
        context: PipelineContext,
    ) -> None:
        """Evaluate a condition and execute the matching branch of tasks."""
        cond_var: str = node.get("condition_variable", "")
        condition_met: bool = False

        if "." in cond_var:
            plugin_part, key_part = cond_var.split(".", 1)
            if plugin_part not in context.environment_state or key_part not in context.environment_state[plugin_part]:
                raise InvaildPipelineStateException(f"condition rely on variable, which doesn't exist '{cond_var}'")
            condition_met = bool(context.environment_state[plugin_part][key_part])
        elif cond_var in context.environment_state:
            condition_met = bool(context.environment_state.get(cond_var, False))
        else: raise InvaildPipelineStateException(f'condition variable doesn\'t exist')

        logger.info(
            "Conditional branch '%s' evaluated to %s",
            cond_var,
            condition_met,
        )
        branch_key: str = "if_true" if condition_met else "if_false"
        next_tasks: list[Any] = node.get(branch_key, [])

        if next_tasks:
            await self._run_sequential_branch(next_tasks, context)

    async def _run_sequential_branch(
        self,
        tasks: list[Any],
        context: PipelineContext,
    ) -> None:
        from aifunckit._internal.orchestrator.pipelines.sequential import SequentialPipeline

        original_tasks = context.decomposed_tasks
        try:
            context.decomposed_tasks = list(tasks)
            await SequentialPipeline(
                routing=self.routing,
                extraction=self.extraction,
                execution=self.execution,
                circuit_breaker=self.circuit_breaker,
            ).execute(context)
        except Exception as exc:
            logger.error("Conditional branch execution failed: %s", exc, exc_info=True)
            raise
        finally:
            context.decomposed_tasks = original_tasks
