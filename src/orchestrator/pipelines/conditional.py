from __future__ import annotations

from typing import Any

from log_config import logger
from schemas.pipeline_context import PipelineContext
from orchestrator.pipelines.base import BasePipeline
from orchestrator.pipelines.strategies.conditional import ConditionalExecutionStrategy


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
            plugin_ctx: dict[str, Any] = context.environment_state.get(plugin_part, {})
            condition_met = bool(plugin_ctx.get(key_part, False))

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
        from orchestrator.pipelines.sequential import SequentialPipeline

        original_tasks = context.decomposed_tasks
        context.decomposed_tasks = list(tasks)
        await SequentialPipeline(
            routing=self.routing,
            extraction=self.extraction,
            execution=self.execution,
        ).execute(context)
        context.decomposed_tasks = original_tasks
