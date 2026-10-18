from __future__ import annotations

from typing import Any

from log_config import logger
from schemas.pipeline_context import PipelineContext
from orchestrator.pipelines.base import BasePipeline
from orchestrator.pipelines.sequential import SequentialPipeline


class ConditionalPipeline(BasePipeline):
    """Route execution into if_true / if_false sequential branches from live state."""

    async def execute(self, context: PipelineContext) -> None:
        remaining: list[Any] = []
        for node in list(context.decomposed_tasks):
            if not self._is_conditional_node(node):
                remaining.append(node)
                continue

            if remaining:
                await self._run_sequential_branch(remaining, context)
                remaining = []

            await self._execute_conditional_node(node, context)

        if remaining:
            await self._run_sequential_branch(remaining, context)

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
        original_tasks = context.decomposed_tasks
        context.decomposed_tasks = list(tasks)
        await SequentialPipeline(
            routing=self.routing,
            extraction=self.extraction,
            execution=self.execution,
        ).execute(context)
        context.decomposed_tasks = original_tasks
