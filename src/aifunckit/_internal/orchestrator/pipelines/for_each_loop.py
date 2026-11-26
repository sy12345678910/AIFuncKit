from __future__ import annotations

from typing import Any

from aifunckit._internal.log_config import logger
from aifunckit._internal.schemas.pipeline_context import PipelineContext
from aifunckit._internal.orchestrator.pipelines.base import BasePipeline
from aifunckit._internal.orchestrator.pipelines.import_strategies.for_each_loop import ForEachLoopExecutionStrategy


class ForEachLoopPipeline(BasePipeline):
    """Execute one or more child tasks for each item in a collection or list."""

    pipeline_type = "for_each_loop"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.strategy = ForEachLoopExecutionStrategy(self)

    async def execute(self, context: PipelineContext) -> None:
        await self.run_nodes(list(context.decomposed_tasks), context)

    async def run_nodes(self, nodes: list[Any], context: PipelineContext) -> None:
        await self.strategy.execute(context, nodes)

    async def execute_for_node(self, node: Any, context: PipelineContext) -> None:
        from aifunckit._internal.orchestrator.pipelines.sequential import SequentialPipeline

        node_dict = node if isinstance(node, dict) else {"task": str(node), "items": []}
        raw_items = (
            node_dict.get("items")
            or node_dict.get("for_each")
            or node_dict.get("collection")
            or []
        )
        item_var = node_dict.get("item_variable") or "item"

        # Resolve items if they reference a variable in environment_state
        resolved_items: list[Any] = []
        if isinstance(raw_items, str):
            if "." in raw_items:
                plugin_part, key_part = raw_items.split(".", 1)
                plugin_ctx: dict[str, Any] = context.environment_state.get(plugin_part, {})
                resolved_items = plugin_ctx.get(key_part, [])
            elif raw_items in context.environment_state:
                resolved_items = context.environment_state.get(raw_items, [])
            elif "," in raw_items:
                resolved_items = [i.strip() for i in raw_items.split(",") if i.strip()]
            else:
                resolved_items = [raw_items]
        elif isinstance(raw_items, (list, tuple)):
            resolved_items = list(raw_items)
        else:
            resolved_items = [raw_items]

        # Extract child tasks to execute
        body = node_dict.get("body") or node_dict.get("tasks")
        if not body:
            single_task = node_dict.get("task")
            body = [single_task] if single_task else []

        original_tasks = context.decomposed_tasks
        original_item_val = context.environment_state.get(item_var)

        try:
            for item in resolved_items:
                logger.debug("ForEach iteration for %s=%r", item_var, item)
                # Expose the current item in environment state
                context.environment_state[item_var] = item

                # Substitute '{item}' placeholder in task descriptions if present
                iter_tasks: list[Any] = []
                for t in body:
                    if isinstance(t, str):
                        replaced = t.replace("{item}", str(item)).replace(f"{{{item_var}}}", str(item))
                        iter_tasks.append(replaced)
                    elif isinstance(t, dict):
                        replaced_dict = dict(t)
                        if isinstance(replaced_dict.get("task"), str):
                            replaced_dict["task"] = (
                                replaced_dict["task"]
                                .replace("{item}", str(item))
                                .replace(f"{{{item_var}}}", str(item))
                            )
                        iter_tasks.append(replaced_dict)
                    else:
                        iter_tasks.append(t)

                context.decomposed_tasks = iter_tasks
                await SequentialPipeline(
                    routing=self.routing,
                    extraction=self.extraction,
                    execution=self.execution,
                    circuit_breaker=self.circuit_breaker,
                ).execute(context)
        except Exception as exc:
            logger.error("ForEachLoopPipeline failed: %s", exc, exc_info=True)
            raise
        finally:
            # Restore original state
            if original_item_val is not None:
                context.environment_state[item_var] = original_item_val
            else:
                context.environment_state.pop(item_var, None)

            context.decomposed_tasks = original_tasks
