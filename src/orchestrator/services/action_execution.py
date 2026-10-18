from __future__ import annotations

import inspect
from typing import Any

from log_config import logger


class ActionExecutionService:
    """Prepare and invoke dynamically selected plugin callables."""

    def build_payload(
        self,
        plugin_meta: dict[str, Any],
        needable_args: dict[str, Any],
    ) -> dict[str, Any]:
        """Assemble the executable payload produced by routing and argument extraction."""
        plugin_name: str = plugin_meta.get("plugin_name", "")
        func_data: dict[str, Any] = {
            "method": plugin_meta["name"],
            "action_name": plugin_meta.get("action_name"),
            "plugin_name": plugin_name,
            "args": needable_args,
        }
        logger.info("Prepared execution payload for action '%s'", func_data["action_name"])
        return func_data

    async def execute(self, task_data: dict[str, Any]) -> Any:
        """Invoke a callable method payload prepared by the planner and argument extraction."""
        target_func: Any = task_data.get("method")
        args: dict[str, Any] = task_data.get("args") or {}
        if not callable(target_func):
            raise TypeError("import_func received a non-callable method payload")
        try:
            if inspect.iscoroutinefunction(target_func):
                return await target_func(**args)
            return target_func(**args)
        except Exception as exc:
            logger.error(
                "Callable import_func execution failed: %s",
                exc,
                exc_info=True,
            )
            raise
