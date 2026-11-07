import inspect
from typing import Any
from log_config import logger

async def run_function(task_data: dict[str, Any]) -> Any:
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
