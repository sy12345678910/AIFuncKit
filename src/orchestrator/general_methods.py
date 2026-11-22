from __future__ import annotations

import inspect
from typing import Any


def metadata_action_name(meta: dict[str, Any]) -> str:
    """Resolve the action name from a metadata record while handling callable names."""
    action_name: Any = meta.get("action_name")
    if action_name:
        return str(action_name)
    stored: Any = meta.get("name")
    if callable(stored):
        return getattr(stored, "__name__", str(stored))
    return str(stored or "")


async def maybe_await(value: Any) -> Any:
    """Await coroutine-like values while leaving eager results unchanged."""
    if inspect.isawaitable(value):
        return await value
    return value
