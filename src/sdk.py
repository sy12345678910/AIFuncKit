from __future__ import annotations

import inspect
from abc import ABC
from typing import Any, Callable, Optional

from log_config import logger


class BasePlugin(ABC):
    """Base class for all runtime plugins that expose action methods."""

    type: str
    plugin_name: str
    file: str

    def __init__(self) -> None:
        """Create an empty live-context dictionary for dynamic plugin state."""
        self.live_context: dict[str, Any] = {}

    def get_methods(self) -> list[dict[str, Any]]:
        """Collect all decorated action methods and their metadata for registration."""
        methods_list: list[dict[str, Any]] = []

        for attr_name in dir(self):
            if attr_name.startswith("_"):
                continue

            attr: Any = getattr(self, attr_name)

            if callable(attr) and hasattr(attr, "_desc"):
                sig: inspect.Signature = inspect.signature(attr)
                args_types: dict[str, str] = {}

                for param_name, param in sig.parameters.items():
                    if param_name in ["self", "args", "kwargs"]:
                        continue
                    p_type: Any = (
                        param.annotation
                        if param.annotation != inspect.Parameter.empty
                        else str
                    )
                    args_types[param_name] = getattr(p_type, "__name__", str(p_type))

                methods_list.append(
                    {
                        "name": attr,
                        "action_name": attr_name,
                        "description": attr._desc,
                        "args": args_types,
                    }
                )

        logger.info(
            "Discovered %s decorated methods for plugin '%s'",
            len(methods_list),
            getattr(self, "plugin_name", type(self).__name__),
        )
        return methods_list

    def verify_environment(self) -> tuple[str, ...]:
        """Return a tuple of required environment checks for the plugin."""
        return ()

    def get_live_context(self) -> dict[str, Any]:
        """Return the current plugin runtime context dictionary."""
        return self.live_context

    def update_contetx(self, key: str, value: Any) -> None:
        """Store a value in the plugin live context under the provided key."""
        self.live_context[key] = value


def decription(text: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Attach a natural-language description to a plugin action method."""

    def wrapper(func: Callable[..., Any]) -> Callable[..., Any]:
        """Store the description metadata on the decorated function."""
        func._desc = text
        return func

    return wrapper


describe = decription
