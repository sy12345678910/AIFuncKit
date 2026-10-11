from __future__ import annotations

import inspect
from abc import ABC
from typing import Any, Callable, Optional

from log_config import logger


class BasePlugin(ABC):
    type: str
    plugin_name: str
    file: str

    def __init__(self) -> None:
        self.live_context: dict[str, Any] = {}

    def get_methods(self) -> list[dict[str, Any]]:
        methods_list: list[dict[str, Any]] = []

        for attr_name in dir(self):
            if attr_name.startswith("_"):
                continue

            attr = getattr(self, attr_name)

            if callable(attr) and hasattr(attr, "_desc"):
                sig = inspect.signature(attr)
                args_types: dict[str, str] = {}

                for param_name, param in sig.parameters.items():
                    if param_name in ["self", "args", "kwargs"]:
                        continue
                    p_type = (
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
        return ()

    def get_live_context(self) -> dict[str, Any]:
        return self.live_context

    def update_contetx(self, key: str, value: Any) -> None:
        self.live_context[key] = value


def decription(text: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def wrapper(func: Callable[..., Any]) -> Callable[..., Any]:
        func._desc = text  
        return func

    return wrapper


describe = decription
