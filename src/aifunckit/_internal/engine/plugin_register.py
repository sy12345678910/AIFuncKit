from __future__ import annotations

import inspect
from typing import Any, Callable, Optional

from aifunckit._internal.log_config import logger
from aifunckit.exceptions.exceptions import PluginForDeleteNotFoundError
from aifunckit.sdk import BasePlugin


class PluginRegisterClass:
    """Manage plugin registration, action decorators, live context, and removal."""

    def __init__(
        self,
        cached_metadata: Optional[list[dict[str, Any]]] = None,
        actions: Optional[dict[str, Callable[..., Any]]] = None,
        plugins: Optional[dict[str, BasePlugin]] = None,
    ) -> None:
        self.cached_metadata: list[dict[str, Any]] = (
            cached_metadata if cached_metadata is not None else []
        )
        self.actions: dict[str, Callable[..., Any]] = (
            actions if actions is not None else {}
        )
        self.plugins: dict[str, BasePlugin] = (
            plugins if plugins is not None else {}
        )

    def register_plugin(self, obj: BasePlugin) -> None:
        """Register a plugin instance and add its decorated methods to the action registry."""
        plugin_name: str = getattr(
            obj, "plugin_name", type(obj).__name__.lower().replace("plugin", "")
        )
        obj.plugin_name = plugin_name

        if plugin_name in self.plugins:
            logger.warning("Plugin '%s' is already registered; skipping", plugin_name)
            return

        try:
            methods: list[dict[str, Any]] = obj.get_methods()
        except Exception as exc:
            logger.error(
                "Failed to collect methods from plugin '%s': %s",
                plugin_name,
                exc,
                exc_info=True,
            )
            raise

        for method in methods:
            method["plugin_name"] = plugin_name
            stored: Any = method["name"]
            action_name: Optional[str] = method.get("action_name")
            if not action_name:
                action_name = stored.__name__ if callable(stored) else str(stored)
                method["action_name"] = action_name
            self.actions[action_name] = stored

        self.cached_metadata.extend(methods)
        self.plugins[plugin_name] = obj
        logger.info(
            "Registered plugin '%s' with %s action(s)",
            plugin_name,
            len(methods),
        )

    def action(
        self,
        description: str,
        name: Optional[str] = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator that registers a function as an engine action with metadata."""

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            """Register the wrapped callable in the engine metadata registry."""
            action_name: str = name or func.__name__
            sig: inspect.Signature = inspect.signature(func)
            args_types: dict[str, str] = {}

            for param_name, param in sig.parameters.items():
                if param_name in ["args", "kwargs", "self"]:
                    continue
                p_type: Any = (
                    param.annotation if param.annotation != inspect.Parameter.empty else str
                )
                args_types[param_name] = getattr(p_type, "__name__", str(p_type))

            metadata: dict[str, Any] = {
                "name": func,
                "action_name": action_name,
                "description": description,
                "args": args_types,
            }
            self.cached_metadata.append(metadata)
            self.actions[action_name] = func
            logger.info("Registered action '%s'", action_name)
            return func

        return decorator

    def get_current_global_context(self) -> dict[str, Any]:
        """Collect the live state from all registered plugins into one context map."""
        global_context: dict[str, Any] = {}
        for name, plugin in self.plugins.items():
            if hasattr(plugin, "get_live_context"):
                try:
                    live_ctx: Any = plugin.get_live_context()
                except Exception as exc:
                    logger.error(
                        "Failed to read live context from plugin '%s': %s",
                        name,
                        exc,
                        exc_info=True,
                    )
                    continue
                if live_ctx:
                    global_context[name] = live_ctx
        return global_context

    def delete_plugin(self, name: str) -> None:
        """Remove a plugin and all of its registered actions from the registry."""
        plugin_name: str = (
            name.lower().replace("plugin", "") if name not in self.plugins else name
        )
        if plugin_name not in self.plugins:
            logger.warning("Plugin '%s' for deletion was not found", name)
            raise PluginForDeleteNotFoundError(
                f"Plugin '{name}' for deleting is not found!"
            )

        self._remove_plugin_from_metadata(plugin_name)
        del self.plugins[plugin_name]
        logger.info("Deleted plugin '%s'", plugin_name)

    def _remove_plugin_from_metadata(self, plugin_name: str) -> None:
        """Delete all metadata entries associated with a plugin name in-place."""
        self.cached_metadata[:] = [
            meta for meta in self.cached_metadata if meta.get("plugin_name") != plugin_name
        ]

        for action_name, func in list(self.actions.items()):
            if (
                hasattr(func, "__self__")
                and getattr(func.__self__, "plugin_name", None) == plugin_name
            ):
                del self.actions[action_name]
