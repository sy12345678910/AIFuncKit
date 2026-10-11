from __future__ import annotations
import inspect
from pathlib import Path
from typing import Any, Callable, Optional, List, Dict
from exceptions import PluginForDeleteNotFoundError
from log_config import configure_logging, logger
from orchestrator import AIOrchestrator
from sdk import BasePlugin


class AIEngine:
    def __init__(
        self,
        ai_model: str = "gemma4:e4b",
        context_window: int = 4096,
        db_path: Path | str = Path("aie_runtine"),
        provider: str = "ollama",
        emb_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
        routing_model: str = "vector",
        verbose: bool = True,
        log_to_file: Optional[str] = "swift_core_ai.log",
    ) -> None:
        configure_logging(verbose=verbose, log_to_file=log_to_file)
        self.verbose: bool = verbose
        self.log_to_file: Optional[str] = log_to_file
        self.embidding_model: str = emb_model
        self.cached_metadata: list[dict[str, Any]] = []
        self.actions: dict[str, Callable[..., Any]] = {}
        self.plugins: dict[str, BasePlugin] = {}

        self.orchestrator = AIOrchestrator(
            cached_metadata=self.cached_metadata,
            provider=provider,
            model_name=ai_model,
            context_window=context_window,
            db_path=str(db_path),
            emb_model=emb_model,
            routing_model=routing_model,
        )
        logger.info(
            "AIEngine created (provider=%s, model=%s, routing=%s)",
            provider,
            ai_model,
            routing_model,
        )

    async def initialize(self) -> None:
        try:
            if hasattr(self.orchestrator, "ensure_embedding_model"):
                await self.orchestrator.ensure_embedding_model()
            if hasattr(self.orchestrator.runtime_memory, "init_and_load_cache"):
                await self.orchestrator.runtime_memory.init_and_load_cache()
            logger.info("AIEngine runtime initialized")
        except Exception as exc:
            logger.error("Failed to initialize AIEngine runtime: %s", exc, exc_info=True)
            raise

    def register_plugin(self, obj: BasePlugin) -> None:
        plugin_name = getattr(
            obj, "plugin_name", type(obj).__name__.lower().replace("plugin", "")
        )
        obj.plugin_name = plugin_name

        if plugin_name in self.plugins:
            logger.warning("Plugin '%s' is already registered; skipping", plugin_name)
            return

        try:
            methods = obj.get_methods()
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
            stored = method["name"]
            action_name = method.get("action_name")
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
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            action_name = name or func.__name__

            sig = inspect.signature(func)
            args_types: dict[str, str] = {}
            for param_name, param in sig.parameters.items():
                if param_name in ["args", "kwargs", "self"]:
                    continue
                p_type = (
                    param.annotation
                    if param.annotation != inspect.Parameter.empty
                    else str
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
        global_context: dict[str, Any] = {}
        for name, plugin in self.plugins.items():
            if hasattr(plugin, "get_live_context"):
                try:
                    live_ctx = plugin.get_live_context()
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

    async def run_engine(self, user_prompt: str) -> List[Dict[str, Any]]:
        logger.info("Engine run started for prompt: %s", user_prompt)
        current_context = self.get_current_global_context()

        try:
            data_for_execute = await self.orchestrator.run_pipeline(
                user_prompt, current_context
            )
        except Exception as exc:
            logger.error("Orchestrator pipeline failed completely: %s", exc, exc_info=True)
            return []

        results: List[Dict[str, Any]] = []

        for data in data_for_execute:
            target_func = data.get("method")
            args = data.get("args", {})
            action_name = data.get("action_name")

            if isinstance(target_func, str):
                target_func = self.actions.get(target_func)

            if not target_func or not callable(target_func):
                logger.error(
                    "Action execution aborted: '%s' is not a registered callable function in memory", 
                    action_name
                )
                continue

            try:
                if inspect.iscoroutinefunction(target_func):
                    res = await target_func(**args)
                else:
                    res = target_func(**args)
                    
                results.append({"action": action_name, "result": res})
                logger.info("Action '%s' completed successfully", action_name)
            except Exception as exc:
                logger.error(
                    "User plugin action '%s' generated an unhandled exception: %s",
                    action_name,
                    exc,
                    exc_info=True,
                )
                continue

        logger.info("Engine run finished with %s result(s)", len(results))
        return results

    async def delete_engine(self) -> None:
        try:
            if hasattr(self.orchestrator.runtime_memory, "vector_list_cache"):
                for layer in self.orchestrator.runtime_memory.vector_list_cache:
                    self.orchestrator.runtime_memory.vector_list_cache[layer] = []

            if hasattr(self.orchestrator.runtime_memory, "close"):
                await self.orchestrator.runtime_memory.close()

            if hasattr(self.orchestrator.client, "close"):
                if inspect.iscoroutinefunction(self.orchestrator.client.close):
                    await self.orchestrator.client.close()
                else:
                    self.orchestrator.client.close()

            self.cached_metadata.clear()
            self.actions.clear()
            self.plugins.clear()
            logger.info("AIEngine resources released")
        except Exception as exc:
            logger.error("Failed while shutting down AIEngine: %s", exc, exc_info=True)
            raise

    def delete_plugin(self, name: str) -> None:
        plugin_name = name.lower().replace("plugin", "") if name not in self.plugins else name
        if plugin_name not in self.plugins:
            raise PluginForDeleteNotFoundError(
                f"Plugin '{name}' for deleting is not found!"
            )

        self._remove_plugin_from_metadata(plugin_name)
        del self.plugins[plugin_name]
        logger.info("Deleted plugin '%s'", plugin_name)

    def _remove_plugin_from_metadata(self, plugin_name: str) -> None:
        self.cached_metadata = [
            m for m in self.cached_metadata if m.get("plugin_name") != plugin_name
        ]

        for action_name, func in list(self.actions.items()):
            if hasattr(func, "__self__") and getattr(func.__self__, "plugin_name", None) == plugin_name:
                del self.actions[action_name]
