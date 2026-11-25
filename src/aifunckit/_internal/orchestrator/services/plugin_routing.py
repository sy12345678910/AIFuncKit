from __future__ import annotations

import asyncio
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from aifunckit.exceptions.exceptions import PluginNotFoundError, RoutingModeDoesNotExist
from aifunckit._internal.log_config import logger
from aifunckit._internal.memory.vector import normalize_vector
from aifunckit._internal.schemas.pipeline_context import PipelineContext
from aifunckit._internal.orchestrator.general_methods import maybe_await, metadata_action_name

class PluginRoutingService:
    """Select the plugin metadata record that best matches a decomposed task."""

    MIN_VECTOR_SIMILARITY: float = 0.25

    def __init__(self, routing_model: str = "vector") -> None:
        self.routing_model = routing_model
        self._routes: dict[str, Callable[..., Any]] = {
            "blm": self.route_blm,
            "text": self.route_text,
            "vector": self.route_vector,
        }

    async def route(self, task: str, context: PipelineContext) -> dict[str, Any]:
        """Select the best matching plugin metadata entry for a particular task text."""
        cached_metadata = context.metadata.cached_metadata
        if not cached_metadata:
            raise PluginNotFoundError("Database with plugins is empty or not found")

        task_lower: str = task.lower()
        handler = self._routes.get(self.routing_model)
        if handler is None:
            raise RoutingModeDoesNotExist(
                f"Routing mode '{self.routing_model}' doesn't exist! "
                f"Available modes: {list(self._routes.keys())}"
            )

        result = handler(task=task_lower, context=context)
        return await maybe_await(result)

    def route_text(self, task: str, context: PipelineContext) -> dict[str, Any]:
        """Match tasks using simple keyword and action-name comparison."""
        cached_metadata = context.metadata.cached_metadata
        for meta in cached_metadata:
            action_name: str = metadata_action_name(meta).lower()
            if action_name and action_name in task:
                logger.info("Text routing matched action '%s'", action_name)
                return meta

            keywords: list[str] = meta.get("keywords") or []
            if any(str(kw).lower() in task for kw in keywords if kw):
                logger.info("Text routing matched keywords for '%s'", action_name)
                return meta

        logger.warning("Text routing found no matches for task: '%s'", task)
        raise PluginNotFoundError(f"No plugin matched text query: '{task}'")

    async def route_vector(self, task: str, context: PipelineContext) -> dict[str, Any]:
        """Match tasks by embedding similarity between the task and action descriptions."""
        cached_metadata = context.metadata.cached_metadata
        await self.ensure_description_embeddings(context)

        try:
            task_vector: NDArray[np.float32] = normalize_vector(
                await maybe_await(context.embedding._encode_text(task))
            )
        except Exception as exc:
            logger.error("Failed to encode task text for routing: %s", exc, exc_info=True)
            raise PluginNotFoundError(f"Failed to encode task '{task}': {exc}") from exc

        def _calculate_best_match() -> tuple[dict[str, Any] | None, int, float]:
            """Compute the best metadata match in a single pass over the cache."""
            best_meta: dict[str, Any] | None = None
            best_index: int = -1
            max_similarity: float = -1.0

            for index, meta in enumerate(cached_metadata):
                desc_vector: NDArray[np.float32] | None = meta.get("description_embedding")
                if desc_vector is None:
                    continue

                similarity: float = float(np.dot(task_vector, desc_vector))

                if similarity > max_similarity:
                    max_similarity = similarity
                    best_meta = meta
                    best_index = index

            return best_meta, best_index, max_similarity

        try:
            if len(cached_metadata) > 100:
                best_meta, best_index, max_similarity = await asyncio.to_thread(
                    _calculate_best_match
                )
            else:
                best_meta, best_index, max_similarity = _calculate_best_match()
        except Exception as exc:
            logger.error("Vector routing calculation failed: %s", exc, exc_info=True)
            raise PluginNotFoundError(
                f"Vector routing failed due to internal error: {exc}"
            ) from exc

        if best_meta is None:
            raise PluginNotFoundError("No valid description embeddings found in registered plugins")

        logger.info(
            "Vector routing match: index=%s, action='%s', numpy score=%.4f",
            best_index,
            metadata_action_name(best_meta),
            max_similarity,
        )

        if max_similarity < self.MIN_VECTOR_SIMILARITY:
            raise PluginNotFoundError(
                f"No suitable plugin found for task (best similarity={max_similarity:.3f} "
                f"< threshold={self.MIN_VECTOR_SIMILARITY}). "
                f"Register a plugin that handles: '{task}'"
            )

        return best_meta

    async def ensure_description_embeddings(self, context: PipelineContext) -> None:
        """Populate missing description embeddings for action metadata records."""
        for meta in context.metadata.cached_metadata:
            if "description_embedding" in meta:
                continue
            description: str = meta.get("description") or metadata_action_name(meta)
            try:
                vector: NDArray[np.float32] = await maybe_await(
                    context.embedding._encode_text(str(description))
                )
                meta["description_embedding"] = normalize_vector(vector)
            except Exception as exc:
                logger.warning(
                    "Could not embed description for '%s': %s",
                    metadata_action_name(meta),
                    exc,
                )

    async def route_blm(self, task: str, context: PipelineContext) -> dict[str, Any]:
        """Route using a model-specific router when the optional laya dependency exists."""
        cached_metadata = context.metadata.cached_metadata
        try:
            import laya
        except ImportError as exc:
            raise ImportError(
                "BLM routing requires 'laya'. Install it via: pip install aifunckit[blm-routing]"
            ) from exc

        try:
            agent = laya.Router(preload=True)
            options: dict[str, dict[str, Any]] = {}
            meta_by_name: dict[str, dict[str, Any]] = {}

            for m in cached_metadata:
                name = metadata_action_name(m)
                if not name:
                    continue
                options[name] = {
                    "description": m.get("description", ""),
                    "keywords": m.get("keywords", []),
                }
                meta_by_name[name] = m

            question: dict[str, Any] = {
                "method": {
                    "type": "choice",
                    "instruction": "Choose the most suitable method by description and keywords.",
                    "options": options,
                }
            }
            answer: Any = await asyncio.to_thread(agent.predict, task, question)
            selected_method_name: Any = (
                answer.get("answers", {}).get("method", {}).get("choice")
            )

            if selected_method_name in meta_by_name:
                logger.info("BLM routing selected action '%s'", selected_method_name)
                return meta_by_name[selected_method_name]

            raise PluginNotFoundError(
                f"Selected method '{selected_method_name}' not found in metadata"
            )
        except PluginNotFoundError:
            raise
        except Exception as exc:
            logger.error("BLM routing failed: %s", exc, exc_info=True)
            raise
