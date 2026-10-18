import asyncio
from typing import Any
import numpy as np
from numpy.typing import NDArray

from exceptions.exceptions import RoutingModeDoesNotExist, PluginNotFoundError
from log_config import logger
from orchestrator.general_methods import metadata_action_name
from memory.vector import normalize_vector
from memory.embedding import EmbeddingService

async def referal_ai_to_correct_plugin(task: str, cached_metadata: list, routing_model: str, emb_service: EmbeddingService) -> dict[str, Any]:
    """Select the best matching plugin metadata entry for a particular task text."""
    if not cached_metadata:
        raise PluginNotFoundError("Database with plugins is not found")
    task_lower: str = task.lower()

    match routing_model:
        case "blm":
            return await __blm_routing(task=task_lower, cached_metadata=cached_metadata)
        case "text":
            return __text_routing(task=task_lower, cached_metadata=cached_metadata)
        case "vector":
            return await __vector_routing(task=task_lower, cached_metadata=cached_metadata, emb_service=emb_service)
        case _:
            raise RoutingModeDoesNotExist("This model of routing doesn't exist!")

def __text_routing(task: str, cached_metadata: list) -> dict[str, Any]:
    """Match tasks using simple keyword and action-name comparison."""
    for meta in cached_metadata:
        action_name: str = metadata_action_name(meta).lower()
        if action_name and action_name in task:
            logger.info("Text routing matched action '%s'", action_name)
            return meta

        keywords: Any = meta.get("keywords", [])
        if any(kw in task for kw in keywords):
            logger.info("Text routing matched keywords for '%s'", action_name)
            return meta
    logger.warning("Text routing fallback to first registered action")
    return cached_metadata[0]

async def __vector_routing(task: str, cached_metadata: list, emb_service: EmbeddingService) -> dict[str, Any]:
    """Match tasks by embedding similarity between the task and action descriptions."""
    task_vector: NDArray[np.float32] = normalize_vector(await emb_service._encode_text(task))
    await ensure_description_embeddings(cached_metadata=cached_metadata, emb_service=emb_service)

    def _calculate_best_match() -> tuple[dict[str, Any], int, float]:
        """Compute the best metadata match in a single pass over the cache."""
        best_meta: dict[str, Any] = cached_metadata[0]
        best_index: int = 0
        max_similarity: float = -1.0

        for index, meta in enumerate(cached_metadata):
            if "description_embedding" not in meta:
                continue

            desc_vector: NDArray[np.float32] = meta["description_embedding"]
            similarity: float = float(np.dot(task_vector, desc_vector))

            if similarity > max_similarity:
                max_similarity = similarity
                best_meta = meta
                best_index = index
        return best_meta, best_index, max_similarity

    try:
        best_meta, best_index, max_similarity = _calculate_best_match()
    except Exception as exc:
        logger.error("Vector routing failed: %s", exc, exc_info=True)
        return cached_metadata[0]

    logger.info(
        "Vector routing match: index=%s, action='%s', numpy score=%.4f",
        best_index,
        metadata_action_name(best_meta),
        max_similarity,
    )
    return best_meta


async def ensure_description_embeddings(cached_metadata: list, emb_service: EmbeddingService) -> None:
    """Populate missing description embeddings for action metadata records."""
    for meta in cached_metadata:
        if "description_embedding" in meta:
            continue
        description: str = meta.get("description") or metadata_action_name(meta)
        try:
            vector: NDArray[np.float32] = await emb_service._encode_text(str(description))
            meta["description_embedding"] = normalize_vector(vector)
        except Exception as exc:
            logger.warning(
                "Could not embed description for '%s': %s",
                metadata_action_name(meta),
                exc,
            )
            
async def __blm_routing(task: str, cached_metadata) -> dict[str, Any]:
    """Route using a model-specific router when the optional laya dependency exists."""
    try:
        import laya
        agent = laya.Router(preload=True)
        options: dict[str, dict[str, Any]] = {
            m["action_name"]: {
                "description": m.get("description", ""),
                "keywords": m.get("keywords", []),
            }
            for m in cached_metadata
        }
        question: dict[str, Any] = {
            "method": {
                "type": "choice",
                "instruction": "Choose the most suitable method by description and keywords.",
                "options": options,
            }
        }
        answer: Any = await asyncio.to_thread(agent.predict, task, question)
        selected_method_name: Any = answer.get("answers", {}).get("method", {}).get("choice")

        for meta in cached_metadata:
            if meta["action_name"] == selected_method_name:
                logger.info("BLM routing selected action '%s'", selected_method_name)
                return meta
        raise PluginNotFoundError(
            f"Selected method '{selected_method_name}' not found in metadata"
        )
    except Exception as exc:
        logger.error("BLM routing failed: %s", exc, exc_info=True)
        raise
