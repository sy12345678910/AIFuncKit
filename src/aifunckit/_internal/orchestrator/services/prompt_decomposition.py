from __future__ import annotations

import json
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from aifunckit._internal.log_config import logger
from aifunckit._internal.model_provide.model_router import get_data_generated_by_model
from aifunckit._internal.schemas.pipeline_context import PipelineContext
from aifunckit._internal.orchestrator.general_methods import maybe_await, metadata_action_name
from aifunckit._internal.schemas.ai_schemas import TaskList
from aifunckit._internal.schemas.memory_schemas import MemoryRecord
from aifunckit._internal.schemas.orchestrator_schemas import MetadataForLLM


class PromptDecompositionService:
    """Split a user prompt into ordered technical tasks using the planner model."""

    async def decompose(self, context: PipelineContext) -> NDArray[np.float32]:
        """Translate a user request into tasks and store them on the pipeline context."""
        prompt_emb: NDArray[np.float32] = await maybe_await(
            context.embedding._encode_text(context.user_prompt)
        )
        memory_prompt_block: str = await self._prepare_memory_block(prompt_emb, context)
        capabilities_block: str = self._prepare_capabilities_block(context.metadata)
        context_block: str = self._set_context_block(context.environment_state)

        prompt: str = (
            "You are an intelligent task planner and workflow orchestrator. "
            "Your ONLY job is to break down the user's request into an ordered execution plan.\n"
            f"{context_block}{memory_prompt_block}{capabilities_block}"
            "PLANNING RULES: 1. Strict Capability Grounding; 2. Language Alignment (match capability language); "
            "3. Abstract & Implicit Intents; 4. Conditional Logic; 5. Iteration Logic; 6. Return ONLY valid JSON."
        )


        try:
            response: Any = await get_data_generated_by_model(
                prompt=prompt,
                provider=context.llm_config.provider,
                user_prompt=context.user_prompt,
                format=TaskList,
                model=context.llm_config.model_name,
                context_window=context.llm_config.context_window,
            )
        except Exception as exc:
            logger.error("Task decomposition LLM call failed: %s", exc, exc_info=True)
            context.decomposed_tasks = []
            return prompt_emb

        try:
            if isinstance(response, str):
                validated_data: TaskList = TaskList.model_validate_json(response)
            else:
                validated_data = response
            tasks: list[Any] = [
                t.model_dump() if hasattr(t, "model_dump") else t
                for t in validated_data.tasks
            ]
            logger.info("Prompt decomposed into %s task(s)", len(tasks))
        except Exception as exc:
            logger.error("Failed to parse decomposed tasks: %s", exc, exc_info=True)
            tasks = []

        context.decomposed_tasks = tasks
        return prompt_emb

    def _set_context_block(self, system_context: Any) -> str:
        return (
            "CURRENT GLOBAL ENVIRONMENT STATE:\n"
            f"{json.dumps(system_context, ensure_ascii=False)}\n\n"
        )

    async def _prepare_memory_block(
        self,
        prompt_emb: np.ndarray,
        context: PipelineContext,
    ) -> str:
        """Fetch a relevant historical decomposition example as a context prompt."""
        try:
            memory_example: Optional[MemoryRecord] = await context.memory.search_relevant_example(
                "decomposition", prompt_emb
            )
        except Exception as exc:
            logger.error("RAG lookup for decomposition failed: %s", exc, exc_info=True)
            return ""

        if not memory_example:
            return ""

        past_prompt = memory_example.user_prompt
        past_json = memory_example.extracted_json
        try:
            past_tasks_dict: Any = json.loads(past_json)
            past_tasks: Any = (
                past_tasks_dict.get("tasks", [])
                if isinstance(past_tasks_dict, dict)
                else past_json
            )
        except (json.JSONDecodeError, TypeError) as exc:
            logger.warning("Could not parse stored decomposition JSON: %s", exc)
            past_tasks = past_json

        return (
            "PAST DECOMPOSITION EXPERIENCE (Use this to understand abstract requests):\n"
            f'- When user previously asked: "{past_prompt}"\n'
            f"- It was successfully split into these tasks: {past_tasks}\n\n"
        )

    def _prepare_capabilities_block(self, metadata: MetadataForLLM) -> str:
        """Describe the currently available plugin capabilities for the planner model."""
        capabilities: dict[str, list[str]] = {}
        for meta in metadata.cached_metadata:
            p_name: str = meta.get("plugin_name", "unknown")
            if p_name not in capabilities:
                capabilities[p_name] = []

            keywords: Any = meta.get("keywords", [])
            kw_str: str = ", ".join(keywords) if isinstance(keywords, list) else str(keywords)
            action_name: str = metadata_action_name(meta)
            capabilities[p_name].append(f"{action_name} (keywords: {kw_str})")

        capabilities_block: str = (
            "AVAILABLE SYSTEM CAPABILITIES (You CANNOT create tasks outside of these plugins):\n"
        )
        for plugin, methods in capabilities.items():
            capabilities_block += f"- Plugin: {plugin} | Methods: {'; '.join(methods)}\n"

        return capabilities_block
