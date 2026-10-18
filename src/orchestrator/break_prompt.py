# import json
# from typing import Any, Optional, Tuple
# import numpy as np
# from numpy.typing import NDArray
# from model_provide.model_router import get_data_generated_by_model
# from log_config import logger
# from src.schemas.ai_schemas import TaskList
# from src.schemas.orchestrator_schemas import LLMConfig, MetadataForLLM
# from src.schemas.memory_schemas import MemoryRecord
# from memory.rag import AIRAGMemoryManager
# from memory.embedding import EmbeddingService
# from orchestrator.general_methods import metadata_action_name

# def set_context_block(system_context: str) -> Tuple:
#     return (
#         "CURRENT GLOBAL ENVIRONMENT STATE:\n"
#         f"{json.dumps(system_context, ensure_ascii=False)}\n\n"
#     )

# async def break_prompt_to_tasks(
#     user_prompt: str,
#     llm_config: LLMConfig,
#     metadata_for_llm: MetadataForLLM,
#     memory: AIRAGMemoryManager,
#     embedding: EmbeddingService
# ) -> tuple[list[Any], NDArray[np.float32]]:
#     """Translate a user request into a list of actions or conditional blocks."""
#     prompt_emb: NDArray[np.float32] = await embedding._encode_text(user_prompt)
#     memory_prompt_block: str = await _prepare_memory_block(prompt_emb, memory)
#     capabilities_block: str = _prepare_capabilities_block(metadata_for_llm)

#     context_block: str = set_context_block(metadata_for_llm.context_block)

#     prompt: str = (
#         "You are an execution scheduler for a local automation core.\n"
#         "Your ONLY job is to take the user's prompt and split it into an ordered "
#         "list of technical tasks or conditional blocks in English.\n\n"
#         f"{context_block}"
#         f"{memory_prompt_block}"
#         f"{capabilities_block}\n"
#         "COMPILATION RULES:\n"
#         "1. Decompose the prompt ONLY into actions that can be handled by the "
#         "AVAILABLE SYSTEM CAPABILITIES.\n"
#         "2. If the user request is abstract (e.g., 'prepare for work', 'clean up'), "
#         "look at CURRENT GLOBAL ENVIRONMENT STATE and PAST DECOMPOSITION EXPERIENCE "
#         "to deduce logical steps, but adapt them to the current capabilities.\n"
#         "3. If the user requests an action not listed in the capabilities, strictly "
#         "IGNORE that action.\n"
#         "4. IF the user prompt contains conditional logic (e.g., 'if/else', "
#         "'otherwise', 'when... then...'), you MUST construct a conditional block "
#         "object using the JSON schema with fields 'condition_variable', 'if_true', "
#         "and 'if_false'. The 'condition_variable' MUST reflect a state variable from "
#         "the AVAILABLE SYSTEM CAPABILITIES in strict format 'plugin_name.variable_name' "
#         "(e.g., 'email.has_unread').\n"
#         "5. Return the result ONLY in accordance with the provided JSON schema."
#     )

#     try:
#         response: Any = await get_data_generated_by_model(
#         prompt=prompt,
#         provider=llm_config.provider,
#         user_prompt=user_prompt,
#         format=TaskList,
#         model=llm_config.model_name,
#         context_window=llm_config.context_window,
#     )
#     except Exception as exc:
#         logger.error("Task decomposition LLM call failed: %s", exc, exc_info=True)
#         return []

#     try:
#         if isinstance(response, str):
#             validated_data: TaskList = TaskList.model_validate_json(response)
#         else:
#             validated_data: Any = response
#             tasks: list[Any] = [
#                 t.model_dump() if hasattr(t, "model_dump") else t
#                 for t in validated_data.tasks
#             ]
#             logger.info("Prompt decomposed into %s task(s)", len(tasks))
#     except Exception as exc:
#         logger.error("Failed to parse decomposed tasks: %s", exc, exc_info=True)
#         tasks = []
#     return tasks

# async def _prepare_memory_block(prompt_emb: np.ndarray, memory: AIRAGMemoryManager) -> str:
#     """Fetch a relevant historical decomposition example as a context prompt."""
#     try:
#         memory_example: Optional[MemoryRecord] = await memory.search_relevant_example(
#             "decomposition", prompt_emb
#         )
#     except Exception as exc:
#         logger.error("RAG lookup for decomposition failed: %s", exc, exc_info=True)
#         return ""

#     if not memory_example:
#         return ""

#     past_prompt = memory_example.user_prompt
#     past_json = memory_example.extracted_json
#     try:
#         past_tasks_dict: Any = json.loads(past_json)
#         past_tasks: Any = (
#             past_tasks_dict.get("tasks", [])
#             if isinstance(past_tasks_dict, dict)
#             else past_json
#         )
#     except (json.JSONDecodeError, TypeError) as exc:
#         logger.warning("Could not parse stored decomposition JSON: %s", exc)
#         past_tasks = past_json

#     return (
#         "PAST DECOMPOSITION EXPERIENCE (Use this to understand abstract requests):\n"
#         f'- When user previously asked: "{past_prompt}"\n'
#         f"- It was successfully split into these tasks: {past_tasks}\n\n"
#     )

# def _prepare_capabilities_block(metadata: MetadataForLLM) -> str:
#     """Describe the currently available plugin capabilities for the planner model."""
#     capabilities: dict[str, list[str]] = {}
#     for meta in metadata.cached_metadata:
#         p_name: str = meta.get("plugin_name", "unknown")
#         if p_name not in capabilities:
#             capabilities[p_name] = []

#     keywords: Any = meta.get("keywords", [])
#     kw_str: str = ", ".join(keywords) if isinstance(keywords, list) else str(keywords)
#     action_name: str = metadata_action_name(meta)
#     capabilities[p_name].append(f"{action_name} (keywords: {kw_str})")

#     capabilities_block: str = (
#         "AVAILABLE SYSTEM CAPABILITIES (You CANNOT create tasks outside of these plugins):\n"
#     )
#     for plugin, methods in capabilities.items():
#         capabilities_block += f"- Plugin: {plugin} | Methods: {'; '.join(methods)}\n"

#     return capabilities_block