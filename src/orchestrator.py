from __future__ import annotations

import asyncio
import inspect
import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import ollama
from numpy.typing import NDArray

from src.schemas.ai_schemas import TaskList
from src.exceptions.exceptions import PluginNotFoundError, RoutingModeDoesNotExist
from log_config import logger
from src.memory.rag import AIRAGMemory
from model_provide.model_router import get_data_generated_by_model

try:
    from sentence_transformers import SentenceTransformer
    import laya
except ImportError:
    logger.warning("You didn't add dependencies blm")

PipelineNode = tuple[str, dict[str, Any]]

# 1
def _as_float32_vector(vec: Any) -> NDArray[np.float32]:
    """Normalize vector-like objects into a flat float32 NumPy array."""
    if hasattr(vec, "detach"):
        vec = vec.detach().cpu().numpy()
    return np.asarray(vec, dtype=np.float32).reshape(-1)


def _metadata_action_name(meta: dict[str, Any]) -> str:
    """Resolve the action name from a metadata record while handling callable names."""
    action_name: Any = meta.get("action_name")
    if action_name:
        return str(action_name)
    stored: Any = meta.get("name")
    if callable(stored):
        return stored.__name__
    return str(stored or "")

# 2
class AIOrchestrator:
    """Plan prompts, choose the right plugin, and execute actions against system context."""

    def __init__(
        self,
        cached_metadata: list[dict[str, Any]],
        provider: str = "ollama",
        model_name: str = "gemma4:e4b",
        context_window: int = 3072,
        db_path: str = "aie_runtime.db",
        emb_model: Any = "",
        routing_model: str = "vector",
    ) -> None:
        """Initialize the orchestrator, memory store, provider client, and route metadata."""
        self.emb_model: Any = emb_model
        self.agent: Any = None
        self.routing_model: str = routing_model
        self.model_name: str = model_name
        self.config: dict[str, Any] = {
            "num_ctx": context_window,
            "temperature": 0.0,
            "num_predict": 256,
        }
        self.provider: str = provider
        self.runtime_memory: AIRAGMemory = AIRAGMemory(db_path)
        self.client: ollama.AsyncClient = ollama.AsyncClient()
        self._modules_cache: dict[str, Any] = {}
        self.cached_metadata: list[dict[str, Any]] = cached_metadata

    async def ensure_embedding_model(self) -> None:
        """Ensure the configured embedding model is loaded for vector encoding."""
        if self.emb_model is None:
            raise RuntimeError("Embedding model is not configured")

        if not isinstance(self.emb_model, str) and hasattr(self.emb_model, "encode"):
            return

        model_name: str = str(self.emb_model)
        try:
            self.emb_model = await asyncio.to_thread(SentenceTransformer, model_name)
            logger.info("Loaded embedding model '%s' via sentence-transformers", model_name)
        except Exception as exc:
            logger.error("Failed to load embedding model '%s': %s", model_name, exc, exc_info=True)
            raise

    async def _encode_text(self, text: str) -> NDArray[np.float32]:
        """Convert a text string into a normalized embedding vector."""
        await self.ensure_embedding_model()

        if not isinstance(self.emb_model, str) and hasattr(self.emb_model, "encode"):
            try:
                raw: Any = await asyncio.to_thread(self.emb_model.encode, text)
                return _as_float32_vector(raw)
            except Exception as exc:
                logger.error("Local Python embedding encode failed: %s", exc, exc_info=True)
                raise

        try:
            response: Any = await self.client.embeddings(
                model=str(self.emb_model),
                prompt=text,
            )
            return _as_float32_vector(response["embedding"])
        except Exception as exc:
            if "not found" in str(exc).lower():
                logger.warning(
                    "Ollama model '%s' not found. Trying main core model '%s'...",
                    self.emb_model,
                    self.model_name,
                )
                try:
                    response = await self.client.embeddings(model=self.model_name, prompt=text)
                    return _as_float32_vector(response["embedding"])
                except Exception as inner_exc:
                    logger.error("Native Ollama fallback embedding failed: %s", inner_exc, exc_info=True)
                    raise
            logger.error("Native Ollama embedding generation failed: %s", exc, exc_info=True)
            raise

    async def break_prompt_to_tasks(
        self,
        user_prompt: str,
        system_context: dict[str, Any],
    ) -> tuple[list[Any], NDArray[np.float32]]:
        """Translate a user request into a list of actions or conditional blocks."""
        prompt_emb: NDArray[np.float32] = await self._encode_text(user_prompt)
        memory_prompt_block: str = await self._prepare_memory_block(prompt_emb)
        capabilities_block: str = self._prepare_capabilities_block()

        context_block: str = (
            "CURRENT GLOBAL ENVIRONMENT STATE:\n"
            f"{json.dumps(system_context, ensure_ascii=False)}\n\n"
        )

        prompt: str = (
            "You are an execution scheduler for a local automation core.\n"
            "Your ONLY job is to take the user's prompt and split it into an ordered "
            "list of technical tasks or conditional blocks in English.\n\n"
            f"{context_block}"
            f"{memory_prompt_block}"
            f"{capabilities_block}\n"
            "COMPILATION RULES:\n"
            "1. Decompose the prompt ONLY into actions that can be handled by the "
            "AVAILABLE SYSTEM CAPABILITIES.\n"
            "2. If the user request is abstract (e.g., 'prepare for work', 'clean up'), "
            "look at CURRENT GLOBAL ENVIRONMENT STATE and PAST DECOMPOSITION EXPERIENCE "
            "to deduce logical steps, but adapt them to the current capabilities.\n"
            "3. If the user requests an action not listed in the capabilities, strictly "
            "IGNORE that action.\n"
            "4. IF the user prompt contains conditional logic (e.g., 'if/else', "
            "'otherwise', 'when... then...'), you MUST construct a conditional block "
            "object using the JSON schema with fields 'condition_variable', 'if_true', "
            "and 'if_false'. The 'condition_variable' MUST reflect a state variable from "
            "the AVAILABLE SYSTEM CAPABILITIES in strict format 'plugin_name.variable_name' "
            "(e.g., 'email.has_unread').\n"
            "5. Return the result ONLY in accordance with the provided JSON schema."
        )

        try:
            response: Any = await get_data_generated_by_model(
                prompt=prompt,
                provider=self.provider,
                user_prompt=user_prompt,
                format=TaskList,
                model=self.model_name,
                context_window=self.config["num_ctx"],
            )
        except Exception as exc:
            logger.error("Task decomposition LLM call failed: %s", exc, exc_info=True)
            return [], prompt_emb

        try:
            if isinstance(response, str):
                validated_data: TaskList = TaskList.model_validate_json(response)
            else:
                validated_data: Any = response
            tasks: list[Any] = [
                t.model_dump() if hasattr(t, "model_dump") else t
                for t in validated_data.tasks
            ]
            logger.info("Prompt decomposed into %s task(s)", len(tasks))
        except Exception as exc:
            logger.error("Failed to parse decomposed tasks: %s", exc, exc_info=True)
            tasks = []
        return tasks, prompt_emb

    async def _prepare_memory_block(self, prompt_emb: np.ndarray) -> str:
        """Fetch a relevant historical decomposition example as a context prompt."""
        try:
            memory_example: Optional[tuple[str, str, str]] = await self.runtime_memory.search_relevant_example(
                "decomposition", prompt_emb
            )
        except Exception as exc:
            logger.error("RAG lookup for decomposition failed: %s", exc, exc_info=True)
            return ""

        if not memory_example:
            return ""

        past_prompt, past_context, past_json = memory_example
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

    def _prepare_capabilities_block(self) -> str:
        """Describe the currently available plugin capabilities for the planner model."""
        capabilities: dict[str, list[str]] = {}
        for meta in self.cached_metadata:
            p_name: str = meta.get("plugin_name", "unknown")
            if p_name not in capabilities:
                capabilities[p_name] = []

            keywords: Any = meta.get("keywords", [])
            kw_str: str = ", ".join(keywords) if isinstance(keywords, list) else str(keywords)
            action_name: str = _metadata_action_name(meta)
            capabilities[p_name].append(f"{action_name} (keywords: {kw_str})")

        capabilities_block: str = (
            "AVAILABLE SYSTEM CAPABILITIES (You CANNOT create tasks outside of these plugins):\n"
        )
        for plugin, methods in capabilities.items():
            capabilities_block += f"- Plugin: {plugin} | Methods: {'; '.join(methods)}\n"

        return capabilities_block

    async def referal_ai_to_correct_plugin(self, task: str) -> dict[str, Any]:
        """Select the best matching plugin metadata entry for a particular task text."""
        if not self.cached_metadata:
            raise PluginNotFoundError("Database with plugins is not found")
        task_lower: str = task.lower()

        match self.routing_model:
            case "blm":
                return await self.__blm_routing(task=task_lower)
            case "text":
                return self.__text_routing(task=task_lower)
            case "vector":
                return await self.__vector_routing(task=task_lower)
            case _:
                raise RoutingModeDoesNotExist("This model of routing doesn't exist!")

    def __text_routing(self, task: str) -> dict[str, Any]:
        """Match tasks using simple keyword and action-name comparison."""
        for meta in self.cached_metadata:
            action_name: str = _metadata_action_name(meta).lower()
            if action_name and action_name in task:
                logger.info("Text routing matched action '%s'", action_name)
                return meta

            keywords: Any = meta.get("keywords", [])
            if any(kw in task for kw in keywords):
                logger.info("Text routing matched keywords for '%s'", action_name)
                return meta
        logger.warning("Text routing fallback to first registered action")
        return self.cached_metadata[0]

    async def __vector_routing(self, task: str) -> dict[str, Any]:
        """Match tasks by embedding similarity between the task and action descriptions."""
        task_vector: NDArray[np.float32] = self._normalize_vector(await self._encode_text(task))
        await self._ensure_description_embeddings()

        def _calculate_best_match() -> tuple[dict[str, Any], int, float]:
            """Compute the best metadata match in a single pass over the cache."""
            best_meta: dict[str, Any] = self.cached_metadata[0]
            best_index: int = 0
            max_similarity: float = -1.0

            for index, meta in enumerate(self.cached_metadata):
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
            if len(self.cached_metadata) > 100:
                best_meta, best_index, max_similarity = await asyncio.to_thread(
                    _calculate_best_match
                )
            else:
                best_meta, best_index, max_similarity = _calculate_best_match()
        except Exception as exc:
            logger.error("Vector routing failed: %s", exc, exc_info=True)
            return self.cached_metadata[0]

        logger.info(
            "Vector routing match: index=%s, action='%s', numpy score=%.4f",
            best_index,
            _metadata_action_name(best_meta),
            max_similarity,
        )
        return best_meta

    async def _ensure_description_embeddings(self) -> None:
        """Populate missing description embeddings for action metadata records."""
        for meta in self.cached_metadata:
            if "description_embedding" in meta:
                continue
            description: str = meta.get("description") or _metadata_action_name(meta)
            try:
                vector: NDArray[np.float32] = await self._encode_text(str(description))
                meta["description_embedding"] = self._normalize_vector(vector)
            except Exception as exc:
                logger.warning(
                    "Could not embed description for '%s': %s",
                    _metadata_action_name(meta),
                    exc,
                )

    async def __blm_routing(self, task: str) -> dict[str, Any]:
        """Route using a model-specific router when the optional laya dependency exists."""
        try:
            self.agent = laya.Router(preload=True)
            options: dict[str, dict[str, Any]] = {
                m["action_name"]: {
                    "description": m.get("description", ""),
                    "keywords": m.get("keywords", []),
                }
                for m in self.cached_metadata
            }
            question: dict[str, Any] = {
                "method": {
                    "type": "choice",
                    "instruction": "Choose the most suitable method by description and keywords.",
                    "options": options,
                }
            }
            answer: Any = await asyncio.to_thread(self.agent.predict, task, question)
            selected_method_name: Any = answer.get("answers", {}).get("method", {}).get("choice")

            for meta in self.cached_metadata:
                if meta["action_name"] == selected_method_name:
                    logger.info("BLM routing selected action '%s'", selected_method_name)
                    return meta
            raise PluginNotFoundError(
                f"Selected method '{selected_method_name}' not found in metadata"
            )
        except Exception as exc:
            logger.error("BLM routing failed: %s", exc, exc_info=True)
            raise

    async def extract_args(
        self,
        task: str,
        expected_args: dict[str, Any],
        current_context: dict[str, Any],
        memory_example: Optional[tuple[str, str, str]],
    ) -> dict[str, Any]:
        """Infer argument values for a plugin method from the active runtime context."""
        memory_prompt_block: str = ""
        if memory_example:
            past_prompt, past_context, past_json = memory_example
            memory_prompt_block = (
                "PAST SUCCESSFUL EXPERIENCE:\n"
                f"- System state: {past_context}\n"
                f'- User said: "{past_prompt}"\n'
                f"- Calculated result: {past_json}\n\n"
            )

        readable_schema: dict[str, str] = {
            key: getattr(val, "__name__", str(val)) for key, val in expected_args.items()
        }

        prompt: str = (
            "You are a strict data extraction compiler for an automation environment (AIE).\n"
            f"REQUIRED ARGUMENTS AND TYPES: {json.dumps(readable_schema)}.\n"
            f"CURRENT LIVE ENVIRONMENT STATE: {json.dumps(current_context)}.\n\n"
            f"{memory_prompt_block}"
            "COMPILATION RULES:\n"
            "1. Extracted JSON keys MUST strictly match the REQUIRED ARGUMENTS.\n"
            "2. Convert types according to expectations (e.g., if type is 'int', provide a "
            "number, if 'bool', provide true/false).\n"
            "3. Resolve relative requests (e.g., 'louder', 'hotter') using calculations based "
            "on CURRENT LIVE ENVIRONMENT STATE.\n"
            "4. Output ONLY raw, valid JSON matching the exact target keys without markdown "
            "formatting or markdown code blocks."
        )

        try:
            response: Any = await get_data_generated_by_model(
                prompt=prompt,
                provider=self.provider,
                user_prompt=task,
                format="json",
                model=self.model_name,
                context_window=self.config["num_ctx"],
            )
        except Exception as exc:
            logger.error("Argument extraction LLM call failed: %s", exc, exc_info=True)
            return {}

        try:
            if isinstance(response, dict):
                return response
            return json.loads(response)
        except (json.JSONDecodeError, TypeError) as exc:
            logger.warning("Could not parse extracted arguments JSON: %s", exc)
            return {}

    async def _process_single_task(
        self,
        task: str,
        plugin_meta: dict[str, Any],
        system_context: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any], NDArray[np.float32]]:
        """Prepare a single executable task payload from the plugin and current context."""
        plugin_name: str = plugin_meta.get("plugin_name", "")
        plugin_context: dict[str, Any] = system_context.get(plugin_name, {})
        expected_schema: dict[str, Any] = plugin_meta.get("args", {})

        task_emb: NDArray[np.float32] = await self._encode_text(task)
        try:
            memory_example: Optional[tuple[str, str, str]] = await self.runtime_memory.search_relevant_example(
                "extraction", task_emb
            )
        except Exception as exc:
            logger.error("RAG lookup for extraction failed: %s", exc, exc_info=True)
            memory_example = None

        needable_args: dict[str, Any] = await self.extract_args(
            task, expected_schema, plugin_context, memory_example
        )

        func_data: dict[str, Any] = {
            "method": plugin_meta["name"],
            "action_name": plugin_meta.get("action_name"),
            "plugin_name": plugin_name,
            "args": needable_args,
        }
        logger.info("Prepared execution payload for action '%s'", func_data["action_name"])
        return func_data, plugin_context, task_emb

    async def run_pipeline(
        self,
        user_prompt: str,
        current_context: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """Build and execute a full task pipeline for a user request."""
        res: list[dict[str, Any]] = []
        system_context: dict[str, Any] = current_context if current_context is not None else {}
        tasks: list[Any]
        prompt_emb: NDArray[np.float32]
        tasks, prompt_emb = await self.break_prompt_to_tasks(user_prompt, system_context)
        if not tasks:
            logger.warning("Pipeline aborted: no tasks produced from prompt")
            return []

        execution_pipeline: list[dict[str, Any]] = await self._build_execution_pipeline(tasks)
        await self._process_pipeline_streams(execution_pipeline, system_context, res)

        decomposition_result: dict[str, list[str]] = {"tasks": [str(task) for task in tasks]}
        try:
            await self.runtime_memory.add_record(
                "decomposition", user_prompt, {}, decomposition_result, prompt_emb
            )
        except Exception as exc:
            logger.error("Failed to persist decomposition memory: %s", exc, exc_info=True)

        logger.info("Pipeline produced %s executable item(s)", len(res))
        return res

    async def _build_execution_pipeline(self, tasks: list[Any]) -> list[dict[str, Any]]:
        """Transform an abstract task list into an executable ordered pipeline."""
        pipeline: list[dict[str, Any]] = []
        current_parallel: list[PipelineNode] = []

        for node in tasks:
            if isinstance(node, dict) and "condition_variable" in node:
                if current_parallel:
                    pipeline.append({"type": "parallel", "nodes": current_parallel})
                    current_parallel = []
                pipeline.append({"type": "conditional", "node": node})
            else:
                task_str: str = str(node)
                try:
                    meta: dict[str, Any] = await self.referal_ai_to_correct_plugin(task_str)
                    is_blocking: bool = meta.get("execution_type") == "blocking"

                    if is_blocking:
                        if current_parallel:
                            pipeline.append({"type": "parallel", "nodes": current_parallel})
                            current_parallel = []
                        pipeline.append({"type": "sequential", "nodes": [(task_str, meta)]})
                    else:
                        current_parallel.append((task_str, meta))
                except Exception as exc:
                    logger.error(
                        "Routing failed for task '%s': %s",
                        task_str,
                        exc,
                        exc_info=True,
                    )
                    current_parallel.append(
                        (
                            task_str,
                            {
                                "name": task_str,
                                "execution_type": "parallel",
                                "type": "general",
                                "file": "main",
                            },
                        )
                    )

        if current_parallel:
            pipeline.append({"type": "parallel", "nodes": current_parallel})

        return pipeline

    async def _process_pipeline_streams(
        self,
        pipeline: list[dict[str, Any]],
        system_context: dict[str, Any],
        res: list[dict[str, Any]],
    ) -> None:
        """Run each stream block in the ordered pipeline."""
        for stream in pipeline:
            stream_type: str = stream["type"]
            if stream_type == "parallel":
                await self._execute_parallel_stream(stream["nodes"], system_context, res)
            elif stream_type == "sequential":
                await self._execute_sequential_stream(stream["nodes"], system_context, res)
            elif stream_type == "conditional":
                await self._execute_conditional_stream(stream["node"], system_context, res)

    async def _execute_parallel_stream(
        self,
        nodes: list[PipelineNode],
        system_context: dict[str, Any],
        res: list[dict[str, Any]],
    ) -> None:
        """Execute multiple tasks concurrently and store their results as memory."""
        task_workers: list[Any] = [
            self._process_single_task(task, meta, system_context) for task, meta in nodes
        ]
        workers_results: list[Any] = await asyncio.gather(*task_workers, return_exceptions=True)

        for (task, _meta), result in zip(nodes, workers_results):
            if isinstance(result, Exception):
                logger.error(
                    "Parallel plugin task '%s' failed: %s",
                    task,
                    result,
                    exc_info=(type(result), result, result.__traceback__),
                )
                continue
            func_data: dict[str, Any]
            task_ctx: dict[str, Any]
            t_emb: NDArray[np.float32]
            func_data, task_ctx, t_emb = result
            res.append(func_data)
            try:
                await self.runtime_memory.add_record(
                    "extraction", task, task_ctx, func_data["args"], t_emb
                )
            except Exception as exc:
                logger.error(
                    "Failed to persist extraction memory for '%s': %s",
                    task,
                    exc,
                    exc_info=True,
                )

    async def _execute_sequential_stream(
        self,
        nodes: list[PipelineNode],
        system_context: dict[str, Any],
        res: list[dict[str, Any]],
    ) -> None:
        """Execute tasks in order so later actions can depend on earlier output."""
        for task, meta in nodes:
            try:
                result: tuple[dict[str, Any], dict[str, Any], NDArray[np.float32]] = await self._process_single_task(
                    task, meta, system_context
                )
                func_data: dict[str, Any]
                task_ctx: dict[str, Any]
                t_emb: NDArray[np.float32]
                func_data, task_ctx, t_emb = result
                res.append(func_data)
                await self.runtime_memory.add_record(
                    "extraction", task, task_ctx, func_data["args"], t_emb
                )
            except Exception as exc:
                logger.error(
                    "Sequential plugin task '%s' failed: %s",
                    task,
                    exc,
                    exc_info=True,
                )
                continue

    async def _execute_conditional_stream(
        self,
        node: dict[str, Any],
        system_context: dict[str, Any],
        res: list[dict[str, Any]],
    ) -> None:
        """Evaluate a condition and execute the matching branch of tasks."""
        cond_var: str = node.get("condition_variable", "")
        condition_met: bool = False

        if "." in cond_var:
            plugin_part, key_part = cond_var.split(".", 1)
            plugin_ctx: dict[str, Any] = system_context.get(plugin_part, {})
            condition_met = bool(plugin_ctx.get(key_part, False))

        logger.info(
            "Conditional branch '%s' evaluated to %s",
            cond_var,
            condition_met,
        )
        branch_key: str = "if_true" if condition_met else "if_false"
        next_tasks: list[Any] = node.get(branch_key, [])

        if next_tasks:
            sub_pipeline: list[dict[str, Any]] = await self._build_execution_pipeline(next_tasks)
            await self._process_pipeline_streams(sub_pipeline, system_context, res)

    async def import_func(self, task_data: dict[str, Any]) -> Any:
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
