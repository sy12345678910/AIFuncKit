import asyncio
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from log_config import logger
from schemas.orchestrator_schemas import PipelineNode, LLMConfig, MetadataForLLM
from schemas.memory_schemas import RecordData
from memory.rag import AIRAGMemoryManager
from memory.embedding import EmbeddingService
from orchestrator.break_prompt import break_prompt_to_tasks
from orchestrator.extract_args import extract_args
from orchestrator.select_method import referal_ai_to_correct_plugin

class Pipeline:
    def __init__(
            self,
            cached_metadata: list[dict[str, Any]],
            provider: str = "ollama",
            model_name: str = "gemma4:e4b",
            context_window: int = 3072,
            db_path: str = "aie_runtime.db",
            emb_model: Any = "sentence_transformers/paraphrase-multilingual-MiniLM-L12-v2",
            routing_model: str = "vector",
        ) -> None:
        self.runtime_memory = AIRAGMemoryManager(db_path=db_path)
        self.llm_config = LLMConfig(
            provider=provider,
            model_name=model_name,
            context_window=context_window,
        )
        self.cached_metadata = cached_metadata
        self.embedding_service = EmbeddingService(
            emb_model=emb_model
        )
        self.routing_model = routing_model
        
    async def run_pipeline(
        self,
        user_prompt: str,
        current_context: Optional[dict[str, Any]] = {},
    ) -> list[dict[str, Any]]:
        """Build and execute a full task pipeline for a user request."""
        res: list[dict[str, Any]] = []
        system_context: dict[str, Any] = current_context 
        tasks: list[Any]
        prompt_emb: NDArray[np.float32]
        meta_for_llm = MetadataForLLM(
            cached_metadata=self.cached_metadata,
            context_block=system_context
        )
        tasks, prompt_emb = await break_prompt_to_tasks(
            user_prompt=user_prompt, 
            llm_config=self.llm_config,
            metadata_for_llm=meta_for_llm,
            memory=self.runtime_memory,
            embedding=self.embedding_service,
        )
        if not tasks:
            logger.warning("Pipeline aborted: no tasks produced from prompt")
            return []

        execution_pipeline: list[dict[str, Any]] = await self._build_execution_pipeline(tasks)
        await self._process_pipeline_streams(execution_pipeline, system_context, res)

        decomposition_result: dict[str, list[str]] = {"tasks": [str(task) for task in tasks]}
        try:
            record_data = RecordData(
                layer="decomposition",
                user_prompt=user_prompt, 
                context_dict= {},
                args=decomposition_result,
                raw_vector=prompt_emb,
            ) 
            await self.runtime_memory.add_record(
                record_data=record_data
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
                    meta: dict[str, Any] = await referal_ai_to_correct_plugin(
                        task_str,
                        self.cached_metadata,
                        self.routing_model,
                        self.embedding_service,
                    )
                    is_blocking: bool = meta.get("execution_type") == "blocking"

                    if not is_blocking:
                        current_parallel.append((task_str, meta))
                    if current_parallel:
                        pipeline.append({"type": "parallel", "nodes": current_parallel})
                        current_parallel = []
                    pipeline.append({"type": "sequential", "nodes": [(task_str, meta)]})
                        
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

        task_emb: NDArray[np.float32] = await self.embedding_service._encode_text(task)
        try:
            memory_example: Optional[tuple[str, str, str]] = await self.runtime_memory.search_relevant_example(
            "extraction", task_emb
        )
        except Exception as exc:
            logger.error("RAG lookup for extraction failed: %s", exc, exc_info=True)
            memory_example = None

        needable_args: dict[str, Any] = await extract_args(
            task, expected_schema, plugin_context, self.llm_config, memory_example
        )

        func_data: dict[str, Any] = {
            "method": plugin_meta["name"],
            "action_name": plugin_meta.get("action_name"),
            "plugin_name": plugin_name,
            "args": needable_args,
        }
        logger.info("Prepared execution payload for action '%s'", func_data["action_name"])
        return func_data, plugin_context, task_emb

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

        for (task, _), result in zip(nodes, workers_results):
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
                record_data = RecordData(
                    layer="extraction",
                    user_prompt=task,
                    context_dict=task_ctx,
                    args=func_data,
                    raw_vector=t_emb
                )
                await self.runtime_memory.add_record(
                    record_data=record_data
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
                    * "extraction", task, task_ctx, func_data["args"], t_emb
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
