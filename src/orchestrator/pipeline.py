from __future__ import annotations

import inspect
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from log_config import logger
from memory.embedding import EmbeddingService
from memory.rag import AIRAGMemoryManager
from schemas.pipeline_context import PipelineContext
from orchestrator.services.action_execution import ActionExecutionService
from orchestrator.services.argument_extraction import ArgumentExtractionService
from orchestrator.services.plugin_routing import PluginRoutingService
from orchestrator.services.prompt_decomposition import PromptDecompositionService
from orchestrator.topology_resolver import TopologyResolver
from schemas.memory_schemas import RecordData
from schemas.orchestrator_schemas import LLMConfig, MetadataForLLM


class MasterOrchestrationEngine:
    """Facade that decomposes a prompt, resolves topology, and boots pipeline execution."""

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
        model_name_or_path = emb_model if isinstance(emb_model, str) else str(emb_model)
        try:
            self.embedding_service = EmbeddingService(model_name=model_name_or_path)
        except TypeError:
            self.embedding_service = EmbeddingService(emb_model=model_name_or_path)
        self.routing_model = routing_model
        self.decomposition = PromptDecompositionService()
        self.routing = PluginRoutingService(routing_model=routing_model)
        self.extraction = ArgumentExtractionService()
        self.execution = ActionExecutionService()
        self.topology_resolver = TopologyResolver()
        self.client: Any = None

    async def ensure_embedding_model(self) -> None:
        """Ensure the configured embedding model is loaded for vector encoding."""
        ensure = getattr(self.embedding_service, "ensure_embedding_model", None)
        if ensure is None:
            return
        if inspect.iscoroutinefunction(ensure):
            await ensure()
        else:
            ensure()

    def _build_context(
        self,
        user_prompt: str,
        current_context: Optional[dict[str, Any]],
    ) -> PipelineContext:
        system_context: dict[str, Any] = current_context if current_context is not None else {}
        return PipelineContext(
            user_prompt=user_prompt,
            llm_config=self.llm_config,
            metadata=MetadataForLLM(
                cached_metadata=self.cached_metadata,
                context_block=system_context,
            ),
            memory=self.runtime_memory,
            embedding=self.embedding_service,
            environment_state=system_context,
        )

    async def run_pipeline(
        self,
        user_prompt: str,
        current_context: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """Decompose a prompt, resolve topology, and execute the selected pipeline."""
        context = self._build_context(user_prompt, current_context)
        prompt_emb: NDArray[np.float32] = await self.decomposition.decompose(context)
        if not context.decomposed_tasks:
            logger.warning("Pipeline aborted: no tasks produced from prompt")
            return []

        pipeline_cls = self.topology_resolver.resolve(context)
        pipeline = pipeline_cls(
            routing=self.routing,
            extraction=self.extraction,
            execution=self.execution,
        )
        await pipeline.execute(context)

        decomposition_result: dict[str, list[str]] = {
            "tasks": [str(task) for task in context.decomposed_tasks]
        }
        try:
            record_data = RecordData(
                layer="decomposition",
                user_prompt=user_prompt,
                context_dict={},
                args=decomposition_result,
                raw_vector=prompt_emb,
            )
            await self.runtime_memory.add_record(record_data=record_data)
        except Exception as exc:
            logger.error("Failed to persist decomposition memory: %s", exc, exc_info=True)

        logger.info("Pipeline produced %s executable item(s)", len(context.execution_results))
        return context.execution_results
