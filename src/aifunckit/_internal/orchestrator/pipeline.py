from __future__ import annotations

import inspect
from typing import Any, Optional, List

import numpy as np
from numpy.typing import NDArray

from aifunckit._internal.log_config import logger
from aifunckit._internal.memory.embedding import EmbeddingService
from aifunckit._internal.memory.rag import AIRAGMemoryManager
from aifunckit._internal.schemas.pipeline_context import PipelineContext
from aifunckit._internal.orchestrator.services.action_execution import ActionExecutionService
from aifunckit._internal.orchestrator.services.argument_extraction import ArgumentExtractionService
from aifunckit._internal.orchestrator.services.plugin_routing import PluginRoutingService
from aifunckit._internal.orchestrator.services.prompt_decomposition import PromptDecompositionService
from aifunckit._internal.orchestrator.services.state_manager import StateManagerService
from aifunckit._internal.orchestrator.services.sanitizer import PromptSanitizer
from aifunckit._internal.orchestrator.services.break_circus import BreakCircusService
from aifunckit._internal.orchestrator.topology_resolver import TopologyResolver
from aifunckit._internal.schemas.memory_schemas import RecordData
from aifunckit._internal.schemas.orchestrator_schemas import LLMConfig, MetadataForLLM, ParamsForExecution, PromptConfig
from aifunckit.exceptions.prompt_exeption import PromptInjectionError, TooLongPromptError
from aifunckit.exceptions.pipeline_exceptions import CircuitBreakerOpenException

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
        failure_threshold: int = 3,
        failure_threashold: Optional[int] = None,
        prompt_injection_protection: Optional[List[str]] = None,
        max_prompt_length: int = 4096,
        guard_model_path: Optional[str] = None,
        guard_tokenizer_path: Optional[str] = None,
        guard_threshold: float = 0.85,
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
        self.prompt_injection_protection = prompt_injection_protection
        self.max_prompt_length = max_prompt_length
        self.sanitizer = PromptSanitizer(
            model_path=guard_model_path,
            tokenizer_path=guard_tokenizer_path,
            custom_patterns=prompt_injection_protection,
            max_length=max_prompt_length,
            threshold=guard_threshold,
        )
        threshold = failure_threashold if failure_threashold is not None else failure_threshold
        self.breaker_circus = BreakCircusService(failure_threshold=threshold)
        self.client: Any = None

    @property
    def circus_breaker(self) -> BreakCircusService:
        return self.breaker_circus

    @circus_breaker.setter
    def circus_breaker(self, value: BreakCircusService) -> None:
        self.breaker_circus = value

    @property
    def circuit_breaker(self) -> BreakCircusService:
        return self.breaker_circus

    @circuit_breaker.setter
    def circuit_breaker(self, value: BreakCircusService) -> None:
        self.breaker_circus = value
        

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
        prompt_config = PromptConfig(
            user_prompt=user_prompt,
            max_prompt_length=self.max_prompt_length,
            prompt_injection_pattern=self.prompt_injection_protection,
        )
        return PipelineContext(
            user_prompt=prompt_config.user_prompt,
            prompt=prompt_config,
            llm_config=self.llm_config,
            metadata=MetadataForLLM(
                cached_metadata=self.cached_metadata,
                context_block=system_context,
            ),
            memory=self.runtime_memory,
            embedding=self.embedding_service,
            environment_state=system_context,
            params_for_execution=ParamsForExecution(async_loop=False),
        )

    async def run_pipeline(
        self,
        user_prompt: str,
        current_context: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """Decompose a prompt, resolve topology, and execute the selected pipeline."""
        self.breaker_circus.check_status()
        context = self._build_context(user_prompt, current_context)
        try:
            sanitized_prompt = self.sanitizer.sanitize(context.user_prompt)
        except (PromptInjectionError, TooLongPromptError):
            logger.error("Rejected user prompt before decomposition")
            raise
        context.user_prompt = sanitized_prompt
        if context.prompt is not None:
            context.prompt.user_prompt = sanitized_prompt
        prompt_emb: NDArray[np.float32] = await self.decomposition.decompose(context)
        if not context.decomposed_tasks:
            logger.warning("Pipeline aborted: no tasks produced from prompt")
            return []

        pipeline_cls = self.topology_resolver.resolve(context)
        pipeline = pipeline_cls(
            routing=self.routing,
            extraction=self.extraction,
            execution=self.execution,
            circuit_breaker=self.breaker_circus,
        )
        pipeline.validate_state(context)
        snapshot = StateManagerService(context=context)
        snapshot.capture()
        try:
            await pipeline.execute(context)
        except CircuitBreakerOpenException:
            snapshot.rollback()
            raise
        except Exception as e:
            snapshot.rollback()
            self.breaker_circus.record_failure()
            raise e

        if context.execution_results:
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
