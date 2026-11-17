from pydantic import BaseModel, ConfigDict
from typing import Any, Dict, List
from schemas.orchestrator_schemas import LLMConfig, MetadataForLLM, ParamsForExecution, PromptConfig


class PipelineContext(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    user_prompt: PromptConfig
    llm_config: LLMConfig
    metadata: MetadataForLLM
    memory: Any
    embedding: Any

    environment_state: Dict[str, Any] = {}
    decomposed_tasks: List[Any] = []
    execution_results: List[Dict[str, Any]] = []
    params_for_execution: ParamsForExecution