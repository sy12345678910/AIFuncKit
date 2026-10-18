from pydantic import BaseModel, ConfigDict
from typing import Any, Dict, List
from schemas.orchestrator_schemas import LLMConfig, MetadataForLLM
from memory.rag import AIRAGMemoryManager
from memory.embedding import EmbeddingService

class PipelineContext(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    user_prompt: str
    llm_config: LLMConfig
    metadata: MetadataForLLM
    memory: AIRAGMemoryManager
    embedding: EmbeddingService
    
    environment_state: Dict[str, Any] = {}
    decomposed_tasks: List[Any] = []  
    execution_results: List[Dict[str, Any]] = []  
