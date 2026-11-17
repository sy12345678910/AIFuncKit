from typing import List, Dict, Any, Optional
from pydantic import BaseModel, ConfigDict

class LLMConfig(BaseModel):
    provider: str = "ollama"
    model_name: str = "gemma4:e4b"
    context_window: int = 3072

class MetadataForLLM(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    cached_metadata: List[dict[str, Any]]
    context_block: Any = ""
    
class PipelineNode(BaseModel):
    type: str
    pipeline_type: str = "sequential"
    description: Dict[str, Any] = {}

class ParamsForExecution(BaseModel):
    async_loop: bool = False
    
class PromptConfig(BaseModel):
    user_prompt: str
    prompt_injection_patern: Optional[List[str]]
    max_prompt_length: int