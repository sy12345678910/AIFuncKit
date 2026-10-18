from typing import List, Dict, Any
from pydantic import BaseModel, ConfigDict

class LLMConfig(BaseModel):
    provider: str = "ollama"
    model_name: str = "gemma4:e4b"
    context_window: int = 3072

class MetadataForLLM(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    cached_metadata: List
    context_block: Any = ""
    
class PipelineNode(BaseModel):
    type: str
    description: Dict[str, Any]    