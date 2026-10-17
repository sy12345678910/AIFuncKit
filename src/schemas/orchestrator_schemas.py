from typing import List, Dict, Any
from pydantic import BaseModel

class LLMConfig(BaseModel):
    provider: str = "ollama"
    model_name: str = "gemma4:e4b"
    context_window: int = 3072

class MetadataForLLM(BaseModel):
    cached_metadata: List
    context_block: str
    
class PipelineNode(BaseModel):
    type: str
    description: Dict[str, Any]    