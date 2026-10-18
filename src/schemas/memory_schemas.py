from typing import Dict
from pydantic import BaseModel, ConfigDict
import numpy as np
from numpy.typing import NDArray

class MemoryRecord(BaseModel):
    user_prompt: str
    context_snapshot: str
    extracted_json: str
    
class ActiveVectors(BaseModel):
    id: int
    layer: str
    embedding: bytes
    
    @property
    def vector(self) -> NDArray[np.float32]:
        return np.frombuffer(self.embedding, dtype=np.float32)
    
class RecordData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    layer: str
    user_prompt: str
    context_dict: Dict
    args: Dict
    raw_vector: np.ndarray