import copy
from typing import Any
from schemas.pipeline_context import PipelineContext

class StateManagerService:
    def __init__(self, context: PipelineContext):
        self.context = context
        self.snapshot: dict[str, Any] = {}
        
    def capture(self) -> None:
        self.snapshot = copy.deepcopy(self.context.environment_state)
        
    def rollback(self) -> None:
        self.context.environment_state = copy.deepcopy(self.snapshot)