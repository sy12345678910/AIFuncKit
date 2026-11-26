import copy
from typing import Any
from aifunckit._internal.log_config import logger
from aifunckit._internal.schemas.pipeline_context import PipelineContext

class StateManagerService:
    def __init__(self, context: PipelineContext):
        self.context = context
        self.snapshot: dict[str, Any] = {}
        
    def capture(self) -> None:
        try:
            self.snapshot = copy.deepcopy(self.context.environment_state)
            logger.debug("Captured environment state snapshot (%d keys)", len(self.snapshot))
        except Exception as exc:
            logger.error("Failed to capture environment state snapshot: %s", exc, exc_info=True)
            raise
        
    def rollback(self) -> None:
        try:
            self.context.environment_state = copy.deepcopy(self.snapshot)
            logger.info("Rolled back environment state to captured snapshot")
        except Exception as exc:
            logger.error("Failed to rollback environment state: %s", exc, exc_info=True)
            raise