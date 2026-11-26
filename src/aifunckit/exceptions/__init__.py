from __future__ import annotations

from aifunckit.exceptions.db_exceptions import BaseDBError, DBArchiveError, DBConnectionError
from aifunckit.exceptions.exceptions import (
    __AIBaseException,
    PluginNotFoundError,
    PluginForDeleteNotFoundError,
    RoutingModeDoesNotExist,
)
from aifunckit.exceptions.pipeline_exceptions import (
    CircuitBreakerOpenException,
    HumansStopPipelineException,
    InvalidPipelineStateException,
    InvaildPipelineStateException,
)
from aifunckit.exceptions.prompt_exeption import PromptInjectionError, TooLongPromptError
from aifunckit.exceptions.vector_exceptions import FetchVectorError, VectorGenerationError

__all__ = [
    "__AIBaseException",
    "BaseDBError",
    "DBConnectionError",
    "DBArchiveError",
    "PluginNotFoundError",
    "PluginForDeleteNotFoundError",
    "RoutingModeDoesNotExist",
    "InvalidPipelineStateException",
    "InvaildPipelineStateException",
    "CircuitBreakerOpenException",
    "HumansStopPipelineException",
    "PromptInjectionError",
    "TooLongPromptError",
    "VectorGenerationError",
    "FetchVectorError",
]
