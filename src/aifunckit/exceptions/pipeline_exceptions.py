from __future__ import annotations

from aifunckit.exceptions.exceptions import __AIBaseException


class __BasePipelineException(__AIBaseException):
    """Base exception type for pipeline errors."""
    def __init__(self, message: str):
        super().__init__(message)


class InvalidPipelineStateException(__BasePipelineException):
    """Raised when pipeline context or task state is invalid."""
    def __init__(self, message: str):
        super().__init__(message)

class CircuitBreakerOpenException(__BasePipelineException):
    "Raised when pipeline error threasholder is open"
    def __init__(self, message: str):
        super().__init__(message)

class HumansStopPipelineException(__BasePipelineException):
    "Raised when human want to stop pipeline"
    def __init__(self, message: str):
        super().__init__(message)

InvaildPipelineStateException = InvalidPipelineStateException