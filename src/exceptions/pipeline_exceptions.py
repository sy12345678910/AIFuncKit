from __future__ import annotations

from exceptions.exceptions import __AIBaseException


class __BasePipelineException(__AIBaseException):
    """Base exception type for pipeline errors."""
    pass


class InvalidPipelineStateException(__BasePipelineException):
    """Raised when pipeline context or task state is invalid."""
    pass

class CircuitBreakerOpenException(__BasePipelineException):
    "Raised when pipeline error threasholder is open"
    pass

class HumansStopPipelineException(__BasePipelineException):
    "Raised when human want to stop pipeline"
    pass

InvaildPipelineStateException = InvalidPipelineStateException