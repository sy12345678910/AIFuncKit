from __future__ import annotations


class __AIBaseException(Exception):
    """Base exception type for all IntentC runtime errors."""

    def __init__(self, message: str) -> None:
        """Initialize the exception with a human-readable message."""
        super().__init__(message)


class PluginNotFoundError(__AIBaseException):
    """Raised when a requested plugin cannot be located in the registry."""

    def __init__(self, message: str) -> None:
        """Create a plugin lookup error for the specified message."""
        super().__init__(message)


class PluginForDeleteNotFoundError(PluginNotFoundError):
    """Raised when the runtime tries to delete a plugin that is not registered."""

    def __init__(self, message: str) -> None:
        """Create a deletion error with the given message."""
        super().__init__(message)


class RoutingModeDoesNotExist(__AIBaseException):
    """Raised when a routing mode name does not match a supported strategy."""

    def __init__(self, message: str) -> None:
        """Create a routing configuration error with the provided message."""
        super().__init__(message)

