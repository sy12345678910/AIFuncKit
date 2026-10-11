from __future__ import annotations


class __AIBaseException(Exception):
    pass


class PluginNotFoundError(__AIBaseException):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class PluginForDeleteNotFoundError(PluginNotFoundError):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class RoutingModeDoesNotExist(__AIBaseException):
    def __init__(self, message: str) -> None:
        super().__init__(message)
