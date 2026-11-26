from __future__ import annotations

__version__ = "0.2.0"

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("aifunckit")
except PackageNotFoundError:
    __version__ = "0.2.0"

__all__ = [
    "AIEngine",
    "describe",
    "decription",
    "BasePlugin",
    "exceptions",
    "engine",
    "sdk",
]


def __dir__() -> list[str]:
    return sorted(__all__)


def __getattr__(name: str):
    if name == "AIEngine":
        from .engine import AIEngine

        return AIEngine
    if name == "BasePlugin":
        from .sdk import BasePlugin

        return BasePlugin
    if name == "decription":
        from .sdk import decription

        return decription
    if name == "describe":
        from .sdk import describe

        return describe
    if name == "exceptions":
        from . import exceptions

        return exceptions
    if name == "engine":
        from . import engine

        return engine
    if name == "sdk":
        from . import sdk

        return sdk
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
