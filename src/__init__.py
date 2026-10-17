from engine import AIEngine
from sdk import BasePlugin, decription, describe
import exceptions

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("your-project-name")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "AIEngine",
    "describe",
    "decription",
    "BasePlugin",
    "exceptions"
]
