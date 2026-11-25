from __future__ import annotations

import pytest


def test_public_top_level_exports():
    import aifunckit

    assert set(aifunckit.__all__) == {
        "AIEngine",
        "BasePlugin",
        "decription",
        "describe",
        "engine",
        "exceptions",
        "sdk",
    }
    assert dir(aifunckit) == sorted(aifunckit.__all__)


def test_public_submodules_importable():
    from aifunckit import AIEngine, BasePlugin, decription, describe, exceptions
    from aifunckit.engine import AIEngine as EngineFromSubmodule
    from aifunckit.sdk import BasePlugin as BasePluginFromSubmodule
    from aifunckit.exceptions import (
        CircuitBreakerOpenException,
        PluginForDeleteNotFoundError,
        PluginNotFoundError,
        PromptInjectionError,
    )

    assert AIEngine is EngineFromSubmodule
    assert BasePlugin is BasePluginFromSubmodule
    assert callable(describe)
    assert callable(decription)
    assert exceptions.PluginNotFoundError is PluginNotFoundError
    assert CircuitBreakerOpenException is not None
    assert PluginForDeleteNotFoundError is not None
    assert PromptInjectionError is not None


@pytest.mark.parametrize(
    "internal_name",
    ["orchestrator", "memory", "model_provide", "schemas", "log_config", "plugins"],
)
def test_internal_modules_not_exposed_on_public_namespace(internal_name: str):
    import aifunckit

    with pytest.raises(AttributeError):
        getattr(aifunckit, internal_name)

    with pytest.raises(ModuleNotFoundError):
        __import__(f"aifunckit.{internal_name}")
