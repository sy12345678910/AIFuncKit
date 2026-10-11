from __future__ import annotations
import inspect
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

ROOT = Path(__file__).resolve().parents[1] / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine import AIEngine
from log_config import logger
from sdk import BasePlugin, decription


@pytest.fixture
def engine_instance() -> AIEngine:
    engine = AIEngine(
        ai_model="model-test",
        context_window=1024,
        db_path=Path("./tmp_engine_test.db"),
        provider="ollama",
        emb_model="dummy-model",
        routing_model="vector",
        verbose=False,
        log_to_file=None,
    )
    yield engine
    engine.orchestrator.runtime_memory.close = AsyncMock()
    engine.cached_metadata.clear()
    engine.actions.clear()
    engine.plugins.clear()


@pytest.mark.asyncio
async def test_initialize_calls_embedding_and_memory_init(engine_instance: AIEngine) -> None:
    engine_instance.orchestrator.ensure_embedding_model = AsyncMock()
    engine_instance.orchestrator.runtime_memory.init_and_load_cache = AsyncMock()

    await engine_instance.initialize()

    engine_instance.orchestrator.ensure_embedding_model.assert_awaited_once()
    engine_instance.orchestrator.runtime_memory.init_and_load_cache.assert_awaited_once()


def test_register_plugin_collects_methods_and_metadata(engine_instance: AIEngine) -> None:
    class DemoPlugin(BasePlugin):
        def __init__(self) -> None:
            super().__init__()
            self.plugin_name = "demo"

        @decription("send demo message")
        def ping(self, count: int, enabled: bool) -> str:
            return f"{count}:{enabled}"

    plugin = DemoPlugin()
    engine_instance.register_plugin(plugin)

    assert plugin.plugin_name == "demo"
    assert "ping" in engine_instance.actions
    assert any(item.get("plugin_name") == "demo" for item in engine_instance.cached_metadata)
    assert engine_instance.plugins["demo"] is plugin


def test_action_decorator_reflects_argument_types(engine_instance: AIEngine) -> None:
    @engine_instance.action("do work", name="custom_action")
    def sample(self, count: int, enabled: bool, ratio: float, *, extras: str = "x") -> str:
        return "done"

    metadata = next(item for item in engine_instance.cached_metadata if item["action_name"] == "custom_action")
    assert metadata["args"] == {"count": "int", "enabled": "bool", "ratio": "float", "extras": "str"}
    assert "self" not in metadata["args"]


@pytest.mark.asyncio
async def test_run_engine_executes_registered_callables_and_skips_invalid_ones(engine_instance: AIEngine) -> None:
    async def async_worker(value: int) -> int:
        return value + 1

    def sync_worker(value: int) -> int:
        return value + 2

    engine_instance.actions["async_worker"] = async_worker
    engine_instance.actions["sync_worker"] = sync_worker
    engine_instance.orchestrator.run_pipeline = AsyncMock(
        return_value=[
            {"method": async_worker, "action_name": "async_worker", "args": {"value": 10}},
            {"method": sync_worker, "action_name": "sync_worker", "args": {"value": 5}},
            {"method": "missing", "action_name": "missing", "args": {}},
        ]
    )

    results = await engine_instance.run_engine("do work")

    assert results[0]["action"] == "async_worker"
    assert results[0]["result"] == 11
    assert results[1]["action"] == "sync_worker"
    assert results[1]["result"] == 7
    assert len(results) == 2


@pytest.mark.asyncio
async def test_run_engine_handles_exception_in_one_action_without_stopping_the_queue(engine_instance: AIEngine) -> None:
    async def boom(value: int) -> int:
        raise RuntimeError("boom")

    async def ok(value: int) -> int:
        return value

    engine_instance.actions["boom"] = boom
    engine_instance.actions["ok"] = ok
    engine_instance.orchestrator.run_pipeline = AsyncMock(
        return_value=[
            {"method": boom, "action_name": "boom", "args": {"value": 1}},
            {"method": ok, "action_name": "ok", "args": {"value": 2}},
        ]
    )

    with patch.object(logger, "error") as mock_error:
        results = await engine_instance.run_engine("fail once")

    assert len(results) == 1
    assert results[0]["action"] == "ok"
    assert results[0]["result"] == 2
    assert mock_error.called


def test_remove_plugin_from_metadata_clears_actions_and_metadata(engine_instance: AIEngine) -> None:
    class DemoPlugin(BasePlugin):
        def __init__(self) -> None:
            super().__init__()
            self.plugin_name = "demo"

        @decription("get status")
        def ping(self, count: int) -> int:
            return count

    plugin = DemoPlugin()
    engine_instance.register_plugin(plugin)
    engine_instance._remove_plugin_from_metadata("demo")

    assert all(item.get("plugin_name") != "demo" for item in engine_instance.cached_metadata)
    assert "ping" not in engine_instance.actions
    assert "demo" in engine_instance.plugins
