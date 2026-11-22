"""
Tests for AIEngine – the high-level public API.

What we test here:
  - register_plugin() adds methods to cached_metadata and actions.
  - register_plugin() skips duplicate plugins (idempotent).
  - delete_plugin() removes plugin data; raises PluginForDeleteNotFoundError for unknown.
  - action() decorator registers a standalone function.
  - get_current_global_context() aggregates live context from plugins.
  - stop_engine() / resume_engine() interact with the circuit breaker.
  - run_engine() calls orchestrator and executes returned callables.
  - run_engine() returns [] on orchestrator failure (defensive).
  - schedule_task / list_scheduled_tasks / unschedule_task round-trip.

All LLM and embedding calls are mocked – no network or model needed.
"""
from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from exceptions.exceptions import PluginForDeleteNotFoundError
from exceptions.pipeline_exceptions import CircuitBreakerOpenException
from sdk import BasePlugin, describe


# ---------------------------------------------------------------------------
# Minimal concrete plugin for testing
# ---------------------------------------------------------------------------

class DemoPlugin(BasePlugin):
    plugin_name = "demo"

    @describe("Say hello to someone")
    def say_hello(self, name: str) -> str:
        return f"Hello, {name}!"

    @describe("Return a fixed number")
    def fixed_number(self) -> int:
        return 42


# ---------------------------------------------------------------------------
# Engine factory (patches away all heavy init)
# ---------------------------------------------------------------------------

def _make_engine(**overrides):
    """Create AIEngine with all external dependencies stubbed out."""
    with (
        patch("engine.configure_logging"),
        patch("engine.MasterOrchestrationEngine") as MockOrch,
        patch("engine.SchedulerService") as MockSched,
    ):
        # Build a reusable mock orchestrator
        orch = MagicMock()
        orch.breaker_circus = MagicMock()
        orch.breaker_circus.check_status = MagicMock()
        orch.breaker_circus.emergency_halt = MagicMock()
        orch.breaker_circus.reset = MagicMock()
        orch.runtime_memory = AsyncMock()
        orch.client = MagicMock()
        orch.run_pipeline = AsyncMock(return_value=[])
        MockOrch.return_value = orch

        sched = MagicMock()
        sched.schedule = MagicMock(return_value="task-123")
        sched.list_tasks = MagicMock(return_value=[])
        sched.stop = MagicMock()
        sched.stop_all = MagicMock()
        MockSched.return_value = sched

        from engine import AIEngine
        engine = AIEngine(provider="ollama", **overrides)
        engine._mock_orch = orch
        engine._mock_sched = sched
        return engine


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------

class TestRegisterPlugin:
    def test_register_adds_methods_to_metadata(self):
        engine = _make_engine()
        plugin = DemoPlugin()
        engine.register_plugin(plugin)
        action_names = [m["action_name"] for m in engine.cached_metadata]
        assert "say_hello" in action_names
        assert "fixed_number" in action_names

    def test_register_adds_to_actions(self):
        engine = _make_engine()
        engine.register_plugin(DemoPlugin())
        assert "say_hello" in engine.actions
        assert "fixed_number" in engine.actions

    def test_register_plugin_idempotent(self):
        engine = _make_engine()
        engine.register_plugin(DemoPlugin())
        count_before = len(engine.cached_metadata)
        engine.register_plugin(DemoPlugin())   # duplicate – should skip
        assert len(engine.cached_metadata) == count_before

    def test_plugin_stored_in_plugins_dict(self):
        engine = _make_engine()
        p = DemoPlugin()
        engine.register_plugin(p)
        assert "demo" in engine.plugins


# ---------------------------------------------------------------------------
# Plugin deletion
# ---------------------------------------------------------------------------

class TestDeletePlugin:
    def test_delete_removes_plugin(self):
        engine = _make_engine()
        engine.register_plugin(DemoPlugin())
        engine.delete_plugin("demo")
        assert "demo" not in engine.plugins

    def test_delete_nonexistent_raises(self):
        engine = _make_engine()
        with pytest.raises(PluginForDeleteNotFoundError):
            engine.delete_plugin("ghost_plugin")

    def test_delete_with_plugin_suffix_in_name(self):
        """delete_plugin strips trailing 'plugin' from name before lookup."""
        engine = _make_engine()
        engine.register_plugin(DemoPlugin())
        # 'demoplugin' → normalised to 'demo'
        engine.delete_plugin("demoplugin")
        assert "demo" not in engine.plugins


# ---------------------------------------------------------------------------
# @action decorator
# ---------------------------------------------------------------------------

class TestActionDecorator:
    def test_action_decorator_registers_function(self):
        engine = _make_engine()

        @engine.action(description="Ping the service")
        def ping() -> str:
            return "pong"

        assert "ping" in engine.actions
        assert engine.actions["ping"] is ping

    def test_action_decorator_with_custom_name(self):
        engine = _make_engine()

        @engine.action(description="Pong back", name="my_pong")
        def pong() -> str:
            return "pong"

        assert "my_pong" in engine.actions

    def test_action_decorator_extracts_arg_types(self):
        engine = _make_engine()

        @engine.action(description="Add numbers")
        def add(x: int, y: int) -> int:
            return x + y

        meta = next(m for m in engine.cached_metadata if m["action_name"] == "add")
        assert meta["args"]["x"] == "int"
        assert meta["args"]["y"] == "int"


# ---------------------------------------------------------------------------
# get_current_global_context
# ---------------------------------------------------------------------------

class TestGetGlobalContext:
    def test_aggregates_plugin_contexts(self):
        engine = _make_engine()
        p = DemoPlugin()
        p.update_context("last_greeting", "hi")
        engine.register_plugin(p)

        ctx = engine.get_current_global_context()
        assert "demo" in ctx
        assert ctx["demo"]["last_greeting"] == "hi"

    def test_empty_when_no_plugins(self):
        engine = _make_engine()
        assert engine.get_current_global_context() == {}


# ---------------------------------------------------------------------------
# Circuit breaker integration via stop/resume
# ---------------------------------------------------------------------------

class TestStopResume:
    def test_stop_engine_calls_emergency_halt(self):
        engine = _make_engine()
        engine.stop_engine("test reason")
        engine._mock_orch.breaker_circus.emergency_halt.assert_called_once()

    def test_stop_engine_calls_stop_all_scheduler(self):
        engine = _make_engine()
        engine.stop_engine()
        engine._mock_sched.stop_all.assert_called_once()

    def test_resume_engine_calls_reset(self):
        engine = _make_engine()
        engine.resume_engine()
        engine._mock_orch.breaker_circus.reset.assert_called_once()


# ---------------------------------------------------------------------------
# run_engine
# ---------------------------------------------------------------------------

class TestRunEngine:
    @pytest.mark.asyncio
    async def test_run_engine_executes_returned_callable(self):
        engine = _make_engine()

        def greet(name: str) -> str:
            return f"Hi {name}"

        engine._mock_orch.run_pipeline = AsyncMock(
            return_value=[
                {"method": greet, "args": {"name": "Alice"}, "action_name": "greet"}
            ]
        )
        results = await engine.run_engine("greet Alice")
        assert results[0]["action"] == "greet"
        assert results[0]["result"] == "Hi Alice"

    @pytest.mark.asyncio
    async def test_run_engine_resolves_action_by_name(self):
        """method stored as string key → looked up in self.actions."""
        engine = _make_engine()

        def my_func() -> str:
            return "done"

        engine.actions["my_func"] = my_func
        engine._mock_orch.run_pipeline = AsyncMock(
            return_value=[
                {"method": "my_func", "args": {}, "action_name": "my_func"}
            ]
        )
        results = await engine.run_engine("run my_func")
        assert results[0]["result"] == "done"

    @pytest.mark.asyncio
    async def test_run_engine_returns_empty_list_on_pipeline_failure(self):
        engine = _make_engine()
        engine._mock_orch.run_pipeline = AsyncMock(side_effect=RuntimeError("boom"))
        results = await engine.run_engine("some prompt")
        assert results == []

    @pytest.mark.asyncio
    async def test_run_engine_skips_missing_callable(self):
        """If action_name points to nothing registered, entry is skipped (no crash)."""
        engine = _make_engine()
        engine._mock_orch.run_pipeline = AsyncMock(
            return_value=[
                {"method": None, "args": {}, "action_name": "ghost"}
            ]
        )
        results = await engine.run_engine("call ghost")
        assert results == []

    @pytest.mark.asyncio
    async def test_run_engine_uses_precomputed_result(self):
        """If 'result' key already in data dict, it must be used as-is."""
        engine = _make_engine()
        engine._mock_orch.run_pipeline = AsyncMock(
            return_value=[
                {
                    "method": lambda: "should not run",
                    "args": {},
                    "action_name": "cached",
                    "result": "pre-cached value",
                }
            ]
        )
        results = await engine.run_engine("use cached")
        assert results[0]["result"] == "pre-cached value"

    @pytest.mark.asyncio
    async def test_run_engine_supports_async_action(self):
        engine = _make_engine()

        async def async_action() -> str:
            return "async result"

        engine._mock_orch.run_pipeline = AsyncMock(
            return_value=[
                {"method": async_action, "args": {}, "action_name": "async_action"}
            ]
        )
        results = await engine.run_engine("run async")
        assert results[0]["result"] == "async result"


# ---------------------------------------------------------------------------
# Scheduler integration
# ---------------------------------------------------------------------------

class TestScheduler:
    def test_schedule_task_returns_task_id(self):
        engine = _make_engine()
        tid = engine.schedule_task("check disk", interval_seconds=60)
        assert tid == "task-123"

    def test_list_scheduled_tasks_delegates(self):
        engine = _make_engine()
        engine._mock_sched.list_tasks.return_value = [{"id": "task-123"}]
        tasks = engine.list_scheduled_tasks()
        assert tasks[0]["id"] == "task-123"

    def test_unschedule_task_calls_stop(self):
        engine = _make_engine()
        engine.unschedule_task("task-123")
        engine._mock_sched.stop.assert_called_once_with("task-123")
