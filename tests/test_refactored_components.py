from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from aifunckit._internal.engine.plugin_register import PluginRegisterClass
from aifunckit._internal.engine.runtime_manager import RuntimeManager
from aifunckit._internal.orchestrator.pipelines.node_solver import NodeSolver
from aifunckit._internal.orchestrator.pipelines.step_executer import StepExecuter
from aifunckit._internal.orchestrator.pipelines.validation_manager import PipelineValidationManager
from aifunckit._internal.orchestrator.services.break_circus import BreakCircusService
from aifunckit.exceptions import (
    CircuitBreakerOpenException,
    InvalidPipelineStateException,
    PluginForDeleteNotFoundError,
)
from aifunckit.sdk import BasePlugin, describe


class _DummyPlugin(BasePlugin):
    plugin_name = "dummy"

    @describe("Ping action")
    def ping(self, target: str) -> str:
        return f"pong:{target}"


# ---------------------------------------------------------------------------
# PluginRegisterClass
# ---------------------------------------------------------------------------


def test_plugin_register_shares_cached_metadata_in_place():
    shared_metadata: list[dict] = []
    orch_view = shared_metadata
    reg = PluginRegisterClass(cached_metadata=shared_metadata)

    reg.register_plugin(_DummyPlugin())
    assert len(orch_view) == 1
    assert orch_view[0]["action_name"] == "ping"

    # delete_plugin must mutate shared_metadata in-place so orch_view stays synced
    reg.delete_plugin("dummy")
    assert orch_view == []
    assert reg.actions == {}


def test_plugin_register_delete_missing_raises():
    reg = PluginRegisterClass()
    with pytest.raises(PluginForDeleteNotFoundError):
        reg.delete_plugin("unknown")


def test_plugin_register_action_decorator():
    reg = PluginRegisterClass()

    @reg.action("Add numbers", name="add_nums")
    def add(a: int, b: int) -> int:
        return a + b

    assert "add_nums" in reg.actions
    assert reg.cached_metadata[0]["args"] == {"a": "int", "b": "int"}


# ---------------------------------------------------------------------------
# RuntimeManager
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_runtime_manager_run_stop_resume():
    orch = MagicMock()
    orch.breaker_circus = BreakCircusService(failure_threshold=2)
    orch.run_pipeline = AsyncMock(
        return_value=[{"action_name": "ping", "method": lambda target: f"hi {target}", "args": {"target": "world"}}]
    )
    sched = MagicMock()
    reg = PluginRegisterClass()

    rm = RuntimeManager(orchestrator=orch, scheduler=sched, plugin_register=reg)
    results = await rm.run_engine("say hi")
    assert results == [{"action": "ping", "result": "hi world"}]

    with pytest.raises(CircuitBreakerOpenException):
        rm.stop_engine("maintenance")
    assert orch.breaker_circus.is_open is True

    with pytest.raises(CircuitBreakerOpenException):
        await rm.run_engine("should fail")

    rm.resume_engine()
    assert not orch.breaker_circus.is_open


# ---------------------------------------------------------------------------
# NodeSolver
# ---------------------------------------------------------------------------


def test_node_solver_dict_and_object_classification():
    assert NodeSolver.is_parallel_node({"execution_type": "parallel"})
    assert NodeSolver.is_conditional_node({"condition_variable": "flag"})
    assert NodeSolver.is_loop_node({"iterations": 3})
    assert NodeSolver.is_for_each_node({"items": [1, 2]})
    assert NodeSolver.is_timeout_node({"timeout_seconds": 5})

    # Plain dict must NOT match for_each via dict.items method
    assert not NodeSolver.is_for_each_node({"description": "plain task"})

    # Object-based nodes
    assert NodeSolver.is_conditional_node(SimpleNamespace(condition_variable="ok"))
    assert NodeSolver.is_loop_node(SimpleNamespace(repetitions=2))
    assert NodeSolver.is_for_each_node(SimpleNamespace(for_each=[1]))
    assert NodeSolver.is_timeout_node(SimpleNamespace(timeout=10))


# ---------------------------------------------------------------------------
# PipelineValidationManager
# ---------------------------------------------------------------------------


def test_validation_manager_checks(make_context, sample_metadata):
    vm = PipelineValidationManager()

    with pytest.raises(InvalidPipelineStateException):
        vm.validate_state(None)  # type: ignore[arg-type]

    empty_ctx = make_context(cached_metadata=[], decomposed_tasks=["t"])
    with pytest.raises(InvalidPipelineStateException):
        vm.validate_state(empty_ctx)

    bad_loop = make_context(cached_metadata=sample_metadata, decomposed_tasks=[{"iterations": -1}])
    with pytest.raises(InvalidPipelineStateException):
        vm.validate_state(bad_loop)

    valid_ctx = make_context(cached_metadata=sample_metadata, decomposed_tasks=["send alert"])
    vm.validate_state(valid_ctx)


# ---------------------------------------------------------------------------
# StepExecuter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_step_executer_runs_and_records_breaker(make_context, sample_metadata):
    breaker = BreakCircusService(failure_threshold=2)
    executer = StepExecuter(circuit_breaker=breaker)
    executer.process_task = AsyncMock(
        return_value=(
            {"action_name": "send_alert", "plugin_name": "sys", "method": lambda message: f"ok:{message}", "args": {"message": "hi"}},
            {},
            np.zeros(8, dtype=np.float32),
        )
    )
    executer.persist_extraction = AsyncMock()

    ctx = make_context(cached_metadata=sample_metadata, decomposed_tasks=["send alert"])
    res = await executer._execute_task_node("send alert", ctx)
    assert res is not None
    assert res["result"] == "ok:hi"
    assert ctx.environment_state["_last_result"] == "ok:hi"
    assert breaker.count_errors == 0
