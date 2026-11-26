from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import numpy as np
import pytest

from aifunckit._internal.orchestrator.pipelines.conditional import ConditionalPipeline
from aifunckit._internal.orchestrator.pipelines.for_each_loop import ForEachLoopPipeline
from aifunckit._internal.orchestrator.pipelines.loop import LoopPipeline
from aifunckit._internal.orchestrator.pipelines.parallel import ParallelPipeline
from aifunckit._internal.orchestrator.pipelines.sequential import SequentialPipeline
from aifunckit._internal.orchestrator.pipelines.timeout import TimeoutPipeline
from aifunckit._internal.orchestrator.services.break_circus import BreakCircusService
from aifunckit.exceptions.pipeline_exceptions import InvaildPipelineStateException


def _stub_process_task(pipeline, results_log=None):
    async def _proc(task, context, plugin_meta=None):
        if results_log is not None:
            results_log.append(str(task))
        fn = lambda: f"done:{task}"
        return (
            {"action_name": str(task), "plugin_name": "test", "method": fn, "args": {}},
            {},
            np.zeros(8, dtype=np.float32),
        )

    pipeline.step_executer.process_task = AsyncMock(side_effect=_proc)
    pipeline.step_executer.persist_extraction = AsyncMock()


@pytest.mark.asyncio
async def test_sequential_pipeline_executes_in_order(make_context, sample_metadata):
    order = []
    pipe = SequentialPipeline()
    _stub_process_task(pipe, order)
    ctx = make_context(cached_metadata=sample_metadata, decomposed_tasks=["step1", "step2", "step3"])

    await pipe.execute(ctx)
    assert order == ["step1", "step2", "step3"]
    assert len(ctx.execution_results) == 3


@pytest.mark.asyncio
async def test_sequential_pipeline_records_failure_and_continues(make_context, sample_metadata):
    breaker = BreakCircusService(failure_threshold=5)
    pipe = SequentialPipeline(circuit_breaker=breaker)
    pipe.step_executer.persist_extraction = AsyncMock()

    async def _proc(task, context, plugin_meta=None):
        if task == "bad":
            return None
        return (
            {"action_name": "good", "plugin_name": "test", "method": lambda: "ok", "args": {}},
            {},
            np.zeros(8, dtype=np.float32),
        )

    pipe.step_executer.process_task = AsyncMock(side_effect=_proc)
    ctx = make_context(cached_metadata=sample_metadata, decomposed_tasks=["bad", "good"])

    await pipe.execute(ctx)
    assert len(ctx.execution_results) == 1
    assert breaker.count_errors == 0  # reset after 'good' succeeded


@pytest.mark.asyncio
async def test_parallel_pipeline_executes_all_tasks(make_context, sample_metadata):
    pipe = ParallelPipeline()
    _stub_process_task(pipe)
    ctx = make_context(cached_metadata=sample_metadata, decomposed_tasks=["p1", "p2", "p3"])

    await pipe.execute(ctx)
    assert len(ctx.execution_results) == 3


@pytest.mark.asyncio
async def test_parallel_pipeline_isolates_task_exceptions(make_context, sample_metadata):
    pipe = ParallelPipeline()
    pipe.step_executer.persist_extraction = AsyncMock()

    async def _proc(task, context, plugin_meta=None):
        if task == "boom":
            raise RuntimeError("boom")
        return (
            {"action_name": str(task), "plugin_name": "test", "method": lambda: "ok", "args": {}},
            {},
            np.zeros(8, dtype=np.float32),
        )

    pipe.step_executer.process_task = AsyncMock(side_effect=_proc)
    ctx = make_context(cached_metadata=sample_metadata, decomposed_tasks=["boom", "safe"])

    await pipe.execute(ctx)
    assert len(ctx.execution_results) == 1
    assert ctx.execution_results[0]["action_name"] == "safe"


@pytest.mark.asyncio
async def test_conditional_pipeline_selects_if_true_branch(make_context, sample_metadata):
    pipe = ConditionalPipeline()
    ctx = make_context(
        cached_metadata=sample_metadata,
        environment_state={"is_ready": True},
        decomposed_tasks=[{"condition_variable": "is_ready", "if_true": ["go"], "if_false": ["stop"]}],
    )

    executed = []

    async def _run_branch(tasks, context):
        executed.extend(tasks)

    pipe._run_sequential_branch = AsyncMock(side_effect=_run_branch)
    await pipe.execute(ctx)
    assert executed == ["go"]


@pytest.mark.asyncio
async def test_conditional_pipeline_selects_if_false_with_dotted_var(make_context, sample_metadata):
    pipe = ConditionalPipeline()
    ctx = make_context(
        cached_metadata=sample_metadata,
        environment_state={"sys": {"ok": False}},
        decomposed_tasks=[{"condition_variable": "sys.ok", "if_true": ["go"], "if_false": ["fallback"]}],
    )

    executed = []

    async def _run_branch(tasks, context):
        executed.extend(tasks)

    pipe._run_sequential_branch = AsyncMock(side_effect=_run_branch)
    await pipe.execute(ctx)
    assert executed == ["fallback"]


@pytest.mark.asyncio
async def test_conditional_pipeline_missing_variable_raises(make_context, sample_metadata):
    pipe = ConditionalPipeline()
    ctx = make_context(
        cached_metadata=sample_metadata,
        environment_state={},
        decomposed_tasks=[{"condition_variable": "missing.key", "if_true": ["go"], "if_false": []}],
    )

    with pytest.raises(InvaildPipelineStateException):
        await pipe.execute(ctx)


@pytest.mark.asyncio
async def test_loop_pipeline_repeats_iterations_and_restores_tasks(make_context, sample_metadata):
    pipe = LoopPipeline()
    original = [{"task": "ping", "iterations": 3}]
    ctx = make_context(cached_metadata=sample_metadata, decomposed_tasks=list(original))

    calls = 0

    async def _fake_exec(self, context):
        nonlocal calls
        calls += 1

    from aifunckit._internal.orchestrator.pipelines.sequential import SequentialPipeline
    orig_exec = SequentialPipeline.execute
    SequentialPipeline.execute = _fake_exec
    try:
        await pipe.execute(ctx)
    finally:
        SequentialPipeline.execute = orig_exec

    assert calls == 3
    assert ctx.decomposed_tasks == original


@pytest.mark.asyncio
async def test_for_each_loop_substitutes_item_and_restores_env(make_context, sample_metadata):
    pipe = ForEachLoopPipeline()
    ctx = make_context(
        cached_metadata=sample_metadata,
        environment_state={"item": "old"},
        decomposed_tasks=[{"items": ["a", "b"], "task": "process {item}"}],
    )

    seen_tasks = []

    async def _fake_exec(self, context):
        seen_tasks.extend(context.decomposed_tasks)

    from aifunckit._internal.orchestrator.pipelines.sequential import SequentialPipeline
    orig_exec = SequentialPipeline.execute
    SequentialPipeline.execute = _fake_exec
    try:
        await pipe.execute(ctx)
    finally:
        SequentialPipeline.execute = orig_exec

    assert seen_tasks == ["process a", "process b"]
    assert ctx.environment_state["item"] == "old"


@pytest.mark.asyncio
async def test_for_each_loop_resolves_dotted_env_collection(make_context, sample_metadata):
    pipe = ForEachLoopPipeline()
    ctx = make_context(
        cached_metadata=sample_metadata,
        environment_state={"files": {"logs": ["a.log", "b.log"]}},
        decomposed_tasks=[{"items": "files.logs", "task": "read {item}", "item_variable": "f"}],
    )

    seen = []

    async def _fake_exec(self, context):
        seen.append(context.environment_state["f"])

    from aifunckit._internal.orchestrator.pipelines.sequential import SequentialPipeline
    orig_exec = SequentialPipeline.execute
    SequentialPipeline.execute = _fake_exec
    try:
        await pipe.execute(ctx)
    finally:
        SequentialPipeline.execute = orig_exec

    assert seen == ["a.log", "b.log"]
    assert "f" not in ctx.environment_state


@pytest.mark.asyncio
async def test_timeout_pipeline_completes_fast_task(make_context, sample_metadata):
    pipe = TimeoutPipeline()
    _stub_process_task(pipe)
    ctx = make_context(
        cached_metadata=sample_metadata,
        decomposed_tasks=[{"task": "quick", "timeout": 1.0}],
    )

    await pipe.execute(ctx)
    assert len(ctx.execution_results) == 1
    assert ctx.execution_results[0]["result"] == "done:quick"


@pytest.mark.asyncio
async def test_timeout_pipeline_records_timeout_result(make_context, sample_metadata):
    pipe = TimeoutPipeline()

    async def _slow(task, context, plugin_meta=None):
        await asyncio.sleep(0.2)
        return None

    pipe.step_executer._execute_task_node = AsyncMock(side_effect=_slow)
    ctx = make_context(
        cached_metadata=sample_metadata,
        decomposed_tasks=[{"task": "slow", "timeout": 0.01}],
    )

    await pipe.execute(ctx)
    assert len(ctx.execution_results) == 1
    assert ctx.execution_results[0]["result"]["status"] == "timed_out"
