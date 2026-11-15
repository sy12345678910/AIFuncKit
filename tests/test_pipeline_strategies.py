from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
MODEL_PROVIDE_ROOT = SRC_ROOT / "model_provide"
for entry in (ROOT, SRC_ROOT, MODEL_PROVIDE_ROOT):
    path = str(entry)
    if path not in sys.path:
        sys.path.insert(0, path)

from schemas.pipeline_context import PipelineContext
from schemas.orchestrator_schemas import LLMConfig, MetadataForLLM, ParamsForExecution
from orchestrator.pipelines.sequential import SequentialPipeline
from orchestrator.pipelines.parallel import ParallelPipeline
from orchestrator.pipelines.conditional import ConditionalPipeline
from orchestrator.pipelines.loop import LoopPipeline
from orchestrator.pipelines.timeout import TimeoutPipeline
from orchestrator.pipelines.for_each_loop import ForEachLoopPipeline
from exceptions.pipeline_exceptions import InvalidPipelineStateException
import pytest


def _build_context() -> PipelineContext:
    return PipelineContext(
        user_prompt="demo prompt",
        llm_config=LLMConfig(),
        metadata=MetadataForLLM(cached_metadata=[], context_block={}),
        memory=MagicMock(),
        embedding=MagicMock(),
        environment_state={},
        decomposed_tasks=[],
        execution_results=[],
        params_for_execution=ParamsForExecution(async_loop=False),
    )


@pytest.mark.asyncio
async def test_pipeline_type_and_strategy_execution() -> None:
    ctx = _build_context()
    ctx.decomposed_tasks = ["first", "second"]

    seq = SequentialPipeline()
    assert seq.pipeline_type == "sequential"
    seq.process_task = AsyncMock(return_value=(
        {"method": "demo", "action_name": "demo", "plugin_name": "demo", "args": {}},
        {},
        MagicMock(),
    ))
    seq.persist_extraction = AsyncMock()
    await seq.execute(ctx)
    assert len(ctx.execution_results) == 2

    par = ParallelPipeline()
    assert par.pipeline_type == "parallel"
    par.process_task = AsyncMock(return_value=(
        {"method": "demo", "action_name": "demo", "plugin_name": "demo", "args": {}},
        {},
        MagicMock(),
    ))
    par.persist_extraction = AsyncMock()
    ctx.execution_results = []
    ctx.decomposed_tasks = ["one", "two"]
    await par.execute(ctx)
    assert len(ctx.execution_results) == 2

    cond = ConditionalPipeline()
    assert cond.pipeline_type == "conditional"
    cond._run_sequential_branch = AsyncMock()
    ctx.decomposed_tasks = [{"condition_variable": "demo.enabled", "if_true": ["yes"], "if_false": ["no"]}]
    ctx.environment_state = {"demo": {"enabled": True}}
    await cond.execute(ctx)
    cond._run_sequential_branch.assert_awaited_once_with(["yes"], ctx)

    loop = LoopPipeline()
    assert loop.pipeline_type == "loop"
    loop.run_nodes = AsyncMock()
    ctx.decomposed_tasks = [{"loop_variable": "demo.counter", "iterations": 2, "body": ["task"]}]
    await loop.execute(ctx)
    loop.run_nodes.assert_awaited_once()

    timeout = TimeoutPipeline()
    assert timeout.pipeline_type == "timeout"
    timeout.execute_with_timeout = AsyncMock()
    ctx.decomposed_tasks = [{"timeout": 1.5, "task": "slow"}]
    await timeout.execute(ctx)
    timeout.execute_with_timeout.assert_awaited_once()

    for_each = ForEachLoopPipeline()
    assert for_each.pipeline_type == "for_each_loop"
    for_each.run_nodes = AsyncMock()
    ctx.decomposed_tasks = [{"items": ["a", "b"], "task": "process {item}"}]
    await for_each.execute(ctx)
    for_each.run_nodes.assert_awaited_once()


def test_pipeline_state_validation() -> None:
    ctx = _build_context()
    seq = SequentialPipeline()

    # 1. Fails if no plugins in cached_metadata
    ctx.decomposed_tasks = ["run backup"]
    with pytest.raises(InvalidPipelineStateException, match="No registered plugins"):
        seq.validate_state(ctx)

    # 2. Fails if no tasks to execute
    ctx.metadata.cached_metadata = [{"action_name": "backup", "name": "backup"}]
    ctx.decomposed_tasks = []
    with pytest.raises(InvalidPipelineStateException, match="No tasks provided"):
        seq.validate_state(ctx)

    # 3. Fails if conditional task is missing condition_variable
    ctx.decomposed_tasks = [{"condition_variable": "", "if_true": ["yes"]}]
    with pytest.raises(InvalidPipelineStateException, match="missing required 'condition_variable'"):
        seq.validate_state(ctx)

    # 4. Fails if loop task has negative repetitions
    ctx.decomposed_tasks = [{"loop_variable": "test", "repetitions": -1}]
    with pytest.raises(InvalidPipelineStateException, match="repetitions must be non-negative"):
        seq.validate_state(ctx)

    # 5. Succeeds when valid
    ctx.decomposed_tasks = ["run backup"]
    seq.validate_state(ctx)


@pytest.mark.asyncio
async def test_for_each_pipeline_iteration_and_variable_substitution() -> None:
    ctx = _build_context()
    ctx.metadata.cached_metadata = [{"action_name": "process", "name": "process"}]
    ctx.environment_state = {"file_list": ["doc.txt", "img.png"]}

    pipeline = ForEachLoopPipeline()
    executed_tasks = []

    async def mock_execute(self_seq, c):
        executed_tasks.extend(c.decomposed_tasks)
        assert c.environment_state.get("item") in ["doc.txt", "img.png"]

    from unittest.mock import patch
    with patch("orchestrator.pipelines.sequential.SequentialPipeline.execute", new=mock_execute):
        node = {"items": "file_list", "task": "process {item}", "item_variable": "item"}
        await pipeline.execute_for_node(node, ctx)

    assert executed_tasks == ["process doc.txt", "process img.png"]


