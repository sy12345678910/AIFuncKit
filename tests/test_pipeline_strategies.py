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

from src.schemas.pipeline_context import PipelineContext
from src.schemas.orchestrator_schemas import LLMConfig, MetadataForLLM, ParamsForExecution
from src.orchestrator.pipelines.sequential import SequentialPipeline
from src.orchestrator.pipelines.parallel import ParallelPipeline
from src.orchestrator.pipelines.conditional import ConditionalPipeline
from src.orchestrator.pipelines.loop import LoopPipeline
from src.orchestrator.pipelines.timeout import TimeoutPipeline


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
