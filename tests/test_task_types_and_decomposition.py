from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from config import TaskList, ConditionalTask, LoopTask, TimeoutTask
from orchestrator.pipelines.conditional import ConditionalPipeline
from orchestrator.pipelines.loop import LoopPipeline
from orchestrator.pipelines.timeout import TimeoutPipeline
from orchestrator.topology_resolver import TopologyResolver
from schemas.orchestrator_schemas import LLMConfig, MetadataForLLM, ParamsForExecution
from schemas.pipeline_context import PipelineContext


@pytest.fixture
def pipeline_context() -> PipelineContext:
    return PipelineContext(
        user_prompt="test prompt",
        llm_config=LLMConfig(provider="ollama", model_name="dummy", context_window=384),
        metadata=MetadataForLLM(cached_metadata=[], context_block={}),
        memory=MagicMock(),
        embedding=MagicMock(),
        environment_state={"demo": {"enabled": True, "count": 2}},
        decomposed_tasks=[],
        execution_results=[],
        params_for_execution=ParamsForExecution(async_loop=False),
    )


def test_topology_resolver_selects_correct_pipeline_by_task_type() -> None:
    assert TopologyResolver.resolve(SimpleNamespace(decomposed_tasks=["sync task"])) is not None

    conditional_tasks = [{"condition_variable": "demo.enabled", "if_true": ["yes"], "if_false": ["no"]}]
    assert TopologyResolver.resolve(SimpleNamespace(decomposed_tasks=conditional_tasks)).__name__ == "ConditionalPipeline"

    loop_tasks = [{"loop_variable": "demo.count", "iterations": 2, "body": ["repeat"]}]
    assert TopologyResolver.resolve(SimpleNamespace(decomposed_tasks=loop_tasks)).__name__ == "LoopPipeline"

    timeout_tasks = [{"timeout": 0.5, "task": "slow task"}]
    assert TopologyResolver.resolve(SimpleNamespace(decomposed_tasks=timeout_tasks)).__name__ == "TimeoutPipeline"


@pytest.mark.asyncio
async def test_conditional_pipeline_executes_only_matching_branch(pipeline_context: PipelineContext) -> None:
    pipeline = ConditionalPipeline()
    pipeline._run_sequential_branch = AsyncMock()

    node = {"condition_variable": "demo.enabled", "if_true": ["allowed"], "if_false": ["blocked"]}
    await pipeline._execute_conditional_node(node, pipeline_context)

    pipeline._run_sequential_branch.assert_awaited_once_with(["allowed"], pipeline_context)


@pytest.mark.asyncio
async def test_loop_pipeline_repeats_body_for_requested_iterations(pipeline_context: PipelineContext) -> None:
    pipeline = LoopPipeline()
    pipeline.strategy.execute = AsyncMock()

    node = {"loop_variable": "demo.count", "iterations": 3, "body": ["task_a"]}
    with patch("orchestrator.pipelines.sequential.SequentialPipeline.execute", AsyncMock()) as mocked_execute:
        await pipeline.execute_for_node(node, pipeline_context)

    assert mocked_execute.await_count == 3
    assert pipeline_context.decomposed_tasks == []


@pytest.mark.asyncio
async def test_timeout_pipeline_marks_task_as_timed_out(pipeline_context: PipelineContext) -> None:
    async def slow_task() -> None:
        await __import__("asyncio").sleep(0.1)
        return None

    pipeline = TimeoutPipeline()
    pipeline._execute_task_node = AsyncMock(side_effect=slow_task)

    node = {"timeout": 0.01, "task": "slow task"}
    await pipeline.execute_for_node(node, pipeline_context)

    assert pipeline_context.execution_results[-1]["result"]["status"] == "timed_out"


@pytest.mark.asyncio
async def test_prompt_decomposition_accepts_schema_like_task_list_and_keeps_order() -> None:
    ctx = PipelineContext(
        user_prompt="if demo.enabled then check status else retry, and repeat the update 2 times",
        llm_config=LLMConfig(provider="ollama", model_name="dummy", context_window=512),
        metadata=MetadataForLLM(cached_metadata=[{"plugin_name": "demo", "action_name": "status", "keywords": ["status"]}], context_block={}),
        memory=MagicMock(),
        embedding=MagicMock(),
        environment_state={"demo": {"enabled": True}},
        decomposed_tasks=[],
        execution_results=[],
        params_for_execution=ParamsForExecution(async_loop=False),
    )
    ctx.embedding._encode_text = AsyncMock(return_value=np.array([1.0, 0.0], dtype=np.float32))
    ctx.memory.search_relevant_example = AsyncMock(return_value=None)

    payload = TaskList(
        tasks=[
            "check status",
            ConditionalTask(condition_variable="demo.enabled", if_true=["report status"], if_false=["retry"]),
            LoopTask(task="update status", repetitions=2),
            TimeoutTask(task="sync data", timeout=1.5),
        ]
    )

    with patch("orchestrator.services.prompt_decomposition.get_data_generated_by_model", AsyncMock(return_value=payload)):
        output = await __import__("orchestrator.services.prompt_decomposition", fromlist=["PromptDecompositionService"]).PromptDecompositionService().decompose(ctx)

    assert len(ctx.decomposed_tasks) == 4
    assert ctx.decomposed_tasks[0] == "check status"
    assert ctx.decomposed_tasks[1]["condition_variable"] == "demo.enabled"
    assert ctx.decomposed_tasks[2]["repetitions"] == 2
    assert ctx.decomposed_tasks[3]["timeout"] == 1.5
    assert output.shape == (2,)
