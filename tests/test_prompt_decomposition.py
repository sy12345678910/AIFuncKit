from __future__ import annotations

from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from aifunckit._internal.orchestrator.services.prompt_decomposition import PromptDecompositionService
from aifunckit._internal.schemas.ai_schemas import ConditionalTask, LoopTask, TaskList
from aifunckit._internal.schemas.memory_schemas import MemoryRecord
from aifunckit._internal.schemas.orchestrator_schemas import MetadataForLLM


@pytest.fixture()
def decomposer():
    return PromptDecompositionService()


@pytest.mark.asyncio
async def test_decompose_parses_task_list_object(decomposer, make_context, sample_metadata):
    ctx = make_context(user_prompt="send alert and check stats", cached_metadata=sample_metadata)
    task_list = TaskList(tasks=["send alert", "get stats"])

    with patch(
        "aifunckit._internal.orchestrator.services.prompt_decomposition.get_data_generated_by_model",
        new=AsyncMock(return_value=task_list),
    ):
        emb = await decomposer.decompose(ctx)

    assert isinstance(emb, np.ndarray)
    assert ctx.decomposed_tasks == ["send alert", "get stats"]


@pytest.mark.asyncio
async def test_decompose_parses_json_string(decomposer, make_context, sample_metadata):
    ctx = make_context(user_prompt="run", cached_metadata=sample_metadata)
    raw_json = '{"tasks": ["step one", "step two"]}'

    with patch(
        "aifunckit._internal.orchestrator.services.prompt_decomposition.get_data_generated_by_model",
        new=AsyncMock(return_value=raw_json),
    ):
        await decomposer.decompose(ctx)

    assert ctx.decomposed_tasks == ["step one", "step two"]


@pytest.mark.asyncio
async def test_decompose_handles_llm_exception_gracefully(decomposer, make_context, sample_metadata):
    ctx = make_context(user_prompt="fail", cached_metadata=sample_metadata)

    with patch(
        "aifunckit._internal.orchestrator.services.prompt_decomposition.get_data_generated_by_model",
        new=AsyncMock(side_effect=RuntimeError("llm down")),
    ):
        emb = await decomposer.decompose(ctx)

    assert isinstance(emb, np.ndarray)
    assert ctx.decomposed_tasks == []


@pytest.mark.asyncio
async def test_decompose_handles_invalid_json_gracefully(decomposer, make_context, sample_metadata):
    ctx = make_context(user_prompt="bad json", cached_metadata=sample_metadata)

    with patch(
        "aifunckit._internal.orchestrator.services.prompt_decomposition.get_data_generated_by_model",
        new=AsyncMock(return_value="not valid json"),
    ):
        await decomposer.decompose(ctx)

    assert ctx.decomposed_tasks == []


@pytest.mark.asyncio
async def test_decompose_serializes_nested_task_models(decomposer, make_context, sample_metadata):
    ctx = make_context(user_prompt="complex", cached_metadata=sample_metadata)
    task_list = TaskList(
        tasks=[
            ConditionalTask(condition_variable="sys.ok", if_true=["a"], if_false=["b"]),
            LoopTask(task="ping", repetitions=2),
        ]
    )

    with patch(
        "aifunckit._internal.orchestrator.services.prompt_decomposition.get_data_generated_by_model",
        new=AsyncMock(return_value=task_list),
    ):
        await decomposer.decompose(ctx)

    assert ctx.decomposed_tasks[0] == {
        "condition_variable": "sys.ok",
        "if_true": ["a"],
        "if_false": ["b"],
    }
    assert ctx.decomposed_tasks[1] == {"task": "ping", "repetitions": 2}


@pytest.mark.asyncio
async def test_prepare_memory_block_formats_record(decomposer, make_context):
    ctx = make_context()
    ctx.memory.search_relevant_example = AsyncMock(
        return_value=MemoryRecord(
            user_prompt="clean up",
            context_snapshot={},
            extracted_json='{"tasks": ["delete tmp"]}',
        )
    )
    block = await decomposer._prepare_memory_block(np.zeros(4, dtype=np.float32), ctx)
    assert "PAST DECOMPOSITION EXPERIENCE" in block
    assert "clean up" in block
    assert "delete tmp" in block


@pytest.mark.asyncio
async def test_prepare_memory_block_handles_missing_and_error(decomposer, make_context):
    ctx = make_context()
    ctx.memory.search_relevant_example = AsyncMock(return_value=None)
    assert await decomposer._prepare_memory_block(np.zeros(4, dtype=np.float32), ctx) == ""

    ctx.memory.search_relevant_example = AsyncMock(side_effect=RuntimeError("db error"))
    assert await decomposer._prepare_memory_block(np.zeros(4, dtype=np.float32), ctx) == ""


def test_prepare_capabilities_block_lists_plugins_and_keywords(decomposer):
    meta = MetadataForLLM(
        cached_metadata=[
            {"plugin_name": "sys", "action_name": "disk", "keywords": ["storage", "gb"]},
            {"plugin_name": "net", "action_name": "ping", "keywords": []},
        ],
        context_block={},
    )
    block = decomposer._prepare_capabilities_block(meta)
    assert "Plugin: sys" in block
    assert "disk (keywords: storage, gb)" in block
    assert "Plugin: net" in block


def test_set_context_block_serializes_state(decomposer):
    block = decomposer._set_context_block({"cpu": 42})
    assert "CURRENT GLOBAL ENVIRONMENT STATE" in block
    assert '"cpu": 42' in block
