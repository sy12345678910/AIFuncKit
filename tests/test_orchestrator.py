from __future__ import annotations

import builtins
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import pytest_asyncio

ROOT = Path(__file__).resolve().parents[1] / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import orchestrator as orch_mod
from config import TaskList
from memory import AIRAGMemory


@pytest_asyncio.fixture
async def orchestrator(tmp_path: Path):
    db_path = tmp_path / "orch.db"
    memory = AIRAGMemory(db_path=str(db_path))
    await memory.init_and_load_cache()

    instance = orch_mod.AIOrchestrator(
        cached_metadata=[],
        db_path=str(db_path),
        emb_model="dummy-model",
        routing_model="text",
    )
    instance.runtime_memory = memory
    yield instance
    await memory.close()


@pytest.mark.asyncio
async def test_break_prompt_to_tasks_accepts_json_string_and_pydantic_object(orchestrator: orch_mod.AIOrchestrator) -> None:
    orchestrator._encode_text = AsyncMock(return_value=np.array([1.0, 0.0], dtype=np.float32))
    orchestrator._prepare_memory_block = AsyncMock(return_value="")
    orchestrator._prepare_capabilities_block = MagicMock(return_value="CAPS")

    with patch("orchestrator.get_data_generated_by_model", AsyncMock(return_value='{"tasks":["task a", "task b"]}')) as mock_llm:
        tasks, emb = await orchestrator.break_prompt_to_tasks("do something", {"device": {"online": True}})
        assert tasks == ["task a", "task b"]
        assert emb.shape == (2,)
        mock_llm.assert_awaited_once()

    with patch("orchestrator.get_data_generated_by_model", AsyncMock(return_value=TaskList(tasks=["task c", "task d"]))) as mock_llm:
        tasks, emb = await orchestrator.break_prompt_to_tasks("do more", {"device": {"online": True}})
        assert tasks == ["task c", "task d"]
        assert emb.shape == (2,)
        mock_llm.assert_awaited_once()


@pytest.mark.asyncio
async def test_break_prompt_to_tasks_handles_llm_failure_gracefully(orchestrator: orch_mod.AIOrchestrator) -> None:
    orchestrator._encode_text = AsyncMock(return_value=np.array([1.0, 0.0], dtype=np.float32))
    orchestrator._prepare_memory_block = AsyncMock(return_value="")
    orchestrator._prepare_capabilities_block = MagicMock(return_value="CAPS")

    with patch("orchestrator.get_data_generated_by_model", AsyncMock(side_effect=RuntimeError("model down"))):
        tasks, emb = await orchestrator.break_prompt_to_tasks("fail", {})

    assert tasks == []
    assert emb.shape == (2,)


@pytest.mark.asyncio
async def test_referal_ai_to_correct_plugin_text_mode_matches_action_name_and_keywords(orchestrator: orch_mod.AIOrchestrator) -> None:
    orchestrator.routing_model = "text"
    orchestrator.cached_metadata = [
        {"action_name": "send_email", "name": "send_email", "keywords": ["notify"]},
        {"action_name": "read_file", "name": "read_file", "keywords": ["scan"]},
    ]

    meta = await orchestrator.referal_ai_to_correct_plugin("send email to admin")
    assert meta["action_name"] == "send_email"

    keyword_match = await orchestrator.referal_ai_to_correct_plugin("please scan the file")
    assert keyword_match["action_name"] == "read_file"


@pytest.mark.asyncio
async def test_referal_ai_to_correct_plugin_vector_mode_uses_to_thread_for_large_metadata(orchestrator: orch_mod.AIOrchestrator) -> None:
    orchestrator.routing_model = "vector"
    orchestrator.cached_metadata = [
        {"action_name": f"action_{idx}", "name": f"action_{idx}", "description": "general action"}
        for idx in range(101)
    ]
    orchestrator.cached_metadata[50]["description_embedding"] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    for meta in orchestrator.cached_metadata:
        if meta is not orchestrator.cached_metadata[50]:
            meta["description_embedding"] = np.array([0.0, 0.5, 0.0], dtype=np.float32)

    orchestrator._encode_text = AsyncMock(return_value=np.array([1.0, 0.0, 0.0], dtype=np.float32))
    orchestrator._ensure_description_embeddings = AsyncMock()

    async def fake_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    with patch("orchestrator.asyncio.to_thread", new=AsyncMock(side_effect=fake_to_thread)) as mock_thread:
        matched = await orchestrator.referal_ai_to_correct_plugin("choose action_50")

    assert matched["action_name"] == "action_50"
    assert mock_thread.await_count == 1


@pytest.mark.asyncio
async def test_referal_ai_to_correct_plugin_blm_mode_raises_when_laya_missing(orchestrator: orch_mod.AIOrchestrator) -> None:
    orchestrator.routing_model = "blm"
    orchestrator.cached_metadata = [{"action_name": "email", "name": "email", "keywords": ["mail"]}]

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "laya":
            raise ImportError("No module named laya")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=fake_import):
        with pytest.raises(ImportError, match="Laya"):
            await orchestrator.referal_ai_to_correct_plugin("send email")


@pytest.mark.asyncio
async def test_extract_args_builds_json_and_falls_back_to_empty_dict(orchestrator: orch_mod.AIOrchestrator) -> None:
    orchestrator._encode_text = AsyncMock(return_value=np.array([1.0, 0.0], dtype=np.float32))

    with patch("orchestrator.get_data_generated_by_model", AsyncMock(return_value='{"amount": 12, "enabled": true}')):
        result = await orchestrator.extract_args(
            "set amount 12",
            {"amount": int, "enabled": bool},
            {"plugin": {"enabled": True}},
            None,
        )
    assert result == {"amount": 12, "enabled": True}

    with patch("orchestrator.get_data_generated_by_model", AsyncMock(return_value='not-json')):
        result = await orchestrator.extract_args(
            "bad payload",
            {"amount": int},
            {},
            None,
        )
    assert result == {}


@pytest.mark.asyncio
async def test_run_pipeline_executes_tasks_and_persists_extraction_memory(orchestrator: orch_mod.AIOrchestrator) -> None:
    orchestrator.break_prompt_to_tasks = AsyncMock(return_value=(
        ["send email"],
        np.array([1.0, 0.0], dtype=np.float32),
    ))
    orchestrator.referal_ai_to_correct_plugin = AsyncMock(
        return_value={
            "plugin_name": "demo",
            "action_name": "send_email",
            "name": "send_email",
            "args": {"to": "a@b.com"},
        }
    )
    orchestrator._build_execution_pipeline = AsyncMock(
        return_value=[
            {
                "type": "parallel",
                "nodes": [
                    (
                        "send email",
                        {"plugin_name": "demo", "action_name": "send_email", "name": "send_email", "args": {"to": "a@b.com"}},
                    )
                ],
            }
        ]
    )

    async def fake_process_single_task(task, meta, system_context):
        return (
            {"method": "send_email", "action_name": "send_email", "plugin_name": "demo", "args": {"to": "a@b.com"}},
            {"demo": {"enabled": True}},
            np.array([1.0, 0.0], dtype=np.float32),
        )

    with patch.object(orchestrator, "_process_single_task", side_effect=fake_process_single_task):
        result = await orchestrator.run_pipeline("send email", {"demo": {"enabled": True}})

    assert len(result) == 1
    assert result[0]["action_name"] == "send_email"
    match = await orchestrator.runtime_memory.search_relevant_example("extraction", np.array([1.0, 0.0], dtype=np.float32))
    assert match is not None
    assert match[0] == "send email"


@pytest.mark.asyncio
async def test_build_execution_pipeline_groups_parallel_and_sequential_streams(orchestrator: orch_mod.AIOrchestrator) -> None:
    orchestrator.referal_ai_to_correct_plugin = AsyncMock(
        side_effect=[
            {"name": "task_a", "plugin_name": "demo", "execution_type": "parallel", "action_name": "task_a"},
            {"name": "task_b", "plugin_name": "demo", "execution_type": "blocking", "action_name": "task_b"},
        ]
    )

    pipeline = await orchestrator._build_execution_pipeline(["one", "two"])

    assert pipeline[0]["type"] == "parallel"
    assert pipeline[1]["type"] == "sequential"


@pytest.mark.asyncio
async def test_execute_parallel_stream_uses_gather_and_continues_after_exceptions(orchestrator: orch_mod.AIOrchestrator) -> None:
    results = []
    calls = []

    async def fake_process(task, meta, system_context):
        calls.append(task)
        if task == "bad":
            raise ValueError("broken")
        return (
            {"method": "ok", "action_name": "ok", "plugin_name": "demo", "args": {}},
            {"demo": {}},
            np.array([1.0, 0.0], dtype=np.float32),
        )

    with patch.object(orchestrator, "_process_single_task", side_effect=fake_process):
        await orchestrator._execute_parallel_stream(
            [("good", {"action_name": "good"}), ("bad", {"action_name": "bad"})],
            {"demo": {}},
            results,
        )

    assert len(results) == 1
    assert results[0]["action_name"] == "ok"
    assert "good" in calls


@pytest.mark.asyncio
async def test_execute_sequential_stream_continues_after_exception(orchestrator: orch_mod.AIOrchestrator) -> None:
    res = []

    async def fake_process(task, meta, system_context):
        if task == "bad":
            raise RuntimeError("nope")
        return (
            {"method": "ok", "action_name": "ok", "plugin_name": "demo", "args": {}},
            {"demo": {}},
            np.array([1.0, 0.0], dtype=np.float32),
        )

    with patch.object(orchestrator, "_process_single_task", side_effect=fake_process):
        await orchestrator._execute_sequential_stream(
            [("bad", {"action_name": "bad"}), ("good", {"action_name": "good"})],
            {"demo": {}},
            res,
        )

    assert len(res) == 1
    assert res[0]["action_name"] == "ok"


@pytest.mark.asyncio
async def test_execute_conditional_stream_evaluates_condition_and_dispatches_branch(orchestrator: orch_mod.AIOrchestrator) -> None:
    node = {"condition_variable": "device.enabled", "if_true": ["enabled task"], "if_false": ["disabled task"]}
    res = []

    with patch.object(orchestrator, "_build_execution_pipeline", AsyncMock(return_value=[{"type": "parallel", "nodes": [("enabled task", {"action_name": "enabled task"})]}])) as build_mock:
        await orchestrator._execute_conditional_stream(node, {"device": {"enabled": True}}, res)

    assert build_mock.await_count == 1
    assert res == []

    with patch.object(orchestrator, "_build_execution_pipeline", AsyncMock(return_value=[{"type": "parallel", "nodes": [("disabled task", {"action_name": "disabled task"})]}])) as build_mock_false:
        await orchestrator._execute_conditional_stream(node, {"device": {"enabled": False}}, res)

    assert build_mock_false.await_count == 1
