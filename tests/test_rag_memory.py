from __future__ import annotations

import numpy as np
import pytest

from aifunckit._internal.memory.rag import AIRAGMemoryManager
from aifunckit._internal.memory.search_engine import VectorSearchEngine
from aifunckit._internal.memory.vector import compute_best_match, normalize_vector
from aifunckit._internal.schemas.memory_schemas import RecordData


def test_normalize_vector_unit_and_zero():
    v = np.array([3.0, 4.0], dtype=np.float32)
    normed = normalize_vector(v)
    assert np.isclose(np.linalg.norm(normed), 1.0)

    zero = np.array([0.0, 0.0], dtype=np.float32)
    zero_normed = normalize_vector(zero)
    assert np.allclose(zero_normed, zero)


def test_compute_best_match_threshold():
    matrix = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    assert compute_best_match(matrix, np.array([0.9, 0.1], dtype=np.float32), threshold=0.5) == 0
    assert compute_best_match(matrix, np.array([0.1, 0.1], dtype=np.float32), threshold=0.5) is None


def test_vector_search_engine_load_and_find():
    engine = VectorSearchEngine()
    vecs = [
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
    ]
    engine.load_initial_cache(ids=[10, 20], layer="decomposition", vectors=vecs)
    match = engine.find_best_match_id(
        "decomposition", np.array([0.95, 0.05, 0.0], dtype=np.float32), threshold=0.5
    )
    assert match == 10


def test_vector_search_engine_add_vector_incremental():
    engine = VectorSearchEngine()
    engine.add_vector_to_cache("extraction", 1, np.array([1.0, 0.0], dtype=np.float32))
    engine.add_vector_to_cache("extraction", 2, np.array([0.0, 1.0], dtype=np.float32))

    assert engine.find_best_match_id("extraction", np.array([0.0, 1.0], dtype=np.float32)) == 2


def test_vector_search_engine_empty_returns_none():
    engine = VectorSearchEngine()
    assert engine.find_best_match_id("decomposition", np.array([1.0, 0.0], dtype=np.float32)) is None


@pytest.mark.asyncio
async def test_rag_memory_initialize_creates_db(tmp_path):
    db_file = str(tmp_path / "rag_init.db")
    mgr = AIRAGMemoryManager(db_path=db_file)
    await mgr.initialize()
    assert (tmp_path / "rag_init.db").exists()
    await mgr.close()


@pytest.mark.asyncio
async def test_rag_memory_add_and_search_roundtrip(tmp_path):
    db_file = str(tmp_path / "rag_roundtrip.db")
    mgr = AIRAGMemoryManager(db_path=db_file)
    await mgr.initialize()

    vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    await mgr.add_record(
        RecordData(
            layer="extraction",
            user_prompt="send alert",
            context_dict={"env": "prod"},
            args={"msg": "hi"},
            raw_vector=vec,
        )
    )

    found = await mgr.search_relevant_example("extraction", vec)
    assert found is not None
    assert found.user_prompt == "send alert"
    await mgr.close()


@pytest.mark.asyncio
async def test_rag_memory_search_empty_returns_none(tmp_path):
    db_file = str(tmp_path / "rag_empty.db")
    mgr = AIRAGMemoryManager(db_path=db_file)
    await mgr.initialize()

    found = await mgr.search_relevant_example("decomposition", np.ones(4, dtype=np.float32))
    assert found is None
    await mgr.close()


@pytest.mark.asyncio
async def test_rag_memory_persists_across_reopen(tmp_path):
    db_file = str(tmp_path / "rag_persist.db")
    mgr1 = AIRAGMemoryManager(db_path=db_file)
    await mgr1.init_and_load_cache()
    vec = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    await mgr1.add_record(
        RecordData(
            layer="decomposition",
            user_prompt="check system",
            context_dict={},
            args={"tasks": ["check cpu"]},
            raw_vector=vec,
        )
    )
    await mgr1.close()

    mgr2 = AIRAGMemoryManager(db_path=db_file)
    await mgr2.init_and_load_cache()
    found = await mgr2.search_relevant_example("decomposition", vec)
    assert found is not None
    assert found.user_prompt == "check system"
    await mgr2.close()


@pytest.mark.asyncio
async def test_rag_memory_close_is_safe(tmp_path):
    db_file = str(tmp_path / "rag_close.db")
    mgr = AIRAGMemoryManager(db_path=db_file)
    await mgr.initialize()
    await mgr.close()
    assert mgr.repository.db is None
