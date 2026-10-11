from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import aiosqlite
import numpy as np
import pytest
import pytest_asyncio

ROOT = Path(__file__).resolve().parents[1] / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memory import AIRAGMemory


@pytest_asyncio.fixture
async def memory_store(tmp_path: Path):
    store = AIRAGMemory(db_path=str(tmp_path / "rag_memory.db"))
    await store.init_and_load_cache()
    yield store
    await store.close()


@pytest.mark.asyncio
async def test_init_and_load_cache_creates_table_and_empty_caches(memory_store: AIRAGMemory) -> None:
    async with memory_store.db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='memory_history'"
    ) as cursor:
        row = await cursor.fetchone()

    assert row == ("memory_history",)
    assert memory_store.id_cache == {"decomposition": [], "extraction": []}
    assert memory_store.vector_list_cache == {"decomposition": [], "extraction": []}
    assert memory_store.compiled_matrix_cache == {"decomposition": None, "extraction": None}


@pytest.mark.asyncio
async def test_normalize_vector_returns_unit_length_and_handles_zero_vector(memory_store: AIRAGMemory) -> None:
    vector = np.array([3.0, 4.0], dtype=np.float32)
    normalized = memory_store._normalize_vector(vector)

    assert normalized.dtype == np.float32
    assert normalized.shape == (2,)
    assert np.isclose(float(np.linalg.norm(normalized)), 1.0, atol=1e-7)

    zero_vector = np.zeros(5, dtype=np.float32)
    zero_norm = memory_store._normalize_vector(zero_vector)
    assert zero_norm.dtype == np.float32
    assert np.allclose(zero_norm, zero_vector, atol=1e-8)


@pytest.mark.asyncio
async def test_add_record_increments_cache_and_builds_compiled_matrix(memory_store: AIRAGMemory) -> None:
    first_vector = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    second_vector = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    await memory_store.add_record(
        "decomposition",
        "first task",
        {"system": "demo"},
        {"plugin": "demo"},
        first_vector,
    )
    await memory_store.add_record(
        "decomposition",
        "second task",
        {"system": "demo"},
        {"plugin": "demo"},
        second_vector,
    )

    assert len(memory_store.id_cache["decomposition"]) == 2
    assert memory_store.compiled_matrix_cache["decomposition"].shape == (2, 3)
    assert memory_store.vector_list_cache["decomposition"][0].shape == (3,)


@pytest.mark.asyncio
async def test_add_record_propagates_sqlite_errors(memory_store: AIRAGMemory) -> None:
    class BrokenExecute:
        async def __aenter__(self):
            raise aiosqlite.DatabaseError("sqlite boom")

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            return False

    with patch.object(memory_store.db, "execute", return_value=BrokenExecute()):
        with pytest.raises(aiosqlite.DatabaseError):
            await memory_store.add_record(
                "extraction",
                "bad task",
                {"plugin": "x"},
                {"arg": 1},
                np.array([1.0, 1.0], dtype=np.float32),
            )


@pytest.mark.asyncio
async def test_search_relevant_example_returns_match_and_none_on_low_score(memory_store: AIRAGMemory) -> None:
    match_vector = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    wrong_vector = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    await memory_store.add_record(
        "decomposition",
        "match prompt",
        {"system": "demo"},
        {"status": "ok"},
        match_vector,
    )
    await memory_store.add_record(
        "decomposition",
        "non match prompt",
        {"system": "demo"},
        {"status": "bad"},
        wrong_vector,
    )

    result = await memory_store.search_relevant_example("decomposition", np.array([1.0, 0.0, 0.0]))
    assert result is not None
    assert result[0] == "match prompt"

    no_match = await memory_store.search_relevant_example("decomposition", np.array([0.0, 0.0, 1.0]))
    assert no_match is None


@pytest.mark.asyncio
async def test_archive_old_records_marks_rows_and_reloads_ram_cache(memory_store: AIRAGMemory) -> None:
    vector = np.array([0.5, 0.5, 0.0], dtype=np.float32)
    await memory_store.add_record(
        "extraction",
        "old prompt",
        {"system": "demo"},
        {"status": "old"},
        vector,
    )

    async with memory_store.db.execute(
        "UPDATE memory_history SET timestamp = datetime('now', '-30 days') WHERE id = 1"
    ):
        pass
    await memory_store.db.commit()

    await memory_store.archive_old_records(days_limit=7)

    async with memory_store.db.execute(
        "SELECT is_archived FROM memory_history WHERE id = 1"
    ) as cursor:
        archived = await cursor.fetchone()

    assert archived == (1,)
    assert len(memory_store.id_cache["extraction"]) == 0
    assert memory_store.compiled_matrix_cache["extraction"] is None
