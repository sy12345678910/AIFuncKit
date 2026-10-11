from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from memory import AIRAGMemory


def test_project_src_path_is_valid() -> None:
    assert SRC.exists(), f"Missing src directory: {SRC}"
    assert (SRC / "memory.py").exists(), "memory.py should live under src/"


@pytest.mark.asyncio
async def test_memory_smoke_init_and_normalize(tmp_path: Path) -> None:
    db_path = tmp_path / "smoke_memory.db"
    memory = AIRAGMemory(str(db_path))

    await memory.init_and_load_cache()
    vector = np.array([3.0, 4.0], dtype=np.float32)
    normalized = memory._normalize_vector(vector)

    assert normalized.dtype == np.float32
    assert np.isclose(float(np.linalg.norm(normalized)), 1.0)
    assert memory.id_cache == {"decomposition": [], "extraction": []}

    await memory.close()