"""
Shared fixtures for the AIFuncKit test suite.
"""
from __future__ import annotations

import sys
import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Make sure "src" is ALWAYS on sys.path, regardless of how pytest is invoked.
# This handles both `pytest tests/` from project root AND `pytest` from src/.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_SRC  = os.path.join(_ROOT, "src")

for _p in (_ROOT, _SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
# Minimal stub for the EmbeddingService so we never load ONNX models in tests.
# ---------------------------------------------------------------------------
class FakeEmbeddingService:
    async def _encode_text(self, text: str):
        import numpy as np
        # Deterministic vector based on length – good enough for routing tests.
        return np.ones(64, dtype=np.float32) * (len(text) % 10 + 1)

    async def ensure_embedding_model(self):
        pass


# ---------------------------------------------------------------------------
# Minimal stub for AIRAGMemoryManager (no SQLite in unit tests).
# ---------------------------------------------------------------------------
class FakeRAGMemory:
    async def init_and_load_cache(self):
        pass

    async def add_record(self, **kwargs):
        pass

    async def close(self):
        pass

    async def search(self, *args, **kwargs):
        return None


# ---------------------------------------------------------------------------
# Shared pytest-asyncio mode
# ---------------------------------------------------------------------------
def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "asyncio: mark test as async (handled by pytest-asyncio)",
    )


# ---------------------------------------------------------------------------
# Helper: build a minimal PipelineContext without real deps
# ---------------------------------------------------------------------------
@pytest.fixture()
def make_context():
    """Factory that returns a PipelineContext populated with fakes."""
    from aifunckit._internal.schemas.pipeline_context import PipelineContext
    from aifunckit._internal.schemas.orchestrator_schemas import (
        LLMConfig,
        MetadataForLLM,
        ParamsForExecution,
        PromptConfig,
    )

    def _make(
        user_prompt: str = "do something",
        cached_metadata: list | None = None,
        decomposed_tasks: list | None = None,
        environment_state: dict | None = None,
    ) -> PipelineContext:
        return PipelineContext(
            user_prompt=user_prompt,
            prompt=PromptConfig(user_prompt=user_prompt),
            llm_config=LLMConfig(),
            metadata=MetadataForLLM(
                cached_metadata=cached_metadata or [],
                context_block={},
            ),
            memory=FakeRAGMemory(),
            embedding=FakeEmbeddingService(),
            environment_state=environment_state or {},
            decomposed_tasks=decomposed_tasks or [],
            params_for_execution=ParamsForExecution(),
        )

    return _make


# ---------------------------------------------------------------------------
# Shared simple action metadata records
# ---------------------------------------------------------------------------
@pytest.fixture()
def sample_metadata():
    """Two metadata records with known action names."""
    def send_alert(message: str) -> str:
        return f"alert: {message}"

    def get_stats() -> dict:
        return {"cpu": 10}

    return [
        {
            "name": send_alert,
            "action_name": "send_alert",
            "description": "Send an alert message to the administrator",
            "args": {"message": "str"},
        },
        {
            "name": get_stats,
            "action_name": "get_stats",
            "description": "Get current system resource statistics",
            "args": {},
        },
    ]
