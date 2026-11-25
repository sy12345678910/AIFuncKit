"""
Tests for ArgumentExtractionService.

Since the LLM call (get_data_generated_by_model) is always mocked, these
tests verify the JSON-parsing and fallback logic without network or model access.

Covers:
  - No expected_args → returns {} immediately (no LLM call).
  - LLM returns a dict → returned as-is.
  - LLM returns a JSON string → parsed correctly.
  - LLM returns JSON wrapped in markdown code block (AI noise) → {} fallback.
  - LLM returns CAPS/noise string → {} fallback.
  - LLM call raises → returns {} gracefully.
  - memory_example block is included in prompt when provided.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from aifunckit._internal.orchestrator.services.argument_extraction import ArgumentExtractionService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _llm_config():
    from aifunckit._internal.schemas.orchestrator_schemas import LLMConfig
    return LLMConfig(provider="ollama", model_name="test-model", context_window=1024)


def _memory_example(prompt="old prompt", context="{}", extracted='{"x": 1}'):
    mem = MagicMock()
    mem.user_prompt = prompt
    mem.context_snapshot = context
    mem.extracted_json = extracted
    return mem


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestArgumentExtraction:

    @pytest.fixture()
    def service(self):
        return ArgumentExtractionService()

    # --- no args needed ---

    @pytest.mark.asyncio
    async def test_no_expected_args_skips_llm(self, service):
        with patch(
            "aifunckit._internal.orchestrator.services.argument_extraction.get_data_generated_by_model"
        ) as mock_llm:
            result = await service.extract(
                task="do something",
                expected_args={},
                current_context={},
                llm_config=_llm_config(),
                memory_example=None,
            )
        assert result == {}
        mock_llm.assert_not_called()

    # --- LLM returns dict ---

    @pytest.mark.asyncio
    async def test_dict_response_returned_directly(self, service):
        payload = {"message": "Hello world", "level": "critical"}
        with patch(
            "aifunckit._internal.orchestrator.services.argument_extraction.get_data_generated_by_model",
            new=AsyncMock(return_value=payload),
        ):
            result = await service.extract(
                task="send alert",
                expected_args={"message": str, "level": str},
                current_context={},
                llm_config=_llm_config(),
                memory_example=None,
            )
        assert result == payload

    # --- LLM returns JSON string ---

    @pytest.mark.asyncio
    async def test_json_string_response_is_parsed(self, service):
        payload = {"volume": 75}
        with patch(
            "aifunckit._internal.orchestrator.services.argument_extraction.get_data_generated_by_model",
            new=AsyncMock(return_value=json.dumps(payload)),
        ):
            result = await service.extract(
                task="set volume",
                expected_args={"volume": int},
                current_context={},
                llm_config=_llm_config(),
                memory_example=None,
            )
        assert result == payload

    # --- AI noise: markdown code block ---

    @pytest.mark.parametrize("bad_response", [
        "```json\n{\"x\": 1}\n```",   # markdown block
        "```\n{\"x\": 1}\n```",       # bare fenced block
        "Here is the JSON:\n{\"x\": 1}",   # prose prefix
        "RESULT: {x: 1}",             # invalid JSON with CAPS prefix
        "Sure! The answer is 42.",    # plain text
        "",                           # empty
        None,                         # NoneType
    ])
    @pytest.mark.asyncio
    async def test_unparseable_response_returns_empty(self, service, bad_response):
        with patch(
            "aifunckit._internal.orchestrator.services.argument_extraction.get_data_generated_by_model",
            new=AsyncMock(return_value=bad_response),
        ):
            result = await service.extract(
                task="some task",
                expected_args={"x": int},
                current_context={},
                llm_config=_llm_config(),
                memory_example=None,
            )
        assert result == {}

    # --- LLM raises ---

    @pytest.mark.asyncio
    async def test_llm_exception_returns_empty(self, service):
        with patch(
            "aifunckit._internal.orchestrator.services.argument_extraction.get_data_generated_by_model",
            new=AsyncMock(side_effect=RuntimeError("connection refused")),
        ):
            result = await service.extract(
                task="run action",
                expected_args={"key": str},
                current_context={},
                llm_config=_llm_config(),
                memory_example=None,
            )
        assert result == {}

    # --- memory example ---

    @pytest.mark.asyncio
    async def test_memory_example_included_in_llm_call(self, service):
        """When memory_example is provided the prompt should reference past experience."""
        captured_prompt = {}

        async def _fake_llm(prompt, **kwargs):
            captured_prompt["prompt"] = prompt
            return {"n": 5}

        with patch(
            "aifunckit._internal.orchestrator.services.argument_extraction.get_data_generated_by_model",
            new=AsyncMock(side_effect=_fake_llm),
        ):
            await service.extract(
                task="repeat 5 times",
                expected_args={"n": int},
                current_context={},
                llm_config=_llm_config(),
                memory_example=_memory_example(prompt="repeat 3 times"),
            )

        assert "PAST SUCCESSFUL EXPERIENCE" in captured_prompt["prompt"]
        assert "repeat 3 times" in captured_prompt["prompt"]
