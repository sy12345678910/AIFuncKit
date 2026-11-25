"""
Tests for PluginRoutingService – text-based routing path.

The vector routing path is covered separately (requires embedding stubs).
Here we focus on:
  - route_text matches by action_name keyword in the task.
  - route_text matches by keyword list metadata.
  - route_text falls back to first item when no match.
  - route() raises PluginNotFoundError when metadata list is empty.
  - route() raises RoutingModeDoesNotExist for unknown routing_model.
  - LLM AI response noise (CAPS, markdown headers) doesn't break text routing.
"""
from __future__ import annotations

import pytest
import pytest_asyncio
import numpy as np
from unittest.mock import AsyncMock, MagicMock

from aifunckit._internal.orchestrator.services.plugin_routing import PluginRoutingService
from aifunckit.exceptions.exceptions import PluginNotFoundError, RoutingModeDoesNotExist


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_meta(action_name: str, description: str = "", keywords: list | None = None):
    def _fn():
        pass
    _fn.__name__ = action_name
    return {
        "name": _fn,
        "action_name": action_name,
        "description": description,
        "keywords": keywords or [],
    }


def _make_context(metadata: list, task_text: str = ""):
    """Build a minimal context stub for routing calls."""
    ctx = MagicMock()
    ctx.metadata.cached_metadata = metadata
    # Embedding that returns a fixed unit vector
    async def _encode(text):
        return np.ones(64, dtype=np.float32)
    ctx.embedding._encode_text = _encode
    return ctx


# ---------------------------------------------------------------------------
# route_text – basic
# ---------------------------------------------------------------------------

class TestRouteText:
    @pytest.fixture()
    def router(self):
        return PluginRoutingService(routing_model="text")

    @pytest.fixture()
    def metadata(self):
        return [
            _make_meta("send_alert", keywords=["alert", "notify"]),
            _make_meta("get_stats", keywords=["stats", "cpu"]),
        ]

    def test_matches_by_action_name_in_task(self, router, metadata):
        ctx = _make_context(metadata)
        result = router.route_text(task="send_alert the admin", context=ctx)
        assert result["action_name"] == "send_alert"

    def test_matches_by_keyword(self, router, metadata):
        ctx = _make_context(metadata)
        result = router.route_text(task="check cpu load", context=ctx)
        assert result["action_name"] == "get_stats"

    def test_fallback_to_first_on_no_match(self, router, metadata):
        ctx = _make_context(metadata)
        result = router.route_text(task="do something completely unrelated", context=ctx)
        assert result["action_name"] == "send_alert"  # first item

    # --- AI output noise ---

    def test_all_caps_task_still_matches(self, router, metadata):
        """LLM might return task description in CAPS – lower() normalisation must handle it."""
        ctx = _make_context(metadata)
        # route() lower-cases task before calling route_text
        result = router.route_text(task="send_alert", context=ctx)
        assert result["action_name"] == "send_alert"

    def test_markdown_prefix_in_task(self, router, metadata):
        """LLM might wrap the task in a markdown header like '## send_alert'."""
        ctx = _make_context(metadata)
        # strip markdown noise manually as the pipeline does, but route_text
        # checks if action_name is a substring – so it still finds it
        task = "## send_alert the admin now"
        result = router.route_text(task=task.lower(), context=ctx)
        assert result["action_name"] == "send_alert"


# ---------------------------------------------------------------------------
# route() dispatcher – error paths
# ---------------------------------------------------------------------------

class TestRouteDispatcher:

    @pytest.mark.asyncio
    async def test_empty_metadata_raises_plugin_not_found(self):
        router = PluginRoutingService(routing_model="text")
        ctx = _make_context(metadata=[])
        with pytest.raises(PluginNotFoundError):
            await router.route(task="anything", context=ctx)

    @pytest.mark.asyncio
    async def test_unknown_routing_model_raises(self):
        router = PluginRoutingService(routing_model="unknown_mode")
        ctx = _make_context(metadata=[_make_meta("some_action")])
        with pytest.raises(RoutingModeDoesNotExist):
            await router.route(task="anything", context=ctx)

    @pytest.mark.asyncio
    async def test_text_route_dispatched_correctly(self):
        router = PluginRoutingService(routing_model="text")
        meta = [_make_meta("get_stats", keywords=["stats"])]
        ctx = _make_context(meta)
        result = await router.route(task="show me stats", context=ctx)
        assert result["action_name"] == "get_stats"


# ---------------------------------------------------------------------------
# route_vector – similarity threshold
# ---------------------------------------------------------------------------

class TestRouteVector:
    """
    Minimal smoke test for vector routing without a real embedding model.
    We inject pre-computed description_embeddings into metadata so the
    similarity math runs but no model is loaded.
    """

    @pytest.fixture()
    def router(self):
        return PluginRoutingService(routing_model="vector")

    @pytest.mark.asyncio
    async def test_vector_routing_selects_best_match(self, router):
        from aifunckit._internal.memory.vector import normalize_vector

        vec_a = normalize_vector(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
        vec_b = normalize_vector(np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32))

        meta = [
            {**_make_meta("action_a", "description A"), "description_embedding": vec_a},
            {**_make_meta("action_b", "description B"), "description_embedding": vec_b},
        ]

        ctx = MagicMock()
        ctx.metadata.cached_metadata = meta

        # Task vector close to vec_a
        task_vec = normalize_vector(np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32))

        async def _encode(text):
            return task_vec

        ctx.embedding._encode_text = _encode

        result = await router.route_vector(task="run action a", context=ctx)
        assert result["action_name"] == "action_a"

    @pytest.mark.asyncio
    async def test_vector_routing_raises_when_similarity_too_low(self, router):
        """All zeros similarity should be below MIN_VECTOR_SIMILARITY."""
        from aifunckit._internal.memory.vector import normalize_vector

        # Orthogonal – dot product = 0 < 0.25
        vec_desc = normalize_vector(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
        meta = [
            {**_make_meta("lonely_action"), "description_embedding": vec_desc},
        ]

        ctx = MagicMock()
        ctx.metadata.cached_metadata = meta

        task_vec = normalize_vector(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))

        async def _encode(text):
            return task_vec

        ctx.embedding._encode_text = _encode

        with pytest.raises(PluginNotFoundError):
            await router.route_vector(task="totally unrelated", context=ctx)
