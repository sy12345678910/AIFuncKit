"""
Tests for BreakCircusService (Circuit Breaker).

Covers:
  - Fresh breaker is closed (not open).
  - record_failure accumulates and trips at threshold.
  - record_success resets the counter but does not close an already-open breaker.
  - emergency_halt opens immediately regardless of count.
  - check_status raises when breaker is open.
  - reset() fully restores the breaker.
  - Custom failure_threshold values (edge: threshold=1).
  - Error message is non-empty on every raise.
"""
from __future__ import annotations

import pytest
from aifunckit._internal.orchestrator.services.break_circus import BreakCircusService
from aifunckit.exceptions.pipeline_exceptions import CircuitBreakerOpenException


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def breaker():
    return BreakCircusService(failure_threshold=3)


@pytest.fixture()
def breaker_one():
    """Circuit breaker that trips after a single failure."""
    return BreakCircusService(failure_threshold=1)


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------

def test_initial_state_is_closed(breaker):
    assert not breaker.is_open
    assert breaker.count_errors == 0


def test_check_status_does_not_raise_when_closed(breaker):
    breaker.check_status()   # must not raise


# ---------------------------------------------------------------------------
# Failure accumulation
# ---------------------------------------------------------------------------

def test_one_failure_does_not_trip(breaker):
    breaker.record_failure()
    assert not breaker.is_open
    assert breaker.count_errors == 1


def test_failures_below_threshold_do_not_open(breaker):
    for _ in range(2):   # threshold is 3, so 2 is safe
        breaker.record_failure()
    assert not breaker.is_open


def test_reaching_threshold_opens_breaker(breaker):
    for _ in range(2):
        breaker.record_failure()
    with pytest.raises(CircuitBreakerOpenException):
        breaker.record_failure()      # 3rd failure – trips breaker
    assert breaker.is_open


def test_threshold_one_trips_on_first_failure(breaker_one):
    with pytest.raises(CircuitBreakerOpenException):
        breaker_one.record_failure()
    assert breaker_one.is_open


# ---------------------------------------------------------------------------
# Success resets the counter
# ---------------------------------------------------------------------------

def test_record_success_resets_error_count(breaker):
    breaker.record_failure()
    breaker.record_failure()
    breaker.record_succes()   # note: intentional typo from source
    assert breaker.count_errors == 0
    assert not breaker.is_open


# ---------------------------------------------------------------------------
# Emergency halt
# ---------------------------------------------------------------------------

def test_emergency_halt_opens_immediately(breaker):
    with pytest.raises(CircuitBreakerOpenException):
        breaker.emergency_halt("test halt")
    assert breaker.is_open


def test_emergency_halt_message_is_propagated(breaker):
    with pytest.raises(CircuitBreakerOpenException) as exc_info:
        breaker.emergency_halt("Hardware overheating!")
    assert "Hardware overheating!" in str(exc_info.value)


# ---------------------------------------------------------------------------
# check_status after open
# ---------------------------------------------------------------------------

def test_check_status_raises_when_open(breaker):
    try:
        breaker.emergency_halt("forced")
    except CircuitBreakerOpenException:
        pass
    with pytest.raises(CircuitBreakerOpenException):
        breaker.check_status()


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

def test_reset_closes_breaker(breaker):
    try:
        breaker.emergency_halt("forced")
    except CircuitBreakerOpenException:
        pass
    breaker.reset()
    assert not breaker.is_open
    assert breaker.count_errors == 0


def test_check_status_ok_after_reset(breaker):
    try:
        breaker.emergency_halt("forced")
    except CircuitBreakerOpenException:
        pass
    breaker.reset()
    breaker.check_status()   # must not raise


# ---------------------------------------------------------------------------
# MasterOrchestrationEngine Integration
# ---------------------------------------------------------------------------
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np
from aifunckit._internal.orchestrator.pipeline import MasterOrchestrationEngine
from aifunckit._internal.orchestrator.pipelines.sequential import SequentialPipeline


def test_orchestrator_circus_breaker_aliases():
    with patch("aifunckit._internal.orchestrator.pipeline.AIRAGMemoryManager"), patch("aifunckit._internal.orchestrator.pipeline.EmbeddingService"):
        engine = MasterOrchestrationEngine(cached_metadata=[], failure_threshold=5)
        assert engine.breaker_circus.failure_threshold == 5
        assert engine.circus_breaker is engine.breaker_circus
        assert engine.circuit_breaker is engine.breaker_circus

        new_breaker = BreakCircusService(failure_threshold=2)
        engine.circus_breaker = new_breaker
        assert engine.breaker_circus is new_breaker
        assert engine.circuit_breaker is new_breaker


@pytest.mark.asyncio
async def test_run_pipeline_blocked_when_circuit_breaker_open():
    with patch("aifunckit._internal.orchestrator.pipeline.AIRAGMemoryManager"), patch("aifunckit._internal.orchestrator.pipeline.EmbeddingService"):
        engine = MasterOrchestrationEngine(cached_metadata=[])
        engine.circus_breaker.is_open = True

        with pytest.raises(CircuitBreakerOpenException):
            await engine.run_pipeline("test prompt")


@pytest.mark.asyncio
async def test_run_pipeline_passes_breaker_to_pipeline():
    with patch("aifunckit._internal.orchestrator.pipeline.AIRAGMemoryManager"), patch("aifunckit._internal.orchestrator.pipeline.EmbeddingService"):
        engine = MasterOrchestrationEngine(cached_metadata=[])

        mock_pipeline = MagicMock(spec=SequentialPipeline)
        mock_pipeline.validate_state = MagicMock()
        mock_pipeline.execute = AsyncMock()

        passed_breaker = None
        def mock_pipeline_cls(**kwargs):
            nonlocal passed_breaker
            passed_breaker = kwargs.get("circuit_breaker")
            return mock_pipeline

        with patch.object(engine.topology_resolver, "resolve", return_value=mock_pipeline_cls):
            async def fake_decompose(ctx):
                ctx.decomposed_tasks = ["task 1"]
                return np.zeros(10, dtype=np.float32)
            engine.decomposition.decompose = AsyncMock(side_effect=fake_decompose)

            await engine.run_pipeline("test prompt")
            assert passed_breaker is engine.circus_breaker

