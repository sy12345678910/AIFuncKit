"""
Tests for StateManagerService.

Covers:
  - capture() takes a deep snapshot of environment_state.
  - rollback() restores the snapshot, discarding mutations.
  - Nested/complex state is cloned, not referenced.
  - Sequential capture → mutate → rollback cycle.
  - Multiple captures overwrite previous snapshot.
"""
from __future__ import annotations

import pytest
from aifunckit._internal.orchestrator.services.state_manager import StateManagerService


# ---------------------------------------------------------------------------
# Minimal PipelineContext stand-in
# ---------------------------------------------------------------------------

class _FakeContext:
    def __init__(self, state: dict):
        self.environment_state = state


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_capture_then_rollback_restores_state():
    ctx = _FakeContext({"temperature": 20, "fan": "off"})
    sm = StateManagerService(context=ctx)
    sm.capture()

    ctx.environment_state["temperature"] = 99
    ctx.environment_state["fan"] = "on"

    sm.rollback()
    assert ctx.environment_state == {"temperature": 20, "fan": "off"}


def test_snapshot_is_deep_copy_not_reference():
    state = {"nested": {"value": 1}}
    ctx = _FakeContext(state)
    sm = StateManagerService(context=ctx)
    sm.capture()

    ctx.environment_state["nested"]["value"] = 999
    sm.rollback()

    assert ctx.environment_state["nested"]["value"] == 1


def test_rollback_before_capture_gives_empty_state():
    """Rollback with no prior capture sets state to {} (empty snapshot)."""
    ctx = _FakeContext({"key": "original"})
    sm = StateManagerService(context=ctx)
    sm.rollback()
    assert ctx.environment_state == {}


def test_multiple_captures_keep_last():
    ctx = _FakeContext({"x": 1})
    sm = StateManagerService(context=ctx)

    sm.capture()                    # snapshot: x=1
    ctx.environment_state["x"] = 2
    sm.capture()                    # snapshot: x=2
    ctx.environment_state["x"] = 3

    sm.rollback()
    assert ctx.environment_state["x"] == 2


def test_capture_rollback_cycle_repeatable():
    ctx = _FakeContext({"step": 0})
    sm = StateManagerService(context=ctx)

    sm.capture()
    ctx.environment_state["step"] = 1
    sm.rollback()
    assert ctx.environment_state["step"] == 0

    sm.capture()
    ctx.environment_state["step"] = 5
    sm.rollback()
    assert ctx.environment_state["step"] == 0


def test_rollback_does_not_affect_non_state_attributes():
    ctx = _FakeContext({"a": 1})
    ctx.other_attr = "untouched"
    sm = StateManagerService(context=ctx)
    sm.capture()
    ctx.environment_state["a"] = 99
    sm.rollback()
    assert ctx.other_attr == "untouched"
