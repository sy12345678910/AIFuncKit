"""
Tests for TopologyResolver.

Covers:
  - Empty task list → SequentialPipeline (default).
  - Tasks with execution_type="parallel" → ParallelPipeline.
  - Tasks with condition_variable key → ConditionalPipeline.
  - Tasks with loop_variable / iterations / repetitions → LoopPipeline.
  - Tasks with timeout / timeout_seconds → TimeoutPipeline.
  - Tasks with items / for_each / collection → ForEachLoopPipeline.
  - Matcher priority: parallel beats conditional when both keys present.
  - Object-based tasks (not just dicts).
  - context object with .decomposed_tasks attribute is handled.

NOTE: TopologyResolver._matchers is a class variable shared across ALL tests in
the process.  We must NOT mutate it; we only read it via resolve().
"""
from __future__ import annotations

import pytest

# Import AFTER conftest has patched sys.path
from aifunckit._internal.orchestrator.topology_resolver import TopologyResolver
from aifunckit._internal.orchestrator.pipelines.sequential import SequentialPipeline
from aifunckit._internal.orchestrator.pipelines.parallel import ParallelPipeline
from aifunckit._internal.orchestrator.pipelines.conditional import ConditionalPipeline
from aifunckit._internal.orchestrator.pipelines.loop import LoopPipeline
from aifunckit._internal.orchestrator.pipelines.timeout import TimeoutPipeline
from aifunckit._internal.orchestrator.pipelines.for_each_loop import ForEachLoopPipeline


# ---------------------------------------------------------------------------
# Helper: resolve from plain list (not a PipelineContext)
# ---------------------------------------------------------------------------

def resolve(tasks):
    return TopologyResolver.resolve(tasks)


# ---------------------------------------------------------------------------
# Default / empty
# ---------------------------------------------------------------------------

def test_empty_list_gives_sequential():
    assert resolve([]) is SequentialPipeline


def test_plain_tasks_give_sequential():
    tasks = [{"description": "do A"}, {"description": "do B"}]
    assert resolve(tasks) is SequentialPipeline


# ---------------------------------------------------------------------------
# Parallel
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key", ["execution_type", "type"])
def test_parallel_key_selects_parallel_pipeline(key):
    tasks = [{key: "parallel", "description": "run both"}]
    assert resolve(tasks) is ParallelPipeline


# ---------------------------------------------------------------------------
# Conditional
# ---------------------------------------------------------------------------

def test_condition_variable_selects_conditional_pipeline():
    tasks = [{"condition_variable": "is_hot", "description": "cool down"}]
    assert resolve(tasks) is ConditionalPipeline


# ---------------------------------------------------------------------------
# Loop
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key", ["loop_variable", "iterations", "repetitions"])
def test_loop_keys_select_loop_pipeline(key):
    tasks = [{key: 5, "description": "repeat"}]
    assert resolve(tasks) is LoopPipeline


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key", ["timeout", "timeout_seconds"])
def test_timeout_key_selects_timeout_pipeline(key):
    tasks = [{key: 30, "description": "slow task"}]
    assert resolve(tasks) is TimeoutPipeline


# ---------------------------------------------------------------------------
# ForEach
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key", ["items", "for_each", "collection"])
def test_foreach_key_selects_foreach_pipeline(key):
    tasks = [{key: [1, 2, 3], "description": "per item"}]
    assert resolve(tasks) is ForEachLoopPipeline


# ---------------------------------------------------------------------------
# Priority: parallel > conditional (registered first)
# ---------------------------------------------------------------------------

def test_parallel_beats_conditional_when_both_present():
    tasks = [
        {"execution_type": "parallel", "condition_variable": "flag"}
    ]
    assert resolve(tasks) is ParallelPipeline


# ---------------------------------------------------------------------------
# Object-based tasks (hasattr path)
# ---------------------------------------------------------------------------

class _FakeTask:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def test_object_task_with_execution_type_parallel():
    tasks = [_FakeTask(execution_type="parallel")]
    assert resolve(tasks) is ParallelPipeline


def test_object_task_with_loop_variable():
    tasks = [_FakeTask(loop_variable="i")]
    assert resolve(tasks) is LoopPipeline


def test_object_task_no_special_keys_gives_sequential():
    tasks = [_FakeTask(description="nothing special")]
    assert resolve(tasks) is SequentialPipeline


# ---------------------------------------------------------------------------
# Context object with decomposed_tasks attribute
# ---------------------------------------------------------------------------

class _FakeContext:
    def __init__(self, tasks):
        self.decomposed_tasks = tasks


def test_context_object_is_unwrapped():
    ctx = _FakeContext([{"execution_type": "parallel"}])
    assert TopologyResolver.resolve(ctx) is ParallelPipeline


def test_context_object_empty_gives_sequential():
    ctx = _FakeContext([])
    assert TopologyResolver.resolve(ctx) is SequentialPipeline
