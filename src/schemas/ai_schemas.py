from __future__ import annotations

from typing import Any, List, Optional, Union

from pydantic import BaseModel, Field


class SimpleTask(BaseModel):
    """A single English-language action that can be executed by the runtime."""

    task: str = Field(description="Pure technical action description in English")


class ConditionalTask(BaseModel):
    """A conditional task block that selects one of two task branches."""

    condition_variable: str = Field(
        description="The context key to check"
    )
    if_true: List[str] = Field(description="List of tasks to execute if condition is true")
    if_false: List[str] = Field(description="List of tasks to execute if condition is false")

class LoopTask(BaseModel):
    task: str = Field(description="Pure technical task, which must repeat")
    repetitions: int = Field(description="How many times task should repeat")

class ForEachTask(BaseModel):
    """A loop task that executes child tasks for each item in a collection."""
    items: Union[List[Any], str] = Field(
        description="Collection of items or environment variable name"
    )
    task: Optional[str] = Field(
        default=None,
        description="Single task to execute for each item, using '{item}' placeholder if needed"
    )
    body: Optional[List[str]] = Field(
        default=None,
        description="List of tasks to execute for each item"
    )
    item_variable: str = Field(
        default="item",
        description="Variable name under which the current item is stored in environment_state"
    )

class TimeoutTask(BaseModel):
    task: str = Field(description="Pure task with timeout")
    timeout: float = Field(description="Time of timeout")

class TaskList(BaseModel):
    """A validated task list produced by the planner model."""

    tasks: List[Union[str, ConditionalTask, LoopTask, TimeoutTask, ForEachTask]] = Field(
        description="Ordered list of execution tasks or conditional blocks"
    )
