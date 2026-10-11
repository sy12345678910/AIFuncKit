from __future__ import annotations

from typing import List, Union

from pydantic import BaseModel, Field


class SimpleTask(BaseModel):
    """A single English-language action that can be executed by the runtime."""

    task: str = Field(description="Pure technical action description in English")


class ConditionalTask(BaseModel):
    """A conditional task block that selects one of two task branches."""

    condition_variable: str = Field(
        description="The context key to check, e.g., 'email.has_unread'"
    )
    if_true: List[str] = Field(description="List of tasks to execute if condition is true")
    if_false: List[str] = Field(description="List of tasks to execute if condition is false")


class TaskList(BaseModel):
    """A validated task list produced by the planner model."""

    tasks: List[Union[str, ConditionalTask]] = Field(
        description="Ordered list of execution tasks or conditional blocks"
    )
