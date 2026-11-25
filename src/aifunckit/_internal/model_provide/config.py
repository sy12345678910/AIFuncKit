from __future__ import annotations

from typing import Any, List, Optional, Union

from pydantic import BaseModel, Field


class SimpleTask(BaseModel):
    """Minimal task representation for a generated execution item."""

    task: str = Field(description="Pure technical action description in English")


class ConditionalTask(BaseModel):
    """Conditional decision block for branching execution flows."""

    condition_variable: str = Field(
        description="The context key to check, e.g., 'email.has_unread'"
    )
    if_true: List[str] = Field(description="List of tasks to execute if condition is true")
    if_false: List[str] = Field(description="List of tasks to execute if condition is false")


class LoopTask(BaseModel):
    """Repeat the same task a fixed number of times."""

    task: str = Field(description="Pure technical task which must repeat")
    repetitions: int = Field(description="How many times the task should repeat")


class ForEachTask(BaseModel):
    """Execute child tasks for each item in a collection."""

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
    """Execute a task with an explicit timeout."""

    task: str = Field(description="Pure task with timeout")
    timeout: float = Field(description="Maximum time in seconds")


class TaskList(BaseModel):
    """Ordered decomposition of a user request into tasks or conditionals."""

    tasks: List[Union[str, ConditionalTask, LoopTask, TimeoutTask, ForEachTask]] = Field(
        description="Ordered list of execution tasks or conditional blocks"
    )


class TaskAnalysis(BaseModel):
    """Result of task parsing into a command name and flat parameter payload."""

    action: str = Field(description="Core command name in lower_snake_case")
    parameters: str = Field(description="Flat technical key-value arguments")
