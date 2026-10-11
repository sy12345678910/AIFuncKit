from __future__ import annotations

from typing import List, Union

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


class TaskList(BaseModel):
    """Ordered decomposition of a user request into tasks or conditionals."""

    tasks: List[Union[str, ConditionalTask]] = Field(
        description="Ordered list of execution tasks or conditional blocks"
    )


class TaskAnalysis(BaseModel):
    """Result of task parsing into a command name and flat parameter payload."""

    action: str = Field(description="Core command name in lower_snake_case")
    parameters: str = Field(description="Flat technical key-value arguments")
