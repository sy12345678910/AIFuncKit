from __future__ import annotations

from typing import Any, Iterable


class NodeSolver:
    """Universal node classifier for pipeline execution and topology resolution."""

    @staticmethod
    def is_parallel_node(node: Any) -> bool:
        """Return True if the node represents parallel execution."""
        if isinstance(node, dict):
            return node.get("execution_type") == "parallel" or node.get("type") == "parallel"
        return hasattr(node, "execution_type") and getattr(node, "execution_type") == "parallel"

    @staticmethod
    def is_conditional_node(node: Any) -> bool:
        """Return True if the node represents a conditional branch."""
        if isinstance(node, dict):
            return "condition_variable" in node
        return hasattr(node, "condition_variable")

    @staticmethod
    def is_loop_node(node: Any) -> bool:
        """Return True if the node represents a counted loop."""
        if isinstance(node, dict):
            return (
                "loop_variable" in node
                or "iterations" in node
                or "repetitions" in node
            )
        return (
            hasattr(node, "loop_variable")
            or hasattr(node, "repetitions")
            or hasattr(node, "iterations")
        )

    @staticmethod
    def is_for_each_node(node: Any) -> bool:
        """Return True if the node represents a collection iteration."""
        if isinstance(node, dict):
            return "items" in node or "for_each" in node or "collection" in node
        return (
            hasattr(node, "items")
            or hasattr(node, "for_each")
            or hasattr(node, "collection")
        )

    @staticmethod
    def is_timeout_node(node: Any) -> bool:
        """Return True if the node has a timeout guard."""
        if isinstance(node, dict):
            return "timeout" in node or "timeout_seconds" in node
        return hasattr(node, "timeout") or hasattr(node, "timeout_seconds")

    @classmethod
    def has_parallel_nodes(cls, tasks: Iterable[Any]) -> bool:
        return any(cls.is_parallel_node(node) for node in (tasks or []))

    @classmethod
    def has_conditional_nodes(cls, tasks: Iterable[Any]) -> bool:
        return any(cls.is_conditional_node(node) for node in (tasks or []))

    @classmethod
    def has_loop_nodes(cls, tasks: Iterable[Any]) -> bool:
        return any(cls.is_loop_node(node) for node in (tasks or []))

    @classmethod
    def has_for_each_nodes(cls, tasks: Iterable[Any]) -> bool:
        return any(cls.is_for_each_node(node) for node in (tasks or []))

    @classmethod
    def has_timeout_nodes(cls, tasks: Iterable[Any]) -> bool:
        return any(cls.is_timeout_node(node) for node in (tasks or []))
