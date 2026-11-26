from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from aifunckit._internal.orchestrator.pipelines.node_solver import NodeSolver
from aifunckit._internal.orchestrator.pipelines.step_executer import StepExecuter
from aifunckit._internal.orchestrator.pipelines.validation_manager import PipelineValidationManager
from aifunckit._internal.orchestrator.services.action_execution import ActionExecutionService
from aifunckit._internal.orchestrator.services.argument_extraction import ArgumentExtractionService
from aifunckit._internal.orchestrator.services.break_circus import BreakCircusService
from aifunckit._internal.orchestrator.services.plugin_routing import PluginRoutingService
from aifunckit._internal.schemas.pipeline_context import PipelineContext


class BasePipeline(ABC):
    """Facade pipeline topology delegating step execution, node solving, and validation."""

    pipeline_type: str = "base"

    def __init__(
        self,
        routing: Optional[PluginRoutingService] = None,
        extraction: Optional[ArgumentExtractionService] = None,
        execution: Optional[ActionExecutionService] = None,
        circuit_breaker: Optional[BreakCircusService] = None,
        circus_breaker: Optional[BreakCircusService] = None,
        breaker_circus: Optional[BreakCircusService] = None,
    ) -> None:
        self.routing = routing or PluginRoutingService()
        self.extraction = extraction or ArgumentExtractionService()
        self.execution = execution or ActionExecutionService()
        self.strategy: Any = None
        self._circuit_breaker = (
            circuit_breaker or circus_breaker or breaker_circus or BreakCircusService()
        )
        self.node_solver = NodeSolver()
        self.validation_manager = PipelineValidationManager(node_solver=self.node_solver)
        self.step_executer = StepExecuter(
            routing=self.routing,
            extraction=self.extraction,
            execution=self.execution,
            circuit_breaker=self._circuit_breaker,
        )

    @property
    def circuit_breaker(self) -> BreakCircusService:
        return self._circuit_breaker

    @circuit_breaker.setter
    def circuit_breaker(self, value: BreakCircusService) -> None:
        self._circuit_breaker = value
        if hasattr(self, "step_executer"):
            self.step_executer.circuit_breaker = value

    @property
    def circus_breaker(self) -> BreakCircusService:
        return self._circuit_breaker

    @circus_breaker.setter
    def circus_breaker(self, value: BreakCircusService) -> None:
        self.circuit_breaker = value

    @property
    def breaker_circus(self) -> BreakCircusService:
        return self._circuit_breaker

    @breaker_circus.setter
    def breaker_circus(self, value: BreakCircusService) -> None:
        self.circuit_breaker = value

    @abstractmethod
    async def execute(self, context: PipelineContext) -> None:
        """Run this topology against the provided context."""

    async def _execute_task_node(
        self,
        task: Any,
        context: PipelineContext,
        plugin_meta: Optional[dict[str, Any]] = None,
    ) -> Optional[dict[str, Any]]:
        return await self.step_executer._execute_task_node(
            task, context, plugin_meta=plugin_meta
        )

    def _update_context_after_execution(
        self,
        func_data: dict[str, Any],
        executed: Any,
        context: PipelineContext,
        task: Any,
    ) -> None:
        self.step_executer._update_context_after_execution(
            func_data, executed, context, task
        )

    def _build_extraction_context(
        self,
        context: PipelineContext,
        plugin_name: str,
    ) -> dict[str, Any]:
        return self.step_executer._build_extraction_context(context, plugin_name)

    async def process_task(
        self,
        task: Any,
        context: PipelineContext,
        plugin_meta: Optional[dict[str, Any]] = None,
    ) -> Optional[tuple[dict[str, Any], dict[str, Any], NDArray[np.float32]]]:
        return await self.step_executer.process_task(
            task, context, plugin_meta=plugin_meta
        )

    async def persist_extraction(
        self,
        context: PipelineContext,
        task: Any,
        task_ctx: dict[str, Any],
        func_data: dict[str, Any],
        task_emb: NDArray[np.float32],
    ) -> None:
        await self.step_executer.persist_extraction(
            context, task, task_ctx, func_data, task_emb
        )

    def _is_conditional_node(self, node: Any) -> bool:
        return self.node_solver.is_conditional_node(node)

    def _is_loop_node(self, node: Any) -> bool:
        return self.node_solver.is_loop_node(node)

    def _is_for_each_node(self, node: Any) -> bool:
        return self.node_solver.is_for_each_node(node)

    def _is_timeout_node(self, node: Any) -> bool:
        return self.node_solver.is_timeout_node(node)

    def validate_state(self, context: PipelineContext) -> None:
        self.validation_manager.validate_state(context)

    def _pre_validation(
        self,
        tasks: Optional[Any] = None,
        context: Optional[PipelineContext] = None,
    ) -> None:
        self.validation_manager._pre_validation(tasks=tasks, context=context)


__all__ = [
    "BasePipeline",
    "NodeSolver",
    "PipelineValidationManager",
    "StepExecuter",
]
