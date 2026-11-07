from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from model_provide.model_router import get_data_generated_by_model as _router_get_data_generated_by_model
from orchestrator.pipeline import MasterOrchestrationEngine
from orchestrator.topology_resolver import TopologyResolver

get_data_generated_by_model = _router_get_data_generated_by_model

_legacy_orchestrator_path = Path(__file__).resolve().parents[1] / "orchestrator.py"
_legacy_spec = spec_from_file_location("legacy_orchestrator_module", _legacy_orchestrator_path)
AIOrchestrator = MasterOrchestrationEngine

if _legacy_spec is not None and _legacy_spec.loader is not None:
    _legacy_module = module_from_spec(_legacy_spec)
    _legacy_spec.loader.exec_module(_legacy_module)
    if hasattr(_legacy_module, "AIOrchestrator"):
        AIOrchestrator = _legacy_module.AIOrchestrator
    if hasattr(_legacy_module, "get_data_generated_by_model"):
        get_data_generated_by_model = _legacy_module.get_data_generated_by_model

__all__ = [
    "AIOrchestrator",
    "MasterOrchestrationEngine",
    "TopologyResolver",
    "get_data_generated_by_model",
]
