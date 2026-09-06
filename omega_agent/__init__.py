"""OMEGA Agent — Multi-domain action-taking orchestrator."""

from omega_agent.agents.omega import OmegaAgent
from omega_agent.core.config import Config
from omega_agent.core.types import AgentResult, ActionDecision, ExecutionContext
from omega_agent.reasoning.types import DynamicDomainProfile

__version__ = "1.1.0"
__all__ = [
    "OmegaAgent",
    "Config",
    "AgentResult",
    "ActionDecision",
    "ExecutionContext",
    "DynamicDomainProfile",
]

# Optional: Import Gradio UI if gradio is available
try:
    from omega_agent.gradio_ui import GradioInterface, create_gradio_ui
    __all__.extend(["GradioInterface", "create_gradio_ui"])
except ImportError:
    pass
