"""Tool registration and discovery — LLM selects tools dynamically."""

import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("omega_agent.tools.registry")

TOOL_ARG_SCHEMAS: Dict[str, Dict[str, str]] = {
    "web_search": {"query": "string — search query", "max_results": "int — max results (default 5)"},
    "crypto_price_api": {"symbol": "string — coin id e.g. solana, bitcoin", "timeframe": "string — e.g. 1h"},
    "arxiv_search": {"query": "string — academic search query", "max_results": "int"},
    "semantic_scholar": {"query": "string", "max_results": "int"},
    "sentiment_analysis": {"text": "string — text to analyze", "domain": "string — context label"},
    "text_synthesizer": {"inputs": "list — prior task outputs", "goal": "string"},
    "code_generator": {"prompt": "string — coding task", "language": "string — e.g. python"},
    "code_validator": {"code": "string — Python source code"},
    "code_executor": {"code": "string — Python source to run", "timeout": "int — seconds"},
    "task_decomposer": {"goal": "string", "context": "any — prior context"},
}


class ToolRegistry:
    """Registry of available tools — all tools visible to LLM planner."""

    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}
        self.handlers: Dict[str, Callable] = {}

    def register(
        self,
        name: str,
        description: str,
        handler: Callable,
        args: Optional[Dict[str, str]] = None,
        usage_hint: str = "",
    ) -> None:
        self.tools[name] = {
            "name": name,
            "description": description,
            "args": args or TOOL_ARG_SCHEMAS.get(name, {}),
            "usage_hint": usage_hint,
        }
        self.handlers[name] = handler
        logger.debug("Registered tool: %s", name)

    def get(self, name: str) -> Optional[Dict[str, Any]]:
        return self.tools.get(name)

    def get_handler(self, name: str) -> Optional[Callable]:
        return self.handlers.get(name)

    def get_catalog_for_llm(self) -> List[Dict[str, Any]]:
        """Full tool catalog for dynamic LLM planning."""
        return [
            {
                "name": t["name"],
                "description": t["description"],
                "args": t.get("args", {}),
                "usage_hint": t.get("usage_hint", ""),
            }
            for t in self.tools.values()
        ]

    def get_tools_for_domain(self, domain: str) -> List[Dict[str, Any]]:
        """Legacy — returns full catalog; domain filtering is LLM-driven now."""
        return self.get_catalog_for_llm()

    def list_tools(self) -> List[str]:
        return list(self.tools.keys())
