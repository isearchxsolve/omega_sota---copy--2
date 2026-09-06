"""Tool execution with timeout and cost tracking."""

import logging
from typing import Any, Dict, Tuple

from omega_agent.tools.registry import ToolRegistry

logger = logging.getLogger("omega_agent.tools.executor")

TOOL_COSTS = {
    "web_search": 0.001,
    "crypto_price_api": 0.0005,
    "arxiv_search": 0.001,
    "semantic_scholar": 0.001,
    "sentiment_analysis": 0.0001,
    "text_synthesizer": 0.0001,
    "code_generator": 0.005,
    "code_validator": 0.0001,
    "code_executor": 0.002,
    "task_decomposer": 0.0001,
}


class ToolExecutor:
    """Execute registered tools with cost tracking."""

    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    async def execute(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        domain: str = "general",
    ) -> Tuple[Any, float]:
        handler = self.registry.get_handler(tool_name)
        if not handler:
            logger.warning("Unknown tool: %s", tool_name)
            return {"error": f"Unknown tool: {tool_name}"}, 0.0

        logger.info("Executing tool %s with args %s", tool_name, list(arguments.keys()))
        result = await handler(**arguments)
        cost = TOOL_COSTS.get(tool_name, 0.001)
        return result, cost
