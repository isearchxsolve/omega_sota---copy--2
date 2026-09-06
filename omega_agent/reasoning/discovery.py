"""Dynamic domain, best-practices, and tool discovery via web search + LLM."""

import logging
from typing import Any, Dict, List, Optional, Tuple

from omega_agent.core.config import Config
from omega_agent.core.orchestrator import ModelOrchestrator
from omega_agent.core.types import ExecutionContext
from omega_agent.reasoning.evidence import (
    build_tool_guidance_from_evidence,
    extract_practices_from_snippets,
    infer_domain_label,
    rank_tools_by_evidence,
)
from omega_agent.reasoning.types import DynamicDomainProfile
from omega_agent.tools.executor import ToolExecutor
from omega_agent.tools.registry import ToolRegistry

logger = logging.getLogger("omega_agent.reasoning.discovery")

DISCOVERY_SYSTEM = """You are OMEGA Discovery Engine. Analyze goals using web evidence and available tools.
NEVER use predefined domain lists. Infer domain, best practices, and tools dynamically from:
- The user's goal
- Web search evidence
- Available tool catalog

Return JSON with keys:
- domain: descriptive slug (e.g. "crypto_options_trading", "ml_research", "react_frontend")
- sub_domain: more specific label
- confidence: 0-1
- best_practices: list of strings from web evidence + reasoning
- execution_style: {urgency, depth, risk_tolerance} each low/medium/high
- recommended_tools: list of tool names FROM THE CATALOG ONLY
- tool_usage_guidance: {tool_name: "how to use for this goal"}
- system_prompt: expert persona prompt for this specific goal
- output_format: what the final output must contain
- quality_criteria: list of measurable success criteria
- model_priority: "speed" or "accuracy"
"""


class DynamicDiscoveryEngine:
    """Discover domain, practices, and tools via web search + LLM reasoning."""

    def __init__(
        self,
        config: Config,
        orchestrator: ModelOrchestrator,
        tool_registry: ToolRegistry,
        tool_executor: ToolExecutor,
    ):
        self.config = config
        self.orchestrator = orchestrator
        self.tools = tool_registry
        self.tool_executor = tool_executor

    async def discover(
        self,
        goal: str,
        ctx: ExecutionContext,
        domain_hint: Optional[str] = None,
    ) -> Tuple[DynamicDomainProfile, float]:
        web_context, web_cost = await self._gather_web_context(goal)
        ctx.web_context = web_context

        catalog = self.tools.get_catalog_for_llm()
        past_practices = ctx.learning_progress.get("known_practices", [])

        prompt = self._build_discovery_prompt(goal, web_context, catalog, domain_hint, past_practices)

        if self.config.has_llm_credentials():
            data, llm_cost = await self.orchestrator.invoke_json(
                prompt=prompt,
                system=DISCOVERY_SYSTEM,
                temperature=0.3,
            )
            profile = self._validate_profile(data, catalog, web_context)
            return profile, web_cost + llm_cost

        profile = self._discover_from_evidence(goal, web_context, catalog, domain_hint)
        return profile, web_cost

    async def _gather_web_context(self, goal: str) -> Tuple[Dict[str, Any], float]:
        total_cost = 0.0
        queries = [
            f"best practices {goal[:120]}",
            f"how to approach {goal[:100]} expert methodology",
            f"{goal[:80]} tools techniques 2024 2025",
        ]
        all_results: List[Dict[str, Any]] = []
        snippets: List[str] = []

        for query in queries:
            result, cost = await self.tool_executor.execute("web_search", {"query": query, "max_results": 4}, "discovery")
            total_cost += cost
            all_results.append(result)
            for item in result.get("results", []):
                snip = item.get("snippet") or item.get("title", "")
                if snip:
                    snippets.append(snip[:300])

        return {"queries": queries, "results": all_results, "snippets": snippets[:15]}, total_cost

    def _build_discovery_prompt(
        self,
        goal: str,
        web_context: Dict[str, Any],
        catalog: List[Dict[str, Any]],
        domain_hint: Optional[str],
        past_practices: List[str],
    ) -> str:
        snippets = "\n".join(f"- {s}" for s in web_context.get("snippets", [])[:10])
        tools_json = "\n".join(
            f"  - {t['name']}: {t['description']} | args: {t.get('args', {})}"
            for t in catalog
        )
        hint = f"\nUser domain hint (optional): {domain_hint}" if domain_hint else ""
        past = "\n".join(f"- {p}" for p in past_practices[:5]) if past_practices else "None"

        return f"""GOAL:
{goal}
{hint}

WEB SEARCH EVIDENCE:
{snippets or "No web results — infer from goal alone."}

AVAILABLE TOOLS (use ONLY these names in recommended_tools):
{tools_json}

PAST LEARNED PRACTICES FOR SIMILAR TASKS:
{past}

Analyze and return JSON profile. Pick tools dynamically based on goal + evidence.
system_prompt must make the agent an expert action-taker for THIS specific goal."""

    def _validate_profile(
        self,
        data: Dict[str, Any],
        catalog: List[Dict[str, Any]],
        web_context: Dict[str, Any],
    ) -> DynamicDomainProfile:
        valid_names = {t["name"] for t in catalog}
        recommended = [t for t in data.get("recommended_tools", []) if t in valid_names]

        if not recommended:
            recommended = rank_tools_by_evidence(
                data.get("domain", "general"),
                web_context,
                catalog,
                top_k=4,
            )

        data["recommended_tools"] = recommended
        data["web_evidence"] = web_context.get("snippets", [])[:5]
        if not data.get("system_prompt"):
            data["system_prompt"] = self._default_system_prompt(data.get("domain", "general"), data.get("best_practices", []))
        if not data.get("quality_criteria"):
            data["quality_criteria"] = ["actionable output", "evidence-based", "complete"]

        return DynamicDomainProfile.from_dict(data)

    def _discover_from_evidence(
        self,
        goal: str,
        web_context: Dict[str, Any],
        catalog: List[Dict[str, Any]],
        domain_hint: Optional[str],
    ) -> DynamicDomainProfile:
        """Mock-mode discovery: web evidence + tool catalog relevance — no hardcoded domains."""
        combined = (goal + " " + " ".join(web_context.get("snippets", []))).lower()
        domain = domain_hint or infer_domain_label(combined, goal)
        best_practices = extract_practices_from_snippets(web_context.get("snippets", []))
        if not best_practices:
            best_practices = [
                "Gather evidence before deciding",
                "Quantify confidence and risk",
                "End with a clear actionable decision",
            ]

        recommended = rank_tools_by_evidence(domain, web_context, catalog, top_k=4)
        urgency = "high" if any(w in combined for w in ["urgent", "crash", "now", "immediately", "5 min"]) else "medium"

        tool_guidance = {}
        for tool_name in recommended:
            tool = next((t for t in catalog if t["name"] == tool_name), None)
            if tool:
                tool_guidance[tool_name] = build_tool_guidance_from_evidence(tool, goal, web_context)

        return DynamicDomainProfile(
            domain=domain,
            sub_domain=domain.split("_")[-1] if "_" in domain else domain,
            confidence=0.65,
            best_practices=best_practices,
            execution_style={"urgency": urgency, "depth": "medium", "risk_tolerance": "moderate"},
            recommended_tools=recommended,
            tool_usage_guidance=tool_guidance,
            system_prompt=self._default_system_prompt(domain, best_practices),
            output_format="action_decision with confidence, rationale, risk parameters, next steps",
            quality_criteria=["actionable", "evidence-based", "risk-aware", "complete"],
            web_evidence=web_context.get("snippets", [])[:5],
            model_priority="speed" if urgency == "high" else "accuracy",
        )

    @staticmethod
    def _default_system_prompt(domain: str, practices: List[str]) -> str:
        practice_block = "\n".join(f"- {p}" for p in practices[:6])
        return (
            f"You are OMEGA — expert action-taker for: {domain.replace('_', ' ')}.\n"
            f"Apply these best practices (from web research + reasoning):\n{practice_block}\n"
            "RULES: Always end with a clear ACTION. Include confidence %, rationale, risk parameters, next steps."
        )
