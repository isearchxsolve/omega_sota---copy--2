"""LLM-generated dynamic execution plans with evidence-driven mock fallback."""

import logging
from typing import Any, Dict, List, Tuple

from omega_agent.core.config import Config
from omega_agent.core.orchestrator import ModelOrchestrator
from omega_agent.core.types import ExecutionContext, TaskNode
from omega_agent.reasoning.arg_inference import EvidenceArgInference
from omega_agent.reasoning.evidence import is_aggregator_tool
from omega_agent.reasoning.types import DynamicDomainProfile
from omega_agent.tools.registry import ToolRegistry

logger = logging.getLogger("omega_agent.reasoning.planner")

PLANNER_SYSTEM = """You are OMEGA Planner. Create an execution DAG using ONLY tools from the catalog.
Return JSON:
{
  "tasks": [
    {
      "id": "unique_id",
      "name": "short name",
      "description": "what this step does",
      "tool_name": "must match catalog exactly",
      "arguments": {"arg": "value"},
      "dependencies": ["other_task_id"],
      "timeout": 30
    }
  ]
}
Rules:
- Independent data-gathering tools (web_search, arxiv_search, crypto_price_api) should have NO dependencies so they run in parallel
- Synthesis/validation tools depend on gather tasks they consume
- Use tool_usage_guidance from profile
- Reference prior task outputs with "$task_id" in arguments
- 2-6 tasks typically
- NEVER invent tool names not in catalog
"""


class DynamicPlanner:
    """Generate task DAG dynamically via LLM or evidence-driven inference."""

    def __init__(
        self,
        config: Config,
        orchestrator: ModelOrchestrator,
        tool_registry: ToolRegistry,
    ):
        self.config = config
        self.orchestrator = orchestrator
        self.tools = tool_registry

    async def plan(
        self,
        goal: str,
        profile: DynamicDomainProfile,
        ctx: ExecutionContext,
    ) -> Tuple[List[TaskNode], float]:
        catalog = self.tools.get_catalog_for_llm()
        catalog_by_name = {t["name"]: t for t in catalog}
        allowed = set(catalog_by_name.keys())

        prompt = self._build_plan_prompt(goal, profile, catalog, ctx.web_context)

        if self.config.has_llm_credentials():
            data, cost = await self.orchestrator.invoke_json(
                prompt=prompt,
                system=PLANNER_SYSTEM,
                temperature=0.2,
            )
            tasks = self._parse_tasks(data, allowed)
            if tasks:
                return tasks, cost

        tasks = self._plan_from_evidence(goal, profile, catalog_by_name, ctx.web_context)
        return tasks, 0.0

    def _build_plan_prompt(
        self,
        goal: str,
        profile: DynamicDomainProfile,
        catalog: List[Dict[str, Any]],
        web_context: Dict[str, Any],
    ) -> str:
        tools_block = "\n".join(
            f"- {t['name']}: {t['description']} | args: {t.get('args', {})}"
            for t in catalog
        )
        guidance = "\n".join(f"- {k}: {v}" for k, v in profile.tool_usage_guidance.items())
        snippets = "\n".join(f"- {s[:200]}" for s in profile.web_evidence[:5])

        return f"""GOAL: {goal}

DOMAIN PROFILE:
- domain: {profile.domain}
- sub_domain: {profile.sub_domain}
- execution_style: {profile.execution_style}
- best_practices: {profile.best_practices[:5]}

TOOL USAGE GUIDANCE:
{guidance or "Infer from tool descriptions."}

WEB EVIDENCE:
{snippets or "None"}

TOOL CATALOG:
{tools_block}

Create optimal task DAG with parallel independent branches where possible.
recommended_tools hint: {profile.recommended_tools}"""

    def _parse_tasks(self, data: Dict[str, Any], allowed: set) -> List[TaskNode]:
        raw_tasks = data.get("tasks", [])
        if not raw_tasks and "raw" in data:
            return []

        nodes = []
        for i, t in enumerate(raw_tasks):
            tool_name = t.get("tool_name", "")
            if tool_name not in allowed:
                logger.warning("Skipping unknown tool %s", tool_name)
                continue
            nodes.append(
                TaskNode(
                    id=t.get("id", f"task_{i+1}"),
                    name=t.get("name", f"Step {i+1}"),
                    description=t.get("description", ""),
                    tool_name=tool_name,
                    arguments=t.get("arguments", {}),
                    dependencies=t.get("dependencies", []),
                    timeout=int(t.get("timeout", self.config.max_task_timeout)),
                )
            )
        return nodes

    def _plan_from_evidence(
        self,
        goal: str,
        profile: DynamicDomainProfile,
        catalog_by_name: Dict[str, Dict[str, Any]],
        web_context: Dict[str, Any],
    ) -> List[TaskNode]:
        """
        LLM-free evidence-driven DAG:
        - Gather tools run in parallel (wave 1, no dependencies)
        - Aggregator/synthesis tools depend on all gather tasks (wave 2)
        - Args inferred from web evidence + catalog schema (no hardcoded templates)
        """
        tool_names = [t for t in profile.recommended_tools if t in catalog_by_name]
        if not tool_names:
            tool_names = ["web_search"]

        gather_tools: List[str] = []
        aggregator_tools: List[str] = []

        for name in tool_names:
            entry = catalog_by_name[name]
            if is_aggregator_tool(name, entry.get("description", "")):
                aggregator_tools.append(name)
            else:
                gather_tools.append(name)

        if not gather_tools:
            gather_tools = [n for n in tool_names if n not in aggregator_tools] or ["web_search"]

        nodes: List[TaskNode] = []
        gather_ids: List[str] = []

        for i, tool_name in enumerate(gather_tools[:5]):
            task_id = f"gather_{i+1}"
            entry = catalog_by_name[tool_name]
            args = EvidenceArgInference.infer_args(entry, goal, profile, web_context, [])
            nodes.append(
                TaskNode(
                    id=task_id,
                    name=f"Gather via {tool_name}",
                    description=profile.tool_usage_guidance.get(tool_name, entry.get("description", "")),
                    tool_name=tool_name,
                    arguments=args,
                    dependencies=[],
                    timeout=self.config.max_task_timeout,
                )
            )
            gather_ids.append(task_id)

        if aggregator_tools:
            for j, tool_name in enumerate(aggregator_tools[:2]):
                task_id = f"aggregate_{j+1}"
                entry = catalog_by_name[tool_name]
                args = EvidenceArgInference.infer_args(entry, goal, profile, web_context, gather_ids)
                nodes.append(
                    TaskNode(
                        id=task_id,
                        name=f"Aggregate via {tool_name}",
                        description=profile.tool_usage_guidance.get(tool_name, entry.get("description", "")),
                        tool_name=tool_name,
                        arguments=args,
                        dependencies=list(gather_ids),
                        timeout=self.config.max_task_timeout,
                    )
                )
        elif len(gather_ids) > 1 and "text_synthesizer" in catalog_by_name:
            entry = catalog_by_name["text_synthesizer"]
            args = EvidenceArgInference.infer_args(entry, goal, profile, web_context, gather_ids)
            nodes.append(
                TaskNode(
                    id="synthesize_all",
                    name="Synthesize parallel results",
                    description="Combine evidence from parallel gather tasks",
                    tool_name="text_synthesizer",
                    arguments=args,
                    dependencies=list(gather_ids),
                    timeout=self.config.max_task_timeout,
                )
            )

        logger.info(
            "Evidence DAG: %d gather (parallel) + %d aggregate task(s)",
            len(gather_ids),
            len(nodes) - len(gather_ids),
        )
        return nodes
