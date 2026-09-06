"""Main OMEGA Agent — dynamic LLM + web-search reasoning."""

import logging
import re
import time
from typing import Any, Dict, List, Optional

from omega_agent.core.config import Config
from omega_agent.core.orchestrator import ModelOrchestrator
from omega_agent.core.execution import DAGExecutor
from omega_agent.core.types import AgentResult, ExecutionContext
from omega_agent.memory.system import MemorySystem
from omega_agent.reasoning.discovery import DynamicDiscoveryEngine
from omega_agent.reasoning.planner import DynamicPlanner
from omega_agent.reasoning.synthesizer import DynamicSynthesizer
from omega_agent.reasoning.types import DynamicDomainProfile
from omega_agent.reflection.analyzer import FailureAnalyzer
from omega_agent.reflection.learner import OmegaLearner
from omega_agent.reflection.quality_gate import SOTAQualityGate
from omega_agent.tools.registry import ToolRegistry
from omega_agent.tools.executor import ToolExecutor
from omega_agent.tools.stdlib import register_all_tools
from omega_agent.utils.async_utils import OmegaRecursionGuard
from omega_agent.utils.metrics import MetricsCollector
from omega_agent.utils.logging import setup_logging

logger = logging.getLogger("omega_agent")


class OmegaAgent:
    """
    OMEGA Agent: Dynamic action-taking orchestrator.

    Web Search → LLM Discover (domain, practices, tools) → LLM Plan (DAG)
    → Execute Tools → LLM Synthesize (ACTION) → Quality Gate → Learn → Store
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        orchestrator: Optional[ModelOrchestrator] = None,
        memory: Optional[MemorySystem] = None,
        tools: Optional[ToolRegistry] = None,
        learner: Optional[OmegaLearner] = None,
    ):
        self.config = config or Config()
        setup_logging(self.config.log_level, self.config.log_file)

        self.orchestrator = orchestrator or ModelOrchestrator(self.config)
        self.memory = memory or MemorySystem(self.config)
        self.tools = tools or ToolRegistry()
        register_all_tools(self.tools)
        self.tool_executor = ToolExecutor(self.tools)
        self.dag_executor = DAGExecutor(self.config, self.tool_executor)

        self.discovery = DynamicDiscoveryEngine(
            self.config, self.orchestrator, self.tools, self.tool_executor
        )
        self.planner = DynamicPlanner(self.config, self.orchestrator, self.tools)
        self.synthesizer = DynamicSynthesizer(self.orchestrator)
        self.quality_gate = SOTAQualityGate(self.config, self.orchestrator, self.tool_executor)

        self.learner = learner or OmegaLearner(self.config) if self.config.enable_learning else None
        self.analyzer = FailureAnalyzer()
        self.metrics = MetricsCollector()
        self._recursion_guard = OmegaRecursionGuard(self.config.recursion_limit)

        logger.info(
            "OmegaAgent initialized (provider=%s, dynamic=%s, tools=%d)",
            self.config.active_llm_provider() or "mock",
            self.config.enable_dynamic_discovery,
            len(self.tools.list_tools()),
        )

    async def run(
        self,
        goal: str,
        domain: Optional[str] = None,
        max_time: Optional[int] = None,
    ) -> AgentResult:
        if not goal or not isinstance(goal, str):
            raise ValueError("goal must be a non-empty string")

        ctx = ExecutionContext(
            goal=goal,
            domain=domain or self.config.domain_focus,
            max_time=max_time or self.config.max_total_time,
        )
        ctx.learning_progress["known_practices"] = self.memory.get_practices_hints()

        self.memory.audit.record("goal_received", goal, {"domain_hint": domain})

        logger.info("Starting OMEGA: %s", goal[:120])
        start = time.time()

        with self._recursion_guard:
            try:
                return await self._execute_with_sota_guarantee(goal, ctx, start)
            except RecursionError as e:
                return self._fail_result(str(e), ctx, start, "recursion_error")
            except Exception as e:
                logger.error("Goal failed: %s", e, exc_info=True)
                return self._fail_result(str(e), ctx, start, "error", {"error": str(e)})

    async def _execute_with_sota_guarantee(
        self,
        goal: str,
        ctx: ExecutionContext,
        start: float,
    ) -> AgentResult:
        """Retry until SOTA quality threshold met or max attempts exhausted."""
        best_result: Optional[AgentResult] = None
        best_quality = 0.0
        last_result: Optional[AgentResult] = None

        while ctx.sota_attempt < self.config.sota_max_retries:
            if ctx.is_timed_out():
                break

            result = await self._execute_once(goal, ctx, start)
            last_result = result
            profile = ctx.dynamic_profile
            qdetails: Dict[str, Any] = {}

            if profile:
                quality, qdetails = self.quality_gate.evaluate(result, profile)
                result.metadata["quality_score"] = quality
                result.metadata["quality_details"] = qdetails
            else:
                quality = result.decision.confidence if result.decision else 0.5

            if quality > best_quality:
                best_quality = quality
                best_result = result

            if not self.quality_gate.should_retry(quality, ctx.sota_attempt):
                break

            ctx.sota_attempt += 1
            ctx.route = self.analyzer.suggest_route_change(
                result.domain, {"issues": qdetails.get("criteria_failed", [])}
            )
            logger.info("SOTA retry %d (quality=%.2f)", ctx.sota_attempt, quality)

        return best_result or last_result

    async def _execute_once(
        self,
        goal: str,
        ctx: ExecutionContext,
        start: float,
    ) -> AgentResult:
        profile, discover_cost = await self.discovery.discover(goal, ctx, domain_hint=ctx.domain)
        ctx.dynamic_profile = profile
        ctx.domain = profile.domain
        ctx.add_cost(discover_cost)

        self.memory.audit.record(
            "domain_discovered",
            goal,
            profile.to_dict(),
            domain=profile.domain,
        )

        best_route = self.learner.get_best_route(profile.domain) if self.learner else profile.model_priority
        ctx.route = best_route if best_route != "default" else profile.model_priority
        model = self.orchestrator.select_model(profile.domain, ctx.route)

        past = await self.memory.recall_similar(goal, profile.domain, limit=3)
        if past:
            ctx.learning_progress["past_experiences"] = past[:3]

        dag, plan_cost = await self.planner.plan(goal, profile, ctx)
        ctx.add_cost(plan_cost)

        self.memory.audit.record("plan_created", goal, {"tasks": [t.id for t in dag], "tools": [t.tool_name for t in dag]}, profile.domain)

        task_results = await self.dag_executor.execute(dag, ctx)

        decision, llm_output, synth_cost = await self.synthesizer.synthesize(
            goal, profile, task_results, self.config.has_llm_credentials()
        )
        ctx.add_cost(synth_cost)

        if self._needs_code_fix(profile, goal, llm_output, task_results):
            code = llm_output
            if "def " not in code:
                code, gen_cost = await self.orchestrator.invoke(
                    prompt=f"Write complete Python code with tests for:\n{goal}\nInclude try/except.",
                    system=profile.system_prompt,
                    temperature=0.2,
                )
                ctx.add_cost(gen_cost)

            fixed_code, exec_meta, fix_cost = await self.quality_gate.ensure_code_quality(
                goal, profile, code, ctx.add_cost
            )
            ctx.add_cost(fix_cost)
            task_results["generated_code"] = fixed_code
            task_results["code_execution"] = exec_meta
            if decision.risk_params is None:
                decision.risk_params = {}
            decision.risk_params["tests_passed"] = exec_meta.get("tests_passed", False)
            decision.risk_params["code_attempts"] = exec_meta.get("attempts", 0)

        output = decision.to_output()
        result = AgentResult(
            success=True,
            output=output,
            domain=profile.domain,
            route=ctx.route,
            cost=ctx.cost_so_far,
            latency=time.time() - start,
            metadata={
                "tasks_completed": len(task_results),
                "tasks_total": len(dag),
                "model": model,
                "errors": ctx.errors,
                "dynamic_profile": profile.to_dict(),
                "best_practices": profile.best_practices[:5],
                "tools_used": [t.tool_name for t in dag],
                "web_evidence_count": len(profile.web_evidence),
            },
            decision=decision,
        )

        quality = self._estimate_quality(result, profile)
        await self.memory.save_result(goal, profile.domain, result)

        if self.learner:
            await self.learner.update_from_outcome(goal, profile.domain, ctx.route, result, quality=quality)

        self.metrics.record(
            domain=profile.domain,
            route=ctx.route,
            success=result.success,
            quality=quality,
            cost=result.cost,
            latency=result.latency,
            goal=goal,
        )

        logger.info(
            "Completed: domain=%s action=%s confidence=%.0f%% latency=%.2fs",
            profile.domain,
            decision.action,
            decision.confidence * 100,
            result.latency,
        )
        return result

    @staticmethod
    def _needs_code_fix(
        profile: DynamicDomainProfile,
        goal: str,
        llm_output: str,
        task_results: Dict[str, Any],
    ) -> bool:
        combined = (profile.domain + " " + goal + " " + profile.output_format).lower()
        coding_signals = ["code", "python", "implement", "function", "class", "program"]
        return any(s in combined for s in coding_signals) or "code_executor" in str(task_results)

    @staticmethod
    def _estimate_quality(result: AgentResult, profile: DynamicDomainProfile) -> float:
        if not result.decision:
            return 0.5
        score = result.decision.confidence
        output = result.output.lower()

        for criterion in profile.quality_criteria:
            if criterion.lower() in output or any(w in output for w in criterion.lower().split()):
                score += 0.03

        if result.decision.risk_params.get("tests_passed"):
            score += 0.15
        if len(profile.web_evidence) > 0:
            score += 0.05
        if len(profile.best_practices) > 2:
            score += 0.05

        return min(1.0, score)

    @staticmethod
    def _fail_result(message: str, ctx: ExecutionContext, start: float, route: str, extra: Optional[Dict] = None) -> AgentResult:
        return AgentResult(
            success=False,
            output=f"Error: {message}",
            domain=ctx.domain or "unknown",
            route=route,
            cost=ctx.cost_so_far,
            latency=time.time() - start,
            metadata={**(extra or {}), "errors": ctx.errors},
        )

    def get_metrics(self) -> Dict[str, Any]:
        return self.metrics.get_summary()

    def get_memory_stats(self) -> Dict[str, Any]:
        return self.memory.stats()

    def list_tools(self) -> List[str]:
        return self.tools.list_tools()
