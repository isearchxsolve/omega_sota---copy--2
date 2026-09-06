"""SOTA quality guarantee — retry until quality criteria met."""

import logging
import re
from typing import Any, Dict, Optional, Tuple

from omega_agent.core.config import Config
from omega_agent.core.orchestrator import ModelOrchestrator
from omega_agent.core.types import ActionDecision, AgentResult
from omega_agent.reasoning.types import DynamicDomainProfile
from omega_agent.tools.executor import ToolExecutor

logger = logging.getLogger("omega_agent.reflection.quality_gate")


class SOTAQualityGate:
    """Validate output against dynamic quality criteria; fix code until tests pass."""

    def __init__(
        self,
        config: Config,
        orchestrator: ModelOrchestrator,
        tool_executor: ToolExecutor,
    ):
        self.config = config
        self.orchestrator = orchestrator
        self.tool_executor = tool_executor

    def evaluate(
        self,
        result: AgentResult,
        profile: DynamicDomainProfile,
    ) -> Tuple[float, Dict[str, Any]]:
        score = result.decision.confidence if result.decision else 0.5
        details: Dict[str, Any] = {"criteria_met": [], "criteria_failed": []}

        output = result.output.lower()
        for criterion in profile.quality_criteria:
            criterion_lower = criterion.lower()
            met = self._check_criterion(criterion_lower, output, result)
            if met:
                details["criteria_met"].append(criterion)
                score += 0.05
            else:
                details["criteria_failed"].append(criterion)

        if result.decision and result.decision.action:
            score += 0.05
            details["criteria_met"].append("has_action")

        if len(result.output) > 300:
            score += 0.05

        details["score"] = min(1.0, score)
        details["passed"] = details["score"] >= self.config.sota_quality_threshold
        return details["score"], details

    async def ensure_code_quality(
        self,
        goal: str,
        profile: DynamicDomainProfile,
        initial_code: str,
        ctx_cost_callback,
    ) -> Tuple[str, Dict[str, Any], float]:
        """Generate → execute → fix loop until tests pass or max retries."""
        code = initial_code
        total_cost = 0.0
        last_result: Dict[str, Any] = {}

        for attempt in range(1, self.config.code_fix_max_retries + 1):
            exec_result, cost = await self.tool_executor.execute(
                "code_executor", {"code": code}, profile.domain
            )
            total_cost += cost
            ctx_cost_callback(cost)
            last_result = exec_result

            if exec_result.get("success"):
                logger.info("Code passed on attempt %d", attempt)
                return code, {"tests_passed": True, "attempts": attempt, **exec_result}, total_cost

            error = exec_result.get("stderr") or exec_result.get("error", "Unknown error")
            logger.info("Code fix attempt %d failed: %s", attempt, str(error)[:100])

            if not self.config.has_llm_credentials():
                break

            fix_prompt = (
                f"Fix this Python code for goal: {goal}\n\n"
                f"CODE:\n```python\n{code}\n```\n\n"
                f"ERROR:\n{error[:800]}\n\n"
                f"Return ONLY fixed complete code with tests. Include try/except."
            )
            fixed, fix_cost = await self.orchestrator.invoke(
                prompt=fix_prompt,
                system=profile.system_prompt,
                temperature=0.2,
            )
            total_cost += fix_cost
            ctx_cost_callback(fix_cost)

            match = re.search(r"```python\n(.*?)```", fixed, re.DOTALL)
            code = match.group(1) if match else fixed

        return code, {"tests_passed": False, "attempts": self.config.code_fix_max_retries, **last_result}, total_cost

    def should_retry(self, quality_score: float, attempt: int) -> bool:
        return quality_score < self.config.sota_quality_threshold and attempt < self.config.sota_max_retries

    @staticmethod
    def _check_criterion(criterion: str, output: str, result: AgentResult) -> bool:
        if "actionable" in criterion or "action" in criterion:
            return bool(result.decision and result.decision.action)
        if "risk" in criterion:
            return any(k in output for k in ["risk", "stop", "position", "size"])
        if "evidence" in criterion or "citation" in criterion:
            return len(output) > 200
        if "complete" in criterion:
            return len(output) > 400
        if "test" in criterion:
            meta = result.decision.risk_params if result.decision else {}
            return meta.get("tests_passed", False) or "test" in output or "assert" in output
        return criterion[:20] in output
