"""Dynamic action decision synthesis via LLM."""

import json
import logging
import re
from typing import Any, Dict, Tuple

from omega_agent.core.orchestrator import ModelOrchestrator
from omega_agent.core.types import ActionDecision
from omega_agent.reasoning.types import DynamicDomainProfile

logger = logging.getLogger("omega_agent.reasoning.synthesizer")


class DynamicSynthesizer:
    """Synthesize ActionDecision using dynamic profile — no hardcoded domain logic."""

    def __init__(self, orchestrator: ModelOrchestrator):
        self.orchestrator = orchestrator

    async def synthesize(
        self,
        goal: str,
        profile: DynamicDomainProfile,
        task_results: Dict[str, Any],
        has_llm: bool,
    ) -> Tuple[ActionDecision, str, float]:
        prompt = self._build_prompt(goal, profile, task_results)

        if has_llm:
            text, cost = await self.orchestrator.invoke(
                prompt=prompt,
                system=profile.system_prompt,
                temperature=0.4,
                max_tokens=4096,
            )
            decision = self._parse_decision(text, profile)
            return decision, text, cost

        text = self._mock_synthesis(goal, profile, task_results)
        decision = self._parse_decision(text, profile)
        return decision, text, 0.0

    def _build_prompt(
        self,
        goal: str,
        profile: DynamicDomainProfile,
        task_results: Dict[str, Any],
    ) -> str:
        results_summary = json.dumps(
            {k: str(v)[:500] for k, v in task_results.items()},
            indent=2,
        )
        practices = "\n".join(f"- {p}" for p in profile.best_practices[:6])
        criteria = "\n".join(f"- {c}" for c in profile.quality_criteria[:6])
        evidence = "\n".join(f"- {e[:200]}" for e in profile.web_evidence[:4])

        return f"""GOAL: {goal}

DOMAIN: {profile.domain} / {profile.sub_domain}
OUTPUT FORMAT: {profile.output_format}

BEST PRACTICES TO APPLY:
{practices}

QUALITY CRITERIA (must satisfy all):
{criteria}

WEB EVIDENCE:
{evidence}

TOOL EXECUTION RESULTS:
{results_summary}

Provide final analysis and ACTION decision.
MUST include: ACTION (verb), Confidence %, Rationale, Risk Parameters, Next Steps."""

    def _parse_decision(self, text: str, profile: DynamicDomainProfile) -> ActionDecision:
        return ActionDecision(
            action=self._extract_action(text),
            confidence=self._extract_confidence(text),
            rationale=text,
            risk_params=self._extract_risk_params(text),
            next_steps=self._extract_next_steps(text),
            domain=profile.domain,
        )

    @staticmethod
    def _mock_synthesis(goal: str, profile: DynamicDomainProfile, task_results: Dict[str, Any]) -> str:
        practices = "; ".join(profile.best_practices[:3])
        data_hint = str(list(task_results.values())[:2])[:400]
        return (
            f"## ACTION: EXECUTE\n"
            f"**Confidence:** 70%\n\n"
            f"### Rationale\n"
            f"Dynamic analysis for {profile.domain.replace('_', ' ')}.\n"
            f"Applied practices: {practices}.\n"
            f"Tool evidence: {data_hint}\n\n"
            f"Goal: {goal[:300]}\n\n"
            f"### Risk Parameters\n"
            f"- **approach:** evidence-based\n"
            f"- **validation:** cross-check tool results\n\n"
            f"### Next Steps\n"
            f"1. Execute primary action\n"
            f"2. Monitor outcomes\n"
            f"3. Adjust based on feedback"
        )

    @staticmethod
    def _extract_action(text: str) -> str:
        text_lower = text.lower()
        for action in [
            "reduce", "close", "hedge", "implement", "fix", "refactor", "deploy",
            "publish_draft", "deep_dive", "summarize", "execute", "delegate",
            "sell", "buy", "hold",
        ]:
            if action in text_lower:
                return action
        match = re.search(r"action[:\s]+(\w+)", text_lower)
        return match.group(1) if match else "execute"

    @staticmethod
    def _extract_confidence(text: str) -> float:
        match = re.search(r"(\d{1,3})\s*%", text)
        if match:
            return min(1.0, int(match.group(1)) / 100)
        return 0.7

    @staticmethod
    def _extract_risk_params(text: str) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        for line in text.split("\n"):
            m = re.match(r"[-*]\s*\**(\w+)\**:\s*(.+)", line.strip())
            if m:
                params[m.group(1).lower()] = m.group(2).strip()
        return params

    @staticmethod
    def _extract_next_steps(text: str) -> list:
        steps = []
        in_steps = False
        for line in text.split("\n"):
            if "next step" in line.lower():
                in_steps = True
                continue
            if in_steps and line.strip():
                cleaned = re.sub(r"^\d+\.\s*", "", line.strip().lstrip("- "))
                if cleaned and not cleaned.startswith("#"):
                    steps.append(cleaned)
        return steps[:7]
