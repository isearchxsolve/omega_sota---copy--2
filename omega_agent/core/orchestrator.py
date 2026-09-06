"""LLM orchestration — Groq-first, with Anthropic/OpenAI fallbacks."""

import json
import logging
import re
from typing import Any, Dict, Optional

from omega_agent.core.config import Config, GROQ_MODELS

logger = logging.getLogger("omega_agent.orchestrator")

MODEL_COSTS = {
    "llama-3.3-70b-versatile": 0.001,
    "llama-3.1-8b-instant": 0.0002,
    "llama-3.1-70b-versatile": 0.001,
    "mixtral-8x7b-32768": 0.0005,
    "claude-sonnet-4-20250514": 0.012,
    "claude-3-5-haiku-20241022": 0.002,
    "gpt-4o": 0.015,
    "gpt-4o-mini": 0.003,
}


class ModelOrchestrator:
    """Routes goals to the best LLM. Groq is the default when GROQ_API_KEY is set."""

    def __init__(self, config: Config):
        self.config = config
        provider = config.active_llm_provider() or "mock"
        logger.info("ModelOrchestrator initialized (provider=%s)", provider)

    def select_model(self, domain: str, priority: str = "accuracy") -> str:
        provider = self.config.active_llm_provider()
        if provider == "groq":
            if priority in ("speed", "fast", "efficient"):
                return self.config.fast_model
            if priority in ("deep", "accuracy"):
                return self.config.reasoning_model
            return self.config.primary_model
        if priority in ("speed", "fast", "efficient"):
            return self.config.fast_model
        if priority in ("deep", "accuracy"):
            return self.config.reasoning_model
        return self.config.primary_model

    def estimate_cost(self, model: str, input_tokens: int = 1000, output_tokens: int = 500) -> float:
        rate = MODEL_COSTS.get(model, 0.001)
        return rate * ((input_tokens + output_tokens) / 1000)

    async def invoke(
        self,
        prompt: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> tuple[str, float]:
        model = self._resolve_model(model)
        logger.info("Invoking %s (prompt=%d chars)", model, len(prompt))

        if not self.config.has_llm_credentials():
            return self._mock_response(prompt, model, system), self.estimate_cost(model)

        provider = self.config.active_llm_provider()
        try:
            if provider == "groq":
                return await self._invoke_groq(prompt, model, system, temperature, max_tokens)
            if provider == "anthropic":
                return await self._invoke_anthropic(prompt, model, system, temperature, max_tokens)
            if provider == "openai":
                return await self._invoke_openai(prompt, model, system, temperature, max_tokens)
        except Exception as e:
            logger.warning("LLM invoke failed (%s): %s — falling back to mock", provider, e)

        return self._mock_response(prompt, model, system), self.estimate_cost(model)

    async def invoke_json(
        self,
        prompt: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        temperature: float = 0.3,
    ) -> tuple[Dict[str, Any], float]:
        text, cost = await self.invoke(
            prompt + "\n\nRespond with valid JSON only, no markdown fences.",
            model=model,
            system=system,
            temperature=temperature,
        )
        return self._parse_json(text), cost

    def _resolve_model(self, model: Optional[str]) -> str:
        """Map legacy Claude/GPT model names to active provider models."""
        model = model or self.config.primary_model
        provider = self.config.active_llm_provider()

        if provider == "groq":
            if model.startswith("llama") or model.startswith("mixtral") or model.startswith("gemma"):
                return model
            if "haiku" in model or "instant" in model or "fast" in model:
                return self.config.fast_model
            return self.config.primary_model

        if provider == "openai":
            if model.startswith("gpt"):
                return model
            return self.config.backup_model if self.config.backup_model.startswith("gpt") else "gpt-4o-mini"

        if provider == "anthropic":
            if "claude" in model:
                return model
            return self.config.primary_model

        return model

    async def _invoke_groq(
        self,
        prompt: str,
        model: str,
        system: Optional[str],
        temperature: float,
        max_tokens: int,
    ) -> tuple[str, float]:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            api_key=self.config.groq_api_key,
            base_url=self.config.groq_base_url,
        )
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        groq_model = self._resolve_model(model)
        response = await client.chat.completions.create(
            model=groq_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = response.choices[0].message.content or ""
        usage = response.usage
        cost = self.estimate_cost(
            groq_model,
            usage.prompt_tokens if usage else 1000,
            usage.completion_tokens if usage else 500,
        )
        return text, cost

    async def _invoke_anthropic(
        self,
        prompt: str,
        model: str,
        system: Optional[str],
        temperature: float,
        max_tokens: int,
    ) -> tuple[str, float]:
        import anthropic

        client = anthropic.AsyncAnthropic(api_key=self.config.anthropic_api_key)
        kwargs: Dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system

        response = await client.messages.create(**kwargs)
        text = response.content[0].text if response.content else ""
        cost = self.estimate_cost(model, response.usage.input_tokens, response.usage.output_tokens)
        return text, cost

    async def _invoke_openai(
        self,
        prompt: str,
        model: str,
        system: Optional[str],
        temperature: float,
        max_tokens: int,
    ) -> tuple[str, float]:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=self.config.openai_api_key)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        openai_model = self._resolve_model(model)
        response = await client.chat.completions.create(
            model=openai_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = response.choices[0].message.content or ""
        usage = response.usage
        cost = self.estimate_cost(
            openai_model,
            usage.prompt_tokens if usage else 1000,
            usage.completion_tokens if usage else 500,
        )
        return text, cost

    def _mock_response(self, prompt: str, model: str, system: Optional[str]) -> str:
        prompt_lower = prompt.lower()
        domain_hint = "general"
        if any(k in prompt_lower for k in ["buy", "sell", "crypto", "btc", "sol", "trade"]):
            domain_hint = "crypto_trading"
        elif any(k in prompt_lower for k in ["research", "paper", "literature", "arxiv"]):
            domain_hint = "research"
        elif any(k in prompt_lower for k in ["code", "python", "implement", "function"]):
            domain_hint = "coding"

        if domain_hint == "crypto_trading":
            return (
                "## ACTION: REDUCE\n"
                "**Confidence:** 72%\n\n"
                "### Rationale\n"
                "Multi-factor analysis: RSI elevated at 75 (overbought), funding rates negative "
                "(-0.05% indicating short bias), BTC dominance at 48% suggests alt rotation risk. "
                "Fed hawkish stance adds macro headwind.\n\n"
                "### Risk Parameters\n"
                "- **position_size:** reduce by 25%\n"
                "- **stop_loss:** 3% below entry\n"
                "- **hedge:** consider short funding capture\n\n"
                "### Next Steps\n"
                "1. Reduce SOL long exposure by 25%\n"
                "2. Set tighter stops at -3%\n"
                "3. Monitor funding rate reversal"
            )
        if domain_hint == "research":
            return (
                "## Structured Analysis\n\n"
                "### Background\n"
                "ML interpretability remains a critical open problem (Ribeiro et al., 2016; "
                "Molnar, 2020).\n\n"
                "### Top 5 Unresolved Problems\n"
                "1. **Faithfulness vs. simplicity trade-off** — however, current methods lack guarantees\n"
                "2. **Causal vs. correlational explanations** — limitation of post-hoc methods\n"
                "3. **Scalability to foundation models** — future work needed\n"
                "4. **Human-aligned evaluation metrics** — gap in standardized benchmarks\n"
                "5. **Regulatory compliance integration** — emerging requirement\n\n"
                "### Conclusion\n"
                "Although progress has been made, significant gaps remain in faithful, scalable interpretability."
            )
        if domain_hint == "coding":
            return (
                "```python\n"
                "import asyncio\n"
                "import aiohttp\n"
                "from typing import Dict, Any\n\n"
                "async def fetch_urls(urls: list[str], timeout: float = 10.0) -> Dict[str, Any]:\n"
                "    async with aiohttp.ClientSession() as session:\n"
                "        async def fetch_one(url: str):\n"
                "            try:\n"
                "                async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:\n"
                "                    return url, await resp.json()\n"
                "            except Exception as e:\n"
                "                return url, {'error': str(e)}\n"
                "        results = await asyncio.gather(*[fetch_one(u) for u in urls])\n"
                "        return dict(results)\n\n"
                "def test_fetch_urls():\n"
                "    assert callable(fetch_urls)\n"
                "```"
            )
        return f"[{model}] Analysis complete for: {prompt[:200]}..."

    @staticmethod
    def _parse_json(text: str) -> Dict[str, Any]:
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\n?", "", text)
            text = re.sub(r"\n?```$", "", text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
        return {"raw": text, "tasks": []}
