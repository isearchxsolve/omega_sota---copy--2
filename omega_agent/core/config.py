"""OMEGA system configuration."""

import os
from dataclasses import dataclass, field
from typing import Optional


# Groq-hosted models (OpenAI-compatible API)
GROQ_MODELS = {
    "primary": "llama-3.3-70b-versatile",
    "fast": "llama-3.1-8b-instant",
    "reasoning": "llama-3.3-70b-versatile",
    "backup": "llama-3.1-8b-instant",
}


@dataclass
class Config:
    """OMEGA system configuration."""

    # Default to Groq models — set GROQ_API_KEY for full LLM mode
    primary_model: str = GROQ_MODELS["primary"]
    backup_model: str = GROQ_MODELS["backup"]
    fast_model: str = GROQ_MODELS["fast"]
    reasoning_model: str = GROQ_MODELS["reasoning"]

    groq_api_key: Optional[str] = field(
        default_factory=lambda: os.environ.get("GROQ_API_KEY")
    )
    groq_base_url: str = "https://api.groq.com/openai/v1"

    anthropic_api_key: Optional[str] = field(
        default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY")
    )
    openai_api_key: Optional[str] = field(
        default_factory=lambda: os.environ.get("OPENAI_API_KEY")
    )

    # Force provider: "groq", "anthropic", "openai", or None (auto: groq > anthropic > openai)
    llm_provider: Optional[str] = field(
        default_factory=lambda: os.environ.get("OMEGA_LLM_PROVIDER")
    )

    max_task_timeout: int = 30
    max_total_time: int = 300
    max_retries: int = 3
    recursion_limit: int = 50

    enable_episodic_memory: bool = True
    enable_semantic_memory: bool = True
    enable_learning: bool = True
    memory_db_path: str = "./data/omega_memory.db"
    routing_db_path: str = "./data/omega_routing.db"

    domain_focus: Optional[str] = None
    auto_detect_domain: bool = True
    use_mock_llm: bool = False

    enable_dynamic_discovery: bool = True
    enable_faiss: bool = True
    sota_max_retries: int = 3
    sota_quality_threshold: float = 0.65
    code_fix_max_retries: int = 3

    log_level: str = "INFO"
    log_file: Optional[str] = None

    api_host: str = "0.0.0.0"
    api_port: int = 8000

    def active_llm_provider(self) -> Optional[str]:
        """Return active LLM provider or None for mock mode."""
        if self.use_mock_llm:
            return None
        forced = (self.llm_provider or "").lower().strip()
        if forced in ("groq", "anthropic", "openai"):
            key_map = {
                "groq": self.groq_api_key,
                "anthropic": self.anthropic_api_key,
                "openai": self.openai_api_key,
            }
            return forced if key_map.get(forced) else None
        if self.groq_api_key:
            return "groq"
        if self.anthropic_api_key:
            return "anthropic"
        if self.openai_api_key:
            return "openai"
        return None

    def has_llm_credentials(self) -> bool:
        return self.active_llm_provider() is not None
