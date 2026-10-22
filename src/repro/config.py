from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

Provider = Literal["openai", "ollama"]

DEFAULT_MODELS: dict[Provider, str] = {
    "openai": "gpt-5-mini",
    "ollama": "gemma4:e4b",
}

# USD per 1M (input, output) tokens, used to enforce agent cost limits. Check these
# against the provider's current pricing page; models not listed here have no cost
# limit (token limits still apply). Local Ollama models are free.
MODEL_PRICES: dict[str, tuple[float, float]] = {
    "gpt-5": (1.25, 10.0),
    "gpt-5-mini": (0.25, 2.0),
    "gpt-5-nano": (0.05, 0.40),
}


def model_cost(provider: str, model: str, prompt_tokens: int, completion_tokens: int):
    """Cost in USD, or None when the price is unknown."""
    if provider == "ollama":
        return 0.0
    if model not in MODEL_PRICES:
        return None
    price_in, price_out = MODEL_PRICES[model]
    return (prompt_tokens * price_in + completion_tokens * price_out) / 1e6


# Small, fast models for high-volume tasks like issue classification.
CHEAP_MODELS: dict[Provider, str] = {
    "openai": "gpt-5-nano",
    "ollama": "gemma4:e4b",
}


class Settings(BaseSettings):
    """Runtime configuration, read from env vars (prefix REPRO_) and .env."""

    model_config = SettingsConfigDict(env_prefix="REPRO_", env_file=".env", extra="ignore")

    agent_provider: Provider = "openai"
    agent_model: str | None = None  # None -> DEFAULT_MODELS[agent_provider]
    judge_provider: Provider = "openai"
    judge_model: str | None = None  # None -> DEFAULT_MODELS[judge_provider]

    # Bug classifier: a cheap model, escalating to a stronger one when unsure.
    classifier_provider: Provider = "openai"
    classifier_model: str | None = None  # None -> CHEAP_MODELS[classifier_provider]
    escalation_provider: Provider = "openai"
    escalation_model: str | None = None  # None -> DEFAULT_MODELS[...]; "" disables escalation
    escalate_below: float = 0.7

    ollama_base_url: str = "http://localhost:11434/v1"

    max_steps: int = 10
    concurrency: int = 8
    request_timeout: float = 120.0
    max_retries: int = 3
    results_dir: str = "results"

    def resolved_judge_model(self) -> str:
        return self.judge_model or DEFAULT_MODELS[self.judge_provider]

    def resolved_classifier_model(self, provider: Provider | None = None) -> str:
        return self.classifier_model or CHEAP_MODELS[provider or self.classifier_provider]

    def resolved_escalation_model(self) -> str:
        if self.escalation_model is None:
            return DEFAULT_MODELS[self.escalation_provider]
        return self.escalation_model


settings = Settings()
