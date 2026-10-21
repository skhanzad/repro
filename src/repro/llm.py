"""Thin wrapper around the OpenAI client. No frameworks, just the SDK.

Ollama exposes an OpenAI-compatible API at /v1, so it uses the same client
with a different base URL.
"""

from functools import cache

from openai import AsyncOpenAI

from repro.config import Provider, settings


@cache
def get_client(provider: Provider = "openai") -> AsyncOpenAI:
    common = {"timeout": settings.request_timeout, "max_retries": settings.max_retries}
    if provider == "ollama":
        # Ollama ignores the key, but the SDK requires one.
        return AsyncOpenAI(base_url=settings.ollama_base_url, api_key="ollama", **common)
    # OPENAI_API_KEY / OPENAI_BASE_URL are picked up from the environment by the SDK.
    return AsyncOpenAI(**common)
