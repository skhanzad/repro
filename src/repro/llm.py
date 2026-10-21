"""Thin wrapper around the OpenAI client. No frameworks, just the SDK."""

from functools import lru_cache

from openai import AsyncOpenAI

from repro.config import settings


@lru_cache(maxsize=1)
def get_client() -> AsyncOpenAI:
    # OPENAI_API_KEY / OPENAI_BASE_URL are picked up from the environment by the SDK.
    return AsyncOpenAI(timeout=settings.request_timeout, max_retries=settings.max_retries)
