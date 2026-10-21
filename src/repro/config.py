from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, read from env vars (prefix REPRO_) and .env."""

    model_config = SettingsConfigDict(env_prefix="REPRO_", env_file=".env", extra="ignore")

    agent_model: str = "gpt-5-mini"
    judge_model: str = "gpt-5-mini"
    max_steps: int = 10
    concurrency: int = 8
    request_timeout: float = 120.0
    max_retries: int = 3
    results_dir: str = "results"


settings = Settings()
