from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime configuration. Read from environment / .env with prefix VERDICT_."""

    model_config = SettingsConfigDict(env_prefix="VERDICT_", env_file=".env", extra="ignore")

    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: str = "ollama"
    llm_model: str = "qwen3:8b"
    llm_temperature: float = 0.1
    llm_timeout_s: float = 120.0

    database_url: str = "sqlite:///verdict.db"

    # court budget defaults (per question)
    max_rounds: int = 6
    max_tool_calls: int = 120
    max_episode_steps: int = 12
    max_wall_clock_s: float = 900.0


def load_settings() -> Settings:
    return Settings()
