from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime configuration. Read from environment / .env with prefix VERDICT_."""

    model_config = SettingsConfigDict(env_prefix="VERDICT_", env_file=".env", extra="ignore")

    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: str = "ollama"
    llm_model: str = "qwen3:8b"
    llm_temperature: float = 0.0
    llm_timeout_s: float = 60.0
    llm_max_tokens: int = 700
    # Hybrid-reasoning backends (qwen3, deepseek-r1 via Ollama) can silently spend
    # minutes on hidden "thinking" tokens per call. We turn that off by default:
    # courts don't need chain-of-thought, they need fast, reproducible tool calls.
    llm_think: bool = False

    database_url: str = "sqlite:///verdict.db"

    # court budget defaults (per question)
    max_rounds: int = 6
    max_tool_calls: int = 120
    max_episode_steps: int = 12
    max_wall_clock_s: float = 900.0


def load_settings() -> Settings:
    return Settings()
