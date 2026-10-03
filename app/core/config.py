"""Centralized configuration for the application.

Loads environment variables strictly from .env for LLM integration
and PostgreSQL database storage.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded strictly from environment / .env file."""

    # Required: Must be defined in .env
    GROQ_API_KEY: str
    DATABASE_URL: str

    # Optional model hyperparameters with safe defaults
    GROQ_MODEL: str = "qwen/qwen3.8-27b"
    GROQ_TEMPERATURE: float = 0.1

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
