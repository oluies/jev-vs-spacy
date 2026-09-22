"""Run configuration from environment variables (or a `.env` file), typed and in one place."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    jev_model: str = "typesafe:jev-latest"  # pin e.g. typesafe:jev-1.13.0 for reproducible runs
    llm_model: str = "anthropic:claude-opus-5"
    braintrust_project: str = "email-classifier-bakeoff"
    limit: int = Field(0, ge=0, description="test rows per class; 0 = the whole test set")
    concurrency: int = Field(8, ge=1)


settings = Settings()
