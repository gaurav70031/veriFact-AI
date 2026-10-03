"""
Central application configuration via Pydantic Settings.
All values read from environment variables or .env file.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ── Application ───────────────────────────────────────────────────────────
    app_env:     Literal["development", "staging", "production"] = "development"
    app_name:    str  = "Fake News Detection API"
    app_version: str  = "1.0.0"
    debug:       bool = True
    secret_key:  str  = "change-me-in-production-use-32-chars-min"

    # ── Server ────────────────────────────────────────────────────────────────
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000

    # ── Request limits ────────────────────────────────────────────────────────
    max_text_length:    int = 50_000    # characters
    max_url_length:     int = 2_000
    max_request_body:   int = 1_048_576 # 1 MB

    # ── PostgreSQL ────────────────────────────────────────────────────────────
    postgres_host:     str = "localhost"
    postgres_port:     int = 5432
    postgres_db:       str = "fakenews_db"
    postgres_user:     str = "fakenews_user"
    postgres_password: str = "password"

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def database_url_sync(self) -> str:
        """Synchronous URL for Alembic."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # ── Redis ─────────────────────────────────────────────────────────────────
    redis_host: str = "localhost"
    redis_port: int = 6379

    @property
    def redis_url(self) -> str:
        return f"redis://{self.redis_host}:{self.redis_port}"

    # ── ML artefacts ──────────────────────────────────────────────────────────
    # Paths are relative to the project root (one level above backend/)
    ml_saved_models_dir: str = "ml/saved_models"

    @property
    def ml_root(self) -> Path:
        # backend/ is one level below project root
        return Path(__file__).resolve().parents[3] / self.ml_saved_models_dir

    # ── CORS ──────────────────────────────────────────────────────────────────
    allowed_origins: str = "http://localhost:5173,http://localhost:3000"

    @property
    def cors_origins(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]

    # ── Evidence / news APIs ──────────────────────────────────────────────────
    newsapi_key:   str = ""
    serpapi_key:   str = ""
    rss_feed_urls: str = ""   # comma-separated

    @property
    def rss_feeds(self) -> list[str]:
        return [u.strip() for u in self.rss_feed_urls.split(",") if u.strip()]

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
