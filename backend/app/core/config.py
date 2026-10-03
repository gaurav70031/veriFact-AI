"""
Central application configuration via Pydantic Settings.
All values are read from environment variables or the .env file.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ── Application ─────────────────────────────────────────────────────────
    app_env: str = "development"
    app_name: str = "Fake News Detection API"
    app_version: str = "1.0.0"
    debug: bool = True
    secret_key: str = "change-me-in-production"

    # ── Server ───────────────────────────────────────────────────────────────
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000

    # ── PostgreSQL ───────────────────────────────────────────────────────────
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "fakenews_db"
    postgres_user: str = "fakenews_user"
    postgres_password: str = "password"

    @property
    def database_url(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # ── Redis ────────────────────────────────────────────────────────────────
    redis_host: str = "localhost"
    redis_port: int = 6379

    @property
    def redis_url(self) -> str:
        return f"redis://{self.redis_host}:{self.redis_port}"

    # ── ML ───────────────────────────────────────────────────────────────────
    model_dir: str = "ml/models/saved"

    # ── Evidence / News APIs ─────────────────────────────────────────────────
    newsapi_key: str = ""
    serpapi_key: str = ""
    rss_feed_urls: str = ""   # comma-separated list

    # ── CORS ─────────────────────────────────────────────────────────────────
    allowed_origins: str = "http://localhost:5173,http://localhost:3000"

    @property
    def cors_origins(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
