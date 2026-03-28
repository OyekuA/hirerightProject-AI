from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""
    API_KEY: str
    GEMINI_API_KEY: str
    QDRANT_HOST: str = "qdrant"
    QDRANT_PORT: int = 6333
    SENTRY_DSN: Optional[str] = None
    CACHE_TTL_SECONDS: int = 86400
    GENERATION_BREAKER_COOLDOWN_SECONDS: int
    EMBEDDING_BREAKER_COOLDOWN_SECONDS: int
    MAX_PROMPT_CHARS: int = 50000
    INGEST_STATUS_STORE_PATH: str
    CALLBACK_HMAC_SECRET: str
    CALLBACK_SIGNATURE_TTL_SECONDS: int = 300
    MAX_INGEST_FILE_MB: int = 10
    INGEST_FETCH_TIMEOUT_SECONDS: int = 20
    ENFORCE_SINGLE_REPLICA: bool = False
    CALLBACK_MAX_RETRIES: int = 3
    CALLBACK_RETRY_BASE_SECONDS: int = 2
    LOG_LEVEL: str = "ERROR"

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )


_settings_instance: Optional[Settings] = None


def get_settings() -> Settings:
    """Return a singleton Settings instance."""
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings()
    return _settings_instance