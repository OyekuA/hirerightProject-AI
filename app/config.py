from typing import Optional
from pydantic_settings import BaseSettings


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
    GEMINI_GENERATION_TIMEOUT_SECONDS: int = 30
    GEMINI_EMBEDDING_TIMEOUT_SECONDS: int = 30
    GEMINI_MAX_RETRIES: int = 2
    GEMINI_RETRY_BACKOFF_BASE_SECONDS: float = 1.0
    MAX_PROMPT_CHARS: int = 50000
    INGEST_STATUS_STORE_PATH: str
    CALLBACK_HMAC_SECRET: str
    CALLBACK_SIGNATURE_TTL_SECONDS: int = 300
    MAX_INGEST_FILE_MB: int = 10
    INGEST_FETCH_TIMEOUT_SECONDS: int = 20
    ENFORCE_SINGLE_REPLICA: bool = False
    CALLBACK_MAX_ATTEMPTS: int = 3
    CALLBACK_RETRY_BASE_SECONDS: int = 2
    CALLBACK_TIMEOUT_SECONDS: int = 10
    LOG_LEVEL: str = "ERROR"

_settings_instance: Optional[Settings] = None


def get_settings() -> Settings:
    """Return a singleton Settings instance."""
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings()
    return _settings_instance