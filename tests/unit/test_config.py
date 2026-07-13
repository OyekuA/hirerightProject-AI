
import os
from contextlib import contextmanager

from app.config import Settings


@contextmanager
def _set_env(**overrides):
    prev = {}
    for k, v in overrides.items():
        prev[k] = os.environ.get(k)
        os.environ[k] = str(v) if v is not None else ""
    try:
        yield
    finally:
        for k, v in prev.items():
            if v is None:
                del os.environ[k]
            else:
                os.environ[k] = v


_REQUIRED = dict(
    API_KEY="test-api-key",
    QDRANT_HOST="qdrant",
    QDRANT_PORT="6333",
    INGEST_STATUS_STORE_PATH="/tmp/test_store",
    CALLBACK_HMAC_SECRET="test-secret",
    GENERATION_BREAKER_COOLDOWN_SECONDS="60",
    EMBEDDING_BREAKER_COOLDOWN_SECONDS="60",
    RECALL_AI_API_KEY="test-recall-key",
    RECALL_AI_WEBHOOK_SECRET="test-webhook-secret",
    INTERVIEW_SESSION_STORE_PATH="/tmp/test_interview_sessions",
)


class TestSettingsBootstrap:

    def test_bootstrap_with_llm_api_key(self):
        with _set_env(LLM_API_KEY="some-llm-key", **_REQUIRED):
            settings = Settings()
            assert settings.LLM_API_KEY == "some-llm-key"

    def test_bootstrap_without_llm_api_key(self):
        with _set_env(LLM_API_KEY="", **_REQUIRED):
            settings = Settings()
            assert settings.LLM_API_KEY in (None, "")

    def test_bootstrap_with_openai_only(self):
        with _set_env(
            LLM_MODEL="openai/gpt-4o",
            EMBEDDING_MODEL="openai/text-embedding-3-small",
            EMBEDDING_DIMENSIONS="1536",
            **_REQUIRED,
        ):
            settings = Settings()
            assert settings.LLM_MODEL == "openai/gpt-4o"
            assert settings.EMBEDDING_MODEL == "openai/text-embedding-3-small"
            assert settings.EMBEDDING_DIMENSIONS == 1536

    def test_bootstrap_with_anthropic_only(self):
        with _set_env(
            LLM_MODEL="anthropic/claude-3-5-sonnet-20241022",
            EMBEDDING_MODEL="gemini/text-embedding-004",
            EMBEDDING_DIMENSIONS="768",
            **_REQUIRED,
        ):
            settings = Settings()
            assert settings.LLM_MODEL == "anthropic/claude-3-5-sonnet-20241022"

    def test_bootstrap_with_embedding_dimensions(self):
        with _set_env(EMBEDDING_DIMENSIONS="1024", **_REQUIRED):
            settings = Settings()
            assert settings.EMBEDDING_DIMENSIONS == 1024

    def test_log_level_default_is_info(self):
        with _set_env(**_REQUIRED):
            settings = Settings()
            assert settings.LOG_LEVEL == "INFO"

    def test_docs_settings_default_to_disabled(self):
        with _set_env(ENABLE_DOCS="false", DOCS_USERNAME="", DOCS_PASSWORD="", **_REQUIRED):
            settings = Settings()
            assert settings.ENABLE_DOCS is False
            assert settings.DOCS_USERNAME in (None, "")
            assert settings.DOCS_PASSWORD in (None, "")
