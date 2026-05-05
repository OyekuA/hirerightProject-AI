"""Unit tests for Settings bootstrap with different provider configurations."""
import os
from unittest import mock

from app.config import Settings


class TestSettingsBootstrap:
    """Verify Settings can be initialised when only the active provider's env vars are present."""

    REQUIRED_ENV = {
        "API_KEY": "test-api-key",
        "QDRANT_HOST": "qdrant",
        "QDRANT_PORT": "6333",
        "INGEST_STATUS_STORE_PATH": "/tmp/test_store",
        "CALLBACK_HMAC_SECRET": "test-secret",
        "GENERATION_BREAKER_COOLDOWN_SECONDS": "60",
        "EMBEDDING_BREAKER_COOLDOWN_SECONDS": "60",
    }

    def test_bootstrap_with_llm_api_key(self):
        """LLM_API_KEY is set when provided."""
        env = {**self.REQUIRED_ENV, "LLM_API_KEY": "some-llm-key"}
        with mock.patch.dict(os.environ, env, clear=True):
            settings = Settings()
            assert settings.LLM_API_KEY == "some-llm-key"

    def test_bootstrap_without_llm_api_key(self):
        """Settings loads successfully when LLM_API_KEY is absent."""
        env = {**self.REQUIRED_ENV}
        with mock.patch.dict(os.environ, env, clear=True):
            settings = Settings()
            assert settings.LLM_API_KEY is None

    def test_bootstrap_with_openai_only(self):
        """Settings loads when only OPENAI_API_KEY is provided (non-Gemini provider)."""
        env = {
            **self.REQUIRED_ENV,
            "OPENAI_API_KEY": "sk-openai-test",
            "LLM_MODEL": "openai/gpt-4o",
            "EMBEDDING_MODEL": "openai/text-embedding-3-small",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            settings = Settings()
            assert settings.LLM_MODEL == "openai/gpt-4o"
            assert settings.EMBEDDING_MODEL == "openai/text-embedding-3-small"
            assert settings.EMBEDDING_DIMENSIONS == 1536

    def test_bootstrap_with_anthropic_only(self):
        """Settings loads when only ANTHROPIC_API_KEY is provided."""
        env = {
            **self.REQUIRED_ENV,
            "ANTHROPIC_API_KEY": "sk-ant-test",
            "LLM_MODEL": "anthropic/claude-3-5-sonnet-20241022",
            "EMBEDDING_MODEL": "gemini/text-embedding-004",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            settings = Settings()
            assert settings.LLM_MODEL == "anthropic/claude-3-5-sonnet-20241022"

    def test_bootstrap_with_embedding_dimensions(self):
        """EMBEDDING_DIMENSIONS is configurable."""
        env = {
            **self.REQUIRED_ENV,
            "EMBEDDING_DIMENSIONS": "1024",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            settings = Settings()
            assert settings.EMBEDDING_DIMENSIONS == 1024
