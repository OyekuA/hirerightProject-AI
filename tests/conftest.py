import pytest
from unittest.mock import patch, MagicMock
import os

def _is_integration_run():
    if os.getenv("INTEGRATION_TEST") == "1":
        return True
    import sys
    for arg in sys.argv:
        if "integration" in arg:
            return True
    return False

is_integration = _is_integration_run()

if not is_integration:
    os.environ.update({
        "API_KEY": "test-api-key",
        "LLM_API_KEY": "test-llm-key",
        "INGEST_STATUS_STORE_PATH": "/tmp/test_store",
        "CALLBACK_HMAC_SECRET": "test-secret",
        "GENERATION_BREAKER_COOLDOWN_SECONDS": "60",
        "EMBEDDING_BREAKER_COOLDOWN_SECONDS": "60",
    })

    import app.config
    app.config._settings_instance = None

def mock_get_settings():
    """Return a MagicMock with all required settings attributes."""
    mock = MagicMock()
    mock.CACHE_TTL_SECONDS = 86400
    mock.GENERATION_BREAKER_COOLDOWN_SECONDS = 60
    mock.EMBEDDING_BREAKER_COOLDOWN_SECONDS = 60
    mock.MAX_PROMPT_CHARS = 50000
    mock.CALLBACK_HMAC_SECRET = "test-secret"
    mock.CALLBACK_MAX_ATTEMPTS = 3
    mock.CALLBACK_RETRY_BASE_SECONDS = 0
    mock.CALLBACK_TIMEOUT_SECONDS = 10
    mock.INGEST_STATUS_STORE_PATH = "/tmp/test_store"
    mock.API_KEY = "test-api-key"
    mock.LLM_API_KEY = "test-llm-key"
    mock.LLM_MODEL = "openai/gpt-4o-mini"
    mock.EMBEDDING_MODEL = "openai/text-embedding-3-small"
    mock.EMBEDDING_DIMENSIONS = 1536
    mock.QDRANT_HOST = "qdrant"
    mock.QDRANT_PORT = 6333
    mock.SENTRY_DSN = None
    mock.CALLBACK_SIGNATURE_TTL_SECONDS = 300
    mock.MAX_INGEST_FILE_MB = 10
    mock.INGEST_FETCH_TIMEOUT_SECONDS = 20
    mock.ENFORCE_SINGLE_REPLICA = False
    mock.LOG_LEVEL = "ERROR"
    mock.LLM_GENERATION_TIMEOUT_SECONDS = 30
    mock.LLM_EMBEDDING_TIMEOUT_SECONDS = 30
    mock.LLM_MAX_RETRIES = 2
    mock.LLM_RETRY_BACKOFF_BASE_SECONDS = 1.0
    mock.INGEST_QUEUE_MAX_RETRIES = 5
    mock.INGEST_QUEUE_BACKOFF_BASE_SECONDS = 60
    mock.INGEST_QUEUE_POLL_INTERVAL_SECONDS = 30
    return mock

@pytest.fixture(scope="session", autouse=True)
def patch_settings():
    """
    Globally patch get_settings in all modules that import it,
    preventing real environment variable reads during tests.
    """
    if is_integration:
        yield
        return

    mock_settings = mock_get_settings()
    modules_to_patch = [
        "app.config.get_settings",
        "app.logging_config.get_settings",
        "app.main.get_settings",
        "app.clients.dependencies.get_settings",
        "app.clients.qdrant.get_settings",
        "app.services.scoring_service.get_settings",
        "app.services.jd_service.get_settings",
    ]
    patches = []
    for target in modules_to_patch:
        patcher = patch(target, return_value=mock_settings)
        patches.append(patcher)
    for p in patches:
        p.start()
    yield
    for p in patches:
        p.stop()