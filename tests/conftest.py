import asyncio
import inspect
import os
import pytest
from unittest.mock import patch, MagicMock

asyncio.iscoroutinefunction = inspect.iscoroutinefunction

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
        "RECALL_AI_API_KEY": "test-recall-key",
        "RECALL_AI_REGION": "us-east-1",
        "RECALL_AI_WEBHOOK_SECRET": "test-webhook-secret",
        "INTERVIEW_SESSION_STORE_PATH": "/tmp/test_interview_sessions",
    })

    import app.config
    app.config._settings_instance = None

def mock_get_settings():
    mock = MagicMock()
    mock.CACHE_TTL_SECONDS = 86400
    mock.GENERATION_BREAKER_COOLDOWN_SECONDS = 60
    mock.EMBEDDING_BREAKER_COOLDOWN_SECONDS = 60
    mock.MAX_PROMPT_CHARS = 50000
    mock.EMBED_MAX_CHARS = 32000
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
    mock.LOG_LEVEL = "INFO"
    mock.ENABLE_DOCS = False
    mock.DOCS_USERNAME = None
    mock.DOCS_PASSWORD = None
    mock.LLM_GENERATION_TIMEOUT_SECONDS = 30
    mock.LLM_EMBEDDING_TIMEOUT_SECONDS = 30
    mock.LLM_MAX_RETRIES = 2
    mock.LLM_RETRY_BACKOFF_BASE_SECONDS = 1.0
    mock.LLM_JSON_MODE_ENABLED = True
    mock.LLM_GENERATION_MAX_TOKENS = 4096
    mock.LLM_SEED = None
    mock.INGEST_QUEUE_MAX_RETRIES = 5
    mock.INGEST_QUEUE_BACKOFF_BASE_SECONDS = 60
    mock.INGEST_QUEUE_POLL_INTERVAL_SECONDS = 30
    mock.DEAD_LETTER_POLL_INTERVAL_SECONDS = 300
    mock.SCORING_WEIGHT_SKILLS = 0.35
    mock.SCORING_WEIGHT_ROLE = 0.25
    mock.SCORING_WEIGHT_EXPERIENCE = 0.20
    mock.SCORING_WEIGHT_LOCATION = 0.12
    mock.SCORING_WEIGHT_EMPLOYMENT = 0.08
    mock.SCORING_STATUS_PASS_THRESHOLD = 75
    mock.SCORING_STATUS_WARNING_THRESHOLD = 50
    mock.DECISION_FIT_WEIGHT = 0.40
    mock.DECISION_ASSESSMENT_WEIGHT = 0.60

    mock.SCREENING_MAX_BATCH_SIZE = 1000
    mock.SCREENING_CONCURRENCY = 10

    mock.RECALL_AI_API_KEY = "test-recall-key"
    mock.RECALL_AI_REGION = "us-east-1"
    mock.RECALL_AI_WEBHOOK_SECRET = "test-webhook-secret"
    mock.INTERVIEW_SESSION_STORE_PATH = "/tmp/test_interview_sessions"

    mock.RECOMMEND_WEIGHT_VECTOR = 0.55
    mock.RECOMMEND_WEIGHT_SKILL = 0.35
    mock.RECOMMEND_WEIGHT_LOCATION = 0.04
    mock.RECOMMEND_WEIGHT_LEVEL = 0.04
    mock.RECOMMEND_WEIGHT_EMPLOYMENT = 0.02
    mock.RAW_COSINE_GATE = 0.30
    mock.SKILL_COSINE_GATE = 0.40
    mock.RECOMMEND_SKILL_RESCALE_LO = 0.30
    mock.RECOMMEND_SKILL_RESCALE_HI = 1.0
    mock.LEVEL_GATE_DISTANCE = 4
    mock.RECOMMEND_MAX_SEARCHES = 5
    mock.RECOMMEND_MAX_COOC_IDS = 20
    return mock

@pytest.fixture(scope="session", autouse=True)
def patch_settings():
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
        "app.services.recommendation_service.get_settings",
        "app.services.screening_service.get_settings",
        "app.utils.ingestion.get_settings",
        "app.routers.interview_webhook.get_settings",
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