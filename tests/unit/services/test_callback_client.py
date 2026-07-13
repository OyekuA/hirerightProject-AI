import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from app.services.callback_client import CallbackClient


@pytest.fixture
def client():
    return CallbackClient(
        hmac_secret="test-secret",
        max_attempts=2,
        retry_base_seconds=0,
        timeout_seconds=5,
        signature_ttl_seconds=300,
    )


def _make_async_client_mock(mock_post_resp=None):
    """Create a proper async context-manager mock for httpx.AsyncClient."""
    client = AsyncMock(spec=["post"])
    if mock_post_resp:
        client.post = AsyncMock(return_value=mock_post_resp)
    return client


@pytest.mark.asyncio
async def test_send_with_extra_payload(client: CallbackClient):
    """extra_payload should be merged into the signed body."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.status_code = 200

    mock_async_client = MagicMock()
    mock_async_client.__aenter__.return_value = mock_async_client
    mock_async_client.post = AsyncMock(return_value=mock_resp)

    with patch("httpx.AsyncClient", return_value=mock_async_client):
        result = await client.send(
            callback_url="https://example.com/callback",
            event_id="evt-1",
            entity_type="interview",
            entity_id=42,
            status="completed",
            error=None,
            extra_payload={"session_id": "sess-abc", "grading_result": {"score": 85}},
        )

    assert result is True
    call_kwargs = mock_async_client.post.call_args[1]
    import json
    sent_body = json.loads(call_kwargs["content"])
    assert sent_body["event_id"] == "evt-1"
    assert sent_body["session_id"] == "sess-abc"
    assert sent_body["grading_result"]["score"] == 85
    assert sent_body["entity_type"] == "interview"


@pytest.mark.asyncio
async def test_send_without_extra_payload_unaffected(client: CallbackClient):
    """Existing callers that don't pass extra_payload should work as before."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.status_code = 200

    mock_async_client = MagicMock()
    mock_async_client.__aenter__.return_value = mock_async_client
    mock_async_client.post = AsyncMock(return_value=mock_resp)

    with patch("httpx.AsyncClient", return_value=mock_async_client):
        result = await client.send(
            callback_url="https://example.com/callback",
            event_id="evt-2",
            entity_type="candidate",
            entity_id=7,
            status="success",
            error=None,
        )

    assert result is True
    call_kwargs = mock_async_client.post.call_args[1]
    import json
    sent_body = json.loads(call_kwargs["content"])
    assert sent_body["event_id"] == "evt-2"
    assert sent_body["entity_type"] == "candidate"
    assert sent_body["entity_id"] == 7
    # extra_payload keys should NOT be present
    assert "session_id" not in sent_body
