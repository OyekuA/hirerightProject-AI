import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.clients.meeting_bot import RecallAIClient, MeetingBaaSClient


@pytest.mark.asyncio
async def test_recall_ai_inject_bot():
    client = RecallAIClient(api_key="test-key", region="us-east-1")
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"id": "bot-123"}
    mock_resp.raise_for_status = MagicMock()

    mock_client = MagicMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.post = AsyncMock(return_value=mock_resp)

    with patch("httpx.AsyncClient", return_value=mock_client):
        bot_id = await client.inject_bot(
            meeting_url="https://zoom.us/j/123",
            session_id="session-abc",
            candidate_id=42,
        )

    assert bot_id == "bot-123"
    call_kwargs = mock_client.post.call_args[1]
    assert "/bot/" in mock_client.post.call_args[0][0]
    assert call_kwargs["json"]["meeting_url"] == "https://zoom.us/j/123"
    assert call_kwargs["json"]["metadata"]["session_id"] == "session-abc"


@pytest.mark.asyncio
async def test_recall_ai_create_transcript():
    """create_transcript uses POST /recording/{id}/create_transcript/ with recallai_async + diarization."""
    client = RecallAIClient(api_key="test-key", region="us-east-1")
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"id": "tr-456"}
    mock_resp.raise_for_status = MagicMock()

    mock_client = MagicMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.post = AsyncMock(return_value=mock_resp)

    with patch("httpx.AsyncClient", return_value=mock_client):
        transcript_id = await client.create_transcript("rec-789")

    assert transcript_id == "tr-456"
    call_url = mock_client.post.call_args[0][0]
    assert "/recording/rec-789/create_transcript/" in call_url
    call_body = mock_client.post.call_args[1]["json"]
    assert call_body["provider"]["recallai_async"]["language_code"] == "auto"
    assert call_body["diarization"]["use_separate_streams_when_available"] is True


@pytest.mark.asyncio
async def test_recall_ai_fetch_transcript():
    """fetch_transcript reads data.download_url from nested response."""
    client = RecallAIClient(api_key="test-key", region="us-east-1")

    mock_resp1 = MagicMock()
    mock_resp1.json.return_value = {
        "data": {"download_url": "https://example.com/transcript.json"}
    }
    mock_resp1.raise_for_status = MagicMock()

    mock_resp2 = MagicMock()
    mock_resp2.json.return_value = [
        {"speaker": {"name": "Candidate"}, "words": [{"text": "Hello"}]},
    ]
    mock_resp2.raise_for_status = MagicMock()

    mock_client = MagicMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.get = AsyncMock(side_effect=[mock_resp1, mock_resp2])

    with patch("httpx.AsyncClient", return_value=mock_client):
        turns = await client.fetch_transcript("tr-456")

    call_url = mock_client.get.call_args_list[0][0][0]
    assert "/transcript/tr-456/" in call_url
    assert len(turns) == 1
    assert turns[0]["words"][0]["text"] == "Hello"


@pytest.mark.asyncio
async def test_meeting_baas_stub_raises_not_implemented():
    client = MeetingBaaSClient()
    with pytest.raises(NotImplementedError):
        await client.inject_bot(meeting_url="url", session_id="s", candidate_id=1)
    with pytest.raises(NotImplementedError):
        await client.create_transcript("rec-1")
    with pytest.raises(NotImplementedError):
        await client.fetch_transcript("tr-1")


@pytest.mark.asyncio
async def test_region_eu_central_1_resolves_correct_host():
    """eu-central-1 region maps to https://eu-central-1.recall.ai/api/v1."""
    client = RecallAIClient(api_key="test-key", region="eu-central-1")
    assert client._base_url == "https://eu-central-1.recall.ai/api/v1"


@pytest.mark.asyncio
async def test_unsupported_region_raises_value_error():
    """Unsupported region string raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported Recall.ai region: mars-1"):
        RecallAIClient(api_key="test-key", region="mars-1")


@pytest.mark.asyncio
async def test_explicit_base_url_overrides_region():
    """Explicit base_url param takes precedence over region-derived URL."""
    client = RecallAIClient(
        api_key="test-key",
        region="us-west-2",
        base_url="https://custom.example.com/api/v1",
    )
    assert client._base_url == "https://custom.example.com/api/v1"
