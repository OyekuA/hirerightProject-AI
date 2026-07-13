import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import FastAPI, Depends
from fastapi.testclient import TestClient

from app.clients.dependencies import (
    get_meeting_bot_client,
    get_interview_session_store,
    get_callback_client,
    get_llm_client,
)
from app.routers.interview_webhook import router


def _build_test_app(overrides: dict = None) -> FastAPI:
    app = FastAPI()
    app.include_router(router, prefix="/api/ai")

    if overrides:
        for dep, mock_fn in overrides.items():
            app.dependency_overrides[dep] = mock_fn

    return app


def _make_recall_payload(event: str, bot_id: str = "bot-123", **event_data) -> dict:
    """Build a webhook payload matching the documented Recall.ai nested shape.

    Real Recall.webhook shape:
      recording.done → data.recording.id
      transcript.done → data.transcript.id
      transcript.failed → data.error
    """
    data = {
        "bot": {"id": bot_id},
    }
    data.update(event_data)
    return {
        "event": event,
        "data": data,
    }


@pytest.fixture
def mock_bot():
    bot = MagicMock()
    bot.create_transcript = AsyncMock(return_value="tr-456")
    bot.fetch_transcript = AsyncMock(
        return_value=[
            {"speaker": {"name": "Candidate"}, "words": [{"text": "Hello"}]},
        ]
    )
    return bot


@pytest.fixture
def mock_store():
    store = MagicMock()
    record = MagicMock()
    record.session_id = "sess-1"
    record.candidate_id = 42
    record.job_id = 10
    record.rubric = ["Communication"]
    record.callback_url = "https://example.com/callback"
    record.status = "pending"
    store.get_by_bot_id = MagicMock(return_value=record)
    store.update = MagicMock()
    return store


@pytest.fixture
def mock_callback():
    cb = MagicMock()
    cb.send = AsyncMock(return_value=True)
    return cb


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.generate = MagicMock(
        return_value=json.dumps({
            "per_criterion_scores": {"Communication": 85},
            "overall_score": 85,
            "strengths": ["Clear communication"],
            "red_flags": [],
            "recommendation": "hire",
        })
    )
    return llm


class TestInterviewWebhookRouter:

    def test_recording_done_reads_nested_recording_id(
        self, mock_bot, mock_store, mock_callback, mock_llm
    ):
        """recording.done must read data.recording.id (nested), not flat recording_id."""
        payload = _make_recall_payload(
            "recording.done",
            recording={"id": "rec-789"},
        )

        app = _build_test_app(overrides={
            get_meeting_bot_client: lambda: mock_bot,
            get_interview_session_store: lambda: mock_store,
            get_callback_client: lambda: mock_callback,
            get_llm_client: lambda: mock_llm,
        })

        with patch("app.routers.interview_webhook.verify_recall_webhook", return_value=payload):
            with TestClient(app) as tc:
                resp = tc.post(
                    "/api/ai/interview/webhook",
                    content=json.dumps(payload),
                    headers={"content-type": "application/json"},
                )

        assert resp.status_code == 204
        mock_bot.create_transcript.assert_awaited_once_with("rec-789")

    def test_transcript_done_reads_nested_transcript_id(
        self, mock_bot, mock_store, mock_callback, mock_llm
    ):
        """transcript.done must read data.transcript.id (nested), not flat transcript_id."""
        payload = _make_recall_payload(
            "transcript.done",
            transcript={"id": "tr-456"},
        )

        app = _build_test_app(overrides={
            get_meeting_bot_client: lambda: mock_bot,
            get_interview_session_store: lambda: mock_store,
            get_callback_client: lambda: mock_callback,
            get_llm_client: lambda: mock_llm,
        })

        with patch("app.routers.interview_webhook.verify_recall_webhook", return_value=payload):
            with TestClient(app) as tc:
                resp = tc.post(
                    "/api/ai/interview/webhook",
                    content=json.dumps(payload),
                    headers={"content-type": "application/json"},
                )

        assert resp.status_code == 204
        mock_bot.fetch_transcript.assert_awaited_once_with("tr-456")
        mock_llm.generate.assert_called_once()

    def test_transcript_failed_marks_session_failed(
        self, mock_bot, mock_store, mock_callback, mock_llm
    ):
        """transcript.failed must mark session as failed and fire failure callback."""
        payload = _make_recall_payload(
            "transcript.failed",
            error="Transcription failed due to poor audio quality",
        )

        app = _build_test_app(overrides={
            get_meeting_bot_client: lambda: mock_bot,
            get_interview_session_store: lambda: mock_store,
            get_callback_client: lambda: mock_callback,
            get_llm_client: lambda: mock_llm,
        })

        with patch("app.routers.interview_webhook.verify_recall_webhook", return_value=payload):
            with TestClient(app) as tc:
                resp = tc.post(
                    "/api/ai/interview/webhook",
                    content=json.dumps(payload),
                    headers={"content-type": "application/json"},
                )

        assert resp.status_code == 204
        update_calls = [c[0] for c in mock_store.update.call_args_list]
        assert any("sess-1" in str(c) for c in update_calls), "Store update must be called for session"
        mock_callback.send.assert_awaited_once()
        call_kwargs = mock_callback.send.call_args[1]
        assert call_kwargs["status"] == "failed"
        assert "poor audio quality" in call_kwargs["error"]

    def test_invalid_signature_returns_400(
        self, mock_bot, mock_store, mock_callback, mock_llm
    ):
        payload = _make_recall_payload("recording.done")

        app = _build_test_app(overrides={
            get_meeting_bot_client: lambda: mock_bot,
            get_interview_session_store: lambda: mock_store,
            get_callback_client: lambda: mock_callback,
            get_llm_client: lambda: mock_llm,
        })

        with patch("app.routers.interview_webhook.verify_recall_webhook", return_value=None):
            with TestClient(app) as tc:
                resp = tc.post(
                    "/api/ai/interview/webhook",
                    content=json.dumps(payload),
                    headers={"content-type": "application/json"},
                )

        assert resp.status_code == 400

    def test_unknown_event_returns_204(
        self, mock_bot, mock_store, mock_callback, mock_llm
    ):
        payload = _make_recall_payload("bot.unknown_event")

        app = _build_test_app(overrides={
            get_meeting_bot_client: lambda: mock_bot,
            get_interview_session_store: lambda: mock_store,
            get_callback_client: lambda: mock_callback,
            get_llm_client: lambda: mock_llm,
        })

        with patch("app.routers.interview_webhook.verify_recall_webhook", return_value=payload):
            with TestClient(app) as tc:
                resp = tc.post(
                    "/api/ai/interview/webhook",
                    content=json.dumps(payload),
                    headers={"content-type": "application/json"},
                )

        assert resp.status_code == 204
        mock_bot.create_transcript.assert_not_called()
        mock_bot.fetch_transcript.assert_not_called()
