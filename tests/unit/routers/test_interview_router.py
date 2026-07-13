import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from app.clients.dependencies import (
    get_meeting_bot_client,
    get_interview_session_store,
    get_rate_limiter,
)
from app.clients.meeting_bot import RecallAIClient
from app.routers.interview import router


def _build_test_app(overrides: dict = None) -> FastAPI:
    from slowapi import Limiter
    from slowapi.util import get_remote_address
    from fastapi.responses import JSONResponse
    from slowapi.errors import RateLimitExceeded

    app = FastAPI()
    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter

    async def _rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded"},
        )

    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # Simulate verify_api_key as a no-op for these tests
    def _fake_verify(request: Request):
        return True

    app.include_router(router, prefix="/api/ai", dependencies=[Depends(_fake_verify)])

    if overrides:
        for dep, mock_fn in overrides.items():
            app.dependency_overrides[dep] = mock_fn

    return app


from fastapi import Depends


class TestInterviewRouter:

    def test_start_interview_uses_same_session_id(self):
        """The session ID returned to caller must match the ID sent to Recall metadata
        and stored on disk — not a different auto-generated ID."""
        mock_bot = MagicMock()
        mock_bot.inject_bot = AsyncMock(return_value="bot-injected-id")

        mock_store = MagicMock()
        captured_session_id = None

        def _create_side_effect(**kwargs):
            nonlocal captured_session_id
            captured_session_id = kwargs.get("session_id")
            mock_record = MagicMock()
            mock_record.session_id = kwargs.get("session_id")
            return mock_record

        mock_store.create = MagicMock(side_effect=_create_side_effect)

        app = _build_test_app(overrides={
            get_meeting_bot_client: lambda: mock_bot,
            get_interview_session_store: lambda: mock_store,
        })

        with patch("app.routers.interview.validate_callback_url"):
            with TestClient(app) as tc:
                resp = tc.post(
                    "/api/ai/interview/start",
                    json={
                        "meeting_url": "https://zoom.us/j/123",
                        "job_id": 10,
                        "candidate_id": 42,
                        "rubric": ["Communication", "Technical"],
                        "callback_url": "https://example.com/callback",
                    },
                )

        assert resp.status_code == 202
        data = resp.json()
        returned_sid = data["session_id"]

        # The returned session_id must match what was sent to store.create
        assert captured_session_id == returned_sid, (
            f"Session ID mismatch: store received {captured_session_id}, "
            f"but API returned {returned_sid}. "
            "The same UUID must be used for both the Recall metadata and the persisted record."
        )

    def test_get_interview_status_found(self):
        mock_store = MagicMock()
        mock_record = MagicMock()
        mock_record.session_id = "sess-1"
        mock_record.status = "completed"
        mock_record.result = {"score": 85}
        mock_record.candidate_id = 42
        mock_record.job_id = 10
        mock_store.get_by_session_id.return_value = mock_record

        app = _build_test_app(overrides={
            get_interview_session_store: lambda: mock_store,
        })

        with TestClient(app) as tc:
            resp = tc.get("/api/ai/interview/sess-1")

        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == "sess-1"
        assert data["status"] == "completed"
        assert data["result"]["score"] == 85

    def test_get_interview_status_not_found(self):
        mock_store = MagicMock()
        mock_store.get_by_session_id.return_value = None

        app = _build_test_app(overrides={
            get_interview_session_store: lambda: mock_store,
        })

        with TestClient(app) as tc:
            resp = tc.get("/api/ai/interview/nonexistent")

        assert resp.status_code == 404

    def test_start_interview_unsupported_region_returns_500(self):
        """An unsupported RECALL_AI_REGION should yield a controlled 500, not a raw crash."""
        mock_store = MagicMock()
        mock_record = MagicMock()
        mock_record.session_id = "sess-err"
        mock_store.create.return_value = mock_record

        app = _build_test_app(overrides={
            get_interview_session_store: lambda: mock_store,
        })

        with patch.object(RecallAIClient, "__init__", side_effect=ValueError("Unsupported Recall.ai region: mars-1")):
            with patch("app.routers.interview.validate_callback_url"):
                with TestClient(app) as tc:
                    resp = tc.post(
                        "/api/ai/interview/start",
                        json={
                            "meeting_url": "https://zoom.us/j/123",
                            "job_id": 10,
                            "candidate_id": 42,
                            "rubric": ["Communication"],
                            "callback_url": "https://example.com/callback",
                        },
                    )

        assert resp.status_code == 500
        detail = resp.json().get("detail", "")
        assert "Unsupported Recall.ai region" in detail

    def test_delete_interview_session(self):
        mock_store = MagicMock()
        mock_record = MagicMock()
        mock_record.session_id = "sess-del"
        mock_store.get_by_session_id.return_value = mock_record
        mock_store.delete.return_value = True

        app = _build_test_app(overrides={
            get_interview_session_store: lambda: mock_store,
        })

        with TestClient(app) as tc:
            resp = tc.delete("/api/ai/interview/sess-del")

        assert resp.status_code == 200
        assert resp.json() == {"deleted": True}
        mock_store.delete.assert_called_once_with("sess-del")

    def test_delete_interview_session_not_found(self):
        mock_store = MagicMock()
        mock_store.get_by_session_id.return_value = None

        app = _build_test_app(overrides={
            get_interview_session_store: lambda: mock_store,
        })

        with TestClient(app) as tc:
            resp = tc.delete("/api/ai/interview/nonexistent")

        assert resp.status_code == 404
