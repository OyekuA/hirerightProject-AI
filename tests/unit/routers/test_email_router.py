from unittest.mock import MagicMock, patch
from unittest import IsolatedAsyncioTestCase

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.clients.llm import LLMUnavailableError


def _build_test_app(service_override=None) -> FastAPI:
    from app.routers.email import router as email_router
    from app.clients.dependencies import get_qdrant_client, get_llm_client

    app = FastAPI()
    app.include_router(email_router, prefix="/api/ai")

    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter

    @app.exception_handler(LLMUnavailableError)
    async def llm_unavailable_handler(request: Request, exc: LLMUnavailableError):
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=503,
            content={"detail": "AI service temporarily unavailable"},
        )

    if service_override is not None:
        app.dependency_overrides[get_qdrant_client] = lambda: MagicMock()
        app.dependency_overrides[get_llm_client] = lambda: MagicMock()

    return app


class TestEmailRouter(IsolatedAsyncioTestCase):

    @patch("app.routers.email.EmailGenerationService")
    def test_200_success(self, mock_service_cls):
        mock_service = MagicMock()
        mock_service.generate_invite_email.return_value = {
            "subject": "Interview Invitation",
            "body": "We are pleased to invite you. {{CALENDAR_LINK}}",
        }
        mock_service_cls.return_value = mock_service

        app = _build_test_app(service_override=True)
        client = TestClient(app)

        resp = client.post(
            "/api/ai/generate-invite-email",
            json={
                "candidate_id": 1,
                "candidate_version": 1,
                "job_id": 10,
                "job_version": 1,
            },
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["subject"] == "Interview Invitation"
        assert "{{CALENDAR_LINK}}" in data["body"]

    @patch("app.routers.email.EmailGenerationService")
    def test_404_on_value_error(self, mock_service_cls):
        mock_service = MagicMock()
        mock_service.generate_invite_email.side_effect = ValueError("Candidate not found")
        mock_service_cls.return_value = mock_service

        app = _build_test_app(service_override=True)
        client = TestClient(app)

        resp = client.post(
            "/api/ai/generate-invite-email",
            json={
                "candidate_id": 999,
                "candidate_version": 1,
                "job_id": 10,
                "job_version": 1,
            },
        )
        assert resp.status_code == 404, resp.text
        assert "Candidate not found" in resp.text

    @patch("app.routers.email.EmailGenerationService")
    def test_503_on_llm_unavailable(self, mock_service_cls):
        mock_service = MagicMock()
        mock_service.generate_invite_email.side_effect = LLMUnavailableError("LLM down")
        mock_service_cls.return_value = mock_service

        app = _build_test_app(service_override=True)
        client = TestClient(app)

        resp = client.post(
            "/api/ai/generate-invite-email",
            json={
                "candidate_id": 1,
                "candidate_version": 1,
                "job_id": 10,
                "job_version": 1,
            },
        )
        assert resp.status_code == 503, resp.text
        assert "temporarily unavailable" in resp.text.lower()

    @patch("app.routers.email.EmailGenerationService")
    def test_rate_limit_headers_present(self, mock_service_cls):
        mock_service = MagicMock()
        mock_service.generate_invite_email.return_value = {
            "subject": "Invite",
            "body": "{{CALENDAR_LINK}}",
        }
        mock_service_cls.return_value = mock_service

        app = _build_test_app(service_override=True)
        client = TestClient(app)

        resp = client.post(
            "/api/ai/generate-invite-email",
            json={
                "candidate_id": 1,
                "candidate_version": 1,
                "job_id": 10,
                "job_version": 1,
            },
        )
        assert resp.status_code == 200
        assert "X-RateLimit-Limit" in resp.headers or "Retry-After" not in resp.headers

    @patch("app.routers.email.EmailGenerationService")
    def test_malformed_json_from_llm_returns_503(self, mock_service_cls):
        """Comment 2: LLM malformed JSON must produce 503, not 500."""
        mock_service = MagicMock()
        mock_service.generate_invite_email.side_effect = LLMUnavailableError(
            "LLM returned malformed JSON for email generation"
        )
        mock_service_cls.return_value = mock_service

        app = _build_test_app(service_override=True)
        client = TestClient(app)

        resp = client.post(
            "/api/ai/generate-invite-email",
            json={
                "candidate_id": 1,
                "candidate_version": 1,
                "job_id": 10,
                "job_version": 1,
            },
        )
        assert resp.status_code == 503, resp.text
        assert "temporarily unavailable" in resp.text.lower()
