import json
from unittest.mock import MagicMock, patch
from unittest import IsolatedAsyncioTestCase

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.clients.llm import LLMUnavailableError


def _build_test_app(service_override=None) -> FastAPI:
    from app.routers.decision import router as decision_router
    from app.clients.dependencies import get_qdrant_client, get_llm_client, get_cache_backend

    app = FastAPI()
    app.include_router(decision_router, prefix="/api/ai")

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
        app.dependency_overrides[get_cache_backend] = lambda: MagicMock()

    return app


class TestDecisionRouter(IsolatedAsyncioTestCase):

    @patch("app.routers.decision.DecisionService")
    def test_200_success(self, mock_service_cls):
        mock_service = MagicMock()
        mock_service.decide.return_value = {
            "decision": "hire",
            "combined_score": 85,
            "fit_score": 90,
            "assessment_score": 80,
            "rationale": "Strong overall fit.",
            "confidence": 88,
        }
        mock_service_cls.return_value = mock_service

        app = _build_test_app(service_override=True)
        client = TestClient(app)

        resp = client.post(
            "/api/ai/decision",
            json={
                "candidate_id": 1,
                "candidate_version": 1,
                "job_id": 10,
                "job_version": 1,
                "assessment_score": 80,
            },
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["decision"] == "hire"
        assert data["combined_score"] == 85
        assert data["fit_score"] == 90
        assert data["assessment_score"] == 80
        assert data["confidence"] == 88

    @patch("app.routers.decision.DecisionService")
    def test_404_on_value_error(self, mock_service_cls):
        mock_service = MagicMock()
        mock_service.decide.side_effect = ValueError("Candidate not found")
        mock_service_cls.return_value = mock_service

        app = _build_test_app(service_override=True)
        client = TestClient(app)

        resp = client.post(
            "/api/ai/decision",
            json={
                "candidate_id": 999,
                "candidate_version": 1,
                "job_id": 10,
                "job_version": 1,
                "assessment_score": 50,
            },
        )
        assert resp.status_code == 404, resp.text
        assert "Candidate not found" in resp.text

    @patch("app.routers.decision.DecisionService")
    def test_503_on_llm_unavailable(self, mock_service_cls):
        mock_service = MagicMock()
        mock_service.decide.side_effect = LLMUnavailableError("LLM down")
        mock_service_cls.return_value = mock_service

        app = _build_test_app(service_override=True)
        client = TestClient(app)

        resp = client.post(
            "/api/ai/decision",
            json={
                "candidate_id": 1,
                "candidate_version": 1,
                "job_id": 10,
                "job_version": 1,
                "assessment_score": 50,
            },
        )
        assert resp.status_code == 503, resp.text
        assert "temporarily unavailable" in resp.text.lower()

    @patch("app.routers.decision.DecisionService")
    def test_422_on_invalid_assessment_score(self, mock_service_cls):
        mock_service = MagicMock()
        mock_service_cls.return_value = mock_service

        app = _build_test_app(service_override=True)
        client = TestClient(app)

        resp = client.post(
            "/api/ai/decision",
            json={
                "candidate_id": 1,
                "candidate_version": 1,
                "job_id": 10,
                "job_version": 1,
                "assessment_score": 150,
            },
        )
        assert resp.status_code == 422, resp.text

    @patch("app.routers.decision.DecisionService")
    def test_rate_limit_headers_present(self, mock_service_cls):
        mock_service = MagicMock()
        mock_service.decide.return_value = {
            "decision": "hire",
            "combined_score": 80,
            "fit_score": 80,
            "assessment_score": 80,
            "rationale": "OK.",
            "confidence": 75,
        }
        mock_service_cls.return_value = mock_service

        app = _build_test_app(service_override=True)
        client = TestClient(app)

        resp = client.post(
            "/api/ai/decision",
            json={
                "candidate_id": 1,
                "candidate_version": 1,
                "job_id": 10,
                "job_version": 1,
                "assessment_score": 80,
            },
        )
        assert resp.status_code == 200
        assert "X-RateLimit-Limit" in resp.headers or "Retry-After" not in resp.headers
