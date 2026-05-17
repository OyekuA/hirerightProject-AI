"""Unit tests for ingestion router background-task wiring and ``/cv-parse`` endpoint.

Verifies that ``background_tasks.add_task()`` is called with the correct
keyword argument (``llm=llm``) matching the refactored service signatures,
so that a ``gemini``→``llm`` rename regression is caught if the parameter
name changes again.

Also covers the ``/cv-parse`` endpoint contract: HTTPS-only validation,
fetch/parse failure mapping, malformed/schema-invalid LLM JSON fallback,
and ``LLMUnavailableError`` propagation as 503.
"""

import json
from unittest.mock import MagicMock, patch
from unittest import IsolatedAsyncioTestCase

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.testclient import TestClient

from app.clients.llm import LLMUnavailableError


class TestIngestionRouterBackgroundTaskWiring(IsolatedAsyncioTestCase):
    """Ensure ``background_tasks.add_task`` receives ``llm``, not ``gemini``."""

    def _make_mock_req(self, **attrs):
        """Build a generic mock request-like object."""
        obj = MagicMock()
        for k, v in attrs.items():
            setattr(obj, k, v)
        return obj

    # ── candidate ingestion ────────────────────────────────────────────

    @patch("app.routers.ingestion.validate_ingest_url")
    async def test_candidate_ingestion_passes_llm_keyword(
        self,
        mock_validate: MagicMock,
    ):
        """``add_task`` for candidate endpoint must use ``llm=llm``."""
        from app.routers.ingestion import ingest_candidate

        # -- fixtures ---------------------------------------------------
        mock_profile = MagicMock()
        mock_profile.model_dump.return_value = {
            "name": "Jane",
            "location": "Remote",
            "experience_level": "Senior",
            "industry": "Tech",
            "employment_type": "Full-time",
            "candidate_version": 1,
        }

        mock_req = self._make_mock_req(
            candidate_id=42,
            cv_url="https://example.com/cv.pdf",
            profile_data=mock_profile,
            callback_url="https://example.com/callback",
        )

        mock_store = MagicMock()
        mock_store.create.return_value = self._make_mock_req(event_id="evt_candidate_001")

        mock_bt = MagicMock(spec=BackgroundTasks)
        mock_qdrant = MagicMock()
        mock_llm = MagicMock()
        mock_callback = MagicMock()
        mock_request = MagicMock(spec=Request)

        # -- exercise ---------------------------------------------------
        await ingest_candidate(
            request=mock_request,
            req=mock_req,
            background_tasks=mock_bt,
            qdrant=mock_qdrant,
            llm=mock_llm,
            store=mock_store,
            callback_client=mock_callback,
            ingest_queue=MagicMock(),
        )

        # -- verify keyword wiring --------------------------------------
        mock_bt.add_task.assert_called_once()
        call_kwargs = mock_bt.add_task.call_args.kwargs

        self.assertIn(
            "llm",
            call_kwargs,
            "Expected 'llm' keyword in background_tasks.add_task() call",
        )
        self.assertNotIn(
            "gemini",
            call_kwargs,
            "Unexpected 'gemini' keyword — should have been renamed to 'llm'",
        )
        self.assertIs(
            call_kwargs["llm"],
            mock_llm,
            "llm keyword must reference the LLMClient dependency",
        )
        self.assertIn(
            "ingest_queue",
            call_kwargs,
            "Expected 'ingest_queue' keyword in background_tasks.add_task() call",
        )

        # -- verify other key args are present --------------------------
        self.assertEqual(call_kwargs["candidate_id"], 42)
        self.assertEqual(call_kwargs["cv_url"], "https://example.com/cv.pdf")
        self.assertEqual(call_kwargs["event_id"], "evt_candidate_001")

        # -- verify store.create payload contract -----------------------
        mock_store.create.assert_called_once_with(
            entity_type="candidate",
            entity_id=42,
            callback_url="https://example.com/callback",
            payload={
                "cv_url": "https://example.com/cv.pdf",
                "profile_data": {
                    "name": "Jane",
                    "location": "Remote",
                    "experience_level": "Senior",
                    "industry": "Tech",
                    "employment_type": "Full-time",
                    "candidate_version": 1,
                },
            },
        )

    @patch("app.routers.ingestion.validate_ingest_url")
    async def test_candidate_ingestion_rejects_gemini_keyword(
        self,
        mock_validate: MagicMock,
    ):
        """Guard: if ``gemini`` somehow reappears, this test must catch it."""
        from app.routers.ingestion import ingest_candidate

        mock_profile = MagicMock()
        mock_profile.model_dump.return_value = {
            "name": "John",
            "location": "Remote",
            "experience_level": "Mid",
            "industry": "Tech",
            "employment_type": "Full-time",
            "candidate_version": 2,
        }

        mock_req = self._make_mock_req(
            candidate_id=99,
            cv_url="https://example.com/cv2.pdf",
            profile_data=mock_profile,
            callback_url="https://example.com/cb",
        )
        mock_store = MagicMock()
        mock_store.create.return_value = self._make_mock_req(event_id="evt_candidate_002")

        mock_bt = MagicMock(spec=BackgroundTasks)
        mock_qdrant = MagicMock()
        mock_llm = MagicMock()
        mock_callback = MagicMock()
        mock_request = MagicMock(spec=Request)

        await ingest_candidate(
            request=mock_request,
            req=mock_req,
            background_tasks=mock_bt,
            qdrant=mock_qdrant,
            llm=mock_llm,
            store=mock_store,
            callback_client=mock_callback,
            ingest_queue=MagicMock(),
        )

        mock_bt.add_task.assert_called_once()
        call_kwargs = mock_bt.add_task.call_args.kwargs

        # Explicit negative assertion
        self.assertNotIn(
            "gemini",
            call_kwargs,
            "gemini keyword must NOT appear in add_task kwargs",
        )
        self.assertIn("llm", call_kwargs)

        # -- verify store.create payload contract -----------------------
        mock_store.create.assert_called_once_with(
            entity_type="candidate",
            entity_id=99,
            callback_url="https://example.com/cb",
            payload={
                "cv_url": "https://example.com/cv2.pdf",
                "profile_data": {
                    "name": "John",
                    "location": "Remote",
                    "experience_level": "Mid",
                    "industry": "Tech",
                    "employment_type": "Full-time",
                    "candidate_version": 2,
                },
            },
        )

    # ── job ingestion ──────────────────────────────────────────────────

    @patch("app.routers.ingestion.validate_ingest_url")
    async def test_job_ingestion_passes_llm_keyword(
        self,
        mock_validate: MagicMock,
    ):
        """``add_task`` for job endpoint must use ``llm=llm``."""
        from app.routers.ingestion import ingest_job

        # -- fixtures ---------------------------------------------------
        mock_metadata = MagicMock()
        mock_metadata.model_dump.return_value = {
            "title": "Engineer",
            "location": "Remote",
            "experience_level": "Senior",
            "industry": "Tech",
            "employment_type": "Full-time",
            "job_version": 1,
            "company_name": "Acme",
            "about": "Great place",
        }

        mock_req = self._make_mock_req(
            job_id=77,
            jd_text="We are looking for…",
            metadata=mock_metadata,
            callback_url="https://example.com/callback-job",
        )

        mock_store = MagicMock()
        mock_store.create.return_value = self._make_mock_req(event_id="evt_job_001")

        mock_bt = MagicMock(spec=BackgroundTasks)
        mock_qdrant = MagicMock()
        mock_llm = MagicMock()
        mock_callback = MagicMock()
        mock_request = MagicMock(spec=Request)

        # -- exercise ---------------------------------------------------
        await ingest_job(
            request=mock_request,
            req=mock_req,
            background_tasks=mock_bt,
            qdrant=mock_qdrant,
            llm=mock_llm,
            store=mock_store,
            callback_client=mock_callback,
            ingest_queue=MagicMock(),
        )

        # -- verify keyword wiring --------------------------------------
        mock_bt.add_task.assert_called_once()
        call_kwargs = mock_bt.add_task.call_args.kwargs

        self.assertIn(
            "llm",
            call_kwargs,
            "Expected 'llm' keyword in background_tasks.add_task() call",
        )
        self.assertNotIn(
            "gemini",
            call_kwargs,
            "Unexpected 'gemini' keyword — should have been renamed to 'llm'",
        )
        self.assertIs(
            call_kwargs["llm"],
            mock_llm,
            "llm keyword must reference the LLMClient dependency",
        )
        self.assertIn(
            "ingest_queue",
            call_kwargs,
            "Expected 'ingest_queue' keyword in background_tasks.add_task() call",
        )

        # -- verify other key args are present --------------------------
        self.assertEqual(call_kwargs["job_id"], 77)
        self.assertEqual(call_kwargs["jd_text"], "We are looking for…")
        self.assertEqual(call_kwargs["event_id"], "evt_job_001")

        # -- verify store.create payload contract -----------------------
        mock_store.create.assert_called_once_with(
            entity_type="job",
            entity_id=77,
            callback_url="https://example.com/callback-job",
            payload={
                "jd_text": "We are looking for…",
                "metadata": {
                    "title": "Engineer",
                    "location": "Remote",
                    "experience_level": "Senior",
                    "industry": "Tech",
                    "employment_type": "Full-time",
                    "job_version": 1,
                    "company_name": "Acme",
                    "about": "Great place",
                },
            },
        )

    @patch("app.routers.ingestion.validate_ingest_url")
    async def test_job_ingestion_rejects_gemini_keyword(
        self,
        mock_validate: MagicMock,
    ):
        """Guard: if ``gemini`` somehow reappears, this test must catch it."""
        from app.routers.ingestion import ingest_job

        mock_metadata = MagicMock()
        mock_metadata.model_dump.return_value = {
            "title": "Dev",
            "location": "NYC",
            "experience_level": "Mid",
            "industry": "FinTech",
            "employment_type": "Contract",
            "job_version": 3,
            "company_name": "Bank",
            "about": "Finance",
        }

        mock_req = self._make_mock_req(
            job_id=88,
            jd_text="Job description…",
            metadata=mock_metadata,
            callback_url="https://example.com/cb-job",
        )
        mock_store = MagicMock()
        mock_store.create.return_value = self._make_mock_req(event_id="evt_job_002")

        mock_bt = MagicMock(spec=BackgroundTasks)
        mock_qdrant = MagicMock()
        mock_llm = MagicMock()
        mock_callback = MagicMock()
        mock_request = MagicMock(spec=Request)

        await ingest_job(
            request=mock_request,
            req=mock_req,
            background_tasks=mock_bt,
            qdrant=mock_qdrant,
            llm=mock_llm,
            store=mock_store,
            callback_client=mock_callback,
            ingest_queue=MagicMock(),
        )

        mock_bt.add_task.assert_called_once()
        call_kwargs = mock_bt.add_task.call_args.kwargs

        self.assertNotIn(
            "gemini",
            call_kwargs,
            "gemini keyword must NOT appear in add_task kwargs",
        )
        self.assertIn("llm", call_kwargs)

        # -- verify store.create payload contract -----------------------
        mock_store.create.assert_called_once_with(
            entity_type="job",
            entity_id=88,
            callback_url="https://example.com/cb-job",
            payload={
                "jd_text": "Job description…",
                "metadata": {
                    "title": "Dev",
                    "location": "NYC",
                    "experience_level": "Mid",
                    "industry": "FinTech",
                    "employment_type": "Contract",
                    "job_version": 3,
                    "company_name": "Bank",
                    "about": "Finance",
                },
            },
        )


# ── /cv-parse endpoint tests ──────────────────────────────────────────


def _build_test_app(llm_client_override=None) -> FastAPI:
    """Build a minimal FastAPI app with the ingestion router and exception handlers.

    Overrides the ``llm`` dependency so tests can control LLM behaviour
    without hitting a real model.
    """
    from app.clients.dependencies import get_llm_client
    from app.routers.ingestion import router as ingestion_router

    app = FastAPI()
    app.include_router(ingestion_router, prefix="/api/ai")

    from slowapi import Limiter
    from slowapi.util import get_remote_address
    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter

    @app.exception_handler(LLMUnavailableError)
    async def llm_unavailable_handler(request: Request, exc: LLMUnavailableError):
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=503,
            content={"detail": "AI service temporarily unavailable"},
        )

    if llm_client_override is not None:
        app.dependency_overrides[get_llm_client] = lambda: llm_client_override

    return app


class TestCvParseEndpoint(IsolatedAsyncioTestCase):
    """Contract tests for ``POST /api/ai/cv-parse``."""

    # ── happy path ────────────────────────────────────────────────────

    @patch("app.routers.ingestion.validate_ingest_url")
    @patch("app.routers.ingestion.fetch_and_parse_cv")
    @patch("app.routers.ingestion.truncate_to_prompt_cap")
    @patch("app.routers.ingestion.CV_AUTOFILL_PROMPT_TEMPLATE")
    async def test_happy_path_returns_valid_response(
        self,
        mock_template: MagicMock,
        mock_truncate: MagicMock,
        mock_fetch: MagicMock,
        mock_validate: MagicMock,
    ):
        """Valid URL + successful LLM response returns a populated CVAutofillResponse
        with all eleven fields."""
        mock_fetch.return_value = "John Doe CV text …"
        mock_truncate.return_value = "John Doe CV text …"
        mock_template.format.return_value = "prompt"

        mock_llm = MagicMock()
        mock_llm.generate.return_value = json.dumps({
            "name": "John Doe",
            "bio": "Experienced engineer",
            "email": "john.doe@example.com",
            "phone": "+1-555-0123",
            "title": "Senior Software Engineer",
            "address": "San Francisco, CA",
            "website": "https://johndoe.dev",
            "experience": [{"title": "Engineer", "company": "Acme", "duration": "2y", "description": "Built stuff"}],
            "education": [{"degree": "BSc", "institution": "MIT", "year": "2020"}],
            "certifications": ["AWS Certified"],
            "social_links": [
                {"platform": "linkedin", "url": "https://linkedin.com/in/johndoe"},
                {"platform": "github", "url": "https://github.com/johndoe"},
            ],
        })

        app = _build_test_app(llm_client_override=mock_llm)
        client = TestClient(app)

        resp = client.post(
            "/api/ai/cv-parse",
            json={"cv_url": "https://example.com/cv.pdf"},
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        # Original five fields
        assert data["name"] == "John Doe"
        assert data["bio"] == "Experienced engineer"
        assert len(data["experience"]) == 1
        assert data["experience"][0]["title"] == "Engineer"
        assert len(data["education"]) == 1
        assert data["education"][0]["degree"] == "BSc"
        assert data["certifications"] == ["AWS Certified"]
        # New onboarding fields
        assert data["email"] == "john.doe@example.com"
        assert data["phone"] == "+1-555-0123"
        assert data["title"] == "Senior Software Engineer"
        assert data["address"] == "San Francisco, CA"
        assert data["website"] == "https://johndoe.dev"
        assert len(data["social_links"]) == 2
        assert data["social_links"][0]["platform"] == "linkedin"
        assert data["social_links"][0]["url"] == "https://linkedin.com/in/johndoe"
        assert data["social_links"][1]["platform"] == "github"
        assert data["social_links"][1]["url"] == "https://github.com/johndoe"

    # ── invalid / non-HTTPS URL ───────────────────────────────────────

    @patch("app.routers.ingestion.validate_ingest_url")
    async def test_rejects_non_https_url(self, mock_validate: MagicMock):
        """Non-HTTPS URL is rejected with 422 (Pydantic ``HttpUrl`` validator catches it first)."""
        mock_llm = MagicMock()
        app = _build_test_app(llm_client_override=mock_llm)
        client = TestClient(app)

        resp = client.post(
            "/api/ai/cv-parse",
            json={"cv_url": "http://example.com/cv.pdf"},
        )
        assert resp.status_code == 422, resp.text
        # Pydantic's HttpUrl validator rejects non-HTTPS before validate_ingest_url runs
        assert "URL scheme must be HTTPS" in resp.text

    # ── fetch / parse failure → 422 ───────────────────────────────────

    @patch("app.routers.ingestion.validate_ingest_url")
    @patch("app.routers.ingestion.fetch_and_parse_cv")
    async def test_fetch_parse_failure_returns_422(
        self,
        mock_fetch: MagicMock,
        mock_validate: MagicMock,
    ):
        """``ValueError`` / ``RuntimeError`` from ``fetch_and_parse_cv`` maps to 422."""
        mock_fetch.side_effect = ValueError("CV exceeds size limit")

        mock_llm = MagicMock()
        app = _build_test_app(llm_client_override=mock_llm)
        client = TestClient(app)

        resp = client.post(
            "/api/ai/cv-parse",
            json={"cv_url": "https://example.com/cv.pdf"},
        )
        assert resp.status_code == 422, resp.text
        assert "CV exceeds size limit" in resp.text


    @patch("app.routers.ingestion.validate_ingest_url")
    @patch("app.routers.ingestion.fetch_and_parse_cv")
    @patch("app.routers.ingestion.truncate_to_prompt_cap")
    @patch("app.routers.ingestion.CV_AUTOFILL_PROMPT_TEMPLATE")
    async def test_malformed_llm_json_returns_empty_response(
        self,
        mock_template: MagicMock,
        mock_truncate: MagicMock,
        mock_fetch: MagicMock,
        mock_validate: MagicMock,
    ):
        mock_fetch.return_value = "CV text"
        mock_truncate.return_value = "CV text"
        mock_template.format.return_value = "prompt"

        mock_llm = MagicMock()
        mock_llm.generate.return_value = "not valid json at all"

        app = _build_test_app(llm_client_override=mock_llm)
        client = TestClient(app)

        resp = client.post(
            "/api/ai/cv-parse",
            json={"cv_url": "https://example.com/cv.pdf"},
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        # All fields should be empty / None
        assert data["name"] is None
        assert data["bio"] is None
        assert data["email"] is None
        assert data["phone"] is None
        assert data["title"] is None
        assert data["address"] is None
        assert data["website"] is None
        assert data["experience"] == []
        assert data["education"] == []
        assert data["certifications"] == []
        assert data["social_links"] == []


    @patch("app.routers.ingestion.validate_ingest_url")
    @patch("app.routers.ingestion.fetch_and_parse_cv")
    @patch("app.routers.ingestion.truncate_to_prompt_cap")
    @patch("app.routers.ingestion.CV_AUTOFILL_PROMPT_TEMPLATE")
    async def test_schema_validation_fallback_returns_empty_response(
        self,
        mock_template: MagicMock,
        mock_truncate: MagicMock,
        mock_fetch: MagicMock,
        mock_validate: MagicMock,
    ):
        mock_fetch.return_value = "CV text"
        mock_truncate.return_value = "CV text"
        mock_template.format.return_value = "prompt"

        mock_llm = MagicMock()
        mock_llm.generate.return_value = json.dumps({"unexpected_field": "value"})

        app = _build_test_app(llm_client_override=mock_llm)
        client = TestClient(app)

        resp = client.post(
            "/api/ai/cv-parse",
            json={"cv_url": "https://example.com/cv.pdf"},
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["name"] is None
        assert data["bio"] is None
        assert data["email"] is None
        assert data["phone"] is None
        assert data["title"] is None
        assert data["address"] is None
        assert data["website"] is None
        assert data["experience"] == []
        assert data["education"] == []
        assert data["certifications"] == []
        assert data["social_links"] == []

    # ── malformed social_links resilience ──────────────────────────────

    @patch("app.routers.ingestion.validate_ingest_url")
    @patch("app.routers.ingestion.fetch_and_parse_cv")
    @patch("app.routers.ingestion.truncate_to_prompt_cap")
    @patch("app.routers.ingestion.CV_AUTOFILL_PROMPT_TEMPLATE")
    async def test_malformed_social_links_preserves_valid_fields(
        self,
        mock_template: MagicMock,
        mock_truncate: MagicMock,
        mock_fetch: MagicMock,
        mock_validate: MagicMock,
    ):
        """When ``social_links`` is malformed (bare URL list), the endpoint
        still returns the other valid fields instead of an empty response."""
        mock_fetch.return_value = "CV text"
        mock_truncate.return_value = "CV text"
        mock_template.format.return_value = "prompt"

        mock_llm = MagicMock()
        mock_llm.generate.return_value = json.dumps({
            "name": "Jane Smith",
            "bio": "A seasoned developer",
            "email": "jane@example.com",
            "phone": "+1-555-9999",
            "title": "Full Stack Developer",
            "address": "New York, NY",
            "website": "https://janesmith.dev",
            "experience": [{"title": "Dev", "company": "Corp", "duration": "3y", "description": "Built things"}],
            "education": [{"degree": "MSc", "institution": "Stanford", "year": "2018"}],
            "certifications": ["Google Cloud Certified"],
            # social_links is a bare list of URL strings, not objects
            "social_links": ["https://linkedin.com/in/janesmith", "https://github.com/janesmith"],
        })

        app = _build_test_app(llm_client_override=mock_llm)
        client = TestClient(app)

        resp = client.post(
            "/api/ai/cv-parse",
            json={"cv_url": "https://example.com/cv.pdf"},
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        # Valid fields must be preserved
        assert data["name"] == "Jane Smith"
        assert data["bio"] == "A seasoned developer"
        assert data["email"] == "jane@example.com"
        assert data["phone"] == "+1-555-9999"
        assert data["title"] == "Full Stack Developer"
        assert data["address"] == "New York, NY"
        assert data["website"] == "https://janesmith.dev"
        assert len(data["experience"]) == 1
        assert data["education"][0]["degree"] == "MSc"
        assert data["certifications"] == ["Google Cloud Certified"]
        assert data["social_links"] == []

    # ── LLMUnavailableError → 503 ─────────────────────────────────────

    @patch("app.routers.ingestion.validate_ingest_url")
    @patch("app.routers.ingestion.fetch_and_parse_cv")
    @patch("app.routers.ingestion.truncate_to_prompt_cap")
    @patch("app.routers.ingestion.CV_AUTOFILL_PROMPT_TEMPLATE")
    async def test_llm_unavailable_returns_503(
        self,
        mock_template: MagicMock,
        mock_truncate: MagicMock,
        mock_fetch: MagicMock,
        mock_validate: MagicMock,
    ):
        """``LLMUnavailableError`` from ``llm.generate`` propagates as 503."""
        mock_fetch.return_value = "CV text"
        mock_truncate.return_value = "CV text"
        mock_template.format.return_value = "prompt"

        mock_llm = MagicMock()
        mock_llm.generate.side_effect = LLMUnavailableError("Circuit breaker is open")

        app = _build_test_app(llm_client_override=mock_llm)
        client = TestClient(app)

        resp = client.post(
            "/api/ai/cv-parse",
            json={"cv_url": "https://example.com/cv.pdf"},
        )
        assert resp.status_code == 503, resp.text
        assert "AI service temporarily unavailable" in resp.text
