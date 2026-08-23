

import json
from unittest.mock import MagicMock, patch
from unittest import IsolatedAsyncioTestCase

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.testclient import TestClient

from app.clients.llm import LLMUnavailableError

class TestIngestionRouterBackgroundTaskWiring(IsolatedAsyncioTestCase):

    def _make_mock_req(self, **attrs):

        obj = MagicMock()
        for k, v in attrs.items():
            setattr(obj, k, v)
        return obj

    @patch("app.routers.ingestion.validate_callback_url")
    @patch("app.routers.ingestion.validate_ingest_url")
    async def test_candidate_ingestion_passes_llm_keyword(
        self,
        mock_validate: MagicMock,
        mock_validate_cb: MagicMock,
    ):

        from app.routers.ingestion import ingest_candidate

        mock_profile = MagicMock()
        mock_profile.model_dump.return_value = {
            "name": "Jane",
            "location": "Remote",
            "experience_level": "Senior",
            "industry": "Tech",
            "employment_type": "full_time",
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

        self.assertEqual(call_kwargs["candidate_id"], 42)
        self.assertEqual(call_kwargs["cv_url"], "https://example.com/cv.pdf")
        self.assertEqual(call_kwargs["event_id"], "evt_candidate_001")

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
                    "employment_type": "full_time",
                    "candidate_version": 1,
                },
            },
        )

    @patch("app.routers.ingestion.validate_callback_url")
    @patch("app.routers.ingestion.validate_ingest_url")
    async def test_candidate_ingestion_rejects_gemini_keyword(
        self,
        mock_validate: MagicMock,
        mock_validate_cb: MagicMock,
    ):

        from app.routers.ingestion import ingest_candidate

        mock_profile = MagicMock()
        mock_profile.model_dump.return_value = {
            "name": "John",
            "location": "Remote",
            "experience_level": "Mid",
            "industry": "Tech",
            "employment_type": "full_time",
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

        self.assertNotIn(
            "gemini",
            call_kwargs,
            "gemini keyword must NOT appear in add_task kwargs",
        )
        self.assertIn("llm", call_kwargs)

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
                    "employment_type": "full_time",
                    "candidate_version": 2,
                },
            },
        )

    @patch("app.routers.ingestion.validate_callback_url")
    async def test_job_ingestion_passes_llm_keyword(
        self,
        mock_validate_cb: MagicMock,
    ):

        from app.routers.ingestion import ingest_job

        mock_metadata = MagicMock()
        mock_metadata.model_dump.return_value = {
            "title": "Engineer",
            "location": "Remote",
            "experience_level": "Senior",
            "industry": "Tech",
            "employment_type": "full_time",
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

        self.assertEqual(call_kwargs["job_id"], 77)
        self.assertEqual(call_kwargs["jd_text"], "We are looking for…")
        self.assertEqual(call_kwargs["event_id"], "evt_job_001")

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
                    "employment_type": "full_time",
                    "job_version": 1,
                    "company_name": "Acme",
                    "about": "Great place",
                },
            },
        )

    @patch("app.routers.ingestion.validate_callback_url")
    async def test_job_ingestion_rejects_gemini_keyword(
        self,
        mock_validate_cb: MagicMock,
    ):

        from app.routers.ingestion import ingest_job

        mock_metadata = MagicMock()
        mock_metadata.model_dump.return_value = {
            "title": "Dev",
            "location": "NYC",
            "experience_level": "Mid",
            "industry": "FinTech",
            "employment_type": "contract",
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
                    "employment_type": "contract",
                    "job_version": 3,
                    "company_name": "Bank",
                    "about": "Finance",
                },
            },
        )

def _build_test_app(llm_client_override=None) -> FastAPI:

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
        assert data["name"] == "John Doe"
        assert data["bio"] == "Experienced engineer"
        assert len(data["experience"]) == 1
        assert data["experience"][0]["title"] == "Engineer"
        assert len(data["education"]) == 1
        assert data["education"][0]["degree"] == "BSc"
        assert data["certifications"] == ["AWS Certified"]
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

    @patch("app.routers.ingestion.validate_ingest_url")
    async def test_rejects_non_https_url(self, mock_validate: MagicMock):

        mock_llm = MagicMock()
        app = _build_test_app(llm_client_override=mock_llm)
        client = TestClient(app)

        resp = client.post(
            "/api/ai/cv-parse",
            json={"cv_url": "http://example.com/cv.pdf"},
        )
        assert resp.status_code == 422, resp.text
        assert "URL scheme must be HTTPS" in resp.text

    @patch("app.routers.ingestion.validate_ingest_url")
    @patch("app.routers.ingestion.fetch_and_parse_cv")
    async def test_fetch_parse_failure_returns_422(
        self,
        mock_fetch: MagicMock,
        mock_validate: MagicMock,
    ):

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

class TestDeleteCandidateEndpoint(IsolatedAsyncioTestCase):

    async def test_delete_candidate_calls_qdrant_delete_and_cache_clear(self):

        from app.routers.ingestion import delete_candidate

        mock_qdrant = MagicMock()
        mock_qdrant.get.return_value = {"name": "Alice", "candidate_version": 2}
        mock_cache = MagicMock()
        mock_interview_store = MagicMock()
        mock_interview_store.get_all_by_candidate_id.return_value = []

        mock_request = MagicMock(spec=Request)

        result = await delete_candidate(
            request=mock_request,
            candidate_id=42,
            qdrant=mock_qdrant,
            cache=mock_cache,
            interview_store=mock_interview_store,
        )

        self.assertTrue(result["deleted"])
        mock_qdrant.get.assert_called_once()
        mock_qdrant.delete.assert_called_once()
        mock_qdrant.delete.assert_called_with("candidates", 42)
        mock_cache.delete_by_prefix.assert_called_once_with("42:")
        mock_interview_store.get_all_by_candidate_id.assert_called_once_with(42)

    async def test_delete_candidate_purges_even_with_data_source(self):

        from app.routers.ingestion import delete_candidate

        mock_qdrant = MagicMock()
        mock_qdrant.get.return_value = {
            "name": "Alice",
            "candidate_version": 2,
            "data_source": "indeed",
        }
        mock_cache = MagicMock()
        mock_interview_store = MagicMock()
        mock_interview_store.get_all_by_candidate_id.return_value = []

        mock_request = MagicMock(spec=Request)

        result = await delete_candidate(
            request=mock_request,
            candidate_id=99,
            qdrant=mock_qdrant,
            cache=mock_cache,
            interview_store=mock_interview_store,
        )

        self.assertTrue(result["deleted"])
        mock_qdrant.get.assert_called_once()
        mock_qdrant.delete.assert_called_with("candidates", 99)
        mock_cache.delete_by_prefix.assert_called_once_with("99:")
        mock_interview_store.get_all_by_candidate_id.assert_called_once_with(99)

    async def test_delete_candidate_not_found_is_idempotent(self):

        from app.routers.ingestion import delete_candidate

        mock_qdrant = MagicMock()
        mock_qdrant.get.return_value = None
        mock_cache = MagicMock()
        mock_interview_store = MagicMock()

        mock_request = MagicMock(spec=Request)

        result = await delete_candidate(
            request=mock_request,
            candidate_id=999,
            qdrant=mock_qdrant,
            cache=mock_cache,
            interview_store=mock_interview_store,
        )
        self.assertTrue(result["deleted"])
        mock_qdrant.delete.assert_not_called()
        mock_cache.delete_by_prefix.assert_not_called()

    async def test_delete_candidate_not_found_is_idempotent_for_missing_sentinel(self):

        from app.routers.ingestion import delete_candidate
        from app.clients.qdrant import MISSING

        mock_qdrant = MagicMock()
        mock_qdrant.get.return_value = MISSING
        mock_cache = MagicMock()
        mock_interview_store = MagicMock()

        mock_request = MagicMock(spec=Request)

        result = await delete_candidate(
            request=mock_request,
            candidate_id=999,
            qdrant=mock_qdrant,
            cache=mock_cache,
            interview_store=mock_interview_store,
        )
        self.assertTrue(result["deleted"])
        mock_qdrant.delete.assert_not_called()
        mock_cache.delete_by_prefix.assert_not_called()

    async def test_delete_job_not_found_is_idempotent(self):

        from app.routers.ingestion import delete_job

        mock_qdrant = MagicMock()
        mock_qdrant.get.return_value = None
        mock_cache = MagicMock()

        mock_request = MagicMock(spec=Request)

        result = await delete_job(
            request=mock_request,
            job_id=888,
            qdrant=mock_qdrant,
            cache=mock_cache,
        )
        self.assertTrue(result["deleted"])
        mock_qdrant.delete.assert_not_called()
        mock_cache.delete_by_job_id.assert_not_called()

    async def test_delete_job_not_found_is_idempotent_for_missing_sentinel(self):

        from app.routers.ingestion import delete_job
        from app.clients.qdrant import MISSING

        mock_qdrant = MagicMock()
        mock_qdrant.get.return_value = MISSING
        mock_cache = MagicMock()

        mock_request = MagicMock(spec=Request)

        result = await delete_job(
            request=mock_request,
            job_id=888,
            qdrant=mock_qdrant,
            cache=mock_cache,
        )
        self.assertTrue(result["deleted"])
        mock_qdrant.delete.assert_not_called()
        mock_cache.delete_by_job_id.assert_not_called()
