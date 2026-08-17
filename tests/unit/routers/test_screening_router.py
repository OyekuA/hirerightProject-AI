

import json
from unittest.mock import MagicMock, patch, ANY
from unittest import IsolatedAsyncioTestCase

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from app.clients.llm import LLMUnavailableError

def _build_test_app(**overrides) -> FastAPI:

    from app.routers.screening import router as screening_router

    app = FastAPI()
    app.include_router(screening_router, prefix="/api/ai")

    from slowapi import Limiter
    from slowapi.util import get_remote_address
    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter

    from app.clients.dependencies import (
        get_qdrant_client,
        get_llm_client,
        get_cache_backend,
        get_callback_client,
        get_screening_store,
    )

    if "qdrant" in overrides:
        app.dependency_overrides[get_qdrant_client] = lambda: overrides["qdrant"]
    if "llm" in overrides:
        app.dependency_overrides[get_llm_client] = lambda: overrides["llm"]
    if "cache" in overrides:
        app.dependency_overrides[get_cache_backend] = lambda: overrides["cache"]
    if "store" in overrides:
        app.dependency_overrides[get_screening_store] = lambda: overrides["store"]
    if "callback" in overrides:
        app.dependency_overrides[get_callback_client] = lambda: overrides["callback"]

    return app

class TestScreeningRouterXorValidation(IsolatedAsyncioTestCase):

    def setUp(self):
        self.mock_qdrant = MagicMock()
        self.mock_llm = MagicMock()
        self.mock_cache = MagicMock()
        self.mock_store = MagicMock()
        self.mock_callback = MagicMock()

    def test_requires_either_job_id_or_jd_text(self):

        app = _build_test_app(
            qdrant=self.mock_qdrant,
            llm=self.mock_llm,
            cache=self.mock_cache,
            store=self.mock_store,
            callback=self.mock_callback,
        )
        client = TestClient(app)

        resp = client.post(
            "/api/ai/screen-batch",
            json={
                "candidates": [
                    {"candidate_ref": "c1", "cv_url": "https://example.com/cv.pdf"}
                ],
            },
        )
        self.assertEqual(resp.status_code, 422, resp.text)
        self.assertIn("Provide either", resp.text)

    def test_rejects_both_job_id_and_jd_text(self):

        app = _build_test_app(
            qdrant=self.mock_qdrant,
            llm=self.mock_llm,
            cache=self.mock_cache,
            store=self.mock_store,
            callback=self.mock_callback,
        )
        client = TestClient(app)

        resp = client.post(
            "/api/ai/screen-batch",
            json={
                "job_id": 1,
                "jd_text": "We need an engineer",
                "candidates": [
                    {"candidate_ref": "c1", "cv_url": "https://example.com/cv.pdf"}
                ],
            },
        )
        self.assertEqual(resp.status_code, 422, resp.text)
        self.assertIn("not both", resp.text)

    def test_requires_job_version_with_job_id(self):

        app = _build_test_app(
            qdrant=self.mock_qdrant,
            llm=self.mock_llm,
            cache=self.mock_cache,
            store=self.mock_store,
            callback=self.mock_callback,
        )
        client = TestClient(app)

        resp = client.post(
            "/api/ai/screen-batch",
            json={
                "job_id": 1,
                "candidates": [
                    {"candidate_ref": "c1", "cv_url": "https://example.com/cv.pdf"}
                ],
            },
        )
        self.assertEqual(resp.status_code, 422, resp.text)
        self.assertIn("job_version is required", resp.text)

    @patch("app.config.get_settings")
    @patch("app.routers.screening.BulkScreeningService.resolve_job_payload")
    @patch("app.routers.screening.validate_ingest_url")
    @patch("app.routers.screening.validate_callback_url")
    async def test_batch_size_ceiling_rejection(
        self,
        mock_validate_cb,
        mock_validate_url,
        mock_resolve,
        mock_get_settings,
    ):

        mock_settings = MagicMock()
        mock_settings.SCREENING_MAX_BATCH_SIZE = 2  # Low ceiling for test
        mock_get_settings.return_value = mock_settings

        app = _build_test_app(
            qdrant=self.mock_qdrant,
            llm=self.mock_llm,
            cache=self.mock_cache,
            store=self.mock_store,
            callback=self.mock_callback,
        )
        client = TestClient(app)

        resp = client.post(
            "/api/ai/screen-batch",
            json={
                "jd_text": "We need an engineer",
                "candidates": [
                    {"candidate_ref": "c1", "cv_url": "https://example.com/cv1.pdf"},
                    {"candidate_ref": "c2", "cv_url": "https://example.com/cv2.pdf"},
                    {"candidate_ref": "c3", "cv_url": "https://example.com/cv3.pdf"},
                ],
            },
        )
        self.assertEqual(resp.status_code, 422, resp.text)
        self.assertIn("exceeds maximum", resp.text)

    def test_rejects_job_metadata_without_jd_text(self):

        app = _build_test_app(
            qdrant=self.mock_qdrant,
            llm=self.mock_llm,
            cache=self.mock_cache,
            store=self.mock_store,
            callback=self.mock_callback,
        )
        client = TestClient(app)

        resp = client.post(
            "/api/ai/screen-batch",
            json={
                "job_metadata": {
                    "title": "Engineer",
                    "location": "Remote",
                    "experience_level": "Senior",
                    "industry": "Tech",
                    "employment_type": "full_time",
                    "job_version": 1,
                },
                "candidates": [
                    {"candidate_ref": "c1", "cv_url": "https://example.com/cv.pdf"}
                ],
            },
        )
        self.assertEqual(resp.status_code, 422, resp.text)
        self.assertIn("jd_text is required", resp.text)

    def test_rejects_http_cv_url(self):

        app = _build_test_app(
            qdrant=self.mock_qdrant,
            llm=self.mock_llm,
            cache=self.mock_cache,
            store=self.mock_store,
            callback=self.mock_callback,
        )
        client = TestClient(app)

        resp = client.post(
            "/api/ai/screen-batch",
            json={
                "jd_text": "We need an engineer",
                "candidates": [
                    {"candidate_ref": "c1", "cv_url": "http://example.com/cv.pdf"}
                ],
            },
        )
        self.assertEqual(resp.status_code, 422, resp.text)
        self.assertIn("URL scheme must be HTTPS", resp.text)

class TestScreeningRouter202Flow(IsolatedAsyncioTestCase):

    def setUp(self):
        self.mock_qdrant = MagicMock()
        self.mock_llm = MagicMock()
        self.mock_cache = MagicMock()
        self.mock_store = MagicMock()
        self.mock_callback = MagicMock()

    @patch("app.routers.screening.validate_ingest_url")
    @patch("app.routers.screening.validate_callback_url")
    @patch("app.routers.screening.BulkScreeningService.resolve_job_payload")
    async def test_returns_202_with_batch_id(
        self,
        mock_resolve,
        mock_validate_cb,
        mock_validate_url,
    ):

        mock_resolve.return_value = {
            "title": "Engineer",
            "location": "Remote",
            "experience_level": "Senior",
            "industry": "Tech",
            "employment_type": "Full-time",
            "required_skills": ["Python"],
            "raw_jd_summary": "Summary",
        }

        self.mock_store.create.return_value = MagicMock(batch_id="test-batch-uuid")

        app = _build_test_app(
            qdrant=self.mock_qdrant,
            llm=self.mock_llm,
            cache=self.mock_cache,
            store=self.mock_store,
            callback=self.mock_callback,
        )
        client = TestClient(app)

        resp = client.post(
            "/api/ai/screen-batch",
            json={
                "jd_text": "We need an engineer with Python skills",
                "candidates": [
                    {"candidate_ref": "c1", "cv_url": "https://example.com/cv.pdf"}
                ],
            },
        )
        self.assertEqual(resp.status_code, 202, resp.text)
        data = resp.json()
        self.assertIn("batch_id", data)
        self.assertEqual(data["batch_id"], "test-batch-uuid")

    async def test_polling_returns_404_for_missing_batch(self):

        self.mock_store.get_by_batch_id.return_value = None

        app = _build_test_app(
            qdrant=self.mock_qdrant,
            llm=self.mock_llm,
            cache=self.mock_cache,
            store=self.mock_store,
            callback=self.mock_callback,
        )
        client = TestClient(app)

        resp = client.get("/api/ai/screen-batch/non-existent")
        self.assertEqual(resp.status_code, 404, resp.text)

    @patch("app.routers.screening.validate_ingest_url")
    @patch("app.routers.screening.BulkScreeningService.resolve_job_payload")
    async def test_polling_returns_status(
        self,
        mock_resolve,
        mock_validate_url,
    ):

        mock_resolve.return_value = {"title": "Engineer"}

        self.mock_store.create.return_value = MagicMock(batch_id="test-batch-1")
        self.mock_store.get_by_batch_id.return_value = MagicMock(
            batch_id="test-batch-1",
            status="running",
            total=2,
            results=[
                {"candidate_ref": "c1", "status": "scored", "fit_score": 85,
                 "category_breakdown": None, "skill_gap_analysis": None, "error": None},
            ],
        )

        app = _build_test_app(
            qdrant=self.mock_qdrant,
            llm=self.mock_llm,
            cache=self.mock_cache,
            store=self.mock_store,
            callback=self.mock_callback,
        )
        client = TestClient(app)

        resp = client.get("/api/ai/screen-batch/test-batch-1")
        self.assertEqual(resp.status_code, 200, resp.text)
        data = resp.json()
        self.assertEqual(data["batch_id"], "test-batch-1")
        self.assertEqual(data["status"], "running")
        self.assertEqual(data["total"], 2)
        self.assertIn("completed_count", data)
        self.assertIn("results", data)

    @patch("app.routers.screening.validate_ingest_url")
    @patch("app.routers.screening.BulkScreeningService.resolve_job_payload")
    async def test_results_sorted_by_fit_score_descending(
        self,
        mock_resolve,
        mock_validate_url,
    ):

        mock_resolve.return_value = {"title": "Engineer"}

        self.mock_store.create.return_value = MagicMock(batch_id="test-batch-2")
        self.mock_store.get_by_batch_id.return_value = MagicMock(
            batch_id="test-batch-2",
            status="completed",
            total=3,
            results=[
                {"candidate_ref": "c3", "status": "scored", "fit_score": 50,
                 "category_breakdown": None, "skill_gap_analysis": None, "error": None},
                {"candidate_ref": "c1", "status": "scored", "fit_score": 90,
                 "category_breakdown": None, "skill_gap_analysis": None, "error": None},
                {"candidate_ref": "c2", "status": "failed", "fit_score": None,
                 "category_breakdown": None, "skill_gap_analysis": None, "error": "Error"},
            ],
        )

        app = _build_test_app(
            qdrant=self.mock_qdrant,
            llm=self.mock_llm,
            cache=self.mock_cache,
            store=self.mock_store,
            callback=self.mock_callback,
        )
        client = TestClient(app)

        resp = client.get("/api/ai/screen-batch/test-batch-2")
        self.assertEqual(resp.status_code, 200, resp.text)
        data = resp.json()
        results = data["results"]
        self.assertEqual(results[0]["candidate_ref"], "c1")
        self.assertEqual(results[0]["fit_score"], 90)
        self.assertEqual(results[1]["candidate_ref"], "c3")
        self.assertEqual(results[1]["fit_score"], 50)
        self.assertEqual(results[2]["candidate_ref"], "c2")
        self.assertIsNone(results[2]["fit_score"])

if __name__ == "__main__":
    unittest.main()
