"""Unit tests for rate‑limit enforcement on all router endpoints.
"""

import json
from unittest.mock import MagicMock, patch
from unittest import IsolatedAsyncioTestCase

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from app.clients.llm import LLMUnavailableError


def _reset_rate_limiter():
    from app.clients.dependencies import get_rate_limiter
    get_rate_limiter().limiter.reset()


def _build_test_app(router, llm_client_override=None, dependency_overrides=None) -> FastAPI:
    """Build a minimal FastAPI app with the given router and rate‑limiter.

    Uses a per‑test Limiter so rate‑limit counters are isolated per test method.
    """
    from app.clients.dependencies import get_llm_client
    from slowapi import Limiter
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    from fastapi.responses import JSONResponse

    app = FastAPI()
    app.include_router(router, prefix="/api/ai")

    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter

    @app.exception_handler(RateLimitExceeded)
    async def rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
        limit = exc.limit
        retry_after = limit.get_expiry() if hasattr(limit, "get_expiry") else 60
        correlation_id = getattr(request.state, "correlation_id", None)

        return JSONResponse(
            status_code=429,
            content={
                "detail": "Rate limit exceeded",
                "error_code": "RATE_LIMIT_EXCEEDED",
                "retry_after_seconds": retry_after,
                "correlation_id": correlation_id,
            },
            headers={
                "Retry-After": str(retry_after),
                "X-Correlation-Id": correlation_id or "",
            },
        )

    @app.exception_handler(LLMUnavailableError)
    async def llm_unavailable_handler(request: Request, exc: LLMUnavailableError):
        return JSONResponse(
            status_code=503,
            content={"detail": "AI service temporarily unavailable"},
        )

    if llm_client_override is not None:
        app.dependency_overrides[get_llm_client] = lambda: llm_client_override

    if dependency_overrides:
        for dep, mock in dependency_overrides.items():
            app.dependency_overrides[dep] = lambda m=mock: m

    return app


class TestCvParseBurstLimit(IsolatedAsyncioTestCase):
    """Rate‑limit enforcement for POST /cv-parse (10/minute burst)."""

    def setUp(self):
        _reset_rate_limiter()
        from app.routers.ingestion import router as ingestion_router
        self.mock_llm = MagicMock()
        self.mock_llm.generate.return_value = json.dumps({"name": "Test"})
        self.app = _build_test_app(ingestion_router, llm_client_override=self.mock_llm)
        self.client = TestClient(self.app)

    @patch("app.routers.ingestion.validate_ingest_url")
    @patch("app.routers.ingestion.fetch_and_parse_cv")
    @patch("app.routers.ingestion.truncate_to_prompt_cap")
    def test_burst_limit_exceeded(self, mock_truncate, mock_fetch, mock_validate):
        """POST /cv-parse has a 10/minute burst; the 11th request should 429."""
        mock_validate.return_value = None
        mock_fetch.return_value = "parsed cv text"
        mock_truncate.return_value = "truncated cv text"

        for i in range(10):
            resp = self.client.post(
                "/api/ai/cv-parse",
                json={"cv_url": "https://example.com/cv.pdf"},
            )
            self.assertEqual(
                resp.status_code, 200,
                msg=f"Request {i+1} should be 200, got {resp.status_code}: {resp.text}",
            )

        # The 11th request should be rate‑limited
        resp = self.client.post(
            "/api/ai/cv-parse",
            json={"cv_url": "https://example.com/cv.pdf"},
        )
        self.assertEqual(resp.status_code, 429, msg=resp.text)

        # Verify 429 response structure
        data = resp.json()
        self.assertEqual(data["detail"], "Rate limit exceeded")
        self.assertEqual(data["error_code"], "RATE_LIMIT_EXCEEDED")
        self.assertIn("retry_after_seconds", data)
        self.assertIsInstance(data["retry_after_seconds"], (int, float))
        self.assertIn("correlation_id", data)

        # Verify headers
        self.assertIn("Retry-After", resp.headers)
        self.assertIn("X-Correlation-Id", resp.headers)


class TestAssessmentGenerateBurstLimit(IsolatedAsyncioTestCase):
    """Rate‑limit enforcement for POST /assessment/generate (10/minute burst)."""

    def setUp(self):
        _reset_rate_limiter()
        from app.routers.assessment import router as assessment_router
        from app.clients.dependencies import get_qdrant_client

        self.mock_llm = MagicMock()
        # AssessmentService.generate_questions expects a JSON array of strings for "single" type
        self.mock_llm.generate.return_value = json.dumps([
            "Describe a time when you had to refactor a large codebase.",
            "Imagine you are faced with a production outage. Walk through your response.",
            "How would you design a scalable notification system?",
        ])
        self.mock_qdrant = MagicMock()
        # AssessmentService.generate_questions calls qdrant.get(candidates, candidate_id)
        self.mock_qdrant.get.return_value = {
            "past_roles": ["Software Engineer at Acme"],
            "skills": ["Python", "FastAPI"],
        }

        self.app = _build_test_app(
            assessment_router,
            llm_client_override=self.mock_llm,
            dependency_overrides={get_qdrant_client: self.mock_qdrant},
        )
        self.client = TestClient(self.app)

    def test_burst_limit_exceeded(self):
        """POST /assessment/generate has a 10/minute burst; the 11th should 429."""
        payload = {
            "candidate_context": {"candidate_id": 1, "target_role": "Engineer"},
            "num_questions": 3,
            "question_type": "single",
        }

        for i in range(10):
            resp = self.client.post("/api/ai/assessment/generate", json=payload)
            self.assertEqual(
                resp.status_code, 200,
                msg=f"Request {i+1} should be 200, got {resp.status_code}: {resp.text}",
            )

        resp = self.client.post("/api/ai/assessment/generate", json=payload)
        self.assertEqual(resp.status_code, 429, msg=resp.text)
        data = resp.json()
        self.assertEqual(data["error_code"], "RATE_LIMIT_EXCEEDED")
        self.assertIn("Retry-After", resp.headers)


class TestAssessmentGradeBurstLimit(IsolatedAsyncioTestCase):
    """Rate‑limit enforcement for POST /assessment/grade (20/minute burst)."""

    def setUp(self):
        _reset_rate_limiter()
        from app.routers.assessment import router as assessment_router

        self.mock_llm = MagicMock()
        self.mock_llm.generate.return_value = json.dumps({
            "overall_score": 85,
            "skill_breakdown": [
                {"category": "Python", "score": 90, "feedback": "Strong"},
                {"category": "System Design", "score": 70, "feedback": "Adequate"},
                {"category": "Communication", "score": 80, "feedback": "Good"},
            ],
            "authenticity_flag": {"is_suspicious": False, "reason": "No issues detected"},
        })
        self.app = _build_test_app(assessment_router, llm_client_override=self.mock_llm)
        self.client = TestClient(self.app)

    def test_burst_limit_exceeded(self):
        """POST /assessment/grade has a 20/minute burst; the 21st should 429."""
        payload = {
            "questions": ["What is Python?"],
            "answers": ["A programming language"],
            "time_taken_seconds": 120,
        }

        for i in range(20):
            resp = self.client.post("/api/ai/assessment/grade", json=payload)
            self.assertEqual(
                resp.status_code, 200,
                msg=f"Request {i+1} should be 200, got {resp.status_code}: {resp.text}",
            )

        resp = self.client.post("/api/ai/assessment/grade", json=payload)
        self.assertEqual(resp.status_code, 429, msg=resp.text)
        self.assertEqual(resp.json()["error_code"], "RATE_LIMIT_EXCEEDED")


class TestScoringBurstLimit(IsolatedAsyncioTestCase):
    """Rate‑limit enforcement for POST /calculate-fit (20/minute burst)."""

    def setUp(self):
        _reset_rate_limiter()
        from app.routers.scoring import router as scoring_router
        from app.clients.dependencies import get_qdrant_client, get_cache_backend

        self.mock_llm = MagicMock()
        self.mock_llm.generate.return_value = json.dumps({
            "overall_score_percentage": 85,
            "category_breakdown": {
                "role_match": {"status": "pass", "short_reason": "Good fit"},
                "experience": {"status": "pass", "short_reason": "Relevant experience"},
                "location": {"status": "pass", "short_reason": "Same city"},
                "employment_type": {"status": "pass", "short_reason": "Full-time"},
            },
            "skill_gap_analysis": "Candidate meets most requirements",
        })
        self.mock_qdrant = MagicMock()
        # ScoringService.calculate_fit calls qdrant.get for both candidate and job
        self.mock_qdrant.get.return_value = {
            "candidate_version": 1,
            "job_version": 1,
            "past_roles": [],
            "skills": [],
            "title": "Engineer",
            "required_skills": [],
            "raw_jd_summary": "",
        }
        self.mock_cache = MagicMock()
        self.mock_cache.get.return_value = None

        self.app = _build_test_app(
            scoring_router,
            llm_client_override=self.mock_llm,
            dependency_overrides={
                get_qdrant_client: self.mock_qdrant,
                get_cache_backend: self.mock_cache,
            },
        )
        self.client = TestClient(self.app)

    def test_burst_limit_exceeded(self):
        """POST /calculate-fit has a 20/minute burst; the 21st should 429."""
        payload = {
            "candidate_id": 1,
            "candidate_version": 1,
            "job_id": 10,
            "job_version": 1,
        }

        for i in range(20):
            resp = self.client.post("/api/ai/calculate-fit", json=payload)
            self.assertEqual(
                resp.status_code, 200,
                msg=f"Request {i+1} should be 200, got {resp.status_code}: {resp.text}",
            )

        resp = self.client.post("/api/ai/calculate-fit", json=payload)
        self.assertEqual(resp.status_code, 429, msg=resp.text)
        self.assertEqual(resp.json()["error_code"], "RATE_LIMIT_EXCEEDED")


class TestRecommendBurstLimit(IsolatedAsyncioTestCase):
    """Rate‑limit enforcement for POST /recommend (10/minute burst)."""

    def setUp(self):
        _reset_rate_limiter()
        from app.routers.recommend import router as recommend_router
        from app.clients.dependencies import get_qdrant_client, get_cache_backend

        self.mock_llm = MagicMock()
        self.mock_llm.generate.return_value = json.dumps({"results": []})
        self.mock_qdrant = MagicMock()
        # RecommendationService.recommend calls:
        # 1. get_with_vector(CANDIDATES_COLLECTION, target_id) -> (payload, vector)
        #    When vector is not None, it proceeds to the weighted-query path
        # 2. search(target_collection, ...) for peer vectors
        # 3. get_with_vector(search_collection, ...) for co-occurrence vectors
        # 4. search(search_collection, ...) for final results
        self.mock_qdrant.get_with_vector.return_value = (
            {"candidate_version": 1, "skills": ["Python"]},
            [0.1, 0.2, 0.3],
        )
        self.mock_qdrant.search.return_value = []
        self.mock_cache = MagicMock()
        self.mock_cache.get.return_value = None

        self.app = _build_test_app(
            recommend_router,
            llm_client_override=self.mock_llm,
            dependency_overrides={
                get_qdrant_client: self.mock_qdrant,
                get_cache_backend: self.mock_cache,
            },
        )
        self.client = TestClient(self.app)

    def test_burst_limit_exceeded(self):
        """POST /recommend has a 10/minute burst; the 11th should 429."""
        payload = {
            "type": "jobs",
            "target_id": 1,
            "target_version": 1,
            "behavioral_signals": {"recent_clicks": []},
        }

        for i in range(10):
            resp = self.client.post("/api/ai/recommend", json=payload)
            self.assertEqual(
                resp.status_code, 200,
                msg=f"Request {i+1} should be 200, got {resp.status_code}: {resp.text}",
            )

        resp = self.client.post("/api/ai/recommend", json=payload)
        self.assertEqual(resp.status_code, 429, msg=resp.text)
        self.assertEqual(resp.json()["error_code"], "RATE_LIMIT_EXCEEDED")


class TestCareerBurstLimit(IsolatedAsyncioTestCase):
    """Rate‑limit enforcement for POST /analyze-career-paths (5/minute burst)."""

    def setUp(self):
        _reset_rate_limiter()
        from app.routers.career import router as career_router
        from app.clients.dependencies import get_qdrant_client

        self.mock_llm = MagicMock()
        # CareerPathService.analyze_career_paths expects exactly 3 paths
        self.mock_llm.generate.return_value = json.dumps({
            "profile_summary": "A skilled software engineer with 5 years of experience.",
            "paths": [
                {
                    "role": "Senior Software Engineer",
                    "match_percentage": 85,
                    "core_skills": ["Python", "System Design", "Leadership"],
                    "reasoning": "Your strong technical background makes this a natural progression.",
                },
                {
                    "role": "Engineering Manager",
                    "match_percentage": 70,
                    "core_skills": ["Team Management", "Agile", "Communication"],
                    "reasoning": "Your experience mentoring juniors indicates management potential.",
                },
                {
                    "role": "Solutions Architect",
                    "match_percentage": 65,
                    "core_skills": ["Cloud Architecture", "Microservices", "API Design"],
                    "reasoning": "Your broad technical knowledge suits an architectural role.",
                },
            ],
        })
        self.mock_qdrant = MagicMock()
        # CareerPathService.analyze_career_paths calls qdrant.get(candidates, candidate_id)
        self.mock_qdrant.get.return_value = {
            "past_roles": ["Software Engineer"],
            "skills": ["Python"],
        }

        self.app = _build_test_app(
            career_router,
            llm_client_override=self.mock_llm,
            dependency_overrides={get_qdrant_client: self.mock_qdrant},
        )
        self.client = TestClient(self.app)

    def test_burst_limit_exceeded(self):
        """POST /analyze-career-paths has a 5/minute burst; the 6th should 429."""
        payload = {"candidate_id": 1}

        for i in range(5):
            resp = self.client.post("/api/ai/analyze-career-paths", json=payload)
            self.assertEqual(
                resp.status_code, 200,
                msg=f"Request {i+1} should be 200, got {resp.status_code}: {resp.text}",
            )

        resp = self.client.post("/api/ai/analyze-career-paths", json=payload)
        self.assertEqual(resp.status_code, 429, msg=resp.text)
        self.assertEqual(resp.json()["error_code"], "RATE_LIMIT_EXCEEDED")


class TestJdGenerateBurstLimit(IsolatedAsyncioTestCase):
    """Rate‑limit enforcement for POST /generate-jd (5/minute burst)."""

    def setUp(self):
        _reset_rate_limiter()
        from app.routers.jd import router as jd_router
        from app.clients.dependencies import get_qdrant_client

        self.mock_llm = MagicMock()
        self.mock_llm.generate.return_value = "Generated job description text"
        self.mock_qdrant = MagicMock()
        self.mock_qdrant.get.return_value = {"points": []}

        self.app = _build_test_app(
            jd_router,
            llm_client_override=self.mock_llm,
            dependency_overrides={get_qdrant_client: self.mock_qdrant},
        )
        self.client = TestClient(self.app)

    def test_burst_limit_exceeded(self):
        """POST /generate-jd has a 5/minute burst; the 6th should 429."""
        payload = {"prompt": "Write a job description for a software engineer"}

        for i in range(5):
            resp = self.client.post("/api/ai/generate-jd", json=payload)
            self.assertEqual(
                resp.status_code, 200,
                msg=f"Request {i+1} should be 200, got {resp.status_code}: {resp.text}",
            )

        resp = self.client.post("/api/ai/generate-jd", json=payload)
        self.assertEqual(resp.status_code, 429, msg=resp.text)
        self.assertEqual(resp.json()["error_code"], "RATE_LIMIT_EXCEEDED")


class TestJdAnalyzeBurstLimit(IsolatedAsyncioTestCase):
    """Rate‑limit enforcement for POST /analyze-jd (5/minute burst)."""

    def setUp(self):
        _reset_rate_limiter()
        from app.routers.jd import router as jd_router

        self.mock_llm = MagicMock()
        # analyze-jd expects the LLM to return a JSON array of strings (critiques)
        self.mock_llm.generate.return_value = json.dumps([
            "The job description is too vague about required qualifications.",
            "Consider adding a section on company culture and values.",
            "The salary range should be included to attract qualified candidates.",
        ])

        self.app = _build_test_app(jd_router, llm_client_override=self.mock_llm)
        self.client = TestClient(self.app)

    def test_burst_limit_exceeded(self):
        """POST /analyze-jd has a 5/minute burst; the 6th should 429."""
        payload = {"jd_text": "We are looking for a software engineer..."}

        for i in range(5):
            resp = self.client.post("/api/ai/analyze-jd", json=payload)
            self.assertEqual(
                resp.status_code, 200,
                msg=f"Request {i+1} should be 200, got {resp.status_code}: {resp.text}",
            )

        resp = self.client.post("/api/ai/analyze-jd", json=payload)
        self.assertEqual(resp.status_code, 429, msg=resp.text)
        self.assertEqual(resp.json()["error_code"], "RATE_LIMIT_EXCEEDED")


class Test429ResponseContract(IsolatedAsyncioTestCase):
    """Verify the 429 response structure matches the B2B contract."""

    def setUp(self):
        _reset_rate_limiter()
        from app.routers.ingestion import router as ingestion_router
        self.mock_llm = MagicMock()
        self.mock_llm.generate.return_value = json.dumps({"name": "Test"})
        self.app = _build_test_app(ingestion_router, llm_client_override=self.mock_llm)
        self.client = TestClient(self.app)

    @patch("app.routers.ingestion.validate_ingest_url")
    @patch("app.routers.ingestion.fetch_and_parse_cv")
    @patch("app.routers.ingestion.truncate_to_prompt_cap")
    def test_429_response_contract(self, mock_truncate, mock_fetch, mock_validate):
        """The 429 response must include Retry-After, X-Correlation-Id, and structured JSON."""
        mock_validate.return_value = None
        mock_fetch.return_value = "parsed cv text"
        mock_truncate.return_value = "truncated cv text"

        # Exhaust the 10/minute burst limit
        for _ in range(10):
            self.client.post(
                "/api/ai/cv-parse",
                json={"cv_url": "https://example.com/cv.pdf"},
            )

        resp = self.client.post(
            "/api/ai/cv-parse",
            json={"cv_url": "https://example.com/cv.pdf"},
        )
        self.assertEqual(resp.status_code, 429)

        # --- JSON body contract ---
        data = resp.json()
        self.assertEqual(data["detail"], "Rate limit exceeded")
        self.assertEqual(data["error_code"], "RATE_LIMIT_EXCEEDED")
        self.assertIn("retry_after_seconds", data)
        self.assertIsInstance(data["retry_after_seconds"], (int, float))
        self.assertGreater(data["retry_after_seconds"], 0)
        # correlation_id may be None if no middleware ran, but key must exist
        self.assertIn("correlation_id", data)

        # --- Header contract ---
        self.assertIn("Retry-After", resp.headers)
        retry_after = int(resp.headers["Retry-After"])
        self.assertGreater(retry_after, 0)
        self.assertIn("X-Correlation-Id", resp.headers)
