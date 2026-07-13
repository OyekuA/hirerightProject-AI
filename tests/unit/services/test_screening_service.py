

import json
import unittest
from unittest import IsolatedAsyncioTestCase
from unittest.mock import MagicMock, patch, AsyncMock, ANY

from app.services.screening_service import BulkScreeningService
from app.schemas.screening import ScreenBatchRequest, ScreeningCandidateInput

def _make_mock_settings():

    mock = MagicMock()
    mock.SCREENING_CONCURRENCY = 10
    mock.SCREENING_MAX_BATCH_SIZE = 1000
    mock.SCORING_WEIGHT_SKILLS = 0.35
    mock.SCORING_WEIGHT_ROLE = 0.25
    mock.SCORING_WEIGHT_EXPERIENCE = 0.20
    mock.SCORING_WEIGHT_LOCATION = 0.12
    mock.SCORING_WEIGHT_EMPLOYMENT = 0.08
    mock.SCORING_STATUS_PASS_THRESHOLD = 75
    mock.SCORING_STATUS_WARNING_THRESHOLD = 50
    mock.CACHE_TTL_SECONDS = 86400
    mock.MAX_PROMPT_CHARS = 50000
    return mock

def _make_hint():

    hint = MagicMock()
    hint.name = ""
    hint.location = ""
    hint.experience_level = ""
    hint.industry = ""
    hint.employment_type = ""
    return hint

class TestBulkScreeningService(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.mock_llm = MagicMock()
        self.mock_qdrant = MagicMock()
        self.mock_cache = MagicMock()
        self.mock_store = MagicMock()
        self.mock_callback = MagicMock()

        self.service = BulkScreeningService(
            llm=self.mock_llm,
            qdrant=self.mock_qdrant,
            cache=self.mock_cache,
            callback_client=self.mock_callback,
        )

    @patch("app.services.screening_service.get_settings")
    async def test_resolve_job_payload_from_existing_job(self, mock_get_settings):

        mock_get_settings.return_value = _make_mock_settings()
        self.mock_qdrant.get.return_value = {
            "title": "Engineer",
            "job_version": 2,
            "location": "Remote",
        }

        req = ScreenBatchRequest(
            job_id=42,
            job_version=2,
            candidates=[ScreeningCandidateInput(
                candidate_ref="c1",
                cv_url="https://example.com/cv.pdf",
            )],
        )

        payload = await self.service.resolve_job_payload(req)
        self.assertEqual(payload["title"], "Engineer")
        self.assertEqual(payload["job_version"], 2)
        self.mock_qdrant.get.assert_called_once()

    @patch("app.services.screening_service.get_settings")
    async def test_resolve_job_payload_missing_job_raises(self, mock_get_settings):

        mock_get_settings.return_value = _make_mock_settings()
        self.mock_qdrant.get.return_value = None

        req = ScreenBatchRequest(
            job_id=99,
            job_version=1,
            candidates=[ScreeningCandidateInput(
                candidate_ref="c1",
                cv_url="https://example.com/cv.pdf",
            )],
        )

        with self.assertRaises(ValueError):
            await self.service.resolve_job_payload(req)

    @patch("app.services.screening_service.get_settings")
    async def test_resolve_job_payload_version_mismatch_raises(self, mock_get_settings):

        mock_get_settings.return_value = _make_mock_settings()
        self.mock_qdrant.get.return_value = {"title": "Engineer", "job_version": 1}

        req = ScreenBatchRequest(
            job_id=42,
            job_version=2,
            candidates=[ScreeningCandidateInput(
                candidate_ref="c1",
                cv_url="https://example.com/cv.pdf",
            )],
        )

        with self.assertRaises(ValueError):
            await self.service.resolve_job_payload(req)

    @patch("app.services.screening_service.get_settings")
    @patch("app.services.screening_service.fetch_and_parse_cv")
    @patch("app.services.screening_service.extract_candidate_entities")
    async def test_process_batch_no_qdrant_upsert_or_llm_embed(
        self,
        mock_extract,
        mock_fetch,
        mock_get_settings,
    ):

        mock_get_settings.return_value = _make_mock_settings()
        mock_fetch.return_value = "CV text content"
        mock_extraction = MagicMock()
        mock_extraction.name = "Extracted Candidate"
        mock_extraction.location = "Remote"
        mock_extraction.experience_level = "Senior"
        mock_extraction.industry = "Tech"
        mock_extraction.employment_type = "Full-time"
        mock_extraction.skills = ["Python"]
        mock_extraction.past_roles = ["Engineer"]
        mock_extraction.raw_profile_summary = "Summary text"
        mock_extract.return_value = mock_extraction

        self.mock_llm.generate.return_value = json.dumps({
            "skills": {"score": 80, "short_reason": "Good match"},
            "role_match": {"score": 75, "short_reason": "Relevant domain"},
            "skill_gap_analysis": "Some gaps in leadership",
        })

        candidates = [
            MagicMock(
                candidate_ref="c1",
                cv_url=MagicMock(__str__=lambda self: "https://example.com/cv1.pdf"),
                profile_hint=_make_hint(),
            ),
        ]

        job_payload = {
            "title": "Engineer",
            "location": "Remote",
            "experience_level": "Senior",
            "industry": "Tech",
            "employment_type": "Full-time",
            "required_skills": ["Python"],
            "raw_jd_summary": "Looking for an engineer",
        }

        await self.service.process_batch(
            batch_id="test-batch-1",
            job_payload=job_payload,
            job_id=42,
            job_version=1,
            candidates=candidates,
            store=self.mock_store,
        )

        self.mock_qdrant.upsert.assert_not_called()
        self.mock_llm.embed.assert_not_called()
        self.mock_store.update.assert_called()
        self.mock_store.append_result.assert_called()

    @patch("app.services.screening_service.get_settings")
    @patch("app.services.screening_service.extract_candidate_entities")
    @patch("app.services.screening_service.fetch_and_parse_cv")
    async def test_per_candidate_failure_isolation(
        self,
        mock_fetch,
        mock_extract,
        mock_get_settings,
    ):

        mock_get_settings.return_value = _make_mock_settings()

        mock_fetch.side_effect = [
            ValueError("CV exceeds size limit"),
            "Second CV text",
        ]

        mock_extraction = MagicMock()
        mock_extraction.name = "John Doe"
        mock_extraction.location = "Remote"
        mock_extraction.experience_level = "Senior"
        mock_extraction.industry = "Tech"
        mock_extraction.employment_type = "Full-time"
        mock_extraction.skills = ["Python", "ML"]
        mock_extraction.past_roles = ["Engineer"]
        mock_extraction.raw_profile_summary = "Experienced engineer."
        mock_extract.return_value = mock_extraction

        self.mock_llm.generate.return_value = json.dumps({
            "skills": {"score": 80, "short_reason": "Good match"},
            "role_match": {"score": 75, "short_reason": "Relevant domain"},
            "skill_gap_analysis": "Some gaps",
        })

        candidates = [
            MagicMock(
                candidate_ref="c1",
                cv_url=MagicMock(__str__=lambda self: "https://example.com/cv1.pdf"),
                profile_hint=_make_hint(),
            ),
            MagicMock(
                candidate_ref="c2",
                cv_url=MagicMock(__str__=lambda self: "https://example.com/cv2.pdf"),
                profile_hint=_make_hint(),
            ),
        ]

        job_payload = {
            "title": "Engineer",
            "location": "Remote",
            "experience_level": "Senior",
            "industry": "Tech",
            "employment_type": "Full-time",
            "required_skills": ["Python"],
            "raw_jd_summary": "Summary",
        }

        await self.service.process_batch(
            batch_id="test-batch-2",
            job_payload=job_payload,
            job_id=0,
            job_version=1,
            candidates=candidates,
            store=self.mock_store,
        )

        self.assertEqual(self.mock_store.append_result.call_count, 2)
        statuses = [call[0][1]["status"] for call in self.mock_store.append_result.call_args_list]
        self.assertIn("failed", statuses)
        self.assertIn("scored", statuses)

    @patch("app.services.screening_service.get_settings")
    @patch("app.services.screening_service.fetch_and_parse_cv")
    @patch("app.services.screening_service.extract_candidate_entities")
    async def test_circuit_breaker_deterministic(
        self,
        mock_extract,
        mock_fetch,
        mock_get_settings,
    ):

        mock_get_settings.return_value = _make_mock_settings()
        mock_fetch.return_value = "Some CV text"

        mock_extraction = MagicMock()
        mock_extraction.name = "John"
        mock_extraction.location = "Remote"
        mock_extraction.experience_level = "Senior"
        mock_extraction.industry = "Tech"
        mock_extraction.employment_type = "Full-time"
        mock_extraction.skills = ["Python"]
        mock_extraction.past_roles = ["Engineer"]
        mock_extraction.raw_profile_summary = "Summary"
        mock_extract.return_value = mock_extraction

        from app.clients.llm import LLMUnavailableError
        self.mock_llm.generate.side_effect = [
            json.dumps({
                "name": "John", "location": "Remote",
                "experience_level": "Senior", "industry": "Tech",
                "employment_type": "Full-time",
                "skills": ["Python"], "past_roles": [],
                "raw_profile_summary": "Summary",
            }),
            LLMUnavailableError("Generation circuit breaker is open"),
        ]

        settings_override = _make_mock_settings()
        settings_override.SCREENING_CONCURRENCY = 1
        mock_get_settings.return_value = settings_override

        candidates = [
            MagicMock(
                candidate_ref="c1",
                cv_url=MagicMock(__str__=lambda self: "https://example.com/cv1.pdf"),
                profile_hint=_make_hint(),
            ),
            MagicMock(
                candidate_ref="c2",
                cv_url=MagicMock(__str__=lambda self: "https://example.com/cv2.pdf"),
                profile_hint=_make_hint(),
            ),
            MagicMock(
                candidate_ref="c3",
                cv_url=MagicMock(__str__=lambda self: "https://example.com/cv3.pdf"),
                profile_hint=_make_hint(),
            ),
        ]

        job_payload = {
            "title": "Engineer",
            "location": "Remote",
            "experience_level": "Senior",
            "industry": "Tech",
            "employment_type": "Full-time",
            "required_skills": ["Python"],
            "raw_jd_summary": "Summary",
        }

        await self.service.process_batch(
            batch_id="test-batch-det",
            job_payload=job_payload,
            job_id=0,
            job_version=1,
            candidates=candidates,
            store=self.mock_store,
        )

        self.assertEqual(self.mock_store.append_result.call_count, 3)

        first_result = self.mock_store.append_result.call_args_list[0][0][1]
        self.assertEqual(first_result["status"], "failed")

        for i in (1, 2):
            result = self.mock_store.append_result.call_args_list[i][0][1]
            self.assertEqual(result["status"], "failed")
            self.assertIsNotNone(result.get("error"))
            self.assertIn("circuit breaker", (result.get("error") or "").lower())

        self.assertEqual(self.mock_llm.generate.call_count, 2)

    @patch("app.services.screening_service.get_settings")
    @patch("app.services.screening_service.fetch_and_parse_cv")
    @patch("app.services.screening_service.extract_candidate_entities")
    async def test_unexpected_exception_batch_goes_to_failed(
        self,
        mock_extract,
        mock_fetch,
        mock_get_settings,
    ):

        mock_get_settings.return_value = _make_mock_settings()
        mock_fetch.return_value = "Some CV text"

        mock_extraction = MagicMock()
        mock_extraction.name = "John"
        mock_extraction.location = "Remote"
        mock_extraction.experience_level = "Senior"
        mock_extraction.industry = "Tech"
        mock_extraction.employment_type = "Full-time"
        mock_extraction.skills = ["Python"]
        mock_extraction.past_roles = ["Engineer"]
        mock_extraction.raw_profile_summary = "Summary"
        mock_extract.return_value = mock_extraction

        candidates = [
            MagicMock(
                candidate_ref="c1",
                cv_url=MagicMock(__str__=lambda self: "https://example.com/cv1.pdf"),
                profile_hint=_make_hint(),
            ),
        ]

        job_payload = {
            "title": "Engineer",
            "location": "Remote",
            "experience_level": "Senior",
            "industry": "Tech",
            "employment_type": "Full-time",
            "required_skills": ["Python"],
            "raw_jd_summary": "Summary",
        }

        self.mock_store.append_result.side_effect = TypeError("Unexpected store I/O failure")

        await self.service.process_batch(
            batch_id="test-batch-unexpected",
            job_payload=job_payload,
            job_id=0,
            job_version=1,
            candidates=candidates,
            store=self.mock_store,
        )

        self.mock_store.update.assert_any_call("test-batch-unexpected", status="running")
        self.mock_store.update.assert_any_call(
            "test-batch-unexpected", status="failed", error_summary=ANY
        )

if __name__ == "__main__":
    unittest.main()
