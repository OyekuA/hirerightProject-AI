"""Unit tests for ingestion service payload composition."""

import json
import unittest
from unittest.mock import MagicMock, patch, ANY, AsyncMock

from app.services.ingestion_service import run_candidate_ingestion, run_job_ingestion


class TestIngestionServicePayloads(unittest.IsolatedAsyncioTestCase):
    """Verify that raw_profile_summary and raw_jd_summary are persisted in Qdrant upsert payloads."""

    async def test_candidate_ingestion_includes_raw_profile_summary(self):
        """Check that candidate payload contains raw_profile_summary."""
        mock_qdrant = MagicMock()
        mock_qdrant.upsert = MagicMock()
        mock_gemini = MagicMock()
        mock_gemini.generate = MagicMock(return_value=json.dumps({
            "name": "John Doe",
            "location": "Remote",
            "experience_level": "Senior",
            "industry": "Tech",
            "employment_type": "Full-time",
            "skills": ["Python", "ML"],
            "past_roles": ["Engineer"],
            "raw_profile_summary": "Experienced software engineer with ML background."
        }))
        mock_gemini.embed = MagicMock(return_value=[0.1] * 768)
        mock_store = MagicMock()
        mock_store.update = MagicMock()
        mock_callback = MagicMock()
        mock_callback.send = AsyncMock(return_value=True)

        with patch("app.services.ingestion_service.fetch_and_parse_cv") as mock_fetch, \
             patch("app.services.ingestion_service.truncate_to_prompt_cap") as mock_truncate:
            mock_fetch.return_value = "CV text"
            mock_truncate.side_effect = lambda x: x
            await run_candidate_ingestion(
                candidate_id=123,
                cv_url="https://example.com/cv.pdf",
                profile_data={
                    "name": "John",
                    "location": "Remote",
                    "experience_level": "Senior",
                    "industry": "Tech",
                    "employment_type": "Full-time",
                    "candidate_version": 1
                },
                callback_url="https://example.com/callback",
                event_id="evt_123",
                qdrant=mock_qdrant,
                llm=mock_gemini,
                store=mock_store,
                callback_client=mock_callback,
            )

        mock_qdrant.upsert.assert_called_once()
        call_args = mock_qdrant.upsert.call_args
        payload = call_args[0][3]
        self.assertIn("raw_profile_summary", payload)
        self.assertEqual(payload["raw_profile_summary"],
                         "Experienced software engineer with ML background.")
        self.assertEqual(payload["candidate_id"], 123)

    async def test_job_ingestion_includes_raw_jd_summary(self):
        """Check that job payload contains raw_jd_summary."""
        mock_qdrant = MagicMock()
        mock_qdrant.upsert = MagicMock()
        mock_gemini = MagicMock()
        mock_gemini.generate = MagicMock(return_value=json.dumps({
            "title": "Software Engineer",
            "location": "Remote",
            "experience_level": "Mid",
            "industry": "Tech",
            "employment_type": "Full-time",
            "required_skills": ["Python", "AWS"],
            "raw_jd_summary": "Looking for a software engineer with Python and AWS experience."
        }))
        mock_gemini.embed = MagicMock(return_value=[0.1] * 768)
        mock_store = MagicMock()
        mock_store.update = MagicMock()
        mock_callback = MagicMock()
        mock_callback.send = AsyncMock(return_value=True)

        with patch("app.services.ingestion_service.truncate_to_prompt_cap") as mock_truncate:
            mock_truncate.side_effect = lambda x: x
            await run_job_ingestion(
                job_id=456,
                jd_text="Job description text",
                metadata={
                    "title": "Software Engineer",
                    "location": "Remote",
                    "experience_level": "Mid",
                    "industry": "Tech",
                    "employment_type": "Full-time",
                    "job_version": 1,
                    "company_name": "ORACLE",
                    "about": "Nice",
                },
                callback_url="https://example.com/callback",
                event_id="evt_456",
                qdrant=mock_qdrant,
                llm=mock_gemini,
                store=mock_store,
                callback_client=mock_callback,
            )

        mock_qdrant.upsert.assert_called_once()
        payload = mock_qdrant.upsert.call_args[0][3]
        self.assertIn("raw_jd_summary", payload)
        self.assertEqual(payload["raw_jd_summary"],
                         "Looking for a software engineer with Python and AWS experience.")
        self.assertEqual(payload["job_id"], 456)
        self.assertEqual(payload["company_name"], "ORACLE")
        self.assertEqual(payload["about"], "Nice")


class TestIngestionHashDeduplication(unittest.IsolatedAsyncioTestCase):
    """Test hash-based deduplication in candidate and job ingestion."""

    async def test_candidate_hash_match_skips_gemini_calls(self):
        """When stored cv_hash matches new hash, Gemini not called, update_payload is called."""
        mock_qdrant = MagicMock()
        mock_qdrant.get = MagicMock()
        mock_qdrant.update_payload = MagicMock()
        mock_qdrant.upsert = MagicMock()
        mock_gemini = MagicMock()
        mock_gemini.generate = MagicMock()
        mock_gemini.embed = MagicMock()
        mock_store = MagicMock()
        mock_store.update = MagicMock()
        mock_callback = MagicMock()
        mock_callback.send = AsyncMock(return_value=True)

        cv_text = "CV text"
        import hashlib
        new_hash = hashlib.sha256(cv_text.encode()).hexdigest()
        mock_qdrant.get.return_value = {"cv_hash": new_hash}

        with patch("app.services.ingestion_service.fetch_and_parse_cv") as mock_fetch, \
             patch("app.services.ingestion_service.truncate_to_prompt_cap") as mock_truncate:
            mock_fetch.return_value = cv_text
            mock_truncate.side_effect = lambda x: x
            await run_candidate_ingestion(
                candidate_id=123,
                cv_url="https://example.com/cv.pdf",
                profile_data={
                    "name": "John",
                    "location": "Remote",
                    "experience_level": "Senior",
                    "industry": "Tech",
                    "employment_type": "Full-time",
                    "candidate_version": 2
                },
                callback_url="https://example.com/callback",
                event_id="evt_123",
                qdrant=mock_qdrant,
                llm=mock_gemini,
                store=mock_store,
                callback_client=mock_callback,
            )

        mock_gemini.generate.assert_not_called()
        mock_gemini.embed.assert_not_called()
        mock_qdrant.upsert.assert_not_called()
        mock_qdrant.update_payload.assert_called_once()
        call_args = mock_qdrant.update_payload.call_args
        self.assertEqual(call_args[0][0], "candidates")
        self.assertEqual(call_args[0][1], 123)
        payload_fields = call_args[0][2]
        self.assertEqual(payload_fields["candidate_version"], 2)
        self.assertIn("ingested_at", payload_fields)
        self.assertEqual(payload_fields["cv_hash"], new_hash)
        mock_store.update.assert_called_with("evt_123", status="success", attempt_count=1)

    async def test_candidate_hash_mismatch_runs_full_pipeline(self):
        """When stored cv_hash differs, full pipeline runs, cv_hash added to upsert payload."""
        mock_qdrant = MagicMock()
        mock_qdrant.get = MagicMock()
        mock_qdrant.update_payload = MagicMock()
        mock_qdrant.upsert = MagicMock()
        mock_gemini = MagicMock()
        mock_gemini.generate = MagicMock(return_value=json.dumps({
            "name": "John Doe",
            "location": "Remote",
            "experience_level": "Senior",
            "industry": "Tech",
            "employment_type": "Full-time",
            "skills": ["Python", "ML"],
            "past_roles": ["Engineer"],
            "raw_profile_summary": "Experienced software engineer with ML background."
        }))
        mock_gemini.embed = MagicMock(return_value=[0.1] * 768)
        mock_store = MagicMock()
        mock_store.update = MagicMock()
        mock_callback = MagicMock()
        mock_callback.send = AsyncMock(return_value=True)

        cv_text = "CV text"
        import hashlib
        new_hash = hashlib.sha256(cv_text.encode()).hexdigest()
        mock_qdrant.get.return_value = {"cv_hash": "old_hash"}

        with patch("app.services.ingestion_service.fetch_and_parse_cv") as mock_fetch, \
             patch("app.services.ingestion_service.truncate_to_prompt_cap") as mock_truncate:
            mock_fetch.return_value = cv_text
            mock_truncate.side_effect = lambda x: x
            await run_candidate_ingestion(
                candidate_id=123,
                cv_url="https://example.com/cv.pdf",
                profile_data={
                    "name": "John",
                    "location": "Remote",
                    "experience_level": "Senior",
                    "industry": "Tech",
                    "employment_type": "Full-time",
                    "candidate_version": 2
                },
                callback_url="https://example.com/callback",
                event_id="evt_123",
                qdrant=mock_qdrant,
                llm=mock_gemini,
                store=mock_store,
                callback_client=mock_callback,
            )

        mock_gemini.generate.assert_called_once()
        mock_gemini.embed.assert_called_once()
        mock_qdrant.upsert.assert_called_once()
        mock_qdrant.update_payload.assert_not_called()
        payload = mock_qdrant.upsert.call_args[0][3]
        self.assertIn("cv_hash", payload)
        self.assertEqual(payload["cv_hash"], new_hash)

    async def test_candidate_no_existing_payload_runs_full_pipeline(self):
        """When no existing payload (first ingest), full pipeline runs, cv_hash added."""
        mock_qdrant = MagicMock()
        mock_qdrant.get = MagicMock(return_value=None)
        mock_qdrant.update_payload = MagicMock()
        mock_qdrant.upsert = MagicMock()
        mock_gemini = MagicMock()
        mock_gemini.generate = MagicMock(return_value=json.dumps({
            "name": "John Doe",
            "location": "Remote",
            "experience_level": "Senior",
            "industry": "Tech",
            "employment_type": "Full-time",
            "skills": ["Python", "ML"],
            "past_roles": ["Engineer"],
            "raw_profile_summary": "Experienced software engineer with ML background."
        }))
        mock_gemini.embed = MagicMock(return_value=[0.1] * 768)
        mock_store = MagicMock()
        mock_store.update = MagicMock()
        mock_callback = MagicMock()
        mock_callback.send = AsyncMock(return_value=True)

        cv_text = "CV text"
        import hashlib
        new_hash = hashlib.sha256(cv_text.encode()).hexdigest()

        with patch("app.services.ingestion_service.fetch_and_parse_cv") as mock_fetch, \
             patch("app.services.ingestion_service.truncate_to_prompt_cap") as mock_truncate:
            mock_fetch.return_value = cv_text
            mock_truncate.side_effect = lambda x: x
            await run_candidate_ingestion(
                candidate_id=123,
                cv_url="https://example.com/cv.pdf",
                profile_data={
                    "name": "John",
                    "location": "Remote",
                    "experience_level": "Senior",
                    "industry": "Tech",
                    "employment_type": "Full-time",
                    "candidate_version": 2
                },
                callback_url="https://example.com/callback",
                event_id="evt_123",
                qdrant=mock_qdrant,
                llm=mock_gemini,
                store=mock_store,
                callback_client=mock_callback,
            )

        mock_gemini.generate.assert_called_once()
        mock_gemini.embed.assert_called_once()
        mock_qdrant.upsert.assert_called_once()
        mock_qdrant.update_payload.assert_not_called()
        payload = mock_qdrant.upsert.call_args[0][3]
        self.assertIn("cv_hash", payload)
        self.assertEqual(payload["cv_hash"], new_hash)

    async def test_job_hash_match_skips_gemini_calls(self):
        """When stored jd_hash matches new hash, Gemini not called, update_payload is called."""
        mock_qdrant = MagicMock()
        mock_qdrant.get = MagicMock()
        mock_qdrant.update_payload = MagicMock()
        mock_qdrant.upsert = MagicMock()
        mock_gemini = MagicMock()
        mock_gemini.generate = MagicMock()
        mock_gemini.embed = MagicMock()
        mock_store = MagicMock()
        mock_store.update = MagicMock()
        mock_callback = MagicMock()
        mock_callback.send = AsyncMock(return_value=True)

        jd_text = "Job description text"
        import hashlib
        new_hash = hashlib.sha256(jd_text.encode()).hexdigest()
        mock_qdrant.get.return_value = {"jd_hash": new_hash}

        with patch("app.services.ingestion_service.truncate_to_prompt_cap") as mock_truncate:
            mock_truncate.side_effect = lambda x: x
            await run_job_ingestion(
                job_id=456,
                jd_text=jd_text,
                metadata={
                    "title": "Software Engineer",
                    "location": "Remote",
                    "experience_level": "Mid",
                    "industry": "Tech",
                    "employment_type": "Full-time",
                    "job_version": 2,
                    "company_name": "ORACLE",
                    "about": "Nice",
                },
                callback_url="https://example.com/callback",
                event_id="evt_456",
                qdrant=mock_qdrant,
                llm=mock_gemini,
                store=mock_store,
                callback_client=mock_callback,
            )

        mock_gemini.generate.assert_not_called()
        mock_gemini.embed.assert_not_called()
        mock_qdrant.upsert.assert_not_called()
        mock_qdrant.update_payload.assert_called_once()
        call_args = mock_qdrant.update_payload.call_args
        self.assertEqual(call_args[0][0], "jobs")
        self.assertEqual(call_args[0][1], 456)
        payload_fields = call_args[0][2]
        self.assertEqual(payload_fields["job_version"], 2)
        self.assertIn("ingested_at", payload_fields)
        self.assertEqual(payload_fields["jd_hash"], new_hash)
        self.assertEqual(payload_fields["company_name"], "ORACLE")
        self.assertEqual(payload_fields["about"], "Nice")
        mock_store.update.assert_called_with("evt_456", status="success", attempt_count=1)

    async def test_job_hash_mismatch_runs_full_pipeline(self):
        """When stored jd_hash differs, full pipeline runs, jd_hash added to upsert payload."""
        mock_qdrant = MagicMock()
        mock_qdrant.get = MagicMock()
        mock_qdrant.update_payload = MagicMock()
        mock_qdrant.upsert = MagicMock()
        mock_gemini = MagicMock()
        mock_gemini.generate = MagicMock(return_value=json.dumps({
            "title": "Software Engineer",
            "location": "Remote",
            "experience_level": "Mid",
            "industry": "Tech",
            "employment_type": "Full-time",
            "required_skills": ["Python", "AWS"],
            "raw_jd_summary": "Looking for a software engineer with Python and AWS experience."
        }))
        mock_gemini.embed = MagicMock(return_value=[0.1] * 768)
        mock_store = MagicMock()
        mock_store.update = MagicMock()
        mock_callback = MagicMock()
        mock_callback.send = AsyncMock(return_value=True)

        jd_text = "Job description text"
        import hashlib
        new_hash = hashlib.sha256(jd_text.encode()).hexdigest()
        mock_qdrant.get.return_value = {"jd_hash": "old_hash"}

        with patch("app.services.ingestion_service.truncate_to_prompt_cap") as mock_truncate:
            mock_truncate.side_effect = lambda x: x
            await run_job_ingestion(
                job_id=456,
                jd_text=jd_text,
                metadata={
                    "title": "Software Engineer",
                    "location": "Remote",
                    "experience_level": "Mid",
                    "industry": "Tech",
                    "employment_type": "Full-time",
                    "job_version": 2,
                    "company_name": "ORACLE",
                    "about": "Nice",
                },
                callback_url="https://example.com/callback",
                event_id="evt_456",
                qdrant=mock_qdrant,
                llm=mock_gemini,
                store=mock_store,
                callback_client=mock_callback,
            )

        mock_gemini.generate.assert_called_once()
        mock_gemini.embed.assert_called_once()
        mock_qdrant.upsert.assert_called_once()
        mock_qdrant.update_payload.assert_not_called()
        payload = mock_qdrant.upsert.call_args[0][3]
        self.assertIn("jd_hash", payload)
        self.assertEqual(payload["jd_hash"], new_hash)
        self.assertEqual(payload["company_name"], "ORACLE")
        self.assertEqual(payload["about"], "Nice")

    async def test_job_no_existing_payload_runs_full_pipeline(self):
        """When no existing payload (first ingest), full pipeline runs, jd_hash added."""
        mock_qdrant = MagicMock()
        mock_qdrant.get = MagicMock(return_value=None)
        mock_qdrant.update_payload = MagicMock()
        mock_qdrant.upsert = MagicMock()
        mock_gemini = MagicMock()
        mock_gemini.generate = MagicMock(return_value=json.dumps({
            "title": "Software Engineer",
            "location": "Remote",
            "experience_level": "Mid",
            "industry": "Tech",
            "employment_type": "Full-time",
            "required_skills": ["Python", "AWS"],
            "raw_jd_summary": "Looking for a software engineer with Python and AWS experience."
        }))
        mock_gemini.embed = MagicMock(return_value=[0.1] * 768)
        mock_store = MagicMock()
        mock_store.update = MagicMock()
        mock_callback = MagicMock()
        mock_callback.send = AsyncMock(return_value=True)

        jd_text = "Job description text"
        import hashlib
        new_hash = hashlib.sha256(jd_text.encode()).hexdigest()

        with patch("app.services.ingestion_service.truncate_to_prompt_cap") as mock_truncate:
            mock_truncate.side_effect = lambda x: x
            await run_job_ingestion(
                job_id=456,
                jd_text=jd_text,
                metadata={
                    "title": "Software Engineer",
                    "location": "Remote",
                    "experience_level": "Mid",
                    "industry": "Tech",
                    "employment_type": "Full-time",
                    "job_version": 2,
                    "company_name": "ORACLE",
                    "about": "Nice",
                },
                callback_url="https://example.com/callback",
                event_id="evt_456",
                qdrant=mock_qdrant,
                llm=mock_gemini,
                store=mock_store,
                callback_client=mock_callback,
            )

        mock_gemini.generate.assert_called_once()
        mock_gemini.embed.assert_called_once()
        mock_qdrant.upsert.assert_called_once()
        mock_qdrant.update_payload.assert_not_called()
        payload = mock_qdrant.upsert.call_args[0][3]
        self.assertIn("jd_hash", payload)
        self.assertEqual(payload["jd_hash"], new_hash)
        self.assertEqual(payload["company_name"], "ORACLE")
        self.assertEqual(payload["about"], "Nice")

    async def test_candidate_hash_differs_when_only_content_beyond_prompt_cap_changes(self):
        """Hash should differ when raw text differs beyond truncation limit."""
        mock_qdrant = MagicMock()
        mock_qdrant.get = MagicMock(return_value=None)
        mock_qdrant.update_payload = MagicMock()
        mock_qdrant.upsert = MagicMock()
        mock_gemini = MagicMock()
        mock_gemini.generate = MagicMock(return_value=json.dumps({
            "name": "John Doe",
            "location": "Remote",
            "experience_level": "Senior",
            "industry": "Tech",
            "employment_type": "Full-time",
            "skills": ["Python", "ML"],
            "past_roles": ["Engineer"],
            "raw_profile_summary": "Experienced software engineer with ML background."
        }))
        mock_gemini.embed = MagicMock(return_value=[0.1] * 768)
        mock_store = MagicMock()
        mock_store.update = MagicMock()
        mock_callback = MagicMock()
        mock_callback.send = AsyncMock(return_value=True)

        # Raw CV text longer than prompt cap
        raw_cv_text = "A" * 1000
        # Different raw text that truncates to same first 500 characters
        raw_cv_text2 = raw_cv_text[:500] + "EXTRA DIFFERENT CONTENT"
        import hashlib
        existing_hash = hashlib.sha256(raw_cv_text.encode()).hexdigest()
        mock_qdrant.get.return_value = {"cv_hash": existing_hash}

        with patch("app.services.ingestion_service.fetch_and_parse_cv") as mock_fetch, \
             patch("app.services.ingestion_service.truncate_to_prompt_cap") as mock_truncate:
            mock_fetch.return_value = raw_cv_text2
            mock_truncate.side_effect = lambda x: x[:500]  # simulate truncation
            await run_candidate_ingestion(
                candidate_id=123,
                cv_url="https://example.com/cv.pdf",
                profile_data={
                    "name": "John",
                    "location": "Remote",
                    "experience_level": "Senior",
                    "industry": "Tech",
                    "employment_type": "Full-time",
                    "candidate_version": 1
                },
                callback_url="https://example.com/callback",
                event_id="evt_123",
                qdrant=mock_qdrant,
                llm=mock_gemini,
                store=mock_store,
                callback_client=mock_callback,
            )

        # Hash mismatch should trigger full pipeline, not skip
        mock_gemini.generate.assert_called_once()
        mock_gemini.embed.assert_called_once()
        mock_qdrant.upsert.assert_called_once()
        mock_qdrant.update_payload.assert_not_called()
        payload = mock_qdrant.upsert.call_args[0][3]
        expected_hash = hashlib.sha256(raw_cv_text2.encode()).hexdigest()
        self.assertEqual(payload["cv_hash"], expected_hash)

    async def test_job_hash_differs_when_only_content_beyond_prompt_cap_changes(self):
        """Hash should differ when raw JD text differs beyond truncation limit."""
        mock_qdrant = MagicMock()
        mock_qdrant.get = MagicMock(return_value=None)
        mock_qdrant.update_payload = MagicMock()
        mock_qdrant.upsert = MagicMock()
        mock_gemini = MagicMock()
        mock_gemini.generate = MagicMock(return_value=json.dumps({
            "title": "Software Engineer",
            "location": "Remote",
            "experience_level": "Mid",
            "industry": "Tech",
            "employment_type": "Full-time",
            "required_skills": ["Python", "AWS"],
            "raw_jd_summary": "Looking for a software engineer with Python and AWS experience."
        }))
        mock_gemini.embed = MagicMock(return_value=[0.1] * 768)
        mock_store = MagicMock()
        mock_store.update = MagicMock()
        mock_callback = MagicMock()
        mock_callback.send = AsyncMock(return_value=True)

        # Raw JD text longer than prompt cap
        raw_jd_text = "B" * 1000
        raw_jd_text2 = raw_jd_text[:500] + "EXTRA DIFFERENT CONTENT"
        import hashlib
        existing_hash = hashlib.sha256(raw_jd_text.encode()).hexdigest()
        mock_qdrant.get.return_value = {"jd_hash": existing_hash}

        with patch("app.services.ingestion_service.truncate_to_prompt_cap") as mock_truncate:
            mock_truncate.side_effect = lambda x: x[:500]  # simulate truncation
            await run_job_ingestion(
                job_id=456,
                jd_text=raw_jd_text2,
                metadata={
                    "title": "Software Engineer",
                    "location": "Remote",
                    "experience_level": "Mid",
                    "industry": "Tech",
                    "employment_type": "Full-time",
                    "job_version": 1,
                    "company_name": "ORACLE",
                    "about": "Nice",
                },
                callback_url="https://example.com/callback",
                event_id="evt_456",
                qdrant=mock_qdrant,
                llm=mock_gemini,
                store=mock_store,
                callback_client=mock_callback,
            )

        # Hash mismatch should trigger full pipeline, not skip
        mock_gemini.generate.assert_called_once()
        mock_gemini.embed.assert_called_once()
        mock_qdrant.upsert.assert_called_once()
        mock_qdrant.update_payload.assert_not_called()
        payload = mock_qdrant.upsert.call_args[0][3]
        expected_hash = hashlib.sha256(raw_jd_text2.encode()).hexdigest()
        self.assertEqual(payload["jd_hash"], expected_hash)
        self.assertEqual(payload["company_name"], "ORACLE")
        self.assertEqual(payload["about"], "Nice")

class TestIngestionRetry(unittest.IsolatedAsyncioTestCase):
    """Verify retry behavior of candidate and job ingestion."""

    async def test_candidate_ingestion_retries_up_to_three_times(self):
        """Candidate ingestion should retry up to 3 times (4 total attempts) on transient failure."""
        mock_qdrant = MagicMock()
        mock_qdrant.get = MagicMock(return_value=None)
        mock_qdrant.update_payload = MagicMock()
        mock_qdrant.upsert = MagicMock()
        mock_gemini = MagicMock()
        mock_gemini.generate = MagicMock(side_effect=Exception("Gemini error"))
        mock_gemini.embed = MagicMock(side_effect=Exception("Embed error"))
        mock_store = MagicMock()
        mock_store.update = MagicMock()
        mock_callback = MagicMock()
        mock_callback.send = AsyncMock(return_value=True)

        with patch("app.services.ingestion_service.fetch_and_parse_cv") as mock_fetch, \
             patch("app.services.ingestion_service.truncate_to_prompt_cap") as mock_truncate, \
             patch("asyncio.sleep") as mock_sleep:
            mock_fetch.side_effect = Exception("CV fetch error")
            mock_truncate.side_effect = lambda x: x

            await run_candidate_ingestion(
                candidate_id=123,
                cv_url="https://example.com/cv.pdf",
                profile_data={
                    "name": "John",
                    "location": "Remote",
                    "experience_level": "Senior",
                    "industry": "Tech",
                    "employment_type": "Full-time",
                    "candidate_version": 1
                },
                callback_url="https://example.com/callback",
                event_id="evt_123",
                qdrant=mock_qdrant,
                llm=mock_gemini,
                store=mock_store,
                callback_client=mock_callback,
            )

        self.assertEqual(mock_fetch.call_count, 4)
        self.assertEqual(mock_sleep.call_count, 3)
        mock_sleep.assert_any_call(2)
        mock_sleep.assert_any_call(4)
        mock_sleep.assert_any_call(8)
        mock_store.update.assert_any_call("evt_123", status="failed", error_summary=ANY)
        mock_callback.send.assert_called_once_with(
            callback_url="https://example.com/callback",
            event_id="evt_123",
            entity_type="candidate",
            entity_id=123,
            status="failed",
            error=ANY,
        )

    async def test_job_ingestion_retries_up_to_three_times(self):
        """Job ingestion should retry up to 3 times (4 total attempts) on transient failure."""
        mock_qdrant = MagicMock()
        mock_qdrant.get = MagicMock(return_value=None)
        mock_qdrant.update_payload = MagicMock()
        mock_qdrant.upsert = MagicMock()
        mock_gemini = MagicMock()
        mock_gemini.generate = MagicMock(side_effect=Exception("Gemini error"))
        mock_gemini.embed = MagicMock(side_effect=Exception("Embed error"))
        mock_store = MagicMock()
        mock_store.update = MagicMock()
        mock_callback = MagicMock()
        mock_callback.send = AsyncMock(return_value=True)

        with patch("app.services.ingestion_service.truncate_to_prompt_cap") as mock_truncate, \
             patch("asyncio.sleep") as mock_sleep:
            mock_truncate.side_effect = lambda x: x

            await run_job_ingestion(
                job_id=456,
                jd_text="Job description text",
                metadata={
                    "title": "Software Engineer",
                    "location": "Remote",
                    "experience_level": "Mid",
                    "industry": "Tech",
                    "employment_type": "Full-time",
                    "job_version": 1,
                    "company_name": "ORACLE",
                    "about": "Nice",
                },
                callback_url="https://example.com/callback",
                event_id="evt_456",
                qdrant=mock_qdrant,
                llm=mock_gemini,
                store=mock_store,
                callback_client=mock_callback,
            )

        self.assertEqual(mock_truncate.call_count, 1)
        self.assertEqual(mock_sleep.call_count, 3)
        mock_sleep.assert_any_call(2)
        mock_sleep.assert_any_call(4)
        mock_sleep.assert_any_call(8)
        mock_store.update.assert_any_call("evt_456", status="failed", error_summary=ANY)
        mock_callback.send.assert_called_once_with(
            callback_url="https://example.com/callback",
            event_id="evt_456",
            entity_type="job",
            entity_id=456,
            status="failed",
            error=ANY,
        )

    async def test_candidate_ingestion_handles_malformed_extraction(self):
        """Verify that malformed extraction (non-list skills, non-string summary) is handled gracefully."""
        mock_qdrant = MagicMock()
        mock_qdrant.get = MagicMock(return_value=None)
        mock_qdrant.upsert = MagicMock()
        mock_gemini = MagicMock()
        mock_gemini.generate = MagicMock(return_value=json.dumps({
            "name": "John",
            "skills": "Python, ML",  # string instead of list
            "raw_profile_summary": {"key": "value"}  # dict instead of string
        }))
        mock_gemini.embed = MagicMock(return_value=[0.1] * 768)
        mock_store = MagicMock()
        mock_store.update = MagicMock()
        mock_callback = MagicMock()
        mock_callback.send = AsyncMock(return_value=True)

        with patch("app.services.ingestion_service.fetch_and_parse_cv") as mock_fetch, \
             patch("app.services.ingestion_service.truncate_to_prompt_cap") as mock_truncate:
            mock_fetch.return_value = "CV text"
            mock_truncate.side_effect = lambda x: x

            await run_candidate_ingestion(
                candidate_id=123,
                cv_url="https://example.com/cv.pdf",
                profile_data={
                    "name": "John",
                    "location": "Remote",
                    "experience_level": "Senior",
                    "industry": "Tech",
                    "employment_type": "Full-time",
                    "candidate_version": 1
                },
                callback_url="https://example.com/callback",
                event_id="evt_123",
                qdrant=mock_qdrant,
                llm=mock_gemini,
                store=mock_store,
                callback_client=mock_callback,
            )

        mock_qdrant.upsert.assert_called_once()
        payload = mock_qdrant.upsert.call_args[0][3]
        # skills should be empty list (fallback)
        self.assertEqual(payload["skills"], [])
        # raw_profile_summary should be string (default empty)
        self.assertIsInstance(payload["raw_profile_summary"], str)

    async def test_job_ingestion_handles_malformed_extraction(self):
        """Verify that malformed extraction (non-list required_skills, non-string summary) is handled gracefully."""
        mock_qdrant = MagicMock()
        mock_qdrant.get = MagicMock(return_value=None)
        mock_qdrant.upsert = MagicMock()
        mock_gemini = MagicMock()
        mock_gemini.generate = MagicMock(return_value=json.dumps({
            "title": "Software Engineer",
            "required_skills": "Python, AWS",  # string instead of list
            "raw_jd_summary": {"key": "value"}  # dict instead of string
        }))
        mock_gemini.embed = MagicMock(return_value=[0.1] * 768)
        mock_store = MagicMock()
        mock_store.update = MagicMock()
        mock_callback = MagicMock()
        mock_callback.send = AsyncMock(return_value=True)

        with patch("app.services.ingestion_service.truncate_to_prompt_cap") as mock_truncate:
            mock_truncate.side_effect = lambda x: x

            await run_job_ingestion(
                job_id=456,
                jd_text="Job description text",
                metadata={
                    "title": "Software Engineer",
                    "location": "Remote",
                    "experience_level": "Mid",
                    "industry": "Tech",
                    "employment_type": "Full-time",
                    "job_version": 1,
                    "company_name": "ORACLE",
                    "about": "Nice",
                },
                callback_url="https://example.com/callback",
                event_id="evt_456",
                qdrant=mock_qdrant,
                llm=mock_gemini,
                store=mock_store,
                callback_client=mock_callback,
            )

        mock_qdrant.upsert.assert_called_once()
        payload = mock_qdrant.upsert.call_args[0][3]
        # required_skills should be empty list (fallback)
        self.assertEqual(payload["required_skills"], [])
        # raw_jd_summary should be string (default empty)
        self.assertIsInstance(payload["raw_jd_summary"], str)

class TestIngestionCallbackSuppression(unittest.IsolatedAsyncioTestCase):
    """Verify that intermediate failure callbacks are suppressed during queue-managed retry flows."""

    async def test_candidate_suppress_callback_skips_callback_on_failure(self):
        """When suppress_callback=True and ingestion fails, callback_client.send should not be called."""
        mock_qdrant = MagicMock()
        mock_qdrant.get = MagicMock(return_value=None)
        mock_gemini = MagicMock()
        mock_gemini.generate = MagicMock(side_effect=Exception("LLM error"))
        mock_gemini.embed = MagicMock(side_effect=Exception("Embed error"))
        mock_store = MagicMock()
        mock_store.update = MagicMock()
        mock_callback = MagicMock()
        mock_callback.send = AsyncMock()

        with patch("app.services.ingestion_service.fetch_and_parse_cv") as mock_fetch, \
             patch("app.services.ingestion_service.truncate_to_prompt_cap") as mock_truncate, \
             patch("asyncio.sleep") as mock_sleep:
            mock_fetch.side_effect = Exception("Fetch error")
            mock_truncate.side_effect = lambda x: x

            await run_candidate_ingestion(
                candidate_id=123,
                cv_url="https://example.com/cv.pdf",
                profile_data={
                    "name": "John", "location": "Remote",
                    "experience_level": "Senior", "industry": "Tech",
                    "employment_type": "Full-time", "candidate_version": 1,
                },
                callback_url="https://example.com/callback",
                event_id="evt_suppress",
                qdrant=mock_qdrant,
                llm=mock_gemini,
                store=mock_store,
                callback_client=mock_callback,
                suppress_callback=True,
            )

        # Callback should NOT have been sent when suppress_callback=True
        mock_callback.send.assert_not_called()

    async def test_candidate_suppress_callback_suppresses_all_callbacks(self):
        """When suppress_callback=True, no callback is sent even on success — the caller handles it."""
        mock_qdrant = MagicMock()
        mock_qdrant.get = MagicMock(return_value=None)
        mock_qdrant.upsert = MagicMock()
        mock_gemini = MagicMock()
        mock_gemini.generate = MagicMock(return_value=json.dumps({
            "name": "John Doe", "location": "Remote",
            "experience_level": "Senior", "industry": "Tech",
            "employment_type": "Full-time", "skills": ["Python"],
            "past_roles": ["Engineer"],
            "raw_profile_summary": "Experienced engineer."
        }))
        mock_gemini.embed = MagicMock(return_value=[0.1] * 768)
        mock_store = MagicMock()
        mock_store.update = MagicMock()
        mock_callback = MagicMock()
        mock_callback.send = AsyncMock(return_value=True)

        with patch("app.services.ingestion_service.fetch_and_parse_cv") as mock_fetch, \
             patch("app.services.ingestion_service.truncate_to_prompt_cap") as mock_truncate:
            mock_fetch.return_value = "CV text"
            mock_truncate.side_effect = lambda x: x

            await run_candidate_ingestion(
                candidate_id=123,
                cv_url="https://example.com/cv.pdf",
                profile_data={
                    "name": "John", "location": "Remote",
                    "experience_level": "Senior", "industry": "Tech",
                    "employment_type": "Full-time", "candidate_version": 1,
                },
                callback_url="https://example.com/callback",
                event_id="evt_suppress_ok",
                qdrant=mock_qdrant,
                llm=mock_gemini,
                store=mock_store,
                callback_client=mock_callback,
                suppress_callback=True,
            )

        # No callback should be sent — the caller (process_queue_entry) handles it
        mock_callback.send.assert_not_called()

    async def test_candidate_enqueue_suppresses_failure_callback(self):
        """When ingest_queue is provided and all retries fail, callback is suppressed because the record is enqueued."""
        mock_qdrant = MagicMock()
        mock_qdrant.get = MagicMock(return_value=None)
        mock_gemini = MagicMock()
        mock_gemini.generate = MagicMock(side_effect=Exception("LLM error"))
        mock_gemini.embed = MagicMock(side_effect=Exception("Embed error"))
        mock_store = MagicMock()
        mock_store.update = MagicMock()
        mock_store.get_by_event_id = MagicMock()
        mock_callback = MagicMock()
        mock_callback.send = AsyncMock()
        mock_queue = MagicMock()
        mock_queue.enqueue = MagicMock()

        with patch("app.services.ingestion_service.fetch_and_parse_cv") as mock_fetch, \
             patch("app.services.ingestion_service.truncate_to_prompt_cap") as mock_truncate, \
             patch("asyncio.sleep") as mock_sleep, \
             patch("app.services.ingestion_service.get_settings") as mock_get_settings:
            mock_fetch.side_effect = Exception("Fetch error")
            mock_truncate.side_effect = lambda x: x
            mock_settings = MagicMock()
            mock_settings.INGEST_QUEUE_BACKOFF_BASE_SECONDS = 60
            mock_get_settings.return_value = mock_settings

            await run_candidate_ingestion(
                candidate_id=123,
                cv_url="https://example.com/cv.pdf",
                profile_data={
                    "name": "John", "location": "Remote",
                    "experience_level": "Senior", "industry": "Tech",
                    "employment_type": "Full-time", "candidate_version": 1,
                },
                callback_url="https://example.com/callback",
                event_id="evt_enqueue",
                qdrant=mock_qdrant,
                llm=mock_gemini,
                store=mock_store,
                callback_client=mock_callback,
                ingest_queue=mock_queue,
            )

        # Record should have been enqueued
        mock_queue.enqueue.assert_called_once()
        # Callback should NOT have been sent (suppressed by enqueue path)
        mock_callback.send.assert_not_called()

    async def test_job_suppress_callback_skips_callback_on_failure(self):
        """When suppress_callback=True and job ingestion fails, callback_client.send should not be called."""
        mock_qdrant = MagicMock()
        mock_qdrant.get = MagicMock(return_value=None)
        mock_gemini = MagicMock()
        mock_gemini.generate = MagicMock(side_effect=Exception("LLM error"))
        mock_gemini.embed = MagicMock(side_effect=Exception("Embed error"))
        mock_store = MagicMock()
        mock_store.update = MagicMock()
        mock_callback = MagicMock()
        mock_callback.send = AsyncMock()

        with patch("app.services.ingestion_service.truncate_to_prompt_cap") as mock_truncate, \
             patch("asyncio.sleep") as mock_sleep:
            mock_truncate.side_effect = lambda x: x

            await run_job_ingestion(
                job_id=456,
                jd_text="Job description text",
                metadata={
                    "title": "Engineer", "location": "Remote",
                    "experience_level": "Mid", "industry": "Tech",
                    "employment_type": "Full-time", "job_version": 1,
                    "company_name": "ACME", "about": "Nice",
                },
                callback_url="https://example.com/callback",
                event_id="evt_job_suppress",
                qdrant=mock_qdrant,
                llm=mock_gemini,
                store=mock_store,
                callback_client=mock_callback,
                suppress_callback=True,
            )

        mock_callback.send.assert_not_called()

    async def test_job_enqueue_suppresses_failure_callback(self):
        """When ingest_queue is provided and all retries fail, job callback is suppressed."""
        mock_qdrant = MagicMock()
        mock_qdrant.get = MagicMock(return_value=None)
        mock_gemini = MagicMock()
        mock_gemini.generate = MagicMock(side_effect=Exception("LLM error"))
        mock_gemini.embed = MagicMock(side_effect=Exception("Embed error"))
        mock_store = MagicMock()
        mock_store.update = MagicMock()
        mock_store.get_by_event_id = MagicMock()
        mock_callback = MagicMock()
        mock_callback.send = AsyncMock()
        mock_queue = MagicMock()
        mock_queue.enqueue = MagicMock()

        with patch("app.services.ingestion_service.truncate_to_prompt_cap") as mock_truncate, \
             patch("asyncio.sleep") as mock_sleep, \
             patch("app.services.ingestion_service.get_settings") as mock_get_settings:
            mock_truncate.side_effect = lambda x: x
            mock_settings = MagicMock()
            mock_settings.INGEST_QUEUE_BACKOFF_BASE_SECONDS = 60
            mock_get_settings.return_value = mock_settings

            await run_job_ingestion(
                job_id=456,
                jd_text="Job description text",
                metadata={
                    "title": "Engineer", "location": "Remote",
                    "experience_level": "Mid", "industry": "Tech",
                    "employment_type": "Full-time", "job_version": 1,
                    "company_name": "ACME", "about": "Nice",
                },
                callback_url="https://example.com/callback",
                event_id="evt_job_enqueue",
                qdrant=mock_qdrant,
                llm=mock_gemini,
                store=mock_store,
                callback_client=mock_callback,
                ingest_queue=mock_queue,
            )

        mock_queue.enqueue.assert_called_once()
        mock_callback.send.assert_not_called()


if __name__ == "__main__":
    unittest.main()