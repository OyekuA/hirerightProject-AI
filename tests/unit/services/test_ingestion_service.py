

import json
import asyncio
import unittest
from unittest.mock import MagicMock, patch, ANY, AsyncMock

from app.services.ingestion_service import (
    run_candidate_ingestion,
    run_job_ingestion,
    extract_candidate_entities,
    extract_job_entities,
)
from app.schemas.ingestion import CandidateExtraction, JobExtraction

class TestIngestionServicePayloads(unittest.IsolatedAsyncioTestCase):

    async def test_candidate_ingestion_includes_raw_profile_summary(self):

        mock_qdrant = MagicMock()
        mock_qdrant.upsert = MagicMock()
        mock_gemini = MagicMock()
        mock_gemini.generate = MagicMock(return_value=json.dumps({
            "name": "John Doe",
            "location": "Remote",
            "experience_level": "Senior",
            "industry": "Tech",
            "employment_type": "full_time",
            "skills": ["Python", "ML"],
            "past_roles": ["Engineer at Acme (Jan 2020 – Present)"],
            "raw_profile_summary": "Experienced software engineer with ML background.",
            "total_years_experience": None
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
                    "employment_type": "full_time",
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
        self.assertIn("total_years_experience", payload)

    async def test_job_ingestion_includes_raw_jd_summary(self):

        mock_qdrant = MagicMock()
        mock_qdrant.upsert = MagicMock()
        mock_gemini = MagicMock()
        mock_gemini.generate = MagicMock(return_value=json.dumps({
            "title": "Software Engineer",
            "location": "Remote",
            "experience_level": "Mid",
            "industry": "Tech",
            "employment_type": "full_time",
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
                    "employment_type": "full_time",
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

    async def test_candidate_hash_match_skips_gemini_calls(self):
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
            "employment_type": "full_time",
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
                    "employment_type": "full_time",
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
        self.assertEqual(mock_gemini.embed.call_count, 2)  # profile vector + skills_vector
        mock_qdrant.upsert.assert_called_once()
        mock_qdrant.update_payload.assert_not_called()
        payload = mock_qdrant.upsert.call_args[0][3]
        self.assertEqual(payload["candidate_version"], 2)
        self.assertIn("ingested_at", payload)
        self.assertEqual(payload["cv_hash"], new_hash)

    async def test_candidate_hash_mismatch_runs_full_pipeline(self):

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
            "employment_type": "full_time",
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
                    "employment_type": "full_time",
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
        self.assertEqual(mock_gemini.embed.call_count, 2)  # profile vector + skills_vector
        mock_qdrant.upsert.assert_called_once()
        mock_qdrant.update_payload.assert_not_called()
        payload = mock_qdrant.upsert.call_args[0][3]
        self.assertIn("cv_hash", payload)
        self.assertEqual(payload["cv_hash"], new_hash)

    async def test_candidate_no_existing_payload_runs_full_pipeline(self):

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
            "employment_type": "full_time",
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
                    "employment_type": "full_time",
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
        self.assertEqual(mock_gemini.embed.call_count, 2)  # profile vector + skills_vector
        mock_qdrant.upsert.assert_called_once()
        mock_qdrant.update_payload.assert_not_called()
        payload = mock_qdrant.upsert.call_args[0][3]
        self.assertIn("cv_hash", payload)
        self.assertEqual(payload["cv_hash"], new_hash)

    async def test_job_hash_match_skips_gemini_calls(self):
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
            "employment_type": "full_time",
            "required_skills": ["Python"],
            "raw_jd_summary": "Job summary."
        }))
        mock_gemini.embed = MagicMock(return_value=[0.1] * 768)
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
                    "employment_type": "full_time",
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
        self.assertEqual(mock_gemini.embed.call_count, 2)  # profile vector + skills_vector
        mock_qdrant.upsert.assert_called_once()
        mock_qdrant.update_payload.assert_not_called()
        payload = mock_qdrant.upsert.call_args[0][3]
        self.assertEqual(payload["job_version"], 2)
        self.assertIn("ingested_at", payload)
        self.assertEqual(payload["jd_hash"], new_hash)
        self.assertEqual(payload["company_name"], "ORACLE")
        self.assertEqual(payload["about"], "Nice")

    async def test_job_hash_mismatch_runs_full_pipeline(self):

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
            "employment_type": "full_time",
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
                    "employment_type": "full_time",
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
        self.assertEqual(mock_gemini.embed.call_count, 2)  # profile vector + skills_vector
        mock_qdrant.upsert.assert_called_once()
        mock_qdrant.update_payload.assert_not_called()
        payload = mock_qdrant.upsert.call_args[0][3]
        self.assertIn("jd_hash", payload)
        self.assertEqual(payload["jd_hash"], new_hash)
        self.assertEqual(payload["company_name"], "ORACLE")
        self.assertEqual(payload["about"], "Nice")

    async def test_job_no_existing_payload_runs_full_pipeline(self):

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
            "employment_type": "full_time",
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
                    "employment_type": "full_time",
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
        self.assertEqual(mock_gemini.embed.call_count, 2)  # profile vector + skills_vector
        mock_qdrant.upsert.assert_called_once()
        mock_qdrant.update_payload.assert_not_called()
        payload = mock_qdrant.upsert.call_args[0][3]
        self.assertIn("jd_hash", payload)
        self.assertEqual(payload["jd_hash"], new_hash)
        self.assertEqual(payload["company_name"], "ORACLE")
        self.assertEqual(payload["about"], "Nice")

    async def test_candidate_hash_differs_when_only_content_beyond_prompt_cap_changes(self):

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
            "employment_type": "full_time",
            "skills": ["Python", "ML"],
            "past_roles": ["Engineer"],
            "raw_profile_summary": "Experienced software engineer with ML background."
        }))
        mock_gemini.embed = MagicMock(return_value=[0.1] * 768)
        mock_store = MagicMock()
        mock_store.update = MagicMock()
        mock_callback = MagicMock()
        mock_callback.send = AsyncMock(return_value=True)

        raw_cv_text = "A" * 1000
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
                    "employment_type": "full_time",
                    "candidate_version": 1
                },
                callback_url="https://example.com/callback",
                event_id="evt_123",
                qdrant=mock_qdrant,
                llm=mock_gemini,
                store=mock_store,
                callback_client=mock_callback,
            )

        mock_gemini.generate.assert_called_once()
        self.assertEqual(mock_gemini.embed.call_count, 2)  # profile vector + skills_vector
        mock_qdrant.upsert.assert_called_once()
        mock_qdrant.update_payload.assert_not_called()
        payload = mock_qdrant.upsert.call_args[0][3]
        expected_hash = hashlib.sha256(raw_cv_text2.encode()).hexdigest()
        self.assertEqual(payload["cv_hash"], expected_hash)

    async def test_job_hash_differs_when_only_content_beyond_prompt_cap_changes(self):

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
            "employment_type": "full_time",
            "required_skills": ["Python", "AWS"],
            "raw_jd_summary": "Looking for a software engineer with Python and AWS experience."
        }))
        mock_gemini.embed = MagicMock(return_value=[0.1] * 768)
        mock_store = MagicMock()
        mock_store.update = MagicMock()
        mock_callback = MagicMock()
        mock_callback.send = AsyncMock(return_value=True)

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
                    "employment_type": "full_time",
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

        mock_gemini.generate.assert_called_once()
        self.assertEqual(mock_gemini.embed.call_count, 2)  # profile vector + skills_vector
        mock_qdrant.upsert.assert_called_once()
        mock_qdrant.update_payload.assert_not_called()
        payload = mock_qdrant.upsert.call_args[0][3]
        expected_hash = hashlib.sha256(raw_jd_text2.encode()).hexdigest()
        self.assertEqual(payload["jd_hash"], expected_hash)
        self.assertEqual(payload["company_name"], "ORACLE")
        self.assertEqual(payload["about"], "Nice")

class TestIngestionRetry(unittest.IsolatedAsyncioTestCase):

    async def test_candidate_ingestion_retries_up_to_three_times(self):

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
                    "employment_type": "full_time",
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
                    "employment_type": "full_time",
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
                    "employment_type": "full_time",
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
        self.assertEqual(payload["skills"], [])
        self.assertIsInstance(payload["raw_profile_summary"], str)

    async def test_job_ingestion_handles_malformed_extraction(self):

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
                    "employment_type": "full_time",
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
        self.assertEqual(payload["required_skills"], [])
        self.assertIsInstance(payload["raw_jd_summary"], str)

class TestIngestionCallbackSuppression(unittest.IsolatedAsyncioTestCase):

    async def test_candidate_suppress_callback_skips_callback_on_failure(self):

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
                    "employment_type": "full_time", "candidate_version": 1,
                },
                callback_url="https://example.com/callback",
                event_id="evt_suppress",
                qdrant=mock_qdrant,
                llm=mock_gemini,
                store=mock_store,
                callback_client=mock_callback,
                suppress_callback=True,
            )

        mock_callback.send.assert_not_called()

    async def test_candidate_suppress_callback_suppresses_all_callbacks(self):

        mock_qdrant = MagicMock()
        mock_qdrant.get = MagicMock(return_value=None)
        mock_qdrant.upsert = MagicMock()
        mock_gemini = MagicMock()
        mock_gemini.generate = MagicMock(return_value=json.dumps({
            "name": "John Doe", "location": "Remote",
            "experience_level": "Senior", "industry": "Tech",
            "employment_type": "full_time", "skills": ["Python"],
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
                    "employment_type": "full_time", "candidate_version": 1,
                },
                callback_url="https://example.com/callback",
                event_id="evt_suppress_ok",
                qdrant=mock_qdrant,
                llm=mock_gemini,
                store=mock_store,
                callback_client=mock_callback,
                suppress_callback=True,
            )

        mock_callback.send.assert_not_called()

    async def test_candidate_enqueue_suppresses_failure_callback(self):

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
                    "employment_type": "full_time", "candidate_version": 1,
                },
                callback_url="https://example.com/callback",
                event_id="evt_enqueue",
                qdrant=mock_qdrant,
                llm=mock_gemini,
                store=mock_store,
                callback_client=mock_callback,
                ingest_queue=mock_queue,
            )

        mock_queue.enqueue.assert_called_once()
        mock_callback.send.assert_not_called()

    async def test_job_suppress_callback_skips_callback_on_failure(self):

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
                    "employment_type": "full_time", "job_version": 1,
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
                    "employment_type": "full_time", "job_version": 1,
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

class TestExtractionHelpers(unittest.IsolatedAsyncioTestCase):

    def test_extract_candidate_entities_valid(self):

        mock_llm = MagicMock()
        mock_llm.generate.return_value = json.dumps({
            "name": "Jane Doe",
            "location": "Remote",
            "experience_level": "Senior",
            "industry": "Tech",
            "employment_type": "full_time",
            "skills": ["Python", "ML"],
            "past_roles": ["Engineer at Acme (Jan 2020 – Present)"],
            "raw_profile_summary": "Experienced engineer.",
        })

        result = extract_candidate_entities(
            cv_text="CV content here",
            profile_data_json=json.dumps({"name": "Jane"}),
            llm=mock_llm,
        )
        self.assertEqual(result.name, "Jane Doe")
        self.assertEqual(result.skills, ["Python", "ML"])
        self.assertEqual(result.raw_profile_summary, "Experienced engineer.")
        self.assertIsNotNone(result.total_years_experience)

    def test_extract_candidate_entities_fallback_on_validation_error(self):

        mock_llm = MagicMock()
        mock_llm.generate.return_value = json.dumps({
            "name": "Jane",
            "skills": "not a list",  # should be list
            "raw_profile_summary": None,
        })

        result = extract_candidate_entities(
            cv_text="CV content",
            profile_data_json="{}",
            llm=mock_llm,
        )
        self.assertIsInstance(result, CandidateExtraction)
        self.assertEqual(result.skills, [])
        self.assertIsInstance(result.raw_profile_summary, str)

    def test_extract_candidate_entities_raw_profile_summary_fallback(self):

        generated_text = json.dumps({
            "name": "John",
            "location": "NYC",
            "experience_level": "Mid",
            "industry": "Finance",
            "employment_type": "full_time",
            "skills": [],
            "past_roles": [],
            "raw_profile_summary": None,
        })

        mock_llm = MagicMock()
        mock_llm.generate.return_value = generated_text

        result = extract_candidate_entities(
            cv_text="Some CV",
            profile_data_json="{}",
            llm=mock_llm,
        )
        self.assertEqual(result.raw_profile_summary, generated_text[:500])

    def test_extract_job_entities_valid(self):

        mock_llm = MagicMock()
        mock_llm.generate.return_value = json.dumps({
            "title": "Software Engineer",
            "location": "Remote",
            "experience_level": "Senior",
            "industry": "Tech",
            "employment_type": "full_time",
            "required_skills": ["Python", "AWS"],
            "raw_jd_summary": "Looking for an engineer.",
        })

        result = extract_job_entities(
            jd_text="JD content",
            metadata_json=json.dumps({"title": "Engineer"}),
            llm=mock_llm,
        )
        self.assertEqual(result.title, "Software Engineer")
        self.assertEqual(result.required_skills, ["Python", "AWS"])
        self.assertEqual(result.raw_jd_summary, "Looking for an engineer.")

    def test_extract_job_entities_fallback_on_validation_error(self):

        mock_llm = MagicMock()
        mock_llm.generate.return_value = json.dumps({
            "title": "Engineer",
            "required_skills": "not a list",  # wrong type
            "raw_jd_summary": None,
        })

        result = extract_job_entities(
            jd_text="JD content",
            metadata_json="{}",
            llm=mock_llm,
        )
        self.assertIsInstance(result, JobExtraction)
        self.assertEqual(result.required_skills, [])
        self.assertIsInstance(result.raw_jd_summary, str)

    def test_extract_job_entities_raw_jd_summary_fallback(self):

        generated_text = json.dumps({
            "title": "Engineer",
            "location": "Remote",
            "experience_level": "Senior",
            "industry": "Tech",
            "employment_type": "full_time",
            "required_skills": ["Python"],
            "raw_jd_summary": None,
        })

        mock_llm = MagicMock()
        mock_llm.generate.return_value = generated_text

        result = extract_job_entities(
            jd_text="Some JD",
            metadata_json="{}",
            llm=mock_llm,
        )
        self.assertEqual(result.raw_jd_summary, generated_text[:500])

class TestYearsOfExperienceComputation(unittest.TestCase):
    """Test the _compute_total_years_experience helper."""

    def test_multiple_roles_non_overlapping(self):
        from app.services.ingestion_service import _compute_total_years_experience
        roles = [
            "Junior Dev at Acme (Jan 2015 – Dec 2017)",
            "Senior Dev at Beta (Jan 2018 – Dec 2020)",
            "Lead at Gamma (Jan 2021 – Present)",
        ]
        result = _compute_total_years_experience(roles)
        self.assertIsNotNone(result)
        self.assertGreater(result, 0)

    def test_overlapping_roles_merged(self):
        from app.services.ingestion_service import _compute_total_years_experience
        roles = [
            "Junior at Acme (Jan 2015 – Dec 2019)",
            "Senior at Acme (Jan 2018 – Dec 2022)",  # overlaps
        ]
        result = _compute_total_years_experience(roles)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result, 7.0, delta=1)

    def test_present_uses_current_year(self):
        from app.services.ingestion_service import _compute_total_years_experience
        roles = [
            "Engineer at Co (Jan 2020 – Present)",
        ]
        result = _compute_total_years_experience(roles)
        self.assertIsNotNone(result)
        self.assertGreater(result, 0)

    def test_no_dates_returns_none(self):
        from app.services.ingestion_service import _compute_total_years_experience
        roles = ["Engineer at Acme", "Manager at Beta"]
        result = _compute_total_years_experience(roles)
        self.assertIsNone(result)

    def test_mixed_dates_and_no_dates(self):
        from app.services.ingestion_service import _compute_total_years_experience
        roles = [
            "Engineer at Acme",  # no date
            "Senior at Beta (Jan 2020 – Dec 2023)",
        ]
        result = _compute_total_years_experience(roles)
        self.assertIsNotNone(result)
        # Only the dated role counts: 2020-2023 = 3 years
        self.assertAlmostEqual(result, 3.0, delta=1)

    def test_various_date_formats(self):
        from app.services.ingestion_service import _compute_total_years_experience
        roles = [
            "Role at A (2020 - 2022)",  # year only
            "Role at B (Jan 2021 – Present)",  # with month
            "Role at C (2018 to 2020)",  # 'to' separator
            "Role at D (2019-2021)",  # hyphen separator
        ]
        result = _compute_total_years_experience(roles)
        self.assertIsNotNone(result)
        self.assertGreater(result, 0)

    def test_no_fabricated_number_when_no_date(self):
        from app.services.ingestion_service import _compute_total_years_experience
        roles = []  # empty list
        result = _compute_total_years_experience(roles)
        self.assertIsNone(result)

    def test_total_years_persisted_in_payload(self):
        from app.services.ingestion_service import (
            _compute_total_years_experience,
            extract_candidate_entities,
        )
        mock_llm = MagicMock()
        mock_llm.generate.return_value = json.dumps({
            "name": "Jane",
            "location": "Remote",
            "experience_level": "Senior",
            "industry": "Tech",
            "employment_type": "full_time",
            "skills": ["Python"],
            "past_roles": ["Engineer at Acme (Jan 2020 – Dec 2023)"],
            "raw_profile_summary": "Engineer.",
        })
        result = extract_candidate_entities(
            cv_text="CV",
            profile_data_json="{}",
            llm=mock_llm,
        )
        self.assertIsNotNone(result.total_years_experience)
        self.assertAlmostEqual(result.total_years_experience, 3.0, delta=1)


class TestCandidateDataSource(unittest.IsolatedAsyncioTestCase):

    async def test_candidate_ingestion_includes_data_source_in_upsert(self):

        mock_qdrant = MagicMock()
        mock_qdrant.upsert = MagicMock()
        mock_gemini = MagicMock()
        mock_gemini.generate = MagicMock(return_value=json.dumps({
            "name": "John Doe",
            "location": "Remote",
            "experience_level": "Senior",
            "industry": "Tech",
            "employment_type": "full_time",
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
                    "employment_type": "full_time",
                    "candidate_version": 1,
                    "data_source": "indeed",
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
        self.assertIn("data_source", payload)
        self.assertEqual(payload["data_source"], "indeed")
        self.assertIn("raw_profile_summary", payload)

    async def test_candidate_hash_match_includes_data_source_in_update_payload(self):
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
            "employment_type": "full_time",
            "skills": ["Python"],
            "past_roles": ["Engineer"],
            "raw_profile_summary": "Summary."
        }))
        mock_gemini.embed = MagicMock(return_value=[0.1] * 768)
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
                    "employment_type": "full_time",
                    "candidate_version": 2,
                    "data_source": "linkedin",
                },
                callback_url="https://example.com/callback",
                event_id="evt_123",
                qdrant=mock_qdrant,
                llm=mock_gemini,
                store=mock_store,
                callback_client=mock_callback,
            )

        mock_qdrant.upsert.assert_called_once()
        payload_fields = mock_qdrant.upsert.call_args[0][3]
        self.assertIn("data_source", payload_fields)
        self.assertEqual(payload_fields["data_source"], "linkedin")
        self.assertEqual(payload_fields["candidate_version"], 2)


class TestEmploymentTypeAndWorkModeValidation(unittest.TestCase):

    def test_work_mode_constants(self):
        from app.constants import WORK_MODES
        self.assertEqual(WORK_MODES, ("remote", "hybrid", "onsite"))

    def test_employment_type_constants(self):
        from app.constants import EMPLOYMENT_TYPES
        self.assertIn("full_time", EMPLOYMENT_TYPES)
        self.assertIn("part_time", EMPLOYMENT_TYPES)
        self.assertIn("contract", EMPLOYMENT_TYPES)
        self.assertIn("self_employed", EMPLOYMENT_TYPES)

    def test_validate_employment_type_accepts_canonical(self):
        from app.schemas.ingestion import _validate_employment_type
        self.assertEqual(_validate_employment_type("full_time"), "full_time")
        self.assertEqual(_validate_employment_type("Full_Time"), "full_time")
        self.assertEqual(_validate_employment_type(" contract "), "contract")
        self.assertEqual(_validate_employment_type("self_employed"), "self_employed")

    def test_validate_employment_type_rejects_non_canonical(self):
        from pydantic import ValidationError
        from app.schemas.ingestion import ProfileData
        with self.assertRaises(ValidationError) as ctx:
            ProfileData(
                name="A", location="Lagos", experience_level="Senior",
                industry="fintech", employment_type="full-time",
                candidate_version=1,
            )
        self.assertIn("full-time", str(ctx.exception))
        with self.assertRaises(ValidationError):
            ProfileData(
                name="A", location="Lagos", experience_level="Senior",
                industry="fintech", employment_type="FT",
                candidate_version=1,
            )

    def test_validate_work_mode_accepts_canonical(self):
        from app.schemas.ingestion import _validate_work_mode
        self.assertEqual(_validate_work_mode("remote"), "remote")
        self.assertEqual(_validate_work_mode("Hybrid"), "hybrid")

    def test_job_metadata_rejects_hyphen_employment_type(self):
        from pydantic import ValidationError
        from app.schemas.ingestion import JobMetadata
        with self.assertRaises(ValidationError):
            JobMetadata(
                title="Engineer",
                location="Remote",
                experience_level="Senior",
                industry="Tech",
                employment_type="full-time",
            )

    def test_job_metadata_accepts_addon_fields(self):
        from app.schemas.ingestion import JobMetadata
        meta = JobMetadata(
            title="Engineer",
            location="Remote",
            experience_level="Senior",
            industry="Tech",
            employment_type="full_time",
            description="desc",
            requirements="req",
            responsibilities="resp",
            benefits="ben",
            salary_min=50000,
            salary_max=90000,
            salary_currency="USD",
            work_mode="hybrid",
            remote_regions=["EMEA"],
        )
        self.assertEqual(meta.salary_min, 50000)
        self.assertEqual(meta.salary_max, 90000)
        self.assertEqual(meta.salary_currency, "USD")
        self.assertEqual(meta.work_mode, "hybrid")
        self.assertEqual(meta.remote_regions, ["EMEA"])
        self.assertEqual(meta.description, "desc")

    def test_profile_data_accepts_canonical_employment_type_and_work_mode(self):
        from app.schemas.ingestion import ProfileData
        profile = ProfileData(
            name="A",
            location="X",
            experience_level="Senior",
            industry="Tech",
            employment_type="part_time",
            candidate_version=1,
            work_mode="remote",
        )
        self.assertEqual(profile.employment_type, "part_time")
        self.assertEqual(profile.work_mode, "remote")

    def test_profile_data_rejects_non_canonical_employment_type(self):
        from app.schemas.ingestion import ProfileData
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            ProfileData(
                name="A",
                location="X",
                experience_level="Senior",
                industry="Tech",
                employment_type="part-time",
                candidate_version=1,
            )

    def test_profile_data_rejects_invalid_work_mode(self):
        from app.schemas.ingestion import ProfileData
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            ProfileData(
                name="A",
                location="X",
                experience_level="Senior",
                industry="Tech",
                employment_type="full_time",
                candidate_version=1,
                work_mode="in-office",
            )

class TestJobMetadataPersistence(unittest.IsolatedAsyncioTestCase):

    _FULL_METADATA = {
        "title": "Software Engineer",
        "location": "Remote",
        "experience_level": "Mid",
        "industry": "Tech",
        "employment_type": "full_time",
        "job_version": 1,
        "company_name": "ORACLE",
        "about": "Nice",
        "description": "We need a Python engineer.",
        "requirements": "Python and AWS.",
        "responsibilities": "Build APIs.",
        "benefits": "Health and 401k.",
        "salary_min": 90000,
        "salary_max": 120000,
        "salary_currency": "USD",
        "work_mode": "remote",
        "remote_regions": ["EMEA", "Americas"],
    }

    _ADDON_KEYS = [
        "description", "requirements", "responsibilities", "benefits",
        "salary_min", "salary_max", "salary_currency", "work_mode", "remote_regions",
    ]

    def _job_mocks(self, existing_payload=None):
        mock_qdrant = MagicMock()
        mock_qdrant.get = MagicMock(return_value=existing_payload)
        mock_qdrant.upsert = MagicMock()
        mock_qdrant.update_payload = MagicMock()
        mock_gemini = MagicMock()
        mock_gemini.generate = MagicMock(return_value=json.dumps({
            "title": "Software Engineer",
            "location": "Remote",
            "experience_level": "Mid",
            "industry": "Tech",
            "employment_type": "full_time",
            "required_skills": ["Python", "AWS"],
            "raw_jd_summary": "Looking for a software engineer with Python and AWS experience."
        }))
        mock_gemini.embed = MagicMock(return_value=[0.1] * 768)
        mock_store = MagicMock()
        mock_store.update = MagicMock()
        mock_callback = MagicMock()
        mock_callback.send = AsyncMock(return_value=True)
        return mock_qdrant, mock_gemini, mock_store, mock_callback

    async def test_job_fresh_ingest_persists_addon_keys(self):
        mock_qdrant, mock_gemini, mock_store, mock_callback = self._job_mocks(existing_payload=None)
        with patch("app.services.ingestion_service.truncate_to_prompt_cap") as mock_truncate:
            mock_truncate.side_effect = lambda x: x
            await run_job_ingestion(
                job_id=456,
                jd_text="Job description text",
                metadata=dict(self._FULL_METADATA),
                callback_url="https://example.com/callback",
                event_id="evt_456",
                qdrant=mock_qdrant,
                llm=mock_gemini,
                store=mock_store,
                callback_client=mock_callback,
            )
        mock_qdrant.upsert.assert_called_once()
        payload = mock_qdrant.upsert.call_args[0][3]
        for key in self._ADDON_KEYS:
            self.assertIn(key, payload)
        self.assertEqual(payload["work_mode"], "remote")
        self.assertEqual(payload["salary_min"], 90000)
        self.assertEqual(payload["salary_max"], 120000)
        self.assertEqual(payload["salary_currency"], "USD")
        self.assertEqual(payload["remote_regions"], ["EMEA", "Americas"])
        self.assertEqual(payload["description"], "We need a Python engineer.")

    async def test_job_hash_match_updates_addon_keys(self):
        jd_text = "Job description text"
        import hashlib
        new_hash = hashlib.sha256(jd_text.encode()).hexdigest()
        mock_qdrant, mock_gemini, mock_store, mock_callback = self._job_mocks(
            existing_payload={"jd_hash": new_hash}
        )
        with patch("app.services.ingestion_service.truncate_to_prompt_cap") as mock_truncate:
            mock_truncate.side_effect = lambda x: x
            await run_job_ingestion(
                job_id=456,
                jd_text=jd_text,
                metadata=dict(self._FULL_METADATA),
                callback_url="https://example.com/callback",
                event_id="evt_456",
                qdrant=mock_qdrant,
                llm=mock_gemini,
                store=mock_store,
                callback_client=mock_callback,
            )
        mock_qdrant.upsert.assert_called_once()
        payload_fields = mock_qdrant.upsert.call_args[0][3]
        for key in self._ADDON_KEYS:
            self.assertIn(key, payload_fields)
        self.assertEqual(payload_fields["work_mode"], "remote")
        self.assertEqual(payload_fields["salary_min"], 90000)
        self.assertEqual(payload_fields["remote_regions"], ["EMEA", "Americas"])

    async def test_job_fresh_ingest_none_safe_addons(self):
        mock_qdrant, mock_gemini, mock_store, mock_callback = self._job_mocks(existing_payload=None)
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
                    "employment_type": "full_time",
                },
                callback_url="https://example.com/callback",
                event_id="evt_456",
                qdrant=mock_qdrant,
                llm=mock_gemini,
                store=mock_store,
                callback_client=mock_callback,
            )
        payload = mock_qdrant.upsert.call_args[0][3]
        for key in self._ADDON_KEYS:
            self.assertIn(key, payload)
        self.assertIsNone(payload["work_mode"])
        self.assertIsNone(payload["salary_min"])
        self.assertIsNone(payload["remote_regions"])

class TestCandidateWorkModePersistence(unittest.IsolatedAsyncioTestCase):

    def _candidate_mocks(self, existing_payload=None):
        mock_qdrant = MagicMock()
        mock_qdrant.get = MagicMock(return_value=existing_payload)
        mock_qdrant.upsert = MagicMock()
        mock_qdrant.update_payload = MagicMock()
        mock_gemini = MagicMock()
        mock_gemini.generate = MagicMock(return_value=json.dumps({
            "name": "John Doe",
            "location": "Remote",
            "experience_level": "Senior",
            "industry": "Tech",
            "employment_type": "full_time",
            "skills": ["Python", "ML"],
            "past_roles": ["Engineer"],
            "raw_profile_summary": "Experienced software engineer with ML background.",
            "total_years_experience": None
        }))
        mock_gemini.embed = MagicMock(return_value=[0.1] * 768)
        mock_store = MagicMock()
        mock_store.update = MagicMock()
        mock_callback = MagicMock()
        mock_callback.send = AsyncMock(return_value=True)
        return mock_qdrant, mock_gemini, mock_store, mock_callback

    async def test_candidate_fresh_ingest_persists_profile_work_mode(self):
        mock_qdrant, mock_gemini, mock_store, mock_callback = self._candidate_mocks(existing_payload=None)
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
                    "employment_type": "full_time",
                    "candidate_version": 1,
                    "work_mode": "hybrid",
                },
                callback_url="https://example.com/callback",
                event_id="evt_123",
                qdrant=mock_qdrant,
                llm=mock_gemini,
                store=mock_store,
                callback_client=mock_callback,
            )
        payload = mock_qdrant.upsert.call_args[0][3]
        self.assertEqual(payload["work_mode"], "hybrid")

    async def test_candidate_fresh_ingest_extraction_work_mode_wins(self):
        mock_qdrant, mock_gemini, mock_store, mock_callback = self._candidate_mocks(existing_payload=None)
        mock_gemini.generate.return_value = json.dumps({
            "name": "John Doe",
            "location": "Remote",
            "experience_level": "Senior",
            "industry": "Tech",
            "employment_type": "full_time",
            "work_mode": "remote",
            "skills": ["Python"],
            "past_roles": ["Engineer"],
            "raw_profile_summary": "Experienced engineer.",
            "total_years_experience": None
        })
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
                    "employment_type": "full_time",
                    "candidate_version": 1,
                },
                callback_url="https://example.com/callback",
                event_id="evt_123",
                qdrant=mock_qdrant,
                llm=mock_gemini,
                store=mock_store,
                callback_client=mock_callback,
            )
        payload = mock_qdrant.upsert.call_args[0][3]
        self.assertEqual(payload["work_mode"], "remote")

    async def test_candidate_hash_match_persists_work_mode_and_fresh_total_years(self):
        cv_text = "CV text"
        import hashlib
        new_hash = hashlib.sha256(cv_text.encode()).hexdigest()
        mock_qdrant, mock_gemini, mock_store, mock_callback = self._candidate_mocks(
            existing_payload={"cv_hash": new_hash, "total_years_experience": 2.0}
        )
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
                    "employment_type": "full_time",
                    "candidate_version": 2,
                    "work_mode": "hybrid",
                    "total_years_experience": 8.5,
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
        self.assertEqual(payload["work_mode"], "hybrid")
        self.assertIsNone(payload["total_years_experience"])

class TestEmbedTextEnrichment(unittest.IsolatedAsyncioTestCase):

    async def test_job_embed_text_enriched_and_labeled(self):
        mock_qdrant = MagicMock()
        mock_qdrant.get = MagicMock(return_value=None)
        mock_qdrant.upsert = MagicMock()
        mock_gemini = MagicMock()
        mock_gemini.generate = MagicMock(return_value=json.dumps({
            "title": "Software Engineer",
            "location": "Remote",
            "experience_level": "Mid",
            "industry": "Tech",
            "employment_type": "full_time",
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
                    "employment_type": "full_time",
                    "work_mode": "remote",
                    "remote_regions": ["EMEA"],
                    "description": "We need a Python engineer.",
                    "requirements": "Python and AWS.",
                    "responsibilities": "Build APIs.",
                    "benefits": "Health and 401k.",
                    "salary_min": 90000,
                    "salary_max": 120000,
                    "salary_currency": "USD",
                },
                callback_url="https://example.com/callback",
                event_id="evt_456",
                qdrant=mock_qdrant,
                llm=mock_gemini,
                store=mock_store,
                callback_client=mock_callback,
            )

        self.assertEqual(mock_gemini.embed.call_count, 2)  # profile vector + skills_vector
        text = mock_gemini.embed.call_args_list[0][0][0]  # first call = profile/job vector
        self.assertIn("Job Title: Software Engineer", text)
        self.assertIn("Required Skills: Python, AWS", text)
        self.assertIn("Summary: Looking for a software engineer", text)
        self.assertIn("Location: Remote", text)
        self.assertIn("Experience Level: Mid", text)
        self.assertIn("Work Mode: remote", text)
        self.assertNotIn("Employment Type:", text)
        self.assertNotIn("Remote Regions:", text)
        self.assertNotIn("Description:", text)
        self.assertNotIn("Requirements:", text)
        self.assertNotIn("Responsibilities:", text)
        self.assertNotIn("Benefits:", text)
        self.assertNotIn("Salary:", text)

    async def test_job_embed_text_none_safe_skips_missing_values(self):
        mock_qdrant = MagicMock()
        mock_qdrant.get = MagicMock(return_value=None)
        mock_qdrant.upsert = MagicMock()
        mock_gemini = MagicMock()
        mock_gemini.generate = MagicMock(return_value=json.dumps({
            "title": "Software Engineer",
            "location": "Remote",
            "experience_level": "Mid",
            "industry": "Tech",
            "employment_type": "full_time",
            "required_skills": ["Python", "AWS"],
            "raw_jd_summary": "Looking for a software engineer."
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
                    "employment_type": "full_time",
                },
                callback_url="https://example.com/callback",
                event_id="evt_456",
                qdrant=mock_qdrant,
                llm=mock_gemini,
                store=mock_store,
                callback_client=mock_callback,
            )

        text = mock_gemini.embed.call_args_list[0][0][0]  # first call = profile/job vector
        self.assertNotIn("Salary:", text)
        self.assertNotIn("Work Mode:", text)
        self.assertNotIn("Benefits:", text)
        self.assertNotIn("Description:", text)
        self.assertIn("Summary: Looking for a software engineer.", text)

    async def test_candidate_embed_text_excludes_name(self):
        mock_qdrant = MagicMock()
        mock_qdrant.get = MagicMock(return_value=None)
        mock_qdrant.upsert = MagicMock()
        mock_gemini = MagicMock()
        mock_gemini.generate = MagicMock(return_value=json.dumps({
            "name": "John Doe",
            "location": "Remote",
            "experience_level": "Senior",
            "industry": "Tech",
            "employment_type": "full_time",
            "work_mode": "hybrid",
            "skills": ["Python", "ML"],
            "past_roles": ["Engineer at Acme"],
            "raw_profile_summary": "Experienced software engineer with ML background.",
            "total_years_experience": 5
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
                    "name": "Alice",
                    "location": "Remote",
                    "experience_level": "Senior",
                    "industry": "Tech",
                    "employment_type": "full_time",
                    "candidate_version": 1,
                },
                callback_url="https://example.com/callback",
                event_id="evt_123",
                qdrant=mock_qdrant,
                llm=mock_gemini,
                store=mock_store,
                callback_client=mock_callback,
            )

        self.assertEqual(mock_gemini.embed.call_count, 2)  # profile vector + skills_vector
        text = mock_gemini.embed.call_args_list[0][0][0]  # first call = profile/job vector
        self.assertNotIn("Alice", text)
        self.assertNotIn("John Doe", text)
        self.assertIn("Role Headline: Engineer", text)
        self.assertIn("Industry: Tech", text)
        self.assertIn("Location: Remote", text)
        self.assertIn("Employment Type: full_time", text)
        self.assertIn("Work Mode: hybrid", text)
        self.assertIn("Total Years Experience: 5", text)
        self.assertIn("Skills: Python, ML", text)
        self.assertIn("Past Roles: Engineer at Acme", text)
        self.assertIn("Summary: Experienced software engineer", text)

    def test_truncate_to_embed_cap_applies_embed_limit(self):
        from app.utils.ingestion import truncate_to_embed_cap
        long_text = "x" * 50000
        result = truncate_to_embed_cap(long_text)
        self.assertEqual(len(result), 32000)

class TestJobMetadataAddonSchemaXor(unittest.TestCase):

    def test_generate_jd_request_both_job_id_and_metadata_raises(self):
        from pydantic import ValidationError
        from app.schemas.jd import GenerateJDRequest
        from app.schemas.ingestion import JobMetadata
        with self.assertRaises(ValidationError):
            GenerateJDRequest(
                prompt="Create a JD",
                job_id=1,
                job_metadata=JobMetadata(
                    title="Engineer",
                    location="Remote",
                    experience_level="Mid",
                    industry="Tech",
                    employment_type="full_time",
                ),
            )

    def test_generate_jd_request_accepts_metadata_only(self):
        from app.schemas.jd import GenerateJDRequest
        from app.schemas.ingestion import JobMetadata
        req = GenerateJDRequest(
            prompt="Create a JD",
            job_metadata=JobMetadata(
                title="Engineer",
                location="Remote",
                experience_level="Mid",
                industry="Tech",
                employment_type="full_time",
            ),
        )
        self.assertIsNotNone(req.job_metadata)
        self.assertIsNone(req.job_id)

class TestMergedSkillsAndProfileOnly(unittest.TestCase):

    def setUp(self):
        self.mock_qdrant = MagicMock()
        self.mock_qdrant.get = MagicMock(return_value=None)
        self.mock_qdrant.upsert = MagicMock()
        self.mock_store = MagicMock()
        self.mock_store.update = MagicMock()
        self.mock_callback = MagicMock()
        self.mock_callback.send = AsyncMock(return_value=True)

    def _mock_llm(self, skills):
        mock_gemini = MagicMock()
        mock_gemini.generate = MagicMock(return_value=json.dumps({
            "name": "John Doe",
            "location": "Remote",
            "experience_level": "Senior",
            "industry": "Tech",
            "employment_type": "full_time",
            "skills": skills,
            "past_roles": ["Engineer"],
            "raw_profile_summary": "Experienced engineer."
        }))
        mock_gemini.embed = MagicMock(return_value=[0.1] * 768)
        return mock_gemini

    def test_merged_skills_union_be_passed_plus_extracted(self):
        """BE-passed skills + extracted skills merge as union, deduped, sorted for embed."""
        mock_gemini = self._mock_llm(["Python", "Docker"])
        with patch("app.services.ingestion_service.fetch_and_parse_cv") as mock_fetch, \
             patch("app.services.ingestion_service.truncate_to_prompt_cap") as mock_truncate:
            mock_fetch.return_value = "CV text"
            mock_truncate.side_effect = lambda x: x
            asyncio.run(run_candidate_ingestion(
                candidate_id=123,
                cv_url="https://example.com/cv.pdf",
                profile_data={
                    "name": "John",
                    "location": "Remote",
                    "experience_level": "Senior",
                    "industry": "Tech",
                    "employment_type": "full_time",
                    "candidate_version": 1,
                    "skills": ["Python", "AWS"],  # BE-passed
                },
                callback_url="https://example.com/callback",
                event_id="evt_123",
                qdrant=self.mock_qdrant,
                llm=mock_gemini,
                store=self.mock_store,
                callback_client=self.mock_callback,
            ))
        payload = self.mock_qdrant.upsert.call_args[0][3]
        # union: Python, AWS (BE) + Python, Docker (extracted) -> dedupe
        self.assertEqual(sorted(payload["skills"]), ["AWS", "Docker", "Python"])
        # skills embed receives SORTED joined list (deterministic)
        skills_call = mock_gemini.embed.call_args_list[1][0][0]
        self.assertEqual(skills_call, "AWS, Docker, Python")
        # skills_vector stored
        self.assertIsNotNone(payload["skills_vector"])

    def test_merged_skills_dedupe_case_insensitive(self):
        mock_gemini = self._mock_llm(["python", "Docker"])
        with patch("app.services.ingestion_service.fetch_and_parse_cv") as mock_fetch, \
             patch("app.services.ingestion_service.truncate_to_prompt_cap") as mock_truncate:
            mock_fetch.return_value = "CV text"
            mock_truncate.side_effect = lambda x: x
            asyncio.run(run_candidate_ingestion(
                candidate_id=123,
                cv_url="https://example.com/cv.pdf",
                profile_data={
                    "name": "John",
                    "location": "Remote",
                    "experience_level": "Senior",
                    "industry": "Tech",
                    "employment_type": "full_time",
                    "candidate_version": 1,
                    "skills": ["Python"],  # case-different from extracted "python"
                },
                callback_url="https://example.com/callback",
                event_id="evt_123",
                qdrant=self.mock_qdrant,
                llm=mock_gemini,
                store=self.mock_store,
                callback_client=self.mock_callback,
            ))
        payload = self.mock_qdrant.upsert.call_args[0][3]
        # BE-passed "Python" seen first -> casing kept; extracted "python" deduped away
        self.assertEqual(sorted(payload["skills"]), ["Docker", "Python"])

    def test_profile_only_ingestion_no_cv_url(self):
        """Profile-only (cv_url=None) must not crash; uses marker text, stores skills_vector."""
        mock_gemini = self._mock_llm(["Emergency response", "Clinical leadership"])
        with patch("app.services.ingestion_service.fetch_and_parse_cv") as mock_fetch, \
             patch("app.services.ingestion_service.truncate_to_prompt_cap") as mock_truncate:
            mock_truncate.side_effect = lambda x: x
            asyncio.run(run_candidate_ingestion(
                candidate_id=123,
                cv_url=None,  # profile-only
                profile_data={
                    "name": "Maria",
                    "location": "Lagos, Nigeria",
                    "experience_level": "senior",
                    "industry": "healthcare",
                    "employment_type": "full_time",
                    "candidate_version": 1,
                    "headline": "Senior Registered Nurse specialising in emergency response",
                },
                callback_url="https://example.com/callback",
                event_id="evt_123",
                qdrant=self.mock_qdrant,
                llm=mock_gemini,
                store=self.mock_store,
                callback_client=self.mock_callback,
            ))
        # fetch must NOT be called (no CV)
        mock_fetch.assert_not_called()
        # extraction ran with the profile-only marker text
        generate_call = mock_gemini.generate.call_args[0][0]
        self.assertIn("No CV provided", generate_call)
        payload = self.mock_qdrant.upsert.call_args[0][3]
        self.assertIsNotNone(payload["skills_vector"])

    def test_skills_vector_embed_failure_degrades(self):
        """skills_vector embed failure must NOT fail ingest (debate F-5)."""
        mock_gemini = self._mock_llm(["Python"])
        # second embed (skills_vector, exact text "Python") raises; first (profile vector) works
        def flaky_embed(text):
            if text == "Python":
                raise Exception("embedding breaker open")
            return [0.1] * 768

        mock_gemini.embed.side_effect = flaky_embed
        with patch("app.services.ingestion_service.fetch_and_parse_cv") as mock_fetch, \
             patch("app.services.ingestion_service.truncate_to_prompt_cap") as mock_truncate:
            mock_fetch.return_value = "CV text"
            mock_truncate.side_effect = lambda x: x
            asyncio.run(run_candidate_ingestion(
                candidate_id=123,
                cv_url="https://example.com/cv.pdf",
                profile_data={
                    "name": "John",
                    "location": "Remote",
                    "experience_level": "Senior",
                    "industry": "Tech",
                    "employment_type": "full_time",
                    "candidate_version": 1,
                },
                callback_url="https://example.com/callback",
                event_id="evt_123",
                qdrant=self.mock_qdrant,
                llm=mock_gemini,
                store=self.mock_store,
                callback_client=self.mock_callback,
            ))
        # ingest still succeeded (upsert called) with skills_vector=None
        self.mock_qdrant.upsert.assert_called_once()
        payload = self.mock_qdrant.upsert.call_args[0][3]
        self.assertIsNone(payload["skills_vector"])
