"""Unit tests for ingestion router background-task wiring.

Verifies that ``background_tasks.add_task()`` is called with the correct
keyword argument (``llm=llm``) matching the refactored service signatures,
so that a ``gemini``→``llm`` rename regression is caught if the parameter
name changes again.
"""

from unittest.mock import MagicMock, patch
from unittest import IsolatedAsyncioTestCase

from fastapi import BackgroundTasks, Request


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
