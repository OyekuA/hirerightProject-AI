import json
from unittest.mock import MagicMock, patch
from unittest import IsolatedAsyncioTestCase

from app.clients.llm import LLMUnavailableError


class TestEmailGenerationService(IsolatedAsyncioTestCase):

    def setUp(self):
        self.mock_llm = MagicMock()
        self.mock_qdrant = MagicMock()

    def _make_service(self):
        from app.services.email_service import EmailGenerationService
        return EmailGenerationService(llm=self.mock_llm, qdrant=self.mock_qdrant)

    def test_successful_generation(self):
        self.mock_qdrant.get.side_effect = [
            {
                "name": "Jane Doe",
                "candidate_version": 1,
                "skills": ["Python", "FastAPI"],
                "raw_profile_summary": "Senior backend engineer.",
            },
            {
                "title": "Software Engineer",
                "job_version": 1,
                "company_name": "Acme Corp",
            },
        ]

        self.mock_llm.generate.return_value = json.dumps({
            "subject": "Interview Invitation",
            "body": "We are pleased to invite you. {{CALENDAR_LINK}}",
        })

        service = self._make_service()
        result = service.generate_invite_email(
            candidate_id=1,
            candidate_version=1,
            job_id=10,
            job_version=1,
        )

        self.assertEqual(result["subject"], "Interview Invitation")
        self.assertIn("{{CALENDAR_LINK}}", result["body"])

    def test_candidate_not_found_raises_value_error(self):
        self.mock_qdrant.get.return_value = None

        service = self._make_service()
        with self.assertRaises(ValueError) as cm:
            service.generate_invite_email(
                candidate_id=999,
                candidate_version=1,
                job_id=10,
                job_version=1,
            )
        self.assertIn("Candidate not found", str(cm.exception))

    def test_candidate_version_mismatch_raises_value_error(self):
        self.mock_qdrant.get.side_effect = [
            {"name": "John", "candidate_version": 2},
            {"title": "Engineer", "job_version": 1, "company_name": "Co"},
        ]

        service = self._make_service()
        with self.assertRaises(ValueError) as cm:
            service.generate_invite_email(
                candidate_id=1,
                candidate_version=1,
                job_id=10,
                job_version=1,
            )
        self.assertIn("version mismatch", str(cm.exception).lower())

    def test_job_not_found_raises_value_error(self):
        self.mock_qdrant.get.side_effect = [
            {"name": "John", "candidate_version": 1},
            None,
        ]

        service = self._make_service()
        with self.assertRaises(ValueError) as cm:
            service.generate_invite_email(
                candidate_id=1,
                candidate_version=1,
                job_id=999,
                job_version=1,
            )
        self.assertIn("Job not found", str(cm.exception))

    def test_job_version_mismatch_raises_value_error(self):
        self.mock_qdrant.get.side_effect = [
            {"name": "John", "candidate_version": 1},
            {"title": "Engineer", "job_version": 3, "company_name": "Co"},
        ]

        service = self._make_service()
        with self.assertRaises(ValueError) as cm:
            service.generate_invite_email(
                candidate_id=1,
                candidate_version=1,
                job_id=10,
                job_version=1,
            )
        self.assertIn("version mismatch", str(cm.exception).lower())

    def test_missing_placeholder_triggers_retry_then_raises(self):
        self.mock_qdrant.get.side_effect = [
            {
                "name": "Jane Doe",
                "candidate_version": 1,
                "skills": ["Python"],
                "raw_profile_summary": "Engineer.",
            },
            {
                "title": "Engineer",
                "job_version": 1,
                "company_name": "Acme",
            },
        ]

        self.mock_llm.generate.side_effect = [
            json.dumps({"subject": "Invite", "body": "No placeholder here."}),
            json.dumps({"subject": "Invite", "body": "Still no placeholder."}),
        ]

        service = self._make_service()
        with self.assertRaises(LLMUnavailableError) as cm:
            service.generate_invite_email(
                candidate_id=1,
                candidate_version=1,
                job_id=10,
                job_version=1,
            )
        self.assertIn("CALENDAR_LINK", str(cm.exception))
        self.assertEqual(self.mock_llm.generate.call_count, 2)

    def test_retry_succeeds_on_second_attempt(self):
        self.mock_qdrant.get.side_effect = [
            {
                "name": "Jane Doe",
                "candidate_version": 1,
                "skills": ["Python"],
                "raw_profile_summary": "Engineer.",
            },
            {
                "title": "Engineer",
                "job_version": 1,
                "company_name": "Acme",
            },
        ]

        self.mock_llm.generate.side_effect = [
            json.dumps({"subject": "Invite", "body": "No placeholder here."}),
            json.dumps({"subject": "Invite", "body": "Schedule here: {{CALENDAR_LINK}}"}),
        ]

        service = self._make_service()
        result = service.generate_invite_email(
            candidate_id=1,
            candidate_version=1,
            job_id=10,
            job_version=1,
        )

        self.assertEqual(result["subject"], "Invite")
        self.assertIn("{{CALENDAR_LINK}}", result["body"])
        self.assertEqual(self.mock_llm.generate.call_count, 2)

    def test_rendered_prompt_contains_double_brace_calendar_token(self):
        """Regression test: the prompt rendered via .format() must contain literal {{CALENDAR_LINK}}."""
        self.mock_qdrant.get.side_effect = [
            {
                "name": "Alice",
                "candidate_version": 1,
                "skills": ["Python"],
                "raw_profile_summary": "Developer.",
            },
            {
                "title": "Engineer",
                "job_version": 1,
                "company_name": "Co",
            },
        ]

        self.mock_llm.generate.return_value = json.dumps({
            "subject": "Invite",
            "body": "{{CALENDAR_LINK}}",
        })

        from app.prompts import EMAIL_GENERATION_PROMPT_TEMPLATE
        rendered = EMAIL_GENERATION_PROMPT_TEMPLATE.format(
            candidate_name="Alice",
            candidate_skills="Python",
            candidate_summary="Developer.",
            job_title="Engineer",
            company="Co",
        )
        self.assertIn(
            "{{CALENDAR_LINK}}",
            rendered,
            "Rendered prompt must contain literal {{CALENDAR_LINK}} so the LLM is instructed to produce it",
        )

    def test_malformed_json_raises_llm_unavailable(self):
        """Comment 2: JSONDecodeError in _parse_and_validate must become LLMUnavailableError."""
        self.mock_qdrant.get.side_effect = [
            {
                "name": "Bob",
                "candidate_version": 1,
                "skills": ["Go"],
                "raw_profile_summary": "Developer.",
            },
            {
                "title": "Engineer",
                "job_version": 1,
                "company_name": "Acme",
            },
        ]

        self.mock_llm.generate.return_value = "not valid json at all"

        service = self._make_service()
        with self.assertRaises(LLMUnavailableError):
            service.generate_invite_email(
                candidate_id=1,
                candidate_version=1,
                job_id=10,
                job_version=1,
            )

    def test_prompt_contains_non_fabrication_guard(self):
        """Comment 3: the rendered prompt must contain the non-fabrication guard instruction."""
        from app.prompts import EMAIL_GENERATION_PROMPT_TEMPLATE
        rendered = EMAIL_GENERATION_PROMPT_TEMPLATE.format(
            candidate_name="Alice",
            candidate_skills="Python",
            candidate_summary="Developer.",
            job_title="Engineer",
            company="Co",
        )
        self.assertIn(
            "NON-FABRICATION GUARD",
            rendered,
            "Prompt must include the non-fabrication guard instruction",
        )
        self.assertIn(
            "Do NOT invent",
            rendered,
            "Prompt must explicitly forbid inventing details",
        )
