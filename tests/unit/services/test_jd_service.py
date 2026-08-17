

import unittest
from unittest.mock import MagicMock, patch
from app.services.jd_service import JDService
from app.clients.llm import LLMUnavailableError
from app.clients.qdrant import QdrantClient, JOBS_COLLECTION

class TestJDService(unittest.TestCase):

    def setUp(self):

        self.gemini_mock = MagicMock()
        self.service = JDService(llm=self.gemini_mock)
        self.truncate_patcher = patch(
            "app.services.jd_service.truncate_to_prompt_cap",
            side_effect=lambda x: x,
        )
        self.truncate_patcher.start()

    def tearDown(self):
        self.truncate_patcher.stop()

    def test_generate_jd_fresh_returns_string(self):

        self.gemini_mock.generate.return_value = "Job Title: Senior Engineer\nResponsibilities: ..."
        result = self.service.generate_jd(
            prompt="We need a senior backend engineer with Python and Kubernetes experience.",
            existing_draft=None,
        )
        self.assertEqual(result, "Job Title: Senior Engineer\nResponsibilities: ...")
        self.gemini_mock.generate.assert_called_once()

    def test_generate_jd_fresh_prompt_forbids_markdown(self):

        self.gemini_mock.generate.return_value = "Job Title: Engineer"
        self.service.generate_jd(
            prompt="Some prompt",
            existing_draft=None,
        )
        prompt = self.gemini_mock.generate.call_args[0][0]
        self.assertIn("Bold", prompt)
        self.assertIn("##", prompt)

    def test_generate_jd_refinement_returns_string(self):

        self.gemini_mock.generate.return_value = "Job Title: Senior Engineer\nResponsibilities: ..."
        result = self.service.generate_jd(
            prompt="Make it more detailed.",
            existing_draft="Some draft",
        )
        self.assertEqual(result, "Job Title: Senior Engineer\nResponsibilities: ...")
        self.gemini_mock.generate.assert_called_once()

    def test_generate_jd_refinement_prompt_forbids_markdown(self):

        self.gemini_mock.generate.return_value = "Job Title: Engineer"
        self.service.generate_jd(
            prompt="Some prompt",
            existing_draft="Some draft",
        )
        prompt = self.gemini_mock.generate.call_args[0][0]
        self.assertIn("Bold", prompt)
        self.assertIn("##", prompt)

    def test_generate_jd_llm_exception_raises_error(self):

        self.gemini_mock.generate.side_effect = LLMUnavailableError("network error")
        with self.assertRaises(LLMUnavailableError) as cm:
            self.service.generate_jd(
                prompt="Some prompt",
                existing_draft=None,
            )
        self.assertIn("network", str(cm.exception).lower())
        self.gemini_mock.generate.assert_called_once()

    def test_analyze_jd_happy_path(self):

        self.gemini_mock.generate.return_value = '{"items": ["Point 1", "Point 2"]}'
        result = self.service.analyze_jd(
            jd_text="Job description text",
        )
        self.assertEqual(result, ["Point 1", "Point 2"])
        self.gemini_mock.generate.assert_called_once()

    def test_analyze_jd_bare_list_tolerated(self):

        self.gemini_mock.generate.return_value = '["Point 1", "Point 2"]'
        result = self.service.analyze_jd(
            jd_text="Job description text",
        )
        self.assertEqual(result, ["Point 1", "Point 2"])
        self.gemini_mock.generate.assert_called_once()

    def test_analyze_jd_non_json_raises_error(self):

        self.gemini_mock.generate.return_value = "not json"
        with self.assertRaises(LLMUnavailableError) as cm:
            self.service.analyze_jd(
                jd_text="Job description text",
            )
        self.assertIn("malformed", str(cm.exception).lower())
        self.assertEqual(self.gemini_mock.generate.call_count, 2)  # original + repair retry

    def test_analyze_jd_non_list_raises_error(self):

        self.gemini_mock.generate.return_value = '{"key": "value"}'
        with self.assertRaises(LLMUnavailableError) as cm:
            self.service.analyze_jd(
                jd_text="Job description text",
            )
        self.assertIn("malformed", str(cm.exception).lower())
        self.assertEqual(self.gemini_mock.generate.call_count, 2)  # original + repair retry

    def test_analyze_jd_non_string_item_raises_error(self):

        self.gemini_mock.generate.return_value = '["valid", 42]'
        with self.assertRaises(LLMUnavailableError) as cm:
            self.service.analyze_jd(
                jd_text="Job description text",
            )
        self.gemini_mock.generate.assert_called_once()

    def test_generate_jd_with_job_id_and_qdrant_success(self):

        qdrant_mock = MagicMock()
        payload = {
            "title": "Senior Software Engineer",
            "location": "Remote",
            "required_skills": ["Python", "AWS"],
            "raw_jd_summary": "Summary of the job.",
            "company_name": "ORACLE",
            "about": "Nice",
        }
        qdrant_mock.get.return_value = payload
        service = JDService(llm=self.gemini_mock, qdrant=qdrant_mock)
        self.gemini_mock.generate.return_value = "Generated JD"

        result = service.generate_jd(
            prompt="Create a JD",
            existing_draft=None,
            job_id=42,
        )

        self.assertEqual(result, "Generated JD")
        qdrant_mock.get.assert_called_once_with(JOBS_COLLECTION, 42)
        prompt = self.gemini_mock.generate.call_args[0][0]
        self.assertIn("Senior Software Engineer", prompt)
        self.assertIn("Remote", prompt)
        self.assertIn("Python, AWS", prompt)
        self.assertIn("ORACLE", prompt)
        self.assertIn("Nice", prompt)

    def test_generate_jd_with_job_id_no_qdrant_raises(self):

        service = JDService(llm=self.gemini_mock, qdrant=None)
        with self.assertRaises(ValueError) as cm:
            service.generate_jd(
                prompt="Create a JD",
                existing_draft=None,
                job_id=42,
            )
        self.assertIn("job_id provided but Qdrant client is not configured", str(cm.exception))

    def test_generate_jd_with_job_id_not_found_raises(self):

        qdrant_mock = MagicMock()
        qdrant_mock.get.return_value = None
        service = JDService(llm=self.gemini_mock, qdrant=qdrant_mock)
        with self.assertRaises(ValueError) as cm:
            service.generate_jd(
                prompt="Create a JD",
                existing_draft=None,
                job_id=42,
            )
        self.assertIn("Job with ID 42 not found in Qdrant", str(cm.exception))

    def test_generate_jd_placeholder_replacement_company_name(self):

        qdrant_mock = MagicMock()
        payload = {
            "title": "Senior Software Engineer",
            "location": "Remote",
            "required_skills": [],
            "raw_jd_summary": "",
            "company_name": "Acme Corp",
            "about": "[About the Company]",
        }
        qdrant_mock.get.return_value = payload
        service = JDService(llm=self.gemini_mock, qdrant=qdrant_mock)
        self.gemini_mock.generate.return_value = "Generated JD"
        result = service.generate_jd(
            prompt="We need a [Company Name] engineer",
            existing_draft=None,
            job_id=42,
        )
        self.assertEqual(result, "Generated JD")
        qdrant_mock.get.assert_called_once_with(JOBS_COLLECTION, 42)
        prompt = self.gemini_mock.generate.call_args[0][0]
        self.assertIn("Acme Corp", prompt)
        self.assertNotIn("[Company Name]", prompt)

    def test_generate_jd_placeholder_replacement_about(self):

        qdrant_mock = MagicMock()
        payload = {
            "title": "Senior Software Engineer",
            "location": "Remote",
            "required_skills": [],
            "raw_jd_summary": "",
            "company_name": "[Company Name]",
            "about": "We are a tech company.",
        }
        qdrant_mock.get.return_value = payload
        service = JDService(llm=self.gemini_mock, qdrant=qdrant_mock)
        self.gemini_mock.generate.return_value = "Generated JD"
        result = service.generate_jd(
            prompt="About: [About]",
            existing_draft="About the company: [About the Company]",
            job_id=42,
        )
        self.assertEqual(result, "Generated JD")
        qdrant_mock.get.assert_called_once_with(JOBS_COLLECTION, 42)
        prompt = self.gemini_mock.generate.call_args[0][0]
        self.assertIn("We are a tech company.", prompt)
        self.assertNotIn("[About]", prompt)
        self.assertNotIn("[About the Company]", prompt)

    def test_generate_jd_placeholder_replacement_both(self):

        qdrant_mock = MagicMock()
        payload = {
            "title": "Senior Software Engineer",
            "location": "Remote",
            "required_skills": [],
            "raw_jd_summary": "",
            "company_name": "Acme",
            "about": "Tech",
        }
        qdrant_mock.get.return_value = payload
        service = JDService(llm=self.gemini_mock, qdrant=qdrant_mock)
        self.gemini_mock.generate.return_value = "Generated JD"
        result = service.generate_jd(
            prompt="Company: [Company Name], About: [About]",
            existing_draft=None,
            job_id=42,
        )
        self.assertEqual(result, "Generated JD")
        qdrant_mock.get.assert_called_once_with(JOBS_COLLECTION, 42)
        prompt = self.gemini_mock.generate.call_args[0][0]
        self.assertIn("Acme", prompt)
        self.assertIn("Tech", prompt)
        self.assertNotIn("[Company Name]", prompt)
        self.assertNotIn("[About]", prompt)

    def test_generate_jd_company_and_about_resolved_from_qdrant(self):

        qdrant_mock = MagicMock()
        payload = {
            "title": "Senior Software Engineer",
            "location": "Remote",
            "required_skills": ["Python"],
            "raw_jd_summary": "Summary",
            "company_name": "ORACLE",
            "about": "Nice",
        }
        qdrant_mock.get.return_value = payload
        service = JDService(llm=self.gemini_mock, qdrant=qdrant_mock)
        self.gemini_mock.generate.return_value = "Generated JD"
        result = service.generate_jd(
            prompt="Create a JD for [Company Name] about [About the Company]",
            existing_draft=None,
            job_id=2003,
        )
        self.assertEqual(result, "Generated JD")
        qdrant_mock.get.assert_called_once_with(JOBS_COLLECTION, 2003)
        prompt = self.gemini_mock.generate.call_args[0][0]
        self.assertIn("ORACLE", prompt)
        self.assertIn("Nice", prompt)
        self.assertNotIn("[Company Name]", prompt)
        self.assertNotIn("[About the Company]", prompt)

class TestJDServiceInline(unittest.TestCase):

    def setUp(self):
        self.gemini_mock = MagicMock()
        self.truncate_patcher = patch(
            "app.services.jd_service.truncate_to_prompt_cap",
            side_effect=lambda x: x,
        )
        self.truncate_patcher.start()

    def tearDown(self):
        self.truncate_patcher.stop()

    def _metadata(self, **overrides):
        from app.schemas.ingestion import JobMetadata
        base = dict(
            title="Software Engineer",
            location="Remote",
            experience_level="Mid",
            industry="Tech",
            employment_type="full_time",
        )
        base.update(overrides)
        return JobMetadata(**base)

    def test_generate_jd_inline_never_touches_qdrant(self):
        qdrant_mock = MagicMock()
        service = JDService(llm=self.gemini_mock, qdrant=qdrant_mock)
        self.gemini_mock.generate.return_value = "Generated JD"
        result = service.generate_jd(
            prompt="Create a JD",
            job_metadata=self._metadata(),
        )
        self.assertEqual(result, "Generated JD")
        qdrant_mock.get.assert_not_called()
        qdrant_mock.upsert.assert_not_called()
        qdrant_mock.update_payload.assert_not_called()

    def test_generate_jd_inline_without_description_uses_placeholders(self):
        service = JDService(llm=self.gemini_mock)
        self.gemini_mock.generate.return_value = "Generated JD"
        service.generate_jd(
            prompt="Create a JD",
            job_metadata=self._metadata(),
        )
        prompt = self.gemini_mock.generate.call_args[0][0]
        self.assertIn("Job Title: Software Engineer", prompt)
        self.assertIn("[Required Skills]", prompt)
        self.assertNotIn("Compensation Range:", prompt)
        self.assertNotIn("Benefits:", prompt)
        self.assertNotIn("Work Mode:", prompt)

    def test_generate_jd_inline_with_description_runs_extraction(self):
        from app.services.ingestion_service import JobExtraction
        service = JDService(llm=self.gemini_mock)
        self.gemini_mock.generate.return_value = "Generated JD"
        with patch(
            "app.services.jd_service.extract_job_entities",
            return_value=JobExtraction(required_skills=["Python"], raw_jd_summary="Inline summary."),
        ) as mock_extract:
            service.generate_jd(
                prompt="Create a JD",
                job_metadata=self._metadata(description="We need Python engineers."),
            )
        mock_extract.assert_called_once()
        prompt = self.gemini_mock.generate.call_args[0][0]
        self.assertIn("Python", prompt)
        self.assertIn("Inline summary.", prompt)
        self.assertNotIn("[Required Skills]", prompt)

    def test_generate_jd_inline_renders_context_blocks(self):
        service = JDService(llm=self.gemini_mock)
        self.gemini_mock.generate.return_value = "Generated JD"
        service.generate_jd(
            prompt="Create a JD",
            job_metadata=self._metadata(
                work_mode="remote",
                remote_regions=["EMEA", "Americas"],
                benefits="Health and 401k.",
                salary_min=80000,
                salary_max=120000,
                salary_currency="USD",
            ),
        )
        prompt = self.gemini_mock.generate.call_args[0][0]
        self.assertIn("Compensation Range: 80000 - 120000 USD", prompt)
        self.assertIn("Benefits: Health and 401k.", prompt)
        self.assertIn("Work Mode: remote (EMEA, Americas)", prompt)

    def test_generate_jd_refinement_inline_passes_context_blocks(self):
        service = JDService(llm=self.gemini_mock)
        self.gemini_mock.generate.return_value = "Generated JD"
        service.generate_jd(
            prompt="Make it more detailed.",
            existing_draft="Some draft",
            job_metadata=self._metadata(work_mode="hybrid"),
        )
        prompt = self.gemini_mock.generate.call_args[0][0]
        self.assertIn("Work Mode: hybrid", prompt)
        self.assertNotIn("Compensation Range:", prompt)