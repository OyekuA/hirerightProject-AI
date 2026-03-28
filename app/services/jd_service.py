"""Job Description service for generating and analyzing job descriptions.

This module provides the JDService class that orchestrates Gemini calls
to generate job descriptions from prompts and to critique existing JDs.
"""

import json
import structlog
from typing import Optional

from app.clients.gemini import GeminiClient, GeminiUnavailableError
from app.config import get_settings
from app.utils import truncate_to_prompt_cap, parse_gemini_json

logger = structlog.get_logger()


class JDService:
    """Service that encapsulates Gemini‑based JD generation and analysis."""

    def __init__(self, gemini: GeminiClient):
        """Initialize the JD service.

        Args:
            gemini: A configured GeminiClient instance.
        """
        self.gemini = gemini

    def generate_jd(self, prompt: str, existing_draft: Optional[str] = None) -> str:
        """Generate or refine a job description.

        Args:
            prompt: Textual guidance for the desired JD (e.g., "We need a senior
                backend engineer with Python and Kubernetes experience").
            existing_draft: Optional existing JD text to refine.

        Returns:
            The generated or refined JD as a plain text string.

        Raises:
            GeminiUnavailableError: If the Gemini circuit breaker is open or the call fails.
        """
        logger.info(
            "Generating JD",
            prompt_length=len(prompt),
            has_existing_draft=existing_draft is not None,
        )

        prompt = truncate_to_prompt_cap(prompt)
        if existing_draft is not None:
            existing_draft = truncate_to_prompt_cap(existing_draft)

        if existing_draft is None:
            prompt_text = f"""
You are a professional HR specialist writing a job description.

Given the following request:

{prompt}

Write a comprehensive, well‑structured job description that includes:

- A clear, enticing job title
- A brief company/team overview
- A concise summary of the role
- Responsibilities (5‑8 bullet points)
- Required qualifications (must‑haves)
- Preferred qualifications (nice‑to‑haves)
- Compensation range (if possible) and benefits
- Equal opportunity statement

Return only the job description text, without any introductory or closing remarks.
Do not wrap the output in markdown fences, JSON, or any other formatting.
"""
        else:
            prompt_text = f"""
You are a professional HR specialist refining an existing job description.

ORIGINAL DRAFT:
```
{existing_draft}
```

REFINEMENT REQUEST:
{prompt}

Please update the draft according to the request above. Keep the same overall structure
and sections, but modify the content as needed. Return only the revised job description,
without any introductory or closing remarks. Do not wrap the output in markdown fences,
JSON, or any other formatting.
"""

        try:
            jd_text = self.gemini.generate(prompt_text)
        except Exception as e:
            logger.error("Gemini call failed during JD generation", error=str(e))
            raise GeminiUnavailableError(f"JD generation failed: {e}")

        logger.info(
            "JD generated successfully",
            output_length=len(jd_text),
        )
        return jd_text

    def analyze_jd(self, jd_text: str) -> list[str]:
        """Analyze a job description and return a list of critique points.

        Args:
            jd_text: The job description text to analyse.

        Returns:
            A list of strings, each being a concrete critique suggestion.

        Raises:
            GeminiUnavailableError: If the Gemini circuit breaker is open or the call fails,
                or if the response is malformed.
        """
        logger.info(
            "Analyzing JD",
            jd_length=len(jd_text),
        )

        jd_text = truncate_to_prompt_cap(jd_text)

        prompt = f"""
You are a seasoned HR consultant reviewing a job description for clarity, inclusivity,
and effectiveness.

JOB DESCRIPTION:
```
{jd_text}
```

Provide a concise list of critique points that could improve this job description.
Each point should be a single sentence, actionable, and focused on a specific aspect
(e.g., "Needs more specific skill requirements", "Consider adding remote‑work flexibility",
"Remove gender‑biased language").

Return **only** a JSON array of strings, where each string is one critique point.
Do not include any extra text, numbering, or markdown fences.

Example format:
[
  "Needs more specific skill requirements",
  "Consider adding remote‑work flexibility",
  "Remove gender‑biased language"
]
"""

        generated = self.gemini.generate(prompt)

        try:
            critiques = parse_gemini_json(generated)
        except json.JSONDecodeError as e:
            logger.error(
                "Gemini returned non‑JSON response",
                error=str(e),
            )
            raise GeminiUnavailableError(
                f"Gemini returned malformed critique JSON: {e}"
            )

        if not isinstance(critiques, list):
            logger.error(
                "Gemini response is not a list",
                response_type=type(critiques),
            )
            raise GeminiUnavailableError("Gemini response is not a list")

        for i, item in enumerate(critiques):
            if not isinstance(item, str):
                logger.error(
                    f"Item {i} is not a string",
                    item=item,
                )
                raise GeminiUnavailableError(f"Item {i} is not a string")

        logger.info(
            "JD analysis completed",
            num_critiques=len(critiques),
        )
        return critiques