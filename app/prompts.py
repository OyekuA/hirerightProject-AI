"""Centralized prompt templates used across the application."""

GENERATE_QUESTIONS_PROMPT_TEMPLATE = """
You are an expert interviewer for a {target_role} position.
{job_block}
{candidate_block}
{instruction}
"""

GRADE_ANSWERS_PROMPT_TEMPLATE = """
You are a strict, top‑tier expert interview grader. Your task is to evaluate a set of candidate answers. Some questions may be multiple‑choice (objective) and some free‑form (subjective).

TIME DATA:
Total time: {time_taken_seconds} seconds.
Words Per Second (WPS): {wps}
System Hard Flag: {hard_flag}

{qa_pairs}

---
## Grading Rules – CRITICAL

### A. Multiple‑Choice (Objective) Questions
- Each MC question includes:
  - `options`: list of possible answers.
  - `user_answer`: the option selected by the candidate.
  - `correct_answer`: the single correct option.
- **Scoring**:
  - If `user_answer` exactly matches `correct_answer` → **1 point**.
  - Otherwise → **0 points**.
- **DO NOT** award partial credit or infer correctness from surrounding text. The match must be exact.
- The **overall MC score** = `(total_correct_MC / total_MC_questions) * 100`.
- If **any** MC question is answered incorrectly, the `overall_score` cannot be 100%.

### B. Single Response (Subjective) Questions
- Evaluate against professional standards for the target role.
- A score of 100% is **only** possible if the answer:
  - Fully addresses all parts of the question.
  - Demonstrates deep domain knowledge.
  - Is clear, concise, and free of errors.
  - Shows no signs of AI‑generated hedging or generic filler.
- Partial answers receive proportional deduction.
- Vague or off‑topic answers receive 0 for that question.

### C. Authenticity Detection
1. **System Speed Check**: If `hard_flag` is TRUE → `is_suspicious: true` with the reason: "Completion time of {time_taken_seconds}s ({wps} words/sec) is impossible for human typing."
2. **AI‑patterned phrasing**: Look for numbered lists of exactly 3‑5 points, "I would…", "Additionally,", "In conclusion,", and overly balanced hedging.
3. **Structural uniformity**: If all answers follow identical paragraph/list templates, flag it.

If any indicator is present (including hard flag), set `is_suspicious: true` and **reduce `overall_score` by 20–30 points** (after computing the raw score from MC + subjective).

---
## Skill Breakdown

- Derive exactly 3‑5 skill categories from the **themes of the questions**.
- For each category:
  - `score`: average of the question scores (MC or subjective) that belong to that category, expressed as 0‑100.
  - `feedback`: 1‑2 sentences in **second‑person** ("You show…", "You tend to…").
- **Special rule for MC**: If overall MC score = 100%, each category that consists **only** of MC questions gets 100. If overall MC score < 100%, category scores are the percentage of correct MC answers within that category.

---
## Required JSON Output

Return **only** a JSON object with the exact structure below:

```json
{{
  "overall_score": <int 0‑100, penalised if suspicious>,
  "skill_breakdown": [
    {{
      "category": "<derived from questions>",
      "score": <int 0‑100>,
      "feedback": "<second‑person, 1‑2 sentences>"
    }}
  ],
  "authenticity_flag": {{
    "is_suspicious": <bool>,
    "reason": "<explicit reference to wps, phrasing, or structural flags>"
  }}
}}
```

### Example (Suspicious Case)

```json
{{
  "overall_score": 88,
  "skill_breakdown": [
    {{
      "category": "System Design",
      "score": 75,
      "feedback": "You demonstrated good knowledge of scalability but missed a key caching concept."
    }},
    {{
      "category": "Debugging",
      "score": 100,
      "feedback": "You correctly identified all troubleshooting steps for memory leaks."
    }}
  ],
  "authenticity_flag": {{
    "is_suspicious": false,
    "reason": "No suspicious indicators detected."
  }}
}}
```

Do not include any other text outside the JSON object.
"""

CAREER_PATHS_PROMPT_TEMPLATE = """
You are a senior career advisor evaluating a candidate's profile.

CANDIDATE PROFILE (ID {candidate_id}):
```json
{candidate_payload_json}
```

Your task is to suggest exactly three career paths that would be a good fit for this candidate.
For each path, produce a JSON object with the following keys:

- "role": a string describing the job title or role (e.g., "Senior Data Engineer")
- "match_percentage": an integer between 0 and 100 indicating how well the candidate's profile matches this role
- "core_skills": a JSON array of 3‑5 short skill strings drawn directly from the candidate's profile that make them suited for this role (e.g., ["Python", "Data Modelling", "ETL pipelines"])
- "reasoning": a single concise sentence in **second‑person** voice explaining the fit (e.g., "Your 5 years of Python experience aligns well with the data engineering demands of this role.")

Write `reasoning` in second‑person voice (use "You" / "Your"), never "the candidate" or "they".

Additionally, provide a single overall profile summary at the top level:
- "profile_summary": a 2‑3 sentence **second‑person** summary of the candidate's overall profile (not tied to a specific path). Example: "You bring a strong foundation in backend systems and have demonstrated ownership of end‑to‑end data pipelines. Your experience in fast‑paced startups positions you well for roles that require adaptability."

Return **only** a JSON object (not an array) with keys `profile_summary` and `paths`. The `paths` key must contain an array of exactly three objects, each with `role`, `match_percentage`, `core_skills`, and `reasoning`. Do not include `profile_summary` inside each path object.

The JSON must be formatted as follows:

{{
  "profile_summary": "...",
  "paths": [
    {{"role": "...", "match_percentage": ..., "core_skills": [...], "reasoning": "..."}},
    {{"role": "...", "match_percentage": ..., "core_skills": [...], "reasoning": "..."}},
    {{"role": "...", "match_percentage": ..., "core_skills": [...], "reasoning": "..."}}
  ]
}}

Now produce the JSON object.
"""

SCORING_FIT_PROMPT_TEMPLATE = """
You are an elite senior recruiter evaluating a candidate for a specific job opening.

CANDIDATE PROFILE (ID {candidate_id}, version {candidate_version}):
```json
{candidate_payload_json}
```

JOB PROFILE (ID {job_id}, version {job_version}):
```json
{job_payload_json}
```

Your task is to evaluate the candidate's fit for this job and produce a JSON object with exactly the following structure:

{{
  "overall_score_percentage": <integer between 0 and 100>,
  "category_breakdown": {{
    "role_match":      {{"status": "pass|warning|fail", "short_reason": "..."}},
    "experience":      {{"status": "pass|warning|fail", "short_reason": "..."}},
    "location":        {{"status": "pass|warning|fail", "short_reason": "..."}},
    "employment_type": {{"status": "pass|warning|fail", "short_reason": "..."}}
  }},
  "skill_gap_analysis": "A concise paragraph describing the most significant gaps between the profile and the job requirements. State what the role requires and what the profile provides. Focus on the most critical 2–3 skills that are missing or underrepresented in the profile compared to the job description. Avoid generic statements; be specific about which skills are lacking and how that impacts suitability for the role."
}}

Rules:
- **Transferable Skills (Crucial):** Do not heavily penalize for missing exact keyword matches if adjacent technologies or paradigms are present. For example, an engineer with "Go and Java" has highly transferable skills to a "Python and FastAPI" role. Treat strong adjacent experience as a near-match, reflecting this in a higher overall_score_percentage and a "pass" or soft "warning" in role_match.
- **Telegraphic / Entity-Neutral Style (Strict):** NEVER use pronouns or possessives like "The candidate", "The candidate's", "The profile", "They", "Their", "You", or "Your". Start sentences directly with nouns or verbs. Remove unnecessary articles ("The", "A").  
  - BAD: "The candidate's 8+ years of experience meets the requirement."  
  - GOOD: "8+ years experience exceeds 5-year requirement."  
  - BAD: "The role requires Python."  
  - GOOD: "Role requires Python; profile demonstrates transferable mastery in Go and Java."
- **Format strictly:** Return only the JSON object, no markdown fences, no extra text.
- **Category Evaluation:** For each category, choose "pass", "warning", or "fail" based on professional judgment.
- **Short Reasons:** Provide a short_reason (1–2 sentences) that describes the match or mismatch factually. Write as a neutral match statement using the telegraphic style.
- **Overall Score:** The overall_score_percentage should reflect the composite suitability (0-100).
- **Skill Gap Analysis:** The skill_gap_analysis must be a single plain-text paragraph using the strict telegraphic style, highlighting transferable skills if exact matches are missing. Do not use pronouns.

Now produce the JSON object.
"""

JD_ANALYSIS_PROMPT_TEMPLATE = """
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

JD_GENERATION_PROMPT_TEMPLATE = """
You are a professional HR specialist writing a job description.

KNOWN CONTEXT (use these values directly — do not use placeholders for them):
Job Title: {title}
Company Name: {company}
Location: {location}
Required Skills: {skills}
About the Company: {about}
Role Summary: {summary}

Use the KNOWN CONTEXT values verbatim where applicable. Only use a `[placeholder]` for fields that are genuinely unknown.

Given the following request:

{prompt}

Write a comprehensive, well‑structured job description that includes:

- A clear, enticing job title
- A brief company/team overview
- A concise summary of the role
- Responsibilities (5‑8 items, each on a new line, without bullet symbols)
- Required qualifications (must‑haves)
- Preferred qualifications (nice‑to‑haves)
- Compensation range (if possible) and benefits
- Equal opportunity statement

Return only the job description text, without any introductory or closing remarks.
FORMATTING RULES — strictly enforced:
- Do NOT use any markdown syntax: no `**bold**`, no `*italic*`, no `#` or `##` headings, no `*` or `-` bullet characters, no backticks, no `\n\n` blank-line sequences.
- Use plain prose only. Separate sections with a single newline and a plain label followed by a colon, e.g. 'Responsibilities:' not '## Responsibilities'.
- Write 'Job Title: Senior Engineer' not '**Job Title:** Senior Engineer'.
- If you are tempted to use any markdown character, write plain text instead.
"""

JD_REFINEMENT_PROMPT_TEMPLATE = """
You are a professional HR specialist refining an existing job description.

KNOWN CONTEXT (use these values directly — do not use placeholders for them):
Job Title: {title}
Company Name: {company}
Location: {location}
Required Skills: {skills}
About the Company: {about}
Role Summary: {summary}

Use the KNOWN CONTEXT values verbatim where applicable. Only use a `[placeholder]` for fields that are genuinely unknown.

ORIGINAL DRAFT:
```
{existing_draft}
```

REFINEMENT REQUEST:
{prompt}

Please update the draft according to the request above. Keep the same overall structure
and sections, but modify the content as needed. Return only the revised job description,
without any introductory or closing remarks.
Responsibilities should be listed as plain text items, each on a new line, without bullet symbols.
FORMATTING RULES — strictly enforced:
- Do NOT use any markdown syntax: no `**bold**`, no `*italic*`, no `#` or `##` headings, no `*` or `-` bullet characters, no backticks, no `\n\n` blank-line sequences.
- Use plain prose only. Separate sections with a single newline and a plain label followed by a colon, e.g. 'Responsibilities:' not '## Responsibilities'.
- Write 'Job Title: Senior Engineer' not '**Job Title:** Senior Engineer'.
- If you are tempted to use any markdown character, write plain text instead.
"""

CV_EXTRACTION_PROMPT_TEMPLATE = """
Extract the following fields from the CV and the provided profile data.
Return a valid JSON object with exactly these keys:
- "name": string
- "location": string
- "experience_level": string
- "industry": string
- "employment_type": string
- "skills": array of strings
- "past_roles": array of strings
- "raw_profile_summary": string (concise summary of the candidate's overall profile)

CV content:
{cv_text}

Profile data:
{profile_data_json}

Return only the JSON object, no other text.
"""

JD_EXTRACTION_PROMPT_TEMPLATE = """
Extract the following fields from the job description.
Return a valid JSON object with exactly these keys:
- "title": string
- "location": string
- "experience_level": string
- "industry": string
- "employment_type": string
- "required_skills": array of strings
- "raw_jd_summary": string (concise summary of the job description)

Job description:
{jd_text}

Metadata:
{metadata_json}

Return only the JSON object, no other text.
"""

CV_AUTOFILL_PROMPT_TEMPLATE = """
You are a CV parser for candidate onboarding autofill. Extract the following fields from the CV text below.

Return a valid JSON object with exactly these five top-level keys:
- "name": string (the candidate's full name)
- "bio": string (a 2–3 sentence first-person professional summary based on the CV content)
- "experience": array of objects, each with keys "title", "company", "duration", "description"
- "education": array of objects, each with keys "degree", "institution", "year"
- "certifications": array of strings

Example structure:
{{
  "name": "Jane Doe",
  "bio": "I am a senior software engineer with 8+ years of experience building scalable backend systems. I specialise in Python, FastAPI, and cloud-native architectures.",
  "experience": [
    {{"title": "Senior Software Engineer", "company": "Acme Corp", "duration": "2020–Present", "description": "Led the design and implementation of a microservices platform."}}
  ],
  "education": [
    {{"degree": "BSc Computer Science", "institution": "University of Tech", "year": "2015"}}
  ],
  "certifications": ["AWS Solutions Architect", "Certified Kubernetes Administrator"]
}}

CV text:
{cv_text}

Return only the JSON object, no other text.
"""
