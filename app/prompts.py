GENERATE_QUESTIONS_PROMPT_TEMPLATE = """
You are an expert Talent Assessment Strategist evaluating a {target_role} position.
Your goal is to generate high-quality assessment questions based on the provided context.

{job_block}
{candidate_block}

ASSESSMENT INTENT:
{intent_block}

{instruction}
"""

GRADE_ANSWERS_PROMPT_TEMPLATE = """
You are a strict, top‑tier expert interview grader. Your task is to evaluate a set of candidate answers. Some questions may be multiple‑choice (objective) and some free‑form (subjective).

TIME DATA:
Total time: {time_taken_seconds} seconds.
Words Per Second (WPS): {wps}
System Hard Flag: {hard_flag}

{qa_pairs}

{grading_rules_block}

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

### SCORING GUIDANCE (applies to subjective questions)

- 0–10: Blank answer, completely off‑topic, or pure nonsense.
- 11–30: Candidate understood the question but demonstrated no relevant knowledge or gave harmful/wrong advice.
- 31–50: Partial understanding, some relevant points but significant gaps or errors.
- 51–70: Solid understanding with minor gaps; answers are mostly correct and relevant.
- 71–85: Strong answers with good depth, clear reasoning, and relevant experience applied.
- 86–100: Exceptional — comprehensive, precise, demonstrates senior‑level mastery.

Do NOT assign 0 unless the answer is blank or completely incoherent. A wrong but coherent answer scores 11–30.

### C. Authenticity Detection

AUTHENTICITY RULES:
- HARD_FLAG = {hard_flag} (system‑computed: True means wps > 2.5 — too fast for human typing)
- Typing speed: {wps} words/sec over {time_taken_seconds} seconds
- SUSPICIOUS means typing was TOO FAST (> 2.5 wps), indicating likely AI‑generated or copy‑pasted answers.
- SLOW typing (< 0.5 wps) is NOT suspicious — it indicates careful, deliberate writing. Do NOT flag slow typing.
- If HARD_FLAG is TRUE, you MUST set is_suspicious to true.
- If HARD_FLAG is FALSE, only set is_suspicious to true if wps > 2.5. Never flag slow typing.

IMPORTANT: Do NOT subtract any authenticity penalty yourself. Always report the RAW overall_score based purely on answer quality. The system applies the authenticity penalty automatically and deterministically when is_suspicious is true; applying it here as well would double‑penalise the candidate.

---
## Skill Breakdown

- Derive exactly 3‑5 skill categories from the **themes of the questions**.
- For each category:
  - `score`: average of the question scores (MC or subjective) that belong to that category, expressed as 0‑100.
  - `feedback`: 1‑2 sentences in **second‑person** ("You show…", "You tend to…").
- **Special rule for MC**: If overall MC score = 100%, each category that consists **only** of MC questions gets 100. If overall MC score < 100%, category scores are the percentage of correct MC answers within that category.

---
## Required JSON Output

Return **only** a JSON object with the exact key order shown below:

```json
{{
  "authenticity_flag": {{
    "is_suspicious": <bool>,
    "reason": "<explicit reference to typing-speed evidence>"
  }},
  "skill_breakdown": [
    {{
      "category": "<derived from questions>",
      "score": <int 0‑100>,
      "feedback": "<second‑person, 1‑2 sentences>"
    }}
  ],
  "grading_reasoning": "<1 sentence explaining how the raw score was calculated>",
  "overall_score": <int 0‑100, RAW score based on answer quality only — do NOT apply any authenticity penalty>
}}
```

### Example (Suspicious Case)

```json
{{
  "authenticity_flag": {{
    "is_suspicious": true,
    "reason": "WPS of 3.2 exceeds 2.5 threshold — typing too fast for human input."
  }},
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
  "grading_reasoning": "Raw score 82 based on answer quality. Typing speed flagged suspicious; the system applies the authenticity penalty automatically.",
  "overall_score": 82
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
- "core_skills": a JSON array of 3‑5 short skill strings drawn directly from the candidate's profile that make them suited for this role (e.g., ["Python", "Data Modelling", "ETL pipelines"])
- "reasoning": a single concise sentence in **second‑person** voice explaining the fit (e.g., "Your 5 years of Python experience aligns well with the data engineering demands of this role.")
- "match_percentage": an integer between 0 and 100 indicating how well the candidate's profile matches this role

Write `reasoning` in second‑person voice (use "You" / "Your"), never "the candidate" or "they".

Additionally, provide a single overall profile summary at the top level:
- "profile_summary": a 2‑3 sentence **second‑person** summary of the candidate's overall profile (not tied to a specific path). Example: "You bring a strong foundation in backend systems and have demonstrated ownership of end‑to‑end data pipelines. Your experience in fast‑paced startups positions you well for roles that require adaptability."

Return **only** a JSON object (not an array) with keys `profile_summary` and `paths`. The `paths` key must contain an array of exactly three objects, each with `role`, `core_skills`, `reasoning`, and `match_percentage` (in this exact order). Do not include `profile_summary` inside each path object.

The JSON must be formatted as follows:

{{
  "profile_summary": "...",
  "paths": [
    {{"role": "...", "core_skills": [...], "reasoning": "...", "match_percentage": ...}},
    {{"role": "...", "core_skills": [...], "reasoning": "...", "match_percentage": ...}},
    {{"role": "...", "core_skills": [...], "reasoning": "...", "match_percentage": ...}}
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

Your task is to evaluate the candidate's fit for this job and produce a JSON object strictly following the rules below.

CATEGORY EVALUATION RULES:

* **role_match**
* "pass": Direct match OR strong transferable stack (e.g., Go/Java experience for a Python/FastAPI role).
* "warning": Related domain, but stack differs WITHOUT direct paradigm overlap.
* "fail": Entirely different domain (e.g., Designer for Engineering, Sales for Data Science).


* **experience**
* "pass": Experience meets or exceeds the requirement.
* "warning": Years of experience is within 1 year below the requirement.
* "fail": Years of experience is more than 1 year below the requirement.


* **location**
* "pass": Candidate location matches job location, OR job is remote and candidate is open to remote.
* "warning": Candidate location is in the same country but different city, and job does not explicitly require on-site.
* "fail": Candidate location is in a different city AND job requires on-site presence, OR candidate is not open to the required employment arrangement.


* **employment_type**
* "pass": Employment type matches exactly (e.g., both full-time, both remote).
* "warning": Minor mismatch that could be negotiated (e.g., candidate prefers hybrid, job is full on-site).
* "fail": Fundamental mismatch.



SCORING RUBRIC (STRICT & MUTUALLY EXCLUSIVE):
Based on the 4 categories evaluated above, determine the score using EXACTLY this tier system:

* **Tier 1 (85–100):** 4 "pass".
* **Tier 2 (70–84):** 3 "pass", 1 "warning". (Use this tier if relying heavily on transferable skills).
* **Tier 3 (50–69):** 0 "fail", but 2 or more "warning".
* **Tier 4 (35–49):** EXACTLY 1 "fail" (regardless of passes).
* **Tier 5 (0–34):** 2 or more "fail".

STYLE RULES — STRICT:

* **Telegraphic / Entity-Neutral Style:** NEVER use pronouns or possessives like "The candidate", "They", "Their", "You", or "Your". Start sentences directly with nouns or verbs.
* BAD: "The candidate's 8+ years of experience meets the requirement."
* GOOD: "8+ years experience exceeds 5-year requirement."


* **Format strictly:** Return only the JSON object, no markdown fences, no extra text.
* **JSON Ordering:** You MUST generate the JSON keys in the exact order shown in the structure below.

OUTPUT JSON STRUCTURE:
{{
"category_breakdown": {{
"role_match":      {{"status": "pass|warning|fail", "short_reason": "1-2 sentences. Telegraphic style."}},
"experience":      {{"status": "pass|warning|fail", "short_reason": "1-2 sentences. Telegraphic style."}},
"location":        {{"status": "pass|warning|fail", "short_reason": "1-2 sentences. Telegraphic style."}},
"employment_type": {{"status": "pass|warning|fail", "short_reason": "1-2 sentences. Telegraphic style."}}
}},
"skill_gap_analysis": "A concise, plain-text paragraph describing the most significant gaps. Telegraphic style. Highlight transferable skills if exact matches are missing.",
"scoring_reasoning": "Briefly list the count of passes/warnings/fails, identify the matching Tier from the Rubric, and justify the final number. Telegraphic style.",
"overall_score_percentage": <integer between 0 and 100>
}}
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
- Responsibilities (5‑8 bulleted items)
- Required qualifications (must‑haves)
- Preferred qualifications (nice‑to‑haves)
- Compensation range (if possible) and benefits
- Equal opportunity statement

Return only the job description text, without any introductory or closing remarks.

FORMATTING RULES:
- Use clean, standard Markdown to format the job description.
- Use `##` for main section headers (e.g., `## Responsibilities`).
- Use bullet points (`*` or `-`) for listing responsibilities and qualifications.
- Bold (`**`) key terms or job titles where appropriate to improve scannability.
- Do NOT wrap your final output in a markdown code block (do not start or end the response with ```). Return the raw markdown text directly.
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

FORMATTING RULES:
- Use clean, standard Markdown to format the job description.
- Use `##` for main section headers (e.g., `## Responsibilities`).
- Use bullet points (`*` or `-`) for listing responsibilities and qualifications.
- Bold (`**`) key terms or job titles where appropriate to improve scannability.
- Do NOT wrap your final output in a markdown code block (do not start or end the response with ```). Return the raw markdown text directly.
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

If a field cannot be found, set its value to null (for strings) or [] (for arrays). Never use placeholder strings like "Not Provided" or "N/A".

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

If a field cannot be found, set its value to null (for strings) or [] (for arrays). Never use placeholder strings like "Not Provided" or "N/A".

Job description:
{jd_text}

Metadata:
{metadata_json}

Return only the JSON object, no other text.
"""

CV_AUTOFILL_PROMPT_TEMPLATE = """
You are a CV parser for candidate onboarding autofill. Extract the following fields from the CV text below.

Return a valid JSON object with exactly these eleven top-level keys:
- "name": string (the candidate's full name)
- "bio": string (a 2–3 sentence first-person professional summary based on the CV content)
- "email": string (candidate's email address)
- "phone": string (candidate's phone number)
- "title": string (professional headline or current job title)
- "address": string (city or location)
- "website": string (personal website URL)
- "experience": array of objects, each with keys "title", "company", "duration", "description"
- "education": array of objects, each with keys "degree", "institution", "year"
- "certifications": array of strings
- "social_links": array of objects, each with keys "platform" and "url" (e.g. linkedin, github)

If a field cannot be found in the CV, set its value to null (for strings) or [] (for arrays). Never use placeholder strings like "[Email Address]" or "[LinkedIn Profile]".

Example structure:
{{
  "name": "Jane Doe",
  "bio": "I am a senior software engineer with 8+ years of experience building scalable backend systems. I specialise in Python, FastAPI, and cloud-native architectures.",
  "email": "jane.doe@example.com",
  "phone": "+1-555-0123",
  "title": "Senior Software Engineer",
  "address": "San Francisco, CA",
  "website": "https://janedoe.dev",
  "experience": [
    {{"title": "Senior Software Engineer", "company": "Acme Corp", "duration": "2020–Present", "description": "Led the design and implementation of a microservices platform."}}
  ],
  "education": [
    {{"degree": "BSc Computer Science", "institution": "University of Tech", "year": "2015"}}
  ],
  "certifications": ["AWS Solutions Architect", "Certified Kubernetes Administrator"],
  "social_links": [
    {{"platform": "linkedin", "url": "https://linkedin.com/in/janedoe"}},
    {{"platform": "github",   "url": "https://github.com/janedoe"}}
  ]
}}

CV text:
{cv_text}

Return only the JSON object, no other text.
"""
