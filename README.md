# HireRight AI Microservice

## Project Overview

The HireRight AI Microservice adds semantic candidate‑to‑job matching, explainable fit scoring, anti‑cheat skill assessments, passive candidate discovery, and generative JD tools to the HireRight recruitment platform — without touching the existing MySQL database or PHP application.

**This is a standalone FastAPI microservice — it does not connect to MySQL. The PHP backend orchestrates all calls.**

| Technology | Role |
|---|---|
| Python 3.12 | Runtime |
| FastAPI | Web framework |
| Qdrant | Vector database |
| Gemini 2.5 Flash Lite | LLM (generation) |
| gemini‑embedding‑001 | Embeddings |
| uv | Package manager |
| Docker Compose | Container orchestration |
| structlog + Sentry | Structured logging + error telemetry |


## Prerequisites

- Docker + Docker Compose installed
- `uv` installed — provide both install methods:
    - `pip install uv`
    - `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Google AI Studio API key (used for both Gemini generation and embedding calls)
- Sentry DSN (optional — for error tracking)
- Google Cloud

## TLS / HTTPS on IONOS VPS

Place your certificate (`fullchain.pem`) and private key (`privkey.pem`) in `nginx/certs/` on the host. Uncomment the `443 ssl` server block in `nginx/nginx.conf` and add the cert volume mount in `docker-compose.yml` under the `nginx` service. Then run `docker compose up -d --no-build nginx` to reload.

## Getting Started (Step‑by‑Step)

```bash
1. Clone the repo
2. cp .env.example .env
3. Fill in required values in .env:
     API_KEY, GEMINI_API_KEY, CALLBACK_HMAC_SECRET, INGEST_STATUS_STORE_PATH
4. docker compose up --build
5. Verify: curl http://localhost/health
   Expected: {"status": "ok"}
```

## Environment Variables Reference Table

Full table sourced from `.env.example`:

| Variable | Required/Optional | Default | Description |
|---|---|---|---|
| `API_KEY` | Required | `change-me-in-production` | Shared secret for `X‑API‑Key` auth between PHP and FastAPI |
| `GEMINI_API_KEY` | Required | _(none)_ | Google AI API key for Gemini LLM and embedding calls |
| `QDRANT_HOST` | Optional | `qdrant` | Qdrant container hostname (Docker Compose service name) |
| `QDRANT_PORT` | Optional | `6333` | Qdrant port |
| `SENTRY_DSN` | Optional | _(empty)_ | Sentry project DSN for error tracking |
| `CACHE_TTL_SECONDS` | Optional | `86400` | In‑memory fit score cache TTL in seconds (24 hours) |
| `GENERATION_BREAKER_COOLDOWN_SECONDS` | Optional | `60` | Cooldown in seconds after generation circuit breaker trips |
| `EMBEDDING_BREAKER_COOLDOWN_SECONDS` | Optional | `60` | Cooldown in seconds after embedding circuit breaker trips |
| `MAX_PROMPT_CHARS` | Optional | `50000` | Maximum prompt characters — inputs are truncated at this limit |
| `INGEST_STATUS_STORE_PATH` | Required | `/data/ingest_status` | Path where ingestion status files are permanently stored (mounted named volume) |
| `CALLBACK_HMAC_SECRET` | Required | `change-me-in-production` | Shared secret used to sign callback payloads |
| `CALLBACK_SIGNATURE_TTL_SECONDS` | Optional | `300` | Max allowed callback timestamp age for replay protection (seconds) |
| `MAX_INGEST_FILE_MB` | Optional | `10` | Maximum file size (MB) allowed for CV/JD ingestion |
| `INGEST_FETCH_TIMEOUT_SECONDS` | Optional | `20` | Timeout in seconds when fetching external URLs during ingestion |
| `ENFORCE_SINGLE_REPLICA` | Optional | `False` | If `True`, warns on startup when multiple replicas are detected |
| `CALLBACK_MAX_ATTEMPTS` | Optional | `3` | Maximum retry attempts for callback delivery |
| `CALLBACK_RETRY_BASE_SECONDS` | Optional | `2` | Base delay (seconds) for exponential backoff of callback retries |
| `LOG_LEVEL` | Optional | `ERROR` | Log level — use `ERROR` for production, `DEBUG` for local dev |

## API Reference

### `POST /api/ai/ingest‑candidate`
- **Purpose:** Asynchronously ingest a candidate CV from a cloud URL into the vector store.
- **Request:** `candidate_id` (int), `cv_url` (HTTPS URL to PDF), `profile_data` (`name`, `location`, `experience_level`, `industry`, `employment_type`, `candidate_version`), `callback_url` (HTTPS)
- **Response:** `202 Accepted` — `{"event_id": "<uuid>"}`
- **curl:** `curl -X POST http://localhost/api/ai/ingest-candidate -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" -d '{"candidate_id": 123, "cv_url": "https://example.com/cv.pdf", "profile_data": {...}, "callback_url": "https://php-backend.example.com/callback"}'`

### `POST /api/ai/ingest‑job`
- **Purpose:** Asynchronously ingest a job description text into the vector store.
- **Request:** `job_id` (int), `jd_text` (string), `metadata` (`title`, `location`, `experience_level`, `industry`, `employment_type`, `job_version`), `callback_url` (HTTPS)
- **Response:** `202 Accepted` — `{"event_id": "<uuid>"}`
- **curl:** `curl -X POST http://localhost/api/ai/ingest-job -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" -d '{"job_id": 456, "jd_text": "...", "metadata": {...}, "callback_url": "https://php-backend.example.com/callback"}'`

### `DELETE /api/ai/candidates/{candidate_id}`
- **Purpose:** Remove a candidate's vector from Qdrant.
- **Request:** Path param `candidate_id` (int)
- **Response:** `200 OK` — `{"deleted": true}` or `404` if not found
- **curl:** `curl -X DELETE http://localhost/api/ai/candidates/123 -H "X-API-Key: $API_KEY"`

### `DELETE /api/ai/jobs/{job_id}`
- **Purpose:** Remove a job's vector from Qdrant.
- **Request:** Path param `job_id` (int)
- **Response:** `200 OK` — `{"deleted": true}` or `404` if not found
- **curl:** `curl -X DELETE http://localhost/api/ai/jobs/456 -H "X-API-Key: $API_KEY"`

### `GET /api/ai/ingestion‑status`
- **Purpose:** Pull‑based fallback to check ingestion status when callback was missed.
- **Query params:** `event_id` OR (`entity_type` + `entity_id`)
- **Response:** `event_id`, `entity_type`, `entity_id`, `status` (`pending|running|success|failed`), `attempt_count`, `callback_delivery_failed`, `error_summary`, `created_at`, `updated_at`
- **curl:** `curl "http://localhost/api/ai/ingestion-status?event_id=abc123" -H "X-API-Key: $API_KEY"`

### `POST /api/ai/assessment/generate`

- **Purpose:** Generate scenario‑based interview questions tailored to a candidate's past experience or a job description.
- **Request:** Provide exactly one of `candidate_context` or `job_context`:
  - `candidate_context` (object, optional): contains `candidate_id` (int) and `target_role` (string).
  - `job_context` (object, optional): contains `job_id` (int). If `target_role` is omitted, it will be derived from the job's title.
  - `num_questions` (int, default 3, max 30) – number of questions to generate (clamped to 1‑30).
  - `question_type` (`"single"` | `"multiple_choice"`, default `"single"`).
- **Response:**
  ```
  {
    "question_type": "single" | "multiple_choice",
    "questions": ["..."]                          // question_type = "single"
    // OR
    "questions": [
      { "question": "...", "options": ["A. ...", "B. ...", "C. ...", "D. ..."], "correct_answer": "A. ..." }
    ]                                             // question_type = "multiple_choice"
  }
  ```
- **curl examples:**

  **Candidate‑centric request:**
  ```bash
  curl -X POST http://localhost/api/ai/assessment/generate \
    -H "X-API-Key: $API_KEY" \
    -H "Content-Type: application/json" \
    -d '{
      "candidate_context": {
        "candidate_id": 123,
        "target_role": "Senior Backend Engineer"
      },
      "num_questions": 3,
      "question_type": "single"
    }'
  ```

  **Job‑centric request (target_role derived from job title):**
  ```bash
  curl -X POST http://localhost/api/ai/assessment/generate \
    -H "X-API-Key: $API_KEY" \
    -H "Content-Type: application/json" \
    -d '{
      "job_context": {
        "job_id": 456
      },
      "num_questions": 5,
      "question_type": "multiple_choice"
    }'
  ```
### `POST /api/ai/assessment/grade`
- **Purpose:** Grade candidate answers for accuracy and authenticity.
- **Request:** `questions` (list of strings **or** multiple‑choice objects), `answers` (list of strings), `time_taken_seconds` (int). Multiple‑choice objects must be full `MultipleChoiceQuestion` objects with `question` (string), `options` (list of exactly 4 strings), and `correct_answer` (must match one of the options).
- **Response:**
  ```
  {
    "overall_score": 0–100,
    "skill_breakdown": [
      { "category": "System Design", "score": 72, "feedback": "You demonstrate..." }
    ],
    "authenticity_flag": { "is_suspicious": false, "reason": "..." }
  }
  ```
- **curl:** `curl -X POST http://localhost/api/ai/assessment/grade -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" -d '{"questions": ["Q1", "Q2"], "answers": ["A1", "A2"], "time_taken_seconds": 120}'`

### `POST /api/ai/calculate‑fit`
- **Purpose:** Calculate an explainable fit score between a candidate and a job.
- **Request:** `candidate_id`, `candidate_version`, `job_id`, `job_version` (all int), `force_refresh` (bool, optional)
- **Response:** `overall_score_percentage` (0–100), `category_breakdown` (`role_match`, `experience`, `location`, `employment_type` — each with `status` and `short_reason`), `skill_gap_analysis` (string)
- **Note:** `skill_gap_analysis` and each `short_reason` are written in neutral, pronoun‑free language — no “the candidate”, “you”, or “they”. Language describes the match between the profile and the role factually.
- **curl:** `curl -X POST http://localhost/api/ai/calculate-fit -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" -d '{"candidate_id": 123, "candidate_version": 1, "job_id": 456, "job_version": 1}'`

### `POST /api/ai/recommend`
- **Purpose:** Return a ranked list of job or candidate recommendations for a target profile.
- **Request:** `type` (`"jobs"|"candidates"`), `target_id` (int), `target_version` (int), `behavioral_signals` (object — `recent_searches` (list[str]), `recent_clicks` (list[`{id: int, dwell_time_seconds: int}`]), `recent_saves` (list[int]), `recent_positive_outcomes` (list[int])), `hard_filters` (dict), `force_refresh` (bool), `limit` (int, max 50)
- **Response:** `{"results": [{"id": int, "similarity_score": float, "llm_score": int|null}, ...]}`
- **curl:** `curl -X POST http://localhost/api/ai/recommend -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" -d '{"type": "jobs", "target_id": 123, "target_version": 1, "behavioral_signals": {"recent_searches": ["python engineer"], "recent_clicks": [{"id": 456, "dwell_time_seconds": 30}], "recent_saves": [789], "recent_positive_outcomes": []}, "limit": 10}'`

### `POST /api/ai/recommend/pool`
- **Purpose:** Rank a pre-filtered candidate pool by fit score (uses `ScoringService.calculate_fit` internally; cache hits are free).
- **Rate limit:** 100/hour.
- **Request:** `job_id` (int), `job_version` (int), `candidate_ids` (list[int], 1–100 items).
- **Response:** `{"results": [{"candidate_id": int, "fit_score": int}]}` — sorted descending by `fit_score`.
- **curl example:**
  ```bash
  curl -X POST http://localhost/api/ai/recommend/pool \
    -H "X-API-Key: $API_KEY" \
    -H "Content-Type: application/json" \
    -d '{"job_id": 456, "job_version": 1, "candidate_ids": [101, 102, 103]}'
  ```

### `POST /api/ai/analyze‑career‑paths`
- **Purpose:** Suggest three career paths based on the candidate's ingested profile.
- **Request:** `candidate_id` (int)
- **Response:**
  ```
  {
    "profile_summary": "You bring a strong foundation...",   // second-person, 2–3 sentences
    "paths": [
      {
        "role": "...",
        "match_percentage": 0–100,
        "core_skills": ["Python", "Data Modelling", "ETL pipelines"],   // 3–5 strings
        "reasoning": "Your 5 years of Python experience..."             // second-person
      }
    ]
  }
  ```
- **curl:** `curl -X POST http://localhost/api/ai/analyze-career-paths -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" -d '{"candidate_id": 123}'`

### `POST /api/ai/generate‑jd`
- **Purpose:** Generate or refine a job description using Gemini.
- **Request:**

| Field | Type | Required | Description |
|---|---|---|---|
| `prompt` | string | Yes | Textual guidance for the desired JD |
| `existing_draft` | string | No | Existing JD text to refine instead of generating from scratch |
| `job_id` | int | No | ID of an already-ingested job — fetches `title`, `location`, `required_skills`, `raw_jd_summary`, and company context (if available) from Qdrant to enrich the prompt; explicit `company_name` and `about` fields are no longer accepted |

If `job_id` is supplied but not found in Qdrant, the endpoint returns `404`.
- **Response:** `{"jd_text": "..."}`
- **Note:** Output is plain text only — no markdown formatting (`**`, `##`, `*` bullets) is used. All sections are written as prose with plain line breaks.
- **curl:**
  ```bash
  curl -X POST http://localhost/api/ai/generate-jd \
    -H "X-API-Key: $API_KEY" \
    -H "Content-Type: application/json" \
    -d '{
      "prompt": "Write a job description for a Senior DevOps Engineer",
      "job_id": 456
    }'
  ```

### `POST /api/ai/analyze‑jd`
- **Purpose:** Analyse a job description and return actionable critique points.
- **Request:** `jd_text` (string)
- **Response:** `{"critiques": ["...", ...]}`
- **curl:** `curl -X POST http://localhost/api/ai/analyze-jd -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" -d '{"jd_text": "..."}'`

### `GET /health`
- **Purpose:** Health check — no authentication required.
- **Response:** `{"status": "ok"}`
- **curl:** `curl http://localhost/health`

## Authentication

All endpoints except `GET /health` require the `X‑API‑Key` header. The value must match the `API_KEY` environment variable.

Example header:

```
X-API-Key: change-me-in-production
```

A missing or mismatched key returns `401 Unauthorized`.

## Callback Webhook Integration (PHP Side)

The FastAPI service sends an HTTP POST callback to the `callback_url` supplied during ingestion. PHP must verify the HMAC signature to ensure the callback originated from this service.

**Headers sent by FastAPI on each callback:**
- `X‑HireRight‑Event‑Id` — UUID of the ingestion event
- `X‑HireRight‑Timestamp` — Unix timestamp (seconds) at time of sending
- `X‑HireRight‑Signature` — `sha256=HMAC‑SHA256(timestamp + "." + raw_body)`

**PHP verification steps:**
1. Compute `sha256=HMAC‑SHA256(timestamp + "." + raw_body)` using `CALLBACK_HMAC_SECRET`
2. Compare with `X‑HireRight‑Signature` using a constant‑time comparison
3. Reject if `now - X‑HireRight‑Timestamp > 300` seconds (replay window)
4. Use `X‑HireRight‑Event‑Id` for idempotency — deduplicate retried callbacks

**Callback body schema:**
```json
{
  "event_id": "uuid",
  "entity_type": "candidate|job",
  "entity_id": 123,
  "status": "success|failed",
  "error": "string or null"
}
```

**Pull fallback:** If a callback is missed after all retries, PHP can poll the ingestion‑status endpoint:
```
GET /api/ai/ingestion‑status?event_id=<event_id>
```

## Running Tests

```bash
# Unit tests (CI — no live services required)
uv run pytest tests/unit/ -v

# Integration / E2E tests (manual — requires live stack: Qdrant + Gemini API)
uv run pytest tests/integration/ -v -s
```

Unit tests mock all external dependencies (Qdrant, Gemini) and run in CI on every push via `.github/workflows/test.yml`. Integration tests require the full Docker Compose stack and a valid `GEMINI_API_KEY`.