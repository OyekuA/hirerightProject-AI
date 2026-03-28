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

## Getting Started (Step‑by‑Step)

```bash
1. Clone the repo
2. cp .env.example .env
3. Fill in required values in .env:
     API_KEY, GEMINI_API_KEY, CALLBACK_HMAC_SECRET, INGEST_STATUS_STORE_PATH
4. docker compose up --build
5. Verify: curl http://localhost:8000/health
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
| `INGEST_STATUS_STORE_PATH` | Required | `/tmp/ingest_status` | Path where ingestion status files are persisted (mounted volume) |
| `CALLBACK_HMAC_SECRET` | Required | `change-me-in-production` | Shared secret used to sign callback payloads |
| `CALLBACK_SIGNATURE_TTL_SECONDS` | Optional | `300` | Max allowed callback timestamp age for replay protection (seconds) |
| `MAX_INGEST_FILE_MB` | Optional | `10` | Maximum file size (MB) allowed for CV/JD ingestion |
| `INGEST_FETCH_TIMEOUT_SECONDS` | Optional | `20` | Timeout in seconds when fetching external URLs during ingestion |
| `ENFORCE_SINGLE_REPLICA` | Optional | `False` | If `True`, warns on startup when multiple replicas are detected |
| `CALLBACK_MAX_RETRIES` | Optional | `3` | Maximum retry attempts for callback delivery |
| `CALLBACK_RETRY_BASE_SECONDS` | Optional | `2` | Base delay (seconds) for exponential backoff of callback retries |
| `LOG_LEVEL` | Optional | `ERROR` | Log level — use `ERROR` for production, `DEBUG` for local dev |

## API Reference

### `POST /api/ai/ingest‑candidate`
- **Purpose:** Asynchronously ingest a candidate CV from a cloud URL into the vector store.
- **Request:** `candidate_id` (int), `cv_url` (HTTPS URL to PDF/DOCX), `profile_data` (`name`, `location`, `experience_level`, `industry`, `employment_type`, `candidate_version`), `callback_url` (HTTPS)
- **Response:** `202 Accepted` — `{"event_id": "<uuid>"}`
- **curl:** `curl -X POST http://localhost:8000/api/ai/ingest-candidate -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" -d '{"candidate_id": 123, "cv_url": "https://example.com/cv.pdf", "profile_data": {...}, "callback_url": "https://php-backend.example.com/callback"}'`

### `POST /api/ai/ingest‑job`
- **Purpose:** Asynchronously ingest a job description text into the vector store.
- **Request:** `job_id` (int), `jd_text` (string), `metadata` (`title`, `location`, `experience_level`, `industry`, `employment_type`, `job_version`), `callback_url` (HTTPS)
- **Response:** `202 Accepted` — `{"event_id": "<uuid>"}`
- **curl:** `curl -X POST http://localhost:8000/api/ai/ingest-job -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" -d '{"job_id": 456, "jd_text": "...", "metadata": {...}, "callback_url": "https://php-backend.example.com/callback"}'`

### `DELETE /api/ai/candidates/{candidate_id}`
- **Purpose:** Remove a candidate's vector from Qdrant.
- **Request:** Path param `candidate_id` (int)
- **Response:** `200 OK` — `{"deleted": true}` or `404` if not found
- **curl:** `curl -X DELETE http://localhost:8000/api/ai/candidates/123 -H "X-API-Key: $API_KEY"`

### `DELETE /api/ai/jobs/{job_id}`
- **Purpose:** Remove a job's vector from Qdrant.
- **Request:** Path param `job_id` (int)
- **Response:** `200 OK` — `{"deleted": true}` or `404` if not found
- **curl:** `curl -X DELETE http://localhost:8000/api/ai/jobs/456 -H "X-API-Key: $API_KEY"`

### `GET /api/ai/ingestion‑status`
- **Purpose:** Pull‑based fallback to check ingestion status when callback was missed.
- **Query params:** `event_id` OR (`entity_type` + `entity_id`)
- **Response:** `event_id`, `entity_type`, `entity_id`, `status` (`pending|running|success|failed`), `attempt_count`, `callback_delivery_failed`, `error_summary`, `created_at`, `updated_at`
- **curl:** `curl "http://localhost:8000/api/ai/ingestion-status?event_id=abc123" -H "X-API-Key: $API_KEY"`

### `POST /api/ai/assessment/generate`
- **Purpose:** Generate scenario‑based interview questions anchored to the candidate's past roles.
- **Request:** `candidate_id` (int), `target_role` (string), `num_questions` (int, default 3, max 5)
- **Response:** `{"questions": ["...", ...]}`
- **curl:** `curl -X POST http://localhost:8000/api/ai/assessment/generate -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" -d '{"candidate_id": 123, "target_role": "Senior Backend Engineer", "num_questions": 3}'`

### `POST /api/ai/assessment/grade`
- **Purpose:** Grade candidate answers for accuracy and authenticity.
- **Request:** `questions` (list of strings), `answers` (list of strings), `time_taken_seconds` (int)
- **Response:** `overall_score` (0–100), `feedback` (string), `authenticity_flag` (`is_suspicious`, `reason`)
- **curl:** `curl -X POST http://localhost:8000/api/ai/assessment/grade -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" -d '{"questions": ["Q1", "Q2"], "answers": ["A1", "A2"], "time_taken_seconds": 120}'`

### `POST /api/ai/calculate‑fit`
- **Purpose:** Calculate an explainable fit score between a candidate and a job.
- **Request:** `candidate_id`, `candidate_version`, `job_id`, `job_version` (all int), `force_refresh` (bool, optional)
- **Response:** `overall_score_percentage` (0–100), `category_breakdown` (`role_match`, `experience`, `location`, `employment_type` — each with `status` and `short_reason`), `skill_gap_analysis` (string)
- **curl:** `curl -X POST http://localhost:8000/api/ai/calculate-fit -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" -d '{"candidate_id": 123, "candidate_version": 1, "job_id": 456, "job_version": 1}'`

### `POST /api/ai/recommend`
- **Purpose:** Return a ranked list of job or candidate recommendations for a target profile.
- **Request:** `type` (`"jobs"|"candidates"`), `target_id` (int), `target_version` (int), `behavioral_signals` (object — `recent_searches` (list[str]), `recent_clicks` (list[`{id: int, dwell_time_seconds: int}`]), `recent_saves` (list[int]), `recent_positive_outcomes` (list[int])), `hard_filters` (dict), `force_refresh` (bool), `limit` (int, max 50)
- **Response:** `{"results": [{"id": int, "similarity_score": float, "llm_score": int|null}, ...]}`
- **curl:** `curl -X POST http://localhost:8000/api/ai/recommend -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" -d '{"type": "jobs", "target_id": 123, "target_version": 1, "behavioral_signals": {"recent_searches": ["python engineer"], "recent_clicks": [{"id": 456, "dwell_time_seconds": 30}], "recent_saves": [789], "recent_positive_outcomes": []}, "limit": 10}'`

### `POST /api/ai/analyze‑career‑paths`
- **Purpose:** Suggest three career paths based on the candidate's ingested profile.
- **Request:** `candidate_id` (int)
- **Response:** `{"paths": [{"role": "...", "match_percentage": 0–100, "reasoning": "..."}, ...]}`
- **curl:** `curl -X POST http://localhost:8000/api/ai/analyze-career-paths -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" -d '{"candidate_id": 123}'`

### `POST /api/ai/generate‑jd`
- **Purpose:** Generate or refine a job description using Gemini.
- **Request:** `prompt` (string), `existing_draft` (string, optional)
- **Response:** `{"jd_text": "..."}`
- **curl:** `curl -X POST http://localhost:8000/api/ai/generate-jd -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" -d '{"prompt": "Write a job description for a Senior DevOps Engineer"}'`

### `POST /api/ai/analyze‑jd`
- **Purpose:** Analyse a job description and return actionable critique points.
- **Request:** `jd_text` (string)
- **Response:** `{"critiques": ["...", ...]}`
- **curl:** `curl -X POST http://localhost:8000/api/ai/analyze-jd -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" -d '{"jd_text": "..."}'`

### `GET /health`
- **Purpose:** Health check — no authentication required.
- **Response:** `{"status": "ok"}`
- **curl:** `curl http://localhost:8000/health`

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
uv run pytest tests/unit -v
```

The test suite covers unit tests for all core services (`ingestion_service`, `scoring_service`, `recommendation_service`, `assessment_service`, `callback_client`, `ingestion_fetch`) and client utilities (`cache`, `circuit_breaker`) using mocked Qdrant and Gemini dependencies. CI runs on every push via `.github/workflows/test.yml`.