# HireRight AI Microservice

## Project Overview

The HireRight AI Microservice adds semantic candidate‑to‑job matching, explainable fit scoring, anti‑cheat skill assessments, passive candidate discovery, and generative JD tools to the HireRight recruitment platform — without touching the existing MySQL database or PHP application.

**This is a standalone FastAPI microservice — it does not connect to MySQL. The PHP backend orchestrates all calls.**

| Technology | Role |
|---|---|
| Python 3.12 | Runtime |
| FastAPI | Web framework |
| Qdrant | Vector database (separate collections for candidates & jobs) |
| OpenAI gpt-4o-mini | LLM (generation) |
| OpenAI text-embedding-3-small | Embeddings |
| uv | Package manager |
| Docker Compose | Container orchestration |
| structlog + Sentry | Structured logging + error telemetry |
| slowapi | Rate limiting (in‑memory sliding window) |
| svix | Webhook signature verification (Recall.ai callbacks) |

## Project Structure

```
.
├── app/
│   ├── __init__.py
│   ├── main.py                     # FastAPI app factory, lifespan, background workers
│   ├── config.py                   # Pydantic Settings (env‑var loading)
│   ├── auth.py                     # API‑key verification dependency
│   ├── constants.py                # Collection names, EMPLOYMENT_TYPES & WORK_MODES vocabularies, experience level ladder
│   ├── logging_config.py           # structlog configuration
│   ├── prompts.py                  # LLM prompt templates (generation, grading, extraction)
│   ├── clients/
│   │   ├── __init__.py
│   │   ├── cache.py                # Abstract cache + TTLCacheBackend
│   │   ├── dependencies.py         # Singleton accessors for all clients
│   │   ├── llm.py                  # LLM client + CircuitBreaker (generation & embedding)
│   │   ├── meeting_bot.py          # MeetingBotClient (abstract), RecallAIClient, MeetingBaaSClient stub
│   │   ├── qdrant.py               # Qdrant vector DB wrapper
│   │   └── rate_limiter.py         # Rate‑limiter abstraction + SlowAPIRateLimiterBackend
│   ├── middleware/
│   │   ├── __init__.py
│   │   └── correlation.py          # Correlation‑ID middleware (UUID per request)
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── _rate_limit_keys.py     # Shared key‑extractor functions for per‑entity limits
│   │   ├── assessment.py           # POST /assessment/generate, /assessment/grade
│   │   ├── career.py               # POST /analyze-career-paths
│   │   ├── decision.py             # POST /decision
│   │   ├── email.py                # POST /generate-invite-email
│   │   ├── ingestion.py            # POST /ingest-candidate, /ingest-job, /cv-parse, DELETEs, GET /ingestion-status
│   │   ├── interview.py            # POST /interview/start, GET/DELETE /interview/{session_id}
│   │   ├── interview_webhook.py    # POST /interview/webhook (Svix-verified, no X-API-Key)
│   │   ├── jd.py                   # POST /generate-jd, /analyze-jd
│   │   ├── recommend.py            # POST /recommend, /recommend/pool
│   │   └── scoring.py              # POST /calculate-fit
│   ├── schemas/
│   │   ├── assessment.py           # Pydantic models for assessment endpoints
│   │   ├── career.py               # Pydantic models for career‑path endpoints
│   │   ├── decision.py             # Pydantic models for decision endpoint
│   │   ├── email.py                # Pydantic models for email endpoint
│   │   ├── ingestion.py            # Pydantic models for ingestion endpoints
│   │   ├── interview.py            # Pydantic models for interview endpoints
│   │   ├── jd.py                   # Pydantic models for JD endpoints
│   │   ├── recommendation.py       # Pydantic models for recommend endpoints
│   │   └── scoring.py              # Pydantic models for scoring endpoints
│   ├── services/
│   │   ├── __init__.py
│   │   ├── assessment_service.py   # Interview question generation & answer grading
│   │   ├── callback_client.py      # HMAC‑SHA256 signed callback delivery with retries
│   │   ├── career_service.py       # Career path analysis
│   │   ├── decision_service.py     # Ensemble Interview Decision Engine
│   │   ├── email_service.py        # Interview invite email generation
│   │   ├── ingest_queue.py         # File‑based persistent retry queue + dead‑letter monitoring
│   │   ├── ingestion_service.py    # Candidate & job ingestion logic
│   │   ├── ingestion_store.py      # Durable ingestion‑status file store
│   │   ├── interview_service.py    # Bot injection, transcript grading, host/mask pipeline
│   │   ├── interview_session_store.py # File-based interview session state persistence
│   │   ├── jd_service.py           # JD generation & analysis
│   │   ├── recommendation_service.py # Hybrid recommendation engine
│   │   └── scoring_service.py      # LLM‑based fit‑score calculation
│   └── utils/
│       ├── __init__.py             # LLM JSON parser (parse_llm_json)
│       ├── ingestion.py            # CV fetch & parse, SSRF validation, prompt truncation
│       └── webhook_verification.py  # Svix signature verification for Recall.ai
├── nginx/
│   └── nginx.conf                  # Reverse‑proxy config (TLS‑ready)
├── tests/
│   ├── __init__.py
│   ├── conftest.py                 # Shared fixtures (mock settings, Qdrant patches)
│   └── unit/
│       ├── __init__.py
│       ├── test_config.py
│       ├── test_utils.py
│       ├── clients/
│       │   ├── __init__.py
│       │   ├── test_cache.py
│       │   ├── test_circuit_breaker.py
│       │   ├── test_llm_client.py
│       │   ├── test_qdrant_client.py
│       │   └── test_meeting_bot.py
│       ├── routers/
│       │   ├── __init__.py
│       │   ├── test_decision_router.py
│       │   ├── test_email_router.py
│       │   ├── test_ingestion_router.py
│       │   ├── test_interview_router.py
│       │   ├── test_interview_webhook_router.py
│       │   ├── test_rate_limit_keys.py
│       │   ├── test_rate_limits.py
│       │   └── test_screening_router.py
│       ├── services/
│       │   ├── __init__.py
│       │   ├── test_assessment_service.py
│       │   ├── test_callback_client.py
│       │   ├── test_career_service.py
│       │   ├── test_decision_service.py
│       │   ├── test_email_service.py
│       │   ├── test_ingest_queue.py
│       │   ├── test_ingestion_fetch.py
│       │   ├── test_ingestion_service.py
│       │   ├── test_ingestion_store.py
│       │   ├── test_interview_service.py
│       │   ├── test_interview_session_store.py
│       │   ├── test_jd_service.py
│       │   ├── test_recommendation_service.py
│       │   ├── test_scoring_service.py
│       │   ├── test_screening_service.py
│       │   └── test_screening_store.py
│       └── utils/
│           ├── __init__.py
│           └── test_bias_masking.py
├── .dockerignore
├── .env.example
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
├── README.md
└── uv.lock
```

## Architecture Overview

```
┌──────────────┐     HTTP (X‑API‑Key)     ┌──────────────────────┐
│   PHP BE     │ ──────────────────────────▶   FastAPI Service    │
│  (Backend)   │ ◀──────────────────────────   (this project)     │
└──────────────┘    Signed Callback (HMAC)  └──────┬───────────────┘
                                                   │
                                    ┌──────────────┼──────────────┐
                                    ▼              ▼              ▼
                              ┌──────────┐  ┌──────────┐  ┌──────────────┐
                              │  Qdrant  │  │  LLM API │  │  File Store  │
                              │ (Vector) │  │ (OpenAI) │  │ (ingest_status
                              └──────────┘  └──────────┘  │  + queue)    │
                                                          └──────────────┘
```

The PHP backend is the **primary rate limiter** and orchestrator. This AI service applies **defense‑in‑depth** rate limits on LLM‑cost endpoints only. Non‑LLM endpoints (DELETE, GET /ingestion-status) have no rate limits.

## Prerequisites

- Docker + Docker Compose installed
- `uv` installed — provide both install methods:
    - `pip install uv`
    - `curl -LsSf https://astral.sh/uv/install.sh | sh`
- LLM API key (set in .env)
- Sentry DSN (optional — for error tracking)

## TLS / HTTPS on Hostinger VPS

Place your certificate (`fullchain.pem`) and private key (`privkey.pem`) in `nginx/certs/` on the host. Uncomment the `443 ssl` server block in [`nginx/nginx.conf`](nginx/nginx.conf) and add the cert volume mount in [`docker-compose.yml`](docker-compose.yml) under the `nginx` service. Then run `docker compose up -d --no-build nginx` to reload.

## Getting Started (Step‑by‑Step)

```bash
1. Clone the repo
2. cp .env.example .env
3. Fill in required values in .env:
      API_KEY, LLM_API_KEY, CALLBACK_HMAC_SECRET, INGEST_STATUS_STORE_PATH, RECALL_AI_API_KEY, RECALL_AI_WEBHOOK_SECRET, INTERVIEW_SESSION_STORE_PATH
4. docker compose up --build
5. Verify: curl http://localhost/health
   Expected: {"status": "ok"}
```

### Local Development Without Docker

```bash
# 1. Create a virtual environment
uv venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows

# 2. Install dependencies (including dev extras)
uv sync --extra dev

# 3. Start Qdrant separately (e.g., via Docker)
docker run -d -p 6333:6333 qdrant/qdrant:v1.17.0

# 4. Run the FastAPI dev server
uv run uvicorn app.main:create_app --factory --reload --port 8000
```

## Environment Variables Reference Table

Full table sourced from [`.env.example`](.env.example):

| Variable | Required/Optional | Default | Description |
|---|---|---|---|
| `API_KEY` | Required | `change-me-in-production` | Shared secret for `X‑API‑Key` auth between PHP and FastAPI |
| `LLM_API_KEY` | Required | _(none)_ | LLM API key — read directly by litellm for `openai/…` models |
| `LLM_MODEL` | Optional | `openai/gpt-4o-mini` | LLM model identifier for generation |
| `EMBEDDING_MODEL` | Optional | `openai/text-embedding-3-small` | Embedding model identifier |
| `EMBEDDING_DIMENSIONS` | Optional | `1536` | Output dimension of the embedding model — must match `EMBEDDING_MODEL`; update and recreate Qdrant collections when switching models |
| `QDRANT_HOST` | Optional | `qdrant` | Qdrant container hostname (Docker Compose service name) |
| `QDRANT_PORT` | Optional | `6333` | Qdrant port |
| `SENTRY_DSN` | Optional | _(empty)_ | Sentry project DSN for error tracking |
| `CACHE_TTL_SECONDS` | Optional | `86400` | In‑memory fit score cache TTL in seconds (24 hours) |
| `GENERATION_BREAKER_COOLDOWN_SECONDS` | **Required** | _(none)_ | Cooldown in seconds after generation circuit breaker trips |
| `EMBEDDING_BREAKER_COOLDOWN_SECONDS` | **Required** | _(none)_ | Cooldown in seconds after embedding circuit breaker trips |
| `LLM_GENERATION_TIMEOUT_SECONDS` | Optional | `30` | Timeout in seconds for LLM generation requests |
| `LLM_EMBEDDING_TIMEOUT_SECONDS` | Optional | `30` | Timeout in seconds for LLM embedding requests |
| `LLM_MAX_RETRIES` | Optional | `2` | Maximum retries for transient failures before recording a circuit breaker failure |
| `LLM_RETRY_BACKOFF_BASE_SECONDS` | Optional | `1.0` | Base backoff seconds for retry exponential backoff |
| `MAX_PROMPT_CHARS` | Optional | `50000` | Maximum prompt characters — inputs are truncated at this limit |
| `EMBED_MAX_CHARS` | Optional | `32000` | Maximum characters for texts sent to the embedding model (≈8k tokens, within `text-embedding-3-small`'s context) — prompt cap is separate |
| `INGEST_STATUS_STORE_PATH` | Required | `/data/ingest_status` | Path where ingestion status files are permanently stored (mounted named volume) |
| `CALLBACK_HMAC_SECRET` | Required | `change-me-in-production` | Shared secret used to sign callback payloads |
| `CALLBACK_SIGNATURE_TTL_SECONDS` | Optional | `300` | Max allowed callback timestamp age for replay protection (seconds) |
| `CALLBACK_TIMEOUT_SECONDS` | Optional | `10` | Timeout in seconds for each callback HTTP request |
| `CALLBACK_MAX_ATTEMPTS` | Optional | `3` | Maximum retry attempts for callback delivery |
| `CALLBACK_RETRY_BASE_SECONDS` | Optional | `2` | Base delay (seconds) for exponential backoff of callback retries |
| `MAX_INGEST_FILE_MB` | Optional | `10` | Maximum file size (MB) allowed for CV/JD ingestion |
| `INGEST_FETCH_TIMEOUT_SECONDS` | Optional | `20` | Timeout in seconds when fetching external URLs during ingestion |
| `INGEST_QUEUE_MAX_RETRIES` | Optional | `5` | Maximum queue‑level retries before an entry moves to dead letter |
| `INGEST_QUEUE_BACKOFF_BASE_SECONDS` | Optional | `60` | Base backoff in seconds between queue retries (doubles each attempt: 60→120→240→480→960) |
| `INGEST_QUEUE_POLL_INTERVAL_SECONDS` | Optional | `30` | How often (seconds) the background queue worker polls for due entries |
| `ENFORCE_SINGLE_REPLICA` | Optional | `False` | If `True`, acquires a startup lock to prevent multi‑instance execution. Logs a warning if another replica is already running. |
| `DEAD_LETTER_POLL_INTERVAL_SECONDS` | Optional | `300` | How often (seconds) the background dead‑letter watcher polls for new entries |
| `LOG_LEVEL` | Optional | `INFO` | Log level — use `INFO` for production, `DEBUG` for local dev. Sentry only receives ERROR-level events regardless of this setting. |
| `ENABLE_DOCS` | Optional | `False` | Set to `true` to enable Swagger UI at `/` and OpenAPI schema at `/openapi.json` |
| `DOCS_USERNAME` | Optional | _(empty)_ | HTTP Basic Auth username for Swagger UI (only enforced when `ENABLE_DOCS=true` and both credentials are set) |
| `DOCS_PASSWORD` | Optional | _(empty)_ | HTTP Basic Auth password for Swagger UI (only enforced when `ENABLE_DOCS=true` and both credentials are set) |
| `DECISION_FIT_WEIGHT` | Optional | `0.40` | Weight applied to the fit score in the Decision Engine's combined score |
| `DECISION_ASSESSMENT_WEIGHT` | Optional | `0.60` | Weight applied to the assessment score in the Decision Engine's combined score |
| `RECALL_AI_API_KEY` | Required | _(none)_ | Recall.ai API key used by `RecallAIClient` for bot injection/transcript calls |
| `RECALL_AI_REGION` | Optional | `us-east-1` | Recall.ai region — determines API host (`us-east-1`, `us-west-2`, `eu-central-1`, `ap-northeast-1`) |
| `RECALL_AI_WEBHOOK_SECRET` | Required | _(none)_ | Svix secret used to verify Recall.ai webhook signatures |
| `INTERVIEW_SESSION_STORE_PATH` | Required | _(none)_ | Path where interview session state files are persisted |

## Swagger / API Documentation

Swagger UI is available when `ENABLE_DOCS=true` is set in your environment.

### Enabling in Production

Set the following in your `.env`:

```
ENABLE_DOCS=true
DOCS_USERNAME=your-docs-username
DOCS_PASSWORD=your-docs-password
```

Access Swagger UI at `https://your-domain/` — your browser will prompt for the Basic Auth credentials above.

> ⚠️ If `ENABLE_DOCS=true` but `DOCS_USERNAME` or `DOCS_PASSWORD` is not set, the Swagger UI is publicly accessible with no authentication. Always set both credentials in production.

### Using Swagger UI

1. Open `https://your-domain/` and enter your `DOCS_USERNAME` / `DOCS_PASSWORD` when prompted.
2. Click the **Authorize** button (🔒) at the top right of the Swagger UI.
3. Enter your `API_KEY` value in the `X-API-Key` field and click **Authorize**.
4. All API calls made from Swagger UI will now include the `X-API-Key` header automatically.

## Rate Limits

Rate limits are enforced **per API key** (SHA‑256 fingerprinted). Endpoints that call the LLM have rate limits; endpoints that don't (DELETE, GET /ingestion-status) are **unlimited**.

| Endpoint | Daily Cap | Burst (per minute) | Per‑Entity Cap | Notes |
|---|---|---|---|---|
| `POST /ingest-candidate` | 500/day | 10/minute | 20/day per candidate | LLM‑cost (embedding) |
| `POST /ingest-job` | 200/day | 10/minute | 20/day per job | LLM‑cost (embedding) |
| `POST /cv-parse` | 500/day | 20/minute | — | LLM‑cost (generation) |
| `DELETE /candidates/{id}` | — | — | — | **No rate limit** (no LLM call) |
| `DELETE /jobs/{id}` | — | — | — | **No rate limit** (no LLM call) |
| `GET /ingestion-status` | — | — | — | **No rate limit** (no LLM call) |
| `POST /assessment/generate` | 500/day | 10/minute | 50/day per entity | LLM‑cost (generation) |
| `POST /assessment/grade` | 1000/day | 30/minute | 100/day per entity | LLM‑cost (generation) |
| `POST /calculate-fit` | 1000/hour | 30/minute | 100/hour per candidate | LLM‑cost (generation) |
| `POST /recommend` | 500/day | 20/minute | 50/hour per target | LLM‑cost (generation) |
| `POST /recommend/pool` | 100/hour | 10/minute | — | LLM‑cost (generation) |
| `POST /analyze-career-paths` | 500/day | 10/minute | 20/day per candidate | LLM‑cost (generation) |
| `POST /generate-jd` | 500/day | 10/minute | 50/day per job | LLM‑cost (generation) |
| `POST /analyze-jd` | 500/day | 10/minute | 50/day per job | LLM‑cost (generation) |
| `POST /decision` | 500/hour | 20/minute | 100/hour per candidate | LLM‑cost (generation, seeded rationale) |
| `POST /generate-invite-email` | 500/hour | 20/minute | 100/hour per candidate | LLM‑cost (generation) |
| `POST /interview/start` | 500/day | 10/minute | 20/day per candidate | Vendor bot injection (Recall.ai) |
| `GET /interview/{session_id}` | — | — | — | **No rate limit** |
| `DELETE /interview/{session_id}` | — | — | — | **No rate limit** |
| `POST /interview/webhook` | — | — | — | **No rate limit** (Svix-verified vendor callback, not `X-API-Key`) |

### 429 Response Contract

When a rate limit is exceeded, the service returns:

```json
{
  "detail": "Rate limit exceeded",
  "error_code": "RATE_LIMIT_EXCEEDED",
  "retry_after_seconds": 42,
  "correlation_id": "a1b2c3d4-..."
}
```

**Headers:**

| Header | Description |
|---|---|
| `Retry-After` | Seconds until the rate limit window resets |
| `X-Correlation-Id` | UUID tracing the request end‑to‑end |

The PHP backend should read `Retry-After` to determine when to retry, and log `X-Correlation-Id` for debugging.

## Ingestion Reliability Architecture

Ingestion failures are handled across **three layers** of retry:

```
Layer 1: In‑Process Retries (immediate)
  └─ 3 attempts, exponential backoff: 2s → 4s → 8s
  └─ Catches transient LLM / Qdrant / network errors

Layer 2: Persistent Queue (deferred)
  └─ Up to 5 retries, exponential backoff: 60s → 120s → 240s → 480s → 960s
  └─ Survives service restarts (file‑based)
  └─ Background worker polls every INGEST_QUEUE_POLL_INTERVAL_SECONDS

Layer 3: Startup Recovery
  └─ On service start, all incomplete (pending/running) entries are marked as failed
  └─ Callbacks are sent for each failed entry
  └─ A recovery budget (5 concurrent callbacks) prevents callback‑server overload

Final: Dead Letter
  └─ After exhausting all retries, entry moves to dead_letter/ directory
  └─ Requires manual investigation
```

### Dead Letter Monitoring

When an ingestion permanently fails (all retries exhausted), the entry is moved to `dead_letter/`. The service monitors this directory:

- **Sentry alert**: A background task polls every 5 minutes. If new dead letters appear, a Sentry event is sent with details (event_id, entity_type, entity_id, error_summary).
- **Health endpoint**: If dead letters exist, [`GET /health`](app/main.py:372) returns `{"status": "degraded", "dead_letter_count": N}` instead of `{"status": "ok"}`.

**Alert flow:**

1. **Sentry** sends a push notification (email/Slack/PagerDuty) when a dead letter is detected.
2. **Infrastructure monitoring** (e.g., UptimeRobot, Prometheus) polls `/health` and alerts on non-`ok` status.
3. **Manual investigation**: Check the dead letter files and use the ingestion-status endpoint to diagnose.

**Dead letter management API (via [`IngestQueue`](app/services/ingest_queue.py)):**

- `dead_letter_count()` — returns the number of dead letter entries
- `get_dead_letter_entries()` — returns all entries with details
- `clear_dead_letter(event_id=None)` — removes entries (all or by event_id)

## Circuit Breaker

The [`LLMClient`](app/clients/llm.py) uses a **circuit breaker** pattern with three states:

| State | Behaviour |
|---|---|
| **CLOSED** | Normal operation. Requests pass through. |
| **OPEN** | After 3 consecutive failures. All requests fail fast with `LLMUnavailableError`. |
| **HALF_OPEN** | After cooldown period (configurable via env vars). Allows a single probe request. |

- Separate circuit breakers for **generation** and **embedding**.
- A successful probe in HALF_OPEN resets the breaker to CLOSED.
- A failed probe in HALF_OPEN reopens the breaker and restarts the cooldown.
- Thread safety: All CircuitBreaker state transitions (`is_open`, `record_failure`, `record_success`) are protected by a `threading.Lock`, making the breaker safe for concurrent use across the `ThreadPoolExecutor` in `rank_pool` and `asyncio.to_thread` calls.

## Correlation ID Middleware

Every request receives a **UUID correlation ID** via [`CorrelationIdMiddleware`](app/middleware/correlation.py):

- Attached to `request.state.correlation_id` for use in handlers.
- Returned as the `X-Correlation-Id` response header.
- Bound to structlog context for structured logging.

## Callback Client

The [`CallbackClient`](app/services/callback_client.py) delivers ingestion‑status callbacks to the PHP backend:

- **HMAC‑SHA256 signing**: Each callback includes `X-HireRight-Signature`, `X-HireRight-Timestamp`, and `X-HireRight-Event-Id` headers.
- **Retries**: Up to `CALLBACK_MAX_ATTEMPTS` with exponential backoff (`CALLBACK_RETRY_BASE_SECONDS`).
- **Timeout**: Each HTTP request times out after `CALLBACK_TIMEOUT_SECONDS`.
- **SSRF safety**: CV URLs are validated via `validate_ingest_url()` (HTTPS only). Callback URLs are validated via `validate_callback_url()` which allows both HTTP and HTTPS but still blocks private, loopback, link-local, reserved, multicast, and unspecified IP ranges.
- **Replay protection**: PHP verifies the timestamp is within `CALLBACK_SIGNATURE_TTL_SECONDS` of the current time.

## Recall.ai Interview Assist Integration

The microservice integrates with **Recall.ai** to provide automated interview recording, transcription, and grading. This feature is fully async and event-driven.

### Bot Injection

When the PHP backend calls `POST /api/ai/interview/start`, the service uses [`RecallAIClient.inject_bot`](app/clients/meeting_bot.py) to inject a meeting bot into a live video interview:

- Posts to Recall.ai's `/bot/` endpoint with `meeting_url`, `recording_config`, and `metadata` (`session_id`, `candidate_id`).
- The target region (`us-east-1`, `us-west-2`, `eu-central-1`, `ap-northeast-1`) is determined by `RECALL_AI_REGION` and mapped to the appropriate API host.
- Returns a `bot.id` and `session_id` (UUID), which is persisted via [`InterviewSessionStore`](app/services/interview_session_store.py).

### Webhook-Driven Async Transcript

Recall.ai sends async events to `POST /api/ai/interview/webhook` (Svix-verified, no `X-API-Key`):

1. **`recording.done`** — Schedules a background task to call `create_transcript` (returns a transcript job id) and updates session status to `transcribing`.
2. **`transcript.done`** — Schedules a background task that calls `fetch_transcript` (downloads the transcript via `download_url`), invokes grading, updates status to `completed`, and delivers a signed callback.
3. **`transcript.failed`** — Marks session status as `failed` and sends a failure callback.

### Grade-at-End Scoring

The grading pipeline in [`interview_service.grade_transcript`](app/services/interview_service.py):

1. **Normalize transcript turns** — `normalize_transcript_turns` excludes host speakers via `_is_host_speaker` and masks candidate names via `mask_transcript_turns`.
2. **LLM call** — Invoked once at `temperature=0` to produce `per_criterion_scores`, `strengths`, `red_flags`, and `recommendation`.
3. **Deterministic overall score** — Python computes `overall_score` as the rounded mean of `per_criterion_scores`, **overriding** any LLM-suggested overall value.

### Delivery

The final grading result is delivered to the PHP backend via [`CallbackClient.send`](app/services/callback_client.py) with an `extra_payload` containing `{"session_id": ..., "grading_result": ...}`, HMAC-signed exactly like ingestion callbacks.

### Swappable Vendor Abstraction

The [`MeetingBotClient`](app/clients/meeting_bot.py) abstract base class defines the injection interface. [`RecallAIClient`](app/clients/meeting_bot.py) is the concrete implementation; [`MeetingBaaSClient`](app/clients/meeting_bot.py) is an unimplemented stub proving the interface is vendor-swappable.

### On-Demand GDPR Cascade

When `DELETE /api/ai/candidates/{candidate_id}` is called, the endpoint also calls `InterviewSessionStore.get_all_by_candidate_id(candidate_id)` and deletes every matching interview session file — see the [API Reference](#delete-apiaicandidatescandidate_id) entry for details. This cascade is **on-demand only**, executed synchronously within the same request.

### Out of Scope

The following are **not** this microservice's responsibility:

- **Calendar integration** — Booking/creating calendar events is handled elsewhere.
- **Email delivery** — The service generates invite emails but does not send them.
- **Recall.ai account/contract** — Vendor setup is managed by the platform team.

## Canonical Enum Values (Contract)

The PHP backend must send these exact values (lowercase). Anything else is rejected with a `422` that lists the allowed values. Empty string / `null` is allowed where a field is optional and means "unspecified".

**`employment_type`** (ingest-job `metadata`, ingest-candidate `profile_data`):

```
full_time | part_time | contract | freelance | internship | temporary | volunteer | apprenticeship | self_employed
```

**`work_mode`** (ingest-job `metadata`, ingest-candidate `profile_data`):

```
remote | hybrid | onsite
```

- Job `work_mode` = the role's requirement; candidate `work_mode` = the candidate's preference.
- `work_mode` and `employment_type` are orthogonal: a job is `full_time` **and** `remote`, `part_time` **and** `onsite`, etc.
- Candidates declaring `remote`/`hybrid` are scored as remote-capable for the location dimension (no harsh location-mismatch fail across cities). Jobs declaring `remote`/`hybrid` never hard-fail a remote-capable candidate on location.

## API Reference

### `POST /api/ai/ingest‑candidate`
- **Purpose:** Asynchronously ingest a candidate CV from a cloud URL into the vector store.
- **Request:** `candidate_id` (int), `cv_url` (HTTPS URL to PDF), `callback_url` (HTTP or HTTPS), `profile_data` (object):
  - `name`, `location`, `experience_level`, `industry`, `employment_type` (canonical values — see [Canonical Enum Values](#canonical-enum-values-contract)), `candidate_version` (int)
  - `work_mode` (optional, canonical: `remote|hybrid|onsite`) — the candidate's preference
  - `total_years_experience` (optional float), `data_source` (optional string)
- **Response:** `202 Accepted` — `{"event_id": "<uuid>"}`
- **Rate limit:** 500/day, 10/minute, 20/day per candidate
- **curl:** `curl -X POST http://localhost/api/ai/ingest-candidate -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" -d '{"candidate_id": 123, "cv_url": "https://example.com/cv.pdf", "profile_data": {"name": "A", "location": "Lagos, Nigeria", "experience_level": "Senior Level", "industry": "fintech", "employment_type": "full_time", "candidate_version": 1, "work_mode": "remote"}, "callback_url": "https://php-backend.example.com/callback"}'`
  > `callback_url` accepts both `http://` and `https://`.

### `POST /api/ai/ingest‑job`
- **Purpose:** Asynchronously ingest a job description text into the vector store.
- **Request:** `job_id` (int), `jd_text` (string), `metadata` (object), `callback_url` (HTTP or HTTPS)
- **`metadata` fields:**
  - Required: `title`, `location`, `experience_level`, `industry`, `employment_type` (canonical values)
  - Optional: `company_name`, `about`, `work_mode` (canonical), `remote_regions` (array of strings, e.g. `["Worldwide", "EMEA"]`), `description`, `requirements`, `responsibilities`, `benefits`, `salary_min` (number), `salary_max` (number), `salary_currency` (string, e.g. `"NGN"`), `job_version` (int, **defaults to 1**)
  - The optional content fields (description/requirements/responsibilities/benefits/salary/work_mode/remote_regions) are stored in the vector payload, enrich the job's embedding, and feed JD generation and scoring context.
- **Response:** `202 Accepted` — `{"event_id": "<uuid>"}`
- **Rate limit:** 200/day, 10/minute, 20/day per job
- **curl:** `curl -X POST http://localhost/api/ai/ingest-job -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" -d '{"job_id": 456, "jd_text": "...", "metadata": {"title": "...", "location": "...", "experience_level": "Senior Level", "industry": "...", "employment_type": "full_time", "work_mode": "hybrid", "salary_min": 8000000, "salary_max": 12000000, "salary_currency": "NGN", "benefits": "Health insurance"}, "callback_url": "http://php-backend.internal/callback"}'`
  > `callback_url` accepts both `http://` and `https://`.

### `POST /api/ai/cv‑parse`
- **Purpose:** Synchronously parse a CV PDF from a URL and return structured autofill data (name, email, phone, skills, experience, education, social links). **No vector store write occurs.**
- **Request:** `cv_url` (HTTPS URL to PDF)
- **Response:** `200 OK` — `CVAutofillResponse` with extracted fields (empty strings for missing data)
- **Rate limit:** 500/day, 20/minute
- **curl:** `curl -X POST http://localhost/api/ai/cv-parse -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" -d '{"cv_url": "https://example.com/cv.pdf"}'`

### `DELETE /api/ai/candidates/{candidate_id}`
- **Purpose:** Remove a candidate's vector from Qdrant and perform a **GDPR cascade** — after deleting from Qdrant and purging cache entries, calls `InterviewSessionStore.get_all_by_candidate_id(candidate_id)` and deletes every matching interview session file.
- **Request:** Path param `candidate_id` (int)
- **Response:** `200 OK` — `{"deleted": true}` or `404` if not found
- **Rate limit:** None (no LLM call)
- **Note:** This cascade is **on-demand only**, executed synchronously within the same request when the caller invokes this DELETE — there is no scheduled/automatic background purge job.
- **curl:** `curl -X DELETE http://localhost/api/ai/candidates/123 -H "X-API-Key: $API_KEY"`

### `DELETE /api/ai/jobs/{job_id}`
- **Purpose:** Remove a job's vector from Qdrant.
- **Request:** Path param `job_id` (int)
- **Response:** `200 OK` — `{"deleted": true}` or `404` if not found
- **Rate limit:** None (no LLM call)
- **curl:** `curl -X DELETE http://localhost/api/ai/jobs/456 -H "X-API-Key: $API_KEY"`

### `GET /api/ai/ingestion‑status`
- **Purpose:** Pull‑based fallback to check ingestion status when callback was missed.
- **Query params:** `event_id` OR (`entity_type` + `entity_id`)
- **Response:** `event_id`, `entity_type`, `entity_id`, `status` (`pending|running|success|failed`), `attempt_count`, `callback_delivery_failed`, `error_summary`, `created_at`, `updated_at`
- **Rate limit:** None (no LLM call)
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
- **Rate limit:** 500/day, 10/minute, 50/day per entity
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
- **Purpose:** Grade candidate answers for accuracy and authenticity. The LLM is called **once** (temperature 0) to produce a deterministic score, skill breakdown, and authenticity flag.
- **Request:** `questions` (list of strings **or** multiple‑choice objects), `answers` (list of strings), `time_taken_seconds` (int). Multiple‑choice objects must be full `MultipleChoiceQuestion` objects with `question` (string), `options` (list of exactly 4 strings), and `correct_answer` (must match one of the options). For multiple‑choice answers, the full labeled option text (e.g. `"A. Yes"`) or just the letter label (e.g. `"A"` or `"b"`) is accepted.
- **Response:**
  ```
  {
    "overall_score": 0–100,
    "skill_breakdown": [
      { "category": "System Design", "score": 72, "feedback": "You demonstrate..." }
    ],
    "authenticity_flag": { "is_suspicious": false, "reason": "..." },
    "needs_review": false
  }
  ```
- **Rate limit:** 1000/day, 30/minute, 100/day per entity
- **curl:** `curl -X POST http://localhost/api/ai/assessment/grade -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" -d '{"questions": ["Q1", "Q2"], "answers": ["A1", "A2"], "time_taken_seconds": 120}'`

### `POST /api/ai/calculate‑fit`
- **Purpose:** Calculate an explainable fit score between a candidate and a job.
- **Request:** `candidate_id`, `candidate_version`, `job_id`, `job_version` (all int), `force_refresh` (bool, optional)
- **Response:** `overall_score_percentage` (0–100), `category_breakdown` (`skills`, `role_match`, `experience`, `location`, `employment_type` — each with `score`, `status`, and `short_reason`), `skill_gap_analysis` (string)
- **Note:** `skills` and `role_match` are LLM-judged; `experience`, `location`, and `employment_type` are deterministic and `work_mode`-aware (explicit `work_mode` wins, legacy employment-type strings are sniffed as fallback). Arrangement compatibility is **directional**: onsite/hybrid candidates satisfy any job arrangement; only a remote-only candidate conflicts with a hybrid or onsite job (warning, not fail). Neutral "insufficient data" messages appear only when the underlying payload field is missing on either side.
- **Note:** `skill_gap_analysis` and each `short_reason` are written in neutral, pronoun‑free language — no "the candidate", "you", or "they". Language describes the match between the profile and the role factually.
- **Errors:** `404` (candidate/job not found or version mismatch), `503` (LLM or vector store unavailable)
- **Rate limit:** 1000/hour, 30/minute, 100/hour per candidate
- **curl:** `curl -X POST http://localhost/api/ai/calculate-fit -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" -d '{"candidate_id": 123, "candidate_version": 1, "job_id": 456, "job_version": 1}'`

### `POST /api/ai/recommend`
- **Purpose:** Return a ranked list of job or candidate recommendations for a target profile.
- **Request:** `type` (`"jobs"|"candidates"`), `target_id` (int), `target_version` (int), `behavioral_signals` (object — `recent_searches` (list[str]), `recent_clicks` (list[`{id: int, dwell_time_seconds: int}`]), `recent_saves` (list[int]), `recent_positive_outcomes` (list[int])), `hard_filters` (dict), `force_refresh` (bool), `limit` (int, max 50)
- **Response:** `{"results": [{"id": int, "similarity_score": float, "llm_score": int|null}, ...]}`
- **Note:** Results are ordered so candidates at or above a composite `similarity_score` floor of **0.35** (`RECOMMEND_MIN_SIMILARITY` in `recommendation_service.py`) rank first, followed by below-floor candidates as a top-up — the floor is a **preference, not a hard drop** (calibrated on live data: good matches 0.44–0.62, weak tail below ~0.35), so the response is never empty while the search returned matches. The cold-start path (missing target vector) is exempt from the floor by construction. Behavioral signals activate at low levels: intent blends in from a single recent search, co-occurrence from a single save or click. Hard filters (`hard_filters`) are exact-match; if they exclude every candidate the request **retries once unfiltered**. Vector search fetches **5×** the requested limit (`min(limit, 50) × 5`, up to 250 points) for recall. `llm_score` is a lazy cache read of a previously computed fit score — it is `null` unless `calculate-fit` ran for that pair; ranking always uses `similarity_score`.
- **Rate limit:** 500/day, 20/minute, 50/hour per target
- **curl:** `curl -X POST http://localhost/api/ai/recommend -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" -d '{"type": "jobs", "target_id": 123, "target_version": 1, "behavioral_signals": {"recent_searches": ["python engineer"], "recent_clicks": [{"id": 456, "dwell_time_seconds": 30}], "recent_saves": [789], "recent_positive_outcomes": []}, "limit": 10}'`

### `POST /api/ai/recommend/pool`
- **Purpose:** Rank a pre-filtered candidate pool by fit score (uses `ScoringService.calculate_fit` internally; cache hits are free).
- **Rate limit:** 100/hour, 10/minute.
- **Request:** `job_id` (int), `job_version` (int), `candidate_ids` (list[int], 1–100 items).
- **Response:** `{"results": [{"candidate_id": int, "fit_score": int, "status": "scored"|"failed"|"timeout"}]}` — sorted descending by `fit_score`. LLM errors are isolated per candidate: a scoring failure marks only that entry `failed` (fit_score 0) while the rest still rank; only missing candidates/jobs or version mismatches reject the whole request.
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
- **Rate limit:** 500/day, 10/minute, 20/day per candidate
- **curl:** `curl -X POST http://localhost/api/ai/analyze-career-paths -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" -d '{"candidate_id": 123}'`

### `POST /api/ai/generate‑jd`
- **Purpose:** Generate or refine a job description using the configured LLM.
- **Request:**

| Field | Type | Required | Description |
|---|---|---|---|
| `prompt` | string | Yes | Textual guidance for the desired JD |
| `existing_draft` | string | No | Existing JD text to refine instead of generating from scratch |
| `job_id` | int | No | ID of an already-ingested job — fetches `title`, `location`, `required_skills`, `raw_jd_summary`, and company context (if available) from Qdrant to enrich the prompt |
| `job_metadata` | object | No | Full `JobMetadata` object (same shape as `ingest-job.metadata`) to generate from an **unsaved** job — supports all the new optional fields (description, requirements, responsibilities, benefits, salary, work_mode, remote_regions). **XOR with `job_id`** — providing both returns `422`. Inline mode never writes to the vector store |

If `job_id` is supplied but not found in Qdrant, the endpoint returns `404`.
- **Response:** `{"jd_text": "..."}`
- **Note:** Output uses clean Markdown formatting — `##` for section headers, `*` for bullet lists, `**` for bold key terms — as instructed in the generation prompt template.
- **Rate limit:** 500/day, 10/minute, 50/day per job
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
- **Rate limit:** 500/day, 10/minute, 50/day per job
- **curl:** `curl -X POST http://localhost/api/ai/analyze-jd -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" -d '{"jd_text": "..."}'`

### `POST /api/ai/decision`
- **Purpose:** Run the Interview Decision Engine combining fit score + assessment score.
- **Request:** `candidate_id` (int), `candidate_version` (int), `job_id` (int), `job_version` (int), `assessment_score` (int, 0–100), `needs_review` (bool, optional, default `false`)
- **Response:**
  ```
  {
    "decision": "hire|no_hire|review",
    "combined_score": 0–100,
    "fit_score": 0–100,
    "assessment_score": 0–100,
    "rationale": "...",
    "confidence": 0–100
  }
  ```
- **Decision rules (deterministic):** `hire` when `combined_score >= 80 AND assessment_score >= 75`; `no_hire` when `combined_score < 50 OR assessment_score < 40`; otherwise `review`. `combined_score = 0.40 * fit_score + 0.60 * assessment_score`.
- **`needs_review`:** pass the assessment's `needs_review` flag through — when `true`, the decision is **forced to `review`** regardless of scores (a flagged candidate can never be returned as `hire`).
- **Note:** `rationale` and `confidence` are LLM-generated (temperature 0, seeded) from the pre-computed decision — the LLM does not re-judge the outcome.
- **Errors:** `404` (candidate/job not found or version mismatch), `503` (LLM or vector store unavailable)
- **Rate limit:** 500/hour, 20/minute, 100/hour per candidate
- **curl:**
  ```bash
  curl -X POST http://localhost/api/ai/decision \
    -H "X-API-Key: $API_KEY" \
    -H "Content-Type: application/json" \
    -d '{
      "candidate_id": 123,
      "candidate_version": 1,
      "job_id": 456,
      "job_version": 1,
      "assessment_score": 75,
      "needs_review": false
    }'
  ```

### `POST /api/ai/screen-batch`
- **Purpose:** Screen a batch of CVs against a job in one call — each candidate is CV-extracted and fit-scored. Supports an existing ingested job **or** a raw JD supplied inline.
- **Request:**
  - Existing job: `job_id` (int) + `job_version` (int) — **XOR** with raw mode
  - Raw mode: `jd_text` (string) + `job_metadata` (object, same shape as `ingest-job.metadata`, incl. all new optional fields)
  - `candidates` (array of `{"candidate_ref": string, "cv_url": HTTPS URL}`), `callback_url` (optional)
- **Response:** `202 Accepted` — `{"batch_id": "<uuid>"}`; poll `GET /api/ai/screen-batch/{batch_id}` for per-candidate `fit_score`, `category_breakdown`, `skill_gap_analysis`, or `error`
- **Rate limit:** 20/day, 5/minute
- **Errors:** `404` (job not found/version mismatch), `422` (invalid CV URL, dual job source)

### `POST /api/ai/generate‑invite‑email`
- **Purpose:** Generate a personalized interview-invite email.
- **Request:** `candidate_id` (int), `candidate_version` (int), `job_id` (int), `job_version` (int)
- **Response:** `{"subject": "...", "body": "..."}`
- **Note:** The mandatory `{{CALENDAR_LINK}}` placeholder must be present in the generated body. If missing from the first generation, the service retries once at the same temperature (0.7); if still missing, raises `503`.
- **Errors:** `404` (not found/version mismatch), `503`
- **Rate limit:** 500/hour, 20/minute, 100/hour per candidate
- **curl:**
  ```bash
  curl -X POST http://localhost/api/ai/generate-invite-email \
    -H "X-API-Key: $API_KEY" \
    -H "Content-Type: application/json" \
    -d '{
      "candidate_id": 123,
      "candidate_version": 1,
      "job_id": 456,
      "job_version": 1
    }'
  ```

### `POST /api/ai/interview/start`
- **Purpose:** Inject a meeting bot into a live interview for recording + later grading.
- **Request:** `meeting_url` (URL), `job_id` (int), `candidate_id` (int), `rubric` (list[str]), `callback_url` (URL), `join_at` (optional string)
- **Response:** `202 Accepted` — `{"session_id": "<uuid>"}`
- **Note:** SSRF validation of `callback_url` via `validate_callback_url` returns `422` on failure. Returns `502` if `MeetingBotClient.inject_bot` fails.
- **Rate limit:** 500/day, 10/minute, 20/day per candidate
- **curl:**
  ```bash
  curl -X POST http://localhost/api/ai/interview/start \
    -H "X-API-Key: $API_KEY" \
    -H "Content-Type: application/json" \
    -d '{
      "meeting_url": "https://meet.google.com/abc-defg-hij",
      "job_id": 456,
      "candidate_id": 123,
      "rubric": ["Communication", "Technical Skills", "Problem Solving"],
      "callback_url": "https://php-backend.example.com/callback"
    }'
  ```

### `GET /api/ai/interview/{session_id}`
- **Purpose:** Poll interview session status.
- **Response:** `session_id`, `status` (`pending|recording|transcribing|grading|completed|failed`), `result` (nullable dict — grading result once completed), `candidate_id`, `job_id`
- **Errors:** `404` if not found
- **Rate limit:** None
- **curl:** `curl http://localhost/api/ai/interview/550e8400-e29b-41d4-a716-446655440000 -H "X-API-Key: $API_KEY"`

### `DELETE /api/ai/interview/{session_id}`
- **Purpose:** Delete an interview session record.
- **Response:** `{"deleted": true}`; `404` if not found
- **Rate limit:** None
- **curl:** `curl -X DELETE http://localhost/api/ai/interview/550e8400-e29b-41d4-a716-446655440000 -H "X-API-Key: $API_KEY"`

### `POST /api/ai/interview/webhook`
- **Purpose:** Recall.ai async event callback receiver (`recording.done`, `transcript.done`, `transcript.failed`).
- **Authentication:** Uses **Svix-signature verification** via `verify_recall_webhook` (in [`app/utils/webhook_verification.py`](app/utils/webhook_verification.py)) against `RECALL_AI_WEBHOOK_SECRET` — **not** the `X-API-Key` scheme. Registered in [`app/main.py`](app/main.py) via `app.include_router(interview_webhook.router, ...)` **without** the `verify_api_key` dependency (unlike every other router).
- **Flow:** Looks up the session by `bot.id` via `store.get_by_bot_id`; on `recording.done`, schedules a background task to call `create_transcript` and mark status `transcribing`; on `transcript.done`, schedules a background task that calls `fetch_transcript`, invokes `grade_transcript`, updates status to `completed`, and delivers a signed callback via `CallbackClient.send(..., extra_payload={"session_id": ..., "grading_result": ...})`; on `transcript.failed`, marks status `failed` and sends a failure callback.
- **Response:** `204 No Content` (or `400` on invalid signature)
- **Rate limit:** None (internal vendor callback, not a caller-facing/LLM-cost endpoint)
- **curl:** _(Not callable externally — Recall.ai posts to this endpoint directly)_

### `GET /health`
- **Purpose:** Health check — no authentication required.
- **Response:** `{"status": "ok", "dead_letter_count": 0}` — normal operation
- **Response (degraded):** `{"status": "degraded", "dead_letter_count": 3}` — when dead letters exist from permanently failed ingestions
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

# Integration / E2E tests (optional — requires live Qdrant + LLM API)
# No integration test suite exists yet; coverage is provided by unit tests.
```

Unit tests mock all external dependencies (Qdrant, OpenAI) and run in CI on every push via [`.github/workflows/test.yml`](.github/workflows/test.yml).

### Test Coverage

| Test File | What It Covers |
|---|---|
| [`tests/unit/routers/test_rate_limits.py`](tests/unit/routers/test_rate_limits.py) | Burst limit enforcement for all endpoints, 429 response contract |
| [`tests/unit/routers/test_rate_limit_keys.py`](tests/unit/routers/test_rate_limit_keys.py) | Per‑entity key extraction (candidate_id, job_id, target_id) |
| [`tests/unit/routers/test_ingestion_router.py`](tests/unit/routers/test_ingestion_router.py) | Ingestion background‑task wiring, `/cv-parse` contract tests |
| [`tests/unit/services/test_ingest_queue.py`](tests/unit/services/test_ingest_queue.py) | File‑based queue enqueue/dequeue/requeue, claiming, dead‑letter monitoring |
| [`tests/unit/services/test_ingestion_store.py`](tests/unit/services/test_ingestion_store.py) | Ingestion status CRUD operations |
| [`tests/unit/services/test_ingestion_service.py`](tests/unit/services/test_ingestion_service.py) | Candidate & job ingestion logic |
| [`tests/unit/services/test_ingestion_fetch.py`](tests/unit/services/test_ingestion_fetch.py) | CV fetch & parse utilities |
| [`tests/unit/services/test_callback_client.py`](tests/unit/services/test_callback_client.py) | HMAC signing, retries, SSRF validation |
| [`tests/unit/services/test_assessment_service.py`](tests/unit/services/test_assessment_service.py) | Question generation & answer grading |
| [`tests/unit/services/test_career_service.py`](tests/unit/services/test_career_service.py) | Career path analysis |
| [`tests/unit/services/test_jd_service.py`](tests/unit/services/test_jd_service.py) | JD generation & analysis |
| [`tests/unit/services/test_recommendation_service.py`](tests/unit/services/test_recommendation_service.py) | Hybrid recommendation engine |
| [`tests/unit/services/test_scoring_service.py`](tests/unit/services/test_scoring_service.py) | Fit‑score calculation |
| [`tests/unit/clients/test_circuit_breaker.py`](tests/unit/clients/test_circuit_breaker.py) | Circuit breaker state transitions |
| [`tests/unit/clients/test_llm_client.py`](tests/unit/clients/test_llm_client.py) | LLM client generation & embedding |
| [`tests/unit/clients/test_qdrant_client.py`](tests/unit/clients/test_qdrant_client.py) | Qdrant transport-error guarding (QdrantUnavailableError → 503) |
| [`tests/unit/clients/test_cache.py`](tests/unit/clients/test_cache.py) | TTL cache backend |
| [`tests/unit/test_config.py`](tests/unit/test_config.py) | Settings loading |
| [`tests/unit/test_utils.py`](tests/unit/test_utils.py) | Utility functions |
| [`tests/unit/routers/test_screening_router.py`](tests/unit/routers/test_screening_router.py) | Screening question generation & JD analysis |
| [`tests/unit/routers/test_decision_router.py`](tests/unit/routers/test_decision_router.py) | Decision endpoint wiring — threshold rules, `needs_review` force-review |
| [`tests/unit/routers/test_email_router.py`](tests/unit/routers/test_email_router.py) | Invite email generation endpoint — calendar-link retry |
| [`tests/unit/routers/test_interview_router.py`](tests/unit/routers/test_interview_router.py) | Bot injection, SSRF validation, 202 flow |
| [`tests/unit/routers/test_interview_webhook_router.py`](tests/unit/routers/test_interview_webhook_router.py) | Svix verification + event routing (recording.done, transcript.done/failed) |
| [`tests/unit/services/test_screening_service.py`](tests/unit/services/test_screening_service.py) | Screening question generation & JD analysis |
| [`tests/unit/services/test_screening_store.py`](tests/unit/services/test_screening_store.py) | Screening session state persistence |
| [`tests/unit/services/test_decision_service.py`](tests/unit/services/test_decision_service.py) | Decision combined-score math, majority-vote tiebreak |
| [`tests/unit/services/test_email_service.py`](tests/unit/services/test_email_service.py) | Email placeholder validation, {{CALENDAR_LINK}} retry logic |
| [`tests/unit/services/test_interview_service.py`](tests/unit/services/test_interview_service.py) | Transcript grading, host-speaker exclusion, candidate-name masking |
| [`tests/unit/services/test_interview_session_store.py`](tests/unit/services/test_interview_session_store.py) | Session CRUD, candidate cascade lookup |
| [`tests/unit/clients/test_meeting_bot.py`](tests/unit/clients/test_meeting_bot.py) | Meeting bot injection & vendor abstraction |
| [`tests/unit/utils/test_bias_masking.py`](tests/unit/utils/test_bias_masking.py) | Bias masking utilities |

## CI/CD

The project uses GitHub Actions (see [`.github/workflows/test.yml`](.github/workflows/test.yml)):

- **Trigger:** Runs on every push and pull request to the main branch.
- **Steps:**
  1. Checkout code
  2. Install `uv`
  3. Run `uv sync --extra dev`
  4. Run `pytest tests/unit/ -v --tb=short -W error::FutureWarning`
- **No live services required** — all external dependencies are mocked.

## Docker Compose Services

| Service | Image | Ports | Resource Limits | Description |
|---|---|---|---|---|
| `fastapi` | Built from [`Dockerfile`](Dockerfile) | `8000:8000` | 1.0 CPU, 1024M RAM | FastAPI application |
| `qdrant` | `qdrant/qdrant:v1.17.0` | `6333:6333`, `6334:6334` | 0.8 CPU, 1280M RAM | Vector database |
| `nginx` | `nginx:1.29.7-alpine` | `80:80`, `443:443` | 0.2 CPU, 128M RAM | Reverse proxy (profile: `with-nginx`) |

### Healthcheck

The `fastapi` service has a Docker healthcheck configured:
```
test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
interval: 30s
timeout: 10s
retries: 3
```

### Volumes

| Volume | Mount Point | Purpose |
|---|---|---|
| `qdrant_data` | `/qdrant/storage` | Persistent Qdrant vector data |
| `ingest_status_data` | `/data/ingest_status` | Ingestion status files + queue + dead letters |

## Nginx Configuration

The [`nginx/nginx.conf`](nginx/nginx.conf) reverse proxy:

- Proxies all requests to `fastapi:8000`
- Maximum client body size: 20 MB
- Proxy read timeout: 120 seconds
- TLS/SSL ready (uncomment the `443 ssl` server block and mount certs)

## Error Handling

| HTTP Status | When | Handler |
|---|---|---|
| `400` | Validation error (Pydantic) | FastAPI default |
| `401` | Missing or invalid `X-API-Key` | [`verify_api_key`](app/auth.py:14) |
| `404` | Entity not found in Qdrant | Router-level |
| `422` | CV fetch/parse failure | Router-level |
| `429` | Rate limit exceeded | [`_rate_limit_exceeded_handler`](app/main.py:382) |
| `500` | Unexpected internal error | [`global_exception_handler`](app/main.py:436) |
| `503` | LLM unavailable (circuit breaker open) | [`llm_unavailable_handler`](app/main.py:427) |

## Logging

Structured logging via `structlog` (configured in [`app/logging_config.py`](app/logging_config.py)):

- **Production:** `LOG_LEVEL=INFO` — info and above; errors are forwarded to Sentry
- **Development:** `LOG_LEVEL=DEBUG` — full request/response tracing
- **Context:** Correlation ID, event ID, entity type/ID are automatically bound to each log entry
- **Sentry:** Errors are forwarded to Sentry when `SENTRY_DSN` is configured