FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-install-project

COPY app/ ./app/

RUN uv sync --frozen

FROM python:3.12-slim-bookworm AS runtime

WORKDIR /app

RUN useradd --create-home appuser

COPY --from=builder /app/.venv /app/.venv

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /data/ingest_status && chown -R appuser:appuser /data/ingest_status

COPY --from=builder /app/app ./app/

ENV PATH="/app/.venv/bin:$PATH"

USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]