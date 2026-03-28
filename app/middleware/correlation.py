import uuid
import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Middleware that generates a correlation ID for each request and attaches it to the response."""

    async def dispatch(self, request: Request, call_next):
        correlation_id = str(uuid.uuid4())
        request.state.correlation_id = correlation_id

        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            correlation_id=correlation_id,
            endpoint=str(request.url.path),
        )

        response = await call_next(request)
        response.headers["X-Correlation-Id"] = correlation_id
        return response