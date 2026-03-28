import hmac
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from app.config import get_settings


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Middleware that validates the X-API-Key header against the configured API_KEY."""

    async def dispatch(self, request: Request, call_next):
        if request.url.path == "/health" or \
           request.url.path.startswith("/docs") or \
           request.url.path.startswith("/openapi.json"):
            return await call_next(request)

        settings = get_settings()
        provided_key = request.headers.get("X-API-Key")
        expected_key = settings.API_KEY

        if (
            provided_key is None
            or not hmac.compare_digest(provided_key.encode(), expected_key.encode())
        ):
            return JSONResponse(
                status_code=401,
                content={"detail": "Unauthorized"},
            )

        return await call_next(request)