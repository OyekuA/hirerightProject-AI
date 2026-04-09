"""Abstract rate‑limiter interface and a slowapi‑based implementation."""

import abc
import hashlib
from typing import Optional

from slowapi import Limiter
from slowapi.util import get_remote_address


def get_api_key_or_ip(request):
    """Return a rate‑limit key based on X‑API‑Key header, falling back to client IP.

    For API‑key authentication, the key is a SHA‑256 fingerprint of the raw key,
    ensuring the original secret never appears in limiter state or logs.
    """
    api_key = request.headers.get("X-API-Key")
    if api_key:
        fingerprint = hashlib.sha256(api_key.encode()).hexdigest()
        return f"api_key:{fingerprint}"
    return get_remote_address(request)


class RateLimiterBackend(abc.ABC):
    """Abstract interface for a rate‑limiting backend.

    This minimal interface serves as a swap point for a future Redis‑backed
    implementation. The concrete MVP implementation delegates to slowapi.
    """

    @property
    @abc.abstractmethod
    def limiter(self):
        """Return a rate‑limiter object suitable for use with FastAPI routes."""
        raise NotImplementedError


class SlowAPIRateLimiterBackend(RateLimiterBackend):
    """Rate‑limiter backend that wraps slowapi.Limiter.

    This class provides a `.limiter` property that routers can use to attach
    `@limiter.limit(...)` decorators.

    Attributes:
        limiter: The underlying slowapi Limiter instance.
    """

    def __init__(self, default_limits: Optional[str] = None):
        """Initialize the slowapi Limiter.

        Args:
            default_limits: Optional default rate‑limit string (e.g., "5/minute").
                If None, no default limit is applied.
        """
        if default_limits is None:
            limits = []
        else:
            limits = [default_limits]
        self._limiter = Limiter(
            key_func=get_api_key_or_ip,
            default_limits=limits,
        )

    @property
    def limiter(self) -> Limiter:
        """Return the slowapi Limiter instance."""
        return self._limiter