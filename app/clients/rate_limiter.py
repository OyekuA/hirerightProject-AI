import abc
import hashlib
from typing import Optional

from slowapi import Limiter
from slowapi.util import get_remote_address


def get_api_key_or_ip(request):
    api_key = request.headers.get("X-API-Key")
    if api_key:
        fingerprint = hashlib.sha256(api_key.encode()).hexdigest()
        return f"api_key:{fingerprint}"
    return get_remote_address(request)


class RateLimiterBackend(abc.ABC):

    @property
    @abc.abstractmethod
    def limiter(self):
        raise NotImplementedError


class SlowAPIRateLimiterBackend(RateLimiterBackend):

    def __init__(self, default_limits: Optional[str] = None):
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
        return self._limiter
