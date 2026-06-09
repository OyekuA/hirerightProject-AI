import abc
import time
import threading
from typing import Any, Optional


class CacheBackend(abc.ABC):

    @abc.abstractmethod
    def get(self, key: str) -> Optional[Any]:
        pass

    @abc.abstractmethod
    def set(self, key: str, value: Any, ttl: int) -> None:
        pass

    @abc.abstractmethod
    def delete(self, key: str) -> None:
        pass

    @abc.abstractmethod
    def delete_by_prefix(self, prefix: str) -> None:
        pass

    @abc.abstractmethod
    def delete_by_job_id(self, job_id: int) -> None:
        pass


class TTLCacheBackend(CacheBackend):

    def __init__(self, maxsize: int = 1000, ttl: int = 3600):
        self._maxsize = maxsize
        self._default_ttl = ttl
        self._cache: dict[str, tuple[Any, float]] = {}
        self._lock = threading.RLock()

    def _is_expired(self, expires_at: float) -> bool:
        return time.monotonic() >= expires_at

    def _evict_expired(self) -> None:
        now = time.monotonic()
        expired_keys = [k for k, (_, exp) in self._cache.items() if now >= exp]
        for k in expired_keys:
            del self._cache[k]

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            value, expires_at = entry
            if self._is_expired(expires_at):
                del self._cache[key]
                return None
            return value

    def set(self, key: str, value: Any, ttl: int) -> None:
        expires_at = time.monotonic() + ttl
        with self._lock:
            self._evict_expired()
            if len(self._cache) >= self._maxsize:
                oldest_key = min(self._cache, key=lambda k: self._cache[k][1])
                del self._cache[oldest_key]
            self._cache[key] = (value, expires_at)

    def delete(self, key: str) -> None:
        with self._lock:
            self._cache.pop(key, None)

    def delete_by_prefix(self, prefix: str) -> None:
        with self._lock:
            keys_to_delete = [k for k in self._cache.keys() if k.startswith(prefix)]
            for key in keys_to_delete:
                self._cache.pop(key, None)

    def delete_by_job_id(self, job_id: int) -> None:
        with self._lock:
            keys_to_delete = []
            for key in self._cache.keys():
                parts = key.split(':')
                if len(parts) == 4 and parts[2] == str(job_id):
                    keys_to_delete.append(key)
            for key in keys_to_delete:
                self._cache.pop(key, None)
