"""Abstract cache interface and a TTLCache‑based implementation."""

import abc
import threading
from typing import Any, Optional

import cachetools


class CacheBackend(abc.ABC):
    """Abstract interface for a key‑value cache with TTL support."""

    @abc.abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Retrieve the value associated with `key`.

        Returns:
            The cached value, or None if the key does not exist or has expired.
        """
        pass

    @abc.abstractmethod
    def set(self, key: str, value: Any, ttl: int) -> None:
        """Store `value` under `key` with a time‑to‑live of `ttl` seconds.

        The implementation may ignore the per‑call `ttl` if the backend uses
        a fixed TTL configured at construction time.
        """
        pass

    @abc.abstractmethod
    def delete(self, key: str) -> None:
        """Remove the key from the cache.

        If the key does not exist, the method does nothing (silent no‑op).
        """
        pass

    @abc.abstractmethod
    def delete_by_prefix(self, prefix: str) -> None:
        """Delete all cache keys that start with the given prefix."""
        pass

    @abc.abstractmethod
    def delete_by_job_id(self, job_id: int) -> None:
        """Delete all cache keys where the third component matches the given job ID."""
        pass

class TTLCacheBackend(CacheBackend):
    """Concrete cache backend built on top of cachetools.TTLCache.

    This implementation uses a single TTL for all entries, set at creation.
    The `ttl` parameter of `set` is accepted for interface compatibility
    but is ignored.

    Attributes:
        maxsize: Maximum number of items the cache can hold.
        ttl: Time‑to‑live in seconds for each entry.
    """

    def __init__(self, maxsize: int = 1000, ttl: int = 3600):
        """Initialize the TTL cache.

        Args:
            maxsize: Maximum number of entries the cache can hold.
            ttl: Default TTL in seconds for each entry.
        """
        self._cache = cachetools.TTLCache(maxsize=maxsize, ttl=ttl)
        self._lock = threading.RLock()

    def get(self, key: str) -> Optional[Any]:
        """Retrieve the value associated with `key`.

        Returns:
            The cached value, or None if the key does not exist or has expired.
        """
        with self._lock:
            return self._cache.get(key)

    def set(self, key: str, value: Any, ttl: int) -> None:
        """Store `value` under `key`.

        The `ttl` parameter is ignored; the cache‑level TTL configured at
        construction is used for expiration.

        Args:
            key: Cache key.
            value: Value to store.
            ttl: Ignored.
        """
        _ = ttl
        with self._lock:
            self._cache[key] = value

    def delete(self, key: str) -> None:
        """Remove the key from the cache.

        If the key does not exist, the method does nothing (silent no‑op).
        """
        with self._lock:
            self._cache.pop(key, None)

    def delete_by_prefix(self, prefix: str) -> None:
        """Delete all cache keys that start with the given prefix."""
        with self._lock:
            keys_to_delete = [k for k in self._cache.keys() if k.startswith(prefix)]
            for key in keys_to_delete:
                self._cache.pop(key, None)

    def delete_by_job_id(self, job_id: int) -> None:
        """Delete all cache keys where the third component matches the given job ID."""
        with self._lock:
            keys_to_delete = []
            for key in self._cache.keys():
                parts = key.split(':')
                if len(parts) == 4 and parts[2] == str(job_id):
                    keys_to_delete.append(key)
            for key in keys_to_delete:
                self._cache.pop(key, None)