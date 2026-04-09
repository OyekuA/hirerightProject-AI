"""Unit tests for TTLCacheBackend."""

import unittest
import concurrent.futures
from app.clients.cache import TTLCacheBackend


class TestTTLCacheBackend(unittest.TestCase):
    """Test the TTLCacheBackend implementation."""

    def test_set_then_get_returns_value(self):
        """Setting a key should allow retrieving the same value."""
        cache = TTLCacheBackend(maxsize=100, ttl=3600)
        cache.set("k", {"x": 1}, 3600)
        self.assertEqual(cache.get("k"), {"x": 1})

    def test_get_missing_key_returns_none(self):
        """Getting a nonexistent key should return None."""
        cache = TTLCacheBackend(maxsize=100, ttl=3600)
        self.assertIsNone(cache.get("nonexistent"))

    def test_delete_existing_key(self):
        """Deleting an existing key should remove it from the cache."""
        cache = TTLCacheBackend(maxsize=100, ttl=3600)
        cache.set("k", "value", 3600)
        cache.delete("k")
        self.assertIsNone(cache.get("k"))

    def test_delete_missing_key_no_exception(self):
        """Deleting a missing key should not raise an exception."""
        cache = TTLCacheBackend(maxsize=100, ttl=3600)
        try:
            cache.delete("missing")
        except Exception as e:
            self.fail(f"delete raised unexpected exception: {e}")

    def test_concurrent_access(self):
        """Multiple threads can safely read/write/delete cache entries."""
        cache = TTLCacheBackend(maxsize=1000, ttl=3600)
        key_prefix = "key_"
        value_prefix = "val_"
        num_threads = 20
        ops_per_thread = 50

        def worker(thread_id):
            for i in range(ops_per_thread):
                key = f"{key_prefix}{thread_id}_{i}"
                value = f"{value_prefix}{thread_id}_{i}"
                cache.set(key, value, 3600)
                retrieved = cache.get(key)
                # May be None if deleted by another thread, but we don't delete in this test
                self.assertEqual(retrieved, value)
                # Delete every 10th operation
                if i % 10 == 0:
                    cache.delete(key)
                    self.assertIsNone(cache.get(key))

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker, tid) for tid in range(num_threads)]
            for future in concurrent.futures.as_completed(futures):
                future.result()  # propagate any exceptions