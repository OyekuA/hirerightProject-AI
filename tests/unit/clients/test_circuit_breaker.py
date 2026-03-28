"""Unit tests for CircuitBreaker."""

import unittest
from unittest.mock import patch
from app.clients.gemini import CircuitBreaker


class TestCircuitBreaker(unittest.TestCase):
    """Test the CircuitBreaker class."""

    def test_below_threshold_is_not_open(self):
        """Breaker should stay closed when failure count is below threshold."""
        breaker = CircuitBreaker(threshold=3, cooldown_seconds=60.0)
        breaker.record_failure()
        breaker.record_failure()
        self.assertFalse(breaker.is_open())

    def test_at_threshold_is_open(self):
        """Breaker should open exactly when threshold is reached."""
        breaker = CircuitBreaker(threshold=3, cooldown_seconds=60.0)
        breaker.record_failure()
        breaker.record_failure()
        breaker.record_failure()
        self.assertTrue(breaker.is_open())

    def test_cooldown_elapsed_resets_breaker(self):
        """Breaker should automatically reset after cooldown period has elapsed."""
        breaker = CircuitBreaker(threshold=3, cooldown_seconds=60.0)
        for _ in range(3):
            breaker.record_failure()
        self.assertTrue(breaker.is_open())
        opened_at = breaker.opened_at
        self.assertIsNotNone(opened_at)
        with patch("app.clients.gemini.time.monotonic", return_value=opened_at + 61):
            self.assertFalse(breaker.is_open())
            self.assertEqual(breaker.failure_count, 0)
            self.assertIsNone(breaker.opened_at)

    def test_record_success_resets_breaker(self):
        """Calling record_success should reset a tripped breaker."""
        breaker = CircuitBreaker(threshold=3, cooldown_seconds=60.0)
        for _ in range(3):
            breaker.record_failure()
        self.assertTrue(breaker.is_open())
        breaker.record_success()
        self.assertFalse(breaker.is_open())
        self.assertEqual(breaker.failure_count, 0)
        self.assertIsNone(breaker.opened_at)