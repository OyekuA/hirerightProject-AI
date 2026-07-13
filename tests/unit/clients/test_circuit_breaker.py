

import threading
import unittest
from unittest.mock import patch
from app.clients.llm import CircuitBreaker

class TestCircuitBreaker(unittest.TestCase):

    def test_below_threshold_is_not_open(self):

        breaker = CircuitBreaker(threshold=3, cooldown_seconds=60.0)
        breaker.record_failure()
        breaker.record_failure()
        self.assertFalse(breaker.is_open())

    def test_at_threshold_is_open(self):

        breaker = CircuitBreaker(threshold=3, cooldown_seconds=60.0)
        breaker.record_failure()
        breaker.record_failure()
        breaker.record_failure()
        self.assertTrue(breaker.is_open())
        self.assertEqual(breaker.state, CircuitBreaker.OPEN)

    def test_cooldown_elapsed_transitions_to_half_open(self):

        breaker = CircuitBreaker(threshold=3, cooldown_seconds=60.0)
        for _ in range(3):
            breaker.record_failure()
        self.assertTrue(breaker.is_open())
        opened_at = breaker.opened_at
        self.assertIsNotNone(opened_at)
        with patch("app.clients.llm.time.monotonic", return_value=opened_at + 61):
            self.assertFalse(breaker.is_open())
            self.assertEqual(breaker.state, CircuitBreaker.HALF_OPEN)
            self.assertIsNone(breaker.opened_at)
            self.assertEqual(breaker.failure_count, 3)

    def test_half_open_allows_single_probe(self):

        breaker = CircuitBreaker(threshold=3, cooldown_seconds=60.0)
        for _ in range(3):
            breaker.record_failure()
        with patch("app.clients.llm.time.monotonic", return_value=breaker.opened_at + 61):
            breaker.is_open()
        self.assertEqual(breaker.state, CircuitBreaker.HALF_OPEN)
        breaker._probe_sent = False
        self.assertFalse(breaker.is_open())
        self.assertTrue(breaker._probe_sent)
        self.assertTrue(breaker.is_open())

    def test_half_open_probe_success_closes_breaker(self):

        breaker = CircuitBreaker(threshold=3, cooldown_seconds=60.0)
        for _ in range(3):
            breaker.record_failure()
        with patch("app.clients.llm.time.monotonic", return_value=breaker.opened_at + 61):
            breaker.is_open()
        self.assertEqual(breaker.state, CircuitBreaker.HALF_OPEN)
        breaker.record_success()
        self.assertEqual(breaker.state, CircuitBreaker.CLOSED)
        self.assertEqual(breaker.failure_count, 0)
        self.assertIsNone(breaker.opened_at)
        self.assertFalse(breaker._probe_sent)

    def test_half_open_probe_failure_reopens(self):

        breaker = CircuitBreaker(threshold=3, cooldown_seconds=60.0)
        for _ in range(3):
            breaker.record_failure()

        original_opened_at = breaker.opened_at
        self.assertIsNotNone(original_opened_at)

        with patch("app.clients.llm.time.monotonic", return_value=original_opened_at + 61):
            breaker.is_open()

        self.assertEqual(breaker.state, CircuitBreaker.HALF_OPEN)
        self.assertIsNone(breaker.opened_at)

        with patch("app.clients.llm.time.monotonic", return_value=original_opened_at + 62):
            breaker.record_failure()

        self.assertEqual(breaker.state, CircuitBreaker.OPEN)
        self.assertIsNotNone(breaker.opened_at)
        self.assertEqual(breaker.failure_count, 3)
        self.assertGreaterEqual(breaker.opened_at, original_opened_at + 62)

    def test_record_success_resets_breaker(self):

        breaker = CircuitBreaker(threshold=3, cooldown_seconds=60.0)
        for _ in range(3):
            breaker.record_failure()
        self.assertTrue(breaker.is_open())
        breaker.record_success()
        self.assertFalse(breaker.is_open())
        self.assertEqual(breaker.state, CircuitBreaker.CLOSED)
        self.assertEqual(breaker.failure_count, 0)
        self.assertIsNone(breaker.opened_at)

    def test_record_success_in_closed_state_resets_failure_count(self):

        breaker = CircuitBreaker(threshold=3, cooldown_seconds=60.0)
        breaker.record_failure()
        breaker.record_failure()
        self.assertEqual(breaker.failure_count, 2)
        breaker.record_success()
        self.assertEqual(breaker.failure_count, 0)
        self.assertEqual(breaker.state, CircuitBreaker.CLOSED)

    def test_record_failure_in_half_open_reopens(self):

        breaker = CircuitBreaker(threshold=3, cooldown_seconds=60.0)
        for _ in range(3):
            breaker.record_failure()
        with patch("app.clients.llm.time.monotonic", return_value=breaker.opened_at + 61):
            breaker.is_open()
        self.assertEqual(breaker.state, CircuitBreaker.HALF_OPEN)
        breaker.record_failure()
        self.assertEqual(breaker.state, CircuitBreaker.OPEN)
        self.assertIsNotNone(breaker.opened_at)
        self.assertEqual(breaker.failure_count, 3)

    def test_record_failure_after_cooldown_resets_opened_at(self):

        breaker = CircuitBreaker(threshold=3, cooldown_seconds=60.0)
        for _ in range(3):
            breaker.record_failure()
        original_opened_at = breaker.opened_at
        with patch("app.clients.llm.time.monotonic", return_value=original_opened_at + 61):
            breaker.is_open()
        with patch("app.clients.llm.time.monotonic", return_value=original_opened_at + 62):
            breaker.record_failure()
        self.assertEqual(breaker.state, CircuitBreaker.OPEN)
        self.assertAlmostEqual(breaker.opened_at, original_opened_at + 62, delta=0.1)

    def test_concurrent_probe_only_one_allowed(self):

        breaker = CircuitBreaker(threshold=3, cooldown_seconds=60.0)
        for _ in range(3):
            breaker.record_failure()
        with patch("app.clients.llm.time.monotonic", return_value=breaker.opened_at + 61):
            breaker.is_open()
        self.assertEqual(breaker.state, CircuitBreaker.HALF_OPEN)
        breaker._probe_sent = False

        barrier = threading.Barrier(10)
        results = []

        def probe():
            barrier.wait()
            results.append(breaker.is_open())

        threads = [threading.Thread(target=probe) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(results.count(False), 1)
        self.assertEqual(results.count(True), 9)

    def test_thread_safe_failure_recording(self):

        breaker = CircuitBreaker(threshold=3, cooldown_seconds=60.0)
        barrier = threading.Barrier(10)

        def record():
            barrier.wait()
            breaker.record_failure()

        threads = [threading.Thread(target=record) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(breaker.state, CircuitBreaker.OPEN)
        self.assertGreaterEqual(breaker.failure_count, 3)