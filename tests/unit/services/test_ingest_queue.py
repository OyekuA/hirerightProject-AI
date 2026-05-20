"""Unit tests for the file-based ingest queue."""

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.services.ingest_queue import IngestQueue, IngestQueueEntry
from app.services.ingestion_store import IngestionRecord


import uuid

_sentinel = object()


def _make_record(
    entity_type: str = "candidate",
    entity_id: int = 1,
    payload: object = _sentinel,
) -> IngestionRecord:
    """Build a minimal IngestionRecord with sensible defaults."""
    final_payload = payload if payload is not _sentinel else {"cv_url": "https://example.com/cv.pdf", "profile_data": {}}
    return IngestionRecord(
        event_id=str(uuid.uuid4()),
        entity_type=entity_type,  # type: ignore
        entity_id=entity_id,
        status="failed",
        attempt_count=4,
        callback_url="https://example.com/callback",
        error_summary="SomeError: something went wrong",
        callback_delivery_failed=False,
        payload=final_payload,
        created_at=datetime.now(timezone.utc).isoformat(),
        updated_at=datetime.now(timezone.utc).isoformat(),
    )


class TestIngestQueue(unittest.TestCase):
    """Verify the ingest queue's file-based enqueue/dequeue/requeue behaviour."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.queue_path = Path(self.temp_dir) / "failed_queue"
        self.dead_letter_path = Path(self.temp_dir) / "dead_letter"
        self.queue = IngestQueue(
            queue_path=str(self.queue_path),
            dead_letter_path=str(self.dead_letter_path),
        )

    # ── helpers ────────────────────────────────────────────────────

    def _assert_file_exists(self, path: Path, msg: str = ""):
        self.assertTrue(path.exists(), msg or f"Expected file to exist: {path}")

    def _assert_file_not_exists(self, path: Path, msg: str = ""):
        self.assertFalse(path.exists(), msg or f"Expected file to be absent: {path}")

    def _read_entry(self, event_id: str) -> dict:
        path = self.queue._path_for(event_id)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # ── tests ─────────────────────────────────────────────────────

    def test_enqueue_creates_file_with_correct_fields(self):
        """After enqueue, the queue file exists and has the expected fields."""
        record = _make_record()
        self.queue.enqueue(record, backoff_base=60)

        path = self.queue._path_for(record.event_id)
        self._assert_file_exists(path)

        data = self._read_entry(record.event_id)
        self.assertEqual(data["event_id"], record.event_id)
        self.assertEqual(data["entity_type"], "candidate")
        self.assertEqual(data["entity_id"], 1)
        self.assertEqual(data["queue_retry_count"], 0)
        self.assertIn("next_retry_at", data)
        # next_retry_at should be in the future
        next_retry = datetime.fromisoformat(data["next_retry_at"])
        self.assertGreater(next_retry, datetime.now(timezone.utc) - timedelta(seconds=1))

    def test_enqueue_without_payload_does_not_create_file(self):
        """When payload is None, enqueue is a no-op and no file is written."""
        record = _make_record(payload=None)
        self.queue.enqueue(record, backoff_base=60)

        path = self.queue._path_for(record.event_id)
        self._assert_file_not_exists(path)

    def test_get_due_entries_returns_only_past_entries(self):
        """Only entries whose next_retry_at has passed are returned."""
        record_past = _make_record(entity_id=1)
        record_future = _make_record(entity_id=2)

        # Enqueue both
        self.queue.enqueue(record_past, backoff_base=0)  # due immediately
        self.queue.enqueue(record_future, backoff_base=3600)  # due in 1 hour

        # Manually set the future entry's next_retry_at far in the future
        future_entry_path = self.queue._path_for(record_future.event_id)
        with open(future_entry_path, "r", encoding="utf-8") as f:
            future_data = json.load(f)
        far_future = (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()
        future_data["next_retry_at"] = far_future
        with open(future_entry_path, "w", encoding="utf-8") as f:
            json.dump(future_data, f, ensure_ascii=False, indent=2)

        due = self.queue.get_due_entries()
        due_event_ids = [e.event_id for e in due]
        self.assertIn(record_past.event_id, due_event_ids)
        self.assertNotIn(record_future.event_id, due_event_ids)

    def test_requeue_increments_count_and_updates_backoff(self):
        """After requeue, retry count is incremented and next_retry_at ≈ now + backoff_base."""
        record = _make_record()
        self.queue.enqueue(record, backoff_base=60)
        path = self.queue._path_for(record.event_id)

        # Read the entry back
        data = self._read_entry(record.event_id)
        entry = IngestQueueEntry.from_dict(data)

        requeued = self.queue.requeue(entry, max_retries=5, backoff_base=60)
        self.assertTrue(requeued)

        updated = self._read_entry(record.event_id)
        self.assertEqual(updated["queue_retry_count"], 1)
        next_retry = datetime.fromisoformat(updated["next_retry_at"])
        # Should be roughly now + 60s
        expected_lower = datetime.now(timezone.utc) + timedelta(seconds=30)
        expected_upper = datetime.now(timezone.utc) + timedelta(seconds=90)
        self.assertGreaterEqual(next_retry, expected_lower)
        self.assertLessEqual(next_retry, expected_upper)

    def test_requeue_second_time_doubles_backoff(self):
        """After two requeues, backoff is approximately 2× the base (60s → 120s)."""
        record = _make_record()
        self.queue.enqueue(record, backoff_base=60)
        data = self._read_entry(record.event_id)
        entry = IngestQueueEntry.from_dict(data)

        # First requeue
        self.queue.requeue(entry, max_retries=5, backoff_base=60)
        # Second requeue — re-read the entry
        data2 = self._read_entry(record.event_id)
        entry2 = IngestQueueEntry.from_dict(data2)
        self.queue.requeue(entry2, max_retries=5, backoff_base=60)

        updated = self._read_entry(record.event_id)
        self.assertEqual(updated["queue_retry_count"], 2)
        next_retry = datetime.fromisoformat(updated["next_retry_at"])
        # Should be roughly now + 120s
        expected_lower = datetime.now(timezone.utc) + timedelta(seconds=90)
        expected_upper = datetime.now(timezone.utc) + timedelta(seconds=150)
        self.assertGreaterEqual(next_retry, expected_lower)
        self.assertLessEqual(next_retry, expected_upper)

    def test_requeue_exceeds_max_moves_to_dead_letter(self):
        """When retry count exceeds max_retries, the entry moves to dead letter."""
        record = _make_record()
        self.queue.enqueue(record, backoff_base=60)
        data = self._read_entry(record.event_id)
        # Set retry count to already equal max_retries
        data["queue_retry_count"] = 5
        path = self.queue._path_for(record.event_id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        entry = IngestQueueEntry.from_dict(data)
        result = self.queue.requeue(entry, max_retries=5, backoff_base=60)

        self.assertFalse(result)
        self._assert_file_not_exists(self.queue._path_for(record.event_id))
        self._assert_file_exists(self.queue._dead_letter_path_for(record.event_id))

    def test_remove_deletes_file(self):
        """After remove, the queue file no longer exists."""
        record = _make_record()
        self.queue.enqueue(record, backoff_base=60)
        self._assert_file_exists(self.queue._path_for(record.event_id))

        self.queue.remove(record.event_id)
        self._assert_file_not_exists(self.queue._path_for(record.event_id))

    def test_get_due_entries_skips_corrupt_files(self):
        """Corrupt JSON files are skipped with a warning instead of raising."""
        # Write a corrupt file directly
        corrupt_path = self.queue_path / "corrupt.json"
        corrupt_path.write_text("{ invalid json", encoding="utf-8")

        due = self.queue.get_due_entries()
        self.assertEqual(len(due), 0)


class TestIngestQueueClaiming(unittest.TestCase):
    """Verify atomic claim behaviour in get_due_entries and related ops."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.queue_path = Path(self.temp_dir) / "failed_queue"
        self.dead_letter_path = Path(self.temp_dir) / "dead_letter"
        self.queue = IngestQueue(
            queue_path=str(self.queue_path),
            dead_letter_path=str(self.dead_letter_path),
        )

    def _make_record(
        self,
        entity_type: str = "candidate",
        entity_id: int = 1,
    ) -> IngestionRecord:
        return IngestionRecord(
            event_id=str(uuid.uuid4()),
            entity_type=entity_type,  # type: ignore
            entity_id=entity_id,
            status="failed",
            attempt_count=4,
            callback_url="https://example.com/callback",
            error_summary="SomeError: something went wrong",
            callback_delivery_failed=False,
            payload={"cv_url": "https://example.com/cv.pdf", "profile_data": {}},
            created_at=datetime.now(timezone.utc).isoformat(),
            updated_at=datetime.now(timezone.utc).isoformat(),
        )

    def test_get_due_entries_moves_files_to_processing(self):
        """After get_due_entries, due files are moved to the .processing sub‑directory."""
        record = self._make_record()
        self.queue.enqueue(record, backoff_base=0)

        due = self.queue.get_due_entries()
        self.assertEqual(len(due), 1)
        self.assertEqual(due[0].event_id, record.event_id)

        # Queue file should no longer exist in the main directory
        self.assertFalse(self.queue._path_for(record.event_id).exists())
        # File should exist in the processing directory
        self.assertTrue(self.queue._processing_path_for(record.event_id).exists())

    def test_claimed_entries_not_returned_by_second_poll(self):
        """A second call to get_due_entries should NOT return already‑claimed entries."""
        record = self._make_record()
        self.queue.enqueue(record, backoff_base=0)

        # First poll claims the entry
        first_due = self.queue.get_due_entries()
        self.assertEqual(len(first_due), 1)

        # Second poll should return nothing (entry is in processing)
        second_due = self.queue.get_due_entries()
        self.assertEqual(len(second_due), 0)

    def test_remove_cleans_processing_file(self):
        """remove() should delete the processing file for a claimed entry."""
        record = self._make_record()
        self.queue.enqueue(record, backoff_base=0)
        self.queue.get_due_entries()  # claim it

        self.queue.remove(record.event_id)
        self.assertFalse(self.queue._processing_path_for(record.event_id).exists())
        self.assertFalse(self.queue._path_for(record.event_id).exists())

    def test_requeue_moves_entry_back_from_processing_to_queue(self):
        """requeue() should write updated data to the main queue dir and remove the processing file."""
        record = self._make_record()
        self.queue.enqueue(record, backoff_base=0)
        due = self.queue.get_due_entries()
        entry = due[0]

        # Processing file should exist
        self.assertTrue(self.queue._processing_path_for(entry.event_id).exists())

        # Requeue the entry
        requeued = self.queue.requeue(entry, max_retries=5, backoff_base=60)
        self.assertTrue(requeued)

        # Processing file should be gone
        self.assertFalse(self.queue._processing_path_for(entry.event_id).exists())
        # Queue file should be back in the main directory with updated retry count
        self.assertTrue(self.queue._path_for(entry.event_id).exists())
        import json
        with open(self.queue._path_for(entry.event_id), "r") as f:
            data = json.load(f)
        self.assertEqual(data["queue_retry_count"], 1)

    def test_dead_letter_from_processing_state(self):
        """When a claimed entry exceeds max_retries, it moves to dead letter from processing."""
        record = self._make_record()
        self.queue.enqueue(record, backoff_base=0)
        due = self.queue.get_due_entries()
        entry = due[0]

        # Set retry count to max so next requeue dead-letters
        entry.queue_retry_count = 5
        result = self.queue.requeue(entry, max_retries=5, backoff_base=60)
        self.assertFalse(result)

        # Processing file should be gone
        self.assertFalse(self.queue._processing_path_for(entry.event_id).exists())
        # Dead letter file should exist
        self.assertTrue(self.queue._dead_letter_path_for(entry.event_id).exists())

    def test_two_poll_cycles_dispatch_once(self):
        """Simulate two poll cycles: one due entry should be dispatched only once until requeued."""
        record = self._make_record()
        self.queue.enqueue(record, backoff_base=0)

        # Cycle 1
        cycle1 = self.queue.get_due_entries()
        self.assertEqual(len(cycle1), 1)

        # Cycle 2 (without having requeued or removed) — should see nothing
        cycle2 = self.queue.get_due_entries()
        self.assertEqual(len(cycle2), 0)

        # Simulate successful processing: remove the claimed entry
        self.queue.remove(record.event_id)

        # Cycle 3 — nothing to do
        cycle3 = self.queue.get_due_entries()
        self.assertEqual(len(cycle3), 0)

    def test_two_poll_cycles_dispatch_once_then_requeue(self):
        """After requeue, the entry becomes visible on the next poll cycle."""
        record = self._make_record()
        self.queue.enqueue(record, backoff_base=0)

        # Cycle 1: claim the entry
        cycle1 = self.queue.get_due_entries()
        self.assertEqual(len(cycle1), 1)
        entry = cycle1[0]

        # Cycle 2: should not see the claimed entry
        cycle2 = self.queue.get_due_entries()
        self.assertEqual(len(cycle2), 0)

        # Simulate failed processing: requeue it
        self.queue.requeue(entry, max_retries=5, backoff_base=0)  # backoff=0 so it's due immediately

        # Cycle 3: should see the requeued entry
        cycle3 = self.queue.get_due_entries()
        self.assertEqual(len(cycle3), 1)
        self.assertEqual(cycle3[0].event_id, record.event_id)


class TestDeadLetterMonitoring(unittest.TestCase):
    """Verify dead‑letter monitoring methods on IngestQueue."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.queue_path = Path(self.temp_dir) / "failed_queue"
        self.dead_letter_path = Path(self.temp_dir) / "dead_letter"
        self.queue = IngestQueue(
            queue_path=str(self.queue_path),
            dead_letter_path=str(self.dead_letter_path),
        )

    def _push_to_dead_letter(self, entity_type: str = "candidate", entity_id: int = 1) -> str:
        """Helper: enqueue a record, then force it to dead letter."""
        record = IngestionRecord(
            event_id=str(uuid.uuid4()),
            entity_type=entity_type,  # type: ignore
            entity_id=entity_id,
            status="failed",
            attempt_count=4,
            callback_url="https://example.com/callback",
            error_summary="SomeError: something went wrong",
            callback_delivery_failed=False,
            payload={"cv_url": "https://example.com/cv.pdf", "profile_data": {}},
            created_at=datetime.now(timezone.utc).isoformat(),
            updated_at=datetime.now(timezone.utc).isoformat(),
        )
        self.queue.enqueue(record, backoff_base=0)
        due = self.queue.get_due_entries()
        entry = due[0]
        entry.queue_retry_count = 5
        self.queue.requeue(entry, max_retries=5, backoff_base=60)
        return record.event_id

    def test_dead_letter_count_returns_zero_when_empty(self):
        """dead_letter_count() returns 0 when no entries exist."""
        self.assertEqual(self.queue.dead_letter_count(), 0)

    def test_dead_letter_count_returns_correct_number(self):
        """dead_letter_count() returns the number of dead letter entries."""
        self._push_to_dead_letter(entity_id=1)
        self._push_to_dead_letter(entity_id=2)
        self.assertEqual(self.queue.dead_letter_count(), 2)

    def test_get_dead_letter_entries_returns_entries(self):
        """get_dead_letter_entries() returns all dead letter entries with correct fields."""
        event_id = self._push_to_dead_letter(entity_id=42)
        entries = self.queue.get_dead_letter_entries()
        self.assertEqual(len(entries), 1)
        entry = entries[0]
        self.assertEqual(entry.event_id, event_id)
        self.assertEqual(entry.entity_type, "candidate")
        self.assertEqual(entry.entity_id, 42)
        self.assertEqual(entry.queue_retry_count, 6)  # incremented by requeue

    def test_get_dead_letter_entries_skips_corrupt_files(self):
        """Corrupt files in dead letter are skipped without raising."""
        # Write a corrupt file directly
        corrupt_path = self.dead_letter_path / "corrupt.json"
        corrupt_path.write_text("{ invalid json", encoding="utf-8")
        entries = self.queue.get_dead_letter_entries()
        self.assertEqual(len(entries), 0)

    def test_clear_dead_letter_removes_single_entry(self):
        """clear_dead_letter(event_id) removes only the specified entry."""
        id1 = self._push_to_dead_letter(entity_id=1)
        id2 = self._push_to_dead_letter(entity_id=2)
        self.assertEqual(self.queue.dead_letter_count(), 2)

        removed = self.queue.clear_dead_letter(event_id=id1)
        self.assertEqual(removed, 1)
        self.assertEqual(self.queue.dead_letter_count(), 1)
        # id2 should still exist
        self.assertTrue(self.queue._dead_letter_path_for(id2).exists())

    def test_clear_dead_letter_removes_all(self):
        """clear_dead_letter() with no args removes all entries."""
        self._push_to_dead_letter(entity_id=1)
        self._push_to_dead_letter(entity_id=2)
        self._push_to_dead_letter(entity_id=3)
        self.assertEqual(self.queue.dead_letter_count(), 3)

        removed = self.queue.clear_dead_letter()
        self.assertEqual(removed, 3)
        self.assertEqual(self.queue.dead_letter_count(), 0)


if __name__ == "__main__":
    unittest.main()
