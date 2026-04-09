"""Durable store for ingestion job status.

Each ingestion job is persisted as a JSON file in a configurable directory.
Thread‑safe via a global lock.
"""

import json
import threading
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional, List, Dict, Any
import structlog

_logger = structlog.get_logger(__name__)


EntityType = Literal["candidate", "job"]
Status = Literal["pending", "running", "success", "failed"]


@dataclass
class IngestionRecord:
    """Immutable representation of an ingestion job's status."""
    event_id: str
    entity_type: EntityType
    entity_id: int
    status: Status
    attempt_count: int
    callback_url: str
    error_summary: Optional[str] = None
    callback_delivery_failed: bool = False
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IngestionRecord":
        return cls(**data)


class IngestionStatusStore:
    """Thread‑safe file‑based store for ingestion status records."""

    def __init__(self, store_path: str):
        self._store_path = Path(store_path)
        self._store_path.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        _logger.info("Ingestion status store initialized", store_path=self._store_path)

    def _path_for(self, event_id: str) -> Path:
        return self._store_path / f"{event_id}.json"

    def create(
        self,
        entity_type: EntityType,
        entity_id: int,
        callback_url: str,
    ) -> IngestionRecord:
        """Create a new pending record and persist it."""
        event_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        record = IngestionRecord(
            event_id=event_id,
            entity_type=entity_type,
            entity_id=entity_id,
            status="pending",
            attempt_count=0,
            callback_url=callback_url,
            error_summary=None,
            callback_delivery_failed=False,
            created_at=now,
            updated_at=now,
        )

        with self._lock:
            with open(self._path_for(event_id), "w", encoding="utf-8") as f:
                json.dump(record.to_dict(), f, ensure_ascii=False, indent=2)

        _logger.debug("Created ingestion record", event_id=event_id, entity_type=entity_type, entity_id=entity_id)
        return record

    def update(self, event_id: str, **kwargs) -> None:
        """Update specific fields of an existing record."""
        with self._lock:
            path = self._path_for(event_id)
            if not path.exists():
                raise KeyError(f"No ingestion record with event_id {event_id}")

            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            data.update(kwargs)
            data["updated_at"] = datetime.now(timezone.utc).isoformat()

            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        _logger.debug("Updated ingestion record", event_id=event_id)

    def get_by_event_id(self, event_id: str) -> Optional[IngestionRecord]:
        """Retrieve a record by its event_id."""
        path = self._path_for(event_id)
        with self._lock:
            if not path.exists():
                return None
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        return IngestionRecord.from_dict(data)

    def get_by_entity(
        self,
        entity_type: EntityType,
        entity_id: int,
    ) -> Optional[IngestionRecord]:
        """Find the most recent record for a given entity."""
        with self._lock:
            candidates = []
            for p in self._store_path.glob("*.json"):
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if (data["entity_type"] == entity_type and
                        data["entity_id"] == entity_id):
                        candidates.append(IngestionRecord.from_dict(data))
                except (json.JSONDecodeError, KeyError, IOError) as e:
                    _logger.warning("Skipping corrupt status file", file_path=str(p), error=str(e))

        if not candidates:
            return None
        return max(candidates, key=lambda r: r.created_at)

    def get_all_incomplete(self) -> List[IngestionRecord]:
        """Return all records with status 'pending' or 'running'."""
        with self._lock:
            incomplete = []
            for p in self._store_path.glob("*.json"):
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if data["status"] in ("pending", "running"):
                        incomplete.append(IngestionRecord.from_dict(data))
                except (json.JSONDecodeError, KeyError, IOError) as e:
                    _logger.warning("Skipping corrupt status file", file_path=str(p), error=str(e))

        return incomplete