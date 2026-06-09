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
    event_id: str
    entity_type: EntityType
    entity_id: int
    status: Status
    attempt_count: int
    callback_url: str
    error_summary: Optional[str] = None
    callback_delivery_failed: bool = False
    payload: Optional[dict] = None
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IngestionRecord":
        data.setdefault("payload", None)
        return cls(**data)


class IngestionStatusStore:

    def __init__(self, store_path: str):
        self._store_path = Path(store_path)
        self._store_path.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._entity_index: dict[tuple[str, int], tuple[str, str]] = {}
        self._build_entity_index()
        _logger.info("Ingestion status store initialized", store_path=self._store_path)

    def _build_entity_index(self) -> None:
        for p in self._store_path.glob("*.json"):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                entity_type = data.get("entity_type")
                entity_id = data.get("entity_id")
                event_id = data.get("event_id")
                created_at = data.get("created_at", "")
                if entity_type is not None and entity_id is not None and event_id is not None:
                    key = (entity_type, entity_id)
                    existing = self._entity_index.get(key)
                    if existing is None or created_at > existing[1]:
                        self._entity_index[key] = (event_id, created_at)
            except (json.JSONDecodeError, KeyError, IOError) as e:
                _logger.warning("Skipping corrupt status file during index build", file_path=str(p), error=str(e))

    def _path_for(self, event_id: str) -> Path:
        return self._store_path / f"{event_id}.json"

    def create(
        self,
        entity_type: EntityType,
        entity_id: int,
        callback_url: str,
        payload: Optional[dict] = None,
    ) -> IngestionRecord:
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
            payload=payload,
            created_at=now,
            updated_at=now,
        )

        with self._lock:
            with open(self._path_for(event_id), "w", encoding="utf-8") as f:
                json.dump(record.to_dict(), f, ensure_ascii=False, indent=2)
            self._entity_index[(entity_type, entity_id)] = (event_id, now)

        _logger.debug("Created ingestion record", event_id=event_id, entity_type=entity_type, entity_id=entity_id)
        return record

    def update(self, event_id: str, **kwargs) -> None:
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
        with self._lock:
            key = (entity_type, entity_id)
            indexed = self._entity_index.get(key)

        if indexed is not None:
            event_id = indexed[0]
            return self.get_by_event_id(event_id)

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
        best = max(candidates, key=lambda r: r.created_at)
        with self._lock:
            self._entity_index[key] = (best.event_id, best.created_at)
        return best

    def get_all_incomplete(self) -> List[IngestionRecord]:
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
