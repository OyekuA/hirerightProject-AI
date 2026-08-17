import functools
import logging
from typing import Any, Optional, Union

import httpx
import qdrant_client
from qdrant_client import QdrantClient as QdrantSDKClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from qdrant_client.models import Distance, VectorParams

from app.config import get_settings

from app.constants import CANDIDATES_COLLECTION, JOBS_COLLECTION


_MISSING = object()
MISSING = _MISSING

QDRANT_TIMEOUT_SECONDS = 15


class QdrantUnavailableError(Exception):
    """Vector store unreachable or failing at the transport level."""


def _guarded(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except (ResponseHandlingException, UnexpectedResponse, httpx.HTTPError, ConnectionError) as exc:
            self._log.warning("Qdrant transport error: %s", exc)
            raise QdrantUnavailableError(f"Qdrant unavailable: {exc}") from exc
    return wrapper


class QdrantClient:

    def __init__(self, host: str, port: int):
        self._client: QdrantSDKClient = qdrant_client.QdrantClient(
            host=host, port=port, timeout=QDRANT_TIMEOUT_SECONDS
        )
        self._log = logging.getLogger(__name__)

    def _recreate_collection(self, collection_name: str, expected_dim: int) -> None:
        collection_info = self._client.get_collection(collection_name)
        existing_dim = collection_info.config.params.vectors.size
        raise RuntimeError(
            f"Collection '{collection_name}' has dim={existing_dim} but config expects "
            f"dim={expected_dim}. Dimension mismatch requires manual migration — "
            f"aborting startup."
        )

    @_guarded
    def ensure_collections(self) -> None:
        settings = get_settings()
        collections = self._client.get_collections().collections
        existing = {c.name for c in collections}

        expected_dim = settings.EMBEDDING_DIMENSIONS

        for collection_name in (CANDIDATES_COLLECTION, JOBS_COLLECTION):
            if collection_name not in existing:
                self._client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=expected_dim, distance=Distance.COSINE),
                )
                self._log.info("Created collection '%s' with dim=%d", collection_name, expected_dim)
            else:
                collection_info = self._client.get_collection(collection_name)
                existing_dim = collection_info.config.params.vectors.size
                if existing_dim != expected_dim:
                    self._log.warning(
                        "Collection '%s' has dim=%d but config expects dim=%d — will recreate",
                        collection_name, existing_dim, expected_dim,
                    )
                    self._recreate_collection(collection_name, expected_dim)
                else:
                    self._log.debug("Collection '%s' already exists with dim=%d", collection_name, existing_dim)

        self._ensure_payload_indexes(CANDIDATES_COLLECTION)
        self._ensure_payload_indexes(JOBS_COLLECTION)

    def _ensure_payload_indexes(self, collection_name: str) -> None:
        if collection_name == CANDIDATES_COLLECTION:
            fields = [
                ("location", models.PayloadSchemaType.KEYWORD),
                ("experience_level", models.PayloadSchemaType.KEYWORD),
                ("industry", models.PayloadSchemaType.KEYWORD),
                ("employment_type", models.PayloadSchemaType.KEYWORD),
                ("candidate_version", models.PayloadSchemaType.INTEGER),
            ]
        else:
            fields = [
                ("location", models.PayloadSchemaType.KEYWORD),
                ("experience_level", models.PayloadSchemaType.KEYWORD),
                ("industry", models.PayloadSchemaType.KEYWORD),
                ("employment_type", models.PayloadSchemaType.KEYWORD),
                ("job_version", models.PayloadSchemaType.INTEGER),
            ]
        for field_name, field_type in fields:
            try:
                self._client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=field_type,
                )
                self._log.debug(
                    "Created payload index for field '%s' in collection '%s'",
                    field_name, collection_name,
                )
            except Exception as e:
                self._log.debug(
                    "Payload index for field '%s' may already exist: %s",
                    field_name, e,
                )

    @_guarded
    def upsert(
        self,
        collection: str,
        point_id: int,
        vector: list[float],
        payload: dict[str, Any],
    ) -> None:
        point = models.PointStruct(id=point_id, vector=vector, payload=payload)
        self._client.upsert(collection_name=collection, points=[point])

    @_guarded
    def get(self, collection: str, point_id: int) -> Union[dict[str, Any], object]:
        points = self._client.retrieve(
            collection_name=collection,
            ids=[point_id],
        )
        if not points:
            return _MISSING
        return points[0].payload or {}

    @_guarded
    def get_with_vector(self, collection: str, point_id: int) -> tuple[Union[dict[str, Any], object], Optional[list[float]]]:
        points = self._client.retrieve(
            collection_name=collection,
            ids=[point_id],
            with_vectors=True,
        )
        if not points:
            return _MISSING, None
        record = points[0]
        return record.payload or {}, record.vector

    @_guarded
    def search(
        self,
        collection: str,
        query_vector: list[float],
        limit: int,
        query_filter: Optional[models.Filter] = None,
    ) -> list[dict[str, Any]]:
        result = self._client.query_points(
            collection_name=collection,
            query=query_vector,
            limit=limit,
            query_filter=query_filter,
        )
        search_result = result.points
        results = []
        for hit in search_result:
            payload = hit.payload or {}
            payload["_point_id"] = hit.id
            payload["score"] = hit.score
            results.append(payload)
        return results

    @_guarded
    def scroll(
        self,
        collection: str,
        query_filter: Optional[models.Filter] = None,
        limit: int = 10,
        order_by: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        scroll_result = self._client.scroll(
            collection_name=collection,
            scroll_filter=query_filter,
            limit=limit,
            order_by=order_by,
        )
        points, _ = scroll_result
        results = []
        for point in points:
            payload = point.payload or {}
            payload["_point_id"] = point.id
            payload["score"] = 0.0
            results.append(payload)
        return results

    @_guarded
    def delete(self, collection: str, point_id: int) -> None:
        self._client.delete(
            collection_name=collection,
            points_selector=models.PointIdsList(points=[point_id]),
        )

    @_guarded
    def update_payload(self, collection: str, point_id: int, fields: dict) -> None:
        self._client.set_payload(
            collection_name=collection,
            payload=fields,
            points=[point_id],
        )
        self._log.debug(
            "Updated payload for point %d in collection '%s'",
            point_id, collection,
        )
