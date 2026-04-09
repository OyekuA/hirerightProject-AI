"""Qdrant vector database client wrapper."""

import logging
from typing import Optional, Any

import qdrant_client
from qdrant_client import QdrantClient as QdrantSDKClient
from qdrant_client.http import models
from qdrant_client.models import Distance, VectorParams

from app.config import get_settings

from app.constants import CANDIDATES_COLLECTION, JOBS_COLLECTION


class QdrantClient:
    """Wrapper around qdrant-client SDK with convenience methods."""

    def __init__(self, host: str, port: int):
        """Initialize a connection to Qdrant.

        Args:
            host: Qdrant server hostname
            port: Qdrant server port
        """
        self._client: QdrantSDKClient = qdrant_client.QdrantClient(
            host=host, port=port
        )
        self._log = logging.getLogger(__name__)

    def ensure_collections(self) -> None:
        """Create candidates and jobs collections if they do not exist."""
        collections = self._client.get_collections().collections
        existing = {c.name for c in collections}

        if CANDIDATES_COLLECTION not in existing:
            self._client.create_collection(
                collection_name=CANDIDATES_COLLECTION,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE),
            )
            self._log.info("Created collection '%s'", CANDIDATES_COLLECTION)
        else:
            self._log.debug("Collection '%s' already exists", CANDIDATES_COLLECTION)

        if JOBS_COLLECTION not in existing:
            self._client.create_collection(
                collection_name=JOBS_COLLECTION,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE),
            )
            self._log.info("Created collection '%s'", JOBS_COLLECTION)
        else:
            self._log.debug("Collection '%s' already exists", JOBS_COLLECTION)

        self._ensure_payload_indexes(CANDIDATES_COLLECTION)
        self._ensure_payload_indexes(JOBS_COLLECTION)

    def _ensure_payload_indexes(self, collection_name: str) -> None:
        """Create payload indexes for filterable fields."""
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

    def upsert(
        self,
        collection: str,
        point_id: int,
        vector: list[float],
        payload: dict[str, Any],
    ) -> None:
        """Insert or replace a point in the given collection.

        Args:
            collection: Name of the collection (candidates or jobs)
            point_id: Unique integer identifier for the point
            vector: 768‑dimensional embedding vector
            payload: Dictionary of fields matching the collection's schema
        """
        point = models.PointStruct(id=point_id, vector=vector, payload=payload)
        self._client.upsert(collection_name=collection, points=[point])

    def get(self, collection: str, point_id: int) -> Optional[dict[str, Any]]:
        """Retrieve a point's payload by ID.

        Args:
            collection: Name of the collection
            point_id: ID of the point to retrieve

        Returns:
            The payload dictionary if the point exists, otherwise None.
        """
        points = self._client.retrieve(
            collection_name=collection,
            ids=[point_id],
        )
        if not points:
            return None
        return points[0].payload or {}

    def get_with_vector(self, collection: str, point_id: int) -> tuple[Optional[dict[str, Any]], Optional[list[float]]]:
        """Retrieve a point's payload and vector by ID.

        Args:
            collection: Name of the collection
            point_id: ID of the point to retrieve

        Returns:
            A tuple (payload, vector). If the point does not exist, returns (None, None).
        """
        points = self._client.retrieve(
            collection_name=collection,
            ids=[point_id],
            with_vectors=True,
        )
        if not points:
            return None, None
        record = points[0]
        return record.payload or {}, record.vector

    def search(
        self,
        collection: str,
        query_vector: list[float],
        limit: int,
        query_filter: Optional[models.Filter] = None,
    ) -> list[dict[str, Any]]:
        """Search for similar vectors in the collection.

        Args:
            collection: Name of the collection
            query_vector: 768‑dimensional query embedding
            limit: Maximum number of results to return
            query_filter: Optional filter to restrict the search

        Returns:
            List of payload dictionaries, each enriched with a 'score' field.
        """
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

    def scroll(
        self,
        collection: str,
        query_filter: Optional[models.Filter] = None,
        limit: int = 10,
        order_by: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Retrieve points by filter without vector similarity.

        Args:
            collection: Name of the collection.
            query_filter: Optional filter to restrict the scroll.
            limit: Maximum number of points to return.
            order_by: Optional ordering field (not supported by all Qdrant versions).

        Returns:
            List of payload dictionaries, each enriched with '_point_id' and 'score' = 0.0.
        """
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

    def delete(self, collection: str, point_id: int) -> None:
        """Delete a point from the collection.

        Args:
            collection: Name of the collection
            point_id: ID of the point to delete
        """
        self._client.delete(
            collection_name=collection,
            points_selector=models.PointIdsList(points=[point_id]),
        )

    def update_payload(self, collection: str, point_id: int, fields: dict) -> None:
        """Update specific payload fields of an existing point.

        Args:
            collection: Name of the collection (candidates or jobs)
            point_id: Unique integer identifier for the point
            fields: Dictionary of payload fields to update (will be merged with existing payload)
        """
        self._client.set_payload(
            collection_name=collection,
            payload=fields,
            points=[point_id],
        )
        self._log.debug(
            "Updated payload for point %d in collection '%s'",
            point_id, collection,
        )