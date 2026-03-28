"""Recommendation service for suggesting jobs to candidates or candidates to jobs.

This module provides the RecommendationService class that combines vector similarity,
intent embeddings, and cached LLM scores to produce a ranked list of recommendations.
"""

import numpy as np
import structlog
from typing import Literal

import qdrant_client.http.models as qdrant_models

from app.clients.dependencies import CANDIDATES_COLLECTION, JOBS_COLLECTION
from app.clients.gemini import GeminiClient
from app.clients.qdrant import QdrantClient
from app.clients.cache import CacheBackend
from app.utils import truncate_to_prompt_cap

logger = structlog.get_logger()


class RecommendationService:
    """Service that encapsulates hybrid recommendation logic."""

    def __init__(
        self,
        gemini: GeminiClient,
        qdrant: QdrantClient,
        cache: CacheBackend,
    ):
        """Initialize the recommendation service.

        Args:
            gemini: A configured GeminiClient instance.
            qdrant: A QdrantClient instance.
            cache: A CacheBackend instance.
        """
        self.gemini = gemini
        self.qdrant = qdrant
        self.cache = cache

    def recommend(
        self,
        rec_type: Literal["jobs", "candidates"],
        target_id: int,
        target_version: int,
        behavioral_signals: dict = None,
        hard_filters: dict = None,
        force_refresh: bool = False,
        limit: int = 10,
    ) -> list[dict]:
        """Generate recommendations for a target profile.

        Args:
            rec_type: "jobs" to recommend jobs to a candidate,
                      "candidates" to recommend candidates to a job.
            target_id: Unique identifier of the target profile in the vector store.
            target_version: Version of the target profile.
            behavioral_signals: Dict with keys:
                - recent_searches: list[str]
                - recent_clicks: list[dict] with id, dwell_time_seconds
                - recent_saves: list[int]
                - recent_positive_outcomes: list[int]
            hard_filters: Dict of filter conditions (location, experience_level,
                          industry, employment_type).
            force_refresh: If True, bypass cached LLM scores for each result.
            limit: Maximum number of recommendations to return (capped at 50).

        Returns:
            A list of dicts, each with keys:
                - id (int): recommended profile ID
                - similarity_score (float): composite similarity score
                - llm_score (Optional[int]): cached overall_score_percentage if available

        Raises:
            ValueError: If the target profile is not found in the vector store.
            GeminiUnavailableError: If the Gemini circuit breaker is open or the call fails.
        """
        if behavioral_signals is None:
            behavioral_signals = {}
        if hard_filters is None:
            hard_filters = {}

        recent_searches = behavioral_signals.get("recent_searches", [])
        recent_clicks = behavioral_signals.get("recent_clicks", [])
        recent_saves = behavioral_signals.get("recent_saves", [])
        recent_positive_outcomes = behavioral_signals.get("recent_positive_outcomes", [])

        logger.info(
            "Generating recommendations",
            rec_type=rec_type,
            target_id=target_id,
            target_version=target_version,
            behavioral_signals_counts={
                "searches": len(recent_searches),
                "clicks": len(recent_clicks),
                "saves": len(recent_saves),
                "positive_outcomes": len(recent_positive_outcomes),
            },
            hard_filters=hard_filters,
            force_refresh=force_refresh,
            limit=limit,
        )

        if rec_type == "jobs":
            search_collection = JOBS_COLLECTION
            target_collection = CANDIDATES_COLLECTION
            id_field = "job_id"
            version_field = "job_version"
        else:
            search_collection = CANDIDATES_COLLECTION
            target_collection = JOBS_COLLECTION
            id_field = "candidate_id"
            version_field = "candidate_version"

        target_payload, profile_vec = self.qdrant.get_with_vector(target_collection, target_id)
        if not target_payload or profile_vec is None:
            if not target_payload:
                logger.warning(
                    "Target profile not found in vector store",
                    target_id=target_id,
                    collection=target_collection,
                )
                raise ValueError(f"Target profile not found: {target_id}")
            else:
                logger.error(
                    "Target profile vector is missing",
                    target_id=target_id,
                    collection=target_collection,
                )
                raise ValueError(f"Target profile vector is missing: {target_id}")

        if target_collection == CANDIDATES_COLLECTION:
            target_skills = target_payload.get("skills", [])
        else:
            target_skills = target_payload.get("required_skills", [])
        if len(target_skills) < 3:
            conditions = []
            filter_mapping = {
                "location": qdrant_models.FieldCondition(
                    key="location",
                    match=qdrant_models.MatchValue(value=hard_filters["location"]),
                ) if "location" in hard_filters else None,
                "experience_level": qdrant_models.FieldCondition(
                    key="experience_level",
                    match=qdrant_models.MatchValue(value=hard_filters["experience_level"]),
                ) if "experience_level" in hard_filters else None,
                "industry": qdrant_models.FieldCondition(
                    key="industry",
                    match=qdrant_models.MatchValue(value=hard_filters["industry"]),
                ) if "industry" in hard_filters else None,
                "employment_type": qdrant_models.FieldCondition(
                    key="employment_type",
                    match=qdrant_models.MatchValue(value=hard_filters["employment_type"]),
                ) if "employment_type" in hard_filters else None,
            }
            for cond in filter_mapping.values():
                if cond is not None:
                    conditions.append(cond)
            filter_obj = qdrant_models.Filter(must=conditions) if conditions else None

            effective_limit = min(limit, 50)
            scroll_results = self.qdrant.scroll(
                collection=search_collection,
                query_filter=filter_obj,
                limit=effective_limit,
            )
            if search_collection == CANDIDATES_COLLECTION:
                skill_field = "skills"
            else:
                skill_field = "required_skills"
            scroll_results.sort(
                key=lambda r: (
                    r.get("ingested_at", ""),
                    len(r.get(skill_field, [])),
                ),
                reverse=True,
            )
            output = []
            for result in scroll_results:
                result_id = result["_point_id"]
                result_version = result.get(version_field)

                if rec_type == "jobs":
                    candidate_id_for_key = target_id
                    candidate_version_for_key = target_version
                    job_id_for_key = result_id
                    job_version_for_key = result_version
                else:
                    candidate_id_for_key = result_id
                    candidate_version_for_key = result_version
                    job_id_for_key = target_id
                    job_version_for_key = target_version

                llm_score = None
                if not force_refresh and result_version is not None:
                    cache_key = f"{candidate_id_for_key}:{candidate_version_for_key}:{job_id_for_key}:{job_version_for_key}"
                    cached = self.cache.get(cache_key)
                    if cached is not None:
                        llm_score = cached.get("overall_score_percentage")
                        if llm_score is not None:
                            logger.debug(
                                "LLM score cache hit",
                                cache_key=cache_key,
                                llm_score=llm_score,
                            )

                output.append({
                    "id": result_id,
                    "similarity_score": 0.0,
                    "llm_score": llm_score,
                })

            logger.info(
                "Cold‑start recommendations (skills < 3)",
                target_id=target_id,
                result_count=len(output),
            )
            return output

        total_signals = len(recent_searches) + len(recent_clicks) + len(recent_saves) + len(recent_positive_outcomes)
        intent_weight = min(0.45, 0.10 + 0.05 * total_signals) if len(recent_searches) >= 3 else 0.0
        cooccurrence_weight = min(0.20, 0.05 * (len(recent_saves) + len(recent_positive_outcomes))) if (len(recent_saves) + len(recent_positive_outcomes)) >= 2 else 0.0
        peer_weight = 0.10
        profile_weight = 1.0 - intent_weight - cooccurrence_weight - peer_weight
        profile_weight = max(0.0, profile_weight)

        dim = len(profile_vec)
        zero_vec = np.zeros(dim)

        profile_vec_np = np.array(profile_vec)

        if len(recent_searches) >= 3:
            intent_embeddings = []
            for search in recent_searches:
                truncated_search = truncate_to_prompt_cap(search)
                embedding = self.gemini.embed(truncated_search)
                intent_embeddings.append(embedding)
            intent_vec = np.mean(intent_embeddings, axis=0)
        else:
            intent_vec = zero_vec

        if len(recent_saves) + len(recent_positive_outcomes) >= 2:
            cooc_ids = []
            for click in recent_clicks:
                cooc_ids.append(click["id"])
            cooc_ids.extend(recent_saves)
            cooc_ids.extend(recent_positive_outcomes)
            cooc_vectors = []
            for cooc_id in cooc_ids:
                _, vec = self.qdrant.get_with_vector(search_collection, cooc_id)
                if vec is not None:
                    cooc_vectors.append(vec)
            if cooc_vectors:
                cooc_vec = np.mean(cooc_vectors, axis=0)
            else:
                cooc_vec = zero_vec
        else:
            cooc_vec = zero_vec

        peer_search_results = self.qdrant.search(
            collection=target_collection,
            query_vector=profile_vec,
            limit=6,
        )
        peer_vectors = []
        for peer in peer_search_results:
            peer_id = peer["_point_id"]
            if peer_id == target_id:
                continue
            _, peer_vec = self.qdrant.get_with_vector(target_collection, peer_id)
            if peer_vec is not None:
                peer_vectors.append(peer_vec)
            if len(peer_vectors) >= 5:
                break
        if peer_vectors:
            peer_vec = np.mean(peer_vectors, axis=0)
        else:
            peer_vec = zero_vec

        query_vec = (
            profile_weight * profile_vec_np
            + intent_weight * intent_vec
            + cooccurrence_weight * cooc_vec
            + peer_weight * peer_vec
        )
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec = query_vec / norm
        query_vec = query_vec.tolist()

        conditions = []
        filter_mapping = {
            "location": qdrant_models.FieldCondition(
                key="location",
                match=qdrant_models.MatchValue(value=hard_filters["location"]),
            ) if "location" in hard_filters else None,
            "experience_level": qdrant_models.FieldCondition(
                key="experience_level",
                match=qdrant_models.MatchValue(value=hard_filters["experience_level"]),
            ) if "experience_level" in hard_filters else None,
            "industry": qdrant_models.FieldCondition(
                key="industry",
                match=qdrant_models.MatchValue(value=hard_filters["industry"]),
            ) if "industry" in hard_filters else None,
            "employment_type": qdrant_models.FieldCondition(
                key="employment_type",
                match=qdrant_models.MatchValue(value=hard_filters["employment_type"]),
            ) if "employment_type" in hard_filters else None,
        }
        for cond in filter_mapping.values():
            if cond is not None:
                conditions.append(cond)
        filter_obj = qdrant_models.Filter(must=conditions) if conditions else None

        effective_limit = min(limit, 50)
        raw_results = self.qdrant.search(
            collection=search_collection,
            query_vector=query_vec,
            limit=effective_limit * 3,
            query_filter=filter_obj,
        )

        def jaccard_similarity(list_a, list_b):
            if not list_a and not list_b:
                return 0.0
            set_a = set(list_a)
            set_b = set(list_b)
            intersection = len(set_a & set_b)
            union = len(set_a | set_b)
            return intersection / union if union > 0 else 0.0

        for result in raw_results:
            result_skills = result.get("skills", []) or result.get("required_skills", [])
            skill_overlap = jaccard_similarity(target_skills, result_skills)
            location_match = 1.0 if target_payload.get("location") == result.get("location") else 0.0
            level_match = 1.0 if target_payload.get("experience_level") == result.get("experience_level") else 0.0
            employment_match = 1.0 if target_payload.get("employment_type") == result.get("employment_type") else 0.0
            final_score = (
                0.55 * result["score"]
                + 0.20 * skill_overlap
                + 0.10 * location_match
                + 0.10 * level_match
                + 0.05 * employment_match
            )
            result["final_score"] = final_score

        raw_results.sort(key=lambda r: r["final_score"], reverse=True)

        def cluster_key(result):
            if rec_type == "jobs":
                return (result.get("title", ""), result.get("location", ""))
            else:
                return (result.get("name", ""), result.get("location", ""))

        clusters = {}
        for result in raw_results:
            key = cluster_key(result)
            if key not in clusters or result["final_score"] > clusters[key]["final_score"]:
                clusters[key] = result

        clustered_results = list(clusters.values())
        clustered_results.sort(key=lambda r: r["final_score"], reverse=True)
        selected = []
        for result in clustered_results:
            if len(selected) >= effective_limit - 2:
                break
            selected.append(result)

        selected_cluster_keys = set(cluster_key(result) for result in selected)
        exploration_candidates = [r for r in clustered_results if cluster_key(r) not in selected_cluster_keys]
        exploration_candidates.sort(key=lambda r: r["final_score"], reverse=True)
        for candidate in exploration_candidates[:2]:
            if len(selected) < effective_limit:
                selected.append(candidate)

        seen_ids = set()
        dedup_selected = []
        for result in selected:
            pid = result["_point_id"]
            if pid not in seen_ids:
                seen_ids.add(pid)
                dedup_selected.append(result)
        selected = dedup_selected

        selected = selected[:effective_limit]

        output = []
        for result in selected:
            result_id = result["_point_id"]
            similarity_score = result["final_score"]
            result_version = result.get(version_field)

            if rec_type == "jobs":
                candidate_id_for_key = target_id
                candidate_version_for_key = target_version
                job_id_for_key = result_id
                job_version_for_key = result_version
            else:
                candidate_id_for_key = result_id
                candidate_version_for_key = result_version
                job_id_for_key = target_id
                job_version_for_key = target_version

            llm_score = None
            if not force_refresh and result_version is not None:
                cache_key = f"{candidate_id_for_key}:{candidate_version_for_key}:{job_id_for_key}:{job_version_for_key}"
                cached = self.cache.get(cache_key)
                if cached is not None:
                    llm_score = cached.get("overall_score_percentage")
                    if llm_score is not None:
                        logger.debug(
                            "LLM score cache hit",
                            cache_key=cache_key,
                            llm_score=llm_score,
                        )

            output.append({
                "id": result_id,
                "similarity_score": similarity_score,
                "llm_score": llm_score,
            })

        logger.info(
            "Recommendations generated",
            target_id=target_id,
            rec_type=rec_type,
            result_count=len(output),
        )
        return output