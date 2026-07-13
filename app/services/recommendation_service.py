import concurrent.futures
import math
import numpy as np
import structlog
from concurrent.futures import wait
from typing import Literal

import qdrant_client.http.models as qdrant_models

from app.clients.dependencies import CANDIDATES_COLLECTION, JOBS_COLLECTION
from app.clients.llm import LLMClient, LLMUnavailableError
from app.clients.qdrant import QdrantClient, MISSING
from app.clients.cache import CacheBackend
from app.constants import EXPERIENCE_LEVEL_LADDER, canonicalize_experience_level
from app.services.scoring_service import ScoringService
from app.utils.ingestion import truncate_to_prompt_cap

logger = structlog.get_logger()

POOL_RANK_CONCURRENCY = 5
POOL_RANK_TIMEOUT_SECONDS = 30
POOL_RANK_DEFAULT_FIT_SCORE = 0


class RecommendationService:

    def __init__(
        self,
        llm: LLMClient,
        qdrant: QdrantClient,
        cache: CacheBackend,
    ):
        self.llm = llm
        self.qdrant = qdrant
        self.cache = cache

    @staticmethod
    def _jaccard_similarity(list_a, list_b):
        if not list_a and not list_b:
            return 0.0
        set_a = {s.lower().strip() for s in list_a}
        set_b = {s.lower().strip() for s in list_b}
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union > 0 else 0.0

    @staticmethod
    def _canonicalize_level(level: str) -> str:
        return canonicalize_experience_level(level)

    def _build_filter(self, hard_filters: dict):
        conditions = [
            qdrant_models.FieldCondition(key=k, match=qdrant_models.MatchValue(value=hard_filters[k]))
            for k in ("location", "experience_level", "industry", "employment_type")
            if k in hard_filters
        ]
        return qdrant_models.Filter(must=conditions) if conditions else None

    def _compute_composite_score(self, target_payload, result, target_skills, vector_score, collection=None):
        if collection is not None:
            skill_field = self._resolve_skill_field(collection)
            result_skills = [s.lower().strip() for s in result.get(skill_field, [])]
        else:
            result_skills = [s.lower().strip() for s in (result.get("skills", []) or result.get("required_skills", []))]
        skill_overlap = self._jaccard_similarity(target_skills, result_skills)
        target_city = target_payload.get("location", "").split(",")[0].strip().lower()
        result_city = result.get("location", "").split(",")[0].strip().lower()
        target_remote = "remote" in target_payload.get("employment_type", "").lower()
        result_remote = "remote" in result.get("employment_type", "").lower()
        location_match = 1.0 if (target_remote or result_remote) or (target_city and target_city == result_city) else 0.0
        target_level = target_payload.get("experience_level", "").lower().strip().replace("-", " ")
        result_level = result.get("experience_level", "").lower().strip().replace("-", " ")
        target_level_canon = self._canonicalize_level(target_level)
        result_level_canon = self._canonicalize_level(result_level)

        if target_level_canon in EXPERIENCE_LEVEL_LADDER and result_level_canon in EXPERIENCE_LEVEL_LADDER:
            idx_a = EXPERIENCE_LEVEL_LADDER.index(target_level_canon)
            idx_b = EXPERIENCE_LEVEL_LADDER.index(result_level_canon)
            level_match = max(0.0, 1.0 - abs(idx_a - idx_b) / 4)
        elif target_level_canon == result_level_canon:
            level_match = 1.0
        else:
            level_match = 0.0
        employment_match = 1.0 if target_payload.get("employment_type") == result.get("employment_type") else 0.0
        return (
            0.60 * vector_score
            + 0.15 * skill_overlap
            + 0.10 * location_match
            + 0.10 * level_match
            + 0.05 * employment_match
        )

    def _resolve_cache_ids(self, rec_type, target_id, target_version, result_id, result_version):
        if rec_type == "jobs":
            return target_id, target_version, result_id, result_version
        else:
            return result_id, result_version, target_id, target_version

    def _lookup_llm_score(self, candidate_id, candidate_version, job_id, job_version, result_version, force_refresh):
        if force_refresh or result_version is None:
            return None
        cache_key = f"{candidate_id}:{candidate_version}:{job_id}:{job_version}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            llm_score = cached.get("overall_score_percentage")
            if llm_score is not None:
                logger.debug(
                    "LLM score cache hit",
                    cache_key=cache_key,
                    llm_score=llm_score,
                )
            return llm_score
        return None

    @staticmethod
    def _resolve_skill_field(collection: str) -> str:
        if collection == CANDIDATES_COLLECTION:
            return "skills"
        else:
            return "required_skills"

    def _compute_weights(self, recent_searches, recent_clicks, recent_saves, recent_positive_outcomes):
        intent_signals = len(recent_searches) + len(recent_clicks) + len(recent_saves) + len(recent_positive_outcomes)
        intent_weight = min(0.45, 0.10 + 0.05 * intent_signals) if len(recent_searches) >= 3 else 0.0
        cooccurrence_weight = min(0.20, 0.05 * (len(recent_saves) + len(recent_positive_outcomes))) if (len(recent_saves) + len(recent_positive_outcomes)) >= 2 else 0.0
        peer_weight = 0.10
        profile_weight = 1.0 - intent_weight - cooccurrence_weight - peer_weight
        profile_weight = max(0.0, profile_weight)

        return intent_weight, cooccurrence_weight, peer_weight, profile_weight

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
            version_field = "job_version"
        else:
            search_collection = CANDIDATES_COLLECTION
            target_collection = JOBS_COLLECTION
            version_field = "candidate_version"

        target_payload, profile_vec = self.qdrant.get_with_vector(target_collection, target_id)
        if not target_payload:
            logger.warning(
                "Target profile not found in vector store",
                target_id=target_id,
                collection=target_collection,
            )
            raise ValueError(f"Target profile not found: {target_id}")
        if target_collection == CANDIDATES_COLLECTION:
            target_version_field = "candidate_version"
        else:
            target_version_field = "job_version"
        stored_target_version = target_payload.get(target_version_field, 1)
        if stored_target_version != target_version:
            logger.warning(
                "Target version mismatch",
                target_id=target_id,
                stored_version=stored_target_version,
                requested_version=target_version,
            )
            raise ValueError(f"Target version mismatch: stored {stored_target_version}, requested {target_version}")
        if profile_vec is None:
            logger.warning(
                "Target profile vector is missing – falling back to cold‑start",
                target_id=target_id,
                collection=target_collection,
            )

        target_skills = [s.lower().strip() for s in target_payload.get(self._resolve_skill_field(target_collection), [])]


        if profile_vec is None:
            filter_obj = self._build_filter(hard_filters)

            effective_limit = min(limit, 50)
            scroll_results = self.qdrant.scroll(
                collection=search_collection,
                query_filter=filter_obj,
                limit=effective_limit,
            )

            scored_results = []
            for result in scroll_results:
                result_id = result["_point_id"]
                result_version = result.get(version_field)

                candidate_id_for_key, candidate_version_for_key, job_id_for_key, job_version_for_key = self._resolve_cache_ids(
                    rec_type, target_id, target_version, result_id, result_version
                )

                similarity_score = self._compute_composite_score(target_payload, result, target_skills, 0.0, search_collection)

                llm_score = self._lookup_llm_score(
                    candidate_id_for_key, candidate_version_for_key,
                    job_id_for_key, job_version_for_key,
                    result_version, force_refresh
                )

                scored_results.append({
                    "id": result_id,
                    "similarity_score": similarity_score,
                    "llm_score": llm_score,
                })
            scored_results.sort(key=lambda x: x["similarity_score"], reverse=True)
            output = scored_results

            logger.info(
                "Cold‑start recommendations (missing vector)",
                target_id=target_id,
                result_count=len(output),
            )
            return output

        intent_weight, cooccurrence_weight, peer_weight, profile_weight = self._compute_weights(
            recent_searches, recent_clicks, recent_saves, recent_positive_outcomes
        )

        dim = len(profile_vec)
        zero_vec = np.zeros(dim)

        profile_vec_np = np.array(profile_vec)

        if len(recent_searches) >= 3:
            intent_embeddings = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=POOL_RANK_CONCURRENCY) as executor:
                fut_to_search = {
                    executor.submit(self.llm.embed, truncate_to_prompt_cap(search)): search
                    for search in recent_searches
                }
                for future in concurrent.futures.as_completed(fut_to_search):
                    try:
                        intent_embeddings.append(future.result())
                    except Exception as e:
                        logger.warning("Intent embedding failed", error=str(e))
            if intent_embeddings:
                intent_vec = np.mean(intent_embeddings, axis=0)
            else:
                intent_vec = zero_vec
        else:
            intent_vec = zero_vec

        if len(recent_saves) + len(recent_positive_outcomes) >= 2:
            cooc_ids = []
            for click in recent_clicks:
                cooc_ids.append(click["id"])
            cooc_ids.extend(recent_saves)
            cooc_ids.extend(recent_positive_outcomes)
            records = self.qdrant._client.retrieve(
                collection_name=search_collection,
                ids=cooc_ids,
                with_vectors=True,
            )
            cooc_vectors = [r.vector for r in records if r.vector is not None]
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
        peer_ids = [p["_point_id"] for p in peer_search_results if p["_point_id"] != target_id][:5]
        peer_vectors = []
        if peer_ids:
            records = self.qdrant._client.retrieve(
                collection_name=target_collection,
                ids=peer_ids,
                with_vectors=True,
            )
            peer_vectors = [r.vector for r in records if r.vector is not None]
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

        filter_obj = self._build_filter(hard_filters)

        effective_limit = min(limit, 50)
        raw_results = self.qdrant.search(
            collection=search_collection,
            query_vector=query_vec,
            limit=effective_limit * 3,
            query_filter=filter_obj,
        )


        for result in raw_results:
            final_score = self._compute_composite_score(target_payload, result, target_skills, result["score"], search_collection)
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
            if len(selected) >= max(0, effective_limit - 2):
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

            candidate_id_for_key, candidate_version_for_key, job_id_for_key, job_version_for_key = self._resolve_cache_ids(
                rec_type, target_id, target_version, result_id, result_version
            )

            llm_score = self._lookup_llm_score(
                candidate_id_for_key, candidate_version_for_key,
                job_id_for_key, job_version_for_key,
                result_version, force_refresh
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

    def rank_pool(
        self,
        job_id: int,
        job_version: int,
        candidate_ids: list[int],
        force_refresh: bool = False,
    ) -> list[dict]:
        job_payload = self.qdrant.get(JOBS_COLLECTION, job_id)
        if job_payload is MISSING or job_payload is None:
            logger.warning(
                "Job not found in vector store",
                job_id=job_id,
                collection=JOBS_COLLECTION,
            )
            raise ValueError(f"Job not found: {job_id}")

        stored_version = job_payload.get("job_version")
        if stored_version is not None and stored_version != job_version:
            logger.warning(
                "Stale job version supplied",
                job_id=job_id,
                supplied_version=job_version,
                stored_version=stored_version,
            )
            raise ValueError(
                f"Job version mismatch: supplied {job_version}, stored {stored_version}"
            )

        scoring_service = ScoringService(
            llm=self.llm,
            qdrant=self.qdrant,
            cache=self.cache,
        )
        candidate_versions = {}
        missing_ids = []
        for candidate_id in candidate_ids:
            payload = self.qdrant.get(CANDIDATES_COLLECTION, candidate_id)
            if payload is MISSING or payload is None:
                missing_ids.append(candidate_id)
            else:
                candidate_versions[candidate_id] = payload.get("candidate_version", 1)
        if missing_ids:
            logger.warning(
                "Candidate(s) not found in vector store, rejecting request",
                missing_ids=missing_ids,
            )
            raise ValueError(
                f"Candidate(s) not found in vector store: {missing_ids}"
            )

        results = []
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=POOL_RANK_CONCURRENCY)
        try:
            future_to_candidate = {}
            for candidate_id, candidate_version in candidate_versions.items():
                future = executor.submit(
                    scoring_service.calculate_fit,
                    candidate_id,
                    candidate_version,
                    job_id,
                    job_version,
                    force_refresh,
                )
                future_to_candidate[future] = candidate_id

            try:
                effective_timeout = POOL_RANK_TIMEOUT_SECONDS * max(
                    1, math.ceil(len(candidate_ids) / POOL_RANK_CONCURRENCY)
                )
                done, not_done = wait(
                    future_to_candidate.keys(),
                    timeout=effective_timeout,
                    return_when=concurrent.futures.ALL_COMPLETED
                )
            except Exception:
                for future in future_to_candidate:
                    future.cancel()
                raise

            for future in done:
                candidate_id = future_to_candidate[future]
                try:
                    fit_result = future.result()
                    results.append({
                        "candidate_id": candidate_id,
                        "fit_score": fit_result["overall_score_percentage"],
                        "status": "scored",
                    })
                except LLMUnavailableError:
                    for f in not_done:
                        f.cancel()
                    raise
                except Exception as e:
                    logger.warning(
                        "Candidate scoring failed",
                        candidate_id=candidate_id,
                        error=str(e),
                    )
                    results.append({
                        "candidate_id": candidate_id,
                        "fit_score": POOL_RANK_DEFAULT_FIT_SCORE,
                        "status": "failed",
                    })

            if not_done:
                logger.warning(
                    "Candidate scoring global timeout",
                    candidate_ids=[future_to_candidate[f] for f in not_done],
                )
                for future in not_done:
                    candidate_id = future_to_candidate[future]
                    future.cancel()
                    results.append({
                        "candidate_id": candidate_id,
                        "fit_score": POOL_RANK_DEFAULT_FIT_SCORE,
                        "status": "timeout",
                    })
        finally:
            executor.shutdown(wait=False)

        results.sort(key=lambda r: r["fit_score"], reverse=True)
        logger.info(
            "Pool ranking completed",
            job_id=job_id,
            job_version=job_version,
            result_count=len(results),
        )
        return results
