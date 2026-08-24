import concurrent.futures
import math
import numpy as np
import structlog
from concurrent.futures import wait
from typing import Literal, Optional

import qdrant_client.http.models as qdrant_models

from app.clients.dependencies import CANDIDATES_COLLECTION, JOBS_COLLECTION
from app.clients.llm import LLMClient
from app.clients.qdrant import QdrantClient, MISSING
from app.clients.cache import CacheBackend
from app.clients import skill_vector_cache
from app.config import get_settings
from app.constants import EXPERIENCE_LEVEL_LADDER, canonicalize_experience_level
from app.services.scoring_service import ScoringService, resolve_work_mode, _extract_employment_category
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
    def _skill_cosine(vec_a, vec_b) -> float:
        """Cosine similarity between two skills_vector payloads. Missing -> 0.0."""
        if vec_a is None or vec_b is None:
            return 0.0
        a = np.asarray(vec_a, dtype=np.float32)
        b = np.asarray(vec_b, dtype=np.float32)
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0.0 or nb == 0.0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    @staticmethod
    def _skill_overlap_scaled(cosine: float, settings) -> float:
        """Rescale dense cosine into the skill-term range (debate F-6)."""
        lo = settings.RECOMMEND_SKILL_RESCALE_LO
        hi = settings.RECOMMEND_SKILL_RESCALE_HI
        if hi <= lo:
            return 0.0
        return max(0.0, min(1.0, (cosine - lo) / (hi - lo)))

    def _per_skill_score(self, target_payload: dict, result: dict, search_collection: str) -> Optional[float]:
        if search_collection == JOBS_COLLECTION:
            target_skills_raw = target_payload.get("skills") or []
            result_skills_raw = result.get("required_skills") or []
        elif search_collection == CANDIDATES_COLLECTION:
            target_skills_raw = target_payload.get("required_skills") or []
            result_skills_raw = result.get("skills") or []
        else:
            target_skills_raw = target_payload.get("skills") or target_payload.get("required_skills") or []
            result_skills_raw = result.get("skills") or result.get("required_skills") or []

        if not isinstance(target_skills_raw, list):
            target_skills_raw = []
        if not isinstance(result_skills_raw, list):
            result_skills_raw = []

        target_skills = [str(s).strip() for s in target_skills_raw if s is not None and str(s).strip()]
        result_skills = [str(s).strip() for s in result_skills_raw if s is not None and str(s).strip()]

        target_vecs = []
        for skill in target_skills:
            vec = skill_vector_cache.lookup(skill)
            if vec is not None:
                target_vecs.append(vec)
        result_vecs = []
        for skill in result_skills:
            vec = skill_vector_cache.lookup(skill)
            if vec is not None:
                result_vecs.append(vec)

        if not target_vecs or not result_vecs:
            return None

        try:
            T = np.asarray(target_vecs, dtype=np.float32)
            R = np.asarray(result_vecs, dtype=np.float32)
            T_norms = np.linalg.norm(T, axis=1, keepdims=True)
            R_norms = np.linalg.norm(R, axis=1, keepdims=True)
            T_norms = np.where(T_norms == 0, 1, T_norms)
            R_norms = np.where(R_norms == 0, 1, R_norms)
            T_unit = T / T_norms
            R_unit = R / R_norms
            cos_matrix = np.dot(T_unit, R_unit.T)
            cand_recall = float(np.mean(np.max(cos_matrix, axis=1)))
            job_recall = float(np.mean(np.max(cos_matrix, axis=0)))
            denom = cand_recall + job_recall
            if denom == 0:
                return 0.0
            return float(2 * cand_recall * job_recall / denom)
        except Exception:
            return None

    @staticmethod
    def _canonicalize_level(level: str) -> str:
        return canonicalize_experience_level(level)

    def _build_filter(self, hard_filters: dict):
        normalized = dict(hard_filters)
        if "experience_level" in normalized:
            normalized["experience_level"] = canonicalize_experience_level(normalized["experience_level"])
        if "employment_type" in normalized:
            normalized["employment_type"] = _extract_employment_category(normalized["employment_type"])
        conditions = [
            qdrant_models.FieldCondition(key=k, match=qdrant_models.MatchValue(value=normalized[k]))
            for k in ("location", "experience_level", "industry", "employment_type")
            if k in normalized
        ]
        return qdrant_models.Filter(must=conditions) if conditions else None

    def _search_with_filter_retry(self, collection, query_vector, limit, filter_obj, hard_filters):
        return self.qdrant.search(
            collection=collection,
            query_vector=query_vector,
            limit=limit,
            query_filter=filter_obj,
        )

    def _scroll_with_filter_retry(self, collection, limit, filter_obj, hard_filters):
        return self.qdrant.scroll(
            collection=collection,
            query_filter=filter_obj,
            limit=limit,
        )

    def _compute_composite_score(self, target_payload, result, vector_score, collection=None):
        settings = get_settings()
        # semantic skill overlap: cosine of skills_vector payloads (missing -> 0.0)
        skill_cosine = self._skill_cosine(
            target_payload.get("skills_vector"),
            result.get("skills_vector"),
        )
        skill_overlap = self._skill_overlap_scaled(skill_cosine, settings)
        target_city = target_payload.get("location", "").split(",")[0].strip().lower()
        result_city = result.get("location", "").split(",")[0].strip().lower()
        target_remote = resolve_work_mode(target_payload) in ("remote", "hybrid")
        result_remote = resolve_work_mode(result) in ("remote", "hybrid")
        location_match = 1.0 if (target_city and target_city == result_city) else (0.5 if (target_remote or result_remote) else 0.0)
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
        employment_match = 1.0 if _extract_employment_category(target_payload.get("employment_type", "")) == _extract_employment_category(result.get("employment_type", "")) else 0.0
        # window-free: raw cosine directly, never zeroed
        return (
            settings.RECOMMEND_WEIGHT_VECTOR * vector_score
            + settings.RECOMMEND_WEIGHT_SKILL * skill_overlap
            + settings.RECOMMEND_WEIGHT_LOCATION * location_match
            + settings.RECOMMEND_WEIGHT_LEVEL * level_match
            + settings.RECOMMEND_WEIGHT_EMPLOYMENT * employment_match
        )

    def _compute_weights(self, recent_searches, recent_clicks, recent_saves, recent_positive_outcomes):
        intent_signals = len(recent_searches) + len(recent_clicks) + len(recent_saves) + len(recent_positive_outcomes)
        intent_weight = min(0.45, 0.10 + 0.05 * intent_signals) if len(recent_searches) >= 1 else 0.0
        cooccurrence_signals = len(recent_saves) + len(recent_positive_outcomes) + len(recent_clicks)
        cooccurrence_weight = min(0.20, 0.05 * cooccurrence_signals) if cooccurrence_signals >= 1 else 0.0
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
        limit: int = 10,
    ) -> list[dict]:
        if behavioral_signals is None:
            behavioral_signals = {}
        if hard_filters is None:
            hard_filters = {}

        recent_searches = behavioral_signals.get("recent_searches", []) or []
        recent_clicks = behavioral_signals.get("recent_clicks", []) or []
        recent_saves = behavioral_signals.get("recent_saves", []) or []
        recent_positive_outcomes = behavioral_signals.get("recent_positive_outcomes", []) or []

        # --- behavioral signal caps (defense in depth) ---
        settings = get_settings()
        max_searches = settings.RECOMMEND_MAX_SEARCHES
        max_cooc = settings.RECOMMEND_MAX_COOC_IDS
        dropped_signals = 0

        if len(recent_searches) > max_searches:
            dropped = len(recent_searches) - max_searches
            dropped_signals += dropped
            recent_searches = recent_searches[-max_searches:]
            logger.info("Truncated behavioral signals", dropped_signals=dropped, field="recent_searches", kept=len(recent_searches))

        total_cooc = len(recent_clicks) + len(recent_saves) + len(recent_positive_outcomes)
        if total_cooc > max_cooc:
            dropped_cooc = total_cooc - max_cooc
            dropped_signals += dropped_cooc
            # Keep most recent cooc ids across the three lists (combined tail).
            # Build ordered combined sequence clicks -> saves -> positive_outcomes, keep last max_cooc.
            # Reconstruct truncated lists preserving relative order/tail.
            # clicks are dicts {"id": int}, saves/pos are ints.
            combined = []
            for c in recent_clicks:
                combined.append(("click", c))
            for s in recent_saves:
                combined.append(("save", s))
            for p in recent_positive_outcomes:
                combined.append(("pos", p))
            kept = combined[-max_cooc:]
            new_clicks = []
            new_saves = []
            new_pos = []
            for kind, val in kept:
                if kind == "click":
                    new_clicks.append(val)
                elif kind == "save":
                    new_saves.append(val)
                else:
                    new_pos.append(val)
            recent_clicks = new_clicks
            recent_saves = new_saves
            recent_positive_outcomes = new_pos
            logger.info("Truncated behavioral signals", dropped_signals=dropped_cooc, field="cooc_ids", kept=len(kept))

        if dropped_signals:
            logger.info("Behavioral signals truncated", dropped_signals=dropped_signals, max_searches=max_searches, max_cooc_ids=max_cooc)

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
            limit=limit,
        )

        if rec_type == "jobs":
            search_collection = JOBS_COLLECTION
            target_collection = CANDIDATES_COLLECTION
        else:
            search_collection = CANDIDATES_COLLECTION
            target_collection = JOBS_COLLECTION

        target_payload, profile_vec = self.qdrant.get_with_vector(target_collection, target_id)
        if target_payload is MISSING or target_payload is None:
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
                "Target profile vector is missing - falling back to cold-start",
                target_id=target_id,
                collection=target_collection,
            )

        if profile_vec is None:
            filter_obj = self._build_filter(hard_filters)

            effective_limit = min(limit, 50)

            intent_vec = None
            if len(recent_searches) >= 1:
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

            if intent_vec is not None:
                raw_results = self._search_with_filter_retry(
                    search_collection, intent_vec.tolist(), effective_limit * 5, filter_obj, hard_filters
                )
            else:
                raw_results = self._scroll_with_filter_retry(
                    search_collection, effective_limit * 5, filter_obj, hard_filters
                )

            fetched = len(raw_results)
            raw_gated = 0  # not applied on cold-start by design
            skill_gated = 0
            level_gated = 0
            extraction_degraded = 0

            scored_results = []
            for result in raw_results:
                result_id = result["_point_id"]
                vector_score = result.get("score", 0.0) or 0.0

                similarity_score = self._compute_composite_score(
                    target_payload, result, vector_score, search_collection
                )

                # --- gates (cold-start: SKILL + LEVEL only, no RAW) ---
                target_vec = target_payload.get("skills_vector")
                result_vec = result.get("skills_vector")
                target_has_vec = target_vec is not None
                result_has_vec = result_vec is not None
                ps = self._per_skill_score(target_payload, result, search_collection)
                if ps is not None:
                    if ps < settings.SKILL_PER_SKILL_GATE:
                        skill_gated += 1
                        logger.debug("SKILL_GATE drop", per_skill_score=ps, target_id=target_id, result_id=result_id, gate=settings.SKILL_PER_SKILL_GATE)
                        continue
                else:
                    skill_cosine = self._skill_cosine(target_vec, result_vec)
                    if target_has_vec and result_has_vec and skill_cosine < settings.SKILL_COSINE_GATE:
                        skill_gated += 1
                        logger.debug("SKILL_GATE drop", raw_cosine=vector_score, skill_cosine=skill_cosine, target_id=target_id, result_id=result_id)
                        continue
                    extraction_degraded += 1

                # level gate
                target_level_raw = target_payload.get("experience_level", "") or ""
                result_level_raw = result.get("experience_level", "") or ""
                target_canon = self._canonicalize_level(target_level_raw.lower().strip().replace("-", " "))
                result_canon = self._canonicalize_level(result_level_raw.lower().strip().replace("-", " "))
                target_in = target_canon in EXPERIENCE_LEVEL_LADDER
                result_in = result_canon in EXPERIENCE_LEVEL_LADDER
                if target_in and result_in:
                    idx_a = EXPERIENCE_LEVEL_LADDER.index(target_canon)
                    idx_b = EXPERIENCE_LEVEL_LADDER.index(result_canon)
                    if abs(idx_a - idx_b) > settings.LEVEL_GATE_DISTANCE:
                        level_gated += 1
                        logger.debug("LEVEL_GATE drop", target_id=target_id, result_id=result_id, target_level=target_canon, result_level=result_canon, distance=abs(idx_a - idx_b))
                        continue
                else:
                    # canonicalization fallback: no gate, but log for observability
                    if not target_in or not result_in:
                        logger.debug("LEVEL_GATE fallback", target_id=target_id, result_id=result_id, target_level_raw=target_level_raw, result_level_raw=result_level_raw, target_canon=target_canon, result_canon=result_canon)

                scored_results.append({
                    "id": result_id,
                    "similarity_score": similarity_score,
                    "_raw": result,
                })

            # gate telemetry: only log once per request; the final line (below) carries the real returned count
            if not scored_results:
                logger.info("Recommendation gates", target_id=target_id, fetched=fetched, raw_gated=raw_gated, skill_gated=skill_gated, level_gated=level_gated, extraction_degraded=extraction_degraded, returned=0)
                logger.info("No cold-start results after gates", target_id=target_id)
                return []
            scored_results.sort(key=lambda x: x["similarity_score"], reverse=True)

            def _cold_cluster_key(x):
                r = x["_raw"]
                if rec_type == "jobs":
                    t = (r.get("title") or "").lower().strip()
                    loc = (r.get("location") or "").lower().strip()
                    comp = r.get("company_id") or r.get("company_name") or r.get("company") or str(r.get("_point_id",""))
                    return (t, loc, str(comp).lower().strip())
                return ((r.get("name") or "").lower().strip(), (r.get("location") or "").lower().strip())
            clusters = {}
            for x in scored_results:
                k = _cold_cluster_key(x)
                if k not in clusters or x["similarity_score"] > clusters[k]["similarity_score"]:
                    clusters[k] = x
            scored_results = list(clusters.values())
            scored_results.sort(key=lambda x: x["similarity_score"], reverse=True)
            scored_results = scored_results[:effective_limit]
            output = [{"id": x["id"], "similarity_score": x["similarity_score"]} for x in scored_results]

            logger.info(
                "Cold-start recommendations (missing vector)",
                target_id=target_id,
                result_count=len(output),
            )
            # single gates telemetry line per request (cold path), returned = actual response length
            logger.info("Recommendation gates", target_id=target_id, fetched=fetched, raw_gated=raw_gated, skill_gated=skill_gated, level_gated=level_gated, extraction_degraded=extraction_degraded, returned=len(output))
            return output

        intent_weight, cooccurrence_weight, peer_weight, profile_weight = self._compute_weights(
            recent_searches, recent_clicks, recent_saves, recent_positive_outcomes
        )

        dim = len(profile_vec)
        zero_vec = np.zeros(dim)

        profile_vec_np = np.array(profile_vec)

        if len(recent_searches) >= 1:
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

        if len(recent_saves) + len(recent_positive_outcomes) + len(recent_clicks) >= 1:
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
        raw_results = self._search_with_filter_retry(
            search_collection, query_vec, effective_limit * 5, filter_obj, hard_filters
        )

        fetched = len(raw_results)
        raw_gated = 0
        skill_gated = 0
        level_gated = 0
        extraction_degraded = 0

        gated_results = []
        for result in raw_results:
            vector_score = result.get("score", 0.0) or 0.0
            # compute composite before gates (gates run after composite, before sort)
            final_score = self._compute_composite_score(target_payload, result, vector_score, search_collection)

            # --- RAW gate ---
            if vector_score < settings.RAW_COSINE_GATE:
                raw_gated += 1
                logger.debug("RAW_GATE drop", target_id=target_id, result_id=result.get("_point_id"), raw_cosine=vector_score, gate=settings.RAW_COSINE_GATE)
                continue

            # --- SKILL gate (three-state, per-skill F1 + fallback) ---
            target_vec = target_payload.get("skills_vector")
            result_vec = result.get("skills_vector")
            target_has_vec = target_vec is not None
            result_has_vec = result_vec is not None
            ps = self._per_skill_score(target_payload, result, search_collection)
            if ps is not None:
                if ps < settings.SKILL_PER_SKILL_GATE:
                    skill_gated += 1
                    logger.debug("SKILL_GATE drop", per_skill_score=ps, target_id=target_id, result_id=result.get("_point_id"), gate=settings.SKILL_PER_SKILL_GATE)
                    continue
            else:
                skill_cosine = self._skill_cosine(target_vec, result_vec)
                if target_has_vec and result_has_vec and skill_cosine < settings.SKILL_COSINE_GATE:
                    skill_gated += 1
                    logger.debug("SKILL_GATE drop", raw_cosine=vector_score, skill_cosine=skill_cosine, target_id=target_id, result_id=result.get("_point_id"))
                    continue
                extraction_degraded += 1

            # --- LEVEL gate ---
            target_level_raw = target_payload.get("experience_level", "") or ""
            result_level_raw = result.get("experience_level", "") or ""
            target_canon = self._canonicalize_level(target_level_raw.lower().strip().replace("-", " "))
            result_canon = self._canonicalize_level(result_level_raw.lower().strip().replace("-", " "))
            target_in = target_canon in EXPERIENCE_LEVEL_LADDER
            result_in = result_canon in EXPERIENCE_LEVEL_LADDER
            if target_in and result_in:
                idx_a = EXPERIENCE_LEVEL_LADDER.index(target_canon)
                idx_b = EXPERIENCE_LEVEL_LADDER.index(result_canon)
                if abs(idx_a - idx_b) > settings.LEVEL_GATE_DISTANCE:
                    level_gated += 1
                    logger.debug("LEVEL_GATE drop", target_id=target_id, result_id=result.get("_point_id"), target_level=target_canon, result_level=result_canon, distance=abs(idx_a - idx_b))
                    continue
            else:
                if not target_in or not result_in:
                    logger.debug("LEVEL_GATE fallback", target_id=target_id, result_id=result.get("_point_id"), target_level_raw=target_level_raw, result_level_raw=result_level_raw, target_canon=target_canon, result_canon=result_canon)

            result["final_score"] = final_score
            gated_results.append(result)

        raw_results = gated_results

        if not raw_results:
            logger.info("Recommendation gates", target_id=target_id, fetched=fetched, raw_gated=raw_gated, skill_gated=skill_gated, level_gated=level_gated, extraction_degraded=extraction_degraded, returned=0)
            logger.info("No results after gates", target_id=target_id, fetched=fetched, raw_gated=raw_gated, skill_gated=skill_gated, level_gated=level_gated)
            return []
        raw_results.sort(key=lambda r: r["final_score"], reverse=True)

        def cluster_key(result):
            if rec_type == "jobs":
                title_norm = (result.get("title") or "").lower().strip()
                loc_norm = (result.get("location") or "").lower().strip()
                company = result.get("company_id")
                if company is None:
                    company = result.get("company_name")
                if company is None:
                    company = result.get("company")
                if company and str(company).strip():
                    return (title_norm, loc_norm, str(company).lower().strip())
                return (title_norm, loc_norm, str(result.get("_point_id", "")))
            else:
                return ((result.get("name") or "").lower().strip(), (result.get("location") or "").lower().strip())

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

            output.append({
                "id": result_id,
                "similarity_score": similarity_score,
            })

        logger.info(
            "Recommendations generated",
            target_id=target_id,
            rec_type=rec_type,
            result_count=len(output),
        )
        # single gates telemetry line per request, returned = actual response length
        logger.info("Recommendation gates", target_id=target_id, fetched=fetched, raw_gated=raw_gated, skill_gated=skill_gated, level_gated=level_gated, extraction_degraded=extraction_degraded, returned=len(output))
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
