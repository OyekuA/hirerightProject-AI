import json
import structlog
from typing import Optional

from app.clients.dependencies import CANDIDATES_COLLECTION, JOBS_COLLECTION
from app.clients.llm import LLMClient, LLMUnavailableError
from app.clients.qdrant import QdrantClient, MISSING
from app.clients.cache import CacheBackend
from app.config import get_settings
from app.constants import EMPLOYMENT_TYPES, EXPERIENCE_LEVEL_LADDER, canonicalize_experience_level
from app.utils.ingestion import truncate_to_prompt_cap
from app.utils import parse_llm_json
from app.utils.bias_masking import mask_candidate_for_scoring
from app.prompts import SCORING_FIT_PROMPT_TEMPLATE

logger = structlog.get_logger()


def _derive_status(score: int, pass_threshold: Optional[int] = None, warning_threshold: Optional[int] = None) -> str:
    if pass_threshold is None or warning_threshold is None:
        settings = get_settings()
        if pass_threshold is None:
            pass_threshold = settings.SCORING_STATUS_PASS_THRESHOLD
        if warning_threshold is None:
            warning_threshold = settings.SCORING_STATUS_WARNING_THRESHOLD
    if score >= pass_threshold:
        return "pass"
    elif score >= warning_threshold:
        return "warning"
    else:
        return "fail"


_EMPLOYMENT_CATEGORIES = set(EMPLOYMENT_TYPES)

_WORK_ARRANGEMENT_KEYWORDS = ["remote", "hybrid", "on-site", "onsite", "in-office", "in office"]


def _extract_employment_category(raw: str) -> str:
    lowered = raw.lower().strip()
    original = lowered
    for kw in _WORK_ARRANGEMENT_KEYWORDS:
        lowered = lowered.replace(kw.replace("-", " "), "").replace(kw, "").strip()
    for ch in ("/", ",", ";", "-"):
        lowered = lowered.replace(ch, " ")
    lowered = lowered.replace("full time", "full_time")
    lowered = lowered.replace("part time", "part_time")
    tokens = lowered.split()
    for token in tokens:
        token = token.strip()
        if token in ("ft", "fulltime"):
            return "full_time"
        if token in ("pt", "parttime"):
            return "part_time"
        if token == "intern":
            return "internship"
        if token == "freelance":
            return "freelance"
        if token in _EMPLOYMENT_CATEGORIES:
            return token
    if not tokens:
        if any(kw in original for kw in _WORK_ARRANGEMENT_KEYWORDS):
            return _extract_work_arrangement(raw)
        return ""
    return raw.lower().strip()


def _extract_work_arrangement(raw: str) -> str:
    lowered = raw.lower().strip()
    if "remote" in lowered:
        return "remote"
    if "hybrid" in lowered:
        return "hybrid"
    return "onsite"


def resolve_work_mode(payload: dict) -> Optional[str]:
    explicit = (payload.get("work_mode") or "").strip().lower()
    if explicit in ("onsite", "on-site", "on site", "in-office", "in office"):
        return "onsite"
    if explicit in ("remote", "hybrid"):
        return explicit
    return _extract_work_arrangement(payload.get("employment_type") or "")


def _arrangement_compatible(cand_arr: str, job_arr: str) -> bool:
    if cand_arr in ("onsite", "hybrid"):
        return True
    return cand_arr == "remote" and job_arr == "remote"


def _compute_deterministic_dimensions(candidate_payload: dict, job_payload: dict) -> dict:
    settings = get_settings()
    pass_threshold = settings.SCORING_STATUS_PASS_THRESHOLD
    warning_threshold = settings.SCORING_STATUS_WARNING_THRESHOLD

    candidate_level = (candidate_payload.get("experience_level") or "").strip()
    job_level = (job_payload.get("experience_level") or "").strip()

    if candidate_level and job_level:
        cand_canon = canonicalize_experience_level(candidate_level)
        job_canon = canonicalize_experience_level(job_level)

        if cand_canon in EXPERIENCE_LEVEL_LADDER and job_canon in EXPERIENCE_LEVEL_LADDER:
            idx_cand = EXPERIENCE_LEVEL_LADDER.index(cand_canon)
            idx_job = EXPERIENCE_LEVEL_LADDER.index(job_canon)
            distance = abs(idx_cand - idx_job)
            exp_score = max(0, 100 - distance * 25)
            if exp_score >= pass_threshold:
                exp_status = "pass"
                exp_reason = f"Experience level '{cand_canon}' aligns with job requirement '{job_canon}'."
            elif exp_score >= warning_threshold:
                exp_status = "warning"
                exp_reason = f"Experience level '{cand_canon}' partially aligns with '{job_canon}'."
            else:
                exp_status = "fail"
                exp_reason = f"Experience level '{cand_canon}' insufficient for '{job_canon}'."
        else:
            exp_score = 50
            exp_status = "warning"
            if cand_canon == job_canon:
                exp_reason = f"Experience level '{cand_canon}' matches '{job_canon}' but level alignment cannot be verified against known levels."
            else:
                exp_reason = f"Experience level '{cand_canon}' partially aligns with '{job_canon}'."
    else:
        exp_score = 50
        exp_status = "warning"
        exp_reason = "Insufficient experience level data; neutral score assigned."

    cand_location = (candidate_payload.get("location") or "").strip()
    job_location = (job_payload.get("location") or "").strip()
    cand_arrangement = resolve_work_mode(candidate_payload)
    job_arrangement = resolve_work_mode(job_payload)
    cand_remote_ok = cand_arrangement in ("remote", "hybrid")
    job_is_remote = job_arrangement == "remote"
    job_is_hybrid = job_arrangement == "hybrid"

    if job_is_remote and cand_remote_ok:
        loc_score = 100
        loc_status = "pass"
        loc_reason = "Job is remote; candidate open to remote arrangement."
    elif job_is_hybrid and cand_remote_ok:
        loc_score = 100
        loc_status = "pass"
        loc_reason = "Job is hybrid; candidate open to flexible arrangement."
    elif cand_location and job_location:
        same_city = cand_location.split(",")[0].strip().lower() == job_location.split(",")[0].strip().lower()
        same_country = cand_location.split(",")[-1].strip().lower() == job_location.split(",")[-1].strip().lower()
        job_onsite_only = not job_is_remote and not job_is_hybrid

        if same_city:
            loc_score = 100
            loc_status = "pass"
            loc_reason = "Same city as job location."
        elif same_country and not job_onsite_only:
            loc_score = 60
            loc_status = "warning"
            loc_reason = "Same country, different city; job does not require on-site exclusively."
        else:
            loc_score = 20
            loc_status = "fail"
            loc_reason = "Location mismatch and job requires on-site or incompatible arrangement."
    else:
        loc_score = 50
        loc_status = "warning"
        loc_reason = "Insufficient location data; neutral score assigned."

    cand_emp = (candidate_payload.get("employment_type") or "").lower().strip()
    job_emp = (job_payload.get("employment_type") or "").lower().strip()

    if cand_emp and job_emp:
        cand_category = _extract_employment_category(cand_emp)
        job_category = _extract_employment_category(job_emp)
        cand_arr = resolve_work_mode(candidate_payload)
        job_arr = resolve_work_mode(job_payload)

        if cand_category == job_category:
            if _arrangement_compatible(cand_arr, job_arr):
                emp_score = 100
                emp_status = "pass"
                emp_reason = f"Employment type '{cand_emp}' matches requirement."
            else:
                emp_score = 60
                emp_status = "warning"
                emp_reason = f"Candidate arrangement '{cand_arr}' cannot satisfy job arrangement '{job_arr}'."
        elif cand_category in ("remote", "hybrid", "onsite") or job_category in ("remote", "hybrid", "onsite"):
            if _arrangement_compatible(cand_arr, job_arr):
                emp_score = 100
                emp_status = "pass"
                emp_reason = f"Employment arrangement '{cand_arr}' satisfies requirement."
            else:
                emp_score = 60
                emp_status = "warning"
                emp_reason = f"Employment category unspecified on one side; arrangement '{cand_arr}' cannot satisfy '{job_arr}'."
        else:
            emp_score = 20
            emp_status = "fail"
            emp_reason = f"Employment category '{cand_category}' incompatible with '{job_category}'."
    else:
        emp_score = 50
        emp_status = "warning"
        emp_reason = "Insufficient employment type data; neutral score assigned."

    return {
        "experience": {"score": exp_score, "status": exp_status, "short_reason": exp_reason},
        "location": {"score": loc_score, "status": loc_status, "short_reason": loc_reason},
        "employment_type": {"score": emp_score, "status": emp_status, "short_reason": emp_reason},
    }


def _is_new_format_cached(cached: dict) -> bool:
    cat = cached.get("category_breakdown")
    if not isinstance(cat, dict):
        return False
    if "skills" not in cat:
        return False
    for key in ("skills", "role_match", "experience", "location", "employment_type"):
        dim = cat.get(key)
        if not isinstance(dim, dict) or "score" not in dim:
            return False
    return True


class ScoringService:

    def __init__(
        self,
        llm: LLMClient,
        qdrant: QdrantClient,
        cache: CacheBackend,
    ):
        self.llm = llm
        self.qdrant = qdrant
        self.cache = cache

    def _run_scoring(self, prompt: str) -> dict:
        settings = get_settings()
        generated = self.llm.generate(prompt, temperature=0, seed=settings.LLM_SEED)
        result = parse_llm_json(generated)

        if not isinstance(result, dict):
            raise LLMUnavailableError("LLM returned a non‑dict payload")

        required_top = {"skills", "role_match", "skill_gap_analysis"}
        if not all(k in result for k in required_top):
            raise LLMUnavailableError("LLM response missing required top‑level keys")

        for key in ("skills", "role_match"):
            sub = result.get(key)
            if not isinstance(sub, dict):
                raise LLMUnavailableError(f"LLM response key '{key}' must be a dict")
            if "score" not in sub or "short_reason" not in sub:
                raise LLMUnavailableError(f"LLM response key '{key}' missing 'score' or 'short_reason'")
            try:
                sub["score"] = int(sub["score"])
            except (TypeError, ValueError):
                raise LLMUnavailableError(f"LLM response key '{key}.score' must be an integer")
            if not (0 <= sub["score"] <= 100):
                raise LLMUnavailableError(f"LLM response key '{key}.score' must be 0‑100")

        if not isinstance(result.get("skill_gap_analysis"), str):
            raise LLMUnavailableError("LLM response 'skill_gap_analysis' must be a string")

        return result

    def score_from_payloads(
        self,
        candidate_payload: dict,
        job_payload: dict,
        candidate_id: int,
        candidate_version: int,
        job_id: int,
        job_version: int,
    ) -> dict:
        masked_candidate_payload = mask_candidate_for_scoring(candidate_payload)
        candidate_payload_json = json.dumps(masked_candidate_payload, indent=2)
        # job payload also carries skills_vector (~21KB); strip it before the prompt
        masked_job_payload = mask_candidate_for_scoring(job_payload)
        job_payload_json = json.dumps(masked_job_payload, indent=2)
        prompt = SCORING_FIT_PROMPT_TEMPLATE.format(
            candidate_id=candidate_id,
            candidate_version=candidate_version,
            candidate_payload_json=candidate_payload_json,
            job_id=job_id,
            job_version=job_version,
            job_payload_json=job_payload_json,
        )
        prompt = truncate_to_prompt_cap(prompt)

        llm_result = self._run_scoring(prompt)
        skills = llm_result["skills"]
        role_match = llm_result["role_match"]

        det = _compute_deterministic_dimensions(candidate_payload, job_payload)

        settings = get_settings()
        overall = round(
            settings.SCORING_WEIGHT_SKILLS * skills["score"]
            + settings.SCORING_WEIGHT_ROLE * role_match["score"]
            + settings.SCORING_WEIGHT_EXPERIENCE * det["experience"]["score"]
            + settings.SCORING_WEIGHT_LOCATION * det["location"]["score"]
            + settings.SCORING_WEIGHT_EMPLOYMENT * det["employment_type"]["score"]
        )
        overall = max(0, min(100, overall))

        def _build_entry(score_val: int, status_val: str, reason: str) -> dict:
            return {"score": score_val, "status": status_val, "short_reason": reason}

        category_breakdown = {
            "skills": _build_entry(skills["score"], _derive_status(skills["score"]), skills["short_reason"]),
            "role_match": _build_entry(role_match["score"], _derive_status(role_match["score"]), role_match["short_reason"]),
            "experience": _build_entry(det["experience"]["score"], det["experience"]["status"], det["experience"]["short_reason"]),
            "location": _build_entry(det["location"]["score"], det["location"]["status"], det["location"]["short_reason"]),
            "employment_type": _build_entry(det["employment_type"]["score"], det["employment_type"]["status"], det["employment_type"]["short_reason"]),
        }

        return {
            "overall_score_percentage": overall,
            "category_breakdown": category_breakdown,
            "skill_gap_analysis": llm_result["skill_gap_analysis"],
        }

    def calculate_fit(
        self,
        candidate_id: int,
        candidate_version: int,
        job_id: int,
        job_version: int,
        force_refresh: bool = False,
    ) -> dict:
        cache_key = f"{candidate_id}:{candidate_version}:{job_id}:{job_version}"
        logger.info(
            "Calculating fit score",
            candidate_id=candidate_id,
            candidate_version=candidate_version,
            job_id=job_id,
            job_version=job_version,
            force_refresh=force_refresh,
            cache_key=cache_key,
        )

        if not force_refresh:
            cached = self.cache.get(cache_key)
            if cached is not None:
                if _is_new_format_cached(cached):
                    logger.info("Cache hit", cache_key=cache_key)
                    return cached
                logger.info(
                    "Cache hit but legacy format detected — recomputing",
                    cache_key=cache_key,
                )

        logger.info("Cache miss or forced refresh", cache_key=cache_key)

        candidate_payload = self.qdrant.get(CANDIDATES_COLLECTION, candidate_id)
        if candidate_payload is MISSING or candidate_payload is None:
            logger.warning(
                "Candidate not found in vector store",
                candidate_id=candidate_id,
                collection=CANDIDATES_COLLECTION,
            )
            raise ValueError("Candidate not found")
        stored_candidate_version = candidate_payload.get("candidate_version", 1)
        if stored_candidate_version != candidate_version:
            logger.warning(
                "Candidate version mismatch",
                candidate_id=candidate_id,
                stored_version=stored_candidate_version,
                requested_version=candidate_version,
            )
            raise ValueError(f"Candidate version mismatch: stored {stored_candidate_version}, requested {candidate_version}")

        job_payload = self.qdrant.get(JOBS_COLLECTION, job_id)
        if job_payload is MISSING or job_payload is None:
            logger.warning(
                "Job not found in vector store",
                job_id=job_id,
                collection=JOBS_COLLECTION,
            )
            raise ValueError("Job not found")
        stored_job_version = job_payload.get("job_version", 1)
        if stored_job_version != job_version:
            logger.warning(
                "Job version mismatch",
                job_id=job_id,
                stored_version=stored_job_version,
                requested_version=job_version,
            )
            raise ValueError(f"Job version mismatch: stored {stored_job_version}, requested {job_version}")

        result = self.score_from_payloads(
            candidate_payload=candidate_payload,
            job_payload=job_payload,
            candidate_id=candidate_id,
            candidate_version=candidate_version,
            job_id=job_id,
            job_version=job_version,
        )

        settings = get_settings()
        self.cache.set(cache_key, result, ttl=settings.CACHE_TTL_SECONDS)
        logger.info(
            "Fit score computed and cached",
            candidate_id=candidate_id,
            job_id=job_id,
            overall_score=result["overall_score_percentage"],
        )

        return result
