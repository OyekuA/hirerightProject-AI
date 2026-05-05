"""
Integration test — requires live stack, not in CI. Run with: `uv run pytest tests/integration/ -v -s`
"""

import os
import time
import httpx
import pytest
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file (same as the live stack)
env_path = Path(__file__).parent.parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path, override=False)
else:
    print(f"Warning: .env file not found at {env_path}")

pytestmark = pytest.mark.integration

BASE_URL = os.getenv("TEST_BASE_URL", "http://localhost:8000")
# Dedicated integration key, falling back to generic API_KEY for backward compatibility
TEST_API_KEY = os.getenv("TEST_API_KEY")
API_KEY = TEST_API_KEY if TEST_API_KEY is not None else os.getenv("API_KEY", "change-me-in-production")
HEADERS = {"X-API-Key": API_KEY, "Content-Type": "application/json"}
WRONG_HEADERS = {"X-API-Key": "wrong-key", "Content-Type": "application/json"}

# Module-level mutable state dict
STATE = {}


def wait_for_ingestion(client, event_id, timeout=120):
    """Poll ingestion status until success or failure."""
    start = time.time()
    while time.time() - start < timeout:
        resp = client.get(
            f"/api/ai/ingestion-status?event_id={event_id}",
            headers=HEADERS,
        )
        resp.raise_for_status()
        data = resp.json()
        if data["status"] == "success":
            return data
        if data["status"] == "failed":
            raise AssertionError(f"Ingestion failed for event {event_id}: {data.get('error_summary')}")
        time.sleep(5)
    raise AssertionError(f"Timeout waiting for ingestion of event {event_id}")


# Realistic test data constants

CANDIDATES = [
    {
        "candidate_id": 1001,
        "cv_url": "https://raw.githubusercontent.com/dX4CY/scaling-adventure/8b2fcc54d56e17c202463c7f0a2d83e4c3b7ad01/Candidate%201.pdf",
        "profile_data": {
            "name": "Adaeze Okonkwo",
            "location": "Lagos",
            "experience_level": "Senior",
            "industry": "fintech",
            "employment_type": "full-time",
            "candidate_version": 1,
        },
        "callback_url": "https://httpbin.org/post",
    },
    {
        "candidate_id": 1002,
        "cv_url": "https://raw.githubusercontent.com/dX4CY/scaling-adventure/8b2fcc54d56e17c202463c7f0a2d83e4c3b7ad01/Candidate%202.pdf",
        "profile_data": {
            "name": "Emeka Nwosu",
            "location": "Abuja",
            "experience_level": "Mid-level",
            "industry": "AI/ML",
            "employment_type": "remote",
            "candidate_version": 1,
        },
        "callback_url": "https://httpbin.org/post",
    },
{
        "candidate_id": 1003,
        "cv_url": "https://raw.githubusercontent.com/dX4CY/scaling-adventure/79e6461243f832859958689bf296fe7005c986d0/Candidate%203.pdf",
        "profile_data": {
            "name": "Blessing Taiwo",
            "location": "Lagos",
            "experience_level": "Junior",
            "industry": "e-commerce",
            "employment_type": "full-time",
            "candidate_version": 1,
        },
        "callback_url": "https://httpbin.org/post",
    },
]

JOBS = [
    {
        "job_id": 2001,
        "jd_text": ( # Note: In swagger you'd pass jd_text(in a single double quote): "We are looking for a Senior Backend Engineer to join our fintech team in Lagos. You will design and implement scalable microservices, optimize database performance, and mentor junior engineers. Requires 5+ years of experience with Python, FastAPI, and cloud platforms." 
            "We are looking for a Senior Backend Engineer to join our fintech team in Lagos. "
            "You will design and implement scalable microservices, optimize database performance, "
            "and mentor junior engineers. Requires 5+ years of experience with Python, FastAPI, and cloud platforms."
        ),
        "metadata": {
            "title": "Senior Backend Engineer",
            "location": "Lagos",
            "experience_level": "mid level",
            "industry": "fintech",
            "employment_type": "full-time",
            "job_version": 1,
        },
        "callback_url": "https://httpbin.org/post",
    },
    {
        "job_id": 2002,
        "jd_text": (
            "Machine Learning Engineer needed for our AI/ML division in Abuja. "
            "Responsibilities include building and deploying ML models, working with large datasets, "
            "and collaborating with data scientists. Experience with PyTorch, TensorFlow, and MLOps required."
        ),
        "metadata": {
            "title": "Machine Learning Engineer",
            "location": "Abuja",
            "experience_level": "mid level",
            "industry": "AI/ML",
            "employment_type": "remote",
            "job_version": 1,
        },
        "callback_url": "https://httpbin.org/post",
    },
    {
        "job_id": 2003,
        "jd_text": (
            "Join our e-commerce platform as a Frontend Developer in Lagos. "
            "You'll build responsive user interfaces with React, optimize frontend performance, "
            "and work closely with UX designers. Suitable for junior developers with 1‑2 years of experience."
        ),
        "metadata": {
            "title": "Frontend Developer",
            "location": "Lagos",
            "experience_level": "junior",
            "industry": "e-commerce",
            "employment_type": "full-time",
            "job_version": 1,
        },
        "callback_url": "https://httpbin.org/post",
    },
]


def test_01_health_check():
    """GET /health (no auth)."""
    with httpx.Client(base_url=BASE_URL, timeout=30.0) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body == {"status": "ok"}


def test_02_auth_guard():
    """POST /api/ai/calculate-fit with wrong key."""
    with httpx.Client(base_url=BASE_URL, timeout=30.0) as client:
        resp = client.post(
            "/api/ai/calculate-fit",
            headers=WRONG_HEADERS,
            json={"candidate_id": 999, "job_id": 888},
        )
        assert resp.status_code == 401


def test_03_ingest_candidates():
    """Ingest three candidates, wait for each to succeed."""
    with httpx.Client(base_url=BASE_URL, timeout=30.0) as client:
        event_ids = []
        for cand in CANDIDATES:
            headers = {**HEADERS, "X-Candidate-ID": str(cand["candidate_id"])}
            resp = client.post("/api/ai/ingest-candidate", headers=headers, json=cand)
            assert resp.status_code == 202
            data = resp.json()
            assert "event_id" in data
            event_ids.append(data["event_id"])
        STATE["candidate_event_ids"] = event_ids
        # Wait for ingestion completion
        for eid in event_ids:
            record = wait_for_ingestion(client, eid)
            assert record["status"] == "success"


def test_04_ingest_jobs():
    """Ingest three jobs, wait for each to succeed."""
    with httpx.Client(base_url=BASE_URL, timeout=30.0) as client:
        event_ids = []
        for job in JOBS:
            headers = {**HEADERS, "X-Job-ID": str(job["job_id"])}
            resp = client.post("/api/ai/ingest-job", headers=headers, json=job)
            assert resp.status_code == 202
            data = resp.json()
            assert "event_id" in data
            event_ids.append(data["event_id"])
        STATE["job_event_ids"] = event_ids
        for eid in event_ids:
            record = wait_for_ingestion(client, eid)
            assert record["status"] == "success"


def test_05_ingestion_status_pull_fallback():
    """GET /api/ai/ingestion-status with entity_type and entity_id."""
    with httpx.Client(base_url=BASE_URL, timeout=30.0) as client:
        resp = client.get(
            "/api/ai/ingestion-status?entity_type=candidate&entity_id=1001",
            headers=HEADERS,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["entity_id"] == 1001
        assert data["status"] == "success"


def test_06_assessment_generate():
    """POST /api/ai/assessment/generate to create three questions."""
    with httpx.Client(base_url=BASE_URL, timeout=30.0) as client:
        payload = {
            "candidate_context": {
                "candidate_id": 1001,
                "target_role": "Senior Backend Engineer"
            },
            "num_questions": 3,
        }
        headers = {**HEADERS, "X-Candidate-ID": "1001"}
        resp = client.post("/api/ai/assessment/generate", headers=headers, json=payload)
        assert resp.status_code == 200
        data = resp.json()
        questions = data.get("questions", [])
        assert isinstance(questions, list)
        assert len(questions) == 3
        for q in questions:
            assert isinstance(q, str)
            assert len(q) > 10
        STATE["questions"] = questions
        print(f"Generated questions: {questions}")


def test_07_assessment_grade():
    """POST /api/ai/assessment/grade with plausible answers."""
    questions = STATE.get("questions", [])
    assert len(questions) == 3
    answers = [
        "When facing a surge in fraudulent transactions, my first step would be to analyze transaction logs and patterns to identify anomalies such as velocity spikes, geolocation mismatches, or device inconsistencies. I would then design a layered fraud detection system combining rules-based checks with machine learning models for behavioral scoring, integrating real-time risk assessment into the payment flow. Implementation would involve a dedicated fraud detection microservice with low-latency APIs and manual review queues for flagged cases. Throughout, I’d ensure PCI-DSS compliance by encrypting sensitive data, maintaining audit trails, and enforcing strict access controls.",
        "For a new payment routing microservice, I’d prioritize scalability and high availability by designing stateless services deployed on Kubernetes across multiple AWS availability zones. The routing logic would abstract gateway integrations behind a unified interface, supporting dynamic rules for cost optimization, failover, and geo-preference. Integration with existing Go and Spring Boot services would rely on gRPC/REST APIs with idempotency guarantees. Challenges include handling inconsistent gateway APIs and ensuring transactional integrity under retries and failovers. I’d mentor the team through design reviews, CI/CD best practices, and resilience testing, fostering confidence in building a robust, production-ready system.",
        "To address a difficult-to-reproduce bug causing intermittent transaction failures, I’d start by enhancing logging with correlation IDs and simulating concurrency through stress testing to expose race conditions. I’d investigate PostgreSQL isolation levels, lock contention, and connection pooling issues while reviewing async job handling. To minimize downtime, I’d use feature flags, canary releases, and fallback paths to stabilize the system while debugging. Long-term fixes would include circuit breakers, retry backoff strategies, and regression tests for concurrency scenarios. All debugging would respect NDPR compliance by anonymizing logs, documenting incident handling, and maintaining strict data access controls.",
    ]
    with httpx.Client(base_url=BASE_URL, timeout=30.0) as client:
        payload = {
            "questions": questions,
            "answers": answers,
            "time_taken_seconds": 300,
        }
        resp = client.post("/api/ai/assessment/grade", headers=HEADERS, json=payload)
        print(f"Grade response status: {resp.status_code}, body: {resp.text}")
        assert resp.status_code == 200
        data = resp.json()
        assert "overall_score" in data
        assert 0 <= data["overall_score"] <= 100
        assert "authenticity_flag" in data
        flag = data["authenticity_flag"]
        assert "is_suspicious" in flag
        assert isinstance(flag["is_suspicious"], bool)
        assert "reason" in flag
        assert isinstance(flag["reason"], str)


def test_08_fit_score_cache_miss():
    """POST /api/ai/calculate-fit (force_refresh=False) should compute a fresh score."""
    with httpx.Client(base_url=BASE_URL, timeout=30.0) as client:
        payload = {
            "candidate_id": 1001,
            "candidate_version": 1,
            "job_id": 2001,
            "job_version": 1,
            "force_refresh": False,
        }
        headers = {**HEADERS, "X-Candidate-ID": "1001"}
        resp = client.post("/api/ai/calculate-fit", headers=headers, json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "overall_score_percentage" in data
        assert 0 <= data["overall_score_percentage"] <= 100
        assert "category_breakdown" in data
        breakdown = data["category_breakdown"]
        required_keys = {"role_match", "experience", "location", "employment_type"}
        assert all(k in breakdown for k in required_keys)
        STATE["fit_result"] = data


def test_09_fit_score_cache_hit():
    """Identical request should return cached result (identical body)."""
    with httpx.Client(base_url=BASE_URL, timeout=30.0) as client:
        payload = {
            "candidate_id": 1001,
            "candidate_version": 1,
            "job_id": 2001,
            "job_version": 1,
            "force_refresh": False,
        }
        headers = {**HEADERS, "X-Candidate-ID": "1001"}
        resp = client.post("/api/ai/calculate-fit", headers=headers, json=payload)
        assert resp.status_code == 200
        data = resp.json()
        # Compare with stored result
        stored = STATE.get("fit_result")
        assert stored is not None
        assert data == stored


def test_10_recommend_jobs():
    """POST /api/ai/recommend with type=jobs."""
    with httpx.Client(base_url=BASE_URL, timeout=30.0) as client:
        payload = {
            "type": "jobs",
            "target_id": 1001,
            "target_version": 1,
            "behavioral_signals": {},
            "hard_filters": {},
            "limit": 5,
        }
        headers = {**HEADERS, "X-Target-ID": "1001"}
        resp = client.post("/api/ai/recommend", headers=headers, json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data
        results = data["results"]
        assert isinstance(results, list)
        for item in results:
            assert "id" in item
            assert "similarity_score" in item
            assert "llm_score" in item
            print(f"Recommended job: id={item['id']}, similarity={item['similarity_score']:.3f}, llm={item.get('llm_score')}")


def test_11_recommend_candidates():
    """POST /api/ai/recommend with type=candidates."""
    with httpx.Client(base_url=BASE_URL, timeout=30.0) as client:
        payload = {
            "type": "candidates",
            "target_id": 2001,
            "target_version": 1,
            "behavioral_signals": {},
            "hard_filters": {},
            "limit": 5,
        }
        headers = {**HEADERS, "X-Target-ID": "2001"}
        resp = client.post("/api/ai/recommend", headers=headers, json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data
        results = data["results"]
        assert isinstance(results, list)
        for item in results:
            # Expect at least these keys (adjust if needed)
            assert "id" in item
            assert "similarity_score" in item
            assert "llm_score" in item
            print(f"Recommended candidate: id={item['id']}, similarity={item['similarity_score']:.3f}, llm={item.get('llm_score')}")


def test_12_career_paths():
    """POST /api/ai/analyze-career-paths returns three suggested paths."""
    with httpx.Client(base_url=BASE_URL, timeout=30.0) as client:
        payload = {"candidate_id": 1001}
        headers = {**HEADERS, "X-Candidate-ID": "1001"}
        resp = client.post("/api/ai/analyze-career-paths", headers=headers, json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "paths" in data
        assert "profile_summary" in data
        assert isinstance(data["profile_summary"], str)
        assert len(data["profile_summary"]) > 0
        paths = data["paths"]
        print(f"Career paths response: {paths}")
        assert isinstance(paths, list)
        assert len(paths) == 3
        for path in paths:
            assert "role" in path
            assert isinstance(path["role"], str)
            assert "match_percentage" in path
            assert isinstance(path["match_percentage"], int)
            assert 0 <= path["match_percentage"] <= 100
            assert "reasoning" in path
            assert isinstance(path["reasoning"], str)
            print(f"  Path: role={path['role']}, match={path['match_percentage']}%, reasoning={path['reasoning'][:50]}...")


def test_13_generate_jd():
    """POST /api/ai/generate-jd creates a job description."""
    with httpx.Client(base_url=BASE_URL, timeout=30.0) as client:
        payload = {
            "prompt": "Write a JD for a Senior Python Backend Engineer at a fintech startup in Lagos"
        }
        resp = client.post("/api/ai/generate-jd", headers=HEADERS, json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "jd_text" in data
        jd_text = data["jd_text"]
        print(f"Generated JD: {jd_text}")
        assert isinstance(jd_text, str)
        assert len(jd_text) > 0
        STATE["generated_jd"] = jd_text


def test_14_analyze_jd():
    """POST /api/ai/analyze-jd critiques a job description."""
    with httpx.Client(base_url=BASE_URL, timeout=30.0) as client:
        # Use the JD from the previous test, or fall back to a sample
        jd_text = STATE.get("generated_jd") or (
            "Senior Python Backend Engineer needed at fintech startup in Lagos. "
            "Must have 5+ years of experience with FastAPI and cloud platforms."
        )
        print(f"Analyzing JD: {jd_text[:200]}...")
        payload = {"jd_text": jd_text}
        resp = client.post("/api/ai/analyze-jd", headers=HEADERS, json=payload)
        print(f"Response status: {resp.status_code}, body: {resp.text}")
        assert resp.status_code == 200
        data = resp.json()
        assert "critiques" in data
        critiques = data["critiques"]
        print(f"JD critiques: {critiques}")
        assert isinstance(critiques, list)
        assert len(critiques) > 0
        for c in critiques:
            assert isinstance(c, str)


def test_15_delete_candidate_and_verify_404():
    """DELETE /api/ai/candidates/1001 then calculate-fit returns 404."""
    with httpx.Client(base_url=BASE_URL, timeout=30.0) as client:
        # Delete candidate 1001
        resp = client.delete("/api/ai/candidates/1001", headers=HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("deleted") is True
        # Attempt fit calculation with deleted candidate
        payload = {
            "candidate_id": 1001,
            "candidate_version": 1,
            "job_id": 2001,
            "job_version": 1,
            "force_refresh": False,
        }
        resp = client.post("/api/ai/calculate-fit", headers=HEADERS, json=payload)
        assert resp.status_code == 404


def test_16_delete_job_and_verify_404():
    """DELETE /api/ai/jobs/2001 then calculate-fit returns 404."""
    with httpx.Client(base_url=BASE_URL, timeout=30.0) as client:
        # Delete job 2001
        resp = client.delete("/api/ai/jobs/2001", headers=HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("deleted") is True
        # Attempt fit calculation with deleted job (using a different candidate)
        payload = {
            "candidate_id": 1002,
            "candidate_version": 1,
            "job_id": 2001,
            "job_version": 1,
            "force_refresh": False,
        }
        resp = client.post("/api/ai/calculate-fit", headers=HEADERS, json=payload)
        assert resp.status_code == 404

def test_17_pool_rank():
    """POST /api/ai/recommend/pool with ranking."""
    with httpx.Client(base_url=BASE_URL, timeout=30.0) as client:
        payload = {"job_id": 2002, "job_version": 1, "candidate_ids": [1002, 1003]}
        resp = client.post("/api/ai/recommend/pool", headers=HEADERS, json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data
        results = data["results"]
        assert isinstance(results, list)
        for item in results:
            assert "candidate_id" in item
            assert isinstance(item["candidate_id"], int)
            assert "fit_score" in item
            fit_score = item["fit_score"]
            assert isinstance(fit_score, int)
            assert 0 <= fit_score <= 100
        for i in range(len(results) - 1):
            assert results[i]["fit_score"] >= results[i + 1]["fit_score"]