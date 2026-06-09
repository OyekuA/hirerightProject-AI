import hashlib
import json
from typing import Optional, Union

from fastapi import Request
from slowapi.util import get_remote_address


def _fingerprint(request: Request, entity_id: Optional[str] = None) -> str:
    api_key = request.headers.get("X-API-Key")
    if api_key:
        base = api_key.encode()
    else:
        base = get_remote_address(request).encode()
    if entity_id is not None:
        base += str(entity_id).encode()
    return hashlib.sha256(base).hexdigest()


def _read_body(request: Request) -> Union[dict, None]:
    raw = getattr(request.state, "raw_body", None)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None


def candidate_id_key(request: Request) -> str:
    body = _read_body(request)
    if not isinstance(body, dict):
        return f"unknown:{_fingerprint(request)}"
    candidate_id = body.get("candidate_id")
    if candidate_id is None:
        return f"unknown:{_fingerprint(request)}"
    return f"candidate:{_fingerprint(request, entity_id=str(candidate_id))}"


def job_id_key(request: Request) -> str:
    body = _read_body(request)
    if not isinstance(body, dict):
        return f"unknown:{_fingerprint(request)}"
    job_id = body.get("job_id")
    if job_id is None:
        return f"unknown:{_fingerprint(request)}"
    return f"job:{_fingerprint(request, entity_id=str(job_id))}"


def target_id_key(request: Request) -> str:
    body = _read_body(request)
    if not isinstance(body, dict):
        return f"unknown:{_fingerprint(request)}"
    target_id = body.get("target_id")
    if target_id is None:
        return f"unknown:{_fingerprint(request)}"
    return f"target:{_fingerprint(request, entity_id=str(target_id))}"


def candidate_or_job_id_key(request: Request) -> str:
    body = _read_body(request)
    if not isinstance(body, dict):
        return f"unknown:{_fingerprint(request)}"
    candidate_context = body.get("candidate_context")
    if isinstance(candidate_context, dict) and candidate_context.get("candidate_id") is not None:
        return f"candidate:{_fingerprint(request, entity_id=str(candidate_context['candidate_id']))}"
    job_context = body.get("job_context")
    if isinstance(job_context, dict) and job_context.get("job_id") is not None:
        return f"job:{_fingerprint(request, entity_id=str(job_context['job_id']))}"
    return f"unknown:{_fingerprint(request)}"
