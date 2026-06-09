import hmac
from typing import Optional
from fastapi import Depends, HTTPException, Request
from fastapi.security import APIKeyHeader
from app.config import get_settings, Settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(
    request: Request,
    api_key: Optional[str] = Depends(api_key_header),
    settings: Settings = Depends(get_settings),
) -> None:
    provided_key = api_key
    expected_key = settings.API_KEY

    if (
        provided_key is None
        or not hmac.compare_digest(provided_key.encode(), expected_key.encode())
    ):
        raise HTTPException(
            status_code=401,
            detail="Unauthorized",
        )
