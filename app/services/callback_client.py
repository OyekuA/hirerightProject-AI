import asyncio
import hashlib
import hmac
import json
import time
import structlog
from typing import Optional
import httpx

from app.utils.ingestion import validate_callback_url


logger = structlog.get_logger()


class CallbackClient:

    def __init__(
        self,
        hmac_secret: str,
        max_attempts: int = 3,
        retry_base_seconds: int = 2,
        timeout_seconds: float = 10.0,
        signature_ttl_seconds: int = 300,
    ):
        self.hmac_secret = hmac_secret
        self.max_attempts = max_attempts
        self.retry_base_seconds = retry_base_seconds
        self.timeout_seconds = timeout_seconds
        self.signature_ttl_seconds = signature_ttl_seconds

    def _sign_payload(self, body_bytes: bytes, timestamp: int) -> str:
        message = f"{timestamp}.".encode() + body_bytes
        digest = hmac.new(
            self.hmac_secret.encode(),
            msg=message,
            digestmod=hashlib.sha256,
        ).hexdigest()
        return f"sha256={digest}"

    async def send(
        self,
        callback_url: str,
        event_id: str,
        entity_type: str,
        entity_id: int,
        status: str,
        error: Optional[str] = None,
    ) -> bool:
        base_payload = {
            "event_id": event_id,
            "entity_type": entity_type,
            "entity_id": entity_id,
            "status": status,
            "error": error,
        }

        for attempt in range(self.max_attempts):
            try:
                validate_callback_url(callback_url)

                timestamp = int(time.time())

                payload = base_payload.copy()
                try:
                    body_bytes = json.dumps(payload, separators=(',', ':')).encode('utf-8')
                except (TypeError, ValueError) as e:
                    logger.error("JSON serialization failed", event_id=event_id, error=str(e))
                    return False

                signature = self._sign_payload(body_bytes, timestamp)

                headers = {
                    "X-HireRight-Event-Id": event_id,
                    "X-HireRight-Timestamp": str(timestamp),
                    "X-HireRight-Signature": signature,
                    "Content-Type": "application/json",
                }

                async with httpx.AsyncClient() as client:
                    resp = await client.post(
                        callback_url,
                        content=body_bytes,
                        headers=headers,
                        timeout=self.timeout_seconds,
                    )
                resp.raise_for_status()

                logger.info(
                    "Callback delivered successfully",
                    event_id=event_id,
                    callback_url="[REDACTED]",
                    attempt=attempt + 1,
                )
                return True

            except (ValueError, httpx.HTTPError, httpx.TimeoutException) as e:
                logger.warning(
                    "Callback attempt failed",
                    event_id=event_id,
                    attempt=attempt + 1,
                    max_attempts=self.max_attempts,
                    error=str(e),
                )
                if attempt == self.max_attempts - 1:
                    break
                backoff = self.retry_base_seconds * (2 ** attempt)
                await asyncio.sleep(backoff)

        logger.error(
            "Callback delivery exhausted all retries",
            event_id=event_id,
            callback_url="[REDACTED]",
        )
        return False
