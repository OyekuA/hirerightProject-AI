"""Client for delivering ingestion‑status callbacks with HMAC‑SHA256 signatures.

Implements SSRF safety, retries with exponential backoff, and configurable timeouts.
"""

import hashlib
import hmac
import json
import time
import structlog
from typing import Optional
import httpx

from app.services.ingestion_fetch import validate_ingest_url


logger = structlog.get_logger()


class CallbackClient:
    """Thread‑safe client for sending signed ingestion‑status callbacks."""

    def __init__(
        self,
        hmac_secret: str,
        max_retries: int = 3,
        retry_base_seconds: int = 2,
    ):
        self.hmac_secret = hmac_secret
        self.max_retries = max_retries
        self.retry_base_seconds = retry_base_seconds

    def _sign_payload(self, body_bytes: bytes, timestamp: int) -> str:
        """Compute HMAC‑SHA256 signature of the payload."""
        message = f"{timestamp}.".encode() + body_bytes
        digest = hmac.new(
            self.hmac_secret.encode(),
            msg=message,
            digestmod=hashlib.sha256,
        ).hexdigest()
        return f"sha256={digest}"

    def send(
        self,
        callback_url: str,
        event_id: str,
        entity_type: str,
        entity_id: int,
        status: str,
        error: Optional[str] = None,
    ) -> bool:
        """Deliver a callback to the external system.

        Returns True if a HTTP 2xx response was received within the retry budget,
        False otherwise.
        """
        base_payload = {
            "event_id": event_id,
            "entity_type": entity_type,
            "entity_id": entity_id,
            "status": status,
            "error": error,
        }

        for attempt in range(self.max_retries):
            try:
                validate_ingest_url(callback_url)

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

                resp = httpx.post(
                    callback_url,
                    content=body_bytes,
                    headers=headers,
                    timeout=10.0,
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
                    max_retries=self.max_retries,
                    error=str(e),
                )
                if attempt == self.max_retries - 1:
                    break
                backoff = self.retry_base_seconds * (2 ** attempt)
                time.sleep(backoff)

        logger.error(
            "Callback delivery exhausted all retries",
            event_id=event_id,
            callback_url="[REDACTED]",
        )
        return False