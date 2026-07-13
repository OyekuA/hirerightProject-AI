from typing import Optional
import structlog

from svix.webhooks import Webhook, WebhookVerificationError

logger = structlog.get_logger()


def verify_recall_webhook(
    raw_body: bytes,
    headers: dict,
    secret: str,
) -> Optional[dict]:
    try:
        wh = Webhook(secret)
        payload = wh.verify(raw_body, headers)
        if not isinstance(payload, dict):
            logger.warning("Recall webhook payload is not a dict")
            return None
        return payload
    except WebhookVerificationError:
        logger.warning("Recall webhook verification failed — invalid signature")
        return None
    except Exception as exc:
        logger.error("Unexpected error verifying Recall webhook", error=str(exc))
        return None
