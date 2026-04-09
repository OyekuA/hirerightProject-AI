"""Structured logging configuration with PII redaction and Sentry integration."""

import logging
from typing import Any, Dict
from app.config import get_settings

import structlog
from structlog_sentry import SentryProcessor


_PII_KEYS = {
    "cv_text", "jd_text", "raw_text", "answer", "callback_url",
    "cv_url", "jd_url", "prompt", "generated", "response", "raw",
}


def _redact_dict(obj: Any) -> Any:
    """Recursively redact sensitive values in dicts and lists."""
    if isinstance(obj, dict):
        redacted = {}
        for key, value in obj.items():
            if key in _PII_KEYS:
                redacted[key] = "[REDACTED]"
            else:
                redacted[key] = _redact_dict(value)
        return redacted
    elif isinstance(obj, list):
        return [_redact_dict(item) for item in obj]
    else:
        return obj


def _redact_pii_processor(
    _logger: logging.Logger, _log_method: str, event_dict: Dict[str, Any]
) -> Dict[str, Any]:
    """Redact values of PII-sensitive keys before emitting logs."""
    return _redact_dict(event_dict)


def configure_logging() -> None:
    """Set up structlog with JSON renderer, PII redaction, and Sentry bridge.

    The processor chain is applied in this order:
        1. merge_contextvars – merges per‑request bound vars (correlation_id, endpoint, entity_id)
        2. PII redaction – replaces sensitive field values with "[REDACTED]"
        3. add_log_level – adds the log level (INFO, ERROR, etc.)
        4. SentryProcessor – forwards error‑level logs to Sentry
        5. TimeStamper – adds an ISO‑8601 timestamp
        6. JSONRenderer – outputs machine‑readable JSON lines in production

    Idempotent: repeated calls do not create duplicate handlers.
    """
    if not structlog.is_configured():
        processors = [
            structlog.contextvars.merge_contextvars,
            _redact_pii_processor,
            structlog.stdlib.add_log_level,
            SentryProcessor(level=logging.ERROR),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ]

        structlog.configure(
            processors=processors,
            wrapper_class=structlog.BoundLogger,
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=True,
        )

    root_logger = logging.getLogger()
    handler_already_added = any(
        isinstance(h, logging.StreamHandler) and
        isinstance(h.formatter, structlog.stdlib.ProcessorFormatter)
        for h in root_logger.handlers
    )
    if not handler_already_added:
        handler = logging.StreamHandler()
        handler.setFormatter(
            structlog.stdlib.ProcessorFormatter(
                processor=structlog.processors.JSONRenderer(),
            )
        )
        root_logger.addHandler(handler)

    settings = get_settings()
    root_logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.ERROR))