import threading
import time
from typing import Optional
import litellm
import structlog


class LLMUnavailableError(Exception):
    pass


class CircuitBreaker:

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(self, threshold: int = 3, cooldown_seconds: float = 60.0):
        self.threshold = threshold
        self.cooldown_seconds = cooldown_seconds
        self.state = self.CLOSED
        self.failure_count = 0
        self.opened_at: Optional[float] = None
        self._probe_sent = False
        self._lock = threading.Lock()
        self._log = structlog.get_logger()

    def is_open(self) -> bool:
        with self._lock:
            if self.state == self.CLOSED:
                return False

            if self.state == self.OPEN:
                if self.opened_at is None:
                    self.opened_at = time.monotonic()
                    return True
                elapsed = time.monotonic() - self.opened_at
                if elapsed >= self.cooldown_seconds:
                    self.state = self.HALF_OPEN
                    self._probe_sent = False
                    self.opened_at = None
                    self._log.info(
                        "Circuit breaker moving to HALF_OPEN",
                        threshold=self.threshold,
                        cooldown_seconds=self.cooldown_seconds,
                    )
                    return False
                return True

            if not self._probe_sent:
                self._probe_sent = True
                return False
            return True

    def record_failure(self) -> None:
        with self._lock:
            if self.state == self.HALF_OPEN:
                self.state = self.OPEN
                self.opened_at = time.monotonic()
                self._probe_sent = False
                self.failure_count = self.threshold
                self._log.warning(
                    "Circuit breaker probe failed, reopening",
                    failure_count=self.failure_count,
                    threshold=self.threshold,
                    cooldown_seconds=self.cooldown_seconds,
                )
                return

            self.failure_count += 1
            if self.failure_count >= self.threshold and self.state == self.CLOSED:
                self.state = self.OPEN
                self.opened_at = time.monotonic()
                self._log.warning(
                    "Circuit breaker opened",
                    failure_count=self.failure_count,
                    threshold=self.threshold,
                    cooldown_seconds=self.cooldown_seconds,
                )

    def record_success(self) -> None:
        with self._lock:
            if self.state == self.HALF_OPEN:
                self.state = self.CLOSED
                self.failure_count = 0
                self.opened_at = None
                self._probe_sent = False
                self._log.info(
                    "Circuit breaker probe succeeded, closing",
                    threshold=self.threshold,
                    cooldown_seconds=self.cooldown_seconds,
                )
                return

            if self.failure_count > 0:
                self._log.debug(
                    "Circuit breaker reset",
                    previous_failures=self.failure_count,
                )
            self.failure_count = 0
            self.state = self.CLOSED
            self.opened_at = None
            self._probe_sent = False


class LLMClient:

    def __init__(
        self,
        model: str,
        embedding_model: str,
        embedding_dimensions: int,
        generation_cooldown: float,
        embedding_cooldown: float,
        generation_timeout: int = 30,
        embedding_timeout: int = 30,
        max_retries: int = 2,
        retry_backoff_base: float = 1.0,
        api_key: Optional[str] = None,
    ):
        self._model = model
        self._embedding_model = embedding_model
        self._embedding_dimensions = embedding_dimensions
        self._generation_breaker = CircuitBreaker(
            cooldown_seconds=generation_cooldown
        )
        self._embedding_breaker = CircuitBreaker(
            cooldown_seconds=embedding_cooldown
        )
        self._generation_timeout = generation_timeout
        self._embedding_timeout = embedding_timeout
        self._max_retries = max_retries
        self._retry_backoff_base = retry_backoff_base
        self._log = structlog.get_logger()

        if api_key is not None:
            litellm.api_key = api_key

    def generate(
        self,
        prompt: str,
        temperature: float = 0,
        response_format: Optional[dict] = None,
        max_tokens: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> str:
        if self._generation_breaker.is_open():
            self._log_token_usage(len(prompt), 0, "generation", outcome="breaker_open")
            raise LLMUnavailableError("Generation circuit breaker is open")

        for attempt in range(self._max_retries + 1):
            try:
                kwargs = {
                    "model": self._model,
                    "messages": [{"role": "user", "content": prompt}],
                    "timeout": self._generation_timeout,
                    "temperature": temperature,
                }

                has_optional = False
                if response_format is not None:
                    supported = litellm.get_supported_openai_params(model=self._model)
                    if "response_format" in supported:
                        kwargs["response_format"] = response_format
                        has_optional = True
                        if response_format.get("type") == "json_schema":
                            if litellm.supports_response_schema(model=self._model):
                                kwargs["response_format"] = response_format
                            else:
                                kwargs["response_format"] = {"type": "json_object"}
                    else:
                        self._log.info(
                            "response_format not supported by model, relying on prompt instruction",
                            model=self._model,
                        )
                    self._log.info(
                        "LLM response format mode",
                        response_format=kwargs.get("response_format"),
                        model=self._model,
                    )

                if max_tokens is not None:
                    kwargs["max_tokens"] = max_tokens
                    has_optional = True
                if seed is not None:
                    kwargs["seed"] = seed
                    has_optional = True

                if has_optional:
                    kwargs["drop_params"] = True

                response = litellm.completion(**kwargs)
                text = response.choices[0].message.content
                if not text or not text.strip():
                    raise ValueError("LLM returned an empty response")
                self._generation_breaker.record_success()
                self._log_token_usage(len(prompt), len(text), "generation", outcome="success")
                return text
            except Exception as exc:
                retry_after = _extract_retry_after(exc)
                if retry_after is not None and attempt < self._max_retries:
                    self._log.warning(
                        "Rate limit hit, sleeping Retry-After",
                        retry_after=retry_after,
                        attempt=attempt + 1,
                    )
                    time.sleep(retry_after)
                elif attempt < self._max_retries:
                    backoff = self._retry_backoff_base * (2 ** attempt)
                    time.sleep(backoff)
                else:
                    self._generation_breaker.record_failure()
                    self._log_token_usage(len(prompt), 0, "generation", outcome="failure")
                    raise LLMUnavailableError(
                        f"LLM generation failed after {self._max_retries + 1} attempts: {exc}"
                    ) from exc

    def embed(self, text: str) -> list[float]:
        if self._embedding_breaker.is_open():
            self._log_token_usage(len(text), 0, "embedding", outcome="breaker_open")
            raise LLMUnavailableError("Embedding circuit breaker is open")

        for attempt in range(self._max_retries + 1):
            try:
                result = litellm.embedding(
                    model=self._embedding_model,
                    input=[text],
                    dimensions=self._embedding_dimensions,
                    timeout=self._embedding_timeout,
                )
                entry = result.data[0]
                vector = entry["embedding"] if isinstance(entry, dict) else entry.embedding
                if not vector:
                    raise ValueError("LLM returned an empty embedding")
                self._embedding_breaker.record_success()
                self._log_token_usage(len(text), 0, "embedding", outcome="success")
                return vector
            except Exception as exc:
                if attempt < self._max_retries:
                    backoff = self._retry_backoff_base * (2 ** attempt)
                    time.sleep(backoff)
                else:
                    self._embedding_breaker.record_failure()
                    self._log_token_usage(len(text), 0, "embedding", outcome="failure")
                    raise LLMUnavailableError(
                        f"LLM embedding failed after {self._max_retries + 1} attempts: {exc}"
                    ) from exc

    @staticmethod
    def get_response_format(model: str, structure_type: str, item_schema: Optional[dict] = None) -> Optional[dict]:
        supported = litellm.get_supported_openai_params(model=model)
        if "response_format" not in supported:
            return None

        if structure_type == "object":
            return {"type": "json_object"}

        if structure_type == "array":
            if litellm.supports_response_schema(model=model):
                items = item_schema or {}
                schema = {
                    "type": "array",
                    "items": items,
                }
                return {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "response",
                        "schema": schema,
                    },
                }
            return None

        return None

    def _log_token_usage(
        self, input_chars: int, output_chars: int, operation: str, outcome: str = "success"
    ) -> None:
        input_tokens = input_chars // 4
        output_tokens = output_chars // 4 if output_chars else 0
        self._log.info(
            "LLM token usage",
            operation=operation,
            input_tokens_approx=input_tokens,
            output_tokens_approx=output_tokens,
            total_tokens_approx=input_tokens + output_tokens,
            outcome=outcome,
        )


def _extract_retry_after(exc: Exception) -> Optional[float]:
    _log = structlog.get_logger()
    try:
        if getattr(exc, "status_code", None) == 429:
            headers = getattr(exc, "response_headers", None) or getattr(exc, "headers", None)
            if headers:
                raw = headers.get("Retry-After") or headers.get("retry-after")
                if raw is not None:
                    return float(raw)
    except (ValueError, TypeError):
        _log.debug("Failed to extract Retry-After from exception headers")
    try:
        if isinstance(exc, litellm.RateLimitError):
            ra = getattr(exc, "retry_after", None)
            if ra is not None:
                return float(ra)
    except (ValueError, TypeError):
        _log.debug("Failed to extract retry_after from RateLimitError")
    return None
