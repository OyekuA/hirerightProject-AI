"""Google Gemini API client with circuit‑breaker protection."""

import time
from typing import Optional
from google import genai
from google.genai import types
import structlog

from app.config import get_settings

class GeminiUnavailableError(Exception):
    """Raised when a Gemini call cannot be performed because a circuit breaker is open."""


class CircuitBreaker:
    """Circuit breaker with CLOSED, OPEN, and HALF_OPEN states and single‑probe recovery.

    Attributes:
        threshold: Number of consecutive failures needed to open the breaker.
        cooldown_seconds: Seconds the breaker stays open before moving to HALF_OPEN.
    """

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
        self._log = structlog.get_logger()

    def is_open(self) -> bool:
        """Return True if the breaker is currently open (no requests should pass).
        
        If the breaker is OPEN and cooldown has elapsed, it transitions to HALF_OPEN
        and allows a single probe (returns False for that probe).
        Subsequent calls while HALF_OPEN and probe already sent return True.
        """
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
                self._probe_sent = True
                return False
            return True
        
        if not self._probe_sent:
            self._probe_sent = True
            return False
        return True

    def record_failure(self) -> None:
        """Record a failure, possibly opening the breaker or keeping it open."""
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
        """Record a success, resetting the breaker if CLOSED or moving to CLOSED if HALF_OPEN."""
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

    def _reset(self) -> None:
        """Internal reset of the breaker's state (kept for compatibility)."""
        if self.failure_count > 0:
            self._log.debug(
                "Circuit breaker reset",
                previous_failures=self.failure_count,
            )
        self.state = self.CLOSED
        self.failure_count = 0
        self.opened_at = None
        self._probe_sent = False


class GeminiClient:
    """Google Gemini API client with separate circuit breakers for generation and embedding."""

    def __init__(
        self,
        api_key: str,
        generation_cooldown: float,
        embedding_cooldown: float,
        generation_timeout: int = 30,
        embedding_timeout: int = 30,
        max_retries: int = 2,
        retry_backoff_base: float = 1.0,
    ):
        """Initialize the Gemini client and its circuit breakers.

        Args:
            api_key: Google AI Studio API key
            generation_cooldown: Seconds the generation breaker stays open
            embedding_cooldown: Seconds the embedding breaker stays open
            generation_timeout: Timeout in seconds for generation requests
            embedding_timeout: Timeout in seconds for embedding requests
            max_retries: Maximum retries for transient failures before recording breaker failure
            retry_backoff_base: Base backoff seconds for exponential retry
        """
        http_options = types.HttpOptions(
            timeout=max(generation_timeout, embedding_timeout) * 1000
        )

        self._client = genai.Client(api_key=api_key, http_options=http_options)
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

    def generate(self, prompt: str) -> str:
        """Generate text from a prompt using Gemini‑2.5‑flash‑lite.

        Args:
            prompt: Input text to send to the model.

        Returns:
            The generated text.

        Raises:
            GeminiUnavailableError: If the generation circuit breaker is open
                or the API call fails after tripping the breaker.
        """
        if self._generation_breaker.is_open():
            self._log_token_usage(len(prompt), 0, "generation", outcome="breaker_open")
            raise GeminiUnavailableError("Generation circuit breaker is open")

        import time
        last_exc = None
        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.models.generate_content(
                    model="gemini-2.5-flash-lite",
                    contents=prompt,
                )
                text = response.text
                self._generation_breaker.record_success()
                self._log_token_usage(len(prompt), len(text), "generation", outcome="success")
                return text
            except Exception as exc:
                last_exc = exc
                if attempt < self._max_retries:
                    backoff = self._retry_backoff_base * (2 ** attempt)
                    time.sleep(backoff)
                else:
                    self._generation_breaker.record_failure()
                    self._log_token_usage(len(prompt), 0, "generation", outcome="failure")
                    raise GeminiUnavailableError(
                        f"Gemini generation failed after {self._max_retries + 1} attempts: {exc}"
                    ) from exc

    def embed(self, text: str) -> list[float]:
        """Embed a piece of text using the gemini‑embedding‑001 model.

        Args:
            text: Input text to embed.

        Returns:
            A 768‑dimensional embedding vector.

        Raises:
            GeminiUnavailableError: If the embedding circuit breaker is open
                or the API call fails after tripping the breaker.
        """
        if self._embedding_breaker.is_open():
            self._log_token_usage(len(text), 0, "embedding", outcome="breaker_open")
            raise GeminiUnavailableError("Embedding circuit breaker is open")

        import time
        last_exc = None
        for attempt in range(self._max_retries + 1):
            try:
                result = self._client.models.embed_content(
                    model="gemini-embedding-001",
                    contents=text,
                    config=types.EmbedContentConfig(output_dimensionality=768),
                )
                vector = result.embeddings[0].values
                self._embedding_breaker.record_success()
                self._log_token_usage(len(text), 0, "embedding", outcome="success")
                return vector
            except Exception as exc:
                last_exc = exc
                if attempt < self._max_retries:
                    backoff = self._retry_backoff_base * (2 ** attempt)
                    time.sleep(backoff)
                else:
                    self._embedding_breaker.record_failure()
                    self._log_token_usage(len(text), 0, "embedding", outcome="failure")
                    raise GeminiUnavailableError(
                        f"Gemini embedding failed after {self._max_retries + 1} attempts: {exc}"
                    ) from exc

    def _log_token_usage(
        self, input_chars: int, output_chars: int, operation: str, outcome: str = "success"
    ) -> None:
        """Log approximate token usage for a Gemini call.

        The approximation uses 4 characters ≈ 1 token.

        Args:
            input_chars: Length of input text in characters.
            output_chars: Length of output text in characters (zero for embedding).
            operation: Either "generation" or "embedding".
            outcome: One of "success", "breaker_open", "failure".
        """
        input_tokens = input_chars // 4
        output_tokens = output_chars // 4 if output_chars else 0
        self._log.info(
            "Gemini token usage",
            operation=operation,
            input_tokens_approx=input_tokens,
            output_tokens_approx=output_tokens,
            total_tokens_approx=input_tokens + output_tokens,
            outcome=outcome,
        )