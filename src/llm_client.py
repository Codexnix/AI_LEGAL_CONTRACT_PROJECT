"""Groq API client wrapper for the AI Contract Analysis Pipeline.

This module is the SINGLE point of contact with the Groq SDK
(``groq``) in the entire codebase. No other module should import
``groq`` directly. All callers interact exclusively with
:class:`LLMClient` and receive back provider-agnostic
:class:`~src.models.LLMResponse` objects, never raw Groq SDK types.

Isolating the SDK here keeps the rest of the pipeline (extractor,
summarizer, pipeline) decoupled from a specific LLM provider, and gives
us one place to control configuration, retries, logging, and error
translation.
"""

from __future__ import annotations

import logging
import time

import groq
from groq import Groq
from groq.types.chat import ChatCompletion
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config import settings
from src.models import LLMResponse

logger = logging.getLogger(__name__)

_MIN_TEMPERATURE = 0.0
_MAX_TEMPERATURE = 1.0


class LLMClientError(Exception):
    """Base exception for all errors raised by :class:`LLMClient`."""


class MissingAPIKeyError(LLMClientError):
    """Raised when no Groq API key is available in configuration."""


class InvalidRequestError(LLMClientError):
    """Raised when caller-supplied generation parameters are invalid."""


class LLMAuthenticationError(LLMClientError):
    """Raised when the Groq API rejects the provided credentials."""


class LLMRateLimitError(LLMClientError):
    """Raised when the Groq API reports that the rate limit was exceeded."""


class LLMConnectionError(LLMClientError):
    """Raised when the Groq API cannot be reached due to a network issue,
    including when a request times out."""


class LLMServerError(LLMClientError):
    """Raised when the Groq API reports a transient server-side failure."""


class LLMResponseError(LLMClientError):
    """Raised when the Groq API returns an unexpected or unusable response."""


_RETRYABLE_EXCEPTIONS = (LLMConnectionError, LLMRateLimitError, LLMServerError)


def _log_before_retry(retry_state: RetryCallState) -> None:
    """Log details about an upcoming retry attempt before it sleeps.

    Called by tenacity immediately before backing off and retrying a
    failed :meth:`LLMClient._send_request_with_retry` call. Logs the
    attempt number that just failed, the type of exception that
    triggered the retry, the exception message, and how long the
    client will wait before the next attempt.

    Args:
        retry_state: The tenacity-provided state describing the
            just-completed attempt, its outcome, and the scheduled
            next action.
    """
    exception = retry_state.outcome.exception() if retry_state.outcome else None
    exception_type = type(exception).__name__ if exception is not None else "unknown"
    exception_message = str(exception) if exception is not None else ""
    wait_seconds = (
        retry_state.next_action.sleep if retry_state.next_action is not None else 0.0
    )
    logger.warning(
        "LLM request retry scheduled (attempt=%d, exception_type=%s, "
        "exception_message=%s, wait_seconds=%.2f)",
        retry_state.attempt_number,
        exception_type,
        exception_message,
        wait_seconds,
    )


class LLMClient:
    """Thin, exception-safe wrapper around the Groq chat completions API.

    Owns the lifecycle of the Groq SDK client and exposes a single
    public method, :meth:`generate`, used by all other pipeline modules
    to obtain text completions. Every Groq-specific concept (SDK
    response objects, SDK exceptions, usage data) is translated into
    plain types or :class:`~src.models.LLMResponse` before leaving this
    class.
    """

    def __init__(self, model: str | None = None) -> None:
        """Initialize the Groq client from application configuration.

        Args:
            model: Optional override for the model identifier. Defaults
                to ``settings.llm_model`` when not provided.

        Raises:
            MissingAPIKeyError: If no Groq API key is configured.
            InvalidRequestError: If the resolved model identifier is
                empty or blank.
        """
        self._api_key = self._resolve_api_key()
        self._model = model or settings.llm_model
        self._validate_model(self._model)
        self._default_max_tokens = settings.llm_default_max_tokens
        self._default_temperature = settings.llm_default_temperature
        self._timeout_seconds = settings.llm_timeout_seconds
        self._client = Groq(
            api_key=self._api_key,
            timeout=self._timeout_seconds,
        )
        logger.info(
            "LLMClient initialized (model=%s, timeout_seconds=%s)",
            self._model,
            self._timeout_seconds,
        )

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        """Generate a completion from the Groq chat completions API.

        Args:
            prompt: The user-facing prompt content to send to the model.
                Must be non-empty.
            system_prompt: Optional system-level instruction that shapes
                the model's behavior for this request. If provided, it
                must not be blank.
            max_tokens: Maximum number of tokens to generate. Must be
                greater than zero. Defaults to ``settings.llm_default_max_tokens``
                when not provided.
            temperature: Sampling temperature in ``[0.0, 1.0]``. Lower
                values produce more deterministic output, preferred for
                extraction tasks. Defaults to
                ``settings.llm_default_temperature`` when not provided.

        Returns:
            An :class:`~src.models.LLMResponse` containing the generated
            text, the model identifier used, and total token usage if
            reported by the API.

        Raises:
            InvalidRequestError: If any input parameter fails validation.
            LLMAuthenticationError: If the API key is rejected.
            LLMRateLimitError: If the request is throttled after retries.
            LLMConnectionError: If the API cannot be reached, or a
                request times out, after retries.
            LLMServerError: If the API reports a persistent server error.
            LLMResponseError: If the API returns a malformed or empty
                response.
            LLMClientError: For any other unexpected failure.
        """
        max_tokens = self._default_max_tokens if max_tokens is None else max_tokens
        temperature = (
            self._default_temperature if temperature is None else temperature
        )
        self._validate_request(prompt, system_prompt, max_tokens, temperature)
        logger.info(
            "LLM request started (model=%s, prompt_length=%d, max_tokens=%d, "
            "temperature=%.2f, timeout_seconds=%s)",
            self._model,
            len(prompt),
            max_tokens,
            temperature,
            self._timeout_seconds,
        )

        start_time = time.perf_counter()
        try:
            response = self._send_request_with_retry(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except LLMAuthenticationError:
            logger.error("LLM request failed: authentication error (model=%s)", self._model)
            raise
        except LLMRateLimitError:
            logger.error(
                "LLM request failed: rate limit exceeded after retries (model=%s)",
                self._model,
            )
            raise
        except LLMConnectionError:
            logger.error(
                "LLM request failed: connection error or timeout after retries "
                "(model=%s, timeout_seconds=%s)",
                self._model,
                self._timeout_seconds,
            )
            raise
        except LLMServerError:
            logger.error(
                "LLM request failed: server error after retries (model=%s)", self._model
            )
            raise
        except LLMClientError:
            logger.error("LLM request failed: unexpected API error (model=%s)", self._model)
            raise

        duration_seconds = time.perf_counter() - start_time
        llm_response = self._to_llm_response(response)
        logger.info(
            "LLM request completed (model=%s, usage_tokens=%s, duration_seconds=%.2f)",
            llm_response.model,
            llm_response.usage_tokens,
            duration_seconds,
        )
        return llm_response

    def _resolve_api_key(self) -> str:
        """Read and validate the Groq API key from configuration.

        Returns:
            The validated API key string.

        Raises:
            MissingAPIKeyError: If the key is missing or empty.
        """
        api_key = getattr(settings, "groq_api_key", None)
        if not api_key:
            logger.error("LLMClient initialization failed: missing Groq API key")
            raise MissingAPIKeyError(
                "GROQ_API_KEY is not configured. Set it in the environment or .env file."
            )
        return api_key

    @staticmethod
    def _validate_model(model: str | None) -> None:
        """Validate the resolved model identifier.

        Args:
            model: The model identifier to validate.

        Raises:
            InvalidRequestError: If the model identifier is missing or
                blank.
        """
        if not model or not model.strip():
            logger.error("LLMClient initialization failed: missing or blank model identifier")
            raise InvalidRequestError(
                "A non-empty model identifier must be configured or provided."
            )

    @staticmethod
    def _validate_request(
        prompt: str,
        system_prompt: str | None,
        max_tokens: int,
        temperature: float,
    ) -> None:
        """Validate caller-supplied generation parameters.

        Args:
            prompt: The prompt to validate.
            system_prompt: The optional system prompt to validate.
            max_tokens: The token limit to validate.
            temperature: The sampling temperature to validate.

        Raises:
            InvalidRequestError: If any parameter is invalid.
        """
        if not prompt or not prompt.strip():
            raise InvalidRequestError("prompt must not be empty.")
        if system_prompt is not None and not system_prompt.strip():
            raise InvalidRequestError(
                "system_prompt must not be blank; pass None to omit it."
            )
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool):
            raise InvalidRequestError(
                f"max_tokens must be an int, got {type(max_tokens).__name__}."
            )
        if max_tokens <= 0:
            raise InvalidRequestError(f"max_tokens must be > 0, got {max_tokens}.")
        if not isinstance(temperature, (int, float)) or isinstance(temperature, bool):
            raise InvalidRequestError(
                f"temperature must be a number, got {type(temperature).__name__}."
            )
        if not (_MIN_TEMPERATURE <= temperature <= _MAX_TEMPERATURE):
            raise InvalidRequestError(
                f"temperature must be between {_MIN_TEMPERATURE} and "
                f"{_MAX_TEMPERATURE}, got {temperature}."
            )

    @retry(
        retry=retry_if_exception_type(_RETRYABLE_EXCEPTIONS),
        stop=stop_after_attempt(settings.retry_attempts),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        before_sleep=_log_before_retry,
        reraise=True,
    )
    def _send_request_with_retry(
        self,
        prompt: str,
        system_prompt: str | None,
        max_tokens: int,
        temperature: float,
    ) -> ChatCompletion:
        """Send a request to Groq, retrying only transient failures.

        Retries on connection errors (including timeouts), rate
        limiting, and server-side (5xx) errors using exponential
        backoff, up to ``settings.retry_attempts`` attempts.
        Authentication and other non-transient errors propagate
        immediately. The retry policy is bound once at class
        definition time rather than rebuilt on every call. Before each
        retry sleeps, :func:`_log_before_retry` logs the attempt
        number, the exception type and message that triggered the
        retry, and the scheduled wait duration.

        Args:
            prompt: The user-facing prompt content.
            system_prompt: Optional system-level instruction.
            max_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature.

        Returns:
            The raw Groq SDK ``ChatCompletion`` object.
        """
        return self._send_request(prompt, system_prompt, max_tokens, temperature)

    def _send_request(
        self,
        prompt: str,
        system_prompt: str | None,
        max_tokens: int,
        temperature: float,
    ) -> ChatCompletion:
        """Send a single request to the Groq chat completions API.

        Args:
            prompt: The user-facing prompt content.
            system_prompt: Optional system-level instruction.
            max_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature.

        Returns:
            The raw Groq SDK ``ChatCompletion`` object.

        Raises:
            LLMAuthenticationError: If the API reports an authentication
                failure.
            LLMRateLimitError: If the API reports that the rate limit
                was exceeded.
            LLMServerError: If the API reports a 5xx server-side error.
            LLMConnectionError: If the request fails due to a network or
                transport-level issue, or times out before completing.
            LLMClientError: If the API reports any other client-side
                error.

        This is the only method in the codebase that touches the SDK's
        request surface directly. Only Groq SDK exceptions are
        translated here; programming errors (e.g. TypeError,
        AttributeError) are allowed to propagate unmodified.
        """
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            return self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except groq.AuthenticationError as exc:
            raise LLMAuthenticationError(
                f"Groq API rejected the provided credentials for model "
                f"'{self._model}': {exc.message}"
            ) from exc
        except groq.RateLimitError as exc:
            raise LLMRateLimitError(
                f"Groq API rate limit was exceeded for model '{self._model}': "
                f"{exc.message}"
            ) from exc
        except groq.APITimeoutError as exc:
            logger.warning(
                "Groq request timed out (model=%s, timeout_seconds=%s)",
                self._model,
                self._timeout_seconds,
            )
            raise LLMConnectionError(
                f"Groq API request for model '{self._model}' timed out after "
                f"{self._timeout_seconds} seconds."
            ) from exc
        except groq.APIConnectionError as exc:
            raise LLMConnectionError(
                f"Could not connect to the Groq API for model '{self._model}': {exc}"
            ) from exc
        except groq.InternalServerError as exc:
            logger.warning(
                "Transient Groq server error (model=%s, status_code=%s)",
                self._model,
                exc.status_code,
            )
            raise LLMServerError(
                f"Groq API server error for model '{self._model}': {exc.message}"
            ) from exc
        except groq.APIStatusError as exc:
            if exc.status_code >= 500:
                logger.warning(
                    "Transient Groq server error (model=%s, status_code=%s)",
                    self._model,
                    exc.status_code,
                )
                raise LLMServerError(
                    f"Groq API server error for model '{self._model}': {exc.message}"
                ) from exc
            raise LLMClientError(
                f"Groq API returned a client error for model '{self._model}' "
                f"(status_code={exc.status_code}): {exc.message}"
            ) from exc
        except groq.APIError as exc:
            raise LLMClientError(
                f"Groq API returned an unexpected error for model "
                f"'{self._model}': {exc.message}"
            ) from exc

    def _to_llm_response(self, response: ChatCompletion) -> LLMResponse:
        """Convert a raw Groq response into a provider-agnostic LLMResponse.

        Args:
            response: The raw Groq SDK ``ChatCompletion`` object.

        Returns:
            A validated :class:`~src.models.LLMResponse`.
        """
        return LLMResponse(
            content=self._extract_text(response),
            model=self._model,
            usage_tokens=self._extract_usage_tokens(response),
        )

    @staticmethod
    def _extract_text(response: ChatCompletion) -> str:
        """Validate and extract text content from a Groq response.

        Explicitly checks for missing choices and missing message
        content before reading any text, so that a malformed response
        produces a clear :class:`LLMResponseError` rather than an
        opaque SDK-internal failure.

        Args:
            response: The raw Groq SDK ``ChatCompletion`` object.

        Returns:
            The stripped text content of the first choice's message.

        Raises:
            LLMResponseError: If the response contains no choices, no
                message, or no usable text content.
        """
        choices = getattr(response, "choices", None)
        if not choices:
            logger.error("LLM response contained no choices")
            raise LLMResponseError("Groq API response contained no choices.")

        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None) if message is not None else None
        if not content or not content.strip():
            finish_reason = getattr(choices[0], "finish_reason", None)
            logger.error(
                "LLM response contained no usable text content (finish_reason=%s)",
                finish_reason,
            )
            raise LLMResponseError(
                "Groq API response contained no usable text content "
                f"(finish_reason={finish_reason})."
            )
        return content.strip()

    @staticmethod
    def _extract_usage_tokens(response: ChatCompletion) -> int | None:
        """Compute total token usage from a Groq response, if available.

        Args:
            response: The raw Groq SDK ``ChatCompletion`` object.

        Returns:
            The total number of tokens consumed by the request, or
            ``None`` if usage data is unavailable.
        """
        usage = getattr(response, "usage", None)
        if usage is None:
            return None
        return getattr(usage, "total_tokens", None)