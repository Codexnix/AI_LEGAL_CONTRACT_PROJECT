"""Centralized configuration for the AI Contract Analysis Pipeline.

This module defines a single :class:`Settings` object that is the sole
source of truth for every configurable value in the project: API
credentials, LLM parameters, and filesystem locations. All other
modules obtain configuration exclusively via::

    from src.config import settings

Values are loaded from environment variables and an optional ``.env``
file at the project root via ``pydantic-settings``. This module
contains no business logic, only configuration definition and
validation.
"""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import Field, computed_field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# A Groq API key is a non-empty string of printable, non-whitespace
# characters. Groq keys conventionally start with "gsk_", but the prefix
# is only used to emit a helpful warning-style error, not to hard-fail,
# since key formats may change over time.
_GROQ_KEY_PATTERN = re.compile(r"^\S+$")
_GROQ_KEY_PREFIX = "gsk_"

# Reasonable upper bounds used purely to catch obvious misconfiguration
# (e.g. a typo like ``retry_attempts=300``) rather than to encode a hard
# business rule.
_MAX_TOKENS_CEILING = 200_000
_MAX_RETRY_ATTEMPTS = 10


class Settings(BaseSettings):
    """Application-wide configuration for the contract analysis pipeline.

    Combines Groq API configuration, LLM generation defaults, and
    all filesystem paths used by the pipeline into a single, validated,
    strongly typed object. Filesystem directories required by the
    pipeline are created automatically once the settings are loaded.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # Groq / LLM configuration
    # ------------------------------------------------------------------
    groq_api_key: str = Field(
        ...,
        description=(
            "Secret API key used to authenticate with the Groq API. Must be "
            "a non-empty string containing no whitespace. Loaded from the "
            "GROQ_API_KEY environment variable or a .env file; never hard-code "
            "this value in source control."
        ),
    )
    llm_model: str = Field(
        default="llama-3.3-70b-versatile",
        description=(
            "Identifier of the Groq-hosted model used for contract extraction "
            "and summarization (e.g. 'llama-3.3-70b-versatile')."
        ),
    )
    llm_default_max_tokens: int = Field(
        default=1000,
        description=(
            "Default maximum number of completion tokens requested per LLM "
            f"call. Must be a positive integer, no greater than {_MAX_TOKENS_CEILING}."
        ),
    )
    llm_default_temperature: float = Field(
        default=0.0,
        description=(
            "Default sampling temperature applied to LLM requests, controlling "
            "randomness of generated output. Must be between 0.0 (deterministic) "
            "and 1.0 (most random), inclusive."
        ),
    )
    llm_timeout_seconds: float = Field(
        default=60.0,
        description=(
            "Maximum time, in seconds, to wait for a Groq API response before "
            "the request is aborted. Must be a positive, finite number."
        ),
    )
    retry_attempts: int = Field(
        default=3,
        description=(
            "Maximum number of attempts for retryable LLM requests, including "
            f"the initial attempt. Must be between 1 and {_MAX_RETRY_ATTEMPTS}, inclusive."
        ),
    )

    # ------------------------------------------------------------------
    # Field validators
    # ------------------------------------------------------------------
    @field_validator("groq_api_key")
    @classmethod
    def _validate_groq_api_key(cls, value: str) -> str:
        """Ensure the Groq API key is a plausible, non-empty credential.

        Args:
            value: The raw API key value.

        Returns:
            The validated (whitespace-stripped) API key.

        Raises:
            ValueError: If the value is empty, contains whitespace, or does
                not look like a Groq API key.
        """
        stripped = value.strip()
        if not stripped:
            raise ValueError(
                "groq_api_key must not be empty. Set the GROQ_API_KEY "
                "environment variable or add it to your .env file."
            )
        if not _GROQ_KEY_PATTERN.match(stripped):
            raise ValueError(
                "groq_api_key must not contain whitespace or control characters."
            )
        if not stripped.startswith(_GROQ_KEY_PREFIX):
            raise ValueError(
                f"groq_api_key does not look like a Groq API key (expected it "
                f"to start with '{_GROQ_KEY_PREFIX}'), got a value starting "
                f"with '{stripped[:4]}...'."
            )
        return stripped

    @field_validator("llm_default_temperature")
    @classmethod
    def _validate_temperature(cls, value: float) -> float:
        """Ensure the default temperature falls within the valid sampling range.

        Args:
            value: The temperature value to validate.

        Returns:
            The validated temperature.

        Raises:
            ValueError: If the value is not between 0 and 1 inclusive, or is
                not a finite number.
        """
        if value != value or value in (float("inf"), float("-inf")):  # NaN/inf check
            raise ValueError(f"llm_default_temperature must be a finite number, got {value}.")
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"llm_default_temperature must be between 0 and 1, got {value}.")
        return value

    @field_validator("llm_default_max_tokens")
    @classmethod
    def _validate_max_tokens(cls, value: int) -> int:
        """Ensure the default max tokens is a sane positive integer.

        Args:
            value: The max tokens value to validate.

        Returns:
            The validated max tokens value.

        Raises:
            ValueError: If the value is not greater than zero, or exceeds the
                configured ceiling.
        """
        if value <= 0:
            raise ValueError(f"llm_default_max_tokens must be > 0, got {value}.")
        if value > _MAX_TOKENS_CEILING:
            raise ValueError(
                f"llm_default_max_tokens must be <= {_MAX_TOKENS_CEILING}, got {value}."
            )
        return value

    @field_validator("retry_attempts")
    @classmethod
    def _validate_retry_attempts(cls, value: int) -> int:
        """Ensure the retry attempt count is within a sane range.

        Args:
            value: The retry attempts value to validate.

        Returns:
            The validated retry attempts value.

        Raises:
            ValueError: If the value is less than 1 or exceeds the configured
                maximum.
        """
        if value < 1:
            raise ValueError(f"retry_attempts must be >= 1, got {value}.")
        if value > _MAX_RETRY_ATTEMPTS:
            raise ValueError(
                f"retry_attempts must be <= {_MAX_RETRY_ATTEMPTS}, got {value}."
            )
        return value

    @field_validator("llm_timeout_seconds")
    @classmethod
    def _validate_timeout(cls, value: float) -> float:
        """Ensure the request timeout is a positive, finite number of seconds.

        Args:
            value: The timeout value to validate.

        Returns:
            The validated timeout value.

        Raises:
            ValueError: If the value is not greater than zero, or is not finite.
        """
        if value != value or value in (float("inf"), float("-inf")):  # NaN/inf check
            raise ValueError(f"llm_timeout_seconds must be a finite number, got {value}.")
        if value <= 0:
            raise ValueError(f"llm_timeout_seconds must be > 0, got {value}.")
        return value

    # ------------------------------------------------------------------
    # Directories
    # ------------------------------------------------------------------
    @computed_field  # type: ignore[prop-decorator]
    @property
    def project_root(self) -> Path:
        """Absolute path to the project root directory."""
        return _PROJECT_ROOT

    @computed_field  # type: ignore[prop-decorator]
    @property
    def data_dir(self) -> Path:
        """Absolute path to the top-level data directory."""
        return self.project_root / "data"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def raw_pdf_dir(self) -> Path:
        """Absolute path to the directory containing raw source PDF contracts."""
        return self.data_dir / "raw_pdfs"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def processed_dir(self) -> Path:
        """Absolute path to the directory containing processed/cleaned text."""
        return self.data_dir / "processed"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def output_dir(self) -> Path:
        """Absolute path to the directory containing final pipeline outputs."""
        return self.project_root / "outputs"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def logs_dir(self) -> Path:
        """Absolute path to the directory containing log files."""
        return self.project_root / "logs"

    # ------------------------------------------------------------------
    # Files
    # ------------------------------------------------------------------
    @computed_field  # type: ignore[prop-decorator]
    @property
    def json_output_file(self) -> Path:
        """Absolute path to the consolidated JSON output file."""
        return self.output_dir / "results.json"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def csv_output_file(self) -> Path:
        """Absolute path to the flattened CSV output file."""
        return self.output_dir / "results.csv"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def log_file(self) -> Path:
        """Absolute path to the pipeline's log file."""
        return self.logs_dir / "pipeline.log"

    # ------------------------------------------------------------------
    # Filesystem initialization
    # ------------------------------------------------------------------
    def create_directories(self) -> None:
        """Create all required project directories if they do not already exist.

        Ensures ``raw_pdf_dir``, ``processed_dir``, ``output_dir``, and
        ``logs_dir`` exist on disk, creating any missing parent
        directories. Does not create any files. Safe to call multiple
        times, as existing directories are left untouched.

        Raises:
            RuntimeError: If a required path exists but is not a directory,
                or if the directory cannot be created due to a filesystem
                error (e.g. insufficient permissions).
        """
        for directory in (
            self.raw_pdf_dir,
            self.processed_dir,
            self.output_dir,
            self.logs_dir,
        ):
            if directory.exists() and not directory.is_dir():
                raise RuntimeError(
                    f"Expected '{directory}' to be a directory, but a file "
                    "with that path already exists."
                )
            try:
                directory.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise RuntimeError(
                    f"Failed to create required directory '{directory}': {exc}"
                ) from exc

    @model_validator(mode="after")
    def _ensure_directories_exist(self) -> "Settings":
        """Automatically create required directories once settings are loaded.

        This runs after all field validation succeeds, guaranteeing that any
        code importing ``settings`` can immediately read/write to
        ``raw_pdf_dir``, ``processed_dir``, ``output_dir``, and ``logs_dir``
        without needing to remember to call :meth:`create_directories`
        manually.

        Returns:
            The validated ``Settings`` instance, unchanged.
        """
        self.create_directories()
        return self


settings = Settings()