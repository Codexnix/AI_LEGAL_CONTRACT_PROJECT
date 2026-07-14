"""Clause extraction for the AI Contract Analysis Pipeline.

This module is responsible solely for extracting the termination,
confidentiality, and limitation-of-liability clauses from a
preprocessed contract document via the LLM. It performs no PDF
loading, no text cleaning, no summarization, and no output
persistence, and it never constructs prompts manually — all base
prompt text comes from :class:`~src.prompts.PromptBuilder`, augmented
here with an explicit precision and anti-hallucination directive.

Context reduction (keyword-based retrieval) is delegated entirely to
:class:`~src.retriever.ContractRetriever`; this module no longer
performs its own keyword search or window merging.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Final

from src.llm_client import LLMClient, LLMClientError
from src.models import ClauseExtraction, ContractDocument
from src.prompts import PromptBuilder
from src.retriever import ContractRetriever, RetrievalResult

logger = logging.getLogger(__name__)

_REQUIRED_CLAUSE_FIELDS: Final[tuple[str, ...]] = (
    "termination_clause",
    "confidentiality_clause",
    "liability_clause",
)

_EXTRACTION_PRECISION_DIRECTIVE: Final[str] = (
    "\n\n---\n"
    "For each of the following clause types, extract the exact "
    "contractual wording verbatim from the source text — do not "
    "paraphrase, summarize, or reword:\n"
    "- Termination\n"
    "- Confidentiality\n"
    "- Limitation of Liability\n\n"
    "Rules:\n"
    "- Only extract text that is actually present in the source "
    "document. Never invent, infer, or hallucinate clause text that "
    "does not appear verbatim in the contract.\n"
    "- If a clause type is genuinely absent from the document, return "
    "null for that field rather than substituting unrelated text, a "
    "placeholder, or a best guess.\n"
    "- If a clause spans multiple non-contiguous sections, extract the "
    "most complete, directly relevant excerpt rather than combining "
    "unrelated passages.\n"
    "- Do not confuse similar clause types (e.g. general liability "
    "language vs. a limitation-of-liability provision, or a "
    "non-disclosure/confidentiality clause vs. a non-compete clause)."
)


class ClauseExtractionError(Exception):
    """Base exception for all errors raised by :class:`ClauseExtractor`."""


class JSONParsingError(ClauseExtractionError):
    """Raised when the LLM's response cannot be parsed into a valid clause JSON object."""


class ClauseExtractor:
    """Extracts key contract clauses from a document using the LLM.

    Orchestrates prompt construction via
    :class:`~src.prompts.PromptBuilder`, delegates generation to an
    injected :class:`~src.llm_client.LLMClient`, and parses the model's
    JSON response into a validated
    :class:`~src.models.ClauseExtraction`.

    Before prompting the LLM, the document's clean text is reduced to
    the passages most likely to contain relevant clauses using a
    single, shared :class:`~src.retriever.ContractRetriever` instance,
    which performs deterministic, keyword-based retrieval. This lowers
    token usage without changing the wording of any prompt.

    The base system prompt from :class:`~src.prompts.PromptBuilder` is
    augmented with an explicit precision directive requiring verbatim
    contractual wording, prohibiting hallucination, and requiring
    ``null`` for any clause type genuinely absent from the document.
    """

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        retriever: ContractRetriever | None = None,
    ) -> None:
        """Initialize the extractor.

        Args:
            llm_client: An optional pre-configured
                :class:`~src.llm_client.LLMClient`. If not provided, a
                new instance is created using default configuration.
            retriever: An optional pre-configured
                :class:`~src.retriever.ContractRetriever`. If not
                provided, a new instance is created using default
                configuration. A single instance is created here and
                reused across every call to :meth:`extract`, rather
                than being instantiated per request.
        """
        self._llm_client = llm_client or LLMClient()
        self._retriever = retriever or ContractRetriever()
        logger.info("ClauseExtractor initialized")

    def extract(self, document: ContractDocument) -> ClauseExtraction:
        """Extract the termination, confidentiality, and liability clauses.

        Args:
            document: The contract document to extract clauses from.
                ``document.clean_text`` must be populated.

        Returns:
            A :class:`~src.models.ClauseExtraction` containing the
            extracted clause text, or ``None`` for any clause not
            present in the contract.

        Raises:
            ValueError: If ``document.clean_text`` is empty or contains
                only whitespace.
            JSONParsingError: If the LLM response is not valid JSON or
                is missing required fields.
            ClauseExtractionError: For any other failure during
                extraction.
        """
        self._validate_document(document)
        logger.info("Extraction started (filename=%s)", document.filename)

        system_prompt = self._build_system_prompt()

        retrieval = self._retriever.retrieve(document.clean_text)
        self._log_retrieval_summary(document.filename, retrieval)

        retrieval_context = retrieval.context if retrieval.context else document.clean_text
        user_prompt = PromptBuilder.build_extraction_prompt(retrieval_context)

        try:
            response = self._llm_client.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
            )
        except LLMClientError as exc:
            logger.error("Extraction failed during LLM generation (filename=%s)", document.filename)
            raise ClauseExtractionError(
                f"Clause extraction failed for '{document.filename}': {exc}"
            ) from exc

        parsed = self._parse_json(response.content.strip())
        clauses = self._build_clause_extraction(parsed)

        logger.info("Extraction completed (filename=%s)", document.filename)
        return clauses

    @staticmethod
    def _build_system_prompt() -> str:
        """Build the system prompt used for extraction.

        Layers an explicit precision and anti-hallucination directive
        on top of the base system prompt supplied by
        :class:`~src.prompts.PromptBuilder`, without modifying that
        class. This directs the model to prefer exact contractual
        wording for the termination, confidentiality, and
        limitation-of-liability clauses, to return ``null`` when a
        clause type is genuinely absent, and to never invent clause
        text that is not present in the source document.

        Returns:
            The base extraction system prompt with the precision
            directive appended.
        """
        base_system_prompt = PromptBuilder.get_extraction_system_prompt()
        return f"{base_system_prompt}{_EXTRACTION_PRECISION_DIRECTIVE}"

    @staticmethod
    def _log_retrieval_summary(filename: str, retrieval: RetrievalResult) -> None:
        """Log a structured summary of a single retrieval pass.

        Args:
            filename: The filename of the document being processed,
                included for traceability across log lines.
            retrieval: The :class:`~src.retriever.RetrievalResult`
                produced by the single retrieval call for this
                document.
        """
        logger.info(
            "Original document characters (filename=%s): %s",
            filename,
            f"{retrieval.original_length:,}",
        )
        logger.info(
            "Retrieved characters (filename=%s): %s",
            filename,
            f"{retrieval.retrieved_length:,}",
        )
        logger.info(
            "Compression ratio (filename=%s): %.0f%%",
            filename,
            retrieval.compression_ratio * 100,
        )
        logger.info(
            "Matched categories (filename=%s): %s",
            filename,
            ", ".join(retrieval.matched_categories) or "none",
        )
        logger.info(
            "Matched keywords (filename=%s): %s",
            filename,
            ", ".join(retrieval.matched_keywords) or "none",
        )
        logger.info(
            "Window count (filename=%s): %d",
            filename,
            retrieval.merged_window_count,
        )
        logger.info(
            "Truncated (filename=%s): %s",
            filename,
            retrieval.truncated,
        )

    @staticmethod
    def _validate_document(document: ContractDocument) -> None:
        """Ensure the document's clean text is present and non-blank.

        Args:
            document: The contract document to validate.

        Raises:
            ValueError: If ``document.clean_text`` is empty or contains
                only whitespace.
        """
        if not document.clean_text or not document.clean_text.strip():
            raise ValueError("document.clean_text is empty or contains only whitespace.")

    @staticmethod
    def _parse_json(content: str) -> dict[str, Any]:
        """Parse and validate the LLM's raw response as a clause JSON object.

        Args:
            content: The raw text content returned by the LLM.

        Returns:
            The parsed JSON object as a dictionary.

        Raises:
            JSONParsingError: If the content is not valid JSON, is not
                a JSON object, or is missing one or more required
                clause fields.
        """
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            logger.error("JSON parsing failed: response was not valid JSON")
            raise JSONParsingError("LLM response was not valid JSON.") from exc

        if not isinstance(parsed, dict):
            logger.error("JSON parsing failed: response was not a JSON object")
            raise JSONParsingError("LLM response did not contain a JSON object.")

        missing_fields = [field for field in _REQUIRED_CLAUSE_FIELDS if field not in parsed]
        if missing_fields:
            logger.error("JSON parsing failed: missing fields %s", missing_fields)
            raise JSONParsingError(
                f"LLM response is missing required field(s): {missing_fields}."
            )

        return parsed

    @staticmethod
    def _build_clause_extraction(parsed: dict[str, Any]) -> ClauseExtraction:
        """Convert a validated parsed JSON object into a ClauseExtraction.

        Args:
            parsed: The parsed JSON object, guaranteed to contain all
                required clause fields.

        Returns:
            A validated :class:`~src.models.ClauseExtraction`.

        Raises:
            JSONParsingError: If any clause field has an unexpected
                type (not a string or null).
        """
        values: dict[str, str | None] = {}
        for field in _REQUIRED_CLAUSE_FIELDS:
            value = parsed[field]
            if value is not None and not isinstance(value, str):
                logger.error("JSON parsing failed: field %r had unexpected type", field)
                raise JSONParsingError(
                    f"Field {field!r} must be a string or null, got {type(value).__name__}."
                )
            values[field] = value

        try:
            return ClauseExtraction(**values)
        except Exception as exc:
            logger.error("JSON parsing failed: could not build ClauseExtraction")
            raise JSONParsingError("Parsed clause data failed model validation.") from exc