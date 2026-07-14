"""Contract summarization for the AI Contract Analysis Pipeline.

This module is responsible solely for generating a concise business
summary of a preprocessed contract document via the LLM. It performs
no PDF loading, no text cleaning, no clause extraction, and no output
persistence. Prompt text for the initial generation attempt comes from
:class:`~src.prompts.PromptBuilder`, augmented here with an explicit
coverage directive (purpose, parties, obligations, payment terms,
termination, confidentiality, liability, governing law) so summaries
are never generic. If the LLM's first response falls outside the
required word-count range, a single corrective retry is issued with an
amended instruction appended to that same base prompt.

Context reduction is delegated entirely to
:class:`~src.retriever.ContractRetriever`: rather than building an
ad-hoc head/tail/clause excerpt of ``document.clean_text``, this
module retrieves keyword-relevant context exactly once per
:meth:`summarize` call and reuses that same retrieved context for both
the initial generation attempt and the corrective retry, if any.
"""

from __future__ import annotations

import logging
from typing import Final

from src.llm_client import LLMClient, LLMClientError
from src.models import ContractDocument, ContractSummary
from src.prompts import PromptBuilder
from src.retriever import ContractRetriever, RetrievalResult

logger = logging.getLogger(__name__)

_MIN_WORDS = 100
_MAX_WORDS = 150

_REQUIRED_COVERAGE_DIRECTIVE = (
    "\n\n---\n"
    "Generate a concise executive summary of approximately 100-150 "
    "words using only the supplied contract context.\n\n"
    "The summary must be specific to this contract and must never be "
    "generic. Wherever the information is present in the source text, "
    "explicitly cover:\n"
    "- The purpose of the contract\n"
    "- The parties involved\n"
    "- The main obligations of each party\n"
    "- Payment terms (if present)\n"
    "- Termination conditions\n"
    "- Confidentiality obligations\n"
    "- Liability provisions\n"
    "- Governing law"
)


class SummaryGenerationError(Exception):
    """Raised when a contract summary cannot be generated or is invalid."""


class ContractSummarizer:
    """Generates concise business-style summaries of contract documents.

    Orchestrates prompt construction via
    :class:`~src.prompts.PromptBuilder`, delegates generation to an
    injected :class:`~src.llm_client.LLMClient`, and validates the
    model's response into a :class:`~src.models.ContractSummary`.

    The base system prompt from :class:`~src.prompts.PromptBuilder` is
    augmented with an explicit coverage directive so every summary
    addresses the contract's purpose, parties, obligations, payment
    terms, termination conditions, confidentiality, liability, and
    governing law rather than reading as generic boilerplate.

    Before prompt construction, ``document.clean_text`` is reduced to
    keyword-relevant context using a single, shared
    :class:`~src.retriever.ContractRetriever` instance. Retrieval is
    performed exactly once per :meth:`summarize` call; the same
    retrieved context is reused for both the initial attempt and the
    corrective retry path, if one is needed.

    If the initial summary's word count falls outside the required
    100-150 word range, exactly one corrective retry is attempted
    before giving up.
    """

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        retriever: ContractRetriever | None = None,
    ) -> None:
        """Initialize the summarizer.

        Args:
            llm_client: An optional pre-configured
                :class:`~src.llm_client.LLMClient`. If not provided, a
                new instance is created using default configuration.
            retriever: An optional pre-configured
                :class:`~src.retriever.ContractRetriever`. If not
                provided, a new instance is created using default
                configuration. A single instance is created here and
                reused across every call to :meth:`summarize`, rather
                than being instantiated per request.
        """
        self._llm_client = llm_client or LLMClient()
        self._retriever = retriever or ContractRetriever()
        logger.info("ContractSummarizer initialized")

    def summarize(self, document: ContractDocument) -> ContractSummary:
        """Generate a concise summary of a contract document.

        Args:
            document: The contract document to summarize.
                ``document.clean_text`` must be populated.

        Returns:
            A :class:`~src.models.ContractSummary` containing the
            generated summary text.

        Raises:
            ValueError: If ``document.clean_text`` is empty or contains
                only whitespace.
            SummaryGenerationError: If the LLM call fails, or if both
                the initial attempt and the single corrective retry
                produce an empty, meaningless, or out-of-range summary.
        """
        self._validate_document(document)
        logger.info("Summary generation started (filename=%s)", document.filename)

        system_prompt = self._build_system_prompt()

        retrieval = self._retriever.retrieve(document.clean_text)
        self._log_retrieval_summary(document.filename, retrieval)

        summary_context = retrieval.context if retrieval.context else document.clean_text
        user_prompt = PromptBuilder.build_summary_prompt(summary_context)

        summary_text = self._generate_summary_text(
            prompt=user_prompt,
            system_prompt=system_prompt,
            filename=document.filename,
        )
        self._validate_summary_content(summary_text, document.filename)

        word_count = len(summary_text.split())
        if _MIN_WORDS <= word_count <= _MAX_WORDS:
            logger.info(
                "Summary generation completed without retry "
                "(filename=%s, word_count=%d)",
                document.filename,
                word_count,
            )
            return ContractSummary(summary=summary_text)

        logger.warning(
            "Initial summary out of range, retrying (filename=%s, word_count=%d)",
            document.filename,
            word_count,
        )

        # The same retrieved context (via user_prompt) is reused for the
        # retry; retrieval is never re-run.
        retry_prompt = self._build_retry_prompt(user_prompt, word_count)
        retry_summary_text = self._generate_summary_text(
            prompt=retry_prompt,
            system_prompt=system_prompt,
            filename=document.filename,
        )
        self._validate_summary_content(retry_summary_text, document.filename)
        self._validate_word_count(retry_summary_text, document.filename)

        retry_word_count = len(retry_summary_text.split())
        logger.info(
            "Summary generation completed via retry path "
            "(filename=%s, word_count=%d)",
            document.filename,
            retry_word_count,
        )
        return ContractSummary(summary=retry_summary_text)

    @staticmethod
    def _build_system_prompt() -> str:
        """Build the system prompt used for all generation attempts.

        Layers an explicit content-coverage directive on top of the
        base system prompt supplied by
        :class:`~src.prompts.PromptBuilder`, without modifying that
        class. This ensures every summary — first attempt or retry —
        is generated as a concise executive summary of approximately
        100-150 words using only the supplied contract context, and
        addresses the contract's purpose, parties, obligations,
        payment terms, termination conditions, confidentiality,
        liability, and governing law, rather than reading as a
        generic, boilerplate summary.

        Returns:
            The base system prompt with the coverage directive
            appended.
        """
        base_system_prompt = PromptBuilder.get_summary_system_prompt()
        return f"{base_system_prompt}{_REQUIRED_COVERAGE_DIRECTIVE}"

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
            "Original characters (filename=%s): %s",
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
            "Matched keywords (filename=%s): %s",
            filename,
            ", ".join(retrieval.matched_keywords) or "none",
        )
        logger.info(
            "Matched categories (filename=%s): %s",
            filename,
            ", ".join(retrieval.matched_categories) or "none",
        )
        logger.info(
            "Merged windows (filename=%s): %d",
            filename,
            retrieval.merged_window_count,
        )
        logger.info(
            "Truncated (filename=%s): %s",
            filename,
            retrieval.truncated,
        )

    def _generate_summary_text(
        self, prompt: str, system_prompt: str, filename: str
    ) -> str:
        """Call the LLM and return the stripped summary text.

        Args:
            prompt: The user prompt to send to the LLM.
            system_prompt: The system prompt to send to the LLM.
            filename: The filename of the source document, used for
                error context and logging.

        Returns:
            The stripped text content of the LLM's response.

        Raises:
            SummaryGenerationError: If the LLM call fails.
        """
        try:
            response = self._llm_client.generate(
                prompt=prompt,
                system_prompt=system_prompt,
            )
        except LLMClientError as exc:
            logger.error(
                "Summary generation failed during LLM generation (filename=%s)", filename
            )
            raise SummaryGenerationError(
                f"Summary generation failed for '{filename}': {exc}"
            ) from exc

        return response.content.strip()

    @staticmethod
    def _build_retry_prompt(original_prompt: str, previous_word_count: int) -> str:
        """Build a corrective retry prompt appended to the original prompt.

        Args:
            original_prompt: The original summary prompt produced by
                :class:`~src.prompts.PromptBuilder`, built from the
                same retrieved context as the initial attempt.
            previous_word_count: The word count of the previous,
                out-of-range summary attempt.

        Returns:
            The original prompt with explicit corrective instructions
            appended, directing the LLM to rewrite the summary so that
            it contains between 100 and 150 words while preserving
            every important contractual detail, inventing no facts,
            and remaining concise and professional.
        """
        correction = (
            "\n\n---\n"
            f"The previous summary contained {previous_word_count} words. "
            "Rewrite the summary. The new summary MUST contain BETWEEN 100 "
            "AND 150 WORDS. Preserve every important contractual detail. "
            "Do not invent facts. Keep the summary concise and professional."
        )
        return f"{original_prompt}{correction}"

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
    def _validate_summary_content(summary_text: str, filename: str) -> None:
        """Ensure the generated summary is non-empty and contains meaningful text.

        Args:
            summary_text: The stripped summary text returned by the
                LLM.
            filename: The filename of the source document, used for
                error context.

        Raises:
            SummaryGenerationError: If the summary is empty or does not
                contain meaningful alphabetic content.
        """
        if not summary_text:
            logger.error("Summary generation failed: empty summary (filename=%s)", filename)
            raise SummaryGenerationError(f"Generated summary for '{filename}' was empty.")

        if not any(character.isalpha() for character in summary_text):
            logger.error(
                "Summary generation failed: summary lacks meaningful text (filename=%s)", filename
            )
            raise SummaryGenerationError(
                f"Generated summary for '{filename}' did not contain meaningful text."
            )

    @staticmethod
    def _validate_word_count(summary_text: str, filename: str) -> None:
        """Ensure the generated summary's word count is within range.

        Args:
            summary_text: The stripped summary text returned by the
                LLM.
            filename: The filename of the source document, used for
                error context.

        Raises:
            SummaryGenerationError: If the word count is outside the
                100-150 word range.
        """
        word_count = len(summary_text.split())
        if word_count < _MIN_WORDS or word_count > _MAX_WORDS:
            logger.error(
                "Summary generation failed: word count out of range "
                "(filename=%s, word_count=%d)",
                filename,
                word_count,
            )
            raise SummaryGenerationError(
                f"Generated summary for '{filename}' contained {word_count} words; "
                "expected 100-150 words."
            )