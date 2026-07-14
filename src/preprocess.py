"""Text preprocessing for the AI Contract Analysis Pipeline.

This module is responsible solely for cleaning and normalizing the raw
text extracted from a contract PDF into a form suitable for downstream
LLM consumption. It performs no PDF loading, no LLM calls, no clause
extraction, no summarization, and no output persistence — those are the
responsibilities of other pipeline stages.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections import Counter

from src.models import ContractDocument

logger = logging.getLogger(__name__)

_BLANK_LINES_PATTERN = re.compile(r"\n{3,}")
_SPACES_AND_TABS_PATTERN = re.compile(r"[ \t]+")

# Matches a line that consists *only* of a page-number marker, in the
# common forms PDFs produce, e.g.:
#   "3", "- 3 -", "Page 3", "Page 3 of 45", "3/45"
# Anchored to the full line (after stripping) so it never matches a page
# number that appears alongside real clause text on the same line.
_PAGE_NUMBER_LINE_PATTERN = re.compile(
    r"^\s*(?:page\s+)?-?\s*\d+\s*(?:(?:of|/)\s*\d+)?\s*-?\s*$",
    re.IGNORECASE,
)

# A repeated short line (company letterhead, "CONFIDENTIAL" watermark,
# running header/footer text, etc.) is only treated as boilerplate if it
# is both short and appears verbatim on at least this many lines. This
# keeps the pass conservative: a single mention of a phrase, or a longer
# line that is actually substantive legal text, is never removed.
_MIN_BOILERPLATE_REPETITIONS = 3
_MAX_BOILERPLATE_LINE_LENGTH = 80


class Preprocessor:
    """Cleans and normalizes raw contract text extracted from a PDF.

    Applies a fixed, ordered sequence of normalization steps to
    ``ContractDocument.raw_text`` and populates
    ``ContractDocument.clean_text`` with the result, leaving the raw
    text untouched.
    """

    def __init__(self) -> None:
        """Initialize the preprocessor."""
        logger.info("Preprocessor initialized")

    def preprocess(self, document: ContractDocument) -> ContractDocument:
        """Clean and normalize a contract document's raw text.

        Runs the raw text through Unicode normalization, line ending
        normalization, page-number/header/footer boilerplate removal,
        whitespace trimming, space collapsing, and blank line
        collapsing, then stores the result in ``document.clean_text``.
        ``document.raw_text`` is never modified.

        Args:
            document: The contract document whose ``raw_text`` should
                be cleaned.

        Returns:
            The same :class:`~src.models.ContractDocument` instance,
            with ``clean_text`` populated.

        Raises:
            ValueError: If ``document.raw_text`` is empty or contains
                only whitespace.
        """
        self._validate_document(document)
        logger.info("Preprocessing started (input_length=%d)", len(document.raw_text))

        document.clean_text = self._clean_text(document.raw_text)

        logger.info("Preprocessing completed (output_length=%d)", len(document.clean_text))
        return document

    @staticmethod
    def _validate_document(document: ContractDocument) -> None:
        """Ensure the document's raw text is present and non-blank.

        Args:
            document: The contract document to validate.

        Raises:
            ValueError: If ``document.raw_text`` is empty or contains
                only whitespace.
        """
        if not document.raw_text or not document.raw_text.strip():
            raise ValueError("raw_text is empty or contains only whitespace.")

    def _clean_text(self, text: str) -> str:
        """Run the full text-cleaning pipeline over a string, in order.

        Applies Unicode normalization, line ending normalization,
        removal of standalone page-number lines and repeated
        header/footer boilerplate, trailing-whitespace trimming, space
        collapsing, blank line collapsing, and finally strips
        leading/trailing whitespace from the whole document.

        Args:
            text: The raw text to clean.

        Returns:
            The fully cleaned and normalized text.
        """
        text = self._normalize_unicode(text)
        text = self._normalize_line_endings(text)
        text = self._remove_page_number_lines(text)
        text = self._remove_repeated_boilerplate_lines(text)
        text = self._trim_line_whitespace(text)
        text = self._collapse_spaces(text)
        text = self._collapse_blank_lines(text)
        return text.strip()

    @staticmethod
    def _normalize_unicode(text: str) -> str:
        """Normalize Unicode text to its NFKC canonical form.

        Collapses visually equivalent but differently encoded
        characters (e.g. full-width digits, ligatures, various dash and
        quote variants) into a single consistent representation,
        without altering the substantive wording of the contract.

        Args:
            text: The text to normalize.

        Returns:
            The Unicode-normalized text.
        """
        return unicodedata.normalize("NFKC", text)

    @staticmethod
    def _normalize_line_endings(text: str) -> str:
        """Convert all line endings to a single ``\\n`` style.

        Args:
            text: The text whose line endings should be normalized.

        Returns:
            The text with ``\\r\\n`` and ``\\r`` converted to ``\\n``.
        """
        return text.replace("\r\n", "\n").replace("\r", "\n")

    @staticmethod
    def _remove_page_number_lines(text: str) -> str:
        """Remove lines that consist solely of a page-number marker.

        Matches common PDF page-number renderings such as ``"3"``,
        ``"- 3 -"``, ``"Page 3"``, and ``"Page 3 of 45"`` only when the
        *entire* line (after stripping) is nothing but that marker. A
        line that happens to contain a number alongside real clause
        text (e.g. a numbered section heading like ``"3. Termination"``
        or a monetary figure) never matches, since the pattern requires
        the whole line to be the page-number marker with no other
        content.

        Args:
            text: The text to strip page-number lines from.

        Returns:
            The text with standalone page-number lines removed
            entirely (including their newline), leaving surrounding
            content and blank-line structure otherwise intact.
        """
        lines = text.split("\n")
        kept_lines = [
            line for line in lines if not _PAGE_NUMBER_LINE_PATTERN.match(line)
        ]
        return "\n".join(kept_lines)

    @staticmethod
    def _remove_repeated_boilerplate_lines(text: str) -> str:
        """Remove short lines that repeat across the document as running headers/footers.

        A line is only treated as boilerplate — and removed everywhere
        it occurs — if it is both short (at most
        ``_MAX_BOILERPLATE_LINE_LENGTH`` characters, ruling out
        substantive clause text) and repeats verbatim at least
        ``_MIN_BOILERPLATE_REPETITIONS`` times (ruling out a legal
        phrase or heading that happens to appear once or twice). This
        keeps the removal conservative: genuine, non-repeating legal
        text is never affected, even if short.

        Typical matches are running letterhead ("ACME CORP"),
        confidentiality watermarks ("CONFIDENTIAL"), or footer text
        repeated on every page.

        Args:
            text: The text to strip repeated boilerplate lines from.

        Returns:
            The text with qualifying repeated short lines removed,
            leaving all other content untouched.
        """
        lines = text.split("\n")
        stripped_lines = [line.strip() for line in lines]

        candidate_counts = Counter(
            stripped for stripped in stripped_lines
            if stripped and len(stripped) <= _MAX_BOILERPLATE_LINE_LENGTH
        )
        boilerplate_lines = {
            candidate
            for candidate, count in candidate_counts.items()
            if count >= _MIN_BOILERPLATE_REPETITIONS
        }

        if not boilerplate_lines:
            return text

        kept_lines = [
            line
            for line, stripped in zip(lines, stripped_lines)
            if stripped not in boilerplate_lines
        ]
        return "\n".join(kept_lines)

    @staticmethod
    def _trim_line_whitespace(text: str) -> str:
        """Remove trailing whitespace from every line.

        Args:
            text: The text whose lines should be trimmed.

        Returns:
            The text with trailing whitespace removed from each line.
        """
        return "\n".join(line.rstrip() for line in text.split("\n"))

    @staticmethod
    def _collapse_spaces(text: str) -> str:
        """Collapse consecutive spaces and tabs into a single space.

        Args:
            text: The text whose horizontal whitespace should be
                collapsed.

        Returns:
            The text with runs of spaces/tabs collapsed. Newlines are
            left unaffected.
        """
        return _SPACES_AND_TABS_PATTERN.sub(" ", text)

    @staticmethod
    def _collapse_blank_lines(text: str) -> str:
        """Collapse multiple consecutive blank lines into a single blank line.

        Args:
            text: The text whose blank lines should be collapsed.

        Returns:
            The text with runs of three or more newlines reduced to
            exactly two, preserving paragraph boundaries.
        """
        return _BLANK_LINES_PATTERN.sub("\n\n", text)