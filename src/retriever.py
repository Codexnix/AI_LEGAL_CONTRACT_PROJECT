"""Standalone keyword-based context retrieval for contract text.

This module is responsible SOLELY for reducing the amount of text that
downstream callers (e.g. a clause extractor or a summarizer) would
otherwise need to send to an LLM. It performs lightweight,
deterministic, keyword-driven retrieval over a raw text string: it
finds legally-significant keywords (termination, confidentiality, and
liability related terms), expands a paragraph-aware window of
surrounding context around each match, merges nearby/overlapping
windows, optionally truncates to a maximum size, and returns the
combined result together with retrieval statistics.

This module intentionally:
    - Contains no LLM calls of any kind.
    - Has no dependency on ``src.extractor`` or ``src.summarizer``.
    - Performs no PDF loading, text cleaning/normalization pipeline
      work, prompt construction, summarization, or output persistence.
    - Has no knowledge of any particular downstream consumer; it
      operates purely on a plain ``str`` and returns a plain ``str``
      (via :meth:`ContractRetriever.retrieve_relevant_context`) or a
      structured :class:`RetrievalResult` (via
      :meth:`ContractRetriever.retrieve`).

Other modules (an extractor, a summarizer, etc.) may choose to use
:class:`ContractRetriever` to shrink their own LLM inputs, but this
module does not know about or reach into them.

Design notes on this revision
------------------------------
* Retrieval is 100% rule-based: no AI, no embeddings, no vector
  databases, no fuzzy matching. Only compiled regular expressions with
  word boundaries.
* All keyword categories are searched in a **single left-to-right pass**
  over the document using one combined, pre-compiled regular
  expression (built once per :class:`ContractRetriever` instance, at
  construction time) rather than one pass per keyword or per category.
* Windows are expanded to the nearest blank line on each side so that
  paragraphs are not cut in half where avoidable.
* Nearby windows (within ``merge_distance`` characters of one another)
  are merged into a single contiguous span, in addition to windows
  that overlap or are adjacent.
* If the merged, retrieved context would exceed ``max_context_chars``,
  it is truncated, preferring to keep the earliest merged windows, and
  the truncation is logged.
* :meth:`ContractRetriever.retrieve` returns a full
  :class:`RetrievalResult` with statistics. The pre-existing public
  method :meth:`ContractRetriever.retrieve_relevant_context` is kept
  byte-for-byte compatible with the previous behaviour (same return
  type, same "return original text unchanged if nothing matched"
  semantics) so existing callers are not broken.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Iterable, Mapping

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievalConfig:
    """Configuration governing how :class:`ContractRetriever` builds context windows.

    Instances are immutable (``frozen=True``) so that a given
    configuration cannot be mutated after a :class:`ContractRetriever`
    has been constructed with it, which keeps retrieval fully
    deterministic for the lifetime of the retriever.

    Attributes:
        chars_before: Number of characters of context to retain
            immediately *before* each keyword match, prior to
            paragraph-aware expansion. Defaults to ``1200``.
        chars_after: Number of characters of context to retain
            immediately *after* each keyword match, prior to
            paragraph-aware expansion. Defaults to ``2500``.
        merge_distance: Two windows that are within this many
            characters of each other (even if not directly
            overlapping or adjacent) are merged into a single window.
            This absorbs keyword matches that are "close together"
            but not close enough to naturally overlap. Defaults to
            ``400``.
        max_context_chars: The maximum total length, in characters, of
            the final assembled context (including window
            separators). If the assembled context would exceed this,
            it is truncated, preferring to keep the earliest merged
            windows in full and dropping/trimming later ones.
            Defaults to ``20000``.
        expand_to_paragraph_boundaries: Whether to expand each raw
            window backward to the previous blank line and forward to
            the next blank line, so paragraphs are not cut in half.
            Defaults to ``True``.
        termination_keywords: Keywords indicating a termination /
            expiration / cancellation clause.
        confidentiality_keywords: Keywords indicating a
            confidentiality / non-disclosure clause.
        liability_keywords: Keywords indicating a liability /
            indemnification clause.
        window_separator: The string inserted between two
            non-contiguous merged windows in the final returned
            context, so that a reader (human or LLM) can tell the
            surrounding text is discontinuous.
        truncation_notice: The string appended to the assembled
            context when truncation occurs, so a reader (human or
            LLM) can tell the context was cut short.
    """

    chars_before: int = 1200
    chars_after: int = 2500
    merge_distance: int = 400
    max_context_chars: int = 20_000
    expand_to_paragraph_boundaries: bool = True

    termination_keywords: tuple[str, ...] = field(
        default_factory=lambda: (
            "termination",
            "terminate",
            "termination date",
            "expiry",
            "expiration",
            "expires",
            "cancel",
            "cancellation",
        )
    )
    confidentiality_keywords: tuple[str, ...] = field(
        default_factory=lambda: (
            "confidential",
            "confidentiality",
            "confidential information",
            "non-disclosure",
            "disclosure",
        )
    )
    liability_keywords: tuple[str, ...] = field(
        default_factory=lambda: (
            "liability",
            "limitation of liability",
            "damages",
            "indemnification",
            "indemnify",
            "hold harmless",
        )
    )

    window_separator: str = "\n\n[...]\n\n"
    truncation_notice: str = "\n\n[... context truncated: maximum size reached ...]"

    @property
    def categories(self) -> Mapping[str, tuple[str, ...]]:
        """Return the configured keyword categories, in a fixed order.

        Returns:
            A mapping from category name (``"termination"``,
            ``"confidentiality"``, ``"liability"``) to its configured
            keyword tuple. Order is fixed and only affects the
            (irrelevant) order in which same-position ties are broken
            during pattern construction; the final result is always
            sorted by position in the source text.
        """
        return {
            "termination": self.termination_keywords,
            "confidentiality": self.confidentiality_keywords,
            "liability": self.liability_keywords,
        }


@dataclass(frozen=True)
class KeywordMatch:
    """A single keyword occurrence found in the source text.

    Attributes:
        category: The keyword category this match belongs to
            (``"termination"``, ``"confidentiality"``, or
            ``"liability"``).
        keyword: The exact configured keyword text that was matched
            (not necessarily the literal casing found in the source,
            since matching is case-insensitive).
        start: The character offset, in the normalized text, where
            the match begins (inclusive).
        end: The character offset, in the normalized text, where the
            match ends (exclusive).
    """

    category: str
    keyword: str
    start: int
    end: int


@dataclass(frozen=True)
class _ContextWindow:
    """A contiguous span of the normalized text to retain in the output.

    Attributes:
        start: The character offset, in the normalized text, where
            this window begins (inclusive).
        end: The character offset, in the normalized text, where this
            window ends (exclusive).
    """

    start: int
    end: int


@dataclass(frozen=True)
class RetrievalResult:
    """The full outcome of a retrieval pass, including statistics.

    Attributes:
        context: The retrieved, merged (and possibly truncated)
            context. This is exactly what
            :meth:`ContractRetriever.retrieve_relevant_context`
            returns.
        original_length: The character length of the input text
            (before normalization).
        retrieved_length: The character length of ``context``.
        compression_ratio: The fraction of the original text that was
            *removed*, i.e. ``1 - retrieved_length / original_length``,
            clamped to ``[0.0, 1.0]``. ``0.0`` if ``original_length``
            is ``0`` or no reduction occurred (e.g. no keywords
            found, so the original text is returned unchanged).
        match_count: The total number of keyword occurrences found
            across all categories.
        merged_window_count: The number of non-overlapping windows
            remaining after merging.
        matched_categories: The set of keyword categories that had at
            least one match, as a sorted tuple.
        matched_keywords: The set of distinct keywords that matched at
            least once, as a sorted tuple.
        truncated: Whether the assembled context had to be truncated
            to respect ``config.max_context_chars``.
    """

    context: str
    original_length: int
    retrieved_length: int
    compression_ratio: float
    match_count: int
    merged_window_count: int
    matched_categories: tuple[str, ...]
    matched_keywords: tuple[str, ...]
    truncated: bool


class ContractRetriever:
    """Performs lightweight, deterministic keyword-based text retrieval.

    :class:`ContractRetriever` reduces an arbitrarily large body of
    contract text down to just the passages surrounding
    legally-significant keywords (termination, confidentiality, and
    liability related terms), so that a caller can send substantially
    fewer characters/tokens to an LLM without omitting the sections
    most likely to contain the clauses it cares about.

    This class performs retrieval only. It does not call an LLM, does
    not build prompts, does not perform clause extraction or
    summarization, and does not depend on any other pipeline module.

    All keyword patterns are compiled exactly once, at construction
    time, into a single combined regular expression, so that
    repeated calls to :meth:`retrieve` / :meth:`retrieve_relevant_context`
    perform only one linear pass over the input text for keyword
    discovery, regardless of how many keywords are configured.
    """

    def __init__(self, config: RetrievalConfig | None = None) -> None:
        """Initialize the retriever and compile its keyword pattern.

        Args:
            config: An optional :class:`RetrievalConfig` controlling
                window sizes, merge distance, maximum context size,
                keyword lists, and separators. If not provided, the
                default configuration is used.
        """
        self._config = config or RetrievalConfig()
        self._pattern = self._compile_pattern(self._config.categories)
        logger.info(
            "ContractRetriever initialized (chars_before=%d, chars_after=%d, "
            "merge_distance=%d, max_context_chars=%d, keyword_count=%d)",
            self._config.chars_before,
            self._config.chars_after,
            self._config.merge_distance,
            self._config.max_context_chars,
            sum(len(keywords) for keywords in self._config.categories.values()),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve_relevant_context(self, text: str) -> str:
        """Reduce ``text`` to the passages surrounding legal keywords.

        This is the original, pre-existing public entry point and is
        kept fully backward compatible: it returns a plain ``str``,
        and returns the original ``text`` unchanged (not normalized)
        if no configured keyword is found anywhere in it, or if
        ``text`` is empty.

        Args:
            text: The raw contract text to retrieve context from.

        Returns:
            The merged, keyword-relevant context, or the original
            ``text`` unchanged if no keywords were found or the input
            was empty.
        """
        return self.retrieve(text).context

    def retrieve(self, text: str) -> RetrievalResult:
        """Retrieve keyword-relevant context from ``text`` and compute statistics.

        Steps performed:
            1. Normalize ``text`` (see :meth:`_normalize_text`).
            2. Search the normalized text, in a single pass, for every
               configured termination, confidentiality, and liability
               keyword (case-insensitively, with word boundaries).
            3. For each match, compute a raw character window of
               ``chars_before``/``chars_after`` around it, then (if
               enabled) expand that window to the nearest blank line
               on each side so paragraphs are not split.
            4. Sort all windows by position and merge any that
               overlap, are adjacent, or fall within
               ``merge_distance`` characters of one another.
            5. Concatenate the merged windows (separated by
               ``config.window_separator`` wherever they are not
               contiguous), truncating (and logging) if the result
               would exceed ``config.max_context_chars``.
            6. Compute and return retrieval statistics alongside the
               resulting context.

        If no keyword is found anywhere in ``text``, the original,
        unmodified ``text`` is returned in full via
        :attr:`RetrievalResult.context` — no normalization is applied
        in that case, since there is nothing to retrieve and the
        caller should see the same text it passed in.

        Args:
            text: The raw contract text to retrieve context from.

        Returns:
            A :class:`RetrievalResult` describing the retrieved
            context and statistics about the retrieval pass. If
            ``text`` is empty, an all-zero result wrapping the empty
            string is returned.
        """
        original_length = len(text)

        if not text:
            logger.info("Retrieval skipped: input text was empty")
            return RetrievalResult(
                context=text,
                original_length=0,
                retrieved_length=0,
                compression_ratio=0.0,
                match_count=0,
                merged_window_count=0,
                matched_categories=(),
                matched_keywords=(),
                truncated=False,
            )

        normalized_text = self._normalize_text(text)
        matches = self._find_keyword_matches(normalized_text)

        if not matches:
            logger.info(
                "Retrieval found no legal keywords in %d character(s); "
                "returning original text unchanged",
                original_length,
            )
            return RetrievalResult(
                context=text,
                original_length=original_length,
                retrieved_length=original_length,
                compression_ratio=0.0,
                match_count=0,
                merged_window_count=0,
                matched_categories=(),
                matched_keywords=(),
                truncated=False,
            )

        raw_windows = [
            self._window_for_match(match, normalized_text) for match in matches
        ]
        sorted_windows = sorted(
            raw_windows, key=lambda window: (window.start, window.end)
        )
        merged_windows = self._merge_windows(sorted_windows)
        context, truncated = self._assemble_context(normalized_text, merged_windows)

        matched_categories = tuple(sorted({match.category for match in matches}))
        matched_keywords = tuple(sorted({match.keyword for match in matches}))
        retrieved_length = len(context)
        compression_ratio = (
            max(0.0, min(1.0, 1 - (retrieved_length / len(normalized_text))))
            if normalized_text
            else 0.0
        )

        self._log_summary(
            original_length=original_length,
            matches=matches,
            merged_window_count=len(merged_windows),
            retrieved_length=retrieved_length,
            compression_ratio=compression_ratio,
            truncated=truncated,
        )

        return RetrievalResult(
            context=context,
            original_length=original_length,
            retrieved_length=retrieved_length,
            compression_ratio=compression_ratio,
            match_count=len(matches),
            merged_window_count=len(merged_windows),
            matched_categories=matched_categories,
            matched_keywords=matched_keywords,
            truncated=truncated,
        )

    # ------------------------------------------------------------------
    # Pattern compilation
    # ------------------------------------------------------------------

    @staticmethod
    def _compile_pattern(categories: Mapping[str, tuple[str, ...]]) -> re.Pattern[str]:
        """Compile a single combined regex covering every category/keyword.

        Every keyword is wrapped with word boundaries (``\\b``) and
        placed into its own uniquely-named capture group so that, for
        any match, the matching group's name reveals both the
        category and the specific keyword that matched. Keywords are
        listed longest-first within each category so that, e.g.,
        ``"confidentiality"`` is preferred over the shorter
        ``"confidential"`` when both could match at the same
        position.

        Group names cannot contain characters like spaces or hyphens,
        so each keyword is referenced by a numeric index rather than
        its literal text; the index is mapped back to the literal
        keyword via a side table built alongside the pattern.

        Args:
            categories: A mapping of category name to its keyword
                tuple, as produced by
                :attr:`RetrievalConfig.categories`.

        Returns:
            A single compiled, case-insensitive regular expression
            whose named groups are of the form
            ``f"{category}__{index}"``.
        """
        group_parts: list[str] = []
        ContractRetriever._group_to_keyword: dict[str, str] = {}
        ContractRetriever._group_to_category: dict[str, str] = {}

        for category, keywords in categories.items():
            for index, keyword in enumerate(
                sorted(keywords, key=len, reverse=True)
            ):
                group_name = f"{category}__{index}"
                group_parts.append(
                    rf"(?P<{group_name}>\b{re.escape(keyword)}\b)"
                )
                ContractRetriever._group_to_keyword[group_name] = keyword
                ContractRetriever._group_to_category[group_name] = category

        combined_pattern = "|".join(group_parts)
        return re.compile(combined_pattern, flags=re.IGNORECASE)

    # ------------------------------------------------------------------
    # Text normalization and keyword search
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize raw text prior to keyword search and extraction.

        Normalization is intentionally conservative: it standardizes
        line endings and strips trailing whitespace from each line and
        from the text as a whole, without otherwise reflowing,
        reordering, or removing content. This keeps keyword search
        (and the resulting character offsets used for windowing)
        simple, predictable, and fully deterministic, while avoiding
        cosmetic artifacts (stray ``\\r`` characters, trailing spaces)
        that could otherwise interfere with keyword matching at line
        boundaries.

        Args:
            text: The raw text to normalize.

        Returns:
            The normalized text. All subsequent search and windowing
            in :meth:`retrieve` operates on this normalized string, so
            offsets stay internally consistent.
        """
        unified_newlines = text.replace("\r\n", "\n").replace("\r", "\n")
        stripped_lines = [line.rstrip() for line in unified_newlines.split("\n")]
        return "\n".join(stripped_lines).strip()

    def _find_keyword_matches(self, normalized_text: str) -> list[KeywordMatch]:
        """Find every configured keyword occurrence in a single pass.

        Uses the combined, pre-compiled pattern built in
        :meth:`_compile_pattern` (once, at construction time) and a
        single ``finditer`` sweep over ``normalized_text`` to discover
        every keyword occurrence across all categories at once,
        rather than performing one pass per keyword or per category.

        Args:
            normalized_text: The already-normalized text to search.

        Returns:
            A list of :class:`KeywordMatch` instances, one per
            occurrence found, ordered by position in the text (this
            falls out naturally from ``finditer`` scanning
            left-to-right).
        """
        matches: list[KeywordMatch] = []
        for found in self._pattern.finditer(normalized_text):
            group_name = found.lastgroup
            if group_name is None:
                continue
            matches.append(
                KeywordMatch(
                    category=self._group_to_category[group_name],
                    keyword=self._group_to_keyword[group_name],
                    start=found.start(),
                    end=found.end(),
                )
            )
        return matches

    # ------------------------------------------------------------------
    # Window computation
    # ------------------------------------------------------------------

    def _window_for_match(
        self, match: KeywordMatch, normalized_text: str
    ) -> _ContextWindow:
        """Compute the paragraph-aware context window for a single match.

        First computes a raw character window of ``chars_before``
        characters before the match and ``chars_after`` characters
        after it, clipped to the bounds of the text. If
        ``config.expand_to_paragraph_boundaries`` is enabled, that raw
        window is then expanded backward to the previous blank line
        and forward to the next blank line, so that paragraphs are not
        cut in half where a blank line is available nearby.

        Args:
            match: The keyword match to build a window around.
            normalized_text: The full normalized text the match was
                found in, used both for clipping and for locating
                paragraph (blank-line) boundaries.

        Returns:
            A :class:`_ContextWindow` describing the final,
            paragraph-aware span to retain for this match.
        """
        text_length = len(normalized_text)
        window_start = max(0, match.start - self._config.chars_before)
        window_end = min(text_length, match.end + self._config.chars_after)

        if self._config.expand_to_paragraph_boundaries:
            window_start = self._expand_backward_to_blank_line(
                normalized_text, window_start
            )
            window_end = self._expand_forward_to_blank_line(
                normalized_text, window_end
            )

        return _ContextWindow(start=window_start, end=window_end)

    @staticmethod
    def _expand_backward_to_blank_line(text: str, position: int) -> int:
        """Move ``position`` backward to just after the previous blank line.

        A "blank line" is a ``"\\n\\n"`` sequence (two consecutive
        newlines) in the normalized text. If one is found before
        ``position``, the window is expanded to start immediately
        after it. If none is found, the window expands all the way to
        the start of the text.

        Args:
            text: The normalized text to search within.
            position: The character offset to expand backward from.

        Returns:
            The (possibly earlier) character offset to use as the new
            window start.
        """
        blank_line_index = text.rfind("\n\n", 0, position)
        if blank_line_index == -1:
            return 0
        return blank_line_index + 2

    @staticmethod
    def _expand_forward_to_blank_line(text: str, position: int) -> int:
        """Move ``position`` forward to just before the next blank line.

        A "blank line" is a ``"\\n\\n"`` sequence (two consecutive
        newlines) in the normalized text. If one is found at or after
        ``position``, the window is expanded to end right before it.
        If none is found, the window expands all the way to the end
        of the text.

        Args:
            text: The normalized text to search within.
            position: The character offset to expand forward from.

        Returns:
            The (possibly later) character offset to use as the new
            window end.
        """
        blank_line_index = text.find("\n\n", position)
        if blank_line_index == -1:
            return len(text)
        return blank_line_index

    # ------------------------------------------------------------------
    # Merging
    # ------------------------------------------------------------------

    def _merge_windows(
        self, sorted_windows: list[_ContextWindow]
    ) -> list[_ContextWindow]:
        """Merge overlapping, adjacent, or nearby windows into contiguous spans.

        Performs a single left-to-right sweep over ``sorted_windows``
        (which must already be sorted by start position), merging any
        window whose start falls at or before
        ``previous_window.end + config.merge_distance`` into the
        current merged span, and starting a new span otherwise. This
        both eliminates duplicate/overlapping windows and absorbs
        windows that are merely "close together" without directly
        touching, per ``config.merge_distance``.

        Args:
            sorted_windows: Windows sorted in ascending order by
                ``(start, end)``.

        Returns:
            A list of non-overlapping :class:`_ContextWindow`
            instances, in ascending order, each separated from the
            next by more than ``config.merge_distance`` characters.
        """
        merged: list[_ContextWindow] = []
        merge_distance = self._config.merge_distance

        for window in sorted_windows:
            if merged and window.start <= merged[-1].end + merge_distance:
                previous = merged[-1]
                merged[-1] = _ContextWindow(
                    start=previous.start, end=max(previous.end, window.end)
                )
            else:
                merged.append(window)

        return merged

    # ------------------------------------------------------------------
    # Assembly and truncation
    # ------------------------------------------------------------------

    def _assemble_context(
        self, normalized_text: str, merged_windows: list[_ContextWindow]
    ) -> tuple[str, bool]:
        """Slice, join, and (if necessary) truncate the merged windows.

        Windows are joined with ``config.window_separator`` wherever
        two windows are not directly adjacent. If the assembled result
        would exceed ``config.max_context_chars``, later windows are
        dropped or trimmed (earlier windows are always preserved in
        full, in line with the requirement to prefer keeping earlier
        merged windows) and ``config.truncation_notice`` is appended.

        Args:
            normalized_text: The normalized text the windows were
                computed against.
            merged_windows: The final, non-overlapping windows to
                extract, in ascending order.

        Returns:
            A tuple of ``(context, truncated)`` where ``context`` is
            the fully assembled (and possibly truncated) string, and
            ``truncated`` indicates whether truncation occurred.
        """
        max_chars = self._config.max_context_chars
        separator = self._config.window_separator
        notice = self._config.truncation_notice

        segments: list[str] = []
        running_length = 0
        truncated = False

        for index, window in enumerate(merged_windows):
            segment = normalized_text[window.start : window.end]
            addition = segment if index == 0 else separator + segment

            if running_length + len(addition) > max_chars:
                remaining_budget = max_chars - running_length - len(notice)
                if index > 0 and remaining_budget > len(separator):
                    trimmed = addition[: max(0, remaining_budget)]
                    segments.append(trimmed)
                elif remaining_budget > 0:
                    segments.append(segment[: max(0, remaining_budget)])
                truncated = True
                break

            segments.append(addition)
            running_length += len(addition)

        context = "".join(segments)
        if truncated:
            context += notice

        return context, truncated

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    @staticmethod
    def _log_summary(
        *,
        original_length: int,
        matches: Iterable[KeywordMatch],
        merged_window_count: int,
        retrieved_length: int,
        compression_ratio: float,
        truncated: bool,
    ) -> None:
        """Log a structured, human-readable summary of a retrieval pass.

        Args:
            original_length: Character length of the original input.
            matches: All keyword matches found during this pass.
            merged_window_count: Number of merged windows produced.
            retrieved_length: Character length of the final context.
            compression_ratio: Fraction of the original text removed.
            truncated: Whether the context had to be truncated.
        """
        matches = list(matches)
        counts_by_category: dict[str, int] = {}
        for match in matches:
            counts_by_category[match.category] = (
                counts_by_category.get(match.category, 0) + 1
            )

        logger.info("Original document size: %s chars", f"{original_length:,}")
        for category, count in sorted(counts_by_category.items()):
            logger.info("%s matches: %d", category.capitalize(), count)
        logger.info("Merged windows: %d", merged_window_count)
        logger.info("Retrieved context: %s chars", f"{retrieved_length:,}")
        logger.info("Compression ratio: %.0f%%", compression_ratio * 100)
        logger.info("Context truncated: %s", truncated)