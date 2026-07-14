"""Strongly typed data models for the AI Contract Analysis Pipeline.

This module defines the Pydantic v2 schemas that flow through every stage
of the pipeline: document ingestion, preprocessing, clause extraction,
summarization, and LLM response handling.

These models contain no business logic. They exist purely to provide
strict, validated, self-documenting data contracts between pipeline
stages (``pdf_loader`` -> ``preprocess`` -> ``extractor`` ->
``summarizer`` -> ``pipeline``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ContractDocument(BaseModel):
    """Represents a single contract document as it moves through the pipeline.

    Captures the document's origin on disk as well as both the raw text
    extracted from the PDF and the normalized/cleaned version produced by
    the preprocessing stage.
    """

    model_config = ConfigDict(extra="forbid", frozen=False)

    filename: str = Field(
        ...,
        min_length=1,
        description="Original filename of the source PDF, e.g. 'contract_01.pdf'.",
    )
    file_path: Path = Field(
        ...,
        description="Absolute or relative filesystem path to the source PDF on disk.",
    )
    raw_text: str = Field(
        default="",
        description=(
            "Unmodified text extracted directly from the PDF, prior to any "
            "normalization or cleaning."
        ),
    )
    clean_text: str = Field(
        default="",
        description=(
            "Normalized text produced by the preprocessing stage (whitespace "
            "collapsed, encoding artifacts removed, ready for LLM input)."
        ),
    )

    @field_validator("filename")
    @classmethod
    def validate_filename_extension(cls, value: str) -> str:
        """Ensure the filename refers to a PDF document.

        Args:
            value: The filename to validate.

        Returns:
            The validated filename unchanged.

        Raises:
            ValueError: If the filename does not end with '.pdf' (case-insensitive).
        """
        if not value.lower().endswith(".pdf"):
            raise ValueError(f"filename must have a '.pdf' extension, got: {value!r}")
        return value


class ClauseDetail(BaseModel):
    """Represents a single extracted contract clause with structured provenance.

    Captures not just the clause's verbatim text but, where the LLM can
    identify them, the section it came from and the model's confidence
    in the extraction. All fields besides ``text`` are optional because
    contracts vary widely in formatting — many have no numbered sections
    or headings at all, and a model may not always report a confidence
    score.
    """

    model_config = ConfigDict(extra="forbid")

    text: str = Field(
        ...,
        min_length=1,
        description="Verbatim text of the extracted clause, as it appears in the contract.",
    )
    section_title: str | None = Field(
        default=None,
        description=(
            "Title or heading of the section the clause was extracted from "
            "(e.g. 'Termination'), if the source document has one and it "
            "could be identified."
        ),
    )
    section_number: str | None = Field(
        default=None,
        description=(
            "Section or clause number the text was extracted from (e.g. "
            "'8.2' or 'Section IV'), if the source document numbers its "
            "sections and it could be identified. Stored as a string "
            "rather than a number since contracts use varied numbering "
            "schemes (e.g. '8.2', 'IV', '3(a)')."
        ),
    )
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Optional model-reported confidence that this text correctly "
            "represents the clause type, on a 0.0-1.0 scale. None if the "
            "extraction process does not produce a confidence score."
        ),
    )


class ClauseExtraction(BaseModel):
    """Holds the three key clauses extracted from a contract by the LLM.

    Each field is nullable because not every contract contains every
    clause type. A ``None`` value indicates the clause was genuinely
    absent from the source document, not that extraction failed.

    Each present clause is a structured :class:`ClauseDetail` capturing
    the verbatim text plus optional section metadata and confidence.
    For backward compatibility with callers that only have a plain
    clause string (e.g. simple extraction pipelines that don't identify
    section metadata), a bare string is also accepted and is
    automatically wrapped into a :class:`ClauseDetail` with only
    ``text`` populated.
    """

    model_config = ConfigDict(extra="forbid")

    termination_clause: ClauseDetail | None = Field(
        default=None,
        description=(
            "The contract's termination clause, or None if no such clause "
            "is present in the document. Accepts either a ClauseDetail "
            "object or a plain string (which is wrapped into a "
            "ClauseDetail with only its text populated)."
        ),
    )
    confidentiality_clause: ClauseDetail | None = Field(
        default=None,
        description=(
            "The contract's confidentiality clause, or None if no such "
            "clause is present in the document. Accepts either a "
            "ClauseDetail object or a plain string (which is wrapped into "
            "a ClauseDetail with only its text populated)."
        ),
    )
    liability_clause: ClauseDetail | None = Field(
        default=None,
        description=(
            "The contract's liability clause, or None if no such clause "
            "is present in the document. Accepts either a ClauseDetail "
            "object or a plain string (which is wrapped into a "
            "ClauseDetail with only its text populated)."
        ),
    )

    @field_validator(
        "termination_clause",
        "confidentiality_clause",
        "liability_clause",
        mode="before",
    )
    @classmethod
    def coerce_plain_string_to_clause_detail(cls, value: Any) -> Any:
        """Wrap a bare clause string into a ClauseDetail for backward compatibility.

        Existing callers (e.g. ``ClauseExtractor``) may only have a
        verbatim clause string with no section metadata or confidence
        score. Rather than requiring every caller to construct a
        :class:`ClauseDetail`, a plain, non-empty string is transparently
        wrapped as ``ClauseDetail(text=value)``. ``None``, ``ClauseDetail``
        instances, and already-structured dicts pass through unchanged
        for normal Pydantic validation to handle.

        Args:
            value: The raw field value being validated: ``None``, a
                plain string, a dict of clause detail fields, or an
                existing ``ClauseDetail`` instance.

        Returns:
            The value unchanged, or a ``{"text": value}`` dict when a
            non-empty plain string was supplied.
        """
        if isinstance(value, str):
            return {"text": value}
        return value


class ContractSummary(BaseModel):
    """Represents an LLM-generated natural language summary of a contract."""

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(
        ...,
        min_length=1,
        description="Human-readable summary of the contract, targeted at 100-150 words.",
    )


class PipelineResult(BaseModel):
    """Aggregates the full output of the pipeline for a single contract.

    Combines the source document, its extracted clauses, and its
    generated summary into one cohesive, serializable record suitable
    for writing to CSV or JSON.
    """

    model_config = ConfigDict(extra="forbid")

    document: ContractDocument = Field(
        ..., description="The source contract document and its extracted/cleaned text."
    )
    clauses: ClauseExtraction = Field(
        ..., description="The clauses extracted from the document by the LLM."
    )
    summary: ContractSummary = Field(
        ..., description="The generated summary of the document."
    )


class LLMResponse(BaseModel):
    """Represents a raw response returned by an LLM provider call.

    Used as a normalized wrapper around provider-specific SDK responses so
    that downstream code (extractor, summarizer) depends on a stable,
    provider-agnostic shape rather than raw SDK objects.
    """

    model_config = ConfigDict(extra="forbid")

    content: str = Field(
        ..., description="The raw text content returned by the LLM for this call."
    )
    model: str = Field(
        ...,
        min_length=1,
        description="Identifier of the LLM model that produced this response, e.g. 'claude-sonnet-4-6'.",
    )
    usage_tokens: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Total number of tokens consumed by this call (prompt + completion), "
            "if reported by the provider."
        ),
    )