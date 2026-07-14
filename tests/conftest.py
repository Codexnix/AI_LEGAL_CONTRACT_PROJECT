"""Shared pytest fixtures for the AI Contract Analysis Pipeline test suite."""

from __future__ import annotations

import pytest

from src.models import (
    ClauseExtraction,
    ContractDocument,
    ContractSummary,
    PipelineResult,
)


@pytest.fixture
def sample_contract_document() -> ContractDocument:
    """Return a ContractDocument with populated raw and clean text."""
    return ContractDocument(
        filename="sample_contract.pdf",
        file_path="data/raw_pdfs/sample_contract.pdf",
        raw_text="This   Agreement   is   entered   into   by   the   parties.",
        clean_text="This Agreement is entered into by the parties.",
    )


@pytest.fixture
def sample_clause_extraction() -> ClauseExtraction:
    """Return a ClauseExtraction with all three clauses populated."""
    return ClauseExtraction(
        termination_clause="Either party may terminate this Agreement with 30 days notice.",
        confidentiality_clause="Each party shall keep confidential information secret.",
        liability_clause="Neither party shall be liable for indirect damages.",
    )


@pytest.fixture
def sample_contract_summary() -> ContractSummary:
    """Return a ContractSummary with a short placeholder summary."""
    return ContractSummary(
        summary="This agreement establishes a business relationship between the parties, "
        "outlining termination rights, confidentiality obligations, and liability limits."
    )


@pytest.fixture
def sample_pipeline_result(
    sample_contract_document: ContractDocument,
    sample_clause_extraction: ClauseExtraction,
    sample_contract_summary: ContractSummary,
) -> PipelineResult:
    """Return a fully populated PipelineResult composed of the other fixtures."""
    return PipelineResult(
        document=sample_contract_document,
        clauses=sample_clause_extraction,
        summary=sample_contract_summary,
    )