"""Unit tests for src.extractor.ClauseExtractor.

The Anthropic API is never called: LLMClient is replaced with a mock
that returns pre-built LLMResponse objects.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from src.extractor import ClauseExtractor, JSONParsingError
from src.llm_client import LLMClient
from src.models import ContractDocument, LLMResponse


def _make_mock_llm_client(response_content: str) -> MagicMock:
    """Build a mock LLMClient whose generate() returns fixed content."""
    mock_client = MagicMock(spec=LLMClient)
    mock_client.generate.return_value = LLMResponse(
        content=response_content,
        model="mock-model",
        usage_tokens=42,
    )
    return mock_client


def test_extract_parses_valid_json(sample_contract_document: ContractDocument) -> None:
    """extract() should return a ClauseExtraction matching valid JSON content."""
    valid_json = json.dumps(
        {
            "termination_clause": "Either party may terminate with notice.",
            "confidentiality_clause": "Information shall remain confidential.",
            "liability_clause": None,
        }
    )
    extractor = ClauseExtractor(llm_client=_make_mock_llm_client(valid_json))

    result = extractor.extract(sample_contract_document)

    assert result.termination_clause == "Either party may terminate with notice."
    assert result.confidentiality_clause == "Information shall remain confidential."
    assert result.liability_clause is None


def test_extract_raises_on_invalid_json(sample_contract_document: ContractDocument) -> None:
    """extract() should raise JSONParsingError when the LLM response is not valid JSON."""
    extractor = ClauseExtractor(llm_client=_make_mock_llm_client("this is not json"))

    with pytest.raises(JSONParsingError):
        extractor.extract(sample_contract_document)


def test_extract_raises_on_missing_required_field(
    sample_contract_document: ContractDocument,
) -> None:
    """extract() should raise JSONParsingError when a required field is missing."""
    incomplete_json = json.dumps(
        {
            "termination_clause": "Either party may terminate with notice.",
            "confidentiality_clause": "Information shall remain confidential.",
        }
    )
    extractor = ClauseExtractor(llm_client=_make_mock_llm_client(incomplete_json))

    with pytest.raises(JSONParsingError):
        extractor.extract(sample_contract_document)


def test_extract_raises_on_wrong_value_type(sample_contract_document: ContractDocument) -> None:
    """extract() should raise JSONParsingError when a clause value is not a string or null."""
    wrong_type_json = json.dumps(
        {
            "termination_clause": 12345,
            "confidentiality_clause": "Information shall remain confidential.",
            "liability_clause": None,
        }
    )
    extractor = ClauseExtractor(llm_client=_make_mock_llm_client(wrong_type_json))

    with pytest.raises(JSONParsingError):
        extractor.extract(sample_contract_document)