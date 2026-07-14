"""Unit tests for src.preprocess.Preprocessor."""

from __future__ import annotations

import unicodedata

import pytest

from src.models import ContractDocument
from src.preprocess import Preprocessor


def test_preprocess_populates_clean_text(sample_contract_document: ContractDocument) -> None:
    """preprocess() should populate clean_text and leave raw_text untouched."""
    preprocessor = Preprocessor()
    original_raw_text = sample_contract_document.raw_text

    result = preprocessor.preprocess(sample_contract_document)

    assert result is sample_contract_document
    assert result.raw_text == original_raw_text
    assert result.clean_text != ""


def test_preprocess_normalizes_unicode() -> None:
    """preprocess() should normalize text to NFKC form."""
    decomposed_text = unicodedata.normalize("NFD", "café agreement")
    document = ContractDocument(
        filename="unicode.pdf",
        file_path="data/raw_pdfs/unicode.pdf",
        raw_text=decomposed_text,
    )

    result = Preprocessor().preprocess(document)

    assert result.clean_text == unicodedata.normalize("NFKC", decomposed_text)


def test_preprocess_collapses_spaces_and_tabs() -> None:
    """preprocess() should collapse consecutive spaces and tabs into one space."""
    document = ContractDocument(
        filename="spacing.pdf",
        file_path="data/raw_pdfs/spacing.pdf",
        raw_text="Party  A   and\t\tParty B agree.",
    )

    result = Preprocessor().preprocess(document)

    assert "  " not in result.clean_text
    assert "\t" not in result.clean_text
    assert result.clean_text == "Party A and Party B agree."


def test_preprocess_collapses_blank_lines() -> None:
    """preprocess() should collapse three or more consecutive newlines into one blank line."""
    document = ContractDocument(
        filename="blank_lines.pdf",
        file_path="data/raw_pdfs/blank_lines.pdf",
        raw_text="Section 1.\n\n\n\nSection 2.",
    )

    result = Preprocessor().preprocess(document)

    assert result.clean_text == "Section 1.\n\nSection 2."


def test_preprocess_raises_value_error_on_empty_raw_text() -> None:
    """preprocess() should raise ValueError when raw_text is empty or whitespace-only."""
    document = ContractDocument(
        filename="empty.pdf",
        file_path="data/raw_pdfs/empty.pdf",
        raw_text="   ",
    )

    with pytest.raises(ValueError):
        Preprocessor().preprocess(document)