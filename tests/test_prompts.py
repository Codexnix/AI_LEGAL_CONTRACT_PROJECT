"""Unit tests for src.prompts.PromptBuilder."""

from __future__ import annotations

import pytest

from src.prompts import PromptBuilder

_CONTRACT_TEXT = "This Agreement is entered into by Acme Corp and Beta LLC."


def test_extraction_prompt_contains_contract_text() -> None:
    """build_extraction_prompt() should embed the provided contract text."""
    prompt = PromptBuilder.build_extraction_prompt(_CONTRACT_TEXT)

    assert _CONTRACT_TEXT in prompt


def test_extraction_prompt_rejects_empty_text() -> None:
    """build_extraction_prompt() should raise ValueError for empty contract text."""
    with pytest.raises(ValueError):
        PromptBuilder.build_extraction_prompt("   ")


def test_summary_prompt_contains_contract_text() -> None:
    """build_summary_prompt() should embed the provided contract text."""
    prompt = PromptBuilder.build_summary_prompt(_CONTRACT_TEXT)

    assert _CONTRACT_TEXT in prompt


def test_summary_prompt_rejects_empty_text() -> None:
    """build_summary_prompt() should raise ValueError for empty contract text."""
    with pytest.raises(ValueError):
        PromptBuilder.build_summary_prompt("")