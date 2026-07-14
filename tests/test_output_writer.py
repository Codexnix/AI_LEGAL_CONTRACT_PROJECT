"""Unit tests for src.utils.OutputWriter.

Output paths are redirected into pytest's tmp_path by monkeypatching the
computed path properties on the Settings class, ensuring no test ever
writes into the real project's outputs/ directory.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from src.config import Settings
from src.models import PipelineResult
from src.utils import OutputWriter


@pytest.fixture
def output_writer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> OutputWriter:
    """Return an OutputWriter whose output files are redirected into tmp_path."""
    monkeypatch.setattr(Settings, "output_dir", property(lambda self: tmp_path))
    return OutputWriter()


def test_write_json_creates_file(
    output_writer: OutputWriter, sample_pipeline_result: PipelineResult
) -> None:
    """write_json() should create a JSON file at the configured path."""
    output_path = output_writer.write_json([sample_pipeline_result])

    assert output_path.exists()


def test_write_csv_creates_file(
    output_writer: OutputWriter, sample_pipeline_result: PipelineResult
) -> None:
    """write_csv() should create a CSV file at the configured path."""
    output_path = output_writer.write_csv([sample_pipeline_result])

    assert output_path.exists()


def test_write_json_raises_on_empty_results(output_writer: OutputWriter) -> None:
    """write_json() should raise ValueError when given an empty results list."""
    with pytest.raises(ValueError):
        output_writer.write_json([])


def test_write_csv_raises_on_empty_results(output_writer: OutputWriter) -> None:
    """write_csv() should raise ValueError when given an empty results list."""
    with pytest.raises(ValueError):
        output_writer.write_csv([])


def test_json_output_contains_expected_keys(
    output_writer: OutputWriter, sample_pipeline_result: PipelineResult
) -> None:
    """The written JSON should contain document, clauses, and summary keys."""
    output_path = output_writer.write_json([sample_pipeline_result])

    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert len(payload) == 1
    record = payload[0]
    assert set(record.keys()) == {"document", "clauses", "summary"}


def test_csv_output_contains_expected_columns(
    output_writer: OutputWriter, sample_pipeline_result: PipelineResult
) -> None:
    """The written CSV should contain exactly the expected column headers."""
    output_path = output_writer.write_csv([sample_pipeline_result])

    with output_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)

    assert reader.fieldnames == [
        "filename",
        "termination_clause",
        "confidentiality_clause",
        "liability_clause",
        "summary",
    ]
    assert len(rows) == 1