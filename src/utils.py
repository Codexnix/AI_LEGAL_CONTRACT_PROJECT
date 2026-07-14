"""Output persistence for the AI Contract Analysis Pipeline.

This module is responsible solely for writing a list of
:class:`~src.models.PipelineResult` objects to disk as JSON and CSV. It
performs no PDF loading, no LLM calls, no prompt construction, no text
preprocessing, and no pipeline orchestration.

The JSON and CSV exports are intentionally lean: the large `raw_text`
and `clean_text` fields captured on :class:`~src.models.ContractDocument`
are excluded from both outputs, since they are not needed downstream and
can be very large. Each output record instead carries the document's
`filename`, its extracted `clauses`, its `summary`, and lightweight
processing metadata (`processing_model`, `processing_timestamp`).
"""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from src.config import settings
from src.models import PipelineResult

logger = logging.getLogger(__name__)

_CSV_FIELDNAMES: tuple[str, ...] = (
    "filename",
    "termination_clause",
    "confidentiality_clause",
    "liability_clause",
    "summary",
    "processing_model",
    "processing_timestamp",
)

_UNKNOWN_MODEL_LABEL = "unknown"


class OutputWriter:
    """Writes pipeline results to JSON and CSV files.

    Reads output file locations from :data:`src.config.settings` and
    assumes the required output directory already exists (created via
    ``settings.create_directories()`` elsewhere in the application).

    Both outputs omit the large `raw_text`/`clean_text` document fields
    and instead include only `filename`, `clauses`, `summary`, and
    processing metadata (`processing_model`, `processing_timestamp`).
    """

    def __init__(self) -> None:
        """Initialize the writer using configured output file paths."""
        self._json_output_file = settings.json_output_file
        self._csv_output_file = settings.csv_output_file
        logger.info("OutputWriter initialized")

    def write_json(self, results: list[PipelineResult]) -> Path:
        """Write pipeline results to a pretty-formatted JSON file.

        Each record contains only `filename`, `clauses`, `summary`, and
        processing metadata (`processing_model`,
        `processing_timestamp`); the large `raw_text` and `clean_text`
        fields are deliberately omitted.

        Args:
            results: The pipeline results to write.

        Returns:
            The path to the written JSON file.

        Raises:
            ValueError: If ``results`` is empty.
            OSError: If the JSON file cannot be written to disk.
        """
        self._validate_results(results)

        processing_model = self._resolve_processing_model()
        processing_timestamp = self._current_timestamp()
        payload = [
            self._to_json_record(result, processing_model, processing_timestamp)
            for result in results
        ]

        try:
            with self._json_output_file.open("w", encoding="utf-8") as json_file:
                json.dump(payload, json_file, indent=2, ensure_ascii=False)
                json_file.write("\n")
        except OSError as exc:
            logger.error(
                "Failed to write JSON output (path=%s): %s",
                self._json_output_file,
                exc,
            )
            raise OSError(
                f"Could not write JSON output to '{self._json_output_file}': {exc}"
            ) from exc

        logger.info(
            "JSON written (path=%s, records=%d)", self._json_output_file, len(results)
        )
        return self._json_output_file

    def write_csv(self, results: list[PipelineResult]) -> Path:
        """Write pipeline results to a flattened CSV file.

        Includes `filename`, each extracted clause, `summary`, and
        processing metadata (`processing_model`,
        `processing_timestamp`) as columns. All fields are quoted to
        safely handle clause and summary text that may contain commas,
        quotes, or embedded newlines.

        Args:
            results: The pipeline results to write.

        Returns:
            The path to the written CSV file.

        Raises:
            ValueError: If ``results`` is empty.
            OSError: If the CSV file cannot be written to disk.
        """
        self._validate_results(results)

        processing_model = self._resolve_processing_model()
        processing_timestamp = self._current_timestamp()

        try:
            with self._csv_output_file.open(
                "w", encoding="utf-8", newline=""
            ) as csv_file:
                writer = csv.DictWriter(
                    csv_file,
                    fieldnames=_CSV_FIELDNAMES,
                    quoting=csv.QUOTE_ALL,
                    lineterminator="\n",
                )
                writer.writeheader()
                for result in results:
                    writer.writerow(
                        self._to_csv_row(result, processing_model, processing_timestamp)
                    )
        except OSError as exc:
            logger.error(
                "Failed to write CSV output (path=%s): %s",
                self._csv_output_file,
                exc,
            )
            raise OSError(
                f"Could not write CSV output to '{self._csv_output_file}': {exc}"
            ) from exc

        logger.info(
            "CSV written (path=%s, records=%d)", self._csv_output_file, len(results)
        )
        return self._csv_output_file

    @staticmethod
    def _validate_results(results: list[PipelineResult]) -> None:
        """Ensure the results list is non-empty.

        Args:
            results: The pipeline results to validate.

        Raises:
            ValueError: If ``results`` is empty.
        """
        if not results:
            raise ValueError(
                "No pipeline results were available because every document "
                "failed processing."
            )

    @staticmethod
    def _resolve_processing_model() -> str:
        """Resolve the model identifier used for this processing run.

        Returns:
            The configured LLM model identifier, or a fallback label if
            it is not available in configuration.
        """
        model = getattr(settings, "llm_model", None)
        if not model:
            logger.warning(
                "Could not resolve a configured LLM model; using fallback "
                "label %r for processing_model metadata.",
                _UNKNOWN_MODEL_LABEL,
            )
            return _UNKNOWN_MODEL_LABEL
        return model

    @staticmethod
    def _current_timestamp() -> str:
        """Build an ISO 8601 UTC timestamp for the current output write.

        Returns:
            The current UTC time formatted as an ISO 8601 string.
        """
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _to_json_record(
        result: PipelineResult, processing_model: str, processing_timestamp: str
    ) -> dict[str, object]:
        """Build a single lean JSON record for a PipelineResult.

        Deliberately excludes ``result.document.raw_text`` and
        ``result.document.clean_text``, keeping only the filename,
        extracted clauses, summary, and processing metadata.

        Args:
            result: The pipeline result to convert.
            processing_model: The model identifier used to process this
                batch.
            processing_timestamp: The ISO 8601 UTC timestamp for this
                output write.

        Returns:
            A JSON-serializable dictionary containing `filename`,
            `clauses`, `summary`, `processing_model`, and
            `processing_timestamp`.
        """
        return {
            "filename": result.document.filename,
            "clauses": result.clauses.model_dump(mode="json"),
            "summary": result.summary.model_dump(mode="json"),
            "processing_model": processing_model,
            "processing_timestamp": processing_timestamp,
        }

    @staticmethod
    def _to_csv_row(
        result: PipelineResult, processing_model: str, processing_timestamp: str
    ) -> dict[str, str | None]:
        """Flatten a single PipelineResult into a CSV row.

        Args:
            result: The pipeline result to flatten.
            processing_model: The model identifier used to process this
                batch.
            processing_timestamp: The ISO 8601 UTC timestamp for this
                output write.

        Returns:
            A dictionary mapping each CSV column name to its value.
        """
        return {
            "filename": result.document.filename,
            "termination_clause": result.clauses.termination_clause,
            "confidentiality_clause": result.clauses.confidentiality_clause,
            "liability_clause": result.clauses.liability_clause,
            "summary": result.summary.summary,
            "processing_model": processing_model,
            "processing_timestamp": processing_timestamp,
        }