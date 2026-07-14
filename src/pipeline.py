"""Pipeline orchestration for the AI Contract Analysis Pipeline.

This module is responsible solely for coordinating the existing
pipeline stages — PDF loading, preprocessing, clause extraction, and
summarization — into a single end-to-end workflow. It contains no
prompt text, no direct PDF parsing, no direct text cleaning, no direct
LLM SDK usage, and no output persistence; all of that is delegated to
the already-completed stage modules.
"""

from __future__ import annotations

import logging
import time

from src.extractor import ClauseExtractionError, ClauseExtractor
from src.models import ContractDocument, PipelineResult
from src.pdf_loader import PDFLoader
from src.preprocess import Preprocessor
from src.summarizer import ContractSummarizer, SummaryGenerationError

logger = logging.getLogger(__name__)


class ContractAnalysisPipeline:
    """Orchestrates the end-to-end contract analysis workflow.

    Coordinates :class:`~src.pdf_loader.PDFLoader`,
    :class:`~src.preprocess.Preprocessor`,
    :class:`~src.extractor.ClauseExtractor`, and
    :class:`~src.summarizer.ContractSummarizer` to turn a directory of
    contract PDFs into a list of :class:`~src.models.PipelineResult`
    objects. A single document's failure is logged and skipped without
    aborting the rest of the batch.
    """

    def __init__(
        self,
        pdf_loader: PDFLoader | None = None,
        preprocessor: Preprocessor | None = None,
        extractor: ClauseExtractor | None = None,
        summarizer: ContractSummarizer | None = None,
    ) -> None:
        """Initialize the pipeline with its stage dependencies.

        Args:
            pdf_loader: An optional pre-configured
                :class:`~src.pdf_loader.PDFLoader`. If not provided, a
                new instance is created using default configuration.
            preprocessor: An optional pre-configured
                :class:`~src.preprocess.Preprocessor`. If not provided,
                a new instance is created.
            extractor: An optional pre-configured
                :class:`~src.extractor.ClauseExtractor`. If not
                provided, a new instance is created using default
                configuration.
            summarizer: An optional pre-configured
                :class:`~src.summarizer.ContractSummarizer`. If not
                provided, a new instance is created using default
                configuration.
        """
        self._pdf_loader = pdf_loader or PDFLoader()
        self._preprocessor = preprocessor or Preprocessor()
        self._extractor = extractor or ClauseExtractor()
        self._summarizer = summarizer or ContractSummarizer()
        logger.info("ContractAnalysisPipeline initialized")

    def run(self, limit: int = 50) -> list[PipelineResult]:
        """Run the full contract analysis workflow.

        Loads up to ``limit`` contract PDFs and processes each one
        through preprocessing, clause extraction, and summarization.
        A document that fails at any stage is logged and skipped; it
        never aborts processing of the remaining documents.

        Args:
            limit: Maximum number of contract PDFs to load and process.

        Returns:
            A list of :class:`~src.models.PipelineResult` objects, one
            per successfully processed document. Empty if every
            document failed to process.
        """
        if limit <= 0:
            raise ValueError("limit must be greater than 0.")

        logger.info("Pipeline started (limit=%d)", limit)
        pipeline_start_time = time.perf_counter()
        documents = self._pdf_loader.load_documents(limit)

        results: list[PipelineResult] = []
        successful_filenames: list[str] = []
        failed_filenames: list[str] = []

        for document in documents:
            document_start_time = time.perf_counter()
            try:
                result = self._process_document(document)
            except (ValueError, ClauseExtractionError, SummaryGenerationError):
                document_seconds = time.perf_counter() - document_start_time
                failed_filenames.append(document.filename)
                logger.exception(
                    "Document processing failed (filename=%s, duration_seconds=%.2f)",
                    document.filename,
                    document_seconds,
                )
                continue

            document_seconds = time.perf_counter() - document_start_time
            results.append(result)
            successful_filenames.append(document.filename)
            logger.info(
                "Document processed (filename=%s, duration_seconds=%.2f)",
                document.filename,
                document_seconds,
            )

        pipeline_seconds = time.perf_counter() - pipeline_start_time

        if results:
            logger.info(
                "Pipeline completed (successful=%d, failed=%d, total=%d, "
                "duration_seconds=%.2f)",
                len(results),
                len(documents) - len(results),
                len(documents),
                pipeline_seconds,
            )
        else:
            logger.warning(
                "Pipeline completed but no documents were successfully processed "
                "(failed=%d, total=%d, duration_seconds=%.2f)",
                len(documents) - len(results),
                len(documents),
                pipeline_seconds,
            )

        if successful_filenames:
            logger.info("Successfully processed documents: %s", successful_filenames)
        if failed_filenames:
            logger.warning("Failed to process documents: %s", failed_filenames)

        return results

    def _process_document(self, document: ContractDocument) -> PipelineResult:
        """Run a single document through preprocessing, extraction, and summarization.

        Args:
            document: The contract document to process.

        Returns:
            A :class:`~src.models.PipelineResult` combining the
            preprocessed document, its extracted clauses, and its
            summary.
        """
        preprocessed_document = self._preprocessor.preprocess(document)
        clauses = self._extractor.extract(preprocessed_document)
        summary = self._summarizer.summarize(preprocessed_document)
        return PipelineResult(
            document=preprocessed_document,
            clauses=clauses,
            summary=summary,
        )