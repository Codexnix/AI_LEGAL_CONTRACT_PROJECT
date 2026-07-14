"""Application entry point for the AI Contract Analysis Pipeline.

This module contains no business logic. It only configures logging,
ensures required directories exist, and coordinates the already-built
:class:`~src.pipeline.ContractAnalysisPipeline` and
:class:`~src.utils.OutputWriter` to run the full contract analysis
workflow end to end.
"""

from __future__ import annotations

import logging
import time

from src.config import settings
from src.pipeline import ContractAnalysisPipeline
from src.utils import OutputWriter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)


def main() -> None:
    """Run the end-to-end contract analysis pipeline.

    Ensures required directories exist, runs the contract analysis
    pipeline, and writes the results to JSON and CSV. If the pipeline
    produces no results (e.g. every document failed processing), this
    is treated as a non-fatal outcome: a warning is logged, output
    writing is skipped entirely, and the application exits gracefully.
    Any unexpected failure is logged with a full traceback and
    re-raised.

    On completion, a summary is logged reporting the number of
    documents processed, the number of documents that failed to
    process, the output location (when files were written), and the
    total execution time.

    Raises:
        Exception: Any exception raised during pipeline execution or
            output writing, after being logged.
    """
    application_start_time = time.perf_counter()
    logger.info("Application starting: AI Contract Analysis Pipeline")
    try:
        settings.create_directories()
        logger.info("Required directories verified/created")

        total_documents_found = _count_input_pdfs()

        pipeline = ContractAnalysisPipeline()
        logger.info("Pipeline started")
        results = pipeline.run()

        documents_processed = len(results)
        documents_failed = max(total_documents_found - documents_processed, 0)

        if not results:
            logger.warning(
                "No documents were successfully processed; output files "
                "(JSON and CSV) will not be generated."
            )
            output_location = "N/A (no output written)"
        else:
            writer = OutputWriter()
            json_path = writer.write_json(results)
            writer.write_csv(results)
            output_location = str(json_path.parent)
            logger.info("Outputs written (directory=%s)", output_location)

        logger.info("Pipeline completed successfully.")

        execution_seconds = time.perf_counter() - application_start_time
        logger.info(
            "Run summary: documents_processed=%d, documents_failed=%d, "
            "output_location=%s, execution_seconds=%.2f",
            documents_processed,
            documents_failed,
            output_location,
            execution_seconds,
        )
    except Exception as exc:
        execution_seconds = time.perf_counter() - application_start_time
        logger.exception(
            "Application failed after %.2f seconds: %s", execution_seconds, exc
        )
        raise

    logger.info(
        "Application finished (total execution time: %.2f seconds)",
        execution_seconds,
    )


def _count_input_pdfs() -> int:
    """Count PDF files available in the configured raw PDF directory.

    Used only to derive an approximate "documents failed" figure for
    the end-of-run summary (``total_found - documents_processed``).
    This does not affect, call into, or duplicate any pipeline
    orchestration logic — it is a read-only, best-effort count over the
    same directory :class:`~src.pdf_loader.PDFLoader` reads from.

    Returns:
        The number of files with a ``.pdf`` extension (case-insensitive)
        found directly in ``settings.raw_pdf_dir``, or ``0`` if the
        directory cannot be read for any reason.
    """
    raw_pdf_dir = settings.raw_pdf_dir
    try:
        return sum(
            1
            for path in raw_pdf_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".pdf"
        )
    except OSError as exc:
        logger.warning(
            "Could not count input PDFs in %s for summary reporting: %s",
            raw_pdf_dir,
            exc,
        )
        return 0


if __name__ == "__main__":
    main()