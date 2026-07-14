"""FastAPI web application for the AI Contract Analysis Pipeline.

This module exposes the existing, unmodified pipeline (``src.pipeline``)
and output writer (``src.utils``) over HTTP so that contracts can be
uploaded, analyzed, and downloaded through a simple browser UI.

No pipeline, model, or configuration code is modified by this module —
it only imports and orchestrates the existing components:

- ``src.config.settings`` for all filesystem paths.
- ``src.pipeline.ContractAnalysisPipeline`` to run the analysis.
- ``src.utils.OutputWriter`` to persist JSON/CSV results.

Every JSON response returned by this API follows one consistent shape::

    {
        "success": bool,
        "message": str,
        "processed": int,
        "failed": int
    }

``processed``/``failed`` are ``0`` for endpoints where they don't apply
(e.g. a plain download or health check) so callers can rely on the shape
being present on every response, including error responses raised via
``HTTPException``.

Run locally with::

    uvicorn api:app --reload

Requires ``fastapi``, ``uvicorn``, ``jinja2``, and ``python-multipart``
(for multipart file uploads) in addition to the pipeline's existing
dependencies.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException, UploadFile, status
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from src.config import settings
from src.pipeline import ContractAnalysisPipeline
from src.utils import OutputWriter

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
# Reuses the log directory already created by src.config.Settings, so no
# existing configuration code needs to change to support this module.
logger = logging.getLogger("api")
logger.setLevel(logging.INFO)

if not logger.handlers:
    _formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    _stream_handler = logging.StreamHandler()
    _stream_handler.setFormatter(_formatter)
    logger.addHandler(_stream_handler)

    try:
        _file_handler = logging.FileHandler(settings.log_file, encoding="utf-8")
        _file_handler.setFormatter(_formatter)
        logger.addHandler(_file_handler)
    except OSError:  # pragma: no cover - filesystem/permission issues
        logger.warning("Could not attach file handler at %s.", settings.log_file)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_APP_ROOT = Path(__file__).resolve().parent
_TEMPLATES_DIR = _APP_ROOT / "frontend" / "templates"
_STATIC_DIR = _APP_ROOT / "frontend" / "static"

_ALLOWED_CONTENT_TYPE = "application/pdf"
_ALLOWED_EXTENSION = ".pdf"
_MAX_FILES_PER_BATCH = 20

# ---------------------------------------------------------------------------
# FastAPI application setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="AI Contract Analysis Pipeline API",
    description=(
        "HTTP interface for uploading legal contract PDFs, running the "
        "clause-extraction and summarization pipeline, and downloading "
        "the resulting JSON/CSV reports."
    ),
    version="1.0.0",
)

templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

# Ensure the static directory exists before mounting, so a missing
# ``frontend/static`` folder doesn't crash app startup.
_STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Response helpers
# ---------------------------------------------------------------------------
def _build_payload(
    success: bool,
    message: str,
    processed: int = 0,
    failed: int = 0,
) -> dict:
    """Build the standard response body used by every JSON endpoint.

    Keeping this in one place guarantees every success and error response
    exposes the same four keys, regardless of which endpoint produced it.

    Args:
        success: Whether the operation succeeded.
        message: A short, human-readable description of the outcome.
        processed: Count of items successfully processed, if applicable.
        failed: Count of items that failed, if applicable.

    Returns:
        A dict with keys ``success``, ``message``, ``processed``, ``failed``.
    """
    return {
        "success": success,
        "message": message,
        "processed": processed,
        "failed": failed,
    }


def _http_error(status_code: int, message: str) -> HTTPException:
    """Raise-ready ``HTTPException`` whose body follows the standard shape.

    FastAPI serializes ``HTTPException.detail`` directly as the response
    body when it's not a string, so passing our standard payload dict here
    ensures error responses have the same shape as success responses.

    Args:
        status_code: The HTTP status code to respond with.
        message: A short, human-readable error description.

    Returns:
        An ``HTTPException`` ready to be raised.
    """
    return HTTPException(
        status_code=status_code,
        detail=_build_payload(success=False, message=message),
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    """Render the application's landing page.

    Args:
        request: The incoming request, required by Jinja2Templates for
            URL generation within the rendered template.

    Returns:
        The rendered ``index.html`` template.
    """
    logger.info("Serving index page.")
    return templates.TemplateResponse(request=request, name="index.html", context={})


@app.post("/upload")
async def upload_pdfs(files: List[UploadFile]) -> JSONResponse:
    """Upload one or more contract PDFs, replacing any existing PDFs.

    All existing files in ``settings.raw_pdf_dir`` are removed before the
    newly uploaded PDFs are saved, so each call to this endpoint fully
    replaces the pipeline's input set.

    Args:
        files: One or more uploaded files, all of which must be non-empty
            PDFs. Up to ``_MAX_FILES_PER_BATCH`` files are accepted per
            call.

    Returns:
        A JSON response following the standard payload shape, with
        ``processed`` set to the number of PDFs saved and ``failed``
        always ``0`` on success (any invalid file rejects the whole
        batch rather than partially succeeding).

    Raises:
        HTTPException: 400 if no files were provided, if too many files
            were provided, if any file is empty, or if any file is not a
            PDF. 500 if the files could not be written to disk.
    """
    if not files:
        logger.warning("Upload rejected: no files provided.")
        raise _http_error(
            status.HTTP_400_BAD_REQUEST,
            "No files were uploaded. Please select at least one PDF.",
        )

    if len(files) > _MAX_FILES_PER_BATCH:
        logger.warning(
            "Upload rejected: %d files exceed the %d-file limit.",
            len(files),
            _MAX_FILES_PER_BATCH,
        )
        raise _http_error(
            status.HTTP_400_BAD_REQUEST,
            f"Too many files: {len(files)} were provided, but the limit is "
            f"{_MAX_FILES_PER_BATCH} PDFs per batch.",
        )

    for upload in files:
        filename = upload.filename or ""

        if not filename:
            logger.warning("Upload rejected: a file was submitted without a filename.")
            raise _http_error(
                status.HTTP_400_BAD_REQUEST,
                "One of the uploaded files is missing a filename.",
            )

        is_pdf_extension = filename.lower().endswith(_ALLOWED_EXTENSION)
        is_pdf_content_type = upload.content_type == _ALLOWED_CONTENT_TYPE

        if not is_pdf_extension or not is_pdf_content_type:
            logger.warning("Upload rejected: '%s' is not a PDF file.", filename)
            raise _http_error(
                status.HTTP_400_BAD_REQUEST,
                f"'{filename}' is not a PDF file. Only .pdf files are accepted.",
            )

    try:
        settings.raw_pdf_dir.mkdir(parents=True, exist_ok=True)

        # Replace old PDFs with the newly uploaded set.
        removed_count = 0
        for existing_pdf in settings.raw_pdf_dir.glob("*.pdf"):
            existing_pdf.unlink()
            removed_count += 1
        logger.info("Removed %d existing PDF(s) from raw_pdf_dir.", removed_count)

        saved_count = 0
        for upload in files:
            contents = await upload.read()

            if not contents:
                logger.warning("Upload rejected: '%s' is empty.", upload.filename)
                raise _http_error(
                    status.HTTP_400_BAD_REQUEST,
                    f"'{upload.filename}' is empty and cannot be processed.",
                )

            destination = settings.raw_pdf_dir / Path(upload.filename).name
            destination.write_bytes(contents)
            saved_count += 1
            logger.info("Saved uploaded PDF to '%s'.", destination)

    except HTTPException:
        raise
    except OSError as exc:
        logger.exception("Failed to save uploaded PDFs.")
        raise _http_error(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            f"Failed to save uploaded PDFs due to a filesystem error: {exc}",
        ) from exc

    logger.info("Upload complete: %d PDF(s) saved.", saved_count)

    plural = "" if saved_count == 1 else "s"
    message = f"{saved_count} PDF{plural} uploaded successfully."

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=_build_payload(success=True, message=message, processed=saved_count, failed=0),
    )


@app.post("/analyze")
async def analyze_contracts() -> JSONResponse:
    """Run the contract analysis pipeline and persist JSON/CSV results.

    Invokes the existing, unmodified ``ContractAnalysisPipeline`` against
    whatever PDFs currently exist in ``settings.raw_pdf_dir``, then writes
    the results using the existing ``OutputWriter``.

    Returns:
        A JSON response following the standard payload shape, where
        ``processed`` is the number of documents successfully analyzed
        and ``failed`` is the number of input PDFs that failed
        processing.

    Raises:
        HTTPException: 400 if there are no PDFs to analyze. 500 with a
            descriptive message if the pipeline run or output writing
            fails.
    """
    try:
        total_inputs = len(list(settings.raw_pdf_dir.glob("*.pdf")))

        if total_inputs == 0:
            logger.warning("Analyze rejected: no PDFs found in raw_pdf_dir.")
            raise _http_error(
                status.HTTP_400_BAD_REQUEST,
                "No PDFs are available to analyze. Please upload contracts first.",
            )

        logger.info("Starting pipeline run on %d input PDF(s).", total_inputs)

        pipeline = ContractAnalysisPipeline()
        results = pipeline.run()

        processed = len(results)
        failed = max(total_inputs - processed, 0)

        if processed == 0:
            logger.warning(
                "Pipeline produced no successful results out of %d input(s); "
                "skipping output writing.",
                total_inputs,
            )
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content=_build_payload(
                    success=True,
                    message=(
                        f"Analysis finished, but all {failed} document(s) failed "
                        "to process. No reports were generated."
                    ),
                    processed=processed,
                    failed=failed,
                ),
            )

        writer = OutputWriter()
        writer.write_json(results)
        writer.write_csv(results)

        logger.info(
            "Pipeline run complete: %d processed, %d failed.", processed, failed
        )

        if failed > 0:
            message = (
                f"Analysis complete: {processed} document(s) processed "
                f"successfully, {failed} failed. Reports are ready to download."
            )
        else:
            message = (
                f"Analysis complete: all {processed} document(s) processed "
                "successfully. Reports are ready to download."
            )

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=_build_payload(
                success=True, message=message, processed=processed, failed=failed
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 - surfaced to the client deliberately
        logger.exception("Pipeline run failed.")
        raise _http_error(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            f"Pipeline run failed: {exc}",
        ) from exc


@app.get("/download/json")
async def download_json() -> FileResponse:
    """Download the most recently generated JSON results file.

    Returns:
        The ``results.json`` file as a downloadable attachment.

    Raises:
        HTTPException: 404 if no JSON results file has been generated yet.
    """
    if not settings.json_output_file.exists():
        logger.warning("JSON download requested but file does not exist.")
        raise _http_error(
            status.HTTP_404_NOT_FOUND,
            "No JSON results are available yet. Run /analyze first.",
        )

    logger.info("Serving JSON results file.")
    return FileResponse(
        path=settings.json_output_file,
        media_type="application/json",
        filename=settings.json_output_file.name,
    )


@app.get("/download/csv")
async def download_csv() -> FileResponse:
    """Download the most recently generated CSV results file.

    Returns:
        The ``results.csv`` file as a downloadable attachment.

    Raises:
        HTTPException: 404 if no CSV results file has been generated yet.
    """
    if not settings.csv_output_file.exists():
        logger.warning("CSV download requested but file does not exist.")
        raise _http_error(
            status.HTTP_404_NOT_FOUND,
            "No CSV results are available yet. Run /analyze first.",
        )

    logger.info("Serving CSV results file.")
    return FileResponse(
        path=settings.csv_output_file,
        media_type="text/csv",
        filename=settings.csv_output_file.name,
    )


@app.get("/health")
async def health_check() -> JSONResponse:
    """Report basic service health.

    Returns:
        A JSON response following the standard payload shape, with
        ``processed``/``failed`` always ``0`` since they don't apply here.
    """
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=_build_payload(success=True, message="Service is healthy."),
    )