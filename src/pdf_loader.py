"""PDF loading for the AI Contract Analysis Pipeline.

This module is responsible solely for discovering contract PDFs on disk
and extracting their raw text into :class:`~src.models.ContractDocument`
instances. It performs no text normalization, no LLM calls, no clause
extraction, no summarization, and no output persistence — those are the
responsibilities of later pipeline stages.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import fitz  # PyMuPDF

from src.config import settings
from src.models import ContractDocument

logger = logging.getLogger(__name__)


class EncryptedPDFError(RuntimeError):
    """Raised when a PDF is password-protected and cannot be opened.

    Subclasses :class:`RuntimeError` so that any existing caller
    catching ``RuntimeError`` around :meth:`PDFLoader.extract_text`
    continues to work without modification, while still allowing
    callers who want to distinguish this specific case to catch
    :class:`EncryptedPDFError` first.
    """


class PDFLoader:
    """Discovers and loads contract PDFs into raw :class:`ContractDocument` objects.

    Reads PDFs from ``settings.raw_pdf_dir``, extracting only their raw
    text content. Text cleaning, LLM interaction, and downstream
    analysis are explicitly out of scope for this class.
    """

    def __init__(self) -> None:
        """Initialize the loader using the configured raw PDF directory."""
        self._raw_pdf_dir = settings.raw_pdf_dir
        logger.info("PDFLoader initialized (raw_pdf_dir=%s)", self._raw_pdf_dir)

    def load_documents(self, limit: int = 50) -> list[ContractDocument]:
        """Load contract PDFs from the configured raw PDF directory.

        Discovers PDF files in ``settings.raw_pdf_dir``, sorts them
        alphabetically, and loads at most ``limit`` of them. Files that
        are unreadable, empty, encrypted, or corrupted are skipped and
        logged; a single bad file never aborts the batch.

        Args:
            limit: Maximum number of PDFs to load.

        Returns:
            A list of :class:`~src.models.ContractDocument` instances,
            one per successfully loaded PDF, with ``clean_text`` left
            empty for later pipeline stages to populate.
        """
        if limit <= 0:
            raise ValueError("limit must be greater than 0.")

        pdf_paths = self._find_pdf_files()
        logger.info("Discovered %d PDF file(s) in %s", len(pdf_paths), self._raw_pdf_dir)

        documents: list[ContractDocument] = []
        for pdf_path in pdf_paths[:limit]:
            document = self._load_single_document(pdf_path)
            if document is not None:
                documents.append(document)

        logger.info(
            "Successfully loaded %d/%d PDFs.",
            len(documents),
            min(limit, len(pdf_paths)),
        )
        return documents

    def extract_text(self, pdf_path: Path) -> str:
        """Extract raw text from every page of a PDF, preserving page order.

        Logs the number of pages in the document and the time taken to
        extract text from it. If the PDF is encrypted, an attempt is
        made to open it with an empty password (common for PDFs that
        are "protected" but not meaningfully secured); if that fails,
        a clear :class:`EncryptedPDFError` is raised.

        Args:
            pdf_path: Path to the PDF file to extract text from.

        Returns:
            The extracted raw text, with individual pages separated by
            ``"\\n\\n"``.

        Raises:
            EncryptedPDFError: If the PDF requires a password that was
                not supplied (or the empty-password attempt failed).
                A subclass of ``RuntimeError``.
            fitz.FileDataError: If the PDF is corrupted or malformed.
            RuntimeError: If PyMuPDF fails to open or read the document
                for any other reason.
            OSError: If the file cannot be accessed due to a filesystem
                or permission error.
        """
        start_time = time.perf_counter()
        try:
            with fitz.open(pdf_path) as document:
                self._ensure_not_encrypted(document, pdf_path)
                page_count = document.page_count
                text = "\n\n".join(page.get_text() for page in document)
        except EncryptedPDFError:
            raise
        except fitz.FileDataError as exc:
            raise fitz.FileDataError(
                f"PDF '{pdf_path.name}' appears to be corrupted or malformed "
                f"and could not be parsed: {exc}"
            ) from exc
        except RuntimeError as exc:
            raise RuntimeError(
                f"PyMuPDF failed to open or read '{pdf_path.name}': {exc}"
            ) from exc
        except OSError as exc:
            raise OSError(
                f"Could not access PDF file '{pdf_path.name}' on disk: {exc}"
            ) from exc

        extraction_seconds = time.perf_counter() - start_time
        logger.info(
            "Extracted text from PDF (filename=%s, pages=%d, extraction_seconds=%.2f)",
            pdf_path.name,
            page_count,
            extraction_seconds,
        )
        return text

    @staticmethod
    def _ensure_not_encrypted(document: fitz.Document, pdf_path: Path) -> None:
        """Verify a PDF is not password-protected, attempting an empty password.

        Some PDFs are marked encrypted but use an empty user password;
        these can be opened transparently. Anything requiring a real
        password is treated as unreadable.

        Args:
            document: The already-opened PyMuPDF document to check.
            pdf_path: Path to the PDF file, used for error/log context.

        Raises:
            EncryptedPDFError: If the document is encrypted and could
                not be authenticated with an empty password.
        """
        if not document.is_encrypted:
            return

        if not document.authenticate(""):
            raise EncryptedPDFError(
                f"PDF '{pdf_path.name}' is encrypted and requires a password; "
                "it cannot be processed without one."
            )

        logger.warning(
            "PDF was encrypted but opened successfully with an empty password "
            "(filename=%s)",
            pdf_path.name,
        )

    def _find_pdf_files(self) -> list[Path]:
        """Discover and alphabetically sort PDF files in the raw PDF directory.

        Returns:
            A sorted list of paths to files with a ``.pdf`` extension.

        Raises:
            FileNotFoundError: If the configured raw PDF directory does
                not exist.
        """
        if not self._raw_pdf_dir.is_dir():
            logger.error("Raw PDF directory does not exist: %s", self._raw_pdf_dir)
            raise FileNotFoundError(f"Raw PDF directory not found: {self._raw_pdf_dir}")

        pdf_paths = sorted(
            path
            for path in self._raw_pdf_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".pdf"
        )
        return pdf_paths

    def _load_single_document(self, pdf_path: Path) -> ContractDocument | None:
        """Load a single PDF into a ContractDocument, skipping it on failure.

        Args:
            pdf_path: Path to the PDF file to load.

        Returns:
            A populated :class:`~src.models.ContractDocument`, or
            ``None`` if the file was skipped due to being empty,
            encrypted, corrupted, unreadable, or otherwise invalid.
        """
        try:
            raw_text = self.extract_text(pdf_path)
        except EncryptedPDFError as exc:
            logger.warning("Skipping encrypted PDF %s: %s", pdf_path.name, exc)
            return None
        except (fitz.FileDataError, RuntimeError, OSError) as exc:
            logger.warning("Skipping unreadable or corrupted PDF %s: %s", pdf_path.name, exc)
            return None

        if not self._validate_pdf(pdf_path, raw_text):
            return None

        logger.info("Loaded PDF: %s", pdf_path.name)
        return ContractDocument(
            filename=pdf_path.name,
            file_path=pdf_path,
            raw_text=raw_text,
            clean_text="",
        )

    @staticmethod
    def _validate_pdf(pdf_path: Path, raw_text: str) -> bool:
        """Validate that extracted PDF text is non-empty.

        Args:
            pdf_path: Path to the PDF file being validated, used for
                logging context.
            raw_text: The text extracted from the PDF.

        Returns:
            ``True`` if the text is non-empty after stripping
            whitespace, ``False`` otherwise.
        """
        if not raw_text.strip():
            logger.warning(
                "Skipping empty PDF (no extractable text found): %s", pdf_path.name
            )
            return False
        return True