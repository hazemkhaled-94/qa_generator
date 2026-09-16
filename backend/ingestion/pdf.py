"""Reading a PDF's page and character counts."""

from __future__ import annotations

import logging

import pymupdf

from ingestion.models import PdfFacts

log = logging.getLogger(__name__)


class UnreadablePdf(Exception):
    """Raised when a file cannot be opened and read as a PDF."""


def read_pdf(data: bytes) -> PdfFacts:
    """Opens a PDF and counts its pages and extractable characters.

    Args:
        data: The raw file bytes.

    Returns:
        The page and character counts.

    Raises:
        UnreadablePdf: If the file is encrypted, corrupt or otherwise cannot
            be opened.
    """
    try:
        with pymupdf.open(stream=data, filetype="pdf") as document:
            if document.needs_pass:
                raise UnreadablePdf("the PDF is password protected")
            # Summed page by page, so no document's text is held all at once.
            return PdfFacts(
                page_count=document.page_count,
                char_count=sum(len(page.get_text()) for page in document),
            )
    except UnreadablePdf:
        raise
    except Exception as exc:
        log.warning("unreadable PDF: %s: %s", type(exc).__name__, exc)
        raise UnreadablePdf(f"the PDF could not be read: {exc}") from exc
